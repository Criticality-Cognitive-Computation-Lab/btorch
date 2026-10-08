"""Products with sparse arrays.

The public operations are :func:`matmul` (``A @ x``), :func:`matvec`,
:func:`rmatvec` and :func:`stack`. How a product is executed (which
representation, which algorithm, which backend) is not part of this API.

The implementation here is the reference one: a gather along the input index
of every stored entry followed by an ``index_add`` along its output index. It
uses only differentiable dense tensor operations, so autograd, ``torch.compile``
and every device work, and its backward never builds a dense matrix.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .base import Sparse
from .coo import COO, _ravel
from .properties import Properties


def _apply(
    out_index: Tensor,
    in_index: Tensor,
    values: Tensor,
    x: Tensor,
    n_out: int,
    value_batch_dim: int,
    dense_dim: int,
) -> Tensor:
    """``y[..., out_index[e], :] += values[..., e, :] * x[..., in_index[e]]``.

    Args:
        out_index: ``[E]`` output coordinate of each entry.
        in_index: ``[E]`` input coordinate of each entry.
        values: ``[*value_batch, E, *dense]``.
        x: ``[*value_batch (broadcastable), *sample, n_in]``.
        n_out: Size of the output axis.
        value_batch_dim: Number of leading batch dimensions of ``values``.
        dense_dim: Number of trailing dense dimensions of ``values``.

    Returns:
        ``[*value_batch, *sample, n_out, *dense]``.
    """
    gathered = x.index_select(-1, in_index)
    n_sample = gathered.ndim - 1 - value_batch_dim
    values = values.reshape(
        *values.shape[:value_batch_dim],
        *(1,) * n_sample,
        *values.shape[value_batch_dim:],
    )
    gathered = gathered.reshape(*gathered.shape, *(1,) * dense_dim)
    contrib = gathered * values
    edge_axis = value_batch_dim + n_sample
    out = contrib.new_zeros(
        *contrib.shape[:edge_axis], n_out, *contrib.shape[edge_axis + 1 :]
    )
    return out.index_add(edge_axis, out_index, contrib)


def _check_batch(A: Sparse, x: Tensor, n_in: int, what: str) -> None:
    if x.shape[-1] != n_in:
        raise ValueError(
            f"{what}: the last dimension of the input is {x.shape[-1]}, "
            f"expected {n_in} for a sparse array of shape {A.shape}."
        )
    bd = A.batch_dim()
    if bd == 0:
        return
    if x.ndim - 1 < bd:
        raise ValueError(
            f"{what}: the sparse array has batch shape {A.batch_shape}, so the "
            f"input needs shape [*batch, ..., {n_in}], got {tuple(x.shape)}. "
            "Batch members are never combined implicitly with every input "
            "sample: add the batch dimensions explicitly, e.g. "
            "x.unsqueeze(0), and they will broadcast."
        )
    for a, b in zip(A.batch_shape, x.shape[:bd]):
        if b != 1 and a != 1 and b != a:
            raise ValueError(
                f"{what}: the leading dimensions {tuple(x.shape[:bd])} of the "
                f"input do not broadcast with the batch shape {A.batch_shape}."
            )


def _matvec(A: Sparse, x: Tensor, transpose: bool) -> Tensor:
    if A.sparse_dim() != 2:
        raise ValueError(
            f"Matrix products need two sparse dimensions, got {A.sparse_dim()}; "
            "use sparse.einsum for higher-order arrays."
        )
    lead, row, col = A._edge_index()
    n_row, n_col = A.sparse_shape
    if transpose:
        row, col, n_row, n_col = col, row, n_col, n_row
    _check_batch(A, x, n_col, "rmatvec" if transpose else "matvec")
    values = A._values
    if not (x.is_floating_point() or x.is_complex()) and values.is_floating_point():
        x = x.to(values.dtype)
    n_vb = A._value_batch_dim
    dd = A.dense_dim()
    if lead is None:
        return _apply(row, col, values, x, n_row, n_vb, dd)

    # Different patterns per batch member: fold the batch coordinate into the
    # row/column index and run one product over the block-diagonal operator.
    bd = A.batch_dim()
    ib_shape = A.batch_shape[n_vb:]
    n_member = math.prod(ib_shape)
    x = x.expand(*x.shape[:n_vb], *ib_shape, *x.shape[bd:])
    n_sample = x.ndim - 1 - bd
    vb_axes = list(range(n_vb))
    ib_axes = list(range(n_vb, bd))
    sample_axes = list(range(bd, bd + n_sample))
    x = x.permute(*vb_axes, *sample_axes, *ib_axes, x.ndim - 1)
    sample_shape = x.shape[n_vb : n_vb + n_sample]
    x = x.reshape(*x.shape[:n_vb], *sample_shape, n_member * n_col)
    member = _ravel(lead, ib_shape)
    y = _apply(
        member * n_row + row,
        member * n_col + col,
        values,
        x,
        n_member * n_row,
        n_vb,
        dd,
    )
    dense = y.shape[y.ndim - dd :] if dd else ()
    y = y.reshape(*y.shape[:n_vb], *sample_shape, *ib_shape, n_row, *dense)
    # [vb, sample, ib, n_row, dense] -> [vb, ib, sample, n_row, dense]
    n_ib = len(ib_shape)
    s0 = n_vb
    i0 = n_vb + n_sample
    order = (
        vb_axes
        + list(range(i0, i0 + n_ib))
        + list(range(s0, s0 + n_sample))
        + list(range(i0 + n_ib, y.ndim))
    )
    return y.permute(*order)


def matvec(A: Sparse, x: Tensor) -> Tensor:
    """Apply a sparse matrix along the last axis of a dense tensor.

    For ``A.shape == (*batch, M, N, *dense)`` and ``x.shape == (*batch, ...,
    N)`` the result has shape ``(*batch, ..., M, *dense)``:

    .. math::
        y_{\\ldots m} = \\sum_n A_{m n}\\, x_{\\ldots n}

    Any number of leading dimensions of ``x`` is treated as independent
    samples. If ``A`` has batch dimensions they align with the *leading*
    dimensions of ``x`` and broadcast (size 1 in ``x`` broadcasts); they are
    never combined with the sample dimensions implicitly.

    Args:
        A: Sparse array with two sparse dimensions.
        x: Dense tensor whose last dimension is ``N``.

    Returns:
        Dense tensor.

    Raises:
        ValueError: On a shape mismatch, or if ``x`` has fewer leading
            dimensions than ``A`` has batch dimensions.
    """
    return _matvec(A, x, transpose=False)


def rmatvec(A: Sparse, x: Tensor) -> Tensor:
    """Apply the transpose of a sparse matrix along the last axis of ``x``.

    ``rmatvec(A, x)`` equals ``matvec(A.T, x)`` (and ``x @ A`` for 1-D ``x``)
    without building the transpose. ``x`` has shape ``(*batch, ..., M)`` and
    the result ``(*batch, ..., N, *dense)``.
    """
    return _matvec(A, x, transpose=True)


def _matmat(A: Sparse, x: Tensor, transpose: bool) -> Tensor:
    """Batched product with ``torch.matmul`` (right-aligned) broadcasting.

    ``x`` is ``[..., K, n]`` with the operator applied along the last axis;
    batch dimensions of ``A`` align with the dimensions right before ``K``.
    """
    bd = A.batch_dim()
    if bd == 0:
        return _matvec(A, x, transpose)
    n_lead = x.ndim - 2
    if n_lead < bd:
        x = x.reshape(*(1,) * (bd - n_lead), *x.shape)
        n_lead = bd
    extra = n_lead - bd
    # [*extra, *batch, K, n] -> [*batch, *extra, K, n]
    order = list(range(extra, n_lead)) + list(range(extra)) + [n_lead, n_lead + 1]
    y = _matvec(A, x.permute(*order), transpose)
    back = list(range(bd, n_lead)) + list(range(bd)) + list(range(n_lead, y.ndim))
    return y.permute(*back)


def matmul(a, b):
    """Matrix product with one sparse operand; the function behind ``@``.

    Shapes follow :func:`torch.matmul`:

    - ``A @ x`` with 1-D ``x`` of length ``N`` returns length ``M``.
    - ``A @ X`` with ``X.shape == (..., N, K)`` returns ``(..., M, K)``.
    - ``x @ A`` with 1-D ``x`` of length ``M`` returns length ``N``.
    - ``X @ A`` with ``X.shape == (..., K, M)`` returns ``(..., K, N)``.

    Batch dimensions of the sparse operand broadcast with the leading
    dimensions of the dense operand as in ``torch.matmul``. To apply a matrix
    to a stack of vectors ``[..., N]`` use :func:`matvec`.

    Args:
        a: Left operand (sparse array or dense tensor).
        b: Right operand (dense tensor or sparse array).

    Returns:
        Dense tensor.
    """
    a_sparse, b_sparse = isinstance(a, Sparse), isinstance(b, Sparse)
    if a_sparse and b_sparse:
        raise NotImplementedError(
            "sparse @ sparse (operator composition / SpGEMM) is not implemented."
        )
    if a_sparse:
        if not isinstance(b, Tensor):
            return NotImplemented
        if a.dense_dim() and b.ndim > 1:
            raise NotImplementedError(
                "A @ X with a matrix right-hand side is not defined for arrays "
                "with dense entry dimensions; use matvec or sparse.einsum."
            )
        if b.ndim == 1:
            if a.batch_dim():
                b = b.reshape(*(1,) * a.batch_dim(), -1)
            return _matvec(a, b, transpose=False)
        return _matmat(a, b.mT, transpose=False).mT
    if b_sparse:
        if not isinstance(a, Tensor):
            return NotImplemented
        if b.dense_dim() and a.ndim > 1:
            raise NotImplementedError(
                "X @ A with a matrix left-hand side is not defined for arrays "
                "with dense entry dimensions; use rmatvec or sparse.einsum."
            )
        if a.ndim == 1:
            if b.batch_dim():
                a = a.reshape(*(1,) * b.batch_dim(), -1)
            return _matvec(b, a, transpose=True)
        return _matmat(b, a, transpose=True)
    return torch.matmul(a, b)


def stack(arrays: Sequence[Sparse]) -> Sparse:
    """Stack sparse arrays along a new leading batch dimension.

    The members must have the same shape. If they all store the same sparsity
    pattern the result shares it (indices stored once, values ``[G, nnz]``)
    and keeps the format of the first member. Otherwise the members may have
    different numbers of entries; the result is a batched COO that stores
    ``sum(nnz)`` entries without padding.

    Args:
        arrays: Sparse arrays of identical shape.

    Returns:
        Sparse array of shape ``(len(arrays), *shape)``.

    Examples:
        >>> A = sparse.stack([A0, A1, A2])       # doctest: +SKIP
        >>> A.batch_shape
        (3,)
        >>> y = A.matvec(x)                      # x: [3, B, N] -> [3, B, M]
    """
    arrays = list(arrays)
    if not arrays:
        raise ValueError("stack needs at least one sparse array.")
    first = arrays[0]
    for a in arrays:
        if not isinstance(a, Sparse):
            raise TypeError(f"stack expects sparse arrays, got {type(a).__name__}.")
        if a.shape != first.shape or a.dense_dim() != first.dense_dim():
            raise ValueError(
                f"All stacked arrays need the same shape, got {a.shape} and "
                f"{first.shape}."
            )
        if a.batch_dim():
            raise NotImplementedError("Stacking already batched arrays.")

    same_pattern = all(a.format == first.format for a in arrays) and all(
        _same_pattern(first, a) for a in arrays[1:]
    )
    if same_pattern:
        return first.with_values(torch.stack([a._values for a in arrays]))

    coos = [a.tocoo() for a in arrays]
    indices = torch.cat(
        [
            torch.cat([torch.full_like(c._indices[:1], g), c._indices])
            for g, c in enumerate(coos)
        ],
        dim=1,
    )
    values = torch.cat([c._values for c in coos])
    canonical = all(c.properties.canonical for c in coos)
    return COO(
        indices,
        values,
        (len(coos), *first.shape),
        batch_dim=1,
        dense_dim=first.dense_dim(),
        properties=Properties(sorted=canonical, unique=canonical),
        check=False,
    )


def _same_pattern(a: Sparse, b: Sparse) -> bool:
    if a.nnz != b.nnz:
        return False
    if a.format == "coo":
        return torch.equal(a._indices, b._indices)
    return torch.equal(a._pointer, b._pointer) and torch.equal(a._minor, b._minor)
