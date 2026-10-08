"""Conversion between btorch sparse arrays, PyTorch and SciPy.

This module is the single place where foreign sparse objects are understood.
Every higher-level API that accepts "a sparse matrix" calls
:func:`as_sparse`, so all of them accept the same inputs and treat them the
same way. Conversions never transpose and never densify.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .base import Sparse
from .coo import COO
from .csr import CSC, CSR
from .properties import Properties


_TORCH_LAYOUTS = {
    "coo": torch.sparse_coo,
    "csr": torch.sparse_csr,
    "csc": torch.sparse_csc,
    "bsr": torch.sparse_bsr,
    "bsc": torch.sparse_bsc,
}
_LAYOUT_NAMES = {v: k for k, v in _TORCH_LAYOUTS.items()}


def _is_scipy_sparse(obj: Any) -> bool:
    # Cheap module check first so SciPy is only imported for SciPy objects.
    if not type(obj).__module__.startswith("scipy.sparse"):
        return False
    import scipy.sparse

    return scipy.sparse.issparse(obj)


def from_torch(tensor: Tensor, *, batch_dim: int = 0) -> Sparse:
    """Wrap a PyTorch sparse tensor without copying or transposing.

    COO, CSR and CSC keep their format. BSR and BSC are converted through
    COO, which is lossless. Values keep their autograd history, so gradients
    flow back to the tensor the sparse tensor was built from.

    An uncoalesced COO tensor keeps its entries as stored (order and
    duplicates), unless it requires grad: PyTorch only exposes the values of
    an uncoalesced tensor to autograd after coalescing, so it is coalesced
    then. A batched CSR/CSC tensor whose members have different patterns
    becomes a batched COO.

    Args:
        tensor: Tensor with a sparse layout.
        batch_dim: For a COO tensor, how many leading sparse dimensions are
            batch coordinates. PyTorch COO has no notion of a batch, so a
            tensor of shape ``(G, M, N)`` is read as a 3-D sparse array
            unless ``batch_dim=1`` says it is ``G`` matrices.

    Returns:
        ``COO``, ``CSR`` or ``CSC`` with ``shape == tensor.shape``.

    Raises:
        TypeError: If ``tensor`` is not sparse.
    """
    if not isinstance(tensor, Tensor) or tensor.layout == torch.strided:
        raise TypeError(
            "from_torch expects a torch sparse tensor; dense tensors are never "
            "sparsified implicitly (use sparse.from_dense)."
        )
    shape = tuple(tensor.shape)
    layout = tensor.layout
    if layout in (torch.sparse_bsr, torch.sparse_bsc):
        tensor = tensor.to_sparse(layout=torch.sparse_coo)
        layout = torch.sparse_coo
    if layout == torch.sparse_coo:
        reordered = False
        if tensor.is_coalesced():
            indices, values = tensor.indices(), tensor.values()
            canonical = True
        elif tensor.requires_grad:
            # PyTorch only exposes the values of an uncoalesced tensor to
            # autograd after coalescing, which merges and reorders entries.
            tensor = tensor.coalesce()
            indices, values = tensor.indices(), tensor.values()
            canonical = reordered = True
        else:
            # Keep the entries exactly as stored (order and duplicates), so
            # arrays aligned with them stay aligned.
            indices, values = tensor._indices(), tensor._values()
            canonical = False
        out = COO(
            indices,
            values,
            shape,
            batch_dim=batch_dim,
            dense_dim=tensor.dense_dim(),
            properties=Properties(sorted=canonical, unique=canonical),
            check=False,
        )
        out._reordered = reordered
        return out
    if batch_dim:
        raise ValueError(
            "batch_dim applies to COO tensors; batched CSR/CSC tensors carry "
            "their batch dimensions themselves."
        )
    if layout == torch.sparse_csr:
        pointer, minor, cls = tensor.crow_indices(), tensor.col_indices(), CSR
    elif layout == torch.sparse_csc:
        pointer, minor, cls = tensor.ccol_indices(), tensor.row_indices(), CSC
    else:
        raise TypeError(f"Unsupported torch sparse layout {layout}.")
    if pointer.ndim > 1:
        # Batched compressed tensor: keep the compressed format when the
        # members share a pattern, otherwise fall back to batched COO.
        n_batch = pointer.ndim - 1
        flat_p = pointer.reshape(-1, pointer.shape[-1])
        flat_m = minor.reshape(-1, minor.shape[-1])
        if bool((flat_p == flat_p[0]).all()) and bool((flat_m == flat_m[0]).all()):
            pointer, minor = flat_p[0], flat_m[0]
        else:
            coo = tensor.to_sparse(layout=torch.sparse_coo).coalesce()
            return COO(
                coo.indices(),
                coo.values(),
                shape,
                batch_dim=n_batch,
                dense_dim=coo.dense_dim(),
                properties=Properties(sorted=True, unique=True),
                check=False,
            )
    # PyTorch does not guarantee sorted or unique indices within a row of a
    # compressed tensor, so nothing is claimed about them.
    return cls(
        pointer,
        minor,
        tensor.values(),
        shape,
        dense_dim=tensor.dense_dim(),
        check=False,
    )


def from_scipy(array: Any, *, device=None, dtype: torch.dtype | None = None) -> Sparse:
    """Convert a SciPy sparse array or matrix.

    Both ``scipy.sparse.sparray`` and the legacy ``spmatrix`` are accepted.
    COO, CSR and CSC keep their format (including unsorted indices and
    duplicate entries, exactly as stored); every other format (BSR, LIL, DOK,
    DIA) is converted through COO. The data is copied from CPU memory.

    Args:
        array: SciPy sparse array or matrix.
        device: Target device (default CPU).
        dtype: Value dtype (default: the array's dtype).

    Returns:
        ``COO``, ``CSR`` or ``CSC`` with ``shape == array.shape``.

    Raises:
        TypeError: If ``array`` is not a SciPy sparse object.
    """
    if not _is_scipy_sparse(array):
        raise TypeError(f"from_scipy expects a SciPy sparse array, got {type(array)}.")
    fmt = array.format
    if fmt not in ("coo", "csr", "csc"):
        array = array.tocoo()
        fmt = "coo"
    shape = tuple(int(s) for s in array.shape)
    if len(shape) != 2:
        raise NotImplementedError("Only 2-D SciPy sparse arrays are supported.")

    # Copies: the result must never alias (and later mutate) the caller's data.
    def index(a) -> Tensor:
        return torch.tensor(np.asarray(a, dtype=np.int64), device=device)

    values = torch.tensor(np.asarray(array.data), device=device)
    if dtype is not None:
        values = values.to(dtype)
    if fmt == "coo":
        indices = torch.stack([index(array.row), index(array.col)])
        # SciPy's ``has_canonical_format`` flag is not reliable (a DOK array
        # converted with ``tocoo()`` sets it for insertion-ordered entries),
        # so sortedness is verified here, once, in O(nnz).
        key = indices[0] * shape[1] + indices[1]
        canonical = bool((key[1:] > key[:-1]).all())
        return COO(
            indices,
            values,
            shape,
            properties=Properties(sorted=canonical, unique=canonical),
            check=False,
        )
    cls = CSR if fmt == "csr" else CSC
    unique = bool(array.has_canonical_format)
    return cls(
        index(array.indptr),
        index(array.indices),
        values,
        shape,
        properties=Properties(sorted=bool(array.has_sorted_indices), unique=unique),
        check=False,
    )


def from_dense(tensor: Tensor) -> COO:
    """Build a COO array from the non-zero entries of a dense tensor.

    Every dimension becomes a sparse dimension. Values stay connected to
    ``tensor`` for autograd.
    """
    if tensor.layout != torch.strided:
        raise TypeError("from_dense expects a dense tensor.")
    indices = tensor.nonzero().T
    return COO(
        indices,
        tensor[tuple(indices)],
        tuple(tensor.shape),
        properties=Properties(sorted=True, unique=True),
        check=False,
    )


def as_sparse(
    obj: Any,
    *,
    format: str | None = None,
    device=None,
    dtype: torch.dtype | None = None,
    batch_dim: int = 0,
) -> Sparse:
    """Interpret ``obj`` as a btorch sparse array.

    This is the one conversion entry point: it accepts a btorch
    :class:`Sparse`, a PyTorch sparse tensor, or a SciPy sparse array/matrix.
    The orientation is never changed.

    Args:
        obj: Sparse object of any supported library.
        format: Optional target format (``"coo"``, ``"csr"`` or ``"csc"``).
        device: Optional target device.
        dtype: Optional value dtype.
        batch_dim: Leading batch dimensions of a PyTorch COO tensor (see
            :func:`from_torch`).

    Returns:
        A :class:`Sparse` (the same object if nothing had to change).

    Raises:
        TypeError: For dense tensors, arrays and unsupported types; dense
            data is never sparsified implicitly.
    """
    if isinstance(obj, Sparse):
        out = obj
    elif isinstance(obj, Tensor):
        out = from_torch(obj, batch_dim=batch_dim)
    elif _is_scipy_sparse(obj):
        out = from_scipy(obj)
    else:
        raise TypeError(
            f"Cannot interpret {type(obj).__name__} as a sparse array. Expected "
            "a btorch Sparse, a torch sparse tensor or a SciPy sparse array."
        )
    if device is not None or dtype is not None:
        out = out.to(device=device, dtype=dtype)
    if format is not None and out.format != format:
        if format not in ("coo", "csr", "csc"):
            raise ValueError(
                f"Unknown or non-native sparse format {format!r}; native formats "
                "are 'coo', 'csr' and 'csc'."
            )
        out = getattr(out, f"to{format}")()
    return out


def _expand_batch_coo(A: COO) -> tuple[Tensor, Tensor, int]:
    """Indices/values of a torch COO tensor with explicit batch coordinates."""
    n_vb = A._value_batch_dim
    indices, values = A._indices, A._values
    if n_vb == 0:
        return indices, values, indices.shape[0]
    batch = A.shape[:n_vb]
    nnz = A.nnz
    grids = torch.meshgrid(
        *[torch.arange(b, device=indices.device) for b in batch], indexing="ij"
    )
    n_member = grids[0].numel()
    batch_idx = torch.stack([g.reshape(-1) for g in grids])  # [n_vb, G]
    batch_idx = batch_idx.repeat_interleave(nnz, dim=1)
    full = torch.cat([batch_idx, indices.repeat(1, n_member)])
    values = values.reshape(n_member * nnz, *values.shape[n_vb + 1 :])
    return full, values, full.shape[0]


def to_torch(A: Sparse, layout: str | torch.layout | None = None, **kwargs) -> Tensor:
    """Export a sparse array as a PyTorch sparse tensor.

    A shared-pattern batch has no PyTorch equivalent, so its pattern is
    repeated for every batch member on export.

    Args:
        A: Sparse array.
        layout: Target layout name or ``torch.sparse_*`` layout. Defaults to
            the stored format.
        **kwargs: ``blocksize`` for ``"bsr"`` / ``"bsc"``.

    Returns:
        Sparse tensor with ``shape == A.shape``; values keep their autograd
        history.
    """
    if layout is None:
        layout = A.format
    if not isinstance(layout, str):
        layout = _LAYOUT_NAMES[layout]
    if layout not in _TORCH_LAYOUTS:
        raise ValueError(f"Unknown torch sparse layout {layout!r}.")

    if layout in ("csr", "csc") and A.format == layout:
        bd = A.batch_dim()
        batch = A.shape[:bd]
        pointer = A._pointer.expand(*batch, -1)
        minor = A._minor.expand(*batch, -1)
        make = torch.sparse_csr_tensor if layout == "csr" else torch.sparse_csc_tensor
        return make(pointer, minor, A._values, size=A.shape)
    if layout in ("csr", "csc") and not getattr(A, "_index_batch_dim", 0):
        return to_torch(getattr(A, f"to{layout}")(), layout)

    coo = A.tocoo()
    indices, values, _ = _expand_batch_coo(coo)
    out = torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        is_coalesced=coo.properties.canonical or None,
    )
    if layout == "coo":
        return out
    if coo.batch_dim() or coo.sparse_dim() != 2:
        raise NotImplementedError(
            f"Export to the torch {layout} layout is implemented for "
            "unbatched matrices (and same-format shared-pattern batches)."
        )
    return out.coalesce().to_sparse(layout=_TORCH_LAYOUTS[layout], **kwargs)


def to_scipy(A: Sparse, format: str | None = None, **kwargs):
    """Export a sparse matrix as a ``scipy.sparse`` array.

    Values are detached and copied to CPU memory.

    Args:
        A: Unbatched sparse matrix without dense entry dimensions.
        format: SciPy format name. Defaults to the stored format.
        **kwargs: Passed to the SciPy constructor for formats reached through
            a SciPy conversion (e.g. ``blocksize`` for ``"bsr"``).

    Returns:
        ``scipy.sparse.sparray`` with ``shape == A.shape``.

    Raises:
        ValueError: If ``A`` is batched, has dense entry dimensions, or does
            not have exactly two sparse dimensions.
    """
    import scipy.sparse

    if A.batch_dim() or A.dense_dim() or A.sparse_dim() != 2:
        raise ValueError(
            "SciPy sparse arrays are 2-D: export one batch member at a time "
            f"(got batch_shape={A.batch_shape}, sparse_shape={A.sparse_shape}, "
            f"dense_shape={A.dense_shape})."
        )
    if format is None:
        format = A.format

    def np_(t: Tensor):
        return t.detach().cpu().numpy()

    if A.format == "coo":
        native = scipy.sparse.coo_array(
            (np_(A._values), (np_(A._indices[0]), np_(A._indices[1]))), shape=A.shape
        )
    else:
        cls = scipy.sparse.csr_array if A.format == "csr" else scipy.sparse.csc_array
        native = cls((np_(A._values), np_(A._minor), np_(A._pointer)), shape=A.shape)
    if format == A.format:
        return native
    if format == "bsr":
        return scipy.sparse.bsr_array(native.tocsr(), **kwargs)
    return native.asformat(format)
