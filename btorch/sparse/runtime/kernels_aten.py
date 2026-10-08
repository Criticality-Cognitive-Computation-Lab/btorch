"""Reference kernels built from stock PyTorch (ATen) operations.

They run on every device and dtype and define the semantics every other
backend must reproduce. Conventions shared by all kernels:

- The operator is destination-major CSR: ``crow [M + 1]``, ``col [E]``; entry
  ``k`` of row ``m`` contributes ``values[k] * x[col[k]]`` to ``y[m]``.
- ``values`` is ``[*vb, E]``; ``x`` is ``[*vb (broadcastable), *sample, N]``.
"""

from __future__ import annotations

import math
import warnings
import weakref

import torch
from torch import Tensor


# The CSR layout is used as a plain kernel input here, never handed to users.
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings(
    "ignore", message="Sparse invariant checks are implicitly disabled"
)


# Entry-times-sample budget for temporaries of the gather-based kernels.
_CHUNK_ELEMENTS = 1 << 25
_NATIVE_CSR_DTYPES = (torch.float32, torch.float64)


_ROW_CACHE: dict = {}


def _row_indices(crow: Tensor) -> Tensor:
    """Row of every entry, expanded from the pointers (memoised).

    An entry is valid only for the very tensor object it was built from, at
    the version it was built at: the weak reference rules out a new tensor
    that reuses a freed address, the version counter an in-place rebuild of
    the pointers (rewiring).
    """
    key = (crow.data_ptr(), crow.shape[0], crow.device)
    hit = _ROW_CACHE.get(key)
    if hit is not None and hit[0]() is crow and hit[1] == crow._version:
        return hit[2]
    counts = crow[1:] - crow[:-1]
    row = torch.repeat_interleave(
        torch.arange(counts.shape[0], device=crow.device), counts
    )
    if len(_ROW_CACHE) >= 16:
        _ROW_CACHE.clear()
    try:
        _ROW_CACHE[key] = (weakref.ref(crow), crow._version, row)
    except (TypeError, RuntimeError):
        pass  # tensors without weak references or version counters
    return row


def _csr_mm(crow: Tensor, col: Tensor, values: Tensor, x: Tensor, n_out: int) -> Tensor:
    """One pattern, one value vector, ``x [*sample, N]`` -> ``[*sample,
    M]``."""
    n_in = x.shape[-1]
    lead = x.shape[:-1]
    if values.dtype in _NATIVE_CSR_DTYPES:
        # The CUDA CSR product reads the value buffer as if it were dense and
        # contiguous; an expanded or strided view would give wrong results.
        values = values.contiguous()
        mat = torch.sparse_csr_tensor(
            crow, col, values, size=(n_out, n_in), check_invariants=False
        )
        if x.ndim == 1:
            return mat @ x
        # (A @ Xᵀ)ᵀ; contiguous so the layout matches a freshly allocated
        # output (the fake kernel used for tracing reports one).
        # A contiguous right-hand side is several times faster than the
        # strided transpose view for the batched product.
        flat = x.reshape(math.prod(lead), n_in).T.contiguous()
        return (mat @ flat).T.contiguous().reshape(*lead, n_out)
    # Dtypes without a native CSR product (half, bfloat16, integers).
    row = _row_indices(crow)
    flat = x.reshape(math.prod(lead), n_in)
    out = flat.new_zeros(flat.shape[0], n_out)
    step = max(1, _CHUNK_ELEMENTS // max(col.shape[0], 1))
    for i in range(0, flat.shape[0], step):
        out[i : i + step].index_add_(1, row, flat[i : i + step, col] * values)
    return out.reshape(*lead, n_out)


def csr_matvec(crow: Tensor, col: Tensor, values: Tensor, x: Tensor) -> Tensor:
    """``y[..., m] = sum_k values[..., k] * x[..., col[k]]`` over row ``m``."""
    n_out = crow.shape[0] - 1
    dtype = torch.promote_types(values.dtype, x.dtype)
    values, x = values.to(dtype), x.to(dtype)
    n_vb = values.ndim - 1
    if x.ndim - 1 < n_vb:
        raise ValueError(
            f"values has batch shape {tuple(values.shape[:-1])}, so x needs at "
            f"least {n_vb + 1} dimensions, got {tuple(x.shape)}."
        )
    if n_vb == 0:
        return _csr_mm(crow, col, values, x, n_out)
    # Shared pattern, one value vector per batch member.
    batch = torch.broadcast_shapes(values.shape[:-1], x.shape[:n_vb])
    values = values.expand(*batch, -1).reshape(math.prod(batch), values.shape[-1])
    x = x.expand(*batch, *x.shape[n_vb:])
    sample = x.shape[n_vb:-1]
    x = x.reshape(values.shape[0], *x.shape[n_vb:])
    out = torch.stack(
        [_csr_mm(crow, col, values[g], x[g], n_out) for g in range(values.shape[0])]
    )
    return out.reshape(*batch, *sample, n_out)


def edge_grad(crow: Tensor, col: Tensor, grad: Tensor, x: Tensor, n_vb: int) -> Tensor:
    """Gradient of ``csr_matvec`` w.r.t. ``values``.

    ``g[..., k] = sum_samples grad[..., row[k]] * x[..., col[k]]`` (a sampled
    dense-dense product). Samples are processed in chunks so the temporary
    stays bounded and no ``M x N`` matrix is ever formed.

    Args:
        crow: ``[M + 1]`` row pointers.
        col: ``[E]`` input index of each entry.
        grad: ``[*vb, *sample, M]`` output gradient.
        x: ``[*vb (broadcastable), *sample, N]`` forward input.
        n_vb: Number of value-batch dimensions.

    Returns:
        ``[*vb, E]``.
    """
    row = _row_indices(crow)
    n_edge = col.shape[0]
    batch = grad.shape[:n_vb]
    x = x.expand(*batch, *x.shape[n_vb:]).to(grad.dtype)
    n_sample = math.prod(grad.shape[n_vb:-1])
    grad = grad.reshape(*batch, n_sample, grad.shape[-1])
    x = x.reshape(*batch, n_sample, x.shape[-1])
    out = grad.new_zeros(*batch, n_edge)
    n_member = max(1, math.prod(batch))
    step = max(1, _CHUNK_ELEMENTS // max(n_edge * n_member, 1))
    for i in range(0, grad.shape[-2], step):
        g = grad[..., i : i + step, :].index_select(-1, row)
        out += (g * x[..., i : i + step, :].index_select(-1, col)).sum(-2)
    return out


def pack_spikes(x: Tensor) -> tuple[Tensor, Tensor]:
    """Pack the non-zero entries of ``x [B, N]`` into per-sample index lists.

    Returns:
        ``(active_idx, ptr)``: the source index of every non-zero entry in
        sample-major order, and ``[B + 1]`` offsets such that sample ``b``
        owns ``active_idx[ptr[b]:ptr[b + 1]]``.
    """
    sample, active_idx = x.nonzero(as_tuple=True)
    counts = torch.bincount(sample, minlength=x.shape[0])
    ptr = torch.zeros(x.shape[0] + 1, dtype=torch.long, device=x.device)
    ptr[1:] = torch.cumsum(counts, 0)
    return active_idx, ptr


def spike_push_values(
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    values: Tensor,
    amplitude: Tensor,
    active_idx: Tensor,
    ptr: Tensor,
    n_out: int,
) -> Tensor:
    """Source-driven propagation of packed events.

    Visits only the out-edges of active sources, so the cost scales with the
    number of delivered events instead of with the number of edges. Built
    from differentiable operations: gradients flow to ``values`` and
    ``amplitude``.

    Args:
        t_crow: ``[N + 1]`` pointers of the source-major (transposed) CSR.
        t_col: ``[E]`` destination of each entry in source-major order.
        t_perm: ``[E]`` position in ``values`` of each source-major entry.
        values: ``[E]`` edge values in destination-major order.
        amplitude: ``[n_active]`` value of each event.
        active_idx: ``[n_active]`` source index of each event, sample-major.
        ptr: ``[B + 1]`` sample offsets into the event list.
        n_out: Number of destinations ``M``.

    Returns:
        ``[B, M]``.
    """
    n_batch = ptr.shape[0] - 1
    device = active_idx.device
    dtype = torch.promote_types(values.dtype, amplitude.dtype)
    out = torch.zeros(n_batch * n_out, dtype=dtype, device=device)
    n_active = active_idx.shape[0]
    if n_active == 0:
        return out.reshape(n_batch, n_out)
    sample = torch.repeat_interleave(
        torch.arange(n_batch, device=device), ptr[1:] - ptr[:-1]
    )
    start = t_crow[active_idx]
    degree = t_crow[active_idx + 1] - start
    # Expand every event into the out-edges of its source.
    owner = torch.repeat_interleave(torch.arange(n_active, device=device), degree)
    first = torch.cumsum(degree, 0) - degree
    edge = start[owner] + (torch.arange(owner.shape[0], device=device) - first[owner])
    contrib = values[t_perm[edge]].to(dtype) * amplitude.to(dtype)[owner]
    out = out.index_add(0, sample[owner] * n_out + t_col[edge], contrib)
    return out.reshape(n_batch, n_out)


def spike_push(
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    values: Tensor,
    x: Tensor,
    active_idx: Tensor,
    ptr: Tensor,
    n_out: int,
) -> Tensor:
    """Source-driven propagation of the packed non-zero entries of ``x``.

    Args:
        t_crow: ``[N + 1]`` pointers of the source-major (transposed) CSR.
        t_col: ``[E]`` destination of each entry in source-major order.
        t_perm: ``[E]`` position in ``values`` of each source-major entry.
        values: ``[E]`` edge values in destination-major order.
        x: ``[B, N]`` dense input (the logical tensor; supplies amplitudes).
        active_idx: Packed source indices from :func:`pack_spikes`.
        ptr: ``[B + 1]`` sample offsets from :func:`pack_spikes`.
        n_out: Number of destinations ``M``.

    Returns:
        ``[B, M]``.
    """
    sample = torch.repeat_interleave(
        torch.arange(x.shape[0], device=x.device), ptr[1:] - ptr[:-1]
    )
    amplitude = x[sample, active_idx]
    return spike_push_values(
        t_crow, t_col, t_perm, values, amplitude, active_idx, ptr, n_out
    )


def torch_sparse_available() -> bool:
    """Whether the optional ``torch_sparse`` package can be imported."""
    try:
        import torch_sparse  # noqa: F401
    except ImportError:
        return False
    return True


def csr_matvec_torch_sparse(
    crow: Tensor, col: Tensor, values: Tensor, x: Tensor
) -> Tensor:
    """``csr_matvec`` through the optional ``torch_sparse`` package.

    Kept as a selectable legacy backend; nothing depends on it.
    """
    if values.ndim != 1:
        return csr_matvec(crow, col, values, x)
    from torch_sparse import spmm

    n_out, n_in = crow.shape[0] - 1, x.shape[-1]
    dtype = torch.promote_types(values.dtype, x.dtype)
    index = torch.stack([_row_indices(crow), col])
    flat = x.reshape(-1, n_in).to(dtype)
    out = spmm(index, values.to(dtype), n_out, n_in, flat.T).T
    return out.contiguous().reshape(*x.shape[:-1], n_out)
