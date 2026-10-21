"""Device spike-list preparation for the existing packed Triton SpMSpV kernel.

Compact active source IDs once per sample, then reuse the edge-balanced
packed fanout kernel. Fixed per-sample capacity covers every source, so
underestimated activity cannot overflow and CUDA graphs need no host-
side count read. Unused slots have zero degree; the prefix sum and
fanout ignore them.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import kernels_triton, kernels_triton_push
from .backend import KernelCache


try:
    import triton
    import triton.language as tl
except ImportError:  # CPU installations do not need Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _compact_spikes_kernel(
        x,
        degree,
        active,
        active_degree,
        counts,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        sample = tl.program_id(1)
        source = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        amplitude = tl.load(
            x + sample.to(tl.int64) * N + source, mask=source < N, other=0.0
        )
        live = (source < N) & (amplitude != 0.0)
        rank = tl.cumsum(live.to(tl.int32)) - 1
        count = tl.sum(live.to(tl.int32))
        if count > 0:
            start = tl.atomic_add(counts + sample, count)
            slot = sample.to(tl.int64) * N + start + rank
            fanout = tl.load(degree + source, mask=live, other=0)
            tl.store(active + slot, source, mask=live)
            tl.store(active_degree + slot, fanout, mask=live)


def prepare_values(
    t_crow, t_col, t_perm, values, batch_size, *, cache: KernelCache, route_key=None
):
    """Prepare source-major weights once per trajectory, outside its timestep
    loop."""
    if values.dtype != torch.float32 or values.ndim != 1 or not values.is_cuda:
        return None, 0
    layout = kernels_triton_push._layout(cache, t_crow, t_col, t_perm)
    if (
        layout is None
        or batch_size > kernels_triton._MAX_GRID_Y
        or batch_size * (t_crow.numel() - 1) > 2**31 - 1
    ):
        return None, 0
    # Sparse backward still owns the canonical differentiable values. This is
    # only a read layout; no second autograd path or independent weight exists.
    if layout.source_values is None:
        layout.source_values = torch.empty_like(values)
        layout.source_value_epoch = kernels_triton._next_epoch()
    # Stable addresses let already-captured graphs observe optimizer updates.
    # The shared layout cache owns lifetime and topology invalidation, just as
    # for the packed fanout's index copies.
    layout.source_values.copy_(values.detach().index_select(0, t_perm.long()))
    return layout.source_values, layout.source_value_epoch


def spike_push_list(
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    values: Tensor,
    x: Tensor,
    n_out: int,
    *,
    task_values: Tensor | None = None,
    require_task: bool = False,
    cache: KernelCache,
) -> Tensor:
    """Propagate device-compacted sources with the shared packed SpMSpV
    kernel."""
    n_batch, n_in = x.shape
    output = x.new_zeros((n_batch, n_out))
    capacity = x.numel()
    if not capacity or not values.numel():
        return output
    layout = kernels_triton_push._layout(cache, t_crow, t_col, t_perm)
    if layout is None or capacity > 2**31 - 1:
        raise RuntimeError("Unsupported spike-list layout; replan the connection")
    active = torch.empty(capacity, device=x.device, dtype=torch.int64)
    active_degree = torch.zeros(capacity, device=x.device, dtype=torch.int64)
    counts = torch.zeros(n_batch, device=x.device, dtype=torch.int32)
    x = x.contiguous()
    _compact_spikes_kernel[(triton.cdiv(n_in, 256), n_batch)](
        x,
        layout.degree,
        active,
        active_degree,
        counts,
        N=n_in,
        BLOCK=256,
    )
    cumulative = torch.cumsum(active_degree, 0)
    # A sample owns N slots, including zero-degree padding. The fanout's
    # existing sample lookup therefore works unchanged for compacted lists.
    pointers = torch.arange(n_batch + 1, device=x.device, dtype=torch.int64) * n_in
    programs = kernels_triton_push._MAX_PROGRAMS
    kernels_triton._launch(
        cache,
        "push_packed",
        kernels_triton_push._kernels(cache)[0],
        x.device.index,
        (programs, 1),
        (
            layout.crow,
            layout.col,
            layout.perm,
            values if task_values is None else task_values,
            x,
            active,
            cumulative,
            pointers,
            output,
            capacity,
            n_in,
            n_out,
            n_batch,
            capacity.bit_length(),
            (n_batch - 1).bit_length(),
            programs,
        ),
        kernels_triton_push._PACKED_CONSTS,
        (kernels_triton_push._BLOCK, task_values is None),
        kernels_triton_push._BLOCK_WARPS,
    )
    return output


def register(registry):
    """Register spike-list preparation and fanout under the Triton backend."""

    def prepare(t_crow, t_col, t_perm, values, batch_size, *, route_key=None):
        return prepare_values(
            t_crow,
            t_col,
            t_perm,
            values,
            batch_size,
            cache=registry.kernels,
            route_key=route_key,
        )

    def push(*args, **kwargs):
        return spike_push_list(*args, **kwargs, cache=registry.kernels)

    for name, implementation in (
        ("spike_push_list", push),
        ("prepare_spike_list_values", prepare),
    ):
        registry.register(
            name,
            "triton",
            implementation,
            device="cuda",
            priority=20,
            available=kernels_triton_push.is_available,
        )
