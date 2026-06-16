from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import triton

from .event_kernels import (
    dense_spike_to_list_kernel,
    post_span_spike_list_forward_kernel,
    pre_span_spike_list_forward_kernel,
)


@dataclass(frozen=True)
class SpikeList:
    """Compact per-batch spike list.

    ``count[b]`` stores the valid number of entries in ``ind[b]``.
    ``ind[b, :count[b]]`` stores the presynaptic neuron ids that spiked.
    """

    count: torch.Tensor
    ind: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.count
        yield self.ind


def dense_spike_to_spike_list(
    spike: torch.Tensor,
    *,
    threshold: float = 0.5,
    max_spikes: int | None = None,
) -> SpikeList:
    """Compact dense spike flags into ``(spike_count, spike_ind)``.

    Args:
        spike: Dense spike tensor of shape ``(batch_size, n_pre)``.
        threshold: Values greater than this are treated as spikes.
        max_spikes: Capacity of each batch row in ``spike_ind``. Defaults to
            ``n_pre``, which cannot overflow for binary spikes.
    """
    if not spike.is_cuda:
        raise ValueError("dense_spike_to_spike_list requires CUDA tensors.")
    if spike.ndim != 2:
        raise ValueError("spike must have shape (batch_size, n_pre).")

    spike = spike.contiguous()
    batch_size, n_pre = spike.shape
    max_spikes = n_pre if max_spikes is None else max_spikes
    if max_spikes <= 0 or max_spikes > n_pre:
        raise ValueError("max_spikes must be in the range [1, n_pre].")

    spike_count = torch.zeros((batch_size,), device=spike.device, dtype=torch.int32)
    spike_ind = torch.empty(
        (batch_size, max_spikes), device=spike.device, dtype=torch.int64
    )

    block_size = 256
    grid = (batch_size, triton.cdiv(n_pre, block_size))
    dense_spike_to_list_kernel[grid](
        spike,
        spike_count,
        spike_ind,
        spike.stride(0),
        spike.stride(1),
        spike_ind.stride(0),
        spike_ind.stride(1),
        n_pre,
        THRESHOLD=threshold,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return SpikeList(spike_count, spike_ind)


def _validate_spike_list_inputs(
    spike_count: torch.Tensor,
    spike_ind: torch.Tensor,
    row_length: torch.Tensor,
    ind: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not spike_count.is_cuda or not spike_ind.is_cuda:
        raise ValueError("Spike-list Triton kernels require CUDA tensors.")
    if spike_count.ndim != 1:
        raise ValueError("spike_count must have shape (batch_size,).")
    if spike_ind.ndim != 2:
        raise ValueError("spike_ind must have shape (batch_size, max_spikes).")
    if spike_ind.shape[0] != spike_count.shape[0]:
        raise ValueError("spike_ind and spike_count batch dimensions must match.")
    if row_length.ndim != 1:
        raise ValueError("row_length must be 1D.")
    if ind.ndim != 2 or weight.ndim != 2:
        raise ValueError("ind and weight must be 2D tensors.")
    if ind.shape != weight.shape:
        raise ValueError("ind and weight must have identical shapes.")
    if row_length.shape[0] != ind.shape[0]:
        raise ValueError("row_length length must match the number of rows in ind.")
    if row_length.device != spike_count.device:
        row_length = row_length.to(device=spike_count.device)
    if ind.device != spike_count.device:
        ind = ind.to(device=spike_count.device)
    if weight.device != spike_count.device:
        weight = weight.to(device=spike_count.device)
    if spike_ind.device != spike_count.device:
        spike_ind = spike_ind.to(device=spike_count.device)
    spike_count = spike_count.contiguous()
    spike_ind = spike_ind.contiguous()
    row_length = row_length.contiguous()
    ind = ind.contiguous()
    weight = weight.contiguous()
    return spike_count, spike_ind, row_length, ind, weight


def pre_span_spmm_from_spike_list(
    spike_count: torch.Tensor,
    spike_ind: torch.Tensor,
    row_length: torch.Tensor,
    ind: torch.Tensor,
    weight: torch.Tensor,
    *,
    size_m: int,
    out: torch.Tensor | None = None,
    block_spike: int = 1,
    block_edge: int = 32,
) -> torch.Tensor:
    """Apply pre-span propagation from a compact spike list."""
    spike_count, spike_ind, row_length, ind, weight = _validate_spike_list_inputs(
        spike_count, spike_ind, row_length, ind, weight
    )
    batch_size, max_spikes = spike_ind.shape
    row_stride = ind.shape[1]

    if out is None:
        out = torch.zeros(
            (batch_size, size_m), device=weight.device, dtype=weight.dtype
        )
    else:
        if out.shape != (batch_size, size_m):
            raise ValueError("out must have shape (batch_size, size_m).")
        if out.device != weight.device or out.dtype != weight.dtype:
            raise ValueError("out must match weight device and dtype.")
        out = out.contiguous()
        out.zero_()

    grid = (
        batch_size,
        triton.cdiv(max_spikes, block_spike),
        triton.cdiv(row_stride, block_edge),
    )
    pre_span_spike_list_forward_kernel[grid](
        spike_count,
        spike_ind,
        row_length,
        ind,
        weight,
        out,
        spike_ind.stride(0),
        spike_ind.stride(1),
        ind.stride(0),
        ind.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        row_stride,
        max_spikes,
        BLOCK_SPIKE=block_spike,
        BLOCK_EDGE=block_edge,
        num_warps=1,
    )
    return out


def post_span_spmm_from_spike_list(
    spike_count: torch.Tensor,
    spike_ind: torch.Tensor,
    row_length: torch.Tensor,
    ind: torch.Tensor,
    weight: torch.Tensor,
    *,
    size_m: int,
    out: torch.Tensor | None = None,
    block_spike: int = 32,
    block_slot: int = 32,
) -> torch.Tensor:
    """Apply GeNN-style post-span propagation from a compact spike list."""
    spike_count, spike_ind, row_length, ind, weight = _validate_spike_list_inputs(
        spike_count, spike_ind, row_length, ind, weight
    )
    batch_size, max_spikes = spike_ind.shape
    row_stride = ind.shape[1]

    if out is None:
        out = torch.zeros(
            (batch_size, size_m), device=weight.device, dtype=weight.dtype
        )
    else:
        if out.shape != (batch_size, size_m):
            raise ValueError("out must have shape (batch_size, size_m).")
        if out.device != weight.device or out.dtype != weight.dtype:
            raise ValueError("out must match weight device and dtype.")
        out = out.contiguous()
        out.zero_()

    grid = (
        batch_size,
        triton.cdiv(max_spikes, block_spike),
        triton.cdiv(row_stride, block_slot),
    )
    post_span_spike_list_forward_kernel[grid](
        spike_count,
        spike_ind,
        row_length,
        ind,
        weight,
        out,
        spike_ind.stride(0),
        spike_ind.stride(1),
        ind.stride(0),
        ind.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        row_stride,
        max_spikes,
        BLOCK_SPIKE=block_spike,
        BLOCK_SLOT=block_slot,
        num_warps=1,
    )
    return out
