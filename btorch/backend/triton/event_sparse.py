from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import triton

from .event_kernels import (
    bucketed_span_forward_kernel,
    dense_spike_to_list_kernel,
    post_span_spike_list_forward_kernel,
    pre_span_spike_list_forward_kernel,
    spike_list_to_bucketed_work_kernel,
)


DEFAULT_BUCKET_SIZES = (32, 64, 128, 256, 512, 1024)
TAIL_MERGE_THRESHOLD = 0.25


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


@dataclass(frozen=True)
class EventBucketPlan:
    """Static mapping from presynaptic rows to bucketed edge segments."""

    bucket_sizes: tuple[int, ...]
    bucket_capacity: tuple[int, ...]
    pre_segment_start: torch.Tensor
    pre_segment_count: torch.Tensor
    segment_bucket: torch.Tensor
    segment_syn_start: torch.Tensor
    segment_syn_len: torch.Tensor
    max_segments_per_pre: int

    def to(self, device: torch.device | str) -> EventBucketPlan:
        """Move plan tensors to ``device`` while keeping Python metadata."""
        return EventBucketPlan(
            bucket_sizes=self.bucket_sizes,
            bucket_capacity=self.bucket_capacity,
            pre_segment_start=self.pre_segment_start.to(device=device),
            pre_segment_count=self.pre_segment_count.to(device=device),
            segment_bucket=self.segment_bucket.to(device=device),
            segment_syn_start=self.segment_syn_start.to(device=device),
            segment_syn_len=self.segment_syn_len.to(device=device),
            max_segments_per_pre=self.max_segments_per_pre,
        )


@dataclass(frozen=True)
class BucketedSpikeList:
    """Runtime work-lists grouped by bucket size.

    Each work row stores ``pre_id, syn_start, syn_len``.
    """

    bucket_counts: torch.Tensor
    bucket_work: tuple[torch.Tensor, ...]
    bucket_sizes: tuple[int, ...]


def _validate_bucket_sizes(bucket_sizes: tuple[int, ...]) -> None:
    if not bucket_sizes:
        raise ValueError("bucket_sizes must not be empty.")
    previous = 0
    for size in bucket_sizes:
        if size <= 0:
            raise ValueError("bucket sizes must be positive.")
        if size <= previous:
            raise ValueError("bucket_sizes must be strictly increasing.")
        previous = size


def _bucket_index_for_len(length: int, bucket_sizes: tuple[int, ...]) -> int:
    for bucket_idx, bucket_size in enumerate(bucket_sizes):
        if length <= bucket_size:
            return bucket_idx
    return len(bucket_sizes) - 1


def _segments_for_row(
    fanout: int,
    bucket_sizes: tuple[int, ...],
    tail_merge_threshold: float,
) -> list[tuple[int, int, int]]:
    if fanout <= 0:
        return []

    if fanout <= bucket_sizes[-1]:
        bucket_idx = _bucket_index_for_len(fanout, bucket_sizes)
        return [(bucket_idx, 0, fanout)]

    base_bucket_idx = 0
    for bucket_idx, bucket_size in enumerate(bucket_sizes):
        if bucket_size <= fanout:
            base_bucket_idx = bucket_idx
        else:
            break
    base_bucket_size = bucket_sizes[base_bucket_idx]

    if fanout <= base_bucket_size:
        return [(base_bucket_idx, 0, fanout)]

    full_segments = fanout // base_bucket_size
    remainder = fanout % base_bucket_size
    if (
        full_segments == 1
        and remainder > 0
        and remainder < base_bucket_size * tail_merge_threshold
        and base_bucket_idx + 1 < len(bucket_sizes)
        and fanout <= bucket_sizes[base_bucket_idx + 1]
    ):
        return [(base_bucket_idx + 1, 0, fanout)]

    segments = [
        (base_bucket_idx, segment_idx * base_bucket_size, base_bucket_size)
        for segment_idx in range(full_segments)
    ]
    if remainder > 0:
        tail_bucket_idx = _bucket_index_for_len(remainder, bucket_sizes)
        segments.append((tail_bucket_idx, full_segments * base_bucket_size, remainder))
    return segments


def build_event_bucket_plan(
    row_length: torch.Tensor,
    *,
    bucket_sizes: tuple[int, ...] = DEFAULT_BUCKET_SIZES,
    tail_merge_threshold: float = TAIL_MERGE_THRESHOLD,
) -> EventBucketPlan:
    """Build static bucketed segments from presynaptic row fanouts."""
    if row_length.ndim != 1:
        raise ValueError("row_length must be 1D.")
    _validate_bucket_sizes(bucket_sizes)
    if tail_merge_threshold <= 0.0 or tail_merge_threshold >= 1.0:
        raise ValueError("tail_merge_threshold must be in the range (0, 1).")

    row_lengths = row_length.detach().cpu().to(torch.int64).tolist()
    pre_segment_start: list[int] = []
    pre_segment_count: list[int] = []
    segment_bucket: list[int] = []
    segment_syn_start: list[int] = []
    segment_syn_len: list[int] = []
    bucket_capacity = [0 for _ in bucket_sizes]
    max_segments_per_pre = 0

    for fanout in row_lengths:
        fanout_i = int(fanout)
        if fanout_i < 0:
            raise ValueError("row_length entries must be non-negative.")
        segments = _segments_for_row(fanout_i, bucket_sizes, tail_merge_threshold)
        pre_segment_start.append(len(segment_bucket))
        pre_segment_count.append(len(segments))
        max_segments_per_pre = max(max_segments_per_pre, len(segments))
        for bucket_idx, syn_start, syn_len in segments:
            segment_bucket.append(bucket_idx)
            segment_syn_start.append(syn_start)
            segment_syn_len.append(syn_len)
            bucket_capacity[bucket_idx] += 1

    device = row_length.device
    return EventBucketPlan(
        bucket_sizes=bucket_sizes,
        bucket_capacity=tuple(bucket_capacity),
        pre_segment_start=torch.tensor(
            pre_segment_start, device=device, dtype=torch.int64
        ),
        pre_segment_count=torch.tensor(
            pre_segment_count, device=device, dtype=torch.int64
        ),
        segment_bucket=torch.tensor(segment_bucket, device=device, dtype=torch.int64),
        segment_syn_start=torch.tensor(
            segment_syn_start, device=device, dtype=torch.int64
        ),
        segment_syn_len=torch.tensor(segment_syn_len, device=device, dtype=torch.int64),
        max_segments_per_pre=max_segments_per_pre,
    )


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


def _ensure_plan_device(
    bucket_plan: EventBucketPlan,
    device: torch.device,
) -> EventBucketPlan:
    if bucket_plan.pre_segment_start.device == device:
        return bucket_plan
    return bucket_plan.to(device)


def bucketed_spike_list_from_spike_list(
    spike_count: torch.Tensor,
    spike_ind: torch.Tensor,
    bucket_plan: EventBucketPlan,
    *,
    block_spike: int = 128,
) -> BucketedSpikeList:
    """Expand compact spiking pre ids into bucketed segment work-lists."""
    if not spike_count.is_cuda or not spike_ind.is_cuda:
        raise ValueError("Bucketed spike-list expansion requires CUDA tensors.")
    if spike_count.ndim != 1:
        raise ValueError("spike_count must have shape (batch_size,).")
    if spike_ind.ndim != 2:
        raise ValueError("spike_ind must have shape (batch_size, max_spikes).")
    if spike_ind.shape[0] != spike_count.shape[0]:
        raise ValueError("spike_ind and spike_count batch dimensions must match.")
    if block_spike <= 0:
        raise ValueError("block_spike must be positive.")

    spike_count = spike_count.contiguous()
    spike_ind = spike_ind.contiguous()
    bucket_plan = _ensure_plan_device(bucket_plan, spike_count.device)
    batch_size, max_spikes = spike_ind.shape
    num_buckets = len(bucket_plan.bucket_sizes)
    bucket_counts = torch.zeros(
        (num_buckets, batch_size), device=spike_count.device, dtype=torch.int32
    )
    bucket_work = tuple(
        torch.empty(
            (batch_size, capacity, 3),
            device=spike_count.device,
            dtype=torch.int64,
        )
        for capacity in bucket_plan.bucket_capacity
    )

    if max_spikes == 0 or bucket_plan.max_segments_per_pre == 0:
        return BucketedSpikeList(bucket_counts, bucket_work, bucket_plan.bucket_sizes)

    grid = (batch_size, triton.cdiv(max_spikes, block_spike))
    for bucket_idx, work in enumerate(bucket_work):
        if work.shape[1] == 0:
            continue
        spike_list_to_bucketed_work_kernel[grid](
            spike_count,
            spike_ind,
            bucket_plan.pre_segment_start,
            bucket_plan.pre_segment_count,
            bucket_plan.segment_bucket,
            bucket_plan.segment_syn_start,
            bucket_plan.segment_syn_len,
            bucket_counts,
            work,
            spike_ind.stride(0),
            spike_ind.stride(1),
            bucket_counts.stride(0),
            bucket_counts.stride(1),
            work.stride(0),
            work.stride(1),
            work.stride(2),
            max_spikes,
            TARGET_BUCKET=bucket_idx,
            MAX_SEGMENTS_PER_PRE=bucket_plan.max_segments_per_pre,
            BLOCK_SPIKE=block_spike,
            num_warps=4,
        )
    return BucketedSpikeList(bucket_counts, bucket_work, bucket_plan.bucket_sizes)


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


def _prepare_out(
    batch_size: int,
    size_m: int,
    weight: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if out is None:
        return torch.zeros(
            (batch_size, size_m), device=weight.device, dtype=weight.dtype
        )
    if out.shape != (batch_size, size_m):
        raise ValueError("out must have shape (batch_size, size_m).")
    if out.device != weight.device or out.dtype != weight.dtype:
        raise ValueError("out must match weight device and dtype.")
    out = out.contiguous()
    out.zero_()
    return out


def _num_warps_for_bucket(bucket_size: int) -> int:
    if bucket_size >= 512:
        return 8
    if bucket_size >= 128:
        return 4
    return 1


def _bucketed_span_spmm_from_spike_list(
    spike_count: torch.Tensor,
    spike_ind: torch.Tensor,
    row_length: torch.Tensor,
    ind: torch.Tensor,
    weight: torch.Tensor,
    bucket_plan: EventBucketPlan,
    *,
    size_m: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    spike_count, spike_ind, row_length, ind, weight = _validate_spike_list_inputs(
        spike_count, spike_ind, row_length, ind, weight
    )
    batch_size = spike_count.shape[0]
    row_stride = ind.shape[1]
    out = _prepare_out(batch_size, size_m, weight, out)
    bucket_plan = _ensure_plan_device(bucket_plan, spike_count.device)
    bucketed = bucketed_spike_list_from_spike_list(
        spike_count,
        spike_ind,
        bucket_plan,
    )

    for bucket_idx, (bucket_size, work) in enumerate(
        zip(bucketed.bucket_sizes, bucketed.bucket_work, strict=True)
    ):
        capacity = work.shape[1]
        if capacity == 0:
            continue
        grid = (batch_size, capacity)
        bucketed_span_forward_kernel[grid](
            bucketed.bucket_counts,
            work,
            ind,
            weight,
            out,
            bucketed.bucket_counts.stride(0),
            bucketed.bucket_counts.stride(1),
            work.stride(0),
            work.stride(1),
            work.stride(2),
            ind.stride(0),
            ind.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            row_stride,
            TARGET_BUCKET=bucket_idx,
            BLOCK_EDGE=bucket_size,
            num_warps=_num_warps_for_bucket(bucket_size),
        )
    return out


def pre_span_bucketed_spmm_from_spike_list(
    spike_count: torch.Tensor,
    spike_ind: torch.Tensor,
    row_length: torch.Tensor,
    ind: torch.Tensor,
    weight: torch.Tensor,
    bucket_plan: EventBucketPlan,
    *,
    size_m: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply bucketed pre-span propagation from a compact spike list."""
    return _bucketed_span_spmm_from_spike_list(
        spike_count,
        spike_ind,
        row_length,
        ind,
        weight,
        bucket_plan,
        size_m=size_m,
        out=out,
    )


def post_span_bucketed_spmm_from_spike_list(
    spike_count: torch.Tensor,
    spike_ind: torch.Tensor,
    row_length: torch.Tensor,
    ind: torch.Tensor,
    weight: torch.Tensor,
    bucket_plan: EventBucketPlan,
    *,
    size_m: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply bucketed post-span propagation from a compact spike list."""
    return _bucketed_span_spmm_from_spike_list(
        spike_count,
        spike_ind,
        row_length,
        ind,
        weight,
        bucket_plan,
        size_m=size_m,
        out=out,
    )


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
    out = _prepare_out(batch_size, size_m, weight, out)

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
    out = _prepare_out(batch_size, size_m, weight, out)

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
