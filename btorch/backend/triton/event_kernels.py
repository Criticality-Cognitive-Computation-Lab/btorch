import triton
import triton.language as tl


@triton.jit
def dense_spike_to_list_kernel(
    spike_ptr,
    spike_count_ptr,
    spike_ind_ptr,
    stride_spike_b,
    stride_spike_n,
    stride_spike_ind_b,
    stride_spike_ind_n,
    n_pre,
    THRESHOLD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compact dense spike flags into a per-batch spike index list."""
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)

    offsets = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pre
    spike_vals = tl.load(
        spike_ptr + pid_batch * stride_spike_b + offsets * stride_spike_n,
        mask=mask,
        other=0.0,
    )
    fired = (spike_vals > THRESHOLD) & mask
    fired_i = fired.to(tl.int32)
    block_count = tl.sum(fired_i, axis=0)
    local_rank = tl.cumsum(fired_i, axis=0) - 1
    base = tl.atomic_add(spike_count_ptr + pid_batch, block_count)

    out_offsets = base + local_rank
    tl.store(
        spike_ind_ptr
        + pid_batch * stride_spike_ind_b
        + out_offsets * stride_spike_ind_n,
        offsets,
        mask=fired,
    )


@triton.jit
def spike_list_to_bucketed_work_kernel(
    spike_count_ptr,
    spike_ind_ptr,
    pre_segment_start_ptr,
    pre_segment_count_ptr,
    segment_bucket_ptr,
    segment_syn_start_ptr,
    segment_syn_len_ptr,
    bucket_count_ptr,
    bucket_work_ptr,
    stride_spike_ind_b,
    stride_spike_ind_n,
    stride_bucket_count_bucket,
    stride_bucket_count_b,
    stride_bucket_work_b,
    stride_bucket_work_row,
    stride_bucket_work_field,
    max_spikes,
    TARGET_BUCKET: tl.constexpr,
    MAX_SEGMENTS_PER_PRE: tl.constexpr,
    BLOCK_SPIKE: tl.constexpr,
):
    """Expand spiking presynaptic ids into one bucket-specific work-list."""
    pid_batch = tl.program_id(0)
    pid_spike_block = tl.program_id(1)

    spike_offsets = pid_spike_block * BLOCK_SPIKE + tl.arange(0, BLOCK_SPIKE)
    spike_count = tl.load(spike_count_ptr + pid_batch)
    spike_mask = (spike_offsets < spike_count) & (spike_offsets < max_spikes)

    pre = tl.load(
        spike_ind_ptr
        + pid_batch * stride_spike_ind_b
        + spike_offsets * stride_spike_ind_n,
        mask=spike_mask,
        other=0,
    )
    segment_start = tl.load(pre_segment_start_ptr + pre, mask=spike_mask, other=0)
    segment_count = tl.load(pre_segment_count_ptr + pre, mask=spike_mask, other=0)

    for segment_offset in tl.static_range(0, MAX_SEGMENTS_PER_PRE):
        has_segment = spike_mask & (segment_offset < segment_count)
        segment_idx = segment_start + segment_offset
        bucket = tl.load(segment_bucket_ptr + segment_idx, mask=has_segment, other=-1)
        keep = has_segment & (bucket == TARGET_BUCKET)
        keep_i = keep.to(tl.int32)
        block_count = tl.sum(keep_i, axis=0)
        local_rank = tl.cumsum(keep_i, axis=0) - 1
        base = tl.atomic_add(
            bucket_count_ptr
            + TARGET_BUCKET * stride_bucket_count_bucket
            + pid_batch * stride_bucket_count_b,
            block_count,
        )
        out_offsets = base + local_rank

        syn_start = tl.load(segment_syn_start_ptr + segment_idx, mask=keep, other=0)
        syn_len = tl.load(segment_syn_len_ptr + segment_idx, mask=keep, other=0)
        row_base = (
            bucket_work_ptr
            + pid_batch * stride_bucket_work_b
            + out_offsets * stride_bucket_work_row
        )
        tl.store(row_base, pre, mask=keep)
        tl.store(
            row_base + stride_bucket_work_field,
            syn_start,
            mask=keep,
        )
        tl.store(
            row_base + 2 * stride_bucket_work_field,
            syn_len,
            mask=keep,
        )


@triton.jit
def bucketed_span_forward_kernel(
    bucket_count_ptr,
    bucket_work_ptr,
    ind_ptr,
    weight_ptr,
    out_ptr,
    stride_bucket_count_bucket,
    stride_bucket_count_b,
    stride_bucket_work_b,
    stride_bucket_work_row,
    stride_bucket_work_field,
    stride_ind_row,
    stride_ind_col,
    stride_weight_row,
    stride_weight_col,
    stride_out_b,
    stride_out_n,
    row_stride,
    TARGET_BUCKET: tl.constexpr,
    BLOCK_EDGE: tl.constexpr,
):
    """Apply one bucketed ``pre_id, syn_start, syn_len`` work row."""
    pid_batch = tl.program_id(0)
    pid_work = tl.program_id(1)

    work_count = tl.load(
        bucket_count_ptr
        + TARGET_BUCKET * stride_bucket_count_bucket
        + pid_batch * stride_bucket_count_b
    )
    active = pid_work < work_count
    row_base = (
        bucket_work_ptr
        + pid_batch * stride_bucket_work_b
        + pid_work * stride_bucket_work_row
    )
    pre = tl.load(row_base, mask=active, other=0)
    syn_start = tl.load(
        row_base + stride_bucket_work_field,
        mask=active,
        other=0,
    )
    syn_len = tl.load(
        row_base + 2 * stride_bucket_work_field,
        mask=active,
        other=0,
    )

    edge_offsets = tl.arange(0, BLOCK_EDGE)
    edge = syn_start + edge_offsets
    valid = active & (edge_offsets < syn_len) & (edge < row_stride)

    post = tl.load(
        ind_ptr + pre * stride_ind_row + edge * stride_ind_col,
        mask=valid,
        other=0,
    )
    weight = tl.load(
        weight_ptr + pre * stride_weight_row + edge * stride_weight_col,
        mask=valid,
        other=0.0,
    )
    out_ptrs = out_ptr + pid_batch * stride_out_b + post * stride_out_n
    tl.atomic_add(out_ptrs, weight, mask=valid)


@triton.jit
def pre_span_spike_list_forward_kernel(
    spike_count_ptr,
    spike_ind_ptr,
    row_length_ptr,
    ind_ptr,
    weight_ptr,
    out_ptr,
    stride_spike_ind_b,
    stride_spike_ind_n,
    stride_ind_row,
    stride_ind_col,
    stride_weight_row,
    stride_weight_col,
    stride_out_b,
    stride_out_n,
    row_stride,
    max_spikes,
    BLOCK_SPIKE: tl.constexpr,
    BLOCK_EDGE: tl.constexpr,
):
    """Presynaptic fan-out over a compact spike list."""
    pid_batch = tl.program_id(0)
    pid_spike_block = tl.program_id(1)
    pid_edge_block = tl.program_id(2)

    spike_offsets = pid_spike_block * BLOCK_SPIKE + tl.arange(0, BLOCK_SPIKE)
    edge_offsets = pid_edge_block * BLOCK_EDGE + tl.arange(0, BLOCK_EDGE)
    spike_count = tl.load(spike_count_ptr + pid_batch)
    spike_mask = (spike_offsets < spike_count) & (spike_offsets < max_spikes)

    pre = tl.load(
        spike_ind_ptr
        + pid_batch * stride_spike_ind_b
        + spike_offsets * stride_spike_ind_n,
        mask=spike_mask,
        other=0,
    )
    row_len = tl.load(row_length_ptr + pre, mask=spike_mask, other=0)

    pre_2d = pre[:, None]
    edge_2d = edge_offsets[None, :]
    valid = spike_mask[:, None] & (edge_2d < row_stride) & (edge_2d < row_len[:, None])

    post = tl.load(
        ind_ptr + pre_2d * stride_ind_row + edge_2d * stride_ind_col,
        mask=valid,
        other=0,
    )
    weight = tl.load(
        weight_ptr + pre_2d * stride_weight_row + edge_2d * stride_weight_col,
        mask=valid,
        other=0.0,
    )
    out_ptrs = out_ptr + pid_batch * stride_out_b + post * stride_out_n
    tl.atomic_add(out_ptrs, weight, mask=valid)


@triton.jit
def post_span_spike_list_forward_kernel(
    spike_count_ptr,
    spike_ind_ptr,
    row_length_ptr,
    ind_ptr,
    weight_ptr,
    out_ptr,
    stride_spike_ind_b,
    stride_spike_ind_n,
    stride_ind_row,
    stride_ind_col,
    stride_weight_row,
    stride_weight_col,
    stride_out_b,
    stride_out_n,
    row_stride,
    max_spikes,
    BLOCK_SPIKE: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
):
    """GeNN-style post-span over spike tiles and row slots."""
    pid_batch = tl.program_id(0)
    pid_spike_block = tl.program_id(1)
    pid_slot_block = tl.program_id(2)

    spike_offsets = pid_spike_block * BLOCK_SPIKE + tl.arange(0, BLOCK_SPIKE)
    slot_offsets = pid_slot_block * BLOCK_SLOT + tl.arange(0, BLOCK_SLOT)
    spike_count = tl.load(spike_count_ptr + pid_batch)
    spike_mask = (spike_offsets < spike_count) & (spike_offsets < max_spikes)

    # This vector plays the role of GeNN's shared-memory spike tile. Triton
    # keeps it close to the program rather than exposing shared memory directly.
    pre = tl.load(
        spike_ind_ptr
        + pid_batch * stride_spike_ind_b
        + spike_offsets * stride_spike_ind_n,
        mask=spike_mask,
        other=0,
    )
    row_len = tl.load(row_length_ptr + pre, mask=spike_mask, other=0)

    pre_2d = pre[:, None]
    slot_2d = slot_offsets[None, :]
    valid = spike_mask[:, None] & (slot_2d < row_stride) & (slot_2d < row_len[:, None])

    post = tl.load(
        ind_ptr + pre_2d * stride_ind_row + slot_2d * stride_ind_col,
        mask=valid,
        other=0,
    )
    weight = tl.load(
        weight_ptr + pre_2d * stride_weight_row + slot_2d * stride_weight_col,
        mask=valid,
        other=0.0,
    )
    out_ptrs = out_ptr + pid_batch * stride_out_b + post * stride_out_n
    tl.atomic_add(out_ptrs, weight, mask=valid)
