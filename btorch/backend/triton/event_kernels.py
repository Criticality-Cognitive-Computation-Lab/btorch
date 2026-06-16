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
