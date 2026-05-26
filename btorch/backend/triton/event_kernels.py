import triton
import triton.language as tl


@triton.jit
def pre_span_forward_kernel(
    spike_ptr,
    row_length_ptr,
    ind_ptr,
    weight_ptr,
    out_ptr,
    stride_spike_b,
    stride_spike_n,
    stride_ind_row,
    stride_ind_col,
    stride_weight_row,
    stride_weight_col,
    stride_out_b,
    stride_out_n,
    n_pre,
    row_stride,
    IS_BOOL_FLOAT: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
):
    """Accumulate event current using rows grouped by presynaptic neuron."""
    pid_batch = tl.program_id(0)
    pid_pre = tl.program_id(1)

    if pid_pre >= n_pre:
        return

    spike_ptrs = spike_ptr + pid_batch * stride_spike_b + pid_pre * stride_spike_n
    spike_val = tl.load(spike_ptrs)
    row_len = tl.load(row_length_ptr + pid_pre)

    row_base_ind = ind_ptr + pid_pre * stride_ind_row
    row_base_weight = weight_ptr + pid_pre * stride_weight_row
    out_batch_base = out_ptr + pid_batch * stride_out_b

    for offset in range(ROW_STRIDE):
        if offset < row_stride and offset < row_len:
            post = tl.load(row_base_ind + offset * stride_ind_col)
            weight = tl.load(row_base_weight + offset * stride_weight_col)
            out_ptrs = out_batch_base + post * stride_out_n
            if IS_BOOL_FLOAT:
                contrib = tl.where(spike_val > 0.5, weight, 0.0)
            else:
                contrib = weight * spike_val
            tl.atomic_add(out_ptrs, contrib)


@triton.jit
def post_span_forward_kernel(
    spike_ptr,
    row_length_ptr,
    ind_ptr,
    weight_ptr,
    out_ptr,
    stride_spike_b,
    stride_spike_n,
    stride_ind_row,
    stride_ind_col,
    stride_weight_row,
    stride_weight_col,
    stride_out_b,
    stride_out_n,
    n_pre,
    row_stride,
    IS_BOOL_FLOAT: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
):
    """Accumulate event current using GeNN-style post-span traversal.

    The sparse storage is still presynaptic-row based. The parallelism changes
    from "one program per presynaptic row" to "one program per slot within each
    row", mirroring the reference PostSpan sparse update strategy.
    """
    pid_batch = tl.program_id(0)
    pid_slot_block = tl.program_id(1)

    slot_offsets = pid_slot_block * BLOCK_SLOT + tl.arange(0, BLOCK_SLOT)
    slot_mask = slot_offsets < row_stride
    out_batch_base = out_ptr + pid_batch * stride_out_b

    for pre in range(0, n_pre):
        spike_ptrs = spike_ptr + pid_batch * stride_spike_b + pre * stride_spike_n
        spike_val = tl.load(spike_ptrs)
        row_len = tl.load(row_length_ptr + pre)

        row_ind_ptrs = ind_ptr + pre * stride_ind_row + slot_offsets * stride_ind_col
        row_weight_ptrs = (
            weight_ptr + pre * stride_weight_row + slot_offsets * stride_weight_col
        )

        valid_mask = slot_mask & (slot_offsets < row_len)
        post = tl.load(row_ind_ptrs, mask=valid_mask, other=0)
        weight = tl.load(row_weight_ptrs, mask=valid_mask, other=0.0)
        if IS_BOOL_FLOAT:
            contrib = tl.where(spike_val > 0.5, weight, 0.0)
        else:
            contrib = weight * spike_val

        out_ptrs = out_batch_base + post * stride_out_n
        tl.atomic_add(out_ptrs, contrib, mask=valid_mask)
