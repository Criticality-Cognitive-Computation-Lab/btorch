import triton
import triton.language as tl


@triton.jit
def dense_event_to_list_kernel(
    values_ptr,
    count_ptr,
    indices_ptr,
    stride_values_b,
    stride_values_n,
    stride_indices_b,
    stride_indices_n,
    n_pre,
    THRESHOLD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pre
    values = tl.load(
        values_ptr + batch * stride_values_b + offsets * stride_values_n,
        mask=mask,
        other=0.0,
    )
    active = (values > THRESHOLD) & mask
    active_i = active.to(tl.int32)
    block_count = tl.sum(active_i, axis=0)
    rank = tl.cumsum(active_i, axis=0) - 1
    base = tl.atomic_add(count_ptr + batch, block_count)
    tl.store(
        indices_ptr + batch * stride_indices_b + (base + rank) * stride_indices_n,
        offsets,
        mask=active,
    )


@triton.jit
def pre_span_event_kernel(
    event_count_ptr,
    event_indices_ptr,
    row_length_ptr,
    row_offset_ptr,
    destination_ptr,
    values_ptr,
    out_ptr,
    stride_event_b,
    stride_event_n,
    stride_destination_row,
    stride_destination_slot,
    stride_out_b,
    stride_out_n,
    row_stride,
    max_events,
    BLOCK_EVENT: tl.constexpr,
    BLOCK_EDGE: tl.constexpr,
):
    batch = tl.program_id(0)
    event_block = tl.program_id(1)
    edge_block = tl.program_id(2)
    event_slot = event_block * BLOCK_EVENT + tl.arange(0, BLOCK_EVENT)
    edge_slot = edge_block * BLOCK_EDGE + tl.arange(0, BLOCK_EDGE)
    count = tl.load(event_count_ptr + batch)
    event_mask = (event_slot < count) & (event_slot < max_events)
    pre = tl.load(
        event_indices_ptr + batch * stride_event_b + event_slot * stride_event_n,
        mask=event_mask,
        other=0,
    )
    row_length = tl.load(row_length_ptr + pre, mask=event_mask, other=0)
    row_offset = tl.load(row_offset_ptr + pre, mask=event_mask, other=0)
    pre_2d = pre[:, None]
    edge_2d = edge_slot[None, :]
    valid = event_mask[:, None] & (edge_2d < row_stride)
    valid &= edge_2d < row_length[:, None]
    destination = tl.load(
        destination_ptr
        + pre_2d * stride_destination_row
        + edge_2d * stride_destination_slot,
        mask=valid,
        other=0,
    )
    edge = row_offset[:, None] + edge_2d
    weight = tl.load(values_ptr + edge, mask=valid, other=0.0)
    tl.atomic_add(
        out_ptr + batch * stride_out_b + destination * stride_out_n,
        weight,
        mask=valid,
    )


@triton.jit
def post_span_event_kernel(
    event_count_ptr,
    event_indices_ptr,
    row_length_ptr,
    row_offset_ptr,
    destination_ptr,
    values_ptr,
    out_ptr,
    stride_event_b,
    stride_event_n,
    stride_destination_row,
    stride_destination_slot,
    stride_out_b,
    stride_out_n,
    row_stride,
    max_events,
    BLOCK_EVENT: tl.constexpr,
    BLOCK_SLOT: tl.constexpr,
):
    batch = tl.program_id(0)
    event_block = tl.program_id(1)
    slot_block = tl.program_id(2)
    event_slot = event_block * BLOCK_EVENT + tl.arange(0, BLOCK_EVENT)
    edge_slot = slot_block * BLOCK_SLOT + tl.arange(0, BLOCK_SLOT)
    count = tl.load(event_count_ptr + batch)
    event_mask = (event_slot < count) & (event_slot < max_events)
    pre = tl.load(
        event_indices_ptr + batch * stride_event_b + event_slot * stride_event_n,
        mask=event_mask,
        other=0,
    )
    row_length = tl.load(row_length_ptr + pre, mask=event_mask, other=0)
    row_offset = tl.load(row_offset_ptr + pre, mask=event_mask, other=0)
    pre_2d = pre[:, None]
    edge_2d = edge_slot[None, :]
    valid = event_mask[:, None] & (edge_2d < row_stride)
    valid &= edge_2d < row_length[:, None]
    destination = tl.load(
        destination_ptr
        + pre_2d * stride_destination_row
        + edge_2d * stride_destination_slot,
        mask=valid,
        other=0,
    )
    edge = row_offset[:, None] + edge_2d
    weight = tl.load(values_ptr + edge, mask=valid, other=0.0)
    tl.atomic_add(
        out_ptr + batch * stride_out_b + destination * stride_out_n,
        weight,
        mask=valid,
    )
