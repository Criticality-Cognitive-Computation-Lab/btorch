from __future__ import annotations

import torch

from btorch.sparse.errors import BackendUnavailableError, UnsupportedCapabilityError
from btorch.sparse.events import BinaryEvents, SpikeListEvents


def _require_triton():
    try:
        import triton
    except ImportError as exc:
        raise BackendUnavailableError("Triton is not installed.") from exc
    return triton


def compact_binary_events(events: BinaryEvents) -> SpikeListEvents:
    triton = _require_triton()
    from .event_kernels import dense_event_to_list_kernel

    values = events.values
    if values.ndim != 2:
        raise ValueError("Triton event compaction expects batched 2D values.")
    values = values.contiguous()
    batch_size, n_pre = values.shape
    count = torch.zeros(batch_size, device=values.device, dtype=torch.int32)
    indices = torch.empty((batch_size, n_pre), device=values.device, dtype=torch.int64)
    block_size = 256
    grid = (batch_size, triton.cdiv(n_pre, block_size))
    dense_event_to_list_kernel[grid](
        values,
        count,
        indices,
        values.stride(0),
        values.stride(1),
        indices.stride(0),
        indices.stride(1),
        n_pre,
        THRESHOLD=events.threshold,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return SpikeListEvents(count=count, indices=indices, size=n_pre)


def event_sparse_mm(
    matrix,
    events: BinaryEvents | SpikeListEvents,
    *,
    schedule: str = "auto",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    triton = _require_triton()
    from .event_kernels import (
        post_span_event_kernel,
        pre_span_event_kernel,
    )

    no_batch = isinstance(events, BinaryEvents) and events.values.ndim == 1
    if isinstance(events, BinaryEvents):
        values = events.values.unsqueeze(0) if no_batch else events.values
        events = compact_binary_events(BinaryEvents(values, threshold=events.threshold))
    eff_values = matrix.effective_values()
    if not events.indices.is_cuda or not eff_values.is_cuda:
        raise UnsupportedCapabilityError("Triton event execution requires CUDA.")
    if events.indices.device != eff_values.device:
        raise ValueError("Events and matrix must be on the same device.")
    if not hasattr(matrix, "padded_csr_layout"):
        raise UnsupportedCapabilityError(
            f"{type(matrix).__name__} has no padded CSR layout."
        )

    layout = matrix.padded_csr_layout()
    batch_size, max_events = events.indices.shape
    output_size = getattr(matrix, "operation_shape", matrix.shape)[1]
    if out is None:
        out = torch.zeros(
            (batch_size, output_size),
            device=eff_values.device,
            dtype=eff_values.dtype,
        )
    else:
        if out.shape != (batch_size, output_size):
            raise ValueError("out has an incompatible shape.")
        out.zero_()
    if layout.row_stride == 0:
        return out[0] if no_batch else out

    schedule = "pre_span" if schedule == "auto" else schedule
    if schedule == "pre_span":
        block_event, block_edge = 1, 32
        grid = (
            batch_size,
            triton.cdiv(max_events, block_event),
            triton.cdiv(layout.row_stride, block_edge),
        )
        kernel = pre_span_event_kernel
        block_kwargs = {
            "BLOCK_EVENT": block_event,
            "BLOCK_EDGE": block_edge,
        }
    elif schedule == "post_span":
        block_event, block_slot = 32, 32
        grid = (
            batch_size,
            triton.cdiv(max_events, block_event),
            triton.cdiv(layout.row_stride, block_slot),
        )
        kernel = post_span_event_kernel
        block_kwargs = {
            "BLOCK_EVENT": block_event,
            "BLOCK_SLOT": block_slot,
        }
    else:
        raise ValueError(f"Unknown event schedule {schedule!r}.")

    kernel[grid](
        events.count.contiguous(),
        events.indices.contiguous(),
        layout.row_length.contiguous(),
        layout.row_offset.contiguous(),
        layout.indices.contiguous(),
        eff_values.reshape(-1).contiguous(),
        out,
        events.indices.stride(0),
        events.indices.stride(1),
        layout.indices.stride(0),
        layout.indices.stride(1),
        out.stride(0),
        out.stride(1),
        layout.row_stride,
        max_events,
        num_warps=1,
        **block_kwargs,
    )
    return out[0] if no_batch else out
