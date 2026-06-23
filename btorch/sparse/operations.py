from __future__ import annotations

from typing import Literal

import torch

from .errors import BackendUnavailableError, UnsupportedCapabilityError
from .events import BinaryEvents, EventRepresentation, SpikeListEvents


SparseBackend = Literal["auto", "native", "torch_sparse", "triton", "warp"]
EventSchedule = Literal["auto", "pre_span", "post_span"]


def sparse_mm(matrix, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``x @ W`` for any SparseTensorBase matrix."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("sparse_mm expects a torch.Tensor input.")
    from .backends.native import sparse_mm as native_impl

    result = native_impl(matrix, x)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError("out shape does not match sparse_mm result.")
        out.copy_(result)
        return out
    return result


def available_backends() -> list[str]:
    from .backends import torch_sparse, triton, warp

    result = ["native"]
    if torch_sparse.is_available():
        result.append("torch_sparse")
    if triton.is_available():
        result.append("triton")
    if warp.is_available():
        result.append("warp")
    return result


def _event_to_numeric(events: EventRepresentation, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(events, BinaryEvents):
        return (events.values > events.threshold).to(dtype)
    return events.to_dense(dtype=dtype)


def compact_events(events: BinaryEvents) -> SpikeListEvents:
    values = events.values
    no_batch = values.ndim == 1
    if no_batch:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError("BinaryEvents must have shape (n,) or (batch, n).")

    if values.is_cuda:
        try:
            from .backends.triton.event import compact_binary_events

            return compact_binary_events(
                BinaryEvents(values, threshold=events.threshold)
            )
        except (ImportError, BackendUnavailableError):
            pass

    active = values > events.threshold
    count = active.sum(dim=1, dtype=torch.int32)
    capacity = max(int(count.max().item()), 1)
    indices = torch.zeros(
        (values.shape[0], capacity),
        device=values.device,
        dtype=torch.long,
    )
    for batch in range(values.shape[0]):
        selected = torch.nonzero(active[batch], as_tuple=False).flatten()
        indices[batch, : selected.numel()] = selected
    return SpikeListEvents(count=count, indices=indices, size=values.shape[1])


def event_sparse_mm(
    matrix,
    events: EventRepresentation,
    *,
    schedule: str = "auto",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if events.size != matrix.shape[0]:
        raise ValueError(
            f"Expected event size {matrix.shape[0]}, received {events.size}."
        )
    eff_values = matrix.effective_values()
    requires_grad = torch.is_grad_enabled() and (
        eff_values.requires_grad
        or (isinstance(events, BinaryEvents) and events.values.requires_grad)
    )
    if requires_grad:
        dense = _event_to_numeric(events, eff_values.dtype)
        return matrix.mm(dense)

    if eff_values.is_cuda and hasattr(matrix, "padded_csr_layout"):
        try:
            from .backends.triton.event import event_sparse_mm as triton_impl

            return triton_impl(matrix, events, schedule=schedule, out=out)
        except (BackendUnavailableError, UnsupportedCapabilityError):
            pass

    dense = _event_to_numeric(events, eff_values.dtype)
    result = matrix.mm(dense)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError("out shape does not match event_sparse_mm result.")
        out.copy_(result)
        return out
    return result
