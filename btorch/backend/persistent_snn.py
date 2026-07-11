"""Persistent SNN operator scaffold.

This module defines the stable Python-side contract for an event-driven
persistent-kernel SNN backend. The current implementation is intentionally a
no-op reference/stub: it validates layout, preserves state, and returns empty
spike outputs. A future C++/CUDA implementation can register
``torch.ops.btorch_cuda.persistent_snn_forward`` without changing benchmark
or caller code.

The first persistent kernel target is intentionally narrow: recurrent RSNN
dynamics with a scalar-parameter LIF neuron and an ``ExponentialPSC`` synapse.
The benchmark and persistent contract share these defaults:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - First-kernel meaning
   * - ``dt``
     - ``1.0``
     - Euler step size for LIF and exponential PSC decay.
   * - ``tau_mem``
     - ``20.0``
     - LIF membrane time constant.
   * - ``tau_syn``
     - ``5.0``
     - Exponential PSC time constant.
   * - ``v_threshold``
     - ``1.0``
     - Spike threshold; spikes are emitted when ``v >= v_threshold``.
   * - ``v_reset``
     - ``0.0``
     - Reset baseline used by the LIF leak and reset delta.
   * - ``c_m``
     - ``1.0``
     - Membrane capacitance divisor for input current.
   * - ``hard_reset``
     - ``False``
     - Soft reset: subtract ``v_threshold - v_reset`` after a spike.
   * - ``window_size``
     - ``128``
     - Suggested processing window length in time steps.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Literal

import torch


Backend = Literal["auto", "torch_stub", "cuda_persistent"]
ReturnMode = Literal["dense", "events", "both"]


@dataclass(frozen=True)
class WindowedSpikeEvents:
    """Time-batch bucketed spike-event input.

    Args:
        offsets: Prefix sum over ``T * B`` buckets, shape ``(T * B + 1,)``.
        indices: Pre-synaptic neuron indices, shape ``(nnz,)``.
        values: Optional per-event values, shape ``(nnz,)``.
        shape: Logical dense event shape ``(T, B, N_pre)``.
    """

    offsets: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor | None
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class EventCSRGraph:
    """Pre-synaptic-row CSR graph for event fanout.

    Args:
        indptr: Row pointer for pre-synaptic neurons, shape ``(N_pre + 1,)``.
        indices: Post-synaptic neuron indices, shape ``(E,)``.
        weight: Edge weights, shape ``(E,)``.
        delay: Optional delay in integer time steps, shape ``(E,)``.
        shape: Logical graph shape ``(N_pre, N_post)``.
    """

    indptr: torch.Tensor
    indices: torch.Tensor
    weight: torch.Tensor
    delay: torch.Tensor | None
    shape: tuple[int, int]


@dataclass(frozen=True)
class PersistentSNNState:
    """State tensors carried across persistent SNN windows."""

    v: torch.Tensor
    psc: torch.Tensor
    refractory: torch.Tensor | None = None
    delay_ring: torch.Tensor | None = None


@dataclass(frozen=True)
class PersistentSNNOutput:
    """Output of the persistent SNN operator scaffold."""

    spikes: torch.Tensor | None
    spike_events: WindowedSpikeEvents | None
    state: PersistentSNNState


@dataclass(frozen=True)
class PersistentSNNParams:
    """Scalar simulation parameters for the persistent SNN operator.

    The first CUDA persistent kernel is scoped to scalar-parameter
    LIF + ExponentialPSC dynamics. Its default reset mode is the same as
    :class:`btorch.models.neurons.lif.LIF`: soft reset
    (``hard_reset=False``).
    """

    dt: float = 1.0
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    c_m: float = 1.0
    hard_reset: bool = False
    window_size: int = 128


def _check_int_tensor(name: str, tensor: torch.Tensor, ndim: int) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tensor.ndim}.")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64, got {tensor.dtype}.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_events(events: WindowedSpikeEvents) -> tuple[int, int, int]:
    offsets, indices = events.offsets, events.indices
    t_steps, batch_size, n_pre = events.shape
    if t_steps <= 0 or batch_size <= 0 or n_pre <= 0:
        raise ValueError(f"events.shape must be positive, got {events.shape}.")
    _check_int_tensor("events.offsets", offsets, ndim=1)
    _check_int_tensor("events.indices", indices, ndim=1)
    if offsets.numel() != t_steps * batch_size + 1:
        raise ValueError(
            "events.offsets must have shape (T * B + 1,), got "
            f"{tuple(offsets.shape)} for shape={events.shape}."
        )
    # One D2H sync for both boundary values instead of two separate `.item()` calls.
    offsets_first, offsets_last = offsets[[0, -1]].tolist()
    if offsets_first != 0:
        raise ValueError("events.offsets[0] must be zero.")
    if offsets_last != indices.numel():
        raise ValueError("events.offsets[-1] must equal events.indices.numel().")
    if events.values is not None:
        if events.values.shape != indices.shape:
            raise ValueError("events.values must have the same shape as indices.")
        if events.values.device != indices.device:
            raise ValueError("events.values and indices must be on the same device.")
    return t_steps, batch_size, n_pre


def _validate_graph(graph: EventCSRGraph, n_pre: int) -> int:
    graph_n_pre, n_post = graph.shape
    if graph_n_pre != n_pre:
        raise ValueError(f"graph has N_pre={graph_n_pre}, events have N_pre={n_pre}.")
    if n_post <= 0:
        raise ValueError(f"graph N_post must be positive, got {n_post}.")
    _check_int_tensor("graph.indptr", graph.indptr, ndim=1)
    _check_int_tensor("graph.indices", graph.indices, ndim=1)
    if graph.indptr.numel() != n_pre + 1:
        raise ValueError("graph.indptr must have shape (N_pre + 1,).")
    if graph.weight.ndim != 1:
        raise ValueError("graph.weight must have shape (E,).")
    if graph.indices.shape != graph.weight.shape:
        raise ValueError("graph.indices and graph.weight must have matching shape.")
    if graph.delay is not None and graph.delay.shape != graph.indices.shape:
        raise ValueError("graph.delay must have the same shape as graph.indices.")
    return n_post


def _validate_state(
    state: PersistentSNNState,
    *,
    batch_size: int,
    n_post: int,
) -> None:
    expected = (batch_size, n_post)
    if tuple(state.v.shape) != expected:
        raise ValueError(f"state.v must have shape {expected}, got {state.v.shape}.")
    if tuple(state.psc.shape) != expected:
        raise ValueError(
            f"state.psc must have shape {expected}, got {state.psc.shape}."
        )
    if state.refractory is not None and tuple(state.refractory.shape) != expected:
        raise ValueError(
            f"state.refractory must have shape {expected}, "
            f"got {state.refractory.shape}."
        )


def make_empty_state(
    batch_size: int,
    n_neuron: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    refractory: bool = True,
    delay_ring_shape: tuple[int, int, int] | None = None,
) -> PersistentSNNState:
    """Create zero-filled persistent SNN state tensors."""

    v = torch.zeros((batch_size, n_neuron), device=device, dtype=dtype)
    psc = torch.zeros_like(v)
    refractory_t = torch.zeros_like(v) if refractory else None
    delay_ring = None
    if delay_ring_shape is not None:
        delay_ring = torch.zeros(delay_ring_shape, device=device, dtype=dtype)
    return PersistentSNNState(
        v=v,
        psc=psc,
        refractory=refractory_t,
        delay_ring=delay_ring,
    )


def torch_stub_persistent_snn_forward(
    events: WindowedSpikeEvents,
    graph: EventCSRGraph,
    state: PersistentSNNState,
    params: PersistentSNNParams | None = None,
    *,
    return_mode: ReturnMode = "dense",
) -> PersistentSNNOutput:
    """Run the no-op persistent SNN reference implementation.

    The stub proves the operator contract without doing neural dynamics:
    state tensors are returned unchanged, dense spikes are all zero, and event
    output contains no spikes. It is intentionally deterministic and device
    preserving, so benchmarks can be used before the CUDA kernel exists.
    """

    del params
    t_steps, batch_size, n_pre = _validate_events(events)
    n_post = _validate_graph(graph, n_pre)
    _validate_state(state, batch_size=batch_size, n_post=n_post)

    dense_spikes = None
    if return_mode in ("dense", "both"):
        dense_spikes = torch.zeros(
            (t_steps, batch_size, n_post),
            device=state.v.device,
            dtype=state.v.dtype,
        )

    spike_events = None
    if return_mode in ("events", "both"):
        event_offsets = torch.zeros_like(events.offsets)
        event_indices = torch.empty(
            (0,), device=events.indices.device, dtype=events.indices.dtype
        )
        spike_events = WindowedSpikeEvents(
            offsets=event_offsets,
            indices=event_indices,
            values=None,
            shape=(t_steps, batch_size, n_post),
        )

    return PersistentSNNOutput(
        spikes=dense_spikes,
        spike_events=spike_events,
        state=state,
    )


def _has_cuda_op() -> bool:
    return hasattr(torch.ops, "btorch_cuda") and hasattr(
        torch.ops.btorch_cuda, "persistent_snn_forward"
    )


def _ensure_cuda_op() -> None:
    if _has_cuda_op():
        return
    from .persistent import plain_version

    plain_version.load()


# Delay tensors already confirmed all-zero, keyed by id(). `graph.delay` is part
# of a graph's fixed structure and is typically reused unchanged across many
# forward() calls (e.g. sequential windows of one long sequence) -- caching this
# avoids re-running an O(E) reduction + device->host sync on every single call
# for a value that never changes.
#
# Keyed by id() rather than stored in a `weakref.WeakSet`/`WeakKeyDictionary`:
# those containers fall back to `==` on hash-bucket lookups, and torch.Tensor's
# `__eq__` is elementwise (returns a Tensor, not a bool), which raises
# "Boolean value of Tensor with more than one value is ambiguous" -- even when
# comparing a tensor against itself, since weakref's own `__eq__` has no
# identity shortcut. Each entry carries a weakref-with-callback so it's evicted
# the moment the tensor is actually garbage collected, which also avoids the
# id()-reuse hazard (a new, unrelated tensor later allocated at the same
# address would otherwise appear to hit the cache).
_zero_delay_cache: dict[int, "weakref.ref[torch.Tensor]"] = {}


def _delay_confirmed_zero(delay: torch.Tensor) -> bool:
    ref = _zero_delay_cache.get(id(delay))
    return ref is not None and ref() is delay


def _remember_delay_confirmed_zero(delay: torch.Tensor) -> None:
    key = id(delay)

    def _evict(_: object, key: int = key) -> None:
        _zero_delay_cache.pop(key, None)

    _zero_delay_cache[key] = weakref.ref(delay, _evict)


def _cuda_persistent_snn_forward(
    events: WindowedSpikeEvents,
    graph: EventCSRGraph,
    state: PersistentSNNState,
    params: PersistentSNNParams,
    *,
    return_mode: ReturnMode,
) -> PersistentSNNOutput:
    t_steps, batch_size, n_pre = _validate_events(events)
    n_post = _validate_graph(graph, n_pre)
    _validate_state(state, batch_size=batch_size, n_post=n_post)
    if n_pre != n_post:
        raise ValueError("cuda_persistent v1 requires a recurrent N x N graph.")
    if params.hard_reset:
        raise ValueError("cuda_persistent v1 only supports hard_reset=False.")
    if state.refractory is not None:
        raise ValueError("cuda_persistent v1 does not support refractory state.")
    if state.delay_ring is not None:
        raise ValueError("cuda_persistent v1 does not support delay_ring state.")
    if events.offsets.dtype != torch.int32 or events.indices.dtype != torch.int32:
        raise TypeError("cuda_persistent v1 requires int32 event tensors.")
    if graph.indptr.dtype != torch.int32 or graph.indices.dtype != torch.int32:
        raise TypeError("cuda_persistent v1 requires int32 graph indices.")
    if graph.delay is not None and graph.delay.dtype != torch.int32:
        raise TypeError("cuda_persistent v1 requires int32 graph delay.")
    delay_validated = False
    if graph.delay is not None:
        if not _delay_confirmed_zero(graph.delay):
            if torch.count_nonzero(graph.delay).item() != 0:
                raise ValueError("cuda_persistent v1 does not support nonzero delay.")
            _remember_delay_confirmed_zero(graph.delay)
        delay_validated = True
    if state.v.dtype != torch.float32 or state.psc.dtype != torch.float32:
        raise TypeError("cuda_persistent v1 requires float32 state tensors.")
    if graph.weight.dtype != torch.float32:
        raise TypeError("cuda_persistent v1 requires float32 graph weights.")
    if events.values is not None and events.values.dtype != torch.float32:
        raise TypeError("cuda_persistent v1 requires float32 event values.")

    _ensure_cuda_op()
    event_values = (
        events.values
        if events.values is not None
        else torch.empty((0,), device=events.indices.device, dtype=state.v.dtype)
    )
    graph_delay = (
        graph.delay
        if graph.delay is not None
        else torch.empty((0,), device=graph.indices.device, dtype=graph.indices.dtype)
    )
    return_events = return_mode in ("events", "both")
    (
        dense_spikes,
        event_offsets,
        event_indices,
        v_out,
        psc_out,
        _overflow,
    ) = torch.ops.btorch_cuda.persistent_snn_forward(
        events.offsets,
        events.indices,
        event_values,
        events.values is not None,
        graph.indptr,
        graph.indices,
        graph.weight,
        graph_delay,
        graph.delay is not None,
        delay_validated,
        state.v,
        state.psc,
        float(params.dt),
        float(params.tau_mem),
        float(params.tau_syn),
        float(params.v_threshold),
        float(params.v_reset),
        float(params.c_m),
        bool(params.hard_reset),
        return_events,
    )

    spikes = dense_spikes if return_mode in ("dense", "both") else None
    spike_events = None
    if return_events:
        spike_events = WindowedSpikeEvents(
            offsets=event_offsets,
            indices=event_indices,
            values=None,
            shape=(t_steps, batch_size, n_post),
        )
    return PersistentSNNOutput(
        spikes=spikes,
        spike_events=spike_events,
        state=PersistentSNNState(v=v_out, psc=psc_out),
    )


def persistent_snn_forward(
    events: WindowedSpikeEvents,
    graph: EventCSRGraph,
    state: PersistentSNNState,
    params: PersistentSNNParams | None = None,
    *,
    backend: Backend = "auto",
    return_mode: ReturnMode = "dense",
) -> PersistentSNNOutput:
    """Dispatch the persistent SNN operator.

    Args:
        events: Windowed input spike events.
        graph: CSR connectivity graph.
        state: Persistent state tensors.
        params: Scalar simulation parameters.
        backend: ``"torch_stub"`` for the reference no-op, ``"cuda_persistent"``
            for the future compiled operator, or ``"auto"`` to use CUDA when
            registered and fall back to the stub otherwise.
        return_mode: Select dense spikes, event spikes, or both.

    Returns:
        Operator output with spikes/events and final state.
    """

    params = params or PersistentSNNParams()
    if backend not in ("auto", "torch_stub", "cuda_persistent"):
        raise ValueError(f"Unknown backend: {backend}.")

    use_cuda = backend == "cuda_persistent" or (backend == "auto" and _has_cuda_op())
    if use_cuda:
        return _cuda_persistent_snn_forward(
            events,
            graph,
            state,
            params,
            return_mode=return_mode,
        )

    return torch_stub_persistent_snn_forward(
        events,
        graph,
        state,
        params,
        return_mode=return_mode,
    )


__all__ = [
    "EventCSRGraph",
    "PersistentSNNOutput",
    "PersistentSNNParams",
    "PersistentSNNState",
    "WindowedSpikeEvents",
    "make_empty_state",
    "persistent_snn_forward",
    "torch_stub_persistent_snn_forward",
]
