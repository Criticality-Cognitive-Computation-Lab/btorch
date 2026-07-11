"""Persistent SNN operator scaffold.

This module defines the stable Python-side contract for an event-driven
persistent-kernel SNN backend. The current implementation is intentionally a
no-op reference/stub: it validates layout, preserves state, and returns empty
spike outputs. A future C++/CUDA implementation can register
``torch.ops.btorch_cuda.persistent_snn_forward`` without changing benchmark
or caller code.

The persistent kernel target is intentionally narrow: recurrent RSNN dynamics
with scalar-parameter ``GLIF3`` neurons and ``AlphaPSC`` synapses. The benchmark
and persistent contract share these defaults:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - First-kernel meaning
   * - ``dt``
     - ``1.0``
     - Euler step size for GLIF and AlphaPSC dynamics.
   * - ``tau``
     - ``20.0``
     - GLIF membrane time constant.
   * - ``tau_syn``
     - ``5.0``
     - AlphaPSC time constant.
   * - ``v_threshold``
     - ``-45.0``
     - Spike threshold; spikes are emitted when ``v >= v_threshold``.
   * - ``v_reset``
     - ``-60.0``
     - Reset baseline used by the GLIF leak and reset delta.
   * - ``c_m``
     - ``2.0``
     - Membrane capacitance divisor for input current.
   * - ``tau_ref``
     - ``2.0``
     - Refractory period in time units.
   * - ``k`` / ``asc_amps``
     - ``(0.1, 0.2)`` / ``(1.0, -2.0)``
     - GLIF3 after-spike current decay rates and increments.
   * - ``hard_reset``
     - ``False``
     - Soft reset: subtract ``v_threshold - v_reset`` after a spike.
   * - ``window_size``
     - ``128``
     - Suggested processing window length in time steps.
"""

from __future__ import annotations

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
    psc_h: torch.Tensor | None = None
    asc: torch.Tensor | None = None
    refractory: torch.Tensor | None = None
    delay_ring: torch.Tensor | None = None


@dataclass(frozen=True)
class PersistentSNNOutput:
    """Output of the persistent SNN operator scaffold."""

    spikes: torch.Tensor | None
    spike_events: WindowedSpikeEvents | None
    state: PersistentSNNState


@dataclass(frozen=True, init=False)
class PersistentSNNParams:
    """Scalar simulation parameters for the persistent SNN operator.

    The CUDA persistent kernel is scoped to scalar-parameter
    :class:`btorch.models.neurons.glif.GLIF3` +
    :class:`btorch.models.synapse.AlphaPSC` dynamics. Its default reset mode is
    the same as ``GLIF3``: soft reset (``hard_reset=False``).
    """

    dt: float = 1.0
    tau: float = 20.0
    tau_syn: float = 5.0
    v_threshold: float = -45.0
    v_reset: float = -60.0
    v_rest: float | None = None
    c_m: float = 2.0
    tau_ref: float = 2.0
    k: tuple[float, ...] = (0.1, 0.2)
    asc_amps: tuple[float, ...] = (1.0, -2.0)
    psc_g_max: float = 1.0
    hard_reset: bool = False
    window_size: int = 128

    def __init__(
        self,
        dt: float = 1.0,
        tau: float = 20.0,
        tau_mem: float | None = None,
        tau_syn: float = 5.0,
        v_threshold: float = -45.0,
        v_reset: float = -60.0,
        v_rest: float | None = None,
        c_m: float = 2.0,
        tau_ref: float = 2.0,
        k: tuple[float, ...] = (0.1, 0.2),
        asc_amps: tuple[float, ...] = (1.0, -2.0),
        psc_g_max: float = 1.0,
        hard_reset: bool = False,
        window_size: int = 128,
    ) -> None:
        if tau_mem is not None:
            tau = tau_mem
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "tau_syn", tau_syn)
        object.__setattr__(self, "v_threshold", v_threshold)
        object.__setattr__(self, "v_reset", v_reset)
        object.__setattr__(self, "v_rest", v_rest)
        object.__setattr__(self, "c_m", c_m)
        object.__setattr__(self, "tau_ref", tau_ref)
        object.__setattr__(self, "k", tuple(k))
        object.__setattr__(self, "asc_amps", tuple(asc_amps))
        object.__setattr__(self, "psc_g_max", psc_g_max)
        object.__setattr__(self, "hard_reset", hard_reset)
        object.__setattr__(self, "window_size", window_size)

    @property
    def tau_mem(self) -> float:
        """Backward-compatible alias for the membrane time constant."""

        return self.tau


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
    if int(offsets[0].item()) != 0:
        raise ValueError("events.offsets[0] must be zero.")
    if int(offsets[-1].item()) != indices.numel():
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
    if state.psc_h is not None and tuple(state.psc_h.shape) != expected:
        raise ValueError(
            f"state.psc_h must have shape {expected}, got {state.psc_h.shape}."
        )
    if state.asc is not None:
        if state.asc.ndim != 3 or tuple(state.asc.shape[:2]) != expected:
            raise ValueError(
                "state.asc must have shape (B, N, n_asc), got "
                f"{state.asc.shape}."
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
    psc_h: bool = True,
    n_asc: int = 2,
    delay_ring_shape: tuple[int, int, int] | None = None,
) -> PersistentSNNState:
    """Create zero-filled persistent SNN state tensors."""

    v = torch.zeros((batch_size, n_neuron), device=device, dtype=dtype)
    psc = torch.zeros_like(v)
    psc_h_t = torch.zeros_like(v) if psc_h else None
    asc = torch.zeros((batch_size, n_neuron, n_asc), device=device, dtype=dtype)
    refractory_t = torch.zeros_like(v) if refractory else None
    delay_ring = None
    if delay_ring_shape is not None:
        delay_ring = torch.zeros(delay_ring_shape, device=device, dtype=dtype)
    return PersistentSNNState(
        v=v,
        psc=psc,
        psc_h=psc_h_t,
        asc=asc,
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
    return (
        hasattr(torch.ops, "btorch_cuda")
        and hasattr(torch.ops.btorch_cuda, "persistent_snn_forward")
    )


def _ensure_cuda_op() -> None:
    if _has_cuda_op():
        return
    from .persistent import plain_version

    plain_version.load()


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
    if state.refractory is None:
        raise ValueError("cuda_persistent v1 requires refractory state.")
    if state.psc_h is None:
        raise ValueError("cuda_persistent v1 requires psc_h state.")
    if state.asc is None:
        raise ValueError("cuda_persistent v1 requires asc state.")
    if len(params.k) != 2 or len(params.asc_amps) != 2:
        raise ValueError("cuda_persistent v1 requires exactly two GLIF3 ASC modes.")
    if state.delay_ring is not None:
        raise ValueError("cuda_persistent v1 does not support delay_ring state.")
    if events.offsets.dtype != torch.int32 or events.indices.dtype != torch.int32:
        raise TypeError("cuda_persistent v1 requires int32 event tensors.")
    if graph.indptr.dtype != torch.int32 or graph.indices.dtype != torch.int32:
        raise TypeError("cuda_persistent v1 requires int32 graph indices.")
    if graph.delay is not None and graph.delay.dtype != torch.int32:
        raise TypeError("cuda_persistent v1 requires int32 graph delay.")
    if graph.delay is not None and torch.count_nonzero(graph.delay).item() != 0:
        raise ValueError("cuda_persistent v1 does not support nonzero delay.")
    if state.v.dtype != torch.float32 or state.psc.dtype != torch.float32:
        raise TypeError("cuda_persistent v1 requires float32 state tensors.")
    if state.psc_h.dtype != torch.float32 or state.asc.dtype != torch.float32:
        raise TypeError("cuda_persistent v1 requires float32 extended state tensors.")
    if state.refractory.dtype != torch.float32:
        raise TypeError("cuda_persistent v1 requires float32 refractory tensor.")
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
        psc_h_out,
        asc_out,
        refractory_out,
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
        state.v,
        state.psc,
        state.psc_h,
        state.asc,
        state.refractory,
        float(params.dt),
        float(params.tau),
        float(params.tau_syn),
        float(params.v_threshold),
        float(params.v_reset),
        float(params.v_reset if params.v_rest is None else params.v_rest),
        float(params.c_m),
        float(params.tau_ref),
        float(params.k[0]),
        float(params.k[1]),
        float(params.asc_amps[0]),
        float(params.asc_amps[1]),
        float(params.psc_g_max),
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
        state=PersistentSNNState(
            v=v_out,
            psc=psc_out,
            psc_h=psc_h_out,
            asc=asc_out,
            refractory=refractory_out,
        ),
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
