"""Arrival-based online learning rules over canonical connection edges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from ...sparse import EdgeMap
from ..base import MemoryModule


PlasticityDomain = Literal["edge", "segment", "post", "scalar"]


@dataclass(frozen=True)
class PlasticityLayout:
    """Describe persistent learning-state domains without owning topology.

    Args:
        n_edge: Number of canonical edges.
        n_post: Number of postsynaptic neurons.
        n_segment: Number of nonempty source/delay groups, when requested.
    """

    n_edge: int
    n_post: int
    n_segment: int = 0

    def shape(self, domain: PlasticityDomain) -> tuple[int, ...]:
        """Look up the leading shape of a declared state or coefficient."""
        if domain == "scalar":
            return ()
        if domain not in ("edge", "segment", "post"):
            raise ValueError(f"Unknown plasticity domain: {domain!r}")
        return (getattr(self, "n_" + domain),)


@dataclass(frozen=True)
class PostState:
    """Bind a live postsynaptic state at a declared neuron stage.

    Args:
        name: Physical state name in the neuron template.
        stage: ``state`` reads the post-step state; ``charged`` reads its
            declared pre-reset record. The neuron must enable that record.
    """

    name: str
    stage: Literal["state", "charged"] = "charged"

    def __post_init__(self):
        if not self.name.isidentifier() or self.stage not in ("state", "charged"):
            raise ValueError("PostState requires a state name and declared stage")


@dataclass(frozen=True)
class ExternalSignal:
    """Declare the layout of a caller-supplied per-step learning signal.

    Args:
        domain: Scalar, canonical edge, postsynaptic or source/delay segment.
        time_first: Supply one value per timestep to RecurrentNN. False means
            the supplied tensor is constant during each multi-step call.
    """

    domain: PlasticityDomain = "scalar"
    time_first: bool = False

    def __post_init__(self):
        if self.domain not in ("scalar", "edge", "post", "segment"):
            raise ValueError(f"Unknown plasticity signal domain: {self.domain!r}")
        if not isinstance(self.time_first, bool):
            raise TypeError("time_first must be a bool")


def _validate_weight_storage(weight: Tensor, n_edge: int) -> None:
    """Require separate writable storage for distinct canonical edge slots."""
    if weight.ndim != 1 or weight.numel() != n_edge:
        raise ValueError(
            f"STDP requires {n_edge} canonical edge weights, "
            f"got shape {tuple(weight.shape)}"
        )
    if weight.numel() > 1 and weight.stride(-1) == 0:
        raise ValueError(
            "STDP requires independent writable edge weights; "
            "a broadcast weight aliases multiple canonical edges"
        )


@dataclass(frozen=True)
class PlasticityEvents:
    """Provide canonical edge events and explicitly supplied rule signals.

    Args:
        arrival: Presynaptic activity arriving now, including each edge delay.
        post_spike: Current postsynaptic activity for each canonical edge.
        signals: Additional named tensor inputs required by a custom rule.
            The caller supplies live values at the rule's specified stage.
        post_index: Postsynaptic neuron index for each canonical edge, allowing
            a rule to gather its declared population signals.
        segment_index: Source/delay segment index for each canonical edge,
            prepared only when a rule declares segment-domain inputs or state.
        segment_arrival: Arriving activity for each nonempty source/delay
            segment, including segments shared by several edges.
        post_activity: Full postsynaptic activity, including neurons without
            incoming edges.
    """

    arrival: Tensor
    post_spike: Tensor
    signals: Mapping[str, Tensor] = field(default_factory=dict)
    post_index: Tensor | None = None
    segment_index: Tensor | None = None
    segment_arrival: Tensor | None = None
    post_activity: Tensor | None = None


class SynapticRule(MemoryModule):
    """Bind persistent per-edge synaptic state to canonical connectivity.

    This base owns coefficient placement and :class:`MemoryModule` lifecycle.
    It deliberately does not define how a rule changes weights or releases
    activity; those semantics belong to :class:`PlasticityRule` and
    :class:`~btorch.models.connection.TransmissionRule` respectively.
    """

    state_domains: Mapping[str, PlasticityDomain] = {}
    parameter_domains: Mapping[str, PlasticityDomain] = {}

    def __init__(self):
        super().__init__()
        self._bound = False
        self._positive_coefficients: set[str] = set()
        self.signals: dict[str, PostState | ExternalSignal] = {}

    def _register_coefficient(
        self, name: str, value: float | Tensor, *, positive=False, nonnegative=False
    ) -> None:
        """Register a finite scalar or edge-aligned coefficient."""
        value = torch.as_tensor(value)
        if value.requires_grad:
            raise ValueError("Synaptic coefficients are fixed, not BPTT parameters")
        if value.ndim > 1 or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite and scalar or edge-aligned")
        if positive and not (value > 0).all():
            raise ValueError(f"{name} must be positive")
        if nonnegative and not (value >= 0).all():
            raise ValueError(f"{name} must be nonnegative")
        if not value.is_floating_point():
            value = value.to(torch.get_default_dtype())
        self.register_buffer(name, value.detach().clone())
        if positive:
            self._positive_coefficients.add(name)

    def _validate_binding(self, edge_map: EdgeMap, layout: PlasticityLayout) -> None:
        """Validate rule-specific coefficient relationships before binding."""

    def _validate_weight_binding(self, edge_map: EdgeMap, weight: Tensor) -> None:
        """Validate weight storage requirements owned by a concrete rule."""

    def bind(self, edge_map: EdgeMap, weight: Tensor, layout: PlasticityLayout) -> None:
        """Bind coefficients and initialize persistent memories once."""
        if self._bound:
            raise ValueError("Create one synaptic rule module per connection")
        if not edge_map.is_permutation:
            raise ValueError(
                "Synaptic rules need distinct canonical edges; "
                "coalesce duplicates first"
            )
        self._validate_weight_binding(edge_map, weight)
        if layout.n_edge != edge_map.n_new:
            raise ValueError("Synaptic layout disagrees with canonical edge count")
        if set(self.parameter_domains) - set(self._buffers):
            raise ValueError(
                "Parameter domains must name registered coefficient buffers"
            )
        for domain in self.parameter_domains.values():
            layout.shape(domain)
        self._validate_binding(edge_map, layout)
        source = edge_map.source()
        for name, value in self.named_buffers(recurse=False):
            if name in self._memories_rv:
                continue
            if value.requires_grad:
                raise ValueError("Synaptic coefficients are fixed, not BPTT parameters")
            if value.ndim > 1 or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite and scalar or edge-aligned")
            value = value.to(
                device=weight.device,
                dtype=weight.dtype if value.is_floating_point() else value.dtype,
            )
            if value.ndim:
                domain = self.parameter_domains.get(name, "edge")
                expected = (
                    (edge_map.n_old,) if domain == "edge" else layout.shape(domain)
                )
                if value.shape != expected:
                    raise ValueError(f"{name} must have shape {expected} for {domain}")
                if domain == "edge":
                    value = value.index_select(0, source)
            setattr(self, name, value)
        self._validate_coefficients()
        if set(self.state_domains) != set(self._memories_rv):
            raise ValueError("Declare a domain for every synaptic memory")
        for name, domain in self.state_domains.items():
            shape = layout.shape(domain)
            # Constructors declare local memory dimensions. Binding supplies
            # the topology-owned prefix through the normal reset interface.
            reset = self._memories_rv[name]
            self.set_reset_value(
                name, reset, sizes=(*shape, *reset.sizes), strict=False
            )
        self.init_state(device=weight.device, dtype=weight.dtype)
        self._layout = layout
        self._bound = True

    def _validate_coefficients(self) -> None:
        """Check finite and positive coefficient constraints after binding."""
        for name, value in self.named_buffers(recurse=False):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must remain finite in the weight dtype")
            if name in self._positive_coefficients and not (value > 0).all():
                raise ValueError(f"{name} must remain positive in the weight dtype")

    def _validate_state(self) -> None:
        """Require one sample with each registered memory shape."""
        for name, reset in self._memories_rv.items():
            value = getattr(self, name)
            rank = len(reset.sizes)
            state_shape = value.shape[-rank:] if rank else ()
            batch_shape = value.shape[:-rank] if rank else value.shape
            if state_shape != reset.sizes or any(size != 1 for size in batch_shape):
                raise NotImplementedError(
                    "Synaptic memories require one sample with registered shapes; "
                    "shared-weight minibatch learning is undefined"
                )

    @staticmethod
    def _time_step(dt: float | Tensor, reference: Tensor) -> Tensor:
        """Bind a positive scalar clock to the rule's device and precision."""
        if isinstance(dt, Tensor):
            if dt.numel() != 1 or dt.requires_grad:
                raise ValueError("Synaptic dt must be a non-differentiable scalar")
            valid = (torch.isfinite(dt) & (dt > 0)).all()
            if dt.device.type == "cpu":
                if not bool(valid):
                    raise ValueError("Synaptic dt must be finite and positive")
            else:
                torch._assert_async(valid, "Synaptic dt must be finite and positive")
            return dt.to(device=reference.device, dtype=reference.dtype).reshape(())
        if not (0 < dt < float("inf")):
            raise ValueError("Synaptic dt must be finite and positive")
        return torch.full((), dt, device=reference.device, dtype=reference.dtype)

    @property
    def capture_version(self) -> tuple:
        """Describe rule declarations and coefficient storage for captures."""
        model = getattr(self, "_kernel_model", None)
        revision = None
        if model is not None:
            from ..kernels.plasticity import (
                plasticity_template_names,
                plasticity_template_revision,
            )

            if model in plasticity_template_names():
                revision = plasticity_template_revision(model)
        return (
            model,
            revision,
            tuple(self.state_domains.items()),
            tuple(self.parameter_domains.items()),
            tuple(self.signals.items()),
            tuple(
                (
                    name,
                    value.shape,
                    value.stride(),
                    value.dtype,
                    value.device,
                    value.data_ptr(),
                )
                for name, value in self.named_buffers(recurse=False)
                if name not in self._memories_rv
            ),
        )


class PlasticityRule(SynapticRule):
    """Own persistent learning state while leaving weights in the connection.

    Subclasses register fixed coefficients with ``register_buffer``, declare
    memories with ``register_memory`` in their constructor, and implement
    :meth:`single_step_forward`. The transition updates only rule memories
    and returns a canonical per-edge weight increment. :meth:`update` applies
    that increment and clips once, after propagation has used the old weights.

    ``state_domains`` declares every registered memory's leading domain.
    Register a scalar per domain member with ``sizes=()``; the connection
    prepends the domain size during binding. Explicit sizes describe trailing
    local dimensions. No separate state-initialization hook is required.
    Coefficients default to scalar or input-edge arrays, sorted once during
    binding; ``parameter_domains`` declares post/segment arrays explicitly.
    Memories follow the standard MemoryModule reset, device, dtype and
    serialization lifecycle. Native execution requires an explicitly supported
    template; subclassing a built-in rule does not inherit its kernel binding.

    Args:
        w_min: Finite lower weight bound, scalar or input-edge array.
        w_max: Finite upper weight bound, scalar or input-edge array.
        signals: Named live neuron-state or external signal declarations.
    """

    def __init__(
        self,
        w_min: float | Tensor = 0.0,
        w_max: float | Tensor = 1.0,
        *,
        signals: Mapping[str, PostState | ExternalSignal] | None = None,
    ):
        super().__init__()
        # Resolve explicit concrete-class opt-in before any compiled forward.
        # Repeated mappingproxy reads during tracing become invalid when
        # preparation mutates module dictionaries.
        self._kernel_model = type(self).__dict__.get("kernel_model")
        self.signals = dict(signals or {})
        if any(
            not name.isidentifier()
            or not isinstance(value, (PostState, ExternalSignal))
            for name, value in self.signals.items()
        ):
            raise ValueError("Declare named PostState or ExternalSignal inputs")
        self._register_coefficient("w_min", w_min)
        self._register_coefficient("w_max", w_max)

    def _validate_binding(self, edge_map: EdgeMap, layout: PlasticityLayout) -> None:
        """Validate weight bounds and declared coefficient domains."""
        if any(
            self.parameter_domains.get(name, "edge") != "edge"
            for name in ("w_min", "w_max")
        ):
            raise ValueError("Weight bounds must be scalar or input-edge arrays")

    def _validate_weight_binding(self, edge_map: EdgeMap, weight: Tensor) -> None:
        """Require independent writable canonical weights for STDP."""
        _validate_weight_storage(weight, edge_map.n_new)

    def _validate_coefficients(self) -> None:
        """Check relationships after coefficients are on the weight device."""
        super()._validate_coefficients()
        if not (self.w_min <= self.w_max).all():
            raise ValueError("w_min must not exceed w_max")

    def update(
        self, weight: Tensor, events: PlasticityEvents, dt: float | Tensor
    ) -> None:
        """Apply one validated online transition and clip its combined
        increment."""
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Online STDP requires no_grad or inference_mode; BPTT is unsupported"
            )
        self._validate_state()
        self.validate_signals(events.signals, weight.device)
        dt = self._time_step(dt, weight)
        delta = self.single_step_forward(weight, events, dt)
        if delta.shape != weight.shape:
            raise ValueError("Plasticity must return one increment per canonical edge")
        weight.copy_(
            torch.maximum(self.w_min, torch.minimum(self.w_max, weight + delta))
        )

    def validate_signals(
        self, values: Mapping[str, Tensor], device: torch.device
    ) -> None:
        """Validate declared live inputs before advancing any rule state."""
        if values.keys() != self.signals.keys():
            raise ValueError(
                f"Plasticity requires signals {tuple(self.signals)}, "
                f"got {tuple(values)}"
            )
        for name, declaration in self.signals.items():
            value = values[name]
            domain = (
                "post" if isinstance(declaration, PostState) else declaration.domain
            )
            shape = self._layout.shape(domain)
            if not isinstance(value, Tensor) or value.device != device:
                raise ValueError(f"{name} must be a tensor on {device}")
            if value.shape != shape:
                raise ValueError(f"{name} must have {domain} shape {shape}")

    @property
    def signal_domains(self) -> tuple[tuple[str, PlasticityDomain], ...]:
        """Look up static domains for live neuron or external learning
        inputs."""
        return tuple(
            (name, "post" if isinstance(source, PostState) else source.domain)
            for name, source in self.signals.items()
        )

    def single_step_forward(
        self, weight: Tensor, events: PlasticityEvents, dt: Tensor
    ) -> Tensor:
        """Update rule memories and return increments without mutating
        weights."""
        raise NotImplementedError


class _PairTraces(PlasticityRule):
    """Share arrival/post trace ownership, ordering and persistence."""

    state_domains = {"pre_trace": "edge", "post_trace": "edge"}

    def _decay_traces(self, dt: Tensor) -> tuple[Tensor, Tensor]:
        """Read strict-past trace snapshots for the concrete rule."""
        raise NotImplementedError

    def _weight_delta(
        self, weight: Tensor, events: PlasticityEvents, pre: Tensor, post: Tensor
    ) -> Tensor:
        """Evaluate the concrete weight law without committing trace state."""
        raise NotImplementedError

    def _configure_trace_scope(
        self, trace_scope: Literal["connection", "neuron"], trace_times: tuple[str, ...]
    ) -> None:
        """Resolve trace domains after registering the rule's time
        constants."""
        if trace_scope not in ("connection", "neuron"):
            raise ValueError("trace_scope must be 'connection' or 'neuron'")
        if trace_scope == "neuron":
            if any(getattr(self, name).ndim for name in trace_times):
                raise ValueError(
                    "Shared traces require scalar time constants; "
                    "use trace_scope='connection' for heterogeneous edge time constants"
                )
            self.state_domains = {"pre_trace": "segment", "post_trace": "post"}
            self.parameter_domains = {
                **self.parameter_domains,
                **dict.fromkeys(trace_times, "scalar"),
            }

        for name in ("pre_trace", "post_trace"):
            self.register_memory(name, 0.0, (), persistent=True)

    @property
    def trace_scope(self) -> Literal["connection", "neuron"]:
        """Look up the declared persistent trace layout."""
        return (
            "neuron" if self.state_domains["pre_trace"] == "segment" else "connection"
        )

    def get_extra_state(self) -> dict[str, bool]:
        """Record the persistent trace layout with the learning memories."""
        return {"neuron_traces": self.trace_scope == "neuron"}

    def set_extra_state(self, state: dict[str, bool]) -> None:
        """Require a matching state layout instead of converting traces."""
        if state != {"neuron_traces": self.trace_scope == "neuron"}:
            raise ValueError(
                "Construct the rule with the checkpoint's trace_scope before loading"
            )

    def single_step_forward(
        self, weight: Tensor, events: PlasticityEvents, dt: Tensor
    ) -> Tensor:
        """Compute a pair update from old traces, then advance their policy."""
        x, y = self._decay_traces(dt)
        if self.trace_scope == "neuron":
            if events.segment_index is None or events.post_index is None:
                raise ValueError("Shared traces require segment and post indices")
            delta = self._weight_delta(
                weight,
                events,
                x[events.segment_index],
                y[events.post_index],
            )
        else:
            delta = self._weight_delta(weight, events, x, y)
        self._advance_traces(events, x, y)
        return delta

    def _advance_traces(
        self, events: PlasticityEvents, pre: Tensor, post: Tensor
    ) -> None:
        """Accumulate current events after evaluating the weight increment."""
        arrival, post_event = events.arrival, events.post_spike
        if self.trace_scope == "neuron":
            if events.segment_arrival is None or events.post_activity is None:
                raise ValueError("Shared traces require domain-owned activity")
            arrival, post_event = events.segment_arrival, events.post_activity
        self.pre_trace.copy_((pre + arrival).reshape_as(self.pre_trace))
        self.post_trace.copy_((post + post_event).reshape_as(self.post_trace))


class PairSTDP(_PairTraces):
    r"""Apply additive pair STDP using strictly earlier arrival traces.

    A presynaptic event is a spike arriving at an edge, including its integer
    delay. At each step, decay both traces, compute both weight increments
    from those old traces, clip their sum once, then increment the traces:

    .. math::
        x^-_e = e^{-dt/\tau_{+,e}} x_e,\quad
        y^-_e = e^{-dt/\tau_{-,e}} y_e

        \Delta w_e = -A_{-,e}a_e y^-_e + A_{+,e}z_{j(e)}x^-_e

    Coincident events do not pair with each other; both may still pair with
    earlier events. Propagation uses the weight before this step's update.
    Edge-local traces allow heterogeneous time constants and independent
    initial values even among edges sharing a source, delay, or target.
    The shared layout instead declares one persistent trace per event stream;
    it does not compress or overwrite independently initialized edge traces.

    The connection owns this module after binding. Reset clears traces, not
    learned weights. Traces and hyperparameters are included in state_dict.
    Online mutation is independent of BPTT: execute under torch.no_grad() or
    torch.inference_mode(). Shared-weight minibatch learning and simultaneous
    optimizer updates are not defined by this rule.

    Args:
        a_plus: Nonnegative LTP amplitude, scalar or input-edge array.
        a_minus: Nonnegative LTD amplitude, scalar or input-edge array.
        tau_plus: Positive pre-trace time constant in the same units as dt.
        tau_minus: Positive post-trace time constant.
        w_min: Lower weight bound, scalar or input-edge array.
        w_max: Upper weight bound, scalar or input-edge array.
        trace_scope: ``connection`` permits independent per-edge traces and time
            constants. ``neuron`` owns one pre trace per source/delay segment
            and one post trace per neuron; both time constants must be scalar.
            Amplitudes and bounds may remain heterogeneous. Construct the same
            layout when restoring a checkpoint.
    """

    def __init__(
        self,
        a_plus: float | Tensor = 0.01,
        a_minus: float | Tensor = 0.012,
        tau_plus: float | Tensor = 20.0,
        tau_minus: float | Tensor = 20.0,
        w_min: float | Tensor = 0.0,
        w_max: float | Tensor = 1.0,
        *,
        trace_scope: Literal["connection", "neuron"] = "connection",
    ):
        super().__init__(w_min=w_min, w_max=w_max)
        for name, value in (
            ("a_plus", a_plus),
            ("a_minus", a_minus),
            ("tau_plus", tau_plus),
            ("tau_minus", tau_minus),
        ):
            self._register_coefficient(
                name,
                value,
                positive=name.startswith("tau_"),
                nonnegative=name.startswith("a_"),
            )
        self._configure_trace_scope(trace_scope, ("tau_plus", "tau_minus"))

    kernel_model = "pair_stdp"

    def _decay_traces(self, dt: Tensor) -> tuple[Tensor, Tensor]:
        """Read the two strict-past exponential snapshots."""
        return (
            self.pre_trace.reshape(-1) * torch.exp(-dt / self.tau_plus),
            self.post_trace.reshape(-1) * torch.exp(-dt / self.tau_minus),
        )

    def _weight_delta(
        self, weight: Tensor, events: PlasticityEvents, pre: Tensor, post: Tensor
    ) -> Tensor:
        """Compute an edge weight law from strict-past pair snapshots."""
        return (
            self.a_plus * events.post_spike * pre - self.a_minus * events.arrival * post
        )


@dataclass(frozen=True)
class ModulatedSTDPOptions:
    """Configure pair eligibility and its conversion into weight changes.

    Args:
        a_plus: Nonnegative potentiating eligibility impulse amplitude.
        a_minus: Nonnegative depressing eligibility impulse amplitude.
        tau_plus: Positive presynaptic trace time constant.
        tau_minus: Positive postsynaptic trace time constant.
        tau_eligibility: Positive eligibility decay time constant.
        eta: Nonnegative learning rate per unit simulation time.
        baseline: Modulator value producing zero weight drift.
    """

    a_plus: float | Tensor = 0.01
    a_minus: float | Tensor = 0.012
    tau_plus: float | Tensor = 20.0
    tau_minus: float | Tensor = 20.0
    tau_eligibility: float | Tensor = 1000.0
    eta: float | Tensor = 0.001
    baseline: float | Tensor = 0.0


class ModulatedSTDP(PairSTDP):
    r"""Convert persistent pair eligibility into modulated weight changes.

    The caller supplies a modulator held constant over each simulation bin.
    Existing eligibility decays and contributes its exact integral to the
    weight update. Current pair events add eligibility at the bin's end:

    .. math::
        p_e &= A_{+,e}z_{j(e)}x_e^- - A_{-,e}a_e y_e^-\\
        \Delta w_e &= \eta_e(m_e-b_e)c_e\tau_{c,e}
            (1-e^{-dt/\tau_{c,e}})\\
        c_e' &= e^{-dt/\tau_{c,e}}c_e+p_e

    Pair traces and coincident events follow :class:`PairSTDP`. A reward in
    the pairing bin does not act on that newly created eligibility; a later
    reward can change weights even without any spikes. Propagation uses old
    weights, followed by one combined, clipped learning update. Eligibility
    evolves while the modulator equals its baseline. The signal is supplied
    directly; this rule does not filter dopamine or create a volume transmitter.

    Eligibility is always canonical-edge memory. Pair traces may be shared
    when their time constants are scalar. Reset clears all three memories
    while preserving weights; checkpoints include memories and coefficients.
    Online execution requires ``no_grad`` or ``inference_mode``.

    Args:
        options: Pair amplitudes, decay times and modulation coefficients.
            Coefficients may be scalars or input-edge arrays.
        modulator: Required caller-supplied scalar, edge, post or segment
            signal, named ``modulator`` in ``plasticity_signals``.
        w_min: Lower weight bound, scalar or input-edge array.
        w_max: Upper weight bound, scalar or input-edge array.
        trace_scope: ``connection`` or ``neuron`` pair-trace ownership.
            Eligibility always belongs to each canonical connection edge.

    Notes:
        This defines a binned three-factor rule, inspired by eligibility-based
        dopamine STDP. It does not claim NEST's dopamine filtering or precise
        event-integration semantics. Native kernel support is declared
        separately rather than inherited from the pair rule.
    """

    kernel_model = "modulated_stdp"

    def __init__(
        self,
        options: ModulatedSTDPOptions = ModulatedSTDPOptions(),
        *,
        modulator: ExternalSignal = ExternalSignal(),
        w_min: float | Tensor = 0.0,
        w_max: float | Tensor = 1.0,
        trace_scope: Literal["connection", "neuron"] = "connection",
    ):
        if not isinstance(modulator, ExternalSignal):
            raise TypeError("ModulatedSTDP requires an ExternalSignal modulator")
        super().__init__(
            options.a_plus,
            options.a_minus,
            options.tau_plus,
            options.tau_minus,
            w_min,
            w_max,
            trace_scope=trace_scope,
        )
        self.signals = {"modulator": modulator}
        self.state_domains = {**self.state_domains, "eligibility": "edge"}
        self._register_coefficient(
            "tau_eligibility", options.tau_eligibility, positive=True
        )
        self._register_coefficient("eta", options.eta, nonnegative=True)
        self._register_coefficient("baseline", options.baseline)
        self.register_memory("eligibility", 0.0, (), persistent=True)

    def get_extra_state(self) -> dict[str, object]:
        """Record trace layout and the caller's teaching-signal convention."""
        signal = self.signals["modulator"]
        return {
            **super().get_extra_state(),
            "modulator": (signal.domain, signal.time_first),
        }

    def set_extra_state(self, state: dict[str, object]) -> None:
        """Require the checkpoint's trace and teaching-signal declarations."""
        if state != self.get_extra_state():
            raise ValueError(
                "Construct ModulatedSTDP with the checkpoint's trace_scope "
                "and modulator declaration before loading"
            )

    def single_step_forward(
        self, weight: Tensor, events: PlasticityEvents, dt: Tensor
    ) -> Tensor:
        """Integrate old eligibility before adding the current pair impulse."""
        signal = events.signals["modulator"]
        domain = self.signals["modulator"].domain
        if domain in ("post", "segment"):
            index = events.post_index if domain == "post" else events.segment_index
            if index is None:
                raise ValueError(f"Modulator requires canonical {domain} indices")
            signal = signal[index]
        eligibility = self.eligibility.reshape(-1)
        # expm1 retains accuracy when dt is much shorter than the eligibility
        # decay time; pair impulses enter only after this bin's integral.
        fraction = -torch.expm1(-dt / self.tau_eligibility)
        delta = (
            self.eta
            * (signal - self.baseline)
            * eligibility
            * self.tau_eligibility
            * fraction
        )
        impulse = super().single_step_forward(weight, events, dt)
        self.eligibility.copy_(
            (eligibility * (1 - fraction) + impulse).reshape_as(self.eligibility)
        )
        return delta


class InhibitorySTDP(_PairTraces):
    r"""Balance inhibitory strength with symmetric spike pairing.

    Use the Vogels--Sprekeler homeostatic law with strictly past arrival/post
    traces and one combined clipping operation. For polarity :math:`p_e`,
    learning rate :math:`\eta_e` and presynaptic offset :math:`\alpha_e`:

    .. math::
        x^-_e = e^{-dt/\tau_e}x_e,\qquad
        y^-_e = e^{-dt/\tau_e}y_e

        \Delta w_e = p_e\eta_e
        \left[a_e(y^-_e-\alpha_e)+z_{j(e)}x^-_e\right]

    Increment traces only after reading both old snapshots. Coincident spikes
    do not pair with one another, but the arrival still applies its constant
    weakening. Silent steps only decay traces. Delays refer to actual arrivals,
    not emission times. Propagation consumes the old weight before learning.

    ``polarity=-1`` represents negative inhibitory current weights; ``+1``
    represents positive inhibitory conductances. Bounds must retain that sign.
    Polarity does not infer receptor identity or select which edges learn;
    use zero eta for edges that should remain static. Reset clears traces,
    while learned weights remain in the connection. This discrete ordering is
    explicit and does not claim timestamp-level equivalence with NEST.
    For independent binary spikes at finite dt, the unclipped drift changes
    sign at rate ``alpha * expm1(dt / tau) / (2 * dt)``. The continuous-time
    target is its small-dt limit; no discretization correction is applied.

    Args:
        eta: Nonnegative learning rate, scalar or input-edge array.
        tau: Positive trace time constant in the same units as dt.
        alpha: Nonnegative constant weakening per arrival. The uncorrelated
            continuous-time rate target is alpha / (2 * tau), in inverse dt units.
            For tau=20 ms and a continuous-time 5 Hz target, use alpha=0.2.
        w_min: Lower signed weight bound, scalar or input-edge array.
        w_max: Upper signed weight bound.
        polarity: Exactly -1 or +1, scalar or input-edge array.
        trace_scope: ``connection`` permits heterogeneous tau; ``neuron`` uses
            segment/post traces and requires scalar tau.

    References:
        Vogels et al., Inhibitory plasticity balances excitation and inhibition
        in sensory pathways and memory networks, Science, 2011.
    """

    kernel_model = "inhibitory_stdp"

    def __init__(
        self,
        eta: float | Tensor = 0.001,
        tau: float | Tensor = 20.0,
        alpha: float | Tensor = 0.2,
        *,
        w_min: float | Tensor = -1.0,
        w_max: float | Tensor = 0.0,
        polarity: float | Tensor = -1.0,
        trace_scope: Literal["connection", "neuron"] = "connection",
    ):
        super().__init__(w_min=w_min, w_max=w_max)
        self._register_coefficient("eta", eta, nonnegative=True)
        self._register_coefficient("tau", tau, positive=True)
        self._register_coefficient("alpha", alpha, nonnegative=True)
        self._register_coefficient("polarity", polarity)
        self._configure_trace_scope(trace_scope, ("tau",))

    def _validate_coefficients(self) -> None:
        """Keep inhibitory sign fixed even when a weight reaches zero."""
        super()._validate_coefficients()
        if not (self.polarity.abs() == 1).all():
            raise ValueError("InhibitorySTDP polarity must be exactly -1 or +1")
        if not (
            (self.polarity * self.w_min >= 0) & (self.polarity * self.w_max >= 0)
        ).all():
            raise ValueError("InhibitorySTDP bounds must agree with polarity")

    def _decay_traces(self, dt: Tensor) -> tuple[Tensor, Tensor]:
        """Reuse one decay factor for both event streams."""
        decay = torch.exp(-dt / self.tau)
        return self.pre_trace.reshape(-1) * decay, self.post_trace.reshape(-1) * decay

    def _weight_delta(
        self, weight: Tensor, events: PlasticityEvents, pre: Tensor, post: Tensor
    ) -> Tensor:
        """Combine symmetric pairing and arrival weakening before clipping."""
        return (
            self.polarity
            * self.eta
            * (events.arrival * (post - self.alpha) + events.post_spike * pre)
        )


class NearestPairSTDP(PairSTDP):
    r"""Pair each arrival or post spike with the nearest earlier opposite event.

    The weight law, arrival delays, strict-past snapshots and one final clamp
    follow :class:`PairSTDP`. Traces replace their value on a current event
    instead of accumulating all past events:

    .. math::
        x_e' = \begin{cases}a_e & a_e\ne0\\x_e^- & \text{otherwise}\end{cases},
        \qquad
        y_e' = \begin{cases}z_j & z_j\ne0\\y_e^- & \text{otherwise}\end{cases}

    ``symmetric`` allows an earlier spike to pair with several opposite events.
    ``restricted`` additionally consumes the opposite trace after pairing:
    a post spike clears the pre trace unless an arrival replaces it in the
    same step, and conversely for the post trace. Each event can then
    contribute at most once to potentiation and once to depression.

    Coincident events only pair with earlier events. Both current events
    establish fresh eligibility after the old snapshots have been consumed.
    Nonzero event amplitudes replace traces; event multiplicities within a
    timestep are not resolved into separate spike times.

    Pairing mode is a persistent buffer, so loading a checkpoint restores the
    policy together with the traces. Reset clears eligibility, not the policy
    or learned weights. These conventions do not cover every nearest-neighbour
    variant or imply identical event timing to another simulator.

    Args:
        a_plus: Nonnegative potentiation amplitude, scalar or input-edge array.
        a_minus: Nonnegative depression amplitude, scalar or input-edge array.
        tau_plus: Positive presynaptic trace time constant, in dt units.
        tau_minus: Positive postsynaptic trace time constant.
        w_min: Lower weight bound, scalar or input-edge array.
        w_max: Upper weight bound, scalar or input-edge array.
        pairing: ``symmetric`` or eligibility-consuming ``restricted`` pairing.
    """

    kernel_model = "nearest_pair_stdp"

    def __init__(
        self,
        a_plus: float | Tensor = 0.01,
        a_minus: float | Tensor = 0.012,
        tau_plus: float | Tensor = 20.0,
        tau_minus: float | Tensor = 20.0,
        w_min: float | Tensor = 0.0,
        w_max: float | Tensor = 1.0,
        *,
        pairing: Literal["symmetric", "restricted"] = "symmetric",
    ):
        if pairing not in ("symmetric", "restricted"):
            raise ValueError("pairing must be 'symmetric' or 'restricted'")
        super().__init__(a_plus, a_minus, tau_plus, tau_minus, w_min, w_max)
        self.register_buffer("restricted", torch.tensor(pairing == "restricted"))

    def _advance_traces(
        self, events: PlasticityEvents, pre: Tensor, post: Tensor
    ) -> None:
        """Replace current traces and optionally consume opposite
        eligibility."""
        pre = torch.where(self.restricted & (events.post_spike != 0), 0, pre)
        post = torch.where(self.restricted & (events.arrival != 0), 0, post)
        self.pre_trace.copy_(
            torch.where(events.arrival != 0, events.arrival, pre).reshape_as(
                self.pre_trace
            )
        )
        self.post_trace.copy_(
            torch.where(events.post_spike != 0, events.post_spike, post).reshape_as(
                self.post_trace
            )
        )


@dataclass(frozen=True)
class PowerPairSTDPOptions:
    """Configure weight-dependent pair amplitudes, traces and exponents.

    Args:
        a_plus: Nonnegative potentiation amplitude in weight units.
        a_minus: Nonnegative depression amplitude in weight units.
        tau_plus: Positive presynaptic trace time constant, in dt units.
        tau_minus: Positive postsynaptic trace time constant.
        mu_plus: Nonnegative upper-bound distance exponent.
        mu_minus: Nonnegative lower-bound distance exponent.

    Each coefficient may be a scalar or an input-edge array. Zero exponents
    recover additive pairing; unit exponents give multiplicative dependence.
    """

    a_plus: float | Tensor = 0.01
    a_minus: float | Tensor = 0.012
    tau_plus: float | Tensor = 20.0
    tau_minus: float | Tensor = 20.0
    mu_plus: float | Tensor = 1.0
    mu_minus: float | Tensor = 1.0


class PowerPairSTDP(PairSTDP):
    r"""Apply a power weight law using the pair rule's arrival trace schedule.

    For normalized weight :math:`q_e`, use the old trace snapshots:

    .. math::
        q_e &= \operatorname{clip}
            \left(\frac{w_e-w_{\min,e}}{w_{\max,e}-w_{\min,e}},0,1\right)\\
        \Delta w_e &=
            A_{+,e} z_j x_e^- (1-q_e)^{\mu_{+,e}}
            - A_{-,e} a_e y_e^- q_e^{\mu_{-,e}}

    Zero exponents use :math:`0^0=1`, recovering :class:`PairSTDP`; unit
    exponents give multiplicative dependence. Equal bounds fix the weight
    after clipping, while traces still evolve. Clipping and coincident-event
    ordering follow the pair rule.

    Bounds use numerical weight order, including negative intervals:
    potentiation moves toward ``w_max`` and depression toward ``w_min``.
    On a negative-only interval, potentiation therefore reduces inhibitory
    magnitude. Receptor identity does not silently change the learning sign.
    Amplitudes are absolute weight increments, not fractions of the bound span.

    Args:
        options: Pair amplitudes, time constants and power exponents.
        w_min: Finite lower weight bound, scalar or input-edge array.
        w_max: Finite upper weight bound, scalar or input-edge array.
    """

    kernel_model = "power_pair_stdp"

    def __init__(
        self,
        options: PowerPairSTDPOptions = PowerPairSTDPOptions(),
        *,
        w_min: float | Tensor = 0.0,
        w_max: float | Tensor = 1.0,
    ):
        super().__init__(
            a_plus=options.a_plus,
            a_minus=options.a_minus,
            tau_plus=options.tau_plus,
            tau_minus=options.tau_minus,
            w_min=w_min,
            w_max=w_max,
        )
        self._register_coefficient("mu_plus", options.mu_plus, nonnegative=True)
        self._register_coefficient("mu_minus", options.mu_minus, nonnegative=True)

    def _weight_delta(
        self, weight: Tensor, events: PlasticityEvents, pre: Tensor, post: Tensor
    ) -> Tensor:
        """Scale the pair terms by normalized distances to their bounds."""
        span = self.w_max - self.w_min
        denominator = torch.where(span > 0, span, torch.ones_like(span))
        q = ((weight - self.w_min) / denominator).clamp(0, 1)
        return self.a_plus * events.post_spike * pre * (1 - q).pow(
            self.mu_plus
        ) - self.a_minus * events.arrival * post * q.pow(self.mu_minus)


@dataclass(frozen=True)
class TripletSTDPOptions:
    """Configure all-to-all pair and triplet amplitudes and trace times.

    Args:
        a_plus: Nonnegative pair potentiation amplitude.
        a_minus: Nonnegative pair depression amplitude.
        a3_plus: Nonnegative triplet potentiation amplitude.
        a3_minus: Nonnegative triplet depression amplitude.
        tau_plus: Fast presynaptic time constant, in dt units.
        tau_minus: Fast postsynaptic time constant.
        tau_x: Slow presynaptic time constant.
        tau_y: Slow postsynaptic time constant.

    Each value may be a scalar or input-edge tensor.
    """

    a_plus: float | Tensor = 0.01
    a_minus: float | Tensor = 0.012
    a3_plus: float | Tensor = 0.005
    a3_minus: float | Tensor = 0.005
    tau_plus: float | Tensor = 20.0
    tau_minus: float | Tensor = 20.0
    tau_x: float | Tensor = 100.0
    tau_y: float | Tensor = 100.0


class TripletSTDP(PlasticityRule):
    r"""Apply all-to-all triplet STDP from strictly earlier arrival traces.

    Fast and slow pre/post traces decay before computing the increment:

    .. math::
        \Delta w_e =
        z_j r_1^- (A_2^+ + A_3^+ o_2^-)
        - a_e o_1^- (A_2^- + A_3^- r_2^-)

    Only after the increment is computed are current arrivals added to
    :math:`r_1,r_2` and current postsynaptic events to :math:`o_1,o_2`.
    Thus coincident events do not pair with one another. Delays act on the
    presynaptic arrival stream, as for :class:`PairSTDP`.

    Args:
        options: Pair/triplet amplitudes and the four trace time constants.
        w_min: Finite lower weight bound, scalar or input-edge array.
        w_max: Finite upper weight bound, scalar or input-edge array.

    References:
        Pfister and Gerstner, Triplets of spikes in a model of spike timing
        dependent plasticity, Journal of Neuroscience, 2006.
    """

    kernel_model = "triplet_stdp"

    def __init__(
        self,
        options: TripletSTDPOptions = TripletSTDPOptions(),
        *,
        w_min: float | Tensor = 0.0,
        w_max: float | Tensor = 1.0,
    ):
        super().__init__(w_min=w_min, w_max=w_max)
        for name in TripletSTDPOptions.__dataclass_fields__:
            self._register_coefficient(
                name,
                getattr(options, name),
                positive=name.startswith("tau_"),
                nonnegative=name.startswith("a"),
            )

        for name in self.state_domains:
            self.register_memory(name, 0.0, (), persistent=True)

    state_domains = {
        "pre_trace": "edge",
        "post_trace": "edge",
        "pre_slow": "edge",
        "post_slow": "edge",
    }

    def single_step_forward(
        self, weight: Tensor, events: PlasticityEvents, dt: Tensor
    ) -> Tensor:
        """Compute pair/triplet contributions before incrementing any trace."""
        pre = self.pre_trace.reshape(-1) * torch.exp(-dt / self.tau_plus)
        post = self.post_trace.reshape(-1) * torch.exp(-dt / self.tau_minus)
        pre_slow = self.pre_slow.reshape(-1) * torch.exp(-dt / self.tau_x)
        post_slow = self.post_slow.reshape(-1) * torch.exp(-dt / self.tau_y)
        delta = events.post_spike * pre * (
            self.a_plus + self.a3_plus * post_slow
        ) - events.arrival * post * (self.a_minus + self.a3_minus * pre_slow)
        self.pre_trace.copy_((pre + events.arrival).reshape_as(self.pre_trace))
        self.post_trace.copy_((post + events.post_spike).reshape_as(self.post_trace))
        self.pre_slow.copy_((pre_slow + events.arrival).reshape_as(self.pre_slow))
        self.post_slow.copy_((post_slow + events.post_spike).reshape_as(self.post_slow))
        return delta


@dataclass(frozen=True)
class VoltageSTDPOptions:
    """Configure voltage-gated potentiation and arrival-triggered depression.

    Args:
        a_plus: Nonnegative LTP scale in weight per voltage squared.
        a_minus: Nonnegative LTD amplitude in weight per voltage per arrival.
        tau_pre: Positive arrival-filter time constant; scalar or input-edge array.
        tau_plus: Positive LTP voltage-filter time constant; scalar or post array.
        tau_minus: Positive LTD voltage-filter time constant; scalar or post array.
        theta_plus: Instantaneous voltage threshold; scalar or post array.
        theta_minus: Filtered voltage threshold; scalar or post array.
        v_rest: Initial/reset voltage of both filters; scalar or post array.

    Time constants use the same units as dt. Both amplitudes may be scalar or
    input-edge arrays. Voltage values must use one consistent voltage unit.
    """

    a_plus: float | Tensor = 0.0001
    a_minus: float | Tensor = 0.001
    tau_pre: float | Tensor = 15.0
    tau_plus: float | Tensor = 7.0
    tau_minus: float | Tensor = 10.0
    theta_plus: float | Tensor = -45.0
    theta_minus: float | Tensor = -70.0
    v_rest: float | Tensor = -70.0


class VoltageSTDP(PlasticityRule):
    r"""Apply a voltage-gated rule with continuous LTP and event-driven LTD.

    This discretizes the unshifted voltage rule of Clopath and Gerstner [1],
    without an additional voltage-filter delay or homeostatic amplitude.
    Let :math:`v_j` be the declared voltage sample, held over the step:

    .. math::
        x_e^- &= e^{-dt/\tau_{\mathrm{pre},e}} x_e\\
        u_{\pm,j}' &= e^{-dt/\tau_{\pm,j}}u_{\pm,j}
            +(1-e^{-dt/\tau_{\pm,j}})v_j\\
        \Delta w_e &= dt\,A_{+,e}x_e^-[v_j-\theta_{+,j}]_+
            [u_{+,j}'-\theta_{-,j}]_+
            -A_{-,e}a_e[u_{-,j}'-\theta_{-,j}]_+\\
        x_e' &= x_e^- + a_e/\tau_{\mathrm{pre},e}

    State filters advance exponentially; LTP uses right-endpoint quadrature.
    The arrival trace has inverse-time units and unit area per arrival.
    A current arrival contributes to LTP starting on the next step. LTD uses
    the updated voltage filter immediately. The combined increment is clipped
    once after old-weight propagation. LTP can continue without any new spike,
    so the pair rule's event-only lazy execution is insufficient.

    Voltage filters are post-owned and updated once, regardless of fanout.
    Reset clears the normalized arrival trace and restores both voltage
    filters to ``v_rest``; it preserves learned weights. All coefficients and
    memories serialize through state_dict. This eager reference does not
    claim numerical identity with NEST's Clopath implementation.

    Args:
        options: Amplitudes, filter times, thresholds and reset voltage.
        voltage: Live neuron state/stage or external postsynaptic voltage.
            The default requires the neuron's charged ``v`` record.
        w_min: Lower weight bound, scalar or input-edge array.
        w_max: Upper weight bound, scalar or input-edge array.

    References:
        [1] Clopath and Gerstner, Voltage and Spike Timing Interact in STDP,
        Frontiers in Synaptic Neuroscience, 2010.
    """

    kernel_model = "voltage_stdp"
    state_domains = {
        "pre_trace": "edge",
        "voltage_plus": "post",
        "voltage_minus": "post",
    }
    parameter_domains = {
        name: "post"
        for name in ("tau_plus", "tau_minus", "theta_plus", "theta_minus", "v_rest")
    }

    def __init__(
        self,
        options: VoltageSTDPOptions = VoltageSTDPOptions(),
        *,
        voltage: PostState | ExternalSignal = PostState("v", stage="charged"),
        w_min: float | Tensor = 0.0,
        w_max: float | Tensor = 1.0,
    ):
        if isinstance(voltage, ExternalSignal) and voltage.domain != "post":
            raise ValueError("External plasticity voltage must use the post domain")
        super().__init__(w_min, w_max, signals={"voltage": voltage})
        for name in VoltageSTDPOptions.__dataclass_fields__:
            self._register_coefficient(
                name,
                getattr(options, name),
                positive=name.startswith("tau_"),
                nonnegative=name.startswith("a_"),
            )

        self.register_memory("pre_trace", 0.0, (), persistent=True)
        for name in ("voltage_plus", "voltage_minus"):
            self.register_memory(name, self._reset_voltage, (), persistent=True)

    def _reset_voltage(self, sizes, *, batch_size=None):
        """Read the current reset buffer, including after load or device
        movement."""
        return self.v_rest.expand(sizes).clone()

    def _validate_coefficients(self) -> None:
        """Require the LTP threshold to exceed the filtered-voltage
        threshold."""
        super()._validate_coefficients()
        if not (self.theta_plus > self.theta_minus).all():
            raise ValueError("theta_plus must exceed theta_minus at every post neuron")

    def single_step_forward(
        self, weight: Tensor, events: PlasticityEvents, dt: Tensor
    ) -> Tensor:
        """Advance held-voltage filters before the strict-past arrival
        update."""
        if events.post_index is None:
            raise ValueError("VoltageSTDP requires canonical postsynaptic indices")
        voltage = events.signals["voltage"]
        pre = self.pre_trace.reshape(-1) * torch.exp(-dt / self.tau_pre)
        plus = self.voltage_plus.reshape(-1) + (
            voltage - self.voltage_plus.reshape(-1)
        ) * (-torch.expm1(-dt / self.tau_plus))
        minus = self.voltage_minus.reshape(-1) + (
            voltage - self.voltage_minus.reshape(-1)
        ) * (-torch.expm1(-dt / self.tau_minus))
        post = events.post_index
        delta = (
            dt
            * self.a_plus
            * pre
            * (voltage - self.theta_plus).clamp_min(0)[post]
            * (plus - self.theta_minus).clamp_min(0)[post]
            - self.a_minus
            * events.arrival
            * (minus - self.theta_minus).clamp_min(0)[post]
        )
        self.pre_trace.copy_(
            (pre + events.arrival / self.tau_pre).reshape_as(self.pre_trace)
        )
        self.voltage_plus.copy_(plus.reshape_as(self.voltage_plus))
        self.voltage_minus.copy_(minus.reshape_as(self.voltage_minus))
        return delta
