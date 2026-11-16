"""Domain-owned plasticity stages using the shared scalar renderers."""

from __future__ import annotations

import ast
import functools
import keyword
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from .backends.engine import (
    _compile,
    _DoubleBufferedState,
    _launch_kernel,
    _Source,
    _tensor_storage,
)
from .backends.renderers import get_runtime


PARAMETERS = ("a_plus", "a_minus", "tau_plus", "tau_minus", "w_min", "w_max")


@dataclass(frozen=True)
class PlasticityStage:
    """Bind a native pointwise helper to a domain-owned learning stage.

    Btorch loads the named inputs, calls the supplied function inside its
    existing kernel and stores the returned values in output order. Functions
    own their math, clipping and precision. They must not launch kernels,
    synchronize blocks or mutate tensor storage themselves.

    Args:
        domain: Domain owning each output: edge, post, segment or scalar.
        reads: Positional input names from state, coefficients, signals or
            weight/arrival/post_event/dt.
        outputs: Names of the updated memories or canonical weight. Return a
            tuple in this order, including for a single output.
        triton: Native ``@triton.jit`` helper, or None if unsupported.
        tilelang: Native TileLang helper accepting scalar expressions, or None.
            It returns a tuple of expressions during kernel construction.
    """

    domain: str
    reads: tuple[str, ...]
    outputs: tuple[str, ...]
    triton: Callable | None = None
    tilelang: Callable | None = None

    def __post_init__(self):
        if self.triton is None and self.tilelang is None:
            raise ValueError("A plasticity stage requires a native backend function")
        if any(
            value is not None and not callable(value)
            for value in (self.triton, self.tilelang)
        ):
            raise TypeError("Plasticity backend functions must be callable")
        if not self.outputs:
            raise ValueError("A plasticity stage must update a memory or weight")


@dataclass(frozen=True)
class PlasticityTrace:
    """Declare exact exponential silent evolution for one additive trace.

    This is a scheduling contract, not a definition of user math. The native
    function must decay the old trace and add the named current event. Btorch
    may then skip silent edges and materialize traces at chunk boundaries.

    Args:
        state: Registered memory name.
        time_constant: Positive time-constant coefficient name.
        event: ``arrival`` or ``post_event``.
    """

    state: str
    time_constant: str
    event: str

    def __post_init__(self):
        if self.event not in ("arrival", "post_event"):
            raise ValueError("A plasticity trace event must be arrival or post_event")


@dataclass(frozen=True)
class PlasticityTemplate:
    """Declare state bindings and native functions for an external rule.

    Args:
        states: Registered memory names in native tensor order.
        parameters: Runtime coefficient names, including bounds used by kernels.
        stages: Native domain-owned stages in execution order.
        state_domains: Explicit memory domains; defaults to edge ownership.
        parameter_domains: Non-edge coefficient domains. Scalars broadcast.
        signals: Named live signal domains matching the MemoryModule rule.
        traces: Optional independent additive exponential trace contracts.
        event_driven: Weights change only on events. Requires one edge stage
            updating weight and all traces. Its native helper receives two
            additional compile-time booleans, include_ltp and include_ltd, after
            the declared inputs. It must gate weight terms with these flags
            while still advancing all traces from the original current events.
        reset_parameters: Reset-only coefficient domains.
        variable_signals: Signals whose domains can specialize with the rule.
    """

    states: tuple[str, ...]
    parameters: tuple[str, ...]
    stages: tuple[PlasticityStage, ...]
    state_domains: tuple[tuple[str, str], ...] = ()
    parameter_domains: tuple[tuple[str, str], ...] = ()
    signals: tuple[tuple[str, str], ...] = ()
    traces: tuple[PlasticityTrace, ...] = ()
    event_driven: bool = False
    reset_parameters: tuple[tuple[str, str], ...] = ()
    variable_signals: tuple[str, ...] = ()


_NATIVE_PLASTICITY: dict[str, PlasticityTemplate] = {}


def _native_alias(model, index, revision=None):
    revision = plasticity_template_revision(model) if revision is None else revision
    return f"_btorch_stdp_{model}_v{revision}_s{index}"


def _native_call(model, index, stage, *, decay="dt", flags=(), revision=None):
    inputs = tuple(decay if name == "dt" else name for name in stage.reads)
    outputs = tuple(
        "next_weight" if name == "weight" else "_stdp_next_" + name
        for name in stage.outputs
    )
    assignment = ", ".join(outputs) + ("," if len(outputs) == 1 else "")
    function = _native_alias(model, index, revision)
    return (f"{assignment} = {function}({', '.join((*inputs, *flags))})",)


def _resolve_native_function(model, backend, index, revision):
    if revision != plasticity_template_revision(model):
        raise RuntimeError("Plasticity registration changed; recreate the kernel")
    function = getattr(_NATIVE_PLASTICITY[model].stages[index], backend, None)
    if function is None:
        raise NotImplementedError(
            f"Plasticity {model!r} has no {backend} function for stage {index}"
        )
    return function


@functools.lru_cache(maxsize=128)
def _native_source(source, backend, model):
    definition = _NATIVE_PLASTICITY.get(model)
    if definition is None:
        return source
    revision = plasticity_template_revision(model)
    lines = ["from btorch.models.kernels.plasticity import _resolve_native_function"]
    for index in range(len(definition.stages)):
        _resolve_native_function(model, backend, index, revision)
        lines.append(
            f"{_native_alias(model, index)} = _resolve_native_function("
            f"{model!r}, {backend!r}, {index}, {revision})"
        )
    return "\n".join(lines) + "\n" + source


def _validate_native_backend(model, backend):
    definition = _NATIVE_PLASTICITY.get(model)
    if definition is not None:
        revision = plasticity_template_revision(model)
        for index in range(len(definition.stages)):
            _resolve_native_function(model, backend, index, revision)


@dataclass(frozen=True, kw_only=True)
class _ExponentialTraces:
    """Declare independent exponential traces without promising event-only
    learning.

    Trace times describe exponential silent evolution. An empty
    ``trace_updates`` uses additive event accumulation. These independent
    transitions may share compact state domains even when another memory or
    weight evolves on silent steps. Event-only traversal requires the separate
    :class:`_ExponentialEvents` contract.

    Args:
        states: Persistent memory names in corresponding trace order.
        trace_times: Positive decay-time coefficient for each trace.
        trace_values: Scalar names for decayed old snapshots.
        trace_events: Arrival/post-event expression for each trace.
        state_outputs: Scalar names for the updated memories.
        trace_updates: Optional explicit next-trace expressions. Empty uses
            decayed old trace plus the corresponding event.
    """

    states: tuple[str, ...]
    trace_times: tuple[str, ...]
    trace_values: tuple[str, ...]
    trace_events: tuple[str, ...]
    state_outputs: tuple[str, ...]
    trace_updates: tuple[str, ...] = ()

    def __post_init__(self):
        count = len(self.states)
        if any(
            len(values) != count
            for values in (
                self.trace_times,
                self.trace_values,
                self.trace_events,
                self.state_outputs,
            )
        ) or (self.trace_updates and len(self.trace_updates) != count):
            raise ValueError("Trace evolution declarations must have matching lengths")

    def decay_statements(self, decay="dt"):
        """Read old trace snapshots without committing any state."""
        return tuple(
            f"{value} = {state} * exp(-({decay}) / {tau})"
            for state, value, tau in zip(
                self.states, self.trace_values, self.trace_times, strict=True
            )
        )

    def trace_statements(self):
        """Advance each trace according to the declared event policy."""
        return tuple(
            f"{output} = {expression}"
            for output, expression in zip(
                self.state_outputs,
                self.trace_updates
                or tuple(
                    f"{value} + {event}"
                    for value, event in zip(
                        self.trace_values, self.trace_events, strict=True
                    )
                ),
                strict=True,
            )
        )


@dataclass(frozen=True, kw_only=True)
class _ExponentialEvents(_ExponentialTraces):
    """Add a weight law that changes only on declared spike events.

    Args:
        potentiation: Nonnegative LTP expression, including its post event.
        depression: Nonnegative LTD expression, including its arrival event.
        weight_statements: Optional intermediate scalar assignments evaluated
            after trace decay and before LTP/LTD and combined clipping.
    """

    potentiation: str
    depression: str
    weight_statements: tuple[str, ...] = ()

    def statements(self, *, decay="dt", include_ltp=True, include_ltd=True):
        """Use old exponential snapshots before applying the event policy."""
        ltp = self.potentiation if include_ltp else "0.0"
        ltd = self.depression if include_ltd else "0.0"
        return (
            *self.decay_statements(decay),
            *self.weight_statements,
            f"delta = {ltp} - {ltd}",
            "next_weight = maximum(w_min, minimum(w_max, weight + delta))",
            *self.trace_statements(),
        )


@dataclass(frozen=True, kw_only=True)
class _NativeEvents(_ExponentialTraces):
    model: str
    stage: PlasticityStage
    revision: int

    def statements(self, *, decay="dt", include_ltp=True, include_ltd=True):
        return _native_call(
            self.model,
            0,
            self.stage,
            decay=decay,
            flags=(repr(include_ltp), repr(include_ltd)),
            revision=self.revision,
        )


@dataclass(frozen=True)
class _PlasticityStage:
    """Declare a transition with one writer per item of its domain.

    Reads name state, coefficients, signals or built-in event inputs.
    Outputs map mutable buffers to scalar results. Stages execute in
    order on the current stream, so a later stage sees earlier committed
    outputs.
    """

    domain: str
    reads: tuple[str, ...]
    statements: tuple[str, ...]
    outputs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _PlasticityTemplate:
    """Describe native math, ownership and optional event-only evolution.

    Unlisted coefficient domains are canonical-edge domains; a one-
    element coefficient always broadcasts. Continuous transitions
    deliberately have no exponential-event contract and cannot enter
    HELION's lazy traversal.

    Args:
        states: Persistent memory names in the rule's native tensor order.
        parameters: Runtime coefficient names, including weight bounds.
        stages: Ordered domain-owned transitions for ordinary execution.
        state_domains: One edge, segment, post or scalar domain per state.
        parameter_domains: Explicit coefficient domains; unspecified is edge.
        signals: Named live signal domains, matching the rule's declarations.
        events: Optional exact event-only contract for sparse/fused execution.
        reset_parameters: Coefficients used only to initialize/reset memories.
        traces: Optional independent exponential trace evolution.
        variable_signals: Signals whose domains may be specialized by the rule.
    """

    states: tuple[str, ...]
    parameters: tuple[str, ...]
    stages: tuple[_PlasticityStage, ...]
    state_domains: tuple[tuple[str, str], ...]
    parameter_domains: tuple[tuple[str, str], ...] = ()
    signals: tuple[tuple[str, str], ...] = ()
    events: _ExponentialEvents | None = None
    reset_parameters: tuple[tuple[str, str], ...] = ()
    traces: _ExponentialTraces | None = None
    variable_signals: tuple[str, ...] = ()

    def __post_init__(self):
        """Reject ambiguous state ownership before any template can launch."""
        state_domains = dict(self.state_domains)
        for names in (self.states, self.parameters, tuple(dict(self.signals))):
            if len(names) != len(set(names)) or any(
                not name.isidentifier() or keyword.iskeyword(name) for name in names
            ):
                raise ValueError("Plasticity declarations need unique identifiers")
        declarations = (*self.states, *self.parameters, *dict(self.signals))
        if len(set(declarations)) != len(declarations):
            raise ValueError(
                "Plasticity states, coefficients and signals must be distinct"
            )
        if set(declarations) & {"weight", "dt", "arrival", "post_event"}:
            raise ValueError("Plasticity declarations must not shadow built-in inputs")
        if any(
            domain not in ("edge", "post", "segment", "scalar")
            for _, domain in (
                *self.state_domains,
                *self.parameter_domains,
                *self.reset_parameters,
                *self.signals,
            )
        ):
            raise ValueError("Unknown plasticity declaration domain")
        if (
            len(state_domains) != len(self.state_domains)
            or len(state_domains) != len(self.states)
            or set(state_domains) != set(self.states)
        ):
            raise ValueError("Each plasticity state needs one declared domain")
        for domains, names in (
            (self.parameter_domains, self.parameters),
            (self.signals, tuple(dict(self.signals))),
            (self.reset_parameters, tuple(dict(self.reset_parameters))),
        ):
            if len(dict(domains)) != len(domains) or set(dict(domains)) - set(names):
                raise ValueError(
                    "Plasticity domain declarations must be unique and named"
                )
        if set(self.parameters) & dict(self.reset_parameters).keys():
            raise ValueError("Runtime and reset-only coefficients must be distinct")
        if set(self.variable_signals) - dict(self.signals).keys():
            raise ValueError("Variable signal domains must name declared signals")
        if self.traces is not None and set(self.traces.states) - set(self.states):
            raise ValueError("Trace evolution must name declared plasticity states")
        for evolution in (self.traces, self.events):
            if evolution is None:
                continue
            if set(evolution.states) - set(self.states) or set(
                evolution.trace_times
            ) - set(self.parameters):
                raise ValueError(
                    "Trace evolution requires declared states and time constants"
                )
        owners = {"weight": "edge", **state_domains}
        inputs = {
            *owners,
            *self.parameters,
            *dict(self.signals),
            "arrival",
            "post_event",
            "dt",
        }
        for stage in self.stages:
            if set(stage.reads) - inputs:
                raise ValueError("Plasticity stages must read declared inputs")
            writes = [name for name, _value in stage.outputs]
            if len(writes) != len(set(writes)) or any(
                owners.get(name) != stage.domain for name in writes
            ):
                raise ValueError(
                    "Plasticity stage outputs must have the stage's writer domain"
                )


def _event_template(parameters, events):
    """Lower the same event equations for separate and HELION execution."""
    return _PlasticityTemplate(
        states=events.states,
        parameters=parameters,
        stages=(
            _PlasticityStage(
                "edge",
                ("weight", *events.states, *parameters, "arrival", "post_event", "dt"),
                events.statements(),
                (
                    ("weight", "next_weight"),
                    *zip(events.states, events.state_outputs, strict=True),
                ),
            ),
        ),
        state_domains=tuple((name, "edge") for name in events.states),
        events=events,
        traces=events,
    )


def _lower_native_template(name, definition, revision):
    if not definition.stages or any(
        not isinstance(stage, PlasticityStage) for stage in definition.stages
    ):
        raise TypeError("Plasticity stages must declare native backend functions")
    traces = None
    if definition.traces:
        if any(
            not isinstance(trace, PlasticityTrace) for trace in definition.traces
        ) or len({trace.state for trace in definition.traces}) != len(
            definition.traces
        ):
            raise ValueError("Each exponential trace must name one distinct memory")
        traces = _ExponentialTraces(
            states=tuple(trace.state for trace in definition.traces),
            trace_times=tuple(trace.time_constant for trace in definition.traces),
            trace_values=tuple(
                "_stdp_old_" + trace.state for trace in definition.traces
            ),
            trace_events=tuple(trace.event for trace in definition.traces),
            state_outputs=tuple(
                "_stdp_next_" + trace.state for trace in definition.traces
            ),
        )
    events = None
    if definition.event_driven:
        if (
            len(definition.stages) != 1
            or definition.stages[0].domain != "edge"
            or traces is None
            or traces.states != definition.states
            or set(definition.stages[0].outputs) != {"weight", *definition.states}
            or not {"weight", "dt", "arrival", "post_event"}.issubset(
                definition.stages[0].reads
            )
        ):
            raise ValueError(
                "Event-driven learning requires one edge stage updating weight "
                "and all declared exponential traces"
            )
        if definition.signals:
            raise NotImplementedError(
                "Event-driven native hooks do not yet bind live signals; "
                "use ordinary stages"
            )
        if not {"w_min", "w_max"}.issubset(definition.parameters):
            raise ValueError(
                "Event-driven learning requires w_min and w_max coefficients"
            )
        events = _NativeEvents(
            **traces.__dict__, model=name, stage=definition.stages[0], revision=revision
        )
    stages = tuple(
        _PlasticityStage(
            stage.domain,
            stage.reads,
            _native_call(
                name,
                index,
                stage,
                revision=revision,
                flags=("True", "True") if events is not None else (),
            ),
            tuple(
                (
                    output,
                    "next_weight" if output == "weight" else "_stdp_next_" + output,
                )
                for output in stage.outputs
            ),
        )
        for index, stage in enumerate(definition.stages)
    )
    return _PlasticityTemplate(
        states=definition.states,
        parameters=definition.parameters,
        stages=stages,
        state_domains=definition.state_domains
        or tuple((state, "edge") for state in definition.states),
        parameter_domains=definition.parameter_domains,
        signals=definition.signals,
        events=events,
        traces=traces,
        reset_parameters=definition.reset_parameters,
        variable_signals=definition.variable_signals,
    )


def stdp_scalar_statements(
    *,
    decay="dt",
    include_ltp=True,
    include_ltd=True,
    model="pair_stdp",
):
    """Return the canonical scalar STDP equations used by all backends.

    ``decay`` is the elapsed physical time since the edge traces were last
    materialized.  Keeping the equations here prevents the fused HELION path
    and the native edge kernel from drifting in their handling of coincident
    pre/post events or heterogeneous parameters.
    """
    events = get_plasticity_template(model).events
    if events is None:
        raise NotImplementedError(
            f"{model!r} has no event-only exponential evolution contract"
        )
    return events.statements(
        decay=decay, include_ltp=include_ltp, include_ltd=include_ltd
    )


_PLASTICITY_TEMPLATES = {
    "pair_stdp": _event_template(
        PARAMETERS,
        _ExponentialEvents(
            states=("pre_trace", "post_trace"),
            trace_times=("tau_plus", "tau_minus"),
            trace_values=("x", "y"),
            trace_events=("arrival", "post_event"),
            potentiation="a_plus * post_event * x",
            depression="a_minus * arrival * y",
            state_outputs=("next_pre", "next_post"),
        ),
    ),
    "triplet_stdp": _event_template(
        (
            "a_plus",
            "a_minus",
            "a3_plus",
            "a3_minus",
            "tau_plus",
            "tau_minus",
            "tau_x",
            "tau_y",
            "w_min",
            "w_max",
        ),
        _ExponentialEvents(
            states=("pre_trace", "post_trace", "pre_slow", "post_slow"),
            trace_times=("tau_plus", "tau_minus", "tau_x", "tau_y"),
            trace_values=("x", "y", "r", "o"),
            trace_events=("arrival", "post_event", "arrival", "post_event"),
            potentiation="post_event * x * (a_plus + a3_plus * o)",
            depression="arrival * y * (a_minus + a3_minus * r)",
            state_outputs=(
                "next_pre_trace",
                "next_post_trace",
                "next_pre_slow",
                "next_post_slow",
            ),
        ),
    ),
}
_PLASTICITY_TEMPLATES["power_pair_stdp"] = _event_template(
    (*PARAMETERS[:-2], "mu_plus", "mu_minus", *PARAMETERS[-2:]),
    replace(
        _PLASTICITY_TEMPLATES["pair_stdp"].events,
        weight_statements=(
            "span = w_max - w_min",
            "normalizer = where(span > 0.0, span, 1.0)",
            "q = maximum(0.0, minimum(1.0, (weight - w_min) / normalizer))",
        ),
        potentiation="a_plus * post_event * x * pow(1.0 - q, mu_plus)",
        depression="a_minus * arrival * y * pow(q, mu_minus)",
    ),
)
_PLASTICITY_TEMPLATES["inhibitory_stdp"] = _event_template(
    ("eta", "tau", "alpha", "polarity", "w_min", "w_max"),
    replace(
        _PLASTICITY_TEMPLATES["pair_stdp"].events,
        trace_times=("tau", "tau"),
        potentiation="polarity * eta * post_event * x",
        depression="polarity * eta * arrival * (alpha - y)",
    ),
)
_PLASTICITY_TEMPLATES["nearest_pair_stdp"] = _event_template(
    (*PARAMETERS, "restricted"),
    replace(
        _PLASTICITY_TEMPLATES["pair_stdp"].events,
        trace_updates=(
            "where(arrival != 0.0, arrival, "
            "where(restricted != 0.0, where(post_event != 0.0, 0.0, x), x))",
            "where(post_event != 0.0, post_event, "
            "where(restricted != 0.0, where(arrival != 0.0, 0.0, y), y))",
        ),
    ),
)
_PAIR_TRACES = _PLASTICITY_TEMPLATES["pair_stdp"].events
_MODULATED_PARAMETERS = (
    *PARAMETERS[:-2],
    "tau_eligibility",
    "eta",
    "baseline",
    *PARAMETERS[-2:],
)
_PLASTICITY_TEMPLATES["modulated_stdp"] = _PlasticityTemplate(
    states=(*_PAIR_TRACES.states, "eligibility"),
    parameters=_MODULATED_PARAMETERS,
    state_domains=tuple(
        (name, "edge") for name in (*_PAIR_TRACES.states, "eligibility")
    ),
    signals=(("modulator", "scalar"),),
    variable_signals=("modulator",),
    traces=_PAIR_TRACES,
    stages=(
        _PlasticityStage(
            "edge",
            (
                "weight",
                *_PAIR_TRACES.states,
                "eligibility",
                *_MODULATED_PARAMETERS,
                "modulator",
                "arrival",
                "post_event",
                "dt",
            ),
            (
                *_PAIR_TRACES.decay_statements(),
                "eligibility_fraction = -expm1(-dt / tau_eligibility)",
                "delta = eta * (modulator - baseline) * eligibility * "
                "tau_eligibility * eligibility_fraction",
                "next_weight = maximum(w_min, minimum(w_max, weight + delta))",
                f"pair_impulse = {_PAIR_TRACES.potentiation} - "
                f"{_PAIR_TRACES.depression}",
                "next_eligibility = eligibility * (1.0 - eligibility_fraction) "
                "+ pair_impulse",
                *_PAIR_TRACES.trace_statements(),
            ),
            (
                ("weight", "next_weight"),
                ("eligibility", "next_eligibility"),
                *zip(_PAIR_TRACES.states, _PAIR_TRACES.state_outputs, strict=True),
            ),
        ),
    ),
)
_PLASTICITY_TEMPLATES["voltage_stdp"] = _PlasticityTemplate(
    states=("pre_trace", "voltage_plus", "voltage_minus"),
    parameters=(
        "a_plus",
        "a_minus",
        "tau_pre",
        "tau_plus",
        "tau_minus",
        "theta_plus",
        "theta_minus",
        "w_min",
        "w_max",
    ),
    state_domains=(
        ("pre_trace", "edge"),
        ("voltage_plus", "post"),
        ("voltage_minus", "post"),
    ),
    parameter_domains=tuple(
        (name, "post")
        for name in ("tau_plus", "tau_minus", "theta_plus", "theta_minus")
    ),
    signals=(("voltage", "post"),),
    reset_parameters=(("v_rest", "post"),),
    stages=(
        _PlasticityStage(
            "post",
            ("voltage_plus", "voltage_minus", "voltage", "tau_plus", "tau_minus", "dt"),
            (
                "next_plus = voltage_plus + (voltage - voltage_plus) * "
                "(-expm1(-dt / tau_plus))",
                "next_minus = voltage_minus + (voltage - voltage_minus) * "
                "(-expm1(-dt / tau_minus))",
            ),
            (("voltage_plus", "next_plus"), ("voltage_minus", "next_minus")),
        ),
        _PlasticityStage(
            "edge",
            (
                "weight",
                "pre_trace",
                "voltage_plus",
                "voltage_minus",
                "voltage",
                "a_plus",
                "a_minus",
                "tau_pre",
                "theta_plus",
                "theta_minus",
                "w_min",
                "w_max",
                "dt",
                "arrival",
            ),
            (
                "old_pre = pre_trace * exp(-dt / tau_pre)",
                "ltp = dt * a_plus * old_pre * maximum(voltage - theta_plus, 0.0) * "
                "maximum(voltage_plus - theta_minus, 0.0)",
                "ltd = a_minus * arrival * maximum(voltage_minus - theta_minus, 0.0)",
                "delta = ltp - ltd",
                "next_weight = maximum(w_min, minimum(w_max, weight + delta))",
                "next_pre = old_pre + arrival / tau_pre",
            ),
            (("weight", "next_weight"), ("pre_trace", "next_pre")),
        ),
    ),
)


_BUILTIN_PLASTICITY = frozenset(_PLASTICITY_TEMPLATES)
_PLASTICITY_REVISIONS: dict[str, int] = {}


def plasticity_template_names() -> tuple[str, ...]:
    """Look up available learning models without importing GPU dependencies."""
    return tuple(_PLASTICITY_TEMPLATES)


def plasticity_template_revision(name: str) -> int:
    """Look up the revision used by captured and compiled recurrent calls."""
    if name not in _PLASTICITY_TEMPLATES:
        raise ValueError(f"Unknown plasticity template {name!r}")
    return _PLASTICITY_REVISIONS.get(name, 1)


def register_plasticity_template(
    name: str, template: PlasticityTemplate, *, replace: bool = False
) -> PlasticityTemplate:
    """Register out-of-tree native functions for Triton and TileLang.

    The concrete rule class opts in with ``kernel_model = name``. Stages
    describe ordinary learning; an exact event-driven trace contract also
    enables sparse event traversal and HELION fusion. Continuous stages use
    the existing supported domain schedule or raise before stepping.

    Args:
        name: Python identifier selecting the learning equations.
        template: Immutable state, coefficient, signal and stage declarations.
        replace: Replace an external registration and increment its revision.
            Recreate compiled callables and standalone graphs afterwards.
            RecurrentNN's graph runner invalidates captures automatically.

    Returns:
        The registered template.

    Raises:
        ValueError: If the name is invalid, built-in or already registered.
        TypeError: If template is not a PlasticityTemplate.
    """
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError("Plasticity template names must be Python identifiers")
    if not isinstance(template, PlasticityTemplate):
        raise TypeError("template must be a PlasticityTemplate")
    if name in _BUILTIN_PLASTICITY:
        raise ValueError(
            f"The built-in plasticity template {name!r} cannot be replaced"
        )
    previous = _PLASTICITY_TEMPLATES.get(name)
    if previous is not None and not replace:
        raise ValueError(f"Plasticity template {name!r} is already registered")
    revision = 1 if previous is None else plasticity_template_revision(name) + 1
    lowered = _lower_native_template(name, template, revision)
    _PLASTICITY_REVISIONS[name] = revision
    _PLASTICITY_TEMPLATES[name] = lowered
    _NATIVE_PLASTICITY[name] = template
    # Render caches resolve names internally. Drop generated metadata, never
    # caller-owned tensors, when an external equation changes.
    for module_name in (__name__, __package__ + ".backends.helion_plasticity"):
        module = sys.modules.get(module_name)
        if module is not None:
            for value in vars(module).values():
                if (
                    callable(value)
                    and getattr(value, "__module__", None) == module_name
                    and hasattr(value, "cache_clear")
                ):
                    value.cache_clear()
    return template


@torch.compiler.disable
def get_plasticity_template(
    model: str,
    state_domains: tuple[tuple[str, str], ...] = (),
    signal_domains: tuple[tuple[str, str], ...] = (),
) -> _PlasticityTemplate:
    """Resolve explicitly implemented equations without inheriting bindings."""
    try:
        template = _PLASTICITY_TEMPLATES[model]
    except KeyError:
        raise NotImplementedError(
            f"No native plasticity template for {model!r}"
        ) from None
    # Model and declared domains are static metadata, just like neuron specs.
    # Resolve their AST outside Dynamo rather than tracing preparation work.
    # Registry values contain OOT callables; treating them as synthetic graph
    # constants gives Dynamo invalid guards after a graph break.
    return _specialize_domains(
        _specialize_signals(template, signal_domains), state_domains
    )


@functools.lru_cache(maxsize=128)
def _specialize_signals(template, signal_domains):
    """Resolve explicitly variable signal domains as static launch metadata."""
    if not signal_domains or not template.variable_signals:
        return template
    requested = dict(signal_domains)
    signals = tuple(
        (
            name,
            requested.get(name, domain)
            if name in template.variable_signals
            else domain,
        )
        for name, domain in template.signals
    )
    if any(
        domain not in ("scalar", "edge", "post", "segment") for _, domain in signals
    ):
        raise ValueError("Unknown native plasticity signal domain")
    return replace(template, signals=signals)


@functools.lru_cache(maxsize=128)
def _specialize_domains(template, state_domains):
    """Assign trace writers without changing the registered learning law.

    Edge weights read old traces; independent segment/post stages
    advance shared traces. The scheduler can combine them using old/next
    snapshots. Eligibility policies depending on the opposite stream
    cannot share a trace merely because its decay constant agrees.
    """
    if not state_domains or dict(state_domains) == dict(template.state_domains):
        return template
    domains = dict(state_domains)
    events = template.traces
    if events is None or set(domains) != set(template.states):
        raise NotImplementedError("This rule has no shared exponential trace contract")
    if any(
        domains[name] != dict(template.state_domains)[name]
        for name in template.states
        if name not in events.states
    ):
        raise NotImplementedError(
            "Only declared independent traces can change ownership"
        )
    if len(template.stages) != 1 or template.stages[0].domain != "edge":
        raise NotImplementedError("Shared event traces require one edge transition")
    stages = {}
    parameters = dict(template.parameter_domains)
    for i, state in enumerate(events.states):
        domain = domains[state]
        if domain == "edge":
            continue
        event = events.trace_events[i]
        if (domain, event) not in (("segment", "arrival"), ("post", "post_event")):
            raise NotImplementedError(
                f"Cannot assign {event!r} trace {state!r} to the {domain!r} domain"
            )
        transition = events.trace_statements()[i]
        names = {
            node.id
            for node in ast.walk(ast.parse(transition.split(" = ", 1)[1]))
            if isinstance(node, ast.Name)
        }
        if names - {events.trace_values[i], event, "where", "maximum", "minimum"}:
            raise NotImplementedError(
                f"Trace {state!r} depends on another event stream or coefficient; "
                "retain its edge ownership"
            )
        tau = events.trace_times[i]
        parameters[tau] = "scalar"
        stage = stages.setdefault(
            domain, {"reads": [], "statements": [], "outputs": []}
        )
        stage["reads"].extend((state, tau, event, "dt"))
        stage["statements"].extend((events.decay_statements()[i], transition))
        stage["outputs"].append((state, events.state_outputs[i]))
    edge = template.stages[0]
    return replace(
        template,
        state_domains=tuple((name, domains[name]) for name in template.states),
        parameter_domains=tuple(parameters.items()),
        stages=(
            replace(
                edge,
                outputs=tuple(
                    output
                    for output in edge.outputs
                    if output[0] == "weight" or domains[output[0]] == "edge"
                ),
            ),
            *(
                _PlasticityStage(
                    domain,
                    tuple(dict.fromkeys(stage["reads"])),
                    tuple(stage["statements"]),
                    tuple(stage["outputs"]),
                )
                for domain, stage in stages.items()
            ),
        ),
    )


def _materialize_equations(model, *, decay="elapsed_dt"):
    template = get_plasticity_template(model)
    if template.events is None:
        raise NotImplementedError(f"{model!r} cannot materialize event-only traces")
    return tuple(
        f"{output} = {state} * exp(-({decay}) / {tau})"
        for state, tau, output in zip(
            template.states,
            template.events.trace_times,
            template.events.state_outputs,
            strict=True,
        )
    )


@dataclass
class PreparedPlasticity:
    """Own learning launches and invocation-local trace snapshots."""

    runtime: object
    stages: tuple
    initial_states: dict
    snapshots: _DoubleBufferedState | None = None

    @property
    def states(self):
        """Look up the latest submitted trace values."""
        return (
            {**self.initial_states, **self.snapshots.current}
            if self.snapshots is not None
            else self.initial_states
        )

    def __call__(self, arrivals, posts, *, records=None, signals=None):
        current = {"arrivals": arrivals, "posts": posts, **(signals or {})}
        current.update(
            (name + "_history", value) for name, value in (records or {}).items()
        )
        for arguments, changing, launch, bindings in self.stages:
            for index, name, size in changing:
                arguments[index] = self.runtime.argument(current[name], size)
            if bindings:
                for index, value in bindings[self.snapshots.steps % 2]:
                    arguments[index] = value
            launch(arguments)
        if self.snapshots is not None:
            self.snapshots.steps += 1

    def finish(self):
        """Commit final traces to persistent storage at the chunk boundary."""
        if self.snapshots is not None:
            self.snapshots.finish()


def prepare_update(
    backend,
    model,
    weight,
    states,
    arrivals,
    posts,
    pre_index,
    post_index,
    delay_index,
    n_delay,
    parameters,
    dt,
    *,
    records=None,
    signals=None,
    state_domains=(),
    signal_domains=(),
    segments=None,
    weight_destination=None,
):
    """Prepare domain-owned transitions for one recurrent invocation.

    Integer topology stays integer; FP64 weights use FP64 arithmetic.
    FP16 and BF16 pointwise arithmetic retain their dtype. Unsupported
    low-precision division/transcendentals promote only that operation and
    round its result immediately.
    Canonical weights and trace buffers stay bound for the invocation.
    The prepared callable accepts current arrivals, posts, records and
    signals with the prepared layouts. Call it after old-weight
    propagation. Validation, source resolution and native launch
    preparation run once.
    An optional flattened ``weight_destination`` and canonical placement map
    receive committed storage values from the weight stage itself, without
    adding a copy or launch. Other stages and static propagation do no work
    for this destination.
    Shared event traces use old/next snapshots to combine their independent
    stages without a global barrier. Read ``prepared.states`` for intermediate
    snapshots and call ``prepared.finish()`` once at the invocation boundary
    to commit persistent storage. Even-length invocations need no trace copy.
    Segment bindings supply ``segment_pre`` and ``segment_delay`` plus either
    ``source_segment_ptr`` for one delay bin or a canonical ``segment_index``
    map for heterogeneous delays. Zero-delay learning needs no per-edge map.
    """
    if backend not in ("triton", "tilelang"):
        raise NotImplementedError("Native STDP supports Triton and TileLang")
    state_domains = tuple(tuple(item) for item in state_domains)
    signal_domains = tuple(tuple(item) for item in signal_domains)
    template = get_plasticity_template(model, state_domains, signal_domains)
    if set(states) != set(template.states) or set(parameters) != set(
        template.parameters
    ):
        raise ValueError("Plasticity buffers disagree with the native template")
    if set(signals or {}) != dict(template.signals).keys():
        raise ValueError("Plasticity signals disagree with the native template")
    domain_sizes = {
        "edge": weight.numel(),
        "post": posts.numel(),
        "scalar": 1,
        "segment": 0 if segments is None else segments["segment_pre"].numel(),
    }
    if (
        "segment"
        in (*dict(template.state_domains).values(), *dict(template.signals).values())
        and segments is None
    ):
        raise ValueError("Segment-owned plasticity requires prepared segment indices")
    domains = {
        "weight": "edge",
        "dt": "scalar",
        **dict(template.state_domains),
        **dict.fromkeys(template.parameters, "edge"),
        **dict(template.parameter_domains),
        **dict(template.signals),
    }
    for name, value in {**states, **parameters, **(signals or {})}.items():
        expected = domain_sizes[domains[name]]
        if value.numel() != expected and not (
            name in template.parameters and value.numel() == 1
        ):
            raise ValueError(
                f"Plasticity {name!r} requires {expected} {domains[name]} values"
            )
    buffers = dict(
        weight=weight,
        arrivals=arrivals,
        posts=posts,
        pre_index=pre_index,
        post_index=post_index,
        dt=dt,
    )
    buffers.update(states)
    shared = (
        template.traces is not None
        and len(template.stages) > 1
        and template.stages[0].domain == "edge"
        and all(stage.domain in ("segment", "post") for stage in template.stages[1:])
    )
    snapshots = (
        _DoubleBufferedState(
            {
                name: states[name]
                for name, domain in template.state_domains
                if domain != "edge"
            }
        )
        if shared
        else None
    )
    if snapshots is not None:
        buffers.update(snapshots.buffers())
    if weight_destination is not None:
        buffers["weight_destination"], buffers["weight_destination_index"] = (
            weight_destination
        )
    if segments is not None:
        buffers.update(segments)
    if delay_index is not None:
        buffers["delay_index"] = delay_index
    buffers.update(parameters)
    buffers.update(signals or {})
    buffers.update(
        {name + "_history": value for name, value in (records or {}).items()}
    )
    runtime = get_runtime(backend)
    stages = []
    dynamic_names = {"arrivals", "posts", *(signals or {})}
    dynamic_names.update(name + "_history" for name in (records or {}))
    groups = (
        (tuple(range(len(template.stages))),)
        if shared
        else tuple((index,) for index in range(len(template.stages)))
    )
    destinations = (
        {} if not shared else {name: "next_" + name for name in snapshots.initial}
    )
    for group in groups:
        count = max(domain_sizes[template.stages[index].domain] for index in group)
        if not count:
            continue
        render = _render_shared if shared else _render
        rendered, declarations = render(
            backend,
            model,
            tuple(domain_sizes.items()) if shared else group[0],
            count,
            n_delay,
            _tensor_storage(buffers),
            tuple((name, value.numel()) for name, value in buffers.items()),
            str(weight.dtype).removeprefix("torch."),
            state_domains,
            signal_domains,
        )
        kernel = _compile(
            backend,
            rendered,
            False,
            weight.dtype != torch.float64,
            device=weight.device,
        )
        arguments = [
            runtime.argument(buffers[name], size) for name, size in declarations
        ]
        changing = tuple(
            (index, name, size)
            for index, (name, size) in enumerate(declarations)
            if name in dynamic_names
        )
        output_names = {
            name for index in group for name, _value in template.stages[index].outputs
        }
        outputs = {destinations.get(name, name) for name in output_names}
        reads = {name for index in group for name in template.stages[index].reads}
        if "weight" in outputs and weight_destination is not None:
            outputs.add("weight_destination")
        outputs.update(name + "_history" for name in output_names)
        bindings = (
            snapshots.bindings(runtime, declarations) if snapshots is not None else ()
        )
        state_slots = tuple(index for index, _value in bindings[0]) if bindings else ()
        launch = runtime.prepare_launch(
            kernel,
            arguments,
            count,
            changing=(*tuple(index for index, _name, _size in changing), *state_slots),
            enable_fp_fusion=False,
            writes=tuple(
                index
                for index, (name, _size) in enumerate(declarations)
                if name in outputs
            ),
            read_writes=tuple(
                index
                for index, (name, _size) in enumerate(declarations)
                if name in outputs and name in reads
            ),
        )
        stages.append((arguments, changing, launch, bindings))

    return PreparedPlasticity(runtime, tuple(stages), states, snapshots)


def update(
    backend,
    model,
    weight,
    states,
    arrivals,
    posts,
    pre_index,
    post_index,
    delay_index,
    n_delay,
    parameters,
    dt,
    *,
    records=None,
    signals=None,
    state_domains=(),
    signal_domains=(),
    segments=None,
):
    """Apply one transition after old-weight propagation.

    Recurrent execution uses :func:`prepare_update` once per chunk. This
    direct entry point is useful for standalone plasticity transitions.
    """
    prepared = prepare_update(
        backend,
        model,
        weight,
        states,
        arrivals,
        posts,
        pre_index,
        post_index,
        delay_index,
        n_delay,
        parameters,
        dt,
        records=records,
        signals=signals,
        state_domains=state_domains,
        signal_domains=signal_domains,
        segments=segments,
    )
    prepared(arrivals, posts, records=records, signals=signals)
    prepared.finish()


def _append_stage(source, template, stage, n_delay, sizes, *, destinations=()):
    """Emit one declared stage without duplicating its equations or
    indexing."""
    domains = {
        "weight": "edge",
        "dt": "scalar",
        "post_event": "post",
        **dict(template.state_domains),
        **dict.fromkeys(template.parameters, "edge"),
        **dict(template.parameter_domains),
        **dict(template.signals),
    }

    def load(local, pointer, index, *, dtype=None):
        source.pointer(pointer, sizes[pointer])
        source.load(local, pointer, index, dtype=dtype)

    if stage.domain == "edge" and any(
        domains.get(name) == "post" for name in stage.reads
    ):
        load("post_item_index", "post_index", "index", dtype="int64")
    if stage.domain == "edge" and any(
        domains.get(name) == "segment" for name in stage.reads
    ):
        if "source_segment_ptr" in sizes:
            # With one delay bin, each connected source owns one segment.
            # Reuse its compact pointer; source IDs need not equal segment IDs
            # when some neurons have no outgoing edges.
            load("pre_item_index", "pre_index", "index", dtype="int64")
            load(
                "segment_item_index",
                "source_segment_ptr",
                "pre_item_index",
                dtype="int64",
            )
        else:
            load("segment_item_index", "segment_index", "index", dtype="int64")
    for name in stage.reads:
        if name == "arrival":
            load(
                "pre_item_index",
                "segment_pre" if stage.domain == "segment" else "pre_index",
                "index",
                dtype="int64",
            )
            if stage.domain == "segment":
                load("delay", "segment_delay", "index", dtype="int64")
                delay = "delay"
            elif "delay_index" in sizes:
                load("delay", "delay_index", "index", dtype="int64")
                delay = "delay"
            else:
                delay = "0"
            load("arrival", "arrivals", f"pre_item_index * {n_delay} + {delay}")
            continue
        domain = domains[name]
        pointer = "posts" if name == "post_event" else name
        if domain == "scalar" or (name in template.parameters and sizes[name] == 1):
            index = "0"
        elif domain == stage.domain:
            index = "index"
        elif stage.domain == "edge" and domain == "post":
            index = "post_item_index"
        elif stage.domain == "edge" and domain == "segment":
            index = "segment_item_index"
        else:
            raise NotImplementedError(
                f"Cannot read {domain}-owned {name!r} from a {stage.domain} stage"
            )
        load(name, pointer, index)
    source.statements(stage.statements)
    destinations = dict(destinations)
    for name, value in stage.outputs:
        destination = destinations.get(name, name)
        source.pointer(destination, sizes[destination])
        source.store(destination, "index", value)
        if name == "weight" and "weight_destination" in sizes:
            # Read the committed storage value so a promoted GEMM operand
            # observes the same FP16/BF16 rounding as the canonical weights.
            load("committed_weight", "weight", "index")
            load(
                "weight_slot",
                "weight_destination_index",
                "index",
                dtype="int64",
            )
            source.pointer("weight_destination", sizes["weight_destination"])
            source.store("weight_destination", "weight_slot", "committed_weight")
        if name + "_history" in sizes:
            source.pointer(name + "_history", sizes[name + "_history"])
            source.store(name + "_history", "index", value)


@functools.lru_cache(maxsize=128)
def _render(
    backend,
    model,
    stage_index,
    n,
    n_delay,
    storage,
    sizes,
    scalar_dtype,
    state_domains,
    signal_domains=(),
):
    """Reuse source for a prepared layout independently of tensor addresses."""
    source = _Source(
        backend, n, 1, 0, storage=storage, precise_exp=True, scalar_dtype=scalar_dtype
    )
    template = get_plasticity_template(model, state_domains, signal_domains)
    _append_stage(source, template, template.stages[stage_index], n_delay, dict(sizes))
    return _native_source(source.render(), backend, model), tuple(source.pointers)


@functools.lru_cache(maxsize=128)
def _render_shared(
    backend,
    model,
    domain_sizes,
    n,
    n_delay,
    storage,
    sizes,
    scalar_dtype,
    state_domains,
    signal_domains=(),
):
    """Combine independent shared trace writers using old/next snapshots."""
    counts = dict(domain_sizes)
    template = get_plasticity_template(model, state_domains, signal_domains)
    source = _Source(
        backend, n, 1, 0, storage=storage, precise_exp=True, scalar_dtype=scalar_dtype
    )
    destinations = tuple(
        (name, "next_" + name)
        for name, domain in template.state_domains
        if domain != "edge"
    )
    for stage in template.stages:
        count = counts[stage.domain]
        if not count:
            continue
        fragment = _Source(
            backend,
            count,
            1,
            0,
            storage=storage,
            precise_exp=True,
            scalar_dtype=scalar_dtype,
        )
        _append_stage(
            fragment, template, stage, n_delay, dict(sizes), destinations=destinations
        )
        source.local_dtypes.update(fragment.local_dtypes)
        for name, size in fragment.pointers:
            source.pointer(name, size)
        if backend == "triton":
            source.statements(
                (
                    f"if tl.program_id(0) * BLOCK < {count}:",
                    f"    mask = index < {count}",
                )
            )
        else:
            source.statements((f"if index < {count}:",))
        source.lines.extend("    " + line for line in fragment.lines)
    return _native_source(source.render(), backend, model), tuple(source.pointers)


@functools.lru_cache(maxsize=128)
def _render_materialize(backend, model, n, storage, sizes, scalar_dtype):
    """Render one edge pass that makes invocation-local lazy traces current."""
    source = _Source(
        backend,
        n,
        1,
        0,
        storage=storage,
        precise_exp=True,
        scalar_dtype=scalar_dtype,
    )
    sizes = dict(sizes)
    for name, size in sizes.items():
        source.pointer(name, size)
    template = get_plasticity_template(model)
    if template.events is None:
        raise NotImplementedError(f"{model!r} has no lazy exponential trace contract")
    for name in template.states:
        source.load(name, name, "index")
    source.load("trace_tick", "trace_tick", "index", dtype="int64")
    source.load("final_step", "stdp_step", "0", dtype="int64")
    source.load("dt", "stdp_dt", "0")
    for name in template.events.trace_times:
        source.load(name, name, "0" if sizes[name] == 1 else "index")
    source.statements(
        (
            "elapsed = final_step - trace_tick",
            "elapsed_dt = elapsed * dt",
            *_materialize_equations(model),
        )
    )
    for name, output in zip(
        template.states, template.events.state_outputs, strict=True
    ):
        source.store(name, "index", output)
    source.local_dtypes.update(
        {"trace_tick": "int64", "final_step": "int64", "elapsed": "int64"}
    )
    return _native_source(source.render(), backend, model), tuple(source.pointers)


def materialize(
    backend,
    model,
    states,
    trace_tick,
    parameters,
    dt,
    steps,
):
    """Materialize lazy edge traces at a chunk boundary in one edge kernel."""
    if backend not in ("triton", "tilelang"):
        raise NotImplementedError(
            "HELION trace materialization supports Triton and TileLang"
        )
    template = get_plasticity_template(model)
    if template.events is None:
        raise NotImplementedError(f"{model!r} has no lazy exponential trace contract")
    sample = states[template.states[0]]
    n_edge = trace_tick.numel()
    if set(states) != set(template.states) or any(
        value.numel() != n_edge for value in states.values()
    ):
        raise ValueError("Trace tensors must have equal edge counts")
    if not n_edge:
        return
    if any(
        parameters[name].numel() not in (1, n_edge)
        for name in template.events.trace_times
    ):
        raise ValueError("Trace time constants must be scalar or edge-sized")
    buffers = {
        **states,
        **{name: parameters[name] for name in template.events.trace_times},
        "trace_tick": trace_tick,
        "stdp_dt": dt,
        "stdp_step": steps,
    }
    sizes = tuple((name, value.numel()) for name, value in buffers.items())
    rendered, declarations = _render_materialize(
        backend,
        model,
        n_edge,
        _tensor_storage(buffers),
        sizes,
        str(sample.dtype).removeprefix("torch."),
    )
    kernel = _compile(
        backend,
        rendered,
        False,
        sample.dtype != torch.float64,
        device=sample.device,
    )
    runtime = get_runtime(backend)
    args = tuple(runtime.argument(buffers[name], size) for name, size in declarations)
    _launch_kernel(
        runtime,
        kernel,
        args,
        n_edge,
        enable_fp_fusion=False,
    )


__all__ = (
    "PARAMETERS",
    "get_plasticity_template",
    "stdp_scalar_statements",
    "update",
    "materialize",
)
