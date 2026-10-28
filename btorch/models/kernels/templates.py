"""Register model-local state, record, transition and reverse contracts."""

import importlib
import math
import pkgutil
import re
from collections.abc import Callable
from dataclasses import dataclass, replace as replace_template

import torch

from .contracts import NeuronSpec


@dataclass(frozen=True)
class RecordCotangent:
    """Join a record adjoint at an explicit reverse stage boundary."""

    name: str


@dataclass(frozen=True)
class BackendFragments:
    """Store directly executable backend statements for named stages.

    Args:
        forward: Statements in the fixed charge, adapt, fire, reset order.
        backward: Ordered scalar assignments interleaved with explicit
            :class:`RecordCotangent` joins for paired reverse composition.
    """

    forward: tuple[str, ...]
    backward: tuple[str | RecordCotangent, ...]

    @property
    def deferred_records(self) -> tuple[str, ...]:
        """Return adjoints whose ordering is declared within reverse stages."""
        return tuple(
            stage.name for stage in self.backward if isinstance(stage, RecordCotangent)
        )


@dataclass(frozen=True)
class PreparedParameter:
    """Declare a reusable computed value and its scalar-template binding.

    Args:
        name: Tensor name returned by the template's preparation callback.
        scalar_name: Scalar register consumed by the existing math fragments.
        shape_source: Declared parameter supplying its logical tensor layout.
    """

    name: str
    scalar_name: str
    shape_source: str


@dataclass(frozen=True)
class ModelTemplate:
    """Declare explicit state and floating parameter ordering.

    Args:
        name: Canonical model name.
        state_names: Module state names in tensor operator order.
        parameter_names: Actually participating floating tensor parameters.
        history_names: Supported post-step or charged records.
        state_extra_dims: Extra trailing state axes beyond input batch/neuron
            axes, as pairs of state names and axis counts.
        history_sources: Shape sources for non-state records, as pairs of
            history names and state names.
        parameter_extra_dims: Extra trailing parameter axes, beyond the shared
            batch/neuron layout, declared like state_extra_dims.
        prepared_parameters: Optional computed tensor declarations shared by
            primitive and recurrent kernel binding.
        prepare_parameters: Optional callback accepting the specification,
            named live parameters and a backward flag. Return declared tensor
            values without changing Module memories or parameter ownership.
        recurrent_checkpoint: Whether this template supports the current paired
            recurrent checkpoint representation. This is a capability, not a
            request to enable checkpointing for every use of the template.
        fragments: Model-local scalar forward/reverse stage builder. None leaves
            execution to explicitly registered tensor callbacks.
        supports_refractory: Permit adding the shared refractory state contract.
        required_dt: Optional exact timestep required by the model equations.
        checkpoint_state: Physical state reconstructed by replay_state.
        checkpoint_interval: Primitive reverse checkpoint block length.
        module_factory: Independent original Torch Module constructor, called on
            CPU with (spec, states, neuron_count); it must not use generated math.
        parameter_attributes: Parameter names mapped to Module storage attributes.
        record_attributes: Auxiliary record names mapped to registered memories.
        replay_state: Ordered scalar reconstruction stage for checkpoint_state.
        reverse_parameter_warps: Declared reverse scheduling hints by parameter,
            used only when input VJPs are omitted by a compatible renderer.
        output_state: State equal to the output, allowing trajectory alias reuse.
        default_precise_division: Accurate division for shorthand backends;
            explicit KernelOptions always take precedence.
        spike_gradient_input: Forward scalar register passed to the spike
            surrogate, if its output adjoint enters only through that
            surrogate. This permits exact support masking without suppressing
            physical-state adjoints or interpreting model-specific equations.
    """

    name: str
    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    history_names: tuple[str, ...]
    state_extra_dims: tuple[tuple[str, int], ...] = ()
    history_sources: tuple[tuple[str, str], ...] = ()
    parameter_extra_dims: tuple[tuple[str, int], ...] = ()
    prepared_parameters: tuple[PreparedParameter, ...] = ()
    prepare_parameters: Callable | None = None
    recurrent_checkpoint: bool = False
    spike_gradient_input: str | None = None
    fragments: Callable | None = None
    supports_refractory: bool = False
    required_dt: float | None = None
    checkpoint_state: str | None = None
    checkpoint_interval: int = 0
    module_factory: Callable | None = None
    parameter_attributes: tuple[tuple[str, str], ...] = ()
    record_attributes: tuple[tuple[str, str], ...] = ()
    replay_state: Callable | None = None
    reverse_parameter_warps: tuple[tuple[str, int], ...] = ()
    output_state: str | None = None
    default_precise_division: bool = False

    def extra_dims(self, name: str) -> int:
        """Return the declared trailing state/parameter axis count."""
        dimensions = dict((*self.state_extra_dims, *self.parameter_extra_dims))
        return dimensions.get(
            name, dimensions.get(dict(self.history_sources).get(name), 0)
        )

    def history_state_name(self, name: str) -> str:
        """Look up the state supplying a declared history's shape.

        Post-step states use their own shape; auxiliary records use the
        explicit history_sources declaration.

        Args:
            name: Declared post-step state or auxiliary history.

        Returns:
            Name of the state supplying the history shape.

        Raises:
            KeyError: If no state shape is declared for the history.
        """
        return name if name in self.state_names else dict(self.history_sources)[name]


@dataclass(frozen=True)
class ResidualTemplate:
    """Declare private storage selected by one generated reverse algorithm.

    Args:
        histories: Full post-step trajectories required by reverse.
        checkpoint_state: State reconstructed from periodic checkpoints.
        checkpoint_interval: Number of steps between saved checkpoints.
        save_output: Retain the forward output tensor for local replay.
    """

    histories: tuple[str, ...]
    checkpoint_state: str | None = None
    checkpoint_interval: int = 0
    save_output: bool = False

    @property
    def encoded(self) -> str:
        """Return stable lowering metadata independent from
        :class:`NeuronSpec`."""
        checkpoint = self.checkpoint_state or ""
        histories = ",".join(self.histories)
        return (
            f"{histories}|{checkpoint}|{self.checkpoint_interval}|"
            f"{int(self.save_output)}"
        )


def decode_residual_template(encoded: str) -> ResidualTemplate:
    """Restore normalized private residual metadata at the operator
    boundary."""
    histories, checkpoint, interval, save_output = encoded.split("|")
    return ResidualTemplate(
        tuple(filter(None, histories.split(","))),
        checkpoint or None,
        int(interval),
        bool(int(save_output)),
    )


def residual_template(spec: NeuronSpec, requested: tuple[str, ...]) -> ResidualTemplate:
    """Select the built-in primitive reverse representation.

    GLIF adaptation is replayed from block checkpoints unless the caller
    already requests its complete trajectory. Other models retain their
    established full-state representation.
    """
    template = get_template(spec)
    if spec.surrogate == "poisson" and spec.poisson.spiking:
        # Reverse reuses the sampled raster. Replaying random draws would change
        # both reset semantics and the user's global generator continuation.
        return ResidualTemplate(template.state_names, save_output=True)
    if (
        template.checkpoint_state is not None
        and template.checkpoint_state not in requested
    ):
        return ResidualTemplate(
            tuple(
                name
                for name in template.state_names
                if name != template.checkpoint_state
            ),
            template.checkpoint_state,
            template.checkpoint_interval,
            True,
        )
    return ResidualTemplate(template.state_names)


_TEMPLATES: dict[str, ModelTemplate] = {}
_EXTERNAL_MODELS: dict[str, tuple[ModelTemplate, Callable, Callable | None]] = {}
_DEFINITIONS_LOADED = False


def _load_definitions():
    global _DEFINITIONS_LOADED
    if _DEFINITIONS_LOADED:
        return
    _DEFINITIONS_LOADED = True
    from . import definitions

    for module in pkgutil.iter_modules(definitions.__path__):
        importlib.import_module(f"{definitions.__name__}.{module.name}")


def has_generated_model(name: str) -> bool:
    """Report whether a registered definition supplies GPU mathematics."""
    _load_definitions()
    return name in _TEMPLATES and _TEMPLATES[name].fragments is not None


_MODEL_REVISIONS: dict[str, int] = {}


def model_template_revision(name: str) -> int:
    """Return the declaration revision used to validate outstanding
    backwards."""
    _load_definitions()
    if name in _MODEL_REVISIONS:
        return _MODEL_REVISIONS[name]
    raise ValueError(f"Unknown cell template {name!r}")


def register_model_template(
    template: ModelTemplate,
    *,
    forward_step: Callable | None = None,
    backward_step: Callable | None = None,
    replace: bool = False,
) -> None:
    """Register explicit model declarations and trusted one-step reference
    VJPs.

    Args:
        template: Names, tensor ordering, history and extra state dimensions.
        forward_step: Trusted tensor recurrence returning output, states, records.
        backward_step: Trusted explicit first-order reverse recurrence. None
            permits inference with the reference callback; training requires
            this callback or an independently registered backend VJP.
        replace: Permit replacing an existing external declaration.

    Raises:
        ValueError: When names or declaration are inconsistent or duplicate.
        TypeError: When reference callbacks are not callable.
    """
    _load_definitions()
    if not isinstance(template, ModelTemplate):
        raise TypeError("template must be a ModelTemplate")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", template.name):
        raise ValueError("Model names must be lowercase Python identifiers")
    if template.name in _TEMPLATES and not replace:
        raise ValueError(f"Model template {template.name!r} is already registered")
    if (
        (forward_step is None and template.fragments is None)
        or (forward_step is not None and not callable(forward_step))
        or (backward_step is not None and not callable(backward_step))
    ):
        raise TypeError("forward_step and any supplied backward_step must be callable")
    for names in (
        template.state_names,
        template.parameter_names,
        template.history_names,
    ):
        if len(names) != len(set(names)) or any(
            not name.isidentifier() for name in names
        ):
            raise ValueError("Tensor declarations require unique identifier names")
    if not set(template.state_names).issubset(template.history_names):
        raise ValueError("Every state must be available as a post-step history")
    if len(dict(template.state_extra_dims)) != len(template.state_extra_dims):
        raise ValueError("Extra state dimension declarations must be unique")
    for name, ndim in template.state_extra_dims:
        if name not in template.state_names or not isinstance(ndim, int) or ndim < 0:
            raise ValueError("Invalid extra state dimension declaration")
    sources = dict(template.history_sources)
    if len(sources) != len(template.history_sources):
        raise ValueError("History shape declarations must be unique")
    if any(
        name not in template.history_names or source not in template.state_names
        for name, source in template.history_sources
    ):
        raise ValueError(
            "History shape sources must name declared histories and states"
        )
    for name in template.history_names:
        if (
            name not in template.state_names
            and sources.get(name) not in template.state_names
        ):
            raise ValueError(f"History {name!r} needs an explicit state shape source")
    for bindings, names, kind in (
        (template.parameter_attributes, template.parameter_names, "parameter"),
        (template.record_attributes, template.history_names, "record"),
    ):
        if len(dict(bindings)) != len(bindings) or any(
            name not in names or not attribute.isidentifier()
            for name, attribute in bindings
        ):
            raise ValueError(f"Invalid {kind} attribute bindings for {template.name}")
    if (
        template.output_state is not None
        and template.output_state not in template.state_names
    ):
        raise ValueError("output_state must name a physical state")
    if template.checkpoint_state is not None and (
        template.checkpoint_state not in template.state_names
        or template.checkpoint_interval < 1
        or not callable(template.replay_state)
    ):
        raise ValueError(
            "Checkpoint state requires a positive interval and replay stage"
        )
    if template.required_dt is not None and (
        not math.isfinite(template.required_dt) or template.required_dt <= 0
    ):
        raise ValueError("required_dt must be finite and positive")
    _TEMPLATES[template.name] = template
    if forward_step is not None:
        _EXTERNAL_MODELS[template.name] = template, forward_step, backward_step
    else:
        _EXTERNAL_MODELS.pop(template.name, None)
    _MODEL_REVISIONS[template.name] = _MODEL_REVISIONS.get(template.name, 0) + 1


def get_reference_callbacks(spec: NeuronSpec):
    """Look up explicit external recurrences without parsing their code."""
    entry = _EXTERNAL_MODELS.get(spec.model)
    return None if entry is None else entry[1:]


def prepare_parameters(spec, parameters, *, backward=False):
    """Prepare template-declared values once for the requested execution
    pass."""
    template = get_template(spec)
    if template.prepare_parameters is None:
        return {}
    return template.prepare_parameters(
        spec, dict(zip(template.parameter_names, parameters)), backward=backward
    )


def get_template(spec: NeuronSpec) -> ModelTemplate:
    """Resolve a named cell and validate its static supported options."""
    _load_definitions()
    if spec.model not in _TEMPLATES:
        raise NotImplementedError(f"Unsupported cell template: {spec.model!r}")
    template = _TEMPLATES[spec.model]
    if template.fragments is None:
        return template
    if spec.surrogate not in (
        "sigmoid",
        "atan",
        "atan_approx",
        "erf",
        "superspike",
        "triangle",
        "poisson",
    ):
        raise NotImplementedError(f"Unsupported surrogate: {spec.surrogate!r}")
    if (spec.surrogate == "poisson") != (spec.poisson is not None):
        raise ValueError("Poisson surrogate and options must be selected together")
    if spec.surrogate_width is not None and (
        not math.isfinite(spec.surrogate_width)
        or spec.surrogate_width <= 0
        or spec.surrogate == "poisson"
    ):
        raise ValueError(
            "surrogate_width requires a finite positive deterministic support"
        )
    state_names, parameter_names = template.state_names, template.parameter_names
    if spec.poisson is not None:
        if "v" not in state_names and "u" not in state_names:
            raise NotImplementedError(
                "Poisson lowering requires a built-in neuron template"
            )
        if not isinstance(spec.poisson.spiking, bool):
            raise TypeError("Poisson spiking must be a bool")
        values = (
            spec.poisson.tau,
            spec.poisson.rho,
            spec.poisson.leak,
            spec.poisson.k,
            spec.damping,
        )
        if any(not isinstance(value, (float, int)) for value in values):
            raise TypeError("Poisson options must be non-trainable Python scalars")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Poisson options must be finite")
        if spec.poisson.tau <= 0:
            raise ValueError("Poisson tau must be positive")
    if not isinstance(spec.dt, torch.Tensor) and (
        not math.isfinite(spec.dt) or spec.dt <= 0
    ):
        raise ValueError("dt must be finite and positive")
    if (
        template.required_dt is not None
        and not isinstance(spec.dt, torch.Tensor)
        and spec.dt != template.required_dt
    ):
        raise ValueError(f"{template.name} requires dt == {template.required_dt:g}")
    if not spec.refractory:
        return template
    if not template.supports_refractory:
        raise ValueError(f"{spec.model} does not support a refractory state")
    return replace_template(
        template,
        state_names=state_names + ("refractory",),
        parameter_names=parameter_names + ("tau_ref",),
        history_names=template.history_names + ("refractory",),
    )
