"""Declare named models and fixed forward/reverse stages without an IR."""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .contracts import CellSpec


@dataclass(frozen=True)
class BackendFragments:
    """Store directly executable backend statements for named stages.

    Args:
        forward: Statements in the fixed charge, adapt, fire, reset order.
        backward: Statements in the reverse reset, fire, adapt, charge order.
    """

    forward: tuple[str, ...]
    backward: tuple[str, ...]


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
    """

    name: str
    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    history_names: tuple[str, ...]
    state_extra_dims: tuple[tuple[str, int], ...] = ()
    history_sources: tuple[tuple[str, str], ...] = ()

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
        :class:`CellSpec`."""
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


def residual_template(spec: CellSpec, requested: tuple[str, ...]) -> ResidualTemplate:
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
    if spec.model == "glif3" and "Iasc" not in requested:
        return ResidualTemplate(
            tuple(name for name in template.state_names if name != "Iasc"),
            "Iasc",
            16,
            True,
        )
    return ResidualTemplate(template.state_names)


_MODELS = {
    "lif": (("v",), ("v_threshold", "v_reset", "c_m", "tau")),
    "if": (("v",), ("v_threshold", "v_reset", "c_m")),
    "alif": (
        ("v", "g_k"),
        (
            "v_threshold",
            "v_reset",
            "c_m",
            "g_leak",
            "E_leak",
            "E_k",
            "tau_adapt",
            "dg_k",
        ),
    ),
    "elif": (
        ("v", "g_k"),
        (
            "v_threshold",
            "v_reset",
            "c_m",
            "g_leak",
            "E_leak",
            "E_k",
            "tau_adapt",
            "dg_k",
            "delta_T",
            "v_T",
        ),
    ),
    "glif3": (
        ("v", "Iasc"),
        ("v_threshold", "v_reset", "c_m", "tau", "v_rest", "k", "asc_amps"),
    ),
    "izhikevich": (
        ("v", "u"),
        ("v_threshold", "v_reset", "c_m", "k", "v_rest", "v_peak", "a", "b", "d"),
    ),
    "exponential": (("psc",), ("tau_syn",)),
    "alpha": (("psc", "h"), ("tau_syn", "g_max")),
    "alpha_billeh": (("psc", "h"), ("tau_syn", "syn_decay")),
    "dual_exponential": (("psc", "g_rise", "g_decay"), ("tau_rise", "tau_decay", "a")),
}


_EXTERNAL_MODELS: dict[str, tuple[ModelTemplate, Callable, Callable | None]] = {}
_MODEL_REVISIONS: dict[str, int] = {}


def model_template_revision(name: str) -> int:
    """Return the declaration revision used to validate outstanding
    backwards."""
    if name in _MODELS:
        return 1
    if name in _MODEL_REVISIONS:
        return _MODEL_REVISIONS[name]
    raise ValueError(f"Unknown cell template {name!r}")


def register_model_template(
    template: ModelTemplate,
    *,
    forward_step: Callable,
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
    if not isinstance(template, ModelTemplate):
        raise TypeError("template must be a ModelTemplate")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", template.name):
        raise ValueError("Model names must be lowercase Python identifiers")
    if template.name in _MODELS or (template.name in _EXTERNAL_MODELS and not replace):
        raise ValueError(f"Model template {template.name!r} is already registered")
    if not callable(forward_step) or (
        backward_step is not None and not callable(backward_step)
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
    _EXTERNAL_MODELS[template.name] = template, forward_step, backward_step
    _MODEL_REVISIONS[template.name] = _MODEL_REVISIONS.get(template.name, 0) + 1


def get_reference_callbacks(spec: CellSpec):
    """Look up explicit external recurrences without parsing their code."""
    entry = _EXTERNAL_MODELS.get(spec.model)
    return None if entry is None else entry[1:]


def get_template(spec: CellSpec) -> ModelTemplate:
    """Resolve a named cell and validate its static supported options."""
    if spec.model in _EXTERNAL_MODELS:
        return _EXTERNAL_MODELS[spec.model][0]
    if spec.model not in _MODELS:
        raise NotImplementedError(f"Unsupported cell template: {spec.model!r}")
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
    state_names, parameter_names = _MODELS[spec.model]
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
        spec.model == "alpha_billeh"
        and not isinstance(spec.dt, torch.Tensor)
        and spec.dt != 1
    ):
        raise ValueError("AlphaPSCBilleh requires dt == 1")
    if spec.refractory:
        if spec.model not in ("lif", "if", "alif", "elif", "glif3"):
            raise ValueError(f"{spec.model} does not support a refractory state")
        state_names += ("refractory",)
        parameter_names += ("tau_ref",)
    history_names = state_names
    if "v" in state_names:
        history_names += ("v_charged",)
    if "u" in state_names:
        history_names += ("u_charged",)
    return ModelTemplate(
        spec.model,
        state_names,
        parameter_names,
        history_names,
        (("Iasc", 1),) if spec.model == "glif3" else (),
        tuple(
            (name, name.removesuffix("_charged"))
            for name in history_names
            if name.endswith("_charged")
        ),
    )
