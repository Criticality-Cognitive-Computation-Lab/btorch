"""Prepare stateful modules for pure tensor sequence operators."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from functools import lru_cache

import torch
from torch import Tensor

from .options import KernelOptions, normalize_backend


_connection_contexts = {}


def _model_layout(module):
    from .templates import model_template_revision

    # A subclass changes equations unless it explicitly redeclares its binding.
    model = type(module).__dict__.get("kernel_model")
    if model is None:
        raise NotImplementedError(
            f"No cell backend template for {type(module).__name__}"
        )
    lookup = (
        _cached_model_layout.__wrapped__
        if torch.compiler.is_compiling()
        else _cached_model_layout
    )
    return lookup(
        model, getattr(module, "_use_refractory", False), model_template_revision(model)
    )


@lru_cache(maxsize=128)
def _cached_model_layout(model, refractory, revision):
    from .contracts import NeuronSpec
    from .templates import get_template

    template = get_template(NeuronSpec(model, refractory=refractory))
    return model, template.state_names, template.parameter_names


@lru_cache(maxsize=128)
def _cell_records(model, refractory, revision):
    from .contracts import NeuronSpec
    from .templates import get_template

    return get_template(NeuronSpec(model, refractory=refractory)).record_attributes


def validate_module(module, options: KernelOptions) -> None:
    """Reject modules without an explicitly bound sequence implementation."""
    # Composite modules own their sequence composition (for example, a
    # conductance PSC needs the voltage sequence).  Require the hook on the
    # concrete class, just as ``kernel_model`` is required below, so a
    # subclass that changes equations cannot silently reuse an incompatible
    # adapter.
    if type(module).__dict__.get("_kernel_sequence") is None:
        _model_layout(module)
    if options.merge_synapse:
        raise ValueError(
            "merge_synapse requires an explicitly compatible receptor population"
        )


@contextmanager
def connection_scope(options: KernelOptions):
    """Select the existing sparse runtime before preparing its cached route."""
    from ...sparse.runtime.backend import PreparedBackendContext
    from ...sparse.runtime.method import resolve_method, use_method

    if options.connection == "auto" and options.connection_method == "auto":
        yield
        return
    backend_scope = nullcontext()
    if options.connection != "auto":
        context = _connection_contexts.get(options.connection)
        if context is None:
            context = PreparedBackendContext(options.connection)
            _connection_contexts[options.connection] = context
        backend_scope = context.enter()
    with backend_scope, use_method(resolve_method(options.connection_method)):
        yield


def _cell_options(module):
    """Read mathematical choices without retaining the module or its
    tensors."""
    if hasattr(module, "surrogate_function"):
        from .. import surrogate

        function = module.surrogate_function
        width = None
        if isinstance(function, surrogate.Truncated):
            width = function.support
            while isinstance(function, surrogate.Truncated):
                function = function.surrogate
        if not isinstance(function.damping_factor, (float, int)):
            raise NotImplementedError(
                "Surrogate hyperparameters must be non-trainable scalars"
            )
        if type(function) is surrogate.PoissonRandomSpike:
            from .contracts import PoissonOptions

            poisson_values = (
                function.tau,
                function.rho,
                function.leak,
                function.k,
            )
            if any(not isinstance(value, (float, int)) for value in poisson_values):
                raise NotImplementedError(
                    "Poisson hyperparameters must be non-trainable scalars"
                )
            poisson = PoissonOptions(
                function.spiking,
                float(function.tau),
                float(function.rho),
                float(function.leak),
                float(function.k),
            )
            alpha = 1.0
        else:
            poisson = None
            if not function.spiking:
                raise NotImplementedError(
                    "Deterministic surrogate kernels require spiking=True"
                )
            if not isinstance(function.alpha, (float, int)):
                raise NotImplementedError(
                    "Surrogate hyperparameters must be non-trainable scalars"
                )
            alpha = float(function.alpha)
        return (
            type(function),
            alpha,
            float(function.damping_factor),
            width,
            poisson,
            module.hard_reset,
            module.detach_reset,
            getattr(module, "_use_refractory", False),
        )
    return ()


@lru_cache(maxsize=128)
def _cell_metadata(model, options, revision):
    """Bind only immutable model semantics, never live state or parameters."""
    from .contracts import NeuronSpec
    from .templates import get_template

    kwargs = {"model": model}
    if options:
        from .. import surrogate

        function, alpha, damping, width, poisson, reset, detach, refractory = options
        names = {
            surrogate.Sigmoid: "sigmoid",
            surrogate.ATan: "atan",
            surrogate.Erf: "erf",
            surrogate.SuperSpike: "superspike",
            surrogate.Triangle: "triangle",
            surrogate.ATanApprox: "atan_approx",
            surrogate.PoissonRandomSpike: "poisson",
        }
        if function not in names:
            raise NotImplementedError(
                "Cell kernels require a supported built-in surrogate"
            )
        kwargs.update(
            surrogate=names[function],
            alpha=alpha,
            damping=damping,
            surrogate_width=width,
            poisson=poisson,
            hard_reset=reset,
            detach_reset=detach,
            refractory=refractory,
        )
    spec = NeuronSpec(**kwargs)
    template = get_template(spec)
    return (
        spec,
        tuple(template.extra_dims(name) for name in template.state_names),
        template.record_attributes,
    )


def _prepare_cell_from_layout(module, x_seq: Tensor, layout):
    """Bind current tensors to a cached, immutable mathematical contract."""
    from .operators import environment_dt
    from .templates import model_template_revision

    model, state_names, parameter_names = layout
    metadata = (
        _cell_metadata.__wrapped__ if torch.compiler.is_compiling() else _cell_metadata
    )
    spec, dimensions, _ = metadata(
        model, _cell_options(module), model_template_revision(model)
    )
    states = []
    for name, ndim in zip(state_names, dimensions):
        value = getattr(module, name)
        if value is None:
            raise RuntimeError(f"Initialize {name} before invoking a cell sequence")
        extra = module._memories_rv[name].sizes[-ndim:] if ndim else ()
        shape = (*x_seq.shape[1:], *extra)
        if not isinstance(value, Tensor):
            value = x_seq.new_full(shape, value, dtype=torch.float32)
        else:
            value = value.to(device=x_seq.device, dtype=torch.float32)
        states.append(value)
    parameters = tuple(
        getattr(module, name).to(device=x_seq.device, dtype=torch.float32)
        for name in parameter_names
    )
    return (
        replace(spec, dt=environment_dt(x_seq, model)),
        state_names,
        tuple(states),
        parameters,
    )


def prepare_cell(module, x_seq: Tensor):
    """Collect model tensors without disconnecting parameter or state
    gradients."""
    return _prepare_cell_from_layout(module, x_seq, _model_layout(module))


def neuron_sequence(module, x_seq: Tensor) -> Tensor:
    """Execute a neuron sequence and retain its differentiable final state."""
    from .contracts import HistoryPlan
    from .operators import run_multi_step

    spec, names, states, parameters = prepare_cell(module, x_seq)
    from .templates import model_template_revision

    record_layout = (
        _cell_records.__wrapped__ if torch.compiler.is_compiling() else _cell_records
    )
    records = record_layout(
        spec.model, spec.refractory, model_template_revision(spec.model)
    )
    requested = tuple(
        name for name, attribute in records if attribute in module._memories_rv
    )
    result = run_multi_step(
        x_seq,
        states,
        parameters,
        spec,
        HistoryPlan(requested=requested),
        backend=module.backend,
    )
    for name, value in zip(names, result[1 : 1 + len(names)]):
        setattr(module, name, value)
    attributes = dict(records)
    for name, value in zip(requested, result[1 + len(names) :]):
        if x_seq.shape[0]:
            setattr(
                module,
                attributes[name],
                value[-1].clone(),
            )
    return result[0]


def _psc_sequence(module, z_seq: Tensor, *, layout=None) -> Tensor:
    """Execute PSC dynamics, optionally with an exact private cell layout."""
    from ..base import unflatten_neuron

    # Reuse the existing connection and delay APIs. A call-scoped sparse route
    # avoids planning or rebuilding topology once for every timestep.
    from ..rnn import _sparse_run_scope
    from .operators import run_multi_step

    options = normalize_backend(module.backend)
    with (
        connection_scope(options),
        _sparse_run_scope(module, z_seq) if z_seq.shape[0] else nullcontext(),
    ):
        source = (
            z_seq.reshape(*z_seq.shape[: -len(module.n_neuron)], -1)
            if len(module.n_neuron) > 1
            and z_seq.shape[-len(module.n_neuron) :] == module.n_neuron
            else z_seq
        )
        if module.delayed_spike is None:
            drive = module.linear(source)
        else:
            drive = module.linear(delay_sequence(module, source))
    drive = unflatten_neuron(drive, drive.shape[:-1], module.n_neuron)
    prepared = (
        prepare_cell(module, drive)
        if layout is None
        else _prepare_cell_from_layout(module, drive, layout)
    )
    spec, names, states, parameters = prepared
    result = run_multi_step(drive, states, parameters, spec, backend=module.backend)
    for name, value in zip(names, result[1 : 1 + len(names)]):
        setattr(module, name, value)
    return result[0]


def psc_sequence(module, z_seq: Tensor) -> Tensor:
    """Propagate connections separately and execute stateful PSC dynamics."""
    hook = type(module).__dict__.get("_kernel_sequence")
    if hook is not None:
        return hook(module, z_seq)
    return _psc_sequence(module, z_seq)


def delay_sequence(module, source: Tensor) -> Tensor:
    """Expand delay zero as this step and preserve initial history
    gradients."""
    history = module.delayed_spike
    if not source.shape[0]:
        return source.new_empty((*source.shape[:-1], source.shape[-1] * module.n_delay))
    if history.use_circular_buffer:
        # The established circular API updates in place and is inference-only.
        # Using it here preserves its cursor and capture behavior exactly.
        return torch.stack([module._delay_expand(step) for step in source])
    initial = history.history.movedim(-len(history.n_neuron) - 1, 0)
    combined = torch.cat((initial.flip(0), source), dim=0)
    n_delay, n_step = module.n_delay, source.shape[0]
    expanded = torch.stack(
        tuple(
            combined[n_delay - delay : n_delay - delay + n_step]
            for delay in range(n_delay)
        ),
        dim=-1,
    ).flatten(-2)
    if n_step:
        history.history = (
            combined[-n_delay:].flip(0).movedim(0, -len(history.n_neuron) - 1)
        )
    return expanded
