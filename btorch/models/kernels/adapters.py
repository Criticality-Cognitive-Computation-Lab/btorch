"""Prepare stateful modules for pure tensor sequence operators."""

from contextlib import contextmanager, nullcontext

import torch
from torch import Tensor

from .options import KernelOptions, normalize_backend


_connection_contexts = {}


def _model_layout(module):
    # Exact classes protect subclass equations from being silently replaced.
    from ..neurons.alif import ALIF, ELIF
    from ..neurons.glif import GLIF3
    from ..neurons.izhikevich import Izhikevich
    from ..neurons.lif import IF, LIF
    from ..synapse import AlphaPSC, AlphaPSCBilleh, DualExponentialPSC, ExponentialPSC
    from .contracts import CellSpec
    from .templates import get_template

    models = {
        LIF: "lif",
        IF: "if",
        ALIF: "alif",
        ELIF: "elif",
        GLIF3: "glif3",
        Izhikevich: "izhikevich",
        AlphaPSC: "alpha",
        ExponentialPSC: "exponential",
        AlphaPSCBilleh: "alpha_billeh",
        DualExponentialPSC: "dual_exponential",
    }
    if type(module) not in models:
        raise NotImplementedError(
            f"No cell backend template for {type(module).__name__}"
        )
    model = models[type(module)]
    template = get_template(
        CellSpec(model, refractory=getattr(module, "_use_refractory", False))
    )
    return model, template.state_names, template.parameter_names


def validate_module(module, options: KernelOptions) -> None:
    """Reject modules without a general or exact-class sequence adapter."""
    from ..synapse import ConductancePSC, HeterSynapsePSC

    # ConductancePSC has an exact-class standalone sequence adapter. Keeping it
    # out of _model_layout makes recurrent preparation reject COBA until the
    # scheduler has an explicit voltage-observation contract.
    if type(module) not in (ConductancePSC, HeterSynapsePSC):
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


def _prepare_cell_from_layout(module, x_seq: Tensor, layout):
    """Normalize tensors for one explicitly selected cell layout."""
    from .contracts import CellSpec
    from .operators import environment_dt

    model, state_names, parameter_names = layout
    kwargs = {"model": model, "dt": environment_dt(x_seq, model)}
    if hasattr(module, "surrogate_function"):
        from .. import surrogate

        function = module.surrogate_function
        names = {
            surrogate.Sigmoid: "sigmoid",
            surrogate.ATan: "atan",
            surrogate.Erf: "erf",
            surrogate.SuperSpike: "superspike",
            surrogate.Triangle: "triangle",
            surrogate.ATanApprox: "atan_approx",
            surrogate.PoissonRandomSpike: "poisson",
        }
        if type(function) not in names:
            raise NotImplementedError(
                "Cell kernels require a supported built-in surrogate"
            )
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
            kwargs["poisson"] = PoissonOptions(
                function.spiking,
                float(function.tau),
                float(function.rho),
                float(function.leak),
                float(function.k),
            )
            alpha = 1.0
        else:
            if not function.spiking:
                raise NotImplementedError(
                    "Deterministic surrogate kernels require spiking=True"
                )
            if not isinstance(function.alpha, (float, int)):
                raise NotImplementedError(
                    "Surrogate hyperparameters must be non-trainable scalars"
                )
            alpha = float(function.alpha)
        kwargs.update(
            hard_reset=module.hard_reset,
            detach_reset=module.detach_reset,
            surrogate=names[type(function)],
            alpha=alpha,
            damping=float(function.damping_factor),
            refractory=getattr(module, "_use_refractory", False),
        )
    states = []
    for name in state_names:
        value = getattr(module, name)
        if value is None:
            raise RuntimeError(f"Initialize {name} before invoking a cell sequence")
        shape = (*x_seq.shape[1:], module.n_Iasc) if name == "Iasc" else x_seq.shape[1:]
        if not isinstance(value, Tensor):
            value = x_seq.new_full(shape, value, dtype=torch.float32)
        else:
            value = value.to(device=x_seq.device, dtype=torch.float32)
        states.append(value)
    parameters = tuple(
        getattr(module, name).to(device=x_seq.device, dtype=torch.float32)
        for name in parameter_names
    )
    return CellSpec(**kwargs), state_names, tuple(states), parameters


def prepare_cell(module, x_seq: Tensor):
    """Collect model tensors without disconnecting parameter or state
    gradients."""
    return _prepare_cell_from_layout(module, x_seq, _model_layout(module))


def neuron_sequence(module, x_seq: Tensor) -> Tensor:
    """Execute a neuron sequence and retain its differentiable final state."""
    from .contracts import HistoryPlan
    from .operators import run_multi_step

    spec, names, states, parameters = prepare_cell(module, x_seq)
    requested = ("v_charged",) if module.pre_spike_v else ()
    if "u_pre_spike" in module._memories_rv:
        requested += ("u_charged",)
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
    for name, value in zip(requested, result[1 + len(names) :]):
        if x_seq.shape[0]:
            setattr(
                module,
                "v_pre_spike" if name == "v_charged" else "u_pre_spike",
                value[-1],
            )
    return result[0]


def _psc_sequence(module, z_seq: Tensor, *, layout=None) -> Tensor:
    """Execute PSC dynamics, optionally with an exact private cell layout."""
    from ..base import unflatten_neuron

    # Reuse the existing connection and delay APIs. A call-scoped sparse route
    # avoids planning or rebuilding topology once for every timestep.
    from ..rnn import _sparse_run_scope
    from ..synapse import HeterSynapsePSC
    from .operators import run_multi_step

    options = normalize_backend(module.backend)
    with (
        connection_scope(options),
        _sparse_run_scope(module, z_seq) if z_seq.shape[0] else nullcontext(),
    ):
        if isinstance(module, HeterSynapsePSC):
            source, leading, has_receptor_axis = module._flatten_input(z_seq)
            if module.delayed_spike is not None:
                module._validate_delayed_input(has_receptor_axis, z_seq.shape)
                source = delay_sequence(module, source)
            output = psc_sequence(module.base_psc, source)
            result = module._reshape_sum_receptor(output, leading)
            if z_seq.shape[0]:
                module.psc = result[-1]
            return result
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
    return _psc_sequence(module, z_seq)


def conductance_sequence(module, z_seq: Tensor, v_seq: Tensor | None = None) -> Tensor:
    """Reuse exponential state kernels for an exact ConductancePSC sequence."""
    from ..synapse import ConductancePSC

    if type(module) is not ConductancePSC:
        raise TypeError("conductance_sequence requires an exact ConductancePSC")
    conductance = _psc_sequence(
        module,
        z_seq,
        layout=("exponential", ("psc",), ("tau_syn",)),
    )
    if v_seq is None:
        return conductance
    padding = conductance.ndim - v_seq.ndim
    voltage = v_seq.reshape(v_seq.shape[0], *((1,) * padding), *v_seq.shape[1:])
    return conductance * (module.E - voltage)


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
