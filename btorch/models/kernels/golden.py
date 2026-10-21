"""Adapt existing Python Modules to the tensor ABI without duplicating
equations."""

import torch
from torch import Tensor

from .. import environ, surrogate
from ..neurons.alif import ALIF, ELIF
from ..neurons.glif import GLIF3
from ..neurons.izhikevich import Izhikevich
from ..neurons.lif import IF, LIF
from ..synapse import AlphaPSC, AlphaPSCBilleh, DualExponentialPSC, ExponentialPSC
from .contracts import CellSpec
from .templates import get_template


def python_module_sequence(
    x_seq: Tensor,
    states: tuple[Tensor, ...],
    parameters: tuple[Tensor, ...],
    spec: CellSpec,
    requested: tuple[str, ...] = (),
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Run the original btorch single-step implementation as the golden
    baseline.

    Tensor bindings retain the caller's autograd graph. Only layout conversion,
    state binding and record collection happen here; all model equations and
    surrogate derivatives belong to the existing Python Modules.

    Args:
        x_seq: Drive with time first, followed by batch and neuron axes.
        states: Initial states in the declared model order.
        parameters: Original parameter tensors, preserving their gradients.
        spec: Scalar integration, reset and surrogate choices.
        requested: Post-step states or charged-state records to collect.

    Returns:
        Output sequence, final states and requested record sequences.

    Raises:
        ValueError: When the model has no built-in Python Module.
    """
    template = get_template(spec)
    n_neuron = x_seq.shape[-1]
    kwargs = dict(n_neuron=n_neuron, backend="torch", step_mode="s")
    neurons = {
        "if": IF,
        "lif": LIF,
        "alif": ALIF,
        "elif": ELIF,
        "glif3": GLIF3,
        "izhikevich": Izhikevich,
    }
    if spec.model in neurons:
        functions = dict(
            sigmoid=surrogate.Sigmoid,
            atan=surrogate.ATan,
            atan_approx=surrogate.ATanApprox,
            erf=surrogate.Erf,
            superspike=surrogate.SuperSpike,
            triangle=surrogate.Triangle,
            poisson=surrogate.PoissonRandomSpike,
        )
        function_kwargs = (
            {
                "spiking": spec.poisson.spiking,
                "tau": spec.poisson.tau,
                "rho": spec.poisson.rho,
                "leak": spec.poisson.leak,
                "k": spec.poisson.k,
                "damping_factor": spec.damping,
            }
            if spec.poisson is not None
            else {"alpha": spec.alpha, "damping_factor": spec.damping}
        )
        kwargs.update(
            surrogate_function=functions[spec.surrogate](**function_kwargs),
            hard_reset=spec.hard_reset,
            detach_reset=spec.detach_reset,
            device="cpu",
            dtype=torch.float32,
        )
        kwargs["pre_spike" if spec.model == "izhikevich" else "pre_spike_v"] = True
        if spec.refractory:
            kwargs["tau_ref"] = 1.0
        if spec.model == "glif3":
            n_modes = states[1].shape[-1]
            kwargs.update(k=[0.2] * n_modes, asc_amps=[0.0] * n_modes)
        # Constructor defaults are irrelevant after binding. Keep them on CPU
        # to avoid synchronous host-to-device copies during CUDA graph capture.
        with torch.device("cpu"):
            module = neurons[spec.model](**kwargs)
    else:
        kwargs["linear"] = torch.nn.Identity()
        if spec.model == "dual_exponential":
            with torch.device("cpu"):
                module = DualExponentialPSC(tau_rise=2.0, tau_decay=4.0, **kwargs)
        else:
            classes = dict(
                exponential=ExponentialPSC, alpha=AlphaPSC, alpha_billeh=AlphaPSCBilleh
            )
            if spec.model not in classes:
                raise ValueError("External model parity requires ParityCase.golden")
            with torch.device("cpu"), environ.context(dt=spec.dt):
                module = classes[spec.model](tau_syn=4.0, **kwargs)
    for name, value in zip(template.parameter_names, parameters):
        # Registering a new Parameter would detach the supplied leaf tensor.
        # Bind the original tensor as a buffer so autograd sees this exact input.
        name = "_v_rest" if spec.model == "glif3" and name == "v_rest" else name
        module._parameters.pop(name, None)
        module._buffers.pop(name, None)
        if name in module.__dict__:
            delattr(module, name)
        module.register_buffer(name, value.float(), persistent=False)
    for name, value in zip(template.state_names, states):
        setattr(module, name, value.float())
    outputs = []
    histories = {name: [] for name in requested}
    with environ.context(dt=spec.dt):
        for x in x_seq:
            outputs.append(module.single_step_forward(x.float()))
            for name in requested:
                attribute = {
                    "v_charged": "v_pre_spike",
                    "u_charged": "u_pre_spike",
                }.get(name, name)
                histories[name].append(getattr(module, attribute))
    output = (
        torch.stack(outputs).to(x_seq.dtype)
        if outputs
        else x_seq.new_empty(x_seq.shape)
    )
    final = tuple(getattr(module, name) for name in template.state_names)
    records = tuple(
        torch.stack(histories[name])
        if histories[name]
        else states[
            template.state_names.index(name.removesuffix("_charged"))
        ].new_empty(
            (
                0,
                *states[
                    template.state_names.index(name.removesuffix("_charged"))
                ].shape,
            )
        )
        for name in requested
    )
    return output, final, records


def python_module_step(
    x: Tensor,
    states: tuple[Tensor, ...],
    parameters: tuple[Tensor, ...],
    spec: CellSpec,
) -> tuple[Tensor, tuple[Tensor, ...], dict[str, Tensor]]:
    """Execute an original Python Module step with all declared records.

    Args:
        x: Single-step drive with batch and neuron axes.
        states: Initial states in the declared model order.
        parameters: Original floating parameter tensors.
        spec: Scalar integration, reset and surrogate choices.

    Returns:
        Output, final states and records indexed by declared name.
    """
    names = get_template(spec).history_names
    output, final, records = python_module_sequence(
        x.unsqueeze(0), states, parameters, spec, names
    )
    return output[0], final, {name: value[0] for name, value in zip(names, records)}
