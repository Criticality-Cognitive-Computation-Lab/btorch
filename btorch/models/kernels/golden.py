"""Adapt existing Python Modules to the tensor ABI without duplicating
equations."""

import torch
from torch import Tensor

from .. import environ
from .contracts import NeuronSpec
from .templates import get_template


def python_module_sequence(
    x_seq: Tensor,
    states: tuple[Tensor, ...],
    parameters: tuple[Tensor, ...],
    spec: NeuronSpec,
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
    if template.module_factory is None:
        raise ValueError("External model parity requires an explicit golden callable")
    with torch.device("cpu"), environ.context(dt=spec.dt):
        module = template.module_factory(spec, states, x_seq.shape[-1])
    parameter_attributes = dict(template.parameter_attributes)
    record_attributes = dict(template.record_attributes)
    for name, value in zip(template.parameter_names, parameters):
        # Registering a new Parameter would detach the supplied leaf tensor.
        # Bind the original tensor as a buffer so autograd sees this exact input.
        name = parameter_attributes.get(name, name)
        module._parameters.pop(name, None)
        module._buffers.pop(name, None)
        if name in module.__dict__:
            delattr(module, name)
        module.register_buffer(name, value, persistent=False)
    for name, value in zip(template.state_names, states):
        setattr(module, name, value)
    outputs = []
    histories = {name: [] for name in requested}
    with environ.context(dt=spec.dt):
        for x in x_seq:
            outputs.append(module.single_step_forward(x))
            for name in requested:
                attribute = record_attributes.get(name, name)
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
            template.state_names.index(template.history_state_name(name))
        ].new_empty(
            (
                0,
                *states[
                    template.state_names.index(template.history_state_name(name))
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
    spec: NeuronSpec,
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
