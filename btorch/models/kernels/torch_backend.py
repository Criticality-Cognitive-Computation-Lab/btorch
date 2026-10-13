"""Adapt the existing Python modules to the tensor cell operator ABI."""

import torch
from torch import Tensor

from .contracts import CellSpec
from .templates import get_reference_callbacks, get_template


def forward_step(
    x: Tensor,
    states: tuple[Tensor, ...],
    parameters: tuple[Tensor, ...],
    spec: CellSpec,
):
    """Execute the existing Python module, or an external model's callback."""
    callbacks = get_reference_callbacks(spec)
    if callbacks is not None:
        return callbacks[0](x, states, parameters, spec)
    from .golden import python_module_step

    return python_module_step(x, states, parameters, spec)


def backward_step(
    x: Tensor,
    states: tuple[Tensor, ...],
    parameters: tuple[Tensor, ...],
    spec: CellSpec,
    grad_output: Tensor,
    grad_states: tuple[Tensor, ...],
    grad_records: dict[str, Tensor],
    parameter_mask: tuple[bool, ...] | None = None,
):
    """Differentiate one existing Python module step with ordinary autograd.

    This is the Torch backend's local VJP, not a second model
    implementation. Only one step is differentiated; the sequence caller
    propagates temporal state adjoints using saved histories. GPU
    backends keep their generated VJPs.
    """
    callbacks = get_reference_callbacks(spec)
    if callbacks is not None:
        if callbacks[1] is None:
            raise NotImplementedError(
                f"Model template {spec.model!r} does not provide a backward_step"
            )
        gx, gs, gp = callbacks[1](
            x, states, parameters, spec, grad_output, grad_states, grad_records
        )
        if parameter_mask is not None:
            gp = tuple(
                value if needed else None for value, needed in zip(gp, parameter_mask)
            )
        return gx, gs, gp
    mask = parameter_mask or (True,) * len(parameters)
    # Custom operator execution disables grad. Enable it locally, preserving
    # all state adjoints even when the original initial state was frozen.
    with torch.inference_mode(False), torch.enable_grad():
        local_x = x.detach().requires_grad_()
        local_states = tuple(value.detach().requires_grad_() for value in states)
        local_parameters = tuple(
            value.detach().requires_grad_(needed)
            for value, needed in zip(parameters, mask)
        )
        output, final, records = forward_step(
            local_x, local_states, local_parameters, spec
        )
        outputs = (output, *final, *(records[name] for name in grad_records))
        cotangents = (grad_output, *grad_states, *grad_records.values())
        differentiable = [
            (value, grad)
            for value, grad in zip(outputs, cotangents)
            if value.requires_grad
        ]
        inputs = (
            local_x,
            *local_states,
            *(value for value, needed in zip(local_parameters, mask) if needed),
        )
        grads = (
            torch.autograd.grad(
                tuple(value for value, _ in differentiable),
                inputs,
                grad_outputs=tuple(grad for _, grad in differentiable),
                allow_unused=True,
            )
            if differentiable
            else (None,) * len(inputs)
        )
    grads = tuple(
        torch.zeros_like(value) if grad is None else grad
        for value, grad in zip(inputs, grads)
    )
    n_states = len(states)
    parameter_grads = iter(grads[1 + n_states :])
    return (
        grads[0],
        grads[1 : 1 + n_states],
        tuple(next(parameter_grads) if needed else None for needed in mask),
    )


def forward_sequence(
    x_seq: Tensor,
    states: tuple[Tensor, ...],
    parameters: tuple[Tensor, ...],
    spec: CellSpec,
    history_names: tuple[str, ...],
):
    """Evaluate a complete sequence with only the requested history storage."""
    x_float = x_seq.float()
    states = tuple(value.float() for value in states)
    parameters = tuple(value.float() for value in parameters)
    template = get_template(spec)
    state_map = dict(zip(template.state_names, states))
    output_seq = torch.empty_like(x_float)
    histories = []
    for name in history_names:
        source = (
            state_map[name]
            if name in state_map
            else state_map[dict(template.history_sources)[name]]
        )
        histories.append(source.new_empty((x_seq.shape[0], *source.shape)))
    # Allocate each sequence once, rather than retaining every step followed
    # by a second full-sequence stack allocation. Slice writes still preserve
    # ordinary eager autograd, which is the independent parity reference.
    for time, x in enumerate(x_float.unbind(0)):
        output, states, records = forward_step(x, states, parameters, spec)
        output_seq[time] = output
        for name, values in zip(history_names, histories):
            values[time] = records[name]
    # Custom operators must return non-aliasing tensors; histories themselves
    # are allocated once and shared with autograd through their output handles.
    return (
        output_seq,
        tuple(value.clone() for value in states),
        tuple(histories),
    )


def backward_sequence(
    x_seq: Tensor,
    initial_states: tuple[Tensor, ...],
    parameters: tuple[Tensor, ...],
    spec: CellSpec,
    history_names: tuple[str, ...],
    histories: tuple[Tensor, ...],
    grad_output: Tensor,
    grad_final_states: tuple[Tensor, ...],
    grad_histories: tuple[Tensor, ...],
    needs_input_grad: tuple[bool, ...] | None = None,
):
    """Reverse a sequence using shared state histories and local
    recomputation."""
    template = get_template(spec)
    history = dict(zip(history_names, histories))
    state_grad = tuple(value.float() for value in grad_final_states)
    parameter_mask = (
        None
        if needs_input_grad is None
        else needs_input_grad[1 + len(initial_states) :]
    )
    parameter_grad = tuple(
        torch.zeros_like(value, dtype=torch.float32)
        if parameter_mask is None or parameter_mask[i]
        else None
        for i, value in enumerate(parameters)
    )
    parameters_float = tuple(value.float() for value in parameters)
    initial_float = tuple(value.float() for value in initial_states)
    needs_x = needs_input_grad is None or needs_input_grad[0]
    grad_x = torch.empty_like(x_seq) if needs_x else None
    for time in range(x_seq.shape[0] - 1, -1, -1):
        previous = (
            initial_float
            if time == 0
            else tuple(history[name][time - 1] for name in template.state_names)
        )
        grad_records = {
            name: value[time].float()
            for name, value in zip(history_names, grad_histories)
        }
        gx, state_grad, gp = backward_step(
            x_seq[time].float(),
            previous,
            parameters_float,
            spec,
            grad_output[time].float(),
            state_grad,
            grad_records,
            parameter_mask=parameter_mask,
        )
        if grad_x is not None:
            grad_x[time] = gx
        parameter_grad = tuple(
            a + b if a is not None else None for a, b in zip(parameter_grad, gp)
        )
    state_grad = tuple(
        value.to(initial.dtype) for value, initial in zip(state_grad, initial_states)
    )
    parameter_grad = tuple(
        value.to(parameter.dtype) if value is not None else None
        for value, parameter in zip(parameter_grad, parameters)
    )
    if needs_input_grad is not None:
        gradients = (grad_x, *state_grad, *parameter_grad)
        gradients = tuple(
            value if needed else None
            for value, needed in zip(gradients, needs_input_grad)
        )
        grad_x = gradients[0]
        state_grad = gradients[1 : 1 + len(initial_states)]
        parameter_grad = gradients[1 + len(initial_states) :]
    return grad_x, state_grad, parameter_grad
