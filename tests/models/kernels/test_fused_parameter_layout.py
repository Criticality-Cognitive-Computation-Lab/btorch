"""Validate fused cells read original broadcast parameter storage."""

import ast
import copy

import pytest
import torch
from torch import nn

from btorch.models.kernels.backends.engine import _layout
from btorch.models.kernels.backends.fused import (
    _prepare_psc_parameters,
    fused_step_backward,
    fused_step_forward,
    generate_fused_source,
)
from btorch.models.kernels.contracts import NeuronSpec
from btorch.models.kernels.templates import get_template
from btorch.models.kernels.torch_backend import forward_step
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


def _leaf(value):
    return value.detach().requires_grad_()


def test_fused_source_specializes_parameter_storage_and_receptor_coordinates():
    neuron_spec, psc_spec = NeuronSpec("lif"), NeuronSpec("alpha")
    neuron_parameters = (
        torch.tensor(0.8),
        torch.arange(7.0)[1::2],
        torch.ones(2, 1),
        torch.tensor(3.0),
    )
    psc_parameters = (torch.arange(5.0)[1::2], torch.ones(3, 1))
    neuron_layouts = tuple(_layout(value, (2, 3)) for value in neuron_parameters)
    _, psc_layouts = _prepare_psc_parameters(
        psc_parameters,
        get_template(psc_spec),
        (2, 6),
        (2, 3, 2),
    )
    assert tuple(layout.target_shape for layout in psc_layouts) == (
        (2, 3, 2),
        (2, 3, 2),
    )
    assert torch.empty(psc_layouts[0].target_shape).sum_to_size(2).shape == (2,)

    # The ABI exposes only reachable storage. The generated receptor loads use
    # the flattened neuron/receptor lane to recover receptor and neuron axes;
    # this is the same coordinate mapping used by primitive sequence kernels.
    source, pointers = generate_fused_source(
        "triton",
        neuron_spec,
        psc_spec,
        6,
        2,
        2,
        neuron_parameter_layouts=neuron_layouts,
        psc_parameter_layouts=psc_layouts,
    )
    sizes = dict(pointers)
    assert sizes["neuron_v_reset"] == 5
    assert sizes["neuron_c_m"] == 2
    assert sizes["psc_tau_syn"] == 3
    assert sizes["psc_g_max"] == 3
    # Evaluate the generated pointer offsets, not formatting-specific
    # parentheses. Receptor tau uses strided storage; gain follows neuron
    # coordinates independently of batch and receptor.
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    for receptor in (0, 1):
        for name, expected in (
            ("tau_syn", [receptor * 2] * 6),
            ("g_max", [0, 1, 2, 0, 1, 2]),
        ):
            load = assignments[f"psc_{receptor}_{name}"]
            offset = load.args[0].right
            expression = compile(ast.Expression(offset), "<parameter-offset>", "eval")
            actual = [
                eval(expression, {"__builtins__": {}}, {"index": lane})
                for lane in range(6)
            ]
            assert actual == expected

    # A dense view has different address arithmetic and must not collide with
    # the strided program in the bounded fused-source cache.
    dense_layouts = tuple(_layout(value.clone(), (2, 3)) for value in neuron_parameters)
    dense_source = generate_fused_source(
        "triton",
        neuron_spec,
        psc_spec,
        6,
        2,
        2,
        neuron_parameter_layouts=dense_layouts,
        psc_parameter_layouts=psc_layouts,
    )
    assert dense_source != (source, pointers)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_fused_cell_broadcast_views_match_eager_forward_and_vjp(backend):
    require_cuda_backend(backend)
    torch.manual_seed(811)
    batch, neurons, receptors = 2, 3, 2
    neuron_spec = NeuronSpec("lif", dt=0.25, hard_reset=True)
    psc_spec = NeuronSpec("alpha", dt=0.5)
    external = _leaf(torch.rand(batch, neurons, device="cuda") * 0.2)
    drive = _leaf(torch.randn(batch, neurons * receptors, device="cuda") * 0.15)
    neuron_states = (_leaf(torch.rand(batch, neurons, device="cuda") * 0.2),)
    psc_states = tuple(
        _leaf(torch.rand(batch, neurons * receptors, device="cuda") * 0.1)
        for _ in range(2)
    )

    reset_storage = torch.linspace(0.0, 0.3, 2 * neurons + 1, device="cuda")
    reset = _leaf(torch._neg_view(reset_storage[1::2]))
    threshold = _leaf(torch.tensor(0.7, device="cuda").expand(neurons))
    assert threshold.stride() == (0,)
    shared_low_precision = torch.tensor(
        2.5, device="cuda", dtype=torch.float16, requires_grad=True
    )
    neuron_parameters = (
        threshold,
        reset,
        shared_low_precision,
        shared_low_precision,
    )
    receptor_storage = torch.tensor([99.0, 1.5, 99.0, 3.0, 99.0], device="cuda")
    tau_syn = _leaf(receptor_storage[1::2])
    gain_storage = torch.linspace(0.8, 1.4, 2 * neurons + 1, device="cuda")
    g_max = _leaf(gain_storage[1::2].view(neurons, 1))
    psc_parameters = (tau_syn, g_max)

    logical_states = tuple(
        value.reshape(batch, neurons, receptors) for value in psc_states
    )
    logical_drive = drive.reshape(batch, neurons, receptors)
    current, expected_psc, psc_records = forward_step(
        logical_drive, logical_states, psc_parameters, psc_spec
    )
    expected_spike, expected_neuron, neuron_records = forward_step(
        external + current.sum(-1), neuron_states, neuron_parameters, neuron_spec
    )
    expected = (
        expected_spike,
        *expected_neuron,
        *(value.reshape_as(psc_states[0]) for value in expected_psc),
        *(neuron_records[name].unsqueeze(0) for name in ("v", "v_charged")),
        *(
            psc_records[name].reshape_as(psc_states[0]).unsqueeze(0)
            for name in ("psc", "h")
        ),
    )
    actual = fused_step_forward(
        external,
        drive,
        neuron_states,
        psc_states,
        neuron_parameters,
        psc_parameters,
        neuron_spec,
        psc_spec,
        ("v", "v_charged"),
        ("psc", "h"),
        backend=backend,
    )
    actual_flat = (actual[0], *actual[1], *actual[2], *actual[3], *actual[4])
    for result, reference in zip(actual_flat, expected):
        torch.testing.assert_close(result, reference, atol=8e-5, rtol=8e-4)

    # Random cotangents make every final state and public one-step record
    # contribute. This catches wrong receptor coordinates and confirms lane
    # gradients still reduce into scalar, neuron, receptor and shared leaves.
    cotangents = tuple(torch.randn_like(value) for value in expected)
    inputs = (
        external,
        drive,
        *neuron_states,
        *psc_states,
        neuron_parameters[0],
        neuron_parameters[1],
        shared_low_precision,
        *psc_parameters,
    )
    trusted = torch.autograd.grad(expected, inputs, cotangents, allow_unused=True)
    reverse = fused_step_backward(
        external,
        drive,
        neuron_states,
        psc_states,
        neuron_parameters,
        psc_parameters,
        neuron_spec,
        psc_spec,
        cotangents[0],
        cotangents[1:2],
        cotangents[2:4],
        grad_neuron_records=dict(zip(("v", "v_charged"), cotangents[4:6])),
        grad_psc_records=dict(zip(("psc", "h"), cotangents[6:])),
        backend=backend,
    )
    actual_gradients = (
        reverse[0],
        reverse[1],
        *reverse[2],
        *reverse[3],
        reverse[4][0],
        reverse[4][1],
        reverse[4][2] + reverse[4][3],
        *reverse[5],
    )
    for result, reference, original in zip(actual_gradients, trusted, inputs):
        if reference is None:
            reference = torch.zeros_like(original)
        torch.testing.assert_close(result, reference, atol=7e-4, rtol=7e-4)


def _replace_tensor(module, name, value, *, parameter=False):
    module._parameters.pop(name, None)
    module._buffers.pop(name, None)
    if name in module.__dict__:
        delattr(module, name)
    if parameter:
        module.register_parameter(name, nn.Parameter(value))
    else:
        module.register_buffer(name, value.requires_grad_(), persistent=False)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_recurrent_module_strided_parameters_match_torch_vjp(backend):
    require_cuda_backend(backend)
    from btorch.models import environ
    from btorch.models.kernels import KernelOptions
    from btorch.models.neurons import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import AlphaPSC

    torch.manual_seed(812)
    neurons, batch = 3, 2
    synapse = AlphaPSC(neurons, 2.0, nn.Linear(neurons, neurons, bias=False))
    model = RecurrentNN(
        LIF(neurons, v_threshold=0.7, trainable_param={"tau"}),
        synapse,
        backend=KernelOptions(neuron=backend),
        update_state_names=("neuron.v", "synapse.psc", "synapse.h"),
    ).to(device="cuda", dtype=torch.float32)
    model.neuron.init_state(batch_size=batch, device="cuda", dtype=torch.float32)
    synapse.init_state(batch_size=batch, device="cuda", dtype=torch.float32)
    reset_storage = torch.linspace(0.0, 0.2, 2 * neurons + 1, device="cuda")
    tau_storage = torch.linspace(2.0, 4.0, 2 * neurons + 1, device="cuda")
    psc_tau_storage = torch.linspace(1.5, 3.0, 2 * neurons + 1, device="cuda")
    _replace_tensor(model.neuron, "v_reset", reset_storage[1::2])
    _replace_tensor(model.neuron, "tau", tau_storage[1::2], parameter=True)
    _replace_tensor(model.synapse, "tau_syn", psc_tau_storage[1::2])
    _replace_tensor(
        model.synapse,
        "g_max",
        torch.tensor([[0.8], [1.2]], device="cuda"),
    )
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = torch.randn(5, batch, neurons, device="cuda").requires_grad_()
    x_reference = x.detach().clone().requires_grad_()

    with environ.context(dt=0.5):
        output, records = model(x)
        expected, expected_records = reference(x_reference)
    torch.testing.assert_close(output, expected, atol=8e-5, rtol=8e-4)
    for name in records:
        torch.testing.assert_close(
            records[name], expected_records[name], atol=8e-5, rtol=8e-4
        )
    weights = torch.randn_like(output)
    loss = (output * weights).sum() + sum(
        value.square().sum() for value in records.values()
    )
    expected_loss = (expected * weights).sum() + sum(
        value.square().sum() for value in expected_records.values()
    )
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(x.grad, x_reference.grad, atol=7e-4, rtol=7e-4)
    for name in ("tau", "v_reset"):
        torch.testing.assert_close(
            getattr(model.neuron, name).grad,
            getattr(reference.neuron, name).grad,
            atol=7e-4,
            rtol=7e-4,
        )
    for name in ("tau_syn", "g_max"):
        torch.testing.assert_close(
            getattr(model.synapse, name).grad,
            getattr(reference.synapse, name).grad,
            atol=7e-4,
            rtol=7e-4,
        )
