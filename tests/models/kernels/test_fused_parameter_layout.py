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


def test_fused_plan_rebinds_values_layouts_and_releases_invocation_storage():
    """Reuse static preparation without retaining states, drives or dt."""
    import gc
    import weakref
    from dataclasses import replace

    from btorch.models.kernels.golden import python_module_sequence
    from tests.models.kernels.cases import cell_inputs

    require_cuda_backend("triton")
    neuron_spec, _, neuron_states, neuron_parameters = cell_inputs("lif")
    psc_spec, _, psc_states, psc_parameters = cell_inputs("alpha")
    neuron_states, psc_states, neuron_parameters, psc_parameters = (
        tuple(value.detach().cuda() for value in group)
        for group in (neuron_states, psc_states, neuron_parameters, psc_parameters)
    )
    stream = torch.cuda.Stream()
    with torch.inference_mode():
        for index, dt in enumerate((0.25, 0.5, 0.75)):
            # The second call reuses the exact layout with fresh buffers and
            # parameters. The third changes the physical input stride without
            # changing its shape. Both must match the original Python modules.
            external = (
                torch.randn(3, 2, device="cuda").T
                if index == 2
                else torch.randn(2, 3, device="cuda")
            )
            drive = torch.randn_like(external)
            runtime_dt = torch.tensor(dt, device="cuda")
            neuron_runtime = replace(neuron_spec, dt=runtime_dt)
            psc_runtime = replace(psc_spec, dt=runtime_dt)
            for parameter in (*neuron_parameters, *psc_parameters):
                parameter.add_(0.01)
            # Executables are shared across calls; their current-stream
            # binding must remain invocation-local, including after a warm hit.
            current = torch.cuda.current_stream()
            target = stream if index == 1 else current
            if target != current:
                target.wait_stream(current)
            with torch.cuda.stream(target):
                external.add_(0.1)
                actual = fused_step_forward(
                    external,
                    drive,
                    neuron_states,
                    psc_states,
                    neuron_parameters,
                    psc_parameters,
                    neuron_runtime,
                    psc_runtime,
                )
            if target != current:
                current.wait_stream(target)
            psc_reference = python_module_sequence(
                drive.unsqueeze(0),
                psc_states,
                psc_parameters,
                replace(psc_spec, dt=dt),
            )
            neuron_reference = python_module_sequence(
                (external + psc_reference[0][0]).unsqueeze(0),
                neuron_states,
                neuron_parameters,
                replace(neuron_spec, dt=dt),
            )
            for value, expected in zip(
                (actual[0], *actual[1], *actual[2]),
                (
                    neuron_reference[0][0],
                    *neuron_reference[1],
                    *psc_reference[1],
                ),
                strict=True,
            ):
                torch.testing.assert_close(value, expected, atol=8e-5, rtol=8e-4)
            references = tuple(
                weakref.ref(value) for value in (external, drive, runtime_dt)
            )
            del external, drive, runtime_dt, neuron_runtime, psc_runtime
            gc.collect()
            assert all(reference() is None for reference in references)


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
    # The first paired stage reads caller-owned PSC views; subsequent stages
    # read the internal recording layout. Preparation must not assume that the
    # initial storage describes every step of the recurrent working set.
    for name in ("psc", "h"):
        storage = torch.zeros(batch, neurons * 2, device="cuda")
        setattr(model.synapse, name, storage[:, 1::2])
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    # Three-neuron rows have alternating 16-byte alignment, and this external
    # input also has a nonzero offset and nonunit stride. Reusing an executable
    # compiled for the first pointer's alignment would be an invalid contract.
    x = torch.randn(5, batch, neurons * 2, device="cuda")[..., 1::2].requires_grad_()
    x_reference = x.detach().clone().requires_grad_()
    # Continue with a time-broadcast view. Its row address must remain fixed,
    # while its input adjoint still sums all logical timesteps into the shared
    # base. Together the windows exercise nonzero and zero temporal strides.
    continuation_base = torch.randn(
        1, batch, neurons * 2, device="cuda"
    ).requires_grad_()
    continuation_reference = continuation_base.detach().clone().requires_grad_()
    continuation = continuation_base[..., 1::2].expand(3, -1, -1)
    reference_continuation = continuation_reference[..., 1::2].expand(3, -1, -1)

    with environ.context(dt=0.5):
        output, records = model(x)
        expected, expected_records = reference(x_reference)
        next_output, next_records = model(continuation)
        next_expected, next_expected_records = reference(reference_continuation)
    torch.testing.assert_close(output, expected, atol=8e-5, rtol=8e-4)
    torch.testing.assert_close(next_output, next_expected, atol=8e-5, rtol=8e-4)
    for name in records:
        torch.testing.assert_close(
            records[name], expected_records[name], atol=8e-5, rtol=8e-4
        )
        torch.testing.assert_close(
            next_records[name], next_expected_records[name], atol=8e-5, rtol=8e-4
        )
    weights = torch.randn_like(output)
    next_weights = torch.randn_like(next_output)
    loss = (output * weights).sum() + sum(
        value.square().sum() for value in records.values()
    )
    loss = (
        loss
        + (next_output * next_weights).sum()
        + sum(value.square().sum() for value in next_records.values())
    )
    expected_loss = (expected * weights).sum() + sum(
        value.square().sum() for value in expected_records.values()
    )
    expected_loss = (
        expected_loss
        + (next_expected * next_weights).sum()
        + sum(value.square().sum() for value in next_expected_records.values())
    )
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(x.grad, x_reference.grad, atol=7e-4, rtol=7e-4)
    torch.testing.assert_close(
        continuation_base.grad, continuation_reference.grad, atol=7e-4, rtol=7e-4
    )
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


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_fused_reverse_reuses_metadata_with_independent_tapes(backend):
    """Fresh tensors, runtime dt and cotangent strides retain every state
    VJP."""
    import gc
    import weakref
    from dataclasses import replace

    from btorch.models.kernels.golden import python_module_sequence
    from tests.models.kernels.cases import cell_inputs

    require_cuda_backend(backend)

    def run(dt, strided):
        neuron_spec, _, neuron_states, neuron_parameters = cell_inputs("lif")
        psc_spec, _, psc_states, psc_parameters = cell_inputs("alpha")
        neuron_states, psc_states, neuron_parameters, psc_parameters = (
            tuple(value.detach().cuda().requires_grad_() for value in group)
            for group in (neuron_states, psc_states, neuron_parameters, psc_parameters)
        )
        external = torch.full((2, 3), 0.6, device="cuda", requires_grad=True)
        drive = torch.full((2, 3), 0.3, device="cuda", requires_grad=True)
        runtime_dt = torch.tensor(dt, device="cuda")
        psc_reference = python_module_sequence(
            drive.unsqueeze(0), psc_states, psc_parameters, replace(psc_spec, dt=dt)
        )
        neuron_reference = python_module_sequence(
            (external + psc_reference[0][0]).unsqueeze(0),
            neuron_states,
            neuron_parameters,
            replace(neuron_spec, dt=dt),
        )
        expected = (neuron_reference[0][0], *neuron_reference[1], *psc_reference[1])
        cotangents = tuple(
            torch.randn((*value.shape, 2), device="cuda")[..., 0]
            if strided
            else torch.randn_like(value)
            for value in expected
        )
        roots = (
            external,
            drive,
            *neuron_states,
            *psc_states,
            *neuron_parameters,
            *psc_parameters,
        )
        golden = torch.autograd.grad(expected, roots, cotangents, allow_unused=True)
        reverse = fused_step_backward(
            external,
            drive,
            neuron_states,
            psc_states,
            neuron_parameters,
            psc_parameters,
            replace(neuron_spec, dt=runtime_dt),
            replace(psc_spec, dt=runtime_dt),
            cotangents[0],
            cotangents[1:2],
            cotangents[2:],
            backend=backend,
        )
        actual = (
            reverse[0],
            reverse[1],
            *reverse[2],
            *reverse[3],
            *reverse[4],
            *reverse[5],
        )
        for value, trusted, root in zip(actual, golden, roots, strict=True):
            torch.testing.assert_close(
                value,
                torch.zeros_like(root) if trusted is None else trusted,
                atol=8e-5,
                rtol=8e-4,
            )
        return tuple(weakref.ref(value) for value in (*roots, runtime_dt))

    # The second call hits the same semantic/layout plan with new operands and
    # dt; the third requires a separate cotangent ABI. No plan may own a tape.
    references = [run(0.25, False), run(0.5, False), run(0.75, True)]
    gc.collect()
    assert all(reference() is None for group in references for reference in group)
