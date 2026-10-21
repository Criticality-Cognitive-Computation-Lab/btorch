"""Verify recurrent training retains only histories needed by reverse."""

import copy

import pytest
import torch

from btorch.models import environ
from btorch.models.kernels import KernelOptions
from btorch.models.neurons.izhikevich import Izhikevich
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC
from tests.models.kernels.helpers import (
    GPU_BACKENDS,
    NewEmptyShapes,
    require_cuda_backend,
)


def _model(neuron_model, backend, requested=(), device="cpu"):
    if neuron_model == "lif":
        neuron = LIF(
            4,
            v_threshold=0.45,
            pre_spike_v=True,
            trainable_param={"c_m", "v_threshold"},
        )
        charged = ("v_pre_spike",)
    else:
        neuron = Izhikevich(
            4,
            v_threshold=0.45,
            v_peak=0.8,
            pre_spike=True,
            trainable_param={"c", "v_threshold"},
        )
        charged = ("v_pre_spike", "u_pre_spike")
    model = RecurrentNN(
        neuron,
        AlphaPSC(4, 2.0, torch.nn.Linear(4, 4)),
        backend=KernelOptions(neuron=backend),
        update_state_names=tuple(f"neuron.{name}" for name in requested),
    ).to(device=device, dtype=torch.float32)
    model.neuron.init_state(batch_size=2, device=device, dtype=torch.float32)
    model.synapse.init_state(batch_size=2, device=device, dtype=torch.float32)
    for name in (*(("v",) if neuron_model == "lif" else ("v", "u")), "psc", "h"):
        owner = model.neuron if hasattr(model.neuron, name) else model.synapse
        getattr(owner, name).requires_grad_()
    return model, charged


def _state_inputs(model, neuron_model):
    names = ("v",) if neuron_model == "lif" else ("v", "u")
    return [
        *(getattr(model.neuron, name) for name in names),
        model.synapse.psc,
        model.synapse.h,
    ]


def _run_two(model, first, second):
    first_output, first_records = model(first)
    first_charged = tuple(
        getattr(model.neuron, name)
        for name in ("v_pre_spike", "u_pre_spike")
        if hasattr(model.neuron, name)
    )
    second_output, second_records = model(second)
    second_charged = tuple(
        getattr(model.neuron, name)
        for name in ("v_pre_spike", "u_pre_spike")
        if hasattr(model.neuron, name)
    )
    loss = first_output.square().sum() + 0.7 * second_output.square().sum()
    loss = loss + sum(value.square().sum() for value in first_charged)
    loss = loss + 0.5 * sum(value.square().sum() for value in second_charged)
    loss = loss + sum(value.square().sum() for value in first_records.values())
    loss = loss + 0.3 * sum(value.square().sum() for value in second_records.values())
    loss = loss + model.neuron.v.square().sum() + model.synapse.psc.square().sum()
    return (
        first_output,
        second_output,
        first_records,
        second_records,
        first_charged,
        second_charged,
        loss,
    )


def _assert_run_matches(model, reference, x1, x2, neuron_model, *, allocation=False):
    xr1 = x1.detach().clone().requires_grad_()
    xr2 = x2.detach().clone().requires_grad_()
    initial = _state_inputs(model, neuron_model)
    initial_reference = _state_inputs(reference, neuron_model)
    parameters = list(model.parameters())
    reference_parameters = list(reference.parameters())
    saved_pointers = []
    allocations = NewEmptyShapes()

    # Saved-tensor hooks expose exactly what the custom backward retains, while
    # the dispatch mode counts the trajectory buffers allocated by forward.
    def pack(value):
        saved_pointers.append(value.untyped_storage().data_ptr())
        return value

    if allocation:
        from btorch.models.kernels import recurrent_operator

        trajectory = recurrent_operator._trajectory

        def instrumented_trajectory(*args, **kwargs):
            with allocations:
                return trajectory(*args, **kwargs)

        recurrent_operator._trajectory = instrumented_trajectory
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            actual = _run_two(model, x1, x2)
    finally:
        if allocation:
            recurrent_operator._trajectory = trajectory
    expected = _run_two(reference, xr1, xr2)

    for value, target in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(value, target, atol=3e-5, rtol=3e-5)
    for records, target_records in zip(actual[2:4], expected[2:4]):
        assert tuple(records) == tuple(target_records)
        for name in records:
            torch.testing.assert_close(
                records[name], target_records[name], atol=3e-5, rtol=3e-5
            )
    for values, targets in zip(actual[4:6], expected[4:6]):
        for value, target in zip(values, targets):
            torch.testing.assert_close(value, target, atol=3e-5, rtol=3e-5)

    actual_gradients = torch.autograd.grad(
        actual[-1], (x1, x2, *initial, *parameters), allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected[-1],
        (xr1, xr2, *initial_reference, *reference_parameters),
        allow_unused=True,
    )
    for value, target, source in zip(
        actual_gradients, expected_gradients, (x1, x2, *initial, *parameters)
    ):
        value = torch.zeros_like(source) if value is None else value
        target = torch.zeros_like(source) if target is None else target
        torch.testing.assert_close(value, target, atol=5e-5, rtol=5e-5)

    # Final charged memories are differentiable saved outputs, but they do not
    # require a T-by-batch-by-neuron allocation unless requested as a record.
    for value in actual[5]:
        assert value.untyped_storage().data_ptr() in saved_pointers
    return allocations


@pytest.mark.parametrize("steps", [1, 5])
@pytest.mark.parametrize("neuron_model", ["lif", "izhikevich"])
@pytest.mark.parametrize("request_charged", [False, True])
def test_torch_fallback_final_charged_values_match_module_vjp_and_storage(
    reference_backend, steps, neuron_model, request_charged
):
    torch.manual_seed(941)
    requested = ("v_pre_spike",) if request_charged else ()
    model, _ = _model(neuron_model, reference_backend, requested)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x1 = torch.rand(steps, 2, 4, requires_grad=True)
    x2 = torch.rand(steps, 2, 4, requires_grad=True)
    with environ.context(dt=1.0):
        allocations = _assert_run_matches(
            model, reference, x1, x2, neuron_model, allocation=steps > 1
        )

    if steps > 1:
        physical_states = 3 if neuron_model == "lif" else 4
        expected_per_forward = physical_states + int(request_charged)
        assert allocations.shapes.count(tuple(x1.shape)) == 2 * expected_per_forward


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("neuron_model", ["lif", "izhikevich"])
def test_native_recurrent_final_charged_memory_matches_module_vjp(
    backend, neuron_model
):
    require_cuda_backend(backend)
    torch.manual_seed(942)
    model, _ = _model(neuron_model, backend, device="cuda")
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x1 = torch.rand(5, 2, 4, device="cuda", requires_grad=True)
    x2 = torch.rand(5, 2, 4, device="cuda", requires_grad=True)
    with environ.context(dt=1.0):
        _assert_run_matches(model, reference, x1, x2, neuron_model)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_final_charged_training_survives_cudagraph_tree_capture(backend):
    require_cuda_backend(backend)
    torch.manual_seed(944)
    model, _ = _model("lif", backend, device="cuda")
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = torch.rand(4, 2, 4, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()

    def loss(value):
        output, _ = model(value)
        return output.square().sum() + model.neuron.v_pre_spike.square().sum()

    # reduce-overhead uses CUDA Graph Trees for the compiled forward and
    # backward. The final charged tensor must remain a stable graph output and
    # its cotangent must still enter only the last recurrent step.
    torch.compiler.cudagraph_mark_step_begin()
    with environ.context(dt=1.0):
        actual = torch.compile(loss, mode="reduce-overhead", fullgraph=True)(x)
        expected_output, _ = reference(xr)
        expected = (
            expected_output.square().sum() + reference.neuron.v_pre_spike.square().sum()
        )
    actual_gradient = torch.autograd.grad(actual, x)[0]
    expected_gradient = torch.autograd.grad(expected, xr)[0]
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=5e-5, rtol=5e-5)


def test_final_charged_training_is_fake_tensor_and_fullgraph_compatible(
    reference_backend,
):
    torch.manual_seed(943)
    model, _ = _model("izhikevich", reference_backend)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = torch.rand(3, 2, 4, requires_grad=True)
    xr = x.detach().clone().requires_grad_()

    # Fullgraph AOTAutograd exercises the registered fake implementation and
    # its custom backward using O(1) charged outputs rather than eager internals.
    def loss(module, value):
        output, _ = module(value)
        return (
            output.square().sum()
            + module.neuron.v_pre_spike.square().sum()
            + module.neuron.u_pre_spike.square().sum()
        )

    with environ.context(dt=1.0):
        actual = torch.compile(
            lambda value: loss(model, value), backend="aot_eager", fullgraph=True
        )(x)
        expected = loss(reference, xr)
    actual_gradient = torch.autograd.grad(actual, x)[0]
    expected_gradient = torch.autograd.grad(expected, xr)[0]
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=5e-5, rtol=5e-5)


def test_zero_length_sequence_preserves_charged_module_memory(
    reference_backend,
):
    model, charged = _model("izhikevich", reference_backend)
    before = tuple(getattr(model.neuron, name).clone() for name in charged)
    x = torch.empty(0, 2, 4, requires_grad=True)
    with environ.context(dt=1.0):
        output, records = model(x)
    assert output.shape == x.shape
    assert records == {}
    assert torch.autograd.grad(output.sum(), x)[0].shape == x.shape
    for name, expected in zip(charged, before):
        torch.testing.assert_close(getattr(model.neuron, name), expected)
