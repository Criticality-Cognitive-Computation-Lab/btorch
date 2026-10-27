"""Exercise private GLIF residual storage independently from public records."""

import pytest
import torch

from btorch.models.kernels.backends.engine import generate_source
from btorch.models.kernels.contracts import HistoryPlan, NeuronSpec
from btorch.models.kernels.golden import python_module_sequence
from btorch.models.kernels.operators import run_multi_step
from btorch.models.kernels.templates import residual_template
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


def _case(device, steps=20, refractory=False, hard_reset=False):
    torch.manual_seed(1204)
    shape = (2, 5)
    spec = NeuronSpec(
        "glif3",
        dt=0.5,
        refractory=refractory,
        hard_reset=hard_reset,
        surrogate="atan",
    )
    x = (torch.rand(steps, *shape, device=device) * 4.0).requires_grad_()
    states = [
        (torch.rand(*shape, device=device) * 0.3).requires_grad_(),
        (torch.rand(*shape, 2, device=device) * 0.2).requires_grad_(),
    ]
    parameters = [
        torch.tensor(value, device=device).requires_grad_()
        for value in (-0.05, -0.8, 1.2, 9.0, -0.2)
    ]
    parameters += [
        torch.tensor([0.04, 0.015], device=device).requires_grad_(),
        torch.tensor([-0.3, -0.1], device=device).requires_grad_(),
    ]
    if refractory:
        states.append(torch.zeros(shape, device=device, requires_grad=True))
        parameters.append(torch.tensor(1.5, device=device, requires_grad=True))
    return spec, x, tuple(states), tuple(parameters)


def test_glif_checkpoint_source_replaces_only_private_adaptation_history():
    spec, x, states, _ = _case("cpu")
    plan = residual_template(spec, ("v",))
    source, pointers = generate_source(
        "triton",
        spec,
        x[0].numel(),
        x.shape[0],
        ("v",),
        states[1].shape[-1],
        residual_template=plan,
    )
    names = dict(pointers)

    # The public voltage record remains a full trajectory. Adaptation is a
    # private block-boundary residual and the existing spike output drives the
    # ordinary stable recurrence during reverse replay.
    assert names["v_seq"] == x.numel()
    assert "Iasc_seq" not in names
    assert names["Iasc_checkpoint"] == ((x.shape[0] - 1) // 16) * states[1].numel()
    assert "Iasc_checkpoint_ptr" in source


def test_torch_glif_training_keeps_full_state_fallback():
    spec, x, states, parameters = _case("cpu", steps=5)
    expected = python_module_sequence(x, states, parameters, spec, ("v",))
    actual = run_multi_step(
        x, states, parameters, spec, HistoryPlan(requested=("v",)), backend="torch"
    )
    expected_flat = (expected[0], *expected[1], *expected[2])
    cotangents = tuple(torch.randn_like(value) for value in actual)
    inputs = (x, *states, *parameters)
    gradients = torch.autograd.grad(actual, inputs, cotangents)
    references = torch.autograd.grad(expected_flat, inputs, cotangents)
    for result, reference in zip(gradients, references):
        torch.testing.assert_close(result, reference)


def test_requested_adaptation_history_reuses_full_record_instead_of_checkpoint():
    spec, x, states, _ = _case("cpu")
    plan = residual_template(spec, ("Iasc",))
    _, pointers = generate_source(
        "triton",
        spec,
        x[0].numel(),
        x.shape[0],
        ("Iasc", "v"),
        states[1].shape[-1],
        residual_template=plan,
    )
    names = dict(pointers)
    assert names["Iasc_seq"] == x.shape[0] * states[1].numel()
    assert "Iasc_checkpoint" not in names


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize(
    "steps,refractory,hard_reset",
    ((0, False, False), (7, False, True), (20, True, False)),
)
def test_checkpointed_glif_matches_module_for_partial_blocks_and_record_losses(
    backend, steps, refractory, hard_reset
):
    require_cuda_backend(backend)
    spec, x, states, parameters = _case("cuda", steps, refractory, hard_reset)
    expected = python_module_sequence(x, states, parameters, spec, ("v",))
    actual = run_multi_step(
        x, states, parameters, spec, HistoryPlan(requested=("v",)), backend=backend
    )
    expected_flat = (expected[0], *expected[1], *expected[2])
    for result, reference in zip(actual, expected_flat):
        torch.testing.assert_close(result, reference, atol=5e-6, rtol=5e-5)

    # Independent random cotangents exercise output, final states and the shared
    # public/residual voltage trajectory. Adaptation state and k gradients need
    # the exact replayed pre-step Iasc values.
    generator = torch.Generator(device="cuda").manual_seed(88)
    cotangents = tuple(
        torch.randn(value.shape, generator=generator, device=value.device)
        for value in actual
    )
    inputs = (x, *states, *parameters)
    actual_gradients = torch.autograd.grad(actual, inputs, cotangents)
    # With no timesteps the Module returns plain empty output/record tensors.
    # Only its final states participate in autograd; omit the empty constants
    # from the VJP and treat unused input/parameter derivatives as zero below.
    differentiable = [
        (value, cotangent)
        for value, cotangent in zip(expected_flat, cotangents)
        if value.requires_grad
    ]
    expected_gradients = torch.autograd.grad(
        tuple(value for value, _ in differentiable),
        inputs,
        tuple(cotangent for _, cotangent in differentiable),
        allow_unused=True,
    )
    for result, reference, value in zip(actual_gradients, expected_gradients, inputs):
        if reference is None:
            reference = torch.zeros_like(value)
        torch.testing.assert_close(result, reference, atol=5e-5, rtol=5e-4)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_two_checkpointed_forwards_keep_independent_residuals(backend):
    require_cuda_backend(backend)
    first = _case("cuda", steps=17)
    second = _case("cuda", steps=19)
    references = [
        python_module_sequence(x, states, parameters, spec, ())
        for spec, x, states, parameters in (first, second)
    ]
    reference_flat = [(value[0], *value[1]) for value in references]
    outputs = [
        run_multi_step(x, states, parameters, spec, backend=backend)
        for spec, x, states, parameters in (first, second)
    ]
    cotangents = [
        tuple(torch.randn_like(value) for value in result) for result in outputs
    ]
    cases = reversed((first, second))
    paired = zip(
        reversed(outputs), reversed(reference_flat), reversed(cotangents), cases
    )
    for actual, expected, gradients, (_, x, states, parameters) in paired:
        inputs = (x, *states, *parameters)
        result = torch.autograd.grad(actual, inputs, gradients, retain_graph=True)
        reference = torch.autograd.grad(expected, inputs, gradients)
        for value, trusted in zip(result, reference):
            torch.testing.assert_close(value, trusted, atol=5e-5, rtol=5e-4)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_saved_tensors_keep_block_checkpoint_without_private_full_iasc(backend):
    require_cuda_backend(backend)
    spec, x, states, parameters = _case("cuda", steps=20)
    saved_shapes = []

    def pack(value):
        saved_shapes.append(tuple(value.shape))
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = run_multi_step(x, states, parameters, spec, backend=backend)
        sum(value.sum() for value in output).backward()

    assert (1, *states[1].shape) in saved_shapes
    assert (x.shape[0], *states[1].shape) not in saved_shapes
