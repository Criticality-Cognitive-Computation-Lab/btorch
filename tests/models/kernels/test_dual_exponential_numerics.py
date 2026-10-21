"""Guard long-horizon DualExponentialPSC state and parameter numerics."""

import pytest
import torch

from btorch.models.kernels import HistoryPlan, KernelOptions
from btorch.models.kernels.backends import engine
from btorch.models.kernels.contracts import CellSpec
from btorch.models.kernels.golden import python_module_sequence
from btorch.models.kernels.operators import run_multi_step
from btorch.models.kernels.templates import get_template
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("fp_policy", ("default", "strict"))
def test_long_dual_exponential_sequence_matches_module_state_and_parameter_vjps(
    backend, fp_policy
):
    """Preserve the Module's exponential-Euler graph over 128 steps.

    Nonzero incoming memories make both decay branches observable from
    the first step. Mixed scalar and per-neuron time constants exercise
    both the elementwise reverse and broadcast reduction, while record
    cotangents keep the stored rise/decay trajectories in the loss
    rather than checking only their normalized PSC readout.
    """
    require_cuda_backend(backend)

    steps, neurons = 128, 7
    spec = CellSpec("dual_exponential", dt=0.3)
    template = get_template(spec)
    time = torch.arange(steps, device="cuda", dtype=torch.float32)[:, None, None]
    lane = torch.arange(neurons, device="cuda", dtype=torch.float32)[None, None]
    source = 0.08 * torch.sin(time * 0.17 + lane * 0.31)
    source = source + 0.05 * torch.cos(time * 0.07 + lane * 0.23)
    initial = (
        0.04 + 0.006 * lane[0],
        -0.03 + 0.009 * lane[0],
        0.07 - 0.005 * lane[0],
    )
    parameters = (
        torch.tensor(1.7, device="cuda"),
        torch.linspace(3.8, 5.0, neurons, device="cuda"),
        torch.tensor(0.73, device="cuda"),
    )

    native_source = source.detach().clone().requires_grad_()
    module_source = source.detach().clone().requires_grad_()
    native_states = tuple(value.detach().clone().requires_grad_() for value in initial)
    module_states = tuple(value.detach().clone().requires_grad_() for value in initial)
    native_parameters = tuple(
        value.detach().clone().requires_grad_() for value in parameters
    )
    module_parameters = tuple(
        value.detach().clone().requires_grad_() for value in parameters
    )
    options = KernelOptions(neuron=backend)
    if fp_policy == "strict":
        options = KernelOptions(
            neuron=backend, enable_fp_fusion=False, precise_division=True
        )
    histories = template.history_names

    actual = run_multi_step(
        native_source,
        native_states,
        native_parameters,
        spec,
        HistoryPlan(requested=histories),
        backend=options,
    )
    expected_output, expected_states, expected_records = python_module_sequence(
        module_source, module_states, module_parameters, spec, histories
    )
    expected = (
        expected_output,
        *expected_states,
        *expected_records,
    )
    for value, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, reference, atol=5e-6, rtol=5e-5)

    generator = torch.Generator(device="cuda").manual_seed(818)
    cotangents = tuple(
        torch.randn(value.shape, device="cuda", generator=generator) for value in actual
    )
    native_roots = (native_source, *native_states, *native_parameters)
    module_roots = (module_source, *module_states, *module_parameters)
    gradients = torch.autograd.grad(actual, native_roots, cotangents, allow_unused=True)
    references = torch.autograd.grad(
        expected, module_roots, cotangents, allow_unused=True
    )
    names = ("source", *template.state_names, *template.parameter_names)
    for name, gradient, reference, root in zip(
        names, gradients, references, native_roots, strict=True
    ):
        if name == "psc":
            # DualExponentialPSC replaces ``psc`` from the two conductance
            # memories at the first step, so its incoming value is not an
            # ancestor in the Module graph. The explicit native reverse returns
            # the equivalent materialized zero for this requested state root.
            assert reference is None
            assert gradient is not None
            torch.testing.assert_close(gradient, torch.zeros_like(root))
            continue
        assert gradient is not None, name
        assert reference is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.isfinite(reference).all(), name
        torch.testing.assert_close(gradient, reference, atol=5e-5, rtol=5e-4, msg=name)
        assert bool(gradient.abs().sum() > 0), name
        assert bool(reference.abs().sum() > 0), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_precise_exp_dual_sequence_matches_actual_module_forward_and_vjp(backend):
    """Match the Module across per-neuron decay, readout, and parameter
    VJPs."""
    require_cuda_backend(backend)

    steps, neurons = 128, 64
    spec = CellSpec("dual_exponential", dt=1.0)
    template = get_template(spec)
    time = torch.arange(steps, device="cuda", dtype=torch.float32)[:, None, None]
    lane = torch.arange(neurons, device="cuda", dtype=torch.float32)[None, None]
    source = 0.09 * torch.sin(time * 0.17 + lane * 0.13)
    initial = (
        0.02 + 0.001 * lane[0],
        -0.03 + 0.002 * lane[0],
        0.08 - 0.001 * lane[0],
    )
    parameters = (
        torch.linspace(1.44, 2.73, neurons, device="cuda"),
        torch.linspace(4.48, 7.01, neurons, device="cuda"),
        torch.linspace(0.7, 1.1, neurons, device="cuda"),
    )
    native_roots = tuple(
        value.detach().clone().requires_grad_()
        for value in (source, *initial, *parameters)
    )
    module_roots = tuple(
        value.detach().clone().requires_grad_()
        for value in (source, *initial, *parameters)
    )
    native_source, *native_tail = native_roots
    module_source, *module_tail = module_roots
    native_states = tuple(native_tail[:3])
    native_parameters = tuple(native_tail[3:])
    module_states = tuple(module_tail[:3])
    module_parameters = tuple(module_tail[3:])
    histories = template.history_names

    actual = engine.forward(
        backend,
        native_source,
        native_states,
        native_parameters,
        spec,
        histories,
        enable_fp_fusion=False,
        precise_division=True,
        precise_exp=True,
    )
    expected_output, expected_states, expected_histories = python_module_sequence(
        module_source, module_states, module_parameters, spec, histories
    )
    actual_flat = (actual[0], *actual[1], *actual[2])
    expected_flat = (expected_output, *expected_states, *expected_histories)
    for value, reference in zip(actual_flat, expected_flat, strict=True):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-5)

    generator = torch.Generator(device="cuda").manual_seed(382)
    cotangents = tuple(
        torch.randn(value.shape, device="cuda", generator=generator)
        for value in actual_flat
    )
    actual_gradients = engine.backward(
        backend,
        native_source,
        native_states,
        native_parameters,
        spec,
        histories,
        actual[2],
        cotangents[0],
        cotangents[1:4],
        cotangents[4:],
        needs_input_grad=(True,) * len(native_roots),
        enable_fp_fusion=False,
        precise_division=True,
        precise_exp=True,
    )
    actual_gradients = (
        actual_gradients[0],
        *actual_gradients[1],
        *actual_gradients[2],
    )
    expected_gradients = torch.autograd.grad(
        expected_flat, module_roots, cotangents, allow_unused=True
    )
    for name, value, reference, root in zip(
        ("input", *template.state_names, *template.parameter_names),
        actual_gradients,
        expected_gradients,
        native_roots,
        strict=True,
    ):
        if reference is None:
            assert name == "psc"
            torch.testing.assert_close(value, torch.zeros_like(root))
        else:
            torch.testing.assert_close(value, reference, atol=5e-5, rtol=5e-4, msg=name)
