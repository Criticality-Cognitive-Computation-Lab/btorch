"""Validate ALIF/ELIF tensor sequences against their actual Modules."""

from dataclasses import replace

import pytest
import torch

from btorch.models import environ, surrogate
from btorch.models.kernels import HistoryPlan, KernelOptions
from btorch.models.kernels.adapters import prepare_cell
from btorch.models.kernels.golden import python_module_sequence
from btorch.models.kernels.operators import run_multi_step
from btorch.models.kernels.templates import get_template
from btorch.models.neurons.alif import ALIF, ELIF
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


def _module_inputs(
    neuron_cls: type[ALIF],
    *,
    hard_reset: bool,
    detach_reset: bool,
    refractory: bool,
    tau_adapt: float | torch.Tensor = 2.7,
    v_T: float | torch.Tensor = 0.24,
):
    parameter_names = {
        "v_threshold",
        "v_reset",
        "c_m",
        "g_leak",
        "E_leak",
        "E_k",
        "tau_adapt",
        "dg_k",
    }
    kwargs = {}
    if neuron_cls is ELIF:
        parameter_names.update(("delta_T", "v_T"))
        kwargs.update(delta_T=torch.tensor([0.22, 0.25, 0.28]), v_T=v_T)
    if refractory:
        parameter_names.add("tau_ref")

    module = neuron_cls(
        3,
        v_threshold=torch.tensor([0.48, 0.56, 0.64]),
        v_reset=torch.tensor(-0.18),
        c_m=torch.tensor([1.0, 1.15, 1.3]),
        g_leak=torch.tensor([0.65, 0.75, 0.85]),
        E_leak=torch.tensor(-0.12),
        E_k=torch.tensor(-0.9),
        tau_adapt=tau_adapt,
        dg_k=torch.tensor([0.07, 0.09, 0.11]),
        tau_ref=0.75 if refractory else None,
        trainable_param=parameter_names,
        surrogate_function=surrogate.ATan(alpha=1.8, damping_factor=0.7),
        hard_reset=hard_reset,
        detach_reset=detach_reset,
        pre_spike_v=True,
        **kwargs,
    )
    module.init_state(batch_size=2, dtype=torch.float32)
    lane = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    module.v = (0.12 + 0.075 * lane).requires_grad_()
    module.g_k = (0.035 + 0.012 * lane).requires_grad_()
    if refractory:
        module.refractory = torch.tensor(
            [[0.0, 0.25, 0.0], [0.5, 0.0, 0.25]], requires_grad=True
        )

    time = torch.arange(7, dtype=torch.float32)[:, None, None]
    drive = 0.14 + 0.16 * torch.sin(time * 0.73 + lane * 0.41)
    drive = drive + (time == 2).to(drive.dtype) * torch.tensor([1.5, 0.8, 1.8])
    drive = drive + (time == 5).to(drive.dtype) * torch.tensor([0.4, 1.4, 0.6])
    x = drive.requires_grad_()
    with environ.context(dt=0.25):
        spec, state_names, states, parameters = prepare_cell(module, x)
    return x, spec, state_names, states, parameters


def _compare_with_module(
    x: torch.Tensor,
    spec,
    states: tuple[torch.Tensor, ...],
    parameters: tuple[torch.Tensor, ...],
    *,
    backend: str | KernelOptions = "torch",
):
    histories = get_template(spec).history_names
    expected = python_module_sequence(x, states, parameters, spec, histories)
    expected_flat = (expected[0], *expected[1], *expected[2])
    actual = run_multi_step(
        x,
        states,
        parameters,
        spec,
        HistoryPlan(requested=histories),
        backend=backend,
    )
    split = 3
    first = run_multi_step(
        x[:split],
        states,
        parameters,
        spec,
        HistoryPlan(requested=histories),
        backend=backend,
    )
    n_states = len(states)
    second = run_multi_step(
        x[split:],
        first[1 : 1 + n_states],
        parameters,
        spec,
        HistoryPlan(requested=histories),
        backend=backend,
    )
    chunked = (
        torch.cat((first[0], second[0])),
        *second[1 : 1 + n_states],
        *(
            torch.cat((first[1 + n_states + i], second[1 + n_states + i]))
            for i in range(len(histories))
        ),
    )

    for values in (actual, chunked):
        for value, reference in zip(values, expected_flat):
            torch.testing.assert_close(value, reference)

    # Independent cotangents on spikes, final memories, and every public trace
    # exercise reset, adaptation, refractory, and charged-voltage VJP paths.
    generator = torch.Generator().manual_seed(902)
    cotangents = tuple(
        torch.randn(value.shape, generator=generator, dtype=value.dtype).to(
            value.device
        )
        for value in actual
    )
    roots = (x, *states, *parameters)
    gradients = torch.autograd.grad(actual, roots, cotangents, allow_unused=True)
    chunk_gradients = torch.autograd.grad(chunked, roots, cotangents, allow_unused=True)
    references = torch.autograd.grad(
        expected_flat, roots, cotangents, allow_unused=True
    )
    for candidate in (gradients, chunk_gradients):
        for index, (value, reference) in enumerate(zip(candidate, references)):
            assert (value is None) == (reference is None), f"root {index}"
            if reference is not None:
                assert torch.isfinite(value).all(), f"root {index}"
                torch.testing.assert_close(
                    value, reference, atol=1e-5, rtol=1e-4, msg=f"root {index}"
                )
    return actual, gradients


@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("refractory", [False, True])
def test_torch_sequence_matches_module_for_reset_and_refractory_options(
    neuron_cls, hard_reset, detach_reset, refractory
):
    """The tensor boundary preserves every supported ALIF-family phase
    option."""
    x, spec, _, states, parameters = _module_inputs(
        neuron_cls,
        hard_reset=hard_reset,
        detach_reset=detach_reset,
        refractory=refractory,
    )
    actual, gradients = _compare_with_module(x, spec, states, parameters)

    # The drive deliberately mixes threshold crossings and quiet lanes. This
    # prevents reset and refractory comparisons from passing on an all-zero or
    # all-one spike sequence.
    assert 0 < int(actual[0].sum()) < actual[0].numel()
    assert all(value is not None for value in gradients)


@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
def test_torch_sequence_matches_module_at_infinite_adaptation_tau(neuron_cls):
    """Infinite adaptation tau follows the Module's finite no-decay limit."""
    x, spec, _, states, parameters = _module_inputs(
        neuron_cls,
        hard_reset=False,
        detach_reset=False,
        refractory=False,
        tau_adapt=float("inf"),
    )
    actual, _ = _compare_with_module(x, spec, states, parameters)
    assert all(torch.isfinite(value).all() for value in actual)


def test_elif_high_soft_threshold_reduces_to_alif_sequence_and_vjp():
    """A negligible ELIF exponential term preserves the ALIF operator limit."""
    alif = _module_inputs(ALIF, hard_reset=True, detach_reset=False, refractory=False)
    elif_ = _module_inputs(
        ELIF,
        hard_reset=True,
        detach_reset=False,
        refractory=False,
        v_T=100.0,
    )
    alif_actual, _ = _compare_with_module(alif[0], alif[1], alif[3], alif[4])
    elif_actual, _ = _compare_with_module(elif_[0], elif_[1], elif_[3], elif_[4])

    # Compare spikes, v/g_k finals, and v/g_k/charged histories. ELIF's two
    # additional parameter roots are intentionally outside this model limit.
    alif_common = (alif_actual[0], *alif_actual[1:3], *alif_actual[3:])
    elif_common = (elif_actual[0], *elif_actual[1:3], *elif_actual[3:])
    for value, reference in zip(elif_common, alif_common):
        torch.testing.assert_close(value, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("refractory", [False, True])
def test_native_sequence_preserves_module_full_bptt(
    backend, neuron_cls, hard_reset, detach_reset, refractory
):
    """Native whole/chunk execution preserves all scientific gradient roots."""
    require_cuda_backend(backend)
    x, spec, _, states, parameters = _module_inputs(
        neuron_cls,
        hard_reset=hard_reset,
        detach_reset=detach_reset,
        refractory=refractory,
    )

    # Move each original root independently: shared scalar and per-neuron
    # parameters must retain their own reductions, and the second forward
    # must backpropagate through the first forward's voltage/adaptation state.
    def cuda_leaf(value):
        return value.detach().cuda().requires_grad_(value.requires_grad)

    actual, gradients = _compare_with_module(
        cuda_leaf(x),
        spec,
        tuple(cuda_leaf(value) for value in states),
        tuple(cuda_leaf(value) for value in parameters),
        backend=backend,
    )
    assert 0 < int(actual[0].sum()) < actual[0].numel()
    assert all(value is not None for value in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
@pytest.mark.parametrize(
    "surrogate_name",
    ["sigmoid", "atan", "atan_approx", "erf", "superspike", "triangle"],
)
def test_native_surrogates_keep_normalized_module_derivatives(
    backend, neuron_cls, surrogate_name
):
    """All deterministic surrogates retain reset/adaptation/refractory VJPs."""
    require_cuda_backend(backend)
    x, spec, _, states, parameters = _module_inputs(
        neuron_cls, hard_reset=False, detach_reset=False, refractory=True
    )
    roots = tuple(
        value.detach().cuda().requires_grad_(value.requires_grad)
        for value in (x, *states, *parameters)
    )
    _compare_with_module(
        roots[0],
        replace(spec, surrogate=surrogate_name),
        roots[1 : 1 + len(states)],
        roots[1 + len(states) :],
        backend=backend,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
@pytest.mark.parametrize("limit", ["zero_linear", "infinite_tau"])
def test_native_exp_euler_limits_preserve_module_branch_vjps(
    backend, neuron_cls, limit
):
    """The exact zero-linear branch and no-decay limit keep Module
    semantics."""
    require_cuda_backend(backend)
    x, spec, _, states, parameters = _module_inputs(
        neuron_cls,
        hard_reset=True,
        detach_reset=False,
        refractory=False,
        tau_adapt=float("inf") if limit == "infinite_tau" else 2.7,
    )
    if limit == "zero_linear":
        # Both conductances start at zero. The first step therefore takes
        # phi(A=0)=dt, whose selected Module branch has zero derivative in A.
        # Later spikes add adaptation, exercising the ordinary branch too.
        states = (states[0], torch.zeros_like(states[1], requires_grad=True))
        parameters = (
            *parameters[:3],
            torch.zeros_like(parameters[3], requires_grad=True),
            *parameters[4:],
        )
    roots = tuple(
        value.detach().cuda().requires_grad_(value.requires_grad)
        for value in (x, *states, *parameters)
    )
    _compare_with_module(
        roots[0],
        spec,
        roots[1 : 1 + len(states)],
        roots[1 + len(states) :],
        backend=backend,
    )
