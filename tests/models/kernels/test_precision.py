"""Qualify native floating-point policies against the actual Module
recurrence."""

import pytest
import torch

from btorch.models.kernels import KernelOptions, NeuronSpec
from btorch.models.kernels.backends import engine
from btorch.models.kernels.backends.engine import backward, forward
from btorch.models.kernels.golden import python_module_sequence
from btorch.models.kernels.templates import get_template
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


PARAMETER_VALUES = (-40.0, -65.0, 100.0, 2.0, -60.0, 30.0, 0.02, 0.2, 8.0)


def _inputs(steps: int, neurons: int):
    # This drive produces repeated spikes and a long nonlinear recurrence. It is
    # deliberately sensitive to contraction but retains identical spike events,
    # so state and VJP comparisons expose numerical policy rather than a trivial
    # threshold-crossing discontinuity.
    offsets = torch.arange(steps * neurons, device="cuda", dtype=torch.float32)
    x_seq = (200.0 + offsets.remainder(40.0) + 0.5).reshape(steps, neurons)
    return (
        x_seq.requires_grad_(),
        torch.full((neurons,), -65.0, device="cuda"),
        torch.zeros(neurons, device="cuda"),
    )


def _parameters():
    # Scalars exercise the direct parameter-storage path used by the external
    # 002HIP comparison that identified the contraction-sensitive trajectory.
    return tuple(
        torch.tensor(value, device="cuda", dtype=torch.float32)
        for value in PARAMETER_VALUES
    )


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((False, 128), id="unfused"),
        pytest.param((True, 4096), id="accurate-division"),
    ],
)
def izh_reference(request):
    """Keep policy-sensitive neuron counts while sharing one physical oracle.

    Both cases use the same Module equations and nontrivial repeated
    spikes. The larger division case exposes long-run FP32 division
    differences.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    precise_division, neurons = request.param
    spec = NeuronSpec("izhikevich", hard_reset=True, alpha=100.0)
    parameters = _parameters()
    x, v, u = _inputs(128, neurons)
    output, final, records = python_module_sequence(
        x, (v, u), parameters, spec, ("v", "u")
    )
    trajectories = (output, *records)
    grad_x = torch.autograd.grad(
        trajectories, x, tuple(torch.ones_like(value) for value in trajectories)
    )[0]
    return (
        neurons,
        spec,
        parameters,
        tuple(value.detach() for value in trajectories),
        tuple(value.detach() for value in final),
        grad_x.detach(),
        dict(enable_fp_fusion=False, precise_division=precise_division),
    )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_long_izhikevich_policy_matches_module_forward_and_vjp(backend, izh_reference):
    require_cuda_backend(backend)
    neurons, spec, parameters, expected, final, expected_grad, policy = izh_reference
    x, v, u = _inputs(128, neurons)
    actual = forward(backend, x, (v, u), parameters, spec, ("v", "u"), **policy)
    gradient = backward(
        backend,
        x,
        (v, u),
        parameters,
        spec,
        ("v", "u"),
        actual[2],
        torch.ones_like(actual[0]),
        (None, None),
        tuple(torch.ones_like(value) for value in actual[2]),
        needs_input_grad=(True, False, False, *(False,) * len(parameters)),
        **policy,
    )[0]
    # A silent or constant trajectory would not exercise reset recurrence.
    assert actual[0].sum() == 2 * neurons
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    for value, reference in zip(
        (*actual[2], *actual[1]), (*expected[1:], *final), strict=True
    ):
        torch.testing.assert_close(value, reference, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gradient, expected_grad, atol=5e-5, rtol=5e-4)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize(
    ("option", "default"),
    [
        ("enable_fp_fusion", True),
        ("precise_division", False),
    ],
)
def test_implicit_math_policy_matches_explicit_forward_and_reverse(
    backend, option, default
):
    require_cuda_backend(backend)
    spec = NeuronSpec("izhikevich", hard_reset=True, alpha=100.0)
    parameters = _parameters()
    x, v, u = _inputs(64, 32)
    # Isolate division policy from contraction when comparing its default.
    base = {"enable_fp_fusion": False} if option == "precise_division" else {}
    explicit_policy = {**base, option: default}
    implicit = forward(backend, x, (v, u), parameters, spec, ("v", "u"), **base)
    explicit = forward(
        backend, x, (v, u), parameters, spec, ("v", "u"), **explicit_policy
    )
    assert implicit[0].sum() > 0
    for value, reference in zip(
        (implicit[0], *implicit[1], *implicit[2]),
        (explicit[0], *explicit[1], *explicit[2]),
        strict=True,
    ):
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-6)
    reverse = []
    for result, policy in ((implicit, base), (explicit, explicit_policy)):
        gradients = backward(
            backend,
            x,
            (v, u),
            parameters,
            spec,
            ("v", "u"),
            result[2],
            torch.ones_like(result[0]),
            tuple(torch.ones_like(value) for value in result[1]),
            tuple(torch.ones_like(value) for value in result[2]),
            **policy,
        )
        reverse.append((gradients[0], *gradients[1], *gradients[2]))
    for value, reference in zip(*reverse, strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "option", ("enable_fp_fusion", "precise_division", "precise_exp")
)
def test_math_policy_rejects_non_boolean_values(option):
    # A truthy integer would create ambiguous compile/capture identities.
    with pytest.raises(TypeError, match=f"{option} must be a bool"):
        KernelOptions(**{option: 1})


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("backward", (False, True))
def test_precise_exp_changes_source_policy_without_changing_default(backend, backward):
    """Keep historical lowering as the default and opt into accurate exp."""
    spec = NeuronSpec("dual_exponential", dt=0.3)
    arguments = (backend, spec, 8, 4, get_template(spec).history_names)

    default, _ = engine.generate_source(*arguments, backward=backward)
    precise, _ = engine.generate_source(*arguments, backward=backward, precise_exp=True)

    if backend == "triton":
        assert "tl.exp(" in default
        assert "libdevice.exp(" not in default
        assert "libdevice.exp(" in precise
        assert "tl.exp(" not in precise
    elif backend == "tilelang":
        assert "T.exp(" in default
        assert 'T.call_pure_extern("float32", "expf", ' not in default
        assert 'T.call_pure_extern("float32", "expf", ' in precise
        assert "T.exp(" not in precise
    else:
        # CuPy's historical lowering is already the standard accurate expf.
        assert "expf(" in default
        assert "expf(" in precise


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_precise_exp_participates_in_generated_source_cache_key(backend):
    """Do not let default and opt-in requests share a source-cache entry."""
    spec = NeuronSpec("dual_exponential", dt=0.3)
    engine._generate_source.cache_clear()

    engine.generate_source(backend, spec, 8, 4, (), precise_exp=False)
    engine.generate_source(backend, spec, 8, 4, (), precise_exp=True)
    engine.generate_source(backend, spec, 8, 4, (), precise_exp=False)

    info = engine._generate_source.cache_info()
    assert info.misses == 2
    assert info.hits == 1
