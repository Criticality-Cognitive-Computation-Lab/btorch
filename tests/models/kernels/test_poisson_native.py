"""Validate native Poisson ALIF/ELIF sequences against their actual Modules."""

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


def _case(neuron_cls, *, device="cpu", spiking=True, refractory=True):
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
        kwargs.update(delta_T=torch.tensor([0.32, 0.35, 0.38, 0.41]), v_T=0.75)
    if refractory:
        parameter_names.add("tau_ref")
    module = neuron_cls(
        4,
        v_threshold=torch.tensor([0.82, 0.90, 0.98, 1.06]),
        v_reset=-0.15,
        c_m=torch.tensor([1.0, 1.1, 1.2, 1.3]),
        g_leak=0.22,
        E_leak=-0.08,
        E_k=-1.0,
        tau_adapt=torch.tensor([4.5, 5.0, 5.5, 6.0]),
        dg_k=torch.tensor([0.08, 0.10, 0.12, 0.14]),
        tau_ref=0.75 if refractory else None,
        trainable_param=parameter_names,
        surrogate_function=surrogate.PoissonRandomSpike(
            spiking=spiking,
            tau=2.4,
            rho=0.75,
            leak=0.2,
            k=1.1,
            damping_factor=0.6,
        ),
        hard_reset=True,
        pre_spike_v=True,
        device=device,
        dtype=torch.float32,
        **kwargs,
    )
    module.init_state(batch_size=2, device=device, dtype=torch.float32)
    lane = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4)
    module.v = (0.20 + 0.035 * lane).requires_grad_()
    module.g_k = (0.025 + 0.008 * lane).requires_grad_()
    if refractory:
        module.refractory = torch.tensor(
            [[0.0, 0.25, 0.0, 0.5], [0.0, 0.0, 0.25, 0.0]],
            device=device,
            requires_grad=True,
        )
    # Slicing the backing lane checks that stochastic lowering accepts the
    # same ordinary strided input layouts as the deterministic cell path.
    backing = torch.empty((7, 2, 8), device=device, dtype=torch.float32)
    time = torch.arange(7, device=device, dtype=torch.float32)[:, None, None]
    backing[..., ::2] = 0.30 + 0.20 * torch.sin(time * 0.8 + lane * 0.35)
    backing[..., 1::2] = -9.0
    x = backing[..., ::2].requires_grad_()
    with environ.context(dt=0.25):
        spec, _, states, parameters = prepare_cell(module, x)
    return x, spec, states, parameters


def _flatten_golden(result):
    output, final, records = result
    return (output, *final, *records)


def _cuda_rng_state():
    return torch.cuda.get_rng_state().clone()


def test_invalid_poisson_requests_do_not_advance_the_cpu_generator():
    """Every host-checkable failure happens before stochastic draws begin."""
    x, spec, states, parameters = _case(ALIF)
    failures = (
        # A valid model on an unsupported device must be rejected before
        # Torch draws the sequence's uniforms, even when CUDA is installed.
        lambda: run_multi_step(x, states, parameters, spec, backend="triton"),
        lambda: run_multi_step(
            x,
            states,
            parameters,
            spec,
            HistoryPlan(requested=("missing",)),
            backend="triton",
        ),
        lambda: run_multi_step(
            x,
            (states[0][..., :3], *states[1:]),
            parameters,
            spec,
            backend="triton",
        ),
        lambda: run_multi_step(
            x,
            states,
            (*parameters[:-1], torch.ones(3, 2)),
            spec,
            backend="triton",
        ),
        lambda: run_multi_step(x, states, parameters, spec, backend="torch"),
        lambda: run_multi_step(
            x,
            states,
            parameters,
            replace(spec, dt=torch.tensor(float("nan"))),
            backend="triton",
        ),
    )
    for call in failures:
        before = torch.get_rng_state().clone()
        with pytest.raises((ValueError, NotImplementedError)):
            call()
        assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("target", ("input", "state", "parameter", "uniform"))
def test_reduced_precision_poisson_requests_fail_before_rng(dtype, target):
    """Native stochastic execution rejects every reduced-precision root.

    Event comparisons are discontinuous, so accepting a dtype whose Torch
    ``rand_like`` stream differs from FP32 would silently select another spike
    raster even when the final generator counter happens to match.
    """
    x, spec, states, parameters = _case(ALIF)
    states = list(states)
    parameters = list(parameters)
    uniforms = None
    if target == "input":
        x = x.to(dtype)
    elif target == "state":
        states[0] = states[0].to(dtype)
    elif target == "parameter":
        parameters[0] = parameters[0].to(dtype)
    else:
        uniforms = torch.empty(x.shape, dtype=dtype)

    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="requires FP32"):
        run_multi_step(
            x,
            tuple(states),
            tuple(parameters),
            spec,
            backend="triton",
            uniform_seq=uniforms,
        )
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
def test_native_poisson_whole_and_chunk_vjp_preserve_module_rng(backend, neuron_cls):
    """Whole and continued calls preserve events, states, VJPs, and RNG
    position."""
    require_cuda_backend(backend)
    x, spec, states, parameters = _case(neuron_cls, device="cuda")
    histories = get_template(spec).history_names
    roots = (x, *states, *parameters)
    options = KernelOptions(neuron=backend)

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(8143)
        initial_rng = _cuda_rng_state()
        expected = _flatten_golden(
            python_module_sequence(x, states, parameters, spec, histories)
        )
        expected_rng = _cuda_rng_state()

        torch.cuda.set_rng_state(initial_rng)
        actual = run_multi_step(
            x,
            states,
            parameters,
            spec,
            HistoryPlan(requested=histories),
            backend=options,
        )
        actual_rng = _cuda_rng_state()

        torch.cuda.set_rng_state(initial_rng)
        split = 3
        first = run_multi_step(
            x[:split],
            states,
            parameters,
            spec,
            HistoryPlan(requested=histories),
            backend=options,
        )
        second = run_multi_step(
            x[split:],
            first[1 : 1 + len(states)],
            parameters,
            spec,
            HistoryPlan(requested=histories),
            backend=options,
        )
        chunked = (
            torch.cat((first[0], second[0])),
            *second[1 : 1 + len(states)],
            *(
                torch.cat((first[1 + len(states) + i], value))
                for i, value in enumerate(second[1 + len(states) :])
            ),
        )
        chunk_rng = _cuda_rng_state()

        assert torch.equal(actual[0], expected[0])
        assert torch.equal(chunked[0], expected[0])
        assert actual[0].eq(0).any() and actual[0].eq(1).any()
        for candidate in (actual[1:], chunked[1:]):
            for value, reference in zip(candidate, expected[1:]):
                torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-5)
        assert torch.equal(actual_rng, expected_rng)
        assert torch.equal(chunk_rng, expected_rng)

        generator = torch.Generator().manual_seed(91)
        cotangents = []
        for index, value in enumerate(actual):
            cotangent = torch.randn(value.shape, generator=generator).to(value.device)
            # Only the second chunk and final memories contribute to the loss.
            # Its initial-state VJP therefore has to traverse the first native
            # call through the carried voltage, adaptation, and refractory state.
            if index == 0 or index >= 1 + len(states):
                cotangent[:split] = 0
            cotangents.append(cotangent)
        cotangents = tuple(cotangents)
        rng_before_backward = _cuda_rng_state()
        actual_grad = torch.autograd.grad(actual, roots, cotangents, allow_unused=True)
        chunk_grad = torch.autograd.grad(chunked, roots, cotangents, allow_unused=True)
        expected_grad = torch.autograd.grad(
            expected, roots, cotangents, allow_unused=True
        )
        assert torch.equal(_cuda_rng_state(), rng_before_backward)
        for candidate in (actual_grad, chunk_grad):
            for index, (value, reference) in enumerate(zip(candidate, expected_grad)):
                assert (value is None) == (reference is None), f"root {index}"
                if reference is not None:
                    torch.testing.assert_close(
                        value,
                        reference,
                        atol=3e-5,
                        rtol=3e-5,
                        msg=f"root {index}",
                    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_nonspiking_poisson_is_differentiable_without_rng_draws(backend):
    """The public analog Poisson mode keeps its Module primitive and VJP."""
    require_cuda_backend(backend)
    x, spec, states, parameters = _case(
        ALIF, device="cuda", spiking=False, refractory=False
    )
    roots = (x, *states, *parameters)
    histories = get_template(spec).history_names
    before = _cuda_rng_state()
    expected = _flatten_golden(
        python_module_sequence(x, states, parameters, spec, histories)
    )
    actual = run_multi_step(
        x,
        states,
        parameters,
        spec,
        HistoryPlan(requested=histories),
        backend=backend,
    )
    assert torch.equal(_cuda_rng_state(), before)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-5)
    cotangents = tuple(torch.randn_like(value) for value in actual)
    gradients = torch.autograd.grad(actual, roots, cotangents, allow_unused=True)
    references = torch.autograd.grad(expected, roots, cotangents, allow_unused=True)
    for value, reference in zip(gradients, references):
        assert (value is None) == (reference is None)
        if reference is not None:
            torch.testing.assert_close(value, reference, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_training_saves_sampled_raster_without_uniforms_or_backward_rng(backend):
    """Reverse uses authoritative events without retaining or redrawing
    uniforms."""
    require_cuda_backend(backend)
    x, spec, states, parameters = _case(ALIF, device="cuda")
    roots = (x, *states, *parameters)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(1701)
        uniforms = torch.stack(
            tuple(
                torch.rand(x.shape[1:], device=x.device, dtype=torch.float32)
                for _ in range(x.shape[0])
            )
        )
        state_after_draws = _cuda_rng_state()
        saved = []

        def pack(value):
            saved.append(value)
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            actual = run_multi_step(
                x,
                states,
                parameters,
                spec,
                backend=backend,
                uniform_seq=uniforms,
            )
        storage = lambda value: value.untyped_storage().data_ptr()
        saved_storage = {storage(value) for value in saved}
        assert storage(actual[0]) in saved_storage
        assert storage(uniforms) not in saved_storage
        assert torch.equal(_cuda_rng_state(), state_after_draws)
        # Cotangent generation must not advance the CUDA generator whose
        # continuation this test is checking. Use an independent CPU generator
        # so both explicit-uniform forward and reverse are held accountable.
        generator = torch.Generator().manual_seed(1702)
        cotangents = tuple(
            torch.randn(value.shape, generator=generator).to(value.device)
            for value in actual
        )
        gradients = torch.autograd.grad(actual, roots, cotangents, allow_unused=True)
        assert torch.equal(_cuda_rng_state(), state_after_draws)
        assert all(
            value is not None and torch.isfinite(value).all() for value in gradients
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_inference_keeps_only_model_outputs_and_advances_rng(backend):
    """Inference samples events without constructing a backward residual
    tape."""
    require_cuda_backend(backend)
    x, spec, states, parameters = _case(ALIF, device="cuda")
    histories = get_template(spec).history_names
    saved = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(992)
        initial_rng = _cuda_rng_state()
        with torch.no_grad():
            expected = _flatten_golden(
                python_module_sequence(x, states, parameters, spec, histories)
            )
        expected_rng = _cuda_rng_state()
        torch.cuda.set_rng_state(initial_rng)
        with torch.autograd.graph.saved_tensors_hooks(
            saved.append, lambda value: value
        ):
            with torch.no_grad():
                actual = run_multi_step(
                    x,
                    states,
                    parameters,
                    spec,
                    HistoryPlan(requested=histories),
                    backend=backend,
                )
        assert not saved
        assert torch.equal(_cuda_rng_state(), expected_rng)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-5)
        assert not value.requires_grad
