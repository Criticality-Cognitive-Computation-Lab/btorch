"""Check that sequence length is invocation data for generated cell kernels."""

import importlib.util

import pytest
import torch

from btorch.models.kernels.backends import engine
from btorch.models.kernels.backends.engine import backward, forward
from btorch.models.kernels.contracts import NeuronSpec
from btorch.models.kernels.templates import ResidualTemplate


def _require_backend(backend: str) -> None:
    if not torch.cuda.is_available() or importlib.util.find_spec(backend) is None:
        pytest.skip(f"{backend} CUDA runtime is unavailable")


def test_runtime_length_reuses_triton_and_cupy_source():
    spec = NeuronSpec("lif", hard_reset=True)
    layouts = tuple(engine._ParameterLayout((), (), (8,), 1) for _ in range(4))
    for backend in ("triton", "cupy"):
        engine._generate_source.cache_clear()
        short = engine.generate_source(
            backend, spec, 8, 1, ("v",), parameter_layouts=layouts
        )
        long = engine.generate_source(
            backend, spec, 8, 17, ("v",), parameter_layouts=layouts
        )
        assert long == short
        assert engine._generate_source.cache_info().hits == 1
        assert "n_steps" in short[0]
        assert "n_steps" not in dict(short[1])


def test_runtime_length_reuses_checkpoint_source():
    spec = NeuronSpec("glif3")
    residual = ResidualTemplate(("v",), "Iasc", 16, True)
    first = engine.generate_source(
        "triton",
        spec,
        8,
        1,
        ("v",),
        modes=2,
        backward=True,
        residual_template=residual,
    )
    second = engine.generate_source(
        "triton",
        spec,
        8,
        17,
        ("v",),
        modes=2,
        backward=True,
        residual_template=residual,
    )
    assert second == first
    assert "n_steps - 1" in first[0]
    assert "if time >= 16:" in first[0]
    assert "Iasc_checkpoint" in dict(first[1])


def test_runtime_length_canonicalizes_noncontiguous_time_storage():
    short = torch.empty((1, 4, 8))[:, :, ::2]
    long = torch.empty((17, 4, 8))[:, :, ::2]
    short_storage = engine._tensor_storage({"x_seq": short}, dynamic_names={"x_seq"})
    long_storage = engine._tensor_storage({"x_seq": long}, dynamic_names={"x_seq"})
    assert short_storage == long_storage

    spec = NeuronSpec("lif")
    first = engine.generate_source("triton", spec, 4, 1, (), storage=short_storage)
    second = engine.generate_source("triton", spec, 4, 17, (), storage=long_storage)
    assert second == first


@pytest.mark.parametrize("backend", ("triton", "cupy"))
def test_runtime_length_preserves_forward_and_full_vjp(backend):
    _require_backend(backend)
    engine._compile_cached.cache_clear()
    engine._generate_source.cache_clear()
    for steps in (1, 7):
        spec = NeuronSpec("lif", dt=0.75, hard_reset=True)
        torch.manual_seed(492)
        x = (torch.rand(steps, 2, 4, device="cuda") * 0.5 + 0.1).requires_grad_()
        state = (torch.zeros(2, 4, device="cuda", requires_grad=True),)
        parameters = tuple(
            value.requires_grad_()
            for value in (
                torch.full((4,), 1.0, device="cuda"),
                torch.zeros(4, device="cuda"),
                torch.full((4,), 1.2, device="cuda"),
                torch.full((4,), 3.0, device="cuda"),
            )
        )
        actual = forward(backend, x, state, parameters, spec, ("v",))
        expected_states = state
        expected_outputs = []
        expected_records = []
        from btorch.models.kernels.torch_backend import forward_step

        for value in x.unbind(0):
            output, expected_states, records = forward_step(
                value, expected_states, parameters, spec
            )
            expected_outputs.append(output)
            expected_records.append(records["v"])
        expected = (
            torch.stack(expected_outputs),
            expected_states,
            (torch.stack(expected_records),),
        )
        for actual_value, expected_value in zip(
            (actual[0], *actual[1], *actual[2]),
            (expected[0], *expected[1], *expected[2]),
            strict=True,
        ):
            torch.testing.assert_close(
                actual_value, expected_value, atol=5e-5, rtol=5e-5
            )

        cotangents = tuple(
            torch.randn_like(value) for value in (actual[0], *actual[1], *actual[2])
        )
        expected_grads = torch.autograd.grad(
            (expected[0], *expected[1], *expected[2]),
            (x, *state, *parameters),
            cotangents,
            allow_unused=True,
        )
        actual_grads = backward(
            backend,
            x,
            state,
            parameters,
            spec,
            ("v",),
            actual[2],
            cotangents[0],
            cotangents[1:2],
            cotangents[2:3],
        )
        for actual_value, expected_value, reference in zip(
            (actual_grads[0], *actual_grads[1], *actual_grads[2]),
            expected_grads,
            (x, *state, *parameters),
            strict=True,
        ):
            if expected_value is None:
                expected_value = torch.zeros_like(reference)
            torch.testing.assert_close(
                actual_value, expected_value, atol=2e-4, rtol=2e-3
            )

    compile_info = engine._compile_cached.cache_info()
    assert compile_info.misses == 2
    assert compile_info.hits >= 2


def test_runtime_length_glif_checkpoint_forward_and_vjp_noncontiguous():
    _require_backend("triton")
    spec = NeuronSpec("glif3", dt=0.25, hard_reset=True)
    residual = ResidualTemplate(("v", "Iasc"), "Iasc", 16, True)
    # Keep Iasc private here so the test exercises compact checkpoint replay;
    # requesting its full record selects the separate full-history contract.
    history_names = ("v", "v_charged")
    for steps in (1, 17):
        torch.manual_seed(919)
        base = torch.rand(steps, 2, 8, device="cuda")
        base.mul_(1.5).add_(0.75)
        x = base[:, :, ::2].detach().requires_grad_()
        assert not x.is_contiguous()
        states = (
            torch.full((2, 4), 0.75, device="cuda", requires_grad=True),
            torch.full((2, 4, 2), 0.3, device="cuda", requires_grad=True),
        )
        parameters = tuple(
            value.requires_grad_()
            for value in (
                torch.full((4,), 1.0, device="cuda"),
                torch.zeros(4, device="cuda"),
                torch.full((4,), 1.2, device="cuda"),
                torch.full((4,), 3.0, device="cuda"),
                torch.full((4,), 0.1, device="cuda"),
                torch.full((4, 2), 0.2, device="cuda"),
                torch.full((4, 2), -0.1, device="cuda"),
            )
        )
        actual = forward(
            "triton",
            x,
            states,
            parameters,
            spec,
            history_names,
            residual_template=residual,
        )
        expected_states = states
        expected_outputs = []
        expected_records = {name: [] for name in (*history_names, "Iasc")}
        from btorch.models.kernels.torch_backend import forward_step

        for value in x.unbind(0):
            output, expected_states, records = forward_step(
                value, expected_states, parameters, spec
            )
            expected_outputs.append(output)
            for name in expected_records:
                expected_records[name].append(records[name])
        torch.testing.assert_close(
            actual[0], torch.stack(expected_outputs), atol=5e-5, rtol=5e-5
        )
        for actual_state, expected_state in zip(
            actual[1], expected_states, strict=True
        ):
            torch.testing.assert_close(
                actual_state, expected_state, atol=5e-5, rtol=5e-5
            )
        for index, name in enumerate(history_names):
            torch.testing.assert_close(
                actual[2][index],
                torch.stack(expected_records[name]),
                atol=5e-5,
                rtol=5e-5,
            )
        if steps > 16:
            assert expected_records["Iasc"][15].abs().max() > 0
            assert torch.stack(expected_outputs).any()
            torch.testing.assert_close(
                actual[2][-1][0],
                expected_records["Iasc"][15],
                atol=5e-5,
                rtol=5e-5,
            )
        else:
            assert not actual[2][-1].numel()

        grad_output = torch.randn_like(actual[0])
        grad_final_states = tuple(torch.randn_like(value) for value in actual[1])
        grad_histories = tuple(
            torch.randn_like(value) for value in actual[2][: len(history_names)]
        )
        expected = (
            torch.stack(expected_outputs),
            expected_states,
            tuple(torch.stack(expected_records[name]) for name in history_names),
        )
        expected_grads = torch.autograd.grad(
            (expected[0], *expected[1], *expected[2]),
            (x, *states, *parameters),
            (grad_output, *grad_final_states, *grad_histories),
            allow_unused=True,
        )
        actual_grads = backward(
            "triton",
            x,
            states,
            parameters,
            spec,
            history_names,
            actual[2][: len(history_names)],
            grad_output,
            grad_final_states,
            grad_histories,
            residual_template=residual,
            residual_output=actual[0],
            residuals=(actual[2][-1],),
        )
        for actual_value, expected_value, reference in zip(
            (actual_grads[0], *actual_grads[1], *actual_grads[2]),
            expected_grads,
            (x, *states, *parameters),
            strict=True,
        ):
            if expected_value is None:
                expected_value = torch.zeros_like(reference)
            torch.testing.assert_close(
                actual_value, expected_value, atol=5e-4, rtol=5e-3
            )
