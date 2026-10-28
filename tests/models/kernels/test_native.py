"""Exercise native whole-operator bindings without a second autograd
registry."""

import pytest
import torch

from btorch.models.kernels.native import (
    KernelBinding,
    load_kernel_package,
)
from btorch.models.kernels.testing import (
    ModelParityCase,
    ModelParityOptions,
    check_model_parity,
)


@torch.library.custom_op("btorch_test_native::inference", mutates_args=())
def _inference(
    x: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return x.cumsum(0) + state, x.sum(0) + state


@_inference.register_fake
def _simulation_fake(x, state):
    return torch.empty_like(x), torch.empty_like(state)


@torch.library.custom_op("btorch_test_native::training", mutates_args=())
def _training(
    x: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return x.cumsum(0) + state, x.sum(0) + state


@_training.register_fake
def _training_fake(x, state):
    return torch.empty_like(x), torch.empty_like(state)


def _setup_context(ctx, inputs, output):
    # This linear recurrence needs only the sequence length, no tensor tape.
    ctx.time_steps = inputs[0].shape[0]


def _backward(ctx, grad_seq, grad_final):
    return grad_seq.flip(0).cumsum(0).flip(0) + grad_final, grad_seq.sum(0) + grad_final


_training.register_autograd(_backward, setup_context=_setup_context)
_BINDING = KernelBinding(_inference, _training)


def _golden(x, state):
    # Ordinary Torch is the oracle, including final-state cotangents.
    return x.cumsum(0) + state, x.sum(0) + state


def test_native_parity_upstream_gradient_and_outstanding_calls():
    x = torch.randn(5, 2, requires_grad=True)
    state = torch.randn(2)  # A frozen RNN state must not cut the encoder path.
    report = check_model_parity(
        [ModelParityCase("recurrence", _golden, _BINDING, (x, state))]
    )
    report.assert_close()
    # Every call owns its backward context, even before any backward executes.
    first, final = _BINDING(x, state)
    second, next_final = _BINDING(x * 2, state)
    (first.sum() + final.sum() + second.sum() + next_final.sum()).backward()
    expected = 3 * (torch.arange(5, 0, -1) + 1).unsqueeze(1).expand_as(x)
    torch.testing.assert_close(x.grad, expected.float())


def test_inference_uses_independent_operator_and_eval_keeps_training():
    class Network(torch.nn.Module):
        def forward(self, x, state):
            return _BINDING(x, state)

    model = Network().eval()
    x = torch.randn(3, 2, requires_grad=True)
    state = torch.zeros(2)
    # eval() must not suppress the upstream encoder gradient.
    model(x, state)[0].sum().backward()
    assert x.grad is not None
    with torch.inference_mode():
        output, final = model(x, state)
        assert not output.requires_grad and not final.requires_grad
    with pytest.raises(RuntimeError, match="no training"):
        KernelBinding(_inference)(x, state)


def test_wrong_backward_is_reported_even_when_forward_matches():
    class WrongGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            return grad * 2

    x = torch.randn(4, requires_grad=True)
    report = check_model_parity(
        [ModelParityCase("wrong_vjp", lambda x: x.clone(), WrongGradient.apply, (x,))]
    )
    assert all(item.passed for item in report.comparisons if "/output/" in item.name)
    failures = [item for item in report.comparisons if not item.passed]
    assert failures[0].name == "wrong_vjp/training/backward/input/0"
    assert failures[0].first_mismatch == (0,)
    with pytest.raises(AssertionError, match="wrong_vjp/training/backward"):
        report.assert_close()


def test_aliases_noncontiguous_views_and_independent_storage():
    base = torch.arange(12.0).reshape(3, 4)
    x = base[:, ::2].requires_grad_()
    state = base[:, 1::2]

    def model(x, state, repeated):
        # An adapter may depend on shared storage and repeated parameter identity.
        assert x is repeated
        assert x.stride() == (4, 2)
        assert x.untyped_storage().data_ptr() == state.untyped_storage().data_ptr()
        return x + state + repeated

    check_model_parity(
        [ModelParityCase("views", model, model, (x, state, x))]
    ).assert_close()
    torch.testing.assert_close(base, torch.arange(12.0).reshape(3, 4))


def test_package_loading_and_fullgraph_native_execution(tmp_path):
    package = tmp_path / "user_kernel"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from btorch.models.kernels.native import KernelBinding\n"
        "def run(x):\n    return x + 1\n"
        "KERNEL_BINDINGS = {'whole_network': KernelBinding(run, run)}\n"
    )
    loaded = load_kernel_package(package)
    x = torch.randn(3)
    torch.testing.assert_close(loaded["whole_network"](x), x + 1)
    # Fake implementations provide the compile boundary; eager backend keeps
    # this registration/fullgraph test independent of GPU compiler availability.
    compiled = torch.compile(_BINDING, backend="eager", fullgraph=True)
    x = torch.randn(3, 2, requires_grad=True)
    actual = compiled(x, torch.zeros(2))
    expected = _golden(x, torch.zeros(2))
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
    actual[0].sum().backward()
    torch.testing.assert_close(
        x.grad, torch.arange(3, 0, -1).float()[:, None].expand_as(x)
    )


def test_native_registration_and_trainable_initial_state():
    # Torch owns registration validation; the harness separately compares
    # mathematical results and gradients of both drive and initial state.
    x = torch.randn(3, 2, requires_grad=True)
    state = torch.randn(2, requires_grad=True)
    torch.library.opcheck(_training, (x, state))
    check_model_parity(
        [ModelParityCase("initial_state", _golden, _BINDING, (x, state))]
    ).assert_close()


def test_alternative_gradient_oracle_and_missing_gradient():
    class AlternativeGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            return grad * 2

    def oracle(inputs, outputs, cotangents):
        # A different gradient definition must be explicit, rather than
        # silently comparing it with ordinary autograd for the forward model.
        return (cotangents[0] * 2,)

    x = torch.randn(4, requires_grad=True)
    check_model_parity(
        [
            ModelParityCase(
                "alternative",
                lambda x: x.clone(),
                AlternativeGradient.apply,
                (x,),
                gradient_oracle=oracle,
            )
        ]
    ).assert_close()
    missing = check_model_parity(
        [ModelParityCase("detached", lambda x: x.clone(), lambda x: x.detach(), (x,))]
    )
    assert not missing.passed
    assert any(
        "candidate grad=False" in item.detail
        for item in missing.comparisons
        if "/training/backward/" in item.name
    )


def test_expanded_zero_stride_input():
    # Expanded drive is valid Torch input; cloning must not compact it or try
    # writing directly into an overlapping view.
    x = torch.randn(1, 2).expand(4, 2).detach().requires_grad_()

    def run(x):
        assert x.stride()[0] == 0
        return x.sin()

    check_model_parity([ModelParityCase("expanded", run, run, (x,))]).assert_close()


@torch.library.custom_op("btorch_test_native::nonlinear_inference", mutates_args=())
def _nonlinear_inference(x: torch.Tensor) -> torch.Tensor:
    return x.square()


@_nonlinear_inference.register_fake
def _nonlinear_inference_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("btorch_test_native::nonlinear_training", mutates_args=())
def _nonlinear_training(x: torch.Tensor) -> torch.Tensor:
    return x.square()


@_nonlinear_training.register_fake
def _nonlinear_training_fake(x):
    return torch.empty_like(x)


def _nonlinear_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _nonlinear_backward(ctx, grad):
    (x,) = ctx.saved_tensors
    return grad * 2 * x


_nonlinear_training.register_autograd(
    _nonlinear_backward, setup_context=_nonlinear_setup
)


def test_inference_does_not_save_training_tape():
    # Unlike the linear example, this backward explicitly needs a tensor tape.
    # Inspect Torch's saved tensors rather than equating no gradient on an
    # output with proof that the forward avoided backward-specific storage.
    binding = KernelBinding(_nonlinear_inference, _nonlinear_training)
    x = torch.randn(4, requires_grad=True)
    saved = []

    def pack(value):
        saved.append(value)
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = binding(x)
        assert len(saved) == 1
        assert saved[0] is x
        output.sum().backward()
    torch.testing.assert_close(x.grad, x.detach() * 2)
    for context in (torch.no_grad, torch.inference_mode):
        saved.clear()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            with context():
                inference = binding(x)
        assert saved == []
        assert not inference.requires_grad
        torch.testing.assert_close(inference, output.detach())


def test_external_package_imports_in_fresh_process(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Relative imports are common for an OOT package with separate model and
    # kernel files. A new interpreter proves no registration or sys.modules
    # state from another test is necessary to load the generated source.
    package = tmp_path / "durable_kernel"
    package.mkdir()
    (package / "model.py").write_text("def run(x):\n    return x.square()\n")
    (package / "__init__.py").write_text(
        "from .model import run\n"
        "from btorch.models.kernels import KernelBinding\n"
        "KERNEL_BINDINGS = {'model': KernelBinding(run, run)}\n"
    )
    code = (
        "import sys, torch\n"
        "from btorch.models.kernels import load_kernel_package\n"
        "binding = load_kernel_package(sys.argv[1])['model']\n"
        "x = torch.tensor([2., 3.], requires_grad=True)\n"
        "out = binding(x)\n"
        "torch.testing.assert_close(out, torch.tensor([4., 9.]))\n"
        "out.sum().backward()\n"
        "torch.testing.assert_close(x.grad, torch.tensor([4., 6.]))\n"
        "with torch.inference_mode():\n"
        "    torch.testing.assert_close(binding(x), out.detach())\n"
    )
    env = dict(os.environ)
    checkout = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join([str(checkout), env.get("PYTHONPATH", "")])
    subprocess.run(
        [sys.executable, "-c", code, str(package)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_unused_parameter_none_and_materialized_zero_vjps():
    x = torch.randn(4, requires_grad=True)
    unused = torch.randn(4, requires_grad=True)

    def golden(x, unused):
        return x.square()

    def materialized(x, unused):
        # Real backward implementations often fill a parameter gradient buffer
        # with zeros rather than returning Torch's unused-gradient sentinel.
        return x.square() + unused * 0

    check_model_parity(
        [
            ModelParityCase("unused", golden, materialized, (x, unused)),
            ModelParityCase("zero_reference", materialized, golden, (x, unused)),
        ]
    ).assert_close()

    class WrongUnusedGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, unused):
            ctx.save_for_backward(x)
            return x.square()

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * 2 * x, grad

    # Forward agrees exactly, but a spurious nonzero gradient for an unused
    # parameter must fail, with useful coordinates rather than a None mismatch.
    bad = check_model_parity(
        [
            ModelParityCase(
                "wrong_unused", golden, WrongUnusedGradient.apply, (x, unused)
            )
        ]
    )
    failures = [item for item in bad.comparisons if not item.passed]
    assert len(failures) == 1
    assert failures[0].name == "wrong_unused/training/backward/input/1"
    assert failures[0].first_mismatch == (0,)
    assert failures[0].max_abs_error > 0


def test_parity_rejects_gradient_permutation_hidden_by_periodic_probe():
    class WrongPermutation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            result = grad.clone()
            result[0], result[13] = grad[13], grad[0]
            return result

    # The previous modulo-13 cotangent gave equal coefficients to these two
    # indices, letting a wrong Jacobian pass even though independent losses
    # have different gradients. The nonperiodic probe exposes the permutation.
    x = torch.arange(14.0).requires_grad_()
    rng = torch.get_rng_state().clone()
    report = check_model_parity(
        [
            ModelParityCase(
                "wrong_permutation", lambda x: x.clone(), WrongPermutation.apply, (x,)
            )
        ]
    )
    torch.testing.assert_close(torch.get_rng_state(), rng)
    failures = [item for item in report.comparisons if not item.passed]
    assert len(failures) == 1
    assert failures[0].first_mismatch == (0,)
    assert "/backward/input/0" in failures[0].name


def test_parity_rejects_integer_routing_off_by_one():
    # Relative floating tolerances must never hide an incorrect integer route,
    # event index or quantized storage value merely because its magnitude grows.
    route = torch.tensor([131072], dtype=torch.int64)
    report = check_model_parity(
        [
            ModelParityCase(
                "route", lambda route: route.clone(), lambda route: route + 1, (route,)
            )
        ]
    )
    assert not report.passed
    assert all(item.first_mismatch == (0,) for item in report.comparisons)


def test_parity_preserves_fp64_sub_float32_gradient_errors():
    class SlightlyWrongGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            # This offset is below float32 resolution near one, but remains a
            # real defect in an FP64 VJP and must not disappear in comparison.
            return grad + 5e-8

    x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    report = check_model_parity(
        [
            ModelParityCase(
                "fp64_vjp",
                lambda value: value.clone(),
                SlightlyWrongGradient.apply,
                (x,),
            )
        ],
        options=ModelParityOptions(atol=1e-9, rtol=0.0, modes=("training",)),
    )
    assert not report.passed
    assert any(
        "fp64_vjp/training/backward/input/0" == item.name for item in report.comparisons
    )


def test_parity_reports_complex_imaginary_mismatch():
    x = torch.tensor([1.0 + 2.0j], dtype=torch.complex128)
    report = check_model_parity(
        [
            ModelParityCase(
                "complex",
                lambda value: value,
                lambda value: value + 1e-3j,
                (x,),
            )
        ],
        options=ModelParityOptions(check_gradients=False, modes=("inference",)),
    )
    assert not report.passed
    assert report.comparisons[0].max_abs_error == pytest.approx(1e-3)


def test_parity_names_first_spike_divergence():
    expected = torch.tensor([[0, 0], [0, 1], [0, 0]], dtype=torch.int64)
    actual = expected.clone()
    actual[2, 1] = 1
    report = check_model_parity(
        [
            ModelParityCase(
                "spikes",
                lambda: {"spikes": expected},
                lambda: {"spikes": actual},
                (),
                exact_outputs=("['spikes']",),
            )
        ],
        options=ModelParityOptions(check_gradients=False, modes=("inference",)),
    )
    assert not report.passed
    assert report.comparisons[0].first_spike_divergence == 2
    assert "['spikes']" in report.comparisons[0].name


def test_parity_preserves_mixed_dtype_alias_views():
    storage = torch.arange(12.0, dtype=torch.float32).reshape(3, 4)
    x = storage[:, ::2]
    state = torch.tensor([0.5, 1.5], dtype=torch.float64)

    def model(value, initial, repeated):
        assert value is repeated
        assert value.stride() == (4, 2)
        return {"value": value.to(torch.float64) + initial}

    report = check_model_parity(
        [
            ModelParityCase(
                "mixed_alias",
                model,
                model,
                (x, state, x),
                input_names=("x", "state", "x_alias"),
            )
        ],
        options=ModelParityOptions(check_gradients=False, modes=("inference",)),
    )
    report.assert_close()


def test_comparison_rejects_wrong_device_before_copying():
    from btorch.models.kernels.testing import _compare

    # Meta has no data to copy. Device rejection must therefore precede the
    # CPU copy; this reproduces the ABI defect without requiring a GPU.
    report = _compare(
        "wrong_device", torch.empty(2, device="meta"), torch.zeros(2), 1e-5, 1e-5
    )
    assert not report.passed
    assert "meta" in report.detail and "cpu" in report.detail


def test_inference_only_parity_with_grad_requiring_encoder_inputs():
    calls = []

    def inference(x):
        # An inference-only kernel can accept tensors produced by a trainable
        # encoder when executed under no_grad/inference_mode. Verification must
        # not invent a training requirement for this declared task.
        assert not torch.is_grad_enabled()
        calls.append("inference")
        return x.square()

    x = torch.randn(4, requires_grad=True)
    report = check_model_parity(
        [
            ModelParityCase(
                "inference_only", lambda x: x.square(), KernelBinding(inference), (x,)
            )
        ],
        options=ModelParityOptions(modes=("inference", "no_grad")),
    )
    report.assert_close()
    assert calls == ["inference", "inference"]
    assert not any("/training/" in item.name for item in report.comparisons)
    # Original gradient intent remains intact; an explicit training invocation
    # is still rejected rather than silently routed to an unregistered VJP.
    assert x.requires_grad
    with pytest.raises(RuntimeError, match="no training"):
        KernelBinding(inference)(x)


def test_frozen_training_context_does_not_claim_training_kernel_evidence():
    x = torch.randn(4)
    report = check_model_parity(
        [
            ModelParityCase(
                "frozen",
                lambda x: x.square(),
                KernelBinding(lambda x: x.square()),
                (x,),
            )
        ]
    )
    report.assert_close()
    assert any(
        "not evidence of training-kernel execution" in item for item in report.skipped
    )
