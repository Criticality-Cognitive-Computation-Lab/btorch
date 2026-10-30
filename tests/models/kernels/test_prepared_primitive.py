"""Focused contracts for reusable primitive preparation.

These tests exercise the tensor-free caches directly through public
sequence execution.  They intentionally cover distinct invalidation and
autograd ownership cases instead of multiplying models, renderers and
dtypes.
"""

import gc
import weakref
from dataclasses import replace

import pytest
import torch

from btorch.models import environ
from btorch.models.kernels import (
    HistoryPlan,
    NeuronSpec,
    adapters,
    operators,
    run_multi_step,
)
from btorch.models.kernels.adapters import prepare_cell
from btorch.models.kernels.backends import engine
from btorch.models.kernels.golden import python_module_sequence
from btorch.models.neurons.lif import LIF
from tests.models.kernels.helpers import require_cuda_backend


def _clear_metadata_caches():
    adapters._cached_model_layout.cache_clear()
    adapters._cell_metadata.cache_clear()
    adapters._cell_records.cache_clear()


def _lif_module():
    module = LIF(
        3,
        v_threshold=0.8,
        v_reset=-0.1,
        c_m=1.2,
        tau=3.0,
        hard_reset=True,
        pre_spike_v=True,
    )
    module.init_state(batch_size=2, dtype=torch.float32)
    return module


def _flat_sequence(result):
    output, final, records = result
    return (output, *final, *records)


def test_prepare_cell_reuses_tensor_free_model_metadata():
    """Repeated preparation reuses metadata while reading current values."""
    _clear_metadata_caches()
    module = _lif_module()
    x = torch.randn(5, 2, 3)
    history = HistoryPlan(requested=("v",))

    with environ.context(dt=0.25):
        first = prepare_cell(module, x)
        first_result = run_multi_step(
            x, first[2], first[3], first[0], history, backend="torch"
        )
        first_expected = python_module_sequence(
            x, first[2], first[3], first[0], history.requested
        )
        first_info = (
            adapters._cached_model_layout.cache_info(),
            adapters._cell_metadata.cache_info(),
        )
        with torch.no_grad():
            module.c_m.add_(0.25)
        second = prepare_cell(module, x)
        second_result = run_multi_step(
            x, second[2], second[3], second[0], history, backend="torch"
        )
        second_expected = python_module_sequence(
            x, second[2], second[3], second[0], history.requested
        )
        second_info = (
            adapters._cached_model_layout.cache_info(),
            adapters._cell_metadata.cache_info(),
        )
    with environ.context(dt=0.5):
        third = prepare_cell(module, x)
        third_info = (
            adapters._cached_model_layout.cache_info(),
            adapters._cell_metadata.cache_info(),
        )

    assert first[0].model == second[0].model == "lif"
    assert first_info[0].misses == second_info[0].misses == 1
    assert first_info[1].misses == second_info[1].misses == 1
    assert second_info[0].hits >= 1 and second_info[1].hits >= 1
    assert third_info[0].misses == third_info[1].misses == 1
    for actual, expected in zip(
        first_result, _flat_sequence(first_expected), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(
        second_result, _flat_sequence(second_expected), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(second[3][2], module.c_m.float())
    assert not torch.equal(first_expected[1][0], second_expected[1][0])
    assert third[0].dt.item() == 0.5


def test_validation_cache_rechecks_changed_layout_and_reuses_it_afterward():
    """A changed noncontiguous layout invalidates once, then stays reusable."""
    operators._validated_sequence_contract.cache_clear()
    spec = NeuronSpec("lif", dt=0.25, hard_reset=True)
    state = (torch.zeros(2, 3),)
    parameters = tuple(torch.tensor(value) for value in (0.8, -0.1, 1.2, 3.0))
    contiguous = torch.randn(4, 2, 3)
    first_result = run_multi_step(contiguous, state, parameters, spec, backend="torch")
    first = operators._validated_sequence_contract.cache_info()

    storage = torch.empty(4, 2, 6)
    strided = storage[..., ::2]
    strided.copy_(contiguous)
    strided_result = run_multi_step(strided, state, parameters, spec, backend="torch")
    changed = operators._validated_sequence_contract.cache_info()
    run_multi_step(strided, state, parameters, spec, backend="torch")
    repeated = operators._validated_sequence_contract.cache_info()

    assert changed.misses == first.misses + 1
    assert repeated.misses == changed.misses
    assert repeated.hits >= changed.hits + 1
    for actual, expected in zip(strided_result, first_result, strict=True):
        torch.testing.assert_close(actual, expected)


def test_cached_validation_does_not_accept_an_invalid_replacement():
    """A prior valid contract must not suppress later shape validation."""
    operators._validated_sequence_contract.cache_clear()
    spec = NeuronSpec("lif", dt=0.25)
    parameters = tuple(torch.tensor(value) for value in (0.8, 0.0, 1.0, 3.0))
    x = torch.randn(3, 2, 3)
    run_multi_step(x, (torch.zeros(2, 3),), parameters, spec, backend="torch")

    with pytest.raises(ValueError, match="Invalid initial v shape"):
        run_multi_step(
            x,
            (torch.zeros(2, 4),),
            parameters,
            spec,
            backend="torch",
        )


@pytest.mark.parametrize("backend", ("torch", "triton"))
def test_outstanding_primitive_forwards_keep_independent_full_vjps(backend):
    """Two live forwards retain independent full VJPs and golden parity."""
    if backend != "torch":
        require_cuda_backend(backend)
    device = "cpu" if backend == "torch" else "cuda"
    spec = NeuronSpec("lif", dt=0.25, hard_reset=True)
    state = torch.zeros(2, 3, device=device, requires_grad=True)
    parameters = tuple(
        torch.tensor(value, device=device, requires_grad=True)
        for value in (0.8, -0.1, 1.2, 3.0)
    )
    x_first = torch.randn(4, 2, 3, device=device, requires_grad=True)
    x_second = torch.randn(3, 2, 3, device=device, requires_grad=True)
    history = HistoryPlan(requested=("v",))

    first = run_multi_step(
        x_first, (state,), parameters, spec, history, backend=backend
    )
    second = run_multi_step(
        x_second, (state,), parameters, spec, history, backend=backend
    )
    first_expected = python_module_sequence(
        x_first, (state,), parameters, spec, history.requested
    )
    second_expected = python_module_sequence(
        x_second, (state,), parameters, spec, history.requested
    )
    for actual, expected in zip(first, _flat_sequence(first_expected), strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(second, _flat_sequence(second_expected), strict=True):
        torch.testing.assert_close(actual, expected)

    first_expected_flat = _flat_sequence(first_expected)
    second_expected_flat = _flat_sequence(second_expected)
    actual_loss = sum(value.square().sum() for value in (*first, *second))
    expected_loss = sum(
        value.square().sum() for value in (*first_expected_flat, *second_expected_flat)
    )
    roots = (x_first, x_second, state, *parameters)
    actual_gradients = torch.autograd.grad(actual_loss, roots, allow_unused=True)
    expected_gradients = torch.autograd.grad(
        expected_loss,
        roots,
        allow_unused=True,
    )
    for actual, expected, root in zip(
        actual_gradients, expected_gradients, roots, strict=True
    ):
        if expected is None:
            expected = torch.zeros_like(root)
        if actual is None:
            actual = torch.zeros_like(root)
        torch.testing.assert_close(actual, expected)

    assert actual_gradients[0].shape == x_first.shape
    assert actual_gradients[1].shape == x_second.shape


@pytest.mark.parametrize("backend", ("triton", "cupy", "tilelang"))
def test_prepared_native_plan_reuses_source_for_new_runtime_values(
    backend, monkeypatch
):
    """Each renderer reuses a warmed plan for new values and outputs."""
    require_cuda_backend(backend)
    engine._primitive_plan.cache_clear()
    engine._kernel_plan.cache_clear()
    engine._generate_source.cache_clear()
    engine._compile_cached.cache_clear()
    generated = 0
    original_generate = engine.generate_source

    def counted_generate(*args, **kwargs):
        nonlocal generated
        generated += 1
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(engine, "generate_source", counted_generate)
    spec = NeuronSpec("lif", dt=0.25, hard_reset=True)
    parameters = tuple(
        torch.tensor(value, device="cuda") for value in (0.8, 0.0, 1.2, 3.0)
    )
    state = (torch.zeros(4, device="cuda"),)
    first = torch.randn(5, 4, device="cuda")
    second = first + 0.25
    second_spec = replace(spec, dt=0.5)

    with torch.inference_mode():
        first_result = run_multi_step(first, state, parameters, spec, backend=backend)
        warmed = engine._primitive_plan.cache_info()
        second_result = run_multi_step(
            second, state, parameters, second_spec, backend=backend
        )
    repeated = engine._primitive_plan.cache_info()
    first_expected = python_module_sequence(first, state, parameters, spec)
    second_expected = python_module_sequence(second, state, parameters, second_spec)
    for actual, expected in zip(
        first_result, _flat_sequence(first_expected), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
    for actual, expected in zip(
        second_result, _flat_sequence(second_expected), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)

    assert generated == 1
    assert repeated.misses == warmed.misses == 1
    assert repeated.hits >= warmed.hits + 1

    # A warmed executable must not own invocation tensors after the caller
    # releases them, even while its outputs and golden comparison stay live.
    references = tuple(
        weakref.ref(value) for value in (second, state[0], parameters[0])
    )
    del second, state, parameters
    gc.collect()
    assert all(reference() is None for reference in references)


def test_native_empty_sequences_preserve_continuation_without_launch(monkeypatch):
    """Empty time preserves initial state; empty neuron axes launch no work."""
    require_cuda_backend("triton")
    spec = NeuronSpec("lif", dt=0.25)
    parameters = tuple(
        torch.tensor(value, device="cuda") for value in (0.8, 0.0, 1.2, 3.0)
    )

    def unexpected_launch(*args, **kwargs):
        pytest.fail("An empty primitive must not launch a CUDA kernel")

    monkeypatch.setattr(engine, "_launch", unexpected_launch)
    for steps, neurons in ((0, 4), (3, 0)):
        state = torch.full((neurons,), 0.3, device="cuda")
        output, final, histories = engine.forward(
            "triton",
            torch.empty(steps, neurons, device="cuda"),
            (state,),
            parameters,
            spec,
            ("v",),
        )
        torch.testing.assert_close(final[0], state)
        assert output.shape == histories[0].shape == (steps, neurons)
        assert output.numel() == histories[0].numel() == 0
        # Continuation storage is independently owned, including T=0.
        if neurons:
            assert final[0].data_ptr() != state.data_ptr()


def test_gradient_producer_rebinds_shared_storage_without_changing_results():
    """Disjoint views in one storage get a distinct, safe producer plan."""
    require_cuda_backend("tilelang")
    from btorch.sparse.runtime.activity import GradientList

    engine._primitive_plan.cache_clear()
    steps, neurons = 2, 4
    tape_width = GradientList.size(neurons)
    shared = torch.zeros(
        steps * (neurons + tape_width), device="cuda", dtype=torch.int32
    )
    x = shared[: steps * neurons].view(torch.float32).reshape(steps, neurons)
    x.copy_(torch.tensor([[0.1, 0.4, 0.8, 1.2]], device="cuda"))
    shared_tape = shared[steps * neurons :].reshape(steps, tape_width)
    separate_tape = torch.zeros_like(shared_tape)
    state = (torch.zeros(neurons, device="cuda"),)
    parameters = tuple(
        torch.tensor(value, device="cuda") for value in (0.8, 0.0, 1.2, 3.0)
    )
    spec = NeuronSpec("lif", dt=0.25, surrogate_width=1.0)
    first = engine.forward(
        "tilelang",
        x.clone(),
        state,
        parameters,
        spec,
        ("v",),
        gradient_activity=separate_tape,
    )
    warmed = engine._primitive_plan.cache_info()
    second = engine.forward(
        "tilelang",
        x,
        state,
        parameters,
        spec,
        ("v",),
        gradient_activity=shared_tape,
    )
    assert engine._primitive_plan.cache_info().misses == warmed.misses + 1
    expected = python_module_sequence(x, state, parameters, spec, ("v",))
    for actual in (first, second):
        for value, reference in zip(
            _flat_sequence(actual), _flat_sequence(expected), strict=True
        ):
            torch.testing.assert_close(value, reference, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(shared_tape, separate_tape, rtol=0, atol=0)


def test_sequence_operator_functional_and_autograd_contracts():
    """Check functional storage, fake shapes and compiled VJPs at the
    boundary."""
    # Without CustomOpDef's eager alias scan, the functional schema is still
    # mandatory. opcheck independently checks writes/aliases, fake metadata
    # and autograd registration, including AOT execution with dynamic shapes.
    # Numerical full-root VJPs are covered by the outstanding-forwards case.
    x = torch.linspace(0.1, 1.2, 24).reshape(4, 2, 3).requires_grad_()
    state = torch.zeros(2, 3, requires_grad=True)
    parameters = [
        torch.tensor(value, requires_grad=True) for value in (0.8, -0.1, 1.2, 3.0)
    ]
    encoded = operators.encode_spec(NeuronSpec("lif"))
    dt = torch.tensor(0.25)
    torch.library.opcheck(
        operators._training, (x, [state], parameters, encoded, "v", "torch", dt)
    )
    torch.library.opcheck(
        operators._inference,
        (
            x.detach(),
            [state.detach()],
            [value.detach() for value in parameters],
            encoded,
            "v",
            "torch",
            dt,
        ),
    )


@pytest.mark.parametrize("backend", ("triton", "cupy", "tilelang"))
def test_prepared_reverse_rebinds_tapes_and_strided_cotangents(backend, monkeypatch):
    """Reuse reverse preparation without owning tensors or losing full VJPs."""
    require_cuda_backend(backend)
    engine._primitive_plan.cache_clear()
    original_launch = engine._launch
    reverse_preparations = 0

    def count_preparation(*args, **kwargs):
        nonlocal reverse_preparations
        reverse_preparations += bool(kwargs.get("backward"))
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(engine, "_launch", count_preparation)
    references = []
    for index in range(4 if backend == "triton" else 3):
        # New storage and dt must bind to the same executable. Later passes
        # change cotangent or saved-history strides and need new reverse plans.
        spec = NeuronSpec("lif", dt=0.25 + index * 0.1, hard_reset=True)
        x = torch.linspace(0.05, 0.9, 24, device="cuda").reshape(4, 2, 3)
        x = (x + index * 0.03).requires_grad_()
        state = torch.full((2, 3), 0.1 * index, device="cuda", requires_grad=True)
        parameters = tuple(
            torch.tensor(value, device="cuda", requires_grad=True)
            for value in (0.8, -0.1, 1.2, 3.0)
        )
        restored_histories = []

        def pack(value, source=x):
            return value, value is not source and value.shape == source.shape

        def unpack(packed):
            value, is_history = packed
            if index == 3 and is_history:
                restored = torch.empty((*value.shape, 2), device=value.device)[..., 0]
                restored.copy_(value)
                restored_histories.append(tuple(restored.stride()))
                return restored
            return value

        # One renderer also exercises real saved-tensor restoration: offload
        # hooks may restore a history in new storage with different strides.
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            actual = run_multi_step(
                x,
                (state,),
                parameters,
                spec,
                HistoryPlan(requested=("v",)),
                backend=backend,
            )
        expected = _flat_sequence(
            python_module_sequence(x, (state,), parameters, spec, ("v",))
        )
        cotangents = tuple(
            torch.ones((*value.shape, 2), device="cuda")[..., 0]
            if index == 2
            else torch.ones_like(value)
            for value in actual
        )
        roots = (x, state, *parameters)
        gradients = torch.autograd.grad(actual, roots, cotangents)
        golden = torch.autograd.grad(expected, roots, cotangents, allow_unused=True)
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, atol=5e-5, rtol=5e-5)
        for value, reference, root in zip(gradients, golden, roots, strict=True):
            torch.testing.assert_close(
                value,
                torch.zeros_like(root) if reference is None else reference,
                atol=5e-5,
                rtol=5e-5,
            )
        assert reverse_preparations == (1 if index < 2 else index)
        if index == 3:
            assert restored_histories
        references.extend(
            weakref.ref(value) for value in (*roots, *actual, *cotangents)
        )
        del roots, x, state, parameters, actual, expected, cotangents, gradients, golden
        del pack, unpack
    # Plans may retain executables and layouts, never a saved autograd tape.
    del value, reference, root
    gc.collect()
    assert all(reference() is None for reference in references)
