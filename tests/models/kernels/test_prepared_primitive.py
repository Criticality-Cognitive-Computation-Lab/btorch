"""Focused contracts for reusable primitive preparation.

These tests exercise the tensor-free caches directly through public
sequence execution.  They intentionally cover distinct invalidation and
autograd ownership cases instead of multiplying models, renderers and
dtypes.
"""

import gc
import weakref

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


def test_outstanding_primitive_forwards_keep_independent_full_vjps():
    """Two live forwards retain independent full VJPs and golden parity."""
    spec = NeuronSpec("lif", dt=0.25, hard_reset=True)
    state = torch.zeros(2, 3, requires_grad=True)
    parameters = tuple(
        torch.tensor(value, requires_grad=True) for value in (0.8, -0.1, 1.2, 3.0)
    )
    x_first = torch.randn(4, 2, 3, requires_grad=True)
    x_second = torch.randn(3, 2, 3, requires_grad=True)
    history = HistoryPlan(requested=("v",))

    first = run_multi_step(
        x_first, (state,), parameters, spec, history, backend="torch"
    )
    second = run_multi_step(
        x_second, (state,), parameters, spec, history, backend="torch"
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

    with torch.inference_mode():
        first_result = run_multi_step(first, state, parameters, spec, backend=backend)
        warmed = engine._kernel_plan.cache_info()
        second_result = run_multi_step(second, state, parameters, spec, backend=backend)
    repeated = engine._kernel_plan.cache_info()
    first_expected = python_module_sequence(first, state, parameters, spec)
    second_expected = python_module_sequence(second, state, parameters, spec)
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
