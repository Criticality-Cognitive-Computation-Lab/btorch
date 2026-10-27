"""Validate direct GPU reads from broadcast parameter storage."""

import pytest
import torch

from btorch.models.kernels.backends.engine import _prepare, backward, forward
from btorch.models.kernels.contracts import NeuronSpec
from btorch.models.kernels.golden import python_module_sequence
from btorch.models.kernels.templates import get_template
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


def _leaf_view(value):
    return value.detach().requires_grad_()


def _assert_forward_and_backward(backend, x_seq, states, parameters, spec):
    template = get_template(spec)
    history_names = template.history_names
    expected = python_module_sequence(x_seq, states, parameters, spec, history_names)
    actual = forward(backend, x_seq, states, parameters, spec, history_names)
    expected_flat = (expected[0], *expected[1], *expected[2])
    actual_flat = (actual[0], *actual[1], *actual[2])
    for result, reference in zip(actual_flat, expected_flat):
        torch.testing.assert_close(result, reference, atol=5e-6, rtol=5e-5)

    # Every public trajectory contributes to the loss. This makes a wrong
    # broadcast address visible in state, input, and parameter VJPs rather than
    # allowing a coincidentally silent spike sequence to pass.
    torch.manual_seed(902)
    cotangents = tuple(torch.randn_like(value) for value in expected_flat)
    reference_gradients = torch.autograd.grad(
        expected_flat,
        (x_seq, *states, *parameters),
        cotangents,
        allow_unused=True,
    )
    n_states = len(states)
    actual_gradients = backward(
        backend,
        x_seq,
        states,
        parameters,
        spec,
        history_names,
        actual[2],
        cotangents[0],
        cotangents[1 : 1 + n_states],
        cotangents[1 + n_states :],
    )
    for result, reference, value in zip(
        (actual_gradients[0], *actual_gradients[1], *actual_gradients[2]),
        reference_gradients,
        (x_seq, *states, *parameters),
    ):
        if reference is None:
            reference = torch.zeros_like(value)
        torch.testing.assert_close(result, reference, atol=5e-5, rtol=5e-4)
    return actual


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_lif_reads_original_scalar_strided_offset_and_broadcast_storage(backend):
    require_cuda_backend(backend)
    torch.manual_seed(74)
    batch, neurons = 2, 5
    spec = NeuronSpec("lif", dt=0.25, hard_reset=True)
    x_seq = (torch.rand(9, batch, neurons, device="cuda") * 0.7).requires_grad_()
    states = ((torch.rand(batch, neurons, device="cuda") * 0.2).requires_grad_(),)

    scalar = torch.tensor(0.65, device="cuda")
    contiguous = torch.linspace(-0.15, 0.05, neurons, device="cuda")
    strided = torch.linspace(0.9, 1.5, 2 * neurons + 1, device="cuda")[1::2]
    batched_offset = torch.tensor(
        [[99.0, 2.5, 99.0], [99.0, 4.0, 99.0]], device="cuda"
    )[:, 1:2]
    parameters = tuple(
        _leaf_view(value) for value in (scalar, contiguous, strided, batched_offset)
    )

    # Preserved preparation must hand the launch path the same views. The
    # strided and batch-broadcast tensors occupy less storage than the ten
    # logical lanes, proving this path did not create expanded replicas.
    prepared = _prepare(x_seq, states, parameters, spec, preserve_parameters=True)
    buffers, layouts = prepared[-2:]
    template = get_template(spec)
    for name, parameter, layout in zip(template.parameter_names, parameters, layouts):
        assert buffers[name] is parameter
        assert buffers[name].data_ptr() == parameter.data_ptr()
        assert layout.shape == tuple(parameter.shape)
        assert layout.stride == tuple(parameter.stride())
    assert buffers["c_m"].stride() == (2,)
    assert buffers["tau"].storage_offset() == batched_offset.storage_offset()
    assert sum(layout.span for layout in layouts) < batch * neurons * len(layouts)

    # Compile and execute the dense layout first with the same values. The
    # following strided call must specialize independently instead of reusing
    # dense address arithmetic from the source cache.
    dense_parameters = (
        parameters[0],
        parameters[1],
        _leaf_view(parameters[2].clone()),
        parameters[3],
    )
    _assert_forward_and_backward(backend, x_seq, states, dense_parameters, spec)
    first_result = _assert_forward_and_backward(
        backend, x_seq, states, parameters, spec
    )
    with torch.no_grad():
        parameters[0].add_(0.08)
        parameters[2].mul_(1.1)
        parameters[3].add_(0.4)
    second_result = _assert_forward_and_backward(
        backend, x_seq, states, parameters, spec
    )
    # The update changes membrane dynamics continuously even if the binary
    # spike sequence happens to stay unchanged.
    assert not torch.equal(first_result[1][0], second_result[1][0])
    assert not torch.equal(first_result[2][0], second_result[2][0])


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_glif_mode_parameters_broadcast_with_original_strides(backend):
    require_cuda_backend(backend)
    torch.manual_seed(75)
    batch, neurons, modes = 2, 4, 3
    spec = NeuronSpec("glif3", dt=0.5, hard_reset=True)
    x_seq = (torch.rand(7, batch, neurons, device="cuda") * 0.4).requires_grad_()
    states = (
        (torch.rand(batch, neurons, device="cuda") * 0.1).requires_grad_(),
        (torch.rand(batch, neurons, modes, device="cuda") * 0.05).requires_grad_(),
    )
    mode_storage = torch.tensor(
        [99.0, 0.08, 99.0, 0.12, 99.0, 0.2, 99.0], device="cuda"
    )
    mode_only = mode_storage[1::2]
    amp_storage = torch.empty(1, neurons, modes * 2 + 1, device="cuda")
    amp_storage.fill_(99.0)
    amp_storage[..., 1::2] = torch.linspace(
        -0.08, -0.02, neurons * modes, device="cuda"
    ).reshape(1, neurons, modes)
    batched_neuron_modes = amp_storage[..., 1::2]
    parameters = tuple(
        _leaf_view(value)
        for value in (
            torch.tensor(0.7, device="cuda"),
            torch.tensor(-0.1, device="cuda"),
            torch.linspace(0.9, 1.2, neurons, device="cuda"),
            torch.tensor([[3.0], [4.5]], device="cuda"),
            torch.tensor(-0.2, device="cuda"),
            mode_only,
            batched_neuron_modes,
        )
    )

    # k broadcasts only over the GLIF mode axis, while asc_amps has a leading
    # singleton batch and nonunit mode stride. Their independent layouts catch
    # both fixed-mode addressing and ordinary batch/neuron decomposition.
    assert parameters[5].shape == (modes,) and parameters[5].stride() == (2,)
    assert parameters[6].shape == (1, neurons, modes)
    assert parameters[6].stride(-1) == 2
    _assert_forward_and_backward(backend, x_seq, states, parameters, spec)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_lazy_negative_and_shared_low_precision_parameters(backend):
    require_cuda_backend(backend)
    torch.manual_seed(76)
    batch, neurons = 2, 4
    spec = NeuronSpec("lif", dt=0.2, hard_reset=True)
    template = get_template(spec)
    history_names = template.history_names
    x_seq = (torch.rand(8, batch, neurons, device="cuda") * 0.6).requires_grad_()
    states = ((torch.rand(batch, neurons, device="cuda") * 0.1).requires_grad_(),)
    threshold = torch.tensor(0.75, device="cuda", requires_grad=True)
    positive_reset = torch.linspace(0.05, 0.2, neurons, device="cuda")
    reset = torch._neg_view(positive_reset).detach().requires_grad_()
    shared = torch.tensor(2.5, device="cuda", dtype=torch.float16).requires_grad_()
    parameters = (threshold, reset, shared, shared)

    # A lazy negative view shares positive physical storage. Direct pointer
    # kernels must resolve that metadata before reading, and the one shared
    # FP16 leaf used for c_m and tau stays in its original storage dtype.
    assert reset.is_neg() and reset.data_ptr() == positive_reset.data_ptr()
    prepared = _prepare(x_seq, states, parameters, spec, preserve_parameters=True)
    buffers = prepared[-2]
    assert not buffers["v_reset"].is_neg()
    assert buffers["c_m"] is buffers["tau"]
    assert buffers["c_m"].dtype == shared.dtype

    expected = python_module_sequence(x_seq, states, parameters, spec, history_names)
    actual = forward(backend, x_seq, states, parameters, spec, history_names)
    expected_flat = (expected[0], *expected[1], *expected[2])
    actual_flat = (actual[0], *actual[1], *actual[2])
    for result, reference in zip(actual_flat, expected_flat):
        torch.testing.assert_close(result, reference, atol=5e-6, rtol=5e-5)

    cotangents = tuple(torch.randn_like(value) for value in expected_flat)
    expected_gradients = torch.autograd.grad(
        expected_flat,
        (x_seq, *states, threshold, reset, shared),
        cotangents,
    )
    actual_gradients = backward(
        backend,
        x_seq,
        states,
        parameters,
        spec,
        history_names,
        actual[2],
        cotangents[0],
        cotangents[1:2],
        cotangents[2:],
    )
    actual_unique = (
        actual_gradients[0],
        *actual_gradients[1],
        actual_gradients[2][0],
        actual_gradients[2][1],
        actual_gradients[2][2] + actual_gradients[2][3],
    )
    for result, reference in zip(actual_unique, expected_gradients):
        torch.testing.assert_close(result, reference, atol=5e-4, rtol=5e-4)
