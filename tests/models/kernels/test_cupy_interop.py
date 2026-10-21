"""Exercise raw CuPy launches against PyTorch-owned CUDA storage."""

import torch

from btorch.models.kernels import CellSpec, HistoryPlan
from btorch.models.kernels.operators import run_multi_step
from tests.models.kernels.helpers import require_cuda_backend


def _leaf(value):
    return value.detach().requires_grad_()


def _inputs():
    steps, batch, neurons = 8, 2, 5
    x_storage = torch.linspace(
        0.7,
        1.5,
        steps * batch * (neurons + 1),
        device="cuda",
        dtype=torch.float16,
    ).reshape(steps, batch, neurons + 1)
    x_seq = _leaf(x_storage[..., 1:])
    state_storage = torch.linspace(
        0.05,
        0.25,
        batch * (2 * neurons + 1),
        device="cuda",
        dtype=torch.float16,
    ).reshape(batch, 2 * neurons + 1)
    state = _leaf(state_storage[:, 1::2])

    reset_storage = torch.tensor([99.0, 0.0, 99.0], device="cuda")
    capacitance_storage = torch.linspace(0.9, 1.4, 2 * neurons + 1, device="cuda")
    tau_storage = torch.linspace(
        2.5,
        4.0,
        2 * neurons + 1,
        device="cuda",
        dtype=torch.float16,
    )
    parameters = (
        _leaf(torch.tensor(0.7, device="cuda")),
        _leaf(reset_storage[1:2]),
        _leaf(capacitance_storage[1::2]),
        _leaf(tau_storage[1::2]),
    )
    assert x_seq.storage_offset() > 0
    assert state.storage_offset() > 0 and state.stride(-1) == 2
    assert parameters[1].storage_offset() > 0
    assert parameters[2].storage_offset() > 0 and parameters[2].stride() == (2,)
    assert parameters[3].dtype == torch.float16 and parameters[3].stride() == (2,)
    return x_seq, (state,), parameters


def _evaluate(backend, cotangents=None):
    x_seq, states, parameters = _inputs()
    result = run_multi_step(
        x_seq,
        states,
        parameters,
        CellSpec("lif", dt=0.5, hard_reset=True),
        HistoryPlan(("v_charged",)),
        backend=backend,
    )
    if cotangents is None:
        generator = torch.Generator(device="cuda").manual_seed(417)
        cotangents = tuple(
            torch.randn(value.shape, generator=generator, device=value.device)
            for value in result
        )
    loss = sum((value * gradient).sum() for value, gradient in zip(result, cotangents))
    inputs = (x_seq, *states, *parameters)
    gradients = torch.autograd.grad(loss, inputs)
    return result, gradients, cotangents


def _assert_close(actual, expected):
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=2e-3, rtol=2e-3)


def _allocator_pressure():
    values = [torch.empty(1 << 18, device="cuda") for _ in range(12)]
    for index, value in enumerate(values):
        value.fill_(index)
    del values


def test_cupy_raw_pointers_survive_repeated_conversion_and_allocator_reuse():
    require_cuda_backend("cupy")
    reference, reference_gradients, cotangents = _evaluate("torch")
    assert 0 < reference[0].count_nonzero() < reference[0].numel()

    # Every call creates fresh FP32 conversion buffers for the FP16 drive,
    # state, and tau parameter. Releasing each graph before allocator pressure
    # makes stale raw pointers fail deterministically instead of being hidden by
    # the original tensors remaining live.
    for _ in range(3):
        actual, gradients, _ = _evaluate("cupy", cotangents)
        _assert_close(actual, reference)
        _assert_close(gradients, reference_gradients)
        del actual, gradients
        _allocator_pressure()


def test_cupy_raw_pointers_replay_forward_and_vjp_in_cuda_graph():
    require_cuda_backend("cupy")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x_seq, states, parameters = _inputs()
        inputs = (x_seq, *states, *parameters)
        cotangents = None
        for _ in range(2):
            warm = run_multi_step(
                x_seq,
                states,
                parameters,
                CellSpec("lif", dt=0.5, hard_reset=True),
                HistoryPlan(("v_charged",)),
                backend="cupy",
            )
            if cotangents is None:
                generator = torch.Generator(device="cuda").manual_seed(419)
                cotangents = tuple(
                    torch.randn(value.shape, generator=generator, device=value.device)
                    for value in warm
                )
            warm_loss = sum(
                (value * gradient).sum() for value, gradient in zip(warm, cotangents)
            )
            torch.autograd.grad(warm_loss, inputs)

    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run_multi_step(
            x_seq,
            states,
            parameters,
            CellSpec("lif", dt=0.5, hard_reset=True),
            HistoryPlan(("v_charged",)),
            backend="cupy",
        )
        captured_loss = sum(
            (value * gradient).sum() for value, gradient in zip(captured, cotangents)
        )
        captured_gradients = torch.autograd.grad(captured_loss, inputs)

    graph.replay()
    torch.cuda.synchronize()
    first_outputs = tuple(value.clone() for value in captured)
    first_gradients = tuple(value.clone() for value in captured_gradients)
    _allocator_pressure()
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(captured, first_outputs)
    _assert_close(captured_gradients, first_gradients)

    reference, reference_gradients, _ = _evaluate("torch", cotangents)
    _assert_close(captured, reference)
    _assert_close(captured_gradients, reference_gradients)
