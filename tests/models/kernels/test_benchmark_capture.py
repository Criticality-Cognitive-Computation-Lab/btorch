"""Regression tests for foreign CUDA kernels in benchmark graph capture."""

import importlib.util

import numpy as np
import pytest
import torch

from btorch.models.kernels.benchmark import CaptureCase, capture_measure, tensor_bytes


_CUDA_CAPTURE_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("cupy") is None,
    reason="CUDA Graph and CuPy are required",
)


_SOURCE = r"""
extern "C" __global__ void recurrence(
    const float* input, float* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float neighbor = input[(index + 1) % count];
    float state = input[index];
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
      state = state * 0.75f + neighbor * 0.125f + float(step) * 0.01f;
    }
    output[index] = state;
  }
}
"""


def _reference(input):
    """Evaluate the same recurrence with ordinary Torch float32 operations."""
    neighbor = input.roll(-1)
    state = input.clone()
    for step in range(8):
        state = state * 0.75 + neighbor * 0.125 + step * 0.01
    return state


def _launch(cupy, kernel, input, output, *, bind_torch_stream):
    """Launch on either Torch's active stream or CuPy's foreign null stream."""
    arguments = (
        np.uintp(input.data_ptr()),
        np.uintp(output.data_ptr()),
        np.int32(input.numel()),
    )
    grid = ((input.numel() + 127) // 128,)
    with cupy.cuda.Device(input.device.index):
        if bind_torch_stream:
            stream = cupy.cuda.Stream.from_external(
                torch.cuda.current_stream(input.device)
            )
        else:
            stream = cupy.cuda.Stream.null
        with stream:
            kernel(grid, (128,), arguments)


def _build(cupy, kernel, *, bind_torch_stream):
    """Build the persistent input/output pair expected by graph capture."""
    capture_x = torch.linspace(0.2, 1.8, 257, device="cuda")
    captured_output = torch.empty_like(capture_x)

    def operation():
        _launch(
            cupy,
            kernel,
            capture_x,
            captured_output,
            bind_torch_stream=bind_torch_stream,
        )
        return (captured_output,)

    # Populate the persistent output before capture, matching the benchmark's
    # contract and reproducing the stale-but-initially-correct failure mode.
    operation()
    return CaptureCase(operation=operation, inputs=(capture_x,))


def _assert_eager_kernel(cupy, kernel, *, bind_torch_stream):
    """Prove compilation and ordinary launch work before testing capture."""
    input = torch.linspace(0.2, 1.8, 257, device="cuda")
    output = torch.empty_like(input)
    torch.cuda.synchronize()
    _launch(
        cupy,
        kernel,
        input,
        output,
        bind_torch_stream=bind_torch_stream,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, _reference(input), atol=1e-6, rtol=1e-6)


@_CUDA_CAPTURE_REQUIRED
def test_benchmark_rejects_uncaptured_cupy_kernel_after_input_change():
    """Accept a bound RawKernel graph and reject the same kernel when
    unbound."""
    cupy = pytest.importorskip("cupy")
    kernel = cupy.RawKernel(_SOURCE, "recurrence", options=("--fmad=false",))
    kernel.compile()

    # Both launch routes are valid outside capture. This prevents a compiler or
    # pointer interop failure from masquerading as the regression under test.
    _assert_eager_kernel(cupy, kernel, bind_torch_stream=True)
    _assert_eager_kernel(cupy, kernel, bind_torch_stream=False)

    result = capture_measure(
        lambda: _build(cupy, kernel, bind_torch_stream=True),
        warmup=2,
        repeats=3,
        validate=lambda actual, expected: torch.testing.assert_close(
            actual, expected, atol=0, rtol=0
        ),
        mutate=lambda inputs: inputs[0].add_(0.25),
    )
    assert "changed-input replay passed" in result.validation

    try:
        capture_measure(
            lambda: _build(cupy, kernel, bind_torch_stream=False),
            warmup=2,
            repeats=3,
            validate=lambda actual, expected: torch.testing.assert_close(
                actual, expected, atol=0, rtol=0
            ),
            mutate=lambda inputs: inputs[0].add_(0.25),
        )
    except AssertionError as exc:
        # A stale output is expected to fail either the bit-exact comparison or
        # the explicit changed-output requirement, both inside replay validation.
        message = str(exc)
        assert (
            "Captured outputs did not respond to changed input" in message
            or "Tensor-likes are not equal" in message
            or "Mismatched elements" in message
        )
    except (RuntimeError, cupy.cuda.runtime.CUDARuntimeError) as exc:
        # Some CUDA runtimes reject work on a foreign stream while global graph
        # capture is active. Accept only that explicit capture diagnostic.
        message = str(exc).lower()
        assert any(
            diagnostic in message
            for diagnostic in (
                "stream capture",
                "stream is capturing",
                "cudaerrorstreamcapture",
                "capture invalidated",
            )
        )
    else:
        pytest.fail("An unbound CuPy kernel was accepted as a captured replay")


def test_tensor_bytes_counts_alias_views_once():
    """Count one allocation when a result exposes both a tensor and its
    view."""
    base = torch.empty(16, dtype=torch.float32)
    clone = torch.empty(4, dtype=torch.float32)
    result = {"full": base, "view": base[::2], "clone": clone}

    expected = base.untyped_storage().nbytes() + clone.untyped_storage().nbytes()
    assert tensor_bytes(result) == expected
