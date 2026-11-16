"""Qualify shared exponential arithmetic at cancellation and dtype
boundaries."""

import pytest
import torch

from btorch.models.kernels.backends.math import expm1_series
from tests.models.kernels.helpers import require_cuda_backend


def test_fast_expm1_series_preserves_tiny_updates_and_interval_accuracy():
    # Exponential Euler needs a nonzero update even when exp(x) rounds to one.
    # A dense independent FP64 oracle also checks the approximation's endpoints.
    x = torch.cat(
        (
            torch.linspace(-0.5, 0.5, 2049, dtype=torch.float64),
            torch.tensor([-1e-30, 0.0, 1e-30], dtype=torch.float64),
        )
    )
    actual = expm1_series(x)
    torch.testing.assert_close(actual, torch.expm1(x), atol=9e-9, rtol=0)
    torch.testing.assert_close(actual[-3:], x[-3:], atol=0, rtol=0)


@pytest.mark.parametrize(
    "backend,dtype",
    (
        ("triton", torch.float64),
        ("triton", torch.float16),
        ("tilelang", torch.float32),
        ("tilelang", torch.bfloat16),
    ),
)
def test_native_exponentials_preserve_dtype_small_updates_and_special_values(
    backend, dtype
):
    require_cuda_backend(backend)
    # Both modes share this single vector, covering the polynomial boundary,
    # tiny signed increments, stiff decay, and IEEE special values. The four
    # backend/dtype cases qualify distinct ABIs without a Cartesian matrix.
    x = torch.tensor(
        [
            -float("inf"),
            -100,
            -20,
            -0.5001,
            -0.5,
            -1e-5,
            -1e-8,
            -0.0,
            0,
            1e-8,
            1e-5,
            0.5,
            0.5001,
            20,
            float("inf"),
            float("nan"),
        ],
        device="cuda",
        dtype=dtype,
    )
    reference = torch.stack((torch.exp(x), torch.expm1(x)))
    for fast_math in (True, False):
        output = torch.empty_like(reference)
        if backend == "triton":
            import triton
            import triton.language as tl

            from btorch.models.kernels.backends.math_triton import exp, expm1

            @triton.jit
            def kernel(x_ptr, y_ptr, N: tl.constexpr, FAST: tl.constexpr):
                index = tl.arange(0, 32)
                value = tl.load(x_ptr + index, index < N, other=0)
                tl.store(y_ptr + index, exp(value, FAST), index < N)
                tl.store(y_ptr + N + index, expm1(value, FAST), index < N)

            kernel[(1,)](x, output, x.numel(), fast_math)
        else:
            import tilelang
            import tilelang.language as T

            from btorch.models.kernels.backends.math_tilelang import exp, expm1

            n, storage = x.numel(), str(dtype).removeprefix("torch.")

            @tilelang.jit
            def build():
                @T.prim_func
                def kernel(
                    x_ptr: T.Tensor((n,), storage), y_ptr: T.Tensor((2 * n,), storage)
                ):
                    with T.Kernel(1, threads=32):
                        index = T.get_thread_binding()
                        if index < n:
                            y_ptr[index] = exp(x_ptr[index], fast_math)
                            y_ptr[n + index] = expm1(x_ptr[index], fast_math)

                return kernel

            build()(x, output)
        assert output.dtype == dtype
        torch.testing.assert_close(
            output,
            reference,
            equal_nan=True,
            atol=0,
            rtol=max(8e-7 if dtype != torch.float64 else 0, 4 * torch.finfo(dtype).eps),
        )
