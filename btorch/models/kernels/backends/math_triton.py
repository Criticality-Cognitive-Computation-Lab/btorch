"""Native Triton exponentials with an explicit fast-math policy."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def exp(x, fast_math: tl.constexpr = True):
    """Evaluate exp, retaining libdevice accuracy for FP64 and precise mode."""
    if x.dtype == tl.float64:
        return libdevice.exp(x)
    else:
        value = x.to(tl.float32)
        if fast_math:
            result = tl.exp(value)
        else:
            result = libdevice.exp(value)
        return result.to(x.dtype)


@triton.jit
def expm1(x):
    """Evaluate cancellation-safe exp(x) - 1 using libdevice in either math mode

    Keep FP64 arithmetic intact and promote half/BF16 only for the
    transcendental ABI, rounding back to the input storage dtype.
    Triton's libdevice lowering can flush FP32 subnormals to zero; this
    helper does not add a separate correction to that backend behavior.
    """
    if x.dtype == tl.float64:
        return libdevice.expm1(x)
    return libdevice.expm1(x.to(tl.float32)).to(x.dtype)
