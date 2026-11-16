"""Native Triton exponentials with an explicit fast-math policy."""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from .math import EXPM1_COEFFICIENTS


_COEFFICIENTS = tl.constexpr(EXPM1_COEFFICIENTS)


@triton.jit
def _expm1_series(x):
    value = tl.full(x.shape, _COEFFICIENTS[0], tl.float32)
    for index in tl.static_range(1, len(_COEFFICIENTS)):
        value = value * x + _COEFFICIENTS[index]
    return x * value


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
def expm1(x, fast_math: tl.constexpr = True):
    """Evaluate exp(x) - 1 without subtractive cancellation near zero"""
    if x.dtype == tl.float64:
        return libdevice.expm1(x)
    else:
        value = x.to(tl.float32)
        if fast_math:
            small = tl.abs(value) <= 0.5
            bounded = tl.where(small, value, 0.0)
            result = tl.where(small, _expm1_series(bounded), exp(value) - 1.0)
        else:
            result = libdevice.expm1(value)
        return result.to(x.dtype)
