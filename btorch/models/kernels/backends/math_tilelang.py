"""Native TileLang exponentials with an explicit fast-math policy."""

import tilelang.language as T

from .math import expm1_series


_LIBDEVICE_DECLARATIONS = """
extern "C" __device__ float __nv_expf(float);
extern "C" __device__ double __nv_exp(double);
extern "C" __device__ float __nv_expm1f(float);
extern "C" __device__ double __nv_expm1(double);
"""


def exp(x, fast_math=True):
    """Evaluate exp, retaining libdevice accuracy for FP64 and precise mode."""
    T.import_source(_LIBDEVICE_DECLARATIONS)
    dtype = x.dtype
    value = T.Cast("float32", x) if dtype in ("float16", "bfloat16") else x
    if fast_math and dtype != "float64":
        result = T.exp(value)
    else:
        result = T.call_pure_extern(
            value.dtype, "__nv_exp" if dtype == "float64" else "__nv_expf", value
        )
    return T.Cast(dtype, result)


def expm1(x, fast_math=True):
    """Evaluate exp(x) - 1 without subtractive cancellation near zero"""
    T.import_source(_LIBDEVICE_DECLARATIONS)
    dtype = x.dtype
    value = T.Cast("float32", x) if dtype in ("float16", "bfloat16") else x
    if fast_math and dtype != "float64":
        small = T.abs(value) <= 0.5
        bounded = T.if_then_else(small, value, T.Cast(value.dtype, 0.0))
        result = T.if_then_else(small, expm1_series(bounded), exp(value) - 1.0)
    else:
        result = T.call_pure_extern(
            value.dtype, "__nv_expm1" if dtype == "float64" else "__nv_expm1f", value
        )
    return T.Cast(dtype, result)
