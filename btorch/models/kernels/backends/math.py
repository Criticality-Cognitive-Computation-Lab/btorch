"""Shared cancellation-safe scalar approximations for native GPU helpers."""

# Descending coefficients of expm1(x) / x, shared by native helpers.
EXPM1_COEFFICIENTS = (1 / 40320, 1 / 5040, 1 / 720, 1 / 120, 1 / 24, 1 / 6, 0.5, 1.0)


def expm1_series(x):
    """Approximate exp(x) - 1 on [-0.5, 0.5] without cancellation

    The degree-eight Taylor remainder is below 9e-9 on this interval.
    Backend helpers use this polynomial only for FP32 arithmetic, retaining
    libdevice for FP64 and for precise mode. Storage rounding belongs to
    the caller, just as it does for other scalar kernel operations.
    """
    value = EXPM1_COEFFICIENTS[0]
    for coefficient in EXPM1_COEFFICIENTS[1:]:
        value = value * x + coefficient
    return x * value
