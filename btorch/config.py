# Optional numba support for accelerated hex grid operations
try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Provide no-op decorator if numba is not available
    def njit(*args, **kwargs):
        """No-op decorator when numba is not installed."""

        def decorator(func):
            return func

        return decorator


__all__ = [
    "HAS_NUMBA",
    "njit",
]
