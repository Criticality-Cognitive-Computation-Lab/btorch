import os


try:
    from torch.jit import _enabled
except ImportError:
    from torch.jit._state import _enabled


_TRUE = {"y", "yes", "t", "true", "on", "1"}
_FALSE = {"n", "no", "f", "false", "off", "0"}


def env_to_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable (replaces removed distutils
    strtobool)

    Accepts y/yes/t/true/on/1 and n/no/f/false/off/0 (case-insensitive).
    Unset or empty falls back to ``default``.

    Raises:
        ValueError: If the variable is set to an unrecognised value.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    val = raw.strip().lower()
    if val in _TRUE:
        return True
    if val in _FALSE:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}")


# Read once at import time; btorch.jit decides at import whether to wrap with jit.
JIT_ENABLED = env_to_bool("BTORCH_JIT", True)

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
    "_enabled",
    "JIT_ENABLED",
    "HAS_NUMBA",
    "njit",
]
