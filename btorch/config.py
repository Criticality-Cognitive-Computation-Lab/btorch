import os
from distutils.util import strtobool


try:
    from torch.jit import _enabled
except ImportError:
    from torch.jit._state import _enabled


def env_to_bool(name, default):
    return bool(strtobool(os.environ.get(name, "{}".format(default))))


JIT_ENABLED = env_to_bool("BTORCH_JIT", True)
SPARSE_BACKEND = os.environ.get("BTORCH_SPARSE_BACKEND", "torch_sparse").lower()
EVENT_SPARSE_MODE = os.environ.get("BTORCH_EVENT_SPARSE_MODE", "pre_span").lower()


def event_sparse_enabled() -> bool:
    """Return whether sparse event propagation is enabled for forward passes."""
    return env_to_bool("BTORCH_EVENT_SPARSE", False)


def event_sparse_mode() -> str:
    """Return the configured sparse event traversal mode."""
    return os.environ.get("BTORCH_EVENT_SPARSE_MODE", EVENT_SPARSE_MODE).lower()


__all__ = [
    "_enabled",
    "event_sparse_enabled",
    "event_sparse_mode",
    "EVENT_SPARSE_MODE",
    "JIT_ENABLED",
    "SPARSE_BACKEND",
]
