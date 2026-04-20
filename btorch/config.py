import os
from distutils.util import strtobool


try:
    from torch.jit import _enabled
except ImportError:
    from torch.jit._state import _enabled


def env_to_bool(name, default):
    return bool(strtobool(os.environ.get(name, "{}".format(default))))


JIT_ENABLED = env_to_bool("BTORCH_JIT", True)
SPARSE_BACKEND = os.environ.get("BTORCH_SPARSE_BACKEND")
SPARSE_BACKEND = SPARSE_BACKEND.lower() if SPARSE_BACKEND else None

__all__ = ["_enabled", "JIT_ENABLED", "SPARSE_BACKEND"]
