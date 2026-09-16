"""Public package entrypoint for Btorch."""

import importlib.metadata

from btorch import config, jit, monitor


__version__ = importlib.metadata.version(__name__)


__all__ = [
    "__version__",
    "config",
    "jit",
    "monitor",
]
