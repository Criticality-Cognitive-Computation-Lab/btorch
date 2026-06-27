"""Public package entrypoint for Btorch."""

import importlib.metadata

from btorch import config, jit


__version__ = importlib.metadata.version(__name__)


__all__ = [
    "__version__",
    "config",
    "jit",
]
