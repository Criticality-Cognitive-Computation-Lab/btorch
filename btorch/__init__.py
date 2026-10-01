"""Public package entrypoint for Btorch."""

from importlib.metadata import PackageNotFoundError, version

from btorch import config, jit


try:
    __version__ = version("btorch")
except PackageNotFoundError:
    # Imported from a source tree that is not installed (the version is derived
    # from git tags at build time, so it only exists in the installed metadata).
    # Same convention as xarray: report a huge version so that downstream minimum
    # version checks never disable features for a development copy.
    __version__ = "9999"


__all__ = [
    "__version__",
    "config",
    "jit",
]
