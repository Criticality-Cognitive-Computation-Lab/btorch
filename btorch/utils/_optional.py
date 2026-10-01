"""Helper for importing optional dependencies lazily."""

import importlib
from types import ModuleType


def require(module_name: str, extra: str, purpose: str | None = None) -> ModuleType:
    """Import an optional dependency, or raise ImportError with an install
    hint.

    Args:
        module_name: Dotted module path to import (e.g. ``"plotly.graph_objects"``).
        extra: Name of the btorch extra that provides the module (e.g. ``"viz"``).
        purpose: Optional short description of the feature needing the module.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        what = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"'{module_name.split('.')[0]}' is required{what} but is not "
            f'installed. Install it with: pip install "btorch[{extra}]"'
        ) from e
