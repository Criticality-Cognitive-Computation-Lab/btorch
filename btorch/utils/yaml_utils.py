"""YAML serialization utilities.

Simple helpers for loading and saving Python objects to YAML files, with
automatic directory creation.
"""

import os
from typing import Any

from btorch.utils._optional import require


yaml = require("yaml", "config", "YAML serialization")


def save_yaml(obj: Any, folder_or_file: str, filename: "str | None" = None) -> None:
    """Save object to YAML file.

    Args:
        obj: Object to serialize. Dumped with ``yaml.safe_dump``; if that
            rejects the object and it has a ``__dict__`` (e.g. an
            ``argparse.Namespace``), its attributes are dumped instead.
        folder_or_file: Directory path if ``filename`` is provided,
            otherwise full file path.
        filename: Optional filename when ``folder_or_file``
            is a directory.

    Raises:
        TypeError: If neither ``obj`` nor ``vars(obj)`` is YAML-safe
            serializable. Nothing is written in that case.
    """
    try:
        obj_text = yaml.safe_dump(obj)
    except yaml.representer.RepresenterError as e:
        if not hasattr(obj, "__dict__"):
            raise TypeError(
                f"Cannot serialize {type(obj).__name__} to YAML and it has no "
                "__dict__ to fall back on"
            ) from e
        try:
            obj_text = yaml.safe_dump(vars(obj))
        except yaml.representer.RepresenterError as e2:
            raise TypeError(
                f"Cannot serialize {type(obj).__name__} to YAML: {e2}"
            ) from e2

    folder = os.path.dirname(folder_or_file) if filename is None else folder_or_file
    os.makedirs(folder, exist_ok=True)
    file = (
        folder_or_file if filename is None else os.path.join(folder_or_file, filename)
    )
    with open(file, "w") as f:
        f.write(obj_text)


def load_yaml(folder_or_file: str, filename: "str | None" = None) -> Any:
    """Load object from YAML file.

    Args:
        folder_or_file: Directory path if ``filename`` is provided,
            otherwise full file path.
        filename: Optional filename when ``folder_or_file``
            is a directory.

    Returns:
        Deserialized Python object.
    """
    file = (
        folder_or_file if filename is None else os.path.join(folder_or_file, filename)
    )
    with open(file, "r") as f:
        return yaml.safe_load(f)
