"""YAML serialization utilities.

Simple helpers for loading and saving Python objects to YAML files, with
automatic directory creation.
"""

import os
from typing import Any

import yaml


def save_yaml(args: Any, folder_or_file: str, filename: "str | None" = None) -> None:
    """Save object to YAML file.

    Args:
        args: Object to serialize. Dumped with ``yaml.safe_dump``; if that
            rejects the object and it has a ``__dict__`` (e.g. an
            ``argparse.Namespace``), its attributes are dumped instead.
        folder_or_file: Directory path if ``filename`` is provided,
            otherwise full file path.
        filename: Optional filename when ``folder_or_file``
            is a directory.

    Raises:
        TypeError: If neither ``args`` nor ``vars(args)`` is YAML-safe
            serializable. Nothing is written in that case.
    """
    try:
        args_text = yaml.safe_dump(args)
    except yaml.representer.RepresenterError as e:
        if not hasattr(args, "__dict__"):
            raise TypeError(
                f"Cannot serialize {type(args).__name__} to YAML and it has no "
                "__dict__ to fall back on"
            ) from e
        try:
            args_text = yaml.safe_dump(vars(args))
        except yaml.representer.RepresenterError as e2:
            raise TypeError(
                f"Cannot serialize {type(args).__name__} to YAML: {e2}"
            ) from e2

    folder = os.path.dirname(folder_or_file) if filename is None else folder_or_file
    os.makedirs(folder, exist_ok=True)
    file = (
        folder_or_file if filename is None else os.path.join(folder_or_file, filename)
    )
    with open(file, "w") as f:
        f.write(args_text)


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
