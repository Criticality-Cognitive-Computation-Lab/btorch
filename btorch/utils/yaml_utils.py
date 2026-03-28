"""YAML serialization utilities.

Simple helpers for loading and saving Python objects to YAML files, with
automatic directory creation.
"""

import os

import yaml


def save_yaml(args, folder_or_file, filename=None):
    """Save object to YAML file.

    Args:
        args: Object to serialize. Tries ``safe_dump`` first, falls back
            to dumping ``args.__dict__``.
        folder_or_file: Directory path if ``filename`` is provided,
            otherwise full file path.
        filename: Optional filename when ``folder_or_file`` is a directory.

    Returns:
        None
    """
    try:
        args_text = yaml.safe_dump(args)
    except Exception:
        args_text = yaml.dump(args.__dict__)

    folder = os.path.dirname(folder_or_file) if filename is None else folder_or_file
    os.makedirs(folder, exist_ok=True)
    file = (
        folder_or_file if filename is None else os.path.join(folder_or_file, filename)
    )
    with open(file, "w") as f:
        f.write(args_text)


def load_yaml(folder_or_file, filename=None):
    """Load object from YAML file.

    Args:
        folder_or_file: Directory path if ``filename`` is provided,
            otherwise full file path.
        filename: Optional filename when ``folder_or_file`` is a directory.

    Returns:
        Deserialized Python object.
    """
    file = (
        folder_or_file if filename is None else os.path.join(folder_or_file, filename)
    )
    with open(file, "r") as f:
        return yaml.safe_load(f)
