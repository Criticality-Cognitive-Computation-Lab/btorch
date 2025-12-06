import os

import yaml


def save_yaml(args, folder_or_file, filename=None):
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
    file = (
        folder_or_file if filename is None else os.path.join(folder_or_file, filename)
    )
    with open(file, "r") as f:
        return yaml.safe_load(f)
