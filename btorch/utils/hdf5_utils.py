"""HDF5 serialization utilities.

Helpers for saving and loading nested dictionaries containing arrays to
HDF5 files with optional Blosc2 compression for large arrays.
"""

import os
from typing import Optional

import h5py
import hdf5plugin


def save_dict_to_hdf5(
    folder_or_filename,
    data,
    compression=hdf5plugin.Blosc2(),
    filename: Optional[str] = None,
    compression_threshold=1024 * 1024,  # 1MiB
):
    """Save nested dictionary with array values to HDF5 file.

    Recursively traverses ``data`` and saves arrays as datasets.
    Datasets larger than ``compression_threshold`` are compressed
    with the specified compression filter.

    Args:
        folder_or_filename: Directory path if ``filename`` is provided,
            otherwise full file path.
        data: Nested dictionary with array-like values to serialize.
        compression: Compression filter (default: Blosc2).
        filename: Optional filename when ``folder_or_filename`` is a directory.
        compression_threshold: Minimum array size in bytes to trigger
            compression (default: 1 MiB).

    Returns:
        None
    """

    def save_array(h5file, path_k, v):
        if v.nbytes > compression_threshold:
            h5file.create_dataset(path_k, data=v, compression=compression)
        else:
            h5file.create_dataset(path_k, data=v)

    def save_group(h5file, path, data):
        for k, v in data.items():
            if v is None:
                continue
            elif isinstance(v, dict):
                h5file.create_group(f"{path}/{k}")
                save_group(h5file, f"{path}/{k}", v)
            elif hasattr(v, "shape") and hasattr(v, "dtype"):
                save_array(h5file, f"{path}/{k}", v)
            else:
                h5file.create_dataset(f"{path}/{k}", data=v)

    file = (
        folder_or_filename
        if filename is None
        else os.path.join(folder_or_filename, filename)
    )
    with h5py.File(file, "w") as f:
        save_group(f, "", data)


def load_dict_from_hdf5(folder_or_filename, filename: Optional[str] = None):
    """Load nested dictionary from HDF5 file.

    Args:
        folder_or_filename: Directory path if ``filename`` is provided,
            otherwise full file path.
        filename: Optional filename when ``folder_or_filename`` is a directory.

    Returns:
        Nested dictionary with restored array values.
    """
    file = (
        folder_or_filename
        if filename is None
        else os.path.join(folder_or_filename, filename)
    )

    def load_group(h5file, path):
        data = {}
        for k, v in h5file[path].items():
            if isinstance(v, h5py.Group):
                data[k] = load_group(h5file, f"{path}/{k}")
            else:
                if v.shape == ():
                    data[k] = v[()]  # Load scalar value
                else:
                    data[k] = v[:]  # Load array value
        return data

    with h5py.File(file, "r") as f:
        return load_group(f, "/")
