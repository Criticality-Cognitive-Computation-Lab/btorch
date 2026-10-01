"""I/O utilities for serializing and loading simulation data.

This module provides helpers for converting between PyTorch tensors,
nested dictionaries, and persistent storage formats (xarray/Zarr).
Key features include:

- Sparse array encoding for efficient spike storage
- Dimension-aware serialization with flexible (time, batch, neuron) grouping
- Automatic handling of partial recordings on neuron subsets
- Compression and chunking for large datasets
- Option dataclasses: `DimLayout`, `SparseOptions`, `ZarrStoreOptions`

Main entry points:
    - [`memories_to_xarray`](btorch/io/serialization.py): Convert nested
      dict to xr.Dataset
    - [`xarray_to_memories`](btorch/io/serialization.py): Restore nested
      dict from xr.Dataset
    - [`save_memories_to_xarray`](btorch/io/serialization.py): Save dict
      directly to Zarr
    - [`load_memories_from_xarray`](btorch/io/serialization.py): Load dict
      from Zarr store
"""

from btorch.io.serialization import (
    DimLayout,
    SparseOptions,
    ZarrStoreOptions,
    from_spike_sparse,
    load_memories_from_xarray,
    memories_to_xarray,
    save_memories_to_xarray,
    to_sparse_repr,
    xarray_to_memories,
)


__all__ = [
    "DimLayout",
    "SparseOptions",
    "ZarrStoreOptions",
    "from_spike_sparse",
    "load_memories_from_xarray",
    "memories_to_xarray",
    "save_memories_to_xarray",
    "to_sparse_repr",
    "xarray_to_memories",
]
