"""Serialization helpers for converting simulation data to xarray/Zarr format.

This module handles conversion of nested dictionaries (typically containing
simulation "memories" like spike trains, voltages, and synaptic states) into
xarray Datasets for storage and analysis.

Sparse Encoding Semantics
-------------------------
Arrays are encoded in sparse COO format when beneficial. For a variable
named ``spikes`` with shape ``(T, B, N)`` and ``nnz`` non-zero entries:

- ``spikes``: scalar marker with attrs ``{"_btorch_sparse": True, ...}``
- ``spikes_idx_time``: indices along time dim, shape ``(nnz,)``
- ``spikes_idx_batch``: indices along batch dim, shape ``(nnz,)``
- ``spikes_idx_neuron``: indices along neuron dim, shape ``(nnz,)``
- ``spikes_data``: actual values, shape ``(nnz,)``

The sparse dimension is named ``_btorch_sparse_idx_{var_name}`` to avoid
collisions. Original dtype and shape are preserved in the marker attrs.

Shape Conventions
-----------------
Dimension groups are specified via :class:`DimLayout`: ``dim_names`` (default:
``("time", "batch", "neuron")``) and ``dim_counts`` (how many physical
dimensions each logical group spans). For example:

- ``dim_counts=(1, 1, 2)`` with ``dim_names=("time", "batch", "neuron")``
  produces physical dims ``["time", "batch", "neuron_0", "neuron_1"]``
- A tensor of shape ``(100, 32, 64, 64)`` would map as
  ``(time=100, batch=32, neuron_0=64, neuron_1=64)``

Partial recordings (only a subset of neurons recorded) are expanded to full
size by filling missing entries with NaN (float) or 0 (integer/bool).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import scipy.sparse as sp

from ..utils._optional import require
from ..utils.array import to_numpy_or_sparse


if TYPE_CHECKING:
    import xarray as xr


@dataclass(frozen=True, kw_only=True)
class DimLayout:
    """Dimension layout of a memories dictionary.

    Attributes:
        dim_counts: Number of physical dimensions per logical group in
            ``dim_names``. If None, inferred from ``hint_field`` or heuristics.
        dim_names: Logical group names, in array order.
        hint_field: Flattened (dot-separated) field name used as the shape
            template for dimension inference.
        strict_dims: If True, every variable must match the global dimension
            structure exactly. If False, lower-rank arrays (e.g. parameters)
            are allowed.

    Raises:
        ValueError: If ``dim_counts`` has negative entries or a length that
            differs from ``dim_names``.
    """

    dim_counts: tuple[int, ...] | None = None
    dim_names: tuple[str, ...] = ("time", "batch", "neuron")
    hint_field: str | None = None
    strict_dims: bool = True

    def __post_init__(self) -> None:
        # Normalise sequences so the dataclass stays hashable and immutable.
        object.__setattr__(self, "dim_names", tuple(self.dim_names))
        if self.dim_counts is not None:
            object.__setattr__(self, "dim_counts", tuple(self.dim_counts))
            if len(self.dim_counts) != len(self.dim_names):
                raise ValueError(
                    f"dim_counts {self.dim_counts} must have one entry per "
                    f"dim_names {self.dim_names}."
                )
            if any(c < 0 for c in self.dim_counts):
                raise ValueError(
                    f"dim_counts must be non-negative, got {self.dim_counts}."
                )


@dataclass(frozen=True, kw_only=True)
class SparseOptions:
    """Sparse COO encoding policy for spike arrays.

    Attributes:
        spike_suffix: Substring (case-insensitive) identifying spike arrays.
        spike_dtype: Dtype dense spike arrays are cast to when sparse-encoded.
        sparse_threshold: Encode sparsely when ``nnz / size`` is below this
            ratio (0 to 1).
        force_sparse: True to force sparse encoding of all spike arrays, or the
            flattened variable names to force.

    Raises:
        ValueError: If ``sparse_threshold`` is outside ``[0, 1]``.
    """

    spike_suffix: str = "spike"
    spike_dtype: Any = bool
    sparse_threshold: float = 0.05
    force_sparse: bool | tuple[str, ...] = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparse_threshold <= 1.0:
            raise ValueError(
                f"sparse_threshold must be in [0, 1], got {self.sparse_threshold}."
            )
        if not isinstance(self.force_sparse, bool):
            object.__setattr__(self, "force_sparse", tuple(self.force_sparse))


@dataclass(frozen=True, kw_only=True)
class ZarrStoreOptions:
    """Compression, chunking and overwrite policy of a Zarr store.

    Attributes:
        compression_level: Zstd compression level (1-9, higher is smaller).
        chunks: Chunk size per dimension, e.g. ``{"time": 100, "neuron": -1}``;
            unlisted dimensions are not chunked (-1).
        overwrite: If True overwrite an existing store, else fail if it exists.

    Raises:
        ValueError: If ``compression_level`` is outside 1-9 or a chunk size is
            neither -1 nor positive.
    """

    compression_level: int = 5
    chunks: dict[str, int] | None = None
    overwrite: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.compression_level <= 9:
            raise ValueError(
                f"compression_level must be in 1-9, got {self.compression_level}."
            )
        for dim, size in (self.chunks or {}).items():
            if size != -1 and size <= 0:
                raise ValueError(
                    f"Chunk size for '{dim}' must be -1 or positive, got {size}."
                )


def to_sparse_repr(
    val: np.ndarray | sp.spmatrix | sp.sparray,
    var_dims: Sequence[str],
    var_name: str,
) -> dict[str, Any]:
    """Convert a dense or sparse array to sparse COO representation for
    storage.

    Supports arbitrary dtypes (float, int, bool). Only non-zero entries are
    stored. The returned dictionary contains index arrays per dimension and
    a data array, suitable for constructing an xr.Dataset.

    Args:
        val: Array to encode. Can be dense numpy or scipy sparse.
        var_dims: Physical dimension names for this variable (e.g.,
            ["time", "batch", "neuron"]).
        var_name: Base name for the variable (used to name output keys).

    Returns:
        Dictionary mapping variable names to (dims, data) tuples or xr.DataArray
        coords. Keys include:
            - ``{var_name}_idx_{dim}`` for each dimension
            - ``{var_name}_data`` for the values
            - ``{var_name}`` as a scalar marker with metadata attrs

    Shape semantics:
        - Input array with shape ``(*var_dims)`` and ``nnz`` non-zeros
        - Output index arrays: each has shape ``(nnz,)``
        - Output data array: shape ``(nnz,)``, dtype preserved from input
    """
    if sp.issparse(val):
        # Handle scipy sparse array
        coo = val.tocoo()
        indices = (coo.row, coo.col)
        nnz = coo.nnz
        data_vals = coo.data
    else:
        # Handle dense numpy array
        indices = np.nonzero(val)
        nnz = len(indices[0])
        data_vals = val[indices]

    ds_vars = {}

    # Use a unique dimension name for this variable's sparse indices
    sparse_dim_name = f"_btorch_sparse_idx_{var_name}"

    for i, d_name in enumerate(var_dims):
        idx_var = f"{var_name}_idx_{d_name}"
        # Indices are always integers
        ds_vars[idx_var] = ([sparse_dim_name], indices[i].astype(np.int32))

    data_var = f"{var_name}_data"
    # Preserve original dtype of values, ensure it's a standard numpy array
    ds_vars[data_var] = ([sparse_dim_name], np.asarray(data_vals))

    # Metadata marker
    ds_vars[var_name] = (
        [],
        np.int32(nnz),
        {
            "_btorch_sparse": True,
            "original_shape": list(val.shape),
            "original_dims": list(var_dims),
            "original_dtype": str(val.dtype),
            "nnz": int(nnz),
        },
    )
    return ds_vars


def from_spike_sparse(
    ds: xr.Dataset,
    var_name: str,
    return_sparse_2d: bool = False,
) -> tuple[np.ndarray | sp.coo_array, set[str]]:
    """Reconstruct a dense or scipy sparse array from btorch sparse encoding.

    Args:
        ds: Dataset containing the sparse-encoded variable.
        var_name: Name of the sparse marker variable (the scalar with attrs).
        return_sparse_2d: If True and the original was 2D, return a scipy
            coo_array instead of dense numpy.

    Returns:
        A tuple of (array, used_variable_names). The array is either dense
        numpy or scipy sparse (if 2D and requested). used_variable_names
        contains all dataset keys consumed during reconstruction.

    Shape semantics:
        - Output array has shape from ``original_shape`` attrs
        - Dense output: numpy array of original dtype
        - Sparse output (2D only): scipy.sparse.coo_array
    """
    attrs = ds[var_name].attrs
    shape = tuple(attrs["original_shape"])
    dims = attrs["original_dims"]
    dtype = np.dtype(attrs["original_dtype"])
    used_vars = {var_name}

    indices = []
    for d_name in dims:
        idx_name = f"{var_name}_idx_{d_name}"
        indices.append(ds[idx_name].values)
        used_vars.add(idx_name)

    data_name = f"{var_name}_data"
    data_vals = ds[data_name].values
    used_vars.add(data_name)

    if return_sparse_2d and len(shape) == 2:
        out = sp.coo_array((data_vals, (indices[0], indices[1])), shape=shape)
    else:
        out = np.zeros(shape, dtype=dtype)
        out[tuple(indices)] = data_vals

    return out, used_vars


def _expand_dim_names(dim_names: Sequence[str], dim_counts: Sequence[int]) -> list[str]:
    """Expand logical dimension groups into physical names.

    Examples:
        - ("time", "neuron"), (1, 2) -> ["time", "neuron_0", "neuron_1"]
        - ("time", "batch"), (1, 1) -> ["time", "batch"]
    """
    all_mapped_dims = []
    for count, name in zip(dim_counts, dim_names):
        if count == 1:
            all_mapped_dims.append(name)
        else:
            for i in range(count):
                all_mapped_dims.append(f"{name}_{i}")
    return all_mapped_dims


def _infer_dim_counts(
    val: np.ndarray,
    neuron_ids: np.ndarray | None,
    dim_names: Sequence[str] = ("time", "batch", "neuron"),
) -> tuple[int, int, int]:
    """Infer dim_counts (T, B, N) from a representative array by its rank.

    This is a heuristic (rank 3 -> T, B, N; rank 2 -> T, N; rank 1 -> N); pass
    ``dim_counts`` or ``hint_field`` explicitly when it is ambiguous. Ranks
    above 3 fall back to ``(1, 1, 1)`` and leave the extra trailing axes to the
    private-dimension logic of :func:`memories_to_xarray`.

    Returns:
        Tuple of (time_dims, batch_dims, neuron_dims) counts.
    """
    ndim = val.ndim
    if ndim == 3:
        return (1, 1, 1)
    if ndim == 2:
        return (1, 0, 1)
    if ndim == 1:
        return (0, 0, 1)
    return (1, 1, 1)


def _validate_and_infer_dims(
    flat_data: dict[str, Any],
    dim_names: Sequence[str],
    dim_counts: Sequence[int] | None,
    hint_field: str | None,
    neuron_ids: np.ndarray | None,
) -> tuple[Sequence[int], list[str], list[str]]:
    """Determine global dimension structure (dim_counts) and physical names.

    Counts come from ``dim_counts`` if given, else from the ``hint_field``
    array, else from the first variable; an empty ``flat_data`` gives
    ``(1, 1, 1)``.

    Returns:
        Tuple of (dim_counts, all_mapped_dims, neuron_group_dims).
    """
    if dim_counts is not None:
        resolved_counts = dim_counts
    elif hint_field and hint_field in flat_data:
        hint_val = to_numpy_or_sparse(flat_data[hint_field])
        resolved_counts = _infer_dim_counts(hint_val, neuron_ids)
    elif flat_data:
        first = to_numpy_or_sparse(next(iter(flat_data.values())))
        resolved_counts = _infer_dim_counts(first, neuron_ids)
    else:
        resolved_counts = (1, 1, 1)

    all_mapped_dims = _expand_dim_names(dim_names, resolved_counts)

    # Physical dims that belong to the logical "neuron" group.
    neuron_group_dims: list[str] = []
    if "neuron" in dim_names:
        neuron_idx = dim_names.index("neuron")
        pre_dims = sum(resolved_counts[:neuron_idx])
        n_dims = resolved_counts[neuron_idx]
        neuron_group_dims = all_mapped_dims[pre_dims : pre_dims + n_dims]

    return resolved_counts, all_mapped_dims, neuron_group_dims


def _expand_partial(
    val: np.ndarray,
    indices: np.ndarray,
    var_name: str,
    neuron_group_dims: Sequence[str],
    dim_registry: dict[str, int],
) -> np.ndarray:
    """Expand a partially recorded variable to the full neuron size.

    Contract: ``val`` holds only the recorded neurons along its trailing neuron
    dims and ``indices`` locates them in the full population. 1D indices
    address the flattened neuron dims; ``val`` must then still carry one
    trailing axis per neuron dim whose sizes multiply to ``len(indices)``
    (e.g. ``(T, B, 1, k)`` for two neuron dims). The leading (time/batch) shape is kept;
    the neuron dims are expanded to the size registered in ``dim_registry``
    (known from ``neuron_ids``) and unrecorded entries are filled with NaN
    (0 for non-float dtypes).

    Raises:
        ValueError: If no neuron dims are defined or their full size is unknown.
    """
    if not neuron_group_dims:
        raise ValueError(
            f"Cannot expand partial variable '{var_name}': no neuron "
            "dimensions are defined. Provide 'hint_field' or 'neuron_ids'."
        )
    if not all(d in dim_registry for d in neuron_group_dims):
        # Storing the variable as-is would mismatch dims later.
        raise ValueError(
            f"Cannot expand partial variable '{var_name}': Full neuron "
            f"dimensions unknown. Provide 'hint_field' or 'neuron_ids'."
        )

    lead_shape = val.shape[: -len(neuron_group_dims)]
    full_neuron_shape = tuple(dim_registry[d] for d in neuron_group_dims)
    full_shape = lead_shape + full_neuron_shape

    fill_val = np.nan if np.issubdtype(val.dtype, np.floating) else 0
    expanded = np.full(full_shape, fill_val, dtype=val.dtype)

    # Index the (possibly flattened) neuron axes with a leading full slice per
    # time/batch axis; ``reshape`` of the contiguous buffer is a view.
    lead = (slice(None),) * len(lead_shape)
    if len(neuron_group_dims) > 1 and indices.ndim == 1:
        flat = expanded.reshape(lead_shape + (-1,))
        flat[(*lead, indices)] = val.reshape(lead_shape + (-1,))
    else:
        expanded[(*lead, indices)] = val
    return expanded


def _resolve_var_dims(
    var_name: str,
    val: np.ndarray,
    all_mapped_dims: list[str],
    dim_registry: dict[str, int],
    hint_field: str | None,
    strict_dims: bool,
) -> list[str]:
    """Name the dimensions of one variable and register their sizes.

    Alignment contract: core dims (e.g. T, B, N) are a fixed prefix of every
    array; extra trailing dims (e.g. a synapse state of shape (T, B, N, 2)) get
    private names; lower-rank arrays are right-aligned to the core dims. A size
    that conflicts with an already registered dim is an error when
    ``hint_field`` is given, otherwise that axis gets a private name.
    ``dim_registry`` is updated in place.

    Raises:
        ValueError: On a size conflict with ``hint_field``, or if
            ``strict_dims`` and the variable has lower rank than the core dims.
    """
    n_core = len(all_mapped_dims)
    if val.ndim >= n_core:
        current_dims = list(all_mapped_dims)
        for i in range(val.ndim - n_core):
            current_dims.append(f"{var_name}_dim_{n_core + i}")
    else:
        current_dims = all_mapped_dims[n_core - val.ndim :]

    final_dims: list[str] = []
    for i, (d_name, size) in enumerate(zip(current_dims, val.shape)):
        if d_name not in dim_registry:
            dim_registry[d_name] = size
            final_dims.append(d_name)
        elif dim_registry[d_name] == size:
            final_dims.append(d_name)
        elif hint_field and d_name in all_mapped_dims:
            raise ValueError(
                f"Dimension mismatch for '{var_name}' on dim "
                f"'{d_name}': expected {dim_registry[d_name]}, "
                f"got {size}."
            )
        else:
            final_dims.append(f"{var_name}_d{i}")

    if strict_dims and len(final_dims) < n_core:
        # Lower-rank arrays (e.g. parameters) cannot be aligned to the global
        # dims.
        raise ValueError(
            f"Strict dimensions required: Variable '{var_name}' has "
            f"rank {len(final_dims)} but global dims are "
            f"{n_core} {all_mapped_dims}."
        )
    return final_dims


def _encode_variable(
    var_name: str,
    val: np.ndarray | sp.spmatrix | sp.sparray,
    var_dims: list[str],
    sparse: SparseOptions,
) -> dict[str, Any]:
    """Return the dataset entries for one variable (dense or sparse COO)."""
    spike_suffix, spike_dtype = sparse.spike_suffix, sparse.spike_dtype
    force_sparse = sparse.force_sparse
    is_spike = spike_suffix in var_name.lower()
    should_sparse = False

    if sp.issparse(val):
        should_sparse = True
    elif is_spike:
        if force_sparse is True or (
            isinstance(force_sparse, (list, tuple)) and var_name in force_sparse
        ):
            should_sparse = True
        else:
            nnz = np.count_nonzero(val)
            should_sparse = val.size == 0 or (nnz / val.size) < sparse.sparse_threshold

    if not should_sparse:
        return {var_name: (var_dims, val)}

    # to_sparse_repr preserves the input dtype; only dense arrays identified as
    # spikes are cast to ``spike_dtype``.
    if is_spike and spike_dtype is not None and not sp.issparse(val):
        val = val.astype(spike_dtype)
    return to_sparse_repr(val, var_dims, var_name)


def _root_id_entry(
    neuron_group_dims: Sequence[str],
    dim_registry: dict[str, int],
    neuron_ids: np.ndarray | None,
) -> tuple[list[str], np.ndarray] | None:
    """Build the ``root_id`` variable, or None if the neuron size is unknown.

    Without ``neuron_ids`` the ids default to ``arange``.

    Raises:
        ValueError: If ``neuron_ids`` cannot fill the neuron dims.
    """
    if not neuron_group_dims or not all(d in dim_registry for d in neuron_group_dims):
        return None
    shape = tuple(dim_registry[d] for d in neuron_group_dims)
    dims = list(neuron_group_dims)
    if neuron_ids is None:
        return dims, np.arange(np.prod(shape)).reshape(shape)
    if neuron_ids.shape == shape:
        return dims, neuron_ids
    if neuron_ids.size == np.prod(shape):
        return dims, neuron_ids.reshape(shape)
    raise ValueError(
        f"neuron_ids of shape {neuron_ids.shape} cannot fill the "
        f"neuron dims {dims} of shape {shape} for 'root_id'."
    )


def memories_to_xarray(
    memories: dict[str, Any],
    layout: DimLayout | None = None,
    *,
    neuron_ids: Any | None = None,
    partial_map: dict[str, Any] | None = None,
    sparse: SparseOptions | None = None,
) -> xr.Dataset:
    """Convert a nested dictionary of simulation results into an xr.Dataset.

    This function flattens a nested dictionary (e.g., from a simulation run
    containing spike trains, voltages, and synaptic states) and converts it
    into an xarray Dataset with consistent dimension naming and optional
    sparse encoding for spike arrays.

    Args:
        memories: Nested dictionary of arrays/tensors. Keys become variable
            names (dot-separated for nested dicts).
        layout: Dimension counts/names, shape hint and strictness (see
            :class:`DimLayout`). Defaults to ``DimLayout()``.
        neuron_ids: Optional neuron identifiers for ``root_id`` coordinate.
        partial_map: Dict of ``{field_name: indices}`` for fields recorded
            on a subset of neurons. Missing values filled with NaN (float)
            or 0 (integer/bool).
        sparse: Sparse spike encoding policy (see :class:`SparseOptions`).
            Defaults to ``SparseOptions()``.

    Returns:
        xr.Dataset with all variables, coordinates, and sparse encodings.

    Raises:
        ValueError: If a ``partial_map`` variable cannot be expanded because no
            neuron dimensions are defined or the full neuron size is unknown
            (provide ``layout.hint_field`` or ``neuron_ids``); if a variable's size
            conflicts with an already registered dimension while
            ``hint_field`` is given; if ``strict_dims`` is True and a variable
            has lower rank than the global dimensions; or if ``neuron_ids``
            cannot fill the neuron dimensions for ``root_id``.
        ImportError: If ``xarray`` is not installed.

    Example:
        >>> memories = {
        ...     "spike": torch.randn(100, 32, 128) > 0,  # (T, B, N)
        ...     "v": torch.randn(100, 32, 128),
        ... }
        >>> ds = memories_to_xarray(memories, DimLayout(dim_counts=(1, 1, 1)))
        >>> ds  # Dataset with dims (time: 100, batch: 32, neuron: 128)
    """
    from btorch.utils.dict_utils import flatten_dict

    layout = layout or DimLayout()
    sparse = sparse or SparseOptions()
    flat_data = flatten_dict(memories, dot=True)
    n_ids_arr = to_numpy_or_sparse(neuron_ids) if neuron_ids is not None else None

    _, all_mapped_dims, neuron_group_dims = _validate_and_infer_dims(
        flat_data,
        layout.dim_names,
        layout.dim_counts,
        layout.hint_field,
        n_ids_arr,
    )

    # Sizes of the core dims, locked by the first variable that uses them;
    # neuron dims are pre-locked by ``neuron_ids`` so partials can be expanded.
    dim_registry: dict[str, int] = {}
    if n_ids_arr is not None and len(neuron_group_dims) == n_ids_arr.ndim:
        for d, size in zip(neuron_group_dims, n_ids_arr.shape):
            dim_registry[d] = size

    ds_vars: dict[str, Any] = {}
    for var_name, val in flat_data.items():
        val = to_numpy_or_sparse(val)
        if partial_map and var_name in partial_map:
            indices = to_numpy_or_sparse(partial_map[var_name])
            val = _expand_partial(
                val, indices, var_name, neuron_group_dims, dim_registry
            )
        var_dims = _resolve_var_dims(
            var_name,
            val,
            all_mapped_dims,
            dim_registry,
            layout.hint_field,
            layout.strict_dims,
        )
        ds_vars.update(_encode_variable(var_name, val, var_dims, sparse))

    root_id = _root_id_entry(neuron_group_dims, dim_registry, n_ids_arr)
    if root_id is not None:
        ds_vars["root_id"] = root_id

    xr = require("xarray", "io", "xarray/Zarr serialization")
    ds = xr.Dataset(ds_vars)
    if "root_id" in ds:
        ds = ds.set_coords("root_id")
    return ds


def xarray_to_memories(
    ds: xr.Dataset,
    return_sparse_2d: bool = False,
) -> dict[str, Any]:
    """Convert an xr.Dataset back to a nested dictionary.

    Reconstructs the original nested dictionary structure from a Dataset
    created by ``memories_to_xarray``. Handles sparse-encoded variables
    automatically.

    Args:
        ds: Dataset to convert (typically loaded from Zarr).
        return_sparse_2d: If True, return 2D arrays as scipy sparse coo_array
            instead of dense numpy.

    Returns:
        Nested dictionary with restored variable names and structure.

    Example:
        >>> ds = xr.open_zarr("simulation.zarr")
        >>> memories = xarray_to_memories(ds)
        >>> memories["spike"].shape  # (T, B, N) or scipy sparse
    """
    flat_res: dict[str, Any] = {}
    reconstructed_vars = set()

    # Identify sparse btorch variables
    sparse_markers = [v for v in ds.variables if ds[v].attrs.get("_btorch_sparse")]
    for v in sparse_markers:
        out, used = from_spike_sparse(ds, v, return_sparse_2d=return_sparse_2d)
        flat_res[v] = out
        reconstructed_vars.update(used)

    # Load everything else
    for v in ds.variables:
        if v not in reconstructed_vars and v not in ds.dims:
            flat_res[v] = ds[v].values

    from btorch.utils.dict_utils import unflatten_dict

    return unflatten_dict(flat_res, dot=True)


def save_memories_to_xarray(
    memories: dict[str, Any],
    path: str | Path,
    layout: DimLayout | None = None,
    *,
    neuron_ids: Any | None = None,
    partial_map: dict[str, Any] | None = None,
    sparse: SparseOptions | None = None,
    store: ZarrStoreOptions | None = None,
) -> None:
    """Save a nested dictionary to a Zarr store via xarray.

    Convenience wrapper that converts the dictionary to a Dataset and saves
    with compression and optional chunking.

    Args:
        memories: Nested dictionary of arrays/tensors to save.
        path: Path to the output Zarr store.
        layout: Dimension layout (see :class:`DimLayout`).
        neuron_ids: Optional neuron identifiers.
        partial_map: Partial recording indices for subset fields.
        sparse: Sparse spike encoding policy (see :class:`SparseOptions`).
        store: Zarr compression, chunking and overwrite policy (see
            :class:`ZarrStoreOptions`). Defaults to ``ZarrStoreOptions()``.

    Raises:
        ImportError: If ``zarr`` or ``xarray`` is not installed, or no Blosc
            codec is available (``numcodecs`` for Zarr v2).
        ValueError: Propagated from :func:`memories_to_xarray` on inconsistent
            dimensions or partial-recording arguments.
        Exception: With ``store.overwrite=False`` an existing store makes
            ``xarray.Dataset.to_zarr`` fail (exception type depends on the
            installed zarr version).
    """
    require("zarr", "io", "Zarr serialization")
    store = store or ZarrStoreOptions()
    ds = memories_to_xarray(
        memories,
        layout,
        neuron_ids=neuron_ids,
        partial_map=partial_map,
        sparse=sparse,
    )

    encoding = {}

    try:
        from zarr.codecs import BloscCodec
    except ImportError:  # Zarr v2
        BloscCodec = None
    try:
        from numcodecs import Blosc
    except ImportError:  # Zarr v3-only environment
        Blosc = None

    if BloscCodec is not None:
        # Zarr v3 expects native codecs in `compressors`.
        compressor: Any = BloscCodec(cname="zstd", clevel=store.compression_level)
        compressor_key = "compressors"
        compressor_value: Any = [compressor]
    elif Blosc is not None:
        # Zarr v2 uses numcodecs and the `compressor` key.
        compressor = Blosc(
            cname="zstd", clevel=store.compression_level, shuffle=Blosc.BITSHUFFLE
        )
        compressor_key = "compressor"
        compressor_value = compressor
    else:
        raise ImportError(
            "No Blosc codec is available. Install `numcodecs` for Zarr v2 "
            '(pip install "btorch[io]"), or use Zarr v3 with '
            "`zarr.codecs.BloscCodec`."
        )

    for v_name in ds.variables:
        v_encoding: dict[str, Any] = {compressor_key: compressor_value}
        if store.chunks:
            v_chunks = [store.chunks.get(d, -1) for d in ds[v_name].dims]
            if any(c != -1 for c in v_chunks):
                v_encoding["chunks"] = v_chunks
        encoding[v_name] = v_encoding

    ds.to_zarr(
        path,
        mode="w" if store.overwrite else "w-",
        encoding=encoding,
        consolidated=True,
    )


def load_memories_from_xarray(
    path: str | Path, dask: bool = False, return_sparse_2d: bool = False
) -> dict[str, Any]:
    """Load a nested dictionary from a Zarr store.

    Args:
        path: Path to the Zarr store.
        dask: If True, return Dask-backed arrays (lazy loading). If False,
            load into memory immediately.
        return_sparse_2d: If True, return 2D arrays as scipy sparse coo_array.

    Returns:
        Nested dictionary with restored structure.

    Raises:
        ImportError: If ``xarray`` or ``zarr`` is not installed.
    """
    xr = require("xarray", "io", "xarray/Zarr serialization")
    require("zarr", "io", "Zarr serialization")
    ds = xr.open_zarr(path, consolidated=True, chunks="auto" if dask else None)
    return xarray_to_memories(ds, return_sparse_2d=return_sparse_2d)
