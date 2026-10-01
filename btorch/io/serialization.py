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
Dimension groups are specified via ``dim_names`` (default:
``("time", "batch", "neuron")``) and ``dim_counts`` (how many physical
dimensions each logical group spans). For example:

- ``dim_counts=(1, 1, 2)`` with ``dim_names=("time", "batch", "neuron")``
  produces physical dims ``["time", "batch", "neuron_0", "neuron_1"]``
- A tensor of shape ``(100, 32, 64, 64)`` would map as
  ``(time=100, batch=32, neuron_0=64, neuron_1=64)``

Partial recordings (only a subset of neurons recorded) are expanded to full
size by filling missing entries with NaN (float) or 0 (integer/bool).
"""

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
import torch
import xarray as xr


try:
    from zarr.codecs import BloscCodec
except ImportError:  # Zarr v2
    BloscCodec = None

try:
    from numcodecs import Blosc
except ImportError:  # Zarr v3-only environment
    Blosc = None


def _to_numpy(val: Any) -> np.ndarray | sp.spmatrix | sp.sparray:
    """Convert torch tensors, lists, or scalars to numpy/scipy arrays.

    Args:
        val: Input value (torch.Tensor, numpy array, scipy sparse, or scalar).

    Returns:
        Numpy array, scipy sparse matrix/array, or the input if already sparse.
    """
    if sp.issparse(val):
        return val
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu()
        if val.is_sparse:
            # Convert sparse torch tensor to scipy sparse
            return val.to_sparse_coo().to_scipy()
        return val.numpy()
    if not isinstance(val, (np.ndarray, np.generic)):
        if hasattr(val, "numpy"):  # Handle other tensor-like
            return val.numpy()
        return np.asarray(val)
    return val


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


def unique_val_dims(dims: Sequence[str]) -> set[str]:
    """Return unique dimension names from a sequence."""
    return set(dims)


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
    partial: bool = False,
    dim_names: Sequence[str] = ("time", "batch", "neuron"),
) -> tuple[int, int, int]:
    """Infer dim_counts (T, B, N) from a representative array.

    This is heuristic and may be ambiguous. Assumes:
    - neuron_ids (if provided) dictates N rank
    - If no neuron_ids, assume N=1
    - Assume T=1, remaining is B

    Returns:
        Tuple of (time_dims, batch_dims, neuron_dims) counts.
    """
    # This is a fallback heuristic. Better to have user hint.
    # If val is (T_d, B_d, N_d)
    ndim = val.ndim

    if partial:
        # Partial arrays might strictly be (T, B, N_partial)
        pass

    # Unspecified counts: fall back to conservative defaults by rank.
    if ndim == 3:
        return (1, 1, 1)
    elif ndim == 2:
        return (1, 0, 1)  # Time, Neuron
    elif ndim == 1:
        return (0, 0, 1)  # Just Neuron? Or Just Time?

    # If we are here, it's ambiguous. Return (1, 1, 1) and let validation fail
    # if mismatch.
    return (1, 1, 1)


def _validate_and_infer_dims(
    flat_data: dict[str, Any],
    dim_names: Sequence[str],
    dim_counts: Sequence[int] | None,
    hint_field: str | None,
    neuron_ids: np.ndarray | None,
) -> tuple[Sequence[int], list[str], list[str]]:
    """Determine global dimension structure (dim_counts) and physical names.

    Returns:
        Tuple of (dim_counts, all_mapped_dims, neuron_group_dims).
    """

    # 1. Resolve dim_counts
    resolved_counts = None

    if dim_counts is not None:
        resolved_counts = dim_counts
    elif hint_field and hint_field in flat_data:
        # Infer from hint
        hint_val = _to_numpy(flat_data[hint_field])
        resolved_counts = _infer_dim_counts(hint_val, neuron_ids)
    else:
        # Infer from first available non-partial, non-sparse array?
        # Or just default strict
        # User requirement: "without it just assume uniform shape" -> implied
        # uniform usually means (1,1,1) or existing logic
        # We will default to (1,1,1) if nothing else guides us, effectively.
        # But we need to handle rank mismatches.

        # Let's find a candidate
        candidate = None
        for k, v in flat_data.items():
            candidate = _to_numpy(v)
            break

        if candidate is not None:
            resolved_counts = _infer_dim_counts(candidate, neuron_ids)
        else:
            resolved_counts = (1, 1, 1)

    if resolved_counts is None:
        resolved_counts = (1, 1, 1)  # Should not happen

    # 2. Expand names
    all_mapped_dims = _expand_dim_names(dim_names, resolved_counts)

    # 3. Identify neuron dims
    neuron_group_dims = []
    # Re-run logic to find which valid physical dims belong to 'neuron'
    # We need to know which index in dim_names corresponds to 'neuron'.
    # Usually it's the last one.
    if "neuron" in dim_names:
        neuron_idx = dim_names.index("neuron")
        # How many dims before it?
        pre_dims = sum(resolved_counts[:neuron_idx])
        n_dims = resolved_counts[neuron_idx]
        neuron_group_dims = all_mapped_dims[pre_dims : pre_dims + n_dims]

    return resolved_counts, all_mapped_dims, neuron_group_dims


def memories_to_xarray(
    memories: dict[str, Any],
    dim_counts: Sequence[int] | None = None,
    dim_names: Sequence[str] = ("time", "batch", "neuron"),
    neuron_ids: Any | None = None,
    hint_field: str | None = None,
    partial_map: dict[str, Any] | None = None,
    strict_dims: bool = True,
    spike_suffix: str = "spike",
    spike_dtype: Any = bool,
    sparse_threshold: float = 0.05,
    force_sparse: bool | Sequence[str] = False,
) -> xr.Dataset:
    """Convert a nested dictionary of simulation results into an xr.Dataset.

    This function flattens a nested dictionary (e.g., from a simulation run
    containing spike trains, voltages, and synaptic states) and converts it
    into an xarray Dataset with consistent dimension naming and optional
    sparse encoding for spike arrays.

    Args:
        memories: Nested dictionary of arrays/tensors. Keys become variable
            names (dot-separated for nested dicts).
        dim_counts: Number of dimensions per logical group (time, batch,
            neuron). If None, inferred from ``hint_field`` or heuristics.
        dim_names: Logical group names for dimensions.
        neuron_ids: Optional neuron identifiers for ``root_id`` coordinate.
        hint_field: Field name (flattened, dot-separated) to use as shape
            template for dimension inference.
        partial_map: Dict of ``{field_name: indices}`` for fields recorded
            on a subset of neurons. Missing values filled with NaN (float)
            or 0 (integer/bool).
        strict_dims: If True, enforce exact dimension structure match.
            If False, allow lower-rank arrays (e.g., parameters).
        spike_suffix: Substring to identify spike arrays for sparse encoding.
        spike_dtype: Data type for spikes when converting dense to sparse.
        sparse_threshold: Sparsity ratio threshold for triggering sparse
            encoding (nnz / total_size < threshold).
        force_sparse: If True, force sparse encoding for all spike arrays.
            Can also be a list of specific field names to force sparse.

    Returns:
        xr.Dataset with all variables, coordinates, and sparse encodings.

    Example:
        >>> memories = {
        ...     "spike": torch.randn(100, 32, 128) > 0,  # (T, B, N)
        ...     "v": torch.randn(100, 32, 128),
        ... }
        >>> ds = memories_to_xarray(memories, dim_counts=(1, 1, 1))
        >>> ds  # Dataset with dims (time: 100, batch: 32, neuron: 128)
    """
    from btorch.utils.dict_utils import flatten_dict

    flat_data = flatten_dict(memories, dot=True)

    # Prepare inputs
    n_ids_arr = _to_numpy(neuron_ids) if neuron_ids is not None else None

    # 1. Determine Global Dimension Layout
    resolved_counts, all_mapped_dims, neuron_group_dims = _validate_and_infer_dims(
        flat_data, dim_names, dim_counts, hint_field, n_ids_arr
    )

    # 2. Establish Reference Shape (Locked Dimensions)
    # We strictly enforce that dimensions mapped by 'resolved_counts' match
    # across variables (except for partials, which we expand).

    # If we have a hint field, get the authoritative shape from it.
    dim_registry: dict[str, int] = {}

    # Pre-lock neuron dims if we have root_ids
    if n_ids_arr is not None and len(neuron_group_dims) == n_ids_arr.ndim:
        for d, s in zip(neuron_group_dims, n_ids_arr.shape):
            dim_registry[d] = s

    # 3. Process Variables
    ds_vars: dict[str, Any] = {}

    for var_name, val in flat_data.items():
        val = _to_numpy(val)

        # Handle Partial Recording Expansion
        if partial_map and var_name in partial_map:
            indices = _to_numpy(partial_map[var_name])

            # Contract: ``val`` holds only the recorded neurons along its trailing
            # neuron dims, and ``indices`` locates them in the full neuron
            # population. 1D indices address the flattened neuron dims. The
            # leading (time/batch) shape is kept from ``val``; the neuron dims are
            # expanded to their full size (known from ``hint_field`` or
            # ``neuron_ids``) and unrecorded entries are filled with NaN (0 for
            # non-float dtypes).
            if not neuron_group_dims:
                raise ValueError(
                    f"Cannot expand partial variable '{var_name}': no neuron "
                    "dimensions are defined. Provide 'hint_field' or 'neuron_ids'."
                )

            non_neuron_shape = val.shape[: -len(neuron_group_dims)]

            if all(d in dim_registry for d in neuron_group_dims):
                full_neuron_shape = tuple(dim_registry[d] for d in neuron_group_dims)
                full_shape = non_neuron_shape + full_neuron_shape

                fill_val = np.nan if np.issubdtype(val.dtype, np.floating) else 0
                expanded = np.full(full_shape, fill_val, dtype=val.dtype)

                if len(neuron_group_dims) > 1 and indices.ndim == 1:
                    # Flatten last K dims of expanded to assign, then reshape
                    # back. This is expensive but safe.
                    flattened_neuron_size = np.prod(full_neuron_shape)
                    temp = expanded.reshape(non_neuron_shape + (flattened_neuron_size,))
                    val_flat = val.reshape(non_neuron_shape + (-1,))

                    # Indexing logic: temp[..., indices] = val_flat
                    # We need to build a slice object
                    slicer = [slice(None)] * len(non_neuron_shape)
                    slicer.append(indices)
                    temp[tuple(slicer)] = val_flat

                    expanded = temp.reshape(full_shape)
                else:
                    # Standard indexing
                    slicer = [slice(None)] * len(non_neuron_shape)
                    slicer.append(indices)
                    expanded[tuple(slicer)] = val

                val = expanded
            else:
                # The full neuron size is unknown, so the variable cannot be
                # expanded; storing it as-is would mismatch dims later.
                raise ValueError(
                    f"Cannot expand partial variable '{var_name}': Full neuron "
                    f"dimensions unknown. Provide 'hint_field' or 'neuron_ids'."
                )

        # Determine dimensions for this variable
        n_dims = val.ndim

        # Alignment contract: core dims (e.g. T, B, N) are a fixed prefix of every
        # array; extra trailing dims (e.g. a synapse state of shape (T, B, N, 2))
        # get private names. Lower-rank arrays are right-aligned to the core dims.
        n_dims = val.ndim

        if n_dims >= len(all_mapped_dims):
            current_dims = list(all_mapped_dims)
            extra_count = n_dims - len(all_mapped_dims)
            for i in range(extra_count):
                # private extra dim
                current_dims.append(f"{var_name}_dim_{len(all_mapped_dims) + i}")
        else:
            # Rank deficiency: right-align against the core dims.
            current_dims = all_mapped_dims[len(all_mapped_dims) - n_dims :]

        # Validation against dim_registry
        final_dims = []
        for i, (d_name, size) in enumerate(zip(current_dims, val.shape)):
            if d_name in dim_registry:
                if dim_registry[d_name] != size:
                    # Size conflict: with a hint field the core dims must agree
                    # across variables (error); otherwise the mismatching axis
                    # falls back to a private per-variable dim.
                    if hint_field and d_name in all_mapped_dims:
                        raise ValueError(
                            f"Dimension mismatch for '{var_name}' on dim "
                            f"'{d_name}': expected {dim_registry[d_name]}, "
                            f"got {size}."
                        )

                    # Fallback to private dimension
                    new_name = f"{var_name}_d{i}"
                    final_dims.append(new_name)
                else:
                    final_dims.append(d_name)
            else:
                dim_registry[d_name] = size
                final_dims.append(d_name)

        # Validation of Rank/Suffix if Strict
        if strict_dims and len(final_dims) < len(all_mapped_dims):
            # Lower-rank arrays (e.g. parameters) cannot be aligned to the
            # global dims.
            raise ValueError(
                f"Strict dimensions required: Variable '{var_name}' has "
                f"rank {len(final_dims)} but global dims are "
                f"{len(all_mapped_dims)} {all_mapped_dims}."
            )

        var_dims = final_dims

        # Sparsity Handling
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
                should_sparse = (
                    (nnz / val.size) < sparse_threshold if val.size > 0 else True
                )

        if should_sparse:
            # to_sparse_repr preserves the input dtype; only dense arrays
            # identified as spikes are cast to ``spike_dtype``.
            if is_spike and spike_dtype is not None and not sp.issparse(val):
                # Only cast dense arrays that we identified as "spikes"
                val = val.astype(spike_dtype)

            ds_vars.update(to_sparse_repr(val, var_dims, var_name))
        else:
            ds_vars[var_name] = (var_dims, val)

    # 4. Add Root ID
    # If the user provided dimension names, we only use them to confirm
    # rank/order. We still rely on `resolved_counts` for the actual chunking
    # logic.
    if neuron_group_dims and all(d in dim_registry for d in neuron_group_dims):
        if neuron_ids is not None:
            # Verify shape
            expected_shape = tuple(dim_registry[d] for d in neuron_group_dims)
            if n_ids_arr.shape == expected_shape:
                ds_vars["root_id"] = (neuron_group_dims, n_ids_arr)
            elif n_ids_arr.size == np.prod(expected_shape):
                ds_vars["root_id"] = (
                    neuron_group_dims,
                    n_ids_arr.reshape(expected_shape),
                )
            else:
                # Mismatch, warning?
                pass
        else:
            # Default IDs
            shape = tuple(dim_registry[d] for d in neuron_group_dims)
            ds_vars["root_id"] = (
                neuron_group_dims,
                np.arange(np.prod(shape)).reshape(shape),
            )

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
    data: dict[str, Any],
    path: str | Path,
    dim_counts: Sequence[int] | None = None,
    dim_names: Sequence[str] = ("time", "batch", "neuron"),
    neuron_ids: Any | None = None,
    hint_field: str | None = None,
    partial_map: dict[str, Any] | None = None,
    strict_dims: bool = True,
    spike_suffix: str = "spike",
    spike_dtype: Any = bool,
    sparse_threshold: float = 0.05,
    compression_level: int = 5,
    chunks: dict[str, int] | None = None,
    overwrite: bool = True,
) -> None:
    """Save a nested dictionary to a Zarr store via xarray.

    Convenience wrapper that converts the dictionary to a Dataset and saves
    with compression and optional chunking.

    Args:
        data: Nested dictionary of arrays/tensors to save.
        path: Path to the output Zarr store.
        dim_counts: Dimension counts per logical group (see
            ``memories_to_xarray``).
        dim_names: Logical dimension names.
        neuron_ids: Optional neuron identifiers.
        hint_field: Field to use for shape inference.
        partial_map: Partial recording indices for subset fields.
        strict_dims: Enforce strict dimension matching.
        spike_suffix: Substring identifying spike arrays.
        spike_dtype: Dtype for spike conversion.
        sparse_threshold: Sparsity threshold for sparse encoding.
        compression_level: Zstd compression level (1-9, higher=smaller).
        chunks: Optional chunk sizes per dimension, e.g.,
            ``{"time": 100, "neuron": -1}``.
        overwrite: If True, overwrite existing store. If False, raise error
            if store exists.
    """
    ds = memories_to_xarray(
        data,
        dim_counts=dim_counts,
        dim_names=dim_names,
        neuron_ids=neuron_ids,
        hint_field=hint_field,
        partial_map=partial_map,
        strict_dims=strict_dims,
        spike_suffix=spike_suffix,
        spike_dtype=spike_dtype,
        sparse_threshold=sparse_threshold,
    )

    encoding = {}

    if BloscCodec is not None:
        # Zarr v3 expects native codecs in `compressors`.
        compressor: Any = BloscCodec(cname="zstd", clevel=compression_level)
        compressor_key = "compressors"
        compressor_value: Any = [compressor]
    elif Blosc is not None:
        # Zarr v2 uses numcodecs and the `compressor` key.
        compressor = Blosc(
            cname="zstd", clevel=compression_level, shuffle=Blosc.BITSHUFFLE
        )
        compressor_key = "compressor"
        compressor_value = compressor
    else:
        raise ImportError(
            "No Blosc codec is available. Install `numcodecs` for Zarr v2, "
            "or use Zarr v3 with `zarr.codecs.BloscCodec`."
        )

    for v_name in ds.variables:
        v_encoding: dict[str, Any] = {compressor_key: compressor_value}
        if chunks:
            v_chunks = [chunks.get(d, -1) for d in ds[v_name].dims]
            if any(c != -1 for c in v_chunks):
                v_encoding["chunks"] = v_chunks
        encoding[v_name] = v_encoding

    ds.to_zarr(
        path,
        mode="w" if overwrite else "w-",
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
    """
    ds = xr.open_zarr(path, consolidated=True, chunks="auto" if dask else None)
    return xarray_to_memories(ds, return_sparse_2d=return_sparse_2d)
