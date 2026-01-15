import numpy as np
import pytest
import scipy.sparse as sp
import torch
import xarray as xr

from btorch.io.serialization import (
    load_memories_from_xarray,
    save_memories_to_xarray,
)


def test_save_load_roundtrip(tmp_path):
    # Prepare mock simulation data
    data = {
        "model": {
            "v": np.random.randn(100, 10, 50).astype(np.float32),
            "spikes": (np.random.rand(100, 10, 50) < 0.02).astype(np.float32),
            # In strict mode, this would fail if dim_counts implies (T,B,N)
            # We'll test strictness separately. Here we ensure roundtrip works for
            # uniform data.
        },
        "input": torch.randn(100, 10, 50),
    }

    save_path = tmp_path / "sim_result.zarr"

    # Save
    save_memories_to_xarray(data, save_path, dim_counts=(1, 1, 1))

    # Check if zarr exists
    assert save_path.exists()

    # Load
    loaded_data = load_memories_from_xarray(save_path)

    # Verify content
    assert "model" in loaded_data
    assert "input" in loaded_data
    np.testing.assert_allclose(loaded_data["model"]["v"], data["model"]["v"])
    np.testing.assert_allclose(loaded_data["model"]["spikes"], data["model"]["spikes"])
    np.testing.assert_allclose(loaded_data["input"], data["input"].numpy())


def test_sparse_matrix_support(tmp_path):
    # Test Scipy Sparse Matrices (COO/CSR) with float data
    shape = (100, 50)  # 2D matrix
    # Create random sparse matrix
    rng = np.random.default_rng(42)
    dense = rng.random(shape).astype(np.float32)
    mask = dense < 0.1
    dense[~mask] = 0

    coo = sp.coo_matrix(dense)
    csr = sp.csr_matrix(dense)

    data = {"weights_coo": coo, "weights_csr": csr}

    save_path = tmp_path / "sparse_matrix.zarr"
    # dim_counts=(1, 1) -> maps to (dim0, dim1)
    save_memories_to_xarray(
        data, save_path, dim_names=("pre", "post"), dim_counts=(1, 1)
    )

    ds = xr.open_zarr(save_path)

    # Check internal representation
    assert ds["weights_coo"].attrs["_btorch_sparse"] is True
    assert ds["weights_coo"].attrs["original_dtype"] == "float32"
    assert "weights_coo_idx_pre" in ds
    assert "weights_coo_data" in ds

    # Roundtrip load
    loaded = load_memories_from_xarray(save_path)

    # Should come back as dense by default unless return_sparse_2d=True
    np.testing.assert_allclose(loaded["weights_coo"], dense)
    np.testing.assert_allclose(loaded["weights_csr"], dense)

    # Test loading as sparse
    loaded_sparse = load_memories_from_xarray(save_path, return_sparse_2d=True)
    assert sp.issparse(loaded_sparse["weights_coo"])
    assert sp.issparse(loaded_sparse["weights_csr"])

    # Check values
    np.testing.assert_allclose(loaded_sparse["weights_coo"].toarray(), dense)


def test_strict_dims_validation(tmp_path):
    # Case: Parameter 'tau' has shape (10, 50) while global map is (100, 10, 50) -> (T,
    # B, N)
    data = {
        "v": np.random.randn(100, 10, 50),  # (T, B, N)
        "tau": np.random.randn(10, 50),  # (B, N) - missing T
    }

    save_path = tmp_path / "strict_fail.zarr"

    # 1. Strict Mode = True (Default) -> Should Raise ValueError
    with pytest.raises(ValueError, match="Strict dimensions required"):
        save_memories_to_xarray(data, save_path, dim_counts=(1, 1, 1), strict_dims=True)

    # 2. Strict Mode = False -> Should Succeed
    save_memories_to_xarray(
        data, save_path, dim_counts=(1, 1, 1), strict_dims=False, overwrite=True
    )

    loaded = load_memories_from_xarray(save_path)
    np.testing.assert_allclose(loaded["tau"], data["tau"])
    # Dimensions of tau in xarray should be (batch, neuron)
    # Since we aligned right (?), it matches B and N.


def test_spike_suffix_auto_sparse(tmp_path):
    # Test legacy behavior and new generic sparse behavior co-existing
    spikes = np.zeros((10, 10), dtype=bool)
    spikes[1, 1] = True

    data = {"neuron.spike": spikes}

    save_path = tmp_path / "spikes.zarr"
    # Ensure dim_names matches dim_counts length to avoid IndexError in inference
    save_memories_to_xarray(
        data, save_path, dim_counts=(1, 1), dim_names=("dim0", "dim1")
    )

    ds = xr.open_zarr(save_path)
    # Should be sparse because of suffix "spike" and low density
    assert ds["neuron.spike"].attrs.get("_btorch_sparse") is True
    # Should be boolean
    # Check data variable type
    data_var_name = "neuron.spike_data"
    assert ds[data_var_name].dtype == bool

    # Roundtrip
    loaded = load_memories_from_xarray(save_path)
    # Correctly access nested dictionary (dot=True unflattening happened)
    np.testing.assert_array_equal(loaded["neuron"]["spike"], spikes)


def test_glif_memory_states_complex(tmp_path):
    """
    Test requested by user:
    neuron.v: (T, B, N)
    neuron.Iasc: (T, B, N, 2)
    synapse.psc: (T, B, N)
    synapse.epsc: (T, B, Subset_N) - partial recording

    Using hint_field='neuron.v' and root_id.
    """
    T, B, N = 10, 5, 20
    subset_indices = np.array([0, 5, 10, 15], dtype=np.int64)  # 4 neurons

    data = {
        "neuron": {
            "v": np.random.randn(T, B, N).astype(np.float32),
            "Iasc": np.random.randn(T, B, N, 2).astype(np.float32),
        },
        "synapse": {
            "psc": np.random.randn(T, B, N).astype(np.float32),
            "epsc": np.random.randn(T, B, len(subset_indices)).astype(np.float32),
        },
    }

    root_id = np.arange(N) * 10
    partial_map = {"synapse.epsc": subset_indices}
    save_path = tmp_path / "glif_complex.zarr"

    # Inference with hint_field
    save_memories_to_xarray(
        data,
        save_path,
        dim_counts=None,  # Should infer (1, 1, 1) from neuron.v
        dim_names=("time", "batch", "neuron"),
        neuron_ids=root_id,
        hint_field="neuron.v",
        partial_map=partial_map,
        strict_dims=True,  # Regular variables must match
    )

    loaded = load_memories_from_xarray(save_path)
    np.testing.assert_allclose(loaded["neuron"]["v"], data["neuron"]["v"])
    # Partial check
    val_orig = data["synapse"]["epsc"]
    val_loaded = loaded["synapse"]["epsc"]
    # We expect val_loaded to be (T, B, N) with NaNs
    for i, idx in enumerate(subset_indices):
        np.testing.assert_allclose(val_loaded[:, :, idx], val_orig[:, :, i])


def test_multidim_dims(tmp_path):
    # time=(10, 5), batch=(2, 3), neuron=(4,)
    # extra dim: (2,)
    shape = (10, 5, 2, 3, 4, 2)
    data = {
        "Iasc": np.random.randn(*shape).astype(np.float32),
        "v": np.random.randn(10, 5, 2, 3, 4).astype(np.float32),
    }

    save_path = tmp_path / "multidim.zarr"
    # dim_counts: time=2, batch=2, neuron=1
    save_memories_to_xarray(
        data, save_path, dim_counts=(2, 2, 1), dim_names=("time", "batch", "neuron")
    )
    loaded = load_memories_from_xarray(save_path)
    np.testing.assert_allclose(loaded["Iasc"], data["Iasc"])


def test_neuron_indices(tmp_path):
    # Test custom neuron IDs
    n_shape = (2, 5)  # 2 subpopulations of 5 neurons
    data = {"v": np.random.randn(10, 2, 5).astype(np.float32)}
    # root_id provided as flat array of unique IDs
    my_ids = np.arange(1000, 1010)

    save_path = tmp_path / "neuron_ids.zarr"
    # Infer dims counts from hint or manual
    save_memories_to_xarray(data, save_path, dim_counts=(1, 0, 2), neuron_ids=my_ids)

    ds = xr.open_zarr(save_path)
    assert "root_id" in ds.coords
    assert ds["root_id"].shape == n_shape
