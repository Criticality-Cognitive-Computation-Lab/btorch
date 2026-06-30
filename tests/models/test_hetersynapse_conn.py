"""Tests for hetersynapse connection and constraint functionality.

This test module covers:
1. make_hetersynapse_conn with dict returns and NaN handling
2. make_hetersynapse_constraint with different constraint modes
3. make_hetersynapse_constrained_conn integration
4. HeterSynapsePSC.get_psc with autodetection
5. SparseConstrainedConn enhancements (from_hetersynapse, helper methods)

Tests also serve as examples and documentation for how to use these features.
"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import torch

from btorch.connectome.connection import (
    make_hetersynapse_conn,
    make_hetersynapse_constrained_conn,
    make_hetersynapse_constraint,
)
from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net
from btorch.models.history import SpikeHistory
from btorch.models.linear import SparseConstrainedConn
from btorch.models.synapse import AlphaPSC, ExponentialPSC, HeterSynapsePSC
from tests.utils.file import save_fig


def create_test_neurons(n_neurons=100, seed=42):
    """Create test neuron DataFrame with cell types and receptor types.

    This helper creates realistic test data for synapse testing.

    Args:
        n_neurons: Number of neurons to create
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: root_id, simple_id, cell_type, EI
    """
    np.random.seed(seed)

    # Create neuron IDs
    neurons = pd.DataFrame(
        {
            "root_id": np.arange(1000, 1000 + n_neurons),
            "simple_id": np.arange(n_neurons),
        }
    )

    # Assign cell types (e.g., different neuron classes)
    n_cell_types = 5
    neurons["cell_type"] = [f"type_{i % n_cell_types}" for i in range(n_neurons)]

    # Assign receptor types (E or I)
    neurons["EI"] = np.random.choice(["E", "I"], size=n_neurons, p=[0.7, 0.3])

    return neurons


def create_test_connections(neurons, density=0.1, seed=42):
    """Create test connection DataFrame.

    Args:
        neurons: DataFrame from create_test_neurons
        density: Connection density (fraction of possible connections)
        seed: Random seed

    Returns:
        DataFrame with columns: pre_root_id, post_root_id, pre_simple_id,
        post_simple_id, syn_count
    """
    np.random.seed(seed)

    n_neurons = len(neurons)
    n_connections = int(n_neurons**2 * density)

    # Create random connections
    pre_idx = np.random.choice(n_neurons, size=n_connections)
    post_idx = np.random.choice(n_neurons, size=n_connections)

    # Remove self-connections
    mask = pre_idx != post_idx
    pre_idx = pre_idx[mask]
    post_idx = post_idx[mask]

    connections = pd.DataFrame(
        {
            "pre_simple_id": pre_idx,
            "post_simple_id": post_idx,
            "syn_count": np.random.randint(1, 10, size=len(pre_idx)),
        }
    )

    # Add root IDs
    connections["pre_root_id"] = connections["pre_simple_id"].map(
        dict(zip(neurons["simple_id"], neurons["root_id"]))
    )
    connections["post_root_id"] = connections["post_simple_id"].map(
        dict(zip(neurons["simple_id"], neurons["root_id"]))
    )

    return connections


def test_make_hetersynapse_conn_dict_return_neuron_mode():
    """Test make_hetersynapse_conn with return_dict=True in neuron mode.

    This demonstrates how to get separate matrices for each receptor
    type pair, allowing manual scaling or manipulation before combining.
    """
    neurons = create_test_neurons(n_neurons=50)
    connections = create_test_connections(neurons, density=0.15)

    # Get dict return
    conn_dict, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        return_dict=True,
    )

    # Verify it's a dict
    assert isinstance(conn_dict, dict)

    # Check structure
    assert len(conn_dict) > 0

    # Each key should be (pre_receptor, post_receptor) tuple
    for key, mat in conn_dict.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert all(isinstance(k, str) for k in key)

        # Each value should be a sparse matrix
        assert scipy.sparse.issparse(mat)
        assert mat.shape == (len(neurons), len(neurons))

    # receptor_idx should have pre/post receptor types
    assert "pre_receptor_type" in receptor_idx.columns
    assert "post_receptor_type" in receptor_idx.columns
    assert "receptor_index" in receptor_idx.columns

    print(f"✓ Dict return created {len(conn_dict)} receptor type pairs")
    print(f"  Receptor pairs: {list(conn_dict.keys())}")


def test_make_hetersynapse_conn_nan_handling():
    """Test NaN handling in make_hetersynapse_conn.

    This demonstrates the dropna parameter and error handling for
    missing receptor type data.
    """
    neurons = create_test_neurons(n_neurons=30)
    connections = create_test_connections(neurons, density=0.2)

    # Add some NaN receptor types
    neurons.loc[5:8, "EI"] = np.nan

    # Should raise error by default (dropna='error')
    with pytest.raises(ValueError, match="NaN receptor types found"):
        make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            dropna="error",
        )

    # Test dropna='filter' - removes connections involving NaN neurons
    with pytest.warns(UserWarning, match="Filtered out .* connections involving"):
        conn_mat_filter, receptor_idx_filter = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            dropna="filter",
        )

    assert scipy.sparse.issparse(conn_mat_filter)
    # Neuron count should be preserved
    assert conn_mat_filter.shape[0] == len(neurons)
    print("✓ dropna='filter' works correctly (preserves neuron count)")

    # Test dropna='unknown' - treats NaN as a separate receptor type
    with pytest.warns(UserWarning, match="Treating .* neurons with NaN receptor types"):
        conn_mat_unknown, receptor_idx_unknown = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            dropna="unknown",
        )

    assert scipy.sparse.issparse(conn_mat_unknown)
    # Neuron count should be preserved
    assert conn_mat_unknown.shape[0] == len(neurons)
    # Should have more receptor type pairs (including NaN combinations)
    assert len(receptor_idx_unknown) > len(receptor_idx_filter)
    print("✓ dropna='unknown' works correctly (NaN as receptor type)")

    print("✓ NaN handling works correctly")


def test_make_hetersynapse_constraint_modes():
    """Test different constraint modes in make_hetersynapse_constraint.

    This demonstrates the three constraint granularity options and
    visualizes the resulting number of constraint groups.
    """
    neurons = create_test_neurons(n_neurons=60)
    connections = create_test_connections(neurons, density=0.15)

    modes = ["full", "cell_only", "cell_and_receptor"]
    results = {}

    for mode in modes:
        constraint = make_hetersynapse_constraint(
            neurons,
            connections,
            cell_type_col="cell_type",
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            constraint_mode=mode,
        )

        # Get number of unique groups
        n_groups = int(constraint.data.max())
        results[mode] = n_groups

        assert scipy.sparse.issparse(constraint)
        assert constraint.nnz > 0  # Should have some constraints

    # Visualize constraint group counts
    fig, ax = plt.subplots(figsize=(8, 5))
    modes_list = list(results.keys())
    counts = [results[m] for m in modes_list]

    bars = ax.bar(modes_list, counts, color=["#3498db", "#e74c3c", "#2ecc71"])
    ax.set_ylabel("Number of Constraint Groups")
    ax.set_title("Constraint Groups by Mode")

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{count}",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, max(counts) * 1.2)
    fig.tight_layout()
    save_fig(fig, "hetersynapse_constraint_modes")
    plt.close(fig)

    # Verify expected ordering: full > cell_and_receptor >= cell_only
    assert results["full"] >= results["cell_and_receptor"]
    assert results["cell_and_receptor"] >= results["cell_only"]

    print("✓ Constraint modes work correctly:")
    for mode, n_groups in results.items():
        print(f"  {mode:20s}: {n_groups} groups")


def test_make_hetersynapse_constrained_conn():
    """Test the convenience function that creates both conn and constraint.

    This demonstrates the recommended workflow for creating
    hetersynaptic connections with constraints ready for
    SparseConstrainedConn.
    """
    neurons = create_test_neurons(n_neurons=40)
    connections = create_test_connections(neurons, density=0.2)

    # Create both matrices in one call
    conn, constraint, receptor_idx = make_hetersynapse_constrained_conn(
        neurons,
        connections,
        cell_type_col="cell_type",
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        constraint_mode="cell_only",  # Share weights across receptor types
    )

    # Verify outputs
    assert scipy.sparse.issparse(conn)
    assert scipy.sparse.issparse(constraint)
    assert isinstance(receptor_idx, pd.DataFrame)

    # Shapes should match
    assert conn.shape == constraint.shape

    # Constraint should have valid group IDs
    assert constraint.nnz > 0
    assert constraint.data.min() >= 1

    print("✓ make_hetersynapse_constrained_conn works correctly")
    print(f"  Connection shape: {conn.shape}")
    print(f"  Constraint groups: {int(constraint.data.max())}")


def test_hetersynapse_psc_get_psc_autodetection():
    """Test HeterSynapsePSC.get_psc with autodetection of receptor modes.

    This demonstrates how the mode is automatically detected from the
    receptor_type_index DataFrame structure.
    """
    neurons = create_test_neurons(n_neurons=30)
    connections = create_test_connections(neurons, density=0.2)

    # Create connection matrix
    conn_sp, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
    )

    n_receptor = len(receptor_idx)
    n_neurons = len(neurons)

    # Create linear layer
    linear = torch.nn.Linear(n_neurons, n_neurons * n_receptor, bias=False)

    with environ.context(dt=1.0):
        hetero_psc = HeterSynapsePSC(
            n_neuron=n_neurons,
            n_receptor=n_receptor,
            receptor_type_index=receptor_idx,
            linear=linear,
            base_psc=AlphaPSC,
            tau_syn=2.0,
        )

        init_net_state(hetero_psc, dtype=torch.float32)

        # Run a forward pass
        z = torch.randn(1, n_neurons)
        hetero_psc.single_step_forward(z)

    # Test autodetection with neuron mode (tuple input)
    if "pre_receptor_type" in receptor_idx.columns:
        # Get unique receptor types
        pre_types = receptor_idx["pre_receptor_type"].unique()
        post_types = receptor_idx["post_receptor_type"].unique()

        if len(pre_types) > 0 and len(post_types) > 0:
            # Try getting PSC by receptor pair
            pre_type = pre_types[0]
            post_type = post_types[0]

            # get_psc should use the psc from base_psc, pass it explicitly
            psc = hetero_psc.get_psc(
                receptor_type=(pre_type, post_type), psc=hetero_psc.base_psc.psc
            )
            assert psc is not None
            assert psc.shape[-1] == n_neurons

            print("✓ Autodetection works for neuron mode")
            print(f"  Retrieved PSC for ({pre_type}, {post_type})")

    # Get total PSC (None argument) - returns base_psc.psc with all receptor types
    psc_total = hetero_psc.get_psc(receptor_type=None)
    # The total PSC includes all receptor types, so shape is (n_neurons * n_receptor,)
    assert psc_total.shape[-1] == n_neurons * n_receptor

    print("✓ get_psc autodetection works correctly")


def test_sparse_constrained_conn_from_hetersynapse():
    """Test SparseConstrainedConn.from_hetersynapse class method.

    This demonstrates the clean workflow for creating constrained
    connections from hetersynapse data.
    """
    neurons = create_test_neurons(n_neurons=50)
    connections = create_test_connections(neurons, density=0.15)

    # Create heterosynapse connection and constraint
    conn, constraint, receptor_idx = make_hetersynapse_constrained_conn(
        neurons,
        connections,
        cell_type_col="cell_type",
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        constraint_mode="full",
    )

    # Use the class method
    linear = SparseConstrainedConn.from_hetersynapse(
        conn=conn,
        constraint=constraint,
        receptor_type_index=receptor_idx,
        enforce_dale=True,
    )

    # Verify constraint_info is populated
    assert linear.constraint_info is not None
    assert "receptor_type_index" in linear.constraint_info
    pd.testing.assert_frame_equal(
        linear.constraint_info["receptor_type_index"], receptor_idx
    )

    print("✓ from_hetersynapse class method works correctly")
    print(f"  Magnitude shape: {linear.magnitude.shape}")


def test_sparse_constrained_conn_helper_methods():
    """Test SparseConstrainedConn helper methods for inspection and
    manipulation.

    This demonstrates:
    1. get_group_info() for inspecting constraint groups
    2. set_group_magnitude() for programmatic weight manipulation
    3. get_weights_by_group() for analysis
    """
    neurons = create_test_neurons(n_neurons=40)
    connections = create_test_connections(neurons, density=0.2)

    conn, constraint, receptor_idx = make_hetersynapse_constrained_conn(
        neurons,
        connections,
        cell_type_col="cell_type",
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        constraint_mode="cell_and_receptor",
    )

    linear = SparseConstrainedConn.from_hetersynapse(
        conn, constraint, receptor_idx, enforce_dale=True
    )

    # Test get_group_info
    group_info = linear.get_group_info(include_weights=True)
    assert isinstance(group_info, pd.DataFrame)
    assert "group_id" in group_info.columns
    assert "num_connections" in group_info.columns
    assert "current_magnitude" in group_info.columns
    assert "mean_initial_weight" in group_info.columns

    print("✓ get_group_info works correctly")
    print(f"  Found {len(group_info)} constraint groups")

    # Test set_group_magnitude by group_id
    linear.set_group_magnitude(group_id=0, value=2.5)
    assert torch.isclose(linear.magnitude[0], torch.tensor(2.5))

    # Test get_weights_by_group
    weights_by_group = linear.get_weights_by_group()
    assert isinstance(weights_by_group, dict)
    assert len(weights_by_group) == len(linear.magnitude)

    # Visualize group sizes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Group sizes
    ax = axes[0]
    group_ids = group_info["group_id"].values
    num_conns = group_info["num_connections"].values
    ax.bar(group_ids, num_conns, color="#3498db")
    ax.set_xlabel("Group ID")
    ax.set_ylabel("Number of Connections")
    ax.set_title("Connections per Constraint Group")

    # Plot 2: Current magnitudes
    ax = axes[1]
    magnitudes = group_info["current_magnitude"].values
    ax.bar(group_ids, magnitudes, color="#e74c3c")
    ax.set_xlabel("Group ID")
    ax.set_ylabel("Magnitude")
    ax.set_title("Current Magnitude per Group")
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Initial (1.0)")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "constraint_group_info")
    plt.close(fig)

    print("✓ All helper methods work correctly")


def test_hetersynapse_workflow_example():
    """Complete workflow example demonstrating the full pipeline.

    This test serves as documentation for the recommended usage pattern.
    """
    # Step 1: Create test data
    neurons = create_test_neurons(n_neurons=60)
    connections = create_test_connections(neurons, density=0.15)

    # Step 2: Create hetersynapse connection with constraints
    conn, constraint, receptor_idx = make_hetersynapse_constrained_conn(
        neurons,
        connections,
        cell_type_col="cell_type",
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        constraint_mode="cell_only",  # Share weights across receptor types
    )

    # Step 3: Initialize constrained connection
    linear = SparseConstrainedConn.from_hetersynapse(
        conn, constraint, receptor_idx, enforce_dale=True
    )

    # Step 4: Inspect constraint groups
    group_info = linear.get_group_info()
    print(f"Created {len(group_info)} constraint groups")

    # Step 5: (Optional) Manually adjust magnitudes
    # For example, boost E→I connections
    linear.set_group_magnitude(group_id=0, value=1.5)

    # Step 6: Use in forward pass
    n_neurons = len(neurons)
    x = torch.randn(10, n_neurons)  # batch_size=10
    y = linear(x)

    assert y.shape == (10, conn.shape[1])

    print("✓ Complete workflow example passed")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")


def test_stack_hetersynapse_dict():
    """Test converting dict format back to stacked matrix format.

    This demonstrates the round-trip conversion and allows for
    modification of individual receptor type matrices before stacking.
    """
    from btorch.connectome.connection import stack_hetersynapse

    neurons = create_test_neurons(n_neurons=40)
    connections = create_test_connections(neurons, density=0.15)

    # Get dict format
    conn_dict, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        return_dict=True,
    )

    # Get stacked format for comparison
    conn_stacked_ref, receptor_idx_ref = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        return_dict=False,
    )

    # Convert dict back to stacked
    conn_stacked = stack_hetersynapse(conn_dict, receptor_idx)

    # Should match the reference
    assert conn_stacked.shape == conn_stacked_ref.shape

    # Convert to same format for comparison
    conn_stacked = conn_stacked.tocsr()
    conn_stacked_ref = conn_stacked_ref.tocsr()

    # Check they're equal (data, indices, indptr)
    assert np.allclose(conn_stacked.data, conn_stacked_ref.data)
    assert np.array_equal(conn_stacked.indices, conn_stacked_ref.indices)
    assert np.array_equal(conn_stacked.indptr, conn_stacked_ref.indptr)

    print("✓ stack_hetersynapse_dict correctly converts dict to stacked format")

    # Test with modification
    conn_dict_modified = {}
    for k, v in conn_dict.items():
        # Convert to float for modification
        v_copy = v.copy()
        v_copy.data = v_copy.data.astype(float)
        conn_dict_modified[k] = v_copy

    # Scale E->I connections by 2.0
    if ("E", "I") in conn_dict_modified:
        conn_dict_modified[("E", "I")].data *= 2.0

    conn_stacked_modified = stack_hetersynapse(conn_dict_modified, receptor_idx)

    # The modified matrix should be different
    conn_stacked_modified = conn_stacked_modified.tocsr()
    assert not np.allclose(conn_stacked_modified.data, conn_stacked_ref.data)

    print("✓ Modifications to dict are preserved after stacking")
    print(f"  Dict keys: {list(conn_dict.keys())}")
    print(f"  Stacked shape: {conn_stacked.shape}")


def test_stack_hetersynapse_collapse_neuron_mode():
    """Test collapsing behavior in neuron mode for 'pre', 'post', and 'all'.

    Uses make_hetersynapse_conn(return_dict=True) to generate receptor-
    pair matrices. Then verifies the output shapes when collapsing
    dimensions.
    """
    from btorch.connectome.connection import stack_hetersynapse

    neurons = create_test_neurons(n_neurons=30)
    connections = create_test_connections(neurons, density=0.2)

    conn_dict, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
        return_dict=True,
    )

    n = len(neurons)
    # Unique receptor types per dimension
    n_pre_types = receptor_idx["pre_receptor_type"].nunique()
    n_post_types = receptor_idx["post_receptor_type"].nunique()

    # No collapse: expect N x (N * num_pairs)
    conn_full = stack_hetersynapse(conn_dict, receptor_idx)
    assert conn_full.shape == (n, n * len(receptor_idx))

    # Collapse post: expect N x (N * n_pre_types)
    conn_post = stack_hetersynapse(conn_dict, receptor_idx, ignore_receptor_type="post")
    assert conn_post[0].shape == (n, n * n_pre_types)

    # Collapse pre: expect N x (N * n_post_types)
    conn_pre = stack_hetersynapse(conn_dict, receptor_idx, ignore_receptor_type="pre")
    assert conn_pre[0].shape == (n, n * n_post_types)

    # Collapse all: expect N x N
    conn_all = stack_hetersynapse(conn_dict, receptor_idx, ignore_receptor_type="all")
    assert conn_all[0].shape == (n, n)

    print("✓ Neuron-mode collapse shapes verified")


def test_stack_hetersynapse_connection_mode_stack_and_collapse():
    """Test stacking and collapse behavior in connection mode.

    In connection mode, receptor types are properties of connections.
    Collapsing with 'pre'/'post' should behave like 'all'.
    """
    from btorch.connectome.connection import stack_hetersynapse

    neurons = create_test_neurons(n_neurons=25)
    connections = create_test_connections(neurons, density=0.25)

    # Add receptor type per connection for connection mode
    conn_df = connections.copy()
    conn_df["EI"] = np.random.choice(["E", "I"], size=len(conn_df))

    # Dict format in connection mode (keys are receptor_type strings)
    conn_dict, receptor_idx = make_hetersynapse_conn(
        neurons,
        conn_df,
        receptor_type_col="EI",
        receptor_type_mode="connection",
        return_dict=True,
    )

    n = len(neurons)
    n_types = len(receptor_idx)

    # No collapse: expect N x (N * n_types)
    conn_full = stack_hetersynapse(conn_dict, receptor_idx)
    assert conn_full.shape == (n, n * n_types)

    # 'pre' behaves like 'all' in connection mode: N x N
    conn_pre = stack_hetersynapse(conn_dict, receptor_idx, ignore_receptor_type="pre")
    assert conn_pre[0].shape == (n, n)

    # 'post' behaves like 'all' in connection mode: N x N
    conn_post = stack_hetersynapse(conn_dict, receptor_idx, ignore_receptor_type="post")
    assert conn_post[0].shape == (n, n)

    # 'all': N x N
    conn_all = stack_hetersynapse(conn_dict, receptor_idx, ignore_receptor_type="all")
    assert conn_all[0].shape == (n, n)

    print("✓ Connection-mode stacking and collapse verified")


def test_stack_hetersynapse_neuron_mode_values():
    """Verify data correctness (not just shapes) for neuron-mode collapse."""
    from btorch.connectome.connection import stack_hetersynapse

    n = 3
    # Build deterministic receptor pair matrices
    mat_ax = scipy.sparse.coo_array(([1], ([0], [1])), shape=(n, n))
    mat_ay = scipy.sparse.coo_array(([3], ([0], [2])), shape=(n, n))
    mat_bx = scipy.sparse.coo_array(([2], ([1], [0])), shape=(n, n))
    mat_by = scipy.sparse.coo_array(([5], ([2], [1])), shape=(n, n))

    conn_dict = OrderedDict(
        [
            (("A", "X"), mat_ax),
            (("A", "Y"), mat_ay),
            (("B", "X"), mat_bx),
            (("B", "Y"), mat_by),
        ]
    )

    receptor_idx = pd.DataFrame(
        [
            (0, "A", "X"),
            (1, "A", "Y"),
            (2, "B", "X"),
            (3, "B", "Y"),
        ],
        columns=["receptor_index", "pre_receptor_type", "post_receptor_type"],
    )

    # No collapse
    conn_full = stack_hetersynapse(conn_dict, receptor_idx).tocoo()
    dense_full = conn_full.toarray()
    # Check a few exact positions
    assert dense_full[0, 1 * 4 + 0] == 1  # AX at col=1
    assert dense_full[0, 2 * 4 + 1] == 3  # AY at col=2
    assert dense_full[1, 0 * 4 + 2] == 2  # BX at col=0
    assert dense_full[2, 1 * 4 + 3] == 5  # BY at col=1

    # Collapse post types -> keep pre types (A, B)
    conn_post, idx_post = stack_hetersynapse(
        conn_dict, receptor_idx, ignore_receptor_type="post"
    )
    dense_post = conn_post.toarray()
    # Expected columns per target neuron multiplied by n_pre_types=2
    # A channel (idx 0): AX + AY -> entries at (0, 2) and (0, 4)
    assert dense_post[0, 2] == 1
    assert dense_post[0, 4] == 3
    # B channel (idx 1): BX + BY -> entries at (1, 1) and (2, 3)
    assert dense_post[1, 1] == 2
    assert dense_post[2, 3] == 5
    assert list(idx_post["receptor_type"]) == ["A", "B"]

    # Collapse pre types -> keep post types (X, Y)
    conn_pre, idx_pre = stack_hetersynapse(
        conn_dict, receptor_idx, ignore_receptor_type="pre"
    )
    dense_pre = conn_pre.toarray()
    # X channel (idx 0): AX + BX -> entries at (0, 2) and (1, 0)
    assert dense_pre[0, 2] == 1
    assert dense_pre[1, 0] == 2
    # Y channel (idx 1): AY + BY -> entries at (0, 5) and (2, 3)
    assert dense_pre[0, 5] == 3
    assert dense_pre[2, 3] == 5
    assert list(idx_pre["receptor_type"]) == ["X", "Y"]

    # Collapse all -> sum everything
    conn_all, idx_all = stack_hetersynapse(
        conn_dict, receptor_idx, ignore_receptor_type="all"
    )
    dense_all = conn_all.toarray()
    expected_all = mat_ax + mat_ay + mat_bx + mat_by
    np.testing.assert_array_equal(dense_all, expected_all.toarray())
    assert list(idx_all["receptor_type"]) == ["all"]

    print("✓ Neuron-mode collapse values verified (data, not just shape)")


def test_stack_hetersynapse_connection_mode_values():
    """Verify data correctness for connection mode and collapse."""
    from btorch.connectome.connection import stack_hetersynapse

    n = 2
    mat_e = scipy.sparse.coo_array(([1], ([0], [1])), shape=(n, n))
    mat_i = scipy.sparse.coo_array(([4], ([1], [0])), shape=(n, n))

    conn_dict = OrderedDict([("E", mat_e), ("I", mat_i)])

    receptor_idx = pd.DataFrame(
        [(0, "E"), (1, "I")], columns=["receptor_index", "receptor_type"]
    )

    # No collapse: expect two channels stacked
    conn_full = stack_hetersynapse(conn_dict, receptor_idx).tocoo()
    dense_full = conn_full.toarray()
    # Channel E (idx 0) col = original_col * 2 + 0
    assert dense_full[0, 1 * 2 + 0] == 1
    # Channel I (idx 1) col = original_col * 2 + 1
    assert dense_full[1, 0 * 2 + 1] == 4

    # Collapse with 'pre' behaves like 'all' in connection mode
    conn_pre, idx_pre = stack_hetersynapse(
        conn_dict, receptor_idx, ignore_receptor_type="pre"
    )
    dense_pre = conn_pre.toarray()
    expected_all = (mat_e + mat_i).toarray()
    np.testing.assert_array_equal(dense_pre, expected_all)
    assert list(idx_pre["receptor_type"]) == ["all"]

    # Collapse with 'post' also behaves like 'all'
    conn_post, idx_post = stack_hetersynapse(
        conn_dict, receptor_idx, ignore_receptor_type="post"
    )
    np.testing.assert_array_equal(conn_post.toarray(), expected_all)
    assert list(idx_post["receptor_type"]) == ["all"]

    print("✓ Connection-mode collapse values verified (data, not just shape)")


def test_hetersynapse_psc_with_max_delay_steps_matches_manual():
    """Test HeterSynapsePSC with max_delay_steps matches manual SpikeHistory.

    This verifies that the built-in delay buffering in HeterSynapsePSC
    produces the same result as a manual SpikeHistory + get_flattened
    pipeline when using a delay-expanded connection matrix.
    """
    from btorch.connectome.connection import expand_conn_for_delays
    from btorch.models.linear import SparseConn

    neurons = create_test_neurons(n_neurons=20)
    connections = create_test_connections(neurons, density=0.2)

    # Create hetersynapse connection first
    conn_sp, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
    )

    n_receptor = len(receptor_idx)
    n_neurons = len(neurons)
    n_delay_bins = 3

    # Generate synthetic delays for each non-zero connection
    rng = np.random.default_rng(42)
    delays = rng.integers(0, n_delay_bins, size=conn_sp.nnz)

    # Expand connection matrix for delays (adds virtual pre-neuron rows)
    conn_d = expand_conn_for_delays(
        conn_sp,
        delays=delays,
        n_delay_bins=n_delay_bins,
    )
    linear = SparseConn(conn_d, enforce_dale=False)

    with environ.context(dt=1.0):
        hetero_psc = HeterSynapsePSC(
            n_neuron=n_neurons,
            n_receptor=n_receptor,
            receptor_type_index=receptor_idx,
            linear=linear,
            base_psc=ExponentialPSC,
            tau_syn=2.0,
            max_delay_steps=n_delay_bins,
            use_circular_buffer=False,
        )
        init_net_state(hetero_psc, batch_size=1, dtype=torch.float32)

        # Manual pipeline with separate SpikeHistory
        manual_history = SpikeHistory(
            n_neuron=n_neurons,
            max_delay_steps=n_delay_bins,
            use_circular_buffer=False,
        )
        manual_history.init_state(batch_size=1, dtype=torch.float32)
        manual_base = ExponentialPSC(
            n_neuron=n_neurons * n_receptor,
            tau_syn=2.0,
            linear=linear,
        )
        init_net_state(manual_base, batch_size=1, dtype=torch.float32)

        z_seq = torch.randn(5, 1, n_neurons)

        for t in range(z_seq.shape[0]):
            out_hetero = hetero_psc.single_step_forward(z_seq[t])

            manual_history.update(z_seq[t])
            z_delayed = manual_history.get_flattened(n_delay_bins)
            out_manual = manual_base.single_step_forward(z_delayed)
            out_manual = out_manual.view(
                *out_manual.shape[:-1], n_neurons, n_receptor
            ).sum(-1)

            torch.testing.assert_close(out_hetero, out_manual, atol=1e-6, rtol=0.0)


def test_hetersynapse_psc_delay_state_init_reset():
    """Test state init and reset for HeterSynapsePSC with delays.

    Verifies both circular buffer and cat buffer modes.
    """
    neurons = create_test_neurons(n_neurons=15)
    connections = create_test_connections(neurons, density=0.2)

    conn_sp, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
    )

    n_receptor = len(receptor_idx)
    n_neurons = len(neurons)
    n_delay_bins = 4
    # Use a linear layer sized for delay-expanded inputs
    linear = torch.nn.Linear(
        n_neurons * n_delay_bins, n_neurons * n_receptor, bias=False
    )

    for use_circular in (True, False):
        with environ.context(dt=1.0):
            hetero_psc = HeterSynapsePSC(
                n_neuron=n_neurons,
                n_receptor=n_receptor,
                receptor_type_index=receptor_idx,
                linear=linear,
                base_psc=AlphaPSC,
                tau_syn=2.0,
                max_delay_steps=n_delay_bins,
                use_circular_buffer=use_circular,
            )
            init_net_state(hetero_psc, batch_size=2, dtype=torch.float32)

            assert hetero_psc.history is not None
            assert hetero_psc.history.history.shape == (2, 4, n_neurons)

            z = torch.ones(2, n_neurons)
            hetero_psc.single_step_forward(z)

            # History should be non-zero after update
            assert hetero_psc.history.history.abs().sum() > 0

            reset_net(hetero_psc, batch_size=2)

            # After reset, history should be zeroed
            assert hetero_psc.history.history.abs().sum() == 0
            assert hetero_psc.psc.abs().sum() == 0


def test_hetersynapse_psc_no_delay_when_max_delay_steps_one():
    """Test HeterSynapsePSC without delays (max_delay_steps=1)."""
    neurons = create_test_neurons(n_neurons=10)
    connections = create_test_connections(neurons, density=0.2)

    conn_sp, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
    )

    n_receptor = len(receptor_idx)
    n_neurons = len(neurons)
    linear = torch.nn.Linear(n_neurons, n_neurons * n_receptor, bias=False)

    with environ.context(dt=1.0):
        hetero_psc = HeterSynapsePSC(
            n_neuron=n_neurons,
            n_receptor=n_receptor,
            receptor_type_index=receptor_idx,
            linear=linear,
            base_psc=AlphaPSC,
            tau_syn=2.0,
            max_delay_steps=1,
        )
        init_net_state(hetero_psc, batch_size=1, dtype=torch.float32)

        assert hetero_psc.history is None

        z = torch.randn(1, n_neurons)
        out = hetero_psc.single_step_forward(z)
        assert out.shape == (1, n_neurons)
