"""Tests for delay expansion in connection matrices."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from btorch.connectome.connection import (
    expand_conn_for_delays,
    make_hetersynapse_conn,
)


class TestExpandConnForDelays:
    """Test suite for delay expansion of connection matrices."""

    def test_delay_expansion_shape(self):
        """Test that delay expansion produces correct matrix shape."""
        # Create simple 3-neuron connection
        conn = scipy.sparse.coo_array(
            ([1.0, 2.0, 3.0], ([0, 1, 2], [1, 2, 0])), shape=(3, 3)
        )

        # Assign delays to each connection
        delays = np.array([0, 2, 4])  # 3 connections
        n_delay_bins = 5

        conn_d = expand_conn_for_delays(conn, delays, n_delay_bins)

        # Shape should be (n_neurons * n_delays, n_neurons)
        assert conn_d.shape == (3 * 5, 3)
        assert conn_d.shape == (15, 3)

    def test_delay_expansion_mapping(self):
        """Test that each connection maps to correct delay-expanded row."""
        # Single connection: neuron 0 -> neuron 1 with weight 5.0
        conn = scipy.sparse.coo_array(([5.0], ([0], [1])), shape=(3, 3))

        # Delay = 2
        delays = np.array([2])
        conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)

        # Convert to dense for easier checking
        dense = conn_d.toarray()

        # Connection should be at row 0*5 + 2 = 2, column 1
        assert dense[2, 1] == 5.0

        # All other entries in column 1 should be 0
        for r in range(15):
            if r != 2:
                assert dense[r, 1] == 0.0

    def test_delay_expansion_preserves_weights(self):
        """Test that all connection weights are preserved correctly."""
        conn = scipy.sparse.coo_array(
            ([1.0, 2.0, 3.0, 4.0], ([0, 0, 1, 2], [1, 2, 0, 1])), shape=(3, 3)
        )

        delays = np.array([0, 1, 2, 3])
        conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)

        dense = conn_d.toarray()

        # Verify each weight is in correct position
        # conn[0,1]=1.0 with delay 0 -> row 0*5+0=0, col 1
        assert dense[0, 1] == 1.0
        # conn[0,2]=2.0 with delay 1 -> row 0*5+1=1, col 2
        assert dense[1, 2] == 2.0
        # conn[1,0]=3.0 with delay 2 -> row 1*5+2=7, col 0
        assert dense[7, 0] == 3.0
        # conn[2,1]=4.0 with delay 3 -> row 2*5+3=13, col 1
        assert dense[13, 1] == 4.0

    def test_empty_delay_bins(self):
        """Test expansion when some delay bins have no connections."""
        conn = scipy.sparse.coo_array(([1.0, 2.0], ([0, 1], [1, 2])), shape=(3, 3))

        # Both connections have delay=0
        delays = np.array([0, 0])
        conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)

        dense = conn_d.toarray()

        # Only rows 0*5+0=0 and 1*5+0=5 should have values
        assert dense[0, 1] == 1.0
        assert dense[5, 2] == 2.0

        # All other rows should be zero
        non_zero_rows = [0, 5]
        for r in range(15):
            if r not in non_zero_rows:
                assert np.all(dense[r, :] == 0)

    def test_delay_clipping(self):
        """Test that delays outside range are clipped to valid bins."""
        conn = scipy.sparse.coo_array(([1.0, 2.0], ([0, 1], [1, 2])), shape=(3, 3))

        # Delays exceed n_delay_bins - should be clipped
        delays = np.array([10, 100])  # Will be clipped to 4 (max valid bin)
        conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)

        dense = conn_d.toarray()

        # Both should be clipped to delay 4
        assert dense[0 * 5 + 4, 1] == 1.0  # row 4
        assert dense[1 * 5 + 4, 2] == 2.0  # row 9

    def test_negative_delay_raises(self):
        """Test that negative delays raise ValueError."""
        conn = scipy.sparse.coo_array(([1.0], ([0], [1])), shape=(3, 3))

        delays = np.array([-1])

        with pytest.raises(ValueError, match="delays must be non-negative"):
            expand_conn_for_delays(conn, delays, n_delay_bins=5)

    def test_single_delay_bin(self):
        """Test expansion with n_delay_bins=1 (no expansion)."""
        conn = scipy.sparse.coo_array(([1.0, 2.0], ([0, 1], [1, 2])), shape=(3, 3))

        delays = np.array([0, 0])
        conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=1)

        # Shape should be (3*1, 3) = (3, 3)
        assert conn_d.shape == (3, 3)

        # Should be equivalent to original (modulo delay which is 0)
        dense = conn_d.toarray()
        assert dense[0, 1] == 1.0
        assert dense[1, 2] == 2.0

    def test_no_connections(self):
        """Test expansion with empty connection matrix."""
        conn = scipy.sparse.coo_array((3, 3))

        delays = np.array([], dtype=int)
        conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)

        assert conn_d.shape == (15, 3)
        assert conn_d.nnz == 0

    def test_large_matrix(self):
        """Test expansion with larger matrix (performance check)."""
        n = 100
        density = 0.1
        n_conns = int(n * n * density)

        rows = np.random.randint(0, n, n_conns)
        cols = np.random.randint(0, n, n_conns)
        data = np.random.randn(n_conns)

        conn = scipy.sparse.coo_array((data, (rows, cols)), shape=(n, n))
        delays = np.random.randint(0, 5, n_conns)

        conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)

        assert conn_d.shape == (n * 5, n)
        assert conn_d.nnz == n_conns


class TestMakeDelayedHeteroConn:
    """Test combined delay + heterosynapse connection creation."""

    def _create_test_data(self):
        """Create test neurons and connections DataFrames."""
        # 3 neurons: 2 excitatory, 1 inhibitory
        neurons = pd.DataFrame(
            {
                "simple_id": [0, 1, 2],
                "root_id": [100, 101, 102],
                "EI": ["E", "E", "I"],
            }
        )

        # Connections:
        # 0 (E) -> 1 (E), delay=1
        # 0 (E) -> 2 (I), delay=2
        # 1 (E) -> 2 (I), delay=0
        # 2 (I) -> 0 (E), delay=3
        connections = pd.DataFrame(
            {
                "pre_simple_id": [0, 0, 1, 2],
                "post_simple_id": [1, 2, 2, 0],
                "pre_root_id": [100, 100, 101, 102],
                "post_root_id": [101, 102, 102, 100],
                "syn_count": [1, 1, 1, 1],
                "delay_steps": [1, 2, 0, 3],
            }
        )

        return neurons, connections

    def test_delays_only(self):
        """Test creating connection with delays but no heterosynapse."""
        neurons, connections = self._create_test_data()

        conn, idx = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col=None,  # No heterosynapse
            delay_col="delay_steps",
            n_delay_bins=5,
        )

        # Shape: (n_neurons * n_delays, n_neurons) = (15, 3)
        assert conn.shape == (15, 3)

        # Verify index describes columns (one per neuron)
        assert len(idx) == 3  # One row per neuron

    def test_combined_delay_and_hetero(self):
        """Test creating connection with both delays and heterosynapse."""
        neurons, connections = self._create_test_data()

        conn, idx = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            delay_col="delay_steps",
            n_delay_bins=5,
        )

        # Shape: (n_neurons * n_delays, n_neurons * n_receptors)
        # n_receptors = 4 (E->E, E->I, I->E, I->I)
        assert conn.shape == (3 * 5, 3 * 4)
        assert conn.shape == (15, 12)

    def test_combined_shape_verification(self):
        """Verify specific connections map correctly in combined matrix."""
        neurons, connections = self._create_test_data()

        conn, idx = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            delay_col="delay_steps",
            n_delay_bins=5,
        )

        dense = conn.toarray()

        # Connection 0(E)->1(E) with delay=1
        # In hetero matrix: post 1, pre E, post E -> receptor index for E->E
        # Column = 1 * 4 + 0 = 4 (assuming E->E is index 0)
        # Row = 0 * 5 + 1 = 1 (delay=1)
        assert dense[1, 4] == 1.0, "Expected connection 0(E)->1(E) at row 1, col 4"

        # Connection 0(E)->2(I) with delay=2
        # Column = 2 * 4 + 1 = 9 (E->I is index 1)
        # Row = 0 * 5 + 2 = 2 (delay=2)
        assert dense[2, 9] == 1.0, "Expected connection 0(E)->2(I) at row 2, col 9"

        # Connection 1(E)->2(I) with delay=0
        # Column = 2 * 4 + 1 = 9 (E->I is index 1)
        # Row = 1 * 5 + 0 = 5 (delay=0)
        assert dense[5, 9] == 1.0, "Expected connection 1(E)->2(I) at row 5, col 9"

        # Connection 2(I)->0(E) with delay=3
        # Column = 0 * 4 + 2 = 2 (I->E is index 2)
        # Row = 2 * 5 + 3 = 13 (delay=3)
        assert dense[13, 2] == 1.0, "Expected connection 2(I)->0(E) at row 13, col 2"

        # Verify total non-zero count matches input connections
        assert conn.nnz == len(connections)

    def test_column_index_matches_heterosynapse(self):
        """Test that column structure matches heterosynapse pattern."""
        neurons, connections = self._create_test_data()

        # First create heterosynapse only
        from btorch.connectome.connection import make_hetersynapse_conn

        conn_hetero, idx_hetero = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col="EI",
            receptor_type_mode="neuron",
        )

        # Then create with delays
        conn_delayed, idx_delayed = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            delay_col="delay_steps",
            n_delay_bins=5,
        )

        # The column dimension should be the same (heterosynapse structure unchanged)
        assert conn_delayed.shape[1] == conn_hetero.shape[1]

        # Column index should match heterosynapse index exactly
        pd.testing.assert_frame_equal(idx_delayed, idx_hetero)

    def test_invalid_delay_column(self):
        """Test error when delay column doesn't exist."""
        neurons, connections = self._create_test_data()

        with pytest.raises(KeyError):
            make_hetersynapse_conn(
                neurons,
                connections,
                receptor_type_col="receptor_type",
                receptor_type_mode="connection",
                delay_col="invalid_delay_col",
                n_delay_bins=5,
            )

    def test_empty_connections(self):
        """Test with empty connections DataFrame."""
        neurons, _ = self._create_test_data()
        empty_connections = pd.DataFrame(
            {
                "pre_simple_id": [],
                "post_simple_id": [],
                "pre_root_id": [],
                "post_root_id": [],
                "syn_count": [],
                "delay_steps": [],
            }
        )

        conn, idx = make_hetersynapse_conn(
            neurons,
            empty_connections,
            delay_col="delay_steps",
            n_delay_bins=5,
            receptor_type_col=None,
        )

        assert conn.shape == (15, 3)
        assert conn.nnz == 0


class TestDelayExpansionRoundTrip:
    """Test that expanded matrices produce correct matmul results."""

    def test_simple_matmul_with_history(self):
        """Test matrix multiply simulates correct delay behavior."""
        import torch

        from btorch.models.history import SpikeHistory

        # Create simple 2-neuron network:
        # Neuron 0 -> Neuron 1 with weight 2.0, delay=2
        conn = scipy.sparse.coo_array(([2.0], ([0], [1])), shape=(2, 2))
        delays = np.array([2])

        # Expand for delays (use 3 bins to match history)
        n_delays = 3
        conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=n_delays)
        # conn_d shape: (2*3, 2) = (6, 2)

        # Create history with specific pattern
        history = SpikeHistory(n_neuron=2, max_delay_steps=n_delays)
        history.init_state(batch_size=1, dtype=torch.float32)

        # t=0: spike at neuron 0
        history.update(torch.tensor([[1.0, 0.0]]))
        # t=1: no spikes
        history.update(torch.tensor([[0.0, 0.0]]))
        # t=2: no spikes
        history.update(torch.tensor([[0.0, 0.0]]))
        # t=3: current (no spike yet)

        # At t=3, delay=2 should pick up spike from t=0
        # Flattened format: [n0_d0, n0_d1, n0_d2, n1_d0, n1_d1, n1_d2]
        # = [0, 0, 1, 0, 0, 0] (n0 had spike 2 timesteps ago)
        z_flat = history.get_flattened(n_delays)  # shape: (1, 6)

        # Convert sparse to torch (use float32 to match default torch dtype)
        indices = torch.tensor(np.vstack([conn_d.row, conn_d.col]))
        values = torch.tensor(conn_d.data, dtype=torch.float32)
        torch_sparse = torch.sparse_coo_tensor(indices, values, conn_d.shape)
        # torch_sparse shape: (6, 2) = (n_neurons * n_delays, n_neurons)

        # Matrix multiply: y = x @ W
        # x (z_flat): (batch, n_neurons * n_delays) = (1, 6)
        # W (torch_sparse): (n_neurons * n_delays, n_neurons) = (6, 2)
        # y: (batch, n_neurons) = (1, 2)
        output = torch.sparse.mm(z_flat, torch_sparse)

        # Output should be [0, 2.0] (neuron 1 receives weight*spike)
        assert output[0, 0].item() == 0.0
        assert output[0, 1].item() == 2.0
