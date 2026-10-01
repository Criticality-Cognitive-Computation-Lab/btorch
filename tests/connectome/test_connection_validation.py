"""Input validation and id-mapping behaviour of ``btorch.connectome``.

These tests use only pandas/scipy, so they are fast and need no torch.
"""

import numpy as np
import pandas as pd
import pytest

from btorch.connectome import simple_id_to_root_id
from btorch.connectome.connection import neuron_subset_to_conn_mat


@pytest.fixture
def neurons():
    # Three neurons; simple ids are contiguous row indices, root ids are large.
    return pd.DataFrame({"root_id": [1000, 2000, 3000], "simple_id": [0, 1, 2]})


def test_simple_id_to_root_id_direction(neurons):
    # Default: simple_id -> root_id (as the function name reads).
    assert simple_id_to_root_id(neurons) == {0: 1000, 1: 2000, 2: 3000}
    # reverse=True: the opposite lookup, root_id -> simple_id.
    assert simple_id_to_root_id(neurons, reverse=True) == {
        1000: 0,
        2000: 1,
        3000: 2,
    }


def test_root_id_subset_maps_to_simple_id(neurons):
    out = neuron_subset_to_conn_mat([3000, 1000], "root_id", 3, neurons=neurons)
    np.testing.assert_array_equal(out, [2, 0])


def test_root_id_subset_without_neurons_raises_value_error():
    # Previously this dereferenced ``None`` and raised AttributeError.
    with pytest.raises(ValueError, match="neurons"):
        neuron_subset_to_conn_mat([1000], "root_id", 3, neurons=None)


def test_dataframe_subset_without_root_id_raises_value_error(neurons):
    with pytest.raises(ValueError, match="root_id"):
        neuron_subset_to_conn_mat(
            pd.DataFrame({"x": [1]}), "root_id", 3, neurons=neurons
        )


def test_unmapped_ids_raise_or_are_dropped(neurons):
    subset = [1000, 9999]  # 9999 is not in the neuron table
    with pytest.raises(ValueError, match="unknown root_id"):
        neuron_subset_to_conn_mat(subset, "root_id", 3, neurons=neurons)
    with pytest.warns(UserWarning, match="removing"):
        out = neuron_subset_to_conn_mat(
            subset, "root_id", 3, neurons=neurons, remove_nan=True
        )
    np.testing.assert_array_equal(out, [0])


def test_sparse_input_rejects_connection_receptor_mode():
    """Sparse-array input only supports per-neuron receptor types.

    Passing ``receptor_type_mode="connection"`` must raise ``ValueError`` (an
    explicit check, not a bare ``assert``).
    """
    import scipy.sparse

    from btorch.connectome.connection import make_hetersynapse_conn

    mat = scipy.sparse.coo_array(np.ones((2, 2)))
    with pytest.raises(ValueError, match="receptor_type_mode='neuron'"):
        make_hetersynapse_conn(
            pd.DataFrame({"simple_id": [0, 1]}), mat, receptor_type_mode="connection"
        )


def test_dataframe_subset_without_simple_id_is_mapped(neurons):
    """A subset DataFrame holding only ``root_id`` gets ``simple_id`` looked up
    from the neuron table."""
    out = neuron_subset_to_conn_mat(
        pd.DataFrame({"root_id": [2000, 3000]}), "root_id", 3, neurons=neurons
    )
    np.testing.assert_array_equal(out, [1, 2])
