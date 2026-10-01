"""Characterisation tests for every branch of ``make_hetersynapse_conn``.

Each test compares the function output against a tiny brute-force reference
written in this file (``_expected_dense``), so the tests double as an executable
specification of the column / row layout:

* **neuron mode**: receptor index of an edge ``pre -> post`` is
  ``type_idx[pre_type] * K + type_idx[post_type]`` (``K`` = number of receptor
  types, sorted); with ``ignore_post_type`` it is ``type_idx[pre_type]``.
* **connection mode**: receptor index is the position of the connection's own
  receptor type in the sorted list of types.
* Column of an edge is ``post * n_receptor + receptor_index``; row is ``pre``
  (or ``pre * n_delay_bins + clip(delay)`` when delays are expanded).
"""

import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from btorch.connectome.connection import make_hetersynapse_conn


N = 4  # number of neurons


@pytest.fixture
def neurons():
    # Neurons 0, 1 are excitatory, 2, 3 inhibitory.
    return pd.DataFrame(
        {"simple_id": range(N), "EI": ["E", "E", "I", "I"]},
    )


@pytest.fixture
def connections():
    # Edges listed deliberately NOT sorted by pre id nor by receptor type, and
    # with a duplicated (pre, post) pair (two neuropils) so that aggregation and
    # row/entry ordering are exercised.  ``rt`` is the per-connection receptor.
    return pd.DataFrame(
        {
            "pre_simple_id": [3, 0, 2, 1, 0, 0],
            "post_simple_id": [0, 1, 1, 3, 3, 1],
            "syn_count": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "rt": ["b", "a", "b", "a", "b", "a"],
            "delay": [4, 1, 0, 2, 3, 1],
        }
    )


def _expected_dense(
    neurons,
    connections,
    mode="neuron",
    ignore_post_type=False,
    n_delay_bins=1,
    rt_col="EI",
):
    """Brute-force reference producing the stacked dense matrix."""
    if mode == "neuron":
        types = sorted(neurons[rt_col].unique())
        k = len(types)
        n_rec = k if ignore_post_type else k * k
        nt = dict(zip(neurons["simple_id"], neurons[rt_col]))
    else:
        types = sorted(connections["rt"].unique())
        n_rec = len(types)
    out = np.zeros((N * n_delay_bins, N * n_rec))
    has_delay = n_delay_bins > 1
    # Aggregate duplicated entries per (pre, post[, receptor]).
    keys = ["pre_simple_id", "post_simple_id"] + (
        ["rt"] if mode == "connection" else []
    )
    agg = {"syn_count": "sum"}
    if has_delay:
        agg["delay"] = "mean"
    for key, row in connections.groupby(keys).agg(agg).iterrows():
        pre, post = key[0], key[1]
        if mode == "neuron":
            r = types.index(nt[pre])
            if not ignore_post_type:
                r = r * len(types) + types.index(nt[post])
        else:
            r = types.index(key[2])
        d = min(int(row["delay"]), n_delay_bins - 1) if has_delay else 0
        out[pre * n_delay_bins + d, post * n_rec + r] += row["syn_count"]
    return out


# ---------------------------------------------------------------- no delays


@pytest.mark.parametrize("ignore_post_type", [False, True])
def test_neuron_mode_dataframe_stacked(neurons, connections, ignore_post_type):
    conn, idx = make_hetersynapse_conn(
        neurons, connections, "EI", "neuron", ignore_post_type=ignore_post_type
    )
    assert isinstance(conn, scipy.sparse.sparray)
    np.testing.assert_allclose(
        conn.toarray(),
        _expected_dense(neurons, connections, ignore_post_type=ignore_post_type),
    )
    if ignore_post_type:
        assert list(idx.columns) == ["receptor_index", "receptor_type"]
        assert idx["receptor_type"].tolist() == ["E", "I"]
    else:
        assert list(idx.columns) == [
            "receptor_index",
            "pre_receptor_type",
            "post_receptor_type",
        ]
        assert list(zip(idx.pre_receptor_type, idx.post_receptor_type)) == [
            ("E", "E"),
            ("E", "I"),
            ("I", "E"),
            ("I", "I"),
        ]
    assert idx["receptor_index"].tolist() == list(range(len(idx)))


def test_neuron_mode_sparse_input_matches_dataframe_input(neurons, connections):
    from btorch.connectome.connection import make_sparse_mat

    sp = make_sparse_mat(connections, (N, N))
    a, idx_a = make_hetersynapse_conn(neurons, sp, "EI", "neuron")
    b, idx_b = make_hetersynapse_conn(neurons, connections, "EI", "neuron")
    np.testing.assert_allclose(a.toarray(), b.toarray())
    pd.testing.assert_frame_equal(idx_a, idx_b)


@pytest.mark.parametrize("ignore_post_type", [False, True])
def test_neuron_mode_return_dict(neurons, connections, ignore_post_type):
    d, idx = make_hetersynapse_conn(
        neurons,
        connections,
        "EI",
        "neuron",
        return_dict=True,
        ignore_post_type=ignore_post_type,
    )
    assert isinstance(d, OrderedDict)
    # Every dict entry is a full (N, N) matrix, empty receptor groups are
    # omitted, and the key layout follows the index frame.
    assert all(v.shape == (N, N) for v in d.values())
    total = sum(v.toarray() for v in d.values())
    np.testing.assert_allclose(total, _agg(connections))
    if ignore_post_type:
        assert set(d) <= {"E", "I"}
    else:
        assert set(d) <= {("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")}
    assert len(idx) == (2 if ignore_post_type else 4)


def _agg(connections):
    """Dense (N, N) matrix of summed syn_count per (pre, post)."""
    g = connections.groupby(["pre_simple_id", "post_simple_id"]).syn_count.sum()
    out = np.zeros((N, N))
    for (pre, post), w in g.items():
        out[pre, post] = w
    return out


def test_neuron_mode_return_dict_matches_stacked(neurons, connections):
    d, _ = make_hetersynapse_conn(
        neurons, connections, "EI", "neuron", return_dict=True
    )
    stacked, idx = make_hetersynapse_conn(neurons, connections, "EI", "neuron")
    dense = stacked.toarray()
    for r, (pre_t, post_t) in enumerate(
        zip(idx.pre_receptor_type, idx.post_receptor_type)
    ):
        if (pre_t, post_t) in d:
            np.testing.assert_allclose(d[(pre_t, post_t)].toarray(), dense[:, r::4])
        else:
            assert not dense[:, r::4].any()


def test_connection_mode_stacked(neurons, connections):
    conn, idx = make_hetersynapse_conn(neurons, connections, "rt", "connection")
    np.testing.assert_allclose(
        conn.toarray(), _expected_dense(neurons, connections, "connection")
    )
    assert list(idx.columns) == ["receptor_index", "receptor_type"]
    assert idx["receptor_type"].tolist() == ["a", "b"]


def test_connection_mode_return_dict(neurons, connections):
    d, idx = make_hetersynapse_conn(
        neurons, connections, "rt", "connection", return_dict=True
    )
    assert isinstance(d, OrderedDict)
    assert list(d) == ["a", "b"]
    # Duplicate (0, 1, 'a') rows are summed: 2 + 6.
    assert d["a"].toarray()[0, 1] == 8.0
    assert idx["receptor_type"].tolist() == ["a", "b"]


def test_connection_mode_dict_matches_stacked(neurons, connections):
    d, _ = make_hetersynapse_conn(
        neurons, connections, "rt", "connection", return_dict=True
    )
    stacked, _ = make_hetersynapse_conn(neurons, connections, "rt", "connection")
    dense = stacked.toarray()
    for i, key in enumerate(d):
        np.testing.assert_allclose(d[key].toarray(), dense[:, i::2])


# ------------------------------------------------------------------- delays


@pytest.mark.parametrize("ignore_post_type", [False, True])
def test_neuron_mode_with_delays_stacked(neurons, connections, ignore_post_type):
    conn, _ = make_hetersynapse_conn(
        neurons,
        connections,
        "EI",
        "neuron",
        ignore_post_type=ignore_post_type,
        delay_col="delay",
        n_delay_bins=5,
    )
    np.testing.assert_allclose(
        conn.toarray(),
        _expected_dense(
            neurons, connections, ignore_post_type=ignore_post_type, n_delay_bins=5
        ),
    )


def test_connection_mode_with_delays_stacked(neurons, connections):
    conn, _ = make_hetersynapse_conn(
        neurons, connections, "rt", "connection", delay_col="delay", n_delay_bins=5
    )
    np.testing.assert_allclose(
        conn.toarray(),
        _expected_dense(neurons, connections, "connection", n_delay_bins=5),
    )


def test_neuron_mode_with_delays_return_dict(neurons, connections):
    d, idx = make_hetersynapse_conn(
        neurons,
        connections,
        "EI",
        "neuron",
        return_dict=True,
        delay_col="delay",
        n_delay_bins=5,
    )
    assert isinstance(d, OrderedDict)
    assert all(v.shape == (N * 5, N) for v in d.values())
    stacked, _ = make_hetersynapse_conn(
        neurons, connections, "EI", "neuron", delay_col="delay", n_delay_bins=5
    )
    dense = stacked.toarray()
    for r, key in enumerate(zip(idx.pre_receptor_type, idx.post_receptor_type)):
        expected = dense[:, r::4]
        got = d[key].toarray() if key in d else np.zeros_like(expected)
        np.testing.assert_allclose(got, expected)


def test_connection_mode_with_delays_return_dict(neurons, connections):
    d, _ = make_hetersynapse_conn(
        neurons,
        connections,
        "rt",
        "connection",
        return_dict=True,
        delay_col="delay",
        n_delay_bins=5,
    )
    stacked, _ = make_hetersynapse_conn(
        neurons, connections, "rt", "connection", delay_col="delay", n_delay_bins=5
    )
    dense = stacked.toarray()
    for i, key in enumerate(d):
        assert d[key].shape == (N * 5, N)
        np.testing.assert_allclose(d[key].toarray(), dense[:, i::2])


def test_delay_clipped_to_last_bin(neurons, connections):
    conn, _ = make_hetersynapse_conn(
        neurons, connections, "EI", "neuron", delay_col="delay", n_delay_bins=3
    )
    # Edge 3 -> 0 has delay 4, which is clipped to bin 2: row 3 * 3 + 2.
    assert conn.toarray()[3 * 3 + 2].sum() == 1.0


def test_single_delay_bin_means_no_expansion(neurons, connections):
    with_col, _ = make_hetersynapse_conn(
        neurons, connections, "EI", "neuron", delay_col="delay", n_delay_bins=1
    )
    without, _ = make_hetersynapse_conn(neurons, connections, "EI", "neuron")
    assert with_col.shape == without.shape == (N, N * 4)
    np.testing.assert_allclose(with_col.toarray(), without.toarray())


# ------------------------------------------------------ receptor_type_col=None


def test_delays_only_stacked(neurons, connections):
    conn, idx = make_hetersynapse_conn(
        neurons, connections, None, delay_col="delay", n_delay_bins=5
    )
    dense = conn.toarray()
    assert dense.shape == (N * 5, N)
    # (0 -> 1): counts 2 and 6 summed, delays 1 and 1 averaged.
    assert dense[0 * 5 + 1, 1] == 8.0
    assert dense[3 * 5 + 4, 0] == 1.0
    assert conn.nnz == 5
    assert idx.columns.tolist() == ["simple_id", "delay_index"]


def test_delays_only_single_bin_returns_plain_matrix(neurons, connections):
    conn, idx = make_hetersynapse_conn(
        neurons, connections, None, delay_col="delay", n_delay_bins=1
    )
    assert conn.shape == (N, N)
    assert idx.columns.tolist() == ["simple_id"]


def test_delays_only_return_dict_is_rejected(neurons, connections):
    # There are no receptor types to key a dict by, so asking for one is an error
    # rather than silently returning a sparse array.
    with pytest.raises(ValueError, match="return_dict"):
        make_hetersynapse_conn(
            neurons, connections, None, return_dict=True, delay_col="delay"
        )


# --------------------------------------------------------------- validation


def test_delay_col_requires_dataframe(neurons):
    sp = scipy.sparse.coo_array(np.eye(N))
    with pytest.raises(ValueError, match="delay_col"):
        make_hetersynapse_conn(neurons, sp, "EI", delay_col="delay")


def test_receptor_none_requires_dataframe_and_delay(neurons, connections):
    sp = scipy.sparse.coo_array(np.eye(N))
    with pytest.raises(ValueError, match="DataFrame"):
        make_hetersynapse_conn(neurons, sp, None, delay_col="d")
    with pytest.raises(ValueError, match="receptor_type_col or delay_col"):
        make_hetersynapse_conn(neurons, connections, None)


def test_missing_delay_column_raises_key_error(neurons, connections):
    with pytest.raises(KeyError):
        make_hetersynapse_conn(neurons, connections, "EI", delay_col="nope")


def test_invalid_connections_type(neurons):
    with pytest.raises(ValueError, match="DataFrame or a scipy"):
        make_hetersynapse_conn(neurons, np.eye(N), "EI")


def test_sparse_nan_data_raises(neurons):
    sp = scipy.sparse.coo_array(([np.nan], ([0], [1])), shape=(N, N))
    with pytest.raises(ValueError, match="NaN"):
        make_hetersynapse_conn(neurons, sp, "EI")


def test_invalid_dropna_mode(neurons, connections):
    neurons = neurons.copy()
    neurons.loc[0, "EI"] = np.nan
    with pytest.raises(ValueError, match="dropna"):
        make_hetersynapse_conn(neurons, connections, "EI", dropna="bogus")


def test_nan_filter_and_unknown_with_delays(neurons, connections):
    neurons = neurons.copy()
    neurons.loc[3, "EI"] = np.nan
    with pytest.warns(UserWarning, match="Filtered"):
        conn, _ = make_hetersynapse_conn(
            neurons,
            connections,
            "EI",
            dropna="filter",
            delay_col="delay",
            n_delay_bins=5,
        )
    # Edges touching neuron 3 are dropped; the rest keep their delay rows.
    kept = connections[
        (connections.pre_simple_id != 3) & (connections.post_simple_id != 3)
    ]
    assert conn.shape == (N * 5, N * 4)
    assert conn.toarray().sum() == pytest.approx(
        kept.groupby(["pre_simple_id", "post_simple_id"]).syn_count.sum().sum()
    )
    with pytest.warns(UserWarning, match="unknown"):
        conn_u, idx_u = make_hetersynapse_conn(
            neurons,
            connections,
            "EI",
            dropna="unknown",
            delay_col="delay",
            n_delay_bins=5,
        )
    assert set(idx_u.pre_receptor_type) == {"E", "I", "unknown"}
    assert conn_u.toarray().sum() == pytest.approx(
        connections.groupby(["pre_simple_id", "post_simple_id"]).syn_count.sum().sum()
    )


def test_empty_connections_neuron_mode(neurons):
    empty = pd.DataFrame(
        {
            "pre_simple_id": pd.Series([], dtype=int),
            "post_simple_id": pd.Series([], dtype=int),
            "syn_count": pd.Series([], dtype=float),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conn, idx = make_hetersynapse_conn(neurons, empty, "EI", "neuron")
    assert conn.shape == (N, N * 4)
    assert conn.nnz == 0
    assert len(idx) == 4
