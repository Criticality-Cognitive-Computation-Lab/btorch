"""Tests for :mod:`btorch.analysis.connectivity`."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from btorch.analysis.connectivity import HopDistanceModel, compute_ie_ratio


def _chain_edges() -> pd.DataFrame:
    # Directed chain a -> b -> c -> d plus a shortcut a -> c.
    return pd.DataFrame(
        {
            "source": ["a", "b", "c", "a"],
            "target": ["b", "c", "d", "c"],
        }
    )


def test_compute_ie_ratio_simple():
    # Neuron 0 receives 2 E and 1 I input -> ratio 0.5; neuron 1 gets 1 E + 3 I.
    exc = sparse.csr_array(np.array([[0, 0], [2, 1]]))
    inh = sparse.csr_array(np.array([[0, 3], [1, 0]]))
    neurons = pd.DataFrame({"simple_id": [0, 1], "EI": ["I", "I"]})
    whole, ratios = compute_ie_ratio(
        exc, inh, excitatory_neuron_only=False, neurons=neurons, warn_strict=False
    )
    np.testing.assert_allclose(ratios, [0.5, 3.0])
    assert whole == pytest.approx(1.75)


def test_hop_distance_edges_and_sparse_agree():
    # Edge-list backend: shortest distance to c is 1 thanks to the shortcut.
    model = HopDistanceModel(edges=_chain_edges())
    df = model.compute_distances(["a"])
    dist = dict(zip(df["node"], df["distance"]))
    assert dist == {"a": 0, "b": 1, "c": 1, "d": 2}

    # Sparse backend on the same graph (a=0, b=1, c=2, d=3).
    adj = sparse.csr_array(
        (np.ones(4), ([0, 1, 2, 0], [1, 2, 3, 2])),
        shape=(4, 4),
    )
    df_s = HopDistanceModel(adjacency=adj).compute_distances([0])
    assert dict(zip(df_s["node"], df_s["distance"])) == {0: 0, 1: 1, 2: 1, 3: 2}


def test_hop_distance_max_hops_and_missing_seed():
    model = HopDistanceModel(edges=_chain_edges())
    df = model.compute_distances(["a", "zzz"], max_hops=1)  # unknown seed is ignored
    assert set(df["node"]) == {"a", "b", "c"}


def test_hop_statistics_cumulative():
    stats = HopDistanceModel(edges=_chain_edges()).hop_statistics(["a"])
    assert list(stats["hops"]) == [0, 1, 2]
    assert list(stats["nodes_count"]) == [1, 2, 1]
    assert stats["cumulative_percentage"].iloc[-1] == pytest.approx(100.0)


def test_reconstruct_path_and_unreachable():
    model = HopDistanceModel(edges=_chain_edges())
    assert model.reconstruct_path("a", "d") == ["a", "c", "d"]
    # d has no outgoing edges, so a is unreachable from d.
    assert model.reconstruct_path("d", "a") == []


def test_requires_edges_or_adjacency():
    with pytest.raises(ValueError):
        HopDistanceModel()
