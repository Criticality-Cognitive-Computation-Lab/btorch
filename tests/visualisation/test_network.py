"""Smoke test for the network graph plot."""

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse  # noqa: E402

from btorch.visualisation.network import plot_network  # noqa: E402


def test_plot_network_uses_given_axes():
    """plot_network draws on the provided axes and returns its figure."""
    pytest.importorskip("networkx")
    mat = scipy.sparse.csr_array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    fig, ax = plt.subplots()
    out = plot_network(mat, ax=ax)
    assert out is fig
    assert ax.get_title() == "Network Graph"
    plt.close("all")
