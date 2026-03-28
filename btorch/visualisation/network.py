"""Network graph visualization utilities."""

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_network(sparse_mat, ax: Axes | None = None) -> Figure:
    """Plot a network graph from a sparse connectivity matrix.

    Uses NetworkX spring layout to visualize the graph structure.
    Nodes are colored skyblue, edges are gray.

    Args:
        sparse_mat: Sparse matrix (scipy.sparse) representing connections.
            Non-zero entries indicate edges.
        ax: Existing axes to plot on. If None, creates new figure.

    Returns:
        Figure containing the network plot.

    Raises:
        ImportError: If networkx is not installed.

    Example:
        >>> from scipy.sparse import random
        >>> mat = random(50, 50, density=0.1, format="csr")
        >>> fig = plot_network(mat)
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.from_scipy_sparse_array(sparse_mat)
    pos = nx.spring_layout(G)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        ax=ax,
    )
    ax.set_title("Network Graph")
    return fig
