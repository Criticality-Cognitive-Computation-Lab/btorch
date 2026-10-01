"""Log-binned histogram plot."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

from ...analysis.statistics import compute_log_hist
from ...utils.array import to_numpy


def plot_log_hist(
    values: np.ndarray | torch.Tensor,
    ax: Axes | None = None,
    title: str = "Distribution",
    xlabel: str = "Value",
    **kwargs: Any,
) -> Axes:
    """Plot log-log histogram with logarithmic binning.

    Creates a scatter plot of histogram counts using logarithmically
    spaced bins. Useful for visualizing heavy-tailed distributions
    (e.g., power laws).

    Args:
        values: Input values to histogram. Flattened if multidimensional.
        ax: Existing axes to plot on. Creates new figure if None.
        title: Plot title.
        xlabel: X-axis label.
        **kwargs: Additional arguments passed to ax.scatter().

    Returns:
        Axes containing the log-log histogram.

    Example:
        >>> ax = plot_log_hist(synapse_weights, title="Weight Distribution")
    """
    vals = to_numpy(values)
    hist, bin_centers = compute_log_hist(vals)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(bin_centers, hist, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)

    return ax
