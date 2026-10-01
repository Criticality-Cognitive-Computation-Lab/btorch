"""Continuous timeseries trace plot."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

from ...utils.array import to_numpy
from ._helpers import (
    _get_time_axis,
)


def plot_traces(
    data: np.ndarray | torch.Tensor,
    dt: float | None = None,
    times: Sequence[float] | None = None,
    ax: Axes | None = None,
    neurons: Sequence[int] | int | None = None,
    labels: Sequence[str] | str | None = None,
    colors: Sequence[Any] | None = None,
    title: str | None = None,
    xlabel: str = "Time (ms)",
    ylabel: str | None = None,
    legend: bool = True,
    alpha: float = 0.8,
) -> Axes:
    """Plot continuous timeseries traces.

    Parameters
    ----------
    data : array-like
        Shape (Time, Neurons) or (Time, Neurons, Features).
    dt : float, optional
        Time step.
    times : array-like, optional
        Explicit time array.
    ax : Axes, optional
        Axis to plot on.
    neurons : list of int or int, optional
        Indices of neurons to plot. If None, plots all (careful with large N).
        If int, samples that many neurons randomly.
    labels : list of str, optional
        Labels for the legend.
    colors : list of colors, optional
        Colors for traces.
    title : str, optional
        Plot title.

    Returns
    -------
    Axes
    """
    data_np = to_numpy(data)
    t = _get_time_axis(data_np.shape[0], dt, times)

    if data_np.ndim == 2:
        data_np = data_np[:, :, np.newaxis]
    elif data_np.ndim != 3:
        raise ValueError("Data must be 2D (T, N) or 3D (T, N, F)")

    n_neurons = data_np.shape[1]
    if neurons is None:
        neuron_indices = np.arange(n_neurons)
    elif isinstance(neurons, int):
        if neurons >= n_neurons:
            neuron_indices = np.arange(n_neurons)
        else:
            neuron_indices = np.sort(
                np.random.choice(n_neurons, neurons, replace=False)
            )
    else:
        neuron_indices = np.array(neurons)

    n_features = data_np.shape[2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if colors is None:
        cmap = plt.get_cmap("turbo", len(neuron_indices))
        colors = [cmap(i) for i in range(len(neuron_indices))]

    for i, idx in enumerate(neuron_indices):
        c = (
            colors[i]
            if isinstance(colors, (list, np.ndarray))
            and len(colors) == len(neuron_indices)
            else None
        )

        for feat in range(n_features):
            trace = data_np[:, idx, feat]

            lbl = None
            if labels is not None:
                if isinstance(labels, str):
                    lbl = f"{labels} {idx}"
                elif len(labels) == len(neuron_indices):
                    lbl = labels[i]
                else:
                    lbl = f"Neuron {idx}"
            else:
                lbl = f"Neuron {idx}"

            if n_features > 1:
                lbl += f" (f{feat})"

            ax.plot(t, trace, label=lbl, color=c, alpha=alpha)

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlim(t[0], t[-1])

    if title:
        ax.set_title(title)

    if legend and len(neuron_indices) <= 20:  # Limit legend clutter
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    return ax
