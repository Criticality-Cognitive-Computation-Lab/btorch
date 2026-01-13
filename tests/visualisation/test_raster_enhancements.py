"""Tests for enhanced raster plotting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btorch.utils.file import save_fig
from btorch.visualisation.timeseries import plot_raster


def test_raster_grouping():
    """Test grouping in raster plot."""
    n_neurons = 20
    n_time = 100
    spikes = np.random.rand(n_time, n_neurons) > 0.9

    # Create metadata
    # 5 neurons in group A, 10 in B, 5 in C
    groups = ["A"] * 5 + ["B"] * 10 + ["C"] * 5
    # Shuffle indices to test reordering
    indices = np.arange(n_neurons)
    np.random.shuffle(indices)
    groups = [groups[i] for i in indices]

    df = pd.DataFrame({"group": groups})

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_raster(
        spikes,
        ax=ax,
        neurons_df=df,
        group_by="group",
        group_order=["A", "B", "C"],
        title="Grouped Raster (A, B, C)",
        color={"A": "red", "B": "green", "C": "blue"},
        show_separators=True,
    )

    save_fig(fig, name="raster_grouped", suffix="png", transparent=False)
    plt.close(fig)


def test_raster_styling_per_neuron():
    """Test individual neuron styling."""
    n_neurons = 10
    n_time = 50
    spikes = np.random.rand(n_time, n_neurons) > 0.8

    specs = [
        {"color": "red"} if i % 2 == 0 else {"color": "blue"} for i in range(n_neurons)
    ]

    fig, ax = plt.subplots()
    plot_raster(
        spikes, ax=ax, neuron_specs=specs, title="Styled Raster (Alternating Colors)"
    )
    save_fig(fig, name="raster_styled", suffix="png", transparent=False)
    plt.close(fig)


def test_raster_pipe_marker():
    """Test raster with pipe marker and integer axis."""
    n_neurons = 10
    n_time = 100
    spikes = np.random.rand(n_time, n_neurons) > 0.9

    fig, ax = plt.subplots()
    plot_raster(
        spikes,
        ax=ax,
        marker="|",
        markersize=10.0,
        title="Raster with Pipe Marker",
    )
    save_fig(fig, name="raster_pipe", suffix="png", transparent=False)
    plt.close(fig)
