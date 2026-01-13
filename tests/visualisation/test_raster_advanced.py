"""Tests for advanced raster features."""

import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import save_fig
from btorch.visualisation.timeseries import NeuronSpec, plot_raster


def test_raster_events_regions_tracks():
    """Test events, regions and tracks."""
    n_neurons = 20
    n_time = 200
    spikes = np.random.rand(n_time, n_neurons) > 0.95

    events = [50, 150]
    regions = [(80, 120)]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_raster(
        spikes,
        ax=ax,
        events=events,
        regions=regions,
        show_tracks=True,
        title="Raster with Events, Regions, Tracks",
    )
    save_fig(fig, name="raster_advanced_features", suffix="png", transparent=False)
    plt.close(fig)


def test_raster_mixed_markers():
    """Test mixed markers via NeuronSpec."""
    n_neurons = 10
    n_time = 100
    spikes = np.random.rand(n_time, n_neurons) > 0.9

    # First 5: red circle, Next 5: blue x
    specs = []
    for i in range(n_neurons):
        if i < 5:
            specs.append(NeuronSpec(color="red", marker="o", markersize=10))
        else:
            specs.append(NeuronSpec(color="blue", marker="x", markersize=15))

    fig, ax = plt.subplots()
    plot_raster(spikes, ax=ax, neuron_specs=specs, title="Mixed Markers")
    save_fig(fig, name="raster_mixed_markers", suffix="png", transparent=False)
    plt.close(fig)


def test_large_population_raster():
    """Test raster with large population (1000 neurons)."""
    n_neurons = 2000
    n_time = 500
    # Sparse spikes
    spikes = np.random.rand(n_time, n_neurons) > 0.999

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_raster(
        spikes,
        ax=ax,
        marker="|",
        markersize=2.0,  # Small marker for density
        color="black",
        title="Large Population Raster (2000 Neurons)",
        show_tracks=False,  # Tracks might be too dense
    )

    # Check if it rendered without error
    save_fig(fig, name="raster_large_population", suffix="png", transparent=False)
    plt.close(fig)
