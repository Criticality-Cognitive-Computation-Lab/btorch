"""Tests for raster plotting, traces, and spectrum visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from btorch.utils.file import save_fig
from btorch.visualisation.timeseries import (
    NeuronSpec,
    plot_raster,
)


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
        marker_size=2.0,  # Small marker for density
        spike_color="black",
        title="Large Population Raster (2000 Neurons)",
        show_tracks=False,  # Tracks might be too dense
    )

    # Check if it rendered without error
    save_fig(fig, name="raster_large_population", suffix="png", transparent=False)
    plt.close(fig)


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
        group_key="group",
        group_sort=["A", "B", "C"],
        title="Grouped Raster (A, B, C)",
        spike_color={"A": "red", "B": "green", "C": "blue"},
        show_group_separators=True,
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


def test_raster_dt_times():
    """Test dt and times parameters."""
    n_neurons = 10
    n_time = 50
    spikes = np.random.rand(n_time, n_neurons) > 0.9

    # Test with dt
    fig, ax = plt.subplots()
    plot_raster(spikes, dt=0.5, ax=ax, title="Raster with dt=0.5")
    save_fig(fig, name="raster_dt", suffix="png", transparent=False)
    plt.close(fig)

    # Test with explicit times
    times = np.linspace(0, 10, n_time)
    fig, ax = plt.subplots()
    plot_raster(spikes, times=times, ax=ax, title="Raster with explicit times")
    save_fig(fig, name="raster_times", suffix="png", transparent=False)
    plt.close(fig)


def test_raster_show_rate():
    """Test show_rate functionality."""
    n_neurons = 20
    n_time = 100
    spikes = np.random.rand(n_time, n_neurons) > 0.9
    rate_series = np.linspace(0.0, 20.0, n_time)
    groups = ["A"] * 10 + ["B"] * 10
    df = pd.DataFrame({"group": groups})

    axes = plot_raster(
        spikes,
        dt=0.1,
        rate=rate_series,
        group_rate=True,
        neurons_df=df,
        group_key="group",
        rate_window_ms=2.0,
        events=[20, 60],
        regions=[(10, 30)],
        title="Raster with Rate",
    )
    assert len(axes) == 2
    fig = axes[0].figure
    save_fig(fig, name="raster_with_rate_dt", suffix="png", transparent=False)
    plt.close(fig)


def test_raster_events_dict():
    """Test events as dict."""
    n_neurons = 10
    n_time = 100
    spikes = np.random.rand(n_time, n_neurons) > 0.9

    events = {"stimulus": [25, 75], "response": [50]}

    fig, ax = plt.subplots()
    plot_raster(spikes, ax=ax, events=events, title="Raster with Dict Events")
    save_fig(fig, name="raster_events_dict", suffix="png", transparent=False)
    plt.close(fig)


def test_raster_regions_dict():
    """Test regions as dict."""
    n_neurons = 10
    n_time = 100
    spikes = np.random.rand(n_time, n_neurons) > 0.9

    regions = {"baseline": [(0, 20)], "stimulus": [(40, 60), (80, 100)]}

    fig, ax = plt.subplots()
    plot_raster(spikes, ax=ax, regions=regions, title="Raster with Dict Regions")
    save_fig(fig, name="raster_regions_dict", suffix="png", transparent=False)
    plt.close(fig)


def test_raster_event_region_kwargs():
    """Test event_kwargs and region_kwargs."""
    n_neurons = 10
    n_time = 100
    spikes = np.random.rand(n_time, n_neurons) > 0.9

    events = [30, 70]
    regions = [(10, 40)]

    fig, ax = plt.subplots()
    plot_raster(
        spikes,
        ax=ax,
        events=events,
        regions=regions,
        event_kwargs={"color": "purple", "linewidth": 2},
        region_kwargs={"color": "green", "alpha": 0.5},
        title="Raster with Custom Event/Region Styling",
    )
    save_fig(fig, name="raster_custom_styling", suffix="png", transparent=False)
    plt.close(fig)


def test_raster_color_dict_no_group():
    """Test color dict without group_key (should warn)."""
    n_neurons = 10
    n_time = 50
    spikes = np.random.rand(n_time, n_neurons) > 0.9

    with pytest.warns(
        UserWarning, match="spike_color dict provided but group_key not set"
    ):
        fig, ax = plt.subplots()
        plot_raster(
            spikes,
            ax=ax,
            spike_color={"group1": "red"},
            title="Raster Color Dict Warn",
        )
        save_fig(fig, name="raster_color_dict_warn", suffix="png", transparent=False)
        plt.close(fig)


def test_raster_separator_kwargs():
    """Test separator_style."""
    n_neurons = 20
    n_time = 100
    spikes = np.random.rand(n_time, n_neurons) > 0.9

    groups = ["A"] * 10 + ["B"] * 10
    df = pd.DataFrame({"group": groups})

    fig, ax = plt.subplots()
    plot_raster(
        spikes,
        ax=ax,
        neurons_df=df,
        group_key="group",
        show_group_separators=True,
        separator_style={"color": "red", "linewidth": 2, "linestyle": "-"},
        title="Raster with Custom Separators",
    )
    save_fig(fig, name="raster_custom_separators", suffix="png", transparent=False)
    plt.close(fig)


def test_raster_torch_tensor():
    """Test raster with torch tensor input."""
    n_neurons = 10
    n_time = 50
    spikes_np = np.random.rand(n_time, n_neurons) > 0.9
    spikes_torch = torch.tensor(spikes_np)

    fig, ax = plt.subplots()
    plot_raster(spikes_torch, ax=ax, title="Raster with Torch Tensor")
    save_fig(fig, name="raster_torch", suffix="png", transparent=False)
    plt.close(fig)


def test_raster_group_strip_small_groups():
    """100-neuron network with 5 groups and group strip."""
    n_neurons = 100
    n_time = 120
    spikes = np.random.rand(n_time, n_neurons) > 0.95

    n_groups = 5
    group_labels = [f"G{i}" for i in range(n_groups) for _ in range(n_neurons // 5)]
    while len(group_labels) < n_neurons:
        group_labels.append("G4")
    df = pd.DataFrame({"group": group_labels})

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_raster(
        spikes,
        ax=ax,
        neurons_df=df,
        group_key="group",
        show_group_strip=True,
        group_color_key="group",
        strip_cmap="tab10",
        group_label_mode="top",
        title="Raster Group Strip (5 Groups)",
    )

    assert len(fig.axes) >= 2
    cax = fig.axes[-1]
    legend = cax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    for i in range(n_groups):
        assert f"G{i}" in labels
    save_fig(fig, name="raster_group_strip_small", suffix="png", transparent=False)
    plt.close(fig)


@pytest.mark.parametrize("side", ["right", "left"])
def test_raster_group_strip_large_many_subgroups(side):
    """Large 2000-neuron raster with 10 top groups and 100 subgroups."""
    n_neurons = 2000
    n_time = 500
    spikes = np.random.rand(n_time, n_neurons) > 0.999

    n_top = 10
    n_sub = 100
    per_sub = n_neurons // n_sub

    top_list = []
    sub_list = []
    for s in range(n_sub):
        tg = f"G{s // (n_sub // n_top)}"
        sub_name = f"s{s:03d}"
        for _ in range(per_sub):
            top_list.append(tg)
            sub_list.append(sub_name)

    # pad if needed
    while len(top_list) < n_neurons:
        top_list.append(f"G{n_top-1}")
        sub_list.append(f"s{n_sub-1:03d}")

    df = pd.DataFrame({"group": top_list, "sub": sub_list})

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_raster(
        spikes,
        ax=ax,
        neurons_df=df,
        group_key="group",
        show_group_strip=True,
        group_color_key="sub",
        strip_cmap="tab20",
        group_label_mode="top",
        group_strip_side=side,
        title="Large Raster Many Subgroups",
    )

    # Expect legend to show top groups
    cax = fig.axes[-1]
    legend = cax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    # should contain the 10 top groups
    for i in range(n_top):
        assert f"G{i}" in labels

    save_fig(
        fig, name=f"raster_large_subgroups_{side}", suffix="png", transparent=False
    )
    plt.close(fig)
