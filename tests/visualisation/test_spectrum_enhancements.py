"""Tests for grouped spectrum plotting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btorch.utils.file import save_fig
from btorch.visualisation.timeseries import (
    SpectrumGrouping,
    SpectrumStyle,
    plot_grouped_spectrum,
)


def generate_data():
    """Generate synthetic oscillatory data."""
    n_time = 1000
    n_neurons = 30
    dt = 1.0
    t = np.arange(n_time) * dt

    data = np.zeros((n_time, n_neurons))

    # Group A: 10 Hz
    for i in range(10):
        data[:, i] = np.sin(2 * np.pi * 10 * t / 1000) + np.random.randn(n_time) * 0.5

    # Group B: 40 Hz
    for i in range(10, 20):
        data[:, i] = np.sin(2 * np.pi * 40 * t / 1000) + np.random.randn(n_time) * 0.5

    # Group C: Noise
    for i in range(20, 30):
        data[:, i] = np.random.randn(n_time)

    df = pd.DataFrame({"type": ["A"] * 10 + ["B"] * 10 + ["C"] * 10})

    return data, df, dt


def test_grouped_spectrum_overlay():
    """Test overlay mode."""
    data, df, dt = generate_data()

    fig = plot_grouped_spectrum(
        data,
        dt=dt,
        grouping=SpectrumGrouping(neurons_df=df, group_by="type"),
        mode="overlay",
        title="Grouped Spectrum (Overlay)",
        style=SpectrumStyle(show_traces=True),
    )

    save_fig(fig, name="spectrum_grouped_overlay", suffix="png", transparent=False)
    plt.close(fig)


def test_grouped_spectrum_subplots():
    """Test subplots mode."""
    data, df, dt = generate_data()

    fig = plot_grouped_spectrum(
        data,
        dt=dt,
        grouping=SpectrumGrouping(neurons_df=df, group_by="type"),
        mode="subplots",
        title="Grouped Spectrum (Subplots)",
    )

    save_fig(fig, name="spectrum_grouped_subplots", suffix="png", transparent=False)
    plt.close(fig)


def test_grouped_spectrum_separate():
    """Test separate figures mode."""
    data, df, dt = generate_data()

    figs = plot_grouped_spectrum(
        data,
        dt=dt,
        grouping=SpectrumGrouping(neurons_df=df, group_by="type"),
        separate_figures=True,
    )

    assert isinstance(figs, dict)
    assert len(figs) == 3

    for name, fig in figs.items():
        save_fig(
            fig, name=f"spectrum_grouped_sep_{name}", suffix="png", transparent=False
        )
        plt.close(fig)


def test_grouped_spectrum_manual_groups():
    """Test manual groups dict."""
    data, _, dt = generate_data()

    groups = {"Low Freq": list(range(10)), "High Freq": list(range(10, 20))}

    fig = plot_grouped_spectrum(
        data,
        dt=dt,
        grouping=SpectrumGrouping(groups=groups),
        mode="overlay",
        title="Manual Groups",
    )

    save_fig(fig, name="spectrum_grouped_manual", suffix="png", transparent=False)
    plt.close(fig)


def test_grouped_spectrum_options_change_output():
    """SpectrumGrouping / SpectrumStyle must actually alter the figure."""
    data, df, dt = generate_data()
    grouping = SpectrumGrouping(neurons_df=df, group_by="type")

    # Default style: 3 groups, each with faint per-neuron traces plus a mean
    # line, so the single axes holds many lines.
    fig_default = plot_grouped_spectrum(data, dt=dt, grouping=grouping)
    n_default = len(fig_default.axes[0].lines)

    # show_traces=False drops the per-neuron lines, keeping only means.
    fig_means = plot_grouped_spectrum(
        data, dt=dt, grouping=grouping, style=SpectrumStyle(show_traces=False)
    )
    ax = fig_means.axes[0]
    assert len(ax.lines) < n_default

    # Manual groups override the metadata column: one group -> one legend entry.
    fig_manual = plot_grouped_spectrum(
        data, dt=dt, grouping=SpectrumGrouping(groups={"only": [0, 1, 2]})
    )
    labels = [t.get_text() for t in fig_manual.axes[0].get_legend().get_texts()]
    assert labels == ["only"]

    # Custom colours and panel size are honoured.
    fig_style = plot_grouped_spectrum(
        data,
        dt=dt,
        grouping=SpectrumGrouping(groups={"g": [0, 1]}),
        style=SpectrumStyle(colors={"g": "red"}, plot_width=3.0, plot_height=2.0),
    )
    assert tuple(fig_style.get_size_inches()) == (3.0, 2.0)
    mean_line = fig_style.axes[0].lines[-1]
    assert mean_line.get_color() == "red"
    plt.close("all")
