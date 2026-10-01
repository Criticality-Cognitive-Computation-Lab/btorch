"""Basic tests for the public timeseries plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from btorch.utils.file import save_fig
from btorch.visualisation.timeseries import (
    plot_log_hist,
    plot_spectrum,
    plot_traces,
)


def test_plot_traces():
    # Seeded (time, neurons) data so line contents are reproducible.
    data = np.random.default_rng(0).standard_normal((100, 5))
    ax = plot_traces(data)

    assert isinstance(ax, Axes)
    # One line per neuron, each with one sample per time step.
    assert len(ax.lines) == 5
    for i, line in enumerate(ax.lines):
        y = np.asarray(line.get_ydata())
        assert y.shape == (100,)
        assert np.all(np.isfinite(y))
        np.testing.assert_allclose(y, data[:, i])

    save_fig(ax.get_figure(), name="traces")
    plt.close(ax.get_figure())


def test_plot_spectrum():
    # White noise, 1 kHz sampling when dt = 1 ms.
    data = np.random.default_rng(0).standard_normal((1000, 1))
    freqs, power, ax = plot_spectrum(data, dt=1.0)

    assert isinstance(ax, Axes)
    freqs = np.asarray(freqs)
    power = np.asarray(power)
    assert freqs.shape[0] == power.shape[0]
    assert np.all(np.isfinite(power))
    # Power is non-negative and frequencies are sorted and non-negative.
    assert np.all(power >= 0)
    assert np.all(freqs >= 0)
    assert np.all(np.diff(freqs) > 0)
    assert len(ax.lines) >= 1

    save_fig(ax.get_figure(), name="spectrum")
    plt.close(ax.get_figure())


def test_plot_log_hist():
    # Log-normal samples (strictly positive) as required for log binning.
    vals = np.exp(np.random.default_rng(0).standard_normal(1000))
    ax = plot_log_hist(vals)

    assert isinstance(ax, Axes)
    # Log-log scatter of bin counts versus bin centres.
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert ax.get_ylabel() == "Count"
    assert ax.get_title() == "Distribution"
    assert len(ax.collections) == 1
    offsets = np.asarray(ax.collections[0].get_offsets())
    assert offsets.shape[1] == 2 and offsets.shape[0] > 0
    assert np.all(np.isfinite(offsets))
    # Bin centres lie within the data range (log-binned, strictly positive).
    assert offsets[:, 0].min() > 0
    assert offsets[:, 0].max() <= vals.max() * 1.01

    save_fig(ax.get_figure(), name="log_hist")
    plt.close(ax.get_figure())
