import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import save_fig
from btorch.visualisation.timeseries import (
    plot_log_hist,
    plot_raster,
    plot_spectrum,
    plot_traces,
)


def test_plot_traces():
    data = np.random.randn(100, 5)
    ax = plot_traces(data)
    save_fig(ax.get_figure(), name="traces")
    plt.close(ax.get_figure())


def test_plot_raster():
    spikes = (np.random.rand(100, 10) > 0.9).astype(float)
    ax = plot_raster(spikes)
    if isinstance(ax, tuple):
        fig = ax[0].get_figure()
    else:
        fig = ax.get_figure()
    save_fig(fig, name="raster")
    plt.close(fig)

    # with rate
    ax_tup = plot_raster(spikes, show_rate=True)
    save_fig(ax_tup[0].get_figure(), name="raster_with_rate")
    plt.close(ax_tup[0].get_figure())


def test_plot_spectrum():
    data = np.random.randn(1000, 1)
    freqs, power, ax = plot_spectrum(data, dt=1.0)
    save_fig(ax.get_figure(), name="spectrum")
    plt.close(ax.get_figure())


def test_plot_log_hist():
    vals = np.exp(np.random.randn(1000))
    ax = plot_log_hist(vals)
    save_fig(ax.get_figure(), name="log_hist")
    plt.close(ax.get_figure())
