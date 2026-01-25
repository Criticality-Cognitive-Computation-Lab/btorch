import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import save_fig
from btorch.visualisation.timeseries import (
    plot_log_hist,
    plot_spectrum,
    plot_traces,
)


def test_plot_traces():
    data = np.random.randn(100, 5)
    ax = plot_traces(data)
    save_fig(ax.get_figure(), name="traces")
    plt.close(ax.get_figure())


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
