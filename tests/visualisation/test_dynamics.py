"""Tests for multiscale dynamics visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from btorch.utils.file import save_fig
from btorch.visualisation.dynamics import (
    DFAConfig,
    DynamicsData,
    DynamicsPlotFormat,
    FanoFactorConfig,
    plot_avalanche_analysis,
    plot_dfa_analysis,
    plot_eigenvalue_spectrum,
    plot_firing_rate_distribution,
    plot_gain_stability,
    plot_isi_cv,
    plot_lyapunov_spectrum,
    plot_micro_dynamics,
    plot_multiscale_fano,
)


def generate_spike_data(n_time=5000, n_neurons=50, rate=0.05):
    """Generate synthetic spike data."""
    spikes = (np.random.rand(n_time, n_neurons) < rate).astype(float)
    return spikes


def test_plot_multiscale_fano_plain_args():
    """Test Fano factor plotting with plain arguments."""
    spikes = generate_spike_data()

    fig = plot_multiscale_fano(
        spikes=spikes, dt=0.1, windows=[10, 50, 100, 500], mode="individual"
    )

    save_fig(fig, name="multiscale_fano_plain_args")
    plt.close(fig)


def test_plot_multiscale_fano_dataclass():
    """Test Fano factor plotting with dataclass interface."""
    spikes = generate_spike_data()

    data = DynamicsData(spikes=spikes, dt=0.1)
    config = FanoFactorConfig(windows=[10, 50, 100, 500, 1000], overlap=5)
    format = DynamicsPlotFormat(mode="individual")

    fig = plot_multiscale_fano(data=data, config=config, format=format)

    save_fig(fig, name="multiscale_fano_dataclass")
    plt.close(fig)


def test_plot_multiscale_fano_distribution():
    """Test Fano factor distribution mode."""
    spikes = generate_spike_data()

    fig = plot_multiscale_fano(spikes=spikes, dt=0.1, mode="distribution")

    save_fig(fig, name="multiscale_fano_distribution")
    plt.close(fig)


def test_plot_multiscale_fano_grouped_by_neuron_type():
    """Test Fano factor grouped by neuron type."""
    spikes = generate_spike_data(n_neurons=30)

    # Create neuron metadata
    neurons_df = pd.DataFrame(
        {
            "simple_id": range(30),
            "cell_type": [f"Type_{i%3}" for i in range(30)],
        }
    )

    data = DynamicsData(spikes=spikes, dt=0.1, neurons_df=neurons_df)
    format = DynamicsPlotFormat(mode="grouped", group_by="neuron_type")

    fig = plot_multiscale_fano(data=data, format=format)

    save_fig(fig, name="multiscale_fano_grouped_neuron_type")
    plt.close(fig)


def test_plot_multiscale_fano_grouped_by_neuropil():
    """Test Fano factor grouped by neuropil."""
    n_neurons = 30
    spikes = generate_spike_data(n_neurons=n_neurons)

    # Create neuron and connection metadata
    neurons_df = pd.DataFrame(
        {
            "simple_id": range(n_neurons),
            "group": [f"neuropil_{i%3}.neuropil_{(i+1)%3}" for i in range(n_neurons)],
        }
    )

    connections_df = pd.DataFrame(
        {
            "pre_simple_id": np.random.choice(n_neurons, 100),
            "post_simple_id": np.random.choice(n_neurons, 100),
            "neuropil": [f"neuropil_{i%3}" for i in range(100)],
        }
    )

    data = DynamicsData(
        spikes=spikes, dt=0.1, neurons_df=neurons_df, connections_df=connections_df
    )
    format = DynamicsPlotFormat(mode="grouped", group_by="neuropil")

    fig = plot_multiscale_fano(data=data, format=format)

    save_fig(fig, name="multiscale_fano_grouped_neuropil")
    plt.close(fig)


def test_plot_dfa_analysis_plain_args():
    """Test DFA analysis plotting with plain arguments."""
    spikes = generate_spike_data()

    fig = plot_dfa_analysis(spikes=spikes, dt=0.1)

    save_fig(fig, name="dfa_analysis_plain_args")
    plt.close(fig)


def test_plot_dfa_analysis_dataclass():
    """Test DFA analysis plotting with dataclass interface."""
    spikes = generate_spike_data()

    data = DynamicsData(spikes=spikes, dt=0.1)
    config = DFAConfig(min_window=4, max_window=100, bin_size=1)

    fig = plot_dfa_analysis(data=data, config=config)

    save_fig(fig, name="dfa_analysis_dataclass")
    plt.close(fig)


def test_plot_isi_cv_distribution():
    """Test ISI CV distribution plotting."""
    spikes = generate_spike_data()

    fig = plot_isi_cv(spikes=spikes, dt=0.1, mode="distribution")

    save_fig(fig, name="isi_cv_distribution")
    plt.close(fig)


def test_plot_isi_cv_grouped():
    """Test ISI CV grouped by neuron type."""
    spikes = generate_spike_data(n_neurons=30)

    neurons_df = pd.DataFrame(
        {
            "simple_id": range(30),
            "cell_type": [f"Type_{i%3}" for i in range(30)],
        }
    )

    data = DynamicsData(spikes=spikes, dt=0.1, neurons_df=neurons_df)
    format = DynamicsPlotFormat(mode="grouped", group_by="neuron_type")

    fig = plot_isi_cv(data=data, format=format)

    save_fig(fig, name="isi_cv_grouped")
    plt.close(fig)


def test_plot_isi_cv_dataclass():
    """Test ISI CV with dataclass interface."""
    spikes = generate_spike_data()

    data = DynamicsData(spikes=spikes, dt=0.1)
    format = DynamicsPlotFormat(mode="individual")

    fig = plot_isi_cv(data=data, format=format)

    save_fig(fig, name="isi_cv_dataclass")
    plt.close(fig)


def test_multiscale_fano_error_no_spikes():
    """Test that error is raised when spikes not provided."""
    with pytest.raises(ValueError, match="spikes is required"):
        plot_multiscale_fano(dt=0.1)


def test_multiscale_fano_error_grouped_no_metadata():
    """Test that error is raised for grouped mode without metadata."""
    spikes = generate_spike_data()

    with pytest.raises(ValueError, match="group_by must be specified"):
        plot_multiscale_fano(spikes=spikes, mode="grouped")


# --- Merged from test_dynamics_plots.py ---


def test_plot_avalanche_analysis():
    # Generate random spike data
    spikes = (np.random.rand(100, 10) > 0.8).astype(int)
    fig, res = plot_avalanche_analysis(spikes, bin_size=1)
    save_fig(fig, name="avalanche_analysis")
    plt.close(fig)


def test_plot_eigenvalue_spectrum():
    W = np.random.randn(50, 50)
    fig, ax, res = plot_eigenvalue_spectrum(W)
    save_fig(fig, name="eigenvalue_spectrum")
    plt.close(fig)


def test_plot_lyapunov_spectrum():
    metrics = np.sort(np.random.randn(10))[::-1]
    fig, ax = plot_lyapunov_spectrum(metrics)
    save_fig(fig, name="lyapunov_spectrum")
    plt.close(fig)


def test_plot_micro_dynamics():
    spikes = (np.random.rand(100, 10) > 0.8).astype(int)
    fig, res = plot_micro_dynamics(spikes)
    save_fig(fig, name="micro_dynamics")
    plt.close(fig)


def test_plot_gain_stability():
    # data: slope, intercept, g_values, lambda_values
    data = (
        0.5,
        -1.0,
        np.linspace(0.5, 5, 10),
        0.5 * np.linspace(0.5, 5, 10) - 1.0 + np.random.randn(10) * 0.01,
    )
    fig, ax = plot_gain_stability(data)
    save_fig(fig, name="gain_stability")
    plt.close(fig)


def test_plot_firing_rate_distribution():
    """Test new firing rate distribution plot."""
    spikes = generate_spike_data()
    fig, stats = plot_firing_rate_distribution(spikes, dt=0.1)
    save_fig(fig, name="firing_rate_distribution")
    plt.close(fig)
