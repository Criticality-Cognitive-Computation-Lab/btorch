"""Tests for multiscale dynamics visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

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


# Optional extras: skip this module when the library is not installed.
pytest.importorskip("nolds")
pytest.importorskip("powerlaw")


def generate_spike_data(n_time=5000, n_neurons=50, rate=0.05, seed=0):
    """Generate seeded synthetic Bernoulli spike data (time, neurons)."""
    rng = np.random.default_rng(seed)
    return (rng.random((n_time, n_neurons)) < rate).astype(float)


def _all_line_data_finite(fig):
    """Return True when every plotted line / scatter has no NaN coordinates."""
    for ax in fig.axes:
        for line in ax.lines:
            if not np.all(np.isfinite(np.asarray(line.get_ydata(), dtype=float))):
                return False
    return True


def test_plot_multiscale_fano_plain_args():
    """Test Fano factor plotting with plain arguments."""
    spikes = generate_spike_data()

    fig = plot_multiscale_fano(
        spikes=spikes, dt=0.1, windows=[10, 50, 100, 500], mode="individual"
    )

    # One axes, one line per neuron, labelled with the window axis.
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_title() == "Multiscale Fano Factor - Individual Neurons"
    assert ax.get_xlabel() == "Time Window (ms)"
    assert ax.get_ylabel() == "Fano Factor"
    assert len(ax.lines) == 10
    assert _all_line_data_finite(fig)

    save_fig(fig, name="multiscale_fano_plain_args")
    plt.close(fig)


def test_plot_multiscale_fano_dataclass():
    """Test Fano factor plotting with dataclass interface."""
    spikes = generate_spike_data()

    data = DynamicsData(spikes=spikes, dt=0.1)
    config = FanoFactorConfig(windows=[10, 50, 100, 500, 1000], overlap=5)
    format = DynamicsPlotFormat(mode="individual")

    fig = plot_multiscale_fano(data=data, config=config, format=format)

    # Dataclass interface must produce the same kind of figure as plain args.
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Multiscale Fano Factor - Individual Neurons"
    assert len(fig.axes[0].lines) > 0

    save_fig(fig, name="multiscale_fano_dataclass")
    plt.close(fig)


def test_plot_multiscale_fano_distribution():
    """Test Fano factor distribution mode."""
    spikes = generate_spike_data()

    fig = plot_multiscale_fano(spikes=spikes, dt=0.1, mode="distribution")

    # Distribution mode draws violins (PolyCollections), not line plots.
    ax = fig.axes[0]
    assert ax.get_title() == "Multiscale Fano Factor - Distribution"
    assert len(ax.lines) == 0
    assert len(ax.collections) > 0

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

    # One mean line per neuron type (Type_0..Type_2).
    ax = fig.axes[0]
    assert ax.get_title() == "Multiscale Fano Factor - Grouped by neuron_type"
    assert ax.get_ylabel() == "Fano Factor (mean)"
    assert len(ax.lines) == 3

    save_fig(fig, name="multiscale_fano_grouped_neuron_type")
    plt.close(fig)


def test_plot_multiscale_fano_grouped_by_neuropil():
    """Test Fano factor grouped by neuropil."""
    n_neurons = 30
    spikes = generate_spike_data(n_neurons=n_neurons)

    # Create neuron and connection metadata (seeded)
    rng = np.random.default_rng(1)
    neurons_df = pd.DataFrame(
        {
            "simple_id": range(n_neurons),
            "group": [f"neuropil_{i%3}.neuropil_{(i+1)%3}" for i in range(n_neurons)],
        }
    )

    connections_df = pd.DataFrame(
        {
            "pre_simple_id": rng.choice(n_neurons, 100),
            "post_simple_id": rng.choice(n_neurons, 100),
            "neuropil": [f"neuropil_{i%3}" for i in range(100)],
        }
    )

    data = DynamicsData(
        spikes=spikes, dt=0.1, neurons_df=neurons_df, connections_df=connections_df
    )
    format = DynamicsPlotFormat(mode="grouped", group_by="neuropil")

    fig = plot_multiscale_fano(data=data, format=format)

    assert isinstance(fig, Figure)
    assert "neuropil" in fig.axes[0].get_title()

    save_fig(fig, name="multiscale_fano_grouped_neuropil")
    plt.close(fig)


def test_plot_dfa_analysis_plain_args():
    """Test DFA analysis plotting with plain arguments."""
    spikes = generate_spike_data()

    fig = plot_dfa_analysis(spikes=spikes, dt=0.1)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "Detrended Fluctuation Analysis"

    save_fig(fig, name="dfa_analysis_plain_args")
    plt.close(fig)


def test_plot_dfa_analysis_dataclass():
    """Test DFA analysis plotting with dataclass interface."""
    spikes = generate_spike_data()

    data = DynamicsData(spikes=spikes, dt=0.1)
    config = DFAConfig(min_window=4, max_window=100, bin_size=1)

    fig = plot_dfa_analysis(data=data, config=config)

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Detrended Fluctuation Analysis"

    save_fig(fig, name="dfa_analysis_dataclass")
    plt.close(fig)


def test_plot_isi_cv_distribution():
    """Test ISI CV distribution plotting."""
    spikes = generate_spike_data()

    fig = plot_isi_cv(spikes=spikes, dt=0.1, mode="distribution")

    # Histogram of CV values: 30 bins drawn as patches plus a mean marker line.
    ax = fig.axes[0]
    assert ax.get_title() == "ISI Coefficient of Variation Distribution"
    assert ax.get_xlabel() == "ISI CV"
    assert ax.get_ylabel() == "Count"
    assert len(ax.patches) == 30
    # Poisson-like (Bernoulli) spiking has CV close to 1 (loose bounds).
    mean_cv = ax.lines[0].get_xdata()[0]
    assert 0.5 < mean_cv < 1.5

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

    # One bar per neuron type.
    ax = fig.axes[0]
    assert ax.get_title() == "ISI CV by cell_type"
    assert ax.get_ylabel() == "Mean ISI CV"
    assert len(ax.patches) == 3

    save_fig(fig, name="isi_cv_grouped")
    plt.close(fig)


def test_plot_isi_cv_dataclass():
    """Test ISI CV with dataclass interface."""
    spikes = generate_spike_data()

    data = DynamicsData(spikes=spikes, dt=0.1)
    format = DynamicsPlotFormat(mode="individual")

    fig = plot_isi_cv(data=data, format=format)

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_xlabel() == "ISI CV"
    assert len(fig.axes[0].patches) == 30

    save_fig(fig, name="isi_cv_dataclass")
    plt.close(fig)


def test_multiscale_fano_error_grouped_no_metadata():
    """Test that error is raised for grouped mode without metadata."""
    spikes = generate_spike_data()

    with pytest.raises(ValueError, match="group_by must be specified"):
        plot_multiscale_fano(spikes=spikes, mode="grouped")


# --- Merged from test_dynamics_plots.py ---


def test_plot_avalanche_analysis():
    rng = np.random.default_rng(0)
    spikes = (rng.random((2000, 10)) < 0.05).astype(int)
    fig, res = plot_avalanche_analysis(spikes, bin_size=1)
    # Result dict carries the raw avalanches and fitted exponents.
    assert isinstance(res, dict)
    assert len(res["sizes"]) == len(res["durations"]) > 0
    assert np.all(res["sizes"] >= 1) and np.all(res["durations"] >= 1)
    for key in ("tau", "alpha", "gamma"):
        assert np.isfinite(res[key])
    # Three panels: P(S), P(T) and <S>(T).
    assert len(fig.axes) == 3
    assert [a.get_ylabel() for a in fig.axes] == ["P(S)", "P(T)", "Average Size <S>"]

    save_fig(fig, name="avalanche_analysis")
    plt.close(fig)


def test_plot_eigenvalue_spectrum():
    W = np.random.default_rng(0).standard_normal((50, 50))
    fig, ax, res = plot_eigenvalue_spectrum(W)

    assert ax.get_title() == "Eigenvalue Spectrum"
    assert res["eigenvalues"].shape == (50,)
    # Spectral radius bounds every eigenvalue modulus up to numerical error.
    assert res["spectral_radius"] > 0
    assert np.abs(res["eigenvalues"]).max() >= res["spectral_radius"] - 1e-9
    # One scatter for the bulk and one for outliers, plus the unit circle patch.
    assert len(ax.collections) == 2
    assert len(ax.patches) == 1
    save_fig(fig, name="eigenvalue_spectrum")
    plt.close(fig)


def test_plot_lyapunov_spectrum():
    metrics = np.sort(np.random.default_rng(0).standard_normal(10))[::-1]
    fig, ax = plot_lyapunov_spectrum(metrics)

    assert ax.get_xlabel() == "Index"
    assert ax.get_ylabel() == "Lyapunov Exponent"
    # Title reports the Kaplan-Yorke dimension.
    assert "D_KY" in ax.get_title()
    assert len(ax.lines) == 2
    # The spectrum line has one point per exponent.
    assert any(len(line.get_ydata()) == 10 for line in ax.lines)
    save_fig(fig, name="lyapunov_spectrum")
    plt.close(fig)


def test_plot_micro_dynamics():
    spikes = (np.random.default_rng(0).random((100, 10)) > 0.8).astype(int)
    fig, res = plot_micro_dynamics(spikes)

    # Two panels: firing-rate and ISI-CV distributions.
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Rate Distribution"
    assert fig.axes[1].get_title() == "CV Distribution"
    assert set(res) == {"fr", "cv"}
    save_fig(fig, name="micro_dynamics")
    plt.close(fig)


def test_plot_gain_stability():
    # data: slope, intercept, g_values, lambda_values (seeded noisy line)
    rng = np.random.default_rng(0)
    data = (
        0.5,
        -1.0,
        np.linspace(0.5, 5, 10),
        0.5 * np.linspace(0.5, 5, 10) - 1.0 + rng.standard_normal(10) * 0.01,
    )
    fig, ax = plot_gain_stability(data)

    assert ax.get_title() == "Gain Stability Analysis"
    assert ax.get_xlabel() == "Gain (g)"
    # Scatter of the measurements (10 points) plus the fitted line.
    assert len(ax.collections) == 1
    assert len(ax.collections[0].get_offsets()) == 10
    assert len(ax.lines) == 1
    save_fig(fig, name="gain_stability")
    plt.close(fig)


def test_plot_firing_rate_distribution():
    """Test new firing rate distribution plot."""
    spikes = generate_spike_data()
    fig, stats = plot_firing_rate_distribution(spikes, dt=0.1)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "Rate Distribution"
    assert fig.axes[0].get_xlabel() == "Firing Rate (Hz)"
    assert len(fig.axes[0].patches) == 30
    assert all(np.isfinite(stats[k]) for k in ("mean", "skew", "kurt"))
    assert np.all(np.isfinite(stats["rates"]))
    assert stats["mean"] > 0

    save_fig(fig, name="firing_rate_distribution")
    plt.close(fig)


def test_fano_grouped_rejects_unknown_group_by():
    """An unsupported ``group_by`` is a ``ValueError``, not a ``NameError``.

    The grouped Fano plot only knows how to aggregate by ``neuron_type`` or
    ``neuropil``; anything else must fail with a clear message up front.
    """
    from btorch.visualisation.dynamics import _plot_fano_grouped

    with pytest.raises(ValueError, match="group_by must be"):
        _plot_fano_grouped(
            {10: np.ones(3)}, [10], 1.0, None, None, None, "bogus", "cell_type"
        )
