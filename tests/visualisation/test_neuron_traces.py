"""Tests for neuron trace visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from btorch.utils.file import save_fig
from btorch.visualisation.timeseries import (
    NeuronSpec,
    SimulationStates,
    TracePlotFormat,
    plot_neuron_traces,
)


def test_plot_neuron_traces_plain_args():
    """Test plotting with plain arguments."""
    # Generate synthetic data
    n_time, n_neurons = 1000, 20
    dt = 0.1
    voltage = -65 + 15 * np.random.randn(n_time, n_neurons)
    psc = 50 * np.random.randn(n_time, n_neurons)

    # Plot with plain args
    fig = plot_neuron_traces(
        voltage=voltage,
        psc=psc,
        dt=dt,
        neuron_indices=[0, 5, 10],
    )

    save_fig(fig, name="neuron_traces_plain_args")
    plt.close(fig)


def test_plot_neuron_traces_dataclass():
    """Test plotting with dataclass interface."""
    # Generate synthetic data
    n_time, n_neurons = 1000, 20
    dt = 0.1

    # Simulate voltage with spikes
    voltage = -65 * np.ones((n_time, n_neurons))
    spikes = np.zeros((n_time, n_neurons))

    for i in range(n_neurons):
        # Random spike times
        spike_times = np.random.choice(n_time, size=5, replace=False)
        spikes[spike_times, i] = 1
        # Voltage spikes
        for t in spike_times:
            if t < n_time - 10:
                voltage[t : t + 10, i] = -65 + 50 * np.exp(-np.arange(10) / 3)

    asc = -10 * np.random.exponential(1, (n_time, n_neurons))
    psc = 50 * np.random.randn(n_time, n_neurons)
    epsc = np.abs(psc) * (psc > 0)
    ipsc = -np.abs(psc) * (psc < 0)

    # Create dataclasses
    states = SimulationStates(
        voltage=voltage,
        asc=asc,
        psc=psc,
        epsc=epsc,
        ipsc=ipsc,
        spikes=spikes,
        dt=dt,
        v_threshold=-40.0,
        v_reset=-65.0,
    )

    format = TracePlotFormat(
        sample_size=5,
        seed=42,
        show_voltage=True,
        show_asc=True,
        show_psc=True,
    )

    # Plot
    fig = plot_neuron_traces(states=states, format=format)

    save_fig(fig, name="neuron_traces_dataclass")
    plt.close(fig)


def test_plot_neuron_traces_mixed():
    """Test plotting with mixed dataclass and plain args."""
    n_time, n_neurons = 1000, 20
    dt = 0.1
    voltage = -65 + 15 * np.random.randn(n_time, n_neurons)
    psc = 50 * np.random.randn(n_time, n_neurons)

    states = SimulationStates(voltage=voltage, psc=psc, dt=dt)

    # Use dataclass for states, plain args for selection
    fig = plot_neuron_traces(states=states, neuron_indices=[0, 3, 7, 12])

    save_fig(fig, name="neuron_traces_mixed")
    plt.close(fig)


def test_plot_neuron_traces_with_metadata():
    """Test plotting with neuron metadata for labels."""
    n_time, n_neurons = 1000, 20
    dt = 0.1
    voltage = -65 + 15 * np.random.randn(n_time, n_neurons)

    # Create neuron metadata
    neurons_df = pd.DataFrame(
        {
            "simple_id": range(n_neurons),
            "cell_type": [f"Type_{i%3}" for i in range(n_neurons)],
        }
    )

    states = SimulationStates(voltage=voltage, dt=dt)

    fig = plot_neuron_traces(
        states=states, neuron_indices=[0, 5, 10], neurons_df=neurons_df
    )

    save_fig(fig, name="neuron_traces_with_metadata")
    plt.close(fig)


def test_plot_neuron_traces_voltage_only():
    """Test plotting voltage only."""
    n_time, n_neurons = 1000, 20
    voltage = -65 + 15 * np.random.randn(n_time, n_neurons)

    fig = plot_neuron_traces(
        voltage=voltage, dt=0.1, neuron_indices=[0, 5], show_asc=False, show_psc=False
    )

    save_fig(fig, name="neuron_traces_voltage_only")
    plt.close(fig)


def test_plot_neuron_traces_error_no_voltage():
    """Test that error is raised when voltage is not provided."""
    with pytest.raises(ValueError, match="voltage is required"):
        plot_neuron_traces(dt=0.1)


def test_plot_neuron_traces_auto_width():
    """Test plotting voltage only with auto-width."""
    n_time, n_neurons = 1000, 20
    voltage = -65 + 15 * np.random.randn(n_time, n_neurons)

    fig = plot_neuron_traces(
        voltage=voltage, dt=0.1, neuron_indices=[0, 5], show_asc=False, show_psc=False
    )

    # Check auto-width (1000*0.1 = 100ms duration. 100*0.025 = 2.5 < 10. So width 10)
    assert fig.get_figwidth() >= 10.0

    save_fig(fig, name="neuron_traces_voltage_only")
    plt.close(fig)


def test_plot_neuron_traces_separate_figures():
    """Test separate figures mode."""
    n_time, n_neurons = 1000, 20
    voltage = -65 + 15 * np.random.randn(n_time, n_neurons)
    psc = 50 * np.random.randn(n_time, n_neurons)

    figs = plot_neuron_traces(
        voltage=voltage,
        psc=psc,
        dt=0.1,
        neuron_indices=[0, 1],
        separate_figures=True,
        show_asc=False,
    )

    assert isinstance(figs, dict)
    assert "voltage" in figs
    assert "psc" in figs
    assert "asc" not in figs

    for name, fig in figs.items():
        save_fig(fig, name=f"neuron_traces_separate_{name}")
        plt.close(fig)


def test_plot_neuron_traces_missing_data_auto_hide():
    """Test that columns are hidden if data is missing, even if show_* is
    True."""
    n_time, n_neurons = 100, 5
    voltage = -65 + 5 * np.random.randn(n_time, n_neurons)

    # We ask to show ASC but provide no ASC data
    fig = plot_neuron_traces(
        voltage=voltage, dt=0.1, show_asc=True, asc=None, show_psc=False
    )

    # Should result in 1 column (voltage), not 2
    # fig.axes shape is (n_plot, n_cols)
    assert len(fig.axes) == 5  # 5 neurons * 1 col

    save_fig(fig, name="neuron_traces_auto_hide")
    plt.close(fig)


def test_plot_neuron_traces_with_specs():
    """Test plotting with scalar and list neuron specs."""
    voltage = np.random.randn(100, 3)  # 3 neurons

    # 1. Scalar Spec (Dict)
    fig1 = plot_neuron_traces(
        voltage=voltage,
        neuron_specs={"color": "red", "linestyle": "--"},
        show_asc=False,
        show_psc=False,
    )
    assert fig1 is not None

    # 2. Scalar Spec (Object)
    spec = NeuronSpec(color="blue", alpha=0.5)
    fig2 = plot_neuron_traces(
        voltage=voltage, neuron_specs=spec, show_asc=False, show_psc=False
    )
    assert fig2 is not None

    # 3. List Spec
    specs = [
        {"color": "red"},  # N0
        NeuronSpec(color="green"),  # N1
        {"linestyle": ":"},  # N2
    ]
    fig3 = plot_neuron_traces(
        voltage=voltage, neuron_specs=specs, show_asc=False, show_psc=False
    )
    assert fig3 is not None
