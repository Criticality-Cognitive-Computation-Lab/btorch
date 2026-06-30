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
    """Test plotting with metadata-derived labels via callable."""
    n_time, n_neurons = 1000, 20
    dt = 0.1
    voltage = -65 + 15 * np.random.randn(n_time, n_neurons)

    # Create neuron metadata
    neurons_df = pd.DataFrame(
        {
            "simple_id": range(n_neurons),  # trace neuron index
            "root_id": [10_000 + i for i in range(n_neurons)],
        }
    )
    root_id_by_simple_id = neurons_df.set_index("simple_id")["root_id"].to_dict()
    selected_indices = [0, 5, 10]

    states = SimulationStates(voltage=voltage, dt=dt)

    fig = plot_neuron_traces(
        states=states,
        neuron_indices=selected_indices,
        neurons_df=neurons_df,
        neuron_labels=lambda idx: f"root_id={root_id_by_simple_id[idx]}",
    )

    labels = [text.get_text() for ax in fig.axes for text in ax.texts]
    expected_labels = [
        f"root_id={root_id_by_simple_id[idx]}" for idx in selected_indices
    ]
    metadata_labels = [label for label in labels if label.startswith("root_id=")]
    assert sorted(metadata_labels) == sorted(expected_labels)

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


def test_plot_neuron_traces_multi_column_neurons():
    """Test multi-column layout when last row is incomplete."""
    n_time, n_neurons = 200, 3
    dt = 0.1
    voltage = -65 + 5 * np.random.randn(n_time, n_neurons)
    asc = -10 * np.random.exponential(1, (n_time, n_neurons))
    psc = 50 * np.random.randn(n_time, n_neurons)

    fig = plot_neuron_traces(
        voltage=voltage,
        asc=asc,
        psc=psc,
        dt=dt,
        neuron_indices=[0, 1, 2],
        show_voltage=False,
        show_asc=True,
        show_psc=True,
        neurons_per_row=2,
    )

    # Fixed grid: 2 rows * (2 neurons_per_row * 2 trace columns) = 8 axes total.
    # Last row has one empty neuron slot => 2 hidden axes.
    assert len(fig.axes) == 8
    invisible = [ax for ax in fig.axes if not ax.get_visible()]
    assert len(invisible) == 2

    # Last row keeps the same per-axis width as first row (no stretching).
    first_row_w = fig.axes[0].get_position().width
    second_row_w = fig.axes[4].get_position().width
    assert np.isclose(first_row_w, second_row_w, rtol=1e-6, atol=1e-6)

    # Side labels are off by default.
    labels = [
        text.get_text()
        for ax in fig.axes
        for text in ax.texts
        if text.get_text().startswith("Neuron ")
    ]
    assert labels == []

    save_fig(fig, name="multi_column_neurons")
    plt.close(fig)


def test_plot_neuron_traces_custom_side_labels_callable():
    """Test custom top labels via callable in multi-column layout."""
    n_time, n_neurons = 400, 3
    voltage = -65 + 2 * np.random.randn(n_time, n_neurons)
    asc = -10 * np.random.exponential(1, (n_time, n_neurons))
    psc = 50 * np.random.randn(n_time, n_neurons)

    fig = plot_neuron_traces(
        voltage=voltage,
        asc=asc,
        psc=psc,
        dt=0.1,
        neuron_indices=[0, 2],
        neuron_labels=lambda idx: f"Cell-{idx}",
        neuron_label_position="top",
        neurons_per_row=2,
    )

    # Top labels are added via ax.text() on label-row axes
    labels = [text for ax in fig.axes for text in ax.texts]
    label_texts = [
        text.get_text() for text in labels if text.get_text().startswith("Cell-")
    ]
    assert sorted(label_texts) == [
        "Cell-0",
        "Cell-2",
    ]
    # For top placement, labels are rendered in dedicated label-row axes.
    assert all(
        len(text.axes.lines) == 0
        for text in labels
        if text.get_text().startswith("Cell-")
    )

    save_fig(fig, name="neuron_traces_top_labels_callable")
    plt.close(fig)


def test_plot_neuron_traces_long_labels_adaptive_width():
    """Test that figure width adapts to accommodate long top labels.

    When labels are very long and positioned at the top, the figure
    width should expand to prevent subfigures from becoming too narrow.

    Note: This test uses a relative comparison (long vs short labels) rather
    than absolute width thresholds because font metrics vary across
    environments (local vs CI, different DPI settings, font availability).
    """
    n_time, n_neurons = 100, 3
    voltage = -65 + 5 * np.random.randn(n_time, n_neurons)
    psc = 50 * np.random.randn(n_time, n_neurons)

    # Create baseline figure with short labels to measure default width
    fig_short = plot_neuron_traces(
        voltage=voltage,
        psc=psc,
        dt=0.1,
        neuron_indices=[0, 1],
        neuron_labels=lambda idx: f"N{idx}",
        neuron_label_position="top",
        neurons_per_row=2,
        show_asc=False,
    )
    width_short = fig_short.get_figwidth()
    save_fig(fig_short, name="neuron_traces_short_labels_adaptive_width")
    plt.close(fig_short)

    # Create figure with extremely long labels to force adaptive sizing
    # This label is ~120 chars, which should require significant width
    long_label_prefix = (
        "rid=720575940606137632|v=-51.6|fr=23.7|cv=0.00|fa=0.00|"
        "eci=650540544.00|lg=-8.0|r_pa=3.15|r_ei=0.00|extra_field=12345"
    )

    fig = plot_neuron_traces(
        voltage=voltage,
        psc=psc,
        dt=0.1,
        neuron_indices=[0, 1],
        neuron_labels=lambda idx: f"{long_label_prefix}|idx={idx}",
        neuron_label_position="top",
        neurons_per_row=2,
        show_asc=False,
    )
    fig_width = fig.get_figwidth()

    # The key assertion: figure with long labels should be significantly
    # wider than baseline (at least 20% wider, typically much more)
    # Using relative comparison makes this test robust to font variations
    assert fig_width > width_short, (
        f"Figure with long labels ({fig_width:.1f}) should be "
        f"significantly wider than baseline ({width_short:.1f})"
    )

    # Verify labels are present (top labels are axis-level on label-row axes)
    labels = [
        text.get_text()
        for ax in fig.axes
        for text in ax.texts
        if text.get_text().startswith("rid=")
    ]
    assert len(labels) == 2, "Expected 2 neuron labels"
    assert all(long_label_prefix in label for label in labels)

    # Verify subplots have reasonable width (not too narrow)
    # Each subplot should be at least ~2.5 inches wide
    axes_positions = [ax.get_position() for ax in fig.axes if ax.get_visible()]
    for pos in axes_positions:
        subplot_width_inches = pos.width * fig_width
        assert (
            subplot_width_inches >= 2.5
        ), f"Subplot width ({subplot_width_inches:.1f}) should be >= 2.5 inches"

    save_fig(fig, name="neuron_traces_long_labels_adaptive_width")
    plt.close(fig)

    n_time, n_neurons = 100, 4
    voltage = -65 + 5 * np.random.randn(n_time, n_neurons)

    fig = plot_neuron_traces(
        voltage=voltage,
        dt=0.1,
        neuron_indices=[0, 1, 2, 3],
        neuron_labels=lambda idx: f"N{idx}",  # Short labels
        neuron_label_position="top",
        neurons_per_row=2,
        show_asc=False,
        show_psc=False,
    )

    # With short labels, figure should use default width (~20 inches)
    fig_width = fig.get_figwidth()
    assert fig_width <= 25.0, (
        f"Figure width ({fig_width:.1f}) should not expand unnecessarily "
        "for short labels"
    )

    save_fig(fig, name="neuron_traces_short_labels_compact")
    plt.close(fig)


def test_plot_neuron_traces_thresholds_and_reset_per_neuron():
    """Test per-neuron v_threshold and v_rest on voltage plots."""
    n_time, n_neurons = 120, 4
    voltage = -65 + 2 * np.random.randn(n_time, n_neurons)
    neuron_indices = [1, 3]

    neurons_df = pd.DataFrame(
        {
            "simple_id": range(n_neurons),  # trace neuron index
            "v_threshold": [-40.0, -41.0, -42.0, -43.0],
            "v_rest": [-60.0, -61.0, -62.0, -63.0],
        }
    )
    v_threshold_by_simple_id = neurons_df.set_index("simple_id")[
        "v_threshold"
    ].to_dict()
    v_rest_by_simple_id = neurons_df.set_index("simple_id")["v_rest"].to_dict()

    v_threshold = np.array(
        [v_threshold_by_simple_id[idx] for idx in range(n_neurons)], dtype=float
    )
    v_rest = np.array(
        [v_rest_by_simple_id[idx] for idx in range(n_neurons)], dtype=float
    )

    fig = plot_neuron_traces(
        voltage=voltage,
        dt=0.1,
        neuron_indices=neuron_indices,
        show_asc=False,
        show_psc=False,
        v_threshold=v_threshold,
        # API parameter is v_reset; pass per-neuron v_rest values here.
        v_reset=v_rest,
    )

    assert len(fig.axes) == 2

    expected_thresholds = [v_threshold_by_simple_id[idx] for idx in neuron_indices]
    expected_v_rest = [v_rest_by_simple_id[idx] for idx in neuron_indices]
    for ax_idx, (ax, exp_th, exp_v_rest) in enumerate(
        zip(fig.axes, expected_thresholds, expected_v_rest)
    ):
        v_th_lines = [line for line in ax.lines if line.get_label() == "V_th"]
        v_reset_lines = [line for line in ax.lines if line.get_label() == "V_reset"]
        assert len(v_th_lines) == 1
        assert len(v_reset_lines) == 1

        if ax_idx == 0:
            legend = ax.get_legend()
            assert legend is not None
            legend_labels = [text.get_text() for text in legend.get_texts()]
            assert "V_th" in legend_labels
            assert "V_reset" in legend_labels

        for ref_line in (v_th_lines[0], v_reset_lines[0]):
            assert np.isclose(ref_line.get_alpha(), 0.9)
            assert np.isclose(ref_line.get_linewidth(), 1.2)

        constant_levels = [
            float(line.get_ydata()[0])
            for line in ax.lines
            if np.allclose(line.get_ydata(), line.get_ydata()[0])
        ]
        assert any(np.isclose(level, exp_th) for level in constant_levels)
        assert any(np.isclose(level, exp_v_rest) for level in constant_levels)

    save_fig(fig, name="neuron_traces_thresholds_and_reset_per_neuron")
    plt.close(fig)


def test_plot_neuron_traces_threshold_invalid_length():
    """Test invalid per-neuron threshold length raises a clear error."""
    n_time, n_neurons = 100, 4
    voltage = -65 + 2 * np.random.randn(n_time, n_neurons)

    with pytest.raises(ValueError, match="v_threshold must be a scalar"):
        plot_neuron_traces(
            voltage=voltage,
            dt=0.1,
            neuron_indices=[0, 2],
            show_asc=False,
            show_psc=False,
            v_threshold=[-40.0, -41.0, -42.0],  # not n_neurons nor n_plot
        )


def test_plot_neuron_traces_axis_units_voltage_and_currents():
    """Test axis units are correct for voltage and current traces."""
    n_time, n_neurons = 120, 3
    voltage = -65 + 2 * np.random.randn(n_time, n_neurons)
    asc = -10 * np.random.exponential(1, (n_time, n_neurons))
    psc = 50 * np.random.randn(n_time, n_neurons)

    fig = plot_neuron_traces(
        voltage=voltage,
        asc=asc,
        psc=psc,
        dt=0.1,
        neuron_indices=[0],
        show_voltage=True,
        show_asc=True,
        show_psc=True,
    )

    assert len(fig.axes) == 3
    assert fig.axes[0].get_ylabel() == "V (mV)"
    assert fig.axes[1].get_ylabel() == "ASC (pA)"
    assert fig.axes[2].get_ylabel() == "PSC (pA)"

    plt.close(fig)


def test_plot_neuron_traces_3column_long_labels():
    """Test 3-column layout with long metadata labels and adaptive width.

    This verifies that long neuron labels (containing root_id,
    cell_type, neurotransmitter, etc.) don't squeeze the subplot columns
    too narrow, and that the figure width scales appropriately.
    """
    n_time, n_neurons = 500, 6
    dt = 0.1
    voltage = -65 + 15 * np.random.randn(n_time, n_neurons)
    asc = -10 * np.random.exponential(1, (n_time, n_neurons))
    psc = 50 * np.random.randn(n_time, n_neurons)
    spikes = np.zeros((n_time, n_neurons))

    # Simulate some spikes for realism
    for i in range(n_neurons):
        spike_times = np.random.choice(n_time, size=5, replace=False)
        spikes[spike_times, i] = 1

    # Create neuron metadata with long descriptive labels
    neurons_df = pd.DataFrame(
        {
            "simple_id": range(n_neurons),
            "root_id": [720575940629199810 + i * 1000 for i in range(n_neurons)],
            "cell_type": ["ORN_VA1v", "Sm02", "R1-6", "R1-6", "BM_lNm", "Tm5f"],
            "nt": ["ACH", "ACH", "ACH", "ACH", "ACH", "ACH"],
            "v_rest": [-60.0 + i * 0.5 for i in range(n_neurons)],
        }
    )

    def make_long_label(idx: int) -> str:
        row = neurons_df.iloc[idx]
        # Long label mimicking connectome metadata format
        return (
            f"rid={row['root_id']}|"
            f"ct={row['cell_type']}|"
            f"nt={row['nt']}|"
            f"v={row['v_rest']:.1f}|"
            f"fr={np.random.uniform(10, 400):.1f}|"
            f"cv={np.random.uniform(0, 2):.2f}|"
            f"fa={np.random.uniform(0, 0.05):.2f}|"
            f"ecl={np.random.uniform(7000, 8000):.2f}|"
            f"cr={np.random.uniform(0, 1):.2f}|"
            f"lg={np.random.uniform(-10, -5):.1f}|"
            f"r_ei=nan"
        )

    states = SimulationStates(
        voltage=voltage,
        asc=asc,
        psc=psc,
        spikes=spikes,
        dt=dt,
    )

    format = TracePlotFormat(
        neuron_indices=[0, 1, 2, 3, 4, 5],
        show_voltage=True,
        show_asc=True,
        show_psc=True,
        neuron_labels=make_long_label,
        neuron_label_position="top",
        neurons_per_row=2,  # 3 neurons per row to fit more columns
        auto_width=True,
    )

    fig = plot_neuron_traces(states=states, format=format)

    # Verify combined figure (not separate_figures mode)
    assert not isinstance(fig, dict), "Expected combined Figure, not dict"

    # Verify figure has reasonable dimensions (not too narrow)
    # With 3 columns x 2 neurons_per_row = 6 subplots wide
    # Should be at least 20 inches wide to accommodate long labels
    assert fig.get_figwidth() >= 20.0, (
        f"Figure width {fig.get_figwidth()} too narrow for 3-column layout "
        f"with long labels"
    )

    # Verify we have the expected number of axes (label rows + plot rows)
    # With 6 neurons and 2 per row = 3 rows * (1 label + 3 traces) = 12 axes
    expected_axes = 3 * 2 * 2  # 3 rows, 2 neurons per row, 2 row types each
    assert (
        len(fig.axes) >= expected_axes
    ), f"Expected at least {expected_axes} axes, got {len(fig.axes)}"

    save_fig(fig, name="neuron_traces_long_labels_adaptive_width")
    plt.close(fig)


def test_plot_neuron_traces_with_input():
    """Test plotting with input current - real world use case.

    Simulates plotting neuron traces with external input current alongside
    EPSC, IPSC and total PSC.
    """
    n_time, n_neurons = 1000, 20
    dt = 0.1
    voltage = -65 + 15 * np.random.randn(n_time, n_neurons)
    psc = 50 * np.random.randn(n_time, n_neurons)
    epsc = np.abs(psc) * (psc > 0)
    ipsc = -np.abs(psc) * (psc < 0)
    input_current = 100 * np.random.randn(n_time, n_neurons)

    # Test with plain args
    fig = plot_neuron_traces(
        voltage=voltage,
        psc=psc,
        epsc=epsc,
        ipsc=ipsc,
        input=input_current,
        dt=dt,
        neuron_indices=[0, 5, 10],
    )
    save_fig(fig, name="neuron_traces_with_input")
    plt.close(fig)

    # Test with dataclass interface
    states = SimulationStates(
        voltage=voltage,
        psc=psc,
        epsc=epsc,
        ipsc=ipsc,
        input=input_current,
        dt=dt,
    )
    fig = plot_neuron_traces(states=states, neuron_indices=[0, 1])
    save_fig(fig, name="neuron_traces_with_input_dataclass")
    plt.close(fig)


def test_plot_neuron_traces_psc_multi_dim():
    """Test PSC with multiple components like ASC - real world use case.

    When psc has shape (time, neurons, n_psc), each component is plotted
    separately with labels. This is analogous to how ASC handles multiple
    components.
    """
    n_time, n_neurons = 1000, 20
    dt = 0.1
    voltage = -65 + 15 * np.random.randn(n_time, n_neurons)
    # PSC with multiple components (e.g., different synapse types)
    psc = np.stack(
        [
            50 * np.random.randn(n_time, n_neurons),
            30 * np.random.randn(n_time, n_neurons),
            20 * np.random.randn(n_time, n_neurons),
            10 * np.random.randn(n_time, n_neurons),
        ],
        axis=2,
    )  # Shape: (time, neurons, 4)

    # Test with custom labels
    custom_labels = ["Excitatory", "Inhibitory", "Modulatory", "Background"]
    fig = plot_neuron_traces(
        voltage=voltage,
        psc=psc,
        psc_labels=custom_labels,
        dt=dt,
        neuron_indices=[0, 5, 10],
    )
    save_fig(fig, name="neuron_traces_psc_multi_dim")
    plt.close(fig)

    # Test with default labels (no psc_labels provided)
    fig = plot_neuron_traces(
        voltage=voltage,
        psc=psc[:, :, :3],  # Use 3 components for brevity
        dt=dt,
        neuron_indices=[0, 1],
    )
    save_fig(fig, name="neuron_traces_psc_multi_dim_default_labels")
    plt.close(fig)


def test_plot_neuron_traces_psc_multi_dim_validation_errors():
    """Test validation errors for incompatible args with multi-dim PSC.

    When psc has additional dimensions (n_psc > 1), epsc/ipsc/input must
    be None since those are already captured in the multi-dim psc.
    """
    n_time, n_neurons = 500, 10
    voltage = -65 + 10 * np.random.randn(n_time, n_neurons)
    psc = np.stack(
        [
            50 * np.random.randn(n_time, n_neurons),
            30 * np.random.randn(n_time, n_neurons),
            20 * np.random.randn(n_time, n_neurons),
        ],
        axis=2,
    )  # Shape: (time, neurons, 3)

    # epsc should raise error
    with pytest.raises(ValueError, match="epsc must be None"):
        plot_neuron_traces(
            voltage=voltage,
            psc=psc,
            epsc=30 * np.random.randn(n_time, n_neurons),
            dt=0.1,
            neuron_indices=[0, 1],
        )

    # ipsc should raise error
    with pytest.raises(ValueError, match="ipsc must be None"):
        plot_neuron_traces(
            voltage=voltage,
            psc=psc,
            ipsc=-20 * np.random.randn(n_time, n_neurons),
            dt=0.1,
            neuron_indices=[0, 1],
        )

    # input should raise error
    with pytest.raises(ValueError, match="input must be None"):
        plot_neuron_traces(
            voltage=voltage,
            psc=psc,
            input=100 * np.random.randn(n_time, n_neurons),
            dt=0.1,
            neuron_indices=[0, 1],
        )


def test_plot_neuron_traces_input_separate_figures():
    """Test input plotting in separate figures mode."""
    n_time, n_neurons = 500, 10
    voltage = -65 + 10 * np.random.randn(n_time, n_neurons)
    psc = 50 * np.random.randn(n_time, n_neurons)
    input_current = 100 * np.random.randn(n_time, n_neurons)

    figs = plot_neuron_traces(
        voltage=voltage,
        psc=psc,
        input=input_current,
        dt=0.1,
        neuron_indices=[0, 1],
        separate_figures=True,
        show_asc=False,
    )

    assert isinstance(figs, dict)
    assert "voltage" in figs
    assert "psc" in figs

    for name, fig in figs.items():
        save_fig(fig, name=f"neuron_traces_separate_input_{name}")
        plt.close(fig)
