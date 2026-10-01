"""Direct tests for the private building blocks behind ``plot_neuron_traces``.

The public function is covered by ``test_neuron_traces.py`` and the
characterization tests; these tests pin the helpers that decide *what* is
drawn (neuron selection, panel kinds, batch handling, label and colour
resolution, grid layout) in isolation.
"""

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from btorch.visualisation.timeseries import (  # noqa: E402
    NeuronSpec,
    SimulationStates,
    TracePlotFormat,
)
from btorch.visualisation.timeseries._neuron_trace_impl import (  # noqa: E402
    _colors_for_spec,
    _create_trace_grid,
    _extract_batch_dim,
    _format_top_neuron_label,
    _make_label_resolver,
    _prepare_trace_data,
    _resolve_neuron_spec,
    _select_trace_neurons,
    _trace_panel_kinds,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _arr(*shape, seed=0):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# Neuron selection
# --------------------------------------------------------------------------- #
def test_select_default_is_first_five_neurons():
    assert _select_trace_neurons(8, None, None, 42) == [0, 1, 2, 3, 4]
    assert _select_trace_neurons(3, None, None, 42) == [0, 1, 2]


def test_select_explicit_indices_win_untouched():
    assert _select_trace_neurons(8, [7, 2], 3, 42) == [7, 2]


def test_select_sample_is_seeded_sorted_and_does_not_touch_global_rng():
    np.random.seed(123)
    expected_next = np.random.rand()
    np.random.seed(123)
    a = _select_trace_neurons(20, None, 5, seed=7)
    b = _select_trace_neurons(20, None, 5, seed=7)
    assert a == b == sorted(a)
    assert len(set(a)) == 5
    # The helper must not consume or reseed numpy's global generator.
    assert np.random.rand() == expected_next
    assert _select_trace_neurons(20, None, 5, seed=8) != a


def test_select_sample_is_capped_at_population_size():
    assert _select_trace_neurons(3, None, 10, 1) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Panel selection
# --------------------------------------------------------------------------- #
def test_panel_kinds_require_flag_and_data():
    states = SimulationStates(voltage=_arr(10, 3), asc=_arr(10, 3))
    data = _prepare_trace_data(states, None)
    # ASC has data, PSC does not: only voltage and asc are drawn.
    assert _trace_panel_kinds(TracePlotFormat(), data) == ["voltage", "asc"]
    assert _trace_panel_kinds(TracePlotFormat(show_asc=False), data) == ["voltage"]
    assert _trace_panel_kinds(TracePlotFormat(show_voltage=False), data) == ["asc"]
    none_shown = TracePlotFormat(show_voltage=False, show_asc=False, show_psc=False)
    assert _trace_panel_kinds(none_shown, data) == []


def test_panel_kinds_full_order_is_voltage_asc_psc():
    states = SimulationStates(
        voltage=_arr(10, 3), asc=_arr(10, 3), psc=_arr(10, 3, seed=1)
    )
    kinds = _trace_panel_kinds(TracePlotFormat(), _prepare_trace_data(states, None))
    assert kinds == ["voltage", "asc", "psc"]


# --------------------------------------------------------------------------- #
# Batch handling and PSC layout
# --------------------------------------------------------------------------- #
def test_extract_batch_dim_by_rank():
    assert _extract_batch_dim(None, 0) is None
    two_d = _arr(5, 3)
    assert _extract_batch_dim(two_d, 4) is two_d  # no batch axis: untouched
    three_d = _arr(5, 2, 3)
    np.testing.assert_array_equal(_extract_batch_dim(three_d, 1), three_d[:, 1])
    four_d = _arr(5, 2, 3, 4)
    np.testing.assert_array_equal(_extract_batch_dim(four_d, 0), four_d[:, 0])
    # Torch tensors are converted.
    assert isinstance(_extract_batch_dim(torch.zeros(5, 2, 3), 0), np.ndarray)


def test_extract_batch_dim_errors():
    with pytest.raises(ValueError, match="out of bounds"):
        _extract_batch_dim(_arr(5, 2, 3), 2)
    with pytest.raises(ValueError, match="2D, 3D, or 4D"):
        _extract_batch_dim(np.zeros(5), 0)


def test_prepare_trace_data_multi_component_psc_gets_default_labels():
    """(time, neurons, n_psc) next to 2D voltage is a multi-component PSC."""
    states = SimulationStates(voltage=_arr(10, 3), psc=_arr(10, 3, 2, seed=1))
    data = _prepare_trace_data(states, None)
    assert data.psc_has_extra_dim
    assert data.psc_labels == ["PSC_0", "PSC_1"]


def test_prepare_trace_data_batched_psc_is_not_multi_component():
    """(time, batch, neurons) next to 3D voltage is a plain batched PSC."""
    states = SimulationStates(voltage=_arr(10, 2, 3), psc=_arr(10, 2, 3, seed=1))
    data = _prepare_trace_data(states, 1)
    assert not data.psc_has_extra_dim
    assert data.psc.shape == (10, 3) and data.voltage.shape == (10, 3)


def test_prepare_trace_data_rejects_epsc_with_multi_component_psc():
    states = SimulationStates(voltage=_arr(10, 3), psc=_arr(10, 3, 2), epsc=_arr(10, 3))
    with pytest.raises(ValueError, match="epsc must be None"):
        _prepare_trace_data(states, None)


# --------------------------------------------------------------------------- #
# Labels, specs and colours
# --------------------------------------------------------------------------- #
def test_label_resolver_callable_sequence_and_none():
    # Callable gets the neuron index, sequences are indexed by plot position.
    assert _make_label_resolver(lambda i: f"n{i}")(0, 7) == "n7"
    seq = _make_label_resolver(["a", "b"])
    assert seq(1, 9) == "b"
    assert seq(2, 9) is None
    assert _make_label_resolver(None)(0, 0) is None


def test_format_top_label_wraps_on_word_boundaries():
    assert _format_top_neuron_label("short", 10) == "short"
    assert _format_top_neuron_label("alpha beta gamma", 10) == "alpha beta\ngamma"
    assert _format_top_neuron_label("anything", 0) == "anything"


def test_resolve_neuron_spec_variants():
    spec = NeuronSpec(color="red")
    assert _resolve_neuron_spec(None, 0) == NeuronSpec()
    assert _resolve_neuron_spec(spec, 5) is spec  # single spec applies to all
    assert _resolve_neuron_spec({"color": "blue"}, 3).color == "blue"
    listed = [spec, {"linewidth": 2.0}]
    assert _resolve_neuron_spec(listed, 0) is spec
    assert _resolve_neuron_spec(listed, 1).linewidth == 2.0
    assert _resolve_neuron_spec(listed, 2) == NeuronSpec()  # past the list


def test_colors_for_spec_single_colour_keeps_spike_colour():
    base = {"voltage": "v", "asc": "a", "spike": "s"}
    out = _colors_for_spec(base, NeuronSpec(color="red"))
    assert out == {"voltage": "red", "asc": "red", "spike": "s"}
    assert base["voltage"] == "v"  # input is not mutated
    out = _colors_for_spec(base, NeuronSpec(color={"asc": "blue"}))
    assert out == {"voltage": "v", "asc": "blue", "spike": "s"}
    assert _colors_for_spec(base, NeuronSpec()) == base


# --------------------------------------------------------------------------- #
# Grid layout
# --------------------------------------------------------------------------- #
def test_trace_grid_rows_and_columns():
    """5 neurons, 2 per row, 3 panels -> 3 rows x 6 columns of axes."""
    grid = _create_trace_grid(5, 3, 2, False, 0, 12.0, 2.5)
    assert grid.n_rows == 3
    assert len(grid.axes) == 3 * 6
    assert grid.label_axes == {}
    assert grid.plot_row(1) == 1


def test_trace_grid_top_labels_interleave_label_rows():
    grid = _create_trace_grid(3, 2, 2, True, 20, 12.0, 2.5)
    assert grid.n_rows == 2
    # One label axes per (row, slot); trace rows are 1 and 3.
    assert len(grid.label_axes) == 4
    assert grid.plot_row(0) == 1 and grid.plot_row(1) == 3
    assert grid.max_label_chars_per_line >= 36
