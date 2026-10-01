"""Direct tests for the private building blocks behind ``plot_raster``.

``test_raster.py`` and ``test_timeseries_characterization.py`` exercise the
public function; the tests here pin the small helpers one by one (neuron
ordering, spike-colour resolution, strip colours/layout, rate panel
computation) so a regression points at the helper that broke.
"""

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from btorch.visualisation.timeseries import (  # noqa: E402
    GroupStripOptions,
    NeuronSpec,
    RasterGrouping,
    RasterStyle,
    RatePanelOptions,
)
from btorch.visualisation.timeseries._raster_impl import (  # noqa: E402
    _create_raster_axes,
    _draw_group_strip,
    _draw_rate_panel,
    _order_neurons_by_group,
    _resolve_raster_groups,
    _resolve_spike_style,
    _resolve_total_rate,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _df() -> pd.DataFrame:
    """Six neurons, groups interleaved on purpose (B, A, B, A, B, A)."""
    return pd.DataFrame(
        {
            "group": ["B", "A"] * 3,
            "sub": ["y", "x", "x", "y", "y", "x"],
        }
    )


def _spikes(n_t=50, n_n=6, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n_t, n_n)) > 0.8).astype(np.float32)


# --------------------------------------------------------------------------- #
# Grouping / ordering
# --------------------------------------------------------------------------- #
def test_order_neurons_by_group_sorts_and_marks_boundaries():
    """Neurons are gathered per group (stable); boundary y sits between
    bands."""
    labels = np.array(["B", "A", "B", "A", "B", "A"], dtype=object)
    order, boundaries = _order_neurons_by_group(
        labels, labels, ["A", "B"], False, True, 6
    )
    assert order.tolist() == [1, 3, 5, 0, 2, 4]
    # Boundary lies half a neuron above the last neuron of each group.
    assert boundaries == [(2.5, "A"), (5.5, "B")]


def test_order_neurons_by_group_keeps_original_order_without_sorting():
    """sort_neurons=False keeps indices; boundaries mark label changes only."""
    labels = np.array(["A", "A", "B", "B", "A"], dtype=object)
    order, boundaries = _order_neurons_by_group(
        labels, labels, ["A", "B"], False, False, 5
    )
    assert order.tolist() == [0, 1, 2, 3, 4]
    assert boundaries == [(1.5, "A"), (3.5, "B")]


def test_order_neurons_by_group_subgroup_order_is_first_appearance():
    """Within a group, subgroups keep the order they first appear in."""
    top = np.array(["A"] * 4, dtype=object)
    sub = np.array(["y", "x", "y", "x"], dtype=object)
    order, _ = _order_neurons_by_group(top, sub, ["A"], True, True, 4)
    assert order.tolist() == [0, 2, 1, 3]


def test_order_neurons_warns_when_group_missing_from_sort():
    """Neurons whose group is not listed are appended with a warning."""
    labels = np.array(["A", "B", "C"], dtype=object)
    with pytest.warns(UserWarning, match="Not all neurons"):
        order, _ = _order_neurons_by_group(labels, labels, ["A", "B"], False, True, 3)
    assert order.tolist() == [0, 1, 2]


def test_resolve_raster_groups_group_sort_puts_unlisted_groups_last():
    """group_sort order wins; unknown names are ignored, others appended."""
    groups = _resolve_raster_groups(
        6,
        RasterGrouping(neurons_df=_df(), group_key="group", group_sort=["B", "Z"]),
        GroupStripOptions(show=False),
    )
    assert groups.groups == ["B", "A"]
    assert groups.sorted_indices.tolist() == [0, 2, 4, 1, 3, 5]


def test_resolve_raster_groups_validation_errors():
    with pytest.raises(ValueError, match="neurons_df must be provided"):
        _resolve_raster_groups(
            6, RasterGrouping(group_key="group"), GroupStripOptions(show=False)
        )
    with pytest.raises(ValueError, match="not found"):
        _resolve_raster_groups(
            6,
            RasterGrouping(neurons_df=_df(), group_key="nope"),
            GroupStripOptions(show=False),
        )
    with pytest.raises(ValueError, match="not found"):
        _resolve_raster_groups(
            6,
            RasterGrouping(neurons_df=_df(), group_key="group"),
            GroupStripOptions(show=False, color_key="nope"),
        )


def test_resolve_raster_groups_short_dataframe_pads_unknown():
    """A dataframe shorter than the population labels the rest 'Unknown'."""
    groups = _resolve_raster_groups(
        8,
        RasterGrouping(neurons_df=_df(), group_key="group"),
        GroupStripOptions(show=False),
    )
    assert groups.group_labels[6:].tolist() == ["Unknown", "Unknown"]


def test_resolve_raster_groups_without_keys_is_identity():
    groups = _resolve_raster_groups(4, RasterGrouping(), GroupStripOptions(show=False))
    assert groups.groups is None
    assert groups.sorted_indices.tolist() == [0, 1, 2, 3]
    assert groups.boundaries == []


# --------------------------------------------------------------------------- #
# Spike colour / marker resolution
# --------------------------------------------------------------------------- #
def _spike_style(
    idx, n_neurons, spike_color="black", neuron_specs=None, group_key=None, labels=()
):
    """Call ``_resolve_spike_style`` with option objects built from scalars."""
    groups = _resolve_raster_groups(
        n_neurons, RasterGrouping(), GroupStripOptions(show=False)
    )
    groups.group_labels = np.array(labels, dtype=object)
    return _resolve_spike_style(
        RasterStyle(spike_color=spike_color, neuron_specs=neuron_specs),
        RasterGrouping(group_key=group_key),
        groups,
        np.asarray(idx),
        n_neurons,
    )


def test_spike_style_plain_colour_is_passed_through():
    idx = np.array([0, 1, 1])
    style = _spike_style(idx, 3, "red")
    assert style.c_array == "red"
    assert not style.per_neuron_colors
    assert style.sizes == 5.0


def test_spike_style_int_keyed_dict_colours_per_neuron_with_black_default():
    idx = np.array([0, 2, 1])
    style = _spike_style(idx, 3, {0: "red", 2: "blue"})
    assert style.c_array.tolist() == ["red", "blue", "black"]
    assert style.per_neuron_colors


def test_spike_style_group_dict_uses_group_labels():
    labels = np.array(["A", "B", "A"], dtype=object)
    idx = np.array([1, 0])
    style = _spike_style(idx, 3, {"A": "red", "B": "blue"}, None, "group", labels)
    assert style.c_array.tolist() == ["blue", "red"]


def test_spike_style_group_dict_without_group_key_warns_and_uses_black():
    with pytest.warns(UserWarning, match="group_key not set"):
        style = _spike_style([0], 2, {"A": "red"})
    assert style.c_array == "black"


def test_spike_style_sequence_length_must_match():
    with pytest.raises(ValueError, match="sequence length"):
        _spike_style([0], 2, ["red"])


def test_spike_style_specs_mixed_markers_set_multi_marker():
    """Different per-neuron markers cannot share one scatter call."""
    specs = [NeuronSpec(color="red", marker="o"), {"color": "blue", "marker": "x"}]
    idx = np.array([0, 1, 1])
    style = _spike_style(idx, 2, "black", specs)
    assert style.multi_marker
    assert style.marker_list.tolist() == ["o", "x", "x"]
    assert style.color_list == ["red", "blue", "blue"]


def test_spike_style_specs_missing_neuron_falls_back_to_defaults():
    """Neurons without a spec get black, the default marker and size."""
    style = _spike_style([0, 1], 2, "black", {0: NeuronSpec(marker="s", markersize=9)})
    assert style.marker_list.tolist() == ["s", "."]
    assert style.size_list.tolist() == [9, 5.0]
    assert style.color_list == ["black", "black"]


# --------------------------------------------------------------------------- #
# Strip layout
# --------------------------------------------------------------------------- #
def _strip_figure(strip: GroupStripOptions, df=None, group_key="group"):
    df = _df() if df is None else df
    groups = _resolve_raster_groups(
        6, RasterGrouping(neurons_df=df, group_key=group_key), strip
    )
    fig, ax = plt.subplots()
    ax.set_ylim(-0.5, 5.5)
    colors = _draw_group_strip(ax, groups, df, group_key, strip, 6)
    return fig, ax, colors


def test_group_strip_adds_axes_with_one_patch_per_neuron():
    fig, _, colors = _strip_figure(GroupStripOptions())
    assert len(fig.axes) == 2
    assert len(fig.axes[1].patches) == 6
    assert not colors.use_subgroups
    assert set(colors.base_colors) == {"A", "B"}


def test_group_strip_subgroups_when_colour_key_differs_from_group_key():
    _, _, colors = _strip_figure(GroupStripOptions(color_key="sub"))
    assert colors.use_subgroups
    # Subgroup colours are keyed by (top group, subgroup).
    assert ("A", "x") in colors.subgroup_colors


def test_group_strip_side_controls_axes_position():
    """The strip axes sit right of the raster by default, left on request."""
    fig_r, ax_r, _ = _strip_figure(GroupStripOptions(side="right"))
    fig_l, ax_l, _ = _strip_figure(GroupStripOptions(side="left"))
    assert fig_r.axes[1].get_position().x0 >= ax_r.get_position().x1 - 1e-9
    assert fig_l.axes[1].get_position().x1 <= ax_l.get_position().x0 + 1e-9


def test_group_strip_requires_dataframe_and_key():
    groups = _resolve_raster_groups(6, RasterGrouping(), GroupStripOptions(show=False))
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="neurons_df must be provided"):
        _draw_group_strip(ax, groups, None, "group", GroupStripOptions(), 6)
    with pytest.raises(ValueError, match="color_key or grouping.group_key"):
        _draw_group_strip(ax, groups, _df(), None, GroupStripOptions(), 6)
    with pytest.raises(ValueError, match="not found"):
        _draw_group_strip(
            ax, groups, _df(), "group", GroupStripOptions(color_key="nope"), 6
        )


# --------------------------------------------------------------------------- #
# Axes creation and rate panel
# --------------------------------------------------------------------------- #
def test_create_raster_axes_reuses_given_axes_without_rate_panel():
    _, ax = plt.subplots()
    ax_raster, ax_rate = _create_raster_axes(ax, 10, False)
    assert ax_raster is ax and ax_rate is None


def test_create_raster_axes_with_rate_panel_ignores_ax_and_warns():
    _, ax = plt.subplots()
    with pytest.warns(UserWarning, match="ax argument is ignored"):
        ax_raster, ax_rate = _create_raster_axes(ax, 10, True)
    assert ax_raster is not ax and ax_rate is not None
    assert ax_raster.figure is ax_rate.figure


def test_resolve_total_rate_modes():
    spikes = _spikes()
    t = np.arange(50.0)
    assert _resolve_total_rate(False, spikes, t, 1.0, 10.0) is None
    given = np.linspace(0, 1, 50)
    # (T, 1) arrays are squeezed to 1D.
    out = _resolve_total_rate(given[:, None], spikes, t, 1.0, 10.0)
    assert out.shape == (50,)
    computed = _resolve_total_rate(True, spikes, t, 1.0, 10.0)
    assert computed.shape == (50,) and np.all(computed >= 0)
    with pytest.raises(ValueError, match="length"):
        _resolve_total_rate(given[:10], spikes, t, 1.0, 10.0)
    with pytest.raises(ValueError, match="1D"):
        _resolve_total_rate(np.zeros((50, 2)), spikes, t, 1.0, 10.0)


def test_draw_rate_panel_total_and_group_lines():
    """Group lines are drawn first (alpha 0.45), the total on top (black)."""
    spikes = _spikes()
    t = np.arange(50.0)
    groups = _resolve_raster_groups(
        6,
        RasterGrouping(neurons_df=_df(), group_key="group"),
        GroupStripOptions(show=False),
    )
    fig, (ax_raster, ax_rate) = plt.subplots(2, 1)
    _draw_rate_panel(
        ax_raster,
        ax_rate,
        t,
        spikes,
        1.0,
        "t (ms)",
        RatePanelOptions(total=True, per_group=True, window_ms=5.0),
        groups,
        RasterStyle(),
        GroupStripOptions(),
    )
    assert len(ax_rate.lines) == 3  # two groups + total
    assert ax_rate.lines[-1].get_color() == "black"
    assert ax_rate.get_xlabel() == "t (ms)"
    assert ax_raster.get_xlabel() == ""
    assert ax_rate.get_ylabel() == "Rate (Hz)"


def test_draw_rate_panel_group_array_must_match_group_count():
    groups = _resolve_raster_groups(
        6,
        RasterGrouping(neurons_df=_df(), group_key="group"),
        GroupStripOptions(show=False),
    )
    _, (ax_raster, ax_rate) = plt.subplots(2, 1)
    with pytest.raises(ValueError, match="number of groups"):
        _draw_rate_panel(
            ax_raster,
            ax_rate,
            np.arange(50.0),
            _spikes(),
            1.0,
            "t",
            RatePanelOptions(per_group=np.zeros((50, 5))),
            groups,
            RasterStyle(),
            GroupStripOptions(),
        )
