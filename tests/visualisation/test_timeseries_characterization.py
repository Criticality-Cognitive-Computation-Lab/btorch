"""Characterization tests for ``plot_raster`` and ``plot_neuron_traces``.

These tests pin the *current* observable behaviour of the two large plotting
functions so that internal refactors can be verified to be behaviour
preserving.  Two complementary layers are used:

1. Explicit, readable assertions on the main option combinations (number of
   axes, artist counts, labels, titles, limits, returned object types).
2. A structural "signature" snapshot for many option combinations.  The
   signature of a figure is a JSON-friendly description of every axes (visibility,
   title, labels, limits, artist counts, checksums of line/scatter data and of
   scatter colours, legend entries, text strings).  Signatures are compared
   against ``timeseries_characterization.golden`` which was recorded from
   the unrefactored implementation.

To regenerate the golden file after an *intentional* behaviour change run::

    BTORCH_UPDATE_GOLDEN=1 pytest \
        tests/visualisation/test_timeseries_characterization.py
"""

from __future__ import annotations

import json
import os
import warnings
import zlib
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from btorch.visualisation.timeseries import (  # noqa: E402
    NeuronSpec,
    SimulationStates,
    TracePlotFormat,
    plot_neuron_traces,
    plot_raster,
)


GOLDEN_PATH = Path(__file__).with_name("timeseries_characterization.golden")


# --------------------------------------------------------------------------- #
# Signature helpers
# --------------------------------------------------------------------------- #
def _checksum(arr) -> float:
    """Order-sensitive scalar checksum of a numeric array (NaN-safe)."""
    a = np.nan_to_num(np.asarray(arr, dtype=float).ravel(), nan=-12345.0)
    if a.size == 0:
        return 0.0
    w = np.arange(1, a.size + 1, dtype=float)
    return round(float((a * w).sum()), 4)


def _color_hash(colors) -> int:
    """Stable hash of an RGBA array / list of colours."""
    colors = np.asarray(colors)
    return zlib.crc32(np.round(colors.astype(float), 4).tobytes()) if colors.size else 0


def _layout(ax) -> dict | None:
    """Grid geometry of an axes (None for axes placed with ``add_axes``)."""
    spec = ax.get_subplotspec()
    if spec is None:
        return None
    gs = spec.get_gridspec()
    ratios = gs.get_height_ratios()
    return {
        "grid": [gs.nrows, gs.ncols],
        "rows": [spec.rowspan.start, spec.rowspan.stop],
        "cols": [spec.colspan.start, spec.colspan.stop],
        "height_ratios": None
        if ratios is None
        else [round(float(r), 4) for r in ratios],
    }


def _ax_signature(ax, include_position: bool = False) -> dict:
    """Describe an axes: labels, limits, artists and data checksums."""
    legend = ax.get_legend()
    sig = {
        "layout": _layout(ax),
        "visible": bool(ax.get_visible()),
        "axis_off": not ax.axison,
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xlim": [round(float(v), 4) for v in ax.get_xlim()],
        "ylim": [round(float(v), 4) for v in ax.get_ylim()],
        "n_lines": len(ax.lines),
        "n_collections": len(ax.collections),
        "n_patches": len(ax.patches),
        "n_images": len(ax.images),
        "texts": [t.get_text() for t in ax.texts],
        "legend": [t.get_text() for t in legend.get_texts()] if legend else None,
        "lines": [
            [
                _checksum(ln.get_xdata()),
                _checksum(ln.get_ydata()),
                to_hex(ln.get_color()),
                ln.get_linestyle(),
                round(float(ln.get_linewidth()), 3),
                round(float(ln.get_alpha() if ln.get_alpha() is not None else 1.0), 3),
                ln.get_label(),
            ]
            for ln in ax.lines
        ],
        "collections": [],
        "patches": [
            [
                round(float(p.get_x()), 4),
                round(float(p.get_y()), 4),
                round(float(p.get_width()), 4),
                round(float(p.get_height()), 4),
                to_hex(p.get_facecolor()),
            ]
            for p in ax.patches
            if hasattr(p, "get_x")
        ],
    }
    for coll in ax.collections:
        offsets = coll.get_offsets()
        sizes = np.asarray(getattr(coll, "get_sizes", lambda: [])(), dtype=float)
        sig["collections"].append(
            {
                "type": type(coll).__name__,
                "n": int(len(offsets)),
                "offsets": _checksum(np.asarray(offsets)),
                "sizes": _checksum(sizes),
                "facecolors": _color_hash(coll.get_facecolor()),
            }
        )
    if include_position:
        # Only deterministic when no tight_layout is involved (raster).
        sig["position"] = [round(float(v), 4) for v in ax.get_position().bounds]
    # Scatter calls per marker follow set-iteration order (hash-randomised for
    # strings), so the collection order is not stable between interpreter runs.
    sig["collections"].sort(key=lambda c: json.dumps(c, sort_keys=True))
    return sig


def _fig_signature(
    fig: Figure, include_size: bool = True, include_position: bool = False
) -> dict:
    sig = {
        "n_axes": len(fig.axes),
        "axes": [_ax_signature(a, include_position) for a in fig.axes],
    }
    if include_size:
        sig["figsize"] = [round(float(v), 2) for v in fig.get_size_inches()]
    return sig


# --------------------------------------------------------------------------- #
# Deterministic test data
# --------------------------------------------------------------------------- #
N_T, N_N = 60, 12


def _spikes(seed: int = 0, n_t: int = N_T, n_n: int = N_N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n_t, n_n)) > 0.85).astype(np.float32)


def _neurons_df(n_n: int = N_N) -> pd.DataFrame:
    """Two top groups (A/B), each with two subtypes, interleaved on purpose."""
    return pd.DataFrame(
        {
            "group": ["B", "A"] * (n_n // 2),
            "sub": ["x", "x", "y", "y"] * (n_n // 4),
        }
    )


def _trace_data(n_t=40, n_n=6, batch=None, seed=1):
    rng = np.random.default_rng(seed)
    shape = (n_t, n_n) if batch is None else (n_t, batch, n_n)
    voltage = rng.normal(-60.0, 5.0, shape).astype(np.float32)
    spikes = (rng.random(shape) > 0.9).astype(np.float32)
    asc = rng.normal(0, 1, shape).astype(np.float32)
    psc = rng.normal(0, 1, shape).astype(np.float32)
    epsc = np.abs(psc)
    ipsc = -np.abs(psc)
    inp = rng.normal(0, 0.5, shape).astype(np.float32)
    return dict(
        voltage=voltage, spikes=spikes, asc=asc, psc=psc, epsc=epsc, ipsc=ipsc, inp=inp
    )


# --------------------------------------------------------------------------- #
# Scenario registries (name -> callable returning a signature)
# --------------------------------------------------------------------------- #
def _raster_result(ret, include_size=True):
    """Signature of whatever plot_raster returned (Axes or (Axes, Axes))."""
    axes = ret if isinstance(ret, tuple) else (ret,)
    sig = _fig_signature(axes[0].figure, include_size, include_position=True)
    sig["returned"] = [type(a).__name__ for a in axes]
    sig["returned_idx"] = [axes[0].figure.axes.index(a) for a in axes]
    return sig


def _raster_scenarios() -> dict:
    sp = _spikes()
    df = _neurons_df()
    rate_arr = np.linspace(0, 10, N_T)
    group_rate_arr = np.stack([np.linspace(0, 5, N_T), np.linspace(5, 0, N_T)], 1)

    def s(**kw):
        return lambda: _raster_result(plot_raster(sp, **kw))

    sc = {
        "plain": s(),
        "dt_title_labels": s(dt=0.5, title="T", xlabel="X", ylabel="Y"),
        "times": s(times=np.arange(N_T) * 2.0 + 5.0),
        "marker_color": s(spike_color="red", marker="|", marker_size=9.0),
        "tracks_events_regions": s(
            events=[10, 20], regions=[(30, 40)], show_tracks=True
        ),
        "events_regions_dict": s(
            events={"a": [10], "b": [20, 25]},
            regions={"r": [(5, 8), (30, 40)]},
            event_kwargs={"color": "blue"},
            region_kwargs={"color": "green", "alpha": 0.5},
        ),
        "rate_true": s(rate=True, dt=1.0),
        "rate_array": s(rate=rate_arr, xlabel="t"),
        "rate_array_col": s(rate=rate_arr[:, None]),
        "grouped": s(neurons_df=df, group_key="group"),
        "grouped_no_sort": s(neurons_df=df, group_key="group", sort_neurons=False),
        "grouped_group_sort": s(
            neurons_df=df, group_key="group", group_sort=["B", "A", "Z"]
        ),
        "grouped_no_separators": s(
            neurons_df=df, group_key="group", show_group_separators=False
        ),
        "grouped_left_labels": s(
            neurons_df=df,
            group_key="group",
            group_strip_side="left",
            separator_style={"color": "red", "linestyle": ":"},
        ),
        "grouped_color_dict": s(
            neurons_df=df, group_key="group", spike_color={"A": "red", "B": "blue"}
        ),
        "color_dict_ints": s(spike_color={0: "red", 3: "green"}),
        "color_dict_no_group_warns": s(spike_color={"A": "red"}),
        "color_sequence": s(spike_color=["red", "blue"] * (N_N // 2)),
        "group_rate_true": s(neurons_df=df, group_key="group", group_rate=True),
        "group_rate_true_and_rate": s(
            neurons_df=df, group_key="group", group_rate=True, rate=True
        ),
        "group_rate_dict": s(
            neurons_df=df,
            group_key="group",
            group_rate={"A": rate_arr, "B": rate_arr * 2},
        ),
        "group_rate_array": s(
            neurons_df=df, group_key="group", group_rate=group_rate_arr
        ),
        "group_rate_spike_color_dict": s(
            neurons_df=df,
            group_key="group",
            group_rate=True,
            spike_color={"A": "red", "B": "blue"},
        ),
        "specs_same_marker": s(
            neuron_specs={0: {"color": "red", "markersize": 12}, 2: NeuronSpec("c")}
        ),
        "specs_list_mixed_markers": s(
            neuron_specs=[
                NeuronSpec(color="red", marker="o", markersize=10),
                {"color": "blue", "marker": "x", "markersize": 15},
            ]
        ),
        "strip_group": s(neurons_df=df, group_key="group", show_group_strip=True),
        "strip_group_left_nolegend": s(
            neurons_df=df,
            group_key="group",
            show_group_strip=True,
            group_strip_side="left",
            group_strip_legend=False,
        ),
        "strip_sub_top_sub": s(
            neurons_df=df,
            group_key="group",
            group_color_key="sub",
            show_group_strip=True,
        ),
        "strip_sub_top": s(
            neurons_df=df,
            group_key="group",
            group_color_key="sub",
            show_group_strip=True,
            group_label_mode="top",
        ),
        "strip_sub_sub": s(
            neurons_df=df,
            group_key="group",
            group_color_key="sub",
            show_group_strip=True,
            group_label_mode="sub",
            group_strip_side="left",
        ),
        "strip_color_key_only": s(
            neurons_df=df, group_color_key="sub", show_group_strip=True
        ),
        "strip_kwargs": s(
            neurons_df=df,
            group_key="group",
            show_group_strip=True,
            group_strip_kwargs={"width": 0.1, "alpha": 0.5, "label_x": 0.9},
            strip_cmap="tab20",
        ),
        "strip_mixed_markers": s(
            neurons_df=df,
            group_key="group",
            show_group_strip=True,
            neuron_specs=[NeuronSpec(marker="o"), NeuronSpec(marker="x")],
        ),
        "strip_same_specs": s(
            neurons_df=df,
            group_key="group",
            show_group_strip=True,
            neuron_specs=[NeuronSpec(marker="o", markersize=11)],
        ),
        "strip_and_rate": s(
            neurons_df=df, group_key="group", show_group_strip=True, rate=True
        ),
        "everything": s(
            neurons_df=df,
            group_key="group",
            group_color_key="sub",
            show_group_strip=True,
            rate=True,
            group_rate=True,
            events=[5],
            regions=[(10, 20)],
            show_tracks=True,
            title="all",
        ),
        # Pinned quirk: with a strip, per-neuron colour sequences are ignored
        # (spikes are drawn later with strip colours).
        "strip_with_color_sequence": s(
            neurons_df=df,
            group_key="group",
            show_group_strip=True,
            spike_color=["red", "blue"] * (N_N // 2),
        ),
        # Pinned quirk: group_rate without group_key yields an empty rate panel.
        "group_rate_without_group_key": s(group_rate=True),
        "torch_input": lambda: _raster_result(plot_raster(torch.tensor(sp))),
        "empty_spikes": lambda: _raster_result(plot_raster(np.zeros((N_T, N_N)))),
    }

    def on_ax():
        fig, ax = plt.subplots(figsize=(4, 3))
        ret = plot_raster(sp, ax=ax)
        assert ret is ax
        return _raster_result(ret)

    def on_ax_with_rate_warns():
        fig, ax = plt.subplots()
        with pytest.warns(UserWarning, match="ax argument is ignored"):
            ret = plot_raster(sp, ax=ax, rate=True)
        assert isinstance(ret, tuple) and ret[0] is not ax
        return _raster_result(ret)

    sc["on_ax"] = on_ax
    sc["on_ax_with_rate_warns"] = on_ax_with_rate_warns
    return sc


def _traces_result(ret, include_size=True):
    if isinstance(ret, dict):
        return {
            "type": "dict",
            "keys": list(ret.keys()),
            "figs": {k: _fig_signature(v, include_size) for k, v in ret.items()},
        }
    return {"type": type(ret).__name__, "fig": _fig_signature(ret, include_size)}


def _traces_scenarios() -> dict:
    d = _trace_data()
    d3 = _trace_data(batch=3)
    ext = np.random.default_rng(5).normal(size=(40, 6, 3)).astype(np.float32)
    base = dict(voltage=d["voltage"])

    def s(include_size=True, **kw):
        def run():
            return _traces_result(plot_neuron_traces(**kw), include_size)

        return run

    full = dict(
        voltage=d["voltage"],
        spikes=d["spikes"],
        asc=d["asc"],
        psc=d["psc"],
        epsc=d["epsc"],
        ipsc=d["ipsc"],
        input=d["inp"],
    )
    sc = {
        "voltage_only": s(**base),
        "full_default": s(**full),
        "neuron_indices_dt": s(**base, neuron_indices=[1, 3], dt=0.5),
        "sample_size": s(**base, sample_size=3, seed=7),
        "thresholds_scalar": s(**base, v_threshold=-50.0, v_reset=-65.0),
        "thresholds_vector": s(
            **base, neuron_indices=[0, 2], v_threshold=np.arange(6) * -1.0 - 40
        ),
        "no_auto_width": s(**base, auto_width=False),
        "hide_panels": s(**full, show_asc=False, show_psc=False),
        "hide_voltage": s(**full, show_voltage=False),
        "hide_all_fallback_voltage": s(
            **base, show_voltage=False, show_asc=False, show_psc=False
        ),
        "asc_only_data_missing": s(**base, show_asc=True, show_psc=True),
        "side_labels_seq": s(**base, neuron_labels=["n0", "n1", "n2"]),
        "side_labels_callable": s(
            **base, neuron_indices=[2, 4], neuron_labels=lambda i: f"id{i}"
        ),
        "top_labels": s(
            include_size=False,
            **base,
            neuron_labels=lambda i: f"neuron number {i}",
            neuron_label_position="top",
        ),
        "top_labels_grid": s(
            include_size=False,
            **full,
            neuron_indices=[0, 1, 2],
            neurons_per_row=2,
            neuron_labels=["a", "b", "c"],
            neuron_label_position="top",
        ),
        "neurons_per_row_2": s(**full, neuron_indices=[0, 1, 2], neurons_per_row=2),
        "neurons_per_row_3_psc_asc": s(
            **full, neuron_indices=[0, 1, 2, 3], neurons_per_row=3
        ),
        "specs_list": s(
            **full,
            neuron_indices=[0, 1, 2],
            neuron_specs=[
                NeuronSpec(label="first", color="red", linestyle="--", alpha=0.5),
                {"color": {"voltage": "green", "psc": "blue"}, "linewidth": 2.0},
            ],
        ),
        "specs_single": s(**base, neuron_specs=NeuronSpec(color="purple", label="L")),
        "specs_dict": s(**base, neuron_specs={"label": "D", "linewidth": 1.5}),
        "psc_multi": s(
            voltage=d["voltage"],
            psc=ext,
            psc_labels=["ampa", "nmda"],
            spikes=d["spikes"],
        ),
        "psc_multi_default_labels": s(voltage=d["voltage"], psc=ext),
        "batch_default": s(voltage=d3["voltage"], asc=d3["asc"], spikes=d3["spikes"]),
        "batch_idx_2": s(
            voltage=d3["voltage"], asc=d3["asc"], epsc=d3["epsc"], batch_idx=2
        ),
        "separate_full": s(**full, separate_figures=True, neuron_labels=["a", "b"]),
        "separate_top_labels": s(
            **base,
            separate_figures=True,
            neuron_labels=["a", "b"],
            neuron_label_position="top",
            v_threshold=-50,
        ),
        "separate_psc_multi": s(voltage=d["voltage"], psc=ext, separate_figures=True),
        "separate_no_panels": s(**base, separate_figures=True, show_voltage=False),
        "states_dataclass": lambda: _traces_result(
            plot_neuron_traces(
                states=SimulationStates(
                    voltage=d["voltage"],
                    dt=2.0,
                    asc=d["asc"],
                    spikes=d["spikes"],
                    v_threshold=-50.0,
                )
            )
        ),
        "format_dataclass": lambda: _traces_result(
            plot_neuron_traces(
                voltage=d["voltage"],
                spikes=d["spikes"],
                format=TracePlotFormat(
                    neuron_indices=[0, 1],
                    show_spikes_on_voltage=False,
                    auto_width=False,
                    figsize_per_neuron=(7, 3),
                    neurons_per_row=2,
                ),
            )
        ),
        "format_separate": lambda: _traces_result(
            plot_neuron_traces(
                voltage=d["voltage"],
                asc=d["asc"],
                format=TracePlotFormat(separate_figures=True, sample_size=2),
            )
        ),
        # Pinned quirk: the separate-figures path ignores neuron_specs.
        "separate_with_specs": s(
            **full,
            neuron_indices=[0, 1],
            separate_figures=True,
            neuron_specs=[
                NeuronSpec(label="L0", color="red", linestyle="--"),
                NeuronSpec(label="L1", color="blue"),
            ],
        ),
        "torch_input": s(
            voltage=torch.tensor(d["voltage"]), asc=torch.tensor(d["asc"])
        ),
    }
    return sc


def _collect_all() -> dict:
    out = {}
    for name, fn in _raster_scenarios().items():
        out[f"raster/{name}"] = fn
    for name, fn in _traces_scenarios().items():
        out[f"traces/{name}"] = fn
    return out


def _run(name: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.close("all")
        try:
            return json.loads(json.dumps(_collect_all()[name]()))
        finally:
            plt.close("all")


_ALL_NAMES = list(_collect_all().keys())


def _load_golden() -> dict:
    if GOLDEN_PATH.exists():
        return json.loads(GOLDEN_PATH.read_text())
    return {}


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_signature_matches_golden(name):
    """Every scenario reproduces the recorded structural signature."""
    sig = _run(name)
    if os.environ.get("BTORCH_UPDATE_GOLDEN"):
        golden = _load_golden()
        golden[name] = sig
        lines = [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in sorted(golden.items())]
        GOLDEN_PATH.write_text("{\n" + ",\n".join(lines) + "\n}\n")
        return
    golden = _load_golden()
    assert name in golden, f"no golden entry for {name}; regenerate golden file"
    assert sig == golden[name]


# --------------------------------------------------------------------------- #
# Explicit, readable assertions on the most important behaviours
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def test_raster_plain_returns_single_axes():
    """Without rate panels a single Axes is returned, with one scatter."""
    ax = plot_raster(_spikes())
    assert not isinstance(ax, tuple)
    assert len(ax.figure.axes) == 1
    assert len(ax.collections) == 1
    n_spikes = int(_spikes().sum())
    assert len(ax.collections[0].get_offsets()) == n_spikes
    assert ax.get_xlim() == (0.0, N_T - 1.0)
    assert ax.get_ylim() == (-0.5, N_N - 0.5)
    assert ax.get_xlabel() == "Time (ms)"
    assert ax.get_ylabel() == "Neuron Index"
    assert ax.get_title().startswith("Spike raster Fired ")
    assert f"Spikes {n_spikes}" in ax.get_title()
    # Spike-count annotation in the upper-left corner.
    assert [t.get_text() for t in ax.texts] == [f"N={n_spikes}"]


def test_raster_rate_returns_pair_and_moves_xlabel():
    """Rate=True gives (raster, rate) axes; x label lives on the rate axis."""
    ret = plot_raster(_spikes(), rate=True, xlabel="t (ms)")
    assert isinstance(ret, tuple) and len(ret) == 2
    ax_r, ax_rate = ret
    assert len(ax_r.figure.axes) == 2
    assert ax_r.get_xlabel() == ""
    assert ax_rate.get_xlabel() == "t (ms)"
    assert ax_rate.get_ylabel() == "Rate (Hz)"
    assert len(ax_rate.lines) == 1  # only the population rate


def test_raster_group_rate_draws_one_line_per_group_plus_total():
    """group_rate=True + rate=True: one line per group then the total line."""
    _, ax_rate = plot_raster(
        _spikes(),
        neurons_df=_neurons_df(),
        group_key="group",
        group_rate=True,
        rate=True,
    )
    assert len(ax_rate.lines) == 3
    assert [ln.get_label() for ln in ax_rate.lines[:2]] == ["A", "B"]


def test_raster_group_separators_and_labels():
    """Grouping draws separators between groups and text labels at the side."""
    ax = plot_raster(_spikes(), neurons_df=_neurons_df(), group_key="group")
    # Two groups -> only the inner boundary gets an axhline.
    assert len(ax.lines) == 1
    labels = [t.get_text() for t in ax.texts]
    assert "A" in labels and "B" in labels


def test_raster_group_strip_adds_axes_and_legend():
    """The group strip is an extra axes with one patch per neuron and
    legend."""
    ax = plot_raster(
        _spikes(), neurons_df=_neurons_df(), group_key="group", show_group_strip=True
    )
    fig = ax.figure
    assert len(fig.axes) == 2
    cax = fig.axes[1]
    assert len(cax.patches) == N_N
    assert cax.get_legend() is not None
    # Pinned current behaviour (not necessarily intended): the default
    # "top_sub" legend mode joins top/sub labels even without subgroups.
    assert [t.get_text() for t in cax.get_legend().get_texts()] == ["A / A", "B / B"]


def test_raster_mixed_markers_split_into_one_scatter_per_marker():
    ax = plot_raster(
        _spikes(),
        neuron_specs=[NeuronSpec(marker="o"), NeuronSpec(marker="x")],
    )
    # Neurons without a spec fall back to the default "." marker: 3 markers.
    assert len(ax.collections) == 3


def test_raster_error_contracts():
    sp = _spikes()
    with pytest.raises(ValueError, match="2D"):
        plot_raster(sp[0])
    with pytest.raises(ValueError, match="neurons_df must be provided"):
        plot_raster(sp, group_key="group")
    with pytest.raises(ValueError, match="not found"):
        plot_raster(sp, neurons_df=_neurons_df(), group_key="nope")
    with pytest.raises(ValueError, match="sequence length"):
        plot_raster(sp, spike_color=["red"])
    with pytest.raises(ValueError, match="neurons_df must be provided for group"):
        plot_raster(sp, show_group_strip=True)
    with pytest.raises(ValueError, match="group_color_key or group_key"):
        plot_raster(sp, neurons_df=_neurons_df(), show_group_strip=True)
    with pytest.raises(ValueError, match="rate length"):
        plot_raster(sp, rate=np.zeros(3))
    with pytest.raises(ValueError, match="rate must be 1D"):
        plot_raster(sp, rate=np.zeros((N_T, 2)))
    with pytest.raises(ValueError, match=r"\(T, G\)"):
        plot_raster(
            sp,
            neurons_df=_neurons_df(),
            group_key="group",
            group_rate=np.zeros(N_T),
        )
    with pytest.raises(ValueError, match="number of groups"):
        plot_raster(
            sp,
            neurons_df=_neurons_df(),
            group_key="group",
            group_rate=np.zeros((N_T, 5)),
        )
    with pytest.raises(ValueError, match="1D and match"):
        plot_raster(
            sp,
            neurons_df=_neurons_df(),
            group_key="group",
            group_rate={"A": np.zeros(3)},
        )
    with pytest.raises(ValueError, match="times"):
        plot_raster(sp, times=[0, 1])


def test_raster_warns_when_color_dict_without_group_key():
    with pytest.warns(UserWarning, match="group_key not set"):
        plot_raster(_spikes(), spike_color={"A": "red"})


def test_traces_combined_returns_figure_with_expected_grid():
    d = _trace_data()
    fig = plot_neuron_traces(
        voltage=d["voltage"], asc=d["asc"], psc=d["psc"], neuron_indices=[0, 1]
    )
    assert isinstance(fig, Figure)
    # 2 neurons (rows) x 3 panels (voltage, asc, psc)
    assert len(fig.axes) == 6
    titles = [a.get_title() for a in fig.axes]
    assert titles == [
        "Voltage",
        "Afterspike Current",
        "Postsynaptic Current",
        "",
        "",
        "",
    ]
    assert [a.get_xlabel() for a in fig.axes] == ["", "", "", *["Time (ms)"] * 3]
    assert fig.axes[0].get_ylabel() == "V (mV)"
    assert fig.axes[1].get_ylabel() == "ASC (pA)"
    assert fig.axes[2].get_ylabel() == "PSC (pA)"


def test_traces_default_plots_first_five_neurons():
    fig = plot_neuron_traces(voltage=_trace_data(n_n=8)["voltage"])
    assert len(fig.axes) == 5


def test_traces_separate_figures_returns_dict_per_trace_type():
    d = _trace_data()
    figs = plot_neuron_traces(
        voltage=d["voltage"],
        asc=d["asc"],
        psc=d["psc"],
        neuron_indices=[0, 1],
        separate_figures=True,
    )
    assert isinstance(figs, dict)
    assert list(figs) == ["voltage", "asc", "psc"]
    assert all(isinstance(f, Figure) and len(f.axes) == 2 for f in figs.values())
    assert figs["voltage"].axes[0].get_title() == "Voltage Traces"
    assert figs["asc"].axes[0].get_title() == "Afterspike Current"
    assert figs["psc"].axes[0].get_title() == "Postsynaptic Current"
    assert figs["psc"].axes[-1].get_xlabel() == "Time (ms)"


def test_traces_neurons_per_row_hides_unused_axes():
    d = _trace_data()
    fig = plot_neuron_traces(
        voltage=d["voltage"], neuron_indices=[0, 1, 2], neurons_per_row=2
    )
    assert len(fig.axes) == 4
    assert [a.get_visible() for a in fig.axes] == [True, True, True, False]


def test_traces_top_labels_add_label_axes():
    d = _trace_data()
    fig = plot_neuron_traces(
        voltage=d["voltage"],
        neuron_indices=[0, 1],
        neuron_labels=["a", "b"],
        neuron_label_position="top",
    )
    # One hidden-axis label row per neuron row + one trace row per neuron.
    assert len(fig.axes) == 4
    assert [a.axison for a in fig.axes] == [False, True, False, True]
    assert [t.get_text() for a in fig.axes[::2] for t in a.texts] == ["a", "b"]


def test_traces_threshold_reference_lines_and_legend():
    d = _trace_data()
    fig = plot_neuron_traces(
        voltage=d["voltage"],
        neuron_indices=[0],
        v_threshold=-50.0,
        v_reset=-65.0,
    )
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["V_th", "V_reset"]


def test_traces_error_contracts():
    d = _trace_data()
    with pytest.raises(ValueError, match="voltage is required"):
        plot_neuron_traces()
    with pytest.raises(ValueError, match="neurons_per_row"):
        plot_neuron_traces(voltage=d["voltage"], neurons_per_row=0)
    ext = np.zeros((40, 6, 2), dtype=np.float32)
    for kw in ("epsc", "ipsc", "input"):
        with pytest.raises(ValueError, match="must be None"):
            plot_neuron_traces(voltage=d["voltage"], psc=ext, **{kw: d["epsc"]})
    with pytest.raises(ValueError, match="out of bounds"):
        plot_neuron_traces(voltage=_trace_data(batch=2)["voltage"], batch_idx=5)
    with pytest.raises(ValueError, match="v_threshold"):
        plot_neuron_traces(voltage=d["voltage"], v_threshold=[1.0, 2.0])
    with pytest.raises(ValueError, match="2D, 3D, or 4D"):
        plot_neuron_traces(voltage=np.zeros(5))


def test_traces_batched_psc_is_misdetected_as_multi_component():
    """Pinned pre-existing bug: batched 3D psc is treated as (T, N, n_psc).

    With 3D voltage ``(time, batch, neurons)`` the neuron count is read from
    ``voltage.shape[1]`` (the batch size), so a batched ``psc`` always matches
    the "extra dimension" test and later fails indexing neurons.  This is not
    fixed by the behaviour-preserving refactor.
    """
    d3 = _trace_data(batch=3)
    with pytest.raises(IndexError):
        plot_neuron_traces(voltage=d3["voltage"], psc=d3["psc"], batch_idx=2)
