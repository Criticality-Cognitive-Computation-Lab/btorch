"""Timeseries visualization utilities for spike trains and continuous traces.

This module provides plotting functions for:
- Spike raster plots with grouping and styling options
- Continuous timeseries traces (voltage, currents)
- Frequency spectrum analysis
- Log-binned histograms

The raster plot supports neuron grouping, color-coded strips, population
firing rates, and event/region annotations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import ceil
from textwrap import wrap
from typing import Any, Callable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import patches as mpatches
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from ..analysis.spiking import compute_raster, compute_spectrum, firing_rate
from ..analysis.statistics import compute_log_hist


def _to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _resolve_per_neuron_values(
    value: float | Sequence[float] | np.ndarray | torch.Tensor | None,
    neuron_indices: list[int],
    n_neurons: int,
    name: str,
) -> list[float | None]:
    """Resolve scalar or vector input to per-plotted-neuron values."""
    n_plot = len(neuron_indices)
    if value is None:
        return [None] * n_plot

    if np.isscalar(value):
        return [float(value)] * n_plot

    arr = _to_numpy(value)
    if arr.ndim == 0:
        return [float(arr)] * n_plot
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a scalar or 1D array-like, got shape {arr.shape}."
        )

    if arr.shape[0] == n_neurons:
        selected = arr[np.asarray(neuron_indices, dtype=int)]
    elif arr.shape[0] == n_plot:
        selected = arr
    else:
        raise ValueError(
            f"{name} must be a scalar, length {n_neurons} (all neurons), or "
            f"length {n_plot} (plotted neurons), got length {arr.shape[0]}."
        )

    try:
        return [float(v) for v in selected]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc


def _get_time_axis(
    length: int, dt: float | None = None, times: Sequence[float] | None = None
) -> np.ndarray:
    if times is not None:
        if len(times) != length:
            raise ValueError(
                f"Length of times ({len(times)}) must match length of data ({length})."
            )
        return _to_numpy(times)

    if dt is None:
        dt = 1.0
    return np.arange(length) * dt


def _sample_cmap_colors(cmap_name: str, n: int) -> list[str]:
    if n <= 1:
        cmap = plt.get_cmap(cmap_name, 1)
        return [to_hex(cmap(0.5))]
    if n <= 10 and cmap_name.startswith("tab"):
        tab10 = plt.get_cmap("tab10")
        return [to_hex(tab10.colors[i]) for i in range(n)]
    # Sample bin centers instead of the endpoints so small group counts do not
    # collapse onto the first/last colors of listed palettes like tab20.
    cmap = plt.get_cmap(cmap_name, n)
    vals = (np.arange(n, dtype=float) + 0.5) / n
    return [to_hex(cmap(v)) for v in vals]


def _auto_raster_height(
    n_neurons: int,
    min_height: float = 3.5,
    max_height: float = 10.0,
    base_height: float = 4.0,
    log_scale: float = 0.8,
) -> float:
    """Compute a raster height that grows gently with neuron count."""
    n = max(int(n_neurons), 1)
    est_height = base_height + log_scale * np.log10(n)
    return float(np.clip(est_height, min_height, max_height))


def _build_group_color_maps(
    top_group_labels: list[str],
    sub_labels: list[str],
    use_subgroups: bool,
    palette_name: str,
    cmap_name: str,
    sub_hue_span: float,
    sub_val_span: float,
) -> tuple[dict[str, str], dict[tuple[str, str], str], dict[str, list[str]], list[str]]:
    top_groups_order = list(dict.fromkeys(top_group_labels))

    if use_subgroups:
        base_list = _sample_cmap_colors(palette_name, len(top_groups_order))
        base_colors = dict(zip(top_groups_order, base_list))
        subgroups_by_top: dict[str, list[str]] = {tg: [] for tg in top_groups_order}
        for top, sub in zip(top_group_labels, sub_labels):
            lst = subgroups_by_top[top]
            if sub not in lst:
                lst.append(sub)

        subgroup_colors: dict[tuple[str, str], str] = {}
        for tg in top_groups_order:
            subs = subgroups_by_top.get(tg, [])
            m = max(1, len(subs))
            base_rgb = np.array(to_rgb(base_colors[tg]))
            base_hsv = rgb_to_hsv(base_rgb)
            if m == 1:
                hue_offsets = [0.0]
                val_offsets = [0.0]
            else:
                hue_offsets = np.linspace(-sub_hue_span, sub_hue_span, m)
                val_offsets = np.linspace(-sub_val_span, sub_val_span, m)

            for sub, h_off, v_off in zip(subs, hue_offsets, val_offsets):
                hsv = base_hsv.copy()
                hsv[0] = (hsv[0] + h_off) % 1.0
                hsv[2] = float(np.clip(hsv[2] + v_off, 0.25, 1.0))
                subgroup_colors[(tg, sub)] = to_hex(hsv_to_rgb(hsv))

        return base_colors, subgroup_colors, subgroups_by_top, top_groups_order

    group_list = list(dict.fromkeys(sub_labels))
    base_list = _sample_cmap_colors(cmap_name, len(group_list))
    base_colors = dict(zip(group_list, base_list))
    subgroups_by_top = {g: [g] for g in group_list}
    subgroup_colors = {(g, g): base_colors[g] for g in group_list}
    return base_colors, subgroup_colors, subgroups_by_top, group_list


_LINE_MARKERS = ("x", "+", "|", "_", "1", "2", "3", "4")

_STRIP_DEFAULTS: dict[str, Any] = {
    "width": 0.06,
    "pad": 0.005,
    "alpha": 0.9,
    "label_fontsize": 7,
    "label_weight": "bold",
    "legend_fontsize": 6,
    "legend_ncol_threshold": 15,
    "min_label_distance": 0.02,
    "min_span_frac": 0.01,
    "strip_x0": 0.3,
    "strip_width": 0.4,
    "label_x": None,
    "label_gap": 0.05,
    "label_sep": " / ",
    "sub_hue_span": 0.12,
    "sub_val_span": 0.28,
    "left_extra_pad": 0.04,
}


def _marker_linewidth(marker: str) -> float:
    """Return the edge width needed for a marker (line markers need > 0)."""
    return 0.5 if marker in _LINE_MARKERS else 0


def _effective_dt(dt: float | None, t: np.ndarray) -> float:
    if dt is not None:
        return dt
    return t[1] - t[0] if len(t) > 1 else 1.0


@dataclass
class _RasterGroups:
    """Per-neuron group labels and the resulting plotting order."""

    group_labels: np.ndarray
    subgroup_labels: np.ndarray
    groups: list | None
    sorted_indices: np.ndarray
    boundaries: list[tuple[float, Any]]


@dataclass
class _SpikeStyle:
    """Resolved per-spike colours, sizes and markers for a raster."""

    c_array: Any
    marker: str
    sizes: Any
    marker_list: np.ndarray | None = None
    size_list: np.ndarray | None = None
    color_list: list | None = None
    multi_marker: bool = False


@dataclass
class _StripColors:
    use_subgroups: bool
    base_colors: dict[str, str]
    subgroup_colors: dict[tuple[str, str], str]


def _create_raster_axes(
    ax: Axes | None, n_neurons: int, with_rate_panel: bool
) -> tuple[Axes, Axes | None]:
    """Create (or reuse) the raster axes and the optional rate axes below."""
    raster_height = _auto_raster_height(n_neurons)
    raster_width = 8.0
    rate_height = 2.6  # keep rate panel at a stable height

    if with_rate_panel:
        if ax is not None:
            warnings.warn(
                "ax argument is ignored when rate/group_rate is enabled. "
                "Creating new figure."
            )
        _, (ax_raster, ax_rate) = plt.subplots(
            2,
            1,
            figsize=(raster_width, raster_height + rate_height),
            gridspec_kw={
                "height_ratios": [raster_height, rate_height],
                "hspace": 0.06,
            },
        )
        return ax_raster, ax_rate

    if ax is None:
        _, ax = plt.subplots(figsize=(raster_width, raster_height))
    return ax, None


def _labels_from_df(neurons_df: pd.DataFrame, column: str, n_neurons: int):
    """Copy a dataframe column into an object array of length ``n_neurons``."""
    labels = np.full(n_neurons, "Unknown", dtype=object)
    values = neurons_df[column].to_numpy()
    n_copy = min(n_neurons, len(values))
    labels[:n_copy] = values[:n_copy]
    labels[pd.isna(labels)] = "Unknown"
    return labels


def _order_neurons_by_group(
    group_labels: np.ndarray,
    subgroup_labels: np.ndarray,
    groups: list,
    by_subgroup: bool,
    sort_neurons: bool,
    n_neurons: int,
) -> tuple[np.ndarray, list[tuple[float, Any]]]:
    """Return plotting order and group boundaries (y coordinate, label)."""
    boundaries: list[tuple[float, Any]] = []
    sorted_indices = np.arange(n_neurons)

    if not sort_neurons:
        prev_g = None
        for i, idx in enumerate(sorted_indices):
            g = group_labels[idx]
            if i == 0:
                prev_g = g
            elif g != prev_g:
                boundaries.append((i - 0.5, prev_g))
                prev_g = g
        return sorted_indices, boundaries

    new_order = []
    current_y = 0
    for g in groups:
        g_indices = np.flatnonzero(group_labels == g)
        if g_indices.size == 0:
            continue
        if by_subgroup:
            subgroup_vals = subgroup_labels[g_indices]
            subgroup_order = list(dict.fromkeys(subgroup_vals.tolist()))
            order_map = {k: i for i, k in enumerate(subgroup_order)}
            subgroup_rank = np.array([order_map[v] for v in subgroup_vals], dtype=int)
            g_indices = g_indices[np.argsort(subgroup_rank, kind="stable")]

        new_order.append(g_indices)
        current_y += g_indices.size
        boundaries.append((current_y - 0.5, g))

    if new_order:
        sorted_indices = np.concatenate(new_order)
    if len(sorted_indices) < n_neurons:
        warnings.warn("Not all neurons were assigned to a group. Appending defaults.")
        missing = np.setdiff1d(np.arange(n_neurons), sorted_indices)
        sorted_indices = np.concatenate([sorted_indices, missing])
    return sorted_indices, boundaries


def _resolve_raster_groups(
    n_neurons: int,
    neurons_df: pd.DataFrame | None,
    group_key: str | None,
    group_color_key: str | None,
    group_sort: list[str] | None,
    sort_neurons: bool,
) -> _RasterGroups:
    """Validate grouping arguments and resolve labels and neuron order."""
    if group_key is not None or group_color_key is not None:
        if neurons_df is None:
            raise ValueError("neurons_df must be provided when grouping is used.")
        if group_key is not None and group_key not in neurons_df.columns:
            raise ValueError(f"Column '{group_key}' not found in neurons_df.")
        if group_color_key is not None and group_color_key not in neurons_df.columns:
            raise ValueError(f"Column '{group_color_key}' not found in neurons_df.")

    group_labels = np.full(n_neurons, "Unknown", dtype=object)
    if group_key is not None:
        group_labels = _labels_from_df(neurons_df, group_key, n_neurons)

    if group_color_key is not None:
        subgroup_labels = _labels_from_df(neurons_df, group_color_key, n_neurons)
    else:
        subgroup_labels = group_labels

    groups = None
    sorted_indices = np.arange(n_neurons)
    boundaries: list[tuple[float, Any]] = []
    if group_key is not None:
        present_groups = set(group_labels.tolist())
        if group_sort:
            groups = [g for g in group_sort if g in present_groups]
            groups.extend(sorted(present_groups - set(groups)))
        else:
            groups = sorted(present_groups)
        sorted_indices, boundaries = _order_neurons_by_group(
            group_labels,
            subgroup_labels,
            groups,
            group_color_key is not None,
            sort_neurons,
            n_neurons,
        )
    return _RasterGroups(
        group_labels, subgroup_labels, groups, sorted_indices, boundaries
    )


def _raster_spec_attrs(
    neuron_specs: dict | list | NeuronSpec,
    idx: int,
    marker: str,
    marker_size: float,
) -> tuple[Any, str, float]:
    """Look up (color, marker, markersize) for a neuron index.

    Lists and dicts are both keyed by the original neuron index; missing
    entries fall back to black and the default marker/size.
    """
    spec = None
    if isinstance(neuron_specs, list):
        if idx < len(neuron_specs):
            spec = neuron_specs[idx]
    elif isinstance(neuron_specs, dict):
        if idx in neuron_specs:
            spec = neuron_specs[idx]

    color = "black"
    if isinstance(spec, NeuronSpec):
        color = spec.color if spec.color is not None else color
        marker = spec.marker if spec.marker is not None else marker
        marker_size = spec.markersize if spec.markersize is not None else marker_size
    elif isinstance(spec, dict):
        color = spec.get("color", color)
        marker = spec.get("marker", marker)
        marker_size = spec.get("markersize", marker_size)
    return color, marker, marker_size


def _resolve_spike_style(
    spike_color: str | dict | Sequence[Any] | None,
    neuron_specs: dict | list | NeuronSpec | None,
    marker: str,
    marker_size: float,
    orig_neuron_indices: np.ndarray,
    n_neurons: int,
    group_key: str | None,
    group_labels: np.ndarray,
) -> _SpikeStyle:
    """Resolve per-spike colours, sizes and markers from colour/spec
    arguments."""
    c_array = spike_color
    color_by_neuron = None

    if isinstance(spike_color, dict):
        has_int_keys = any(isinstance(k, (int, np.integer)) for k in spike_color)
        if has_int_keys:
            color_by_neuron = np.array(
                [spike_color.get(i, "black") for i in range(n_neurons)],
                dtype=object,
            )
        elif group_key is not None:
            color_by_neuron = np.array(
                [spike_color.get(g, "black") for g in group_labels],
                dtype=object,
            )
        else:
            warnings.warn(
                "spike_color dict provided but group_key not set. Using black."
            )
            c_array = "black"
    elif isinstance(spike_color, (list, tuple, np.ndarray)):
        if len(spike_color) != n_neurons:
            raise ValueError(
                "spike_color sequence length must match number of neurons."
            )
        color_by_neuron = np.array(spike_color, dtype=object)

    if color_by_neuron is not None:
        return _SpikeStyle(color_by_neuron[orig_neuron_indices], marker, marker_size)
    if neuron_specs is None:
        return _SpikeStyle(c_array, marker, marker_size)

    attrs = [
        _raster_spec_attrs(neuron_specs, idx, marker, marker_size)
        for idx in orig_neuron_indices
    ]
    c_list = [a[0] for a in attrs]
    m_list = [a[1] for a in attrs]
    ms_list = [a[2] for a in attrs]
    style = _SpikeStyle(
        c_list,
        marker,
        ms_list,
        marker_list=np.array(m_list),
        size_list=np.array(ms_list),
        color_list=c_list,
    )
    if len(set(m_list)) > 1:
        # scatter() takes a single marker style, so markers are drawn per group.
        style.multi_marker = True
    else:
        style.marker = m_list[0] if m_list else marker
    return style


def _scatter_spikes(ax, x, y, sizes, colors, marker) -> None:
    ax.scatter(
        x, y, s=sizes, c=colors, marker=marker, linewidths=_marker_linewidth(marker)
    )


def _scatter_by_marker(ax, x, y, sizes, colors, markers, order) -> None:
    """Draw one scatter per marker value, visiting markers in ``order``."""
    for um in order:
        mask = markers == um
        _scatter_spikes(ax, x[mask], y[mask], sizes[mask], colors[mask], um)


def _draw_spikes(ax, spike_times, plot_neuron_indices, style: _SpikeStyle) -> None:
    """Draw the spikes with their resolved style (no group strip)."""
    if style.multi_marker:
        _scatter_by_marker(
            ax,
            spike_times,
            plot_neuron_indices,
            style.size_list,
            np.array(style.color_list, dtype=object),
            style.marker_list,
            set(style.marker_list.tolist()),
        )
        return
    _scatter_spikes(
        ax,
        spike_times,
        plot_neuron_indices,
        style.sizes,
        style.c_array,
        style.marker,
    )


def _draw_raster_annotations(
    ax: Axes,
    t: np.ndarray,
    n_neurons: int,
    show_tracks: bool,
    events: Sequence[float] | dict[str, Sequence[float]] | None,
    regions: Sequence[tuple[float, float]]
    | dict[str, Sequence[tuple[float, float]]]
    | None,
    event_kwargs: dict | None,
    region_kwargs: dict | None,
) -> None:
    """Draw neuron tracks, event lines and shaded regions."""
    if show_tracks:
        track_alpha = 0.1 if n_neurons > 100 else 0.2
        ax.hlines(
            y=np.arange(n_neurons),
            xmin=t[0],
            xmax=t[-1],
            colors="gray",
            alpha=track_alpha,
            linewidth=0.5,
            zorder=0,
        )

    if events is not None:
        evt_kwargs = {
            "color": "red",
            "linestyle": "--",
            "alpha": 0.8,
            "linewidth": 1.0,
        }
        if event_kwargs:
            evt_kwargs.update(event_kwargs)
        event_times = (
            [et for ets in events.values() for et in ets]
            if isinstance(events, dict)
            else events
        )
        for et in event_times:
            ax.axvline(x=et, **evt_kwargs)

    if regions is not None:
        reg_kwargs = {"color": "yellow", "alpha": 0.2}
        if region_kwargs:
            reg_kwargs.update(region_kwargs)
        intervals = (
            [iv for ivs in regions.values() for iv in ivs]
            if isinstance(regions, dict)
            else regions
        )
        for start, end in intervals:
            ax.axvspan(start, end, **reg_kwargs)


def _draw_group_separators(
    ax: Axes,
    boundaries: list[tuple[float, Any]],
    n_neurons: int,
    separator_style: dict | None,
    label_groups: bool,
    group_strip_side: str,
) -> None:
    """Draw lines between groups and, optionally, group labels at the side."""
    sep_args = (
        separator_style
        if separator_style
        else {"color": "gray", "linestyle": "--", "alpha": 0.5, "linewidth": 0.8}
    )
    prev_y = -0.5
    for y_limit, label in boundaries:
        if y_limit < n_neurons - 0.5:  # skip the line at the very top
            ax.axhline(y_limit, **sep_args)

        if label_groups:
            left = group_strip_side == "left"
            ax.text(
                -0.02 if left else 1.01,
                (prev_y + y_limit) / 2,
                str(label),
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="right" if left else "left",
                fontsize=8,
                color=sep_args.get("color", "black"),
            )
        prev_y = y_limit


def _add_strip_axes(ax_raster: Axes, side: str, cb_args: dict[str, Any]) -> Axes:
    """Add the narrow axes holding the group strip next to the raster."""
    pos = ax_raster.get_position()
    if side == "right":
        cax_x0 = pos.x1 + cb_args["pad"]
    else:
        cax_x0 = pos.x0 - cb_args["pad"] - cb_args["width"] - cb_args["left_extra_pad"]
    cax = ax_raster.figure.add_axes([cax_x0, pos.y0, cb_args["width"], pos.height])

    if side == "left":
        ylabel_x = (cax_x0 - cb_args["pad"] - pos.x0) / pos.width
        ax_raster.yaxis.set_label_coords(ylabel_x, 0.5)
    return cax


def _add_strip_labels(
    cax: Axes,
    label_list: list[str],
    n_neurons: int,
    side: str,
    cb_args: dict[str, Any],
) -> None:
    """Write one text label per contiguous label run, skipping crowded ones."""
    type_ranges: dict[str, dict[str, int]] = {}
    for i, label in enumerate(label_list):
        if label not in type_ranges:
            type_ranges[label] = {"start": i, "end": i}
        else:
            type_ranges[label]["end"] = i

    unique_types = list(dict.fromkeys(label_list))
    sorted_types = sorted(unique_types, key=lambda x: type_ranges[x]["start"])
    label_positions: list[float] = []
    for label in sorted_types:
        start_idx = type_ranges[label]["start"]
        end_idx = type_ranges[label]["end"]
        mid_y = (start_idx + end_idx) / 2

        min_distance = n_neurons * cb_args["min_label_distance"]
        too_close = any(abs(mid_y - pos) < min_distance for pos in label_positions)
        if too_close and (end_idx - start_idx) <= n_neurons * cb_args["min_span_frac"]:
            continue

        label_x = cb_args["label_x"]
        if label_x is None:
            if side == "right":
                label_x = (
                    cb_args["strip_x0"] + cb_args["strip_width"] + cb_args["label_gap"]
                )
                label_ha = "left"
            else:
                label_x = cb_args["strip_x0"] - cb_args["label_gap"]
                label_ha = "right"
        else:
            label_ha = "left"
        cax.text(
            label_x,
            mid_y,
            str(label),
            ha=label_ha,
            va="center",
            fontsize=cb_args["label_fontsize"],
            transform=cax.transData,
            weight=cb_args["label_weight"],
        )
        label_positions.append(mid_y)


def _add_strip_legend(
    cax: Axes,
    colors: _StripColors,
    top_groups_order: list[str],
    subgroups_by_top: dict[str, list[str]],
    label_mode: str,
    cb_args: dict[str, Any],
) -> None:
    base_colors = colors.base_colors
    subgroup_colors = colors.subgroup_colors
    if label_mode == "top":
        legend_elements = [
            mpatches.Patch(color=base_colors[tg], label=str(tg))
            for tg in top_groups_order
        ]
    elif label_mode == "sub":
        legend_elements = [
            mpatches.Patch(
                color=subgroup_colors.get((tg, sub), base_colors.get(tg)),
                label=str(sub),
            )
            for tg in top_groups_order
            for sub in subgroups_by_top[tg]
        ]
    else:  # top_sub
        legend_elements = [
            mpatches.Patch(
                color=subgroup_colors.get((tg, sub), base_colors.get(tg)),
                label=f"{tg}{cb_args['label_sep']}{sub}",
            )
            for tg in top_groups_order
            for sub in subgroups_by_top[tg]
        ]
    ncol = 2 if len(legend_elements) > cb_args["legend_ncol_threshold"] else 1
    cax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1, 1),
        fontsize=cb_args["legend_fontsize"],
        ncol=ncol,
        frameon=True,
        shadow=True,
    )


def _draw_group_strip(
    ax_raster: Axes,
    groups: _RasterGroups,
    neurons_df: pd.DataFrame | None,
    group_key: str | None,
    group_color_key: str | None,
    strip_cmap: str,
    group_strip_kwargs: dict | None,
    group_strip_legend: bool,
    group_label_mode: str,
    group_strip_side: str,
    n_neurons: int,
) -> _StripColors:
    """Draw the colour strip (patches, labels, legend) beside the raster.

    Returns:
        The colour maps used, so spikes can be coloured consistently.
    """
    if neurons_df is None:
        raise ValueError("neurons_df must be provided for group strip.")
    group_col = group_color_key or group_key
    if group_col is None:
        raise ValueError("group_color_key or group_key must be set for group strip.")
    if group_col not in neurons_df.columns:
        raise ValueError(f"Column '{group_col}' not found in neurons_df.")

    cb_args = dict(_STRIP_DEFAULTS)
    if group_strip_kwargs:
        cb_args.update(group_strip_kwargs)

    cax = _add_strip_axes(ax_raster, group_strip_side, cb_args)

    # Resolve subgroup and top-group labels per neuron in plotting order.
    sub_labels_raw = groups.subgroup_labels[groups.sorted_indices]
    top_group_labels = groups.group_labels[groups.sorted_indices]
    label_list = sub_labels_raw.tolist()

    use_subgroups = group_key is not None and group_col != group_key
    if use_subgroups:
        if group_label_mode == "top":
            label_list = [str(top) for top in top_group_labels]
        elif group_label_mode == "sub":
            label_list = [str(sub) for sub in label_list]
        else:
            label_list = [
                f"{top}{cb_args['label_sep']}{sub}"
                for top, sub in zip(top_group_labels, label_list)
            ]

    base_colors, subgroup_colors, subgroups_by_top, top_groups_order = (
        _build_group_color_maps(
            top_group_labels,
            sub_labels_raw,
            use_subgroups,
            strip_cmap,
            strip_cmap,
            cb_args["sub_hue_span"],
            cb_args["sub_val_span"],
        )
    )
    colors = _StripColors(use_subgroups, base_colors, subgroup_colors)

    for i in range(len(label_list)):
        tg = top_group_labels[i]
        sub = sub_labels_raw[i]
        if use_subgroups:
            color = subgroup_colors.get((tg, sub), base_colors.get(tg, "#cccccc"))
        else:
            color = base_colors.get(sub, "#cccccc")
        cax.add_patch(
            Rectangle(
                (cb_args["strip_x0"], i - 0.5),
                cb_args["strip_width"],
                1.0,
                facecolor=color,
                edgecolor="none",
                alpha=cb_args["alpha"],
            )
        )

    _add_strip_labels(cax, label_list, n_neurons, group_strip_side, cb_args)

    cax.set_xlim(0, 1)
    cax.set_ylim(ax_raster.get_ylim())
    cax.set_xticks([])
    cax.set_yticks([])
    cax.set_frame_on(False)
    for spine in cax.spines.values():
        spine.set_visible(False)

    if group_strip_legend:
        _add_strip_legend(
            cax, colors, top_groups_order, subgroups_by_top, group_label_mode, cb_args
        )
    return colors


def _draw_strip_spikes(
    ax: Axes,
    spike_times: np.ndarray,
    plot_neuron_indices: np.ndarray,
    orig_neuron_indices: np.ndarray,
    groups: _RasterGroups,
    colors: _StripColors,
    style: _SpikeStyle,
    marker_size: float,
) -> None:
    """Draw spikes coloured by the group strip colours (strip mode)."""
    top_vals = groups.group_labels[orig_neuron_indices]
    sub_vals = groups.subgroup_labels[orig_neuron_indices]
    spike_colors = []
    for top, sub in zip(top_vals, sub_vals):
        if colors.use_subgroups:
            spike_colors.append(
                colors.subgroup_colors.get(
                    (top, sub), colors.base_colors.get(top, "black")
                )
            )
        else:
            spike_colors.append(colors.base_colors.get(sub, "black"))
    spike_colors = np.array(spike_colors, dtype=object)

    sizes = style.size_list if style.size_list is not None else marker_size
    marker_list = style.marker_list
    if marker_list is not None and len(set(marker_list)) > 1:
        _scatter_by_marker(
            ax,
            spike_times,
            plot_neuron_indices,
            sizes,
            spike_colors,
            marker_list,
            sorted(set(marker_list)),
        )
    else:
        marker_use = marker_list[0] if marker_list is not None else style.marker
        _scatter_spikes(
            ax, spike_times, plot_neuron_indices, sizes, spike_colors, marker_use
        )


def _resolve_total_rate(
    rate: bool | np.ndarray | torch.Tensor | None,
    spikes_np: np.ndarray,
    t: np.ndarray,
    dt: float | None,
    rate_window_ms: float,
) -> np.ndarray | None:
    """Return the population rate trace (given array or computed) or None."""
    n_time = spikes_np.shape[0]
    if isinstance(rate, (np.ndarray, torch.Tensor)):
        fr = _to_numpy(rate)
        if fr.ndim == 2 and fr.shape[1] == 1:
            fr = fr[:, 0]
        if fr.ndim != 1:
            raise ValueError("rate must be 1D or shape (T, 1).")
        if fr.shape[0] != n_time:
            raise ValueError("rate length must match time axis length.")
        return fr
    if rate is True:
        eff_dt = _effective_dt(dt, t)
        return firing_rate(
            spikes_np, width=rate_window_ms / eff_dt, dt=eff_dt * 1e-3, axis=-1
        )
    return None


def _plot_group_rates(
    ax_rate: Axes,
    t: np.ndarray,
    spikes_np: np.ndarray,
    dt: float | None,
    rate_window_ms: float,
    group_rate: bool | dict[str, np.ndarray | torch.Tensor] | np.ndarray,
    groups: _RasterGroups,
    spike_color: Any,
    strip_cmap: str,
) -> None:
    """Plot one rate line per group (computed, dict-given or array-given)."""
    n_time = len(t)
    group_color_map: dict[str, Any] = {}
    if isinstance(spike_color, dict):
        if not any(isinstance(k, (int, np.integer)) for k in spike_color):
            group_color_map = dict(spike_color)
    if not group_color_map:
        group_palette = _sample_cmap_colors(strip_cmap, len(groups.groups))
        group_color_map = dict(zip(groups.groups, group_palette))

    def plot_line(g, values) -> None:
        ax_rate.plot(
            t,
            values,
            color=group_color_map.get(g, "black"),
            alpha=0.45,
            lw=0.9,
            zorder=1,
            label=str(g),
        )

    if isinstance(group_rate, dict):
        group_rates = {k: _to_numpy(v) for k, v in group_rate.items()}
        for g in groups.groups:
            if g not in group_rates:
                continue
            g_rate = group_rates[g]
            if g_rate.ndim == 2 and g_rate.shape[1] == 1:
                g_rate = g_rate[:, 0]
            if g_rate.ndim != 1 or g_rate.shape[0] != n_time:
                raise ValueError("group_rate values must be 1D and match time axis.")
            plot_line(g, g_rate)
    elif isinstance(group_rate, (np.ndarray, torch.Tensor)):
        group_rate_arr = _to_numpy(group_rate)
        if group_rate_arr.ndim != 2 or group_rate_arr.shape[0] != n_time:
            raise ValueError("group_rate array must have shape (T, G).")
        if group_rate_arr.shape[1] != len(groups.groups):
            raise ValueError("group_rate array must match number of groups.")
        for idx, g in enumerate(groups.groups):
            plot_line(g, group_rate_arr[:, idx])
    elif group_rate is True:
        eff_dt = _effective_dt(dt, t)
        for g in groups.groups:
            g_indices = np.flatnonzero(groups.group_labels == g)
            if g_indices.size == 0:
                continue
            plot_line(
                g,
                firing_rate(
                    spikes_np[:, g_indices],
                    width=rate_window_ms / eff_dt,
                    dt=eff_dt * 1e-3,
                    axis=-1,
                ),
            )


def _draw_rate_panel(
    ax_raster: Axes,
    ax_rate: Axes,
    t: np.ndarray,
    spikes_np: np.ndarray,
    dt: float | None,
    xlabel: str,
    rate: bool | np.ndarray | torch.Tensor | None,
    group_rate: bool | dict[str, np.ndarray | torch.Tensor] | np.ndarray | None,
    show_group_rate: bool,
    rate_window_ms: float,
    group_key: str | None,
    groups: _RasterGroups,
    spike_color: Any,
    strip_cmap: str,
) -> None:
    """Fill the rate axes and move the x label from the raster to it."""
    fr = _resolve_total_rate(rate, spikes_np, t, dt, rate_window_ms)

    if show_group_rate and group_key is not None:
        _plot_group_rates(
            ax_rate,
            t,
            spikes_np,
            dt,
            rate_window_ms,
            group_rate,
            groups,
            spike_color,
            strip_cmap,
        )

    if fr is not None:
        ax_rate.plot(t, fr, color="black", lw=1.8, alpha=0.9, zorder=2)
    ax_rate.set_xlim(t[0], t[-1])
    ax_rate.set_ylabel("Rate (Hz)")
    ax_rate.set_xlabel(xlabel)
    ax_raster.set_xticklabels([])
    ax_raster.set_xlabel("")


def plot_raster(
    spikes: np.ndarray | torch.Tensor,
    dt: float | None = None,
    times: Sequence[float] | None = None,
    ax: Axes | None = None,
    # Grouping and Metadata
    neurons_df: pd.DataFrame | None = None,
    group_key: str | None = None,
    group_sort: list[str] | None = None,
    # Styling
    spike_color: str | dict | Sequence[Any] | None = "black",
    marker: str = ".",
    marker_size: float = 5.0,
    neuron_specs: dict | list | NeuronSpec | None = None,
    show_group_separators: bool = True,
    separator_style: dict | None = None,
    # Standard Plot Args
    title: str | None = None,
    xlabel: str = "Time (ms)",
    ylabel: str = "Neuron Index",
    rate: bool | np.ndarray | torch.Tensor | None = False,
    group_rate: bool | dict[str, np.ndarray | torch.Tensor] | np.ndarray | None = False,
    rate_window_ms: float = 10.0,
    show_group_strip: bool = False,
    group_color_key: str | None = None,
    strip_cmap: str = "tab10",
    group_strip_kwargs: dict | None = None,
    group_strip_legend: bool = True,
    group_label_mode: Literal["top", "sub", "top_sub"] = "top_sub",
    group_strip_side: Literal["left", "right"] = "right",
    sort_neurons: bool = True,
    events: Sequence[float] | dict[str, Sequence[float]] | None = None,
    regions: Sequence[tuple[float, float]]
    | dict[str, Sequence[tuple[float, float]]]
    | None = None,
    show_tracks: bool = False,
    event_kwargs: dict | None = None,
    region_kwargs: dict | None = None,
) -> Axes | tuple[Axes, Axes]:
    """Plot spike raster with optional grouping and styling.

    Parameters
    ----------
    spikes : np.ndarray or torch.Tensor
        Spike matrix of shape (time, neurons).
    dt : float, optional
        Time step in ms. Default is 1.0 if times is not provided.
    times : array-like, optional
        Explicit time array.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure is created.
    neurons_df : pd.DataFrame, optional
        Dataframe containing neuron metadata, required for grouping.
    group_key : str, optional
        Column name in neurons_df to group neurons by.
    group_sort : list[str], optional
        Specific order for the groups.
    spike_color : str or dict or sequence, optional
        Default color for spikes. Can be a dict mapping group names or neuron
        indices to colors, or a per-neuron color sequence.
    marker : str
        Marker type.
    marker_size : float
        Size of the markers.
    neuron_specs : dict, list, or NeuronSpec, optional
        Specific styling per neuron.
    show_group_separators : bool
        Whether to draw lines separating groups.
    separator_style : dict, optional
        Arguments for separator lines (color, linewidth, etc.).
    title : str, optional
        Plot title.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    rate : bool or array-like, optional
        If True, compute and plot the population firing rate. If array-like,
        use it directly with length matching the time axis.
    group_rate : bool or dict or array-like, optional
        If True, compute and plot per-group firing rates when grouping is
        available. If dict, map group names to per-group rate arrays. If
        array-like, interpret as (T, G) in the order of resolved groups.
    rate_window_ms : float
        Window size for firing rate smoothing in ms.
    show_group_strip : bool
        If True, draw a colorbar-like group strip on the side.
    group_color_key : str, optional
        Column name in neurons_df to color the group strip. Defaults to group_key.
    strip_cmap : str
        Matplotlib colormap name used to derive both top-group and subgroup colors.
    group_strip_kwargs : dict, optional
        Additional options for colorbar layout and labels.
    group_strip_legend : bool
        If True, add a legend for group colors.
    group_label_mode : {"top", "sub", "top_sub"}
        Label mode for the colorbar when using subgroups.
    group_strip_side : {"left", "right"}
        Side on which to draw the group strip and labels.
    sort_neurons : bool
        If True (default), neurons are reordered by group and subgroup so bands are
        continuous. If False, original order is preserved.

    Returns
    -------
    ax or (ax_raster, ax_rate)
        The axis object(s).
    """
    spikes_np = _to_numpy(spikes)
    if spikes_np.ndim != 2:
        raise ValueError("spikes must be 2D (time, neurons)")

    n_time, n_neurons = spikes_np.shape
    t = _get_time_axis(n_time, dt, times)

    # Check isinstance first: bool() on a multi-element array/tensor raises.
    show_rate = isinstance(rate, (np.ndarray, torch.Tensor)) or bool(rate)
    show_group_rate = isinstance(group_rate, (dict, np.ndarray, torch.Tensor)) or bool(
        group_rate
    )
    with_rate_panel = show_rate or show_group_rate

    ax_raster, ax_rate = _create_raster_axes(ax, n_neurons, with_rate_panel)
    groups = _resolve_raster_groups(
        n_neurons, neurons_df, group_key, group_color_key, group_sort, sort_neurons
    )

    # sorted_indices[0] is plotted at y=0 (bottom).
    idx_map = np.empty(n_neurons)
    idx_map[groups.sorted_indices] = np.arange(len(groups.sorted_indices))

    orig_neuron_indices, spike_times = compute_raster(spikes_np, t)
    plot_neuron_indices = idx_map[orig_neuron_indices]

    style = _resolve_spike_style(
        spike_color,
        neuron_specs,
        marker,
        marker_size,
        orig_neuron_indices,
        n_neurons,
        group_key,
        groups.group_labels,
    )
    # With a group strip, spikes are drawn later using the strip colours.
    if not show_group_strip:
        _draw_spikes(ax_raster, spike_times, plot_neuron_indices, style)

    ax_raster.set_xlim(t[0], t[-1])
    ax_raster.set_ylim(-0.5, n_neurons - 0.5)
    ax_raster.set_ylabel(ylabel)
    ax_raster.yaxis.set_major_locator(MaxNLocator(integer=True))

    _draw_raster_annotations(
        ax_raster,
        t,
        n_neurons,
        show_tracks,
        events,
        regions,
        event_kwargs,
        region_kwargs,
    )

    spike_count = len(spike_times)
    fired_neurons = len(np.unique(orig_neuron_indices)) if spike_count > 0 else 0
    stats_title = f"Fired {fired_neurons}/{n_neurons}, Spikes {spike_count}"
    ax_raster.set_title(title if title else f"Spike raster {stats_title}")

    if group_key and show_group_separators:
        _draw_group_separators(
            ax_raster,
            groups.boundaries,
            n_neurons,
            separator_style,
            label_groups=not show_group_strip,
            group_strip_side=group_strip_side,
        )

    if show_group_strip:
        strip_colors = _draw_group_strip(
            ax_raster,
            groups,
            neurons_df,
            group_key,
            group_color_key,
            strip_cmap,
            group_strip_kwargs,
            group_strip_legend,
            group_label_mode,
            group_strip_side,
            n_neurons,
        )
        _draw_strip_spikes(
            ax_raster,
            spike_times,
            plot_neuron_indices,
            orig_neuron_indices,
            groups,
            strip_colors,
            style,
            marker_size,
        )

    ax_raster.text(
        0.01,
        0.99,
        f"N={spike_count}",
        transform=ax_raster.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        fontsize=8,
    )

    if with_rate_panel:
        assert ax_rate is not None
        _draw_rate_panel(
            ax_raster,
            ax_rate,
            t,
            spikes_np,
            dt,
            xlabel,
            rate,
            group_rate,
            show_group_rate,
            rate_window_ms,
            group_key,
            groups,
            spike_color,
            strip_cmap,
        )
        return ax_raster, ax_rate

    ax_raster.set_xlabel(xlabel)
    return ax_raster


def plot_traces(
    data: np.ndarray | torch.Tensor,
    dt: float | None = None,
    times: Sequence[float] | None = None,
    ax: Axes | None = None,
    neurons: Sequence[int] | int | None = None,
    labels: Sequence[str] | str | None = None,
    colors: Sequence[Any] | None = None,
    title: str | None = None,
    xlabel: str = "Time (ms)",
    ylabel: str | None = None,
    legend: bool = True,
    alpha: float = 0.8,
) -> Axes:
    """Plot continuous timeseries traces.

    Parameters
    ----------
    data : array-like
        Shape (Time, Neurons) or (Time, Neurons, Features).
    dt : float, optional
        Time step.
    times : array-like, optional
        Explicit time array.
    ax : Axes, optional
        Axis to plot on.
    neurons : list of int or int, optional
        Indices of neurons to plot. If None, plots all (careful with large N).
        If int, samples that many neurons randomly.
    labels : list of str, optional
        Labels for the legend.
    colors : list of colors, optional
        Colors for traces.
    title : str, optional
        Plot title.

    Returns
    -------
    Axes
    """
    data_np = _to_numpy(data)
    t = _get_time_axis(data_np.shape[0], dt, times)

    if data_np.ndim == 2:
        # (Time, Neurons)
        data_np = data_np[:, :, np.newaxis]  # make it (Time, Neurons, 1)
    elif data_np.ndim != 3:
        raise ValueError("Data must be 2D (T, N) or 3D (T, N, F)")

    n_neurons = data_np.shape[1]
    if neurons is None:
        neuron_indices = np.arange(n_neurons)
    elif isinstance(neurons, int):
        if neurons >= n_neurons:
            neuron_indices = np.arange(n_neurons)
        else:
            neuron_indices = np.sort(
                np.random.choice(n_neurons, neurons, replace=False)
            )
    else:
        neuron_indices = np.array(neurons)

    n_features = data_np.shape[2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if colors is None:
        cmap = plt.get_cmap("turbo", len(neuron_indices))
        colors = [cmap(i) for i in range(len(neuron_indices))]

    for i, idx in enumerate(neuron_indices):
        c = (
            colors[i]
            if isinstance(colors, (list, np.ndarray))
            and len(colors) == len(neuron_indices)
            else None
        )

        for feat in range(n_features):
            trace = data_np[:, idx, feat]

            lbl = None
            if labels is not None:
                if isinstance(labels, str):
                    lbl = f"{labels} {idx}"
                elif len(labels) == len(neuron_indices):
                    lbl = labels[i]
                else:
                    lbl = f"Neuron {idx}"
            else:
                lbl = f"Neuron {idx}"

            if n_features > 1:
                lbl += f" (f{feat})"

            ax.plot(t, trace, label=lbl, color=c, alpha=alpha)

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlim(t[0], t[-1])

    if title:
        ax.set_title(title)

    if legend and len(neuron_indices) <= 20:  # Limit legend clutter
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    return ax


def plot_spectrum(
    data: np.ndarray | torch.Tensor,
    dt: float | None = None,
    nperseg: int | None = None,
    ax: Axes | None = None,
    mode: str = "loglog",
    show_mean: bool = True,
    title: str = "Frequency Spectrum",
    color: str | None = None,
    label: str | None = "Mean",
    alpha: float = 0.2,
    mean_linewidth: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, Axes]:
    """Plot frequency spectrum of timeseries data.

    Computes power spectral density using Welch's method and visualizes
    the frequency content. For 2D input (time, neurons), plots individual
    traces with optional mean overlay.

    Args:
        data: Input timeseries with shape (time,) or (time, neurons).
        dt: Sampling interval in ms. Default 1.0.
        nperseg: Length of FFT segments. Default is min(256, time//4).
        ax: Existing axes to plot on. Creates new figure if None.
        mode: Plot scale - "loglog" (default) or "semilogx".
        show_mean: Whether to overlay the mean spectrum (for 2D data).
        title: Plot title.
        color: Color for traces. Uses default if None.
        label: Legend label for mean trace.
        alpha: Opacity for individual traces.
        mean_linewidth: Line width for mean trace.

    Returns:
        Tuple of (frequencies, power_spectrum, axes).

    Example:
        >>> freqs, power, ax = plot_spectrum(spikes, dt=1.0, mode="loglog")
    """
    data_np = _to_numpy(data)
    if dt is None:
        dt = 1.0

    freqs, power = compute_spectrum(data_np, dt=dt, nperseg=nperseg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    power_db = 10 * np.log10(power)

    y_data = power if "log" in mode else power_db

    trace_color = color if color else "blue"
    mean_color = color if color else "black"

    if show_mean and data_np.ndim > 1:
        if alpha > 0:
            ax.plot(freqs, y_data, color=trace_color, alpha=alpha, lw=0.5)
        mean_power = y_data.mean(axis=1) if y_data.ndim > 1 else y_data
        ax.plot(freqs, mean_power, color=mean_color, lw=mean_linewidth, label=label)
    else:
        ax.plot(freqs, y_data, color=mean_color, label=label)

    if mode == "loglog":
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Power")
    elif mode == "semilogx":
        ax.set_xscale("log")
        ax.set_ylabel("Power (dB)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(title)

    return freqs, power, ax


def plot_grouped_spectrum(
    data: np.ndarray | torch.Tensor,
    dt: float = 1.0,
    neurons_df: pd.DataFrame | None = None,
    group_by: str | None = None,
    groups: dict[str, list[int]] | None = None,  # Manual override
    mode: Literal["overlay", "subplots"] = "overlay",
    separate_figures: bool = False,
    nperseg: int | None = None,
    show_traces: bool = True,
    show_mean: bool = True,
    colors: dict[str, str] | None = None,
    title: str | None = "Grouped Spectrum",
    plot_width: float = 6.0,
    plot_height: float = 4.0,
) -> Figure | dict[str, Figure]:
    """Plot spectrum for multiple groups.

    Args:
        data: (Time, Neurons)
        dt: Timestep
        neurons_df: Metadata
        group_by: Column to group by
        groups: Manual dict of {group_label: [neuron_indices]}
        mode: "overlay" (all in one) or "subplots" (rows)
        separate_figures: Return dict of figs
        colors: Dict of {group_label: color}
    """
    data_np = _to_numpy(data)
    if groups is None:
        if neurons_df is None or group_by is None:
            # No grouping, treat as one group "All"
            groups = {"All": list(range(data_np.shape[1]))}
        else:
            if group_by not in neurons_df.columns:
                raise ValueError(f"Column {group_by} missing")

            groups = {}
            unique_groups = neurons_df[group_by].unique()
            for g in sorted(unique_groups):
                indices = neurons_df.index[neurons_df[group_by] == g].tolist()
                valid_indices = [i for i in indices if i < data_np.shape[1]]
                if valid_indices:
                    groups[g] = valid_indices

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = {g: cmap(i % 10) for i, g in enumerate(groups.keys())}

    if separate_figures:
        figs = {}
        for g_name, indices in groups.items():
            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            group_data = data_np[:, indices]

            c = colors.get(g_name, "black")
            plot_spectrum(
                group_data,
                dt=dt,
                nperseg=nperseg,
                ax=ax,
                color=c,
                label=str(g_name),
                show_mean=show_mean,
                alpha=0.2 if show_traces else 0.0,
            )
            ax.set_title(f"Spectrum: {g_name}")
            figs[str(g_name)] = fig
        return figs

    elif mode == "subplots":
        n_groups = len(groups)
        fig, axes = plt.subplots(
            n_groups, 1, figsize=(plot_width, plot_height * n_groups), squeeze=False
        )
        axes = axes.flatten()

        for i, (g_name, indices) in enumerate(groups.items()):
            ax = axes[i]
            group_data = data_np[:, indices]
            c = colors.get(g_name, "black")

            plot_spectrum(
                group_data,
                dt=dt,
                nperseg=nperseg,
                ax=ax,
                color=c,
                label=str(g_name),
                show_mean=show_mean,
                alpha=0.2 if show_traces else 0.0,
            )
            ax.set_title(str(g_name))
            ax.legend(loc="upper right")

        plt.tight_layout()
        return fig

    else:  # Overlay
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))

        for g_name, indices in groups.items():
            group_data = data_np[:, indices]
            c = colors.get(g_name, "black")

            plot_spectrum(
                group_data,
                dt=dt,
                nperseg=nperseg,
                ax=ax,
                color=c,
                label=str(g_name),
                show_mean=show_mean,
                alpha=0.1 if show_traces else 0.0,  # lighter alpha for overlay
            )

        ax.set_title(title)
        ax.legend()
        return fig


def plot_log_hist(
    values: np.ndarray | torch.Tensor,
    ax: Axes | None = None,
    title: str = "Distribution",
    xlabel: str = "Value",
    **kwargs: Any,
) -> Axes:
    """Plot log-log histogram with logarithmic binning.

    Creates a scatter plot of histogram counts using logarithmically
    spaced bins. Useful for visualizing heavy-tailed distributions
    (e.g., power laws).

    Args:
        values: Input values to histogram. Flattened if multidimensional.
        ax: Existing axes to plot on. Creates new figure if None.
        title: Plot title.
        xlabel: X-axis label.
        **kwargs: Additional arguments passed to ax.scatter().

    Returns:
        Axes containing the log-log histogram.

    Example:
        >>> ax = plot_log_hist(synapse_weights, title="Weight Distribution")
    """
    vals = _to_numpy(values)
    hist, bin_centers = compute_log_hist(vals)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(bin_centers, hist, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)

    return ax


@dataclass
class NeuronSpec:
    """Specification for neuron plotting style.

    Attributes:
        label: Custom label text
        color: Main color string or dict of colors for 'voltage', 'asc', 'psc'
        linestyle: Line style string (e.g. '-', '--', ':')
        linewidth: Line width
        alpha: Plot opacity
    """

    label: str | None = None
    color: str | dict[str, str] | None = None
    linestyle: str = "-"
    linewidth: float = 0.8
    alpha: float = 1.0
    marker: str | None = None
    markersize: float | None = None


@dataclass
class SimulationStates:
    """Container for simulation state data and configs.

    Attributes:
        voltage: Membrane voltage traces (time, neurons) or
            (time, batch, neurons) if batch dimension present
        dt: Simulation timestep in ms
        asc: Afterspike current traces (time, neurons), (time, batch, neurons),
            or (time, batch, neurons, n_asc) for multiple ASC components
        psc: Total postsynaptic current (time, neurons), (time, batch, neurons),
            or (time, batch, neurons, n_psc) for multiple PSC components
        epsc: Excitatory PSC (time, neurons) or (time, batch, neurons)
        ipsc: Inhibitory PSC (time, neurons) or (time, batch, neurons)
        input: Input current (time, neurons) or (time, batch, neurons)
        spikes: Spike trains (time, neurons) or (time, batch, neurons)
        v_threshold: Spike threshold voltage(s), scalar or per-neuron
        v_reset: Reset voltage(s), scalar or per-neuron
    """

    voltage: np.ndarray | torch.Tensor
    dt: float = 1.0
    asc: np.ndarray | torch.Tensor | None = None
    psc: np.ndarray | torch.Tensor | None = None
    epsc: np.ndarray | torch.Tensor | None = None
    ipsc: np.ndarray | torch.Tensor | None = None
    input: np.ndarray | torch.Tensor | None = None
    spikes: np.ndarray | torch.Tensor | None = None
    v_threshold: float | Sequence[float] | np.ndarray | torch.Tensor | None = None
    v_reset: float | Sequence[float] | np.ndarray | torch.Tensor | None = None


@dataclass
class TracePlotFormat:
    """Figure formatting configuration.

    Attributes:
        neuron_indices: Specific neuron indices to plot
        sample_size: Number of neurons to randomly sample
        seed: Random seed for sampling
        show_voltage: Whether to show voltage subplot
        show_asc: Whether to show ASC subplot
        show_psc: Whether to show PSC subplot
        show_spikes_on_voltage: Mark spikes on voltage trace
        separate_figures: Return dict of figures (one per trace type) if True
        auto_width: Adjust figure width based on simulation duration
        colors: Color mapping for different traces
        figsize_per_neuron: Figure size per neuron row (width, height)
        neuron_labels: Side labels as sequence or callable(neuron_idx) -> str.
            Default None disables side labels.
        neuron_label_position: Position for neuron labels when enabled.
            "side" places labels at the right of each neuron slot; "top"
            places labels above each neuron slot.
        neurons_per_row: Number of neurons to place per row in combined mode
        batch_idx: Batch index to plot when data has shape (time, batch, neurons).
            If None and data is 3D, defaults to 0 (first batch sample).
    """

    neuron_indices: list[int] | None = None
    sample_size: int | None = None
    seed: int = 42
    show_voltage: bool = True
    show_asc: bool = True
    show_psc: bool = True
    show_spikes_on_voltage: bool = True
    separate_figures: bool = False
    auto_width: bool = True
    colors: dict[str, str] | None = None
    figsize_per_neuron: tuple[float, float] = (12, 2.5)
    neuron_labels: Sequence[str] | Callable[[int], str] | None = None
    neuron_label_position: Literal["side", "top"] = "side"
    neuron_specs: list[NeuronSpec | dict] | NeuronSpec | dict | None = None
    neurons_per_row: int | None = None
    batch_idx: int | None = None


def _extract_batch_dim(
    data: np.ndarray | torch.Tensor | None, batch_idx: int
) -> np.ndarray | None:
    """Extract a single batch from 3D/4D data (time, batch, neurons, ...).

    Args:
        data: Array with shape (time, neurons), (time, batch, neurons),
            or (time, batch, neurons, n_asc) for ASC currents
        batch_idx: Index of the batch sample to extract

    Returns:
        Array with batch dimension removed or None if input is None
    """
    if data is None:
        return None

    arr = _to_numpy(data)
    if arr.ndim == 2:
        # No batch dimension: (time, neurons)
        return arr
    elif arr.ndim == 3:
        # Has batch dimension: (time, batch, neurons)
        if batch_idx >= arr.shape[1]:
            raise ValueError(
                f"batch_idx {batch_idx} is out of bounds for batch dim {arr.shape[1]}"
            )
        return arr[:, batch_idx]
    elif arr.ndim == 4:
        # Has batch and ASC dimension: (time, batch, neurons, n_asc)
        if batch_idx >= arr.shape[1]:
            raise ValueError(
                f"batch_idx {batch_idx} is out of bounds for batch dim {arr.shape[1]}"
            )
        return arr[:, batch_idx]
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D data, got shape {arr.shape}")


def _format_top_neuron_label(label: str, max_chars_per_line: int) -> str:
    """Wrap long top labels to reduce horizontal crowding."""
    if max_chars_per_line <= 0 or len(label) <= max_chars_per_line:
        return label
    return "\n".join(wrap(label, width=max_chars_per_line, break_long_words=False))


_DEFAULT_TRACE_COLORS = {
    "voltage": "#2E86AB",
    "asc": "#A23B72",
    "psc": "#F18F01",
    "epsc": "#06A77D",
    "ipsc": "#D62246",
    "input": "#9467bd",
    "spike": "#000000",
}
_COMBINED_TITLES = {
    "voltage": "Voltage",
    "asc": "Afterspike Current",
    "psc": "Postsynaptic Current",
}
_SEPARATE_TITLES = {**_COMBINED_TITLES, "voltage": "Voltage Traces"}


@dataclass
class _TraceConfig:
    """Arguments of :func:`plot_neuron_traces` before/after merging
    dataclasses."""

    voltage: Any
    dt: float
    asc: Any
    psc: Any
    epsc: Any
    ipsc: Any
    input: Any
    psc_labels: Sequence[str] | None
    spikes: Any
    v_threshold: Any
    v_reset: Any
    neuron_indices: list[int] | None
    sample_size: int | None
    seed: int
    show_voltage: bool
    show_asc: bool
    show_psc: bool
    neuron_labels: Sequence[str] | Callable[[int], str] | None
    neuron_label_position: str
    neuron_specs: list[NeuronSpec | dict] | NeuronSpec | dict | None
    separate_figures: bool
    auto_width: bool
    neurons_per_row: int | None
    batch_idx: int | None


@dataclass
class _TraceData:
    """Numpy trace arrays with the batch dimension removed."""

    voltage: np.ndarray
    spikes: np.ndarray | None
    asc: np.ndarray | None
    psc: np.ndarray | None
    epsc: np.ndarray | None
    ipsc: np.ndarray | None
    input: np.ndarray | None
    psc_labels: Sequence[str] | None
    psc_has_extra_dim: bool


def _merge_trace_config(
    cfg: _TraceConfig,
    states: SimulationStates | pd.DataFrame | None,
    format: TracePlotFormat | None,
) -> _TraceConfig:
    """Fill unset arguments from ``states`` and override from ``format``.

    The sentinels (``dt == 1.0``, ``seed == 42``) mean "not explicitly given".
    """
    if states is not None:
        cfg.voltage = states.voltage if cfg.voltage is None else cfg.voltage
        cfg.dt = states.dt if cfg.dt == 1.0 else cfg.dt
        cfg.asc = states.asc if cfg.asc is None else cfg.asc
        cfg.psc = states.psc if cfg.psc is None else cfg.psc
        cfg.epsc = states.epsc if cfg.epsc is None else cfg.epsc
        cfg.ipsc = states.ipsc if cfg.ipsc is None else cfg.ipsc
        cfg.input = states.input if cfg.input is None else cfg.input
        cfg.spikes = states.spikes if cfg.spikes is None else cfg.spikes
        if cfg.v_threshold is None:
            cfg.v_threshold = states.v_threshold
        cfg.v_reset = states.v_reset if cfg.v_reset is None else cfg.v_reset

    if format is not None:
        if cfg.neuron_indices is None:
            cfg.neuron_indices = format.neuron_indices
        if cfg.sample_size is None:
            cfg.sample_size = format.sample_size
        cfg.seed = format.seed if cfg.seed == 42 else cfg.seed
        cfg.show_voltage = format.show_voltage
        cfg.show_asc = format.show_asc
        cfg.show_psc = format.show_psc
        if cfg.neuron_labels is None:
            cfg.neuron_labels = format.neuron_labels
        cfg.neuron_label_position = format.neuron_label_position
        if cfg.neuron_specs is None:
            cfg.neuron_specs = format.neuron_specs
        cfg.separate_figures = format.separate_figures
        cfg.auto_width = format.auto_width
        if cfg.neurons_per_row is None:
            cfg.neurons_per_row = format.neurons_per_row
        if cfg.batch_idx is None:
            cfg.batch_idx = format.batch_idx
    return cfg


def _prepare_trace_data(cfg: _TraceConfig) -> _TraceData:
    """Validate PSC layout and strip the batch dimension from every array."""
    batch_idx = 0 if cfg.batch_idx is None else cfg.batch_idx
    psc_labels = cfg.psc_labels

    # A 3D PSC matching the neuron count is (time, neurons, n_psc). This must be
    # decided before batch extraction, which would read it as (time, batch, neurons).
    psc_has_extra_dim = False
    psc_raw = _to_numpy(cfg.psc) if cfg.psc is not None else None
    if psc_raw is not None and psc_raw.ndim == 3:
        n_neurons_from_v = _to_numpy(cfg.voltage).shape[1]
        if psc_raw.shape[1] == n_neurons_from_v:
            psc_has_extra_dim = True
            for name, value in (
                ("epsc", cfg.epsc),
                ("ipsc", cfg.ipsc),
                ("input", cfg.input),
            ):
                if value is not None:
                    raise ValueError(
                        f"{name} must be None when psc has additional dimension "
                        "(n_psc > 1)"
                    )
            if psc_labels is None:
                psc_labels = [f"PSC_{i}" for i in range(psc_raw.shape[2])]

    voltage = _extract_batch_dim(cfg.voltage, batch_idx)
    spikes = _extract_batch_dim(cfg.spikes, batch_idx)
    asc = _extract_batch_dim(cfg.asc, batch_idx)
    if psc_has_extra_dim:
        psc = psc_raw
    else:
        psc = _extract_batch_dim(cfg.psc, batch_idx)
    epsc = _extract_batch_dim(cfg.epsc, batch_idx)
    ipsc = _extract_batch_dim(cfg.ipsc, batch_idx)
    input_current = _extract_batch_dim(cfg.input, batch_idx)
    return _TraceData(
        voltage=_to_numpy(voltage),
        spikes=spikes,
        asc=asc,
        psc=psc,
        epsc=epsc,
        ipsc=ipsc,
        input=input_current,
        psc_labels=psc_labels,
        psc_has_extra_dim=psc_has_extra_dim,
    )


def _select_trace_neurons(
    n_neurons: int,
    neuron_indices: list[int] | None,
    sample_size: int | None,
    seed: int,
) -> list[int]:
    """Choose neurons to plot: explicit, random sample, or the first five."""
    if neuron_indices is None and sample_size is None:
        return list(range(min(5, n_neurons)))
    if neuron_indices is None:
        np.random.seed(seed)
        return sorted(
            np.random.choice(n_neurons, min(sample_size, n_neurons), replace=False)
        )
    return neuron_indices


def _make_label_resolver(
    neuron_labels: Sequence[str] | Callable[[int], str] | None,
) -> Callable[[int, int], str | None]:
    """Return ``resolve(plot_idx, neuron_idx)`` for callable/sequence
    labels."""

    def resolve(plot_idx: int, neuron_idx: int) -> str | None:
        if callable(neuron_labels):
            return str(neuron_labels(neuron_idx))
        if neuron_labels is not None and plot_idx < len(neuron_labels):
            return str(neuron_labels[plot_idx])
        return None

    return resolve


def _resolve_neuron_spec(
    neuron_specs: list[NeuronSpec | dict] | NeuronSpec | dict | None, plot_idx: int
) -> NeuronSpec:
    """Return the spec for a plotted neuron (list by position, dict for
    all)."""
    spec = NeuronSpec()
    if isinstance(neuron_specs, list):
        if plot_idx < len(neuron_specs):
            s = neuron_specs[plot_idx]
            spec = NeuronSpec(**s) if isinstance(s, dict) else s
    elif isinstance(neuron_specs, dict):
        spec = NeuronSpec(**neuron_specs)
    elif isinstance(neuron_specs, NeuronSpec):
        spec = neuron_specs
    return spec


def _colors_for_spec(colors: dict[str, str], spec: NeuronSpec) -> dict[str, str]:
    """Apply a spec colour (single colour or per-trace dict) over
    ``colors``."""
    local_colors = colors.copy()
    if spec.color is not None:
        if isinstance(spec.color, dict):
            local_colors.update(spec.color)
        else:
            for k in local_colors:
                if k != "spike":
                    local_colors[k] = spec.color
    return local_colors


def _trace_panel_kinds(
    cfg: _TraceConfig, data: _TraceData
) -> list[Literal["voltage", "asc", "psc"]]:
    """Panels to draw: requested by the user and backed by data."""
    kinds: list[Literal["voltage", "asc", "psc"]] = []
    if cfg.show_voltage:
        kinds.append("voltage")
    if cfg.show_asc and data.asc is not None:
        kinds.append("asc")
    if cfg.show_psc and data.psc is not None:
        kinds.append("psc")
    return kinds


def _draw_psc_panel(
    ax: Axes,
    times: np.ndarray,
    data: _TraceData,
    neuron_idx: int,
    colors: dict[str, str],
    style: dict[str, Any],
) -> bool:
    """Plot PSC traces; return whether the panel should get a legend."""
    if data.psc_has_extra_dim:
        psc_traces = data.psc[:, neuron_idx, :]  # (time, n_psc)
        _plot_multi_psc_on_ax(ax, times, psc_traces, data.psc_labels, colors, **style)
        return True
    _plot_psc_on_ax(
        ax,
        times,
        data.psc[:, neuron_idx],
        data.epsc[:, neuron_idx] if data.epsc is not None else None,
        data.ipsc[:, neuron_idx] if data.ipsc is not None else None,
        data.input[:, neuron_idx] if data.input is not None else None,
        colors,
        **style,
    )
    return data.epsc is not None or data.ipsc is not None or data.input is not None


def _draw_trace_panel(
    ax: Axes,
    kind: str,
    data: _TraceData,
    neuron_idx: int,
    times: np.ndarray,
    colors: dict[str, str],
    format: TracePlotFormat | None,
    v_th: float | None,
    v_reset: float | None,
    style: dict[str, Any],
    title: str | None,
) -> None:
    """Draw one trace panel; ``title`` (first row only) also adds a legend."""
    if kind == "voltage":
        _plot_voltage_on_ax(
            ax,
            times,
            data.voltage[:, neuron_idx],
            data.spikes[:, neuron_idx] if data.spikes is not None else None,
            colors,
            format,
            v_th,
            v_reset,
            **style,
        )
        ax.set_ylabel("V (mV)")
        with_legend = v_th is not None or v_reset is not None
    elif kind == "asc":
        _plot_simple_trace_on_ax(
            ax, times, data.asc[:, neuron_idx], colors["asc"], "ASC (pA)", **style
        )
        with_legend = False
    else:
        with_legend = _draw_psc_panel(ax, times, data, neuron_idx, colors, style)

    if title is not None:
        ax.set_title(title)
        if with_legend:
            ax.legend(loc="upper right", fontsize=8)


def _finish_trace_axis(ax: Axes, is_last_row: bool) -> None:
    if is_last_row:
        ax.set_xlabel("Time (ms)")
    ax.grid(alpha=0.3, linewidth=0.5)


def _add_side_label(ax: Axes, label: str) -> None:
    """Bold label to the right of an axes."""
    ax.text(
        1.02,
        0.5,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="center",
        ha="left",
    )


def _plot_separate_trace_figures(
    kinds: list[str],
    data: _TraceData,
    neuron_indices: list[int],
    times: np.ndarray,
    colors: dict[str, str],
    format: TracePlotFormat | None,
    v_thresholds: list[float | None],
    v_resets: list[float | None],
    resolve_label: Callable[[int, int], str | None],
    label_position: str,
    figsize: tuple[float, float],
) -> dict[str, Figure]:
    """Create one figure per trace type with one row per neuron."""
    n_plot = len(neuron_indices)
    figures: dict[str, Figure] = {}
    for kind in kinds:
        fig, axes = plt.subplots(n_plot, 1, figsize=figsize, squeeze=False)
        for i, neuron_idx in enumerate(neuron_indices):
            ax = axes[i, 0]
            _draw_trace_panel(
                ax,
                kind,
                data,
                neuron_idx,
                times,
                colors,
                format,
                v_thresholds[i],
                v_resets[i],
                {},
                _SEPARATE_TITLES[kind] if i == 0 else None,
            )
            _finish_trace_axis(ax, i == n_plot - 1)
            label = resolve_label(i, neuron_idx)
            if label is None:
                continue
            if label_position == "top":
                ax.text(
                    0.5,
                    1.12,
                    label,
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    va="bottom",
                    ha="center",
                )
            else:
                _add_side_label(ax, label)

        plt.tight_layout()
        figures[kind] = fig
    return figures


@dataclass
class _TraceGrid:
    """Figure and axes of the combined trace layout."""

    fig: Figure
    axes: dict[tuple[int, int], Axes]
    label_axes: dict[tuple[int, int], Axes]
    n_rows: int
    n_cols: int
    neurons_per_row: int
    top_labels: bool
    max_label_chars_per_line: int

    def plot_row(self, row_idx: int) -> int:
        """Grid row holding the traces of neuron row ``row_idx``."""
        return row_idx * 2 + 1 if self.top_labels else row_idx


def _create_trace_grid(
    n_plot: int,
    n_cols: int,
    neurons_per_row: int,
    top_labels: bool,
    max_label_len: int,
    base_width: float,
    height_per_row: float,
) -> _TraceGrid:
    """Create the figure with one trace axes per (neuron row, slot, panel)."""
    n_rows = int(ceil(n_plot / neurons_per_row))
    total_cols = n_cols * neurons_per_row
    label_height_ratio = 0.22
    max_label_chars_per_line = 0
    if top_labels and max_label_len > 0:
        # Rough estimate for wrapping purposes only (not for sizing)
        max_label_chars_per_line = max(36, int(base_width * 8))
        label_line_count = max(
            1, int(ceil(max_label_len / max(max_label_chars_per_line, 1)))
        )
        label_height_ratio = 0.22 + 0.12 * (label_line_count - 1)
    total_height = (
        height_per_row * n_rows * (1.0 + label_height_ratio if top_labels else 1.0)
    )
    # Keep enough width per trace column to avoid label crowding.
    base_width = max(base_width, 4.0 * n_cols)
    fig = plt.figure(figsize=(base_width * neurons_per_row, total_height))
    gridspec_kw = (
        {"height_ratios": [v for _ in range(n_rows) for v in (label_height_ratio, 1.0)]}
        if top_labels
        else {}
    )
    grid_spec = fig.add_gridspec(
        n_rows * 2 if top_labels else n_rows, total_cols, **gridspec_kw
    )
    grid = _TraceGrid(
        fig,
        {},
        {},
        n_rows,
        n_cols,
        neurons_per_row,
        top_labels,
        max_label_chars_per_line,
    )

    for row_idx in range(n_rows):
        if top_labels:
            for slot_idx in range(neurons_per_row):
                col_base = slot_idx * n_cols
                label_ax = fig.add_subplot(
                    grid_spec[row_idx * 2, col_base : col_base + n_cols]
                )
                label_ax.set_axis_off()
                grid.label_axes[(row_idx, slot_idx)] = label_ax

        plot_row = grid.plot_row(row_idx)
        for c in range(total_cols):
            grid.axes[(plot_row, c)] = fig.add_subplot(grid_spec[plot_row, c])
    return grid


def _draw_combined_traces(
    grid: _TraceGrid,
    kinds: list[str],
    data: _TraceData,
    neuron_indices: list[int],
    times: np.ndarray,
    colors: dict[str, str],
    format: TracePlotFormat | None,
    v_thresholds: list[float | None],
    v_resets: list[float | None],
    neuron_specs: list[NeuronSpec | dict] | NeuronSpec | dict | None,
    resolved_labels: list[str | None],
) -> set[tuple[int, int]]:
    """Draw every neuron's panels and labels; return the axes that were
    used."""
    used_axes: set[tuple[int, int]] = set()
    n_cols = grid.n_cols

    for plot_idx, neuron_idx in enumerate(neuron_indices):
        row_idx, slot_idx = divmod(plot_idx, grid.neurons_per_row)
        plot_row = grid.plot_row(row_idx)
        spec = _resolve_neuron_spec(neuron_specs, plot_idx)
        label = spec.label if spec.label is not None else resolved_labels[plot_idx]
        local_colors = _colors_for_spec(colors, spec)
        style = {
            "linestyle": spec.linestyle,
            "linewidth": spec.linewidth,
            "alpha": spec.alpha,
        }

        col_base = slot_idx * n_cols
        for col_idx, kind in enumerate(kinds):
            ax = grid.axes[(plot_row, col_base + col_idx)]
            _draw_trace_panel(
                ax,
                kind,
                data,
                neuron_idx,
                times,
                local_colors,
                format,
                v_thresholds[plot_idx],
                v_resets[plot_idx],
                style,
                _COMBINED_TITLES[kind] if row_idx == 0 else None,
            )
            _finish_trace_axis(ax, row_idx == grid.n_rows - 1)
            used_axes.add((plot_row, col_base + col_idx))

        if label is None:
            continue
        if grid.top_labels:
            label_ax = grid.label_axes[(row_idx, slot_idx)]
            label_ax.text(
                0.5,
                0.5,
                _format_top_neuron_label(label, grid.max_label_chars_per_line),
                transform=label_ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="center",
                ha="center",
            )
        else:
            # Label the rightmost subplot in this neuron slot.
            _add_side_label(grid.axes[(plot_row, col_base + n_cols - 1)], label)
    return used_axes


def _hide_unused_trace_axes(
    grid: _TraceGrid, used_axes: set[tuple[int, int]], n_plot: int
) -> None:
    """Hide axes of empty neuron slots in the final row."""
    for r in range(grid.n_rows):
        plot_row = grid.plot_row(r)
        for c in range(grid.n_cols * grid.neurons_per_row):
            if (plot_row, c) not in used_axes:
                grid.axes[(plot_row, c)].set_visible(False)

        if grid.top_labels:
            for slot_idx in range(grid.neurons_per_row):
                if r * grid.neurons_per_row + slot_idx >= n_plot:
                    grid.label_axes[(r, slot_idx)].set_visible(False)


def _widen_for_top_labels(grid: _TraceGrid) -> None:
    """Grow the figure width so the widest top label fits in its slot."""
    fig = grid.fig
    fig.canvas.draw()  # text must be rendered to measure it
    max_label_width_inches = 0.0
    for label_ax in grid.label_axes.values():
        for text in label_ax.texts:
            bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
            max_label_width_inches = max(max_label_width_inches, bbox.width / fig.dpi)
    if max_label_width_inches > 0:
        required_slot_width = max_label_width_inches * 1.2 + 1.0  # 20% pad + margin
        min_fig_width = required_slot_width * grid.neurons_per_row
        if min_fig_width > fig.get_figwidth():
            fig.set_figwidth(min_fig_width)


def plot_neuron_traces(
    # Dataclass interface
    states: SimulationStates | pd.DataFrame | None = None,
    format: TracePlotFormat | None = None,
    # Plain args interface
    voltage: np.ndarray | torch.Tensor | None = None,
    dt: float = 1.0,
    asc: np.ndarray | torch.Tensor | None = None,
    psc: np.ndarray | torch.Tensor | None = None,
    epsc: np.ndarray | torch.Tensor | None = None,
    ipsc: np.ndarray | torch.Tensor | None = None,
    input: np.ndarray | torch.Tensor | None = None,
    psc_labels: Sequence[str] | None = None,
    spikes: np.ndarray | torch.Tensor | None = None,
    v_threshold: float | Sequence[float] | np.ndarray | torch.Tensor | None = None,
    v_reset: float | Sequence[float] | np.ndarray | torch.Tensor | None = None,
    neuron_indices: list[int] | None = None,
    sample_size: int | None = None,
    seed: int = 42,
    show_voltage: bool = True,
    show_asc: bool = True,
    show_psc: bool = True,
    neuron_labels: Sequence[str] | Callable[[int], str] | None = None,
    neuron_label_position: Literal["side", "top"] = "side",
    neuron_specs: list[NeuronSpec | dict] | NeuronSpec | dict | None = None,
    neurons_df: pd.DataFrame | None = None,
    separate_figures: bool = False,
    auto_width: bool = True,
    neurons_per_row: int | None = None,
    batch_idx: int | None = None,
) -> Figure | dict[str, Figure]:
    """Plot neuron state traces with flexible interface.

    Supports both dataclass and plain argument interfaces. Each neuron gets
    a row of subplots showing voltage, ASC, and PSC traces.

    Args:
        states: SimulationStates dataclass with all state data
        format: TracePlotFormat dataclass with formatting options
        voltage: Voltage traces (time, neurons) or (time, batch, neurons)
        dt: Timestep in ms
        asc: Afterspike current traces (time, neurons), (time, batch, neurons),
            or (time, batch, neurons, n_asc) for multiple ASC components
        psc: Postsynaptic current traces (time, neurons), (time, batch, neurons),
            or (time, batch, neurons, n_psc) for multiple PSC components.
            If psc has additional dims (n_psc > 1), epsc, ipsc, and input
            should be None.
        epsc: Excitatory PSC traces (time, neurons) or (time, batch, neurons)
        ipsc: Inhibitory PSC traces (time, neurons) or (time, batch, neurons)
        input: Input current traces (time, neurons) or (time, batch, neurons)
        psc_labels: Labels for PSC components when psc has shape
            (time, neurons, n_psc) or (time, batch, neurons, n_psc).
            If None, defaults to ["PSC_0", "PSC_1", ...].
        spikes: Spike trains (time, neurons) or (time, batch, neurons)
        v_threshold: Spike threshold(s), scalar or per-neuron values
        v_reset: Reset voltage reference line(s), scalar or per-neuron values
        neuron_indices: Specific neurons to plot
        sample_size: Number of neurons to randomly sample
        seed: Random seed for sampling
        show_voltage: Show voltage subplot
        show_asc: Show ASC subplot
        show_psc: Show PSC subplot
        neuron_labels: Side labels as sequence or callable(neuron_idx) -> str.
            Default None disables side labels.
        neuron_label_position: Position for neuron labels when enabled.
            "side" or "top".
        neuron_specs: Specifications for per-neuron styling (scalar or list)
        neurons_df: DataFrame with neuron metadata for labels
        separate_figures: Return dict of figures (one per trace type)
        auto_width: Adjust width based on duration
        neurons_per_row: Number of neurons per row in combined figure
        batch_idx: Batch index to plot when data has shape (time, batch, neurons).
            If None and data is 3D, defaults to 0.

    Returns:
        Figure with neuron trace subplots OR dict of Figures
    """
    cfg = _merge_trace_config(
        _TraceConfig(
            voltage=voltage,
            dt=dt,
            asc=asc,
            psc=psc,
            epsc=epsc,
            ipsc=ipsc,
            input=input,
            psc_labels=psc_labels,
            spikes=spikes,
            v_threshold=v_threshold,
            v_reset=v_reset,
            neuron_indices=neuron_indices,
            sample_size=sample_size,
            seed=seed,
            show_voltage=show_voltage,
            show_asc=show_asc,
            show_psc=show_psc,
            neuron_labels=neuron_labels,
            neuron_label_position=neuron_label_position,
            neuron_specs=neuron_specs,
            separate_figures=separate_figures,
            auto_width=auto_width,
            neurons_per_row=neurons_per_row,
            batch_idx=batch_idx,
        ),
        states,
        format,
    )
    if cfg.voltage is None:
        raise ValueError("voltage is required (provide via states or direct arg)")

    data = _prepare_trace_data(cfg)
    n_time, n_neurons = data.voltage.shape
    times = np.arange(n_time) * cfg.dt
    duration_ms = n_time * cfg.dt

    neuron_indices = _select_trace_neurons(
        n_neurons, cfg.neuron_indices, cfg.sample_size, cfg.seed
    )
    n_plot = len(neuron_indices)
    neurons_per_row = 1 if cfg.neurons_per_row is None else cfg.neurons_per_row
    if neurons_per_row < 1:
        raise ValueError("neurons_per_row must be >= 1")

    v_thresholds = _resolve_per_neuron_values(
        cfg.v_threshold, neuron_indices, n_neurons, "v_threshold"
    )
    v_resets = _resolve_per_neuron_values(
        cfg.v_reset, neuron_indices, n_neurons, "v_reset"
    )

    base_width = 12.0
    if cfg.auto_width:
        # ~1 inch per 40 ms, bounded to [10, 30]
        base_width = max(10.0, min(duration_ms * 0.025, 30.0))
    elif format:
        base_width = format.figsize_per_neuron[0]
    height_per_row = format.figsize_per_neuron[1] if format else 2.5

    colors = format.colors if format and format.colors else _DEFAULT_TRACE_COLORS
    resolve_label = _make_label_resolver(cfg.neuron_labels)
    kinds = _trace_panel_kinds(cfg, data)

    if cfg.separate_figures:
        return _plot_separate_trace_figures(
            kinds,
            data,
            neuron_indices,
            times,
            colors,
            format,
            v_thresholds,
            v_resets,
            resolve_label,
            cfg.neuron_label_position,
            (base_width, height_per_row * n_plot),
        )

    if not kinds:
        # Nothing requested/available: fall back to the required voltage panel.
        kinds = ["voltage"]

    resolved_labels = [
        resolve_label(plot_idx, neuron_idx)
        for plot_idx, neuron_idx in enumerate(neuron_indices)
    ]
    max_label_len = max((len(label) for label in resolved_labels if label), default=0)
    top_labels = cfg.neuron_label_position == "top"

    grid = _create_trace_grid(
        n_plot,
        len(kinds),
        neurons_per_row,
        top_labels,
        max_label_len,
        base_width,
        height_per_row,
    )
    used_axes = _draw_combined_traces(
        grid,
        kinds,
        data,
        neuron_indices,
        times,
        colors,
        format,
        v_thresholds,
        v_resets,
        cfg.neuron_specs,
        resolved_labels,
    )
    _hide_unused_trace_axes(grid, used_axes, n_plot)
    if top_labels and grid.label_axes:
        _widen_for_top_labels(grid)

    right_margin = 0.96 if cfg.neuron_label_position == "side" else 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid.fig.tight_layout(rect=(0.0, 0.0, right_margin, 1.0), w_pad=0.8, h_pad=0.8)
    return grid.fig


def _plot_voltage_on_ax(
    ax,
    times,
    voltage_trace,
    spike_trace,
    colors,
    format,
    v_th,
    v_reset,
    linestyle="-",
    linewidth=0.8,
    alpha=1.0,
):
    """Helper to plot voltage trace on axis."""
    ax.plot(
        times,
        voltage_trace,
        color=colors["voltage"],
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )

    if spike_trace is not None and (not format or format.show_spikes_on_voltage):
        spike_times = times[spike_trace > 0]
        spike_vals = voltage_trace[spike_trace > 0]
        ax.scatter(
            spike_times, spike_vals, color=colors["spike"], s=20, marker="^", zorder=5
        )

    if v_th is not None:
        ax.axhline(
            v_th,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            zorder=4,
            label="V_th",
        )
    if v_reset is not None:
        ax.axhline(
            v_reset,
            color="#555555",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            zorder=4,
            label="V_reset",
        )


def _plot_simple_trace_on_ax(
    ax, times, trace, color, ylabel, linestyle="-", linewidth=0.8, alpha=1.0
):
    """Helper for simple line trace."""
    ax.plot(
        times,
        trace,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.set_ylabel(ylabel)


def _plot_psc_on_ax(
    ax,
    times,
    psc_trace,
    epsc_trace,
    ipsc_trace,
    input_trace,
    colors,
    linestyle="-",
    linewidth=0.8,
    alpha=1.0,
):
    """Helper for PSC trace with EPSC, IPSC, and input components."""
    ax.plot(
        times,
        psc_trace,
        color=colors["psc"],
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label="Total PSC",
    )

    if epsc_trace is not None:
        ax.plot(
            times,
            epsc_trace,
            color=colors["epsc"],
            linewidth=0.6,
            alpha=0.7,
            linestyle="--",
            label="EPSC",
        )
    if ipsc_trace is not None:
        ax.plot(
            times,
            ipsc_trace,
            color=colors["ipsc"],
            linewidth=0.6,
            alpha=0.7,
            linestyle="--",
            label="IPSC",
        )
    if input_trace is not None:
        ax.plot(
            times,
            input_trace,
            color=colors.get("input", "#9467bd"),  # Default purple
            linewidth=0.6,
            alpha=0.7,
            linestyle=":",
            label="Input",
        )
    ax.set_ylabel("PSC (pA)")


def _plot_multi_psc_on_ax(
    ax,
    times,
    psc_traces,
    labels,
    colors,
    linestyle="-",
    linewidth=0.8,
    alpha=1.0,
):
    """Helper for PSC traces with multiple components (n_psc > 1).

    Args:
        ax: Matplotlib axis
        times: Time array
        psc_traces: Array of shape (time, n_psc)
        labels: List of labels for each PSC component
        colors: Color dictionary
        linestyle: Line style
        linewidth: Line width
        alpha: Plot opacity
    """
    n_components = psc_traces.shape[1]
    base_colors = [
        colors.get("psc", "#F18F01"),
        colors.get("epsc", "#06A77D"),
        colors.get("ipsc", "#D62246"),
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # yellow-green
        "#17becf",  # cyan
    ]

    for i in range(n_components):
        color = base_colors[i % len(base_colors)]
        label = labels[i] if labels and i < len(labels) else f"PSC_{i}"
        ax.plot(
            times,
            psc_traces[:, i],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label=label,
        )
    ax.set_ylabel("PSC (pA)")
