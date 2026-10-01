"""Private raster-plot building blocks (grouping, styling, strips)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import patches as mpatches
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from ...analysis.spiking import firing_rate
from ...utils.array import to_numpy
from ._helpers import (
    _STRIP_DEFAULTS,
    _auto_raster_height,
    _build_group_color_maps,
    _effective_dt,
    _marker_linewidth,
    _sample_cmap_colors,
)
from .neuron_specs import (
    NeuronSpec,
)
from .raster_options import (
    GroupStripOptions,
    RasterAnnotations,
    RasterGrouping,
    RasterStyle,
    RatePanelOptions,
)


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
    # True when the caller supplied explicit per-neuron colours (sequence or
    # neuron-/group-keyed dict) that must also win over strip colours.
    per_neuron_colors: bool = False


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
    n_neurons: int, grouping: RasterGrouping, strip: GroupStripOptions
) -> _RasterGroups:
    """Validate grouping options and resolve labels and neuron order."""
    neurons_df = grouping.neurons_df
    group_key = grouping.group_key
    group_color_key = strip.color_key
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
        if grouping.group_sort:
            groups = [g for g in grouping.group_sort if g in present_groups]
            groups.extend(sorted(present_groups - set(groups)))
        else:
            groups = sorted(present_groups)
        sorted_indices, boundaries = _order_neurons_by_group(
            group_labels,
            subgroup_labels,
            groups,
            group_color_key is not None,
            grouping.sort_neurons,
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
    style: RasterStyle,
    grouping: RasterGrouping,
    groups: _RasterGroups,
    orig_neuron_indices: np.ndarray,
    n_neurons: int,
) -> _SpikeStyle:
    """Resolve per-spike colours, sizes and markers from the style options."""
    spike_color = style.spike_color
    neuron_specs = style.neuron_specs
    marker = style.marker
    marker_size = style.marker_size
    group_key = grouping.group_key
    group_labels = groups.group_labels
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
        return _SpikeStyle(
            color_by_neuron[orig_neuron_indices],
            marker,
            marker_size,
            per_neuron_colors=True,
        )
    if neuron_specs is None:
        return _SpikeStyle(c_array, marker, marker_size)

    attrs = [
        _raster_spec_attrs(neuron_specs, idx, marker, marker_size)
        for idx in orig_neuron_indices
    ]
    c_list = [a[0] for a in attrs]
    m_list = [a[1] for a in attrs]
    ms_list = [a[2] for a in attrs]
    resolved = _SpikeStyle(
        c_list,
        marker,
        ms_list,
        marker_list=np.array(m_list),
        size_list=np.array(ms_list),
        color_list=c_list,
    )
    if len(set(m_list)) > 1:
        # scatter() takes a single marker style, so markers are drawn per group.
        resolved.multi_marker = True
    else:
        resolved.marker = m_list[0] if m_list else marker
    return resolved


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
    annotations: RasterAnnotations,
) -> None:
    """Draw neuron tracks, event lines and shaded regions."""
    events = annotations.events
    regions = annotations.regions
    if annotations.show_tracks:
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
        if annotations.event_kwargs:
            evt_kwargs.update(annotations.event_kwargs)
        event_times = (
            [et for ets in events.values() for et in ets]
            if isinstance(events, dict)
            else events
        )
        for et in event_times:
            ax.axvline(x=et, **evt_kwargs)

    if regions is not None:
        reg_kwargs = {"color": "yellow", "alpha": 0.2}
        if annotations.region_kwargs:
            reg_kwargs.update(annotations.region_kwargs)
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
    # Without subgroups top and sub labels coincide; avoid "A / A".
    if label_mode == "top" or not colors.use_subgroups:
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
    strip: GroupStripOptions,
    n_neurons: int,
) -> _StripColors:
    """Draw the colour strip (patches, labels, legend) beside the raster.

    Returns:
        The colour maps used, so spikes can be coloured consistently.
    """
    if neurons_df is None:
        raise ValueError("neurons_df must be provided for group strip.")
    group_col = strip.color_key or group_key
    if group_col is None:
        raise ValueError(
            "GroupStripOptions.color_key or grouping.group_key must be set for "
            "the group strip."
        )
    if group_col not in neurons_df.columns:
        raise ValueError(f"Column '{group_col}' not found in neurons_df.")

    cb_args = dict(_STRIP_DEFAULTS)
    if strip.layout:
        cb_args.update(strip.layout)

    cax = _add_strip_axes(ax_raster, strip.side, cb_args)

    # Resolve subgroup and top-group labels per neuron in plotting order.
    sub_labels_raw = groups.subgroup_labels[groups.sorted_indices]
    top_group_labels = groups.group_labels[groups.sorted_indices]
    label_list = sub_labels_raw.tolist()

    use_subgroups = group_key is not None and group_col != group_key
    if use_subgroups:
        if strip.label_mode == "top":
            label_list = [str(top) for top in top_group_labels]
        elif strip.label_mode == "sub":
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
            strip.cmap,
            strip.cmap,
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

    _add_strip_labels(cax, label_list, n_neurons, strip.side, cb_args)

    cax.set_xlim(0, 1)
    cax.set_ylim(ax_raster.get_ylim())
    cax.set_xticks([])
    cax.set_yticks([])
    cax.set_frame_on(False)
    for spine in cax.spines.values():
        spine.set_visible(False)

    if strip.legend:
        _add_strip_legend(
            cax, colors, top_groups_order, subgroups_by_top, strip.label_mode, cb_args
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
    if style.per_neuron_colors:
        # Explicit per-neuron colours from the caller override the strip colours.
        spike_colors = np.asarray(style.c_array, dtype=object)
    else:
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
        # marker_list is empty when there are no spikes at all.
        marker_use = (
            marker_list[0]
            if marker_list is not None and len(marker_list) > 0
            else style.marker
        )
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
        fr = to_numpy(rate)
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
            spikes_np, width=rate_window_ms / eff_dt, dt=eff_dt * 1e-3, batch_axis=-1
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
        group_rates = {k: to_numpy(v) for k, v in group_rate.items()}
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
        group_rate_arr = to_numpy(group_rate)
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
                    batch_axis=-1,
                ),
            )


def _wants_total_rate(rate: RatePanelOptions) -> bool:
    # isinstance first: bool() on a multi-element array/tensor raises.
    return isinstance(rate.total, (np.ndarray, torch.Tensor)) or bool(rate.total)


def _wants_group_rate(rate: RatePanelOptions) -> bool:
    return isinstance(rate.per_group, (dict, np.ndarray, torch.Tensor)) or bool(
        rate.per_group
    )


def _draw_rate_panel(
    ax_raster: Axes,
    ax_rate: Axes,
    t: np.ndarray,
    spikes_np: np.ndarray,
    dt: float | None,
    xlabel: str,
    rate: RatePanelOptions,
    groups: _RasterGroups,
    style: RasterStyle,
    strip: GroupStripOptions,
) -> None:
    """Fill the rate axes and move the x label from the raster to it."""
    fr = _resolve_total_rate(rate.total, spikes_np, t, dt, rate.window_ms)

    if _wants_group_rate(rate):
        _plot_group_rates(
            ax_rate,
            t,
            spikes_np,
            dt,
            rate.window_ms,
            rate.per_group,
            groups,
            style.spike_color,
            strip.cmap,
        )

    if fr is not None:
        ax_rate.plot(t, fr, color="black", lw=1.8, alpha=0.9, zorder=2)
    ax_rate.set_xlim(t[0], t[-1])
    ax_rate.set_ylabel("Rate (Hz)")
    ax_rate.set_xlabel(xlabel)
    ax_raster.set_xticklabels([])
    ax_raster.set_xlabel("")
