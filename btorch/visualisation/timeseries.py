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
from typing import Any, Callable, Literal, Sequence, Union

import matplotlib.gridspec as gridspec
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


def _estimate_text_width_inches(
    text: str, fontsize: float = 10, dpi: float = 100
) -> float:
    """Estimate text width in inches based on character count and font size.

    Uses a heuristic approximation: each character is roughly 0.6 * fontsize
    in width at standard DPI.
    """
    if not text:
        return 0.0
    # Approximate character width: 0.6 * fontsize points per char
    # 1 inch = 72 points, 1 point = 1/dpi inches
    char_width_points = 0.6 * fontsize
    total_width_points = len(text) * char_width_points
    return total_width_points / 72.0


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
        return [to_hex(cmap(0.0))]
    # Request a colormap with N=n to avoid duplicate bins for ListedColormap
    # (e.g., tab10) when n exceeds the base number of colors.
    cmap = plt.get_cmap(cmap_name, n)
    vals = np.linspace(0, 1, n, endpoint=True)
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


def plot_raster(
    spikes: Union[np.ndarray, torch.Tensor],
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
) -> Union[Axes, tuple[Axes, Axes]]:
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

    # Evaluate isinstance first to avoid calling bool() on arrays/tensors
    # (which raises ValueError for >1-element numpy arrays).
    show_rate = isinstance(rate, (np.ndarray, torch.Tensor)) or bool(rate)
    show_group_rate = isinstance(group_rate, (dict, np.ndarray, torch.Tensor)) or bool(
        group_rate
    )

    raster_height = _auto_raster_height(n_neurons)
    raster_width = 8.0
    rate_height = 2.6  # keep rate panel at a stable height

    if show_rate or show_group_rate:
        if ax is not None:
            warnings.warn(
                "ax argument is ignored when rate/group_rate is enabled. "
                "Creating new figure."
            )
        fig, (ax_raster, ax_rate) = plt.subplots(
            2,
            1,
            figsize=(raster_width, raster_height + rate_height),
            gridspec_kw={
                "height_ratios": [raster_height, rate_height],
                "hspace": 0.06,
            },
        )
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(raster_width, raster_height))
        ax_raster = ax
        ax_rate = None

    # Handle Grouping
    sorted_indices = np.arange(n_neurons)
    group_boundaries = []  # List of (y_coord, label)

    group_labels = np.full(n_neurons, "Unknown", dtype=object)
    subgroup_labels = None

    if group_key is not None or group_color_key is not None:
        if neurons_df is None:
            raise ValueError("neurons_df must be provided when grouping is used.")
        if group_key is not None and group_key not in neurons_df.columns:
            raise ValueError(f"Column '{group_key}' not found in neurons_df.")
        if group_color_key is not None and group_color_key not in neurons_df.columns:
            raise ValueError(f"Column '{group_color_key}' not found in neurons_df.")

    if group_key is not None:
        group_values = neurons_df[group_key].to_numpy()
        n_copy = min(n_neurons, len(group_values))
        group_labels[:n_copy] = group_values[:n_copy]
        group_labels[pd.isna(group_labels)] = "Unknown"

    if group_color_key is not None:
        subgroup_labels = np.full(n_neurons, "Unknown", dtype=object)
        sub_values = neurons_df[group_color_key].to_numpy()
        n_copy = min(n_neurons, len(sub_values))
        subgroup_labels[:n_copy] = sub_values[:n_copy]
        subgroup_labels[pd.isna(subgroup_labels)] = "Unknown"
    else:
        subgroup_labels = group_labels

    if group_key is not None:
        present_groups = set(group_labels.tolist())
        if group_sort:
            groups = [g for g in group_sort if g in present_groups]
            remaining = sorted(present_groups - set(groups))
            groups.extend(remaining)
        else:
            groups = sorted(present_groups)

        if sort_neurons:
            new_order = []
            current_y = 0
            for g in groups:
                g_indices = np.flatnonzero(group_labels == g)
                if g_indices.size == 0:
                    continue
                if group_color_key is not None:
                    subgroup_vals = subgroup_labels[g_indices]
                    subgroup_order = list(dict.fromkeys(subgroup_vals.tolist()))
                    order_map = {k: i for i, k in enumerate(subgroup_order)}
                    subgroup_rank = np.array(
                        [order_map[v] for v in subgroup_vals], dtype=int
                    )
                    g_indices = g_indices[np.argsort(subgroup_rank, kind="stable")]

                new_order.append(g_indices)
                current_y += g_indices.size
                group_boundaries.append((current_y - 0.5, g))

            if new_order:
                sorted_indices = np.concatenate(new_order)
            if len(sorted_indices) < n_neurons:
                warnings.warn(
                    "Not all neurons were assigned to a group. Appending defaults."
                )
                missing = np.setdiff1d(np.arange(n_neurons), sorted_indices)
                sorted_indices = np.concatenate([sorted_indices, missing])
        else:
            sorted_indices = np.arange(n_neurons)
            prev_g = None
            for i, idx in enumerate(sorted_indices):
                g = group_labels[idx]
                if i == 0:
                    prev_g = g
                else:
                    if g != prev_g:
                        group_boundaries.append((i - 0.5, prev_g))
                        prev_g = g

    # Mapping from original index to plot y-index
    # y-axis: 0 at bottom, N-1 at top.
    # If we want group 0 at top, we should reverse? Standard raster usually 0 at bottom.
    # Let's stick to 0 at bottom.
    # sorted_indices[0] is plotted at y=0.

    # We need a map: original_idx -> y_coord
    idx_map = np.empty(n_neurons)
    idx_map[sorted_indices] = np.arange(len(sorted_indices))

    # Compute raster coordinates
    # spike indices are row indices in spikes_np (time)??
    # No, usually spikes is (time, neurons).
    # compute_raster returns (neuron_indices, spike_times) where indices are 0..N-1
    orig_neuron_indices, spike_times = compute_raster(spikes_np, t)

    # Map neuron indices to sorted plot positions
    plot_neuron_indices = idx_map[orig_neuron_indices]

    # Handle Colors
    c_array = spike_color
    skip_main_scatter = False
    draw_spikes_later = show_group_strip
    ms_array = marker_size  # default fallback if no specs
    marker_list = None
    size_list = None

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
        c_array = color_by_neuron[orig_neuron_indices]
    elif neuron_specs is not None:
        c_list = []
        m_list = []
        ms_list = []

        # Helper to get spec for an index
        def get_spec_attrs(idx):
            s = None
            if isinstance(neuron_specs, list):
                if idx < len(neuron_specs):
                    s = neuron_specs[idx]
            elif isinstance(neuron_specs, dict):
                if idx in neuron_specs:
                    s = neuron_specs[idx]

            c = "black"
            m = marker
            ms = marker_size

            if s is not None:
                if isinstance(s, NeuronSpec):
                    c = s.color if s.color is not None else c
                    m = s.marker if s.marker is not None else m
                    ms = s.markersize if s.markersize is not None else ms
                elif isinstance(s, dict):
                    c = s.get("color", c)
                    m = s.get("marker", m)
                    ms = s.get("markersize", ms)
            return c, m, ms

        for orig_idx in orig_neuron_indices:
            c, m, ms = get_spec_attrs(orig_idx)
            c_list.append(c)
            m_list.append(m)
            ms_list.append(ms)

        c_array = c_list
        marker_list = np.array(m_list)
        size_list = np.array(ms_list)
        # If markers vary, we might need multiple scatter calls or loop.
        # Matplotlib scatter accepts list of colors/sizes
        # but SINGLE marker style usually.
        # Actually scatter does NOT accept list of markers.
        # We must group by marker type if markers vary.

        # Check if multiple markers used
        unique_markers = set(m_list)
        if len(unique_markers) > 1:
            if not show_group_strip:
                # We need to loop
                for um in unique_markers:
                    mask = np.array(m_list) == um
                    # Line-based markers (x, +, |, _) need linewidths > 0
                    lw = 0.5 if um in ("x", "+", "|", "_", "1", "2", "3", "4") else 0
                    ax_raster.scatter(
                        spike_times[mask],
                        plot_neuron_indices[mask],
                        s=np.array(ms_list)[mask],
                        c=np.array(c_list, dtype=object)[mask],
                        marker=um,
                        linewidths=lw,
                    )
            # Skip the main scatter call
            skip_main_scatter = True
        else:
            marker = m_list[0] if m_list else marker
            ms_array = ms_list
            skip_main_scatter = False

    if not skip_main_scatter:
        # If sizes vary? scatter accepts array of sizes 's'
        if neuron_specs is not None:
            # attributes were collected above
            s_arg = ms_array
        else:
            s_arg = marker_size

        # Line-based markers need linewidths > 0
        lw = 0.5 if marker in ("x", "+", "|", "_", "1", "2", "3", "4") else 0

        if not draw_spikes_later:
            ax_raster.scatter(
                spike_times,
                plot_neuron_indices,
                s=s_arg,
                c=c_array,
                marker=marker,
                linewidths=lw,
            )
    ax_raster.set_xlim(t[0], t[-1])
    ax_raster.set_ylim(-0.5, n_neurons - 0.5)
    ax_raster.set_ylabel(ylabel)
    ax_raster.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Advanced Annotations
    # 1. Tracks (Horizontal lines for each neuron)
    if show_tracks:
        # For large N, this might be heavy. Use LineCollection?
        # Or just simple axhlines if N is not too huge.
        # For very large N, maybe skip or use alpha.
        track_alpha = 0.1 if n_neurons > 100 else 0.2
        track_lw = 0.5
        # Draw lines at 0, 1, ... N-1
        # range(n_neurons) maps to y positions.
        # But we actually want lines at integer positions.
        ax_raster.hlines(
            y=np.arange(n_neurons),
            xmin=t[0],
            xmax=t[-1],
            colors="gray",
            alpha=track_alpha,
            linewidth=track_lw,
            zorder=0,
        )

    # 2. Events (Vertical lines)
    if events is not None:
        def_evt_kwargs = {
            "color": "red",
            "linestyle": "--",
            "alpha": 0.8,
            "linewidth": 1.0,
        }
        if event_kwargs:
            def_evt_kwargs.update(event_kwargs)

        if isinstance(events, dict):
            # Cycle colors if not specified? Or just use default.
            # Ideally one color per key if user wants?
            # For now use default kwargs for all
            for label, times in events.items():
                for et in times:
                    ax_raster.axvline(x=et, **def_evt_kwargs)
        else:
            # Sequence
            for et in events:
                ax_raster.axvline(x=et, **def_evt_kwargs)

    # 3. Regions (Shaded intervals)
    if regions is not None:
        def_reg_kwargs = {"color": "yellow", "alpha": 0.2}
        if region_kwargs:
            def_reg_kwargs.update(region_kwargs)

        if isinstance(regions, dict):
            for label, intervals in regions.items():
                for start, end in intervals:
                    ax_raster.axvspan(start, end, **def_reg_kwargs)
        else:
            for start, end in regions:
                ax_raster.axvspan(start, end, **def_reg_kwargs)

    spike_count = len(spike_times)
    fired_neurons = len(np.unique(orig_neuron_indices)) if spike_count > 0 else 0
    stats_title = f"Fired {fired_neurons}/{n_neurons}, Spikes {spike_count}"

    if title:
        ax_raster.set_title(title)
    else:
        ax_raster.set_title(f"Spike raster {stats_title}")

    # Add separators and group labels
    if group_key and show_group_separators:
        sep_args = (
            separator_style
            if separator_style
            else {"color": "gray", "linestyle": "--", "alpha": 0.5, "linewidth": 0.8}
        )

        # We have boundaries at the TOP of groups.
        # We also need to label them. Ideally label is centered in the group band.

        prev_y = -0.5
        for y_limit, label in group_boundaries:
            if y_limit < n_neurons - 0.5:  # Don't draw line at very top if fully filled
                ax_raster.axhline(y_limit, **sep_args)

            # Add text label only when no strip is shown (strip draws labels itself)
            if not show_group_strip:
                mid_y = (prev_y + y_limit) / 2
                label_x = -0.02 if group_strip_side == "left" else 1.01
                label_ha = "right" if group_strip_side == "left" else "left"
                ax_raster.text(
                    label_x,
                    mid_y,
                    str(label),
                    transform=ax_raster.get_yaxis_transform(),
                    va="center",
                    ha=label_ha,
                    fontsize=8,
                    color=sep_args.get("color", "black"),
                )

            prev_y = y_limit

    # Optional group strip
    if show_group_strip:
        if neurons_df is None:
            raise ValueError("neurons_df must be provided for group strip.")

        group_col = group_color_key or group_key
        if group_col is None:
            raise ValueError(
                "group_color_key or group_key must be set for group strip."
            )
        if group_col not in neurons_df.columns:
            raise ValueError(f"Column '{group_col}' not found in neurons_df.")

        cb_args = {
            "width": 0.06,
            "pad": 0.005,
            "alpha": 0.9,
            "label_fontsize": 7,
            "label_weight": "bold",
            "legend_fontsize": 6,
            "legend_ncol_threshold": 15,
            "min_label_distance": 0.02,
            "min_span_frac": 0.01,
            "span_line_frac": 0.005,
            "strip_x0": 0.3,
            "strip_width": 0.4,
            "label_x": None,
            "label_gap": 0.05,
            "bracket_x0": 0.78,
            "bracket_x1": 0.95,
            "label_sep": " / ",
            "group_sep_color": "black",
            "group_sep_lw": 1.4,
            "sub_hue_span": 0.06,
            "sub_val_span": 0.18,
            "left_extra_pad": 0.04,
        }
        if group_strip_kwargs:
            cb_args.update(group_strip_kwargs)

        fig = ax_raster.figure
        pos = ax_raster.get_position()
        if group_strip_side == "right":
            cax_x0 = pos.x1 + cb_args["pad"]
        else:
            cax_x0 = (
                pos.x0 - cb_args["pad"] - cb_args["width"] - cb_args["left_extra_pad"]
            )
        cax = fig.add_axes([cax_x0, pos.y0, cb_args["width"], pos.height])

        if group_strip_side == "left":
            ylabel_x = (cax_x0 - cb_args["pad"] - pos.x0) / pos.width
            ax_raster.yaxis.set_label_coords(ylabel_x, 0.5)

        # Resolve subgroup and top-group labels per neuron in sorted order
        sub_labels_raw = subgroup_labels[sorted_indices]
        top_group_labels = group_labels[sorted_indices]
        group_labels_list = sub_labels_raw.tolist()

        use_subgroups = group_key is not None and group_col != group_key
        if use_subgroups:
            if group_label_mode == "top":
                group_labels_list = [str(top) for top in top_group_labels]
            elif group_label_mode == "sub":
                group_labels_list = [str(sub) for sub in group_labels_list]
            else:
                group_labels_list = [
                    f"{top}{cb_args['label_sep']}{sub}"
                    for top, sub in zip(top_group_labels, group_labels_list)
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

        # Draw patches using group/subgroup colors
        for i, _ in enumerate(group_labels_list):
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

        # Compute ranges for labels
        type_ranges: dict[str, dict[str, int]] = {}
        for i, label in enumerate(group_labels_list):
            if label not in type_ranges:
                type_ranges[label] = {"start": i, "end": i}
            else:
                type_ranges[label]["end"] = i

        unique_types = list(dict.fromkeys(group_labels_list))
        sorted_types = sorted(unique_types, key=lambda x: type_ranges[x]["start"])
        label_positions: list[float] = []
        for label in sorted_types:
            start_idx = type_ranges[label]["start"]
            end_idx = type_ranges[label]["end"]
            mid_y = (start_idx + end_idx) / 2

            min_distance = n_neurons * cb_args["min_label_distance"]
            too_close = any(abs(mid_y - pos) < min_distance for pos in label_positions)

            if (not too_close) or (
                (end_idx - start_idx) > (n_neurons * cb_args["min_span_frac"])
            ):
                label_x = cb_args["label_x"]
                if label_x is None:
                    if group_strip_side == "right":
                        label_x = (
                            cb_args["strip_x0"]
                            + cb_args["strip_width"]
                            + cb_args["label_gap"]
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

        cax.set_xlim(0, 1)
        cax.set_ylim(ax_raster.get_ylim())
        cax.set_xticks([])
        cax.set_yticks([])
        cax.set_frame_on(False)
        for spine in cax.spines.values():
            spine.set_visible(False)

        if group_strip_legend:
            if group_label_mode == "top":
                legend_elements = [
                    mpatches.Patch(color=base_colors[tg], label=str(tg))
                    for tg in top_groups_order
                ]
            elif group_label_mode == "sub":
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

        # If we postponed spike drawing earlier, now draw spikes with
        # matching group/subgroup colors derived above.
        if draw_spikes_later:
            # Build color list per spike (orig_neuron_indices order)
            top_vals = group_labels[orig_neuron_indices]
            sub_vals = subgroup_labels[orig_neuron_indices]
            spike_colors = []
            for top, sub in zip(top_vals, sub_vals):
                if use_subgroups:
                    spike_colors.append(
                        subgroup_colors.get((top, sub), base_colors.get(top, "black"))
                    )
                else:
                    spike_colors.append(base_colors.get(sub, "black"))

            spike_colors = np.array(spike_colors, dtype=object)
            if size_list is not None:
                s_arg = size_list
            else:
                s_arg = marker_size

            if marker_list is not None and len(set(marker_list)) > 1:
                for um in sorted(set(marker_list)):
                    mask = marker_list == um
                    lw = 0.5 if um in ("x", "+", "|", "_", "1", "2", "3", "4") else 0
                    ax_raster.scatter(
                        spike_times[mask],
                        plot_neuron_indices[mask],
                        s=s_arg[mask],
                        c=spike_colors[mask],
                        marker=um,
                        linewidths=lw,
                    )
            else:
                marker_use = marker_list[0] if marker_list is not None else marker
                lw = (
                    0.5 if marker_use in ("x", "+", "|", "_", "1", "2", "3", "4") else 0
                )

                ax_raster.scatter(
                    spike_times,
                    plot_neuron_indices,
                    s=s_arg,
                    c=spike_colors,
                    marker=marker_use,
                    linewidths=lw,
                )

    ax_raster.text(
        0.01,
        0.99,  # Move to top left to avoid conflict with right-side group labels
        f"N={spike_count}",
        transform=ax_raster.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        fontsize=8,
    )

    if show_rate or show_group_rate:
        assert ax_rate is not None
        fr = None
        if isinstance(rate, (np.ndarray, torch.Tensor)):
            fr = _to_numpy(rate)
            if fr.ndim == 2 and fr.shape[1] == 1:
                fr = fr[:, 0]
            if fr.ndim != 1:
                raise ValueError("rate must be 1D or shape (T, 1).")
            if fr.shape[0] != n_time:
                raise ValueError("rate length must match time axis length.")
        elif rate is True:
            eff_dt = dt if dt is not None else (t[1] - t[0] if len(t) > 1 else 1.0)
            fr = firing_rate(
                spikes_np, width=rate_window_ms / eff_dt, dt=eff_dt * 1e-3, axis=-1
            )

        group_alpha = 0.45
        group_lw = 0.9
        group_zorder = 1
        total_lw = 1.8
        total_zorder = 2

        if show_group_rate and group_key is not None:
            if group_key not in (neurons_df.columns if neurons_df is not None else []):
                raise ValueError(
                    "neurons_df with group_key is required for group_rate."
                )
            group_color_map: dict[str, Any] = {}
            if isinstance(spike_color, dict):
                if not any(isinstance(k, (int, np.integer)) for k in spike_color):
                    group_color_map = dict(spike_color)

            if not group_color_map:
                group_palette = _sample_cmap_colors(strip_cmap, len(groups))
                group_color_map = dict(zip(groups, group_palette))

            if isinstance(group_rate, dict):
                group_rates = {k: _to_numpy(v) for k, v in group_rate.items()}
                for g in groups:
                    if g not in group_rates:
                        continue
                    g_rate = group_rates[g]
                    if g_rate.ndim == 2 and g_rate.shape[1] == 1:
                        g_rate = g_rate[:, 0]
                    if g_rate.ndim != 1 or g_rate.shape[0] != n_time:
                        raise ValueError(
                            "group_rate values must be 1D and match time axis."
                        )
                    ax_rate.plot(
                        t,
                        g_rate,
                        color=group_color_map.get(g, "black"),
                        alpha=group_alpha,
                        lw=group_lw,
                        zorder=group_zorder,
                        label=str(g),
                    )
            elif isinstance(group_rate, (np.ndarray, torch.Tensor)):
                group_rate_arr = _to_numpy(group_rate)
                if group_rate_arr.ndim != 2 or group_rate_arr.shape[0] != n_time:
                    raise ValueError("group_rate array must have shape (T, G).")
                if group_rate_arr.shape[1] != len(groups):
                    raise ValueError("group_rate array must match number of groups.")
                for idx, g in enumerate(groups):
                    ax_rate.plot(
                        t,
                        group_rate_arr[:, idx],
                        color=group_color_map.get(g, "black"),
                        alpha=group_alpha,
                        lw=group_lw,
                        zorder=group_zorder,
                        label=str(g),
                    )
            elif group_rate is True:
                eff_dt = dt if dt is not None else (t[1] - t[0] if len(t) > 1 else 1.0)
                for g in groups:
                    g_indices = np.flatnonzero(group_labels == g)
                    if g_indices.size == 0:
                        continue
                    g_rate = firing_rate(
                        spikes_np[:, g_indices],
                        width=rate_window_ms / eff_dt,
                        dt=eff_dt * 1e-3,
                        axis=-1,
                    )
                    ax_rate.plot(
                        t,
                        g_rate,
                        color=group_color_map.get(g, "black"),
                        alpha=group_alpha,
                        lw=group_lw,
                        zorder=group_zorder,
                        label=str(g),
                    )

        if fr is not None:
            ax_rate.plot(
                t,
                fr,
                color="black",
                lw=total_lw,
                alpha=0.9,
                zorder=total_zorder,
            )
        ax_rate.set_xlim(t[0], t[-1])
        ax_rate.set_ylabel("Rate (Hz)")
        ax_rate.set_xlabel(xlabel)
        # Hide x-labels of raster
        ax_raster.set_xticklabels([])
        ax_raster.set_xlabel("")

        return ax_raster, ax_rate
    else:
        ax_raster.set_xlabel(xlabel)
        return ax_raster


def plot_traces(
    data: Union[np.ndarray, torch.Tensor],
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

    # Select neurons
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
        # Generate distinct colors for each neuron
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

            # Construct label
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
    data: Union[np.ndarray, torch.Tensor],
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

    # Defaults
    trace_color = color if color else "blue"
    mean_color = color if color else "black"

    if show_mean and data_np.ndim > 1:
        # Plot individual traces
        if alpha > 0:
            ax.plot(freqs, y_data, color=trace_color, alpha=alpha, lw=0.5)
        # Plot mean
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
    data: Union[np.ndarray, torch.Tensor],
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

    # 2. Defaults
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = {g: cmap(i % 10) for i, g in enumerate(groups.keys())}

    # 3. Plotting
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
    values: Union[np.ndarray, torch.Tensor],
    ax: Axes | None = None,
    title: str = "Distribution",
    xlabel: str = "Value",
    **kwargs,
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
        psc: Total postsynaptic current (time, neurons) or (time, batch, neurons)
        epsc: Excitatory PSC (time, neurons) or (time, batch, neurons)
        ipsc: Inhibitory PSC (time, neurons) or (time, batch, neurons)
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
        psc: Postsynaptic current traces (time, neurons) or (time, batch, neurons)
        epsc: Excitatory PSC traces (time, neurons) or (time, batch, neurons)
        ipsc: Inhibitory PSC traces (time, neurons) or (time, batch, neurons)
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
    # Resolve dataclass vs plain args
    if states is not None:
        voltage = states.voltage if voltage is None else voltage
        dt = states.dt if dt == 1.0 else dt
        asc = states.asc if asc is None else asc
        psc = states.psc if psc is None else psc
        epsc = states.epsc if epsc is None else epsc
        ipsc = states.ipsc if ipsc is None else ipsc
        spikes = states.spikes if spikes is None else spikes
        v_threshold = states.v_threshold if v_threshold is None else v_threshold
        v_reset = states.v_reset if v_reset is None else v_reset

    if format is not None:
        neuron_indices = (
            format.neuron_indices if neuron_indices is None else neuron_indices
        )
        sample_size = format.sample_size if sample_size is None else sample_size
        seed = format.seed if seed == 42 else seed
        show_voltage = format.show_voltage
        show_asc = format.show_asc
        show_psc = format.show_psc
        neuron_labels = format.neuron_labels if neuron_labels is None else neuron_labels
        neuron_label_position = format.neuron_label_position
        neuron_specs = format.neuron_specs if neuron_specs is None else neuron_specs
        separate_figures = format.separate_figures
        auto_width = format.auto_width
        neurons_per_row = (
            format.neurons_per_row if neurons_per_row is None else neurons_per_row
        )
        batch_idx = format.batch_idx if batch_idx is None else batch_idx

    # Validate required data
    if voltage is None:
        raise ValueError("voltage is required (provide via states or direct arg)")

    # Default batch_idx to 0 if data is 3D and no index specified
    if batch_idx is None:
        batch_idx = 0

    # Extract batch dimension from all data arrays
    voltage = _extract_batch_dim(voltage, batch_idx)
    spikes = _extract_batch_dim(spikes, batch_idx)
    asc = _extract_batch_dim(asc, batch_idx)
    psc = _extract_batch_dim(psc, batch_idx)
    epsc = _extract_batch_dim(epsc, batch_idx)
    ipsc = _extract_batch_dim(ipsc, batch_idx)

    # Convert to numpy (preserve None for optional arrays)
    voltage = _to_numpy(voltage)
    spikes = _to_numpy(spikes) if spikes is not None else None
    n_time, n_neurons = voltage.shape
    times = np.arange(n_time) * dt
    duration_ms = n_time * dt

    # Select neurons to plot
    if neuron_indices is None and sample_size is None:
        # Default: plot first 5 neurons
        neuron_indices = list(range(min(5, n_neurons)))
    elif neuron_indices is None:
        # Random sample
        np.random.seed(seed)
        neuron_indices = sorted(
            np.random.choice(n_neurons, min(sample_size, n_neurons), replace=False)
        )

    n_plot = len(neuron_indices)
    if neurons_per_row is None:
        neurons_per_row = 1
    if neurons_per_row < 1:
        raise ValueError("neurons_per_row must be >= 1")

    v_threshold_per_neuron = _resolve_per_neuron_values(
        v_threshold, neuron_indices, n_neurons, "v_threshold"
    )
    v_reset_per_neuron = _resolve_per_neuron_values(
        v_reset, neuron_indices, n_neurons, "v_reset"
    )

    # Determine figure dimensions
    base_width = 12.0
    if auto_width:
        # Scale: ~1 inch per 40ms, bounded [10, 30]
        base_width = max(10.0, min(duration_ms * 0.025, 30.0))
    elif format:
        base_width = format.figsize_per_neuron[0]

    height_per_row = format.figsize_per_neuron[1] if format else 2.5
    total_height = height_per_row * n_plot

    # Default colors
    default_colors = {
        "voltage": "#2E86AB",
        "asc": "#A23B72",
        "psc": "#F18F01",
        "epsc": "#06A77D",
        "ipsc": "#D62246",
        "spike": "#000000",
    }
    colors = format.colors if format and format.colors else default_colors

    label_values: Sequence[str] | None = None
    label_fn: Callable[[int], str] | None = None
    if callable(neuron_labels):
        label_fn = neuron_labels
    elif neuron_labels is not None:
        label_values = neuron_labels

    def _resolve_side_label(
        plot_idx: int, neuron_idx: int, spec: NeuronSpec | None = None
    ) -> str | None:
        if spec is not None and spec.label is not None:
            return spec.label
        if label_fn is not None:
            return str(label_fn(neuron_idx))
        if label_values is not None and plot_idx < len(label_values):
            return str(label_values[plot_idx])
        return None

    # Determine subplot layout based on data availability
    # Only show columns if requested AND data is present
    _show_v = show_voltage and (voltage is not None)
    _show_asc = show_asc and (asc is not None)
    _show_psc = show_psc and (psc is not None)

    if separate_figures:
        figures = {}
        trace_types = []
        if _show_v:
            trace_types.append("voltage")
        if _show_asc:
            trace_types.append("asc")
        if _show_psc:
            trace_types.append("psc")

        for t_type in trace_types:
            fig, axes = plt.subplots(
                n_plot, 1, figsize=(base_width, total_height), squeeze=False
            )

            for i, neuron_idx in enumerate(neuron_indices):
                ax = axes[i, 0]
                label = _resolve_side_label(i, neuron_idx)

                if t_type == "voltage":
                    _plot_voltage_on_ax(
                        ax,
                        times,
                        voltage[:, neuron_idx],
                        spikes[:, neuron_idx] if spikes is not None else None,
                        colors,
                        format,
                        v_threshold_per_neuron[i],
                        v_reset_per_neuron[i],
                    )
                    ax.set_ylabel("V (mV)")
                    if i == 0:
                        ax.set_title("Voltage Traces")
                        if (
                            v_threshold_per_neuron[i] is not None
                            or v_reset_per_neuron[i] is not None
                        ):
                            ax.legend(loc="upper right", fontsize=8)

                elif t_type == "asc":
                    asc_arr = _to_numpy(asc)
                    _plot_simple_trace_on_ax(
                        ax, times, asc_arr[:, neuron_idx], colors["asc"], "ASC (pA)"
                    )
                    if i == 0:
                        ax.set_title("Afterspike Current")

                elif t_type == "psc":
                    psc_arr = _to_numpy(psc)
                    epsc_arr = (
                        _to_numpy(epsc[:, neuron_idx]) if epsc is not None else None
                    )
                    ipsc_arr = (
                        _to_numpy(ipsc[:, neuron_idx]) if ipsc is not None else None
                    )
                    _plot_psc_on_ax(
                        ax, times, psc_arr[:, neuron_idx], epsc_arr, ipsc_arr, colors
                    )
                    if i == 0:
                        ax.set_title("Postsynaptic Current")
                        if epsc is not None or ipsc is not None:
                            ax.legend(loc="upper right", fontsize=8)

                if i == n_plot - 1:
                    ax.set_xlabel("Time (ms)")
                ax.grid(alpha=0.3, linewidth=0.5)
                if label is not None:
                    if neuron_label_position == "top":
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

            plt.tight_layout()
            figures[t_type] = fig

        return figures

    # Combined figure
    n_cols = sum([_show_v, _show_asc, _show_psc])
    if n_cols == 0:
        # Default fallback: if nothing strictly requested by data presence,
        # but voltage is required arg, show voltage
        if voltage is not None:
            _show_v = True
            n_cols = 1
        else:
            raise ValueError(
                "No data available to plot (voltage, asc, or psc required)"
            )

    n_rows = int(ceil(n_plot / neurons_per_row))
    use_top_label_rows = neuron_label_position == "top"
    total_height_grid = height_per_row * n_rows * (1.2 if use_top_label_rows else 1.0)
    # Keep enough width per trace column to avoid label crowding.
    base_width = max(base_width, 4.0 * n_cols)
    fig_width = base_width * neurons_per_row

    # Pre-resolve labels to estimate width for adaptive sizing
    # when using top-positioned labels
    if use_top_label_rows and (label_fn is not None or label_values is not None):
        resolved_labels = [
            _resolve_side_label(i, neuron_indices[i]) for i in range(n_plot)
        ]
        max_label_width_inches = max(
            (
                _estimate_text_width_inches(label, fontsize=10)
                for label in resolved_labels
                if label
            ),
            default=0.0,
        )
        # Each neuron slot must be wide enough for its label
        # Add padding (1.5x) to ensure labels don't crowd
        min_slot_width = max_label_width_inches * 1.5
        slot_width = fig_width / neurons_per_row
        if min_slot_width > slot_width:
            # Expand figure width to accommodate labels
            fig_width = min_slot_width * neurons_per_row

    fig = plt.figure(figsize=(fig_width, total_height_grid))

    if use_top_label_rows:
        # Single GridSpec: 2 rows per neuron row (label + plot)
        # Use spanning cells for labels
        total_gs_rows = n_rows * 2
        total_gs_cols = neurons_per_row * n_cols
        gs = gridspec.GridSpec(
            total_gs_rows,
            total_gs_cols,
            figure=fig,
            height_ratios=[v for _ in range(n_rows) for v in (0.22, 1.0)],
            hspace=0.3,
            wspace=0.3,
        )
        axes: dict[tuple[int, int, int], Axes] = {}
        label_axes: dict[tuple[int, int], Axes] = {}

        for row in range(n_rows):
            label_row = row * 2
            plot_row = row * 2 + 1
            for slot in range(neurons_per_row):
                col_start = slot * n_cols
                col_end = (slot + 1) * n_cols

                # Label axis spans all columns for this neuron slot
                label_ax = fig.add_subplot(gs[label_row, col_start:col_end])
                label_ax.set_axis_off()
                label_axes[(row, slot)] = label_ax

                # Plot axes for each column
                for col in range(n_cols):
                    ax = fig.add_subplot(gs[plot_row, col_start + col])
                    axes[(row, slot, col)] = ax
    else:
        # Simple grid without label rows
        gs = gridspec.GridSpec(
            n_rows,
            neurons_per_row * n_cols,
            figure=fig,
            height_ratios=[1.0] * n_rows,
            hspace=0.3,
            wspace=0.3,
        )
        axes = {}
        label_axes = {}
        for row in range(n_rows):
            for slot in range(neurons_per_row):
                for col in range(n_cols):
                    axes[(row, slot, col)] = fig.add_subplot(
                        gs[row, slot * n_cols + col]
                    )

    asc_arr = _to_numpy(asc) if _show_asc else None
    psc_arr = _to_numpy(psc) if _show_psc else None
    used_axes: set[tuple[int, int, int]] = set()

    for plot_idx, neuron_idx in enumerate(neuron_indices):
        row_idx = plot_idx // neurons_per_row
        slot_idx = plot_idx % neurons_per_row
        # In the new nested gridspec structure, each row is a separate
        # subgridspec with columns 0 to n_cols-1
        plot_row = row_idx

        # Resolve spec
        spec = NeuronSpec()
        if neuron_specs is not None:
            if isinstance(neuron_specs, list):
                if plot_idx < len(neuron_specs):
                    s = neuron_specs[plot_idx]
                    spec = NeuronSpec(**s) if isinstance(s, dict) else s
            elif isinstance(neuron_specs, dict):
                spec = NeuronSpec(**neuron_specs)
            elif isinstance(neuron_specs, NeuronSpec):
                spec = neuron_specs

        label = _resolve_side_label(plot_idx, neuron_idx, spec)

        # Color resolution
        local_colors = colors.copy()
        if spec.color is not None:
            if isinstance(spec.color, dict):
                local_colors.update(spec.color)
            else:
                for k in local_colors:
                    if k != "spike":
                        local_colors[k] = spec.color

        col_idx = 0

        # Voltage subplot
        if _show_v:
            ax = axes[(plot_row, slot_idx, col_idx)]
            _plot_voltage_on_ax(
                ax,
                times,
                voltage[:, neuron_idx],
                spikes[:, neuron_idx] if spikes is not None else None,
                local_colors,
                format,
                v_threshold_per_neuron[plot_idx],
                v_reset_per_neuron[plot_idx],
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                alpha=spec.alpha,
            )
            ax.set_ylabel("V (mV)")
            if row_idx == 0:
                ax.set_title("Voltage")
                if (
                    v_threshold_per_neuron[plot_idx] is not None
                    or v_reset_per_neuron[plot_idx] is not None
                ):
                    ax.legend(loc="upper right", fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (ms)")
            ax.grid(alpha=0.3, linewidth=0.5)
            used_axes.add((plot_row, slot_idx, col_idx))
            col_idx += 1

        # ASC subplot
        if _show_asc and asc_arr is not None:
            ax = axes[(plot_row, slot_idx, col_idx)]
            _plot_simple_trace_on_ax(
                ax,
                times,
                asc_arr[:, neuron_idx],
                local_colors["asc"],
                "ASC (pA)",
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                alpha=spec.alpha,
            )
            if row_idx == 0:
                ax.set_title("Afterspike Current")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (ms)")
            ax.grid(alpha=0.3, linewidth=0.5)
            used_axes.add((plot_row, slot_idx, col_idx))
            col_idx += 1

        # PSC subplot
        if _show_psc and psc_arr is not None:
            ax = axes[(plot_row, slot_idx, col_idx)]
            epsc_arr = _to_numpy(epsc[:, neuron_idx]) if epsc is not None else None
            ipsc_arr = _to_numpy(ipsc[:, neuron_idx]) if ipsc is not None else None
            _plot_psc_on_ax(
                ax,
                times,
                psc_arr[:, neuron_idx],
                epsc_arr,
                ipsc_arr,
                local_colors,
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                alpha=spec.alpha,
            )
            if row_idx == 0:
                ax.set_title("Postsynaptic Current")
                if epsc is not None or ipsc is not None:
                    ax.legend(loc="upper right", fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (ms)")
            ax.grid(alpha=0.3, linewidth=0.5)
            used_axes.add((plot_row, slot_idx, col_idx))
            col_idx += 1

        if label is not None:
            if use_top_label_rows:
                # Use the spanning label axis for this neuron slot
                label_ax = label_axes[(row_idx, slot_idx)]
                label_ax.text(
                    0.5,
                    0.5,
                    label,
                    transform=label_ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    ha="center",
                )
            else:
                # Add label to the rightmost subplot in this neuron slot.
                last_ax = axes[(plot_row, slot_idx, n_cols - 1)]
                last_ax.text(
                    1.02,
                    0.5,
                    label,
                    transform=last_ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    ha="left",
                )

    # Hide unused plot axes for empty neuron slots in the final row.
    for r in range(n_rows):
        for slot in range(neurons_per_row):
            for c in range(n_cols):
                if (r, slot, c) not in used_axes:
                    if (r, slot, c) in axes:
                        axes[(r, slot, c)].set_visible(False)

    right_margin = 0.96 if neuron_label_position == "side" else 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=(0.0, 0.0, right_margin, 1.0), w_pad=0.8, h_pad=0.8)
    return fig


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

    # Mark spikes
    if spike_trace is not None and (not format or format.show_spikes_on_voltage):
        spike_times = times[spike_trace > 0]
        spike_vals = voltage_trace[spike_trace > 0]
        ax.scatter(
            spike_times, spike_vals, color=colors["spike"], s=20, marker="^", zorder=5
        )

    # Reference lines
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
    colors,
    linestyle="-",
    linewidth=0.8,
    alpha=1.0,
):
    """Helper for PSC trace."""
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
    ax.set_ylabel("PSC (pA)")
