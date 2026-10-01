"""Spike raster plot with grouping, strips and annotations."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from ...analysis.spiking import compute_raster
from ...utils.array import to_numpy
from ._helpers import (
    _get_time_axis,
)
from ._raster_impl import (
    _create_raster_axes,
    _draw_group_separators,
    _draw_group_strip,
    _draw_raster_annotations,
    _draw_rate_panel,
    _draw_spikes,
    _draw_strip_spikes,
    _resolve_raster_groups,
    _resolve_spike_style,
)
from .neuron_specs import (
    NeuronSpec,
)


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
    spikes_np = to_numpy(spikes)
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
    if show_group_rate and group_key is None:
        raise ValueError(
            "group_rate requires group_key (and neurons_df) so per-group "
            "rates can be computed."
        )

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
