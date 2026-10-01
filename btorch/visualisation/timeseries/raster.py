"""Spike raster plot with grouping, strips and annotations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
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
    _wants_group_rate,
    _wants_total_rate,
)
from .raster_options import (
    GroupStripOptions,
    RasterAnnotations,
    RasterGrouping,
    RasterStyle,
    RatePanelOptions,
)


def plot_raster(
    spikes: np.ndarray | torch.Tensor,
    *,
    dt: float | None = None,
    times: Sequence[float] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Time (ms)",
    ylabel: str = "Neuron Index",
    style: RasterStyle | None = None,
    grouping: RasterGrouping | None = None,
    strip: GroupStripOptions | None = None,
    rate: RatePanelOptions | None = None,
    annotations: RasterAnnotations | None = None,
) -> Axes | tuple[Axes, Axes]:
    """Plot spike raster with optional grouping and styling.

    Args:
        spikes: Spike matrix of shape (time, neurons).
        dt: Time step in ms. Default is 1.0 if ``times`` is not provided.
        times: Explicit time array.
        ax: Axis to plot on. If None, a new figure is created. Ignored (with a
            warning) when a rate panel is requested.
        title: Plot title; defaults to a "Fired/Spikes" summary.
        xlabel: Label for the x-axis (moved to the rate panel if present).
        ylabel: Label for the y-axis.
        style: Marker, colour and per-neuron styling (:class:`RasterStyle`).
        grouping: Neuron grouping and separators (:class:`RasterGrouping`).
        strip: Group colour strip (:class:`GroupStripOptions`); None draws no
            strip.
        rate: Firing-rate panel (:class:`RatePanelOptions`); None draws no
            panel.
        annotations: Events, regions and tracks (:class:`RasterAnnotations`).

    Returns:
        ``ax_raster``, or ``(ax_raster, ax_rate)`` when a rate panel is drawn.

    Raises:
        ValueError: If ``spikes`` is not 2D, or per-group rates are requested
            without ``grouping.group_key``.
    """
    style_opts = style or RasterStyle()
    grouping = grouping or RasterGrouping()
    strip = strip or GroupStripOptions(show=False)
    rate = rate or RatePanelOptions()
    annotations = annotations or RasterAnnotations()
    group_key = grouping.group_key
    neurons_df = grouping.neurons_df

    spikes_np = to_numpy(spikes)
    if spikes_np.ndim != 2:
        raise ValueError("spikes must be 2D (time, neurons)")

    n_time, n_neurons = spikes_np.shape
    t = _get_time_axis(n_time, dt, times)

    show_group_rate = _wants_group_rate(rate)
    with_rate_panel = _wants_total_rate(rate) or show_group_rate
    if show_group_rate and group_key is None:
        raise ValueError(
            "RatePanelOptions.per_group requires grouping.group_key (and "
            "neurons_df) so per-group rates can be computed."
        )

    ax_raster, ax_rate = _create_raster_axes(ax, n_neurons, with_rate_panel)
    groups = _resolve_raster_groups(n_neurons, grouping, strip)

    # sorted_indices[0] is plotted at y=0 (bottom).
    idx_map = np.empty(n_neurons)
    idx_map[groups.sorted_indices] = np.arange(len(groups.sorted_indices))

    orig_neuron_indices, spike_times = compute_raster(spikes_np, t)
    plot_neuron_indices = idx_map[orig_neuron_indices]

    spike_style = _resolve_spike_style(
        style_opts, grouping, groups, orig_neuron_indices, n_neurons
    )
    # With a group strip, spikes are drawn later using the strip colours.
    if not strip.show:
        _draw_spikes(ax_raster, spike_times, plot_neuron_indices, spike_style)

    ax_raster.set_xlim(t[0], t[-1])
    ax_raster.set_ylim(-0.5, n_neurons - 0.5)
    ax_raster.set_ylabel(ylabel)
    ax_raster.yaxis.set_major_locator(MaxNLocator(integer=True))

    _draw_raster_annotations(ax_raster, t, n_neurons, annotations)

    spike_count = len(spike_times)
    fired_neurons = len(np.unique(orig_neuron_indices)) if spike_count > 0 else 0
    stats_title = f"Fired {fired_neurons}/{n_neurons}, Spikes {spike_count}"
    ax_raster.set_title(title if title else f"Spike raster {stats_title}")

    if group_key and grouping.show_separators:
        _draw_group_separators(
            ax_raster,
            groups.boundaries,
            n_neurons,
            grouping.separator_style,
            label_groups=not strip.show,
            group_strip_side=strip.side,
        )

    if strip.show:
        strip_colors = _draw_group_strip(
            ax_raster, groups, neurons_df, group_key, strip, n_neurons
        )
        _draw_strip_spikes(
            ax_raster,
            spike_times,
            plot_neuron_indices,
            orig_neuron_indices,
            groups,
            strip_colors,
            spike_style,
            style_opts.marker_size,
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

    if ax_rate is None:
        ax_raster.set_xlabel(xlabel)
        return ax_raster

    _draw_rate_panel(
        ax_raster,
        ax_rate,
        t,
        spikes_np,
        dt,
        xlabel,
        rate,
        groups,
        style_opts,
        strip,
    )
    return ax_raster, ax_rate
