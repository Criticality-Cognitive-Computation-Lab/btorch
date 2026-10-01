"""Option dataclasses for
:func:`~btorch.visualisation.timeseries.plot_raster`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import torch

from .neuron_specs import NeuronSpec


@dataclass(frozen=True, kw_only=True)
class RasterStyle:
    """Spike marker appearance.

    Attributes:
        spike_color: Default spike colour. A dict maps group names or neuron
            indices to colours; a sequence gives one colour per neuron.
        marker: Matplotlib marker for spikes.
        marker_size: Marker size in points squared.
        neuron_specs: Per-neuron styling (dict keyed by neuron index, list, or
            a single :class:`NeuronSpec`).
    """

    spike_color: str | dict | Sequence[Any] | None = "black"
    marker: str = "."
    marker_size: float = 5.0
    neuron_specs: dict | list | NeuronSpec | None = None


@dataclass(frozen=True, kw_only=True)
class RasterGrouping:
    """Neuron grouping, ordering and group separators.

    Attributes:
        neurons_df: Neuron metadata, required for any grouping.
        group_key: Column of ``neurons_df`` to group neurons by.
        group_sort: Explicit group order (unknown groups are ignored).
        sort_neurons: Reorder neurons by group/subgroup so bands are
            contiguous; if False the original order is kept.
        show_separators: Draw lines (and labels) between groups.
        separator_style: Extra keyword arguments for the separator lines.
    """

    neurons_df: pd.DataFrame | None = None
    group_key: str | None = None
    group_sort: list[str] | None = None
    sort_neurons: bool = True
    show_separators: bool = True
    separator_style: dict | None = None


@dataclass(frozen=True, kw_only=True)
class GroupStripOptions:
    """Colour strip beside the raster (also sets the group label side).

    Attributes:
        show: Draw the strip. With ``show=False`` only ``color_key`` and
            ``side`` take effect (grouping colour column and label side).
        color_key: Column of ``neurons_df`` used for strip colours; defaults to
            ``RasterGrouping.group_key``. A different column gives subgroups.
        cmap: Matplotlib colormap name for top-group and subgroup colours.
        layout: Overrides for the strip geometry and label styling.
        legend: Add a legend for the group colours.
        label_mode: Which labels to print when subgroups are used.
        side: Side of the raster on which strip and labels are drawn.
    """

    show: bool = True
    color_key: str | None = None
    cmap: str = "tab10"
    layout: dict | None = None
    legend: bool = True
    label_mode: Literal["top", "sub", "top_sub"] = "top_sub"
    side: Literal["left", "right"] = "right"


@dataclass(frozen=True, kw_only=True)
class RatePanelOptions:
    """Firing-rate panel below the raster.

    Attributes:
        total: If True compute the population rate; an array of length ``T``
            (or ``(T, 1)``) is plotted as given.
        per_group: If True compute one rate per group (needs a group key); a
            dict maps group names to arrays, an array is ``(T, G)`` in the
            order of the resolved groups.
        window_ms: Smoothing window for computed rates, in ms.
    """

    total: bool | np.ndarray | torch.Tensor | None = False
    per_group: bool | dict[str, np.ndarray | torch.Tensor] | np.ndarray | None = False
    window_ms: float = 10.0


@dataclass(frozen=True, kw_only=True)
class RasterAnnotations:
    """Events, shaded regions and optional event tracks.

    Attributes:
        events: Event times, or a dict mapping names to event times.
        regions: ``(start, end)`` spans, or a dict mapping names to spans.
        show_tracks: Draw events/regions as tracks below the raster.
        event_kwargs: Extra keyword arguments for event lines.
        region_kwargs: Extra keyword arguments for region patches.
    """

    events: Sequence[float] | dict[str, Sequence[float]] | None = None
    regions: (
        Sequence[tuple[float, float]] | dict[str, Sequence[tuple[float, float]]] | None
    ) = None
    show_tracks: bool = False
    event_kwargs: dict | None = None
    region_kwargs: dict | None = None
