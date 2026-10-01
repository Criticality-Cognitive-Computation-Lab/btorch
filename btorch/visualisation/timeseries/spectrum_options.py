"""Option dataclasses for
:func:`~btorch.visualisation.timeseries.plot_grouped_spectrum`."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, kw_only=True)
class SpectrumGrouping:
    """Which neurons belong to which group.

    Either give ``neurons_df`` plus ``group_by``, or an explicit ``groups``
    mapping (which takes precedence). With neither, all neurons form one
    group called ``"All"``.

    Attributes:
        neurons_df: Neuron metadata; its index holds neuron indices.
        group_by: Column of ``neurons_df`` to group neurons by.
        groups: Manual ``{group_label: [neuron_indices]}`` mapping.
    """

    neurons_df: pd.DataFrame | None = None
    group_by: str | None = None
    groups: dict[str, list[int]] | None = None


@dataclass(frozen=True, kw_only=True)
class SpectrumStyle:
    """Appearance and figure size of the grouped spectrum.

    Attributes:
        show_traces: Draw the individual neuron spectra (faint lines).
        show_mean: Draw the group-mean spectrum.
        colors: ``{group_label: colour}``; defaults to the ``tab10`` cycle.
        plot_width: Width of one panel in inches.
        plot_height: Height of one panel in inches.
    """

    show_traces: bool = True
    show_mean: bool = True
    colors: dict[str, str] | None = None
    plot_width: float = 6.0
    plot_height: float = 4.0
