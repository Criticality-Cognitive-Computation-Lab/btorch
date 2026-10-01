"""Timeseries visualization utilities for spike trains and continuous traces.

This package provides plotting functions for:
- Spike raster plots with grouping and styling options
- Continuous timeseries traces (voltage, currents)
- Frequency spectrum analysis
- Log-binned histograms

The raster plot supports neuron grouping, color-coded strips, population
firing rates, and event/region annotations.

Layout: ``raster``, ``traces``, ``spectrum``, ``histogram``, ``neuron_traces``
and ``neuron_specs`` hold the public API; modules prefixed with ``_`` are
private implementation details shared between them.
"""

from .histogram import plot_log_hist
from .neuron_specs import NeuronSpec, SimulationStates, TracePlotFormat
from .neuron_traces import plot_neuron_traces
from .raster import plot_raster
from .spectrum import plot_grouped_spectrum, plot_spectrum
from .traces import plot_traces


__all__ = [
    "NeuronSpec",
    "SimulationStates",
    "TracePlotFormat",
    "plot_grouped_spectrum",
    "plot_log_hist",
    "plot_neuron_traces",
    "plot_raster",
    "plot_spectrum",
    "plot_traces",
]
