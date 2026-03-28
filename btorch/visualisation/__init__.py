"""Visualization tools for neuromorphic data analysis.

This module provides plotting utilities for spike trains, network dynamics,
connectome structure, and neuron state traces. The API is organized into
five plotting families:

**Aggregation plots** (`aggregation`):
- Grouped distributions:
  [`plot_group_distribution`][btorch.visualisation.aggregation.plot_group_distribution],
  [`plot_group_violin`][btorch.visualisation.aggregation.plot_group_violin],
  [`plot_group_box`][btorch.visualisation.aggregation.plot_group_box],
  [`plot_group_ecdf`][btorch.visualisation.aggregation.plot_group_ecdf]
- Neuropil timeseries:
  [`plot_neuropil_timeseries_overview`][btorch.visualisation.aggregation.plot_neuropil_timeseries_overview],
  [`plot_neuropil_timeseries_panels`][btorch.visualisation.aggregation.plot_neuropil_timeseries_panels]

**Dynamics plots** (`dynamics`):
- Multiscale analysis:
  [`plot_multiscale_fano`][btorch.visualisation.dynamics.plot_multiscale_fano],
  [`plot_dfa_analysis`][btorch.visualisation.dynamics.plot_dfa_analysis],
  [`plot_isi_cv`][btorch.visualisation.dynamics.plot_isi_cv]
- Criticality and attractors:
  [`plot_avalanche_analysis`][btorch.visualisation.dynamics.plot_avalanche_analysis],
  [`plot_eigenvalue_spectrum`][btorch.visualisation.dynamics.plot_eigenvalue_spectrum],
  [`plot_lyapunov_spectrum`][btorch.visualisation.dynamics.plot_lyapunov_spectrum]
- Micro-dynamics:
  [`plot_firing_rate_distribution`][btorch.visualisation.dynamics.plot_firing_rate_distribution],
  [`plot_micro_dynamics`][btorch.visualisation.dynamics.plot_micro_dynamics],
  [`plot_gain_stability`][btorch.visualisation.dynamics.plot_gain_stability]

**Timeseries plots** (`timeseries`):
- Spike visualization:
  [`plot_raster`][btorch.visualisation.timeseries.plot_raster]
- Continuous traces:
  [`plot_traces`][btorch.visualisation.timeseries.plot_traces],
  [`plot_neuron_traces`][btorch.visualisation.timeseries.plot_neuron_traces]
- Spectral analysis:
  [`plot_spectrum`][btorch.visualisation.timeseries.plot_spectrum],
  [`plot_log_hist`][btorch.visualisation.timeseries.plot_log_hist]

**Network plots** (`network`, `hexmap`):
- Graph layout: [`plot_network`][btorch.visualisation.network.plot_network]
- Hexagonal heatmaps: [`hex_heatmap`][btorch.visualisation.hexmap.hex_heatmap]

**Tuning plots** (`tuning`):
- Response curves: [`plot_fi_vi_curve`][btorch.visualisation.tuning.plot_fi_vi_curve]
"""

from .aggregation import (
    plot_group_box,
    plot_group_distribution,
    plot_group_ecdf,
    plot_group_violin,
    plot_neuropil_timeseries_overview,
    plot_neuropil_timeseries_panels,
)
from .dynamics import (
    DFAConfig,
    DynamicsData,
    DynamicsPlotFormat,
    FanoFactorConfig,
    plot_avalanche_analysis,
    plot_dfa_analysis,
    plot_eigenvalue_spectrum,
    plot_firing_rate_distribution,
    plot_gain_stability,
    plot_isi_cv,
    plot_lyapunov_spectrum,
    plot_micro_dynamics,
    plot_multiscale_fano,
)
from .hexmap import hex_heatmap
from .network import plot_network
from .timeseries import (
    SimulationStates,
    TracePlotFormat,
    plot_log_hist,
    plot_neuron_traces,
    plot_raster,
    plot_spectrum,
    plot_traces,
)


__all__ = [
    "animate_3d_activity",
    "animate_mem_potential",
    "hex_heatmap",
    "plot_network",
    "plot_group_box",
    "plot_group_distribution",
    "plot_group_ecdf",
    "plot_group_violin",
    "plot_neuropil_timeseries_overview",
    "plot_neuropil_timeseries_panels",
    "plot_agg_by_neuropil",
    "plot_neuropil_comparison",
    "plot_3d_activities",
    "plot_log_hist",
    "plot_raster",
    "plot_spectrum",
    "plot_traces",
    "plot_neuron_traces",
    "SimulationStates",
    "TracePlotFormat",
    "plot_multiscale_fano",
    "plot_dfa_analysis",
    "plot_isi_cv",
    "DynamicsData",
    "DFAConfig",
    "DynamicsPlotFormat",
    "FanoFactorConfig",
    "plot_avalanche_analysis",
    "plot_eigenvalue_spectrum",
    "plot_firing_rate_distribution",
    "plot_gain_stability",
    "plot_lyapunov_spectrum",
    "plot_micro_dynamics",
]
