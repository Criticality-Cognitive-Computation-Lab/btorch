"""Per-neuron voltage / current trace grids."""

from __future__ import annotations

import warnings
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from ._helpers import (
    _resolve_per_neuron_values,
)
from ._neuron_trace_impl import (
    _DEFAULT_TRACE_COLORS,
    _create_trace_grid,
    _draw_combined_traces,
    _hide_unused_trace_axes,
    _make_label_resolver,
    _merge_trace_config,
    _plot_separate_trace_figures,
    _prepare_trace_data,
    _select_trace_neurons,
    _trace_panel_kinds,
    _TraceConfig,
    _widen_for_top_labels,
)
from .neuron_specs import (
    NeuronSpec,
    SimulationStates,
    TracePlotFormat,
)


def plot_neuron_traces(
    # Dataclass interface
    states: SimulationStates | pd.DataFrame | None = None,
    format: TracePlotFormat | None = None,
    # Plain args interface
    voltage: np.ndarray | torch.Tensor | None = None,
    dt: float | None = None,
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
        dt: Timestep in ms. If None, ``states.dt`` is used when ``states`` is
            given, otherwise 1.0. An explicit value always wins.
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
            cfg.neuron_specs,
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
