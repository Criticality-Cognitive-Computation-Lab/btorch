"""Per-neuron voltage / current trace grids."""

from __future__ import annotations

import warnings

import numpy as np
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
    _plot_separate_trace_figures,
    _prepare_trace_data,
    _select_trace_neurons,
    _trace_panel_kinds,
    _widen_for_top_labels,
)
from .neuron_specs import SimulationStates, TracePlotFormat


def plot_neuron_traces(
    states: SimulationStates,
    format: TracePlotFormat | None = None,
) -> Figure | dict[str, Figure]:
    """Plot neuron state traces.

    Each selected neuron gets a row of subplots showing voltage, ASC and PSC
    traces. All data goes into ``states`` and all presentation options
    (neuron selection, panels, labels, layout) into ``format``.

    Args:
        states: Simulation state arrays, ``dt``, thresholds and PSC labels
            (:class:`SimulationStates`). Arrays are (time, neurons) or
            (time, batch, neurons); ASC/PSC may add a trailing component
            axis. If ``psc`` has a component axis (n_psc > 1), ``epsc``,
            ``ipsc`` and ``input`` must be None.
        format: Neuron selection, panel visibility, labels, per-neuron styles
            and figure layout (:class:`TracePlotFormat`). Defaults to
            ``TracePlotFormat()``.

    Returns:
        Figure with neuron trace subplots, or a dict of figures (one per trace
        type) when ``format.separate_figures`` is True.

    Raises:
        ValueError: On invalid shapes, ``neurons_per_row < 1``, conflicting PSC
            layouts, or a bad ``batch_idx`` / threshold length.
    """
    fmt = format or TracePlotFormat()
    data = _prepare_trace_data(states, fmt.batch_idx)
    n_time, n_neurons = data.voltage.shape
    times = np.arange(n_time) * states.dt
    duration_ms = n_time * states.dt

    neuron_indices = _select_trace_neurons(
        n_neurons, fmt.neuron_indices, fmt.sample_size, fmt.seed
    )
    n_plot = len(neuron_indices)
    neurons_per_row = 1 if fmt.neurons_per_row is None else fmt.neurons_per_row
    if neurons_per_row < 1:
        raise ValueError("neurons_per_row must be >= 1")

    v_thresholds = _resolve_per_neuron_values(
        states.v_threshold, neuron_indices, n_neurons, "v_threshold"
    )
    v_resets = _resolve_per_neuron_values(
        states.v_reset, neuron_indices, n_neurons, "v_reset"
    )

    if fmt.auto_width:
        # ~1 inch per 40 ms, bounded to [10, 30]
        base_width = max(10.0, min(duration_ms * 0.025, 30.0))
    else:
        base_width = fmt.figsize_per_neuron[0]
    height_per_row = fmt.figsize_per_neuron[1]

    colors = fmt.colors or _DEFAULT_TRACE_COLORS
    resolve_label = _make_label_resolver(fmt.neuron_labels)
    kinds = _trace_panel_kinds(fmt, data)

    if fmt.separate_figures:
        return _plot_separate_trace_figures(
            kinds,
            data,
            neuron_indices,
            times,
            colors,
            fmt,
            v_thresholds,
            v_resets,
            resolve_label,
            fmt.neuron_label_position,
            (base_width, height_per_row * n_plot),
            fmt.neuron_specs,
        )

    if not kinds:
        # Nothing requested/available: fall back to the required voltage panel.
        kinds = ["voltage"]

    resolved_labels = [
        resolve_label(plot_idx, neuron_idx)
        for plot_idx, neuron_idx in enumerate(neuron_indices)
    ]
    max_label_len = max((len(label) for label in resolved_labels if label), default=0)
    top_labels = fmt.neuron_label_position == "top"

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
        fmt,
        v_thresholds,
        v_resets,
        fmt.neuron_specs,
        resolved_labels,
    )
    _hide_unused_trace_axes(grid, used_axes, n_plot)
    if top_labels and grid.label_axes:
        _widen_for_top_labels(grid)

    right_margin = 0.96 if fmt.neuron_label_position == "side" else 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid.fig.tight_layout(rect=(0.0, 0.0, right_margin, 1.0), w_pad=0.8, h_pad=0.8)
    return grid.fig
