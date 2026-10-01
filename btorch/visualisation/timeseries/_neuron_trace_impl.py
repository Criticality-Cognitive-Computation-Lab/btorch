"""Private building blocks for :func:`plot_neuron_traces`."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from textwrap import wrap
from typing import Any, Callable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...utils.array import to_numpy
from .neuron_specs import (
    NeuronSpec,
    SimulationStates,
    TracePlotFormat,
)


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

    arr = to_numpy(data)
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


def _format_top_neuron_label(label: str, max_chars_per_line: int) -> str:
    """Wrap long top labels to reduce horizontal crowding."""
    if max_chars_per_line <= 0 or len(label) <= max_chars_per_line:
        return label
    return "\n".join(wrap(label, width=max_chars_per_line, break_long_words=False))


_DEFAULT_TRACE_COLORS = {
    "voltage": "#2E86AB",
    "asc": "#A23B72",
    "psc": "#F18F01",
    "epsc": "#06A77D",
    "ipsc": "#D62246",
    "input": "#9467bd",
    "spike": "#000000",
}
_COMBINED_TITLES = {
    "voltage": "Voltage",
    "asc": "Afterspike Current",
    "psc": "Postsynaptic Current",
}
_SEPARATE_TITLES = {**_COMBINED_TITLES, "voltage": "Voltage Traces"}


@dataclass
class _TraceConfig:
    """Arguments of :func:`plot_neuron_traces` before/after merging
    dataclasses."""

    voltage: Any
    dt: float | None
    asc: Any
    psc: Any
    epsc: Any
    ipsc: Any
    input: Any
    psc_labels: Sequence[str] | None
    spikes: Any
    v_threshold: Any
    v_reset: Any
    neuron_indices: list[int] | None
    sample_size: int | None
    seed: int
    show_voltage: bool
    show_asc: bool
    show_psc: bool
    neuron_labels: Sequence[str] | Callable[[int], str] | None
    neuron_label_position: str
    neuron_specs: list[NeuronSpec | dict] | NeuronSpec | dict | None
    separate_figures: bool
    auto_width: bool
    neurons_per_row: int | None
    batch_idx: int | None


@dataclass
class _TraceData:
    """Numpy trace arrays with the batch dimension removed."""

    voltage: np.ndarray
    spikes: np.ndarray | None
    asc: np.ndarray | None
    psc: np.ndarray | None
    epsc: np.ndarray | None
    ipsc: np.ndarray | None
    input: np.ndarray | None
    psc_labels: Sequence[str] | None
    psc_has_extra_dim: bool


def _merge_trace_config(
    cfg: _TraceConfig,
    states: SimulationStates | pd.DataFrame | None,
    format: TracePlotFormat | None,
) -> _TraceConfig:
    """Fill unset arguments from ``states`` and override from ``format``.

    ``dt=None`` means "not given" (taken from ``states``, else 1.0 ms); the
    sentinel ``seed == 42`` still means "not explicitly given".
    """
    if states is not None:
        cfg.voltage = states.voltage if cfg.voltage is None else cfg.voltage
        cfg.dt = states.dt if cfg.dt is None else cfg.dt
        cfg.asc = states.asc if cfg.asc is None else cfg.asc
        cfg.psc = states.psc if cfg.psc is None else cfg.psc
        cfg.epsc = states.epsc if cfg.epsc is None else cfg.epsc
        cfg.ipsc = states.ipsc if cfg.ipsc is None else cfg.ipsc
        cfg.input = states.input if cfg.input is None else cfg.input
        cfg.spikes = states.spikes if cfg.spikes is None else cfg.spikes
        if cfg.v_threshold is None:
            cfg.v_threshold = states.v_threshold
        cfg.v_reset = states.v_reset if cfg.v_reset is None else cfg.v_reset

    if format is not None:
        if cfg.neuron_indices is None:
            cfg.neuron_indices = format.neuron_indices
        if cfg.sample_size is None:
            cfg.sample_size = format.sample_size
        cfg.seed = format.seed if cfg.seed == 42 else cfg.seed
        cfg.show_voltage = format.show_voltage
        cfg.show_asc = format.show_asc
        cfg.show_psc = format.show_psc
        if cfg.neuron_labels is None:
            cfg.neuron_labels = format.neuron_labels
        cfg.neuron_label_position = format.neuron_label_position
        if cfg.neuron_specs is None:
            cfg.neuron_specs = format.neuron_specs
        cfg.separate_figures = format.separate_figures
        cfg.auto_width = format.auto_width
        if cfg.neurons_per_row is None:
            cfg.neurons_per_row = format.neurons_per_row
        if cfg.batch_idx is None:
            cfg.batch_idx = format.batch_idx
    if cfg.dt is None:
        cfg.dt = 1.0
    return cfg


def _prepare_trace_data(cfg: _TraceConfig) -> _TraceData:
    """Validate PSC layout and strip the batch dimension from every array."""
    batch_idx = 0 if cfg.batch_idx is None else cfg.batch_idx
    psc_labels = cfg.psc_labels

    # Multi-component PSC layouts: (time, neurons, n_psc) next to 2D voltage
    # (no batch dimension), or (time, batch, neurons, n_psc) (4D).  A 3D PSC
    # next to 3D voltage is a plain batched PSC (time, batch, neurons).
    psc_has_extra_dim = False
    psc_raw = to_numpy(cfg.psc) if cfg.psc is not None else None
    if psc_raw is not None:
        voltage_shape = to_numpy(cfg.voltage).shape
        if psc_raw.ndim == 4:
            psc_has_extra_dim = True
        elif (
            psc_raw.ndim == 3
            and len(voltage_shape) == 2
            and psc_raw.shape[1] == voltage_shape[1]
        ):
            psc_has_extra_dim = True
        if psc_has_extra_dim:
            for name, value in (
                ("epsc", cfg.epsc),
                ("ipsc", cfg.ipsc),
                ("input", cfg.input),
            ):
                if value is not None:
                    raise ValueError(
                        f"{name} must be None when psc has additional dimension "
                        "(n_psc > 1)"
                    )
            if psc_labels is None:
                psc_labels = [f"PSC_{i}" for i in range(psc_raw.shape[-1])]

    voltage = _extract_batch_dim(cfg.voltage, batch_idx)
    spikes = _extract_batch_dim(cfg.spikes, batch_idx)
    asc = _extract_batch_dim(cfg.asc, batch_idx)
    if psc_has_extra_dim and psc_raw.ndim == 3:
        psc = psc_raw
    else:
        psc = _extract_batch_dim(cfg.psc, batch_idx)
    epsc = _extract_batch_dim(cfg.epsc, batch_idx)
    ipsc = _extract_batch_dim(cfg.ipsc, batch_idx)
    input_current = _extract_batch_dim(cfg.input, batch_idx)
    return _TraceData(
        voltage=to_numpy(voltage),
        spikes=spikes,
        asc=asc,
        psc=psc,
        epsc=epsc,
        ipsc=ipsc,
        input=input_current,
        psc_labels=psc_labels,
        psc_has_extra_dim=psc_has_extra_dim,
    )


def _select_trace_neurons(
    n_neurons: int,
    neuron_indices: list[int] | None,
    sample_size: int | None,
    seed: int,
) -> list[int]:
    """Choose neurons to plot: explicit, random sample, or the first five."""
    if neuron_indices is None and sample_size is None:
        return list(range(min(5, n_neurons)))
    if neuron_indices is None:
        np.random.seed(seed)
        return sorted(
            np.random.choice(n_neurons, min(sample_size, n_neurons), replace=False)
        )
    return neuron_indices


def _make_label_resolver(
    neuron_labels: Sequence[str] | Callable[[int], str] | None,
) -> Callable[[int, int], str | None]:
    """Return ``resolve(plot_idx, neuron_idx)`` for callable/sequence
    labels."""

    def resolve(plot_idx: int, neuron_idx: int) -> str | None:
        if callable(neuron_labels):
            return str(neuron_labels(neuron_idx))
        if neuron_labels is not None and plot_idx < len(neuron_labels):
            return str(neuron_labels[plot_idx])
        return None

    return resolve


def _resolve_neuron_spec(
    neuron_specs: list[NeuronSpec | dict] | NeuronSpec | dict | None, plot_idx: int
) -> NeuronSpec:
    """Return the spec for a plotted neuron (list by position, dict for
    all)."""
    spec = NeuronSpec()
    if isinstance(neuron_specs, list):
        if plot_idx < len(neuron_specs):
            s = neuron_specs[plot_idx]
            spec = NeuronSpec(**s) if isinstance(s, dict) else s
    elif isinstance(neuron_specs, dict):
        spec = NeuronSpec(**neuron_specs)
    elif isinstance(neuron_specs, NeuronSpec):
        spec = neuron_specs
    return spec


def _colors_for_spec(colors: dict[str, str], spec: NeuronSpec) -> dict[str, str]:
    """Apply a spec colour (single colour or per-trace dict) over
    ``colors``."""
    local_colors = colors.copy()
    if spec.color is not None:
        if isinstance(spec.color, dict):
            local_colors.update(spec.color)
        else:
            for k in local_colors:
                if k != "spike":
                    local_colors[k] = spec.color
    return local_colors


def _trace_panel_kinds(
    cfg: _TraceConfig, data: _TraceData
) -> list[Literal["voltage", "asc", "psc"]]:
    """Panels to draw: requested by the user and backed by data."""
    kinds: list[Literal["voltage", "asc", "psc"]] = []
    if cfg.show_voltage:
        kinds.append("voltage")
    if cfg.show_asc and data.asc is not None:
        kinds.append("asc")
    if cfg.show_psc and data.psc is not None:
        kinds.append("psc")
    return kinds


def _draw_psc_panel(
    ax: Axes,
    times: np.ndarray,
    data: _TraceData,
    neuron_idx: int,
    colors: dict[str, str],
    style: dict[str, Any],
) -> bool:
    """Plot PSC traces; return whether the panel should get a legend."""
    if data.psc_has_extra_dim:
        psc_traces = data.psc[:, neuron_idx, :]  # (time, n_psc)
        _plot_multi_psc_on_ax(ax, times, psc_traces, data.psc_labels, colors, **style)
        return True
    _plot_psc_on_ax(
        ax,
        times,
        data.psc[:, neuron_idx],
        data.epsc[:, neuron_idx] if data.epsc is not None else None,
        data.ipsc[:, neuron_idx] if data.ipsc is not None else None,
        data.input[:, neuron_idx] if data.input is not None else None,
        colors,
        **style,
    )
    return data.epsc is not None or data.ipsc is not None or data.input is not None


def _draw_trace_panel(
    ax: Axes,
    kind: str,
    data: _TraceData,
    neuron_idx: int,
    times: np.ndarray,
    colors: dict[str, str],
    format: TracePlotFormat | None,
    v_th: float | None,
    v_reset: float | None,
    style: dict[str, Any],
    title: str | None,
) -> None:
    """Draw one trace panel; ``title`` (first row only) also adds a legend."""
    if kind == "voltage":
        _plot_voltage_on_ax(
            ax,
            times,
            data.voltage[:, neuron_idx],
            data.spikes[:, neuron_idx] if data.spikes is not None else None,
            colors,
            format,
            v_th,
            v_reset,
            **style,
        )
        ax.set_ylabel("V (mV)")
        with_legend = v_th is not None or v_reset is not None
    elif kind == "asc":
        _plot_simple_trace_on_ax(
            ax, times, data.asc[:, neuron_idx], colors["asc"], "ASC (pA)", **style
        )
        with_legend = False
    else:
        with_legend = _draw_psc_panel(ax, times, data, neuron_idx, colors, style)

    if title is not None:
        ax.set_title(title)
        if with_legend:
            ax.legend(loc="upper right", fontsize=8)


def _finish_trace_axis(ax: Axes, is_last_row: bool) -> None:
    if is_last_row:
        ax.set_xlabel("Time (ms)")
    ax.grid(alpha=0.3, linewidth=0.5)


def _add_side_label(ax: Axes, label: str) -> None:
    """Bold label to the right of an axes."""
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


def _plot_separate_trace_figures(
    kinds: list[str],
    data: _TraceData,
    neuron_indices: list[int],
    times: np.ndarray,
    colors: dict[str, str],
    format: TracePlotFormat | None,
    v_thresholds: list[float | None],
    v_resets: list[float | None],
    resolve_label: Callable[[int, int], str | None],
    label_position: str,
    figsize: tuple[float, float],
    neuron_specs: list[NeuronSpec | dict] | NeuronSpec | dict | None = None,
) -> dict[str, Figure]:
    """Create one figure per trace type with one row per neuron."""
    n_plot = len(neuron_indices)
    figures: dict[str, Figure] = {}
    for kind in kinds:
        fig, axes = plt.subplots(n_plot, 1, figsize=figsize, squeeze=False)
        for i, neuron_idx in enumerate(neuron_indices):
            ax = axes[i, 0]
            spec = _resolve_neuron_spec(neuron_specs, i)
            _draw_trace_panel(
                ax,
                kind,
                data,
                neuron_idx,
                times,
                _colors_for_spec(colors, spec),
                format,
                v_thresholds[i],
                v_resets[i],
                {
                    "linestyle": spec.linestyle,
                    "linewidth": spec.linewidth,
                    "alpha": spec.alpha,
                },
                _SEPARATE_TITLES[kind] if i == 0 else None,
            )
            _finish_trace_axis(ax, i == n_plot - 1)
            label = (
                spec.label if spec.label is not None else resolve_label(i, neuron_idx)
            )
            if label is None:
                continue
            if label_position == "top":
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
                _add_side_label(ax, label)

        plt.tight_layout()
        figures[kind] = fig
    return figures


@dataclass
class _TraceGrid:
    """Figure and axes of the combined trace layout."""

    fig: Figure
    axes: dict[tuple[int, int], Axes]
    label_axes: dict[tuple[int, int], Axes]
    n_rows: int
    n_cols: int
    neurons_per_row: int
    top_labels: bool
    max_label_chars_per_line: int

    def plot_row(self, row_idx: int) -> int:
        """Grid row holding the traces of neuron row ``row_idx``."""
        return row_idx * 2 + 1 if self.top_labels else row_idx


def _create_trace_grid(
    n_plot: int,
    n_cols: int,
    neurons_per_row: int,
    top_labels: bool,
    max_label_len: int,
    base_width: float,
    height_per_row: float,
) -> _TraceGrid:
    """Create the figure with one trace axes per (neuron row, slot, panel)."""
    n_rows = int(ceil(n_plot / neurons_per_row))
    total_cols = n_cols * neurons_per_row
    label_height_ratio = 0.22
    max_label_chars_per_line = 0
    if top_labels and max_label_len > 0:
        # Rough estimate for wrapping purposes only (not for sizing)
        max_label_chars_per_line = max(36, int(base_width * 8))
        label_line_count = max(
            1, int(ceil(max_label_len / max(max_label_chars_per_line, 1)))
        )
        label_height_ratio = 0.22 + 0.12 * (label_line_count - 1)
    total_height = (
        height_per_row * n_rows * (1.0 + label_height_ratio if top_labels else 1.0)
    )
    # Keep enough width per trace column to avoid label crowding.
    base_width = max(base_width, 4.0 * n_cols)
    fig = plt.figure(figsize=(base_width * neurons_per_row, total_height))
    gridspec_kw = (
        {"height_ratios": [v for _ in range(n_rows) for v in (label_height_ratio, 1.0)]}
        if top_labels
        else {}
    )
    grid_spec = fig.add_gridspec(
        n_rows * 2 if top_labels else n_rows, total_cols, **gridspec_kw
    )
    grid = _TraceGrid(
        fig,
        {},
        {},
        n_rows,
        n_cols,
        neurons_per_row,
        top_labels,
        max_label_chars_per_line,
    )

    for row_idx in range(n_rows):
        if top_labels:
            for slot_idx in range(neurons_per_row):
                col_base = slot_idx * n_cols
                label_ax = fig.add_subplot(
                    grid_spec[row_idx * 2, col_base : col_base + n_cols]
                )
                label_ax.set_axis_off()
                grid.label_axes[(row_idx, slot_idx)] = label_ax

        plot_row = grid.plot_row(row_idx)
        for c in range(total_cols):
            grid.axes[(plot_row, c)] = fig.add_subplot(grid_spec[plot_row, c])
    return grid


def _draw_combined_traces(
    grid: _TraceGrid,
    kinds: list[str],
    data: _TraceData,
    neuron_indices: list[int],
    times: np.ndarray,
    colors: dict[str, str],
    format: TracePlotFormat | None,
    v_thresholds: list[float | None],
    v_resets: list[float | None],
    neuron_specs: list[NeuronSpec | dict] | NeuronSpec | dict | None,
    resolved_labels: list[str | None],
) -> set[tuple[int, int]]:
    """Draw every neuron's panels and labels; return the axes that were
    used."""
    used_axes: set[tuple[int, int]] = set()
    n_cols = grid.n_cols

    for plot_idx, neuron_idx in enumerate(neuron_indices):
        row_idx, slot_idx = divmod(plot_idx, grid.neurons_per_row)
        plot_row = grid.plot_row(row_idx)
        spec = _resolve_neuron_spec(neuron_specs, plot_idx)
        label = spec.label if spec.label is not None else resolved_labels[plot_idx]
        local_colors = _colors_for_spec(colors, spec)
        style = {
            "linestyle": spec.linestyle,
            "linewidth": spec.linewidth,
            "alpha": spec.alpha,
        }

        col_base = slot_idx * n_cols
        for col_idx, kind in enumerate(kinds):
            ax = grid.axes[(plot_row, col_base + col_idx)]
            _draw_trace_panel(
                ax,
                kind,
                data,
                neuron_idx,
                times,
                local_colors,
                format,
                v_thresholds[plot_idx],
                v_resets[plot_idx],
                style,
                _COMBINED_TITLES[kind] if row_idx == 0 else None,
            )
            _finish_trace_axis(ax, row_idx == grid.n_rows - 1)
            used_axes.add((plot_row, col_base + col_idx))

        if label is None:
            continue
        if grid.top_labels:
            label_ax = grid.label_axes[(row_idx, slot_idx)]
            label_ax.text(
                0.5,
                0.5,
                _format_top_neuron_label(label, grid.max_label_chars_per_line),
                transform=label_ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="center",
                ha="center",
            )
        else:
            _add_side_label(grid.axes[(plot_row, col_base + n_cols - 1)], label)
    return used_axes


def _hide_unused_trace_axes(
    grid: _TraceGrid, used_axes: set[tuple[int, int]], n_plot: int
) -> None:
    """Hide axes of empty neuron slots in the final row."""
    for r in range(grid.n_rows):
        plot_row = grid.plot_row(r)
        for c in range(grid.n_cols * grid.neurons_per_row):
            if (plot_row, c) not in used_axes:
                grid.axes[(plot_row, c)].set_visible(False)

        if grid.top_labels:
            for slot_idx in range(grid.neurons_per_row):
                if r * grid.neurons_per_row + slot_idx >= n_plot:
                    grid.label_axes[(r, slot_idx)].set_visible(False)


def _widen_for_top_labels(grid: _TraceGrid) -> None:
    """Grow the figure width so the widest top label fits in its slot."""
    fig = grid.fig
    fig.canvas.draw()  # text must be rendered to measure it
    max_label_width_inches = 0.0
    for label_ax in grid.label_axes.values():
        for text in label_ax.texts:
            bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
            max_label_width_inches = max(max_label_width_inches, bbox.width / fig.dpi)
    if max_label_width_inches > 0:
        required_slot_width = max_label_width_inches * 1.2 + 1.0  # 20% pad + margin
        min_fig_width = required_slot_width * grid.neurons_per_row
        if min_fig_width > fig.get_figwidth():
            fig.set_figwidth(min_fig_width)


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

    if spike_trace is not None and (not format or format.show_spikes_on_voltage):
        spike_times = times[spike_trace > 0]
        spike_vals = voltage_trace[spike_trace > 0]
        ax.scatter(
            spike_times, spike_vals, color=colors["spike"], s=20, marker="^", zorder=5
        )

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
    input_trace,
    colors,
    linestyle="-",
    linewidth=0.8,
    alpha=1.0,
):
    """Helper for PSC trace with EPSC, IPSC, and input components."""
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
    if input_trace is not None:
        ax.plot(
            times,
            input_trace,
            color=colors.get("input", "#9467bd"),  # Default purple
            linewidth=0.6,
            alpha=0.7,
            linestyle=":",
            label="Input",
        )
    ax.set_ylabel("PSC (pA)")


def _plot_multi_psc_on_ax(
    ax,
    times,
    psc_traces,
    labels,
    colors,
    linestyle="-",
    linewidth=0.8,
    alpha=1.0,
):
    """Helper for PSC traces with multiple components (n_psc > 1).

    Args:
        ax: Matplotlib axis
        times: Time array
        psc_traces: Array of shape (time, n_psc)
        labels: List of labels for each PSC component
        colors: Color dictionary
        linestyle: Line style
        linewidth: Line width
        alpha: Plot opacity
    """
    n_components = psc_traces.shape[1]
    base_colors = [
        colors.get("psc", "#F18F01"),
        colors.get("epsc", "#06A77D"),
        colors.get("ipsc", "#D62246"),
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # yellow-green
        "#17becf",  # cyan
    ]

    for i in range(n_components):
        color = base_colors[i % len(base_colors)]
        label = labels[i] if labels and i < len(labels) else f"PSC_{i}"
        ax.plot(
            times,
            psc_traces[:, i],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label=label,
        )
    ax.set_ylabel("PSC (pA)")
