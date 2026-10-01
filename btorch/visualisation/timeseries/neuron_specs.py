"""Dataclasses describing neuron trace plotting inputs and format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import torch


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
        psc: Total postsynaptic current (time, neurons), (time, batch, neurons),
            or (time, batch, neurons, n_psc) for multiple PSC components
        epsc: Excitatory PSC (time, neurons) or (time, batch, neurons)
        ipsc: Inhibitory PSC (time, neurons) or (time, batch, neurons)
        input: Input current (time, neurons) or (time, batch, neurons)
        spikes: Spike trains (time, neurons) or (time, batch, neurons)
        v_threshold: Spike threshold voltage(s), scalar or per-neuron
        v_reset: Reset voltage(s), scalar or per-neuron
        psc_labels: Labels for multi-component PSC (defaults to ``PSC_<i>``)
    """

    voltage: np.ndarray | torch.Tensor
    dt: float = 1.0
    asc: np.ndarray | torch.Tensor | None = None
    psc: np.ndarray | torch.Tensor | None = None
    epsc: np.ndarray | torch.Tensor | None = None
    ipsc: np.ndarray | torch.Tensor | None = None
    input: np.ndarray | torch.Tensor | None = None
    spikes: np.ndarray | torch.Tensor | None = None
    v_threshold: float | Sequence[float] | np.ndarray | torch.Tensor | None = None
    v_reset: float | Sequence[float] | np.ndarray | torch.Tensor | None = None
    psc_labels: Sequence[str] | None = None


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
