"""Frequency spectrum plots."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...analysis.spiking import compute_spectrum
from ...utils.array import to_numpy


def plot_spectrum(
    data: np.ndarray | torch.Tensor,
    dt: float | None = None,
    nperseg: int | None = None,
    ax: Axes | None = None,
    mode: str = "loglog",
    show_mean: bool = True,
    title: str = "Frequency Spectrum",
    color: str | None = None,
    label: str | None = "Mean",
    alpha: float = 0.2,
    mean_linewidth: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, Axes]:
    """Plot frequency spectrum of timeseries data.

    Computes power spectral density using Welch's method and visualizes
    the frequency content. For 2D input (time, neurons), plots individual
    traces with optional mean overlay.

    Args:
        data: Input timeseries with shape (time,) or (time, neurons).
        dt: Sampling interval in ms. Default 1.0.
        nperseg: Length of FFT segments. Default is min(256, time//4).
        ax: Existing axes to plot on. Creates new figure if None.
        mode: Plot scale - "loglog" (default) or "semilogx".
        show_mean: Whether to overlay the mean spectrum (for 2D data).
        title: Plot title.
        color: Color for traces. Uses default if None.
        label: Legend label for mean trace.
        alpha: Opacity for individual traces.
        mean_linewidth: Line width for mean trace.

    Returns:
        Tuple of (frequencies, power_spectrum, axes).

    Example:
        >>> freqs, power, ax = plot_spectrum(spikes, dt=1.0, mode="loglog")
    """
    data_np = to_numpy(data)
    if dt is None:
        dt = 1.0

    freqs, power = compute_spectrum(data_np, dt=dt, nperseg=nperseg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    power_db = 10 * np.log10(power)

    y_data = power if "log" in mode else power_db

    trace_color = color if color else "blue"
    mean_color = color if color else "black"

    if show_mean and data_np.ndim > 1:
        if alpha > 0:
            ax.plot(freqs, y_data, color=trace_color, alpha=alpha, lw=0.5)
        mean_power = y_data.mean(axis=1) if y_data.ndim > 1 else y_data
        ax.plot(freqs, mean_power, color=mean_color, lw=mean_linewidth, label=label)
    else:
        ax.plot(freqs, y_data, color=mean_color, label=label)

    if mode == "loglog":
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Power")
    elif mode == "semilogx":
        ax.set_xscale("log")
        ax.set_ylabel("Power (dB)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(title)

    return freqs, power, ax


def plot_grouped_spectrum(
    data: np.ndarray | torch.Tensor,
    dt: float = 1.0,
    neurons_df: pd.DataFrame | None = None,
    group_by: str | None = None,
    groups: dict[str, list[int]] | None = None,  # Manual override
    mode: Literal["overlay", "subplots"] = "overlay",
    separate_figures: bool = False,
    nperseg: int | None = None,
    show_traces: bool = True,
    show_mean: bool = True,
    colors: dict[str, str] | None = None,
    title: str | None = "Grouped Spectrum",
    plot_width: float = 6.0,
    plot_height: float = 4.0,
) -> Figure | dict[str, Figure]:
    """Plot spectrum for multiple groups.

    Args:
        data: (Time, Neurons)
        dt: Timestep
        neurons_df: Metadata
        group_by: Column to group by
        groups: Manual dict of {group_label: [neuron_indices]}
        mode: "overlay" (all in one) or "subplots" (rows)
        separate_figures: Return dict of figs
        colors: Dict of {group_label: color}
    """
    data_np = to_numpy(data)
    if groups is None:
        if neurons_df is None or group_by is None:
            # No grouping, treat as one group "All"
            groups = {"All": list(range(data_np.shape[1]))}
        else:
            if group_by not in neurons_df.columns:
                raise ValueError(f"Column {group_by} missing")

            groups = {}
            unique_groups = neurons_df[group_by].unique()
            for g in sorted(unique_groups):
                indices = neurons_df.index[neurons_df[group_by] == g].tolist()
                valid_indices = [i for i in indices if i < data_np.shape[1]]
                if valid_indices:
                    groups[g] = valid_indices

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = {g: cmap(i % 10) for i, g in enumerate(groups.keys())}

    if separate_figures:
        figs = {}
        for g_name, indices in groups.items():
            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            group_data = data_np[:, indices]

            c = colors.get(g_name, "black")
            plot_spectrum(
                group_data,
                dt=dt,
                nperseg=nperseg,
                ax=ax,
                color=c,
                label=str(g_name),
                show_mean=show_mean,
                alpha=0.2 if show_traces else 0.0,
            )
            ax.set_title(f"Spectrum: {g_name}")
            figs[str(g_name)] = fig
        return figs

    elif mode == "subplots":
        n_groups = len(groups)
        fig, axes = plt.subplots(
            n_groups, 1, figsize=(plot_width, plot_height * n_groups), squeeze=False
        )
        axes = axes.flatten()

        for i, (g_name, indices) in enumerate(groups.items()):
            ax = axes[i]
            group_data = data_np[:, indices]
            c = colors.get(g_name, "black")

            plot_spectrum(
                group_data,
                dt=dt,
                nperseg=nperseg,
                ax=ax,
                color=c,
                label=str(g_name),
                show_mean=show_mean,
                alpha=0.2 if show_traces else 0.0,
            )
            ax.set_title(str(g_name))
            ax.legend(loc="upper right")

        plt.tight_layout()
        return fig

    else:  # Overlay
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))

        for g_name, indices in groups.items():
            group_data = data_np[:, indices]
            c = colors.get(g_name, "black")

            plot_spectrum(
                group_data,
                dt=dt,
                nperseg=nperseg,
                ax=ax,
                color=c,
                label=str(g_name),
                show_mean=show_mean,
                alpha=0.1 if show_traces else 0.0,  # lighter alpha for overlay
            )

        ax.set_title(title)
        ax.legend()
        return fig
