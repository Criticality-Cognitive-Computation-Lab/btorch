"""Multiscale dynamics visualization with unified dual interface.

This module provides plotting functions for multiscale dynamics analysis
including Fano factor, DFA (Detrended Fluctuation Analysis), ISI CV
(Coefficient of Variation), and criticality analysis. Supports both
dataclass and plain argument interfaces for flexible usage.

The visualization modes include:
- Individual neuron traces
- Grouped aggregations (by neuron type or neuropil)
- Distribution summaries (violin, histogram)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from ..analysis.aggregation import agg_by_neuron, agg_by_neuropil
from ..analysis.dynamic_tools.micro_scale import calculate_cv_isi
from ..analysis.spiking import fano


if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _to_numpy(data) -> np.ndarray:
    """Convert torch.Tensor to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


@dataclass
class DynamicsData:
    """Container for dynamics analysis data and configs.

    Attributes:
        spikes: Spike trains with shape (time, neurons).
        dt: Simulation timestep in milliseconds.
        neurons_df: DataFrame with neuron metadata for grouping.
        connections_df: DataFrame with connection metadata for neuropil
            aggregation.
    """

    spikes: np.ndarray | torch.Tensor
    dt: float = 1.0
    neurons_df: pd.DataFrame | None = None
    connections_df: pd.DataFrame | None = None


@dataclass
class DynamicsPlotFormat:
    """Figure formatting for dynamics plots.

    Attributes:
        mode: Visualization mode - "individual" for specific neurons,
            "grouped" for aggregated groups, "distribution" for summary stats.
        group_by: Grouping method for aggregation ("neuropil" or "neuron_type").
        neuron_type_column: Column name in neurons_df for neuron classification.
        neuron_indices: Specific neuron indices for individual mode.
        colors: Color mapping dictionary.
        figsize: Figure size as (width, height) in inches.
    """

    mode: Literal["individual", "grouped", "distribution"] = "individual"
    group_by: Literal["neuropil", "neuron_type", None] = None
    neuron_type_column: str = "cell_type"
    neuron_indices: list[int] | None = None
    colors: dict | None = None
    figsize: tuple[float, float] | None = None


@dataclass
class FanoFactorConfig:
    """Configuration for Fano factor analysis.

    Attributes:
        windows: Time windows in timesteps for multiscale analysis.
            If None, logarithmically spaced windows are auto-generated.
        overlap: Overlap between consecutive windows in timesteps.
    """

    windows: list[int] | None = None
    overlap: int = 0


@dataclass
class DFAConfig:
    """Configuration for DFA (Detrended Fluctuation Analysis).

    Attributes:
        min_window: Minimum window size for DFA in timesteps.
        max_window: Maximum window size. If None, auto-calculated.
        bin_size: Bin size for spike binning in timesteps.
    """

    min_window: int = 4
    max_window: int | None = None
    bin_size: int = 1


def plot_multiscale_fano(
    # Dataclass interface
    data: DynamicsData | None = None,
    config: FanoFactorConfig | None = None,
    format: DynamicsPlotFormat | None = None,
    # Plain args interface
    spikes: np.ndarray | torch.Tensor | None = None,
    dt: float = 1.0,
    windows: list[int] | None = None,
    overlap: int = 0,
    mode: Literal["individual", "grouped", "distribution"] = "individual",
    neurons_df: pd.DataFrame | None = None,
    connections_df: pd.DataFrame | None = None,
    group_by: Literal["neuropil", "neuron_type", None] = None,
    neuron_type_column: str = "cell_type",
    neuron_indices: list[int] | None = None,
    **kwargs,
) -> Figure:
    """Plot multiscale Fano factor analysis.

    Computes and visualizes Fano factor (spike count variance/mean) across
    multiple time windows. Supports three visualization modes:
    - "individual": Line plots for selected neurons
    - "grouped": Aggregated by neuron type or neuropil
    - "distribution": Violin plots showing population distribution

    Supports both dataclass and plain argument interfaces. Dataclass
    arguments take precedence when both are provided.

    Args:
        data: DynamicsData container with spikes and metadata.
        config: FanoFactorConfig with window settings.
        format: DynamicsPlotFormat with visualization options.
        spikes: Spike trains with shape (time, neurons). Required if
            `data` is not provided.
        dt: Timestep in milliseconds. Default 1.0.
        windows: List of window sizes in timesteps. Auto-generated if None.
        overlap: Window overlap in timesteps. Default 0.
        mode: Visualization mode - "individual", "grouped", "distribution".
        neurons_df: DataFrame with neuron metadata for grouping.
        connections_df: DataFrame with connection metadata for neuropil
            grouping.
        group_by: Grouping method - "neuropil" or "neuron_type".
        neuron_type_column: Column name for neuron types in neurons_df.
        neuron_indices: Specific neuron indices for "individual" mode.
            If None, first 10 neurons are plotted.
        **kwargs: Additional arguments passed to plotting functions.

    Returns:
        Figure with Fano factor plots.

    Raises:
        ValueError: If spikes are not provided through either `data` or
            `spikes` argument.

    Example:
        >>> # Plain args interface
        >>> fig = plot_multiscale_fano(spikes, dt=1.0, mode="distribution")
        >>>
        >>> # Dataclass interface
        >>> data = DynamicsData(spikes=spikes, dt=1.0, neurons_df=df)
        >>> config = FanoFactorConfig(windows=[10, 50, 100])
        >>> fig = plot_multiscale_fano(data=data, config=config)
    """
    # Resolve dataclass vs plain args
    if data is not None:
        spikes = data.spikes if spikes is None else spikes
        dt = data.dt if dt == 1.0 else dt
        neurons_df = data.neurons_df if neurons_df is None else neurons_df
        connections_df = (
            data.connections_df if connections_df is None else connections_df
        )

    if config is not None:
        windows = config.windows if windows is None else windows
        overlap = config.overlap if overlap == 0 else overlap

    if format is not None:
        mode = format.mode
        group_by = format.group_by if group_by is None else group_by
        neuron_type_column = format.neuron_type_column
        neuron_indices = (
            format.neuron_indices if neuron_indices is None else neuron_indices
        )

    # Validate
    if spikes is None:
        raise ValueError("spikes is required")

    spikes = _to_numpy(spikes)
    n_time, n_neurons = spikes.shape

    # Default windows: logarithmically spaced
    if windows is None:
        windows = [int(w) for w in np.logspace(1, np.log10(n_time // 4), 10)]

    # Compute Fano factor for each window
    fano_results = {}
    for w in windows:
        fano_values, info = fano(spikes, window=w, overlap=overlap)
        fano_results[w] = fano_values

    # Create figure based on mode
    if mode == "individual":
        return _plot_fano_individual(
            fano_results, windows, dt, neuron_indices, n_neurons
        )
    elif mode == "grouped":
        if group_by is None:
            raise ValueError("group_by must be specified for grouped mode")
        return _plot_fano_grouped(
            fano_results,
            windows,
            dt,
            spikes,
            neurons_df,
            connections_df,
            group_by,
            neuron_type_column,
        )
    elif mode == "distribution":
        return _plot_fano_distribution(fano_results, windows, dt)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _plot_fano_individual(fano_results, windows, dt, neuron_indices, n_neurons):
    """Plot Fano factor for individual neurons."""
    if neuron_indices is None:
        neuron_indices = list(range(min(10, n_neurons)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for neuron_idx in neuron_indices:
        fano_values = [fano_results[w][neuron_idx] for w in windows]
        ax.plot(
            np.array(windows) * dt,
            fano_values,
            marker="o",
            label=f"N{neuron_idx}",
            alpha=0.7,
        )

    ax.set_xlabel("Time Window (ms)")
    ax.set_ylabel("Fano Factor")
    ax.set_title("Multiscale Fano Factor - Individual Neurons")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig


def _plot_fano_grouped(
    fano_results,
    windows,
    dt,
    spikes,
    neurons_df,
    connections_df,
    group_by,
    neuron_type_column,
):
    """Plot Fano factor grouped by neuropil or neuron type."""
    if neurons_df is None and group_by == "neuron_type":
        raise ValueError("neurons_df required for neuron_type grouping")
    if connections_df is None and group_by == "neuropil":
        raise ValueError("connections_df required for neuropil grouping")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Organize data by group first: {group_name: [val_w1, val_w2, ...]}
    group_data = {}

    # Initialize groups from first window to ensure consistency
    first_window_data = fano_results[windows[0]]
    if group_by == "neuron_type":
        grouped_init = agg_by_neuron(
            first_window_data,
            neurons_df,
            agg="mean",
            neuron_type_column=neuron_type_column,
        )
    elif group_by == "neuropil":
        grouped_init, _ = agg_by_neuropil(
            first_window_data,
            neurons=neurons_df,
            connections=connections_df,
            mode="top_innervated",
            agg="mean",
        )
        grouped_init = grouped_init if grouped_init else {}

    if grouped_init:
        for group_name in grouped_init.keys():
            group_data[group_name] = []

        # Collect data for all windows
        for w in windows:
            fano_arr = fano_results[w]

            if group_by == "neuron_type":
                grouped = agg_by_neuron(
                    fano_arr,
                    neurons_df,
                    agg="mean",
                    neuron_type_column=neuron_type_column,
                )
            elif group_by == "neuropil":
                pre_grouped, _ = agg_by_neuropil(
                    fano_arr,
                    neurons=neurons_df,
                    connections=connections_df,
                    mode="top_innervated",
                    agg="mean",
                )
                grouped = pre_grouped

            for group_name in group_data:
                if group_name in grouped:
                    group_data[group_name].append(grouped[group_name])
                else:
                    group_data[group_name].append(np.nan)

        # Plot lines for each group
        times = np.array(windows) * dt
        for group_name, values in group_data.items():
            ax.plot(times, values, marker="o", label=group_name)
    else:
        # Fallback if no groups found
        pass

    ax.set_xlabel("Time Window (ms)")
    ax.set_ylabel("Fano Factor (mean)")
    ax.set_title(f"Multiscale Fano Factor - Grouped by {group_by}")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig


def _plot_fano_distribution(fano_results, windows, dt):
    """Plot Fano factor distribution across neurons."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Use ordinal positions for violins to avoid log-scale width artifacts
    positions = np.arange(len(windows))
    data_list = [fano_results[w] for w in windows]

    ax.violinplot(
        data_list,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showmedians=True,
    )

    # Set x-ticks to show actual time windows
    ax.set_xticks(positions)
    # Format labels: integer if whole number, else 1 decimal
    labels = [f"{w * dt:.1f}" if (w * dt) % 1 else f"{int(w * dt)}" for w in windows]
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel("Time Window (ms)")
    ax.set_ylabel("Fano Factor")
    ax.set_title("Multiscale Fano Factor - Distribution")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_dfa_analysis(
    # Dataclass interface
    data: DynamicsData | None = None,
    config: DFAConfig | None = None,
    format: DynamicsPlotFormat | None = None,
    # Plain args interface
    spikes: np.ndarray | torch.Tensor | None = None,
    dt: float = 1.0,
    min_window: int = 4,
    max_window: int | None = None,
    bin_size: int = 1,
    mode: Literal["individual", "grouped", "distribution"] = "individual",
    neurons_df: pd.DataFrame | None = None,
    **kwargs,
) -> Figure:
    """Plot DFA (Detrended Fluctuation Analysis) results.

    DFA quantifies long-range temporal correlations in spike trains.
    The scaling exponent (alpha) indicates:
    - alpha ≈ 0.5: Uncorrelated (random) activity
    - alpha > 0.5: Long-range positive correlations
    - alpha < 0.5: Long-range anti-correlations

    Supports both dataclass and plain argument interfaces.

    Args:
        data: DynamicsData container with spikes.
        config: DFAConfig with window and bin settings.
        format: DynamicsPlotFormat (mode affects plot style).
        spikes: Spike trains (time, neurons). Required if `data` not provided.
        dt: Timestep in milliseconds (for label consistency).
        min_window: Minimum window size for DFA in timesteps.
        max_window: Maximum window size. Auto-calculated if None.
        bin_size: Bin size for spike binning in timesteps.
        mode: Visualization mode (affects annotation style).
        neurons_df: Neuron metadata for potential grouping.
        **kwargs: Additional arguments.

    Returns:
        Figure with DFA summary and interpretation guide.

    Raises:
        ValueError: If spikes are not provided.

    Example:
        >>> fig = plot_dfa_analysis(spikes, bin_size=10)
    """
    # Resolve dataclass vs plain args
    if data is not None:
        spikes = data.spikes if spikes is None else spikes
        dt = data.dt if dt == 1.0 else dt
        neurons_df = data.neurons_df if neurons_df is None else neurons_df

    if config is not None:
        max_window = config.max_window if max_window is None else max_window
        bin_size = config.bin_size

    # Validate
    if spikes is None:
        raise ValueError("spikes is required")

    spikes = _to_numpy(spikes)

    # Compute DFA
    from ..analysis.dynamic_tools.criticality import calculate_dfa

    alpha = calculate_dfa(spikes, bin_size=bin_size)

    # Create simple plot showing the result
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.text(
        0.5,
        0.5,
        f"DFA Exponent (α): {alpha:.3f}",
        ha="center",
        va="center",
        fontsize=16,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.4,
        "α ≈ 0.5: uncorrelated\n"
        "α > 0.5: long-range correlations\n"
        "α < 0.5: anti-correlations",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.set_title("Detrended Fluctuation Analysis")
    ax.axis("off")
    return fig


def plot_isi_cv(
    # Dataclass interface
    data: DynamicsData | None = None,
    format: DynamicsPlotFormat | None = None,
    # Plain args interface
    spikes: np.ndarray | torch.Tensor | None = None,
    dt: float = 1.0,
    mode: Literal["individual", "grouped", "distribution"] = "individual",
    neurons_df: pd.DataFrame | None = None,
    group_by: Literal["neuropil", "neuron_type", None] = None,
    neuron_type_column: str = "cell_type",
    **kwargs,
) -> Figure:
    """Plot ISI CV (Coefficient of Variation) distribution.

    ISI CV measures spike train irregularity:
    - CV = 1: Poisson-like (irregular) firing
    - CV < 1: Regular, periodic firing
    - CV > 1: Bursty, irregular firing

    Supports histogram view for distributions and bar plots for grouped
    comparisons.

    Args:
        data: DynamicsData container with spikes and metadata.
        format: DynamicsPlotFormat with visualization settings.
        spikes: Spike trains (time, neurons). Required if `data` not provided.
        dt: Timestep in milliseconds for ISI calculation.
        mode: Visualization mode - "distribution", "individual", or "grouped".
        neurons_df: DataFrame with neuron metadata for grouping.
        group_by: Grouping method - "neuropil" or "neuron_type".
        neuron_type_column: Column name for neuron types in neurons_df.
        **kwargs: Additional arguments.

    Returns:
        Figure with ISI CV histogram or grouped bar plot.

    Raises:
        ValueError: If spikes are not provided, or if grouped mode is
            requested without required metadata.

    Example:
        >>> fig = plot_isi_cv(spikes, dt=1.0, mode="distribution")
        >>>
        >>> # Grouped by cell type
        >>> fig = plot_isi_cv(spikes, neurons_df=df,
        ...                   mode="grouped", group_by="neuron_type")
    """
    # Resolve dataclass vs plain args
    if data is not None:
        spikes = data.spikes if spikes is None else spikes
        dt = data.dt if dt == 1.0 else dt
        neurons_df = data.neurons_df if neurons_df is None else neurons_df

    if format is not None:
        mode = format.mode
        group_by = format.group_by if group_by is None else group_by
        neuron_type_column = format.neuron_type_column

    # Validate
    if spikes is None:
        raise ValueError("spikes is required")

    spikes = _to_numpy(spikes)

    # Compute ISI CV
    cv_results = calculate_cv_isi(spikes, dt=dt)
    cv_values = cv_results["cv_isi"]

    # Create figure based on mode
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if mode == "distribution" or mode == "individual":
        # Histogram
        valid_cv = cv_values[~np.isnan(cv_values)]
        ax.hist(valid_cv, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
        ax.axvline(
            cv_results["mean"],
            color="red",
            linestyle="--",
            label=f"Mean={cv_results['mean']:.2f}",
        )
        ax.set_xlabel("ISI CV")
        ax.set_ylabel("Count")
        ax.set_title("ISI Coefficient of Variation Distribution")
        ax.legend()
        ax.grid(alpha=0.3)

    elif mode == "grouped":
        if group_by is None or neurons_df is None:
            raise ValueError("group_by and neurons_df required for grouped mode")

        # Group by neuron type
        grouped = agg_by_neuron(
            cv_values, neurons_df, agg="mean", neuron_type_column=neuron_type_column
        )

        # Bar plot
        names = list(grouped.keys())
        values = list(grouped.values())
        ax.bar(names, values, color="teal", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Neuron Type")
        ax.set_ylabel("Mean ISI CV")
        ax.set_title(f"ISI CV by {neuron_type_column}")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    return fig


# --- Ported from legacy dynamics.py ---


def plot_avalanche_analysis(
    spikes: np.ndarray | torch.Tensor,
    bin_size: int = 1,
    dt: float = 1.0,  # Added for consistency
) -> tuple[Figure, dict]:
    """Plot avalanche size and duration distributions to analyze criticality.

    Creates a 3-panel figure showing:
    1. Avalanche size distribution P(S) with power-law fit
    2. Avalanche duration distribution P(T) with power-law fit
    3. Average size vs duration scaling relation <S>(T)

    Criticality is indicated by power-law distributions and specific
    scaling exponents (tau, alpha, gamma).

    Args:
        spikes: Spike matrix with shape (time, neurons).
        bin_size: Bin size for avalanche detection in timesteps.
        dt: Timestep in ms (unused, kept for interface consistency).

    Returns:
        Tuple of (figure, results) where results contains fitted exponents
        (tau, alpha, gamma), CCC (criticality consistency check), and
        power-law fit objects.

    Example:
        >>> fig, results = plot_avalanche_analysis(spikes, bin_size=5)
        >>> print(f"Tau: {results['tau']:.2f}, CCC: {results['CCC']:.2f}")
    """
    from ..analysis.dynamic_tools.criticality import compute_avalanche_statistics

    spikes = _to_numpy(spikes)
    results = compute_avalanche_statistics(spikes, bin_size=bin_size)

    fig = plt.figure(figsize=(15, 4))

    # 1. Size Distribution P(S)
    ax1 = fig.add_subplot(1, 3, 1)
    if results["fit_S"]:
        results["fit_S"].plot_pdf(color="b", linewidth=2, ax=ax1, label="Data")
        results["fit_S"].power_law.plot_pdf(
            color="b", linestyle="--", ax=ax1, label=f"Fit (tau={results['tau']:.2f})"
        )
    ax1.set_xlabel("Avalanche Size (S)")
    ax1.set_ylabel("P(S)")
    ax1.set_title("Size Distribution")
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend()

    # 2. Duration Distribution P(T)
    ax2 = fig.add_subplot(1, 3, 2)
    if results["fit_T"]:
        results["fit_T"].plot_pdf(color="r", linewidth=2, ax=ax2, label="Data")
        results["fit_T"].power_law.plot_pdf(
            color="r",
            linestyle="--",
            ax=ax2,
            label=f"Fit (alpha={results['alpha']:.2f})",
        )
    ax2.set_xlabel("Avalanche Duration (T)")
    ax2.set_ylabel("P(T)")
    ax2.set_title("Duration Distribution")
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend()

    # 3. Average Size vs Duration <S>(T)
    ax3 = fig.add_subplot(1, 3, 3)
    if (
        "avg_size_by_duration" in results
        and results["avg_size_by_duration"] is not None
    ):
        durations, mean_sizes = results["avg_size_by_duration"]
        ax3.loglog(durations, mean_sizes, "ko", markersize=4, label="Data")

        # Plot fit
        if not np.isnan(results["gamma"]):
            if "gamma_stats" in results and "popt" in results["gamma_stats"]:
                popt = results["gamma_stats"]["popt"]
                a, gamma = popt
                x_fit = np.logspace(
                    np.log10(durations.min()), np.log10(durations.max()), 100
                )
                y_fit = a * np.power(x_fit, gamma)
                ax3.loglog(
                    x_fit,
                    y_fit,
                    "g--",
                    label=f"Fit (gamma={results['gamma']:.2f})",
                )

    # Annotate CCC
    if not np.isnan(results["CCC"]):
        txt = f"CCC = {results['CCC']:.2f}\nPred gamma = {results['gamma_pred']:.2f}"
        ax3.text(
            0.05,
            0.95,
            txt,
            transform=ax3.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax3.set_xlabel("Duration (T)")
    ax3.set_ylabel("Average Size <S>")
    ax3.set_title("Scaling Relation")
    if ax3.get_legend_handles_labels()[0]:
        ax3.legend()

    plt.tight_layout()
    return fig, results


def plot_eigenvalue_spectrum(
    weight_matrix: np.ndarray | torch.Tensor, ax: Axes | None = None
) -> tuple[Figure, Axes, dict]:
    """Plot the eigenvalue spectrum of a weight matrix.

    Visualizes eigenvalues in the complex plane with the spectral radius
    indicated by a dashed circle. Outliers (eigenvalues outside the bulk)
    are highlighted in red.

    Args:
        weight_matrix: Square connectivity matrix (N, N).
        ax: Existing axes to plot on. Creates new figure if None.

    Returns:
        Tuple of (figure, axes, results) where results contains:
        - "eigenvalues": Complex array of all eigenvalues
        - "spectral_radius": Radius of spectral bulk
        - "outliers": Array of outlier eigenvalues

    Example:
        >>> fig, ax, results = plot_eigenvalue_spectrum(W)
        >>> print(f"Spectral radius: {results['spectral_radius']:.2f}")
    """
    from ..analysis.dynamic_tools.attractor_dynamics import (
        calculate_structural_eigenvalue_outliers,
    )

    W = _to_numpy(weight_matrix)
    results = calculate_structural_eigenvalue_outliers(W)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    evals = results["eigenvalues"]
    r_spec = results["spectral_radius"]

    # Draw unit circle / spectral radius
    circle = plt.Circle(
        (0, 0),
        r_spec,
        color="black",
        fill=False,
        linestyle="--",
        alpha=0.5,
        label=f"R={r_spec:.2f}",
    )
    ax.add_artist(circle)

    # Scatter eigenvalues
    ax.scatter(evals.real, evals.imag, s=10, alpha=0.6, c="gray", edgecolors="none")

    # Highlight outliers
    outliers = results["outliers"]
    if len(outliers) > 0:
        ax.scatter(
            outliers.real,
            outliers.imag,
            s=30,
            c="red",
            label=f"Outliers ({len(outliers)})",
        )

    ax.set_aspect("equal")
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title("Eigenvalue Spectrum")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    return fig, ax, results


def plot_lyapunov_spectrum(
    spectrum: list[float] | np.ndarray, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """Plot the Lyapunov exponents spectrum.

    Displays Lyapunov exponents sorted by magnitude. Positive exponents
    indicate chaos; the number of non-negative exponents relates to the
    Kaplan-Yorke dimension.

    Args:
        spectrum: List or array of Lyapunov exponents.
        ax: Existing axes to plot on. Creates new figure if None.

    Returns:
        Tuple of (figure, axes) with the spectrum plot.

    Example:
        >>> fig, ax = plot_lyapunov_spectrum(lyap_spectrum)
        >>> # Positive exponents indicate chaotic dynamics
    """
    from ..analysis.dynamic_tools.attractor_dynamics import (
        calculate_kaplan_yorke_dimension,
    )

    spec = _to_numpy(spectrum)
    # Sort descending just in case, though standard is descending
    spec = np.sort(spec)[::-1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    x = np.arange(1, len(spec) + 1)
    ax.plot(x, spec, "o-", markersize=4, linewidth=1, color="black")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)

    # Calculate Kaplan-Yorke Dim
    ky_dim = calculate_kaplan_yorke_dimension(spec)

    title = f"Lyapunov Spectrum (D_KY = {ky_dim:.2f})"
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Lyapunov Exponent")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax


def plot_firing_rate_distribution(
    spikes: np.ndarray | torch.Tensor,
    dt: float = 1.0,
    ax: Axes | None = None,
) -> tuple[Figure, dict]:
    """Plot the distribution of firing rates across neurons.

    Computes per-neuron firing rates and displays as a histogram with
    mean indicator.

    Args:
        spikes: Spike matrix with shape (time, neurons).
        dt: Timestep in milliseconds for rate calculation.
        ax: Existing axes to plot on. Creates new figure if None.

    Returns:
        Tuple of (figure, stats) where stats contains:
        - "rates": Array of firing rates per neuron (Hz)
        - "mean", "std", "median": Summary statistics

    Example:
        >>> fig, stats = plot_firing_rate_distribution(spikes, dt=1.0)
        >>> print(f"Mean rate: {stats['mean']:.1f} Hz")
    """
    from ..analysis.dynamic_tools.micro_scale import calculate_fr_distribution

    spikes = _to_numpy(spikes)
    stats = calculate_fr_distribution(spikes, dt=dt)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    rates = stats["rates"]
    ax.hist(rates, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    ax.axvline(
        stats["mean"],
        color="red",
        linestyle="--",
        label=f"Mean={stats['mean']:.1f} Hz",
    )
    ax.set_xlabel("Firing Rate (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Rate Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    if ax is None:
        plt.tight_layout()

    return fig, stats


def plot_micro_dynamics(
    spikes: np.ndarray | torch.Tensor,
    dt: float = 1.0,
    ax: Axes | None = None,
) -> tuple[Figure, dict]:
    """Plot firing rate and ISI CV distributions side-by-side.

    Creates a 2-panel figure summarizing micro-scale dynamics:
    - Left: Firing rate distribution histogram
    - Right: ISI CV distribution histogram

    Args:
        spikes: Spike matrix with shape (time, neurons).
        dt: Timestep in milliseconds.
        ax: Unused parameter (kept for API compatibility).

    Returns:
        Tuple of (figure, stats) where stats is a dict with keys:
        - "fr": Firing rate statistics
        - "cv": ISI CV statistics

    Example:
        >>> fig, stats = plot_micro_dynamics(spikes, dt=1.0)
        >>> print(f"Rate: {stats['fr']['mean']:.1f} Hz, CV: {stats['cv']['mean']:.2f}")
    """
    from ..analysis.dynamic_tools.micro_scale import calculate_cv_isi

    # Plot FR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    _, fr_stats = plot_firing_rate_distribution(spikes, dt=dt, ax=ax1)

    # Plot CV (Re-implementing simplified version or using plot_isi_cv logic)
    # reusing logic from plot_isi_cv for consistency but without full overhead
    spikes_np = _to_numpy(spikes)
    cv_results = calculate_cv_isi(spikes_np, dt=dt)
    cv_values = cv_results["cv_isi"]
    valid_cv = cv_values[~np.isnan(cv_values)]

    if len(valid_cv) > 0:
        ax2.hist(valid_cv, bins=30, color="orange", edgecolor="black", alpha=0.7)
        ax2.axvline(
            cv_results["mean"],
            color="red",
            linestyle="--",
            label=f"Mean={cv_results['mean']:.2f}",
        )
        ax2.set_xlabel("CV (ISI)")
        ax2.set_ylabel("Count")
        ax2.set_title("CV Distribution")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No valid CVs", ha="center")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig, {"fr": fr_stats, "cv": cv_results}


def plot_gain_stability(data: tuple) -> tuple[Figure, Axes]:
    """Plot gain stability analysis results.

    Visualizes the relationship between network gain (g) and stability
    metrics (e.g., maximum Lyapunov exponent or spectral abscissa).
    A linear fit indicates consistent scaling behavior.

    Args:
        data: Tuple of (slope, intercept, g_values, lambda_values) where:
            - slope, intercept: Linear fit parameters
            - g_values: Array of gain values tested
            - lambda_values: Corresponding stability metrics

    Returns:
        Tuple of (figure, axes) with scatter plot and fit line.

    Example:
        >>> data = (slope, intercept, g_vals, lyap_vals)
        >>> fig, ax = plot_gain_stability(data)
    """
    slope, intercept, g_values, lambda_values = data

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot scatter of metrics
    ax.scatter(g_values, lambda_values, label="Data", color="blue", alpha=0.6)

    # Plot fit line
    x_range = np.linspace(min(g_values), max(g_values), 100)
    y_fit = slope * x_range + intercept
    ax.plot(x_range, y_fit, "r--", label=f"Fit: y={slope:.2f}x+{intercept:.2f}")

    ax.set_xlabel("Gain (g)")
    ax.set_ylabel("Lyapunov / Eigenvalue metric")
    ax.set_title("Gain Stability Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax
