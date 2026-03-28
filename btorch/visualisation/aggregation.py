from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..analysis.aggregation import agg_by_neuropil, group_ecdf, group_values
from ..types import TensorLike


GroupPlotKind = Literal["violin", "box", "ecdf"]


def plot_group_distribution(
    values: TensorLike,
    neurons_df: pd.DataFrame,
    group_by: str,
    *,
    kind: GroupPlotKind = "violin",
    simple_id_col: str = "simple_id",
    value_name: str = "value",
    group_order: Sequence | None = None,
    dropna: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (9.0, 5.0),
    title: str | None = None,
    showfliers: bool = False,
    linewidth: float = 1.5,
    alpha: float = 0.8,
) -> tuple[Figure, Axes]:
    """Plot grouped value distributions as violin, box, or ECDF.

    Groups per-neuron values by a categorical column in `neurons_df` and
    visualizes the distribution using the specified plot type.

    Args:
        values: Per-neuron values with shape (neurons,) or (time, neurons).
            If 2D, values are flattened across time for each neuron.
        neurons_df: DataFrame with neuron metadata. Must contain `simple_id_col`
            and `group_by` columns.
        group_by: Column name in `neurons_df` to group by (e.g., "cell_type").
        kind: Plot type - "violin", "box", or "ecdf".
        simple_id_col: Column name for neuron identifiers in `neurons_df`.
        value_name: Label for the y-axis / value dimension.
        group_order: Explicit order for groups. If None, uses natural sort.
        dropna: Whether to drop NaN values before plotting.
        ax: Existing axes to plot on. If None, creates new figure.
        figsize: Figure size (width, height) in inches.
        title: Plot title. If None, auto-generated from `kind` and `group_by`.
        showfliers: For box plots, whether to show outlier points.
        linewidth: Line width for ECDF curves.
        alpha: Opacity for violin/box fill.

    Returns:
        Tuple of (figure, axes) containing the plot.

    Raises:
        ValueError: If `kind` is not one of "violin", "box", "ecdf".

    Example:
        >>> fig, ax = plot_group_distribution(
        ...     firing_rates, neurons_df, group_by="cell_type", kind="violin"
        ... )
    """
    fig, ax = _resolve_figure_ax(ax=ax, figsize=figsize)

    if kind in {"violin", "box"}:
        grouped = group_values(
            values,
            neurons_df,
            group_by,
            simple_id_col=simple_id_col,
            value_name=value_name,
            group_order=group_order,
            dropna=dropna,
        )
        order = list(grouped.keys())
        grouped_arrays = [grouped[group] for group in order]

        if kind == "violin":
            _plot_violin(ax, grouped_arrays, order, alpha=alpha)
        else:
            _plot_box(
                ax,
                grouped_arrays,
                order,
                showfliers=showfliers,
                alpha=alpha,
            )

        ax.set_xlabel(group_by)
        ax.set_ylabel(value_name)
    elif kind == "ecdf":
        ecdf_by_group = group_ecdf(
            values,
            neurons_df,
            group_by,
            simple_id_col=simple_id_col,
            value_name=value_name,
            group_order=group_order,
            dropna=dropna,
        )
        _plot_ecdf(
            ax,
            ecdf_by_group,
            group_by=group_by,
            value_name=value_name,
            linewidth=linewidth,
        )
    else:
        raise ValueError(f"Unsupported kind `{kind}`.")

    if title is None:
        title = f"{kind.upper()} grouped by {group_by}"
    ax.set_title(title)
    ax.grid(alpha=0.2)

    return fig, ax


def plot_group_violin(
    values: TensorLike,
    neurons_df: pd.DataFrame,
    group_by: str,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot grouped value distributions as violin plots.

    Convenience wrapper around `plot_group_distribution` with `kind="violin"`.

    Args:
        values: Per-neuron values with shape (neurons,) or (time, neurons).
        neurons_df: DataFrame with neuron metadata.
        group_by: Column name to group by.
        **kwargs: Additional arguments passed to `plot_group_distribution`.

    Returns:
        Tuple of (figure, axes) containing the violin plot.
    """
    return plot_group_distribution(
        values,
        neurons_df,
        group_by,
        kind="violin",
        **kwargs,
    )


def plot_group_box(
    values: TensorLike,
    neurons_df: pd.DataFrame,
    group_by: str,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot grouped value distributions as box plots.

    Convenience wrapper around `plot_group_distribution` with `kind="box"`.

    Args:
        values: Per-neuron values with shape (neurons,) or (time, neurons).
        neurons_df: DataFrame with neuron metadata.
        group_by: Column name to group by.
        **kwargs: Additional arguments passed to `plot_group_distribution`.

    Returns:
        Tuple of (figure, axes) containing the box plot.
    """
    return plot_group_distribution(
        values,
        neurons_df,
        group_by,
        kind="box",
        **kwargs,
    )


def plot_group_ecdf(
    values: TensorLike,
    neurons_df: pd.DataFrame,
    group_by: str,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot grouped value distributions as ECDF curves.

    Convenience wrapper around `plot_group_distribution` with `kind="ecdf"`.

    Args:
        values: Per-neuron values with shape (neurons,) or (time, neurons).
        neurons_df: DataFrame with neuron metadata.
        group_by: Column name to group by.
        **kwargs: Additional arguments passed to `plot_group_distribution`.

    Returns:
        Tuple of (figure, axes) containing the ECDF plot.
    """
    return plot_group_distribution(
        values,
        neurons_df,
        group_by,
        kind="ecdf",
        **kwargs,
    )


def plot_neuropil_timeseries_overview(
    data: TensorLike | Mapping[str, TensorLike],
    *,
    dt: float,
    mode: Literal["top_innervated", "all_innervated"] = "all_innervated",
    agg: Literal["mean", "sum", "std"] = "mean",
    connections: pd.DataFrame | None = None,
    neurons: pd.DataFrame | None = None,
    kind: Literal["wave", "heatmap"] = "wave",
    figsize: tuple[float, float] = (12, 8),
    cmap: str = "viridis",
    top_n: int = 50,
    use_polars: bool = False,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot aggregated neuropil traces as a single overview figure.

    Visualizes neural activity aggregated by brain regions (neuropils).
    Can display as stacked waveforms or a heatmap.

    Args:
        data: Spike matrix (time, neurons) or pre-computed dict of
            {region_name: activity_array}.
        dt: Time step in seconds for x-axis scaling.
        mode: Aggregation mode - "top_innervated" uses primary neuropil per
            neuron, "all_innervated" includes all neuropil connections.
        agg: Aggregation function applied per neuropil ("mean", "sum", "std").
        connections: DataFrame with connection metadata (required if `data`
            is a matrix, not a dict).
        neurons: DataFrame with neuron metadata (required if `data` is a
            matrix, not a dict).
        kind: Visualization style - "wave" for stacked traces, "heatmap"
            for 2D intensity map.
        figsize: Figure size (width, height) in inches.
        cmap: Colormap for heatmap style.
        top_n: Number of top regions to show in wave style (ranked by
            maximum absolute activity).
        use_polars: Whether to use Polars for aggregation (faster for large
            datasets).
        show: Whether to call `plt.show()` before returning.

    Returns:
        Tuple of (figure, axes) containing the plot.

    Raises:
        ValueError: If no neuropil traces can be computed from inputs.

    Example:
        >>> fig, ax = plot_neuropil_timeseries_overview(
        ...     spikes, dt=0.001, connections=conn_df, neurons=neurons_df
        ... )
    """
    traces = _resolve_neuropil_traces(
        data,
        mode=mode,
        agg=agg,
        connections=connections,
        neurons=neurons,
        use_polars=use_polars,
    )
    if not traces:
        raise ValueError("No neuropil traces available for plotting.")

    n_time = len(next(iter(traces.values())))
    time_points = np.arange(n_time, dtype=float) * dt

    fig, ax = plt.subplots(figsize=figsize)
    if kind == "wave":
        _plot_wave_style(ax, traces, time_points, top_n=top_n)
    else:
        _plot_heatmap_style(ax, traces, time_points, cmap=cmap)

    ax.set_ylabel("Neuropil Activity (z-scored)", fontsize=12)
    ax.set_title("Neuropil Activity Traces", fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=12)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_neuropil_timeseries_panels(
    data: TensorLike | Mapping[str, TensorLike],
    *,
    dt: float,
    mode: Literal["top_innervated", "all_innervated"] = "all_innervated",
    agg: Literal["mean", "sum", "std"] = "mean",
    connections: pd.DataFrame | None = None,
    neurons: pd.DataFrame | None = None,
    regions: Sequence[str] | None = None,
    figsize: tuple[float, float] = (15, 10),
    cols: int = 3,
    use_polars: bool = False,
    show: bool = False,
) -> tuple[Figure, np.ndarray]:
    """Plot selected neuropil traces as a subplot grid.

    Creates a grid of subplots showing individual neuropil activity traces,
    with optional statistics annotations.

    Args:
        data: Spike matrix (time, neurons) or pre-computed dict of
            {region_name: activity_array}.
        dt: Time step in seconds for x-axis scaling.
        mode: Aggregation mode - "top_innervated" or "all_innervated".
        agg: Aggregation function applied per neuropil ("mean", "sum", "std").
        connections: DataFrame with connection metadata.
        neurons: DataFrame with neuron metadata.
        regions: List of region names to plot. If None, top 9 regions by
            maximum activity are selected automatically.
        figsize: Figure size (width, height) in inches.
        cols: Number of columns in the subplot grid.
        use_polars: Whether to use Polars for aggregation.
        show: Whether to call `plt.show()` before returning.

    Returns:
        Tuple of (figure, axes_array) containing the subplot grid.

    Raises:
        ValueError: If no neuropil traces available or regions is empty.
    """
    traces = _resolve_neuropil_traces(
        data,
        mode=mode,
        agg=agg,
        connections=connections,
        neurons=neurons,
        use_polars=use_polars,
    )
    if not traces:
        raise ValueError("No neuropil traces available for plotting.")

    if regions is None:
        ranked = sorted(
            traces.items(),
            key=lambda x: float(np.max(np.abs(np.asarray(x[1])))),
            reverse=True,
        )[:9]
        regions = [region for region, _ in ranked]

    n_regions = len(regions)
    if n_regions == 0:
        raise ValueError("`regions` must contain at least one region.")

    rows = int(np.ceil(n_regions / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = np.asarray(axes).reshape(1, -1)
    if cols == 1:
        axes = np.asarray(axes).reshape(-1, 1)

    n_time = len(next(iter(traces.values())))
    time_points = np.arange(n_time, dtype=float) * dt

    for i, region in enumerate(regions):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        if region in traces:
            activity = np.asarray(traces[region])
            ax.plot(time_points, activity, linewidth=1.5, color="darkblue")
            ax.set_title(str(region), fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Activity", fontsize=8)
            ax.grid(True, alpha=0.3)

            mean_act = float(np.mean(activity))
            std_act = float(np.std(activity))
            ax.text(
                0.02,
                0.98,
                f"μ={mean_act:.2f}\nσ={std_act:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=8,
            )

    for i in range(n_regions, rows * cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col])

    fig.suptitle("Neuropil Activity Comparison", fontsize=16)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def _resolve_neuropil_traces(
    data: TensorLike | Mapping[str, TensorLike],
    *,
    mode: Literal["top_innervated", "all_innervated"],
    agg: Literal["mean", "sum", "std"],
    connections: pd.DataFrame | None,
    neurons: pd.DataFrame | None,
    use_polars: bool,
) -> dict[str, np.ndarray]:
    if isinstance(data, Mapping):
        traces = {key: np.asarray(value) for key, value in data.items()}
    else:
        pre_traces, _ = agg_by_neuropil(
            data,
            mode=mode,
            connections=connections,
            neurons=neurons,
            agg=agg,
            use_polars=use_polars,
        )
        traces = {key: np.asarray(value) for key, value in pre_traces.items()}

    if not traces:
        return {}

    lengths = {trace.shape[0] for trace in traces.values()}
    if len(lengths) != 1:
        raise ValueError("All neuropil traces must have the same time length.")

    return traces


def _plot_violin(
    ax: Axes,
    grouped_values: list[np.ndarray],
    order: list[object],
    *,
    alpha: float,
) -> None:
    parts = ax.violinplot(grouped_values, showmeans=False, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(alpha)

    positions = np.arange(1, len(order) + 1)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(group) for group in order], rotation=45, ha="right")


def _plot_box(
    ax: Axes,
    grouped_values: list[np.ndarray],
    order: list[object],
    *,
    showfliers: bool,
    alpha: float,
) -> None:
    box = ax.boxplot(
        grouped_values,
        tick_labels=[str(group) for group in order],
        patch_artist=True,
        showfliers=showfliers,
    )
    for patch in box["boxes"]:
        patch.set_alpha(alpha)

    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def _plot_ecdf(
    ax: Axes,
    ecdf_by_group: dict[object, pd.DataFrame],
    *,
    group_by: str,
    value_name: str,
    linewidth: float,
) -> None:
    for group, ecdf in ecdf_by_group.items():
        ax.step(
            ecdf[value_name].to_numpy(),
            ecdf["ecdf"].to_numpy(),
            where="post",
            label=str(group),
            linewidth=linewidth,
        )

    ax.set_xlabel(value_name)
    ax.set_ylabel("ECDF")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title=group_by)


def _plot_wave_style(
    ax: Axes,
    traces: Mapping[str, np.ndarray],
    time_points: np.ndarray,
    *,
    top_n: int,
) -> None:
    ranked = sorted(
        traces.items(),
        key=lambda x: float(np.max(np.abs(x[1]))),
        reverse=True,
    )[:top_n]
    if not ranked:
        raise ValueError("No regions available for wave plot.")

    offset_step = 3.0
    stacked = np.concatenate([trace for _, trace in ranked])
    mean = float(stacked.mean())
    std = float(stacked.std())

    for i, (region, trace) in enumerate(ranked):
        offset = i * offset_step
        normalized = (trace - mean) / (std + 1e-8)
        ax.plot(
            time_points,
            normalized + offset,
            linewidth=0.8,
            alpha=0.7,
            label=str(region),
        )

    yticks = np.arange(len(ranked), dtype=float) * offset_step
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(region) for region, _ in ranked], fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_heatmap_style(
    ax: Axes,
    traces: Mapping[str, np.ndarray],
    time_points: np.ndarray,
    *,
    cmap: str,
) -> None:
    regions = list(traces.keys())
    if not regions:
        raise ValueError("No regions available for heatmap plot.")

    activity_matrix = np.array([traces[region] for region in regions])
    im = ax.imshow(
        activity_matrix,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=[time_points[0], time_points[-1], 0, len(regions)],
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Activity Magnitude", fontsize=10)

    n_time_ticks = min(10, len(time_points))
    time_ticks = np.linspace(time_points[0], time_points[-1], n_time_ticks)
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in time_ticks])
    ax.set_yticks(np.arange(len(regions)) + 0.5)
    ax.set_yticklabels(regions, fontsize=8)


def _resolve_figure_ax(
    *,
    ax: Axes | None,
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax

    fig, created_ax = plt.subplots(figsize=figsize)
    return fig, created_ax


__all__ = [
    "GroupPlotKind",
    "plot_group_distribution",
    "plot_group_violin",
    "plot_group_box",
    "plot_group_ecdf",
    "plot_neuropil_timeseries_overview",
    "plot_neuropil_timeseries_panels",
]
