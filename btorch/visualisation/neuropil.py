from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..analysis.aggregation import agg_by_neuropil
from ..models.types import TensorLike


def plot_agg_by_neuropil(
    y: TensorLike | dict[str, TensorLike],
    dt: float,
    mode: Literal["top_innervated", "all_innervated"] = "all_innervated",
    agg: Literal["mean", "sum", "std"] = "mean",
    connections: pd.DataFrame | None = None,
    neurons: pd.DataFrame | None = None,
    style: Literal["wave", "heatmap"] = "wave",
    figsize: tuple = (12, 8),
    cmap: str = "viridis",
    show_top_n: int = 50,
    use_polars: bool = False,
) -> None:
    """Plot averaged traces by neuropil."""
    if isinstance(y, TensorLike):
        pre_ret, _ = agg_by_neuropil(
            y,
            mode=mode,
            connections=connections,
            neurons=neurons,
            agg=agg,
            use_polars=use_polars,
        )
    else:
        pre_ret = y

    v = next(iter(pre_ret.values()))
    if isinstance(v, np.ndarray):
        time_points = np.arange(len(v)) * dt
    else:
        time_points = torch.arange(len(v), device=v.device, dtype=v.dtype) * dt

    fig, ax = plt.subplots(figsize=figsize)

    if style == "wave":
        _plot_wave_style(ax, pre_ret, time_points, show_top_n)
    else:
        _plot_heatmap_style(ax, pre_ret, time_points, cmap)

    ax.set_ylabel("Neuropil Activity (z-scored)", fontsize=12)
    ax.set_title("Neuropil Activity Traces", fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=12)

    plt.tight_layout()
    plt.show()


def _plot_wave_style(ax, data_dict, time_points, show_top_n=50):
    """Plot wave-like traces for neural activity."""
    sorted_regions = sorted(
        data_dict.items(),
        key=lambda x: np.max(np.abs(x[1])),
        reverse=True,
    )[:show_top_n]

    offset_step = 3.0
    mean = np.concatenate([activity for _, activity in sorted_regions]).mean()
    std = np.concatenate([activity for _, activity in sorted_regions]).std()

    for i, (region, activity) in enumerate(sorted_regions):
        offset = i * offset_step
        normalized_activity = (activity - mean) / (std + 1e-8)
        ax.plot(
            time_points,
            normalized_activity + offset,
            linewidth=0.8,
            alpha=0.7,
            label=region,
        )

    yticks = np.arange(len(sorted_regions)) * offset_step
    ax.set_yticks(yticks)
    ax.set_yticklabels([r for r, _ in sorted_regions], fontsize=8)

    ax.grid(True, alpha=0.3)
    # ax.legend(loc="upper right", fontsize=8)


def _plot_heatmap_style(ax, data_dict, time_points, cmap="viridis"):
    """Plot heatmap style visualization for neural activity."""
    regions = list(data_dict.keys())
    activity_matrix = np.array([data_dict[region] for region in regions])

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


def plot_neuropil_comparison(
    data_dict: dict[str, np.ndarray],
    dt: float,
    selected_regions: list[str] = None,
    figsize: tuple = (15, 10),
    n_cols: int = 3,
) -> None:
    """Create comparison plots for selected neuropils."""
    if selected_regions is None:
        sorted_regions = sorted(
            data_dict.items(), key=lambda x: np.max(np.abs(x[1])), reverse=True
        )[:9]
        selected_regions = [region for region, _ in sorted_regions]

    n_regions = len(selected_regions)
    n_rows = int(np.ceil(n_regions / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    time_points = np.arange(len(next(iter(data_dict.values())))) * dt

    for i, region in enumerate(selected_regions):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        if region in data_dict:
            activity = data_dict[region]
            ax.plot(time_points, activity, linewidth=1.5, color="darkblue")
            ax.set_title(f"{region}", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Activity", fontsize=8)
            ax.grid(True, alpha=0.3)

            mean_act = np.mean(activity)
            std_act = np.std(activity)
            ax.text(
                0.02,
                0.98,
                f"μ={mean_act:.2f}\nσ={std_act:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=8,
            )

    for i in range(n_regions, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])

    plt.suptitle("Neuropil Activity Comparison", fontsize=16)
    plt.tight_layout()
