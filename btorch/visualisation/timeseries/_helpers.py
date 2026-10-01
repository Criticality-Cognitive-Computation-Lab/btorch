"""Shared private helpers for the timeseries plotting modules."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb

from ...utils.array import to_numpy


def _resolve_per_neuron_values(
    value: float | Sequence[float] | np.ndarray | torch.Tensor | None,
    neuron_indices: list[int],
    n_neurons: int,
    name: str,
) -> list[float | None]:
    """Resolve scalar or vector input to per-plotted-neuron values."""
    n_plot = len(neuron_indices)
    if value is None:
        return [None] * n_plot

    if np.isscalar(value):
        return [float(value)] * n_plot

    arr = to_numpy(value)
    if arr.ndim == 0:
        return [float(arr)] * n_plot
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a scalar or 1D array-like, got shape {arr.shape}."
        )

    if arr.shape[0] == n_neurons:
        selected = arr[np.asarray(neuron_indices, dtype=int)]
    elif arr.shape[0] == n_plot:
        selected = arr
    else:
        raise ValueError(
            f"{name} must be a scalar, length {n_neurons} (all neurons), or "
            f"length {n_plot} (plotted neurons), got length {arr.shape[0]}."
        )

    try:
        return [float(v) for v in selected]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc


def _get_time_axis(
    length: int, dt: float | None = None, times: Sequence[float] | None = None
) -> np.ndarray:
    if times is not None:
        if len(times) != length:
            raise ValueError(
                f"Length of times ({len(times)}) must match length of data ({length})."
            )
        return to_numpy(times)

    if dt is None:
        dt = 1.0
    return np.arange(length) * dt


def _sample_cmap_colors(cmap_name: str, n: int) -> list[str]:
    if n <= 1:
        cmap = plt.get_cmap(cmap_name, 1)
        return [to_hex(cmap(0.5))]
    if n <= 10 and cmap_name.startswith("tab"):
        tab10 = plt.get_cmap("tab10")
        return [to_hex(tab10.colors[i]) for i in range(n)]
    # Sample bin centers instead of the endpoints so small group counts do not
    # collapse onto the first/last colors of listed palettes like tab20.
    cmap = plt.get_cmap(cmap_name, n)
    vals = (np.arange(n, dtype=float) + 0.5) / n
    return [to_hex(cmap(v)) for v in vals]


def _auto_raster_height(
    n_neurons: int,
    min_height: float = 3.5,
    max_height: float = 10.0,
    base_height: float = 4.0,
    log_scale: float = 0.8,
) -> float:
    """Compute a raster height that grows gently with neuron count."""
    n = max(int(n_neurons), 1)
    est_height = base_height + log_scale * np.log10(n)
    return float(np.clip(est_height, min_height, max_height))


def _build_group_color_maps(
    top_group_labels: list[str],
    sub_labels: list[str],
    use_subgroups: bool,
    palette_name: str,
    cmap_name: str,
    sub_hue_span: float,
    sub_val_span: float,
) -> tuple[dict[str, str], dict[tuple[str, str], str], dict[str, list[str]], list[str]]:
    top_groups_order = list(dict.fromkeys(top_group_labels))

    if use_subgroups:
        base_list = _sample_cmap_colors(palette_name, len(top_groups_order))
        base_colors = dict(zip(top_groups_order, base_list))
        subgroups_by_top: dict[str, list[str]] = {tg: [] for tg in top_groups_order}
        for top, sub in zip(top_group_labels, sub_labels):
            lst = subgroups_by_top[top]
            if sub not in lst:
                lst.append(sub)

        subgroup_colors: dict[tuple[str, str], str] = {}
        for tg in top_groups_order:
            subs = subgroups_by_top.get(tg, [])
            m = max(1, len(subs))
            base_rgb = np.array(to_rgb(base_colors[tg]))
            base_hsv = rgb_to_hsv(base_rgb)
            if m == 1:
                hue_offsets = [0.0]
                val_offsets = [0.0]
            else:
                hue_offsets = np.linspace(-sub_hue_span, sub_hue_span, m)
                val_offsets = np.linspace(-sub_val_span, sub_val_span, m)

            for sub, h_off, v_off in zip(subs, hue_offsets, val_offsets):
                hsv = base_hsv.copy()
                hsv[0] = (hsv[0] + h_off) % 1.0
                hsv[2] = float(np.clip(hsv[2] + v_off, 0.25, 1.0))
                subgroup_colors[(tg, sub)] = to_hex(hsv_to_rgb(hsv))

        return base_colors, subgroup_colors, subgroups_by_top, top_groups_order

    group_list = list(dict.fromkeys(sub_labels))
    base_list = _sample_cmap_colors(cmap_name, len(group_list))
    base_colors = dict(zip(group_list, base_list))
    subgroups_by_top = {g: [g] for g in group_list}
    subgroup_colors = {(g, g): base_colors[g] for g in group_list}
    return base_colors, subgroup_colors, subgroups_by_top, group_list


_LINE_MARKERS = ("x", "+", "|", "_", "1", "2", "3", "4")

_STRIP_DEFAULTS: dict[str, Any] = {
    "width": 0.06,
    "pad": 0.005,
    "alpha": 0.9,
    "label_fontsize": 7,
    "label_weight": "bold",
    "legend_fontsize": 6,
    "legend_ncol_threshold": 15,
    "min_label_distance": 0.02,
    "min_span_frac": 0.01,
    "strip_x0": 0.3,
    "strip_width": 0.4,
    "label_x": None,
    "label_gap": 0.05,
    "label_sep": " / ",
    "sub_hue_span": 0.12,
    "sub_val_span": 0.28,
    "left_extra_pad": 0.04,
}


def _marker_linewidth(marker: str) -> float:
    """Return the edge width needed for a marker (line markers need > 0)."""
    return 0.5 if marker in _LINE_MARKERS else 0


def _effective_dt(dt: float | None, t: np.ndarray) -> float:
    if dt is not None:
        return dt
    return t[1] - t[0] if len(t) > 1 else 1.0
