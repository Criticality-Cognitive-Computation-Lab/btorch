from collections.abc import Sequence

import numpy as np
import torch
from scipy.ndimage import convolve1d


def cv_from_spikes(spike_data: np.ndarray, dt_ms: float = 1.0):
    """Calculate coefficient of variation of ISIs per neuron."""
    orig_shape = spike_data.shape
    T = orig_shape[0]
    flat_data = spike_data.reshape(T, -1)
    n_flat = flat_data.shape[1]

    # 1. Vectorized Spike Extraction
    t_idx, n_idx = np.where(flat_data > 0)
    n_spikes_all = np.bincount(n_idx, minlength=n_flat)

    # 2. Sort by neuron index primarily, then time secondarily
    # np.lexsort sorts by the LAST key provided in the tuple first
    sort_order = np.lexsort((t_idx, n_idx))
    t_sorted = t_idx[sort_order]
    n_sorted = n_idx[sort_order]

    # 3. Calculate all global ISIs
    diffs = np.diff(t_sorted) * dt_ms

    # 4. Valid ISIs are those where the neuron index didn't change
    valid_mask = n_sorted[:-1] == n_sorted[1:]
    valid_isis = diffs[valid_mask]
    valid_n = n_sorted[:-1][valid_mask]

    # 5. Fast aggregation for CV calculation
    count_isi = np.bincount(valid_n, minlength=n_flat)
    sum_isi = np.bincount(valid_n, weights=valid_isis, minlength=n_flat)
    sum_isi_sq = np.bincount(valid_n, weights=valid_isis**2, minlength=n_flat)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_isi_arr = sum_isi / count_isi
        var_isi_arr = (sum_isi_sq / count_isi) - mean_isi_arr**2
        std_isi_arr = np.sqrt(np.maximum(var_isi_arr, 0.0))
        cv_values_flat = std_isi_arr / mean_isi_arr

    # NaN out neurons with < 2 spikes
    cv_values_flat[count_isi == 0] = np.nan

    # 6. Reconstruct the original dictionaries exactly as expected
    isi_stats = {}

    # Fast grouping of ISIs by neuron for the dictionary
    if len(valid_n) > 0:
        split_idx = np.flatnonzero(np.diff(valid_n)) + 1
        grouped_isis = np.split(valid_isis, split_idx)
        unique_n = valid_n[np.r_[0, split_idx]]
        isi_dict_map = {int(k): v for k, v in zip(unique_n, grouped_isis)}
    else:
        isi_dict_map = {}

    for i in range(n_flat):
        n_spk = int(n_spikes_all[i])
        if n_spk < 2:
            isi_stats[i] = {
                "n_spikes": n_spk,
                "mean_isi": np.nan,
                "std_isi": np.nan,
                "cv": np.nan,
                "isi_values": [],
            }
        else:
            iv = isi_dict_map.get(i, np.array([]))
            isi_stats[i] = {
                "n_spikes": n_spk,
                "mean_isi": float(mean_isi_arr[i]),
                "std_isi": float(std_isi_arr[i]),
                "cv": float(cv_values_flat[i]),
                "isi_values": iv,
            }

    # Reconstruct isi_total
    if len(valid_isis) == 0:
        isi_total = {"mean_isi": np.nan, "std_isi": np.nan, "cv": np.nan}
    else:
        m_tot = np.mean(valid_isis)
        s_tot = np.std(valid_isis)
        isi_total = {
            "mean_isi": m_tot,
            "std_isi": s_tot,
            "cv": s_tot / m_tot if m_tot > 0 else np.nan,
        }

    cv_values = cv_values_flat.reshape(orig_shape[1:])
    return cv_values, isi_total, isi_stats


def fano_factor_from_spikes(
    spike: np.ndarray,
    window: int | None = None,
    overlap: int = 0,
    sweep_window: bool = False,
):
    """Compute Fano factor for spike trains using optimized cumulative sums."""
    orig_shape = spike.shape
    T = orig_shape[0]

    if sweep_window:
        out = np.zeros(orig_shape)
        for w in range(1, T + 1):
            out[w - 1] = fano_factor_from_spikes(
                spike, window=w, overlap=overlap, sweep_window=False
            )
        return out

    if window is None:
        window = T

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap

    flat_spike = spike.reshape(T, -1)
    n_flat = flat_spike.shape[1]

    # VECTORIZED WINDOWING via Cumulative Sum
    # This replaces the slow for-loop entirely, even with overlaps
    cumsum_spike = np.zeros((T + 1, n_flat), dtype=np.float64)
    np.cumsum(flat_spike, axis=0, dtype=np.float64, out=cumsum_spike[1:])

    t_starts = np.arange(0, T - window + 1, step)
    t_ends = t_starts + window
    counts = cumsum_spike[t_ends] - cumsum_spike[t_starts]

    mean_counts = counts.mean(axis=0)
    var_counts = counts.var(axis=0, ddof=1)

    fano = np.zeros(n_flat, dtype=float)
    valid = np.isfinite(mean_counts) & (mean_counts > 0) & np.isfinite(var_counts)
    if np.any(valid):
        fano[valid] = var_counts[valid] / mean_counts[valid]

    return fano.reshape(orig_shape[1:])


def kurtosis_from_spikes(
    spike: np.ndarray,
    window: int | None = None,
    overlap: int = 0,
    sweep_window: bool = False,
    dt_ms: float = 1.0,
    fisher: bool = True,
):
    """Compute kurtosis of spike counts using optimized cumulative sums."""
    orig_shape = spike.shape
    T = orig_shape[0]

    if sweep_window:
        out = np.zeros(orig_shape)
        for w in range(1, T + 1):
            out[w - 1] = kurtosis_from_spikes(
                spike,
                window=w,
                overlap=overlap,
                sweep_window=False,
                dt_ms=dt_ms,
                fisher=fisher,
            )
        return out

    if window is None:
        window = T

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap

    flat_spike = spike.reshape(T, -1)
    n_flat = flat_spike.shape[1]

    # VECTORIZED WINDOWING via Cumulative Sum
    cumsum_spike = np.zeros((T + 1, n_flat), dtype=np.float64)
    np.cumsum(flat_spike, axis=0, dtype=np.float64, out=cumsum_spike[1:])

    t_starts = np.arange(0, T - window + 1, step)
    t_ends = t_starts + window
    counts = cumsum_spike[t_ends] - cumsum_spike[t_starts]

    m1 = counts.mean(axis=0)
    m2 = counts.var(axis=0, ddof=1)
    m4 = np.mean((counts - m1) ** 4, axis=0)

    eps = 1e-12
    kurt = m4 / (m2 + eps) ** 2

    if fisher:
        kurt = kurt - 3.0

    return kurt.reshape(orig_shape[1:])


def compute_raster(sp_matrix: np.ndarray, times: np.ndarray):
    """Get spike raster plot which displays the spiking activity of a group of
    neurons over time."""
    times = np.asarray(times)
    elements = np.where(sp_matrix > 0.0)
    index = elements[1]
    time = times[elements[0]]
    return index, time


def firing_rate(
    spikes: np.ndarray | torch.Tensor,
    width: int | float | None = 4,
    dt: int | float | None = None,
    axis: int | Sequence[int] | None = None,
):
    """Smooth spikes into firing rates.

    Supports input shapes like [T, ...].
    If axis is not None, averages over the specified dimensions before smoothing.
    """
    if dt is None:
        dt = 1.0

    if axis is not None:
        if isinstance(spikes, np.ndarray):
            spikes = spikes.mean(axis=axis)
        else:
            spikes = spikes.mean(dim=axis)

    if width is None or width == 0:
        return spikes / dt

    width1 = int(width // 2) * 2 + 1

    if isinstance(spikes, np.ndarray):
        window = np.ones(width1, dtype=float) / width1
        # Convolve along time axis (0) for all other dimensions
        out = convolve1d(spikes, window, axis=0, mode="constant", cval=0.0)
        return out / dt

    else:
        # torch implementation for arbitrary dimensions [T, *others]
        orig_shape = spikes.shape
        T = orig_shape[0]

        # Flatten others to treat as batches for conv1d: [T, B] -> [B, 1, T]
        x = spikes.reshape(T, -1).T.unsqueeze(1)

        window = torch.ones(width1, device=spikes.device, dtype=spikes.dtype) / width1
        weight = window.view(1, 1, -1)

        y = torch.conv1d(x, weight, padding="same")

        # [B, 1, T] -> [B, T] -> [T, B] -> [T, *others]
        return y.squeeze(1).T.reshape(orig_shape) / dt


def compute_spectrum(y, dt, nperseg=None):
    from scipy.signal import welch

    freqs, Y_mag = welch(y, fs=1 / dt, nperseg=nperseg, axis=0)
    return freqs, Y_mag
