from collections.abc import Sequence

import numpy as np
import torch
from scipy.ndimage import convolve1d


def cv_from_spikes(spike_data: np.ndarray, dt_ms: float = 1.0):
    """Calculate coefficient of variation of ISIs per neuron.

    Supports input shapes like [T, ...]. CV is calculated for each neuron/batch
    dimension.
    """
    orig_shape = spike_data.shape
    T = orig_shape[0]
    # Flatten all but time dimension
    flat_data = spike_data.reshape(T, -1)
    n_flat = flat_data.shape[1]

    cv_values = np.full(n_flat, np.nan)
    isi_stats = {}

    for i in range(n_flat):
        spike_times = np.where(flat_data[:, i] > 0)[0]

        if len(spike_times) < 2:
            isi_stats[i] = {
                "n_spikes": len(spike_times),
                "mean_isi": np.nan,
                "std_isi": np.nan,
                "cv": np.nan,
                "isi_values": [],
            }
            continue

        isi_values = np.diff(spike_times) * dt_ms
        mean_isi = np.mean(isi_values)
        std_isi = np.std(isi_values)
        cv = std_isi / mean_isi if mean_isi > 0 else np.nan

        cv_values[i] = cv
        isi_stats[i] = {
            "n_spikes": len(spike_times),
            "mean_isi": mean_isi,
            "std_isi": std_isi,
            "cv": cv,
            "isi_values": isi_values,
        }

    # Reshape cv_values back to match original non-time dimensions
    cv_values = cv_values.reshape(orig_shape[1:])

    all_isi_list = [
        s["isi_values"] for s in isi_stats.values() if len(s["isi_values"]) > 0
    ]
    if not all_isi_list:
        return (
            cv_values,
            {"mean_isi": np.nan, "std_isi": np.nan, "cv": np.nan},
            isi_stats,
        )

    isi_concatenated = np.concatenate(all_isi_list)
    isi_total = {
        "mean_isi": np.mean(isi_concatenated),
        "std_isi": np.std(isi_concatenated),
    }
    isi_total["cv"] = (
        isi_total["std_isi"] / isi_total["mean_isi"]
        if isi_total["mean_isi"] > 0
        else np.nan
    )

    return cv_values, isi_total, isi_stats


def fano_factor_from_spikes(
    spike: np.ndarray,
    window: int | None = None,
    overlap: int = 0,
    sweep_window: bool = False,
):
    """Compute Fano factor for spike trains.

    Supports input shapes like [T, ...].
    Returns Fano factor for each non-time dimension.
    """
    orig_shape = spike.shape
    T = orig_shape[0]

    if sweep_window:
        # Returns [T, ...]
        out = np.zeros(orig_shape)
        for w in range(1, T + 1):
            out[w - 1] = fano_factor_from_spikes(
                spike,
                window=w,
                overlap=overlap,
                sweep_window=False,
            )
        return out

    if window is None:
        window = T

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap
    num_win = 1 + (T - window) // step

    # Flatten others
    flat_spike = spike.reshape(T, -1)
    n_flat = flat_spike.shape[1]

    counts = np.zeros((num_win, n_flat))

    idx = 0
    for t0 in range(0, T - window + 1, step):
        t1 = t0 + window
        counts[idx] = flat_spike[t0:t1].sum(axis=0)
        idx += 1

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
    """Compute kurtosis of spike counts across windows.

    Supports input shapes like [T, ...].
    """
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

    assert 1 <= window <= T
    assert overlap < window

    step = window - overlap
    num_win = 1 + (T - window) // step

    # Flatten others
    flat_spike = spike.reshape(T, -1)
    n_flat = flat_spike.shape[1]

    counts = np.zeros((num_win, n_flat))

    idx = 0
    for t0 in range(0, T - window + 1, step):
        t1 = t0 + window
        counts[idx] = flat_spike[t0:t1].sum(axis=0)
        idx += 1

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
