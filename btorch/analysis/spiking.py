import numpy as np
import torch
from scipy.ndimage import convolve1d


def cv_from_spikes(spike_data: np.ndarray, dt_ms: float = 1.0):
    """Calculate coefficient of variation of ISIs per neuron."""
    if spike_data.ndim > 2:
        spike_data = spike_data.reshape((-1, spike_data.shape[-1]))
    n_timesteps, n_neurons = spike_data.shape
    cv_values = np.full(n_neurons, np.nan)
    isi_stats = {}

    for neuron_idx in range(n_neurons):
        spike_times = np.where(spike_data[:, neuron_idx] > 0)[0]

        if len(spike_times) < 2:
            cv_values[neuron_idx] = np.nan
            isi_stats[neuron_idx] = {
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

        cv_values[neuron_idx] = cv
        isi_stats[neuron_idx] = {
            "n_spikes": len(spike_times),
            "mean_isi": mean_isi,
            "std_isi": std_isi,
            "cv": cv,
            "isi_values": isi_values,
        }

    isi_values = np.concatenate([s["isi_values"] for s in isi_stats.values()])
    if len(isi_values) == 0:
        return (
            cv_values,
            {"mean_isi": np.nan, "std_isi": np.nan, "cv": np.nan},
            isi_stats,
        )

    isi_total = {
        "mean_isi": np.mean(isi_values),
        "std_isi": np.std(isi_values),
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
    """Compute Fano factor for spike trains."""
    T, B, N = spike.shape

    if sweep_window:
        out = np.zeros((T, B, N))
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

    counts = np.zeros((num_win, B, N))

    idx = 0
    for t0 in range(0, T - window + 1, step):
        t1 = t0 + window
        counts[idx] = spike[t0:t1].sum(axis=0)
        idx += 1

    mean_counts = counts.mean(axis=0)
    var_counts = counts.var(axis=0, ddof=1)

    # Safely compute Fano: only divide where mean_counts is finite and > 0
    fano = np.zeros_like(var_counts, dtype=float)
    valid = np.isfinite(mean_counts) & (mean_counts > 0) & np.isfinite(var_counts)
    if np.any(valid):
        fano[valid] = var_counts[valid] / mean_counts[valid]
    return fano


def kurtosis_from_spikes(
    spike: np.ndarray,
    window: int | None = None,
    overlap: int = 0,
    sweep_window: bool = False,
    dt_ms: float = 1.0,
    fisher: bool = True,
):
    """Compute kurtosis of spike counts across windows."""
    T, B, N = spike.shape

    if sweep_window:
        out = np.zeros((T, B, N))
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

    counts = np.zeros((num_win, B, N))

    idx = 0
    for t0 in range(0, T - window + 1, step):
        t1 = t0 + window
        counts[idx] = spike[t0:t1].sum(axis=0)
        idx += 1

    m1 = counts.mean(axis=0)
    m2 = counts.var(axis=0, ddof=1)
    m4 = np.mean((counts - m1) ** 4, axis=0)

    eps = 1e-12
    kurt = m4 / (m2 + eps) ** 2

    if fisher:
        kurt = kurt - 3.0

    return kurt


def raster_plot(sp_matrix: np.ndarray, times: np.ndarray):
    """Get spike raster plot which displays the spiking activity of a group of
    neurons over time."""
    times = np.asarray(times)
    elements = np.where(sp_matrix > 0.0)
    index = elements[1]
    time = times[elements[0]]
    return index, time


def firing_rate(
    spikes: np.ndarray | torch.Tensor,
    width: int | float,
    dt: int | float | None = None,
    per_neuron: bool = False,
):
    """Smooth spikes into firing rates."""
    if dt is None:
        dt = 1.0

    width1 = int(width // 2) * 2 + 1

    if isinstance(spikes, np.ndarray):
        if spikes.ndim == 2:
            if not per_neuron:
                spikes = spikes.mean(axis=-1)
        elif spikes.ndim != 1:
            raise ValueError("NumPy spikes must be 1D or 2D")

        window = np.ones(width1, dtype=float) / width1
        # Convolve along time axis for every neuron simultaneously
        out = convolve1d(spikes, window, axis=0, mode="constant", cval=0.0)

        return out / dt

    else:
        if spikes.ndim >= 2:
            if not per_neuron:
                spikes = spikes.mean(dim=-1)
        elif spikes.ndim != 1:
            raise ValueError("Torch spikes must be 1D or 2D")

        window = torch.ones(width1, device=spikes.device, dtype=spikes.dtype) / width1
        weight = window.view(1, 1, -1)

        if spikes.ndim == 1:
            x = spikes.view(1, 1, -1)
            y = torch.conv1d(x, weight, padding="same")[0, 0]
        else:
            x = spikes.T.unsqueeze(1)
            y = torch.conv1d(x, weight, padding="same")[:, 0].T

        return y / dt
