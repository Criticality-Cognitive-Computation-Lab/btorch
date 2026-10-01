import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ...utils._optional import require
from ...utils.array import to_numpy


def compute_continuous_spiking_rate(
    spikes: np.ndarray | torch.Tensor,
    dt: float,
    sigma: float = 20.0,
) -> np.ndarray | torch.Tensor:
    """Convert discrete spike trains into continuous firing rates using
    Gaussian smoothing.

    The output follows the input type: a ``torch.Tensor`` input gives a
    ``float64`` tensor on the input device, anything else a ``float64`` NumPy
    array. The smoothing itself runs in SciPy on the CPU.

    Args:
        spikes: Spike matrix of shape ``(time_steps, n_neurons)``.
        dt: Simulation time step in ms.
        sigma: Standard deviation of the Gaussian kernel in ms. Default 20 ms.

    Returns:
        Continuous firing rate traces of shape ``(time_steps, n_neurons)``,
        as a tensor for tensor input and as an array otherwise.
        There is no NaN failure return; NaN/inf in the input propagate.
    """
    # sigma is in ms, dt in ms: convert to bins
    sigma_bins = sigma / dt

    rates = gaussian_filter1d(to_numpy(spikes).astype(float), sigma=sigma_bins, axis=0)

    if isinstance(spikes, torch.Tensor):
        return torch.from_numpy(rates).to(spikes.device)
    return rates


def compute_max_lyapunov_exponent(
    time_series: np.ndarray,
    emb_dim: int = 6,
    lag: int = 1,
    tau: int = 1,
) -> float:
    """Compute the largest Lyapunov exponent of a given time series using the
    nolds library.

    Args:
        time_series: A 1D numpy array representing the time series data.
        emb_dim: Embedding dimension. Default 6.
        lag: Lag between samples. Default 1.
        tau: Time delay. Default 1.

    Returns:
        The estimated largest Lyapunov exponent. No NaN sentinel is added
        here; degenerate series give whatever ``nolds.lyap_r`` returns.
    """
    nolds = require("nolds", "analysis", "Lyapunov/entropy estimation")
    lyapunov_exponent = nolds.lyap_r(time_series, emb_dim=emb_dim, lag=lag, tau=tau)
    return lyapunov_exponent


def compute_lyapunov_exponent_spectrum(
    time_series: np.ndarray,
    emb_dim: int = 6,
    matrix_dim: int = 4,
    tau: int = 1,
) -> list:
    """Compute the full Lyapunov spectrum of a given time series using the
    nolds library.

    Args:
        time_series: A 1D numpy array representing the time series data.
        emb_dim: Embedding dimension. Default 6.
        matrix_dim: Matrix dimension. Default 4.
        tau: Time delay. Default 1.

    Returns:
        A list of estimated Lyapunov exponents. No NaN sentinel is added
        here; degenerate series give whatever ``nolds.lyap_e`` returns.
    """
    nolds = require("nolds", "analysis", "Lyapunov/entropy estimation")
    lyapunov_spectrum = nolds.lyap_e(
        time_series, emb_dim=emb_dim, matrix_dim=matrix_dim, tau=tau
    )
    return lyapunov_spectrum


def compute_ks_entropy(
    time_series: np.ndarray, emb_dim: int = 6, lag: int = 1
) -> float:
    """Compute the Kolmogorov-Sinai (KS) entropy of a given time series using
    the nolds library.

    Args:
        time_series: A 1D numpy array representing the time series data.
        emb_dim: Embedding dimension. Default 6.
        lag: Lag between samples. Default 1.

    Returns:
        The estimated KS entropy. No NaN sentinel is added here; degenerate
        series give whatever ``nolds.sampen`` returns.
    """
    nolds = require("nolds", "analysis", "Lyapunov/entropy estimation")
    ks_entropy = nolds.sampen(time_series, emb_dim=emb_dim, lag=lag)
    return ks_entropy


def compute_expansion_to_contraction_ratio(
    lyapunov_spectrum: list | np.ndarray,
) -> float:
    """Compute the ratio of expansion to contraction from the Lyapunov
    spectrum.

    Args:
        lyapunov_spectrum: A list or numpy array of Lyapunov exponents.

    Returns:
        The ratio of the sum of positive exponents to the absolute sum of
        negative exponents. ``np.inf`` when there is no negative exponent
        (pure expansion; for an all-zero spectrum this is also ``inf``). No
        NaN is returned.
    """
    lyapunov_spectrum = np.array(lyapunov_spectrum)
    positive_sum = np.sum(lyapunov_spectrum[lyapunov_spectrum > 0])
    negative_sum = np.sum(np.abs(lyapunov_spectrum[lyapunov_spectrum < 0]))

    if negative_sum == 0:
        return np.inf  # Avoid division by zero; indicates pure expansion

    ratio = positive_sum / negative_sum
    return ratio
