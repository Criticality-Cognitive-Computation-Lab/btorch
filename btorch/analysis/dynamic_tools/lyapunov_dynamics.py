import nolds
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


def get_continuous_spiking_rate(spikes, dt, sigma=20.0):
    """Convert discrete spike trains into continuous firing rates using
    Gaussian smoothing.

    Args:
        spikes (np.ndarray or torch.Tensor): Spike matrix of shape (time_steps,
        n_neurons).
        dt (float): Simulation time step in ms.
        sigma (float): Standard deviation of the Gaussian kernel in ms. Default 20ms.

    Returns:
        np.ndarray: Continuous firing rate traces of shape (time_steps, n_neurons).
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()

    # Convert sigma from ms to bins
    sigma_bins = sigma / dt

    # Apply Gaussian filter along the time axis (axis 0)
    rates = gaussian_filter1d(spikes.astype(float), sigma=sigma_bins, axis=0)

    return rates


def compute_max_lyapunov_exponent(time_series, emb_dim=6, lag=1, tau=1):
    """Compute the largest Lyapunov exponent of a given time series using the
    nolds library.

    Parameters:
    - time_series: A 1D numpy array representing the time series data.

    Returns:
    - lyapunov_exponent: The estimated largest Lyapunov exponent.
    """
    lyapunov_exponent = nolds.lyap_r(time_series, emb_dim=emb_dim, lag=lag, tau=tau)
    return lyapunov_exponent


def compute_lyapunov_exponent_spectrum(time_series, emb_dim=6, matrix_dim=4, tau=1):
    """Compute the full Lyapunov spectrum of a given time series using the
    nolds library.

    Parameters:
    - time_series: A 1D numpy array representing the time series data.

    Returns:
    - lyapunov_spectrum: A list of estimated Lyapunov exponents.
    """
    lyapunov_spectrum = nolds.lyap_e(
        time_series, emb_dim=emb_dim, matrix_dim=matrix_dim, tau=tau
    )
    return lyapunov_spectrum


def compute_ks_entropy(time_series, emb_dim=6, lag=1):
    """Compute the Kolmogorov-Sinai (KS) entropy of a given time series using
    the nolds library.

    Parameters:
    - time_series: A 1D numpy array representing the time series data.

    Returns:
    - ks_entropy: The estimated KS entropy.
    """
    ks_entropy = nolds.sampen(time_series, emb_dim=emb_dim, lag=lag)
    return ks_entropy


def compute_expansion_to_contraction_ratio(lyapunov_spectrum):
    """Compute the ratio of expansion to contraction from the Lyapunov
    spectrum.

    Parameters:
    - lyapunov_spectrum: A list or numpy array of Lyapunov exponents.

    Returns:
    - ratio: The ratio of the sum of positive exponents to the absolute sum of
    negative exponents.
    """
    lyapunov_spectrum = np.array(lyapunov_spectrum)
    positive_sum = np.sum(lyapunov_spectrum[lyapunov_spectrum > 0])
    negative_sum = np.sum(np.abs(lyapunov_spectrum[lyapunov_spectrum < 0]))

    if negative_sum == 0:
        return np.inf  # Avoid division by zero; indicates pure expansion

    ratio = positive_sum / negative_sum
    return ratio
