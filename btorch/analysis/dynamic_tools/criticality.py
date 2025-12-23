import warnings

import nolds
import numpy as np
import powerlaw
from scipy.optimize import curve_fit


def _fit_distribution(data):
    """Helper to fit power law distribution using powerlaw package."""
    if len(data) < 10:
        return np.nan, None
    try:
        # discrete=True because sizes/durations are counts (integers)
        fit = powerlaw.Fit(data, discrete=True, verbose=False)
        return fit.alpha, fit
    except Exception as e:
        warnings.warn(f"Failed to fit power law distribution: {e}")
        return np.nan, None


def _power_law_func(x, a, gamma):
    return a * np.power(x, gamma)


def _fit_scaling(x, y):
    """Helper to fit power law scaling y = a * x^gamma using curve_fit."""
    if len(x) < 3:
        return np.nan, None

    try:
        # Initial guess: a=1, gamma=1.5
        popt, pcov = curve_fit(_power_law_func, x, y, p0=[1, 1.5], maxfev=2000)
        gamma = popt[1]

        # Calculate R^2
        residuals = y - _power_law_func(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        stats = {"r_squared": r_squared, "popt": popt, "pcov": pcov}
        return gamma, stats
    except Exception as e:
        warnings.warn(f"Failed to fit scaling relation: {e}")
        return np.nan, None


def compute_avalanche_statistics(spike_train: np.ndarray, bin_size: int = 1):
    """Calculate avalanche size (S) and duration (T) distributions and their
    power-law exponents.

    Definition: An avalanche is defined as a continuous sequence of time bins
    (width bin_size) containing at least one spike, flanked by empty bins.

    Args:
        spike_train (np.ndarray): Binary spike matrix of shape (time_steps, n_neurons).
        bin_size (int): Width of time bin in number of time steps.

    Returns:
        dict: Dictionary containing:
            - 'tau': Power-law exponent for avalanche size distribution P(S) ~
              S^-tau
            - 'alpha': Power-law exponent for avalanche duration distribution
              P(T) ~ T^-alpha
            - 'gamma': Power-law exponent for average size vs duration <S>(T) ~
              T^gamma
            - 'gamma_pred': Predicted gamma based on tau and alpha:
              (alpha-1)/(tau-1)
            - 'CCC': Criticality Consistency Coefficient: 1 - |gamma -
              gamma_pred| / gamma
            - 'sizes': List of avalanche sizes
            - 'durations': List of avalanche durations
            - 'avg_size_by_duration': Tuple (unique_durations, mean_sizes)
            - 'fit_S': powerlaw.Fit object for sizes
            - 'fit_T': powerlaw.Fit object for durations
    """
    # Ensure input is numpy array
    spike_train = np.array(spike_train)

    # Check dimensions. We expect (Time, Neurons).
    if spike_train.ndim != 2:
        raise ValueError("spike_train must be a 2D matrix (time_steps, n_neurons)")

    # 1. Calculate population activity (sum spikes across neurons)
    population_activity = np.sum(spike_train, axis=1)  # Shape: (T,)

    # 2. Binning
    if bin_size > 1:
        n_bins = len(population_activity) // bin_size
        # Truncate to multiple of bin_size
        population_activity = population_activity[: n_bins * bin_size]
        # Reshape and sum
        population_activity = population_activity.reshape(-1, bin_size).sum(axis=1)

    # 3. Identify avalanches
    # Active bins are those with > 0 spikes
    is_active = population_activity > 0

    # Find continuous sequences of active bins
    # Pad with False to detect start/end at boundaries
    padded_active = np.concatenate(([False], is_active, [False]))
    diff = np.diff(padded_active.astype(int))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    sizes = []
    durations = []

    for start, end in zip(starts, ends):
        # segment from start to end (exclusive)
        segment = population_activity[start:end]

        # Size (S): Total number of spikes in the avalanche
        s = np.sum(segment)

        # Duration (T): Number of time bins the avalanche lasts
        t = len(segment)  # equivalent to end - start

        sizes.append(s)
        durations.append(t)

    sizes = np.array(sizes)
    durations = np.array(durations)

    results = {
        "sizes": sizes,
        "durations": durations,
        "tau": np.nan,
        "alpha": np.nan,
        "gamma": np.nan,
        "gamma_pred": np.nan,
        "CCC": np.nan,
        "fit_S": None,
        "fit_T": None,
    }

    if len(sizes) < 10:
        warnings.warn(
            f"Not enough avalanches to fit power law. Found {len(sizes)} avalanches."
        )
        return results

    # 4. Fit power laws using MLE (powerlaw package)
    results["tau"], results["fit_S"] = _fit_distribution(sizes)
    results["alpha"], results["fit_T"] = _fit_distribution(durations)

    # 5. Average Size vs. Duration Scaling (<S>(T) ~ T^gamma)
    if len(durations) > 0:
        # Use bincount for fast grouping by integer duration
        counts = np.bincount(durations)
        sum_sizes = np.bincount(durations, weights=sizes)

        # Filter out durations that didn't occur
        mask = counts > 0
        unique_durations = np.arange(len(counts))[mask]
        mean_sizes = sum_sizes[mask] / counts[mask]

        results["avg_size_by_duration"] = (unique_durations, mean_sizes)

        # Fit scaling relation using curve_fit (non-linear least squares)
        results["gamma"], results["gamma_stats"] = _fit_scaling(
            unique_durations, mean_sizes
        )

    # 6. Calculate Criticality Consistency Coefficient (CCC)
    # gamma_pred = (alpha - 1) / (tau - 1)
    # CCC = 1 - |gamma_obs - gamma_pred| / gamma_obs
    if (
        not np.isnan(results["tau"])
        and not np.isnan(results["alpha"])
        and not np.isnan(results["gamma"])
    ):
        try:
            if results["tau"] != 1:
                gamma_pred = (results["alpha"] - 1) / (results["tau"] - 1)
                results["gamma_pred"] = gamma_pred

                if results["gamma"] != 0:
                    ccc = 1 - abs(results["gamma"] - gamma_pred) / results["gamma"]
                    results["CCC"] = ccc
        except Exception as e:
            warnings.warn(f"Failed to calculate CCC: {e}")

    return results


def calculate_dfa(spike_train: np.ndarray, bin_size: int = 1):
    """Calculate Detrended Fluctuation Analysis (DFA) exponent alpha.

    Meaning of alpha:
    - 0.5: White noise (no memory)
    - 0.5 < alpha < 1.0: Long-range memory (fractal structure)
    - 1.0: 1/f noise (Pink noise)
    - 1.5: Brownian motion (Random walk)

    Args:
        spike_train (np.ndarray): Binary spike matrix of shape (time_steps, n_neurons).
        bin_size (int): Width of time bin in number of time steps.

    Returns:
        float: The DFA exponent alpha.
    """
    # Ensure input is numpy array
    spike_train = np.array(spike_train)

    # 1. Calculate population activity (sum spikes across neurons)
    if spike_train.ndim == 2:
        population_activity = np.sum(spike_train, axis=1)
    else:
        population_activity = spike_train

    # 2. Binning
    if bin_size > 1:
        n_bins = len(population_activity) // bin_size
        population_activity = population_activity[: n_bins * bin_size]
        population_activity = population_activity.reshape(-1, bin_size).sum(axis=1)

    # 3. Calculate DFA using nolds
    # nolds.dfa expects the time series (it performs integration internally)
    try:
        alpha = nolds.dfa(population_activity)
        return alpha
    except Exception as e:
        warnings.warn(f"Failed to calculate DFA: {e}")
        return np.nan
