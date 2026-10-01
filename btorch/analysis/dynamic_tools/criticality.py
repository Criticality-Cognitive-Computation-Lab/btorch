import warnings

import numpy as np
from scipy.optimize import curve_fit

from ...utils._optional import require


def _fit_distribution(data):
    """Fit a discrete power law with the ``powerlaw`` package.

    Returns:
        ``(alpha, fit)``; ``(np.nan, None)`` when there are fewer than 10
        samples or the fit fails (a warning is emitted for failures).
    """
    if len(data) < 10:
        return np.nan, None
    powerlaw = require("powerlaw", "analysis", "power-law fitting")
    try:
        # discrete=True because sizes/durations are counts (integers)
        fit = powerlaw.Fit(data, discrete=True, verbose=False)
        return fit.alpha, fit
    except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError) as e:
        warnings.warn(f"Failed to fit power law distribution: {e}", stacklevel=2)
        return np.nan, None


def _power_law_func(x, a, gamma):
    return a * np.power(x, gamma)


def _fit_scaling(x, y):
    """Fit the power law scaling y = a * x^gamma using curve_fit.

    Returns:
        ``(gamma, stats)`` with ``stats`` a dict (``r_squared``, ``popt``,
        ``pcov``); ``(np.nan, None)`` when there are fewer than 3 points or
        the fit fails (a warning is emitted for failures).
    """
    if len(x) < 3:
        return np.nan, None

    try:
        # Initial guess: a=1, gamma=1.5. curve_fit raises RuntimeError when it
        # fails to converge and ValueError on invalid/NaN input.
        popt, pcov = curve_fit(_power_law_func, x, y, p0=[1, 1.5], maxfev=2000)
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Failed to fit scaling relation: {e}", stacklevel=2)
        return np.nan, None

    gamma = popt[1]
    residuals = y - _power_law_func(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    stats = {"r_squared": r_squared, "popt": popt, "pcov": pcov}
    return gamma, stats


def compute_avalanche_statistics(spike_train: np.ndarray, bin_size: int = 1) -> dict:
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
            - 'avg_size_by_duration': Tuple (unique_durations, mean_sizes);
              empty arrays when no avalanche was found
            - 'gamma_stats': Dict with 'r_squared', 'popt', 'pcov' of the
              scaling fit, or None
            - 'fit_S': powerlaw.Fit object for sizes, or None
            - 'fit_T': powerlaw.Fit object for durations, or None

    Notes:
        The result always has the same keys. When a quantity cannot be
        estimated (fewer than 10 avalanches, failed fit) its exponent is
        ``np.nan`` and the associated fit object / stats are ``None``; a
        :class:`UserWarning` is emitted.
    """
    spike_train = np.array(spike_train)

    # Check dimensions. We expect (Time, Neurons).
    if spike_train.ndim != 2:
        raise ValueError("spike_train must be a 2D matrix (time_steps, n_neurons)")

    # 1. Calculate population activity (sum spikes across neurons)
    population_activity = np.sum(spike_train, axis=1)  # Shape: (T,)

    if bin_size > 1:
        n_bins = len(population_activity) // bin_size
        # Truncate to multiple of bin_size
        population_activity = population_activity[: n_bins * bin_size]
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

    # Average size per duration (<S>(T)); grouped with bincount over the
    # integer durations. Empty arrays when there is no avalanche.
    if len(durations) > 0:
        counts = np.bincount(durations)
        sum_sizes = np.bincount(durations, weights=sizes)
        mask = counts > 0
        unique_durations = np.arange(len(counts))[mask]
        mean_sizes = sum_sizes[mask] / counts[mask]
    else:
        unique_durations = np.array([], dtype=int)
        mean_sizes = np.array([], dtype=float)

    results = {
        "sizes": sizes,
        "durations": durations,
        "tau": np.nan,
        "alpha": np.nan,
        "gamma": np.nan,
        "fit_S": None,
        "fit_T": None,
        "avg_size_by_duration": (unique_durations, mean_sizes),
        "gamma_stats": None,
    }

    if len(sizes) < 10:
        warnings.warn(
            f"Not enough avalanches to fit power law. Found {len(sizes)} avalanches.",
            stacklevel=2,
        )
        results.update(gamma_pred=np.nan, CCC=np.nan)
        return results

    # 4. Fit power laws using MLE (powerlaw package)
    results["tau"], results["fit_S"] = _fit_distribution(sizes)
    results["alpha"], results["fit_T"] = _fit_distribution(durations)

    # 5. Average Size vs. Duration Scaling (<S>(T) ~ T^gamma), computed above.
    # Fit scaling relation using curve_fit (non-linear least squares)
    results["gamma"], results["gamma_stats"] = _fit_scaling(
        unique_durations, mean_sizes
    )

    # 6. Calculate Criticality Consistency Coefficient (CCC)
    # gamma_pred = (alpha - 1) / (tau - 1)
    # CCC = 1 - |gamma_obs - gamma_pred| / gamma_obs
    gamma_pred = np.nan
    ccc = np.nan
    tau, alpha, gamma = results["tau"], results["alpha"], results["gamma"]
    if not (np.isnan(tau) or np.isnan(alpha) or np.isnan(gamma)):
        if tau != 1:
            gamma_pred = (alpha - 1) / (tau - 1)
            if gamma != 0:
                ccc = 1 - abs(gamma - gamma_pred) / gamma
    results.update(gamma_pred=gamma_pred, CCC=ccc)

    return results


def compute_dfa(spike_train: np.ndarray, bin_size: int = 1) -> float:
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
        float: The DFA exponent alpha, or ``np.nan`` (with a warning) when
        the computation fails.
    """
    nolds = require("nolds", "analysis", "DFA analysis")

    spike_train = np.array(spike_train)

    # 1. Calculate population activity (sum spikes across neurons)
    if spike_train.ndim == 2:
        population_activity = np.sum(spike_train, axis=1)
    else:
        population_activity = spike_train

    if bin_size > 1:
        n_bins = len(population_activity) // bin_size
        population_activity = population_activity[: n_bins * bin_size]
        population_activity = population_activity.reshape(-1, bin_size).sum(axis=1)

    # 3. Calculate DFA using nolds
    # nolds.dfa expects the time series (it performs integration internally)
    try:
        alpha = nolds.dfa(population_activity)
        return alpha
    except (ValueError, RuntimeError, AssertionError, FloatingPointError) as e:
        warnings.warn(f"Failed to calculate DFA: {e}", stacklevel=2)
        return np.nan
