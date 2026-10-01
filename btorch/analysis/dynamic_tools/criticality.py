import warnings
from typing import Any, TypedDict

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
    # A constant ``y`` has no variance to explain; report NaN instead of 0/0.
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    stats = {"r_squared": r_squared, "popt": popt, "pcov": pcov}
    return gamma, stats


class AvalancheStatistics(TypedDict):
    """Result of :func:`compute_avalanche_statistics` (keys are always all
    present; unestimable quantities are NaN/None)."""

    tau: float
    alpha: float
    gamma: float
    gamma_pred: float
    CCC: float
    sizes: np.ndarray
    durations: np.ndarray
    avg_size_by_duration: tuple[np.ndarray, np.ndarray]
    gamma_stats: dict[str, Any] | None
    fit_S: Any | None  # powerlaw.Fit
    fit_T: Any | None  # powerlaw.Fit


def _population_activity(spike_train: np.ndarray, bin_size: int) -> np.ndarray:
    """Population spike count per bin, ``(T,)`` or ``(T // bin_size,)``.

    A 2D ``(time_steps, n_neurons)`` input is summed over neurons, a 1D input
    is taken as the activity already. Trailing steps that do not fill a whole
    bin are dropped.
    """
    activity = spike_train.sum(axis=1) if spike_train.ndim == 2 else spike_train
    if bin_size > 1:
        n_bins = len(activity) // bin_size
        activity = activity[: n_bins * bin_size].reshape(-1, bin_size).sum(axis=1)
    return activity


def _extract_avalanches(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sizes (total spikes) and durations (bins) of runs of active bins."""
    # Pad with False so runs touching the boundaries get a start and an end.
    padded = np.concatenate(([False], activity > 0, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]  # exclusive

    sizes = np.array([activity[s:e].sum() for s, e in zip(starts, ends)])
    durations = np.array([e - s for s, e in zip(starts, ends)])
    return sizes, durations


def _mean_size_by_duration(
    sizes: np.ndarray, durations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average avalanche size per distinct duration, ``<S>(T)``.

    Grouped with ``bincount`` over the integer durations; empty arrays when
    there is no avalanche.
    """
    if len(durations) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    counts = np.bincount(durations)
    sum_sizes = np.bincount(durations, weights=sizes)
    mask = counts > 0
    return np.arange(len(counts))[mask], sum_sizes[mask] / counts[mask]


def _criticality_consistency(
    tau: float, alpha: float, gamma: float
) -> tuple[float, float]:
    """Return ``(gamma_pred, CCC)``; NaN where undefined.

    ``gamma_pred = (alpha - 1) / (tau - 1)`` (needs ``tau != 1``) and
    ``CCC = 1 - |gamma - gamma_pred| / gamma`` (needs ``gamma != 0``).
    """
    if np.isnan(tau) or np.isnan(alpha) or np.isnan(gamma) or tau == 1:
        return np.nan, np.nan
    gamma_pred = (alpha - 1) / (tau - 1)
    ccc = 1 - abs(gamma - gamma_pred) / gamma if gamma != 0 else np.nan
    return gamma_pred, ccc


def compute_avalanche_statistics(
    spike_train: np.ndarray, bin_size: int = 1
) -> AvalancheStatistics:
    """Calculate avalanche size (S) and duration (T) distributions and their
    power-law exponents.

    Definition: An avalanche is defined as a continuous sequence of time bins
    (width bin_size) containing at least one spike, flanked by empty bins.

    Args:
        spike_train (np.ndarray): Binary spike matrix of shape (time_steps, n_neurons).
        bin_size (int): Width of time bin in number of time steps.

    Returns:
        AvalancheStatistics: Dictionary containing:
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

    Raises:
        ValueError: If ``spike_train`` is not 2D.

    Notes:
        The result always has the same keys. When a quantity cannot be
        estimated (fewer than 10 avalanches, failed fit) its exponent is
        ``np.nan`` and the associated fit object / stats are ``None``; a
        :class:`UserWarning` is emitted.
    """
    spike_train = np.array(spike_train)
    if spike_train.ndim != 2:
        raise ValueError("spike_train must be a 2D matrix (time_steps, n_neurons)")

    activity = _population_activity(spike_train, bin_size)
    sizes, durations = _extract_avalanches(activity)
    unique_durations, mean_sizes = _mean_size_by_duration(sizes, durations)

    results: AvalancheStatistics = {
        "sizes": sizes,
        "durations": durations,
        "tau": np.nan,
        "alpha": np.nan,
        "gamma": np.nan,
        "gamma_pred": np.nan,
        "CCC": np.nan,
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
        return results

    # Power laws by MLE (powerlaw package); <S>(T) ~ T^gamma by least squares.
    results["tau"], results["fit_S"] = _fit_distribution(sizes)
    results["alpha"], results["fit_T"] = _fit_distribution(durations)
    results["gamma"], results["gamma_stats"] = _fit_scaling(
        unique_durations, mean_sizes
    )
    results["gamma_pred"], results["CCC"] = _criticality_consistency(
        results["tau"], results["alpha"], results["gamma"]
    )
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

    population_activity = _population_activity(spike_train, bin_size)

    # nolds.dfa takes the raw series (it integrates internally).
    try:
        alpha = nolds.dfa(population_activity)
        return alpha
    except (ValueError, RuntimeError, AssertionError, FloatingPointError) as e:
        warnings.warn(f"Failed to calculate DFA: {e}", stacklevel=2)
        return np.nan
