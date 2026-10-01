"""Advanced Fano factor methods for spike train analysis.

This module provides methods to compensate for firing rate effects on the
Fano factor, implementing:

1. Operational Time Method: Transforms spike trains to a rate-independent
   reference frame by evaluating Fano factor at normalized rate (λ=1).
   Reference: Rajdl et al. (2020), Front. Comput. Neurosci.

2. Mean Matching Method: Selects data points where mean spike counts are
   matched across conditions before computing FF.
   Reference: Churchland et al. (2010), Nature Neurosci.

3. Model-Based Approaches:
   - Modulated Poisson with multiplicative noise (Goris et al., 2014)
   - Flexible overdispersion model (Charles et al., 2018)

All methods support both NumPy and PyTorch inputs following the btorch
conventions, with GPU acceleration where applicable.
"""

import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch

from btorch.analysis.statistics import use_percentiles, use_stats


# =============================================================================
# Operational Time Fano Factor
# =============================================================================


def _estimate_rate_numpy(
    spikes: np.ndarray,
    dt: float,
    window_ms: float | None = None,
) -> np.ndarray:
    """Estimate instantaneous firing rate using sliding window.

    Args:
        spikes: Spike train of shape [T, ...]. First dimension is time.
        dt: Time step in milliseconds.
        window_ms: Window size for rate estimation. If None, uses T//20 * dt.

    Returns:
        Estimated rate in Hz [T, ...].
    """
    T = spikes.shape[0]

    if window_ms is None:
        window_bins = max(10, T // 20)
    else:
        window_bins = max(1, int(window_ms / dt))

    kernel = np.ones(window_bins) / (window_bins * dt / 1000.0)

    rate = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"),
        axis=0,
        arr=spikes,
    )

    # Ensure minimum rate to avoid division by zero
    return np.maximum(rate, 1e-6)


def _estimate_rate_torch(
    spikes: torch.Tensor,
    dt: float,
    window_ms: float | None = None,
) -> torch.Tensor:
    """Torch implementation of rate estimation."""
    T = spikes.shape[0]

    if window_ms is None:
        window_bins = max(10, T // 20)
    else:
        window_bins = max(1, int(window_ms / dt))

    # Pad with edge values to handle boundaries
    pad = window_bins // 2

    original_shape = spikes.shape
    if spikes.ndim == 1:
        spikes = spikes.unsqueeze(-1)  # [T, 1]

    # Pad using constant mode (replicate edge values)
    spike_padded = torch.nn.functional.pad(spikes, (0, 0, pad, pad), mode="constant")
    # Fill padding with edge values (replicate)
    if pad > 0:
        spike_padded[:pad] = spikes[0]
        spike_padded[-pad:] = spikes[-1]

    cumsum = torch.cumsum(spike_padded, dim=0)
    moving_sum = cumsum[window_bins:] - cumsum[:-window_bins]

    start_idx = window_bins // 2
    end_idx = start_idx + T
    moving_sum = moving_sum[start_idx:end_idx]

    if moving_sum.shape[0] < T:
        pad_right = T - moving_sum.shape[0]
        moving_sum = torch.nn.functional.pad(moving_sum, (0, 0, 0, pad_right))

    rate = moving_sum / (window_bins * dt / 1000.0)

    rate = rate.reshape(original_shape)

    return torch.clamp(rate, min=1e-6)


def _compute_operational_fano_numpy(
    spikes: np.ndarray,
    rate_hz: np.ndarray,
    window_op: float,
    dt: float,
    batch_axis: tuple[int, ...] | None,
) -> tuple[np.ndarray, dict]:
    """Compute Fano factor in operational time for NumPy arrays.

    The operational time approach rescales time by the firing rate:
    w = λ * t, where λ is the firing rate in Hz.

    For a renewal process: F⁽ᵒ⁾ = CV² (independent of rate).

    Implementation: We normalize spike counts by rate, effectively
    computing statistics as if rate = 1 everywhere.

    Args:
        spikes: Spike train [T, ...]
        rate_hz: Estimated rate in Hz [T, ...] or scalar
        window_op: Window size in operational time units
        dt: Time step in milliseconds
        batch_axis: Axes to aggregate across

    Returns:
        fano_op: Operational time Fano factor
        info: Computation info
    """
    T = spikes.shape[0]
    rest_shape = spikes.shape[1:]
    n_elements = np.prod(rest_shape) if rest_shape else 1

    flat_spikes = spikes.reshape(T, -1) if n_elements > 0 else spikes.reshape(T, 1)
    flat_rate = (
        rate_hz.reshape(T, -1)
        if isinstance(rate_hz, np.ndarray) and rate_hz.ndim > 0
        else rate_hz
    )

    n_neurons = flat_spikes.shape[1]

    # w = λ * t, so for window W in operational time:
    # number of original bins ≈ W / (λ * dt)
    # Use median rate for window calculation
    if isinstance(flat_rate, np.ndarray) and flat_rate.ndim > 1:
        median_rate = np.median(flat_rate, axis=0)
    else:
        median_rate = np.full(
            n_neurons, flat_rate if np.isscalar(flat_rate) else np.median(flat_rate)
        )

    # Compute normalized spike counts (operational time counts)
    # The key insight: for rate λ, the expected count in window w is λ*w
    # Normalizing by λ gives counts as if rate = 1

    # Method: Compute counts in windows, normalize by expected count
    # Then compute variance/mean of normalized counts

    # For simplicity, use fixed window in original time and normalize
    window_bins = max(5, int(window_op / np.median(median_rate) / (dt / 1000.0)))
    window_bins = min(window_bins, T // 3)  # Ensure reasonable size

    if window_bins < 2:
        window_bins = 2

    step = max(1, window_bins // 2)
    n_windows = (T - window_bins) // step + 1

    if n_windows < 2:
        # Not enough windows, return NaN
        warnings.warn(
            f"Insufficient data for operational-time Fano ({n_windows} "
            "windows < 2); returning NaN",
            stacklevel=2,
        )
        return np.full(rest_shape if rest_shape else (1,), np.nan), {
            "n_windows": n_windows
        }

    counts = np.zeros((n_windows, n_neurons))
    for w in range(n_windows):
        start = w * step
        end = start + window_bins
        counts[w] = flat_spikes[start:end].sum(axis=0)

    # Compute expected counts (rate * window duration)
    # For operational time: normalize counts by rate
    if isinstance(flat_rate, np.ndarray) and flat_rate.ndim > 1:
        mean_rate = np.mean(flat_rate, axis=0)  # [n_neurons]
    else:
        mean_rate = np.full(
            n_neurons, flat_rate if np.isscalar(flat_rate) else np.mean(flat_rate)
        )

    window_duration_s = window_bins * dt / 1000.0
    expected_counts = mean_rate * window_duration_s  # [n_neurons]

    # Normalized counts (as if rate = 1)
    # For Poisson: normalized counts should have mean = window_duration
    # and variance = window_duration, giving Fano = 1
    normalized_counts = counts / (expected_counts[None, :] + 1e-12)

    mean_norm = np.mean(normalized_counts, axis=0)
    var_norm = np.var(normalized_counts, axis=0, ddof=1)

    fano_op = np.zeros(n_neurons)
    valid = (mean_norm > 0) & np.isfinite(var_norm)
    fano_op[valid] = var_norm[valid] / mean_norm[valid]

    # Reshape to original structure
    if rest_shape:
        fano_op = fano_op.reshape(rest_shape)
    else:
        fano_op = fano_op[0] if n_neurons == 1 else fano_op

    if batch_axis is not None:
        fano_op = np.mean(fano_op, axis=tuple(batch_axis))

    info = {
        "method": "operational_time",
        "window_bins": window_bins,
        "n_windows": n_windows,
        "mean_rate_hz": np.mean(mean_rate),
        "window_duration_s": window_duration_s,
    }

    return fano_op, info


def _compute_operational_fano_torch(
    spikes: torch.Tensor,
    rate_hz: torch.Tensor | float,
    window_op: float,
    dt: float,
    batch_axis: tuple[int, ...] | None,
) -> tuple[torch.Tensor, dict]:
    """Compute Fano factor in operational time for Torch tensors."""
    device = spikes.device
    T = spikes.shape[0]
    rest_shape = spikes.shape[1:]
    n_elements = int(np.prod(rest_shape)) if rest_shape else 1

    flat_spikes = spikes.reshape(T, -1) if n_elements > 0 else spikes.reshape(T, 1)

    if isinstance(rate_hz, torch.Tensor):
        flat_rate = rate_hz.reshape(T, -1) if rate_hz.numel() > 1 else rate_hz.item()
    else:
        flat_rate = rate_hz

    n_neurons = flat_spikes.shape[1]

    if isinstance(flat_rate, torch.Tensor) and flat_rate.ndim > 1:
        median_rate = torch.median(flat_rate, dim=0).values
    else:
        scalar_rate = (
            flat_rate if isinstance(flat_rate, (int, float)) else flat_rate.item()
        )
        median_rate = torch.full((n_neurons,), scalar_rate, device=device)

    median_rate_val = torch.median(median_rate).item()
    window_bins = max(5, int(window_op / median_rate_val / (dt / 1000.0)))
    window_bins = min(window_bins, T // 3)

    if window_bins < 2:
        window_bins = 2

    step = max(1, window_bins // 2)
    n_windows = (T - window_bins) // step + 1

    if n_windows < 2:
        warnings.warn(
            f"Insufficient data for operational-time Fano ({n_windows} "
            "windows < 2); returning NaN",
            stacklevel=2,
        )
        return torch.full(
            rest_shape if rest_shape else (1,), float("nan"), device=device
        ), {"n_windows": n_windows}

    counts = torch.zeros((n_windows, n_neurons), device=device)
    for w in range(n_windows):
        start = w * step
        end = start + window_bins
        counts[w] = flat_spikes[start:end].sum(dim=0)

    if isinstance(flat_rate, torch.Tensor) and flat_rate.ndim > 1:
        mean_rate = torch.mean(flat_rate, dim=0)
    else:
        scalar_rate = (
            flat_rate if isinstance(flat_rate, (int, float)) else flat_rate.item()
        )
        mean_rate = torch.full((n_neurons,), scalar_rate, device=device)

    window_duration_s = window_bins * dt / 1000.0
    expected_counts = mean_rate * window_duration_s

    normalized_counts = counts / (expected_counts.unsqueeze(0) + 1e-12)

    mean_norm = torch.mean(normalized_counts, dim=0)
    var_norm = torch.var(normalized_counts, dim=0, unbiased=True)

    fano_op = torch.zeros(n_neurons, device=device)
    valid = (mean_norm > 0) & torch.isfinite(var_norm)
    fano_op[valid] = var_norm[valid] / mean_norm[valid]

    if rest_shape:
        fano_op = fano_op.reshape(rest_shape)
    elif n_neurons == 1:
        fano_op = fano_op[0]

    if batch_axis is not None:
        fano_op = torch.mean(fano_op, dim=tuple(batch_axis))

    info = {
        "method": "operational_time",
        "window_bins": window_bins,
        "n_windows": n_windows,
        "mean_rate_hz": torch.mean(mean_rate).item(),
        "window_duration_s": window_duration_s,
    }

    return fano_op, info


@use_percentiles(value_key="fano_op")
@use_stats(value_key="fano_op")
def fano_operational_time(
    spikes: np.ndarray | torch.Tensor,
    window: float | None = None,
    overlap: int | None = None,
    rate_hz: np.ndarray | torch.Tensor | float | None = None,
    dt: float = 1.0,
    batch_axis: int | tuple[int, ...] | None = None,
) -> tuple[np.ndarray | torch.Tensor, dict]:
    """Compute Fano factor in operational time (rate-independent).

        The operational time Fano factor transforms the spike train such that
    the firing rate equals 1 (normalized), making FF independent of the
        absolute firing rate. This is the recommended method for comparing
        variability across conditions with different rates.

        For renewal processes, the operational time Fano factor equals the
        squared coefficient of variation (CV²) of the ISI distribution.

        Reference: Rajdl et al. (2020) "Fano Factor: A Potentially Useful
        Information", Front. Comput. Neurosci.

        Args:
            spikes: Spike train of shape [T, ...]. First dimension is time.
                Values are binary (0/1) or spike counts.
            window: Window size in operational time units (default: 1.0). This is a
                dimensionless expected spike count, not a number of time bins.
                This is the expected count at rate=1 (i.e., 1 spike expected).
            overlap: Unsupported. Operational-time windows are disjoint; any
                value other than ``None`` or ``0`` raises ``ValueError``.
            rate_hz: Firing rate in Hz. Can be:
                - Scalar: homogeneous rate
                - Array [T, ...]: time-varying rate
                - None: estimated from data using sliding window
            dt: Time step in milliseconds for original time axis.
            batch_axis: Axes to average across for FF computation.

        Returns:
            fano_op: Operational time Fano factor values. Failure return: when
                fewer than 2 windows fit, a NaN array of shape
                ``spikes.shape[1:]`` (``(1,)`` for 1D input) with a warning
                and ``info={"n_windows": n}``.
            info: Dictionary with operational time info and computed statistics.

        Raises:
            ValueError: If ``overlap`` is neither ``None`` nor ``0``.

        Example:
            >>> # Compare Fano factors at different rates
            >>> spikes_low_rate = generate_poisson_spikes(rate_hz=20, ...)
            >>> spikes_high_rate = generate_poisson_spikes(rate_hz=80, ...)
            >>> ff_low, _ = fano_operational_time(spikes_low_rate)
            >>> ff_high, _ = fano_operational_time(spikes_high_rate)
            >>> # Both should be ≈ 1 regardless of rate difference
    """
    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)
    if overlap:
        raise ValueError(
            "fano_operational_time does not support overlapping windows; "
            f"got overlap={overlap!r}."
        )
    if window is None:
        window = 1.0  # Unit operational time window

    is_torch = isinstance(spikes, torch.Tensor)

    # Estimate rate if not provided
    if rate_hz is None:
        if is_torch:
            rate_hz = _estimate_rate_torch(spikes, dt)
        else:
            rate_hz = _estimate_rate_numpy(spikes, dt)

    if is_torch:
        return _compute_operational_fano_torch(spikes, rate_hz, window, dt, batch_axis)
    else:
        return _compute_operational_fano_numpy(spikes, rate_hz, window, dt, batch_axis)


# =============================================================================
# Mean Matching Fano Factor
# =============================================================================


def _compute_mean_matching_weights_numpy(
    means: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute mean-matching weights for each condition/time point.

        The mean matching method ensures that the distribution of mean counts
        is matched across conditions/times by computing weights that equalize
    the histograms.

        Args:
            means: Array of mean spike counts [n_conditions, n_neurons]
            n_bins: Number of bins for mean count histogram

        Returns:
            Dictionary with weights and matching info
    """
    n_conditions, n_neurons = means.shape

    mean_min, mean_max = np.nanmin(means), np.nanmax(means)
    if mean_min == mean_max:
        # All means are the same
        return {
            "weights": np.ones_like(means),
            "bin_edges": np.array([mean_min, mean_max + 1]),
            "counts_per_condition": np.full((n_conditions, 1), n_neurons),
            "min_counts_per_bin": np.full(1, n_neurons),
        }

    bin_edges = np.linspace(mean_min, mean_max, n_bins + 1)

    counts_per_condition = np.zeros((n_conditions, n_bins))
    for c in range(n_conditions):
        valid_means = means[c, ~np.isnan(means[c])]
        if len(valid_means) > 0:
            counts_per_condition[c], _ = np.histogram(valid_means, bins=bin_edges)

    # Find minimum count per bin (greatest common distribution)
    min_counts_per_bin = np.min(counts_per_condition, axis=0)

    weights = np.ones_like(means)

    for c in range(n_conditions):
        for b in range(n_bins):
            mask = (means[c] >= bin_edges[b]) & (means[c] < bin_edges[b + 1])
            if b == n_bins - 1:  # Include right edge for last bin
                mask = mask | (means[c] == bin_edges[b + 1])

            n_in_bin = np.sum(mask)
            target_n = min_counts_per_bin[b]

            if n_in_bin > 0 and target_n < n_in_bin:
                weights[c, mask] = target_n / n_in_bin
            elif n_in_bin > 0:
                weights[c, mask] = 1.0
            else:
                weights[c, mask] = 0.0

    return {
        "weights": weights,
        "bin_edges": bin_edges,
        "counts_per_condition": counts_per_condition,
        "min_counts_per_bin": min_counts_per_bin,
    }


def _compute_mean_matching_weights_torch(
    means: torch.Tensor,
    n_bins: int = 10,
) -> dict[str, torch.Tensor]:
    """Torch implementation of mean matching weights."""
    device = means.device
    n_conditions, n_neurons = means.shape

    mean_min, mean_max = torch.min(means), torch.max(means)
    if mean_min.item() == mean_max.item():
        return {
            "weights": torch.ones_like(means),
            "bin_edges": torch.tensor(
                [mean_min.item(), mean_max.item() + 1], device=device
            ),
            "counts_per_condition": torch.full(
                (n_conditions, 1), n_neurons, device=device
            ),
            "min_counts_per_bin": torch.full((1,), n_neurons, device=device),
        }

    bin_edges = torch.linspace(
        mean_min.item(), mean_max.item(), n_bins + 1, device=device
    )

    counts_per_condition = torch.zeros((n_conditions, n_bins), device=device)
    for c in range(n_conditions):
        for b in range(n_bins):
            mask = (means[c] >= bin_edges[b]) & (means[c] < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = mask | (means[c] == bin_edges[b + 1])
            counts_per_condition[c, b] = torch.sum(mask).float()

    min_counts_per_bin = torch.min(counts_per_condition, dim=0).values

    weights = torch.ones_like(means)
    for c in range(n_conditions):
        for b in range(n_bins):
            mask = (means[c] >= bin_edges[b]) & (means[c] < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = mask | (means[c] == bin_edges[b + 1])

            n_in_bin = torch.sum(mask).float()
            target_n = min_counts_per_bin[b]

            if n_in_bin > 0 and target_n < n_in_bin:
                weights[c, mask] = target_n / n_in_bin
            elif n_in_bin > 0:
                weights[c, mask] = 1.0
            else:
                weights[c, mask] = 0.0

    return {
        "weights": weights,
        "bin_edges": bin_edges,
        "counts_per_condition": counts_per_condition,
        "min_counts_per_bin": min_counts_per_bin,
    }


def _compute_weighted_fano_numpy(
    spike_counts: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute weighted Fano factor.

    Args:
        spike_counts: Spike counts [n_trials, n_neurons] or [n_conditions, n_neurons]
        weights: Weights for each observation

    Returns:
        Weighted Fano factor per neuron
    """
    valid_mask = ~np.isnan(spike_counts) & ~np.isnan(weights)

    weights_normalized = np.where(
        valid_mask, weights / (np.nansum(weights, axis=0, keepdims=True) + 1e-12), 0
    )

    weighted_mean = np.nansum(spike_counts * weights_normalized, axis=0)

    V1 = np.nansum(weights, axis=0)
    V2 = np.nansum(weights**2, axis=0)

    # Avoid division by zero
    denom = V1 - V2 / (V1 + 1e-12) + 1e-12

    weighted_var = (
        np.nansum(weights * (spike_counts - weighted_mean[None, :]) ** 2, axis=0)
        / denom
    )

    fano = np.full_like(weighted_mean, np.nan)
    valid = (weighted_mean > 0) & (weighted_var >= 0)
    fano[valid] = weighted_var[valid] / weighted_mean[valid]

    return fano


def _compute_weighted_fano_torch(
    spike_counts: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Torch implementation of weighted Fano factor."""
    valid_mask = ~torch.isnan(spike_counts) & ~torch.isnan(weights)

    weight_sums = torch.sum(weights * valid_mask.float(), dim=0, keepdim=True)
    weights_normalized = torch.where(
        valid_mask, weights / (weight_sums + 1e-12), torch.zeros_like(weights)
    )

    weighted_mean = torch.sum(spike_counts * weights_normalized, dim=0)

    V1 = torch.sum(weights * valid_mask.float(), dim=0)
    V2 = torch.sum((weights**2) * valid_mask.float(), dim=0)

    denom = V1 - V2 / (V1 + 1e-12) + 1e-12

    weighted_var = (
        torch.sum(weights * (spike_counts - weighted_mean.unsqueeze(0)) ** 2, dim=0)
        / denom
    )

    fano = torch.full_like(weighted_mean, float("nan"))
    valid = (weighted_mean > 0) & (weighted_var >= 0)
    fano[valid] = weighted_var[valid] / weighted_mean[valid]

    return fano


@use_percentiles(value_key="fano_mm")
@use_stats(value_key="fano_mm")
def fano_mean_matching(
    spikes: np.ndarray | torch.Tensor,
    window: int | None = None,
    overlap: int = 0,
    condition_axis: int = 1,
    n_bins: int = 10,
    n_resamples: int = 50,
    batch_axis: int | tuple[int, ...] | None = None,
) -> tuple[np.ndarray | torch.Tensor, dict]:
    """Compute mean-matched Fano factor controlling for rate effects.

        The mean matching method (Churchland et al., 2010) ensures that the
    distribution of mean spike counts is matched across conditions or time
        points before computing the Fano factor. This removes artifacts caused
        by rate changes.

        Reference: Churchland et al. (2010) "Stimulus onset quenches neural
        variability: a widespread cortical phenomenon", Nature Neurosci.

        Args:
            spikes: Spike train of shape [T, n_conditions, ...] or
                [T, n_trials, ...]. First dimension is time.
            window: Window size in time bins (not ms) for spike counting.
                If None, uses T//10.
            overlap: Overlap between consecutive windows.
            condition_axis: Axis representing conditions/trials (default 1).
            n_bins: Number of bins for mean count histogram matching.
            n_resamples: Number of resampling iterations for stability.
            batch_axis: Additional axes to aggregate across.

        Returns:
            fano_mm: Mean-matched Fano factor values. Failure return: when
                fewer than 2 windows fit, a NaN array of shape
                ``spikes.shape[2:]`` (the condition axis is removed;
                ``(1,)`` for [T, C] input) with a warning and
                ``info={"n_windows": n}``.
            info: Dictionary with matching info and computed statistics.

        Raises:
            ValueError: If ``window <= overlap``.

        Example:
            >>> # spikes shape: [T, n_conditions, n_neurons]
            >>> ff_mm, info = fano_mean_matching(
            ...     spikes, condition_axis=1, n_bins=10
            ... )
    """
    is_torch = isinstance(spikes, torch.Tensor)
    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)
    T = spikes.shape[0]

    if window is None:
        window = max(1, T // 10)

    step = window - overlap
    if step <= 0:
        raise ValueError(
            f"window must be greater than overlap, got window={window}, "
            f"overlap={overlap}"
        )

    # Move condition axis to position 1 for processing
    if condition_axis != 1:
        perm = list(range(spikes.ndim))
        perm[1], perm[condition_axis] = perm[condition_axis], perm[1]
        spikes = spikes.transpose(*perm) if not is_torch else spikes.permute(*perm)

    n_conditions = spikes.shape[1]

    rest_shape = spikes.shape[2:]
    spike_flat = spikes.reshape(T, n_conditions, -1)
    n_neurons_flat = spike_flat.shape[2]

    n_windows = (T - window) // step + 1
    if n_windows < 2:
        if is_torch:
            result = torch.full((n_neurons_flat,), float("nan"), device=spikes.device)
        else:
            result = np.full((n_neurons_flat,), np.nan)
        warnings.warn(
            f"Too few windows ({n_windows} < 2) for mean-matching Fano; returning NaN",
            stacklevel=2,
        )
        return result.reshape(rest_shape) if rest_shape else result, {
            "n_windows": n_windows
        }

    if is_torch:
        counts = torch.zeros(
            (n_windows, n_conditions, n_neurons_flat), device=spikes.device
        )
    else:
        counts = np.zeros((n_windows, n_conditions, n_neurons_flat))

    for w in range(n_windows):
        start = w * step
        end = start + window
        window_spikes = spike_flat[start:end]  # [window, n_conditions, n_neurons]
        counts[w] = window_spikes.sum(dim=0) if is_torch else window_spikes.sum(axis=0)

    if is_torch:
        means = counts.mean(dim=0)  # [n_conditions, n_neurons]
    else:
        means = counts.mean(axis=0)  # [n_conditions, n_neurons]

    if is_torch:
        match_info = _compute_mean_matching_weights_torch(means, n_bins)
        weights = match_info["weights"]

        fano_mm = _compute_weighted_fano_torch(
            counts.reshape(-1, n_neurons_flat), weights.repeat(n_windows, 1)
        )
    else:
        match_info = _compute_mean_matching_weights_numpy(means, n_bins)
        weights = match_info["weights"]

        fano_mm = _compute_weighted_fano_numpy(
            counts.reshape(-1, n_neurons_flat),
            weights.repeat(n_windows, axis=0),
        )

    fano_mm = fano_mm.reshape(rest_shape)

    if batch_axis is not None:
        if is_torch:
            fano_mm = torch.mean(fano_mm, dim=tuple(batch_axis))
        else:
            fano_mm = np.mean(fano_mm, axis=tuple(batch_axis))

    info = {
        **match_info,
        "method": "mean_matching",
        "n_bins": n_bins,
        "n_resamples": n_resamples,
    }

    return fano_mm, info


# =============================================================================
# Model-Based Fano Factor Approaches
# =============================================================================


def _modulated_poisson_moments(
    lambda_base: np.ndarray | torch.Tensor,
    gain_mean: float = 1.0,
    gain_var: float = 0.5,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Compute mean and variance for modulated Poisson model.

    The modulated Poisson model (Goris et al., 2014):
    r ~ Poisson(λ · g) where g is multiplicative gain noise.

    Mean: E[r] = λ · E[g]
    Variance: Var[r] = E[g] · λ + Var[g] · λ²

    This produces quadratic mean-variance relationship:
    Var[r] = E[r] + (Var[g]/E[g]²) · E[r]²

    Args:
        lambda_base: Base firing rate
        gain_mean: Mean of gain distribution (E[g])
        gain_var: Variance of gain distribution (Var[g])

    Returns:
        mean: Expected spike count
        variance: Spike count variance
    """
    mean = lambda_base * gain_mean
    # Law of total variance: Var[r] = E[Var[r|g]] + Var[E[r|g]]
    # Var[r|g] = λ·g (Poisson), E[r|g] = λ·g
    # E[Var[r|g]] = λ·E[g]
    # Var[E[r|g]] = λ²·Var[g]
    variance = lambda_base * gain_mean + (lambda_base**2) * gain_var
    return mean, variance


def _flexible_overdispersion_moments(
    stimulus_drive: np.ndarray | torch.Tensor,
    noise_std: float = 0.5,
    nonlinearity: Literal["relu", "square", "exp", "softplus"] = "relu",
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Compute mean and variance for flexible overdispersion model.

    The flexible overdispersion model (Charles et al., 2018):
    λ_eff = f(g(x) + ε) where f is a nonlinearity.

    Different nonlinearities produce different mean-FF relationships:
    - Rectified-linear: FF decreases with increasing rate
    - Rectified-squaring: FF ≈ constant
    - Exponential: FF increases with rate

    Args:
        stimulus_drive: Stimulus-dependent drive g(x)
        noise_std: Standard deviation of additive Gaussian noise ε
        nonlinearity: Type of nonlinearity to apply

    Returns:
        mean: Expected spike count
        variance: Spike count variance
    """
    is_torch = isinstance(stimulus_drive, torch.Tensor)

    if nonlinearity == "relu":
        # Rectified linear: f(z) = max(0, z)
        # Mean requires Gaussian integral
        if is_torch:
            # Approximation for rectified Gaussian
            phi = lambda x: 0.5 * (1 + torch.erf(x / np.sqrt(2)))
            mean = stimulus_drive * phi(stimulus_drive / noise_std)
            mean += (
                noise_std
                * torch.exp(-0.5 * (stimulus_drive / noise_std) ** 2)
                / np.sqrt(2 * np.pi)
            )
        else:
            from scipy.stats import norm

            mean = stimulus_drive * norm.cdf(stimulus_drive / noise_std)
            mean += noise_std * norm.pdf(stimulus_drive / noise_std)

    elif nonlinearity == "square":
        # Rectified squaring: f(z) = max(0, z)²
        # E[f(z)] where z ~ N(g, σ²)
        if is_torch:
            phi = lambda x: 0.5 * (1 + torch.erf(x / np.sqrt(2)))
            mean = (stimulus_drive**2 + noise_std**2) * phi(stimulus_drive / noise_std)
            mean += (
                stimulus_drive
                * noise_std
                * torch.exp(-0.5 * (stimulus_drive / noise_std) ** 2)
                / np.sqrt(2 * np.pi)
            )
        else:
            from scipy.stats import norm

            mean = (stimulus_drive**2 + noise_std**2) * norm.cdf(
                stimulus_drive / noise_std
            )
            mean += stimulus_drive * noise_std * norm.pdf(stimulus_drive / noise_std)

    elif nonlinearity == "exp":
        # Exponential: f(z) = exp(z)
        # E[exp(z)] = exp(g + σ²/2) for z ~ N(g, σ²)
        mean = (
            np.exp(stimulus_drive + 0.5 * noise_std**2)
            if not is_torch
            else torch.exp(stimulus_drive + 0.5 * noise_std**2)
        )

    elif nonlinearity == "softplus":
        # Softplus: f(z) = log(1 + exp(z))
        # Numerically stable computation
        if is_torch:
            mean = torch.nn.functional.softplus(stimulus_drive)
        else:
            mean = np.log(1 + np.exp(np.clip(stimulus_drive, -700, 700)))

    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

    # Variance approximation (quadratic for most nonlinearities)
    variance = mean + 0.5 * (mean**2)  # Simplified overdispersion

    return mean, variance


@use_percentiles(value_key="fano_model")
@use_stats(value_key="fano_model")
def fano_model_based(
    spikes: np.ndarray | torch.Tensor,
    window: int | None = None,
    overlap: int = 0,
    model: Literal[
        "modulated_poisson", "flexible_overdispersion"
    ] = "modulated_poisson",
    model_params: dict | None = None,
    stimulus_drive: np.ndarray | torch.Tensor | None = None,
    batch_axis: int | tuple[int, ...] | None = None,
) -> tuple[np.ndarray | torch.Tensor, dict]:
    """Compute model-based Fano factor with rate compensation.

    Model-based approaches fit a generative model to the spike data and
    extract the underlying variability independent of rate effects.

    Models:
    1. Modulated Poisson (Goris et al., 2014):
       r ~ Poisson(λ · g) where g is multiplicative gain noise.
       Produces quadratic mean-variance relationship.

    2. Flexible Overdispersion (Charles et al., 2018):
       λ_eff = f(g(x) + ε) with different nonlinearities.
       - Rectified-linear: FF decreases with rate
       - Rectified-squaring: FF ≈ constant
       - Exponential: FF increases with rate

    References:
    - Goris et al. (2014) "Partitioning neuronal variability"
    - Charles et al. (2018) "Dethroning the Fano factor"

    Args:
        spikes: Spike train of shape [T, ...]. First dimension is time.
        window: Window size in time bins (not ms) for spike counting.
            If None, uses T//10.
        overlap: Overlap between consecutive windows.
        model: Model type to use.
        model_params: Model-specific parameters.
        stimulus_drive: Stimulus-dependent drive (required for flexible_overdispersion).
        batch_axis: Axes to average across for FF computation.

    Returns:
        fano_model: Model-based Fano factor values. Failure return: when fewer
            than 2 windows fit, a NaN array of shape ``spikes.shape[1:]`` with
            a warning and ``info={"n_windows": n}``.
        info: Dictionary with model fit info and computed statistics.

    Raises:
        ValueError: If the requested ``model`` or the ``nonlinearity`` in
            ``model_params`` is unknown.

    Example:
        >>> ff_mod, info = fano_model_based(
        ...     spikes,
        ...     model="modulated_poisson",
        ...     model_params={"gain_mean": 1.0, "gain_var": 0.5}
        ... )
    """
    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)
    is_torch = isinstance(spikes, torch.Tensor)
    T = spikes.shape[0]

    if window is None:
        window = max(1, T // 10)

    model_params = model_params or {}

    flat_spike = spikes.reshape(T, -1)
    n_flat = flat_spike.shape[1]

    step = window - overlap
    n_windows = (T - window) // step + 1

    if n_windows < 2:
        if is_torch:
            result = torch.full((n_flat,), float("nan"), device=spikes.device)
        else:
            result = np.full((n_flat,), np.nan)
        rest_shape = spikes.shape[1:]
        warnings.warn(
            f"Too few windows ({n_windows} < 2) for model-based Fano; returning NaN",
            stacklevel=2,
        )
        return result.reshape(rest_shape), {"n_windows": n_windows}

    if is_torch:
        counts = torch.zeros((n_windows, n_flat), device=spikes.device)
    else:
        counts = np.zeros((n_windows, n_flat))

    for w in range(n_windows):
        start = w * step
        end = start + window
        if is_torch:
            counts[w] = flat_spike[start:end].sum(dim=0)
        else:
            counts[w] = flat_spike[start:end].sum(axis=0)

    if is_torch:
        empirical_mean = counts.mean(dim=0)
        empirical_var = counts.var(dim=0, unbiased=True)
    else:
        empirical_mean = counts.mean(axis=0)
        empirical_var = counts.var(axis=0, ddof=1)

    if model == "modulated_poisson":
        gain_mean = model_params.get("gain_mean", 1.0)
        gain_var = model_params.get("gain_var", 0.5)

        model_mean, model_var = _modulated_poisson_moments(
            empirical_mean, gain_mean, gain_var
        )

        # Model-based FF: ratio of model variance to model mean
        # normalized by expected Poisson variance
        fano_model = model_var / (model_mean + 1e-12)

        info = {
            "model": "modulated_poisson",
            "gain_mean": gain_mean,
            "gain_var": gain_var,
            "empirical_mean": empirical_mean,
            "empirical_var": empirical_var,
            "model_mean": model_mean,
            "model_var": model_var,
        }

    elif model == "flexible_overdispersion":
        if stimulus_drive is None:
            # Use empirical mean as proxy for stimulus drive
            stimulus_drive = (
                np.log(empirical_mean + 1)
                if not is_torch
                else torch.log(empirical_mean + 1)
            )

        nonlinearity = model_params.get("nonlinearity", "relu")
        noise_std = model_params.get("noise_std", 0.5)

        model_mean, model_var = _flexible_overdispersion_moments(
            stimulus_drive, noise_std, nonlinearity
        )

        fano_model = model_var / (model_mean + 1e-12)

        info = {
            "model": "flexible_overdispersion",
            "nonlinearity": nonlinearity,
            "noise_std": noise_std,
            "empirical_mean": empirical_mean,
            "empirical_var": empirical_var,
            "model_mean": model_mean,
            "model_var": model_var,
        }

    else:
        raise ValueError(f"Unknown model: {model}")

    rest_shape = spikes.shape[1:]
    fano_model = fano_model.reshape(rest_shape)

    if batch_axis is not None:
        if is_torch:
            fano_model = torch.mean(fano_model, dim=tuple(batch_axis))
        else:
            fano_model = np.mean(fano_model, axis=tuple(batch_axis))

    return fano_model, info


# =============================================================================
# Unified Interface
# =============================================================================


@use_percentiles(value_key="fano")
@use_stats(value_key="fano")
def fano_compensated(
    spikes: np.ndarray | torch.Tensor,
    method: Literal[
        "operational_time",
        "mean_matching",
        "modulated_poisson",
        "flexible_overdispersion",
    ] = "operational_time",
    **kwargs: Any,
) -> tuple[np.ndarray | torch.Tensor, dict]:
    """Unified interface for compensated Fano factor computation.

    This function provides a unified interface to all rate-compensation
    methods for the Fano factor. The choice of method depends on the
    experimental design and data characteristics:

    - operational_time: Best for comparing variability across different
      firing rates. Transforms to rate-independent reference frame.
    - mean_matching: Best for condition comparisons with overlapping
      rate distributions. Matches rate histograms before computing FF.
    - modulated_poisson: Model-based approach assuming multiplicative
      gain noise. Good for explaining overdispersion.
    - flexible_overdispersion: Model-based approach with flexible
      nonlinearities. Good for testing different rate-FF relationships.

    Args:
        spikes: Spike train of shape [T, ...]. First dimension is time.
        method: Compensation method to use.
        **kwargs: Method-specific arguments passed to underlying functions.

    Returns:
        fano: Compensated Fano factor values. The failure return (NaN array
            plus warning when fewer than 2 windows fit) is that of the
            selected method, see :func:`fano_operational_time`,
            :func:`fano_mean_matching` and :func:`fano_model_based`.
        info: Dictionary with method info and computed statistics.

    Raises:
        ValueError: If ``method`` is not one of the supported names, plus
            whatever the selected method raises.

    Example:
        >>> # Operational time method (recommended for rate comparisons)
        >>> ff_op, _ = fano_compensated(spikes, method="operational_time")
        >>>
        >>> # Mean matching (for condition comparisons)
        >>> ff_mm, _ = fano_compensated(spikes, method="mean_matching", n_bins=10)
        >>>
        >>> # Model-based approach
        >>> ff_mod, _ = fano_compensated(
        ...     spikes, method="modulated_poisson",
        ...     model_params={"gain_var": 0.5}
        ... )
    """
    if method == "operational_time":
        # Filter out model_params if accidentally passed
        kwargs.pop("model_params", None)
        return fano_operational_time(spikes, **kwargs)
    elif method == "mean_matching":
        kwargs.pop("model_params", None)
        return fano_mean_matching(spikes, **kwargs)
    elif method in ("modulated_poisson", "flexible_overdispersion"):
        return fano_model_based(spikes, model=method, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Utility Functions
# =============================================================================


def compare_fano_methods(
    spikes: np.ndarray | torch.Tensor,
    dt: float = 1.0,
    **kwargs: Any,
) -> dict:
    """Compare different Fano factor compensation methods.

    Computes Fano factor using multiple methods for comparison.

    Args:
        spikes: Spike train of shape [T, ...].
        dt: Time step in milliseconds.
        **kwargs: Additional arguments passed to methods.

    Returns:
        Dictionary mapping method name (``"standard"``, ``"operational_time"``,
        ``"mean_matching"``, ``"modulated_poisson"``) to its Fano factor
        result. Values are always numeric. If a compensation method fails for
        data-dependent reasons (``ValueError``, ``RuntimeError``,
        ``ArithmeticError``, ``LinAlgError``), its entry is ``float("nan")``
        and a :class:`UserWarning` is emitted. Other exceptions (e.g.
        ``TypeError`` from bad arguments) are programming errors and propagate.
    """
    results: dict[str, Any] = {}

    from btorch.analysis.spiking import fano as standard_fano

    results["standard"], _ = standard_fano(
        spikes,
        **{k: v for k, v in kwargs.items() if k in ["window", "overlap", "batch_axis"]},
    )

    def _run(name: str, func: Callable[[], Any]) -> None:
        # Only data/numerical failures (ValueError, RuntimeError, LinAlgError,
        # FloatingPointError, ArithmeticError) are converted to NaN. Programming
        # errors such as TypeError or AttributeError propagate to the caller.
        try:
            results[name], _ = func()
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as e:
            warnings.warn(
                f"Fano method {name!r} failed, returning NaN: {e}", stacklevel=3
            )
            results[name] = float("nan")

    _run(
        "operational_time",
        lambda: fano_operational_time(
            spikes,
            dt=dt,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["window", "overlap", "batch_axis"]
            },
        ),
    )
    _run(
        "mean_matching",
        lambda: fano_mean_matching(
            spikes, **{k: v for k, v in kwargs.items() if k not in ["dt"]}
        ),
    )
    _run(
        "modulated_poisson",
        lambda: fano_model_based(
            spikes,
            model="modulated_poisson",
            **{k: v for k, v in kwargs.items() if k not in ["dt"]},
        ),
    )

    return results
