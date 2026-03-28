"""Spike train analysis utilities for computing various metrics.

This module provides functions for analyzing spike train data, including:

- ISI (inter-spike interval) statistics: CV, local variation
- Fano factor (spike count variability)
- Kurtosis (spike count distribution shape)
- Population-level metrics (pooled across neurons)
- Temporal windowed analysis
- Firing rate smoothing

All functions support both NumPy arrays and PyTorch tensors, with
GPU-optimized implementations where applicable. The time dimension is
assumed to be the first dimension (axis=0) for all inputs.

Shape conventions:
    - Input spike trains: [T, ...] where T is time steps, remaining
      dimensions can be arbitrary (neurons, trials, batches, etc.)
    - Per-neuron outputs: [...] (same as input without time dimension)
    - Temporal outputs: [n_windows, ...] for sliding window functions

Batch axis behavior:
    - Most functions support `batch_axis` parameter for aggregating
      across specific dimensions (e.g., trials) before computing metrics
    - Aggregation is typically done via sum (for spike counts) or mean

The module uses decorators from `statistics.py` to add optional
aggregation (`stat`), additional statistics (`stat_info`), and
percentile computation (`percentiles`) to many functions.
"""

from collections.abc import Sequence

import numpy as np
import torch
from scipy.ndimage import convolve1d

from .statistics import use_percentiles, use_stats


# =============================================================================
# Internal Helper Functions (without percentile handling)
# =============================================================================


def _cv_numpy(
    spike_data: np.ndarray,
    dt_ms: float,
    batch_axis: tuple | None,
    dtype: np.dtype | None = None,
):
    """NumPy implementation of ISI CV."""
    orig_shape = spike_data.shape
    T = orig_shape[0]

    # Aggregate across batch dimensions if specified
    if batch_axis is not None:
        axes_to_sum = tuple(batch_axis)
        spike_aggregated = np.sum(spike_data, axis=axes_to_sum, keepdims=False)
        work_shape = (T,) + spike_aggregated.shape[1:]
    else:
        spike_aggregated = spike_data
        work_shape = orig_shape

    flat_data = spike_aggregated.reshape(T, -1)
    n_flat = flat_data.shape[1]

    # Vectorized spike extraction
    t_idx, n_idx = np.where(flat_data > 0)
    n_spikes_all = np.bincount(n_idx, minlength=n_flat)

    # Sort by neuron index primarily, then time secondarily
    sort_order = np.lexsort((t_idx, n_idx))
    t_sorted = t_idx[sort_order]
    n_sorted = n_idx[sort_order]

    # Calculate all global ISIs
    diffs = np.diff(t_sorted) * dt_ms

    # Valid ISIs are those where the neuron index didn't change
    valid_mask = n_sorted[:-1] == n_sorted[1:]
    valid_isis = diffs[valid_mask]
    valid_n = n_sorted[:-1][valid_mask]

    # Fast aggregation for CV calculation
    count_isi = np.bincount(valid_n, minlength=n_flat)
    sum_isi = np.bincount(valid_n, weights=valid_isis, minlength=n_flat)
    sum_isi_sq = np.bincount(valid_n, weights=valid_isis**2, minlength=n_flat)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_isi_arr = sum_isi / count_isi
        var_isi_arr = (sum_isi_sq / count_isi) - mean_isi_arr**2
        std_isi_arr = np.sqrt(np.maximum(var_isi_arr, 0.0))
        cv_values_flat = std_isi_arr / mean_isi_arr

    # NaN out neurons with < 2 ISIs (need at least 2 spikes for 1 ISI),
    # and 2 ISI to compute cv
    cv_values_flat[count_isi < 2] = np.nan
    cv_values = cv_values_flat.reshape(work_shape[1:])
    return cv_values, {
        "spike_count": n_spikes_all,
        "isi_mean": mean_isi_arr,
        "isi_std": std_isi_arr,
        "isi_var": var_isi_arr,
    }


def _cv_torch(
    spike_data: torch.Tensor,
    dt_ms: float,
    batch_axis: tuple | None,
    dtype: torch.dtype | None = None,
):
    """Torch implementation of ISI CV with GPU optimization.

    Strategy: Aggregate on GPU if needed, transfer to CPU for ISI extraction
    (more efficient than pure GPU due to divergent control flow), then return.

    Note: Uses specified dtype or input dtype for accumulation.
    """
    device = spike_data.device
    orig_shape = spike_data.shape
    T = orig_shape[0]

    # Aggregate across batch dimensions if specified
    if batch_axis is not None:
        spike_aggregated = torch.sum(spike_data, dim=batch_axis, keepdim=False)
        work_shape = (T,) + spike_aggregated.shape[1:]
    else:
        spike_aggregated = spike_data
        work_shape = orig_shape

    # For ISI extraction, CPU is more efficient due to sequential nature
    spike_cpu = spike_aggregated.cpu()
    T = spike_cpu.shape[0]
    flat_data = spike_cpu.reshape(T, -1)
    n_flat = flat_data.shape[1]

    # Vectorized spike extraction on CPU
    t_idx, n_idx = torch.nonzero(flat_data, as_tuple=True)
    n_spikes_all = torch.bincount(n_idx, minlength=n_flat)

    # Sort by neuron, then time (torch.lexsort doesn't exist,
    #   use argsort on packed values)
    # Pack: neuron * (T+1) + time ensures sorting by neuron first, then time
    packed = n_idx * (T + 1) + t_idx
    sort_order = torch.argsort(packed, stable=True)
    t_sorted = t_idx[sort_order].float() * dt_ms
    n_sorted = n_idx[sort_order]

    # Calculate ISIs
    isis = torch.diff(t_sorted)
    valid_mask = n_sorted[:-1] == n_sorted[1:]
    valid_isis = isis[valid_mask]
    valid_n = n_sorted[:-1][valid_mask]

    # Aggregate per neuron
    count_isi = torch.bincount(valid_n, minlength=n_flat).float()
    sum_isi = torch.bincount(valid_n, weights=valid_isis, minlength=n_flat)
    sum_isi_sq = torch.bincount(valid_n, weights=valid_isis**2, minlength=n_flat)
    mean_isi_arr = sum_isi / (count_isi + 1e-12)
    var_isi_arr = (sum_isi_sq / (count_isi + 1e-12)) - mean_isi_arr**2
    std_isi_arr = torch.sqrt(torch.clamp(var_isi_arr, min=0.0))
    cv_values_flat = std_isi_arr / (mean_isi_arr + 1e-12)

    cv_values_flat[count_isi < 2] = float("nan")

    # Return to original device
    cv_values = cv_values_flat.reshape(work_shape[1:]).to(device)
    mean_isi_arr = mean_isi_arr.reshape(work_shape[1:]).to(device)
    std_isi_arr = std_isi_arr.reshape(work_shape[1:]).to(device)
    var_isi_arr = var_isi_arr.reshape(work_shape[1:]).to(device)
    n_spikes_all = n_spikes_all.reshape(work_shape[1:]).to(device)
    return cv_values, {
        "spike_count": n_spikes_all,
        "isi_mean": mean_isi_arr,
        "isi_std": std_isi_arr,
        "isi_var": var_isi_arr,
    }


def _fano_numpy(
    spike: np.ndarray,
    window: int | None,
    overlap: int,
    batch_axis: tuple | None,
    dtype: np.dtype | None = None,
):
    """NumPy implementation of Fano factor."""
    orig_shape = spike.shape
    T = orig_shape[0]

    if window is None:
        # Default window size to get ~10 bins for variance computation
        # Need at least 2 bins for valid variance with ddof=1
        window = max(1, T // 10)

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap

    # Average across batch dimensions if specified
    if batch_axis is not None:
        spike = np.mean(spike, axis=batch_axis, keepdims=False)

    flat_spike = spike.reshape(T, -1)
    n_flat = flat_spike.shape[1]

    # VECTORIZED WINDOWING via Cumulative Sum
    # dtype=None lets numpy use its default (float64)
    cumsum_spike = np.zeros((T + 1, n_flat), dtype=dtype)
    np.cumsum(flat_spike, axis=0, dtype=dtype, out=cumsum_spike[1:])

    t_starts = np.arange(0, T - window + 1, step)
    t_ends = t_starts + window
    counts = cumsum_spike[t_ends] - cumsum_spike[t_starts]

    mean_counts = counts.mean(axis=0)
    var_counts = counts.var(axis=0, ddof=1)

    fano = np.zeros(n_flat, dtype=dtype)
    valid = np.isfinite(mean_counts) & (mean_counts > 0) & np.isfinite(var_counts)
    if np.any(valid):
        fano[valid] = var_counts[valid] / mean_counts[valid]

    fano_result = fano.reshape(spike.shape[1:])
    return fano_result, {}


def _fano_torch(
    spike: torch.Tensor,
    window: int,
    overlap: int,
    batch_axis: tuple | None,
    dtype: torch.dtype | None = None,
):
    """Torch implementation of Fano factor (GPU-friendly).

    Note: Uses specified dtype or input dtype for accumulation.
    """
    device = spike.device
    orig_shape = spike.shape
    T = orig_shape[0]

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap

    # Cast to specified dtype or preserve input dtype for accumulation
    if dtype is not None:
        spike_work = spike.to(dtype)
    elif spike.dtype in (torch.float16, torch.bfloat16):
        spike_work = spike.float()
    else:
        spike_work = spike

    if batch_axis is not None:
        spike_work = torch.mean(spike_work, dim=batch_axis, keepdim=False)

    flat_spike = spike_work.reshape(T, -1)
    n_flat = flat_spike.shape[1]

    # Determine accumulator dtype
    accum_dtype = dtype if dtype is not None else spike_work.dtype

    # Vectorized windowing via cumulative sum (very GPU-friendly)
    cumsum_spike = torch.zeros(T + 1, n_flat, device=device, dtype=accum_dtype)
    cumsum_spike[1:] = torch.cumsum(flat_spike, dim=0)

    t_starts = torch.arange(0, T - window + 1, step, device=device)
    t_ends = t_starts + window
    counts = cumsum_spike[t_ends] - cumsum_spike[t_starts]

    mean_counts = counts.mean(dim=0)
    var_counts = counts.var(dim=0, unbiased=True)

    fano = torch.zeros(n_flat, device=device, dtype=accum_dtype)
    valid = torch.isfinite(mean_counts) & (mean_counts > 0) & torch.isfinite(var_counts)
    if valid.any():
        fano[valid] = var_counts[valid] / mean_counts[valid]

    # Use spike_work.shape which accounts for batch_axis aggregation
    fano_result = fano.reshape(spike_work.shape[1:])
    return fano_result, {}


def _kurtosis_numpy(
    spike: np.ndarray,
    window: int,
    overlap: int,
    fisher: bool,
    batch_axis: tuple | None,
    dtype: np.dtype | None = None,
):
    """NumPy implementation of kurtosis."""
    orig_shape = spike.shape
    T = orig_shape[0]

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap

    # Average across batch dimensions if specified
    if batch_axis is not None:
        spike = np.mean(spike, axis=batch_axis, keepdims=False)

    flat_spike = spike.reshape(T, -1)
    n_flat = flat_spike.shape[1]

    # VECTORIZED WINDOWING via Cumulative Sum
    # dtype=None lets numpy use its default (float64)
    cumsum_spike = np.zeros((T + 1, n_flat), dtype=dtype)
    np.cumsum(flat_spike, axis=0, dtype=dtype, out=cumsum_spike[1:])

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

    # Use spike.shape which accounts for batch_axis aggregation (modified in place)
    kurt_result = kurt.reshape(spike.shape[1:])
    return kurt_result, {}


def _kurtosis_torch(
    spike: torch.Tensor,
    window: int,
    overlap: int,
    fisher: bool,
    batch_axis: tuple | None,
):
    """Torch implementation of kurtosis (GPU-friendly)."""
    device = spike.device
    orig_shape = spike.shape
    T = orig_shape[0]

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap

    # Cast to float64 for accumulation accuracy
    if spike.dtype in (torch.float16, torch.float32):
        spike_work = spike.double()
    else:
        spike_work = spike

    if batch_axis is not None:
        spike_work = torch.mean(spike_work, dim=tuple(batch_axis), keepdim=False)

    flat_spike = spike_work.reshape(T, -1)
    n_flat = flat_spike.shape[1]

    # Vectorized windowing via cumulative sum
    cumsum_spike = torch.zeros(T + 1, n_flat, device=device, dtype=torch.float64)
    cumsum_spike[1:] = torch.cumsum(flat_spike, dim=0)

    t_starts = torch.arange(0, T - window + 1, step, device=device)
    t_ends = t_starts + window
    counts = cumsum_spike[t_ends] - cumsum_spike[t_starts]

    m1 = counts.mean(dim=0)
    m2 = counts.var(dim=0, unbiased=True)
    m4 = torch.mean((counts - m1) ** 4, dim=0)

    eps = 1e-12
    kurt = m4 / (m2 + eps) ** 2

    if fisher:
        kurt = kurt - 3.0

    # Use spike_work.shape which accounts for batch_axis aggregation
    kurt_result = kurt.reshape(spike_work.shape[1:])
    return kurt_result, {}


# =============================================================================
# Per-Neuron Metrics with @use_stat and @use_percentiles
# =============================================================================


def _cv(
    spike_data: np.ndarray | torch.Tensor,
    dt_ms: float = 1.0,
    batch_axis: tuple[int, ...] | None = None,
    dtype: np.dtype | torch.dtype | None = None,
):
    if isinstance(spike_data, torch.Tensor):
        return _cv_torch(spike_data, dt_ms, batch_axis, dtype)
    else:
        return _cv_numpy(spike_data, dt_ms, batch_axis, dtype)


@use_percentiles(value_key="cv")
@use_stats(value_key="cv")
def isi_cv(
    spike_data: np.ndarray | torch.Tensor,
    dt_ms: float = 1.0,
    batch_axis: tuple[int, ...] | int | None = None,
    dtype: np.dtype | torch.dtype | None = None,
):
    """Calculate coefficient of variation of ISIs per neuron.

    Supports both NumPy and PyTorch inputs. For GPU tensors, uses a hybrid
    approach: aggregates on GPU, transfers to CPU for ISI extraction, then
    returns to GPU.

    This function is decorated with `@use_stats` and `@use_percentiles`.
    See [`use_stats()`](btorch/analysis/statistics.py:483) and
    [`use_percentiles()`](btorch/analysis/statistics.py:777) for detailed usage.

    Args:
        spike_data: Spike train array of shape [T, ...]. First dimension is time.
            Values are binary (0/1) or spike counts.
        dt_ms: Time step in milliseconds for converting ISI to ms.
        batch_axis: Axes to aggregate ISIs across (e.g., (1, 2) for trials).
            If None, computes CV per element in the non-time dimensions.
        dtype: Data type for accumulation.
        stat: Aggregation statistic to return instead of per-neuron values.
            Options: "mean", "median", "max", "min", "std", "var", "argmax",
            "argmin", "cv". See [`use_stats()`](btorch/analysis/statistics.py:483).
        stat_info: Additional statistics to compute and store in info dict.
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        nan_policy: How to handle NaN values ("skip", "warn", "assert").
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        inf_policy: How to handle Inf values ("propagate", "skip", "warn",
            "assert"). See [`use_stats()`](btorch/analysis/statistics.py:483).
        percentiles: Percentile level(s) in [0, 100] to compute.
            See [`use_percentiles()`](btorch/analysis/statistics.py:777).

    Returns:
        cv_values: CV values reshaped to match input without time dimension.
            Shape: [...] (original shape without T, aggregated over batch_axis).
            If `stat` is provided, returns the aggregated statistic instead.
        info: Dictionary with 'isi_total' (aggregate ISI statistics),
            'isi_stats' (per-neuron statistics), and optional percentile data.
    """
    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)
    return _cv(spike_data, dt_ms, batch_axis, dtype)


@use_percentiles(value_key="fano")
@use_stats(value_key="fano")
def fano(
    spike: np.ndarray | torch.Tensor,
    window: int | None = None,
    overlap: int = 0,
    batch_axis: tuple[int, ...] | int | None = None,
    dtype: np.dtype | torch.dtype | None = None,
):
    """Compute Fano factor for spike trains using optimized cumulative sums.

    Supports both NumPy and PyTorch inputs. GPU-friendly operation.

    This function is decorated with `@use_stats` and `@use_percentiles`.
    See [`use_stats()`](btorch/analysis/statistics.py:483) and
    [`use_percentiles()`](btorch/analysis/statistics.py:777) for detailed usage.

    Args:
        spike: Spike train of shape [T, ...]. First dimension is time.
        window: Window size for spike counting. If None, uses full duration T.
        overlap: Overlap between consecutive windows.
        batch_axis: Axes to average across for FF computation (e.g., trials).
        dtype: Data type for accumulation. If None, uses float64 for NumPy
            and input dtype (or float32 for float16/bfloat16) for Torch.
        stat: Aggregation statistic to return instead of per-neuron values.
            Options: "mean", "median", "max", "min", "std", "var", "argmax",
            "argmin", "cv". See [`use_stats()`](btorch/analysis/statistics.py:483).
        stat_info: Additional statistics to compute and store in info dict.
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        nan_policy: How to handle NaN values ("skip", "warn", "assert").
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        inf_policy: How to handle Inf values ("propagate", "skip", "warn",
            "assert"). See [`use_stats()`](btorch/analysis/statistics.py:483).
        percentiles: Percentile level(s) in [0, 100] to compute.
            See [`use_percentiles()`](btorch/analysis/statistics.py:777).

    Returns:
        fano: Fano factor values with shape [...]
            (input shape without time dimension).
            If `stat` is provided, returns the aggregated statistic instead.
        info: Dictionary with optional computed statistics and percentile data.
    """
    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)
    if window is None:
        # Default window size to get ~10 bins for variance computation
        # Need at least 2 bins for valid variance with unbiased=True
        window = max(1, spike.shape[0] // 10)
    if isinstance(spike, torch.Tensor):
        return _fano_torch(spike, window, overlap, batch_axis, dtype)
    else:
        return _fano_numpy(spike, window, overlap, batch_axis, dtype)


@use_percentiles(value_key="kurtosis")
@use_stats(value_key="kurtosis")
def kurtosis(
    spike: np.ndarray | torch.Tensor,
    window: int | None = None,
    overlap: int = 0,
    fisher: bool = True,
    batch_axis: tuple[int, ...] | int | None = None,
):
    """Compute kurtosis of spike counts using optimized cumulative sums.

    Supports both NumPy and PyTorch inputs. GPU-friendly operation.

    This function is decorated with `@use_stats` and `@use_percentiles`.
    See [`use_stats()`](btorch/analysis/statistics.py:483) and
    [`use_percentiles()`](btorch/analysis/statistics.py:777) for detailed usage.

    Args:
        spike: Spike train of shape [T, ...]. First dimension is time.
        window: Window size for spike counting. If None, uses full duration T.
        overlap: Overlap between consecutive windows.
        fisher: If True, return excess kurtosis (subtract 3).
        batch_axis: Axes to average across for kurtosis computation.
        stat: Aggregation statistic to return instead of per-neuron values.
            Options: "mean", "median", "max", "min", "std", "var", "argmax",
            "argmin", "cv". See [`use_stats()`](btorch/analysis/statistics.py:483).
        stat_info: Additional statistics to compute and store in info dict.
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        nan_policy: How to handle NaN values ("skip", "warn", "assert").
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        inf_policy: How to handle Inf values ("propagate", "skip", "warn",
            "assert"). See [`use_stats()`](btorch/analysis/statistics.py:483).
        percentiles: Percentile level(s) in [0, 100] to compute.
            See [`use_percentiles()`](btorch/analysis/statistics.py:777).

    Returns:
        kurt: Kurtosis values with shape [...]
            (input shape without time dimension).
            If `stat` is provided, returns the aggregated statistic instead.
        info: Dictionary with optional computed statistics and percentile data.
    """
    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)
    if window is None:
        # Default window size to get ~10 bins for variance computation
        # Need at least 2 bins for valid variance with unbiased=True
        window = max(1, spike.shape[0] // 10)
    if isinstance(spike, torch.Tensor):
        return _kurtosis_torch(spike, window, overlap, fisher, batch_axis)
    else:
        return _kurtosis_numpy(spike, window, overlap, fisher, batch_axis)


# =============================================================================
# Population-Level Metrics (pooled, only percentiles)
# =============================================================================


def _isis_population_numpy(spike_data: np.ndarray, dt_ms: float):
    """NumPy implementation of pooled CV across all neurons."""
    T = spike_data.shape[0]
    flat_data = spike_data.reshape(T, -1)

    # Extract all spike times
    t_idx, n_idx = np.where(flat_data > 0)

    # Sort by time only (pool across neurons)
    t_sorted = t_idx[np.argsort(t_idx)].astype(np.float64) * dt_ms

    if len(t_sorted) < 2:
        return np.array(np.nan), {}

    # Compute ISIs from pooled spikes
    isis = np.diff(t_sorted)
    return isis


def _isis_population_torch(spike_data: torch.Tensor, dt_ms: float):
    """Torch implementation of pooled CV across all neurons."""
    device = spike_data.device
    T = spike_data.shape[0]

    # Transfer to CPU for ISI extraction
    flat_data = spike_data.cpu().reshape(T, -1)

    # Extract all spike times
    t_idx = torch.nonzero(flat_data, as_tuple=True)[0]

    if len(t_idx) < 2:
        return torch.tensor(float("nan"), device=device), {}

    # Sort by time only (pool across neurons)
    t_sorted = t_idx.sort().values.float() * dt_ms

    # Compute ISIs from pooled spikes
    isis = torch.diff(t_sorted)

    return isis


@use_stats(value_key="isi_population", default_stat="cv")
def isi_cv_population(
    spike_data: np.ndarray | torch.Tensor,
    dt_ms: float = 1.0,
):
    """Calculate coefficient of variation of ISIs pooled across all neurons.

    This computes CV from the pooled ISI distribution across the entire
    population, giving a single population-level metric.

    This function is decorated with `@use_stats`.
    See [`use_stats()`](btorch/analysis/statistics.py:483) for detailed usage.

    Args:
        spike_data: Spike train array of shape [T, ...]. First dimension is time.
        dt_ms: Time step in milliseconds for converting ISI to ms.
        stat: Aggregation statistic to return. Default is "cv".
            Options: "mean", "median", "max", "min", "std", "var", "argmax",
            "argmin", "cv". See [`use_stats()`](btorch/analysis/statistics.py:483).
        stat_info: Additional statistics to compute and store in info dict.
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        nan_policy: How to handle NaN values ("skip", "warn", "assert").
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        inf_policy: How to handle Inf values ("propagate", "skip", "warn",
            "assert"). See [`use_stats()`](btorch/analysis/statistics.py:483).

    Returns:
        cv_pop: Single scalar CV value for the population, or aggregated
            statistic if `stat` is provided.
        info: Dictionary with computed statistics.
    """
    if isinstance(spike_data, torch.Tensor):
        return _isis_population_torch(spike_data, dt_ms)
    else:
        return _isis_population_numpy(spike_data, dt_ms)


def _fano_population_numpy(
    spike: np.ndarray, window: int | None, overlap: int, dtype: np.dtype | None = None
):
    """NumPy implementation of pooled Fano factor."""
    T = spike.shape[0]

    if window is None:
        window = T

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap

    # Pool across all non-time dimensions
    flat_spike = spike.reshape(T, -1)

    # Sum across all neurons to get population spike count
    pop_counts = flat_spike.sum(axis=1)

    # VECTORIZED WINDOWING via Cumulative Sum
    # dtype=None lets numpy use its default (float64)
    cumsum = np.zeros(T + 1, dtype=dtype)
    np.cumsum(pop_counts, dtype=dtype, out=cumsum[1:])

    t_starts = np.arange(0, T - window + 1, step)
    t_ends = t_starts + window
    counts = cumsum[t_ends] - cumsum[t_starts]

    mean_counts = counts.mean()
    var_counts = counts.var(ddof=1)

    fano_pop = var_counts / mean_counts if mean_counts > 0 else np.nan

    return np.array(fano_pop), {"window": window, "n_windows": len(counts)}


def _fano_population_torch(
    spike: torch.Tensor,
    window: int | None,
    overlap: int,
    dtype: torch.dtype | None = None,
):
    """Torch implementation of pooled Fano factor."""
    device = spike.device
    T = spike.shape[0]

    if window is None:
        window = T

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap

    # Pool across all non-time dimensions
    flat_spike = spike.reshape(T, -1)

    # Sum across all neurons to get population spike count
    pop_counts = flat_spike.sum(dim=1)

    # Use specified dtype or default to float64
    accum_dtype = dtype if dtype is not None else torch.float64

    # Vectorized windowing via cumulative sum
    cumsum = torch.zeros(T + 1, device=device, dtype=accum_dtype)
    cumsum[1:] = torch.cumsum(pop_counts, dim=0)

    t_starts = torch.arange(0, T - window + 1, step, device=device)
    t_ends = t_starts + window
    counts = cumsum[t_ends] - cumsum[t_starts]

    mean_counts = counts.mean().item()
    var_counts = counts.var(unbiased=True).item()

    fano_pop = var_counts / mean_counts if mean_counts > 0 else float("nan")

    return torch.tensor(fano_pop, device=device), {
        "window": window,
        "n_windows": len(counts),
    }


def fano_population(
    spike: np.ndarray | torch.Tensor,
    window: int | None = None,
    overlap: int = 0,
):
    """Compute Fano factor for the pooled population activity.

    This computes Fano factor from the summed population spike count,
    giving a single population-level metric.

    Args:
        spike: Spike train of shape [T, ...]. First dimension is time.
        window: Window size for spike counting. If None, uses full duration T.
        overlap: Overlap between consecutive windows.

    Returns:
        fano_pop: Single scalar Fano factor for the population.
        info: Dictionary with 'window' and 'n_windows'.
    """
    if isinstance(spike, torch.Tensor):
        return _fano_population_torch(spike, window, overlap)
    else:
        return _fano_population_numpy(spike, window, overlap)


def _kurtosis_population_numpy(
    spike: np.ndarray, window: int | None, overlap: int, fisher: bool
):
    """NumPy implementation of pooled kurtosis."""
    T = spike.shape[0]

    if window is None:
        window = T

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap

    # Pool across all non-time dimensions
    flat_spike = spike.reshape(T, -1)

    # Sum across all neurons to get population spike count
    pop_counts = flat_spike.sum(axis=1)

    # VECTORIZED WINDOWING via Cumulative Sum
    cumsum = np.zeros(T + 1, dtype=np.float64)
    np.cumsum(pop_counts, dtype=np.float64, out=cumsum[1:])

    t_starts = np.arange(0, T - window + 1, step)
    t_ends = t_starts + window
    counts = cumsum[t_ends] - cumsum[t_starts]

    m1 = counts.mean()
    m2 = counts.var(ddof=1)
    m4 = np.mean((counts - m1) ** 4)

    eps = 1e-12
    kurt = m4 / (m2 + eps) ** 2

    if fisher:
        kurt = kurt - 3.0

    return np.array(kurt)


def _kurtosis_population_torch(
    spike: torch.Tensor, window: int | None, overlap: int, fisher: bool
):
    """Torch implementation of pooled kurtosis."""
    device = spike.device
    T = spike.shape[0]

    if window is None:
        window = T

    assert 1 <= window <= T, "window must be in [1, T]"
    assert overlap < window, "overlap must be smaller than window"

    step = window - overlap

    # Pool across all non-time dimensions
    flat_spike = spike.reshape(T, -1)

    # Sum across all neurons to get population spike count
    pop_counts = flat_spike.sum(dim=1)

    # Vectorized windowing via cumulative sum
    cumsum = torch.zeros(T + 1, device=device, dtype=torch.float64)
    cumsum[1:] = torch.cumsum(pop_counts, dim=0)

    t_starts = torch.arange(0, T - window + 1, step, device=device)
    t_ends = t_starts + window
    counts = cumsum[t_ends] - cumsum[t_starts]

    m1 = counts.mean()
    m2 = counts.var(unbiased=True)
    m4 = torch.mean((counts - m1) ** 4)

    eps = 1e-12
    kurt = m4 / (m2 + eps) ** 2

    if fisher:
        kurt = kurt - 3.0

    kurt_val = kurt.item() if isinstance(kurt, torch.Tensor) else kurt
    return torch.tensor(kurt_val, device=device)


def kurtosis_population(
    spike: np.ndarray | torch.Tensor,
    window: int | None = None,
    overlap: int = 0,
    fisher: bool = True,
):
    """Compute kurtosis for the pooled population activity.

    This computes kurtosis from the summed population spike count,
    giving a single population-level metric.

    Args:
        spike: Spike train of shape [T, ...]. First dimension is time.
        window: Window size for spike counting. If None, uses full duration T.
        overlap: Overlap between consecutive windows.
        fisher: If True, return excess kurtosis (subtract 3).

    Returns:
        kurt_pop: Single scalar kurtosis for the population.
    """
    if isinstance(spike, torch.Tensor):
        return _kurtosis_population_torch(spike, window, overlap, fisher)
    else:
        return _kurtosis_population_numpy(spike, window, overlap, fisher)


# TODO: dim=1 means stat over neurons, should instead be [1:] to allow multidim
@use_stats(value_key="cv_temporal", dim=1)
def cv_temporal(
    spike_data: np.ndarray | torch.Tensor,
    dt_ms: float = 1.0,
    window: int = 100,
    step: int = 1,
    batch_axis: tuple[int, ...] | int | None = None,
    dtype: np.dtype | torch.dtype | None = None,
):
    """Compute CV in sliding temporal windows.

    Calculates the coefficient of variation of ISIs within sliding windows
    over time, giving a time-resolved measure of spike train irregularity.

    Args:
        spike_data: Spike train array of shape [T, ...]. First dimension is time.
        dt_ms: Time step in milliseconds.
        window: Size of the sliding window in time steps.
        step: Step size between consecutive windows.
        batch_axis: Axes to aggregate across (e.g., (1, 2) for trials).
        dtype: Data type for output arrays. If None, uses float64 for NumPy
            and float32 for Torch.

    Returns:
        cv_temporal: CV values for each window. Shape: [n_windows, ...]
            where n_windows = (T - window) // step + 1.
        info: Dictionary with window boundaries.
    """
    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)

    T = spike_data.shape[0]
    n_windows = (T - window) // step + 1

    # Determine output dtype
    if isinstance(spike_data, torch.Tensor):
        out_dtype = dtype if dtype is not None else torch.float32
        cv_values = torch.full(
            (n_windows,) + spike_data.shape[1:],
            float("nan"),
            dtype=out_dtype,
            device=spike_data.device,
        )
    else:
        out_dtype = dtype if dtype is not None else float
        cv_values = np.full(
            (n_windows,) + spike_data.shape[1:], np.nan, dtype=out_dtype
        )

    for i in range(n_windows):
        start = i * step
        end = start + window
        if end > T:
            break

        window_data = spike_data[start:end]

        cv_window, _ = _cv(window_data, dt_ms, batch_axis, dtype)
        cv_values[i] = cv_window

    window_starts = np.arange(n_windows) * step * dt_ms
    window_ends = window_starts + window * dt_ms

    info = {
        "window": window,
        "step": step,
        "window_starts_ms": window_starts,
        "window_ends_ms": window_ends,
    }
    return cv_values, info


@use_stats(value_key="fano_temporal", dim=1)
def fano_temporal(
    spike: np.ndarray | torch.Tensor,
    window: int = 100,
    step: int = 1,
    batch_axis: tuple[int, ...] | None = None,
):
    """Compute Fano factor in sliding temporal windows.

    Calculates the Fano factor within sliding windows over time,
    giving a time-resolved measure of spike count variability.

    Args:
        spike: Spike train of shape [T, ...]. First dimension is time.
        window: Size of the sliding window in time steps for Fano computation.
        step: Step size between consecutive windows.
        batch_axis: Axes to average across (e.g., trials).

    Returns:
        fano_temporal: Fano factor values for each window. Shape: [n_windows, ...]
        info: Dictionary with window boundaries.
    """
    T = spike.shape[0]
    n_windows = (T - window) // step + 1

    if isinstance(spike, torch.Tensor):
        fano_values = torch.full(
            (n_windows,) + spike.shape[1:],
            float("nan"),
            dtype=torch.float64,
            device=spike.device,
        )
    else:
        fano_values = np.full((n_windows,) + spike.shape[1:], np.nan, dtype=float)

    for i in range(n_windows):
        start = i * step
        end = start + window
        if end > T:
            break

        window_data = spike[start:end]
        bin = max(1, window_data.shape[0] // 10)

        if isinstance(window_data, torch.Tensor):
            fano_window, _ = _fano_torch(window_data, bin, 0, batch_axis)
            fano_values[i] = fano_window
        else:
            fano_window, _ = _fano_numpy(window_data, bin, 0, batch_axis)
            fano_values[i] = fano_window

    window_starts = np.arange(n_windows) * step
    window_ends = window_starts + window

    info = {
        "window": window,
        "step": step,
        "window_starts": window_starts,
        "window_ends": window_ends,
    }
    return fano_values, info


def fano_sweep(
    spike: np.ndarray | torch.Tensor,
    window: int | tuple[int, ...] | None = None,
    overlap: int = 0,
    batch_axis: tuple[int, ...] | int | None = None,
    dtype: np.dtype | torch.dtype | None = None,
):
    """Compute Fano factor sweeping over window sizes.

    This sweeps through window sizes and computes the Fano factor for each,
    useful for analyzing how variability scales with counting window size.

    The window parameter follows numpy.arange semantics:
        - window=10: windows from 1 to 10 (step=1)
        - window=(5, 20): windows from 5 to 20 (step=1)
        - window=(5, 20, 2): windows 5, 7, 9, ..., 19 (step=2)
        - window=None: defaults to range(1, T//20 + 1, 1)

    Args:
        spike: Spike train of shape [T, ...]. First dimension is time.
        window: Window size specification following arange convention:
            - int: stop value (start=1, step=1)
            - tuple (start, stop): range with step=1
            - tuple (start, stop, step): full range specification
            - None: auto-determine as (1, T//20, 1)
        overlap: Overlap between consecutive windows.
        batch_axis: Axes to average across (e.g., trials).
        dtype: Data type for output arrays. If None, uses float64 for NumPy
            and float32 for Torch.

    Returns:
        fano_sweep: Fano factor values for each window size.
            Shape: [n_windows, ...] where n_windows depends on range.
        info: Dictionary with 'window_sizes' array and 'window' spec.

    Examples:
        >>> # Sweep window sizes 1 to 50
        >>> fano_sweep(spike, window=50)
        >>> # Sweep window sizes 10, 20, 30, ..., 100
        >>> fano_sweep(spike, window=(10, 101, 10))
        >>> # Sweep window sizes 20, 30, 40, 50
        >>> fano_sweep(spike, window=(20, 51, 1))
    """
    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)
    T = spike.shape[0]

    # Parse window specification following arange semantics
    if window is None:
        start, stop, step = 1, T // 20 + 1, 1
    elif isinstance(window, int):
        start, stop, step = 1, window + 1, 1
    elif len(window) == 2:
        start, stop, step = window[0], window[1], 1
    elif len(window) == 3:
        start, stop, step = window[0], window[1], window[2]
    else:
        raise ValueError("window must be int, (start, stop), or (start, stop, step)")

    if step <= 0:
        raise ValueError("step must be positive")
    if start < 1:
        raise ValueError("start must be >= 1")
    if stop > T + 1:
        raise ValueError(f"stop must be <= T+1 ({T+1})")

    window_sizes = np.arange(start, stop, step)
    n_windows = len(window_sizes)

    if n_windows == 0:
        raise ValueError("window range produces no valid window sizes")

    if isinstance(spike, torch.Tensor):
        device = spike.device
        out = torch.zeros(
            (n_windows,) + spike.shape[1:], device=device, dtype=torch.float64
        )
        for i, w in enumerate(window_sizes):
            fano_val, _ = _fano_torch(spike, int(w), overlap, batch_axis)
            out[i] = fano_val
    else:
        out = np.zeros((n_windows,) + spike.shape[1:])
        for i, w in enumerate(window_sizes):
            fano_val, _ = _fano_numpy(spike, int(w), overlap, batch_axis)
            out[i] = fano_val

    info = {
        "window": (start, stop, step),
        "window_sizes": window_sizes,
        "n_windows": n_windows,
    }
    return out, info


# =============================================================================
# Local Variation (LV) - Keeping existing implementation
# =============================================================================


def _lv_numpy(
    spike_data: np.ndarray,
    dt_ms: float,
    batch_axis: tuple | None,
    dtype: np.dtype | None = None,
):
    """NumPy implementation of Local Variation."""
    orig_shape = spike_data.shape
    T = orig_shape[0]

    # Aggregate across batch dimensions if specified
    if batch_axis is not None:
        spike_aggregated = np.sum(spike_data, axis=batch_axis, keepdims=False)
        work_shape = (T,) + spike_aggregated.shape[1:]
    else:
        spike_aggregated = spike_data
        work_shape = orig_shape

    flat_data = spike_aggregated.reshape(T, -1)
    n_flat = flat_data.shape[1]

    # Extract spike times
    t_idx, n_idx = np.where(flat_data > 0)

    # Sort by neuron, then time
    sort_order = np.lexsort((t_idx, n_idx))
    t_sorted = t_idx[sort_order].astype(np.float64) * dt_ms
    n_sorted = n_idx[sort_order]

    # Compute ISIs
    isis = np.diff(t_sorted)

    # Valid consecutive ISI pairs (same neuron for 3 consecutive spikes)
    valid_mask = (n_sorted[:-2] == n_sorted[1:-1]) & (n_sorted[1:-1] == n_sorted[2:])

    isi_i = isis[:-1][valid_mask]
    isi_j = isis[1:][valid_mask]
    valid_n = n_sorted[:-2][valid_mask]

    # LV formula: 3*(ISI_i - ISI_{i+1})^2 / (ISI_i + ISI_{i+1})^2
    sum_isis = isi_i + isi_j
    diff_isis = isi_i - isi_j
    lv_terms = 3.0 * (diff_isis**2) / (sum_isis**2 + 1e-12)

    # Count valid pairs per neuron
    count_pairs = np.bincount(valid_n, minlength=n_flat)
    sum_lv = np.bincount(valid_n, weights=lv_terms, minlength=n_flat)

    with np.errstate(divide="ignore", invalid="ignore"):
        lv_values_flat = sum_lv / count_pairs

    # NaN for neurons with insufficient ISI pairs
    lv_values_flat[count_pairs < 1] = np.nan

    # Build stats
    lv_stats = {}
    for i in range(n_flat):
        if count_pairs[i] < 1:
            lv_stats[i] = {"n_pairs": 0, "lv": np.nan}
        else:
            lv_stats[i] = {
                "n_pairs": int(count_pairs[i]),
                "lv": float(lv_values_flat[i]),
            }

    lv_values = lv_values_flat.reshape(work_shape[1:])
    return lv_values, lv_stats


def _lv_torch(
    spike_data: torch.Tensor,
    dt_ms: float,
    batch_axis: tuple | None,
):
    """Torch implementation of Local Variation with GPU optimization."""
    device = spike_data.device
    orig_shape = spike_data.shape
    T = orig_shape[0]

    # Aggregate across batch dimensions if specified
    if batch_axis is not None:
        axes_to_sum = tuple(batch_axis)
        spike_aggregated = torch.sum(spike_data, dim=axes_to_sum, keepdim=False)
        work_shape = (T,) + spike_aggregated.shape[1:]
    else:
        spike_aggregated = spike_data
        work_shape = orig_shape

    # Transfer to CPU for ISI extraction
    spike_cpu = spike_aggregated.cpu()
    T = spike_cpu.shape[0]
    flat_data = spike_cpu.reshape(T, -1)
    n_flat = flat_data.shape[1]

    # Extract spike times
    t_idx, n_idx = torch.nonzero(flat_data, as_tuple=True)

    # Sort by neuron, then time (torch.lexsort doesn't exist,
    #   use argsort on packed values)
    packed = n_idx * (T + 1) + t_idx
    sort_order = torch.argsort(packed, stable=True)
    t_sorted = t_idx[sort_order].float() * dt_ms
    n_sorted = n_idx[sort_order]

    # Compute ISIs
    isis = torch.diff(t_sorted)

    # Valid consecutive ISI pairs (same neuron for 3 consecutive spikes)
    valid_mask = (n_sorted[:-2] == n_sorted[1:-1]) & (n_sorted[1:-1] == n_sorted[2:])

    isi_i = isis[:-1][valid_mask]
    isi_j = isis[1:][valid_mask]
    valid_n = n_sorted[:-2][valid_mask]

    # LV formula
    sum_isis = isi_i + isi_j
    diff_isis = isi_i - isi_j
    lv_terms = 3.0 * (diff_isis**2) / (sum_isis**2 + 1e-12)

    # Aggregate per neuron
    count_pairs = torch.bincount(valid_n, minlength=n_flat).float()
    sum_lv = torch.bincount(valid_n, weights=lv_terms, minlength=n_flat)

    with torch.no_grad():
        lv_values_flat = sum_lv / (count_pairs + 1e-12)

    lv_values_flat[count_pairs < 1] = float("nan")

    # Build stats
    lv_stats = {}
    for i in range(n_flat):
        if count_pairs[i].item() < 1:
            lv_stats[i] = {"n_pairs": 0, "lv": float("nan")}
        else:
            lv_stats[i] = {
                "n_pairs": int(count_pairs[i].item()),
                "lv": float(lv_values_flat[i].item()),
            }

    lv_values = lv_values_flat.reshape(work_shape[1:]).to(device)
    return lv_values, lv_stats


@use_percentiles(value_key="lv")
@use_stats(value_key="lv")
def local_variation(
    spike_data: np.ndarray | torch.Tensor,
    dt_ms: float = 1.0,
    batch_axis: tuple[int, ...] | None = None,
):
    """Calculate Local Variation (LV) of ISIs per neuron.

    LV is a measure of spike train irregularity that is less sensitive to
    slow rate fluctuations than CV. For a Poisson process, LV = 1.

    LV = (1/(N-1)) * sum(3*(ISI_i - ISI_{i+1})^2 / (ISI_i + ISI_{i+1})^2)

    Supports both NumPy and PyTorch inputs.

    Args:
        spike_data: Spike train array of shape [T, ...]. First dimension is time.
        dt_ms: Time step in milliseconds.
        batch_axis: Axes to aggregate ISIs across (e.g., (1, 2) for trials).

    Returns:
        lv_values: LV values reshaped to match input without time dimension.
        lv_stats: Dictionary with per-neuron LV statistics.
    """
    if isinstance(spike_data, torch.Tensor):
        return _lv_torch(spike_data, dt_ms, batch_axis)
    else:
        return _lv_numpy(spike_data, dt_ms, batch_axis)


# =============================================================================
# Utility Functions
# =============================================================================


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

    Args:
        spikes: Spike train array of shape [T, ...].
        width: Smoothing window width. If None or 0, no smoothing is applied.
        dt: Time step in milliseconds. If None, defaults to 1.0.
        axis: Axes to average over before smoothing. Can be int or tuple of ints.

    Returns:
        firing_rates: Smoothed firing rates with same shape as input (minus
            averaged axes if axis is specified).
    """
    if dt is None:
        dt = 1.0

    if axis is not None:
        # Normalize axis to tuple for consistent handling
        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(spikes, np.ndarray):
            spikes = spikes.mean(axis=axis)
        else:
            spikes = spikes.mean(dim=axis)

    if width is None or width == 0:
        return spikes / dt

    width1 = int(width // 2) * 2 + 1

    if isinstance(spikes, np.ndarray):
        if spikes.dtype == np.float16:
            spikes = spikes.astype(np.float32)
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
