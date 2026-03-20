"""E/I balance analysis tools for spiking neural networks."""

import numpy as np
import torch

from ..statistics import use_percentiles, use_stats


# TODO: handle multidim neuron axes and batch axes correctly, it is a mess currently


@use_percentiles(value_key="eci")
@use_stats(value_key="eci")
def compute_eci(
    I_e: torch.Tensor | np.ndarray,
    I_i: torch.Tensor | np.ndarray,
    *,
    I_ext: torch.Tensor | np.ndarray | None = None,
    batch_axis: tuple[int, ...] | int | None = None,
    eps: float = 1e-8,
    dtype: torch.dtype | np.dtype | None = None,
) -> torch.Tensor | np.ndarray:
    """Compute Excitatory-Inhibitory Cancellation Index (ECI).

    ECI measures the degree of cancellation between excitatory and inhibitory
    currents. ECI = 0 indicates perfect cancellation, ECI = 1 indicates no
    cancellation.

    Formula: ECI = |I_rec + I_ext| / (|I_e| + |I_i|)
    where I_rec = I_e + I_i

    This function is decorated with `@use_stats` and `@use_percentiles`.
    See [`use_stats()`](btorch/analysis/statistics.py:483) and
    [`use_percentiles()`](btorch/analysis/statistics.py:777) for detailed usage.

    Args:
        I_e: Excitatory current [T, ...] or [T, B, ...]
        I_i: Inhibitory current [T, ...] or [T, B, ...]. Note: assumed to be
            negative (inhibitory).
        I_ext: External current [T, ...] or [T, B, ...] (optional)
        batch_axis: Axes to aggregate over (e.g., trials). If None, averages
            over all non-time dimensions.
        eps: Small constant for numerical stability
        dtype: Data type for aggregation.
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
        eci: ECI values per neuron (shape depends on batch_axis).
            If `stat` is provided, returns the aggregated statistic instead.
        info: Dictionary with additional statistics and optional percentile data.
    """
    return _compute_eci(
        I_e, I_i, I_ext=I_ext, batch_axis=batch_axis, eps=eps, dtype=dtype
    )


def _compute_eci(
    I_e: torch.Tensor | np.ndarray,
    I_i: torch.Tensor | np.ndarray,
    *,
    I_ext: torch.Tensor | np.ndarray | None = None,
    batch_axis: tuple[int, ...] | int | None = None,
    eps: float = 1e-8,
    dtype: torch.dtype | np.dtype | None = None,
) -> torch.Tensor | np.ndarray:
    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)
    if isinstance(I_e, torch.Tensor):
        I_i = torch.as_tensor(I_i, dtype=I_e.dtype, device=I_e.device)
        I_ext = (
            torch.as_tensor(I_ext, dtype=I_e.dtype, device=I_e.device)
            if I_ext is not None
            else None
        )
        return _eci_torch(I_e, I_i, I_ext, batch_axis, eps, dtype)
    else:
        I_i = (
            I_i.numpy(force=True) if isinstance(I_i, torch.Tensor) else np.asarray(I_i)
        )
        if I_ext is not None:
            I_ext = (
                I_ext.numpy(force=True)
                if isinstance(I_ext, torch.Tensor)
                else np.asarray(I_ext)
            )
        return _eci_numpy(I_e, I_i, I_ext, batch_axis, eps, dtype)


def _eci_numpy(
    I_e: np.ndarray,
    I_i: np.ndarray,
    I_ext: np.ndarray | None,
    batch_axis: tuple[int, ...] | None,
    eps: float,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """NumPy implementation of ECI."""
    # Compute recurrent current
    I_rec = I_e + I_i

    # Total imbalance signal
    if I_ext is not None:
        Iext = np.asarray(I_ext)
        imbalance = I_rec + Iext
    else:
        imbalance = I_rec

    # Determine axes for aggregation
    if batch_axis is not None:
        # Average over specified batch axes
        axes = (0,) + tuple(batch_axis)
    else:
        # Average over all dimensions except last (neurons)
        axes = tuple(range(I_e.ndim - 1))

    # Compute ECI
    numer = np.abs(imbalance).mean(axis=axes, dtype=dtype)
    denom = (np.abs(I_e) + np.abs(I_i)).mean(axis=axes, dtype=dtype) + eps
    eci = numer / denom

    return eci


def _eci_torch(
    I_e: torch.Tensor,
    I_i: torch.Tensor,
    I_ext: torch.Tensor | None,
    batch_axis: tuple[int, ...] | None,
    eps: float,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Torch implementation of ECI with dtype handling."""
    with torch.inference_mode():
        # Compute recurrent current
        I_rec = I_e + I_i

        # Total imbalance signal
        if I_ext is not None:
            imbalance = I_rec + I_ext
        else:
            imbalance = I_rec

        # Determine dimensions for aggregation
        if batch_axis is not None:
            dims = (0,) + tuple(batch_axis)
        else:
            # Average over all dimensions except last
            dims = tuple(range(I_e.ndim - 1))

        # Compute ECI
        numer = torch.abs(imbalance).mean(dim=dims, dtype=dtype)
        denom = (torch.abs(I_e) + torch.abs(I_i)).mean(dim=dims, dtype=dtype) + eps
        eci = numer / denom

    return eci


@use_stats(value_key={0: "peak_corr", 1: "best_lag"})
@use_percentiles(value_key={0: "peak_corr", 1: "best_lag"})
def compute_lag_correlation(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    *,
    dt: float = 1.0,
    max_lag_ms: float = 30.0,
    batch_axis: tuple[int, ...] | int | None = None,
    use_fft: bool = True,
):
    """Compute lagged cross-correlation between two signals.

    Uses FFT-based correlation for efficiency. Returns correlation values
    and best lag per neuron.

    This function is decorated with `@use_stats` and `@use_percentiles`.
    See [`use_stats()`](btorch/analysis/statistics.py:483) and
    [`use_percentiles()`](btorch/analysis/statistics.py:777) for detailed usage.

    Args:
        x: First signal [T, ...] or [T, B, ...]
        y: Second signal [T, ...] or [T, B, ...]
        dt: Time step in ms
        max_lag_ms: Maximum lag for correlation in ms
        batch_axis: Axes to aggregate over (e.g., trials). If None, averages
            over all non-time dimensions.
        use_fft: If True, use FFT-based correlation (faster for long signals)
        stat: Aggregation statistic per return position. Can be a single stat
            or dict mapping position to stat (e.g., {0: "mean", 1: "median"}).
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        stat_info: Additional statistics to compute and store in info dict.
            Can be a single stat, iterable, or dict mapping position to stat(s).
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        nan_policy: How to handle NaN values ("skip", "warn", "assert").
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        inf_policy: How to handle Inf values ("propagate", "skip", "warn",
            "assert"). See [`use_stats()`](btorch/analysis/statistics.py:483).
        percentiles: Percentile level(s) in [0, 100] to compute per position.
            Can be a single value or dict mapping position to percentile(s).
            See [`use_percentiles()`](btorch/analysis/statistics.py:777).

    Returns:
        peak_corr: Correlation values at best lag per neuron.
            If `stat` is provided, returns aggregated value(s) instead.
        best_lag_ms: Best lag in ms per neuron.
            If `stat` is provided, returns aggregated value(s) instead.
        info: Dictionary with correlation over lags, best lags, etc.

    Example:
        # Get per-neuron values
        peak, lag, info = compute_lag_correlation(x, y)

        # Aggregate: max peak correlation, mean best lag
        peak_max, lag_mean, info = compute_lag_correlation(
            x, y, stat={0: "max", 1: "mean"}
        )
    """
    return _compute_lag_correlation(
        x, y, dt=dt, max_lag_ms=max_lag_ms, batch_axis=batch_axis, use_fft=use_fft
    )


def _compute_lag_correlation(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    *,
    dt: float = 1.0,
    max_lag_ms: float = 30.0,
    batch_axis: tuple[int, ...] | int | None = None,
    use_fft: bool = True,
):
    if isinstance(x, torch.Tensor):
        y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return _lag_corr_torch(x, y, dt, max_lag_ms, batch_axis, use_fft)
    else:
        y = y.numpy(force=True) if isinstance(y, torch.Tensor) else np.asarray(y)
        return _lag_corr_numpy(x, y, dt, max_lag_ms, batch_axis, use_fft)


def _lag_corr_numpy(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    max_lag_ms: float,
    batch_axis: tuple[int, ...] | int | None,
    use_fft: bool,
):
    """NumPy implementation of lagged correlation."""
    T = x.shape[0]
    lag_bins = int(max_lag_ms / dt)
    max_lag = min(lag_bins, T - 1)

    # Reshape to [T, -1] for correlation computation
    flat_x = x.reshape(T, -1)
    flat_y = y.reshape(T, -1)

    # Compute correlation over lags
    if use_fft and T > 100:
        corr_over_lags = _cross_correlation_fft_numpy(flat_x, flat_y, max_lag)
    else:
        corr_over_lags = _cross_correlation_direct_numpy(flat_x, flat_y, max_lag)

    # Aggregate over batch dimensions if specified
    shape_without_time = x.shape[1:]
    corr_reshaped = corr_over_lags.reshape(-1, *shape_without_time)
    if batch_axis is not None:
        corr_reshaped = corr_reshaped.mean(axis=batch_axis)

    # Find peak
    best_lag_idx = np.argmax(corr_reshaped, axis=0)
    peak_corr = corr_reshaped[best_lag_idx, np.arange(corr_reshaped.shape[1])]

    # Create lag values
    max_lag_actual = min(lag_bins, T - 1)
    best_lags = best_lag_idx - max_lag_actual  # Convert index to lag value
    best_lag_ms = best_lags * dt

    info = {
        "corr_over_lags": corr_reshaped,
        "lag_values_ms": np.arange(-max_lag_actual, max_lag_actual + 1) * dt,
    }

    return peak_corr, best_lag_ms, info


def _cross_correlation_fft_numpy(
    x: np.ndarray, y: np.ndarray, max_lag: int
) -> np.ndarray:
    """FFT-based cross-correlation for NumPy."""
    T, N = x.shape
    n_fft = 2 * T

    # Demean signals (required for proper correlation coefficient)
    x_demean = x - x.mean(axis=0, keepdims=True)
    y_demean = y - y.mean(axis=0, keepdims=True)

    # Compute FFT
    X = np.fft.rfft(x_demean, n=n_fft, axis=0)
    Y = np.fft.rfft(y_demean, n=n_fft, axis=0)

    # Cross-spectrum
    cross_spec = X * np.conj(Y)

    # Inverse FFT
    corr_full = np.fft.irfft(cross_spec, n=n_fft, axis=0)

    # Normalize to get correlation coefficient [-1, 1]
    x_std = x.std(axis=0, keepdims=True) + 1e-8
    y_std = y.std(axis=0, keepdims=True) + 1e-8

    corr_norm = corr_full / (x_std * y_std * T)

    # Extract valid lags
    neg_lags = corr_norm[-max_lag:, :]
    pos_lags = corr_norm[: max_lag + 1, :]
    corr_all = np.concatenate([neg_lags, pos_lags], axis=0)

    return corr_all


def _cross_correlation_direct_numpy(
    x: np.ndarray, y: np.ndarray, max_lag: int
) -> np.ndarray:
    """Direct cross-correlation for NumPy (simpler for short signals)."""
    T, N = x.shape

    # Normalize
    x_norm = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    y_norm = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)

    n_lags = 2 * max_lag + 1
    corr = np.zeros((n_lags, N))

    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag < 0:
            # x lags y: compare x[:lag] with y[-lag:]
            c = (x_norm[:lag] * y_norm[-lag:]).mean(axis=0)
        elif lag > 0:
            # x leads y: compare x[lag:] with y[:-lag]
            c = (x_norm[lag:] * y_norm[:-lag]).mean(axis=0)
        else:
            c = (x_norm * y_norm).mean(axis=0)
        corr[i, :] = c

    return corr


def _lag_corr_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    dt: float,
    max_lag_ms: float,
    batch_axis: tuple[int, ...] | None,
    use_fft: bool,
):
    """Torch implementation of lagged correlation."""
    T = x.shape[0]
    lag_bins = int(max_lag_ms / dt)
    max_lag = min(lag_bins, T - 1)

    with torch.inference_mode():
        # Flatten non-time dimensions
        flat_x = x.reshape(T, -1)
        flat_y = y.reshape(T, -1)

        # Compute correlation
        if use_fft and T > 100:
            corr_over_lags = _cross_correlation_fft_torch(flat_x, flat_y, max_lag)
        else:
            corr_over_lags = _cross_correlation_direct_torch(flat_x, flat_y, max_lag)

        # Reshape to original non-time shape
        orig_shape = x.shape[1:]
        corr_reshaped = corr_over_lags.reshape(-1, *orig_shape)

        # Aggregate over batch dimensions if specified
        if batch_axis is not None:
            dims = tuple(a + 1 for a in batch_axis)  # Offset for lag dimension
            corr_agg = corr_reshaped.mean(dim=dims)
        else:
            corr_agg = corr_reshaped

        # Find peak
        peak_corr, best_lag_idx = torch.max(corr_agg, dim=0)

        # Create lag values
        max_lag_actual = min(lag_bins, T - 1)
        best_lags = best_lag_idx - max_lag_actual  # Convert index to lag value
        best_lag_ms = best_lags * dt

    info = {
        "corr_over_lags": corr_agg,
        "lag_values_ms": torch.arange(-max_lag_actual, max_lag_actual + 1) * dt,
    }

    return peak_corr, best_lag_ms, info


def _cross_correlation_fft_torch(
    x: torch.Tensor, y: torch.Tensor, max_lag: int
) -> torch.Tensor:
    """FFT-based cross-correlation for Torch."""
    T, N = x.shape
    n_fft = 2 * T

    # Ensure float32 for FFT
    if x.dtype == torch.float16:
        x = x.float()
        y = y.float()

    # Demean signals (required for proper correlation coefficient)
    x_demean = x - x.mean(dim=0, keepdim=True)
    y_demean = y - y.mean(dim=0, keepdim=True)

    # Compute FFT
    X = torch.fft.rfft(x_demean, n=n_fft, dim=0)
    Y = torch.fft.rfft(y_demean, n=n_fft, dim=0)

    # Cross-spectrum
    cross_spec = X * Y.conj()

    # Inverse FFT
    corr_full = torch.fft.irfft(cross_spec, n=n_fft, dim=0)

    # Normalize to get correlation coefficient [-1, 1]
    x_std = x.std(dim=0, keepdim=True) + 1e-8
    y_std = y.std(dim=0, keepdim=True) + 1e-8

    corr_norm = corr_full / (x_std * y_std * T)

    # Extract valid lags
    max_lag_actual = min(max_lag, T - 1)
    neg_lags = corr_norm[-max_lag_actual:, :]
    pos_lags = corr_norm[: max_lag_actual + 1, :]
    corr_all = torch.cat([neg_lags, pos_lags], dim=0)

    return corr_all


def _cross_correlation_direct_torch(
    x: torch.Tensor, y: torch.Tensor, max_lag: int
) -> torch.Tensor:
    """Direct cross-correlation for Torch."""
    T, N = x.shape
    device = x.device

    # Normalize
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True) + 1e-8
    y_std = y.std(dim=0, keepdim=True) + 1e-8

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    max_lag_actual = min(max_lag, T - 1)
    n_lags = 2 * max_lag_actual + 1
    corr = torch.zeros(n_lags, N, device=device, dtype=x.dtype)

    for i, lag in enumerate(range(-max_lag_actual, max_lag_actual + 1)):
        if lag < 0:
            c = (x_norm[:lag] * y_norm[-lag:]).mean(dim=0)
        elif lag > 0:
            c = (x_norm[lag:] * y_norm[:-lag]).mean(dim=0)
        else:
            c = (x_norm * y_norm).mean(dim=0)
        corr[i, :] = c

    return corr


@use_percentiles(
    value_key={0: "eci", 1: "peak_corr", 2: "best_lag_ms"},
    default_percentiles=(10, 50, 90),
)
@use_stats(
    value_key={0: "eci", 1: "peak_corr", 2: "best_lag_ms"},
    default_stat_info={0: "mean", 1: "mean", 2: "mean"},
)
def compute_ei_balance(
    I_e: torch.Tensor | np.ndarray,
    I_i: torch.Tensor | np.ndarray,
    *,
    I_ext: torch.Tensor | np.ndarray | None = None,
    dt: float = 1.0,
    max_lag_ms: float = 8.0,
    batch_axis: tuple[int, ...] | int | None = None,
    use_fft: bool = True,
):
    """Compute full E/I balance metrics including ECI and lag correlation.

    Convenience function that combines compute_eci and compute_lag_correlation.

    This function is decorated with `@use_stats` and `@use_percentiles`.
    See [`use_stats()`](btorch/analysis/statistics.py:483) and
    [`use_percentiles()`](btorch/analysis/statistics.py:777) for detailed usage.

    Args:
        I_e: Excitatory current [T, ...] or [T, B, ...]
        I_i: Inhibitory current [T, ...] or [T, B, ...]. Note: assumed to be
            negative (inhibitory). The lag correlation is computed between
            I_e and -I_i.
        I_ext: External current [T, ...] or [T, B, ...] (optional)
        dt: Time step in ms
        max_lag_ms: Maximum lag for correlation in ms
        batch_axis: Axes to aggregate over (e.g., trials)
        use_fft: Use FFT-based correlation
        stat: Aggregation statistic per return position. Can be a single stat
            or dict mapping position to stat (e.g., {0: "mean", 1: "max"}).
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        stat_info: Additional statistics to compute and store in info dict.
            Can be a single stat, iterable, or dict mapping position to stat(s).
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        nan_policy: How to handle NaN values ("skip", "warn", "assert").
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        inf_policy: How to handle Inf values ("propagate", "skip", "warn",
            "assert"). See [`use_stats()`](btorch/analysis/statistics.py:483).
        percentiles: Percentile level(s) in [0, 100] to compute per position.
            Can be a single value or dict mapping position to percentile(s).
            Default is (10, 50, 90).
            See [`use_percentiles()`](btorch/analysis/statistics.py:777).

    Returns:
        eci: ECI values per neuron, or aggregated if `stat` provided.
        peak_corr: Peak correlation values, or aggregated if `stat` provided.
        best_lag_ms: Best lag in ms, or aggregated if `stat` provided.
        info: Dictionary with per-neuron values and correlation data.
    """
    # Compute ECI
    eci = _compute_eci(I_e, I_i, I_ext=I_ext, batch_axis=batch_axis)
    Ii_neg = -I_i

    peak_corr, best_lag_ms, info = _compute_lag_correlation(
        I_e,
        Ii_neg,
        dt=dt,
        max_lag_ms=max_lag_ms,
        batch_axis=batch_axis,
        use_fft=use_fft,
    )

    return eci, peak_corr, best_lag_ms, info
