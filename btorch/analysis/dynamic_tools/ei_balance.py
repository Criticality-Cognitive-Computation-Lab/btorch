"""E/I balance analysis tools for spiking neural networks."""

import numpy as np
import torch

from ..statistics import use_percentiles, use_stats


# TODO: handle multidim neuron axes and batch axes correctly, it is a mess currently


def _torch_dtype(dtype: torch.dtype | np.dtype | None) -> torch.dtype | None:
    if isinstance(dtype, torch.dtype) or dtype is None:
        return dtype
    return torch.from_numpy(np.empty((), dtype=dtype)).dtype


def _numpy_dtype(dtype: torch.dtype | np.dtype | None) -> np.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return np.dtype(str(dtype).removeprefix("torch."))
    return np.dtype(dtype)


@use_percentiles(value_key="eci")
@use_stats(value_key="eci")
def compute_eci(
    I_e: torch.Tensor | np.ndarray,
    I_i: torch.Tensor | np.ndarray,
    *,
    I_ext: torch.Tensor | np.ndarray | None = None,
    batch_axis: tuple[int, ...] | int | None = None,
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
        I_e: Excitatory current [T,..., N]
        I_i: Inhibitory current [T,..., N]. Note: assumed to be
            negative (inhibitory).
        I_ext: External current [T,..., N] (optional)
        batch_axis: Axes to aggregate over (e.g., trials) in addition to the time axis.
            If None, averages over all non-neuron dimensions.
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
    return _compute_eci(I_e, I_i, I_ext=I_ext, batch_axis=batch_axis, dtype=dtype)


def _compute_eci(
    I_e: torch.Tensor | np.ndarray,
    I_i: torch.Tensor | np.ndarray,
    *,
    I_ext: torch.Tensor | np.ndarray | None = None,
    batch_axis: tuple[int, ...] | int | None = None,
    dtype: torch.dtype | np.dtype | None = None,
) -> torch.Tensor | np.ndarray:
    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)

    # Check if all inputs are zero - return ones in that case
    if (I_e == 0).all() and (I_i == 0).all() and (I_ext is None or (I_ext == 0).all()):
        # Determine which axes are aggregated to compute output shape
        if batch_axis is not None:
            agg_axes = {0} | set(batch_axis)  # time (0) + batch axes
        else:
            agg_axes = set(range(I_e.ndim - 1))  # all except last (neurons)
        output_shape = tuple(I_e.shape[i] for i in range(I_e.ndim) if i not in agg_axes)
        if len(output_shape) == 0:
            output_shape = (1,)
        if isinstance(I_e, torch.Tensor):
            return torch.ones(output_shape, dtype=I_e.dtype, device=I_e.device)
        return np.ones(output_shape, dtype=I_e.dtype)

    # Handle I_ext by splitting into excitatory and inhibitory parts
    if I_ext is not None:
        if isinstance(I_e, torch.Tensor):
            I_ext_pos = torch.clamp(I_ext, min=0)
            I_ext_neg = torch.clamp(I_ext, max=0)
        else:
            I_ext_pos = np.clip(I_ext, 0, None)
            I_ext_neg = np.clip(I_ext, None, 0)
        I_e_eff = I_e + I_ext_pos
        I_i_eff = I_i + I_ext_neg
    else:
        I_e_eff = I_e
        I_i_eff = I_i

    # Compute recurrent current
    I_rec = I_e_eff + I_i_eff

    # Determine axes/dims for aggregation
    if batch_axis is not None:
        agg_dims = (0,) + tuple(batch_axis)
    else:
        agg_dims = tuple(range(I_e.ndim - 1))

    if isinstance(I_e, torch.Tensor):
        numer = torch.abs(I_rec).mean(dim=agg_dims, dtype=dtype)
        denom = (torch.abs(I_e_eff) + torch.abs(I_i_eff)).mean(
            dim=agg_dims, dtype=dtype
        )
        denom = denom + torch.finfo(denom.dtype).eps
        return numer / denom
    else:
        numer = np.abs(I_rec).mean(axis=agg_dims, dtype=dtype)
        denom = (np.abs(I_e_eff) + np.abs(I_i_eff)).mean(axis=agg_dims, dtype=dtype)
        denom = denom + np.finfo(denom.dtype).eps
        return numer / denom


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
    dtype: torch.dtype | np.dtype | None = None,
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
        dtype: Data type for correlation normalization/reduction. Use this to
            avoid float16 overflow in the variance/std normalization path.
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
        x,
        y,
        dt=dt,
        max_lag_ms=max_lag_ms,
        batch_axis=batch_axis,
        use_fft=use_fft,
        dtype=dtype,
    )


def _compute_lag_correlation(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    *,
    dt: float = 1.0,
    max_lag_ms: float = 30.0,
    batch_axis: tuple[int, ...] | int | None = None,
    use_fft: bool = True,
    dtype: torch.dtype | np.dtype | None = None,
):
    assert max_lag_ms >= dt
    if isinstance(x, torch.Tensor):
        y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
    else:
        y = y.numpy(force=True) if isinstance(y, torch.Tensor) else np.asarray(y)

    is_torch = isinstance(x, torch.Tensor)
    T = x.shape[0]
    n_neurons = x.shape[-1]
    lag_bins = int(max_lag_ms / dt)
    max_lag = min(lag_bins, T - 1)

    if isinstance(batch_axis, int):
        batch_axis = (batch_axis,)

    # Common reshape logic (works for both torch and numpy)
    x_3d = x.reshape(T, -1, n_neurons)
    y_3d = y.reshape(T, -1, n_neurons)
    flat_x = x_3d.reshape(T, -1)
    flat_y = y_3d.reshape(T, -1)

    if use_fft or T > 100:
        corr_flat = _cross_correlation_fft(flat_x, flat_y, max_lag, dtype)
    else:
        corr_flat = _cross_correlation_direct(flat_x, flat_y, max_lag, dtype)

    n_lags = corr_flat.shape[0]
    corr_3d = corr_flat.reshape(n_lags, -1, n_neurons)

    if is_torch:
        compute_dtype = _torch_dtype(dtype)
        if batch_axis is not None:
            corr_agg = corr_3d.mean(dim=1, dtype=compute_dtype)
        else:
            corr_agg = corr_3d.reshape(n_lags, -1)

        peak_corr, best_lag_idx = torch.max(corr_agg, dim=0)
        max_lag_actual = min(lag_bins, T - 1)
        best_lags = best_lag_idx - max_lag_actual
        best_lag_ms = best_lags * dt

        info = {
            "corr_over_lags": corr_agg,
            "lag_values_ms": torch.arange(
                -max_lag_actual, max_lag_actual + 1, device=x.device
            )
            * dt,
        }
        return peak_corr, best_lag_ms, info
    else:
        compute_dtype = _numpy_dtype(dtype)
        if batch_axis is not None:
            corr_agg = corr_3d.mean(axis=1, dtype=compute_dtype)
        else:
            corr_agg = corr_3d.reshape(n_lags, -1)

        best_lag_idx = np.argmax(corr_agg, axis=0)
        peak_corr = corr_agg[best_lag_idx, np.arange(corr_agg.shape[1])]
        max_lag_actual = min(lag_bins, T - 1)
        best_lags = best_lag_idx - max_lag_actual
        if compute_dtype is None:
            best_lag_ms = best_lags * dt
            lag_values_ms = np.arange(-max_lag_actual, max_lag_actual + 1) * dt
        else:
            best_lag_ms = best_lags.astype(compute_dtype, copy=False) * np.asarray(
                dt, dtype=compute_dtype
            )
            lag_values_ms = np.arange(
                -max_lag_actual, max_lag_actual + 1, dtype=compute_dtype
            ) * np.asarray(dt, dtype=compute_dtype)

        info = {
            "corr_over_lags": corr_agg,
            "lag_values_ms": lag_values_ms,
        }
        return peak_corr, best_lag_ms, info


def _cross_correlation_fft(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    max_lag: int,
    dtype: torch.dtype | np.dtype | None = None,
) -> torch.Tensor | np.ndarray:
    """FFT-based cross-correlation."""
    is_torch = isinstance(x, torch.Tensor)
    T, N = x.shape
    n_fft = 2 * T

    if is_torch:
        compute_dtype = _torch_dtype(dtype)
        fft_dtype = compute_dtype
        if fft_dtype is None and x.dtype in {torch.float16, torch.bfloat16}:
            fft_dtype = torch.float32
        if fft_dtype is not None:
            x = x.to(dtype=fft_dtype)
            y = y.to(dtype=fft_dtype)
        # Demean and compute FFT
        x_demean = x - x.mean(dim=0, keepdim=True, dtype=fft_dtype)
        y_demean = y - y.mean(dim=0, keepdim=True, dtype=fft_dtype)
        X = torch.fft.rfft(x_demean, n=n_fft, dim=0)
        Y = torch.fft.rfft(y_demean, n=n_fft, dim=0)
        cross_spec = X * Y.conj()
        corr_full = torch.fft.irfft(cross_spec, n=n_fft, dim=0)
        # Normalize
        x_std = x.std(dim=0, keepdim=True) + torch.finfo(x.dtype).eps
        y_std = y.std(dim=0, keepdim=True) + torch.finfo(y.dtype).eps
        corr_norm = corr_full / (x_std * y_std * T)
        # Extract valid lags
        max_lag_actual = min(max_lag, T - 1)
        neg_lags = corr_norm[-max_lag_actual:, :]
        pos_lags = corr_norm[: max_lag_actual + 1, :]
        return torch.cat([neg_lags, pos_lags], dim=0)
    else:
        compute_dtype = _numpy_dtype(dtype)
        if compute_dtype is not None:
            x = x.astype(compute_dtype, copy=False)
            y = y.astype(compute_dtype, copy=False)
        # Demean and compute FFT
        x_demean = x - x.mean(axis=0, keepdims=True, dtype=compute_dtype)
        y_demean = y - y.mean(axis=0, keepdims=True, dtype=compute_dtype)
        X = np.fft.rfft(x_demean, n=n_fft, axis=0)
        Y = np.fft.rfft(y_demean, n=n_fft, axis=0)
        cross_spec = X * np.conj(Y)
        corr_full = np.fft.irfft(cross_spec, n=n_fft, axis=0)
        # Normalize
        eps = np.finfo(x.dtype).eps
        x_std = x.std(axis=0, keepdims=True, dtype=compute_dtype) + eps
        y_std = y.std(axis=0, keepdims=True, dtype=compute_dtype) + eps
        corr_norm = corr_full / (x_std * y_std * T)
        if compute_dtype is not None:
            corr_norm = corr_norm.astype(compute_dtype, copy=False)
        # Extract valid lags
        neg_lags = corr_norm[-max_lag:, :]
        pos_lags = corr_norm[: max_lag + 1, :]
        return np.concatenate([neg_lags, pos_lags], axis=0)


def _cross_correlation_direct(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    max_lag: int,
    dtype: torch.dtype | np.dtype | None = None,
) -> torch.Tensor | np.ndarray:
    """Direct cross-correlation (simpler for short signals)."""
    is_torch = isinstance(x, torch.Tensor)
    T, N = x.shape

    if is_torch:
        device = x.device
        compute_dtype = _torch_dtype(dtype)
        if compute_dtype is not None:
            x = x.to(dtype=compute_dtype)
            y = y.to(dtype=compute_dtype)
        # Normalize
        x_mean = x.mean(dim=0, keepdim=True, dtype=compute_dtype)
        y_mean = y.mean(dim=0, keepdim=True, dtype=compute_dtype)
        x_std = x.std(dim=0, keepdim=True) + torch.finfo(x.dtype).eps
        y_std = y.std(dim=0, keepdim=True) + torch.finfo(y.dtype).eps
        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std
        # Compute correlations for each lag
        max_lag_actual = min(max_lag, T - 1)
        n_lags = 2 * max_lag_actual + 1
        corr = torch.zeros(n_lags, N, device=device, dtype=x.dtype)
        for i, lag in enumerate(range(-max_lag_actual, max_lag_actual + 1)):
            if lag < 0:
                c = (x_norm[:lag] * y_norm[-lag:]).mean(dim=0, dtype=compute_dtype)
            elif lag > 0:
                c = (x_norm[lag:] * y_norm[:-lag]).mean(dim=0, dtype=compute_dtype)
            else:
                c = (x_norm * y_norm).mean(dim=0, dtype=compute_dtype)
            corr[i, :] = c
        return corr
    else:
        compute_dtype = _numpy_dtype(dtype)
        if compute_dtype is not None:
            x = x.astype(compute_dtype, copy=False)
            y = y.astype(compute_dtype, copy=False)
        # Normalize
        x_norm = (x - x.mean(axis=0, dtype=compute_dtype)) / (
            x.std(axis=0, dtype=compute_dtype) + np.finfo(x.dtype).eps
        )
        y_norm = (y - y.mean(axis=0, dtype=compute_dtype)) / (
            y.std(axis=0, dtype=compute_dtype) + np.finfo(y.dtype).eps
        )
        # Compute correlations for each lag
        n_lags = 2 * max_lag + 1
        corr = []
        for i, lag in enumerate(range(-max_lag, max_lag + 1)):
            if lag < 0:
                c = (x_norm[:lag] * y_norm[-lag:]).mean(axis=0, dtype=compute_dtype)
            elif lag > 0:
                c = (x_norm[lag:] * y_norm[:-lag]).mean(axis=0, dtype=compute_dtype)
            else:
                c = (x_norm * y_norm).mean(axis=0, dtype=compute_dtype)
            corr.append(c)
        return np.stack(corr, axis=0)


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
    max_lag_ms: float = 30.0,
    batch_axis: tuple[int, ...] | int | None = None,
    dtype: torch.dtype | np.dtype | None = None,
):
    """Compute E/I balance metrics including ECI and lag correlation.

    This function is decorated with `@use_stats` and `@use_percentiles`.
    See [`use_stats()`](btorch/analysis/statistics.py:483) and
    [`use_percentiles()`](btorch/analysis/statistics.py:777) for detailed usage.

    Args:
        I_e: Excitatory current [T, ..., N]
        I_i: Inhibitory current [T, ..., N]
        I_ext: External current [T, ..., N] (optional)
        dt: Time step in ms
        max_lag_ms: Maximum lag for correlation analysis
        batch_axis: Axes to aggregate over (e.g., trials). If None, averages
            over all non-time dimensions.
        dtype: Data type for ECI aggregation and lag-correlation normalization.
        stat: Aggregation statistic per return position.
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        stat_info: Additional statistics per position.
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        nan_policy: How to handle NaN values ("skip", "warn", "assert").
            See [`use_stats()`](btorch/analysis/statistics.py:483).
        inf_policy: How to handle Inf values ("propagate", "skip", "warn",
            "assert"). See [`use_stats()`](btorch/analysis/statistics.py:483).
        percentiles: Percentile level(s) in [0, 100] to compute per position.
            See [`use_percentiles()`](btorch/analysis/statistics.py:777).

    Returns:
        eci: ECI values per neuron
        peak_corr: Peak correlation between E and I currents
        best_lag_ms: Best lag in ms (positive = I lags E)
        info: Dictionary with detailed analysis results
    """
    # Compute ECI
    eci, eci_info = compute_eci(
        I_e,
        I_i,
        I_ext=I_ext,
        batch_axis=batch_axis,
        dtype=dtype,
        stat=None,
    )

    # Compute lag correlation between E and I
    peak_corr, best_lag_ms, lag_info = compute_lag_correlation(
        I_e,
        -I_i,
        dt=dt,
        max_lag_ms=max_lag_ms,
        batch_axis=batch_axis,
        dtype=dtype,
        stat=None,
    )

    info = {
        "eci_info": eci_info,
        "lag_info": lag_info,
    }

    return eci, peak_corr, best_lag_ms, info
