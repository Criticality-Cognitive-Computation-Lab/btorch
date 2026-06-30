"""Branching-process simulation and MR branching-ratio estimation.

This module provides lightweight simulation helpers and a multiple-regression
(MR) estimator for branching ratio under subsampling.

Citation:
Wilting, J., & Priesemann, V. (2018). Inferring collective dynamical states
from widely unobserved systems. Nature Communications, 9(1), 2325.
https://doi.org/10.1038/s41467-018-04725-4
"""

from __future__ import annotations

import numpy as np


def simulate_branching(
    length: int = 10000,
    m: float = 0.9,
    activity: float = 100,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a branching process with immigration.

    Args:
        length: Number of time points.
        m: True branching ratio.
        activity: Target mean activity.
        rng: Seed or random generator.

    Returns:
        Integer count time series of shape ``(length,)``.
    """
    rng = np.random.default_rng(rng)
    h = activity * (1 - m)
    a_t = np.empty(length, dtype=np.int64)
    a_t[0] = rng.poisson(lam=activity)
    for t in range(1, length):
        tmp = rng.poisson(lam=h)
        if a_t[t - 1] > 0 and m > 0:
            tmp += rng.poisson(lam=m * a_t[t - 1])
        a_t[t] = tmp
    return a_t


def simulate_binomial_subsampling(
    a_t: np.ndarray,
    alpha: float,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Apply binomial subsampling to count data.

    Args:
        a_t: Integer activity time series.
        alpha: Sampling probability in ``[0, 1]``.
        rng: Seed or random generator.

    Returns:
        Subsampled counts as ``float`` for compatibility with existing tests.
    """
    rng = np.random.default_rng(rng)
    counts = np.asarray(a_t, dtype=np.int64)
    out = rng.binomial(counts, alpha)
    return out.astype(float)


def _to_1d_numeric_array(values: np.ndarray | list[float] | list[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Each trial must be a 1D numeric array")
    if arr.size == 0:
        raise ValueError("Empty trials are not supported")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains non-finite values")
    return arr


def input_handler(items: object) -> list[np.ndarray]:
    """Normalize supported input forms to list-of-trials arrays."""
    if isinstance(items, np.ndarray):
        if items.ndim == 1 and items.dtype.kind in "iuf":
            return [_to_1d_numeric_array(items)]
        if items.ndim == 2 and items.dtype.kind in "iuf":
            return [_to_1d_numeric_array(row) for row in items]
        raise ValueError("Unsupported numpy array input")
    if isinstance(items, list):
        if all(isinstance(x, (np.ndarray, list)) for x in items):
            return [_to_1d_numeric_array(x) for x in items]
        if all(isinstance(x, str) for x in items):
            return [_to_1d_numeric_array(np.load(x)) for x in items]
        raise ValueError("List of mixed types not supported")
    if isinstance(items, str):
        return [_to_1d_numeric_array(np.load(items))]
    raise ValueError("Input type not recognized")


def get_slopes(
    all_counts: list[np.ndarray],
    k_max: int,
    min_points: int = 50,
    scatterpoints: bool = False,
    eps: float = 1e-12,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    float,
    list[np.ndarray],
    list[np.ndarray],
]:
    """Compute lagged regression slopes ``r_k`` for ``k = 1 ..

    k_max-1``.
    """
    k_arr = np.arange(1, k_max, dtype=int)
    r_k = np.zeros(len(k_arr), dtype=np.float64)
    stderr = np.zeros(len(k_arr), dtype=np.float64)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for idx, k in enumerate(k_arr):
        x_concat = []
        y_concat = []
        for counts in all_counts:
            if counts.size <= k:
                continue
            x_block = counts[:-k]
            y_block = counts[k:]
            x_block = x_block - x_block.mean()
            y_block = y_block - y_block.mean()
            x_concat.append(x_block)
            y_concat.append(y_block)

        if not x_concat:
            r_k[idx] = np.nan
            stderr[idx] = np.nan
            continue

        x = np.concatenate(x_concat)
        y = np.concatenate(y_concat)
        n_samples = x.size

        if n_samples < min_points:
            r_k[idx] = np.nan
            stderr[idx] = np.nan
            continue

        cov = np.mean(x * y)
        var0 = np.mean(x * x)
        if var0 <= eps:
            r_k[idx] = np.nan
            stderr[idx] = np.nan
            continue

        slope = cov / var0
        prod = x * y
        stderr_prod = prod.std(ddof=1) / np.sqrt(n_samples)
        stderr_slope = stderr_prod / (var0 + eps)

        r_k[idx] = slope
        stderr[idx] = stderr_slope

        if scatterpoints:
            xs.append(x.copy())
            ys.append(y.copy())

    valid_idx = np.where(~np.isnan(r_k))[0]
    if valid_idx.size:
        first = valid_idx[0]
        data_length = int(np.sum([len(a) for a in all_counts]) - k_arr[first])
        mean_activity = float(np.mean([c.mean() for c in all_counts]))
    else:
        data_length = 0
        mean_activity = np.nan

    return k_arr, r_k, stderr, data_length, mean_activity, xs, ys


def _ar1_fallback(
    counts_list: list[np.ndarray], data_length: int, mean_activity: float, eps: float
) -> dict:
    x0 = []
    x1 = []
    for c in counts_list:
        if len(c) > 1:
            x0.append(c[:-1] - c[:-1].mean())
            x1.append(c[1:] - c[1:].mean())

    if not x0:
        raise RuntimeError("Insufficient data for AR(1) fallback")

    x0_cat = np.concatenate(x0)
    x1_cat = np.concatenate(x1)
    var0 = np.mean(x0_cat * x0_cat)
    if var0 <= eps:
        raise RuntimeError("Insufficient variance for AR(1) fallback")

    br_hat = float(np.mean(x0_cat * x1_cat) / var0)
    return {
        "branching_ratio": br_hat,
        "a_fit": np.nan,
        "autocorrelationtime": float(-1.0 / np.log(br_hat)) if br_hat > 0 else np.nan,
        "naive_branching_ratio": br_hat,
        "k": np.array([1], dtype=int),
        "r_k": np.array([br_hat], dtype=np.float64),
        "stderr": np.array([np.nan], dtype=np.float64),
        "fit_slope": np.nan,
        "fit_intercept": np.nan,
        "fit_points_used": 1,
        "data_length": data_length,
        "mean_activity": mean_activity,
        "note": "AR1 fallback used (insufficient valid MR points)",
    }


def branching_ratio(
    all_counts: np.ndarray | list[np.ndarray] | list[list[float]] | str,
    k_max: int = 40,
    *,
    maxslopes: int | None = None,
    scatterpoints: bool = False,
    eps: float = 1e-12,
    ar1_fallback: bool = True,
) -> dict:
    """Estimate branching ratio via subsampling-invariant MR fitting.

    Citation:
    Wilting, J., & Priesemann, V. (2018). Inferring collective dynamical states
    from widely unobserved systems. Nature Communications, 9(1), 2325.
    https://doi.org/10.1038/s41467-018-04725-4

    Args:
        all_counts: One trial, stacked trials, list of trials, or path input.
        k_max: Maximum lag used for the MR fit.
        maxslopes: Legacy synonym for ``k_max``.
        scatterpoints: Keep centered lagged ``x`` and ``y`` samples in result.
        eps: Small stabilizer for divisions and weights.
        ar1_fallback: Use AR(1) fallback when MR fit is ill-posed.

    Returns:
        Dictionary with ``branching_ratio``, ``naive_branching_ratio``, ``k``,
        ``r_k``, ``stderr`` and fit metadata.

    Raises:
        ValueError: If input data are invalid or too short.
        RuntimeError: If MR and fallback estimation both fail.
    """
    counts_list = input_handler(all_counts)

    if maxslopes is not None:
        k_max = maxslopes
    k_max = int(max(2, k_max))

    shortest = min(len(c) for c in counts_list)
    max_possible = min(k_max, shortest - 1)
    if max_possible < 1:
        raise ValueError("Time series too short for branching-ratio estimation")

    k, r_k_raw, stderr_raw, data_length, mean_activity, xs, ys = get_slopes(
        counts_list,
        max_possible + 1,
        scatterpoints=scatterpoints,
        eps=eps,
    )

    r_k = np.asarray(r_k_raw, dtype=np.float64)
    stderr = np.asarray(stderr_raw, dtype=np.float64)

    valid = np.isfinite(r_k) & np.isfinite(stderr) & (r_k > 0) & (stderr > 0)
    if np.sum(valid) < 2:
        if ar1_fallback:
            result = _ar1_fallback(counts_list, data_length, mean_activity, eps)
            if scatterpoints:
                result["xs"] = xs
                result["ys"] = ys
            return result
        raise RuntimeError("Insufficient valid r_k values for MR estimation")

    k_valid = k[valid].astype(np.float64)
    r_valid = np.clip(r_k[valid], 1e-12, 1e12)
    stderr_valid = stderr[valid]

    y = np.log(r_valid)
    w = 1.0 / (stderr_valid + eps)
    w = w / np.mean(w)

    slope, intercept = np.polyfit(k_valid, y, 1, w=w)
    m_hat = float(np.exp(slope))
    a_fit = float(np.exp(intercept))

    if not np.isfinite(m_hat) or m_hat <= 0:
        raise RuntimeError("Fitted branching ratio is non-positive or invalid")

    result = {
        "branching_ratio": m_hat,
        "a_fit": a_fit,
        "autocorrelationtime": float(-1.0 / np.log(m_hat))
        if (m_hat > 0 and m_hat != 1.0)
        else np.inf,
        "naive_branching_ratio": float(r_k[0]) if len(r_k) > 0 else np.nan,
        "k": k,
        "r_k": r_k,
        "stderr": stderr,
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "fit_points_used": int(np.sum(valid)),
        "data_length": data_length,
        "mean_activity": mean_activity,
    }
    if scatterpoints:
        result["xs"] = xs
        result["ys"] = ys
    return result
