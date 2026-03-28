"""Statistical utilities for analysis.

This module provides statistical computation utilities that work with both
NumPy arrays and PyTorch tensors, with consistent APIs across both backends.

Key features:
    - Unified API for common statistics (mean, std, var, median, etc.)
    - Configurable NaN and Inf handling policies
    - Batch computation optimization (reuses mean/std for CV)
    - Decorators for adding aggregation and percentile support

Decorators:
    - `use_stats`: Adds `stat`, `stat_info`, `nan_policy`, `inf_policy` args
    - `use_percentiles`: Adds `percentiles` arg for computing percentiles

Policy options:
    - nan_policy: "skip" (default), "warn", "assert"
    - inf_policy: "propagate" (default), "skip", "warn", "assert"

Example:
    >>> from btorch.analysis.statistics import compute_stat, use_stats
    >>> import numpy as np
    >>> data = np.random.randn(100, 10)  # 100 samples, 10 neurons
    >>> compute_stat(data, "mean", dim=0)  # mean per neuron
    >>> @use_stats
    ... def compute_metric(x): return x.mean(axis=0), {}
    >>> compute_metric(data, stat="median")  # returns median instead
"""

import inspect
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, Literal

import numpy as np
import torch


StatChoice = Literal[
    "mean", "median", "max", "min", "std", "var", "argmax", "argmin", "cv"
]


def describe_array(array: np.ndarray):
    """Print descriptive statistics for a 1D array.

    Displays mean, median, std, min, max, and quartiles.

    Args:
        array: 1D NumPy array to describe.

    Example:
        >>> describe_array(np.random.randn(100))
        Mean: 0.05
        Median: 0.12
        ...
    """
    mean = np.mean(array)
    median = np.median(array)
    std_dev = np.std(array)
    min_val = np.min(array)
    max_val = np.max(array)
    q25 = np.percentile(array, 25)
    q50 = np.percentile(array, 50)  # This is the same as the median
    q75 = np.percentile(array, 75)

    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"25th Percentile (Q1): {q25}")
    print(f"50th Percentile (Q2/Median): {q50}")
    print(f"75th Percentile (Q3): {q75}")


def compute_log_hist(data, bins=1000, edge_pos: Literal["mid", "sep"] = "mid"):
    """Compute histogram with logarithmically-spaced bins.

    Useful for visualizing heavy-tailed distributions like synaptic weights
    or degree distributions.

    Args:
        data: Input data array (must be positive).
        bins: Number of histogram bins.
        edge_pos: Position to return for bin edges.
            "mid": Return bin centers (midpoints).
            "sep": Return bin separators (edges).

    Returns:
        Tuple of (hist, bin_edges) where hist is the count array and
        bin_edges are the positions (centers or edges based on edge_pos).

    Raises:
        ValueError: If data contains non-positive values (required for log scale).
    """
    bin_edges = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), num=bins)
    hist, edges = np.histogram(data, bins=bin_edges)
    if edge_pos == "mid":
        bin_edges = 0.5 * (edges[:-1] + edges[1:])
    return hist, bin_edges


def compute_percentiles(
    values: np.ndarray | torch.Tensor,
    percentiles: float | tuple[float, ...],
) -> dict[str, list[float] | tuple[float, ...]]:
    """Compute percentiles of values.

    Works with both numpy arrays and PyTorch tensors, preserving the input type.

    Args:
        values: Input array or tensor
        percentiles: Percentile level(s) in [0, 100] range (e.g., 50 for median)

    Returns:
        Dictionary with "levels" and "percentiles" keys
    """
    # Normalize percentiles to a tuple
    if isinstance(percentiles, (int, float)):
        levels = (float(percentiles),)
    else:
        levels = tuple(float(p) for p in percentiles)

    # Validate levels
    for p in levels:
        if not 0 <= p <= 100:
            raise ValueError(f"Percentile must be in [0, 100], got {p}")

    # Compute percentiles using native functions
    if isinstance(values, torch.Tensor):
        # Use torch.quantile for tensor inputs (preserves device/dtype)
        values_flat = values.flatten()
        if values_flat.dtype == torch.float16:
            values_flat = values_flat.to(torch.float32)
        # torch.quantile takes quantiles in [0, 1] range directly
        perc_values = torch.quantile(
            values_flat,
            torch.tensor([p / 100 for p in levels], device=values.device),
        ).tolist()
    else:
        # Use numpy for array inputs
        values_flat = np.asarray(values).flatten()
        perc_values = [np.percentile(values_flat, p) for p in levels]

    return {
        "levels": levels,
        "percentiles": perc_values,
    }


def compute_stat(
    values: np.ndarray | torch.Tensor,
    stat: StatChoice,
    *,
    nan_policy: Literal["skip", "warn", "assert"] = "skip",
    inf_policy: Literal["propagate", "skip", "warn", "assert"] = "propagate",
    dim: int | tuple[int, ...] | None = None,
) -> Any:
    """Compute a single statistic on an array or tensor.

    Works with both numpy arrays and PyTorch tensors, preserving the input type.

    Args:
        values: Input array or tensor
        stat: Statistic to compute
        nan_policy: How to handle NaN values:
            - "skip": Ignore NaN values (default)
            - "warn": Warn if NaN values found but continue
            - "assert": Raise error if NaN values found
        inf_policy: How to handle Inf values:
            - "propagate": Keep Inf values (default)
            - "skip": Ignore Inf values
            - "warn": Warn if Inf values found but continue
            - "assert": Raise error if Inf values found
        dim: Dimension(s) to aggregate over. If None, flattens all dimensions.

    Returns:
        Computed statistic value

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(100)
        >>> compute_stat(data, "mean")
        0.1
    """
    return _compute_stat(values, stat, nan_policy, inf_policy, dim)


def _compute_stat(
    values: np.ndarray | torch.Tensor,
    stat: str,
    nan_policy: str,
    inf_policy: str,
    dim: int | tuple[int, ...] | None,
) -> Any:
    """Internal implementation of compute_stat."""
    # Import warnings here to avoid issues with torch.jit
    import warnings

    is_tensor = isinstance(values, torch.Tensor)

    # Handle NaN values
    if nan_policy != "propagate":
        has_nan = (
            torch.isnan(values).any().item() if is_tensor else np.isnan(values).any()
        )
        if has_nan:
            if nan_policy == "assert":
                raise ValueError("NaN values found in input")
            elif nan_policy == "warn":
                warnings.warn("NaN values found in input", UserWarning)
            # "skip" - filter out NaN values
            if is_tensor:
                values = values[~torch.isnan(values)]
            else:
                values = values[~np.isnan(values)]

    # Handle Inf values
    if inf_policy != "propagate":
        has_inf = (
            torch.isinf(values).any().item() if is_tensor else np.isinf(values).any()
        )
        if has_inf:
            if inf_policy == "assert":
                raise ValueError("Inf values found in input")
            elif inf_policy == "warn":
                warnings.warn("Inf values found in input", UserWarning)
            # "skip" - filter out Inf values
            if is_tensor:
                values = values[~torch.isinf(values)]
            else:
                values = values[~np.isinf(values)]

    # Flatten if dim is None
    if dim is None:
        if is_tensor:
            values = values.flatten()
        else:
            values = values.flatten()
        dim = 0

    if stat == "mean":
        if is_tensor:
            return (
                values.mean(dim=dim).item()
                if isinstance(dim, int)
                else values.mean().item()
            )
        return values.mean(axis=dim)
    elif stat == "median":
        if is_tensor:
            return (
                values.median(dim=dim).values.item()
                if isinstance(dim, int)
                else values.median().item()
            )
        return np.median(values, axis=dim)
    elif stat == "max":
        if is_tensor:
            return (
                values.max(dim=dim).values.item()
                if isinstance(dim, int)
                else values.max().item()
            )
        return values.max(axis=dim)
    elif stat == "min":
        if is_tensor:
            return (
                values.min(dim=dim).values.item()
                if isinstance(dim, int)
                else values.min().item()
            )
        return values.min(axis=dim)
    elif stat == "std":
        if is_tensor:
            return (
                values.std(dim=dim).item()
                if isinstance(dim, int)
                else values.std().item()
            )
        return values.std(axis=dim)
    elif stat == "var":
        if is_tensor:
            return (
                values.var(dim=dim).item()
                if isinstance(dim, int)
                else values.var().item()
            )
        return values.var(axis=dim)
    elif stat == "argmax":
        if is_tensor:
            return (
                values.argmax(dim=dim).item()
                if isinstance(dim, int)
                else values.argmax().item()
            )
        return values.argmax(axis=dim)
    elif stat == "argmin":
        if is_tensor:
            return (
                values.argmin(dim=dim).item()
                if isinstance(dim, int)
                else values.argmin().item()
            )
        return values.argmin(axis=dim)
    elif stat == "cv":
        # Coefficient of variation = std / mean
        if is_tensor:
            mean_val = values.mean(dim=dim) if isinstance(dim, int) else values.mean()
            std_val = values.std(dim=dim) if isinstance(dim, int) else values.std()
            cv = std_val / (mean_val + 1e-10)
            return cv.item() if cv.numel() == 1 else cv
        mean_val = values.mean(axis=dim)
        std_val = values.std(axis=dim)
        return std_val / (mean_val + 1e-10)
    else:
        raise ValueError(f"Unknown stat: {stat}")


def _compute_stats_batch(
    values: np.ndarray | torch.Tensor,
    stats: list[str],
    nan_policy: str,
    inf_policy: str,
    dim: int | tuple[int, ...] | None,
) -> dict[str, Any]:
    """Compute multiple stats efficiently, reusing mean/std for cv.

    Returns dict mapping stat name to computed value.
    """
    # Import warnings here to avoid issues with torch.jit
    import warnings

    is_tensor = isinstance(values, torch.Tensor)

    # Handle NaN values
    if nan_policy != "propagate":
        has_nan = (
            torch.isnan(values).any().item() if is_tensor else np.isnan(values).any()
        )
        if has_nan:
            if nan_policy == "assert":
                raise ValueError("NaN values found in input")
            elif nan_policy == "warn":
                warnings.warn("NaN values found in input", UserWarning)
            if is_tensor:
                values = values[~torch.isnan(values)]
            else:
                values = values[~np.isnan(values)]

    # Handle Inf values
    if inf_policy != "propagate":
        has_inf = (
            torch.isinf(values).any().item() if is_tensor else np.isinf(values).any()
        )
        if has_inf:
            if inf_policy == "assert":
                raise ValueError("Inf values found in input")
            elif inf_policy == "warn":
                warnings.warn("Inf values found in input", UserWarning)
            if is_tensor:
                values = values[~torch.isinf(values)]
            else:
                values = values[~np.isinf(values)]

    # Flatten if dim is None
    if dim is None:
        if is_tensor:
            values = values.flatten()
        else:
            values = values.flatten()
        dim = 0

    results = {}

    # Pre-compute mean and std if needed (for cv optimization)
    needs_mean = any(s in stats for s in ["mean", "cv"])
    needs_std = "cv" in stats

    if needs_mean:
        if is_tensor:
            mean_val = values.mean(dim=dim) if isinstance(dim, int) else values.mean()
        else:
            mean_val = values.mean(axis=dim)

    if needs_std:
        if is_tensor:
            std_val = values.std(dim=dim) if isinstance(dim, int) else values.std()
        else:
            std_val = values.std(axis=dim)

    # Compute each stat
    for stat in stats:
        if stat == "mean":
            if is_tensor:
                results[stat] = mean_val.item() if mean_val.numel() == 1 else mean_val
            else:
                results[stat] = mean_val
        elif stat == "std":
            if is_tensor:
                val = values.std(dim=dim) if isinstance(dim, int) else values.std()
                results[stat] = val.item() if val.numel() == 1 else val
            else:
                results[stat] = values.std(axis=dim)
        elif stat == "var":
            if is_tensor:
                val = values.var(dim=dim) if isinstance(dim, int) else values.var()
                results[stat] = val.item() if val.numel() == 1 else val
            else:
                results[stat] = values.var(axis=dim)
        elif stat == "median":
            if is_tensor:
                val = (
                    values.median(dim=dim).values
                    if isinstance(dim, int)
                    else values.median()
                )
                results[stat] = val.item() if val.numel() == 1 else val
            else:
                results[stat] = np.median(values, axis=dim)
        elif stat == "max":
            if is_tensor:
                val = (
                    values.max(dim=dim).values if isinstance(dim, int) else values.max()
                )
                results[stat] = val.item() if val.numel() == 1 else val
            else:
                results[stat] = values.max(axis=dim)
        elif stat == "min":
            if is_tensor:
                val = (
                    values.min(dim=dim).values if isinstance(dim, int) else values.min()
                )
                results[stat] = val.item() if val.numel() == 1 else val
            else:
                results[stat] = values.min(axis=dim)
        elif stat == "argmax":
            if is_tensor:
                val = (
                    values.argmax(dim=dim) if isinstance(dim, int) else values.argmax()
                )
                results[stat] = val.item() if val.numel() == 1 else val
            else:
                results[stat] = values.argmax(axis=dim)
        elif stat == "argmin":
            if is_tensor:
                val = (
                    values.argmin(dim=dim) if isinstance(dim, int) else values.argmin()
                )
                results[stat] = val.item() if val.numel() == 1 else val
            else:
                results[stat] = values.argmin(axis=dim)
        elif stat == "cv":
            cv = std_val / (mean_val + 1e-10)
            if is_tensor:
                results[stat] = cv.item() if cv.numel() == 1 else cv
            else:
                results[stat] = cv

    return results


def compute_stats_batch(
    values: np.ndarray | torch.Tensor,
    stats: list[StatChoice],
    *,
    nan_policy: Literal["skip", "warn", "assert"] = "skip",
    inf_policy: Literal["propagate", "skip", "warn", "assert"] = "propagate",
    dim: int | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Compute multiple statistics efficiently on an array or tensor.

    This function optimizes computation by reusing mean and std calculations
    when computing cv (coefficient of variation).

    Works with both numpy arrays and PyTorch tensors, preserving the input type.

    Args:
        values: Input array or tensor
        stats: List of statistics to compute
        nan_policy: How to handle NaN values
        inf_policy: How to handle Inf values
        dim: Dimension(s) to aggregate over. If None, flattens all dimensions.

    Returns:
        Dict mapping each stat to its computed value

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(100)
        >>> compute_stats_batch(data, ["mean", "std", "cv"])
        {'mean': 0.1, 'std': 1.0, 'cv': 10.0}
    """
    return _compute_stats_batch(
        values, [str(s) for s in stats], nan_policy, inf_policy, dim
    )


def _unpack_result(
    result: Any, value_key: str | dict[int, str]
) -> tuple[tuple[Any, ...], dict]:
    """Unpack function result into values tuple and info dict.

    Handles the logic for detecting whether the last element of a tuple
    result is an info dict or an actual return value.

    When value_key is a dict, we know exactly how many values to expect
    (max position + 1). Only treat the last element as info if there's
    an extra element beyond what's expected.

    When value_key is a string, use the original heuristic: if the last
    element is a dict, treat it as info.

    Args:
        result: The return value from a decorated function
        value_key: The value_key from the decorator (str or dict)

    Returns:
        Tuple of (values_tuple, info_dict)
    """
    if isinstance(result, tuple):
        if isinstance(value_key, dict):
            # Dict value_key: only extract info if there's an extra element
            expected_values = max(value_key.keys()) + 1
            if len(result) > expected_values and isinstance(result[-1], dict):
                return result[:-1], result[-1]
            else:
                return result, {}
        else:
            # String value_key: original heuristic
            if len(result) >= 2 and isinstance(result[-1], dict):
                return result[:-1], result[-1]
            else:
                return result, {}
    else:
        return (result,), {}


def use_stats(
    func: Callable | None = None,
    *,
    value_key: str | dict[int, str] = "values",
    dim: int | tuple[int, ...] | dict[int, int | tuple[int, ...] | None] | None = None,
    default_stat: StatChoice | dict[int, StatChoice] | None = None,
    default_stat_info: (
        StatChoice
        | Iterable[StatChoice]
        | dict[int, StatChoice | Iterable[StatChoice]]
        | None
    ) = None,
    default_nan_policy: Literal["skip", "warn", "assert"] = "skip",
    default_inf_policy: Literal["propagate", "skip", "warn", "assert"] = "propagate",
) -> Callable:
    """Decorator to add stat and stat_info args for aggregation.

    This decorator adds `stat`, `stat_info`, `nan_policy`, and `inf_policy`
    parameters to a function that returns per-neuron values.

    - `stat`: If not None, returns the aggregated value instead of per-neuron
      values. The aggregation is stored in info[f"{value_key}_stat"].
      Can be a StatChoice, or a dict mapping return position to label
      (e.g., {1: "eci", 3: "lag"}) for functions returning multiple values.
    - `stat_info`: Additional stats to compute and store in info dict without
      affecting the return value. Can be a single StatChoice, Iterable of
      StatChoice, a dict mapping position to label(s), or None.
      If dict, format is {position: stat_or_stats} where stat_or_stats can be
      a single StatChoice or Iterable of StatChoice.
    - `dim`: Dimension(s) to aggregate over. Can be:
        - None: Flatten all dimensions (default)
        - int: Aggregate over this dimension for all outputs
        - tuple[int, ...]: Aggregate over these dimensions for all outputs
        - dict[int, int | tuple[int, ...] | None]: Different dim for each
          output position (e.g., {0: 1, 1: 2, 2: None, 3: (1, 3, 4)})
    - `nan_policy`: How to handle NaN values:
        - "skip": Ignore NaN values (default)
        - "warn": Warn if NaN values found but continue
        - "assert": Raise error if NaN values found
    - `inf_policy`: How to handle Inf values:
        - "propagate": Keep Inf values (default)
        - "skip": Ignore Inf values
        - "warn": Warn if Inf values found but continue
        - "assert": Raise error if Inf values found

    The decorated function should return either:
    - A tuple of (values, info_dict) where values are per-neuron metrics
    - Just the per-neuron values (will be wrapped in a tuple with empty dict)
    - A tuple of multiple values with info as the last element

    Args:
        func: The function to decorate (or None if using with parentheses)
        value_key: Key prefix to use in info dict for stat results
        dim: Dimension(s) to aggregate over for each output
        default_nan_policy: Default nan_policy for this decorated function
        default_inf_policy: Default inf_policy for this decorated function
        default_stat: Default stat for this decorated function

    Returns:
        Decorated function with added stat, stat_info, nan_policy, and
        inf_policy parameters

    Example:
        ```python
        @use_stat
        def compute_metric(
            data,
            *,
            stat=None,
            stat_info=None,
            nan_policy="skip",
            inf_policy="propagate",
        ):
            values = some_computation(data)  # per-neuron values
            return values, {"raw": values}

        # Usage:
        values, info = compute_metric(data)  # returns per-neuron values
        mean_val, info = compute_metric(data, stat="mean")  # returns aggregated
        values, info = compute_metric(
            data, stat_info=["mean", "max"]
        )  # extra stats in info

        # Multi-value return with dict stat:
        @use_stat
        def compute_multiple(data, *, stat=None, stat_info=None):
            eci = compute_eci(data)  # per-neuron
            lag = compute_lag(data)  # per-neuron
            return eci, lag, {}  # multiple values

        # Aggregate specific positions with dict stat:
        eci_mean, lag_mean, info = compute_multiple(
            data, stat={0: "eci", 1: "lag"}
        )
        ```
    """

    def decorator(f: Callable) -> Callable:
        # Inspect the wrapped function to determine what arguments it accepts
        sig = inspect.signature(f)
        f_accepts_nan_policy = "nan_policy" in sig.parameters
        f_accepts_inf_policy = "inf_policy" in sig.parameters

        @wraps(f)
        def wrapper(
            *args,
            stat: StatChoice | dict[int, StatChoice] | None = default_stat,
            stat_info: StatChoice
            | Iterable[StatChoice]
            | dict[int, StatChoice | Iterable[StatChoice]]
            | None = default_stat_info,
            nan_policy: Literal["skip", "warn", "assert"] | None = None,
            inf_policy: Literal["propagate", "skip", "warn", "assert"] | None = None,
            **kwargs,
        ) -> tuple[Any, ...]:
            # Use effective policies (passed value > decorator default > "skip")
            effective_nan_policy = (
                nan_policy if nan_policy is not None else default_nan_policy
            )
            effective_inf_policy = (
                inf_policy if inf_policy is not None else default_inf_policy
            )

            # Pass policies to the wrapped function if it accepts them
            if f_accepts_nan_policy:
                kwargs["nan_policy"] = effective_nan_policy
            if f_accepts_inf_policy:
                kwargs["inf_policy"] = effective_inf_policy

            # Call the original function
            result = f(*args, **kwargs)

            # Unpack result using shared helper
            values_tuple, info = _unpack_result(result, value_key)

            # Ensure info is a dict
            if info is None:
                info = {}

            updated_info = dict(info)

            # Helper to get value key name for a position
            def _get_value_key_name(pos: int) -> str:
                if isinstance(value_key, dict):
                    return value_key.get(pos, f"values{pos}")
                if len(values_tuple) > 1:
                    return f"{value_key}{pos}"
                return value_key

            # Helper to get values at a position
            def _get_values(pos: int) -> Any:
                if pos < 0 or pos >= len(values_tuple):
                    raise IndexError(
                        f"Position {pos} out of range for return tuple "
                        f"of length {len(values_tuple)}"
                    )
                return values_tuple[pos]

            # Helper to get effective dim for a position
            def _get_dim_for_pos(pos: int) -> int | tuple[int, ...] | None:
                if dim is None:
                    return None
                if isinstance(dim, dict):
                    return dim.get(pos, None)
                return dim

            # Handle stat parameter
            if stat is not None:
                # Check if stat is a dict mapping positions to stats
                if isinstance(stat, dict):
                    # Multiple position aggregation with dict stat
                    results = []
                    for pos, stat_choice in stat.items():
                        values = _get_values(pos)
                        key_name = _get_value_key_name(pos)
                        effective_dim = _get_dim_for_pos(pos)
                        stat_value = _compute_stat(
                            values,
                            stat_choice,
                            effective_nan_policy,
                            effective_inf_policy,
                            effective_dim,  # type: ignore
                        )
                        results.append(stat_value)
                        updated_info[key_name] = values
                        updated_info[f"{key_name}_{stat_choice}"] = stat_value
                    return tuple(results) + (updated_info,)
                else:
                    # Single stat - apply to position 0
                    values = _get_values(0)
                    key_name = _get_value_key_name(0)
                    effective_dim = _get_dim_for_pos(0)
                    stat_value = _compute_stat(
                        values,
                        stat,
                        effective_nan_policy,
                        effective_inf_policy,
                        effective_dim,  # type: ignore
                    )
                    updated_info[key_name] = values
                    updated_info[f"{key_name}_{stat}"] = stat_value
                    return stat_value, updated_info

            # Handle stat_info parameter
            if stat_info is not None:
                # Check if stat_info is a dict mapping positions to stats
                if isinstance(stat_info, dict):
                    # Dict format: {position: stat_or_stats}
                    for pos, stats in stat_info.items():
                        # Normalize to iterable
                        if isinstance(stats, str):
                            stats_list = [stats]
                        else:
                            stats_list = list(stats)

                        # Use batch computation for efficiency
                        values = _get_values(pos)
                        key_name = _get_value_key_name(pos)
                        effective_dim = _get_dim_for_pos(pos)
                        if len(stats_list) > 1:
                            batch_results = _compute_stats_batch(
                                values,
                                [str(s) for s in stats_list],
                                effective_nan_policy,
                                effective_inf_policy,
                                effective_dim,
                            )
                            for s in stats_list:
                                updated_info[f"{key_name}_{s}"] = batch_results[str(s)]
                        else:
                            # Single stat - no need for batch optimization
                            stat_value = _compute_stat(
                                values,
                                stats_list[0],
                                effective_nan_policy,
                                effective_inf_policy,
                                effective_dim,  # type: ignore
                            )
                            updated_info[f"{key_name}_{stats_list[0]}"] = stat_value
                else:
                    # Original format: apply to position 0
                    # Normalize to iterable
                    if isinstance(stat_info, str):
                        stat_info_list = [stat_info]
                    else:
                        stat_info_list = list(stat_info)

                    # Use batch computation for efficiency (reuses mean/std for cv)
                    values = _get_values(0)
                    key_name = _get_value_key_name(0)
                    effective_dim = _get_dim_for_pos(0)
                    if len(stat_info_list) > 1:
                        batch_results = _compute_stats_batch(
                            values,
                            [str(s) for s in stat_info_list],
                            effective_nan_policy,
                            effective_inf_policy,
                            effective_dim,
                        )
                        for s in stat_info_list:
                            updated_info[f"{key_name}_{s}"] = batch_results[str(s)]
                    else:
                        # Single stat - no need for batch optimization
                        stat_value = _compute_stat(
                            values,
                            stat_info_list[0],
                            effective_nan_policy,
                            effective_inf_policy,
                            effective_dim,  # type: ignore
                        )
                        updated_info[f"{key_name}_{stat_info_list[0]}"] = stat_value

                # Return original values with updated info
                if len(values_tuple) > 1:
                    return values_tuple + (updated_info,)
                else:
                    return (values_tuple[0], updated_info)

            # No stat or stat_info - return original values with info
            if len(values_tuple) > 1:
                return values_tuple + (updated_info,)
            else:
                return (values_tuple[0], updated_info)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


# TODO: compat with use_stats
#   the return value may become a scalar e.g. if use_stats(f)(stat="mean"),
#   need to get the original value from info
def use_percentiles(
    func: Callable | None = None,
    *,
    value_key: str | dict[int, str] = "values",
    default_percentiles: float | tuple[float, ...] | None = None,
) -> Callable:
    """Decorator to add percentiles arg and optionally compute percentiles.

    This decorator adds a `percentiles` parameter to a function that returns
    per-neuron values. Percentiles are only computed if percentiles is not None.
    Results are stored in info[f"{value_key}_percentile"].

    Can also accept a dict mapping return positions to labels for functions
    returning multiple values (e.g., {1: "eci", 3: "lag"}).

    The decorated function should return either:
    - A tuple of (values, info_dict) where values are per-neuron metrics
    - Just the per-neuron values (will be wrapped in a tuple with empty dict)
    - A tuple of multiple values with info as the last element

    Args:
        func: The function to decorate (or None if using with parentheses)
        value_key: Key to use in info dict for the percentile result

    Returns:
        Decorated function with added percentiles parameter

    Example:
        ```python
        @use_percentiles
        def compute_metric(data, *, percentiles=None):
            values = some_computation(data)  # per-neuron values
            return values, {"raw": values}

        # Usage:
        values, info = compute_metric(data)  # no percentiles computed
        values, info = compute_metric(data, percentiles=0.5)  # compute median
        values, info = compute_metric(
            data, percentiles=(0.25, 0.5, 0.75)
        )  # compute quartiles
        ```
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(
            *args,
            percentiles: float
            | tuple[float, ...]
            | dict[int, float | tuple[float, ...]]
            | None = default_percentiles,
            **kwargs,
        ) -> tuple[Any, ...]:
            # Call the original function
            result = f(*args, **kwargs)
            if percentiles is None:
                return result

            # Unpack result using shared helper
            values_tuple, info = _unpack_result(result, value_key)

            # Ensure info is a dict
            if info is None:
                info = {}

            updated_info = dict(info)

            # Helper to get value key name for a position
            def _get_value_key_name(pos: int) -> str:
                if isinstance(value_key, dict):
                    return value_key.get(pos, f"values{pos}")
                return f"{value_key}{pos}"

            # Helper to get values at a position
            def _get_values(pos: int) -> Any:
                if pos < 0 or pos >= len(values_tuple):
                    raise IndexError(
                        f"Position {pos} out of range for return tuple "
                        f"of length {len(values_tuple)}"
                    )
                return values_tuple[pos]

            # Compute percentiles only if requested
            if isinstance(percentiles, dict):
                # Dict format: {position: percentile_value(s)}
                # Allows different percentiles for different return values
                for pos, perc_value in percentiles.items():
                    values = _get_values(pos)
                    key_name = _get_value_key_name(pos)
                    perc_result = compute_percentiles(values, perc_value)
                    updated_info[f"{key_name}_percentiles"] = perc_result["percentiles"]
                    updated_info[f"{key_name}_levels"] = perc_result["levels"]
            elif isinstance(value_key, dict):
                # Dict value_key with single percentiles value:
                # Apply same percentiles to all positions in value_key
                for pos, label in value_key.items():
                    values = _get_values(pos)
                    perc_result = compute_percentiles(values, percentiles)
                    updated_info[f"{label}_percentiles"] = perc_result["percentiles"]
                    updated_info[f"{label}_levels"] = perc_result["levels"]
            else:
                # Single percentiles format - apply to position 0
                values = _get_values(0)
                perc_result = compute_percentiles(values, percentiles)
                updated_info[f"{value_key}_percentiles"] = perc_result["percentiles"]
                updated_info[f"{value_key}_levels"] = perc_result["levels"]

            if len(values_tuple) > 1:
                return values_tuple + (updated_info,)
            else:
                return values_tuple[0], updated_info

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
