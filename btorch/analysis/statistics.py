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
import warnings
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, Literal, Protocol, overload

import numpy as np
import torch


StatChoice = Literal[
    "mean", "median", "max", "min", "std", "var", "argmax", "argmin", "cv"
]
NanPolicy = Literal["skip", "warn", "assert"]
InfPolicy = Literal["propagate", "skip", "warn", "assert"]
DimSpec = int | tuple[int, ...] | dict[int, int | tuple[int, ...] | None] | None
BatchAxis = int | tuple[int, ...] | None
"""Axes (e.g. trials) a spike/current statistic aggregates over before it is
computed; an ``int`` is shorthand for a one-element tuple, ``None`` keeps all
non-time axes."""
StatsResult = tuple[Any, dict[str, Any]]
"""``(value, info)`` returned by a single-output function wrapped by
:func:`use_stats`/:func:`use_percentiles`: ``value`` is the per-element array
(or the aggregate if ``stat`` is given) and ``info`` the statistics dict."""
MultiStatsResult = tuple[Any, ...]
"""``(*values, info)`` returned by a multi-output function wrapped by
:func:`use_stats`/:func:`use_percentiles`: one entry per wrapped return
position followed by the ``info`` dict."""
StatSpec = StatChoice | dict[int, StatChoice] | None
StatInfoSpec = (
    StatChoice
    | Iterable[StatChoice]
    | dict[int, StatChoice | Iterable[StatChoice]]
    | None
)
PercentileSpec = float | tuple[float, ...] | dict[int, float | tuple[float, ...]] | None


def describe_array(array: np.ndarray, verbose: bool = True) -> dict[str, float]:
    """Compute descriptive statistics for an array.

    Computes mean, median, std, min, max, and quartiles, optionally printing
    them.

    Args:
        array: NumPy array to describe (statistics are over all elements).
        verbose: If True (default), also print the statistics.

    Returns:
        Dictionary with keys ``mean``, ``median``, ``std``, ``min``, ``max``,
        ``q25``, ``q50`` and ``q75``.

    Example:
        >>> stats = describe_array(np.random.randn(100), verbose=False)
        >>> sorted(stats)
        ['max', 'mean', 'median', 'min', 'q25', 'q50', 'q75', 'std']
    """
    q25, q50, q75 = np.percentile(array, [25, 50, 75])  # q50 equals the median
    stats = {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
    }

    if verbose:
        print(f"Mean: {stats['mean']}")
        print(f"Median: {stats['median']}")
        print(f"Standard Deviation: {stats['std']}")
        print(f"Min: {stats['min']}")
        print(f"Max: {stats['max']}")
        print(f"25th Percentile (Q1): {stats['q25']}")
        print(f"50th Percentile (Q2/Median): {stats['q50']}")
        print(f"75th Percentile (Q3): {stats['q75']}")
    return stats


def compute_log_hist(
    data: np.ndarray,
    bins: int = 1000,
    edge_pos: Literal["mid", "sep"] = "mid",
) -> tuple:
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

    Raises:
        ValueError: If any requested percentile is outside ``[0, 100]``.
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


_TORCH_REDUCERS: dict[str, Callable[[torch.Tensor, Any], torch.Tensor]] = {
    # Non-int ``dim`` (None was already flattened, tuples) reduces over all
    # elements, matching the historical behaviour of this module.
    "mean": lambda v, d: v.mean(dim=d) if isinstance(d, int) else v.mean(),
    "median": lambda v, d: v.median(dim=d).values if isinstance(d, int) else v.median(),
    "max": lambda v, d: v.max(dim=d).values if isinstance(d, int) else v.max(),
    "min": lambda v, d: v.min(dim=d).values if isinstance(d, int) else v.min(),
    "std": lambda v, d: v.std(dim=d) if isinstance(d, int) else v.std(),
    "var": lambda v, d: v.var(dim=d) if isinstance(d, int) else v.var(),
    "argmax": lambda v, d: v.argmax(dim=d) if isinstance(d, int) else v.argmax(),
    "argmin": lambda v, d: v.argmin(dim=d) if isinstance(d, int) else v.argmin(),
}
_NUMPY_REDUCERS: dict[str, Callable[[np.ndarray, Any], np.ndarray]] = {
    "mean": lambda v, d: v.mean(axis=d),
    "median": lambda v, d: np.median(v, axis=d),
    "max": lambda v, d: v.max(axis=d),
    "min": lambda v, d: v.min(axis=d),
    "std": lambda v, d: v.std(axis=d),
    "var": lambda v, d: v.var(axis=d),
    "argmax": lambda v, d: v.argmax(axis=d),
    "argmin": lambda v, d: v.argmin(axis=d),
}


def _apply_value_policy(
    values: np.ndarray | torch.Tensor,
    *,
    name: str,
    policy: str,
    detect: Callable[[Any], Any],
) -> np.ndarray | torch.Tensor:
    """Apply a NaN/Inf policy (``name`` is ``"NaN"`` or ``"Inf"``).

    ``detect`` is ``isnan``/``isinf`` of the matching backend. Offending
    elements are dropped for ``"skip"`` and ``"warn"`` (which also warns).
    """
    if policy == "propagate":
        return values
    bad = detect(values)
    if not bad.any():
        return values
    if policy == "assert":
        raise ValueError(f"{name} values found in input")
    if policy == "warn":
        warnings.warn(f"{name} values found in input", UserWarning, stacklevel=4)
    return values[~bad]


def compute_stats_batch(
    values: np.ndarray | torch.Tensor,
    stats: Iterable[StatChoice],
    *,
    nan_policy: NanPolicy = "skip",
    inf_policy: InfPolicy = "propagate",
    dim: int | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Compute multiple statistics efficiently on an array or tensor.

    This function optimizes computation by reusing mean and std calculations
    when computing cv (coefficient of variation).

    Works with both numpy arrays and PyTorch tensors, preserving the input type.
    Torch results with a single element become Python scalars; NumPy results
    are returned as produced by NumPy. Note that ``std``/``var``/``cv`` follow
    each backend's native convention (torch: unbiased, NumPy: population).

    Args:
        values: Input array or tensor
        stats: Statistics to compute
        nan_policy: How to handle NaN values
        inf_policy: How to handle Inf values
        dim: Dimension(s) to aggregate over. If None, flattens all dimensions.

    Returns:
        Dict mapping each stat to its computed value

    Raises:
        ValueError: If a stat is unknown or a policy is ``"assert"`` and the
            offending values are present.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(100)
        >>> compute_stats_batch(data, ["mean", "std", "cv"])
        {'mean': 0.1, 'std': 1.0, 'cv': 10.0}
    """
    stats = [str(s) for s in stats]
    for name in stats:
        if name != "cv" and name not in _NUMPY_REDUCERS:
            raise ValueError(f"Unknown stat: {name}")

    is_tensor = isinstance(values, torch.Tensor)
    nan_detect, inf_detect = (
        (torch.isnan, torch.isinf) if is_tensor else (np.isnan, np.isinf)
    )
    values = _apply_value_policy(
        values, name="NaN", policy=nan_policy, detect=nan_detect
    )
    values = _apply_value_policy(
        values, name="Inf", policy=inf_policy, detect=inf_detect
    )
    if dim is None:
        values = values.flatten()
        dim = 0
        if (values.numel() if is_tensor else values.size) == 0:
            # Nothing left to aggregate (e.g. fewer than two spikes): every
            # statistic is undefined, so report NaN instead of warning/raising.
            return {name: float("nan") for name in stats}

    reducers = _TORCH_REDUCERS if is_tensor else _NUMPY_REDUCERS
    cache: dict[str, Any] = {}

    def reduce(name: str) -> Any:
        # mean/std are shared between their own request and ``cv``.
        if name not in cache:
            cache[name] = reducers[name](values, dim)
        return cache[name]

    results: dict[str, Any] = {}
    for name in stats:
        value = (
            reduce("std") / (reduce("mean") + 1e-10) if name == "cv" else reduce(name)
        )
        if is_tensor and value.numel() == 1:
            value = value.item()
        results[name] = value
    return results


def compute_stat(
    values: np.ndarray | torch.Tensor,
    stat: StatChoice,
    *,
    nan_policy: NanPolicy = "skip",
    inf_policy: InfPolicy = "propagate",
    dim: int | tuple[int, ...] | None = None,
) -> Any:
    """Compute a single statistic on an array or tensor.

    Works with both numpy arrays and PyTorch tensors, preserving the input type.
    See :func:`compute_stats_batch` for the shared implementation.

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

    Raises:
        ValueError: If ``stat`` is unknown or a policy is ``"assert"`` and the
            offending values are present.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(100)
        >>> compute_stat(data, "mean")
        0.1
    """
    return compute_stats_batch(
        values, [stat], nan_policy=nan_policy, inf_policy=inf_policy, dim=dim
    )[stat]


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


class _Outputs:
    """Resolve per-position values, info keys and dims of a decorated
    result."""

    def __init__(
        self,
        values: tuple[Any, ...],
        value_key: str | dict[int, str],
        dim: DimSpec = None,
    ) -> None:
        self.values = values
        self.value_key = value_key
        self.dim = dim

    def get(self, pos: int) -> Any:
        if pos < 0 or pos >= len(self.values):
            raise IndexError(
                f"Position {pos} out of range for return tuple "
                f"of length {len(self.values)}"
            )
        return self.values[pos]

    def key(self, pos: int) -> str:
        """Info-key prefix of output ``pos`` used by :func:`use_stats`."""
        if isinstance(self.value_key, dict):
            return self.value_key.get(pos, f"values{pos}")
        if len(self.values) > 1:
            return f"{self.value_key}{pos}"
        return self.value_key

    def dim_for(self, pos: int) -> int | tuple[int, ...] | None:
        if isinstance(self.dim, dict):
            return self.dim.get(pos, None)
        return self.dim


def _percentile_key(value_key: str | dict[int, str], pos: int) -> str:
    """Info-key prefix of output ``pos`` used by :func:`use_percentiles`."""
    if isinstance(value_key, dict):
        return value_key.get(pos, f"values{pos}")
    return f"{value_key}{pos}"


class StatsDecorated(Protocol):
    """Signature of a function wrapped by :func:`use_stats`.

    The positional/keyword arguments of the wrapped function are forwarded
    unchanged; ``stat``, ``stat_info``, ``nan_policy`` and ``inf_policy`` are
    added. The call always returns ``(*values, info)``.
    """

    def __call__(
        self,
        *args: Any,
        stat: StatSpec = ...,
        stat_info: StatInfoSpec = ...,
        nan_policy: NanPolicy | None = ...,
        inf_policy: InfPolicy | None = ...,
        **kwargs: Any,
    ) -> tuple[Any, ...]: ...


class PercentilesDecorated(Protocol):
    """Signature of a function wrapped by :func:`use_percentiles`.

    The arguments of the wrapped function are forwarded unchanged and
    ``percentiles`` is added. The call returns ``(*values, info)``.
    """

    def __call__(
        self,
        *args: Any,
        percentiles: PercentileSpec = ...,
        **kwargs: Any,
    ) -> tuple[Any, ...]: ...


@overload
def use_stats(func: Callable[..., Any], /) -> StatsDecorated: ...


@overload
def use_stats(
    func: None = None,
    /,
    *,
    value_key: str | dict[int, str] = ...,
    dim: DimSpec = ...,
    default_stat: StatSpec = ...,
    default_stat_info: StatInfoSpec = ...,
    default_nan_policy: NanPolicy = ...,
    default_inf_policy: InfPolicy = ...,
) -> Callable[[Callable[..., Any]], StatsDecorated]: ...


def use_stats(
    func: Callable[..., Any] | None = None,
    /,
    *,
    value_key: str | dict[int, str] = "values",
    dim: DimSpec = None,
    default_stat: StatSpec = None,
    default_stat_info: StatInfoSpec = None,
    default_nan_policy: NanPolicy = "skip",
    default_inf_policy: InfPolicy = "propagate",
) -> StatsDecorated | Callable[[Callable[..., Any]], StatsDecorated]:
    """Decorator to add stat and stat_info args for aggregation.

    This decorator adds `stat`, `stat_info`, `nan_policy`, and `inf_policy`
    parameters to a function that returns per-neuron values. The decorated
    function always returns ``(*values, info)`` (see :class:`StatsDecorated`).

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
        default_stat_info: Default stat_info for this decorated function

    Returns:
        Decorated function with added stat, stat_info, nan_policy, and
        inf_policy parameters

    Example:
        ```python
        @use_stats
        def compute_metric(
            data,
            *,
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
        @use_stats
        def compute_multiple(data):
            eci = compute_eci(data)  # per-neuron
            lag = compute_lag(data)  # per-neuron
            return eci, lag, {}  # multiple values

        # Aggregate specific positions with dict stat:
        eci_mean, lag_mean, info = compute_multiple(
            data, stat={0: "eci", 1: "lag"}
        )
        ```
    """

    def decorator(f: Callable[..., Any]) -> StatsDecorated:
        # Inspect the wrapped function to determine what arguments it accepts
        sig = inspect.signature(f)
        f_accepts_nan_policy = "nan_policy" in sig.parameters
        f_accepts_inf_policy = "inf_policy" in sig.parameters

        @wraps(f)
        def wrapper(
            *args: Any,
            stat: StatSpec = default_stat,
            stat_info: StatInfoSpec = default_stat_info,
            nan_policy: NanPolicy | None = None,
            inf_policy: InfPolicy | None = None,
            **kwargs: Any,
        ) -> tuple[Any, ...]:
            # Use effective policies (passed value > decorator default)
            nan_pol = nan_policy if nan_policy is not None else default_nan_policy
            inf_pol = inf_policy if inf_policy is not None else default_inf_policy
            policies = {"nan_policy": nan_pol, "inf_policy": inf_pol}

            # Pass policies to the wrapped function if it accepts them
            if f_accepts_nan_policy:
                kwargs["nan_policy"] = nan_pol
            if f_accepts_inf_policy:
                kwargs["inf_policy"] = inf_pol

            values_tuple, info = _unpack_result(f(*args, **kwargs), value_key)
            outputs = _Outputs(values_tuple, value_key, dim)
            updated_info = dict(info or {})

            if stat is not None:
                # A single stat applies to output 0; a dict maps position->stat.
                stat_map = stat if isinstance(stat, dict) else {0: stat}
                aggregated = []
                for pos, choice in stat_map.items():
                    values = outputs.get(pos)
                    key_name = outputs.key(pos)
                    stat_value = compute_stat(
                        values, choice, dim=outputs.dim_for(pos), **policies
                    )
                    aggregated.append(stat_value)
                    updated_info[key_name] = values
                    updated_info[f"{key_name}_{choice}"] = stat_value
                return (*aggregated, updated_info)

            if stat_info is not None:
                # A bare spec applies to output 0; a dict maps position->spec.
                info_map = stat_info if isinstance(stat_info, dict) else {0: stat_info}
                for pos, spec in info_map.items():
                    names = [spec] if isinstance(spec, str) else list(spec)
                    key_name = outputs.key(pos)
                    batch = compute_stats_batch(
                        outputs.get(pos), names, dim=outputs.dim_for(pos), **policies
                    )
                    for name in names:
                        updated_info[f"{key_name}_{name}"] = batch[str(name)]

            # Original values with the (possibly extended) info
            return (*values_tuple, updated_info)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


# Note on stacking with use_stats: use_percentiles works on whatever the wrapped
# callable returns. If it wraps use_stats and ``stat`` is set, that value is the
# already aggregated scalar (the per-neuron values are only kept in ``info``), so
# percentiles are then taken over the scalar rather than over neurons.
@overload
def use_percentiles(func: Callable[..., Any], /) -> PercentilesDecorated: ...


@overload
def use_percentiles(
    func: None = None,
    /,
    *,
    value_key: str | dict[int, str] = ...,
    default_percentiles: float | tuple[float, ...] | None = ...,
) -> Callable[[Callable[..., Any]], PercentilesDecorated]: ...


def use_percentiles(
    func: Callable[..., Any] | None = None,
    /,
    *,
    value_key: str | dict[int, str] = "values",
    default_percentiles: float | tuple[float, ...] | None = None,
) -> PercentilesDecorated | Callable[[Callable[..., Any]], PercentilesDecorated]:
    """Decorator to add percentiles arg and optionally compute percentiles.

    This decorator adds a `percentiles` parameter to a function that returns
    per-neuron values. Percentiles are only computed if percentiles is not None.
    Results are stored in info[f"{value_key}_percentiles"] and
    info[f"{value_key}_levels"]. The decorated function returns whatever the
    wrapped function returns when ``percentiles`` is None, otherwise
    ``(*values, info)`` (see :class:`PercentilesDecorated`).

    Can also accept a dict mapping return positions to labels for functions
    returning multiple values (e.g., {1: "eci", 3: "lag"}).

    The decorated function should return either:
    - A tuple of (values, info_dict) where values are per-neuron metrics
    - Just the per-neuron values (will be wrapped in a tuple with empty dict)
    - A tuple of multiple values with info as the last element

    Args:
        func: The function to decorate (or None if using with parentheses)
        value_key: Key to use in info dict for the percentile result
        default_percentiles: Percentiles used when the caller passes none.

    Returns:
        Decorated function with added percentiles parameter

    Example:
        ```python
        @use_percentiles
        def compute_metric(data):
            values = some_computation(data)  # per-neuron values
            return values, {"raw": values}

        # Usage:
        values, info = compute_metric(data)  # no percentiles computed
        values, info = compute_metric(data, percentiles=50)  # compute median
        values, info = compute_metric(
            data, percentiles=(25, 50, 75)
        )  # compute quartiles
        ```
    """

    def decorator(f: Callable[..., Any]) -> PercentilesDecorated:
        @wraps(f)
        def wrapper(
            *args: Any,
            percentiles: PercentileSpec = default_percentiles,
            **kwargs: Any,
        ) -> tuple[Any, ...]:
            result = f(*args, **kwargs)
            if percentiles is None:
                return result

            values_tuple, info = _unpack_result(result, value_key)
            outputs = _Outputs(values_tuple, value_key)
            updated_info = dict(info or {})

            # Resolve ``{position: (info key prefix, percentile levels)}``.
            if isinstance(percentiles, dict):
                # Different percentiles for different return values
                targets = {
                    pos: (_percentile_key(value_key, pos), levels)
                    for pos, levels in percentiles.items()
                }
            elif isinstance(value_key, dict):
                # One percentiles value applied to every labelled output
                targets = {
                    pos: (label, percentiles) for pos, label in value_key.items()
                }
            else:
                targets = {0: (value_key, percentiles)}

            for pos, (key_name, levels) in targets.items():
                perc = compute_percentiles(outputs.get(pos), levels)
                updated_info[f"{key_name}_percentiles"] = perc["percentiles"]
                updated_info[f"{key_name}_levels"] = perc["levels"]

            return (*values_tuple, updated_info)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
