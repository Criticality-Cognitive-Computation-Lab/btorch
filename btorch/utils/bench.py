"""Benchmarking utilities.

Performance measurement tools for PyTorch code, supporting both CPU
wall-clock and GPU event-based timing with warmup and statistical
summarization.
"""

import time
import warnings
from typing import Callable, Literal

import torch

from btorch.utils._optional import require


class PerfTimer:
    """Context manager for measuring execution time.

    Example:
        >>> with PerfTimer() as timer:
        ...     result = some_function()
        >>> print(f"Took {timer.elapsed_ms():.2f} ms")
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds.

        Returns:
            Elapsed time from ``__enter__`` to ``__exit__`` (or now
            if ``__exit__`` hasn't been called).

        Raises:
            RuntimeError: If timer was never started.
        """
        if self.start_time is None:
            raise RuntimeError("Timer never started")
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return (end - self.start_time) * 1000


def _resolve_timing_method(timing_method: str) -> str:
    """Validate ``timing_method`` and fall back to cpu if CUDA is missing."""
    if timing_method == "total":
        timing_method = "cpu"
    if timing_method not in ["gpu", "cpu"]:
        raise ValueError("timing_method must be either 'gpu' or 'cpu'")

    if timing_method == "gpu" and not torch.cuda.is_available():
        warnings.warn(
            "GPU timing requested but CUDA is not available. "
            "Falling back to cpu timing.",
            stacklevel=3,
        )
        return "cpu"
    return timing_method


def _resolve_budget(
    warmup: int | float, rep: int | float
) -> tuple[bool, int | float, int | float]:
    """Return ``(use_reps, warmup, rep)``; ints mean iterations, else ms."""
    use_reps = isinstance(warmup, int) and isinstance(rep, int)
    if use_reps:
        if warmup < 0 or rep < 1:
            raise ValueError("warmup must be >= 0 and rep must be >= 1")
        return True, warmup, rep
    warmup = float(warmup)
    rep = float(rep)
    if warmup < 0.0 or rep <= 0.0:
        raise ValueError("warmup and rep must be positive durations in ms")
    return False, warmup, rep


def _reset_grads(grad_to_none) -> None:
    if grad_to_none is not None:
        for x in grad_to_none:
            x.grad = None


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _warmup_cpu(fn: Callable, warmup: int | float, use_reps: bool) -> None:
    if use_reps:
        for _ in range(warmup):
            fn()
        return
    start = time.perf_counter()
    while _elapsed_ms(start) < warmup:
        fn()


def _time_once_cpu(fn: Callable, grad_to_none, sync_cuda: bool) -> float:
    """Time one call of ``fn`` in ms, optionally synchronizing CUDA."""
    sync = sync_cuda and torch.cuda.is_available()
    _reset_grads(grad_to_none)
    if sync:
        torch.cuda.synchronize()
    with PerfTimer() as timer:
        fn()
        if sync:
            torch.cuda.synchronize()
    return timer.elapsed_ms()


def _measure_cpu(
    fn: Callable,
    rep: int | float,
    use_reps: bool,
    grad_to_none,
    sync_cuda: bool,
) -> list[float]:
    """Collect wall-clock samples for ``rep`` iterations or ``rep`` ms."""
    if use_reps:
        return [_time_once_cpu(fn, grad_to_none, sync_cuda) for _ in range(rep)]

    # Always take at least one sample, even for a tiny ``rep`` budget.
    start = time.perf_counter()
    times = [_time_once_cpu(fn, grad_to_none, sync_cuda)]
    while _elapsed_ms(start) < rep:
        times.append(_time_once_cpu(fn, grad_to_none, sync_cuda))
    return times


def _summarize_cpu(
    times: list[float],
    quantiles: list[float] | None,
    return_mode: str,
) -> float | list[float] | dict[str, float]:
    """Reduce samples (ms) to quantiles, all statistics, or one statistic."""
    t = torch.tensor(times)
    if quantiles is not None:
        ret = torch.quantile(t, torch.tensor(quantiles, dtype=torch.float)).tolist()
        return ret[0] if len(ret) == 1 else ret
    if return_mode == "all":
        return {
            "min": t.min().item(),
            "max": t.max().item(),
            "mean": t.mean().item(),
            "median": t.median().item(),
        }
    return getattr(torch, return_mode)(t).item()


def _bench_gpu(
    fn: Callable,
    warmup: int | float,
    rep: int | float,
    use_reps: bool,
    grad_to_none,
    quantiles: list[float] | None,
    return_mode: str,
):
    """Time ``fn`` with CUDA events via ``triton.testing`` internals."""
    testing = require("triton.testing", "gpu", "GPU event timing")
    _summarize_statistics, runtime = testing._summarize_statistics, testing.runtime

    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    if use_reps:
        n_warmup = warmup
        n_repeat = rep
    else:
        # Estimate per-call time from 5 calls to turn ms budgets into counts.
        start_event = di.Event(enable_timing=True)
        end_event = di.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            runtime.driver.active.clear_cache(cache)
            fn()
        end_event.record()
        di.synchronize()
        # Clamp so a sub-resolution (0 ms) measurement cannot divide by zero.
        estimate_ms = max(start_event.elapsed_time(end_event) / 5, 1e-6)
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))

    start_event = [di.Event(enable_timing=True) for _ in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for _ in range(n_repeat)]
    for _ in range(n_warmup):
        fn()
    for i in range(n_repeat):
        _reset_grads(grad_to_none)
        runtime.driver.active.clear_cache(cache)
        start_event[i].record()
        fn()
        end_event[i].record()
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


def do_bench(
    fn: Callable,
    warmup: int | float = 25,
    rep: int | float = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: Literal["min", "max", "mean", "median", "all"] = "mean",
    timing_method: Literal["gpu", "cpu"] = "cpu",
    sync_cuda: bool = True,
) -> float | dict[str, float]:
    """Benchmark function runtime with warmup and statistics.

    Supports both CPU wall-clock timing and GPU CUDA event timing.
    Warmup and repetition can be specified as iteration counts (int)
    or durations in milliseconds (float). In duration mode at least one
    measurement is always taken, even if ``rep`` is shorter than a single
    call.

    Args:
        fn: Function to benchmark (callable with no arguments).
        warmup: Warmup iterations (int) or duration in ms (float).
        rep: Measurement iterations (int) or duration in ms (float).
        grad_to_none: Optional tensor whose gradient is reset to None
            between repetitions.
        quantiles: Optional quantiles to compute (e.g., [0.05, 0.95]).
        return_mode: Central statistic to return:
            "min", "max", "mean", "median", or "all" for all stats.
        timing_method: "gpu" for CUDA events (if available) or "cpu"
            for wall-clock timing.
        sync_cuda: Whether to synchronize CUDA before/after timing
            (only applies to CPU timing).

    Returns:
        Timing result. Float for single statistics, dict for "all"
        or when quantiles are specified.

    Example:
        >>> def bench_fn():
        ...     return torch.mm(a, b)
        >>> do_bench(bench_fn, warmup=10, rep=100, return_mode="median")
        0.523
    """
    if not callable(fn):
        raise TypeError("The 'fn' parameter must be callable")

    timing_method = _resolve_timing_method(timing_method)
    use_reps, warmup, rep = _resolve_budget(warmup, rep)

    if timing_method == "gpu":
        return _bench_gpu(
            fn, warmup, rep, use_reps, grad_to_none, quantiles, return_mode
        )

    _warmup_cpu(fn, warmup, use_reps)
    times = _measure_cpu(fn, rep, use_reps, grad_to_none, sync_cuda)
    return _summarize_cpu(times, quantiles, return_mode)
