import time
from typing import Callable, Dict, List, Literal, Optional, Union

import torch


class PerfTimer:
    """A simple performance timer for measuring execution time."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Returns elapsed time in milliseconds."""
        if self.start_time is None:
            raise RuntimeError("Timer never started")
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return (end - self.start_time) * 1000


def _benchmark_total_time(
    fn: Callable,
    warmup_ms: int,
    rep_ms: int,
    grad_to_none: Optional[torch.Tensor] = None,
    sync_cuda: bool = True,
) -> List[float]:
    """Benchmark a function using total wall-clock time measurements.

    Args:
        fn: Function to benchmark
        warmup_ms: Warmup time in milliseconds
        rep_ms: Repetition time in milliseconds
        grad_to_none: Reset gradient of this tensor to None between runs
        sync_cuda: Whether to synchronize CUDA devices before/after execution

    Returns:
        List of execution times in milliseconds
    """
    # Warmup phase
    warmup_start = time.perf_counter()
    while (time.perf_counter() - warmup_start) * 1000 < warmup_ms:
        fn()

    # Measurement phase
    times = []
    rep_start = time.perf_counter()
    while (time.perf_counter() - rep_start) * 1000 < rep_ms:
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        with PerfTimer() as timer:
            fn()
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

        times.append(timer.elapsed_ms())

    return times


def do_bench(
    fn: Callable,
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: Optional[torch.Tensor] = None,
    quantiles: Optional[List[float]] = None,
    return_mode: Literal["min", "max", "mean", "median", "all"] = "mean",
    timing_method: Literal["gpu", "total"] = "total",
    sync_cuda: bool = True,
) -> Union[float, Dict[str, float]]:
    """Benchmark the runtime of the provided function.

    Args:
        fn: Function to benchmark
        warmup: Warmup time (in ms)
        rep: Repetition time (in ms)
        grad_to_none: Reset the gradient of the provided tensor to None
        quantiles: Performance percentiles to return
                   in addition to the central statistic
        return_mode: The statistical measure to return.
            Options are "min", "max", "mean", "median", or "all"
        timing_method: Method to use for timing - "gpu" for CUDA event timing or
            "total" for wall clock timing
        sync_cuda: Whether to synchronize CUDA devices before/after execution
            (only used with timing_method="total")
    """
    if not callable(fn):
        raise TypeError("The 'fn' parameter must be callable")

    if timing_method not in ["gpu", "total"]:
        raise ValueError("timing_method must be either 'gpu' or 'total'")

    if timing_method == "gpu" and not torch.cuda.is_available():
        print(
            "Warning: GPU timing requested but CUDA is not available. "
            "Falling back to total timing."
        )
        timing_method = "total"

    if timing_method == "gpu":
        import triton

        return triton.testing.do_bench(
            fn, warmup, rep, grad_to_none, return_mode=return_mode, quantiles=quantiles
        )

    times = torch.tensor(
        _benchmark_total_time(fn, warmup, rep, grad_to_none, sync_cuda)
    )

    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()
