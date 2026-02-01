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


def do_bench(
    fn: Callable,
    warmup: int | float = 25,
    rep: int | float = 100,
    grad_to_none: Optional[torch.Tensor] = None,
    quantiles: Optional[List[float]] = None,
    return_mode: Literal["min", "max", "mean", "median", "all"] = "mean",
    timing_method: Literal["gpu", "cpu"] = "cpu",
    sync_cuda: bool = True,
) -> Union[float, Dict[str, float]]:
    """Benchmark the runtime of the provided function.

    Args:
        fn: Function to benchmark
        warmup: Warmup time in ms (float) or warmup iterations (int)
        rep: Repetition time in ms (float) or measured iterations (int)
        grad_to_none: Reset the gradient of the provided tensor to None
        quantiles: Performance percentiles to return
                   in addition to the central statistic
        return_mode: The statistical measure to return.
            Options are "min", "max", "mean", "median", or "all"
        timing_method: Method to use for timing - "gpu" for CUDA event timing or
            "cpu" for wall clock timing
        sync_cuda: Whether to synchronize CUDA devices before/after execution
            (only used with timing_method="cpu")
    """
    if not callable(fn):
        raise TypeError("The 'fn' parameter must be callable")

    if timing_method == "total":
        timing_method = "cpu"
    if timing_method not in ["gpu", "cpu"]:
        raise ValueError("timing_method must be either 'gpu' or 'cpu'")

    if timing_method == "gpu" and not torch.cuda.is_available():
        print(
            "Warning: GPU timing requested but CUDA is not available. "
            "Falling back to cpu timing."
        )
        timing_method = "cpu"

    if timing_method == "gpu":
        use_reps = isinstance(warmup, int) and isinstance(rep, int)
        if use_reps:
            if warmup < 0 or rep < 1:
                raise ValueError("warmup must be >= 0 and rep must be >= 1")
        else:
            warmup = float(warmup)
            rep = float(rep)
            if warmup < 0.0 or rep <= 0.0:
                raise ValueError("warmup and rep must be positive durations in ms")

        from triton.testing import _summarize_statistics, runtime

        di = runtime.driver.active.get_device_interface()

        fn()
        di.synchronize()

        cache = runtime.driver.active.get_empty_cache_for_benchmark()

        if use_reps:
            n_warmup = warmup
            n_repeat = rep
        else:
            start_event = di.Event(enable_timing=True)
            end_event = di.Event(enable_timing=True)
            start_event.record()
            for _ in range(5):
                runtime.driver.active.clear_cache(cache)
                fn()
            end_event.record()
            di.synchronize()
            estimate_ms = start_event.elapsed_time(end_event) / 5
            n_warmup = max(1, int(warmup / estimate_ms))
            n_repeat = max(1, int(rep / estimate_ms))

        start_event = [di.Event(enable_timing=True) for _ in range(n_repeat)]
        end_event = [di.Event(enable_timing=True) for _ in range(n_repeat)]
        for _ in range(n_warmup):
            fn()
        for i in range(n_repeat):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            runtime.driver.active.clear_cache(cache)
            start_event[i].record()
            fn()
            end_event[i].record()
        di.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
        return _summarize_statistics(times, quantiles, return_mode)

    use_reps = isinstance(warmup, int) and isinstance(rep, int)
    if use_reps:
        if warmup < 0 or rep < 1:
            raise ValueError("warmup must be >= 0 and rep must be >= 1")
    else:
        warmup = float(warmup)
        rep = float(rep)
        if warmup < 0.0 or rep <= 0.0:
            raise ValueError("warmup and rep must be positive durations in ms")

    if use_reps:
        for _ in range(warmup):
            fn()
    else:
        warmup_start = time.perf_counter()
        while (time.perf_counter() - warmup_start) * 1000 < warmup:
            fn()

    times = []
    if use_reps:
        rep_start = None
    else:
        rep_start = time.perf_counter()
    while True:
        if use_reps:
            if len(times) >= rep:
                break
        else:
            if (time.perf_counter() - rep_start) * 1000 >= rep:
                break
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

    times = torch.tensor(times)

    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()
