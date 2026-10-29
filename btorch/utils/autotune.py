"""Bound native GPU autotuning to preparation, never repeated submission.

Only execution parameters are searched. Native decorators retain tuning
results; prepared launches own executables and tensors. Trial buffers
are lazy, invocation-local and never retained by cached decorators.
Mathematical tiles and queue ownership remain fixed; capture reuses
warmed preparation.
"""

from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from threading import RLock

import torch

from .kernel import tensor_layout


# Only profiler-based paths lack a native selection cache.
_PROFILED = OrderedDict()
_ACTIVE_TRIAL = None
_LOCK = RLock()
_MAX_PROFILED = 128
_MAX_SCRATCH_BYTES = 64 * 1024 * 1024


class _FactoryIdentity:
    """Keep an unhashable public factory alive under an identity cache key."""

    def __init__(self, factory):
        self.factory = factory

    def __hash__(self):
        return id(self.factory)

    def __eq__(self, other):
        return isinstance(other, _FactoryIdentity) and self.factory is other.factory


def _remember_profiled(key, value):
    _PROFILED[key] = value
    _PROFILED.move_to_end(key)
    while len(_PROFILED) > _MAX_PROFILED:
        _PROFILED.popitem(last=False)
    return value


def _signature(args):
    return tuple(
        tensor_layout(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    )


def _ranges(args, writes):
    """Identify only reachable writable storage, including overlapping
    inputs."""
    groups = {}
    writable = {(args[i].device, args[i].untyped_storage().data_ptr()) for i in writes}
    for index, arg in enumerate(args):
        if not isinstance(arg, torch.Tensor):
            continue
        key = (arg.device, arg.untyped_storage().data_ptr())
        if key not in writable:
            continue
        first = arg.storage_offset() * arg.element_size()
        span = (
            0
            if not arg.numel()
            else 1
            + sum((size - 1) * stride for size, stride in zip(arg.shape, arg.stride()))
        )
        last = first + span * arg.element_size()
        groups.setdefault(key, []).append((index, first, last))
    return tuple(groups.values())


def scratch_arguments(args, writes):
    """Shadow writable storage while preserving dtype, strides and aliases.

    Read-only storage stays shared. Copy only the span reachable through
    the supplied arguments, not a whole sequence allocation behind a
    timestep view. Return a reset callback for atomic accumulation and
    in-place state updates.
    """
    scratch = list(args)
    copies = []
    for group in _ranges(args, writes):
        # Tensor views sharing storage can have different element sizes.
        # Byte alignment must permit reconstructing each typed view.
        alignment = max(args[index].element_size() for index, _, _ in group)
        start = min(first for _, first, _ in group) // alignment * alignment
        end = max(last for _, _, last in group)
        owner = args[group[0][0]]
        original = torch.empty(0, dtype=torch.uint8, device=owner.device).set_(
            owner.untyped_storage(), start, (end - start,), (1,)
        )
        saved = original.clone()
        shadow = saved.clone()
        copies.append((shadow, saved))
        for index, first, _ in group:
            arg = args[index]
            scratch[index] = torch.empty(0, dtype=arg.dtype, device=arg.device).set_(
                shadow.untyped_storage(),
                (first - start) // arg.element_size(),
                arg.shape,
                arg.stride(),
            )

    def reset():
        for shadow, saved in copies:
            shadow.copy_(saved)

    return scratch, reset


def _eligible(args, writes, work):
    if not writes or work < 4096:
        return False
    size = sum(
        max(last for _, _, last in group) - min(first for _, first, _ in group)
        for group in _ranges(args, writes)
    )
    return size <= _MAX_SCRATCH_BYTES


def outputs_are_disjoint(args, writes):
    """Check storage independence, not whether a stage accumulates outputs."""
    writable = {args[i].untyped_storage().data_ptr() for i in writes}
    return not any(
        isinstance(arg, torch.Tensor)
        and i not in writes
        and arg.untyped_storage().data_ptr() in writable
        for i, arg in enumerate(args)
    )


def execution_candidates(default):
    """Search the heuristic and one neighboring legal warp count."""
    return (default, default // 2 if default > 1 else 2)


class _Trial:
    """Materialize scratch only if the native tuner actually benchmarks."""

    def __init__(self, args, writes):
        self.args = args
        self.writes = writes
        self.scratch = None
        self.reset = None

    def inputs(self):
        if self.scratch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm the workload before CUDA Graph capture")
            self.scratch, self.reset = scratch_arguments(self.args, self.writes)
        return self.scratch


@contextmanager
def _trial_inputs(args, writes):
    # Native TileLang profiling can access these callbacks from its worker.
    # Serialize preparation; never retain trial tensors on a cached decorator.
    global _ACTIVE_TRIAL
    with _LOCK:
        previous = _ACTIVE_TRIAL
        _ACTIVE_TRIAL = _Trial(args, writes)
        try:
            yield _ACTIVE_TRIAL
        finally:
            _ACTIVE_TRIAL = previous


def _reset_trial(*_args, **_kwargs):
    _ACTIVE_TRIAL.inputs()
    _ACTIVE_TRIAL.reset()


def _supply_trial(_params):
    return _ACTIVE_TRIAL.inputs()


def _triton_bench(call, **_):
    from triton.testing import do_bench

    return do_bench(call, warmup=1, rep=3, quantiles=(0.5, 0.2, 0.8))


@lru_cache(maxsize=128)
def _triton_autotuner(kernel, signature, constants, options, writes, default):
    import triton

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm the workload before CUDA Graph capture")
    # Retain the decorated callable, not a second copy of its chosen Config.
    # This object is scoped to one static execution contract, so best_config
    # belongs to this contract. Runtime sequence length is deliberately absent.
    return triton.autotune(
        configs=[triton.Config({}, num_warps=w) for w in execution_candidates(default)],
        key=[],
        pre_hook=_reset_trial,
        do_bench=_triton_bench,
    )(kernel)


def triton_config(kernel, args, grid, constants, options, *, writes, work):
    """Read the native tuner's selected configuration during preparation."""
    if not writes or work < 4096:
        return options
    default = options.get("num_warps", 4)
    # Pointer addressing is already bound in source, scalar args and constexprs;
    # Triton's ABI does not consume tensor shape/stride objects. Runtime sequence
    # length likewise does not change the execution specialization.
    signature = tuple(
        (value.dtype, value.device) if isinstance(value, torch.Tensor) else value
        for name, value in zip(kernel.arg_names, args)
        if name != "n_steps"
    )
    tuner = _triton_autotuner(
        kernel,
        (signature, grid),
        tuple(constants),
        tuple(options.items()),
        tuple(writes),
        default,
    )
    with _LOCK:
        if not hasattr(tuner, "best_config"):
            if not _eligible(args, writes, work):
                return options
            with _trial_inputs(args, writes) as trial:
                scratch = trial.inputs()
                for index, name in enumerate(kernel.arg_names[: len(scratch)]):
                    if name == "n_steps":
                        scratch[index] = min(scratch[index], 1)
                launch_options = {
                    name: value
                    for name, value in options.items()
                    if name != "num_warps"
                }
                tuner[grid](*scratch, **dict(constants), **launch_options)
        return {**options, "num_warps": tuner.best_config.num_warps}


@lru_cache(maxsize=128)
def _tilelang_autotuner(identity, device, warps):
    import tilelang

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm the workload before CUDA Graph capture")
    # Native AutoTuneImpl owns (kernel, config) per factory argument tuple.
    # This public supplier resolves invocation-local scratch on cache misses.
    return tilelang.autotune(
        configs=[{"threads": w * 32} for w in execution_candidates(warps)],
        warmup=1,
        rep=3,
        supply_prog=_supply_trial,
        skip_check=True,
        cache_input_tensors=True,
    )(identity.factory)


def tilelang_kernel(factory, metadata, args, writes, warps, *, work, overwrite):
    """Prepare through TileLang's native cache or reset-aware public
    profiler."""
    if not _eligible(args, writes, work):
        return factory(*metadata, warps * 32)
    identity = _FactoryIdentity(factory)
    if overwrite:
        tuner = _tilelang_autotuner(identity, args[0].device, warps)
        with _trial_inputs(args, writes):
            return tuner.compile(*metadata)

    # TileLang's decorator has no per-repeat reset hook. Restrict this separate
    # memo to accumulating stages timed through its public profiler.
    key = (identity, metadata, _signature(args), warps, tuple(writes))
    with _LOCK:
        if key in _PROFILED:
            return _PROFILED[key]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the workload before CUDA Graph capture")
        from tilelang.profiler import do_bench

        scratch, reset = scratch_arguments(args, writes)
        trials = []
        for candidate in execution_candidates(warps):
            executable = factory(*metadata, candidate * 32)

            def trial():
                reset()
                executable(*scratch)

            trials.append((do_bench(trial, warmup=1, rep=3), executable))
        return _remember_profiled(key, min(trials, key=lambda result: result[0])[1])


@lru_cache(maxsize=1)
def _cupy_profiler():
    # Optional imports may retain tracebacks. Load without a tensor-owning
    # caller stack; subsequent preparation reuses the native profiler.
    from concurrent.futures import ThreadPoolExecutor
    from importlib import import_module

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(import_module, "cupyx.profiler").result().benchmark


def cupy_block(kernel, args, n, writes, *, steps=None):
    """Choose a RawKernel block with CuPy's public CUDA-event profiler."""
    import cupy
    import numpy

    key = ("cupy", kernel, _signature(args), n, tuple(writes))
    with _LOCK:
        if key in _PROFILED:
            return _PROFILED[key]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the CuPy workload before CUDA Graph capture")
        if not _eligible(args, writes, n):
            return _remember_profiled(key, 128)
        benchmark = _cupy_profiler()
        scratch, reset = scratch_arguments(args, writes)
        pointers = tuple(numpy.uintp(arg.data_ptr()) for arg in scratch)
        if steps is not None:
            pointers += (numpy.int32(min(steps, 1)),)
        stream = cupy.cuda.Stream.from_external(
            torch.cuda.current_stream(args[0].device)
        )
        trials = []
        for block in (128, 256):

            def trial():
                reset()
                kernel((-(-n // block),), (block,), pointers, stream=stream)

            with cupy.cuda.Device(args[0].device.index), stream:
                result = benchmark(trial, n_warmup=1, n_repeat=8, max_duration=0.01)
            trials.append((result.gpu_times.mean(), block))
        return _remember_profiled(key, min(trials)[1])
