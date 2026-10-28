"""Small CUDA benchmark helpers used by the optional model benchmarks.

The helpers keep measurement policy in one place without adding a benchmark
runner or an optional dependency to btorch.  They are intentionally not
re-exported from :mod:`btorch.models.kernels`: benchmark scripts should state
which measurements they collect and how their independent reference is built.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class TimingSummary:
    """Summarize repeated measurements in milliseconds."""

    median_ms: float
    min_ms: float
    max_ms: float
    samples: int


@dataclass(frozen=True)
class CudaTiming:
    """Report device elapsed time and synchronized host end-to-end time."""

    device: TimingSummary
    host_end_to_end: TimingSummary


@dataclass(frozen=True)
class MemoryUsage:
    """Report allocations made by one warmed operation."""

    baseline_allocated_bytes: int
    peak_allocated_bytes: int
    peak_incremental_allocated_bytes: int
    returned_bytes: int


@dataclass(frozen=True)
class CaptureCase:
    """Describe a graph-capture operation and its mutable input storage.

    ``operation`` must return persistent output tensors whose storage remains
    valid after graph replay.  ``inputs`` contains the device tensors mutated
    by the changed-input validation.  Stateful models should reset their state
    inside ``operation`` or in the validation callback supplied to
    :func:`capture_measure`. Optional ``reference`` evaluates an independent
    oracle using the current inputs; otherwise validation uses eager execution.
    """

    operation: Callable[[], Any]
    inputs: tuple[Tensor, ...]
    reference: Callable[[], Any] | None = None


@dataclass(frozen=True)
class CaptureMeasurement:
    """Report a validated CUDA Graph replay measurement."""

    timing: CudaTiming
    validation: str


@dataclass(frozen=True)
class LaunchProfile:
    """Report profiler event counts for a bounded operation.

    ``cuda_events`` counts CUDA-device events emitted by PyTorch's profiler;
    it is a profiler event count, not a claim about driver-level kernel launch
    packets.  Use Nsight Systems for an authoritative launch count.
    """

    cpu_events: int
    cuda_events: int
    device_time_ms: float


def _summary(values: Iterable[float]) -> TimingSummary:
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError("at least one timing sample is required")
    return TimingSummary(
        median_ms=float(statistics.median(samples)),
        min_ms=min(samples),
        max_ms=max(samples),
        samples=len(samples),
    )


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark helper")


def tensor_bytes(value: Any) -> int:
    """Count unique backing-storage bytes in a nested benchmark result.

    Views and repeated references count once because the result owns one
    allocation, even when several recorded values reach the same
    storage.
    """
    seen: set[tuple[str, int | None, int, int]] = set()

    def visit(item: Any) -> int:
        if isinstance(item, Tensor):
            storage = item.untyped_storage()
            key = (
                item.device.type,
                item.device.index,
                storage.data_ptr(),
                storage.nbytes(),
            )
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(item, Mapping):
            return sum(visit(child) for child in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(child) for child in item)
        return 0

    return visit(value)


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a source or fixture file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(
    paths: Iterable[Path], *, root: Path | None = None
) -> dict[str, str]:
    """Return deterministic source paths and SHA-256 identities."""
    resolved_root = root.resolve() if root is not None else None
    manifest = {}
    for path in sorted({Path(path).resolve() for path in paths}):
        if not path.is_file():
            raise FileNotFoundError(path)
        name = str(path.relative_to(resolved_root)) if resolved_root else str(path)
        manifest[name] = sha256_file(path)
    return manifest


def measure_cuda(
    operation: Callable[[], Any], *, warmup: int = 1, repeats: int = 3
) -> CudaTiming:
    """Measure a warmed operation with CUDA events and synchronized host time.

    The device value covers work between the two events.  The host value
    includes Python enqueue time and the final synchronization.
    Construction, compilation and graph capture belong outside this
    function.
    """
    _require_cuda()
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        result = operation()
        del result
    torch.cuda.synchronize()
    device_ms = []
    host_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        host_start = time.perf_counter()
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        device_ms.append(start.elapsed_time(end))
        host_ms.append((time.perf_counter() - host_start) * 1000.0)
        del result
    return CudaTiming(_summary(device_ms), _summary(host_ms))


def measure_peak_memory(
    operation: Callable[[], Any], *, warmup: int = 1
) -> MemoryUsage:
    """Measure incremental peak allocation after warming an operation."""
    _require_cuda()
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    for _ in range(warmup):
        result = operation()
        del result
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    returned = tensor_bytes(result)
    del result
    return MemoryUsage(
        baseline_allocated_bytes=baseline,
        peak_allocated_bytes=peak,
        peak_incremental_allocated_bytes=max(peak - baseline, 0),
        returned_bytes=returned,
    )


def _clone_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    return value


def _changed(before: Any, after: Any) -> bool:
    if isinstance(before, Tensor) and isinstance(after, Tensor):
        return not torch.equal(before, after)
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        return any(_changed(before[key], after[key]) for key in before.keys())
    if isinstance(before, (tuple, list)) and isinstance(after, (tuple, list)):
        return any(_changed(old, new) for old, new in zip(before, after, strict=True))
    return before != after


def capture_measure(
    factory: Callable[[], CaptureCase],
    *,
    warmup: int = 1,
    repeats: int = 3,
    validate: Callable[[Any, Any], None] | None = None,
    mutate: Callable[[tuple[Tensor, ...]], None] | None = None,
) -> CaptureMeasurement:
    """Capture, validate and measure one persistent CUDA Graph operation.

    ``validate(actual, expected)`` compares graph output with the case's
    reference or eager output. The reference must start from the same physical
    state. If ``mutate`` is given,
    the helper mutates the declared inputs, replays the graph, requires at
    least one output to change, and restores the inputs before returning.
    """
    _require_cuda()
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        case = factory()
        for _ in range(warmup):
            result = case.operation()
            del result
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = case.operation()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()

    if validate is not None:
        actual = _clone_tree(captured)
        expected = (case.reference or case.operation)()
        validate(actual, expected)
        del actual, expected
        # Validation may execute an eager stateful operation and overwrite the
        # persistent output. Restore the graph-produced output before testing
        # the changed-input replay contract.
        graph.replay()
        torch.cuda.synchronize()

    if mutate is not None:
        # Reuse parity's storage-preserving input ownership for aliases and
        # expanded views; restoring these with Tensor.copy_ can overlap lanes.
        from .testing import _clone_inputs, _copy_inputs

        originals = _clone_inputs(case.inputs)
        before = _clone_tree(captured)
        try:
            mutate(case.inputs)
            graph.replay()
            torch.cuda.synchronize()
            if not _changed(before, captured):
                raise AssertionError(
                    "Captured outputs did not respond to changed input"
                )
            if validate is not None:
                actual = _clone_tree(captured)
                expected = (case.reference or case.operation)()
                validate(actual, expected)
                del actual, expected
        finally:
            _copy_inputs(case.inputs, originals)

    timing = measure_cuda(graph.replay, warmup=0, repeats=repeats)
    checks = []
    if validate is not None:
        checks.append(
            "reference parity passed" if case.reference else "eager parity passed"
        )
    if mutate is not None:
        checks.append("changed-input replay passed")
    return CaptureMeasurement(
        timing=timing,
        validation="; ".join(checks)
        or "capture replay measured; correctness validation not requested",
    )


def profile_launches(
    operation: Callable[[], Any], *, warmup: int = 1, repeats: int = 1
) -> LaunchProfile:
    """Collect bounded PyTorch profiler event counts for an operation."""
    _require_cuda()
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        result = operation()
        del result
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        for _ in range(repeats):
            result = operation()
            del result
        torch.cuda.synchronize()
    events = trace.events()
    cpu_events = sum(
        event.device_type == torch.profiler.DeviceType.CPU for event in events
    )
    cuda_events = [
        event for event in events if event.device_type == torch.profiler.DeviceType.CUDA
    ]
    # CPU parents include their children's device durations. Sum device events
    # only so nested CPU scopes cannot count the same GPU work more than once.
    device_time_ms = sum(event.device_time_total for event in cuda_events) / 1000.0
    return LaunchProfile(cpu_events, len(cuda_events), device_time_ms)
