"""Exercise cyclic CUDA graph destruction at the public runner boundary."""

import gc
import weakref

import pytest
import torch

from btorch.models.cudagraph import CudaGraphRunner


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA"
)


def _completed_runner() -> tuple[CudaGraphRunner, weakref.ReferenceType]:
    """Return a runner with a completed graph and a weak reference to it."""
    module = torch.nn.Module().cuda()
    runner = CudaGraphRunner(warmup=1)
    value = torch.linspace(-0.5, 0.7, 32, device="cuda")

    with torch.inference_mode():
        actual = runner(module, lambda x: x.square() + 0.25, (value,))
    torch.testing.assert_close(actual, value.square() + 0.25, atol=0, rtol=0)

    graph = next(iter(runner.entries.values())).graph
    return runner, weakref.ref(graph)


def test_capture_defers_stale_graph_collection_while_recording():
    """Defer cyclic graph destruction until recording has finished.

    A completed ``torch.cuda.CUDAGraph`` destroys its capture resources when
    its owner is reclaimed. Destroying one while this runner is recording
    invalidates the new capture, so automatic cyclic GC must stay off until
    recording has left the capture context. Stale cycles can safely survive
    capture and be reclaimed afterward, without a forced full-heap sweep.
    """
    gc_was_enabled = gc.isenabled()
    thresholds = gc.get_threshold()
    gc.disable()
    try:
        stale_runner, stale_graph = _completed_runner()
        # Ordinary reference counting cannot reclaim this runner. It represents
        # a discarded model whose completed graph survives until cyclic GC.
        stale_runner._test_cycle = stale_runner
        del stale_runner
        assert stale_graph() is not None

        late_runner, late_graph = _completed_runner()
        module = torch.nn.Module().cuda()
        runner = CudaGraphRunner(warmup=1)
        first = torch.linspace(0.1, 0.9, 32, device="cuda")
        second = torch.linspace(-0.8, 0.4, 32, device="cuda")
        recording_gc_states = []
        stale_graph_alive_while_recording = []
        late_graph_alive_while_recording = []
        calls = 0

        def operation(value):
            nonlocal calls, late_runner
            calls += 1
            recording_gc_states.append(gc.isenabled())

            # Call one is the uncaptured settle pass. Call two runs inside
            # ``torch.cuda.graph``. Make another completed graph owner cyclic
            # only during recording and lower all thresholds to force the
            # automatic-GC condition. If the runner failed to suspend it, an
            # automatic collection can reclaim this real graph owner and
            # invalidate the active capture.
            if calls == 2:
                # Both pre-existing and newly discarded graph cycles must
                # survive the active capture even under aggressive GC.
                stale_graph_alive_while_recording.append(stale_graph() is not None)
                late_runner._test_cycle = late_runner
                late_runner = None
                gc.set_threshold(1, 1, 1)
                for _ in range(64):
                    cycle = []
                    cycle.append(cycle)
                late_graph_alive_while_recording.append(late_graph() is not None)
            return 1.5 * value - 0.125

        # Keep automatic collection out of the uncaptured settle pass so the
        # recording pass is what encounters the aggressive-GC condition.
        gc.set_threshold(10**9, 10**9, 10**9)
        gc.enable()
        with torch.inference_mode():
            first_actual = runner(module, operation, (first,))
            second_actual = runner(module, operation, (second,))

        assert stale_graph_alive_while_recording == [True]
        assert late_graph_alive_while_recording == [True]
        assert recording_gc_states == [True, False]
        assert gc.isenabled()
        assert runner.n_captures == 1
        torch.testing.assert_close(first_actual, 1.5 * first - 0.125, atol=0, rtol=0)
        torch.testing.assert_close(second_actual, 1.5 * second - 0.125, atol=0, rtol=0)
        assert not torch.equal(first_actual, second_actual)

        # The cycle created during recording is ordinary garbage once capture
        # has ended, and an explicit collection may reclaim its graph safely.
        gc.collect()
        assert stale_graph() is None
        assert late_graph() is None
    finally:
        gc.set_threshold(*thresholds)
        if gc_was_enabled:
            gc.enable()
        else:
            gc.disable()


def test_capture_preserves_disabled_gc_and_defers_graph_collection():
    """Respect caller-managed collection while safely recording a new graph."""
    gc_was_enabled = gc.isenabled()
    thresholds = gc.get_threshold()
    gc.disable()
    try:
        stale_runner, stale_graph = _completed_runner()
        stale_runner._test_cycle = stale_runner
        del stale_runner
        assert stale_graph() is not None

        module = torch.nn.Module().cuda()
        runner = CudaGraphRunner(warmup=1)
        value = torch.linspace(-0.6, 0.8, 32, device="cuda")
        gc_states = []

        def operation(x):
            gc_states.append(gc.isenabled())
            return 0.75 * x + 0.5

        with torch.inference_mode():
            actual = runner(module, operation, (value,))

        # A caller disabling cyclic GC retains control of collection. This
        # stale graph must not invalidate capture or force an expensive sweep.
        assert stale_graph() is not None
        assert gc_states == [False, False]
        assert not gc.isenabled()
        torch.testing.assert_close(actual, 0.75 * value + 0.5, atol=0, rtol=0)
        gc.collect()
        assert stale_graph() is None
    finally:
        gc.set_threshold(*thresholds)
        if gc_was_enabled:
            gc.enable()
        else:
            gc.disable()
