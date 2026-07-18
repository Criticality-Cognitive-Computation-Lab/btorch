"""Benchmarks for the recurrent time loop in ``btorch/models/rnn.py``.

Two strategy families, each a strategy dict + a ``_measure_*`` helper + a parity
test; ``__main__`` runs the sweeps and prints tables (plus a compile-time plot).

Compile strategies -- how to handle the T-step loop under ``torch.compile``
(compile-time vs inference; CUDA, unroll=8, B=2, H=8):

       T    compile: plain disable region scan | infer: plain disable region scan
      64             7.15    1.83   7.66  3.04 |        0.40    0.37   0.59  1.88
     512            85.01    1.92  68.16  1.26 |        8.41   11.97  26.46 12.37

    Keep ``disable``: compile stays ~flat in T (plain is O(T): ~335s at T=1000),
    inference within ~1.4x of plain. region is worst on both axes; scan is
    O(1)-compile but wins no inference and is training-blocked (test_scan_rnn.py).

CUDA-graph strategies -- inference acceleration for the shipped loop (CUDA,
unroll=16, B=1, H=32, chunk_size=64; inference ms):

       T   eager  reduce-oh  cudagraph  cg+compile  offload  cg+offload
     128    4.27       1.52       1.45        1.23     6.06        1.33
    1024   40.35       8.45       8.89        6.11    56.73        7.66

    ``cudagraph=True`` + ``torch.compile`` is fastest (6.6x over eager at T=1024):
    fusion and launch-collapsing stack. ``cpu_offload`` alone is a slowdown but
    composes with capture (per-chunk, D2H stays eager) and keeps its memory win.
    Launch-bound regime -- shrinks to ~1.3x once wide enough to be compute-bound.

``cudagraph=True`` is inference-only; train with ``torch.compile(mode=
"reduce-overhead")`` (see ``tests/models/rnn/test_cudagraph.py``). Run:
    python benchmarks/rnn/test_compile_strategies.py
"""

import inspect
import platform
import sys
import time
from pathlib import Path


# Make the repo's ``tests`` package importable (and win over any shadowing one)
# when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from btorch.models.functional import reset_net_state
from btorch.models.rnn import RecurrentNNAbstract, make_rnn
from btorch.utils.bench import do_bench
from tests.models.rnn.rnn_utils import HAS_SCAN, ScanRNN, SimpleRNNCell, requires_scan


pytestmark = pytest.mark.skipif(
    platform.system() != "Linux", reason="torch.compile/scan only supported on Linux"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Compile strategies: how to handle the time loop under torch.compile.
# The loop variants subclass rnn.py's loop directly and differ only in guards.
# ===========================================================================
UNROLL = 8


class LoopRNN(RecurrentNNAbstract):
    """``SimpleRNNCell`` driven by rnn.py's chunked time loop (the **disable**
    strategy: ``multi_step_forward`` is already ``disable(recursive=False)``, so
    the loop stays in eager Python)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__(unroll=UNROLL)
        self.cell = SimpleRNNCell(input_size, hidden_size)

    def single_step_forward(self, x_t):
        h = self.cell(x_t)
        return h, {"h": h}


class PlainRNN(LoopRNN):
    """Same loop with the disable guards removed -> Dynamo inlines all T steps.

    ``torch.compiler.disable`` keeps the undecorated function on ``__wrapped__``
    (the ``functools.wraps`` contract), so re-declaring these names with the
    unwrapped originals is the *only* difference from ``LoopRNN``.
    """

    multi_step_forward = inspect.unwrap(RecurrentNNAbstract.multi_step_forward)
    _multi_step_forward_impl = inspect.unwrap(
        RecurrentNNAbstract._multi_step_forward_impl
    )
    _process_large_chunk_impl = inspect.unwrap(
        RecurrentNNAbstract._process_large_chunk_impl
    )


class RegionRNN(PlainRNN):
    """Plain + the step marked as a reusable ``nested_compile_region``."""

    single_step_forward = torch.compiler.nested_compile_region(
        LoopRNN.single_step_forward
    )


COMPILE_STRATEGIES = {
    "plain": PlainRNN,
    "disable": LoopRNN,
    "region": RegionRNN,
}

# Feature-gated rather than version-gated: without the scan HOP the strategy just
# drops out of the sweep and the parity parametrization instead of erroring.
if HAS_SCAN:
    COMPILE_STRATEGIES["scan"] = lambda input_size, hidden_size: ScanRNN(
        SimpleRNNCell(input_size, hidden_size)
    )

# scan needs fullgraph (a break around it hits an unsupported data-dependent op,
# pytorch#153437); the disable variant inherently breaks -> must NOT be fullgraph.
FULLGRAPH = {"scan": True}


def _compile_output(out):
    """Loop variants return ``(stacked, states)``; ScanRNN returns the
    tensor."""
    return out[0] if isinstance(out, tuple) else out


def _measure_compile(name, T, B=2, I=4, H=8):  # noqa: E741 - I is the input dim
    """Compile time (first call, s) and steady-state inference time (ms)."""
    torch._dynamo.reset()
    model = COMPILE_STRATEGIES[name](I, H).to(DEVICE)
    compiled = torch.compile(model, fullgraph=FULLGRAPH.get(name, False))
    x = torch.randn(T, B, I, device=DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        compiled(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0

    with torch.no_grad():  # now warm: measures the generated code, not compilation
        infer_ms = do_bench(
            lambda: compiled(x),
            timing_method="gpu" if DEVICE.type == "cuda" else "cpu",
        )
    return compile_s, infer_ms


@pytest.mark.parametrize("strategy", list(COMPILE_STRATEGIES))
def test_compile_parity(strategy):
    """Every compile strategy matches rnn.py's eager loop (cheap, CPU)."""
    torch.manual_seed(0)
    T, B, I, H = 16, 2, 4, 8  # noqa: E741 - I is the input dim
    x = torch.randn(T, B, I)

    ref = LoopRNN(I, H)  # rnn.py's loop, uncompiled
    reset_net_state(ref, batch_size=B)
    with torch.no_grad():
        expected = _compile_output(ref(x))

    model = COMPILE_STRATEGIES[strategy](I, H)
    model.cell.load_state_dict(ref.cell.state_dict())  # every variant holds .cell
    reset_net_state(model, batch_size=B)
    with torch.no_grad():
        compiled = torch.compile(model, fullgraph=FULLGRAPH.get(strategy, False))
        out = _compile_output(compiled(x))

    assert torch.allclose(out, expected, atol=1e-5)


@requires_scan
def test_scan_compiles_to_single_graph():
    """Scan is O(1) in T: the step is traced once -> one graph capture."""
    counter = {"n": 0}

    def backend(gm, example_inputs):
        counter["n"] += 1
        from torch._inductor.compile_fx import compile_fx

        return compile_fx(gm, example_inputs)

    model = COMPILE_STRATEGIES["scan"](4, 8)
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    with torch.no_grad():
        compiled(torch.randn(512, 2, 4))
    assert counter["n"] == 1, f"expected 1 graph capture, got {counter['n']}"


# ===========================================================================
# CUDA-graph strategies: inference acceleration for the shipped loop, via
# make_rnn (cudagraph=True capture and/or torch.compile). CUDA-only.
# ===========================================================================
CG_UNROLL = 16
CG_CHUNK = 64

# name -> (make_rnn kwargs, torch.compile mode or None)
CUDAGRAPH_STRATEGIES = {
    "eager": (dict(), None),
    "compile": (dict(), "default"),
    "reduce-oh": (dict(), "reduce-overhead"),
    "cudagraph": (dict(cudagraph=True), None),
    "cg+compile": (dict(cudagraph=True), "default"),
    # cpu_offload chunks the sequence and moves each chunk to host memory. Capture
    # is per-chunk, so the D2H moves stay in eager python where replay cannot
    # swallow them -- see btorch/models/cudagraph.py.
    "offload": (dict(cpu_offload=True, chunk_size=CG_CHUNK), None),
    "cg+offload": (dict(cudagraph=True, cpu_offload=True, chunk_size=CG_CHUNK), None),
}


def _build_cudagraph(I, H, **kwargs):  # noqa: E741 - I is the input dim
    torch.manual_seed(0)
    return make_rnn(SimpleRNNCell, unroll=CG_UNROLL, **kwargs)(
        input_size=I, hidden_size=H
    ).to(DEVICE)


def _run_cudagraph(model, compiled, x, mode):
    with torch.no_grad():
        if mode == "reduce-overhead":  # Trees reuse buffers -> mark the boundary
            torch.compiler.cudagraph_mark_step_begin()
        reset_net_state(model, batch_size=x.shape[1])
        return compiled(x)


def _measure_cudagraph(name, T, B=1, I=8, H=32):  # noqa: E741 - I is the input dim
    """Steady-state inference time (ms)."""
    torch._dynamo.reset()
    kwargs, mode = CUDAGRAPH_STRATEGIES[name]
    model = _build_cudagraph(I, H, **kwargs)
    compiled = model if mode is None else torch.compile(model, mode=mode)
    x = torch.randn(T, B, I, device=DEVICE)

    run = lambda: _run_cudagraph(model, compiled, x, mode)  # noqa: E731
    run()  # absorb compile + capture
    return do_bench(run, warmup=5, rep=20, return_mode="median")


# No peak-memory column: every strategy shares one process, so earlier models and
# their graph pools poison max_memory_allocated (cg+offload reads ~42MB in-sweep vs
# ~23MB alone) and cudagraph-trees pools escape that accounting. The memory figures
# in the module docstring were taken one config per process.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="capture requires CUDA")
@pytest.mark.parametrize("strategy", [s for s in CUDAGRAPH_STRATEGIES if s != "eager"])
def test_cudagraph_parity(strategy):
    """Every cudagraph strategy matches the eager loop.

    Guards the benchmark: a strategy that silently skipped work or replayed a
    stale graph would post a great number here for free.
    """
    torch._dynamo.reset()
    T, B, I, H = 64, 1, 8, 32  # noqa: E741 - I is the input dim
    x = torch.randn(T, B, I, device=DEVICE)

    eager = _build_cudagraph(I, H)
    kwargs, mode = CUDAGRAPH_STRATEGIES[strategy]
    model = _build_cudagraph(I, H, **kwargs)
    model.rnn_cell.load_state_dict(eager.rnn_cell.state_dict())
    compiled = model if mode is None else torch.compile(model, mode=mode)

    with torch.no_grad():
        reset_net_state(eager, batch_size=B)
        ref, _ = eager(x)
    out, _ = _run_cudagraph(model, compiled, x, mode)

    # The offload strategies deliberately return on the host; compare there.
    assert torch.allclose(out.cpu(), ref.cpu(), atol=1e-5)


def sweep_table(title, strategies, ts, measure):
    """Print a ``strategy x T`` table.

    ``measure(name, T)`` returns a float, or
    raises (its exception name is shown, e.g. a strategy unsupported at that T).
    """
    print(f"\n{title}")
    print(f"{'T':>6} " + "".join(f"{n:>12}" for n in strategies))
    for t in ts:
        cells = []
        for name in strategies:
            try:
                cells.append(f"{measure(name, t):>12.2f}")
            except Exception as exc:  # noqa: BLE001
                cells.append(f"{type(exc).__name__:>12}")
        print(f"{t:>6} " + "".join(cells))


# --- Copy-paste experiment: edit the strategy dicts / Ts above and re-run. ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from btorch.utils.file import save_fig

    # torch._dynamo.reset() does NOT clear Inductor's on-disk FX cache, so a rerun
    # would report warm (misleadingly fast) compile times. Force cold compiles.
    torch._inductor.config.force_disable_caches = True
    print(f"device: {DEVICE}")

    # Compile strategies: how the loop is handled under torch.compile, vs T.
    compile_ts = [64, 128, 256, 512]  # plain/region compile grows with T
    for name in COMPILE_STRATEGIES:  # warm Inductor once, off the clock
        _measure_compile(name, 8)
    measured = {
        (n, t): _measure_compile(n, t) for n in COMPILE_STRATEGIES for t in compile_ts
    }
    sweep_table(
        "compile strategies -- first compile (s)",
        COMPILE_STRATEGIES,
        compile_ts,
        lambda n, t: measured[(n, t)][0],
    )
    sweep_table(
        "compile strategies -- inference (ms)",
        COMPILE_STRATEGIES,
        compile_ts,
        lambda n, t: measured[(n, t)][1],
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for idx, (ax, ylabel) in enumerate(
        zip(axes, ["first-call compile time (s)", "inference time (ms)"])
    ):
        for name in COMPILE_STRATEGIES:
            ax.plot(
                compile_ts,
                [measured[(name, t)][idx] for t in compile_ts],
                marker="o",
                label=name,
            )
        ax.set(xlabel="sequence length T", ylabel=ylabel, xscale="log", yscale="log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"RNN loop under torch.compile ({DEVICE.type})")
    fig.tight_layout()
    save_fig(fig, name="rnn_compile_vs_inference_vs_T")
    plt.close(fig)

    # CUDA-graph strategies: inference acceleration for the shipped loop, vs T.
    if DEVICE.type == "cuda":
        sweep_table(
            f"cudagraph strategies -- inference (ms), unroll={CG_UNROLL}",
            CUDAGRAPH_STRATEGIES,
            [128, 256, 512, 1024],
            _measure_cudagraph,
        )
