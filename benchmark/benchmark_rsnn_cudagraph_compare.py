"""Compare torch.compile(reduce-overhead), manual CUDA graphs, and the CUDA
persistent kernel on the same recurrent LIF + ExponentialPSC RSNN dynamics.

All three providers share one forward function (``native_sparse_step``) built
on ``btorch.sparse``'s native CSR backend (``sparse_mm``), so the only thing
that differs between providers is *how the per-timestep Python loop gets
dispatched to the GPU*:

  * ``torch_compile_reduce_overhead`` -- ``torch.compile(fn, mode="reduce-overhead")``.
    Inductor's cudagraph-trees feature captures a CUDA graph per compiled
    region internally; we just call the compiled callable.
  * ``cudagraph_native_sparse`` -- a hand-rolled ``torch.cuda.CUDAGraph()``
    capture of the whole T-step loop (warmup on a side stream, capture once,
    replay with fresh input copied into the static input buffer). Same
    pattern as ``tests/models/test_cudagraph.py``.
  * ``persistent`` -- the CUDA cooperative-kernel backend from
    ``btorch.backend.persistent_snn`` (single kernel launch does all T steps
    with in-kernel ``grid.sync()`` barriers instead of separate per-step
    kernel launches).
  * ``eager_native_sparse`` -- no compile, no graph; the same Python loop run
    directly. Included as a baseline to show what compile/graph buy you.

Usage::

    python benchmark/benchmark_rsnn_cudagraph_compare.py \
        --n-neuron 8192 --t-steps 16 32 64 128 256 --fanout 32 --event-rate 0.01
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Literal

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.benchmark_persistent_snn import (  # noqa: E402
    BenchCase,
    RSNNResult,
    csr_to_dense,
    csr_to_persistent_graph,
    dense_rsnn_forward,
    dense_to_windowed_events,
    make_input_sequence,
    make_recurrent_csr,
)
from btorch.backend.persistent_snn import (  # noqa: E402
    PersistentSNNParams,
    make_empty_state,
    persistent_snn_forward,
)
from btorch.sparse import CSR, BinaryEvents, event_sparse_mm, sparse_mm  # noqa: E402


Provider = Literal[
    "eager_native_sparse",
    "torch_compile_reduce_overhead",
    "cudagraph_native_sparse",
    "eager_prespan",
    "cudagraph_prespan",
    "persistent",
]

PROVIDERS: tuple[Provider, ...] = (
    "eager_native_sparse",
    "torch_compile_reduce_overhead",
    "cudagraph_native_sparse",
    "eager_prespan",
    "cudagraph_prespan",
    "persistent",
)


def precompute_csr_row(matrix: CSR) -> torch.Tensor:
    """Expand ``indptr`` into a per-edge source-row index vector.

    ``CSR.mm`` recomputes this with ``torch.repeat_interleave`` on every
    call. That op is *not* stream-capture safe (its output size is resolved
    with an internal sync), so for the CUDA-graph provider we compute it once
    outside the graph -- it only depends on the matrix's fixed structure, not
    on any per-call data.
    """

    counts = matrix.indptr[1:] - matrix.indptr[:-1]
    return torch.repeat_interleave(
        torch.arange(matrix.shape[0], device=matrix.indptr.device), counts
    )


def csr_mm_with_cached_row(
    matrix: CSR, x: torch.Tensor, row: torch.Tensor
) -> torch.Tensor:
    """Same contraction as ``CSR.mm``, but taking a precomputed ``row``.

    Graph-capture-safe version of ``btorch.sparse.sparse_mm`` for a native
    CSR matrix: no ``repeat_interleave``, only gather / elementwise / static
    ``scatter_add_``, all of which are capturable.
    """

    leading = x.shape[:-1]
    x2d = x.reshape(-1, matrix.shape[0])
    contributions = x2d[:, row] * matrix.effective_values()
    result = torch.zeros(x2d.shape[0], matrix.shape[1], device=x.device, dtype=x.dtype)
    result.scatter_add_(1, matrix.indices.expand(x2d.shape[0], -1), contributions)
    return result.reshape(*leading, matrix.shape[1])


def native_sparse_rsnn_forward(
    x_seq: torch.Tensor,
    matrix: CSR,
    v0: torch.Tensor,
    psc0: torch.Tensor,
    *,
    dt: float,
    tau_mem: float,
    tau_syn: float,
    v_threshold: float,
    v_reset: float,
    c_m: float,
    t_steps: int,
    row: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference RSNN step function shared by all non-persistent providers.

    Pure tensor ops, static shapes for a fixed ``matrix``/``t_steps`` -- safe
    to trace with ``torch.compile``. Pass ``row`` (see
    ``precompute_csr_row``) to make this capture-safe for a raw
    ``torch.cuda.graph()`` capture too.
    """

    decay = math.exp(-dt / tau_syn)
    reset_delta = v_threshold - v_reset
    v = v0
    psc = psc0
    spikes = []
    for t in range(t_steps):
        current = psc + x_seq[t]
        v_pre = v + dt * (-(v - v_reset) / tau_mem + current / c_m)
        z = (v_pre >= v_threshold).to(v.dtype)
        v = v_pre - reset_delta * z
        recurrent = (
            csr_mm_with_cached_row(matrix, z, row)
            if row is not None
            else sparse_mm(matrix, z)
        )
        psc = psc * decay + recurrent
        spikes.append(z)
    return torch.stack(spikes, dim=0), v, psc


def _zero_state(
    case: BenchCase, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    v0 = torch.zeros(case.batch_size, case.n_neuron, device=device, dtype=torch.float32)
    psc0 = torch.zeros_like(v0)
    return v0, psc0


def run_eager_native_sparse(x_seq, matrix, case: BenchCase) -> RSNNResult:
    v0, psc0 = _zero_state(case, x_seq.device)
    spikes, v, psc = native_sparse_rsnn_forward(
        x_seq,
        matrix,
        v0,
        psc0,
        dt=case.dt,
        tau_mem=case.tau_mem,
        tau_syn=case.tau_syn,
        v_threshold=case.v_threshold,
        v_reset=case.v_reset,
        c_m=case.c_m,
        t_steps=case.t_steps,
    )
    return RSNNResult(spikes=spikes, v=v, psc=psc)


def prespan_rsnn_forward(
    x_seq: torch.Tensor,
    matrix: CSR,
    v0: torch.Tensor,
    psc0: torch.Tensor,
    *,
    dt: float,
    tau_mem: float,
    tau_syn: float,
    v_threshold: float,
    v_reset: float,
    c_m: float,
    t_steps: int,
    max_events: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same RSNN dynamics, but the recurrent fan-out uses the Triton
    ``pre_span`` event kernel -- the same "pre-synaptic spike list is a task
    queue, fan out over that neuron's CSR edges" algorithm the persistent CUDA
    kernel implements (see ``plan.md``: "prespan算法"), just dispatched as
    ordinary per-timestep kernel launches instead of one cooperative kernel
    with in-kernel barriers.

    ``max_events`` must be a fixed capacity chosen up front (not derived from
    data inside this function) for two reasons: it fixes the Triton kernels'
    launch grid size, and ``compact_binary_events``/``event_sparse_mm`` only
    take the data-dependent-shape branch (a capture-unsafe ``.item()``-sized
    allocation) when ``max_events`` is left as ``None``. Passing a fixed value
    keeps every intermediate shape static, which is what makes this
    capture-safe for the CUDA-graph provider below.
    """

    decay = math.exp(-dt / tau_syn)
    reset_delta = v_threshold - v_reset
    v = v0
    psc = psc0
    spikes = []
    for t in range(t_steps):
        current = psc + x_seq[t]
        v_pre = v + dt * (-(v - v_reset) / tau_mem + current / c_m)
        z = (v_pre >= v_threshold).to(v.dtype)
        v = v_pre - reset_delta * z
        recurrent = event_sparse_mm(
            matrix, BinaryEvents(z), schedule="pre_span", max_events=max_events
        )
        psc = psc * decay + recurrent
        spikes.append(z)
    return torch.stack(spikes, dim=0), v, psc


def run_eager_prespan(x_seq, matrix, case: BenchCase, *, max_events: int) -> RSNNResult:
    v0, psc0 = _zero_state(case, x_seq.device)
    spikes, v, psc = prespan_rsnn_forward(
        x_seq,
        matrix,
        v0,
        psc0,
        dt=case.dt,
        tau_mem=case.tau_mem,
        tau_syn=case.tau_syn,
        v_threshold=case.v_threshold,
        v_reset=case.v_reset,
        c_m=case.c_m,
        t_steps=case.t_steps,
        max_events=max_events,
    )
    return RSNNResult(spikes=spikes, v=v, psc=psc)


class CompiledNativeSparseProvider:
    """Caches one ``torch.compile`` region per (matrix identity, t_steps)."""

    def __init__(self) -> None:
        self._compiled: dict[int, object] = {}

    def __call__(self, x_seq: torch.Tensor, matrix: CSR, case: BenchCase) -> RSNNResult:
        key = id(matrix)
        compiled = self._compiled.get(key)
        if compiled is None:
            fn = torch.compile(native_sparse_rsnn_forward, mode="reduce-overhead")
            self._compiled[key] = compiled = fn
        v0, psc0 = _zero_state(case, x_seq.device)
        spikes, v, psc = compiled(
            x_seq,
            matrix,
            v0,
            psc0,
            dt=case.dt,
            tau_mem=case.tau_mem,
            tau_syn=case.tau_syn,
            v_threshold=case.v_threshold,
            v_reset=case.v_reset,
            c_m=case.c_m,
            t_steps=case.t_steps,
        )
        return RSNNResult(spikes=spikes.clone(), v=v.clone(), psc=psc.clone())


class CUDAGraphNativeSparseProvider:
    """Manual CUDA graph capture of the whole T-step recurrent loop.

    Mirrors ``tests/models/test_cudagraph.py``: static input/state buffers,
    a few warmup iterations on a side stream, one capture, then replay with
    the real input copied into the static buffer.
    """

    def __init__(self) -> None:
        self._graphs: dict[tuple, dict] = {}

    def _build(self, x_seq: torch.Tensor, matrix: CSR, case: BenchCase) -> dict:
        device = x_seq.device
        static_x = torch.zeros_like(x_seq)
        static_v0, static_psc0 = _zero_state(case, device)
        # Precomputed once outside the graph -- see precompute_csr_row().
        row = precompute_csr_row(matrix)

        # Warmup on a side stream so the capture doesn't observe the first-run
        # allocator/cuBLAS/cuSPARSE workspace setup (those calls are not
        # capturable / would bake in stale addresses).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                native_sparse_rsnn_forward(
                    static_x,
                    matrix,
                    static_v0,
                    static_psc0,
                    dt=case.dt,
                    tau_mem=case.tau_mem,
                    tau_syn=case.tau_syn,
                    v_threshold=case.v_threshold,
                    v_reset=case.v_reset,
                    c_m=case.c_m,
                    t_steps=case.t_steps,
                    row=row,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_spikes, static_v, static_psc = native_sparse_rsnn_forward(
                static_x,
                matrix,
                static_v0,
                static_psc0,
                dt=case.dt,
                tau_mem=case.tau_mem,
                tau_syn=case.tau_syn,
                v_threshold=case.v_threshold,
                v_reset=case.v_reset,
                c_m=case.c_m,
                t_steps=case.t_steps,
                row=row,
            )
        return {
            "graph": graph,
            "static_x": static_x,
            "static_spikes": static_spikes,
            "static_v": static_v,
            "static_psc": static_psc,
            # Referenced only inside the captured region above, with no
            # Python-visible output -- if we don't keep them alive here too,
            # the allocator is free to reclaim their memory once _build()
            # returns and hand it to something else. On replay the graph
            # still reads/writes those *addresses*, so a later allocation
            # landing there silently corrupts the "constant" zero initial
            # state and the CSR row-index buffer (which then feeds a gather
            # -> device-side out-of-bounds assert a replay or two later).
            "static_v0": static_v0,
            "static_psc0": static_psc0,
            "row": row,
        }

    def __call__(self, x_seq: torch.Tensor, matrix: CSR, case: BenchCase) -> RSNNResult:
        key = (id(matrix), case.t_steps, case.batch_size, x_seq.shape)
        state = self._graphs.get(key)
        if state is None:
            state = self._build(x_seq, matrix, case)
            self._graphs[key] = state
        state["static_x"].copy_(x_seq)
        state["graph"].replay()
        return RSNNResult(
            spikes=state["static_spikes"].clone(),
            v=state["static_v"].clone(),
            psc=state["static_psc"].clone(),
        )


def _max_events_for_case(
    case: BenchCase, reference: RSNNResult, *, margin: int = 0
) -> int:
    """Fixed per-timestep spike capacity for the pre_span Triton kernels.

    Derived once from the dense reference's realized spike counts (same
    methodology ``benchmark_persistent_snn.py`` already uses for its eager
    Triton providers) -- NOT computed inside the timed/captured region.
    Exceeding this capacity silently truncates spikes in
    ``dense_event_to_list_kernel`` (masked writes, no error), so a margin can
    be added for inputs that vary at replay time.
    """

    observed = int(reference.spikes.count_nonzero(dim=2).max().item())
    return max(1, min(case.n_neuron, observed + margin))


class CUDAGraphPreSpanProvider:
    """Manual CUDA graph capture of the pre_span (Triton) recurrent loop.

    Same capture pattern as ``CUDAGraphNativeSparseProvider``, but wraps the
    *actual algorithm the persistent CUDA kernel implements* (prespan
    event-driven fan-out) instead of a dense CSR gather/scatter -- this is
    the apples-to-apples comparison of "one cooperative kernel with in-kernel
    barriers" vs. "the same algorithm as ordinary per-timestep kernels,
    captured once and replayed" using the same underlying computation.
    """

    def __init__(self) -> None:
        self._graphs: dict[tuple, dict] = {}

    def _build(
        self, x_seq: torch.Tensor, matrix: CSR, case: BenchCase, *, max_events: int
    ) -> dict:
        device = x_seq.device
        static_x = torch.zeros_like(x_seq)
        static_v0, static_psc0 = _zero_state(case, device)
        # Force the padded-CSR layout to be built and cached *before* capture
        # -- computing it lazily inside the graph would not be capture-safe
        # (it involves data-dependent sizing), and it never changes for a
        # fixed matrix, so precomputing once here is exactly the same
        # "structural, not data-dependent -> hoist out of the graph" pattern
        # as precompute_csr_row() for the native-sparse provider.
        matrix.padded_csr_layout()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                prespan_rsnn_forward(
                    static_x,
                    matrix,
                    static_v0,
                    static_psc0,
                    dt=case.dt,
                    tau_mem=case.tau_mem,
                    tau_syn=case.tau_syn,
                    v_threshold=case.v_threshold,
                    v_reset=case.v_reset,
                    c_m=case.c_m,
                    t_steps=case.t_steps,
                    max_events=max_events,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_spikes, static_v, static_psc = prespan_rsnn_forward(
                static_x,
                matrix,
                static_v0,
                static_psc0,
                dt=case.dt,
                tau_mem=case.tau_mem,
                tau_syn=case.tau_syn,
                v_threshold=case.v_threshold,
                v_reset=case.v_reset,
                c_m=case.c_m,
                t_steps=case.t_steps,
                max_events=max_events,
            )
        return {
            "graph": graph,
            "static_x": static_x,
            "static_spikes": static_spikes,
            "static_v": static_v,
            "static_psc": static_psc,
            # See CUDAGraphNativeSparseProvider._build for why these must be
            # kept alive for the graph's lifetime.
            "static_v0": static_v0,
            "static_psc0": static_psc0,
        }

    def __call__(
        self, x_seq: torch.Tensor, matrix: CSR, case: BenchCase, *, max_events: int
    ) -> RSNNResult:
        key = (id(matrix), case.t_steps, case.batch_size, x_seq.shape, max_events)
        state = self._graphs.get(key)
        if state is None:
            state = self._build(x_seq, matrix, case, max_events=max_events)
            self._graphs[key] = state
        state["static_x"].copy_(x_seq)
        state["graph"].replay()
        return RSNNResult(
            spikes=state["static_spikes"].clone(),
            v=state["static_v"].clone(),
            psc=state["static_psc"].clone(),
        )


def run_persistent(x_seq, matrix, case: BenchCase, *, backend: str) -> RSNNResult:
    events = dense_to_windowed_events(x_seq)
    graph = csr_to_persistent_graph(matrix)
    state = make_empty_state(
        case.batch_size, case.n_neuron, device=x_seq.device, refractory=False
    )
    params = PersistentSNNParams(
        dt=case.dt,
        tau_mem=case.tau_mem,
        tau_syn=case.tau_syn,
        v_threshold=case.v_threshold,
        v_reset=case.v_reset,
        c_m=case.c_m,
        hard_reset=case.hard_reset,
        window_size=case.t_steps,
    )
    out = persistent_snn_forward(
        events, graph, state, params, backend=backend, return_mode="dense"
    )
    assert out.spikes is not None
    return RSNNResult(spikes=out.spikes, v=out.state.v, psc=out.state.psc)


def time_ms(fn, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(torch.median(torch.tensor(samples, dtype=torch.float64)).item())


def max_abs_diff(a: RSNNResult, b: RSNNResult) -> float:
    return float(
        max(
            (a.spikes - b.spikes).abs().max().item(),
            (a.v - b.v).abs().max().item(),
            (a.psc - b.psc).abs().max().item(),
        )
    )


def bench_case(
    case: BenchCase,
    *,
    device: torch.device,
    providers: tuple[Provider, ...],
    compiled_provider: CompiledNativeSparseProvider,
    graph_provider: CUDAGraphNativeSparseProvider,
    prespan_graph_provider: CUDAGraphPreSpanProvider,
    warmup: int,
    repeat: int,
    check_correctness: bool,
) -> list[dict]:
    x_seq = make_input_sequence(case, device)
    matrix = make_recurrent_csr(case, device)
    weight_dense = csr_to_dense(matrix)
    reference = dense_rsnn_forward(x_seq, weight_dense, case)
    # Fixed spike-capacity for the Triton pre_span kernels, derived once from
    # the (deterministic) dense reference for this case -- see
    # prespan_rsnn_forward's docstring for why this must be static.
    max_events = _max_events_for_case(case, reference)

    rows = []
    for provider in providers:
        try:
            if provider == "eager_native_sparse":
                op = lambda: run_eager_native_sparse(x_seq, matrix, case)
            elif provider == "torch_compile_reduce_overhead":
                op = lambda: compiled_provider(x_seq, matrix, case)
            elif provider == "cudagraph_native_sparse":
                op = lambda: graph_provider(x_seq, matrix, case)
            elif provider == "eager_prespan":
                op = lambda: run_eager_prespan(
                    x_seq, matrix, case, max_events=max_events
                )
            elif provider == "cudagraph_prespan":
                op = lambda: prespan_graph_provider(
                    x_seq, matrix, case, max_events=max_events
                )
            elif provider == "persistent":
                op = lambda: run_persistent(
                    x_seq, matrix, case, backend="cuda_persistent"
                )
            else:
                raise ValueError(provider)

            result = op()
            status, diff = "not_checked", float("nan")
            if check_correctness:
                diff = max_abs_diff(result, reference)
                status = "passed" if diff < 1.0 else "diff_ge_1_spike"
            latency = time_ms(op, warmup=warmup, repeat=repeat)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "provider": provider,
                    "n_neuron": case.n_neuron,
                    "t_steps": case.t_steps,
                    "batch_size": case.batch_size,
                    "fanout": case.fanout,
                    "status": f"error:{type(exc).__name__}: {exc}",
                    "max_abs_diff": float("nan"),
                    "latency_ms": float("nan"),
                }
            )
            print(
                f"[error] provider={provider} N={case.n_neuron} T={case.t_steps}: {exc}"
            )
            continue

        rows.append(
            {
                "provider": provider,
                "n_neuron": case.n_neuron,
                "t_steps": case.t_steps,
                "batch_size": case.batch_size,
                "fanout": case.fanout,
                "status": status,
                "max_abs_diff": diff,
                "latency_ms": latency,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neuron", type=int, default=2**13)
    parser.add_argument(
        "--t-steps",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fanout", type=int, default=32)
    parser.add_argument("--event-rate", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--tau-syn", type=float, default=5.0)
    parser.add_argument("--v-threshold", type=float, default=1.0)
    parser.add_argument("--c-m", type=float, default=1.0)
    parser.add_argument("--input-amplitude", type=float, default=30.0)
    parser.add_argument("--weight-scale", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS)
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this comparison.")
    device = torch.device("cuda")

    compiled_provider = CompiledNativeSparseProvider()
    graph_provider = CUDAGraphNativeSparseProvider()
    prespan_graph_provider = CUDAGraphPreSpanProvider()

    all_rows: list[dict] = []
    for t_steps in args.t_steps:
        case = BenchCase(
            n_neuron=args.n_neuron,
            batch_size=args.batch_size,
            t_steps=t_steps,
            fanout=args.fanout,
            event_rate=args.event_rate,
            dt=args.dt,
            tau_mem=args.tau_mem,
            tau_syn=args.tau_syn,
            v_threshold=args.v_threshold,
            c_m=args.c_m,
            input_amplitude=args.input_amplitude,
            weight_scale=args.weight_scale,
        )
        print(f"=== N={case.n_neuron} T={t_steps} fanout={case.fanout} ===")
        rows = bench_case(
            case,
            device=device,
            providers=tuple(args.providers),
            compiled_provider=compiled_provider,
            graph_provider=graph_provider,
            prespan_graph_provider=prespan_graph_provider,
            warmup=args.warmup,
            repeat=args.repeat,
            check_correctness=not args.skip_correctness,
        )
        for row in rows:
            print(
                f"  {row['provider']:<32} {row['status']:<20} "
                f"latency={row['latency_ms']:.4f} ms"
            )
        all_rows.extend(rows)

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
