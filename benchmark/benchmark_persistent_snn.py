"""RSNN benchmark for event span, torch.compile, and persistent backends.

The benchmark compares four providers under the same recurrent LIF +
ExponentialPSC dynamics:

    z_t = LIF(psc_t + x_t)
    psc_{t+1} = exp(-dt / tau_syn) * psc_t + recurrent(z_t)

``event_pre_span`` and ``event_post_span`` call the main btorch sparse event
path directly via ``event_sparse_mm(..., schedule=...)``. ``persistent`` uses
the current persistent SNN scaffold so the benchmark contract is ready for the
future CUDA persistent kernel.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from btorch.backend.persistent_snn import (  # noqa: E402
    EventCSRGraph,
    PersistentSNNParams,
    WindowedSpikeEvents,
    make_empty_state,
    persistent_snn_forward,
)
from btorch.sparse import BinaryEvents, CSR, event_sparse_mm  # noqa: E402


Provider = Literal[
    "event_pre_span",
    "event_post_span",
    "torch_compile_dense",
    "persistent",
]

PROVIDERS: tuple[Provider, ...] = (
    "event_pre_span",
    "event_post_span",
    "torch_compile_dense",
    "persistent",
)


@dataclass(frozen=True)
class BenchCase:
    """RSNN benchmark parameter bundle."""

    n_neuron: int
    batch_size: int
    t_steps: int
    fanout: int
    event_rate: float
    dt: float = 1.0
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    input_amplitude: float = 30.0
    weight_scale: float = 0.15

    @property
    def edge_count(self) -> int:
        return self.n_neuron * self.fanout

    @property
    def expected_external_events(self) -> int:
        return int(round(self.t_steps * self.batch_size * self.n_neuron * self.event_rate))


@dataclass(frozen=True)
class RSNNResult:
    """Output and final state for one RSNN provider."""

    spikes: torch.Tensor
    v: torch.Tensor
    psc: torch.Tensor


def benchmark_fig_path() -> Path:
    """Return this benchmark script's figure output directory."""

    path = REPO_ROOT / "fig" / "benchmark" / Path(__file__).with_suffix("").name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def provider_available(provider: Provider, device: torch.device) -> tuple[bool, str]:
    """Return whether a provider can run in this environment."""

    if provider in ("event_pre_span", "event_post_span"):
        if device.type != "cuda":
            return False, "requires_cuda"
        if not _triton_available():
            return False, "requires_triton"
        return True, "ok"
    if provider == "torch_compile_dense":
        if not hasattr(torch, "compile"):
            return False, "torch_compile_missing"
        if platform.system() != "Linux":
            return False, "torch_compile_unsupported_platform"
        return True, "ok"
    return True, "ok"


def make_input_sequence(case: BenchCase, device: torch.device) -> torch.Tensor:
    """Create deterministic sparse external input currents."""

    total = case.t_steps * case.batch_size * case.n_neuron
    n_active = max(0, min(total, int(round(total * case.event_rate))))
    flat = torch.zeros(total, device=device, dtype=torch.float32)
    if n_active > 0:
        # A low-discrepancy deterministic stride avoids RNG noise between runs.
        stride = max(1, total // max(n_active, 1))
        active = (torch.arange(n_active, device=device) * stride) % total
        flat[active.long()] = case.input_amplitude
    return flat.reshape(case.t_steps, case.batch_size, case.n_neuron)


def make_recurrent_csr(case: BenchCase, device: torch.device) -> CSR:
    """Create one fixed-fanout recurrent matrix used by all providers."""

    if case.fanout == 0:
        row = torch.empty((0,), device=device, dtype=torch.long)
        col = torch.empty((0,), device=device, dtype=torch.long)
        data = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        edge = torch.arange(case.edge_count, device=device, dtype=torch.long)
        row = torch.div(edge, case.fanout, rounding_mode="floor")
        slot = edge.remainder(case.fanout)
        col = (row + slot + 1).remainder(case.n_neuron)
        data = torch.full(
            (case.edge_count,),
            case.weight_scale / max(case.fanout, 1),
            device=device,
            dtype=torch.float32,
        )
    return CSR.from_edges(row=row, col=col, data=data, shape=(case.n_neuron, case.n_neuron))


def csr_to_dense(matrix: CSR) -> torch.Tensor:
    """Materialize a CSR matrix as dense ``(N_pre, N_post)`` weights."""

    n_pre, n_post = matrix.shape
    dense = torch.zeros(n_pre, n_post, device=matrix.data.device, dtype=matrix.data.dtype)
    indptr = matrix.indptr
    for pre in range(n_pre):
        start = int(indptr[pre].item())
        end = int(indptr[pre + 1].item())
        if end > start:
            dense[pre, matrix.indices[start:end]] = matrix.effective_values()[start:end]
    return dense


def csr_to_persistent_graph(matrix: CSR) -> EventCSRGraph:
    """Convert the benchmark CSR graph to the persistent scaffold graph."""

    return EventCSRGraph(
        indptr=matrix.indptr.to(torch.int32).contiguous(),
        indices=matrix.indices.to(torch.int32).contiguous(),
        weight=matrix.effective_values().to(torch.float32).contiguous(),
        delay=torch.zeros_like(matrix.indices, dtype=torch.int32).contiguous(),
        shape=matrix.shape,
    )


def dense_to_windowed_events(x_seq: torch.Tensor, threshold: float = 0.0) -> WindowedSpikeEvents:
    """Convert dense ``(T, B, N)`` events to time-batch bucketed events."""

    t_steps, batch_size, n_neuron = x_seq.shape
    active = x_seq > threshold
    counts = active.sum(dim=2, dtype=torch.int32).reshape(-1)
    offsets = torch.zeros(counts.numel() + 1, device=x_seq.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0)
    indices = torch.nonzero(active.reshape(-1, n_neuron), as_tuple=False)[:, 1]
    indices = indices.to(torch.int32).contiguous()
    values = x_seq[active].to(torch.float32).contiguous()
    return WindowedSpikeEvents(
        offsets=offsets.contiguous(),
        indices=indices,
        values=values,
        shape=(t_steps, batch_size, n_neuron),
    )


def lif_fire_and_update(
    v: torch.Tensor,
    current: torch.Tensor,
    case: BenchCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step deterministic LIF update used by benchmark providers."""

    dv = case.dt * (-(v - case.v_reset) / case.tau_mem + current)
    v_pre = v + dv
    spikes = (v_pre >= case.v_threshold).to(v.dtype)
    v_next = torch.where(
        spikes > 0,
        torch.full_like(v_pre, case.v_reset),
        v_pre,
    )
    return spikes, v_next


def dense_rsnn_forward(
    x_seq: torch.Tensor,
    weight_dense: torch.Tensor,
    case: BenchCase,
) -> RSNNResult:
    """Dense reference RSNN forward."""

    batch_size, n_neuron = x_seq.shape[1], x_seq.shape[2]
    v = torch.zeros(batch_size, n_neuron, device=x_seq.device, dtype=x_seq.dtype)
    psc = torch.zeros_like(v)
    decay = math.exp(-case.dt / case.tau_syn)
    spikes = []
    for t in range(case.t_steps):
        z, v = lif_fire_and_update(v, psc + x_seq[t], case)
        psc = psc * decay + z @ weight_dense
        spikes.append(z)
    return RSNNResult(spikes=torch.stack(spikes, dim=0), v=v, psc=psc)


def event_rsnn_forward(
    x_seq: torch.Tensor,
    matrix: CSR,
    case: BenchCase,
    *,
    schedule: Literal["pre_span", "post_span"],
) -> RSNNResult:
    """Event-driven RSNN forward using the main btorch sparse event path."""

    batch_size, n_neuron = x_seq.shape[1], x_seq.shape[2]
    v = torch.zeros(batch_size, n_neuron, device=x_seq.device, dtype=x_seq.dtype)
    psc = torch.zeros_like(v)
    decay = math.exp(-case.dt / case.tau_syn)
    spikes = []
    for t in range(case.t_steps):
        z, v = lif_fire_and_update(v, psc + x_seq[t], case)
        recurrent = event_sparse_mm(matrix, BinaryEvents(z), schedule=schedule)
        psc = psc * decay + recurrent
        spikes.append(z)
    return RSNNResult(spikes=torch.stack(spikes, dim=0), v=v, psc=psc)


def persistent_rsnn_forward(
    x_seq: torch.Tensor,
    matrix: CSR,
    case: BenchCase,
    *,
    backend: str,
) -> RSNNResult:
    """Persistent scaffold provider.

    The current scaffold is a no-op and therefore reports ``stub_only`` in
    correctness metadata. The data conversion here is the future CUDA contract.
    """

    events = dense_to_windowed_events(x_seq)
    graph = csr_to_persistent_graph(matrix)
    state = make_empty_state(case.batch_size, case.n_neuron, device=x_seq.device)
    params = PersistentSNNParams(
        dt=case.dt,
        tau_mem=case.tau_mem,
        tau_syn=case.tau_syn,
        v_threshold=case.v_threshold,
        v_reset=case.v_reset,
        window_size=case.t_steps,
    )
    out = persistent_snn_forward(
        events,
        graph,
        state,
        params,
        backend=backend,
        return_mode="dense",
    )
    assert out.spikes is not None
    return RSNNResult(spikes=out.spikes, v=out.state.v, psc=out.state.psc)


def provider_forward(
    provider: Provider,
    x_seq: torch.Tensor,
    matrix: CSR,
    weight_dense: torch.Tensor,
    case: BenchCase,
    *,
    persistent_backend: str,
):
    """Dispatch one provider forward."""

    if provider == "event_pre_span":
        return event_rsnn_forward(x_seq, matrix, case, schedule="pre_span")
    if provider == "event_post_span":
        return event_rsnn_forward(x_seq, matrix, case, schedule="post_span")
    if provider == "torch_compile_dense":
        compiled = torch.compile(dense_rsnn_forward)
        return compiled(x_seq, weight_dense, case)
    if provider == "persistent":
        return persistent_rsnn_forward(x_seq, matrix, case, backend=persistent_backend)
    raise ValueError(f"Unknown provider: {provider}.")


def correctness_status(
    provider: Provider,
    result: RSNNResult | None,
    reference: RSNNResult,
) -> tuple[str, float]:
    """Compare a provider result to dense eager reference."""

    if result is None:
        return "skipped", float("nan")
    if provider == "persistent":
        return "stub_only", float("nan")
    spike_diff = (result.spikes - reference.spikes).abs().max().item()
    v_diff = (result.v - reference.v).abs().max().item()
    psc_diff = (result.psc - reference.psc).abs().max().item()
    max_diff = float(max(spike_diff, v_diff, psc_diff))
    torch.testing.assert_close(result.spikes, reference.spikes, atol=0, rtol=0)
    torch.testing.assert_close(result.v, reference.v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.psc, reference.psc, atol=1e-5, rtol=1e-5)
    return "passed", max_diff


def time_ms(fn, *, warmup: int, repeat: int, device: torch.device) -> float:
    """Benchmark a callable with CUDA synchronization when needed."""

    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(torch.median(torch.tensor(samples, dtype=torch.float64)).item())


def _empty_row(
    case: BenchCase,
    provider: Provider,
    device: torch.device,
    status: str,
) -> dict[str, float | int | str]:
    return {
        "provider": provider,
        "device": device.type,
        "n_neuron": case.n_neuron,
        "batch_size": case.batch_size,
        "t_steps": case.t_steps,
        "fanout": case.fanout,
        "event_rate": case.event_rate,
        "event_count": 0,
        "edge_count": case.edge_count,
        "correctness_status": status,
        "correctness_max_abs_diff": float("nan"),
        "latency_ms": float("nan"),
        "speedup_vs_compile": float("nan"),
    }


def bench_case(
    case: BenchCase,
    *,
    device: torch.device,
    providers: tuple[Provider, ...],
    persistent_backend: str,
    warmup: int,
    repeat: int,
    skip_correctness: bool,
) -> list[dict[str, float | int | str]]:
    """Measure all requested providers for one case."""

    x_seq = make_input_sequence(case, device)
    matrix = make_recurrent_csr(case, device)
    weight_dense = csr_to_dense(matrix)
    reference = dense_rsnn_forward(x_seq, weight_dense, case)
    event_count = int(torch.count_nonzero(x_seq).item())

    rows: list[dict[str, float | int | str]] = []
    for provider in providers:
        available, reason = provider_available(provider, device)
        if not available:
            row = _empty_row(case, provider, device, f"skipped:{reason}")
            row["event_count"] = event_count
            rows.append(row)
            continue

        def op():
            return provider_forward(
                provider,
                x_seq,
                matrix,
                weight_dense,
                case,
                persistent_backend=persistent_backend,
            )

        try:
            result = None if skip_correctness else op()
            if skip_correctness:
                status, max_diff = "not_checked", float("nan")
            else:
                status, max_diff = correctness_status(provider, result, reference)
            latency = time_ms(op, warmup=warmup, repeat=repeat, device=device)
        except Exception as exc:
            row = _empty_row(case, provider, device, f"error:{type(exc).__name__}")
            row["event_count"] = event_count
            rows.append(row)
            print(f"Provider {provider} failed for N={case.n_neuron}: {exc}")
            continue

        rows.append(
            {
                "provider": provider,
                "device": device.type,
                "n_neuron": case.n_neuron,
                "batch_size": case.batch_size,
                "t_steps": case.t_steps,
                "fanout": case.fanout,
                "event_rate": case.event_rate,
                "event_count": event_count,
                "edge_count": case.edge_count,
                "correctness_status": status,
                "correctness_max_abs_diff": max_diff,
                "latency_ms": latency,
                "speedup_vs_compile": float("nan"),
            }
        )

    compile_latencies = {
        (
            row["n_neuron"],
            row["batch_size"],
            row["t_steps"],
            row["fanout"],
            row["event_rate"],
        ): float(row["latency_ms"])
        for row in rows
        if row["provider"] == "torch_compile_dense"
        and not math.isnan(float(row["latency_ms"]))
    }
    for row in rows:
        key = (
            row["n_neuron"],
            row["batch_size"],
            row["t_steps"],
            row["fanout"],
            row["event_rate"],
        )
        baseline = compile_latencies.get(key, float("nan"))
        latency = float(row["latency_ms"])
        if not math.isnan(baseline) and not math.isnan(latency) and latency > 0:
            row["speedup_vs_compile"] = baseline / latency
    return rows


def sweep_cases(args) -> list[BenchCase]:
    """Build a sweep grid for latency comparison plots."""

    return [
        BenchCase(
            n_neuron=n_neuron,
            batch_size=args.batch_size,
            t_steps=t_steps,
            fanout=fanout,
            event_rate=event_rate,
            dt=args.dt,
            tau_mem=args.tau_mem,
            tau_syn=args.tau_syn,
            v_threshold=args.v_threshold,
            input_amplitude=args.input_amplitude,
            weight_scale=args.weight_scale,
        )
        for n_neuron, t_steps, fanout, event_rate in itertools.product(
            args.n_neuron,
            args.t_steps,
            args.fanout,
            args.event_rate,
        )
    ]


def save_csv(rows: list[dict[str, float | int | str]], output_dir: Path) -> Path:
    """Save benchmark rows as CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "persistent_snn_latency.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _finite_rows(rows, y_key):
    return [row for row in rows if not math.isnan(float(row[y_key]))]


def plot_latency(rows: list[dict[str, float | int | str]], output_dir: Path) -> Path:
    """Plot latency and speedup comparisons across providers."""

    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    specs = [
        ("n_neuron", "latency_ms", "Latency vs N", "N", "Median latency (ms)"),
        (
            "event_rate",
            "latency_ms",
            "Latency vs input event rate",
            "Input event rate",
            "Median latency (ms)",
        ),
        (
            "event_rate",
            "speedup_vs_compile",
            "Speedup vs torch.compile",
            "Input event rate",
            "Speedup",
        ),
    ]
    colors = {
        "event_pre_span": "#1f77b4",
        "event_post_span": "#ff7f0e",
        "torch_compile_dense": "#2ca02c",
        "persistent": "#9467bd",
    }
    markers = {
        "event_pre_span": "o",
        "event_post_span": "s",
        "torch_compile_dense": "^",
        "persistent": "x",
    }

    for ax, (x_key, y_key, title, xlabel, ylabel) in zip(axes, specs):
        for provider in PROVIDERS:
            provider_rows = [
                row for row in _finite_rows(rows, y_key) if row["provider"] == provider
            ]
            if not provider_rows:
                continue
            if x_key == "n_neuron":
                fixed = min(
                    {
                        (row["event_rate"], row["fanout"], row["t_steps"])
                        for row in provider_rows
                    },
                    key=str,
                )
                provider_rows = [
                    row
                    for row in provider_rows
                    if (row["event_rate"], row["fanout"], row["t_steps"]) == fixed
                ]
            else:
                fixed = min(
                    {
                        (row["n_neuron"], row["fanout"], row["t_steps"])
                        for row in provider_rows
                    },
                    key=str,
                )
                provider_rows = [
                    row
                    for row in provider_rows
                    if (row["n_neuron"], row["fanout"], row["t_steps"]) == fixed
                ]
            provider_rows = sorted(provider_rows, key=lambda row: float(row[x_key]))
            ax.plot(
                [float(row[x_key]) for row in provider_rows],
                [float(row[y_key]) for row in provider_rows],
                label=provider,
                marker=markers[provider],
                linewidth=2,
                color=colors[provider],
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if x_key == "n_neuron":
            ax.set_xscale("log", base=2)
        if y_key == "latency_ms":
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)

    fig.suptitle("RSNN LIF + ExponentialPSC Provider Comparison", fontweight="bold")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "persistent_snn_latency_sweep.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--persistent-backend",
        default="auto",
        choices=["auto", "torch_stub", "cuda_persistent"],
        help="Persistent operator backend.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-neuron", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--t-steps", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--fanout", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument(
        "--event-rate",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02],
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--tau-syn", type=float, default=5.0)
    parser.add_argument("--v-threshold", type=float, default=1.0)
    parser.add_argument("--input-amplitude", type=float, default=30.0)
    parser.add_argument("--weight-scale", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)
    output_dir = benchmark_fig_path()
    cases = sweep_cases(args)
    providers = tuple(args.providers)

    rows = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}/{len(cases)}] N={case.n_neuron} T={case.t_steps} "
            f"fanout={case.fanout} event_rate={case.event_rate}"
        )
        rows.extend(
            bench_case(
                case,
                device=device,
                providers=providers,
                persistent_backend=args.persistent_backend,
                warmup=args.warmup,
                repeat=args.repeat,
                skip_correctness=args.skip_correctness,
            )
        )

    csv_path = save_csv(rows, output_dir)
    fig_path_out = plot_latency(rows, output_dir)
    print(f"Benchmark CSV saved to {csv_path}")
    print(f"Latency sweep plot saved to {fig_path_out}")


if __name__ == "__main__":
    main()
