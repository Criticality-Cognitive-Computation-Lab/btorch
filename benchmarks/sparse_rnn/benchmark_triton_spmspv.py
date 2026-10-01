"""Compare native and Triton sparse paths on a large connectome.

The benchmark uses the local Hemibrain edge list by default.  It reports the
one-off connection construction/preparation cost separately from steady-state
GPU execution for both a pure SpMSpV trace and a complete recurrent
LIF--exponential-PSC simulation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import statistics
import sys
import time
import zipfile
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.sparse
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from btorch._sparse_config import TritonSparseConfig  # noqa: E402
from btorch.models import environ  # noqa: E402
from btorch.models.functional import (  # noqa: E402
    init_net_state,
    prepare_sparse_modules,
    reset_net,
)
from btorch.models.linear import SparseConn  # noqa: E402
from btorch.models.neurons import LIF  # noqa: E402
from btorch.models.rnn import RecurrentNN  # noqa: E402
from btorch.models.synapse import ExponentialPSC  # noqa: E402


DEFAULT_DATASET = Path(
    "/home/lenovo/connectome_dataset/data/skewed/"
    "fly_hemibrain/fly_hemibrain.csv.zip"
)
DEFAULT_RATES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument("--t-steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=0.01,
        help=(
            "Mean absolute recurrent fanout strength. The unsigned Hemibrain "
            "graph requires a subcritical value to preserve the rate sweep."
        ),
    )
    parser.add_argument("--external-amplitude", type=float, default=1.05)
    parser.add_argument("--tau-m", type=float, default=20.0)
    parser.add_argument("--tau-syn", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/sparse_rnn/results/hemibrain_rtx4060.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("benchmarks/sparse_rnn/results/hemibrain_rtx4060.json"),
    )
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")
    if any(rate <= 0.0 or rate >= 1.0 for rate in args.rates):
        parser.error("rates must lie in (0, 1)")
    if args.t_steps <= 0 or args.warmup < 0 or args.repeat <= 0:
        parser.error("invalid timing counts")
    return args


def load_csv_zip(path: Path) -> scipy.sparse.csr_array:
    """Load a networks.skewed.de source,target,weight edge table."""

    with zipfile.ZipFile(path) as archive:
        member = next(
            name for name in archive.namelist() if "edges" in name.lower()
        )
        rows: list[int] = []
        columns: list[int] = []
        weights: list[float] = []
        node_ids: dict[str, int] = {}
        with archive.open(member) as edge_file:
            for raw_line in edge_file:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split(",")
                source_name, target_name = fields[0].strip(), fields[1].strip()
                source = node_ids.setdefault(source_name, len(node_ids))
                target = node_ids.setdefault(target_name, len(node_ids))
                weight = 1.0
                for value in fields[2:]:
                    try:
                        weight = float(value.strip())
                        break
                    except ValueError:
                        continue
                rows.append(source)
                columns.append(target)
                weights.append(weight)

    n_neuron = len(node_ids)
    matrix = scipy.sparse.csr_array(
        (
            np.asarray(weights, dtype=np.float32),
            (np.asarray(rows), np.asarray(columns)),
        ),
        shape=(n_neuron, n_neuron),
    )
    matrix.sum_duplicates()
    return matrix


def normalize_weights(
    matrix: scipy.sparse.csr_array, recurrent_strength: float
) -> scipy.sparse.csr_array:
    matrix = matrix.copy()
    mean_abs_fanout = float(abs(matrix).sum()) / matrix.shape[0]
    matrix.data *= recurrent_strength / max(mean_abs_fanout, 1.0)
    return matrix


def make_trace(
    t_steps: int, n_neuron: int, rate: float, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.rand(
            (t_steps, 1, n_neuron), device=device, generator=generator
        )
        < rate
    ).to(torch.float32)


def percentile(samples: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(samples), q))


def time_cuda(
    run,
    *,
    warmup: int,
    repeat: int,
    before_each=None,
) -> list[float]:
    """Time one callable with CUDA events, returning milliseconds."""

    for _ in range(warmup):
        if before_each is not None:
            before_each()
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        if before_each is not None:
            before_each()
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def capture_cuda_graph(run, *, capture_warmup: int = 3) -> torch.cuda.CUDAGraph:
    """Capture a fixed-shape callable after allocator/kernel warmup."""

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(capture_warmup):
            run()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


def make_connection(
    matrix: scipy.sparse.csr_array,
    backend: str,
    device: torch.device,
) -> tuple[SparseConn, float, float]:
    config = TritonSparseConfig() if backend == "triton" else None
    start = time.perf_counter()
    connection = SparseConn(
        matrix,
        enforce_dale=False,
        sparse_backend=backend,
        sparse_config=config,
        device=device,
        dtype=torch.float32,
    )
    torch.cuda.synchronize()
    construction_ms = (time.perf_counter() - start) * 1_000.0

    start = time.perf_counter()
    connection.prepare_sparse(batch_size=1)
    torch.cuda.synchronize()
    prepare_ms = (time.perf_counter() - start) * 1_000.0
    connection.finish_sparse()
    return connection, construction_ms, prepare_ms


def summarize_samples(samples: list[float]) -> dict[str, float]:
    return {
        "latency_ms": statistics.median(samples),
        "latency_mean_ms": statistics.mean(samples),
        "latency_min_ms": min(samples),
        "latency_p95_ms": percentile(samples, 95.0),
        "latency_std_ms": statistics.pstdev(samples),
    }


def operator_case(
    connection: SparseConn,
    trace: torch.Tensor,
    *,
    warmup: int,
    repeat: int,
) -> tuple[dict[str, float], torch.Tensor]:
    output = None

    def run() -> None:
        nonlocal output
        for timestep in range(trace.shape[0]):
            output = connection(trace[timestep])

    context = (
        prepare_sparse_modules(connection, batch_size=1)
        if connection.sparse_backend == "triton"
        else nullcontext()
    )
    with torch.inference_mode(), context:
        graph = capture_cuda_graph(run)
        samples = time_cuda(
            graph.replay, warmup=warmup, repeat=repeat
        )
        output = output.clone()
    assert output is not None
    summary = summarize_samples(samples)
    summary["latency_per_step_us"] = summary["latency_ms"] * 1_000 / len(trace)
    return summary, output


def make_snn(
    matrix: scipy.sparse.csr_array,
    backend: str,
    device: torch.device,
    *,
    tau_m: float,
    tau_syn: float,
) -> tuple[RecurrentNN, float, float]:
    connection, construction_ms, prepare_ms = make_connection(
        matrix, backend, device
    )
    neuron = LIF(
        matrix.shape[0],
        tau=tau_m,
        hard_reset=True,
        device=device,
        dtype=torch.float32,
    )
    synapse = ExponentialPSC(
        matrix.shape[0], tau_syn=tau_syn, linear=connection
    ).to(device=device, dtype=torch.float32)
    model = RecurrentNN(neuron, synapse, unroll=8, cudagraph=True).to(
        device=device, dtype=torch.float32
    )
    init_net_state(
        model, batch_size=1, device=device, dtype=torch.float32
    )
    return model, construction_ms, prepare_ms


def snn_case(
    model: RecurrentNN,
    drive: torch.Tensor,
    *,
    warmup: int,
    repeat: int,
) -> tuple[dict[str, float], torch.Tensor]:
    output = None

    def reset() -> None:
        reset_net(model, batch_size=1, inplace=True)

    def run() -> None:
        nonlocal output
        output, _ = model.multi_step_forward(drive)

    connection = model.synapse.linear
    context = (
        prepare_sparse_modules(model, batch_size=1)
        if connection.sparse_backend == "triton"
        else nullcontext()
    )
    with torch.inference_mode(), environ.context(dt=1.0), context:
        samples = time_cuda(
            run, warmup=warmup, repeat=repeat, before_each=reset
        )
    assert output is not None
    summary = summarize_samples(samples)
    summary["latency_per_step_us"] = summary["latency_ms"] * 1_000 / len(drive)
    return summary, output


def device_metadata() -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "gpu_memory_bytes": props.total_memory,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": importlib.metadata.version("triton"),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")

    load_start = time.perf_counter()
    matrix = normalize_weights(load_csv_zip(args.dataset), args.weight_scale)
    load_ms = (time.perf_counter() - load_start) * 1_000.0
    n_neuron = matrix.shape[0]
    if n_neuron < 10_000:
        raise ValueError(f"dataset is too small: n_neuron={n_neuron}")

    traces = {
        rate: make_trace(args.t_steps, n_neuron, rate, args.seed, device)
        for rate in args.rates
    }
    rows: list[dict[str, object]] = []
    correctness: dict[str, dict[str, float]] = {}

    for scope in ("spmspv", "snn"):
        reference_by_rate: dict[float, torch.Tensor] = {}
        for backend in ("native", "triton"):
            if scope == "spmspv":
                module, construction_ms, prepare_ms = make_connection(
                    matrix, backend, device
                )
            else:
                module, construction_ms, prepare_ms = make_snn(
                    matrix,
                    backend,
                    device,
                    tau_m=args.tau_m,
                    tau_syn=args.tau_syn,
                )
            for rate, trace in traces.items():
                measured_input_rate = float(trace.mean())
                if scope == "spmspv":
                    metrics, output = operator_case(
                        module,
                        trace,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                    measured_output_rate = float("nan")
                else:
                    metrics, output = snn_case(
                        module,
                        trace * args.external_amplitude,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                    measured_output_rate = float(output.mean())

                key = f"{scope}:{rate:g}"
                if backend == "native":
                    reference_by_rate[rate] = output.detach().clone()
                    max_abs_error = 0.0
                else:
                    difference = (output - reference_by_rate[rate]).abs()
                    max_abs_error = float(difference.max())
                    mismatch_rate = float(
                        (~torch.isclose(
                            output,
                            reference_by_rate[rate],
                            atol=1e-6,
                            rtol=1e-5,
                        ))
                        .float()
                        .mean()
                    )
                    correctness[key] = {
                        "max_abs_error": max_abs_error,
                        "mismatch_rate": mismatch_rate,
                    }

                rows.append(
                    {
                        "scope": scope,
                        "execution_mode": "cuda_graph",
                        "backend": backend,
                        "requested_spike_rate": rate,
                        "measured_input_rate": measured_input_rate,
                        "measured_snn_output_rate": measured_output_rate,
                        "n_neuron": n_neuron,
                        "n_edges": matrix.nnz,
                        "t_steps": args.t_steps,
                        "construction_ms": construction_ms,
                        "prepare_ms": prepare_ms,
                        "max_abs_error_vs_native": max_abs_error,
                        **metrics,
                    }
                )
            del module
            torch.cuda.empty_cache()

    native_latency = {
        (row["scope"], row["requested_spike_rate"]): row["latency_ms"]
        for row in rows
        if row["backend"] == "native"
    }
    for row in rows:
        baseline = native_latency[(row["scope"], row["requested_spike_rate"])]
        row["speedup_vs_native"] = baseline / row["latency_ms"]

    output_path = (REPO_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        **device_metadata(),
        "dataset": str(args.dataset.resolve()),
        "dataset_kind": "real Hemibrain connectome",
        "n_neuron": n_neuron,
        "n_edges": matrix.nnz,
        "mean_fanout": matrix.nnz / n_neuron,
        "dataset_load_ms": load_ms,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "triton_config": asdict(TritonSparseConfig()),
        "correctness": correctness,
        "results_csv": str(output_path),
    }
    summary_path = (REPO_ROOT / args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
