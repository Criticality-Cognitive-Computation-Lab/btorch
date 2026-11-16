"""Compare fresh cold/warm Izhikevich inference, BPTT and online STDP.

Run from the repository root with ``python -m benchmarks.rsnn_izhikevich.run``.
Each case runs in a fresh process. Disk compilation caches are intentionally
retained; no complete sequence is run before the cold measurement.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from btorch.models import environ
from examples.rsnn_common import ExampleOptions, stimulus
from examples.rsnn_izhikevich import build_model as build_training_model
from examples.rsnn_izhikevich_stdp import NetworkOptions, build_model, initialize


@dataclass(frozen=True)
class Case:
    """Keep physical parameters identical across execution strategies."""

    neurons: int = 4096
    edges: int = round(4096**2 * 33_072_348 / 60_000**2)
    mode: str = "inference"
    strategy: str = "torch-eager"
    backend: str = "triton"
    device: str = "cuda"
    dtype: str = "float32"
    steps: int = 5000
    repeats: int = 3
    chunk_size: int = 200
    chunk_mode: str = "split"
    seed: int = 42
    dt: float = 1.0
    input_scale: float = 0.6640625
    grad_checkpoint: bool = True
    cpu_offload: bool = False


STRATEGIES = ("torch-eager", "torch-graph", "push-eager", "push-graph", "helion-graph")


def measure(case: Case) -> dict:
    """Measure complete calls, resetting states and weights outside timers.

    Training includes forward, loss, full cross-chunk backward and one
    SGD update. Online STDP includes its weight updates and never
    invokes BPTT. All paths retain the example's neuron/PSC recordings.
    CUDA Graph capture uses RecurrentNN's bounded chunk preparation,
    including remainder shapes.
    """
    result = {"case": asdict(case)}
    if case.mode == "training" and case.strategy == "torch-graph":
        return dict(
            result,
            status="unsupported",
            reason=(
                "RecurrentNN's Torch CUDA Graph runner is inference-only; "
                "bounded forward/reverse training graphs require a native backend"
            ),
        )
    if case.mode == "training" and case.strategy == "helion-graph":
        return dict(result, status="unsupported", reason="HELION has no BPTT/VJP")
    if case.strategy.endswith("graph") and torch.device(case.device).type != "cuda":
        return dict(result, status="unsupported", reason="CUDA Graph requires CUDA")
    if case.strategy.startswith("push") and case.device == "cpu":
        return dict(result, status="unsupported", reason="Native push requires CUDA")
    device = torch.device(case.device)
    native = not case.strategy.startswith("torch")
    backend = case.backend if native else "torch"
    graph = case.strategy.endswith("graph")
    dtype = getattr(torch, case.dtype)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    started = time.perf_counter()
    options = NetworkOptions(
        neurons=case.neurons,
        edges=case.edges,
        seed=case.seed,
        device=case.device,
        dtype=dtype,
        neuron_backend=backend,
        sparse_backend=backend,
        method="push" if native else "pull",
        cudagraph=graph,
        chunk_size=case.chunk_size,
        chunk_mode=case.chunk_mode,
    )
    if case.mode == "training":
        # Reuse the example's sparse surrogate and learnable neuron/edge
        # parameters. Static and STDP builders share the physical circuit.
        training_options = ExampleOptions(
            **{key: value for key, value in asdict(options).items() if key != "dtype"}
        )
        network = build_training_model(training_options).to(dtype=dtype)
        network.grad_checkpoint = case.grad_checkpoint
        network.cpu_offload = case.cpu_offload
        network.train()
        optimizer = torch.optim.SGD(network.parameters(), lr=1e-3, foreach=False)
    else:
        network = build_model(options, plasticity=case.mode == "stdp")
        network.eval()
        optimizer = None
    network.backend = replace(
        network.backend,
        recurrent="helion" if case.strategy.startswith("helion") else "standard",
        spike_selection="spike_list" if native else "auto",
    )
    # Evaluate the spatial phase in FP32 before rounding to half/BF16 storage.
    inputs = (
        (stimulus("izhikevich", 1, case.neurons, case.device) * case.input_scale)
        .to(dtype=dtype)
        .expand(case.steps, -1, -1)
    )
    parameters = [
        (parameter, parameter.detach().clone()) for parameter in network.parameters()
    ]
    weight = network.synapse.linear.weight.value
    initial_weight = weight.detach().clone()
    synchronize()
    result["construction_ms"] = (time.perf_counter() - started) * 1000
    timings, peaks, resets, captures = [], [], [], []
    validation_ms = 0.0
    rate = None
    context = nullcontext if case.mode == "training" else torch.no_grad
    with environ.context(dt=case.dt), context():
        for _ in range(1 + case.repeats):
            reset_started = time.perf_counter()
            with torch.no_grad():
                for parameter, initial in parameters:
                    parameter.copy_(initial)
                weight.copy_(initial_weight)
                initialize(network, case.device)
            # SGD has no momentum or other state to reset. Graph caches stay
            # live across trials; physical/learning memories do not.
            synchronize()
            resets.append((time.perf_counter() - reset_started) * 1000)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            spikes, records = network(inputs)
            if optimizer is not None:
                target = 5.0 * case.dt / 1000
                loss = (spikes.mean(0) - target).square().mean()
                loss = loss + 1e-5 * records["neuron.v"].square().mean()
                loss.backward()
                optimizer.step()
            synchronize()
            timings.append((time.perf_counter() - started) * 1000)
            captures.append(network._cudagraph_runner.n_captures)
            peaks.append(
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            )
            started = time.perf_counter()
            # Validation happens after each timed call. This is a health check,
            # not a claim of bitwise parity for atomic sparse propagation.
            if not all(
                torch.isfinite(value).all().item()
                for value in (spikes, weight, *records.values())
            ):
                raise ArithmeticError("Non-finite spikes, weights or recorded state")
            rate = spikes.float().mean().item() * 1000 / case.dt
            if optimizer is not None:
                gradients = [p.grad for p in network.parameters() if p.grad is not None]
                if not gradients or not all(
                    torch.isfinite(g).all().item() for g in gradients
                ):
                    raise ArithmeticError("Missing or non-finite training gradients")
                result["loss"] = loss.item()
                del loss
            synchronize()
            validation_ms += (time.perf_counter() - started) * 1000
            del spikes, records
    result.update(
        status="ok",
        edges=network.synapse.linear.nnz,
        firing_hz=rate,
        cold_ms=timings[0],
        warm_ms=statistics.median(timings[1:]),
        warm_samples_ms=timings[1:],
        warm_us_per_step=statistics.median(timings[1:]) * 1000 / case.steps,
        peak_cuda_allocated_bytes=max(
            (p for p in peaks if p is not None), default=None
        ),
        reset_ms=resets,
        forward_captures_after_each_call=captures,
        validation_ms=validation_ms,
        weight_abs_change_max=(weight - initial_weight).abs().max().item(),
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )
    return result


def summary(results: list[dict]) -> str:
    """Report every case and matched HELION/push comparisons, including
    slowdowns."""
    lines = [
        "| Neurons | Mode | Backend | Strategy | Cold ms | Warm ms | Hz | Status |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for row in results:
        case = row["case"]
        prefix = (
            f"| {case['neurons']} | {case['mode']} | "
            f"{case['backend']} | {case['strategy']} |"
        )
        if row["status"] == "ok":
            lines.append(
                prefix
                + f" {row['cold_ms']:.3f} | {row['warm_ms']:.3f} | "
                + f"{row['firing_hz']:.3f} | ok |"
            )
        else:
            lines.append(prefix + f" — | — | — | {row['status']}: {row['reason']} |")
    ratios = []
    for row in results:
        if row["status"] != "ok" or row["case"]["strategy"] != "helion-graph":
            continue
        paired = dict(row["case"], strategy="push-graph")
        reference = next(
            (r for r in results if r["case"] == paired and r["status"] == "ok"), None
        )
        if reference:
            ratio = reference["warm_ms"] / row["warm_ms"]
            ratios.append(ratio)
            lines.append(
                f"\nHELION / push graph, {paired['neurons']} {paired['mode']} "
                f"{paired['backend']}: {ratio:.3f}×"
            )
    if ratios:
        lines.append(
            f"\nMatched HELION warm speedup geometric mean ({len(ratios)} cases): "
            f"{statistics.geometric_mean(ratios):.3f}×"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Launch each configuration in a fresh process and save raw
    measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=int, nargs="+", default=[4096, 60000])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("inference", "training", "stdp"),
        default=["inference", "training", "stdp"],
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("triton", "tilelang"),
        default=["triton", "tilelang"],
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-mode", choices=("split", "chunk"), default="split")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile/rsnn_izhikevich") / time.strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="per-case seconds, including construction",
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        case = Case(**json.loads(args.worker))
        try:
            row = measure(case)
        except Exception as error:
            traceback.print_exc()
            row = dict(
                case=asdict(case),
                status="error",
                reason=f"{type(error).__name__}: {error}",
            )
        print("RESULT " + json.dumps(row), flush=True)
        return
    if (
        min(*args.populations, args.steps, args.repeats, args.chunk_size) < 1
        or args.timeout <= 0
    ):
        parser.error(
            "positive populations, steps, repeats, chunk size and timeout required"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for neurons in args.populations:
        for mode in args.modes:
            for strategy in args.strategies:
                # Torch cases have no native backend; run them once per mode.
                backends = ["torch"] if strategy.startswith("torch") else args.backends
                for backend in backends:
                    case = Case(
                        neurons=neurons,
                        edges=round(neurons**2 * 33_072_348 / 60_000**2),
                        mode=mode,
                        strategy=strategy,
                        backend=backend,
                        device=args.device,
                        dtype=args.dtype,
                        steps=args.steps,
                        repeats=args.repeats,
                        chunk_size=args.chunk_size,
                        chunk_mode=args.chunk_mode,
                        grad_checkpoint=args.grad_checkpoint,
                        cpu_offload=args.cpu_offload,
                    )
                    label = f"{neurons}-{mode}-{backend}-{strategy}"
                    log = args.output / f"{label}.log"
                    with log.open("w") as stream:
                        try:
                            completed = subprocess.run(
                                [
                                    sys.executable,
                                    "-m",
                                    "benchmarks.rsnn_izhikevich.run",
                                    "--worker",
                                    json.dumps(asdict(case)),
                                ],
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                                timeout=args.timeout,
                                check=False,
                            )
                            rows = [
                                line.removeprefix("RESULT ")
                                for line in log.read_text().splitlines()
                                if line.startswith("RESULT ")
                            ]
                            row = (
                                json.loads(rows[-1])
                                if rows
                                else dict(
                                    case=asdict(case),
                                    status="error",
                                    reason=(
                                        f"Worker exit {completed.returncode}; see {log}"
                                    ),
                                )
                            )
                        except subprocess.TimeoutExpired:
                            row = dict(
                                case=asdict(case),
                                status="error",
                                reason=f"Timeout after {args.timeout}s; see {log}",
                            )
                    results.append(row)
                    (args.output / "results.json").write_text(
                        json.dumps(results, indent=2) + "\n"
                    )
                    (args.output / "summary.md").write_text(summary(results))
                    print(f"{label}: {row['status']}", flush=True)
    print(summary(results), end="")
    if any(row["status"] == "error" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
