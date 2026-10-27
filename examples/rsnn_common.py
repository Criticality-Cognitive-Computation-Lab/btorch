"""Share execution and CLI plumbing for the two concrete RSNN examples."""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from btorch.analysis import compute_input_scale
from btorch.models import KernelOptions, environ, functional
from btorch.models.connection import FixedIndegree, SparseConnection, Synapse
from btorch.models.connection.weight import EdgeWeight
from btorch.models.constrain import constrain_net
from btorch.models.neurons import GLIF3, Izhikevich
from btorch.models.rnn import RecurrentNN
from btorch.utils.bench import do_bench
from examples.rsnn_sonata import RSNNSonataIO


@dataclass(frozen=True)
class ExampleOptions:
    """Configure construction and execution independently of circuit IO."""

    neurons: int = 256
    density: float = 0.05
    edges: int | None = None
    seed: int = 0
    device: str = "cpu"
    neuron_backend: str = "torch"
    sparse_backend: str = "auto"
    method: str = "auto"
    cudagraph: bool = False


@dataclass(frozen=True)
class RunOptions:
    """Set example defaults independently of circuit construction."""

    steps: int = 256
    target_rate_hz: float = 7.5
    mode: str = "both"
    compare: bool = False
    recordings: Path | None = None


def build_connection(
    options: ExampleOptions,
    *,
    weight_scale: float,
    delay_bins: int = 1,
    dale: bool = False,
    inhibitory_ratio: float = 2.0,
) -> SparseConnection:
    """Build fixed-indegree connectivity with signed receptor-labelled edges.

    Node IDs equal tensor indices; pre is source and post is target.
    Receptor 0 is initially excitatory, receptor 1 inhibitory. Edge
    weights are trainable with optional Dale projection applied after
    each optimizer update.
    """
    n = options.neurons
    count = options.edges
    if count is None:
        count = n * max(1, round(n * options.density))
    if not 0 <= count <= n * n:
        raise ValueError("edges must be between zero and neurons squared")
    degree, extra = divmod(count, n)
    width = degree + bool(extra)
    # Reuse btorch's bounded-memory sampler. Uniform row prefixes remain
    # uniform subsets when some targets need one fewer source for an exact E.
    pre, post = FixedIndegree(width, allow_multapses=False).edges(
        n, n, generator=torch.Generator().manual_seed(options.seed)
    )
    if extra:
        keep = (
            torch.arange(width)[None, :] < (degree + (torch.arange(n) < extra))[:, None]
        )
        pre, post = pre[keep.flatten()], post[keep.flatten()]
    receptor = (pre % 5 == 0).long()  # receptor0 excitatory; receptor1 inhibitory
    # Keep total recurrent drive comparable as network size/density changes.
    weight_scale *= 13 / max(count / n, 1)  # reference: 256 neurons at density .05
    weights = torch.where(receptor == 0, weight_scale, -weight_scale * inhibitory_ratio)
    return SparseConnection.from_edges(
        pre,
        post,
        n,
        n,
        synapse=Synapse(
            weight=EdgeWeight(trainable=True),
            dale=dale,
            receptor=receptor,
            n_receptor=2,
            delay=(pre + post) % delay_bins if delay_bins > 1 else None,
            n_delay=delay_bins if delay_bins > 1 else None,
        ),
        values=weights,
    )


def stimulus(kind: str, steps: int, neurons: int, device: str) -> torch.Tensor:
    """Generate a deterministic sustained current with temporal/spatial
    variation."""
    lane = torch.arange(neurons, device=device)[None, None, :]
    if kind == "izhikevich":
        return (300.0 + 30.0 * torch.sin(lane * 0.01)).expand(steps, -1, -1)
    time = torch.arange(steps, device=device)[:, None, None]
    return 4.0 + 0.5 * torch.sin(time * 0.01) + 0.5 * torch.sin(lane * 0.01)


def initialize(network: RecurrentNN, device: str) -> None:
    """Initialize fresh physical memories without retaining an earlier
    graph."""
    functional.init_net_state(network, batch_size=1, device=device, dtype=torch.float32)
    with torch.no_grad():
        network.neuron.v.copy_(network.neuron.v_rest)


def compare_inference(network: RecurrentNN, inputs: torch.Tensor) -> dict[str, float]:
    """Time public push/pull calls with identical recordings and reset
    states."""
    from dataclasses import replace

    initial = functional.named_hidden_states(network)
    backend, captured = network.backend, network.cudagraph
    timings = {}
    try:
        with torch.inference_mode():
            for label, method, graph in (
                ("push eager", "push", False),
                ("pull CUDA Graph", "pull", True),
                ("push CUDA Graph", "push", True),
            ):
                functional.set_hidden_states(network, initial)
                network.clear_cudagraph()
                network.backend = replace(backend, connection_method=method)
                network.cudagraph = graph

                def trial():
                    functional.set_hidden_states(network, initial)
                    return network(inputs)

                # Settle compilation/capture before timing. Returned histories
                # are deliberately released between calls, including warm-up.
                trial()
                timings[label] = do_bench(
                    trial, warmup=1, rep=3, return_mode="median", timing_method="cpu"
                )
                print(f"{label}: {timings[label]:.3f} ms", flush=True)
        fastest = timings["push CUDA Graph"]
        print(
            f"push CUDA Graph speedup: {timings['pull CUDA Graph'] / fastest:.2f}x "
            f"over pull, {timings['push eager'] / fastest:.2f}x over push eager",
            flush=True,
        )
        print("FP32 push timing does not imply bitwise spike/state reproducibility.")
        return timings
    finally:
        functional.set_hidden_states(network, initial)
        network.backend, network.cudagraph = backend, captured


def main(
    kind: str,
    build_model: Callable[[ExampleOptions], RecurrentNN],
    *,
    defaults: ExampleOptions = ExampleOptions(),
    execution: RunOptions = RunOptions(),
    recording_view: Callable[
        [RecurrentNN, torch.Tensor, dict[str, torch.Tensor]], dict[str, torch.Tensor]
    ]
    | None = None,
) -> None:
    """Run inference/training and optional SONATA circuit import/export."""
    parser = argparse.ArgumentParser(
        description=f"{kind} heterogeneous RSNN; SONATA circuit IO"
    )
    parser.add_argument("--neurons", type=int, default=defaults.neurons)
    parser.add_argument(
        "--edges",
        type=int,
        default=defaults.edges,
        help="exact edge count; overrides density",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=defaults.density,
        help="approximate fixed-indegree fraction",
    )
    parser.add_argument("--steps", type=int, default=execution.steps)
    parser.add_argument(
        "--target-rate-hz",
        type=float,
        default=execution.target_rate_hz,
        help="population mean rate for drive calibration and training",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument(
        "--neuron-backend",
        choices=("torch", "triton", "cupy", "tilelang"),
        default=defaults.neuron_backend,
    )
    parser.add_argument(
        "--sparse-backend",
        choices=("auto", "torch", "triton", "cusparse"),
        default=defaults.sparse_backend,
    )
    parser.add_argument(
        "--method", choices=("auto", "pull", "push"), default=defaults.method
    )
    parser.add_argument(
        "--mode", choices=("inference", "train", "both"), default=execution.mode
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--dt", type=float, default=1.0, help="timestep in ms for a new circuit"
    )
    parser.add_argument(
        "--import-sonata",
        type=Path,
        metavar="CIRCUIT_CONFIG",
        help="restore circuit and dt; keep caller-selected execution backends",
    )
    parser.add_argument("--export-sonata", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--cudagraph",
        action=argparse.BooleanOptionalAction,
        default=defaults.cudagraph,
        help="capture inference; training uses --compile",
    )
    parser.add_argument(
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=execution.compare,
        help="compare push eager and push/pull CUDA Graph",
    )
    parser.add_argument(
        "--recordings",
        type=Path,
        default=execution.recordings,
        help="save the inference recordings outside timing",
    )
    args = parser.parse_args()
    if (
        args.neurons < 1
        or args.steps < 1
        or args.epochs < 1
        or not 0 < args.density <= 1
        or args.dt <= 0
        or not 0 < args.target_rate_hz < 1000 / args.dt
    ):
        parser.error("positive sizes/dt and density in (0,1] required")
    if args.cudagraph and args.device.split(":")[0] != "cuda":
        parser.error("--cudagraph requires --device cuda")
    options = ExampleOptions(
        **{name: getattr(args, name) for name in ExampleOptions.__dataclass_fields__}
    )
    adapter = RSNNSonataIO()
    if args.import_sonata:
        loaded = adapter.import_network(
            args.import_sonata,
            device=args.device,
            backend=KernelOptions(
                neuron=args.neuron_backend,
                connection=args.sparse_backend,
                connection_method=args.method,
            ),
        )
        network, dt = loaded.network, loaded.dt
        expected_type = GLIF3 if kind == "glif3" else Izhikevich
        if type(network.neuron) is not expected_type:
            parser.error("Imported circuit does not match this example")
        network.cudagraph = args.cudagraph
    else:
        network, dt = build_model(options), args.dt
    n = network.neuron.n_neuron[0]
    # Inputs and spikes are [time, batch, neuron]; currents are in pA.
    inputs = stimulus(kind, args.steps, n, args.device)
    initialize(network, args.device)
    # Constant Izh drive needs only one stored row. Calibrate a short window;
    # the full inference run below separately checks its measured rate.
    scale, rate = compute_input_scale(
        network, inputs[:512], target_hz=args.target_rate_hz, dt_ms=dt
    )
    inputs = (
        (inputs[:1] * scale).expand_as(inputs)
        if kind == "izhikevich"
        else inputs * scale
    )
    print(
        f"calibration: target={args.target_rate_hz:g}Hz measured={rate:.3f}Hz",
        flush=True,
    )
    # Recording every state is useful for this tutorial but not required for
    # native fused inference. Large runs should disable unneeded records.
    run = (
        torch.compile(network, fullgraph=True, mode="reduce-overhead")
        if args.compile
        else network
    )
    with environ.context(dt=dt):
        if args.mode in ("inference", "both"):
            network.eval()
            timings = {}
            if args.compare:
                if (
                    args.device.split(":")[0] != "cuda"
                    or args.neuron_backend == "torch"
                ):
                    parser.error("--compare requires CUDA and a native neuron backend")
                print(
                    f"N={n}, E={network.synapse.linear.nnz}, T={args.steps}; "
                    "identical state recordings",
                    flush=True,
                )
                timings = compare_inference(network, inputs)
            initialize(network, args.device)
            with torch.inference_mode():
                spikes, records = run(inputs)
                rate = spikes.mean().item() * 1000 / dt
                print(f"inference: firing={rate:.3f}Hz", flush=True)
                if abs(rate - args.target_rate_hz) > 0.5:
                    raise RuntimeError(
                        f"Full trajectory rate {rate:.3f} Hz differs from target"
                    )
                if args.recordings is not None:
                    saved = (
                        recording_view(network, spikes, records)
                        if recording_view
                        else {
                            "spike": spikes.cpu(),
                            **{k: v.cpu() for k, v in records.items()},
                        }
                    )
                    args.recordings.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(saved, args.recordings)
                    args.recordings.with_suffix(".json").write_text(
                        json.dumps(
                            dict(
                                model=kind,
                                neurons=n,
                                edges=network.synapse.linear.nnz,
                                steps=args.steps,
                                dt_ms=dt,
                                target_hz=args.target_rate_hz,
                                realized_hz=rate,
                                input_scale=scale,
                                seed=args.seed,
                                cudagraph=network.cudagraph,
                                neuron_backend=args.neuron_backend,
                                sparse_backend=args.sparse_backend,
                                method=args.method,
                                axes=["time", "batch", "neuron"],
                                timings_ms=timings,
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
                    print(
                        f"recordings: {args.recordings} ({', '.join(saved)})",
                        flush=True,
                    )
                    del saved
                del spikes, records
        if args.mode in ("train", "both"):
            # Explicit runner owns inference buffers. Training remains native
            # autograd, or uses compile's own graph trees when requested.
            initialize(network, args.device)
            network.clear_cudagraph()
            network.cudagraph = False
            network.train()
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            for epoch in range(args.epochs):
                # Each epoch is an independent trial. Replace memories so no
                # state or autograd graph leaks from inference/the last epoch.
                initialize(network, args.device)
                optimizer.zero_grad(set_to_none=True)
                spikes, records = run(inputs)
                # Synthetic fitting objective: rate plus physical state loss
                # exercises surrogate, neuron parameters and recurrent weights.
                # Convert Hz to spikes/bin using the circuit dt (ms).
                target = args.target_rate_hz * dt / 1000.0
                loss = (spikes.mean(0) - target).square().mean()
                loss = loss + 1e-5 * records["neuron.v"].square().mean()
                loss.backward()
                if not all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in network.parameters()
                ):
                    raise RuntimeError("Nonfinite training gradient")
                optimizer.step()
                # Project GLIF Dale weights back onto their reference signs.
                # This is outside the compiled trajectory/backward boundary.
                constrain_net(network)
                rate = spikes.detach().mean().item() * 1000 / dt
                print(f"train {epoch + 1}: loss={loss.item():.6f} firing={rate:.3f}Hz")
    if args.export_sonata:
        path = adapter.export_network(network, args.export_sonata, dt=dt)
        print(f"SONATA circuit: {path}")
