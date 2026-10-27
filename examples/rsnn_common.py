"""Share execution and CLI plumbing for the two concrete RSNN examples."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from btorch.models import KernelOptions, environ, functional
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.connection.weight import EdgeWeight
from btorch.models.constrain import constrain_net
from btorch.models.neurons import GLIF3, Izhikevich
from btorch.models.rnn import RecurrentNN
from examples.rsnn_sonata import RSNNSonataIO


@dataclass(frozen=True)
class ExampleOptions:
    """Configure construction and execution independently of circuit IO."""

    neurons: int = 256
    density: float = 0.05
    seed: int = 0
    device: str = "cpu"
    neuron_backend: str = "torch"
    sparse_backend: str = "auto"
    method: str = "auto"
    cudagraph: bool = False


def build_connection(
    options: ExampleOptions,
    *,
    weight_scale: float,
    delay_bins: int = 1,
    dale: bool = False,
) -> SparseConnection:
    """Build fixed-indegree connectivity with signed receptor-labelled edges.

    Node IDs equal tensor indices; pre is source and post is target.
    Receptor 0 is initially excitatory, receptor 1 inhibitory. Edge
    weights are trainable with optional Dale projection applied after
    each optimizer update.
    """
    n = options.neurons
    rng = np.random.default_rng(options.seed)
    indegree = max(1, round(n * options.density))
    # Fixed indegree sampling has O(E) storage and only O(N) scratch; the
    # Izhikevich example can therefore retain its N50k/E125M configuration.
    pre = np.empty((n, indegree), dtype=np.int64)
    for row in pre:
        row[:] = np.sort(rng.choice(n, size=indegree, replace=False))
    pre = torch.from_numpy(pre.reshape(-1))
    post = torch.arange(n).repeat_interleave(indegree)
    receptor = (pre % 5 == 0).long()  # receptor0 excitatory; receptor1 inhibitory
    weights = torch.where(receptor == 0, weight_scale, -weight_scale * 2)
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
    time = torch.arange(steps, device=device)[:, None, None]
    lane = torch.arange(neurons, device=device)[None, None, :]
    baseline, amplitude = (4.0, 0.5) if kind == "glif3" else (300.0, 30.0)
    return (
        baseline
        + amplitude * torch.sin(time * 0.01)
        + amplitude * torch.sin(lane * 0.01)
    )


def initialize(network: RecurrentNN, device: str) -> None:
    """Initialize fresh physical memories without retaining an earlier
    graph."""
    functional.init_net_state(network, batch_size=1, device=device, dtype=torch.float32)
    with torch.no_grad():
        network.neuron.v.copy_(network.neuron.v_rest)


def main(kind: str, build_model: Callable[[ExampleOptions], RecurrentNN]) -> None:
    """Run inference/training and optional SONATA circuit import/export."""
    parser = argparse.ArgumentParser(
        description=f"{kind} heterogeneous RSNN; SONATA circuit IO"
    )
    parser.add_argument("--neurons", type=int, default=256)
    parser.add_argument(
        "--density",
        type=float,
        default=0.05,
        help="approximate fixed-indegree fraction",
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--neuron-backend",
        choices=("torch", "triton", "cupy", "tilelang"),
        default="torch",
    )
    parser.add_argument(
        "--sparse-backend",
        choices=("auto", "torch", "triton", "cusparse"),
        default="auto",
    )
    parser.add_argument("--method", choices=("auto", "pull", "push"), default="auto")
    parser.add_argument(
        "--mode", choices=("inference", "train", "both"), default="both"
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
        "--cudagraph", action="store_true", help="explicit inference-only capture"
    )
    args = parser.parse_args()
    if (
        args.neurons < 1
        or args.steps < 1
        or args.epochs < 1
        or not 0 < args.density <= 1
        or args.dt <= 0
    ):
        parser.error("positive sizes/dt and density in (0,1] required")
    if args.cudagraph and args.device.split(":")[0] != "cuda":
        parser.error("--cudagraph requires --device cuda")
    if args.cudagraph and args.mode != "inference":
        parser.error(
            "--cudagraph requires --mode inference; use --compile for training"
        )
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
            initialize(network, args.device)
            with torch.inference_mode():
                spikes, _ = run(inputs)
                print(f"inference: firing={spikes.mean().item() * 1000 / dt:.3f}Hz")
        if args.mode in ("train", "both"):
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
                # A 20 Hz target becomes spikes/bin using the circuit dt (ms).
                target = 20.0 * dt / 1000.0
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
                print(f"train {epoch + 1}: loss={loss.item():.6f}")
    if args.export_sonata:
        path = adapter.export_network(network, args.export_sonata, dt=dt)
        print(f"SONATA circuit: {path}")
