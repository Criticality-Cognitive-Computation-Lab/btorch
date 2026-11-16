"""Learn recurrent weights in the existing low-rate Izhikevich circuit.

    python -m examples.rsnn_izhikevich_stdp

Match rsnn_izhikevich's 60,000 neurons, 33,072,348 edges, heterogeneous
DualExponentialPSC receptors and fixed drive scale 0.6640625 (~5 Hz before
learning). Excitatory edges learn; inhibitory weights remain fixed. CUDA Graphs
capture bounded 200-step chunks. Full neuron/receptor recordings are enabled,
but no per-step edge-weight or trace histories are allocated.

Choose --recurrent helion for a fused timestep, or --recurrent standard for
the existing separate schedule. --spike-selection scan and spike_list select
arrival discovery independently. --no-plasticity gives the matched static
circuit; --no-cudagraph measures ordinary submission. Cold time includes the
first chunk's JIT/warmup/capture; construction, validation and writes are separate.

For a small CPU walkthrough (the fixed drive is calibrated for the large case):

    python -m examples.rsnn_izhikevich_stdp --neurons 128 --density .05 \
        --steps 256 --device cpu --neuron-backend torch --method pull \
        --no-cudagraph --no-recordings

Online STDP runs under no_grad and is separate from BPTT: use rsnn_izhikevich
for checkpointed/offloaded gradient training. Learning can change firing rate;
the input is deliberately not recalibrated during simulation. FP32 push is
not bitwise reproducible; see the sparse propagation guide.
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from btorch.models import KernelOptions, environ, functional
from btorch.models.connection import (
    EdgeWeight,
    FixedIndegree,
    PairSTDP,
    PlasticityRule,
    SparseConnection,
    Synapse,
)
from btorch.models.neurons import Izhikevich
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import DualExponentialPSC, HeterSynapsePSC


@dataclass(frozen=True)
class NetworkOptions:
    """Configure the same physical circuit as rsnn_izhikevich."""

    neurons: int = 60_000
    density: float = 33_072_348 / 60_000**2
    edges: int | None = None
    seed: int = 42
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    neuron_backend: str = "triton"
    sparse_backend: str = "auto"
    method: str = "push"
    cudagraph: bool = True
    chunk_size: int = 200
    chunk_mode: str = "split"


NETWORK_DEFAULTS = NetworkOptions()


def excitatory_stdp(weight: torch.Tensor) -> PairSTDP:
    """Keep edge signs and fixed inhibition while learning excitatory edges.

    Amplitudes scale with the initial edge magnitude, so the same
    relative learning rate applies at different indegrees. Arrays are in
    input-edge order; the connection performs canonical remapping once.
    """
    amplitude = weight.clamp_min(0) * 1e-4
    return PairSTDP(
        a_plus=amplitude,
        a_minus=amplitude * 1.05,
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=weight.clamp_max(0) * 1.5,
        w_max=weight.clamp_min(0) * 1.5,
        trace_scope="neuron",
    )


def build_model(
    options: NetworkOptions,
    *,
    plasticity: bool = True,
    plasticity_factory: Callable[[torch.Tensor], PlasticityRule] = excitatory_stdp,
) -> RecurrentNN:
    """Build fixed-indegree signed connectivity and heterogeneous receptor
    dynamics.

    Args:
        options: Physical circuit and execution settings.
        plasticity: Enable online learning.
        plasticity_factory: Construct a rule from input-edge weights. The
            connection binds its memories and canonical coefficient ordering.
    """
    n = options.neurons
    count = round(n * n * options.density) if options.edges is None else options.edges
    if not 0 < count <= n * n:
        raise ValueError("edges must be positive and no larger than neurons squared")
    degree, extra = divmod(count, n)
    width = degree + bool(extra)
    pre, post = FixedIndegree(width, allow_multapses=False).edges(
        n, n, generator=torch.Generator().manual_seed(options.seed)
    )
    if extra:
        keep = (
            torch.arange(width)[None, :] < (degree + (torch.arange(n) < extra))[:, None]
        )
        pre, post = pre[keep.flatten()], post[keep.flatten()]
    receptor = (pre % 5 == 0).long()
    magnitude = 100 / max(count / n, 1)
    weight = torch.where(receptor == 0, magnitude, -4 * magnitude)
    connection = SparseConnection.from_edges(
        pre,
        post,
        n,
        n,
        values=weight,
        synapse=Synapse(
            weight=EdgeWeight(trainable=False),
            receptor=receptor,
            n_receptor=2,
            plasticity=plasticity_factory(weight) if plasticity else None,
        ),
    )
    neuron = Izhikevich(
        n,
        v_threshold=-40.0,
        v_reset=-65.0,
        v_rest=-60.0,
        v_peak=30.0,
        c_m=100.0,
        k=2.0,
        a=torch.linspace(0.018, 0.022, n),
        b=0.2,
        d=8.0,
        hard_reset=True,
    )
    synapse = HeterSynapsePSC(
        n,
        n_receptor=2,
        receptor_type_index=None,
        linear=connection,
        base_psc=DualExponentialPSC,
        tau_decay=torch.tensor([5.0, 10.0]).repeat(n),
        tau_rise=torch.tensor([1.0, 2.0]).repeat(n),
        A=1.0,
    )
    return RecurrentNN(
        neuron,
        synapse,
        backend=KernelOptions(
            neuron=options.neuron_backend,
            connection=options.sparse_backend,
            connection_method=options.method,
        ),
        update_state_names=(
            "neuron.v",
            "neuron.u",
            "synapse.base_psc.psc",
            "synapse.psc",
        ),
        cudagraph=options.cudagraph,
        chunk_size=options.chunk_size,
        chunk_mode=options.chunk_mode,
    ).to(device=options.device, dtype=options.dtype)


def initialize(network: RecurrentNN, device: str) -> None:
    """Reset physical and learning memories while preserving learned
    weights."""
    functional.init_net_state(
        network,
        batch_size=1,
        device=device,
        dtype=network.synapse.linear.weight.value.dtype,
    )
    with torch.no_grad():
        network.neuron.v.copy_(network.neuron.v_rest)


def recording_view(network, spikes, records) -> dict[str, torch.Tensor]:
    """Transfer requested states, sharing signed receptor storage for
    EPSC/IPSC."""
    receptor_psc = records["synapse.base_psc.psc"].cpu()
    return {
        "v": records["neuron.v"].cpu(),
        "u": records["neuron.u"].cpu(),
        "spike": spikes.cpu(),
        "psc": records["synapse.psc"].cpu(),
        "epsc": network.synapse.get_psc(0, psc=receptor_psc, validate_nan=False),
        "ipsc": network.synapse.get_psc(1, psc=receptor_psc, validate_nan=False),
    }


def main() -> None:
    """Run cold/replayed online learning from identical initialized states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=NETWORK_DEFAULTS.neurons)
    parser.add_argument("--density", type=float, default=NETWORK_DEFAULTS.density)
    parser.add_argument("--edges", type=int)
    parser.add_argument("--seed", type=int, default=NETWORK_DEFAULTS.seed)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=1.0, help="timestep in ms")
    parser.add_argument("--input-scale", type=float, default=0.6640625)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float64", "float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--neuron-backend", choices=("torch", "triton", "tilelang"), default="triton"
    )
    parser.add_argument(
        "--sparse-backend",
        choices=("auto", "torch", "triton", "tilelang", "cusparse"),
        default="auto",
    )
    parser.add_argument("--method", choices=("auto", "push", "pull"), default="push")
    parser.add_argument(
        "--recurrent", choices=("standard", "helion"), default="standard"
    )
    parser.add_argument("--spike-selection", choices=("auto", "scan", "spike_list"))
    parser.add_argument("--chunk-size", type=int, default=NETWORK_DEFAULTS.chunk_size)
    parser.add_argument("--chunk-mode", choices=("split", "chunk"), default="split")
    parser.add_argument(
        "--cudagraph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--plasticity", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile tensor/operator work, retaining Python chunk scheduling",
    )
    parser.add_argument(
        "--recordings",
        type=Path,
        default=Path("fig/rsnn_izhikevich_stdp/recordings.pt"),
    )
    parser.add_argument(
        "--no-recordings", dest="recordings", action="store_const", const=None
    )
    args = parser.parse_args()
    if (
        args.neurons < 1
        or args.steps < 1
        or args.chunk_size < 1
        or not 0 < args.density <= 1
        or args.dt <= 0
        or args.input_scale <= 0
    ):
        parser.error("positive sizes/dt/input scale and density in (0, 1] required")
    if args.cudagraph and torch.device(args.device).type != "cuda":
        parser.error("--cudagraph requires CUDA")
    options = replace(
        NETWORK_DEFAULTS,
        **{
            name: getattr(torch, args.dtype) if name == "dtype" else getattr(args, name)
            for name in NETWORK_DEFAULTS.__dataclass_fields__
        },
    )

    def synchronize():
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)

    started = time.perf_counter()
    network = build_model(options, plasticity=args.plasticity)
    selection = args.spike_selection
    if selection is None:
        selection = (
            "spike_list"
            if args.neuron_backend in ("triton", "tilelang") and args.method == "push"
            else "auto"
        )
    network.backend = replace(
        network.backend, recurrent=args.recurrent, spike_selection=selection
    )
    network.eval()
    initialize(network, args.device)
    # Store a single input row and expand its time axis, just like the base circuit.
    # Compute spatial phase before rounding to storage precision: BF16 cannot
    # represent consecutive neuron indices across the 60K population.
    lane = torch.arange(
        args.neurons,
        device=args.device,
        dtype=torch.float64 if options.dtype == torch.float64 else torch.float32,
    )[None, None, :]
    inputs = (
        ((300.0 + 30.0 * torch.sin(lane * 0.01)) * args.input_scale)
        .to(dtype=options.dtype)
        .expand(args.steps, -1, -1)
    )
    weight = network.synapse.linear.weight.value
    initial_weight = weight.detach().clone()
    synchronize()
    construction_s = time.perf_counter() - started
    run = (
        torch.compile(
            network,
            mode="default" if args.cudagraph else "reduce-overhead",
        )
        if args.compile
        else network
    )
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    timings = {}
    with torch.no_grad(), environ.context(dt=args.dt):
        for label in ("cold", "replay"):
            # Reset outside timing. Graph captures restore mutable weights and
            # rule memories during preparation; replay performs real learning.
            initialize(network, args.device)
            weight.copy_(initial_weight)
            synchronize()
            started = time.perf_counter()
            spikes, records = run(inputs)
            synchronize()
            timings[label + "_ms"] = (time.perf_counter() - started) * 1000
            if label == "cold":
                del spikes, records
        simulation_peak = (
            torch.cuda.max_memory_allocated(args.device)
            if torch.device(args.device).type == "cuda"
            else None
        )
        started = time.perf_counter()
        rate_hz = spikes.mean().item() * 1000 / args.dt
        weight_change = (weight - initial_weight).abs()
        summary = dict(
            neurons=args.neurons,
            edges=network.synapse.linear.nnz,
            steps=args.steps,
            dt_ms=args.dt,
            dtype=args.dtype,
            input_scale=args.input_scale,
            firing_hz=rate_hz,
            plasticity=args.plasticity,
            neuron_backend=args.neuron_backend,
            sparse_backend=args.sparse_backend,
            method=args.method,
            recurrent=args.recurrent,
            spike_selection=selection,
            cudagraph=args.cudagraph,
            chunk_size=args.chunk_size,
            chunk_mode=args.chunk_mode,
            construction_s=construction_s,
            **timings,
            replay_us_per_step=timings["replay_ms"] * 1000 / args.steps,
            weight_abs_change_mean=weight_change.mean().item(),
            weight_abs_change_max=weight_change.max().item(),
            peak_cuda_allocated_bytes=simulation_peak,
        )
        saved = recording_view(network, spikes, records) if args.recordings else None
        synchronize()
        summary["validation_and_transfer_s"] = time.perf_counter() - started
    print(json.dumps(summary, indent=2), flush=True)
    if args.recordings:
        args.recordings.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, args.recordings)
        args.recordings.with_suffix(".json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(f"Recordings: {args.recordings}", flush=True)


if __name__ == "__main__":
    main()
