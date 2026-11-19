"""Run a small deterministic E/I conductance circuit with optional pair STDP.

    python -m examples.rsnn_izhikevich_conductance_stdp
    python -m examples.rsnn_izhikevich_conductance_stdp --no-plasticity

This synthetic ring is an API walkthrough, not a hippocampal topology
reproduction. Weights are positive conductance increments (nS); reversal
potentials determine current sign. No STP, fused cell lowering, CUDA Graphs,
or automatic state/weight histories are used. Only --output writes a JSON
summary. Created 2026-11-18.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from btorch.models import KernelOptions, environ, functional
from btorch.models.connection import EdgeWeight, PairSTDP, SparseConnection, Synapse
from btorch.models.kernels.adapters import connection_scope
from btorch.models.neurons import Izhikevich
from btorch.models.rnn import ConductanceRecurrentNN
from btorch.models.synapse import HeterConductancePSC


def build_model(args: argparse.Namespace) -> ConductanceRecurrentNN:
    """Build a deterministic ring with two distinct outgoing targets per source."""
    n = args.neurons
    pre = torch.arange(n).repeat_interleave(2)
    post = (pre + torch.tensor([1, 2]).repeat(n)) % n
    receptor = (pre % 5 == 0).long()  # Source identity: I=1, E=0.
    weight = torch.where(receptor == 0, 0.6, 1.2)
    excitatory = receptor == 0
    amplitude = torch.where(excitatory, weight * 0.005, 0.0)
    rule = (
        PairSTDP(
            a_plus=amplitude,
            a_minus=amplitude * 1.05,
            tau_plus=20.0,
            tau_minus=20.0,
            w_min=torch.where(excitatory, weight * 0.25, weight),
            w_max=torch.where(excitatory, weight * 2.0, weight),
            trace_scope="neuron",
        )
        if args.plasticity
        else None
    )
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
            plasticity=rule,
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
        backend="torch",
    )
    synapse = HeterConductancePSC(
        n,
        n_receptor=2,
        receptor_type_index=None,
        linear=connection,
        tau_syn=torch.tensor([5.0, 10.0]),
        reversal_potential=torch.tensor([0.0, -80.0]),
        backend="torch",
    )
    return ConductanceRecurrentNN(
        neuron,
        synapse,
        backend=KernelOptions(
            neuron="torch",
            connection=args.sparse_backend,
            connection_method=args.method,
        ),
        update_state_names=(),
        cudagraph=False,
    ).to(device=args.device, dtype=torch.float32)


def simulate(args: argparse.Namespace) -> dict:
    """Continue bounded input blocks without resetting physical or learning state."""
    device = torch.device(args.device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # dt also covers initialization and all simulation calls, not just construction.
    with environ.context(dt=args.dt), torch.no_grad():
        network = build_model(args)
        functional.init_net_state(
            network,
            batch_size=1,
            device=device,
            dtype=torch.float32,
        )
        network.neuron.v.copy_(network.neuron.v_rest)
        connection = network.synapse.linear
        owner = connection.weight
        canonical_weight = owner.value
        before = canonical_weight.clone()
        receptors = connection.receptor.clone()
        state_names = tuple(functional.named_hidden_states(network))
        # Distinct DC levels provide deterministic, noncoincident firing.
        drive = torch.linspace(250.0, 400.0, args.neurons, device=device).view(1, 1, -1)
        spike_count = torch.zeros(args.neurons, device=device)
        synchronize()
        started = time.perf_counter()
        for start in range(0, args.steps, 32):
            inputs = drive.expand(min(32, args.steps - start), 1, args.neurons)
            spikes, _ = network(inputs)
            spike_count.add_(spikes.sum(dim=(0, 1)))
        synchronize()
        elapsed = time.perf_counter() - started
        # Explain under the same scoped sparse options as recurrent propagation.
        with connection_scope(network.backend):
            routing = connection.explain()
        after = owner.value
        delta = after - before
        excitatory = receptors == 0
        finite = all(
            torch.isfinite(value).all().item()
            for value in functional.named_hidden_states(network).values()
            if isinstance(value, torch.Tensor)
        )
        summary = {
            "circuit": "synthetic two-target ring; every fifth source inhibitory",
            "neurons": args.neurons,
            "edges": connection.nnz,
            "steps": args.steps,
            "dt_ms": args.dt,
            "device": str(device),
            "dtype": "float32",
            "plasticity": args.plasticity,
            "cell_backend": "torch",
            "sparse_backend_requested": args.sparse_backend,
            "method_requested": args.method,
            "routing": routing,
            "simulation_seconds": elapsed,
            "timing_scope": (
                "one eager run including first-use routing, input blocks and spike "
                "counting; synchronized on CUDA; excludes build/reset, validation "
                "and output"
            ),
            "spikes": int(spike_count.sum().item()),
            "mean_rate_hz": spike_count.mean().item() * 1000 / (args.steps * args.dt),
            "canonical_weight_owner_preserved": connection.weight is owner,
            "canonical_weight_tensor_preserved": connection.weight.value
            is canonical_weight,
            "state_names_preserved": state_names
            == tuple(functional.named_hidden_states(network)),
            "receptors_preserved": torch.equal(connection.receptor, receptors),
            "excitatory_edges_changed": int(
                torch.count_nonzero(delta[excitatory]).item()
            ),
            "inhibitory_edges_changed": int(
                torch.count_nonzero(delta[~excitatory]).item()
            ),
            "max_abs_weight_change_ns": delta.abs().max().item(),
            "weights_positive": bool((after > 0).all().item()),
            "final_states_finite": finite,
            "final_voltage_range_mv": [
                network.neuron.v.min().item(),
                network.neuron.v.max().item(),
            ],
            "final_conductance_range_ns": [
                network.synapse.base_psc.psc.min().item(),
                network.synapse.base_psc.psc.max().item(),
            ],
        }
        if not (
            finite and summary["weights_positive"] and summary["receptors_preserved"]
        ):
            raise RuntimeError(
                "Nonfinite state, nonpositive weight or changed receptor routing"
            )
        if summary["inhibitory_edges_changed"] or (
            not args.plasticity and torch.count_nonzero(delta)
        ):
            raise RuntimeError("Fixed weights changed")
        return summary


def main() -> None:
    """Parse the small CPU walkthrough or explicitly selected sparse execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--neurons", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.2, help="timestep in ms")
    parser.add_argument("--method", choices=("pull", "push"), default="pull")
    parser.add_argument(
        "--sparse-backend", choices=("auto", "torch", "triton"), default="auto"
    )
    parser.add_argument(
        "--plasticity", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--output", type=Path, help="optional JSON summary (no recordings)"
    )
    args = parser.parse_args()
    if args.neurons < 3 or args.steps < 1 or not math.isfinite(args.dt) or args.dt <= 0:
        parser.error("neurons >= 3, steps >= 1 and finite dt > 0 required")
    if args.sparse_backend == "triton" and torch.device(args.device).type != "cuda":
        parser.error("Triton sparse propagation requires --device cuda")
    summary = simulate(args)
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
