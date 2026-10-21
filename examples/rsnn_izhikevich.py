"""Simulate an Izhikevich RSNN with device spike-list sparse propagation.

The default population has 50,000 neurons and 2,500 distinct incoming sources
per neuron: 125 million edges, or 5% connectivity. Construction uses CPU CSR
storage, without a dense N-by-N matrix. Run a smaller example with::

    python -m examples.rsnn_izhikevich --neurons 256 --density 0.05 --steps 32

The default execution uses Triton neurons and explicit device spike-list push.
Select CSR pull with ``--method pull --sparse-backend cusparse``. These are
execution choices independent of the adjacency storage format. This is a
simulation example, not a cross-implementation benchmark.
"""

import argparse

import numpy as np
import torch

from btorch.models import environ
from btorch.models.connection import SparseConnection
from btorch.models.functional import init_net_state
from btorch.models.kernels import KernelOptions
from btorch.models.neurons.izhikevich import Izhikevich
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import Sigmoid
from btorch.models.synapse import ExponentialPSC
from btorch.sparse import CSR


def build_model(args) -> RecurrentNN:
    """Build fixed-indegree connectivity and the ordinary btorch recurrence."""
    n = args.neurons
    indegree = round(n * args.density)
    # Sampling one row at a time bounds scratch space by the population size.
    # Distinct sources preserve the requested edge count; self-edges are allowed.
    rng = np.random.default_rng(args.seed)
    sources = np.empty((n, indegree), dtype=np.int32)
    for row in sources:
        row[:] = np.sort(rng.choice(n, size=indegree, replace=False))
    adjacency = CSR(
        torch.arange(n + 1, dtype=torch.int64) * indegree,
        torch.from_numpy(sources.reshape(-1)),
        torch.full((n * indegree,), 0.04, dtype=torch.float32),
        (n, n),
    )
    connection = SparseConnection.from_adjacency(adjacency, orientation="post_pre").to(
        args.device
    )
    neuron = Izhikevich(
        n,
        v_threshold=-40.0,
        v_reset=-65.0,
        v_rest=-60.0,
        v_peak=30.0,
        c_m=100.0,
        k=2.0,
        a=0.02,
        b=0.2,
        d=8.0,
        trainable_param=set(),
        surrogate_function=Sigmoid(alpha=100.0),
        hard_reset=True,
    )
    model = (
        RecurrentNN(
            neuron,
            ExponentialPSC(n, tau_syn=5.0, linear=connection),
            backend=KernelOptions(
                neuron=args.neuron_backend,
                connection=args.sparse_backend,
                connection_method=args.method,
            ),
            update_state_names=(),
            cudagraph=args.cudagraph,
        )
        .to(args.device)
        .eval()
    )
    init_net_state(model, batch_size=1, device=args.device, dtype=torch.float32)
    model.neuron.v.fill_(-60.0)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=50_000)
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--neuron-backend",
        choices=("torch", "triton", "cupy", "tilelang"),
        default="triton",
    )
    parser.add_argument(
        "--sparse-backend",
        choices=("auto", "torch", "cusparse", "triton"),
        default="triton",
    )
    parser.add_argument("--method", choices=("auto", "pull", "push"), default="push")
    parser.add_argument("--cudagraph", action="store_true")
    args = parser.parse_args()
    if args.neurons < 1 or args.steps < 1 or not 0 < args.density <= 1:
        parser.error("positive neurons/steps and density in (0, 1] required")
    model = build_model(args)
    time = torch.arange(args.steps, device=args.device, dtype=torch.float32)
    lanes = torch.arange(args.neurons, device=args.device, dtype=torch.float32)
    drive = (
        300.0
        + 30.0 * torch.sin(lanes[None, None, :] * 0.01)
        + 30.0 * torch.sin(time[:, None, None] * 0.01)
    )
    with torch.inference_mode(), environ.context(dt=1.0):
        spikes, _ = model(drive)
    # Report realized activity, not the connectivity density or a firing hint.
    rate_hz = spikes.float().mean().item() * 1000.0
    print(
        f"N={args.neurons}, edges={model.synapse.linear.nnz}, firing={rate_hz:.3f} Hz"
    )


if __name__ == "__main__":
    main()
