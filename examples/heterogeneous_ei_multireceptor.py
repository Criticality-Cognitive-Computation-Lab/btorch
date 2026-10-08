"""Train and inspect an E/I recurrent SNN with semantic receptor edges.

This example keeps receptor identity on each semantic edge. ``SparseConnection``
lowers those attributes for execution, while ``HeterSynapsePSC`` keeps separate
PSC state for every receptor channel and sums the channels at each target
neuron. The model therefore does not use a physically expanded connectome as
its primary representation.

Run a small CPU example with::

    PYTHONPATH=. python examples/heterogeneous_ei_multireceptor.py \
        --device cpu --epochs 2
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from btorch.models import environ, functional
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.constrain import constrain_net
from btorch.models.neurons import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC, HeterSynapsePSC
from btorch.sparse import Hints


RECEPTOR_NAMES = ("E_to_E", "E_to_I", "I_to_E", "I_to_I")
RECEPTOR_TAU = (3.0, 5.0, 8.0, 12.0)


def make_receptor_index() -> pd.DataFrame:
    """Return the table used to inspect neuron-mode receptor channels."""
    pre_type = ("E", "E", "I", "I")
    post_type = ("E", "I", "E", "I")
    return pd.DataFrame(
        {
            "pre_receptor_type": pre_type,
            "post_receptor_type": post_type,
            "receptor_type": RECEPTOR_NAMES,
            "receptor_index": range(len(RECEPTOR_NAMES)),
        }
    )


def make_semantic_edges(
    n_exc: int,
    n_inh: int,
    density: float,
    device: torch.device,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create unique E/I edges, receptor labels, and Dale-signed weights."""
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")

    n_neuron = n_exc + n_inh
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pre_grid = torch.arange(n_neuron).repeat_interleave(n_neuron)
    post_grid = torch.arange(n_neuron).repeat(n_neuron)
    keep = torch.rand(n_neuron * n_neuron, generator=generator) < density
    keep &= pre_grid != post_grid

    pre = pre_grid[keep]
    post = post_grid[keep]
    if pre.numel() == 0:
        raise ValueError("density produced no recurrent edges")

    pre_is_inh = pre >= n_exc
    post_is_inh = post >= n_exc
    receptor = pre_is_inh.to(torch.long) * 2 + post_is_inh.to(torch.long)
    magnitude = 0.25 + 0.1 * torch.rand(pre.numel(), generator=generator)
    values = torch.where(pre_is_inh, -magnitude, magnitude)
    return (
        pre.to(device),
        post.to(device),
        receptor.to(device),
        values.to(device=device, dtype=torch.float32),
    )


class HeterogeneousEIRSNN(nn.Module):
    """GLIF recurrent network with semantic E/I receptor attributes."""

    def __init__(
        self,
        n_exc: int,
        n_inh: int,
        recurrent_density: float,
        expected_spike_density: float,
        device: torch.device,
        seed: int,
    ) -> None:
        super().__init__()
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_neuron = n_exc + n_inh
        self.receptor_type_index = make_receptor_index()

        pre, post, receptor, values = make_semantic_edges(
            n_exc, n_inh, recurrent_density, device, seed
        )
        synapse = Synapse(
            receptor=receptor,
            n_receptor=len(RECEPTOR_NAMES),
            dale=True,
        )
        self.connection = SparseConnection.from_edges(
            pre,
            post,
            self.n_neuron,
            self.n_neuron,
            synapse=synapse,
            values=values,
            hints=Hints(expected_density=expected_spike_density),
            device=device,
            dtype=torch.float32,
        )

        neuron = GLIF3(
            n_neuron=self.n_neuron,
            v_threshold=-50.0,
            v_reset=-60.0,
            c_m=2.0,
            tau=20.0,
            tau_ref=2.0,
            k=[0.1, 0.2],
            asc_amps=[1.0, -2.0],
            step_mode="s",
            device=device,
            dtype=torch.float32,
        )
        channel_tau = torch.tensor(RECEPTOR_TAU, device=device).repeat(self.n_neuron)
        psc = HeterSynapsePSC(
            n_neuron=self.n_neuron,
            n_receptor=len(RECEPTOR_NAMES),
            receptor_type_index=self.receptor_type_index,
            linear=self.connection,
            base_psc=AlphaPSC,
            tau_syn=channel_tau,
            step_mode="s",
        )
        self.brain = RecurrentNN(
            neuron=neuron,
            synapse=psc,
            step_mode="m",
            update_state_names=("neuron.v", "synapse.psc"),
        )
        self.readout = nn.Linear(self.n_neuron, 2, device=device)

    def forward(self, drive: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Run ``drive`` with shape ``[time, batch, neuron]``."""
        spikes, states = self.brain(drive)
        logits = self.readout(spikes.mean(dim=0))
        return logits, spikes, states


def make_batch(
    n_exc: int,
    n_inh: int,
    timesteps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Create a two-class drive that targets E versus I populations."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.randint(2, (batch_size,), generator=generator).to(device)
    drive = 0.15 * torch.randn(
        timesteps, batch_size, n_exc + n_inh, generator=generator
    ).to(device)
    e_drive = labels.eq(0).to(drive.dtype).view(1, batch_size, 1)
    i_drive = labels.eq(1).to(drive.dtype).view(1, batch_size, 1)
    drive[:, :, :n_exc] += 8.0 * e_drive
    drive[:, :, n_exc:] += 8.0 * i_drive
    return drive, F.one_hot(labels, num_classes=2).to(drive.dtype)


def train(
    model: HeterogeneousEIRSNN,
    n_exc: int,
    n_inh: int,
    timesteps: int,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> None:
    """Train a small classifier and project Dale constraints after each
    step."""
    functional.init_net_state(
        model, batch_size=batch_size, device=device, dtype=torch.float32
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for epoch in range(epochs):
        drive, target = make_batch(
            n_exc, n_inh, timesteps, batch_size, device, seed=epoch + 10
        )
        functional.reset_net(
            model, batch_size=batch_size, device=device, dtype=torch.float32
        )
        optimizer.zero_grad()
        with environ.context(dt=1.0):
            logits, _, _ = model(drive)
            loss = F.mse_loss(logits, target)
        loss.backward()
        optimizer.step()
        constrain_net(model)
        print(f"epoch={epoch + 1:02d} loss={loss.item():.5f}")


def simulate_and_inspect(
    model: HeterogeneousEIRSNN,
    timesteps: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Simulate once and print semantic edge and receptor-state summaries."""
    drive, _ = make_batch(
        model.n_exc, model.n_inh, timesteps, batch_size, device, seed=100
    )
    functional.reset_net(
        model, batch_size=batch_size, device=device, dtype=torch.float32
    )
    model.eval()
    with torch.no_grad(), environ.context(dt=1.0):
        logits, spikes, states = model(drive)

    table = model.connection.edge_table()
    print(f"edges={model.connection.nnz} spike_rate={spikes.mean().item():.3%}")
    print(f"edge fields={sorted(table)}")
    print(f"runtime plan:\n{model.connection.explain(spikes[0])}")
    print(f"logits_shape={tuple(logits.shape)} state_fields={sorted(states)}")
    for name, pair in zip(
        RECEPTOR_NAMES, (("E", "E"), ("E", "I"), ("I", "E"), ("I", "I"))
    ):
        channel = model.brain.synapse.get_psc(receptor_type=pair)
        print(f"psc[{name}] mean_abs={channel.abs().mean().item():.5f}")


def main() -> None:
    """Parse options, train, simulate, and inspect the semantic model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-exc", type=int, default=8)
    parser.add_argument("--n-inh", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--recurrent-density", type=float, default=0.35)
    parser.add_argument("--expected-spike-density", type=float, default=0.02)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    environ.set(dt=1.0)
    model = HeterogeneousEIRSNN(
        args.n_exc,
        args.n_inh,
        args.recurrent_density,
        args.expected_spike_density,
        device,
        args.seed,
    )
    train(
        model,
        args.n_exc,
        args.n_inh,
        args.timesteps,
        args.batch_size,
        args.epochs,
        device,
    )
    simulate_and_inspect(model, args.timesteps, args.batch_size, device)


if __name__ == "__main__":
    main()
