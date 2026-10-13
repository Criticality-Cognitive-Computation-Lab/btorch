"""Train an encoder, sparse RSNN and readout with differentiable state
records."""

import argparse

import torch
from torch import nn

from btorch.models import KernelOptions, environ, functional
from btorch.models.connection import SparseConnection
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC


def run_example(backend: str = "torch", device: str = "cpu"):
    """Run simulation and two contiguous training chunks without truncating
    BPTT."""
    torch.manual_seed(8)
    device = torch.device(device)
    pre = torch.tensor([0, 1, 2, 3], device=device)
    post = torch.tensor([1, 2, 3, 0], device=device)
    connection = SparseConnection.from_edges(
        pre, post, 4, 4, values=torch.full((4,), 0.2, device=device)
    )
    model = RecurrentNN(
        LIF(4, v_threshold=0.6, trainable_param={"tau"}, device=device),
        AlphaPSC(4, tau_syn=2.0, linear=connection),
        backend=KernelOptions(cell=backend),
        update_state_names=("neuron.v", "synapse.psc"),
    )
    functional.init_net_state(model, batch_size=2, device=device, dtype=torch.float32)
    encoder = nn.Linear(3, 4, device=device, dtype=torch.float32)
    readout = nn.Linear(4, 2, device=device, dtype=torch.float32)
    source = torch.randn(8, 2, 3, device=device, dtype=torch.float32)
    target = torch.ones(8, 2, 2, device=device, dtype=torch.float32)
    with environ.context(dt=1.0):
        with torch.inference_mode():
            simulated, _ = model(encoder(source))
        # Reset before optimizing an independent trajectory. eval() affects
        # ordinary nn layers but must not select an inference-only cell kernel.
        functional.reset_net(model)
        model.eval()
        first, first_records = model(encoder(source[:3]))
        second, second_records = model(encoder(source[3:]))
        spikes = torch.cat((first, second))
        voltage = torch.cat((first_records["neuron.v"], second_records["neuron.v"]))
        current = torch.cat(
            (first_records["synapse.psc"], second_records["synapse.psc"])
        )
        loss = (readout(spikes) - target).square().mean()
        loss = loss + 0.01 * (voltage.square().mean() + current.square().mean())
        loss.backward()
    for name, parameter in (
        ("encoder", encoder.weight),
        ("readout", readout.weight),
        ("neuron", model.neuron.tau),
    ):
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Missing or invalid {name} gradients")
    # Explicit detach ends this BPTT trajectory before an optimizer step or
    # unrelated minibatch. No backend inserts it automatically at chunk edges.
    functional.detach_net(model)
    return loss.detach(), simulated, spikes.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="torch")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    loss, _, _ = run_example(args.backend, args.device)
    print(f"Loss: {loss.item():.6f}")
