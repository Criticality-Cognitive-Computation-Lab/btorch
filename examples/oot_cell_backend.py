"""Register, validate and use an external cell kernel through public APIs.

Run on CPU with ``python examples/oot_cell_backend.py``. To inspect and validate
real generated GPU code, use ``--backend triton --device cuda --source /tmp/cell.py``
(or cupy/tilelang). Optional compile/capture checks run only when requested.
"""

import argparse
from pathlib import Path

import torch

from btorch.models import KernelOptions, environ, functional
from btorch.models.connection import SparseConnection
from btorch.models.kernels import (
    CellSpec,
    HistoryPlan,
    register_cell_backend,
    resolve_cell_backend,
)
from btorch.models.kernels.testing import (
    ParityCase,
    ParityOptions,
    check_backend_parity,
)
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC


def run_example(
    backend: str = "torch",
    device: str = "cpu",
    *,
    source_path: str | None = None,
    compile: bool = False,
    capture: bool = False,
):
    """Validate a separately registered implementation before neuron
    training."""
    device = torch.device(device)
    spec = CellSpec("lif", dt=1.0)
    implementation = resolve_cell_backend(backend, spec, device)
    # An external package can reuse the shared generator/runtime while evolving
    # scheduling independently. Registering does not mutate model semantics or
    # connection routes. Replacement here supports repeating this tutorial.
    register_cell_backend(
        "tutorial_cell",
        forward=implementation.forward,
        backward=implementation.backward,
        supports=lambda choice, target: (
            choice.model in ("lif", "alpha")
            and target.type == device.type
            and (device.index is None or target.index == device.index)
        ),
        replace=True,
    )
    x_seq = torch.linspace(0.15, 0.85, 192, device=device, dtype=torch.float32).reshape(
        12, 2, 8
    )
    states = (torch.full((2, 8), 0.15, device=device, dtype=torch.float32),)
    parameters = tuple(
        torch.tensor(value, device=device, dtype=torch.float32)
        for value in (1.0, 0.0, 1.0, 4.0)
    )
    history = HistoryPlan(requested=("v", "v_charged"))
    if source_path is not None:
        if backend == "torch":
            raise ValueError("Choose triton, cupy or tilelang to generate DSL source")
        from btorch.models.kernels import generate_cell_source

        source, _ = generate_cell_source(backend, spec, 16, 12, history.names)
        Path(source_path).write_text(source)
    report = check_backend_parity(
        "tutorial_cell",
        [
            ParityCase(x_seq, states, parameters, spec, history),
            ParityCase(
                x_seq,
                (torch.zeros_like(states[0]), torch.zeros_like(states[0])),
                (parameters[3], parameters[2]),
                CellSpec("alpha"),
                HistoryPlan(requested=("h",)),
                name="psc",
            ),
        ],
        options=ParityOptions(compile=compile, capture=capture),
    )
    report.assert_close()
    # Exercise the unchanged stateful Module API, not only standalone callbacks.
    # eval() retains the upstream gradient because the encoder scale is trainable.
    neuron = LIF(8, tau=4.0, step_mode="m", backend="tutorial_cell", device=device)
    neuron.float()
    neuron.init_state(batch_size=2, device=device, dtype=torch.float32)
    neuron.eval()
    scale = torch.tensor(0.8, device=device, requires_grad=True, dtype=torch.float32)
    with environ.context(dt=1.0):
        spikes = neuron(scale * x_seq)
        loss = spikes.sum() + neuron.v.square().sum()
        loss.backward()
    if scale.grad is None or not torch.isfinite(scale.grad):
        raise AssertionError("The registered kernel lost the upstream gradient")
    # The same external callback family handles both LIF and AlphaPSC, so it
    # also composes with an unchanged sparse connection in a whole RSNN.
    pre = torch.arange(8, device=device)
    connection = SparseConnection.from_edges(
        pre,
        (pre + 1) % 8,
        8,
        8,
        values=torch.full((8,), 0.2, device=device, dtype=torch.float32),
    )
    recurrent = RecurrentNN(
        LIF(8, tau=4.0),
        AlphaPSC(8, tau_syn=4.0, linear=connection),
        backend=KernelOptions(neuron="tutorial_cell"),
        update_state_names=("neuron.v", "synapse.psc"),
    )
    functional.init_net_state(
        recurrent, batch_size=2, device=device, dtype=torch.float32
    )
    encoded = x_seq.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        recurrent_spikes, records = recurrent(encoded)
        (
            recurrent_spikes.sum()
            + records["neuron.v"].square().sum()
            + records["synapse.psc"].square().sum()
        ).backward()
    if encoded.grad is None or not torch.isfinite(encoded.grad).all():
        raise AssertionError("The registered kernel lost the sparse RSNN gradient")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("torch", "triton", "tilelang", "cupy"), default="torch"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    report = run_example(
        args.backend,
        args.device,
        source_path=args.source,
        compile=args.compile,
        capture=args.capture,
    )
    print(f"Passed {len(report.comparisons)} numerical comparisons")
    for skip in report.skipped:
        print(f"Skipped: {skip}")
