"""Run a recorded, low-rate Izhikevich RSNN and compare sparse execution.

    python -m examples.rsnn_izhikevich

Defaults: CUDA/Triton, 60,000 neurons, exactly 33,072,348 edges, 5000 steps
at 1 ms, calibrated 5 Hz mean activity, and 200-step CUDA Graph chunks. The script
prints matched push-eager / pull-graph / push-graph timings with all recordings
on, then saves v, u, spike, total PSC, signed EPSC and IPSC under fig/.
FP32 push is not bitwise reproducible, even with fixed seeds or CUDA Graphs.

The two heterogeneous dual-exponential receptors retain independent dynamics.
For a small training run (inference capture is disabled during training):

    python -m examples.rsnn_izhikevich --neurons 256 --density .05 \
        --steps 256 --mode train --compile

Use --no-compare to skip timing comparisons, --no-cudagraph for eager final
inference, and --recordings PATH to choose the saved tensor archive. SONATA
circuit import/export is available through --import-sonata/--export-sonata.
"""

from dataclasses import replace

import torch

from btorch.models import BPTT, KernelOptions
from btorch.models.neurons import Izhikevich
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import Sigmoid, Truncated
from btorch.models.synapse import DualExponentialPSC, HeterSynapsePSC
from examples.rsnn_common import ExampleOptions, RunOptions, build_connection, main


def build_model(options: ExampleOptions) -> RecurrentNN:
    """Build the Izhikevich/DualExponentialPSC circuit with two receptor
    channels."""
    n = options.neurons
    if options.edges is None:
        options = replace(options, edges=round(n * n * options.density))
    connection = build_connection(options, weight_scale=100 / 13, inhibitory_ratio=4.0)
    # Sparse BPTT uses the same truncated surrogate: zero derivative outside
    # the declared support in normalized threshold-distance units.
    function = Truncated(Sigmoid(alpha=100.0), width=0.25)
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
        trainable_param={"a"},
        surrogate_function=function,
        hard_reset=True,
    )
    # PSC lanes are flattened as post * 2 + receptor: repeat each
    # receptor pair once per neuron to match the connection output layout.
    psc = HeterSynapsePSC(
        n,
        n_receptor=2,
        receptor_type_index=None,
        linear=connection,
        base_psc=DualExponentialPSC,
        tau_decay=torch.tensor([5.0, 10.0]).repeat(n),
        tau_rise=torch.tensor([1.0, 2.0]).repeat(n),
        A=1.0,
    )
    records = (
        "neuron.v",
        "neuron.u",
        "synapse.base_psc.psc",
        "synapse.psc",
    )
    return RecurrentNN(
        neuron,
        psc,
        gradient=BPTT(sparse=True),
        backend=KernelOptions(
            neuron=options.neuron_backend,
            connection=options.sparse_backend,
            connection_method=options.method,
        ),
        update_state_names=records,
        cudagraph=options.cudagraph,
        chunk_size=options.chunk_size,
    ).to(options.device)


def recording_view(
    network: RecurrentNN, spikes: torch.Tensor, records: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Keep signed receptor histories as EPSC/IPSC views of one CPU
    transfer."""
    receptor_psc = records["synapse.base_psc.psc"].cpu()
    return {
        "v": records["neuron.v"].cpu(),
        "u": records["neuron.u"].cpu(),
        "spike": spikes.cpu(),
        "psc": records["synapse.psc"].cpu(),
        "epsc": network.synapse.get_psc(0, psc=receptor_psc, validate_nan=False),
        "ipsc": network.synapse.get_psc(1, psc=receptor_psc, validate_nan=False),
    }


if __name__ == "__main__":
    from pathlib import Path

    main(
        "izhikevich",
        build_model,
        defaults=ExampleOptions(
            neurons=60_000,
            density=33_072_348 / 60_000**2,
            seed=42,
            device="cuda",
            neuron_backend="triton",
            sparse_backend="triton",
            method="push",
            cudagraph=True,
            chunk_size=200,
        ),
        execution=RunOptions(
            steps=5000,
            target_rate_hz=5.0,
            mode="inference",
            compare=True,
            recordings=Path("fig/rsnn_izhikevich/recordings.pt"),
        ),
        recording_view=recording_view,
    )
