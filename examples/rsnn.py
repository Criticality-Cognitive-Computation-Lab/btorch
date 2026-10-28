"""Train/infer a GLIF3 + heterogeneous AlphaPSC RSNN with delay bins.

    python -m examples.rsnn
    python -m examples.rsnn --mode both --export-sonata /tmp/glif-circuit
    python -m examples.rsnn --import-sonata /tmp/glif-circuit/circuit_config.json

Two receptors use excitation/inhibition and distinct PSC time constants.
Dale constraints preserve each source neuron's sign after optimizer updates.
Delays span 0..3 bins; SONATA stores their physical values in milliseconds.
Use --device cuda --neuron-backend triton --sparse-backend triton for native
execution. --compile supports training; --cudagraph is inference-only.
"""

import torch

from btorch.models import BPTT, KernelOptions
from btorch.models.neurons import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import Sigmoid, Truncated
from btorch.models.synapse import AlphaPSC, HeterSynapsePSC
from examples.rsnn_common import ExampleOptions, build_connection, main


def build_model(options: ExampleOptions) -> RecurrentNN:
    """Build the GLIF3/AlphaPSC circuit with two receptor channels."""
    n = options.neurons
    connection = build_connection(options, weight_scale=0.15, delay_bins=4, dale=True)
    # Sparse BPTT uses the same truncated surrogate: zero derivative outside
    # the declared support in normalized threshold-distance units.
    function = Truncated(Sigmoid(alpha=100.0), width=0.25)
    neuron = GLIF3(
        n,
        v_threshold=-45.0,
        v_reset=-60.0,
        v_rest=-60.0,
        c_m=2.0,
        tau=torch.linspace(15.0, 25.0, n),
        tau_ref=2.0,
        k=[0.1, 0.2],
        asc_amps=[-0.1, -0.2],
        trainable_param={"tau"},
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
        base_psc=AlphaPSC,
        tau_syn=torch.tensor([3.0, 7.0]).repeat(n),
        g_max=1.0,
    )
    records = (
        "neuron.v",
        "neuron.Iasc",
        "synapse.base_psc.psc",
        "synapse.base_psc.h",
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
        chunk_mode=options.chunk_mode,
    ).to(options.device)


if __name__ == "__main__":
    main("glif3", build_model)
