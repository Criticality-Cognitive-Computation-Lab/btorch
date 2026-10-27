"""Train/infer an Izhikevich + heterogeneous dual-exponential PSC RSNN.

    python -m examples.rsnn_izhikevich
    python -m examples.rsnn_izhikevich --mode both --export-sonata /tmp/izh-circuit
    python -m examples.rsnn_izhikevich --import-sonata \
        /tmp/izh-circuit/circuit_config.json

Two signed receptor channels have distinct rise/decay times, with no additional
transmission delays. Connectivity is built sparsely; the larger reference
configuration is --neurons 50000 --density 0.05. Use --device cuda
--neuron-backend triton --sparse-backend triton --method push for native push.
"""

import torch

from btorch.models import BPTT, KernelOptions
from btorch.models.neurons import Izhikevich
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import Sigmoid, Truncated
from btorch.models.synapse import DualExponentialPSC, HeterSynapsePSC
from examples.rsnn_common import ExampleOptions, build_connection, main


def build_model(options: ExampleOptions) -> RecurrentNN:
    """Build the Izhikevich/DualExponentialPSC circuit with two receptor
    channels."""
    n = options.neurons
    connection = build_connection(options, weight_scale=0.04)
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
        "synapse.base_psc.g_rise",
        "synapse.base_psc.g_decay",
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
    ).to(options.device)


if __name__ == "__main__":
    main("izhikevich", build_model)
