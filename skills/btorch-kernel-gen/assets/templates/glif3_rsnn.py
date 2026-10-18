"""Build actual btorch models at single, pair and network scales.

Copy this factory into the task workspace and customize
parameters/topology. Sequence execution is delegated to RecurrentNN, not
another simulator.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from btorch.models import environ, functional
from btorch.models.connection import SparseConnection
from btorch.models.neurons import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC


def build_model(
    n_neuron: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> RecurrentNN:
    """Build one GLIF3/AlphaPSC network factory for every validation scale.

    Args:
        n_neuron: Population size; use 1, 2 and the target network size.
        device: Model and topology device.
        dtype: Floating parameter and state dtype.

    Returns:
        Stateful recurrent model recording membrane, adaptation and PSC.
    """
    target = torch.device(device)
    # One neuron has no recurrent edges. For a pair, the only edge is 0 -> 1;
    # larger examples use the same feedforward-chain topology construction.
    pre = torch.arange(n_neuron - 1, device=target)
    post = pre + 1
    connection = SparseConnection.from_edges(
        pre,
        post,
        n_neuron,
        n_neuron,
        values=torch.full((pre.numel(),), 2000.0, device=target, dtype=dtype),
    )
    neuron = GLIF3(
        n_neuron,
        v_threshold=-45.0,
        v_reset=-65.0,
        c_m=200.0,
        tau=20.0,
        # Explicit [neuron, channel] avoids a length-two list being interpreted
        # as one heterogeneous component when the population also has two cells.
        k=torch.tensor([0.05, 0.01], device=target, dtype=dtype).expand(n_neuron, 2),
        asc_amps=torch.tensor([-50.0, -10.0], device=target, dtype=dtype).expand(
            n_neuron, 2
        ),
        tau_ref=2.0,
        device=target,
        dtype=dtype,
    )
    psc = AlphaPSC(n_neuron, tau_syn=5.0, linear=connection).to(
        device=target, dtype=dtype
    )
    return RecurrentNN(
        neuron,
        psc,
        update_state_names=("neuron.v", "neuron.Iasc", "synapse.psc"),
        unroll=1,
    )


def run_response(
    model: RecurrentNN,
    x_seq: Float[Tensor, "time batch neuron"],
    *,
    dt: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Run the model's native sequence interface from freshly initialized
    state.

    Args:
        model: Actual recurrent model under validation.
        x_seq: Input current with explicit time, batch and neuron axes.
        dt: Inference timestep in the model's time units.

    Returns:
        Spikes and requested state sequences.
    """
    functional.init_net_state(
        model, batch_size=x_seq.shape[1], device=x_seq.device, dtype=x_seq.dtype
    )
    with environ.context(dt=dt):
        return model(x_seq)
