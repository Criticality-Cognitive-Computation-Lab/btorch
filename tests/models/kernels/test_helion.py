"""Compare genuine fused timesteps with the ordinary recurrent schedule."""

import pytest
import torch

from btorch.models import KernelOptions, environ, functional
from btorch.models.connection import EdgeWeight, SparseConnection, Synapse
from btorch.models.neurons import Izhikevich
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import DualExponentialPSC, HeterSynapsePSC
from tests.models.kernels.helpers import require_cuda_backend


def _network(backend, dtype=torch.float32):
    # 129 neurons leave 127 inactive lanes in the final CTA. Every lane still
    # has to reach the cooperative barrier; putting it inside the neuron mask
    # would deadlock. Signed receptor channels also expose drive-layout errors.
    n = 129
    source = torch.arange(n).repeat(2)
    target = torch.cat(((torch.arange(n) + 1) % n, (torch.arange(n) + 7) % n))
    receptor = source % 2
    connection = SparseConnection.from_edges(
        source,
        target,
        n,
        n,
        Synapse(
            weight=EdgeWeight(torch.where(receptor == 0, 0.03, -0.02), trainable=False),
            receptor=receptor,
            n_receptor=2,
        ),
    )
    model = RecurrentNN(
        Izhikevich(
            n,
            v_threshold=0.7,
            v_reset=0.0,
            v_rest=0.0,
            v_peak=1.0,
            c_m=1.0,
            k=0.1,
            a=torch.linspace(0.025, 0.035, n),
            b=0.1,
            d=0.05,
            hard_reset=True,
        ),
        HeterSynapsePSC(
            n,
            2,
            None,
            connection,
            base_psc=DualExponentialPSC,
            tau_rise=torch.tensor([1.0, 1.5]).repeat(n),
            tau_decay=torch.tensor([4.0, 6.0]).repeat(n),
        ),
        backend=backend,
        update_state_names=(
            "neuron.v",
            "neuron.u",
            "synapse.psc",
            "synapse.base_psc.psc",
            "synapse.base_psc.g_rise",
            "synapse.base_psc.g_decay",
        ),
        chunk_size=200,
    ).to(device="cuda", dtype=dtype)
    functional.init_net_state(model, batch_size=1, device="cuda", dtype=dtype)
    return model


@pytest.mark.parametrize(
    ("backend", "dtype"),
    (("triton", torch.float32), ("tilelang", torch.float32), ("triton", torch.float64)),
)
def test_helion_izhikevich_dual_psc_preserves_physics_and_continuation(backend, dtype):
    """Fuse the schedule without changing PSC timing or recorded memories."""
    require_cuda_backend(backend)
    expected = _network(
        KernelOptions(neuron="torch", connection="torch", connection_method="pull"),
        dtype,
    )
    actual = _network(
        KernelOptions(
            neuron=backend,
            recurrent="helion",
            connection_method="push",
            enable_fp_fusion=False,
            precise_division=True,
            precise_exp=True,
        ),
        dtype,
    )
    actual.load_state_dict(expected.state_dict())
    # A noncontiguous time sequence exercises storage addressing without a
    # forced contiguous conversion. The changed second chunk must continue
    # both adaptation and all three PSC memories from the first chunk.
    inputs = torch.linspace(0.3, 0.9, 129, device="cuda", dtype=dtype).expand(
        31, 1, 129
    )
    tolerance = 2e-10 if dtype == torch.float64 else 4e-4
    with torch.no_grad(), environ.context(dt=1.0):
        for current in (inputs[:13], inputs[13:] * 1.1):
            spikes, records = expected(current)
            observed, actual_records = actual(current)
            torch.testing.assert_close(observed, spikes, atol=0, rtol=0)
            torch.testing.assert_close(
                actual_records, records, atol=tolerance, rtol=tolerance
            )
            actual_states = functional.named_hidden_states(actual)
            for name, value in functional.named_hidden_states(expected).items():
                torch.testing.assert_close(
                    actual_states[name],
                    value,
                    atol=tolerance,
                    rtol=tolerance,
                )
    assert spikes.count_nonzero() > 0
