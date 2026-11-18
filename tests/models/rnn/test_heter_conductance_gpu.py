"""GPU contracts for receptor conductances and actual Triton push routing.

Run only in a valid GPU allocation. The fixed-event oracle uses an independent
FP64 matrix; recurrent checks compare separately selected push/pull execution.
"""

import copy

import pytest
import torch

from btorch.models import KernelOptions, environ
from btorch.models.connection import PairSTDP, SparseConnection, Synapse
from btorch.models.functional import init_net_state
from btorch.models.kernels.adapters import connection_scope
from btorch.models.neurons import LIF
from btorch.models.rnn import ConductanceRecurrentNN
from btorch.models.synapse import HeterConductancePSC


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _edges(n=128):
    pre = torch.arange(n).repeat_interleave(8)
    post = (pre + torch.arange(1, 9).repeat(n)) % n
    receptor = (pre % 5 == 0).long()
    weight = 0.01 + torch.arange(len(pre)).remainder(7).float() * 0.001
    return pre, post, receptor, weight


def _synapse(*, plasticity=False):
    n = 128
    pre, post, receptor, weight = _edges(n)
    plus = torch.where(receptor == 0, 0.002, 0.0)
    rule = (
        PairSTDP(
            a_plus=plus,
            a_minus=plus * 1.05,
            tau_plus=20.0,
            tau_minus=20.0,
            w_min=torch.where(receptor == 0, weight * 0.25, weight),
            w_max=torch.where(receptor == 0, weight * 2, weight),
            trace_scope="neuron",
        )
        if plasticity
        else None
    )
    connection = SparseConnection.from_edges(
        pre,
        post,
        n,
        n,
        Synapse(weight=weight, receptor=receptor, n_receptor=2, plasticity=rule),
    )
    return HeterConductancePSC(
        n,
        2,
        None,
        connection,
        tau_syn=[5.0, 10.0],
        reversal_potential=[1.5, -1.0],
    ).to(device="cuda", dtype=torch.float32)


def _assert_route(connection, options, method):
    with connection_scope(options):
        explanation = connection.explain()
    assert f"algorithm = {method}" in explanation
    assert "backend = triton" in explanation


def test_fixed_events_push_matches_independent_double_matrix():
    """Sparse source accumulation must route to receptor slots before COBA."""
    torch.set_num_threads(1)
    pre, post, receptor, weight = _edges()
    matrix = torch.zeros(256, 128, dtype=torch.float64)
    matrix[post * 2 + receptor, pre] = weight.double()
    generator = torch.Generator().manual_seed(51)
    events = (torch.rand(12, 1, 128, generator=generator) < 0.07).double()
    voltage = torch.rand(12, 1, 128, generator=generator).double() * 2 - 0.7
    expected = torch.full((1, 256), 0.03, dtype=torch.float64)
    synapse = _synapse()
    options = KernelOptions(connection="triton", connection_method="push")
    with torch.no_grad(), environ.context(dt=0.2), connection_scope(options):
        init_net_state(synapse, batch_size=1, dtype=torch.float32, device="cuda")
        synapse.base_psc.psc.fill_(0.03)
        decay = torch.exp(
            -0.2 / torch.tensor([5.0, 10.0], dtype=torch.float64).repeat(128)
        )
        reversal = torch.tensor([1.5, -1.0], dtype=torch.float64).repeat(128)
        for event, v in zip(events, voltage):
            expected = expected * decay + event @ matrix.T
            per_slot = expected * (reversal - v.repeat_interleave(2, -1))
            expected_current = per_slot.reshape(1, 128, 2).sum(-1)
            actual = synapse.single_step_forward(event.float().cuda(), v.float().cuda())
            torch.testing.assert_close(
                synapse.base_psc.psc.cpu().double(), expected, atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                actual.cpu().double(), expected_current, atol=2e-6, rtol=2e-5
            )
    _assert_route(synapse.linear, options, "push")


@pytest.mark.parametrize("plasticity", [False, True])
def test_recurrent_push_pull_and_learning_keep_receptor_identity(plasticity):
    """The voltage-aware scheduler also updates weights after GPU propagation."""
    synapse = _synapse(plasticity=plasticity)
    neuron = LIF(128, v_threshold=1.0, v_reset=0.0, c_m=1.0, tau=20.0).cuda()
    push_options = KernelOptions(connection="triton", connection_method="push")
    push = ConductanceRecurrentNN(
        neuron,
        synapse,
        backend=push_options,
        step_mode="m",
        unroll=1,
        chunk_size=16,
        cudagraph=False,
        update_state_names=("neuron.v", "synapse.base_psc.psc"),
    )
    with torch.no_grad(), environ.context(dt=0.2):
        init_net_state(push, batch_size=1, dtype=torch.float32, device="cuda")
        pull = copy.deepcopy(push)
        pull.backend = KernelOptions(connection="triton", connection_method="pull")
        before = push.synapse.linear.weight.value.clone()
        receptors = push.synapse.linear.receptor.clone()
        drive = torch.full((64, 1, 128), 3.0, device="cuda")
        actual, records = push(drive)
        expected, reference = pull(drive)
        assert actual.sum() > 0
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for name in records:
            torch.testing.assert_close(
                records[name], reference[name], atol=2e-5, rtol=2e-5
            )
        torch.testing.assert_close(
            push.synapse.linear.weight.value,
            pull.synapse.linear.weight.value,
            atol=2e-6,
            rtol=2e-5,
        )
        delta = push.synapse.linear.weight.value - before
        assert torch.equal(push.synapse.linear.receptor, receptors)
        assert torch.equal(
            delta[receptors == 1], torch.zeros_like(delta[receptors == 1])
        )
        if plasticity:
            assert delta[receptors == 0].abs().max() > 0
        else:
            assert torch.count_nonzero(delta) == 0
    _assert_route(push.synapse.linear, push_options, "push")
    _assert_route(pull.synapse.linear, pull.backend, "pull")
