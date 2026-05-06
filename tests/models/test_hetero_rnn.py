import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.linear import DenseConn
from btorch.models.neurons import GLIF3, TwoCompartmentGLIF
from btorch.models.neurons.mixed import MixedNeuronPopulation
from btorch.models.rnn import ApicalRecurrentNN, RecurrentNN
from btorch.models.synapse import AlphaPSC


DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Equivalence: ApicalRecurrentNN must match RecurrentNN when no apical used
# ---------------------------------------------------------------------------


def test_apical_rnn_matches_standard_rnn_without_apical():
    torch.manual_seed(0)
    T, batch_size, n = 6, 2, 8
    n_glif, n_tc = 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    conn = DenseConn(n, n, bias=None)
    psc = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=conn, step_mode="s")

    brain_apical = ApicalRecurrentNN(neuron=mixed, synapse=psc, step_mode="m", unroll=2)
    init_net_state(brain_apical, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes_a, states_a = brain_apical(x)

    assert spikes_a.shape == (T, batch_size, n)
    assert any("group_0.v" in k for k in states_a)
    assert any("group_1.v" in k for k in states_a)


def test_apical_rnn_without_apical_equivalent_to_standard():
    torch.manual_seed(100)
    T, batch_size, n = 4, 2, 6

    # Build two identical networks: one with ApicalRecurrentNN, one with
    # RecurrentNN. They should produce identical output when no apical is used.
    glif_a = GLIF3(n_neuron=n, step_mode="s")
    glif_b = GLIF3(n_neuron=n, step_mode="s")

    mixed_a = MixedNeuronPopulation([(n, glif_a)], step_mode="s")

    conn_a = DenseConn(n, n, bias=None)
    conn_b = DenseConn(n, n, bias=None)
    psc_a = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=conn_a, step_mode="s")
    psc_b = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=conn_b, step_mode="s")

    brain_apical = ApicalRecurrentNN(
        neuron=mixed_a, synapse=psc_a, step_mode="m", unroll=2
    )
    brain_standard = RecurrentNN(neuron=glif_b, synapse=psc_b, step_mode="m", unroll=2)

    # Sync all weights and states
    with torch.no_grad():
        conn_b.weight.copy_(conn_a.weight)
        glif_b.tau.copy_(glif_a.tau)

    init_net_state(brain_apical, batch_size=batch_size, dtype=DTYPE)
    init_net_state(brain_standard, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n, dtype=DTYPE)
    with environ.context(dt=1.0):
        out_a, _ = brain_apical(x)
        out_b, _ = brain_standard(x)

    torch.testing.assert_close(out_a, out_b)


# ---------------------------------------------------------------------------
# Apical input routing in recurrent context
# ---------------------------------------------------------------------------


def test_apical_rnn_changes_tc_state():
    torch.manual_seed(1)
    T, batch_size, n = 6, 2, 10
    n_glif, n_tc = 5, 5

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    conn = DenseConn(n, n, bias=None)
    psc = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=conn, step_mode="s")

    brain = ApicalRecurrentNN(neuron=mixed, synapse=psc, step_mode="m", unroll=2)
    init_net_state(brain, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n, dtype=DTYPE)
    x_apical = torch.randn(T, batch_size, n, dtype=DTYPE)

    with environ.context(dt=1.0):
        _, _ = brain(x, None, x_apical)
    i_a_with = tc.i_a.clone()

    reset_net_state(brain, batch_size=batch_size)
    with environ.context(dt=1.0):
        _, _ = brain(x)
    i_a_without = tc.i_a.clone()

    assert not torch.allclose(
        i_a_with, i_a_without
    ), "Apical drive should alter TC apical current"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_apical_rnn_gradient_flows():
    torch.manual_seed(2)
    T, batch_size, n = 4, 2, 8
    n_glif, n_tc = 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s", trainable_param={"tau"})
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s", trainable_param={"tau_s"})
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    conn = DenseConn(n, n, bias=None)
    psc = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=conn, step_mode="s")

    brain = ApicalRecurrentNN(neuron=mixed, synapse=psc, step_mode="m", unroll=2)
    init_net_state(brain, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n, dtype=DTYPE, requires_grad=True)
    x_apical = torch.randn(T, batch_size, n, dtype=DTYPE, requires_grad=True)

    with environ.context(dt=1.0):
        spikes, _ = brain(x, None, x_apical)
    loss = spikes.sum()
    loss.backward()

    assert x.grad is not None, "Gradient should reach somatic input"
    assert x_apical.grad is not None, "Gradient should reach apical input"
    assert glif.tau.grad is not None, "Gradient should reach GLIF3 tau"
    assert tc.tau_s.grad is not None, "Gradient should reach TC tau_s"


def test_apical_rnn_gradient_matches_standard_rnn():
    torch.manual_seed(110)
    T, batch_size, n = 4, 2, 6

    baseline_neuron = GLIF3(n_neuron=n, step_mode="s", trainable_param={"tau"})
    baseline_conn = DenseConn(n, n, bias=None)
    baseline_psc = AlphaPSC(
        n_neuron=n, tau_syn=5.0, linear=baseline_conn, step_mode="s"
    )
    baseline_rnn = RecurrentNN(
        neuron=baseline_neuron,
        synapse=baseline_psc,
        step_mode="m",
        unroll=2,
    )

    mixed_neuron = MixedNeuronPopulation(
        [(n, GLIF3(n_neuron=n, step_mode="s", trainable_param={"tau"}))],
        step_mode="s",
    )
    mixed_conn = DenseConn(n, n, bias=None)
    mixed_psc = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=mixed_conn, step_mode="s")
    mixed_rnn = ApicalRecurrentNN(
        neuron=mixed_neuron,
        synapse=mixed_psc,
        step_mode="m",
        unroll=2,
    )

    with torch.no_grad():
        mixed_conn.weight.copy_(baseline_conn.weight)

    init_net_state(baseline_rnn, batch_size=batch_size, dtype=DTYPE)
    init_net_state(mixed_rnn, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n, dtype=DTYPE, requires_grad=True)

    with environ.context(dt=1.0):
        out_b, _ = baseline_rnn(x)
        out_m, _ = mixed_rnn(x)

    torch.testing.assert_close(out_b, out_m)

    out_b.sum().backward()
    out_m.sum().backward()

    mixed_glif = mixed_rnn.neuron.group_0
    assert mixed_glif.tau.grad is not None
    torch.testing.assert_close(baseline_neuron.tau.grad, mixed_glif.tau.grad)


# ---------------------------------------------------------------------------
# Parametrized: various unroll / checkpoint / offload configurations
# ---------------------------------------------------------------------------

RNN_CONFIGS = [
    {"unroll": 2, "chunk_size": None, "grad_checkpoint": False, "cpu_offload": False},
    {"unroll": 2, "chunk_size": 4, "grad_checkpoint": True, "cpu_offload": False},
    {"unroll": 1, "chunk_size": None, "grad_checkpoint": False, "cpu_offload": False},
    {
        "unroll": False,
        "chunk_size": None,
        "grad_checkpoint": False,
        "cpu_offload": False,
    },
]


@pytest.mark.parametrize("cfg", RNN_CONFIGS)
def test_apical_rnn_configs(cfg):
    torch.manual_seed(60)
    T, batch_size, n = 6, 2, 8
    n_glif, n_tc = 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")

    conn = DenseConn(n, n, bias=None)
    psc = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=conn, step_mode="s")

    brain = ApicalRecurrentNN(
        neuron=mixed,
        synapse=psc,
        step_mode="m",
        **cfg,
    )
    init_net_state(brain, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n, dtype=DTYPE)
    x_a = torch.randn(T, batch_size, n, dtype=DTYPE)

    with environ.context(dt=1.0):
        spikes, states = brain(x, None, x_a)

    assert spikes.shape == (T, batch_size, n)
    assert len(states) > 0, "Should collect states from sub-populations"


# ---------------------------------------------------------------------------
# State persistence across resets
# ---------------------------------------------------------------------------


def test_apical_rnn_state_consistent_across_resets():
    torch.manual_seed(120)
    T, batch_size, n = 6, 2, 8

    glif = GLIF3(n_neuron=n, step_mode="s")
    mixed = MixedNeuronPopulation([(n, glif)], step_mode="s")
    conn = DenseConn(n, n, bias=None)
    psc = AlphaPSC(n_neuron=n, tau_syn=5.0, linear=conn, step_mode="s")

    brain = ApicalRecurrentNN(neuron=mixed, synapse=psc, step_mode="m", unroll=2)
    init_net_state(brain, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes_1, _ = brain(x)

    reset_net_state(brain, batch_size=batch_size)
    with environ.context(dt=1.0):
        spikes_2, _ = brain(x)

    torch.testing.assert_close(spikes_1, spikes_2)
