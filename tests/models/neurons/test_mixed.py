"""Tests for MixedNeuronPopulation — heterogeneous neuron population container.

Covers functional equivalence, slicing/concatenation order, multi-step
dispatch, apical routing, gradient flow, state management, and edge
cases.
"""

import pytest
import torch

from btorch.models import environ
from btorch.models.functional import detach_net, init_net_state, reset_net_state
from btorch.models.neurons import GLIF3, TwoCompartmentGLIF
from btorch.models.neurons.mixed import MixedNeuronPopulation


DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Functional equivalence: single-group mixed must match raw neuron
# ---------------------------------------------------------------------------


def test_single_group_equivalence_glif():
    torch.manual_seed(10)
    batch_size, n = 3, 7

    raw = GLIF3(n_neuron=n, step_mode="s")
    mixed = MixedNeuronPopulation(
        [(n, GLIF3(n_neuron=n, step_mode="s"))], step_mode="s"
    )

    init_net_state(raw, batch_size=batch_size, dtype=DTYPE)
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n, dtype=DTYPE)
    with environ.context(dt=1.0):
        s_raw = raw(x)
        s_mixed = mixed(x)

    torch.testing.assert_close(s_raw, s_mixed)


def test_single_group_equivalence_tc():
    torch.manual_seed(11)
    batch_size, n = 3, 7

    raw = TwoCompartmentGLIF(n_neuron=n, step_mode="s")
    mixed = MixedNeuronPopulation(
        [(n, TwoCompartmentGLIF(n_neuron=n, step_mode="s"))], step_mode="s"
    )

    init_net_state(raw, batch_size=batch_size, dtype=DTYPE)
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n, dtype=DTYPE)
    x_a = torch.randn(batch_size, n, dtype=DTYPE)

    with environ.context(dt=1.0):
        s_raw, _ = raw(x, x_a)
        s_mixed = mixed(x, x_a)

    torch.testing.assert_close(s_raw, s_mixed)


# ---------------------------------------------------------------------------
# Slicing and concatenation order
# ---------------------------------------------------------------------------


def test_concatenation_order_preserves_group_boundaries():
    torch.manual_seed(20)
    batch_size = 2
    n1, n2 = 3, 4

    glif1 = GLIF3(n_neuron=n1, v_threshold=-50.0, step_mode="s")
    glif2 = GLIF3(n_neuron=n2, v_threshold=-50.0, step_mode="s")

    mixed = MixedNeuronPopulation([(n1, glif1), (n2, glif2)], step_mode="s")
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    # Set group 0 above threshold (spikes) and group 1 below (silent)
    with torch.no_grad():
        glif1.v.fill_(0.0)
        glif2.v.fill_(-60.0)

    x = torch.zeros(batch_size, n1 + n2, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes = mixed(x)

    # First n1 neurons should spike, last n2 should stay silent.
    torch.testing.assert_close(spikes[:, :n1], torch.ones_like(spikes[:, :n1]))
    torch.testing.assert_close(spikes[:, n1:], torch.zeros_like(spikes[:, n1:]))


def test_input_slices_match_group_counts():
    torch.manual_seed(21)
    batch_size = 2
    n_glif, n_tc = 3, 5

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    # Set TC voltage above threshold, GLIF below — verify correct slicing
    with torch.no_grad():
        glif.v.fill_(-60.0)  # below -50 threshold => silent
        tc.v.fill_(0.0)  # above -50 threshold => spike

    x = torch.zeros(batch_size, n_glif + n_tc, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes = mixed(x)

    # GLIF slice (first n_glif) should be silent, TC slice should fire
    torch.testing.assert_close(spikes[:, :n_glif], torch.zeros_like(spikes[:, :n_glif]))
    torch.testing.assert_close(spikes[:, n_glif:], torch.ones_like(spikes[:, n_glif:]))


# ---------------------------------------------------------------------------
# Multi-step dispatch
# ---------------------------------------------------------------------------


def test_multi_step_forward_matches_manual_loop():
    torch.manual_seed(30)
    T, batch_size, n_glif, n_tc = 8, 2, 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="m")

    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n_glif + n_tc, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes_multi = mixed(x)

    # Compare against manual single-step loop
    reset_net_state(mixed, batch_size=batch_size)
    spikes_loop = []
    with environ.context(dt=1.0):
        for t in range(T):
            spikes_loop.append(mixed.single_step_forward(x[t]))
    spikes_loop = torch.stack(spikes_loop, dim=0)

    torch.testing.assert_close(spikes_multi, spikes_loop)


# ---------------------------------------------------------------------------
# Apical input routing
# ---------------------------------------------------------------------------


def test_apical_only_reaches_two_compartment_groups():
    torch.manual_seed(40)
    batch_size = 2
    n_glif, n_tc = 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s", detach_reset=True)
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s", detach_reset=True)
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.zeros(batch_size, n_glif + n_tc, dtype=DTYPE)
    x_apical = torch.zeros(batch_size, n_glif + n_tc, dtype=DTYPE)
    x_apical[:, n_glif:] = 10.0  # strong apical only in TC slice

    with environ.context(dt=1.0):
        mixed(x, x_apical)
    i_a_with = tc.i_a.clone()

    reset_net_state(mixed, batch_size=batch_size)
    x_apical_zero = x_apical.clone()
    x_apical_zero[:, n_glif:] = 0.0
    with environ.context(dt=1.0):
        mixed(x, x_apical_zero)
    i_a_without = tc.i_a.clone()

    assert not torch.allclose(
        i_a_with, i_a_without
    ), "Apical input should change TC apical current"


def test_no_apical_falls_back_gracefully():
    torch.manual_seed(41)
    batch_size, n_glif, n_tc = 2, 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n_glif + n_tc, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes = mixed(x)

    assert spikes.shape == (batch_size, n_glif + n_tc)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_through_all_subpopulations():
    torch.manual_seed(50)
    batch_size, n_glif, n_tc = 2, 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s", trainable_param={"tau"})
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s", trainable_param={"tau_s"})
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n_glif + n_tc, dtype=DTYPE, requires_grad=True)
    with environ.context(dt=1.0):
        spikes = mixed(x)
    loss = spikes.sum()
    loss.backward()

    assert x.grad is not None, "Gradient should reach input"
    assert glif.tau.grad is not None, "Gradient should reach GLIF3 tau"
    assert tc.tau_s.grad is not None, "Gradient should reach TC tau_s"


def test_gradient_with_apical_flows_to_input():
    torch.manual_seed(51)
    batch_size, n_glif, n_tc = 2, 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s", trainable_param={"tau_s"})
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="s")
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, n_glif + n_tc, dtype=DTYPE, requires_grad=True)
    x_apical = torch.randn(batch_size, n_glif + n_tc, dtype=DTYPE, requires_grad=True)
    with environ.context(dt=1.0):
        spikes = mixed(x, x_apical)
    loss = spikes.sum()
    loss.backward()

    assert x.grad is not None
    assert x_apical.grad is not None
    assert tc.tau_s.grad is not None


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def test_init_state_propagates_to_all_subpopulations():
    batch_size = 3
    glif = GLIF3(n_neuron=5, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=5, step_mode="s")
    mixed = MixedNeuronPopulation({"glif": (5, glif), "tc": (5, tc)})
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    assert glif.v.shape == (batch_size, 5)
    assert glif.Iasc.shape == (batch_size, 5, glif.n_Iasc)
    assert tc.v.shape == (batch_size, 5)
    assert tc.i_a.shape == (batch_size, 5)


def test_detach_net_detaches_all_memories():
    batch_size = 2
    glif = GLIF3(n_neuron=5, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=5, step_mode="s")
    mixed = MixedNeuronPopulation([(5, glif), (5, tc)])
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    glif.v.requires_grad_(True)
    tc.i_a.requires_grad_(True)

    detach_net(mixed)

    assert glif.v.grad_fn is None, "GLIF3 v should be detached"
    assert tc.i_a.grad_fn is None, "TC i_a should be detached"


# ---------------------------------------------------------------------------
# Naming and repr
# ---------------------------------------------------------------------------


def test_dict_groups_accessible_by_name():
    batch_size = 2
    glif = GLIF3(n_neuron=3, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=3, step_mode="s")
    mixed = MixedNeuronPopulation({"exc": (3, glif), "inh": (3, tc)}, step_mode="s")
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, 6, dtype=DTYPE)
    with environ.context(dt=1.0):
        mixed(x)

    assert hasattr(mixed, "exc"), "Group should be accessible by dict key"
    assert hasattr(mixed, "inh"), "Group should be accessible by dict key"


def test_extra_repr_shows_group_names_and_counts():
    glif = GLIF3(n_neuron=10)
    tc = TwoCompartmentGLIF(n_neuron=20)
    mixed = MixedNeuronPopulation({"fast": (10, glif), "slow": (20, tc)})
    repr_str = repr(mixed)
    assert "MixedNeuronPopulation" in repr_str
    assert "fast=GLIF3" in repr_str
    assert "slow=TwoCompartmentGLIF" in repr_str
    assert "n_neuron=(30,)" in repr_str


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_groups_raises():
    with pytest.raises(ValueError, match="at least one population"):
        MixedNeuronPopulation([])


def test_non_positive_count_raises():
    with pytest.raises(ValueError, match="count must be positive"):
        MixedNeuronPopulation([(0, GLIF3(n_neuron=1))])


def test_negative_count_raises():
    with pytest.raises(ValueError, match="count must be positive"):
        MixedNeuronPopulation([(-3, GLIF3(n_neuron=3))])


def test_dtype_consistency_across_subpopulations():
    torch.manual_seed(60)
    batch_size = 2
    glif = GLIF3(n_neuron=4, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=4, step_mode="s")
    mixed = MixedNeuronPopulation([(4, glif), (4, tc)], step_mode="s")
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(batch_size, 8, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes = mixed(x)

    assert spikes.dtype == DTYPE, "Output dtype should match input"


def test_step_mode_m_with_s_mode_neurons_works():
    torch.manual_seed(70)
    T, batch_size, n_glif, n_tc = 5, 2, 4, 4

    glif = GLIF3(n_neuron=n_glif, step_mode="s")
    tc = TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")
    mixed = MixedNeuronPopulation([(n_glif, glif), (n_tc, tc)], step_mode="m")
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)

    x = torch.randn(T, batch_size, n_glif + n_tc, dtype=DTYPE)
    with environ.context(dt=1.0):
        spikes = mixed(x)

    assert spikes.shape == (T, batch_size, n_glif + n_tc)
