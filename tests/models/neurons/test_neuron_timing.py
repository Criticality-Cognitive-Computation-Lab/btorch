"""Neuron timing audit: at which step does an input / a spike effect show up?

Convention shared by every neuron (and every PSC, see
``tests/models/test_synapse.py``):

* an input current delivered at step ``t`` already affects ``v`` and the
  emitted spike of step ``t`` (no extra-dt latency);
* a spike's reset / adaptation (``g_k``, ``Iasc``, ``u``, ``i_bap``,
  ``theta_th``) is applied at the end of step ``t`` and therefore first
  affects step ``t + 1``.

Each test uses a *pair* of otherwise identical neurons that differ only in the
parameter under test (e.g. ``dg_k = 0`` vs ``dg_k > 0``).  Their outputs must
be identical at the step of the spike and differ from the next step onward.
"""

import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons import (
    ALIF,
    ELIF,
    GLIF3,
    LIF,
    Izhikevich,
    MixedNeuronPopulation,
    TwoCompartmentGLIF,
)
from btorch.models.neurons.dlif import DBNN, DLIF
from btorch.models.synapse import AlphaPSC, ExponentialPSC


T = 6


@pytest.fixture(autouse=True)
def _dt_context():
    with environ.context(dt=1.0):
        yield


def _run(neuron, x_seq, **kw):
    """Step ``neuron`` over ``x_seq`` and return (spikes, voltages)."""
    init_net_state(neuron, dtype=torch.float32)
    spikes, vs = [], []
    for x in x_seq:
        out = neuron.single_step_forward(x, **kw)
        s = out[0] if isinstance(out, tuple) else out
        spikes.append(s.flatten()[0].item())
        vs.append(neuron.v.flatten()[0].item())
    return torch.tensor(spikes), torch.tensor(vs)


def _pulse(amp: float, n: int = 1) -> torch.Tensor:
    """Input that is ``amp`` at step 0 and zero afterwards."""
    x = torch.zeros(T, n)
    x[0] = amp
    return x


@pytest.mark.parametrize(
    "make",
    [
        lambda: LIF(1, v_threshold=1.0, tau=20.0, hard_reset=True),
        lambda: ALIF(1, v_threshold=1.0, E_k=0.0, hard_reset=True),
        lambda: ELIF(1, v_threshold=1.0, E_k=0.0, hard_reset=True),
        lambda: GLIF3(1, v_threshold=1.0, v_reset=0.0, c_m=1.0, hard_reset=True),
        lambda: Izhikevich(
            1,
            v_threshold=-50.0,
            v_reset=-65.0,
            v_rest=-65.0,
            v_peak=0.0,
            c_m=1.0,
            hard_reset=True,
        ),
    ],
    ids=["LIF", "ALIF", "ELIF", "GLIF3", "Izhikevich"],
)
def test_input_current_spikes_in_same_step(make):
    """A supra-threshold input at step 0 spikes at step 0 (not step 1)."""
    neuron = make()
    amp = 500.0  # large enough for every model's threshold
    spikes, vs = _run(neuron, _pulse(amp))
    assert spikes[0].item() > 0.5
    # The reset is applied inside the same step: the returned state is the
    # post-reset one, far below the huge pre-reset voltage.
    assert vs[0].item() < 50.0


@pytest.mark.parametrize("cls", [LIF, ALIF, GLIF3])
def test_subthreshold_input_changes_v_in_same_step(cls):
    """V returned after step 0 already contains the step-0 input."""
    neuron = cls(1, v_threshold=100.0)
    _, v_in = _run(neuron, _pulse(1.0))
    _, v_zero = _run(cls(1, v_threshold=100.0), _pulse(0.0))
    assert abs(v_in[0].item() - v_zero[0].item()) > 1e-4


def test_alif_spike_adaptation_acts_from_next_step():
    """dg_k added at step 0 is first seen in v at step 1, not step 0."""
    # Pulse makes both neurons spike at step 0 and hard-reset to v_reset;
    # v at step 1 then differs only through g_k * (v - E_k).  dg_k is small
    # enough to keep step 1 sub-threshold so v is not reset again.
    kw = dict(v_threshold=1.0, v_reset=0.0, E_k=-70.0, hard_reset=True)
    base = ALIF(1, dg_k=0.0, **kw)
    adapt = ALIF(1, dg_k=0.005, **kw)
    x = _pulse(500.0)
    s0, v0 = _run(base, x)
    s1, v1 = _run(adapt, x)
    assert s0[0] == s1[0] == 1.0  # spike at step 0 for both
    assert v0[0].item() == pytest.approx(v1[0].item())  # no effect at step 0
    assert abs(v0[1].item() - v1[1].item()) > 1e-3  # effect at step 1


def test_glif_asc_acts_from_next_step():
    """Iasc incremented at step 0 first shows in v at step 1."""
    kw = dict(v_threshold=1.0, v_reset=0.0, hard_reset=True, c_m=1.0)
    base = GLIF3(1, asc_amps=[0.0], **kw)
    asc = GLIF3(1, asc_amps=[0.3], **kw)  # keeps step 1 sub-threshold
    x = _pulse(500.0)
    s0, v0 = _run(base, x)
    s1, v1 = _run(asc, x)
    assert s0[0] == s1[0] == 1.0
    assert v0[0].item() == pytest.approx(v1[0].item())
    assert abs(v0[1].item() - v1[1].item()) > 1e-3


def test_izhikevich_recovery_jump_acts_from_next_step():
    """D added to u at the spike step first shows in v at step 1."""
    kw = dict(v_threshold=-50.0, v_reset=-65.0, v_rest=-65.0, v_peak=0.0, c_m=1.0)
    base = Izhikevich(1, d=0.0, hard_reset=True, **kw)
    jump = Izhikevich(1, d=50.0, hard_reset=True, **kw)
    x = _pulse(500.0)
    s0, v0 = _run(base, x)
    s1, v1 = _run(jump, x)
    assert s0[0] == s1[0] == 1.0
    assert v0[0].item() == pytest.approx(v1[0].item())
    assert abs(v0[1].item() - v1[1].item()) > 1e-3


def test_refractory_blocks_from_next_step():
    """tau_ref blocks spikes after, never at, the spike step."""
    neuron = LIF(1, v_threshold=1.0, tau_ref=3.0)
    spikes, _ = _run(neuron, torch.full((T, 1), 5.0))
    assert spikes[0] == 1.0
    assert spikes[1] == 0.0 and spikes[2] == 0.0


def test_two_compartment_timing():
    """Soma/apical input act in the same step; bAP / threshold at next."""
    kw = dict(v_threshold=1.0, v_reset=0.0, E_L=0.0, delta_th=0.0)
    # Soma pulse -> same-step spike.
    n = TwoCompartmentGLIF(1, **kw)
    spikes, _ = _run(n, _pulse(500.0))
    assert spikes[0] == 1.0

    # Apical pulse reaches the soma voltage in the same step (w_as * i_a_next).
    na = TwoCompartmentGLIF(1, w_as=1.0, **kw)
    zeros = torch.zeros(T, 1)
    init_net_state(na, dtype=torch.float32)
    na.single_step_forward(zeros[0], _pulse(1.0)[0])
    v_with_apical = na.v.item()
    nb = TwoCompartmentGLIF(1, w_as=1.0, **kw)
    init_net_state(nb, dtype=torch.float32)
    nb.single_step_forward(zeros[0], zeros[0])
    assert abs(v_with_apical - nb.v.item()) > 1e-4

    # Back-propagating AP current (w_sa * spike) first acts at step 1.
    base = TwoCompartmentGLIF(1, w_sa=0.0, **kw)
    bap = TwoCompartmentGLIF(1, w_sa=5.0, w_as=1.0, **kw)
    x = _pulse(500.0)
    _, v0 = _run(base, x)
    _, v1 = _run(bap, x)
    assert v0[0].item() == pytest.approx(v1[0].item())
    assert abs(v0[1].item() - v1[1].item()) > 1e-4


def test_mixed_population_matches_member_timing():
    """MixedNeuronPopulation adds no latency over its members."""
    lif = LIF(1, v_threshold=1.0)
    mixed = MixedNeuronPopulation([(1, LIF(1, v_threshold=1.0))])
    s_ref, _ = _run(lif, _pulse(5.0))
    init_net_state(mixed, dtype=torch.float32)
    s_mix = torch.stack(
        [mixed.single_step_forward(x).flatten()[0] for x in _pulse(5.0)]
    )
    torch.testing.assert_close(s_mix, s_ref)
    assert s_mix[0] == 1.0


def _neutral_mixing(cell):
    """Zero the random bilinear weights/bias so the mixing is the identity."""
    with torch.no_grad():
        cell.bilinear.weight.zero_()
        cell.bilinear.bias.zero_()


@pytest.mark.parametrize("psc_cls", [ExponentialPSC, AlphaPSC])
def test_dlif_input_spikes_in_same_step(psc_cls):
    """Dendritic LIF: receptor input at step t drives the soma at step t.

    With the same-step PSC convention (see test_synapse.py) a DBNN with any PSC
    has no extra-dt latency in front of the soma.
    """
    # DLIF has no synaptic dynamics: pure bilinear + linear mixing.
    cell = DLIF(1, n_receptor=1)
    _neutral_mixing(cell)
    init_net_state(cell, dtype=torch.float32)
    x = torch.zeros(T, 1, 1)
    x[0] = 500.0
    s = torch.stack([cell.single_step_forward(xi).flatten()[0] for xi in x])
    assert s[0] == 1.0

    # DBNN with an explicit PSC: input at step 0 must reach the soma at step 0.
    cell = DBNN(1, 1, synapse_cls=psc_cls, synapse_kwargs={"tau_syn": 3.0})
    _neutral_mixing(cell)
    init_net_state(cell, dtype=torch.float32)
    s = torch.stack([cell.single_step_forward(xi).flatten()[0] for xi in x])
    assert s[0] == 1.0
