"""Default-argument hygiene for neuron constructors.

Neurons must not share mutable/instance default arguments: a default
``surrogate_function=Sigmoid()`` evaluated once at import time would be one
``nn.Module`` instance registered as a submodule of every neuron, and a
default ``trainable_param=set()`` would be one shared set object.  Defaults
are therefore ``None`` and the objects are built inside ``__init__``.

``tau_ref`` default is ``None`` (refractory period disabled, no refractory
state allocated) for every neuron that has it.
"""

import inspect

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
    TwoCompartmentGLIF,
)
from btorch.models.neurons.lif import IF


NEURONS = [LIF, IF, ALIF, ELIF, GLIF3, Izhikevich, TwoCompartmentGLIF]


@pytest.mark.parametrize("cls", NEURONS)
def test_no_instance_or_mutable_defaults_in_signature(cls):
    """No constructor default may be a mutable container or a Module."""
    for name, p in inspect.signature(cls.__init__).parameters.items():
        d = p.default
        assert not isinstance(
            d, (list, dict, set, torch.nn.Module)
        ), f"{cls.__name__}.__init__({name}=...) has a shared mutable default"


@pytest.mark.parametrize("cls", NEURONS)
def test_default_surrogate_and_trainable_set_not_shared(cls):
    """Two neurons built with defaults share no surrogate module or set."""
    a, b = cls(4), cls(4)
    assert a.surrogate_function is not b.surrogate_function
    assert a.trainable_param is not b.trainable_param
    # Registered submodules must be distinct objects too.
    assert a.surrogate_function in a.children()
    assert a.surrogate_function not in list(b.children())
    # Mutating one neuron's set must not leak into the other.
    a.trainable_param.add("leaked")
    assert "leaked" not in b.trainable_param
    assert "leaked" not in cls(4).trainable_param


def test_glif_default_after_spike_current_not_shared():
    """GLIF3 default k/asc_amps are fresh per instance (single ASC, zero)."""
    a, b = GLIF3(3), GLIF3(3)
    assert a.n_Iasc == 1
    assert a.asc_amps is not b.asc_amps
    torch.testing.assert_close(a.asc_amps, torch.zeros_like(a.asc_amps))


@pytest.mark.parametrize("cls", [LIF, ALIF, ELIF, GLIF3])
def test_tau_ref_defaults_to_none_and_disables_refractory(cls):
    """tau_ref=None is the single default: refractory fully disabled."""
    assert inspect.signature(cls.__init__).parameters["tau_ref"].default is None
    n = cls(2)
    assert n.tau_ref is None
    assert n._use_refractory is False
    assert not hasattr(n, "refractory") or n.refractory is None


@pytest.mark.parametrize("cls", [LIF, ALIF, GLIF3])
def test_tau_ref_zero_equals_none_spike_train(cls):
    """tau_ref=0.0 never blocks a spike, so it matches the None default."""
    torch.manual_seed(0)
    x = torch.rand(60, 3) * 3.0
    outs = []
    for tau_ref in (None, 0.0):
        with environ.context(dt=1.0):
            n = cls(3, tau_ref=tau_ref)
            init_net_state(n, dtype=torch.float32)
            outs.append(torch.stack([n.single_step_forward(xi) for xi in x]))
    torch.testing.assert_close(outs[0], outs[1])


def test_if_neuron_steps_without_error():
    """IF.neuronal_charge used a non-existent ``self.V`` (AttributeError)."""
    with environ.context(dt=1.0):
        n = IF(2, v_threshold=1.0)
        init_net_state(n, dtype=torch.float32)
        s = [n.single_step_forward(torch.full((2,), 0.4)) for _ in range(3)]
    # No leak: 0.4, 0.8, 1.2 -> first spike on the third step.
    assert [x.sum().item() for x in s] == [0.0, 0.0, 2.0]
