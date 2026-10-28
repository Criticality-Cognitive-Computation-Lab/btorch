"""Model-local forward and reverse scalar definitions."""

from functools import partial

from ..templates import ModelTemplate, register_model_template
from .conductance import (
    after_fire,
    charge as conductance_charge,
    reverse_charge as conductance_reverse_charge,
    reverse_reset,
)
from .neuron import NeuronMath, neuron_fragments


charge = partial(
    conductance_charge, membrane=["v_linear_numerator = -g_leak - g_k_prev"]
)
reverse_charge = partial(
    conductance_reverse_charge,
    voltage=["grad_s_v = gcharged - grad_v_leak - grad_v_potassium"],
)
MATH = NeuronMath(
    charge,
    reverse_charge,
    after_fire,
    reverse_reset,
    reset_after_adaptation=True,
    refractory_first=True,
    ordered_normalization=True,
)


def fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
    return neuron_fragments(spec, modes, prepared_values, authoritative_output, MATH)


def module_factory(spec, states, neurons):
    from ...neurons.alif import ALIF
    from .reference import neuron_module

    return neuron_module(ALIF, spec, neurons, pre_spike_v=True)


register_model_template(
    ModelTemplate(
        "alif",
        ("v", "g_k"),
        (
            "v_threshold",
            "v_reset",
            "c_m",
            "g_leak",
            "E_leak",
            "E_k",
            "tau_adapt",
            "dg_k",
        ),
        ("v", "g_k", "v_charged"),
        history_sources=(("v_charged", "v"),),
        fragments=fragments,
        default_precise_division=True,
        module_factory=module_factory,
        record_attributes=(("v_charged", "v_pre_spike"),),
        spike_gradient_input="norm",
        supports_refractory=True,
    )
)
