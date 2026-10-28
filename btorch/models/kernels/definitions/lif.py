"""Model-local forward and reverse scalar definitions."""

from ..templates import ModelTemplate, register_model_template
from .integration import _lines, empty_stage as after_fire, empty_stage as reverse_reset
from .neuron import NeuronMath, neuron_fragments


def charge(spec, modes, prepared_values, authoritative_output):
    dt = "dt"
    f = []
    f += [f"v_charged = v_prev + {dt} * (-(v_prev - v_reset) / tau + x / c_m)"]
    return f


def reverse_charge(spec, modes, prepared_values, authoritative_output):
    dt = "dt"
    b = []
    b += _lines(f"""
                grad_x = gcharged * {dt} / c_m
                grad_p_c_m = -gcharged * {dt} * x / (c_m * c_m)
            """)
    b += _lines(f"""
                    grad_p_tau = gcharged * {dt} * (v_prev - v_reset) / (tau * tau)
                    grad_p_v_reset = grad_p_v_reset + gcharged * {dt} / tau
                    grad_s_v = gcharged * (1.0 - {dt} / tau)
                """)
    return b


MATH = NeuronMath(charge, reverse_charge, after_fire, reverse_reset)


def fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
    return neuron_fragments(spec, modes, prepared_values, authoritative_output, MATH)


def module_factory(spec, states, neurons):
    from ...neurons.lif import LIF
    from .reference import neuron_module

    return neuron_module(LIF, spec, neurons, pre_spike_v=True)


register_model_template(
    ModelTemplate(
        "lif",
        ("v",),
        ("v_threshold", "v_reset", "c_m", "tau"),
        ("v", "v_charged"),
        history_sources=(("v_charged", "v"),),
        fragments=fragments,
        module_factory=module_factory,
        record_attributes=(("v_charged", "v_pre_spike"),),
        recurrent_checkpoint=True,
        spike_gradient_input="norm",
        supports_refractory=True,
    )
)
