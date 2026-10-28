"""Model-local forward and reverse scalar definitions."""

from ..templates import ModelTemplate, register_model_template
from .integration import _lines, empty_stage as after_fire, empty_stage as reverse_reset
from .neuron import NeuronMath, neuron_fragments


def charge(spec, modes, prepared_values, authoritative_output):
    dt = "dt"
    f = []
    f += [f"v_charged = v_prev + {dt} * x / c_m"]
    return f


def reverse_charge(spec, modes, prepared_values, authoritative_output):
    dt = "dt"
    b = []
    b += _lines(f"""
                grad_x = gcharged * {dt} / c_m
                grad_p_c_m = -gcharged * {dt} * x / (c_m * c_m)
            """)
    b += ["grad_s_v = gcharged"]
    return b


MATH = NeuronMath(charge, reverse_charge, after_fire, reverse_reset)


def fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
    return neuron_fragments(spec, modes, prepared_values, authoritative_output, MATH)


def module_factory(spec, states, neurons):
    from ...neurons.lif import IF
    from .reference import neuron_module

    return neuron_module(IF, spec, neurons, pre_spike_v=True)


register_model_template(
    ModelTemplate(
        "if",
        ("v",),
        ("v_threshold", "v_reset", "c_m"),
        ("v", "v_charged"),
        history_sources=(("v_charged", "v"),),
        fragments=fragments,
        module_factory=module_factory,
        record_attributes=(("v_charged", "v_pre_spike"),),
        spike_gradient_input="norm",
        supports_refractory=True,
    )
)
