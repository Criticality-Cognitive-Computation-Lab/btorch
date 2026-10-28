"""Model-local forward and reverse scalar definitions."""

from ..templates import ModelTemplate, register_model_template
from .integration import _lines
from .neuron import NeuronMath, neuron_fragments


def charge(spec, modes, prepared_values, authoritative_output):
    dt = "dt"
    f = []
    f += _lines(f"""
                quadratic = k * (v_prev - v_rest) * (v_prev - v_threshold)
                v_charged = v_prev + {dt} * (x + quadratic - u_prev) / c_m
                u_charged = u_prev + {dt} * a * (b * (v_charged - v_rest) - u_prev)
            """)
    return f


def reverse_charge(spec, modes, prepared_values, authoritative_output):
    dt = "dt"
    b = []
    b += _lines(f"""
                gu = grad_s_u + grad_r_u_charged
                delta = b * (v_charged - v_rest) - u_prev
                grad_p_a = gu * {dt} * delta
                grad_p_b = gu * {dt} * a * (v_charged - v_rest)
                grad_p_v_rest = -gu * {dt} * a * b
                gcharged = gcharged + gu * {dt} * a * b
                gu_prev = gu * (1.0 - {dt} * a)
                current = x + k * (v_prev - v_rest) * (v_prev - v_threshold) - u_prev
                grad_x = gcharged * {dt} / c_m
                grad_p_c_m = -gcharged * {dt} * current / (c_m * c_m)
                grad_p_k = grad_x * (v_prev - v_rest) * (v_prev - v_threshold)
                grad_p_v_rest = grad_p_v_rest - grad_x * k * (v_prev - v_threshold)
                grad_p_v_threshold = grad_p_v_threshold - grad_x * k * (v_prev - v_rest)
                grad_s_v = gcharged + grad_x * k * (2.0 * v_prev - v_rest - v_threshold)
                grad_s_u = gu_prev - grad_x
            """)
    return b


def after_fire(spec, modes, prepared_values, authoritative_output):
    f = []
    f += ["u_next = u_charged + d * output"]
    return f


def reverse_reset(spec, modes, prepared_values, authoritative_output):
    b = []
    b += ["grad_p_d = grad_s_u * output"]
    if not spec.detach_reset:
        b += ["gspike = gspike + grad_s_u * d"]
    return b


MATH = NeuronMath(charge, reverse_charge, after_fire, reverse_reset, cutoff="v_peak")


def fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
    return neuron_fragments(spec, modes, prepared_values, authoritative_output, MATH)


def module_factory(spec, states, neurons):
    from ...neurons.izhikevich import Izhikevich
    from .reference import neuron_module

    return neuron_module(Izhikevich, spec, neurons, pre_spike=True)


register_model_template(
    ModelTemplate(
        "izhikevich",
        ("v", "u"),
        ("v_threshold", "v_reset", "c_m", "k", "v_rest", "v_peak", "a", "b", "d"),
        ("v", "u", "v_charged", "u_charged"),
        history_sources=(("v_charged", "v"), ("u_charged", "u")),
        fragments=fragments,
        module_factory=module_factory,
        record_attributes=(("v_charged", "v_pre_spike"), ("u_charged", "u_pre_spike")),
        spike_gradient_input="norm",
    )
)
