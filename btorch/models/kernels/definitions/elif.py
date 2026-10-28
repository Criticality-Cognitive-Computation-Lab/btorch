"""Model-local forward and reverse scalar definitions."""

from functools import partial

from ..templates import ModelTemplate, register_model_template
from .conductance import (
    after_fire,
    charge as conductance_charge,
    reverse_charge as conductance_reverse_charge,
    reverse_reset,
)
from .integration import _lines
from .neuron import NeuronMath, neuron_fragments


charge = partial(
    conductance_charge,
    membrane=_lines("""
                    exp_arg = (v_prev - v_T) / delta_T
                    exp_value = exp(exp_arg)
                    exp_scale = g_leak * delta_T
                    exp_term = exp_scale * exp_value
                    membrane_sum = membrane_sum + exp_term
                    exp_linear_term = exp_term / delta_T
                    v_linear_numerator = -g_leak - g_k_prev + exp_linear_term
                """),
)
reverse_charge = partial(
    conductance_reverse_charge,
    voltage=_lines("""
                    grad_exp_term_linear = grad_v_linear_numerator / delta_T
                    grad_exp_term = grad_v_numerator + grad_exp_term_linear
                    exp_linear_over_delta = exp_linear_term / delta_T
                    neg_grad_v_linear = -grad_v_linear_numerator
                    grad_delta_T_linear = neg_grad_v_linear * exp_linear_over_delta
                    grad_p_delta_T = grad_delta_T_linear
                    grad_exp_scale = grad_exp_term * exp_value
                    grad_exp_value = grad_exp_term * exp_scale
                    grad_p_g_leak = grad_p_g_leak + grad_exp_scale * delta_T
                    grad_p_delta_T = grad_p_delta_T + grad_exp_scale * g_leak
                    grad_exp_arg = grad_exp_value * exp_value
                    grad_exp_difference = grad_exp_arg / delta_T
                    grad_v_dynamics = -grad_v_leak - grad_v_potassium
                    grad_v_dynamics = grad_v_dynamics + grad_exp_difference
                    grad_s_v = gcharged + grad_v_dynamics
                    grad_p_v_T = -grad_exp_difference
                    exp_arg_over_delta_T = exp_arg / delta_T
                    neg_grad_exp_arg = -grad_exp_arg
                    grad_delta_T_exp_arg = neg_grad_exp_arg * exp_arg_over_delta_T
                    grad_p_delta_T = grad_p_delta_T + grad_delta_T_exp_arg
                """),
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
    from ...neurons.alif import ELIF
    from .reference import neuron_module

    return neuron_module(ELIF, spec, neurons, pre_spike_v=True)


register_model_template(
    ModelTemplate(
        "elif",
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
            "delta_T",
            "v_T",
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
