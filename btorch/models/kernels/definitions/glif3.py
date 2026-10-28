"""Model-local forward and reverse scalar definitions."""

from ...ode import _EXP_EULER_SERIES_CUTOFF
from ..templates import ModelTemplate, PreparedParameter, register_model_template
from .integration import _exp_euler_backward, _exp_euler_forward, _lines
from .neuron import NeuronMath, neuron_fragments


def charge(spec, modes, prepared_values, authoritative_output):
    dt = "dt"
    glif_small = f"abs(glif_z) <= {_EXP_EULER_SERIES_CUTOFF!r}"
    f = []
    f += [
        "drive = x"
        + (
            " + (" + " + ".join((f"Iasc_{i}_prev" for i in range(modes))) + ")"
            if modes
            else ""
        ),
        "glif_derivative = -(v_prev - v_rest) / tau + drive / c_m",
        "glif_linear = -1.0 / tau",
        f"glif_z = {dt} * glif_linear",
        f"glif_series_z = where({glif_small}, glif_z, 0.0)",
        f"glif_safe_linear = where({glif_small}, 1.0, glif_linear)",
    ]
    if "glif_phi" not in prepared_values:
        f += _exp_euler_forward("glif", "glif_linear", dt)
    f += ["v_charged = v_prev + glif_phi * glif_derivative"]
    for i in range(modes):
        f += [
            f"adapt_derivative_{i} = -k_{i} * Iasc_{i}_prev",
            f"adapt_linear_{i} = -k_{i}",
            *_exp_euler_forward(f"adapt_{i}", f"adapt_linear_{i}", dt),
            f"Iasc_{i}_next = Iasc_{i}_prev + adapt_{i}_phi * adapt_derivative_{i}",
        ]
    return f


def reverse_charge(spec, modes, prepared_values, authoritative_output):
    dt = "dt"
    glif_small = f"abs(glif_z) <= {_EXP_EULER_SERIES_CUTOFF!r}"
    b = []
    b += _lines(f"""
                grad_glif_derivative = gcharged * glif_phi
                grad_glif_phi = gcharged * glif_derivative
                grad_glif_phi_active = where({glif_small}, 0.0, grad_glif_phi)
                grad_glif_delta = grad_glif_phi_active / glif_safe_linear
                glif_safe_square = glif_safe_linear * glif_safe_linear
                grad_glif_safe = -grad_glif_phi_active * glif_delta / glif_safe_square
                grad_glif_exp = grad_glif_delta * {dt} * (glif_delta + 1.0)
                grad_glif_linear_direct = grad_glif_exp + grad_glif_safe
                glif_z = {dt} * glif_linear
                glif_dexprel_tail = 1.0 / 144.0 + glif_series_z / 840.0
                glif_dexprel_inner = 1.0 / 30.0 + glif_series_z * glif_dexprel_tail
                glif_dexprel_mid = 1.0 / 8.0 + glif_series_z * glif_dexprel_inner
                glif_dexprel_head = 1.0 / 3.0 + glif_series_z * glif_dexprel_mid
                glif_dexprel = 0.5 + glif_series_z * glif_dexprel_head
                grad_glif_linear_series = grad_glif_phi * {dt} * {dt} * glif_dexprel
                grad_glif_linear_phi = where({glif_small}, \
                    grad_glif_linear_series, grad_glif_linear_direct)
                grad_x = grad_glif_derivative / c_m
                tau_square = tau * tau
                grad_tau_direct = grad_glif_derivative * (v_prev - v_rest) / tau_square
                grad_tau_linear = grad_glif_linear_phi / tau_square
                grad_p_tau = grad_tau_direct + grad_tau_linear
                grad_p_c_m = -grad_glif_derivative * drive / (c_m * c_m)
                grad_p_v_rest = grad_glif_derivative / tau
                grad_s_v = gcharged - grad_glif_derivative / tau
            """)
    for i in range(modes):
        b += [
            f"grad_adapt_derivative_{i} = grad_s_Iasc_{i} * adapt_{i}_phi",
            f"grad_adapt_{i}_phi = grad_s_Iasc_{i} * adapt_derivative_{i}",
            *_exp_euler_backward(f"adapt_{i}", f"adapt_linear_{i}", dt),
            f"grad_adapt_linear_phi_{i} = grad_adapt_{i}_linear",
            f"grad_adapt_k_direct_{i} = -grad_adapt_derivative_{i} * Iasc_{i}_prev",
            f"grad_p_k_{i} = grad_adapt_k_direct_{i} - grad_adapt_linear_phi_{i}",
            f"grad_s_Iasc_{i} = grad_s_Iasc_{i} "
            f"+ grad_adapt_derivative_{i} * adapt_linear_{i} + grad_x",
        ]
    return b


def after_fire(spec, modes, prepared_values, authoritative_output):
    f = []
    f += [
        f"Iasc_{i}_next = Iasc_{i}_next + asc_amps_{i} * output" for i in range(modes)
    ]
    return f


def reverse_reset(spec, modes, prepared_values, authoritative_output):
    reset_gap = "v_charged - v_reset" if spec.hard_reset else "v_threshold - v_reset"
    b = []
    b += [f"grad_p_asc_amps_{i} = grad_s_Iasc_{i} * output" for i in range(modes)]
    if not spec.detach_reset:
        if modes:
            b += ["gspike_adapt = grad_s_Iasc_0 * asc_amps_0"]
            b += [
                f"gspike_adapt = gspike_adapt + grad_s_Iasc_{i} * asc_amps_{i}"
                for i in range(1, modes)
            ]
            b += ["gspike = gspike + gspike_adapt"]
        b += [
            f"gspike_reset = (-grad_s_v) * ({reset_gap})",
            "gspike = gspike + gspike_reset",
        ]
    return b


MATH = NeuronMath(
    charge, reverse_charge, after_fire, reverse_reset, reset_after_adaptation=True
)


def fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
    return neuron_fragments(spec, modes, prepared_values, authoritative_output, MATH)


def prepare(spec, parameters, *, backward):
    from ...ode import _exp_euler_coefficient

    phi, delta = _exp_euler_coefficient(-1.0 / parameters["tau"], spec.dt)
    return {
        "glif_phi_coefficient": phi,
        **({"glif_delta_coefficient": delta} if backward else {}),
    }


def module_factory(spec, states, neurons):
    from ...neurons.glif import GLIF3
    from .reference import neuron_module

    return neuron_module(
        GLIF3,
        spec,
        neurons,
        pre_spike_v=True,
        k=[0.2] * states[1].shape[-1],
        asc_amps=[0.0] * states[1].shape[-1],
    )


def replay_state(name, component, output, condition):
    """Reconstruct adaptation with the same ordered exponential-Euler graph."""
    return (
        f"replay_L_{component} = -k_{component}",
        f"replay_D_{component} = replay_L_{component} * {name}_prev",
        *_exp_euler_forward(f"replay_{component}", f"replay_L_{component}", "dt"),
        f"{name}_prev = where({condition}, {name}_prev + "
        f"replay_{component}_phi * replay_D_{component}, {name}_prev)",
        f"{name}_prev = where({condition}, {name}_prev + "
        f"asc_amps_{component} * {output}, {name}_prev)",
    )


register_model_template(
    ModelTemplate(
        "glif3",
        ("v", "Iasc"),
        ("v_threshold", "v_reset", "c_m", "tau", "v_rest", "k", "asc_amps"),
        ("v", "Iasc", "v_charged"),
        history_sources=(("v_charged", "v"),),
        fragments=fragments,
        module_factory=module_factory,
        record_attributes=(("v_charged", "v_pre_spike"),),
        parameter_attributes=(("v_rest", "_v_rest"),),
        state_extra_dims=(("Iasc", 1),),
        parameter_extra_dims=(("k", 1), ("asc_amps", 1)),
        prepared_parameters=(
            PreparedParameter("glif_phi_coefficient", "glif_phi", "tau"),
            PreparedParameter("glif_delta_coefficient", "glif_delta", "tau"),
        ),
        prepare_parameters=prepare,
        checkpoint_state="Iasc",
        checkpoint_interval=16,
        replay_state=replay_state,
        spike_gradient_input="norm",
        supports_refractory=True,
    )
)
