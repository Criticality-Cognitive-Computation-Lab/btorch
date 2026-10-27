"""Declare scalar forward and hand-derived reverse stages for cell DSLs.

These statements are direct templates, not an expression IR. All three
GPU backends substitute only their elementary math functions and storage
syntax. GLIF adaptation modes are explicitly unrolled from the static
state shape.
"""

import math

from ...ode import _EXP_EULER_SERIES_CUTOFF
from ...surrogate.sigmoid import _SIG_K
from ...surrogate.superspike import _SS_K
from ..contracts import NeuronSpec
from ..templates import _MODELS, BackendFragments, get_template


def scalar_layout(spec: NeuronSpec, modes: int = 0):
    """Expand the small GLIF mode axis into named scalar registers."""
    template = get_template(spec)
    states = tuple(
        n
        for name in template.state_names
        for n in (
            tuple(f"Iasc_{i}" for i in range(modes)) if name == "Iasc" else (name,)
        )
    )
    parameters = tuple(
        n
        for name in template.parameter_names
        for n in (
            tuple(f"{name}_{i}" for i in range(modes))
            if name in ("k", "asc_amps") and spec.model == "glif3"
            else (name,)
        )
    )
    return states, parameters


def _lines(value: str):
    return tuple(line.strip() for line in value.strip().splitlines() if line.strip())


def _exp_euler_forward(prefix: str, linear: str, dt: str) -> tuple[str, ...]:
    """Emit the Module integrator's stable exponential-Euler coefficient."""
    return (
        f"{prefix}_z = {dt} * {linear}",
        (
            f"{prefix}_safe_linear = where(abs({prefix}_z) <= "
            f"{_EXP_EULER_SERIES_CUTOFF!r}, 1.0, {linear})"
        ),
        f"{prefix}_series_z = where(abs({prefix}_z) <= "
        f"{_EXP_EULER_SERIES_CUTOFF!r}, {prefix}_z, 0.0)",
        f"{prefix}_delta = expm1({prefix}_z)",
        f"{prefix}_ratio = {prefix}_delta / {prefix}_safe_linear",
        (
            f"{prefix}_exprel = 1.0 + {prefix}_series_z * (0.5 + "
            f"{prefix}_series_z * (1.0 / 6.0 + "
            f"{prefix}_series_z * (1.0 / 24.0 + {prefix}_series_z * (1.0 "
            f"/ 120.0 + {prefix}_series_z * "
            f"(1.0 / 720.0 + {prefix}_series_z / 5040.0)))))"
        ),
        (
            f"{prefix}_phi = where(abs({prefix}_z) <= {_EXP_EULER_SERIES_CUTOFF!r}, "
            f"{dt} * {prefix}_exprel, {prefix}_ratio)"
        ),
    )


def _exp_euler_backward(prefix: str, linear: str, dt: str) -> tuple[str, ...]:
    """Emit a stable coefficient VJP while preserving the far-field graph."""
    return (
        (
            f"grad_{prefix}_ratio = where(abs({prefix}_z) <= "
            f"{_EXP_EULER_SERIES_CUTOFF!r}, 0.0, grad_{prefix}_phi)"
        ),
        f"grad_{prefix}_delta = grad_{prefix}_ratio / {prefix}_safe_linear",
        f"grad_{prefix}_ratio_again = {prefix}_delta / {prefix}_safe_linear",
        (
            f"grad_{prefix}_ratio_square = grad_{prefix}_ratio_again / "
            f"{prefix}_safe_linear"
        ),
        (f"grad_{prefix}_safe = -grad_{prefix}_ratio * grad_{prefix}_ratio_square"),
        (
            f"grad_{prefix}_linear_safe = where({linear} == 0.0, 0.0, "
            f"grad_{prefix}_safe)"
        ),
        (
            f"grad_{prefix}_linear_delta = (grad_{prefix}_delta * ({prefix}_delta + "
            f"1.0)) * {dt}"
        ),
        (
            f"grad_{prefix}_linear_direct = grad_{prefix}_linear_delta + "
            f"grad_{prefix}_linear_safe"
        ),
        (
            f"{prefix}_dexprel = 0.5 + {prefix}_series_z * (1.0 / 3.0 + "
            f"{prefix}_series_z * (1.0 "
            f"/ 8.0 + {prefix}_series_z * (1.0 / 30.0 + {prefix}_series_z "
            f"* (1.0 / 144.0 + "
            f"{prefix}_series_z / 840.0))))"
        ),
        (
            f"grad_{prefix}_linear_series = grad_{prefix}_phi * {dt} * {dt} * "
            f"{prefix}_dexprel"
        ),
        (
            f"grad_{prefix}_linear = where(abs({prefix}_z) <= "
            f"{_EXP_EULER_SERIES_CUTOFF!r}, grad_{prefix}_linear_series, "
            f"grad_{prefix}_linear_direct)"
        ),
    )


def get_fragments(
    spec: NeuronSpec,
    modes: int = 0,
    *,
    prepared_values: tuple[str, ...] = (),
    authoritative_output: bool = False,
) -> BackendFragments:
    """Compose fixed charge, adapt, fire, reset and reverse stages."""
    states, parameters = scalar_layout(spec, modes)
    model = spec.model
    if model not in _MODELS:
        raise NotImplementedError(f"No built-in GPU fragments for {model!r}")
    dt = "dt"
    glif_small = f"abs(glif_z) <= {_EXP_EULER_SERIES_CUTOFF!r}"
    f = []
    b = [f"grad_p_{name} = 0.0" for name in parameters]
    if model == "exponential":
        # Follow ExponentialPSC's exponential-Euler graph. Collapsing this to
        # exp(-dt / tau) * psc changes FP32 states and long recurrent VJPs.
        f += _lines(f"""
            psc_negative = -psc_prev
            psc_derivative = psc_negative / tau_syn
            psc_reciprocal = 1.0 / tau_syn
            psc_linear = psc_reciprocal * -1.0
            {chr(10).join(_exp_euler_forward("psc", "psc_linear", dt))}
            psc_decayed = psc_prev + psc_phi * psc_derivative
            psc_next = psc_decayed + x
            output = psc_next
        """)
        b += _lines(f"""
            grad_s_psc = grad_s_psc + grad_output
            grad_x = grad_s_psc
            grad_psc_derivative = grad_s_psc * psc_phi
            grad_psc_phi = grad_s_psc * psc_derivative
            {chr(10).join(_exp_euler_backward("psc", "psc_linear", dt))}
            grad_psc_reciprocal = grad_psc_linear * -1.0
            grad_tau_linear = (-grad_psc_reciprocal) * (psc_reciprocal * psc_reciprocal)
            grad_tau_derivative = (-grad_psc_derivative) * (psc_derivative / tau_syn)
            grad_p_tau_syn = grad_tau_linear + grad_tau_derivative
            grad_s_psc = grad_s_psc + (-grad_psc_derivative / tau_syn)
        """)
    elif model == "alpha":
        f += _lines(f"""
            h_derivative = -h_prev / tau_syn
            alpha_linear = -1.0 / tau_syn
            {chr(10).join(_exp_euler_forward("alpha", "alpha_linear", dt))}
            h_decayed = h_prev + alpha_phi * h_derivative
            h_next = h_decayed + g_max * x
            psc_derivative = -psc_prev / tau_syn + h_next / tau_syn
            psc_next = psc_prev + alpha_phi * psc_derivative
            output = psc_next
        """)
        b += _lines(f"""
            grad_s_psc = grad_s_psc + grad_output
            grad_psc_derivative = grad_s_psc * alpha_phi
            grad_alpha_phi_psc = grad_s_psc * psc_derivative
            tau_syn_square = tau_syn * tau_syn
            grad_tau_psc_decay = grad_psc_derivative * psc_prev / tau_syn_square
            grad_tau_psc_drive = -grad_psc_derivative * h_next / tau_syn_square
            grad_h_next = grad_s_h + grad_psc_derivative / tau_syn
            grad_x = grad_h_next * g_max
            grad_p_g_max = grad_h_next * x
            grad_h_derivative = grad_h_next * alpha_phi
            grad_alpha_phi_h = grad_h_next * h_derivative
            grad_alpha_phi = grad_alpha_phi_psc + grad_alpha_phi_h
            grad_alpha_phi_active = where(alpha_linear == 0.0, 0.0, grad_alpha_phi)
            {chr(10).join(_exp_euler_backward("alpha", "alpha_linear", dt))}
            grad_tau_h_decay = grad_h_derivative * h_prev / tau_syn_square
            grad_tau_linear = grad_alpha_linear / tau_syn_square
            grad_tau_psc = grad_tau_psc_decay + grad_tau_psc_drive
            grad_tau_h = grad_tau_h_decay + grad_tau_linear
            grad_p_tau_syn = grad_tau_psc + grad_tau_h
            grad_s_h = grad_h_next - grad_h_derivative / tau_syn
            grad_s_psc = grad_s_psc - grad_psc_derivative / tau_syn
        """)
    elif model == "alpha_billeh":
        f += _lines(f"""
            h_next = syn_decay * h_prev + {math.e} / tau_syn * x
            psc_next = syn_decay * (psc_prev + h_next)
            output = psc_next
        """)
        b += _lines(f"""
            grad_s_psc = grad_s_psc + grad_output
            gh = grad_s_h + grad_s_psc * syn_decay
            grad_p_syn_decay = grad_s_psc * (psc_prev + h_next) + gh * h_prev
            grad_p_tau_syn = -gh * {math.e} * x / (tau_syn * tau_syn)
            grad_x = gh * {math.e} / tau_syn
            grad_s_h = gh * syn_decay
            grad_s_psc = grad_s_psc * syn_decay
        """)
    elif model == "dual_exponential":
        f += _lines(f"""
            rise_negative = -g_rise_prev
            rise_derivative = rise_negative / tau_rise
            rise_reciprocal = 1.0 / tau_rise
            rise_linear = rise_reciprocal * -1.0
            {chr(10).join(_exp_euler_forward("rise", "rise_linear", dt))}
            g_rise_decayed = g_rise_prev + rise_phi * rise_derivative
            decay_negative = -g_decay_prev
            decay_derivative = decay_negative / tau_decay
            decay_reciprocal = 1.0 / tau_decay
            decay_linear = decay_reciprocal * -1.0
            {chr(10).join(_exp_euler_forward("decay", "decay_linear", dt))}
            g_decay_decayed = g_decay_prev + decay_phi * decay_derivative
            rise = exp(-{dt} / tau_rise)
            decay = exp(-{dt} / tau_decay)
            g_rise_next = g_rise_decayed + x
            g_decay_next = g_decay_decayed + x
            psc_next = a * (decay * g_decay_next - rise * g_rise_next)
            output = psc_next
        """)
        b += _lines(f"""
            grad_s_psc = grad_s_psc + grad_output
            grad_g_rise_next = grad_s_g_rise - grad_s_psc * a * rise
            grad_g_decay_next = grad_s_g_decay + grad_s_psc * a * decay
            grad_p_a = grad_s_psc * (decay * g_decay_next - rise * g_rise_next)
            grad_rise = -grad_s_psc * a * g_rise_next
            grad_tau_rise_readout = grad_rise * rise * {dt} / (tau_rise * tau_rise)
            grad_decay = grad_s_psc * a * g_decay_next
            grad_tau_decay_readout = grad_decay * decay * {dt} / (tau_decay * tau_decay)
            grad_x = grad_g_rise_next + grad_g_decay_next
            grad_rise_derivative = grad_g_rise_next * rise_phi
            grad_rise_phi = grad_g_rise_next * rise_derivative
            {chr(10).join(_exp_euler_backward("rise", "rise_linear", dt))}
            grad_rise_reciprocal = grad_rise_linear * -1.0
            rise_reciprocal_square = rise_reciprocal * rise_reciprocal
            grad_tau_rise_linear = (-grad_rise_reciprocal) * rise_reciprocal_square
            rise_derivative_over_tau = rise_derivative / tau_rise
            negative_grad_rise = -grad_rise_derivative
            grad_tau_rise_derivative = negative_grad_rise * rise_derivative_over_tau
            grad_tau_rise_state = grad_tau_rise_linear + grad_tau_rise_derivative
            grad_p_tau_rise = grad_tau_rise_readout + grad_tau_rise_state
            grad_s_g_rise = grad_g_rise_next + (-grad_rise_derivative / tau_rise)
            grad_decay_derivative = grad_g_decay_next * decay_phi
            grad_decay_phi = grad_g_decay_next * decay_derivative
            {chr(10).join(_exp_euler_backward("decay", "decay_linear", dt))}
            grad_decay_reciprocal = grad_decay_linear * -1.0
            decay_reciprocal_square = decay_reciprocal * decay_reciprocal
            grad_tau_decay_linear = (-grad_decay_reciprocal) * decay_reciprocal_square
            decay_derivative_over_tau = decay_derivative / tau_decay
            negative_grad_decay = -grad_decay_derivative
            grad_tau_decay_derivative = negative_grad_decay * decay_derivative_over_tau
            grad_tau_decay_state = grad_tau_decay_linear + grad_tau_decay_derivative
            grad_p_tau_decay = grad_tau_decay_readout + grad_tau_decay_state
            grad_s_g_decay = grad_g_decay_next + (-grad_decay_derivative / tau_decay)
            grad_s_psc = 0.0
        """)
    else:
        if model == "if":
            f += [f"v_charged = v_prev + {dt} * x / c_m"]
        elif model == "lif":
            f += [f"v_charged = v_prev + {dt} * (-(v_prev - v_reset) / tau + x / c_m)"]
        elif model == "glif3":
            f += [
                "drive = x"
                + (
                    " + (" + " + ".join(f"Iasc_{i}_prev" for i in range(modes)) + ")"
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
                    f"Iasc_{i}_next = Iasc_{i}_prev + "
                    f"adapt_{i}_phi * adapt_derivative_{i}",
                ]
        elif model in ("alif", "elif"):
            f += _lines("""
                leak_gap = v_prev - E_leak
                leak_term = -g_leak * leak_gap
                potassium_gap = v_prev - E_k
                potassium_term = -g_k_prev * potassium_gap
                membrane_sum = leak_term + potassium_term
            """)
            if model == "elif":
                f += _lines("""
                    exp_arg = (v_prev - v_T) / delta_T
                    exp_value = exp(exp_arg)
                    exp_scale = g_leak * delta_T
                    exp_term = exp_scale * exp_value
                    membrane_sum = membrane_sum + exp_term
                    exp_linear_term = exp_term / delta_T
                    v_linear_numerator = -g_leak - g_k_prev + exp_linear_term
                """)
            else:
                f += ["v_linear_numerator = -g_leak - g_k_prev"]
            f += _lines(f"""
                v_numerator = membrane_sum + x
                v_derivative = v_numerator / c_m
                v_linear = v_linear_numerator / c_m
                {chr(10).join(_exp_euler_forward("v", "v_linear", dt))}
                v_charged = v_prev + v_phi * v_derivative
                g_k_negative = -g_k_prev
                g_k_derivative = g_k_negative / tau_adapt
                g_k_reciprocal = 1.0 / tau_adapt
                g_k_linear = g_k_reciprocal * -1.0
                {chr(10).join(_exp_euler_forward("g_k", "g_k_linear", dt))}
                g_k_decayed = g_k_prev + g_k_phi * g_k_derivative
            """)
        else:
            f += _lines(f"""
                quadratic = k * (v_prev - v_rest) * (v_prev - v_threshold)
                v_charged = v_prev + {dt} * (x + quadratic - u_prev) / c_m
                u_charged = u_prev + {dt} * a * (b * (v_charged - v_rest) - u_prev)
            """)
        cutoff = "v_peak" if model == "izhikevich" else "v_threshold"
        f += [
            "gap = v_threshold - v_reset",
            f"norm = (v_charged - {cutoff}) / gap",
        ]
        if spec.surrogate == "poisson":
            if not authoritative_output:
                if spec.poisson.spiking:
                    f += [
                        f"rate = exp({spec.poisson.rho!r} * norm) / "
                        f"{spec.poisson.tau!r}",
                        "output = where(rate > uniform, 1.0, 0.0)",
                    ]
                else:
                    f += [
                        "output = where(norm >= 0.0, "
                        f"{spec.poisson.k!r}, {spec.poisson.leak!r}) * norm"
                    ]
        else:
            f += ["output = where(norm >= 0.0, 1.0, 0.0)"]
        if spec.refractory:
            f += ["output = output * where(refractory_prev == 0.0, 1.0, 0.0)"]
        reset_gap = "v_charged - v_reset" if spec.hard_reset else f"{cutoff} - v_reset"
        f += [f"v_next = v_charged - ({reset_gap}) * output"]
        if model == "glif3":
            f += [
                f"Iasc_{i}_next = Iasc_{i}_next + asc_amps_{i} * output"
                for i in range(modes)
            ]
        if model == "izhikevich":
            f += ["u_next = u_charged + d * output"]
        if model in ("alif", "elif"):
            f += ["g_k_next = g_k_decayed + dg_k * output"]
        if spec.refractory:
            f += [
                f"ref_charged = refractory_prev + output * tau_ref - {dt}",
                "refractory_next = maximum(ref_charged, 0.0)",
            ]
        b += [
            "gcharged = grad_s_v * (1.0 - output)"
            if spec.hard_reset
            else "gcharged = grad_s_v",
            "gcharged = gcharged + grad_r_v_charged",
            "grad_p_v_reset = grad_s_v * output",
            "gspike = grad_output",
        ]
        if spec.refractory and model in ("alif", "elif"):
            b += [
                "ref_gate = where(refractory_next > 0.0, 1.0, 0.0)",
                "grad_s_refractory = grad_s_refractory * ref_gate",
                "grad_p_tau_ref = grad_s_refractory * output",
            ]
            if not spec.detach_reset:
                b += ["gspike = gspike + grad_s_refractory * tau_ref"]
        if not spec.detach_reset and model not in ("glif3", "alif", "elif"):
            b += [f"gspike = gspike - grad_s_v * ({reset_gap})"]
        if not spec.hard_reset:
            b += [f"grad_p_{cutoff} = -grad_s_v * output"]
        if model in ("alif", "elif"):
            b += ["grad_p_dg_k = grad_s_g_k * output"]
            if not spec.detach_reset:
                b += [
                    "gspike = gspike + grad_s_g_k * dg_k",
                    f"gspike_reset = (-grad_s_v) * ({reset_gap})",
                    "gspike = gspike + gspike_reset",
                ]
        if model == "glif3":
            b += [
                f"grad_p_asc_amps_{i} = grad_s_Iasc_{i} * output" for i in range(modes)
            ]
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
        if model == "izhikevich":
            b += ["grad_p_d = grad_s_u * output"]
            if not spec.detach_reset:
                b += ["gspike = gspike + grad_s_u * d"]
        if spec.refractory:
            if model not in ("alif", "elif"):
                b += [
                    "ref_gate = where(refractory_next > 0.0, 1.0, 0.0)",
                    "grad_s_refractory = grad_s_refractory * ref_gate",
                    "grad_p_tau_ref = grad_s_refractory * output",
                ]
                if not spec.detach_reset:
                    b += ["gspike = gspike + grad_s_refractory * tau_ref"]
            b += ["gspike = gspike * where(refractory_prev == 0.0, 1.0, 0.0)"]
        if spec.surrogate == "poisson":
            b += [
                "surrogate_grad = where(norm >= 0.0, "
                f"{spec.poisson.k!r}, {spec.poisson.leak!r})"
            ]
        elif spec.surrogate != "sigmoid" or model not in ("alif", "elif"):
            b += [f"z = {spec.alpha!r} * norm"]
        if spec.surrogate == "poisson":
            pass
        elif spec.surrogate == "sigmoid":
            if model in ("alif", "elif"):
                sigmoid_scale = _SIG_K * spec.alpha
                sigmoid_amplitude = spec.damping * 4.0
                b += [
                    f"sigmoid_arg = norm * {sigmoid_scale!r}",
                    "sigma = 1.0 / (1.0 + exp(-sigmoid_arg))",
                    f"sigmoid_grad = sigma * {sigmoid_amplitude!r}",
                    "sigmoid_complement = 1.0 - sigma",
                    "surrogate_grad = sigmoid_grad * sigmoid_complement",
                ]
            else:
                b += [
                    f"sigma = 1.0 / (1.0 + exp(-{_SIG_K} * z))",
                    "surrogate_grad = 4.0 * sigma * (1.0 - sigma)",
                ]
        elif spec.surrogate == "atan":
            b += [
                "atan_square = z * z",
                "atan_denominator = 1.0 + atan_square",
                "surrogate_grad = 1.0 / atan_denominator",
                f"atan_damped_grad = surrogate_grad * {spec.damping!r}",
            ]
        elif spec.surrogate == "atan_approx":
            b += [
                "denom = 1.0 + 0.28 * z * z",
                "surrogate_grad = (1.0 - 0.28 * z * z) / (denom * denom)",
            ]
        elif spec.surrogate == "erf":
            b += [f"surrogate_grad = exp(-{math.log(2)} * z * z)"]
        elif spec.surrogate == "superspike":
            b += [
                f"denom = 1.0 + {_SS_K} * abs(z)",
                "surrogate_grad = 1.0 / (denom * denom)",
            ]
        else:
            b += ["surrogate_grad = maximum(1.0 - 0.5 * abs(z), 0.0)"]
        if spec.surrogate == "poisson":
            # Match _poisson_grad's operation graph before multiplying by the
            # incoming cotangent; the association is observable in FP32.
            gnorm = f"gspike * (surrogate_grad * {spec.damping!r})"
        elif spec.surrogate == "atan":
            gnorm = "gspike * atan_damped_grad"
        elif spec.surrogate == "sigmoid" and model in ("alif", "elif"):
            gnorm = "gspike * surrogate_grad"
        else:
            gnorm = f"gspike * {spec.damping!r} * surrogate_grad"
        b += [f"gnorm = {gnorm}"]
        if spec.surrogate_width is not None:
            b += [f"gnorm = where(abs(norm) < {spec.surrogate_width!r}, gnorm, 0.0)"]
        if model in ("alif", "elif"):
            b += _lines("""
                grad_norm_numerator = gnorm / gap
                grad_norm_gap = (-gnorm) * (norm / gap)
                gcharged = gcharged + grad_norm_numerator
                grad_p_v_threshold = grad_p_v_threshold + grad_norm_gap
                grad_p_v_reset = grad_p_v_reset - grad_norm_gap
                grad_p_v_threshold = grad_p_v_threshold - grad_norm_numerator
            """)
        else:
            b += _lines(f"""
                gcharged = gcharged + gnorm / gap
                grad_p_v_threshold = grad_p_v_threshold - gnorm * norm / gap
                grad_p_v_reset = grad_p_v_reset + gnorm * norm / gap
                grad_p_{cutoff} = grad_p_{cutoff} - gnorm / gap
            """)
        if model in ("if", "lif"):
            b += _lines(f"""
                grad_x = gcharged * {dt} / c_m
                grad_p_c_m = -gcharged * {dt} * x / (c_m * c_m)
            """)
            if model == "lif":
                b += _lines(f"""
                    grad_p_tau = gcharged * {dt} * (v_prev - v_reset) / (tau * tau)
                    grad_p_v_reset = grad_p_v_reset + gcharged * {dt} / tau
                    grad_s_v = gcharged * (1.0 - {dt} / tau)
                """)
            else:
                b += ["grad_s_v = gcharged"]
        elif model == "glif3":
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
                    f"grad_adapt_k_direct_{i} = -grad_adapt_derivative_{i} "
                    f"* Iasc_{i}_prev",
                    f"grad_p_k_{i} = grad_adapt_k_direct_{i} "
                    f"- grad_adapt_linear_phi_{i}",
                    f"grad_s_Iasc_{i} = grad_s_Iasc_{i} "
                    f"+ grad_adapt_derivative_{i} * adapt_linear_{i} + grad_x",
                ]
        elif model in ("alif", "elif"):
            b += _lines(f"""
                grad_v_derivative = gcharged * v_phi
                grad_v_phi = gcharged * v_derivative
                {chr(10).join(_exp_euler_backward("v", "v_linear", dt))}
                grad_v_numerator = grad_v_derivative / c_m
                grad_v_linear_numerator = grad_v_linear / c_m
                grad_x = grad_v_numerator
                grad_c_m_derivative = -grad_v_derivative * (v_derivative / c_m)
                grad_c_m_linear = -grad_v_linear * (v_linear / c_m)
                grad_p_c_m = grad_c_m_derivative + grad_c_m_linear
                grad_g_leak_derivative = -grad_v_numerator * leak_gap
                grad_p_g_leak = grad_g_leak_derivative - grad_v_linear_numerator
                grad_p_E_leak = grad_v_numerator * g_leak
                grad_g_k_derivative_voltage = -grad_v_numerator * potassium_gap
                grad_g_k_voltage = grad_g_k_derivative_voltage - grad_v_linear_numerator
                grad_p_E_k = grad_v_numerator * g_k_prev
                grad_v_leak = grad_v_numerator * g_leak
                grad_v_potassium = grad_v_numerator * g_k_prev
            """)
            if model == "elif":
                b += _lines("""
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
                """)
            else:
                b += ["grad_s_v = gcharged - grad_v_leak - grad_v_potassium"]
            b += _lines(f"""
                grad_g_k_derivative = grad_s_g_k * g_k_phi
                grad_g_k_phi = grad_s_g_k * g_k_derivative
                {chr(10).join(_exp_euler_backward("g_k", "g_k_linear", dt))}
                grad_g_k_reciprocal = grad_g_k_linear * -1.0
                g_k_reciprocal_square = g_k_reciprocal * g_k_reciprocal
                negative_grad_g_k_reciprocal = -grad_g_k_reciprocal
                grad_tau_linear = negative_grad_g_k_reciprocal * g_k_reciprocal_square
                gk_derivative_over_tau = g_k_derivative / tau_adapt
                neg_grad_gk_derivative = -grad_g_k_derivative
                grad_tau_derivative = neg_grad_gk_derivative * gk_derivative_over_tau
                grad_p_tau_adapt = grad_tau_derivative + grad_tau_linear
                grad_g_k_adapt = grad_s_g_k + (-grad_g_k_derivative / tau_adapt)
                grad_s_g_k = grad_g_k_adapt + grad_g_k_voltage
            """)
        else:
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
    record_vjp_after = (
        (("psc", "grad_s_psc = grad_s_psc + grad_output"),) if model == "alpha" else ()
    )
    return BackendFragments(tuple(f), tuple(b), record_vjp_after)
