"""Declare scalar forward and hand-derived reverse stages for cell DSLs.

These statements are direct templates, not an expression IR. All three
GPU backends substitute only their elementary math functions and storage
syntax. GLIF adaptation modes are explicitly unrolled from the static
state shape.
"""

import math

from ..contracts import CellSpec
from ..templates import BackendFragments, get_template


def scalar_layout(spec: CellSpec, modes: int = 0):
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


def get_fragments(spec: CellSpec, modes: int = 0) -> BackendFragments:
    """Compose fixed charge, adapt, fire, reset and reverse stages."""
    states, parameters = scalar_layout(spec, modes)
    model = spec.model
    if model not in (
        "if",
        "lif",
        "glif3",
        "izhikevich",
        "exponential",
        "alpha",
        "alpha_billeh",
        "dual_exponential",
    ):
        raise NotImplementedError(f"No built-in GPU fragments for {model!r}")
    dt = "dt"
    f = []
    b = [f"grad_p_{name} = 0.0" for name in parameters]
    if model == "exponential":
        f += _lines(f"""
            decay = exp(-{dt} / tau_syn)
            psc_next = decay * psc_prev + x
            output = psc_next
        """)
        b += _lines(f"""
            grad_s_psc = grad_s_psc + grad_output
            grad_x = grad_s_psc
            grad_p_tau_syn = grad_s_psc * psc_prev * decay * {dt} / (tau_syn * tau_syn)
            grad_s_psc = grad_s_psc * decay
        """)
    elif model == "alpha":
        f += _lines(f"""
            decay = exp(-{dt} / tau_syn)
            h_next = decay * h_prev + g_max * x
            psc_next = decay * psc_prev + (1.0 - decay) * h_next
            output = psc_next
        """)
        b += _lines(f"""
            grad_s_psc = grad_s_psc + grad_output
            gh = grad_s_h + grad_s_psc * (1.0 - decay)
            gd = grad_s_psc * (psc_prev - h_next) + gh * h_prev
            grad_x = gh * g_max
            grad_p_g_max = gh * x
            grad_p_tau_syn = gd * decay * {dt} / (tau_syn * tau_syn)
            grad_s_h = gh * decay
            grad_s_psc = grad_s_psc * decay
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
            rise = exp(-{dt} / tau_rise)
            decay = exp(-{dt} / tau_decay)
            g_rise_next = rise * g_rise_prev + x
            g_decay_next = decay * g_decay_prev + x
            psc_next = a * (decay * g_decay_next - rise * g_rise_next)
            output = psc_next
        """)
        b += _lines(f"""
            grad_s_psc = grad_s_psc + grad_output
            gr = grad_s_g_rise - grad_s_psc * a * rise
            gd = grad_s_g_decay + grad_s_psc * a * decay
            grad_p_a = grad_s_psc * (decay * g_decay_next - rise * g_rise_next)
            grad_rise = -grad_s_psc * a * g_rise_next + gr * g_rise_prev
            grad_p_tau_rise = grad_rise * rise * {dt} / (tau_rise * tau_rise)
            grad_decay = grad_s_psc * a * g_decay_next + gd * g_decay_prev
            grad_p_tau_decay = grad_decay * decay * {dt} / (tau_decay * tau_decay)
            grad_x = gr + gd
            grad_s_g_rise = gr * rise
            grad_s_g_decay = gd * decay
            grad_s_psc = 0.0
        """)
    else:
        if model == "if":
            f += [f"v_charged = v_prev + {dt} * x / c_m"]
        elif model == "lif":
            f += [f"v_charged = v_prev + {dt} * (-(v_prev - v_reset) / tau + x / c_m)"]
        elif model == "glif3":
            f += [
                f"decay = exp(-{dt} / tau)",
                "drive = x"
                + (
                    " + " + " + ".join(f"Iasc_{i}_prev" for i in range(modes))
                    if modes
                    else ""
                ),
                "target = v_rest + tau * drive / c_m",
                "v_charged = decay * v_prev + (1.0 - decay) * target",
            ]
            f += [
                f"Iasc_{i}_next = exp(-{dt} * k_{i}) * Iasc_{i}_prev"
                for i in range(modes)
            ]
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
            "output = where(norm >= 0.0, 1.0, 0.0)",
        ]
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
        if not spec.detach_reset:
            b += [f"gspike = gspike - grad_s_v * ({reset_gap})"]
        if not spec.hard_reset:
            b += [f"grad_p_{cutoff} = -grad_s_v * output"]
        if model == "glif3":
            b += [
                f"grad_p_asc_amps_{i} = grad_s_Iasc_{i} * output" for i in range(modes)
            ]
            if not spec.detach_reset:
                b += [
                    f"gspike = gspike + grad_s_Iasc_{i} * asc_amps_{i}"
                    for i in range(modes)
                ]
        if model == "izhikevich":
            b += ["grad_p_d = grad_s_u * output"]
            if not spec.detach_reset:
                b += ["gspike = gspike + grad_s_u * d"]
        if spec.refractory:
            b += [
                "ref_gate = where(refractory_next > 0.0, 1.0, 0.0)",
                "grad_s_refractory = grad_s_refractory * ref_gate",
                "grad_p_tau_ref = grad_s_refractory * output",
            ]
            if not spec.detach_reset:
                b += ["gspike = gspike + grad_s_refractory * tau_ref"]
            b += ["gspike = gspike * where(refractory_prev == 0.0, 1.0, 0.0)"]
        b += [f"z = {spec.alpha!r} * norm"]
        if spec.surrogate == "sigmoid":
            b += [
                f"sigma = 1.0 / (1.0 + exp(-{2 * math.log(math.sqrt(2) + 1)} * z))",
                "surrogate_grad = 4.0 * sigma * (1.0 - sigma)",
            ]
        elif spec.surrogate == "atan":
            b += ["surrogate_grad = 1.0 / (1.0 + z * z)"]
        elif spec.surrogate == "atan_approx":
            b += [
                "denom = 1.0 + 0.28 * z * z",
                "surrogate_grad = (1.0 - 0.28 * z * z) / (denom * denom)",
            ]
        elif spec.surrogate == "erf":
            b += [f"surrogate_grad = exp(-{math.log(2)} * z * z)"]
        elif spec.surrogate == "superspike":
            b += [
                f"denom = 1.0 + {math.sqrt(2) - 1} * abs(z)",
                "surrogate_grad = 1.0 / (denom * denom)",
            ]
        else:
            b += ["surrogate_grad = maximum(1.0 - 0.5 * abs(z), 0.0)"]
        b += _lines(f"""
            gnorm = gspike * {spec.damping!r} * surrogate_grad
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
                target = v_rest + tau * drive / c_m
                grad_x = gcharged * (1.0 - decay) * tau / c_m
                d_decay_tau = decay * {dt} / (tau * tau)
                d_target_tau = (1.0 - decay) * drive / c_m
                grad_p_tau = gcharged * ((v_prev - target) * d_decay_tau + d_target_tau)
                grad_p_c_m = -gcharged * (1.0 - decay) * tau * drive / (c_m * c_m)
                grad_p_v_rest = gcharged * (1.0 - decay)
                grad_s_v = gcharged * decay
            """)
            for i in range(modes):
                b += [
                    f"adapt_decay_{i} = exp(-{dt} * k_{i})",
                    f"grad_p_k_{i} = -{dt} * grad_s_Iasc_{i} "
                    f"* adapt_decay_{i} * Iasc_{i}_prev",
                    f"grad_s_Iasc_{i} = grad_s_Iasc_{i} * exp(-{dt} * k_{i}) + grad_x",
                ]
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
    return BackendFragments(tuple(f), tuple(b))
