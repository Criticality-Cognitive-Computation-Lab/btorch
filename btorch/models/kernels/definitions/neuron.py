"""Compose model-local voltage dynamics with firing and reset contracts."""

import math as scalar_math
from collections.abc import Callable
from dataclasses import dataclass

from ...surrogate.sigmoid import _SIG_K
from ...surrogate.superspike import _SS_K
from ..templates import BackendFragments
from .integration import _lines


@dataclass(frozen=True)
class NeuronMath:
    charge: Callable
    reverse_charge: Callable
    after_fire: Callable
    reverse_reset: Callable
    cutoff: str = "v_threshold"
    reset_after_adaptation: bool = False
    refractory_first: bool = False
    ordered_normalization: bool = False


def neuron_fragments(spec, modes, prepared_values, authoritative_output, math):
    from ..backends.fragments import scalar_layout

    dt = "dt"
    f = list(math.charge(spec, modes, prepared_values, authoritative_output))
    b = [f"grad_p_{name} = 0.0" for name in scalar_layout(spec, modes)[1]]
    cutoff = math.cutoff
    f += ["gap = v_threshold - v_reset", f"norm = (v_charged - {cutoff}) / gap"]
    if spec.surrogate == "poisson":
        if not authoritative_output:
            if spec.poisson.spiking:
                f += [
                    f"rate = exp({spec.poisson.rho!r} * norm) / {spec.poisson.tau!r}",
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
    f += math.after_fire(spec, modes, prepared_values, authoritative_output)
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
    if spec.refractory and math.refractory_first:
        b += [
            "ref_gate = where(refractory_next > 0.0, 1.0, 0.0)",
            "grad_s_refractory = grad_s_refractory * ref_gate",
            "grad_p_tau_ref = grad_s_refractory * output",
        ]
        if not spec.detach_reset:
            b += ["gspike = gspike + grad_s_refractory * tau_ref"]
    if not spec.detach_reset and (not math.reset_after_adaptation):
        b += [f"gspike = gspike - grad_s_v * ({reset_gap})"]
    if not spec.hard_reset:
        b += [f"grad_p_{cutoff} = -grad_s_v * output"]
    b += math.reverse_reset(spec, modes, prepared_values, authoritative_output)
    if spec.refractory:
        if not math.refractory_first:
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
    elif spec.surrogate != "sigmoid" or not math.ordered_normalization:
        b += [f"z = {spec.alpha!r} * norm"]
    if spec.surrogate == "poisson":
        pass
    elif spec.surrogate == "sigmoid":
        if math.ordered_normalization:
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
        b += [f"surrogate_grad = exp(-{scalar_math.log(2)} * z * z)"]
    elif spec.surrogate == "superspike":
        b += [
            f"denom = 1.0 + {_SS_K} * abs(z)",
            "surrogate_grad = 1.0 / (denom * denom)",
        ]
    else:
        b += ["surrogate_grad = maximum(1.0 - 0.5 * abs(z), 0.0)"]
    if spec.surrogate == "poisson":
        gnorm = f"gspike * (surrogate_grad * {spec.damping!r})"
    elif spec.surrogate == "atan":
        gnorm = "gspike * atan_damped_grad"
    elif spec.surrogate == "sigmoid" and math.ordered_normalization:
        gnorm = "gspike * surrogate_grad"
    else:
        gnorm = f"gspike * {spec.damping!r} * surrogate_grad"
    b += [f"gnorm = {gnorm}"]
    if spec.surrogate_width is not None:
        b += [f"gnorm = where(abs(norm) < {spec.surrogate_width!r}, gnorm, 0.0)"]
    if math.ordered_normalization:
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
    b += math.reverse_charge(spec, modes, prepared_values, authoritative_output)
    return BackendFragments(tuple(f), tuple(b))
