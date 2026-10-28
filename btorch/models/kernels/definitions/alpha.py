"""Model-local forward and reverse scalar definitions."""

from ..templates import (
    BackendFragments,
    ModelTemplate,
    RecordCotangent,
    register_model_template,
)
from .integration import _exp_euler_backward, _exp_euler_forward, _lines


def fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
    dt = "dt"
    f, b = ([], [])
    from ..backends.fragments import scalar_layout

    b = [f"grad_p_{name} = 0.0" for name in scalar_layout(spec, modes)[1]]
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
    b += ("grad_s_psc = grad_s_psc + grad_output", RecordCotangent("psc"))
    b += _lines(f"""
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
    return BackendFragments(tuple(f), tuple(b))


def module_factory(spec, states, neurons):
    import torch

    from ...synapse import AlphaPSC

    return AlphaPSC(
        neurons, linear=torch.nn.Identity(), tau_syn=4.0, backend="torch", step_mode="s"
    )


register_model_template(
    ModelTemplate(
        "alpha",
        ("psc", "h"),
        ("tau_syn", "g_max"),
        ("psc", "h"),
        history_sources=(),
        fragments=fragments,
        output_state="psc",
        reverse_parameter_warps=(("tau_syn", 2),),
        module_factory=module_factory,
    )
)
