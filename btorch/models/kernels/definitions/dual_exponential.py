"""Model-local forward and reverse scalar definitions."""

from ..templates import BackendFragments, ModelTemplate, register_model_template
from .integration import _exp_euler_backward, _exp_euler_forward, _lines


def fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
    dt = "dt"
    f, b = ([], [])
    from ..backends.fragments import scalar_layout

    b = [f"grad_p_{name} = 0.0" for name in scalar_layout(spec, modes)[1]]
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
    return BackendFragments(tuple(f), tuple(b))


def module_factory(spec, states, neurons):
    import torch

    from ...synapse import DualExponentialPSC

    return DualExponentialPSC(
        neurons,
        linear=torch.nn.Identity(),
        tau_rise=2.0,
        tau_decay=4.0,
        backend="torch",
        step_mode="s",
    )


register_model_template(
    ModelTemplate(
        "dual_exponential",
        ("psc", "g_rise", "g_decay"),
        ("tau_rise", "tau_decay", "a"),
        ("psc", "g_rise", "g_decay"),
        history_sources=(),
        fragments=fragments,
        output_state="psc",
        module_factory=module_factory,
    )
)
