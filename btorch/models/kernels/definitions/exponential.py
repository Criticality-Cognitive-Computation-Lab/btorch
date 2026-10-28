"""Model-local forward and reverse scalar definitions."""

from ..templates import BackendFragments, ModelTemplate, register_model_template
from .integration import _exp_euler_backward, _exp_euler_forward, _lines


def fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
    dt = "dt"
    f, b = ([], [])
    from ..backends.fragments import scalar_layout

    b = [f"grad_p_{name} = 0.0" for name in scalar_layout(spec, modes)[1]]
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
    return BackendFragments(tuple(f), tuple(b))


def module_factory(spec, states, neurons):
    import torch

    from ...synapse import ExponentialPSC

    return ExponentialPSC(
        neurons, linear=torch.nn.Identity(), tau_syn=4.0, backend="torch", step_mode="s"
    )


register_model_template(
    ModelTemplate(
        "exponential",
        ("psc",),
        ("tau_syn",),
        ("psc",),
        history_sources=(),
        fragments=fragments,
        output_state="psc",
        module_factory=module_factory,
        recurrent_checkpoint=True,
    )
)
