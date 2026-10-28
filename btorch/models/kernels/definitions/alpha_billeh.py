"""Model-local forward and reverse scalar definitions."""

import math

from ..templates import BackendFragments, ModelTemplate, register_model_template
from .integration import _lines


def fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
    f, b = ([], [])
    from ..backends.fragments import scalar_layout

    b = [f"grad_p_{name} = 0.0" for name in scalar_layout(spec, modes)[1]]
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
    return BackendFragments(tuple(f), tuple(b))


def module_factory(spec, states, neurons):
    import torch

    from ...synapse import AlphaPSCBilleh

    return AlphaPSCBilleh(
        neurons, linear=torch.nn.Identity(), tau_syn=4.0, backend="torch", step_mode="s"
    )


register_model_template(
    ModelTemplate(
        "alpha_billeh",
        ("psc", "h"),
        ("tau_syn", "syn_decay"),
        ("psc", "h"),
        history_sources=(),
        fragments=fragments,
        output_state="psc",
        module_factory=module_factory,
        required_dt=1.0,
    )
)
