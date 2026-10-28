"""Small reusable cell workloads for kernel contract tests."""

import torch

from btorch.models.kernels import NeuronSpec
from btorch.models.kernels.templates import get_template


def _if_parameters():
    return {"v_threshold": 1.0, "v_reset": 0.0, "c_m": 1.2}


def _lif_parameters():
    return {**_if_parameters(), "tau": 3.0}


def _glif3_parameters():
    return {
        **_lif_parameters(),
        "v_rest": -0.2,
        "k": 0.1,
        "asc_amps": 0.2,
    }


def _izhikevich_parameters():
    return {
        **_if_parameters(),
        "k": 0.1,
        "v_rest": -0.2,
        "v_peak": 1.3,
        "a": 0.1,
        "b": 0.2,
        "d": 0.3,
    }


def _alpha_parameters():
    return {"tau_syn": 2.0, "g_max": 0.7}


def _exponential_parameters():
    return {"tau_syn": 2.0}


def _alpha_billeh_parameters():
    return {"tau_syn": 2.0, "syn_decay": 0.6}


def _dual_exponential_parameters():
    return {"tau_rise": 2.0, "tau_decay": 4.0, "a": 0.1}


_PARAMETER_FACTORIES = {
    "if": _if_parameters,
    "lif": _lif_parameters,
    "glif3": _glif3_parameters,
    "izhikevich": _izhikevich_parameters,
    "alpha": _alpha_parameters,
    "exponential": _exponential_parameters,
    "alpha_billeh": _alpha_billeh_parameters,
    "dual_exponential": _dual_exponential_parameters,
}


def cell_inputs(model: str, refractory: bool = False):
    """Build one valid, differentiable workload for a registered cell model.

    The fixture preserves the template's state and parameter ordering.
    It is intentionally small so tests can select explicit
    model/trajectory cases without constructing a second model factory
    for every contract.
    """
    try:
        parameter_factory = _PARAMETER_FACTORIES[model]
    except KeyError:
        raise KeyError(f"unknown cell model: {model!r}")
    spec = NeuronSpec(model, dt=1, refractory=refractory)
    template = get_template(spec)
    values = parameter_factory()
    x = (torch.randn(2, 3) * 0.4 + 0.6).requires_grad_()
    states = []
    for name in template.state_names:
        if name == "refractory" and refractory:
            value = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0]])
        else:
            shape = (*x.shape, *(2,) * template.extra_dims(name))
            value = torch.randn(shape) * 0.2
        states.append(value.requires_grad_())
    parameters = []
    for name in template.parameter_names:
        shape = (*x.shape[1:], *(2,) * template.extra_dims(name))
        parameters.append(torch.full(shape, values[name]).requires_grad_())
    return spec, x, tuple(states), tuple(parameters)


def glif_alpha_case(*, backend: str = "torch", neurons: int = 4, edges: int = 8):
    """Build the public delayed heterogeneous GLIF3/AlphaPSC example case."""
    from examples.rsnn import build_model
    from examples.rsnn_common import ExampleOptions

    return build_model(
        ExampleOptions(
            neurons=neurons,
            edges=edges,
            seed=17,
            device="cpu",
            neuron_backend=backend,
            sparse_backend="torch",
            method="pull",
            cudagraph=False,
            chunk_size=None,
        )
    )


def izh_dual_case(*, backend: str = "torch", neurons: int = 4, edges: int = 8):
    """Build the public heterogeneous Izhikevich/DualExponential case."""
    from examples.rsnn_common import ExampleOptions
    from examples.rsnn_izhikevich import build_model

    return build_model(
        ExampleOptions(
            neurons=neurons,
            edges=edges,
            seed=23,
            device="cpu",
            neuron_backend=backend,
            sparse_backend="torch",
            method="pull",
            cudagraph=False,
            chunk_size=None,
        )
    )


def register_external_psc_template():
    """Register the test-only third-party PSC used by extension contracts."""
    from btorch.models import environ
    from btorch.models.kernels.templates import (
        BackendFragments,
        ModelTemplate,
        RecordCotangent,
        register_model_template,
    )

    name = "test_external_psc"

    def fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
        del spec, modes, prepared_values, authoritative_output
        return BackendFragments(
            (
                "psc_decay = exp(-dt / tau_syn)",
                "psc_next = psc_prev * psc_decay + x",
                "output = psc_next",
            ),
            (
                "grad_p_tau_syn = 0.0",
                "grad_s_psc = grad_s_psc + grad_output",
                RecordCotangent("psc"),
                "grad_p_tau_syn = grad_s_psc * psc_prev * psc_decay * dt / "
                "(tau_syn * tau_syn)",
                "grad_x = grad_s_psc",
                "grad_s_psc = grad_s_psc * psc_decay",
            ),
        )

    class ExternalPSC(torch.nn.Module):
        kernel_model = name

        def __init__(self, neurons):
            super().__init__()
            self.register_buffer("tau_syn", torch.tensor(2.5), persistent=False)
            self.psc = torch.zeros(neurons)

        def single_step_forward(self, x):
            dt = torch.as_tensor(
                environ.get("dt"), device=self.tau_syn.device, dtype=self.tau_syn.dtype
            )
            self.psc = self.psc * torch.exp(-dt / self.tau_syn) + x
            return self.psc

    def module_factory(spec, states, neurons):
        del spec, states
        return ExternalPSC(neurons)

    def forward_step(x, states, parameters, spec):
        dt = torch.as_tensor(spec.dt, device=x.device, dtype=x.dtype)
        decay = torch.exp(-dt / parameters[0])
        psc = states[0] * decay + x
        return psc, (psc,), {"psc": psc}

    def backward_step(
        x,
        states,
        parameters,
        spec,
        grad_output,
        grad_states,
        grad_records,
    ):
        dt = torch.as_tensor(spec.dt, device=x.device, dtype=x.dtype)
        tau_syn = parameters[0]
        decay = torch.exp(-dt / tau_syn)
        grad_psc = grad_output + grad_states[0] + grad_records.get("psc", 0.0)
        grad_tau = (
            grad_psc * states[0] * decay * dt / (tau_syn * tau_syn)
        ).sum_to_size(tau_syn.shape)
        return grad_psc, (grad_psc * decay,), (grad_tau,)

    template = ModelTemplate(
        name,
        ("psc",),
        ("tau_syn",),
        ("psc",),
        fragments=fragments,
        module_factory=module_factory,
        output_state="psc",
    )
    register_model_template(
        template, forward_step=forward_step, backward_step=backward_step, replace=True
    )
    return template, NeuronSpec(name, dt=0.25), ExternalPSC
