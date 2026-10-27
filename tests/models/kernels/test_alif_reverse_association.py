"""Protect ALIF-family reverse associations that are observable in FP32."""

import pytest
import torch

from btorch.models import environ, surrogate
from btorch.models.kernels import KernelOptions
from btorch.models.kernels.adapters import prepare_cell
from btorch.models.kernels.backends.fragments import get_fragments
from btorch.models.kernels.contracts import NeuronSpec
from btorch.models.kernels.operators import run_multi_step
from btorch.models.neurons.alif import ELIF


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_elif_voltage_cotangents_follow_the_module_accumulation_order():
    """Preserve the cancellation-sensitive ordering found in a long VJP.

    The values are the terminal local cut from the changed-input T=128
    qualification. Retaining tensors from the actual Module keeps this
    test tied to its autograd graph rather than a separately implemented
    model.
    """
    module = ELIF(
        1,
        v_threshold=0.62,
        v_reset=-0.12,
        c_m=1.15,
        g_leak=0.72,
        E_leak=-0.08,
        E_k=-0.75,
        tau_adapt=3.8,
        dg_k=0.09,
        delta_T=0.16,
        v_T=0.31,
        hard_reset=True,
        pre_spike_v=True,
        surrogate_function=surrogate.Sigmoid(alpha=2.0, damping_factor=1.0),
        device="cuda",
    )
    drive = torch.tensor([[0.03852575272321701]], device="cuda", requires_grad=True)
    voltage = torch.tensor([[0.20708854496479034]], device="cuda", requires_grad=True)
    adaptation = torch.tensor(
        [[0.07182629406452179]], device="cuda", requires_grad=True
    )
    module.v = voltage
    module.g_k = adaptation
    with environ.context(dt=0.25):
        spec, _, states, parameters = prepare_cell(module, drive.unsqueeze(0))

    observed = {}
    original_dv = module.dV
    original_reset = module.neuronal_reset
    original_exp = torch.exp
    exp_count = 0

    def observe_dv(v, g_k, x):
        derivative, linear = original_dv(v, g_k, x)
        observed["derivative"] = derivative
        derivative.retain_grad()
        return derivative, linear

    def observe_reset(spike):
        observed["charged"] = module.v
        module.v.retain_grad()
        return original_reset(spike)

    def observe_exp(value, *args, **kwargs):
        nonlocal exp_count
        result = original_exp(value, *args, **kwargs)
        if exp_count == 0:
            observed["exp_arg"] = value
            value.retain_grad()
        exp_count += 1
        return result

    module.dV = observe_dv
    module.neuronal_reset = observe_reset
    torch.exp = observe_exp
    try:
        with environ.context(dt=0.25):
            output = module.single_step_forward(drive)
    finally:
        torch.exp = original_exp

    (actual,) = torch.autograd.grad(output, voltage, retain_graph=True)
    gcharged = observed["charged"].grad
    grad_v_numerator = observed["derivative"].grad / module.c_m
    grad_v_leak = grad_v_numerator * module.g_leak
    grad_v_potassium = grad_v_numerator * adaptation
    grad_exp_difference = observed["exp_arg"].grad / module.delta_T

    dynamics = (-grad_v_leak - grad_v_potassium) + grad_exp_difference
    expected = gcharged + dynamics
    former = (gcharged - grad_v_leak - grad_v_potassium) + grad_exp_difference
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, former)

    native = run_multi_step(
        drive.unsqueeze(0),
        states,
        parameters,
        spec,
        backend=KernelOptions(
            neuron="triton", enable_fp_fusion=False, precise_division=True
        ),
    )
    (native_voltage,) = torch.autograd.grad(native[0], voltage)
    assert torch.equal(native_voltage, actual)

    reverse = get_fragments(NeuronSpec("elif", dt=0.25, hard_reset=True)).backward
    passive_index = reverse.index("grad_v_dynamics = -grad_v_leak - grad_v_potassium")
    exponential_index = reverse.index(
        "grad_v_dynamics = grad_v_dynamics + grad_exp_difference"
    )
    identity_index = reverse.index("grad_s_v = gcharged + grad_v_dynamics")
    assert passive_index < exponential_index < identity_index


def test_spike_cotangents_follow_the_module_accumulation_order():
    """Compose shared spike uses in the same order as the actual Module
    graph."""
    torch.manual_seed(0)
    batch, neurons = 4, 9
    module = ELIF(
        neurons,
        v_threshold=0.62,
        v_reset=-0.12,
        c_m=1.15,
        g_leak=0.72,
        E_leak=-0.08,
        E_k=-0.75,
        tau_adapt=3.8,
        dg_k=0.37,
        delta_T=0.16,
        v_T=0.31,
        tau_ref=0.83,
        hard_reset=True,
        detach_reset=False,
        pre_spike_v=True,
        surrogate_function=surrogate.Sigmoid(alpha=2.0, damping_factor=1.0),
    )
    v = (torch.rand(batch, neurons) * 0.7).requires_grad_()
    g_k = (torch.rand(batch, neurons) * 0.2).requires_grad_()
    refractory = (torch.rand(batch, neurons) * 0.4).requires_grad_()
    drive = (torch.rand(batch, neurons) * 1.2).requires_grad_()
    module.v = v
    module.g_k = g_k
    module.refractory = refractory

    observed = {}
    reset = module.neuronal_reset

    def observe_reset(spike):
        # The retained cotangent belongs to the one spike tensor shared by the
        # public output and all three reset updates.
        observed["spike"] = spike
        observed["charged"] = module.v
        spike.retain_grad()
        return reset(spike)

    module.neuronal_reset = observe_reset
    with environ.context(dt=0.25):
        output = module.single_step_forward(drive)

    output_cotangent = torch.randn_like(output)
    voltage_cotangent = torch.randn_like(v)
    adaptation_cotangent = torch.randn_like(g_k) * 100.0
    refractory_cotangent = torch.randn_like(refractory) * 10.0
    torch.autograd.grad(
        (output, module.v, module.g_k, module.refractory),
        (drive, v, g_k, refractory),
        (
            output_cotangent,
            voltage_cotangent,
            adaptation_cotangent,
            refractory_cotangent,
        ),
    )

    spike = observed["spike"]
    charged = observed["charged"]
    refractory_gate = (refractory + spike * module.tau_ref - 0.25 > 0.0).to(spike.dtype)
    refractory_term = refractory_cotangent * refractory_gate * module.tau_ref
    adaptation_term = adaptation_cotangent * module.dg_k
    reset_term = -voltage_cotangent * (charged - module.v_reset)

    # Autograd visits the refractory update, adaptation update, and voltage
    # reset in this order. The former native order loses one ULP in this case.
    expected = ((output_cotangent + refractory_term) + adaptation_term) + reset_term
    former = ((output_cotangent + reset_term) + adaptation_term) + refractory_term
    assert torch.equal(spike.grad, expected)
    assert not torch.equal(spike.grad, former)

    reverse = get_fragments(
        NeuronSpec("elif", dt=0.25, hard_reset=True, refractory=True)
    ).backward
    refractory_index = reverse.index("gspike = gspike + grad_s_refractory * tau_ref")
    adaptation_index = reverse.index("gspike = gspike + grad_s_g_k * dg_k")
    reset_index = reverse.index("gspike = gspike + gspike_reset")
    firing_gate_index = reverse.index(
        "gspike = gspike * where(refractory_prev == 0.0, 1.0, 0.0)"
    )
    assert refractory_index < adaptation_index < reset_index < firing_gate_index
