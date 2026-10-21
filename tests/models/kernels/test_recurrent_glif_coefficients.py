"""Validate recurrent GLIF coefficient hoisting against the real Module."""

import copy

import pytest
import torch
from torch import nn

from btorch.models import environ
from btorch.models.kernels import CellSpec, KernelOptions
from btorch.models.kernels.backends import engine, fused
from btorch.models.kernels.recurrent_operator import _glif_membrane_coefficients
from btorch.models.kernels.templates import get_template
from btorch.models.neurons.glif import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


STATE_NAMES = ("neuron.v", "neuron.Iasc", "synapse.psc")


def _tau(layout: str, device: str = "cuda") -> nn.Parameter:
    if layout == "scalar":
        value = torch.tensor(2.75, device=device)
    elif layout == "broadcast":
        # One time constant per batch broadcasts across all four neurons.
        value = torch.tensor([[2.25], [3.5]], device=device)
    else:
        storage = torch.tensor(
            [99.0, 2.1, 99.0, 2.6, 99.0, 3.2, 99.0, 4.0, 99.0],
            device=device,
        )
        value = storage[1::2]
        assert value.shape == (4,) and value.stride() == (2,)
    return nn.Parameter(value)


def _model(backend: str, tau_layout: str, device: str = "cuda") -> RecurrentNN:
    neuron = GLIF3(
        4,
        v_threshold=0.62,
        v_reset=0.0,
        v_rest=0.0,
        c_m=1.0,
        tau=2.0,
        k=[0.13, 0.04],
        asc_amps=[-0.08, 0.03],
        trainable_param={"tau"},
    ).to(device)
    # Assigning the leaf after construction preserves layouts that the public
    # size-normalizing constructor intentionally turns into contiguous values.
    neuron.tau = _tau(tau_layout, device)
    connection = nn.Linear(4, 4, bias=False, device=device)
    with torch.no_grad():
        connection.weight.copy_(
            torch.tensor(
                [
                    [0.11, -0.03, 0.02, 0.00],
                    [0.04, 0.09, -0.02, 0.01],
                    [-0.01, 0.03, 0.10, 0.02],
                    [0.02, 0.00, 0.04, 0.08],
                ],
                device=device,
            )
        )
    model = RecurrentNN(
        neuron,
        ExponentialPSC(4, 2.4, connection),
        backend=KernelOptions(neuron=backend),
        update_state_names=STATE_NAMES,
    ).float()
    neuron.init_state(batch_size=2, dtype=torch.float32, device=device)
    model.synapse.init_state(batch_size=2, dtype=torch.float32, device=device)
    lane = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4)
    neuron.v = (0.07 + lane * 0.006).requires_grad_()
    neuron.Iasc = torch.stack(
        (0.015 * torch.sin(lane), -0.01 * torch.cos(lane)), dim=-1
    ).requires_grad_()
    model.synapse.psc = (0.02 * torch.cos(lane * 0.3)).requires_grad_()
    return model


def _input(steps: int, device: str = "cuda") -> torch.Tensor:
    lane = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4)
    time = torch.arange(steps, device=device, dtype=torch.float32)[:, None, None]
    return (0.31 + 0.19 * torch.sin(time * 0.37 + lane * 0.23)).requires_grad_()


def _loss(output: torch.Tensor, records: dict[str, torch.Tensor]) -> torch.Tensor:
    # Continuous state losses expose coefficient and state-adjoint differences
    # even when two executions happen to emit the same binary spike sequence.
    return (
        output.sum()
        + records["neuron.v"].square().sum()
        + 0.3 * records["neuron.Iasc"].square().sum()
        + 0.2 * records["synapse.psc"].square().sum()
    )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("tau_layout", ("scalar", "broadcast", "strided"))
def test_recurrent_glif_tau_layouts_match_module_forward_and_vjp(backend, tau_layout):
    require_cuda_backend(backend)
    model = _model(backend, tau_layout)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = _input(20)
    x_reference = x.detach().clone().requires_grad_()
    initial = (model.neuron.v, model.neuron.Iasc, model.synapse.psc)
    initial_reference = (
        reference.neuron.v,
        reference.neuron.Iasc,
        reference.synapse.psc,
    )

    with environ.context(dt=0.125):
        actual, records = model(x)
        expected, expected_records = reference(x_reference)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    for name in STATE_NAMES:
        torch.testing.assert_close(
            records[name], expected_records[name], atol=3e-5, rtol=3e-5
        )

    _loss(actual, records).backward()
    _loss(expected, expected_records).backward()
    torch.testing.assert_close(x.grad, x_reference.grad, atol=5e-5, rtol=5e-4)
    for value, expected_value in zip(initial, initial_reference):
        torch.testing.assert_close(
            value.grad, expected_value.grad, atol=5e-5, rtol=5e-4
        )
    torch.testing.assert_close(
        model.neuron.tau.grad,
        reference.neuron.tau.grad,
        atol=5e-5,
        rtol=5e-4,
    )
    torch.testing.assert_close(
        model.synapse.linear.weight.grad,
        reference.synapse.linear.weight.grad,
        atol=5e-5,
        rtol=5e-4,
    )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_recurrent_glif_recomputes_coefficients_after_tau_and_dt_updates(backend):
    require_cuda_backend(backend)
    model = _model(backend, "strided")
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    first = _input(7).detach()
    second = (_input(9) * 0.83).detach()

    with torch.no_grad(), environ.context(dt=0.0625):
        actual_first, records_first = model(first)
        expected_first, expected_records_first = reference(first)
    torch.testing.assert_close(actual_first, expected_first, atol=3e-5, rtol=3e-5)
    for name in STATE_NAMES:
        torch.testing.assert_close(
            records_first[name],
            expected_records_first[name],
            atol=3e-5,
            rtol=3e-5,
        )

    # Both values are runtime inputs to coefficient preparation. Updating the
    # original strided tau storage and changing dt must affect the continued
    # trajectory without relying on a stale generated-source cache entry.
    with torch.no_grad():
        model.neuron.tau.mul_(1.35)
        reference.neuron.tau.mul_(1.35)
    with torch.no_grad(), environ.context(dt=0.4):
        actual_second, records_second = model(second)
        expected_second, expected_records_second = reference(second)
    torch.testing.assert_close(actual_second, expected_second, atol=3e-5, rtol=3e-5)
    for name in STATE_NAMES:
        torch.testing.assert_close(
            records_second[name],
            expected_records_second[name],
            atol=3e-5,
            rtol=3e-5,
        )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_recurrent_glif_infinite_tau_matches_zero_linear_module_vjp(backend):
    require_cuda_backend(backend)
    model = _model(backend, "scalar")
    model.neuron.tau = nn.Parameter(torch.tensor(float("inf"), device="cuda"))
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = _input(12)
    x_reference = x.detach().clone().requires_grad_()

    with environ.context(dt=0.01):
        actual, records = model(x)
        expected, expected_records = reference(x_reference)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    for name in STATE_NAMES:
        torch.testing.assert_close(
            records[name], expected_records[name], atol=3e-5, rtol=3e-5
        )
    _loss(actual, records).backward()
    _loss(expected, expected_records).backward()

    assert torch.isfinite(model.neuron.tau.grad).all()
    assert torch.isfinite(reference.neuron.tau.grad).all()
    torch.testing.assert_close(
        model.neuron.tau.grad,
        reference.neuron.tau.grad,
        atol=5e-5,
        rtol=5e-4,
    )
    torch.testing.assert_close(x.grad, x_reference.grad, atol=5e-5, rtol=5e-4)


def test_recurrent_glif_inference_private_abi_contains_phi_without_delta():
    neuron_spec = CellSpec("glif3", dt=torch.tensor(0.25))
    psc_spec = CellSpec("exponential", dt=torch.tensor(0.25))
    n_neuron, modes = 8, 2
    neuron_layouts = tuple(
        engine._ParameterLayout(
            (),
            (),
            (n_neuron, modes) if name in ("k", "asc_amps") else (n_neuron,),
            1,
        )
        for name in get_template(neuron_spec).parameter_names
    )
    psc_layouts = tuple(
        engine._ParameterLayout((), (), (n_neuron,), 1)
        for _ in get_template(psc_spec).parameter_names
    )
    phi_layout = engine._ParameterLayout((n_neuron,), (1,), (n_neuron,), n_neuron)
    source, pointers = fused.generate_fused_source(
        "triton",
        neuron_spec,
        psc_spec,
        n_neuron,
        1,
        1,
        modes=modes,
        neuron_parameter_layouts=neuron_layouts,
        psc_parameter_layouts=psc_layouts,
        neuron_coefficient_layouts=(phi_layout,),
    )
    pointer_names = tuple(name for name, _ in pointers)

    # Inference needs only the precomputed multiplier. Delta is part of the
    # analytic reverse formula and must not add an inference pointer or load.
    assert "neuron_glif_phi_coefficient" in pointer_names
    assert "neuron_glif_delta_coefficient" not in pointer_names
    assert "neuron_glif_delta_coefficient_ptr" not in source
    assert (
        len(
            _glif_membrane_coefficients(
                neuron_spec,
                (
                    torch.tensor(1.0),
                    torch.tensor(0.0),
                    torch.tensor(1.0),
                    torch.tensor(2.0),
                    torch.tensor(0.0),
                    torch.ones(2),
                    torch.zeros(2),
                ),
                backward=False,
            )
        )
        == 1
    )
