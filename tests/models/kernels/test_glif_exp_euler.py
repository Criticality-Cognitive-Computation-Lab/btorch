"""Validate stable GLIF exponential-Euler kernels against the Module."""

import pytest
import torch

from btorch.models.kernels import CellSpec
from tests.models.kernels.helpers import (
    GPU_BACKENDS,
    assert_module_sequence,
    require_cuda_backend,
)


def _leaves(steps: int, *, tau: float | torch.Tensor = 20.0):
    shape = (2, 4)
    lane = torch.arange(8, device="cuda", dtype=torch.float32).reshape(shape)
    time = torch.arange(steps, device="cuda", dtype=torch.float32)[:, None, None]
    x = (0.45 + 0.12 * torch.sin(time * 0.19 + lane * 0.07)).requires_grad_()
    states = (
        (-63.0 + 0.8 * torch.sin(lane * 0.11)).requires_grad_(),
        torch.stack(
            (0.02 * torch.cos(lane * 0.13), -0.015 * torch.sin(lane * 0.17)),
            dim=-1,
        ).requires_grad_(),
    )
    tau_value = torch.as_tensor(tau, device="cuda", dtype=torch.float32)
    parameters = (
        torch.full((4,), 1000.0, device="cuda").requires_grad_(),
        torch.tensor(-70.0, device="cuda", requires_grad=True),
        torch.tensor(0.05, device="cuda", requires_grad=True),
        tau_value.requires_grad_(),
        torch.tensor(-70.0, device="cuda", requires_grad=True),
        torch.stack(
            (torch.zeros_like(lane), 0.13 + lane * 0.001), dim=-1
        ).requires_grad_(),
        torch.stack(
            (0.01 * torch.sin(lane), -0.02 * torch.cos(lane)), dim=-1
        ).requires_grad_(),
    )
    return x, states, parameters


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("steps", (20, 128))
def test_glif_small_updates_match_module_histories_and_vjps(backend, steps):
    require_cuda_backend(backend)
    x, states, parameters = _leaves(steps)
    histories = ("v", "Iasc", "v_charged") if steps == 20 else ("v_charged",)

    # T=128 omits the public Iasc trajectory so reverse must reconstruct it
    # from block checkpoints using exactly the same exponential-Euler update.
    assert_module_sequence(
        backend,
        CellSpec("glif3", dt=0.001, surrogate="atan"),
        x,
        states,
        parameters,
        histories,
    )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_glif_zero_adaptation_rate_and_infinite_tau_match_module(backend):
    require_cuda_backend(backend)
    x, states, parameters = _leaves(20, tau=float("inf"))
    parameters = (
        *parameters[:5],
        torch.zeros_like(parameters[5], requires_grad=True),
        parameters[6],
    )

    # At tau=inf the membrane update is the Euler limit; at k=0 each ASC is
    # unchanged before the spike increment. The reverse path must avoid the
    # former inf*0 target derivative and retain the Module's finite gradients.
    assert_module_sequence(
        backend,
        CellSpec("glif3", dt=0.01, surrogate="atan"),
        x,
        states,
        parameters,
        ("v", "Iasc", "v_charged"),
    )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_glif_sums_adaptation_modes_before_adding_external_current(backend):
    require_cuda_backend(backend)
    x, states, parameters = _leaves(1)
    x = torch.full_like(x, 1e8, requires_grad=True)
    states = (
        states[0],
        torch.stack(
            (torch.full_like(states[0], -1e8), torch.ones_like(states[0])),
            dim=-1,
        ).requires_grad_(),
    )

    # The Module adds x to the sum over adaptation modes. With cancellation,
    # x + (I0 + I1) is zero in FP32, whereas (x + I0) + I1 is one. Preserving
    # this grouping matters even though the real-valued equations agree.
    assert_module_sequence(
        backend,
        CellSpec("glif3", surrogate="atan"),
        x,
        states,
        parameters,
        ("v", "Iasc", "v_charged"),
    )
