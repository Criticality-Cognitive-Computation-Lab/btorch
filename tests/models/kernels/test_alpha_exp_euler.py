"""Validate stable AlphaPSC exponential-Euler kernels against the Module."""

import pytest
import torch

from btorch.models.kernels import CellSpec
from tests.models.kernels.helpers import (
    GPU_BACKENDS,
    assert_module_sequence,
    require_cuda_backend,
)


def _leaf(value: torch.Tensor) -> torch.Tensor:
    return value.detach().requires_grad_()


def _leaves(
    steps: int,
    layout: str,
    *,
    tau: float = 3.0,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    batch, neurons = 2, 4
    lane = torch.arange(batch * neurons, device="cuda", dtype=torch.float32).reshape(
        batch, neurons
    )
    time = torch.arange(steps, device="cuda", dtype=torch.float32)[:, None, None]
    x = _leaf(0.18 + 0.04 * torch.sin(time * 0.17 + lane * 0.11))
    states = (
        _leaf(0.03 * torch.cos(lane * 0.13)),
        _leaf(-0.02 * torch.sin(lane * 0.19)),
    )
    if layout == "scalar":
        parameters = (
            _leaf(torch.tensor(tau, device="cuda")),
            _leaf(torch.tensor(0.7, device="cuda")),
        )
    elif layout == "broadcast":
        tau_values = torch.linspace(tau, tau + 0.9, neurons, device="cuda")[None]
        gain_storage = torch.linspace(0.5, 1.3, 2 * neurons + 1, device="cuda")
        parameters = (_leaf(tau_values), _leaf(gain_storage[1::2]))
    elif layout == "shared":
        shared = _leaf(torch.tensor(tau, device="cuda"))
        parameters = (shared, shared)
    else:
        raise AssertionError(layout)
    return x, states, parameters


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("layout", ("scalar", "broadcast", "shared"))
def test_alpha_tiny_step_long_sequence_matches_module_vjp(backend, layout):
    require_cuda_backend(backend)
    x, states, parameters = _leaves(128, layout)

    # Small dt makes ``1 - exp(-dt/tau)`` lose the update that the Module
    # computes through expm1. T=128 also exercises reverse replay/history use.
    assert_module_sequence(
        backend, CellSpec("alpha", dt=0.0001), x, states, parameters, ("psc", "h")
    )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_alpha_infinite_tau_matches_module_limit_and_vjp(backend):
    require_cuda_backend(backend)
    x, states, parameters = _leaves(16, "scalar", tau=float("inf"))

    # At infinite tau both ODE derivatives and their linear coefficient are
    # zero. The exp-Euler coefficient takes its finite dt limit, while input
    # still updates h in the delivery step.
    actual = assert_module_sequence(
        backend, CellSpec("alpha", dt=0.01), x, states, parameters, ("psc", "h")
    )
    assert torch.isfinite(actual[0]).all()


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_alpha_input_affects_psc_in_delivery_step(backend):
    require_cuda_backend(backend)
    x, states, parameters = _leaves(3, "broadcast")
    x = torch.zeros_like(x)
    x[0] = 1.0
    x.requires_grad_()
    states = tuple(_leaf(torch.zeros_like(value)) for value in states)

    # AlphaPSC first adds the connection input to h, then advances psc from
    # that updated h. A spike therefore produces nonzero current immediately.
    actual = assert_module_sequence(
        backend, CellSpec("alpha", dt=0.25), x, states, parameters, ("psc", "h")
    )
    assert torch.count_nonzero(actual[0][0]) == actual[0][0].numel()
