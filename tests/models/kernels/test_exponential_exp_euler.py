"""Validate ExponentialPSC exponential-Euler kernels against the Module."""

import pytest
import torch

from btorch.models.kernels import NeuronSpec
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
    large_initial_state: bool = False,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor],
    tuple[torch.Tensor],
    tuple[torch.Tensor],
]:
    batch, neurons = 2, 4
    lane = torch.arange(batch * neurons, device="cuda", dtype=torch.float32).reshape(
        batch, neurons
    )
    time = torch.arange(steps, device="cuda", dtype=torch.float32)[:, None, None]
    x = _leaf(0.18 + 0.04 * torch.sin(time * 0.17 + lane * 0.11))
    scale = 1e4 if large_initial_state else 0.3
    states = (_leaf(scale + 0.03 * torch.cos(lane * 0.13)),)

    if layout == "scalar":
        tau_value = _leaf(torch.tensor(tau, device="cuda"))
        parameters = (tau_value,)
        parameter_roots = (tau_value,)
    elif layout == "broadcast":
        tau_value = _leaf(torch.linspace(tau, tau + 0.9, neurons, device="cuda")[None])
        parameters = (tau_value,)
        parameter_roots = (tau_value,)
    elif layout == "strided":
        # Differentiate through the backing storage, not only through the view.
        # This catches kernels that silently replace a noncontiguous tau view
        # without scattering its VJP back to the original tensor.
        storage = torch.linspace(
            tau - 0.4, tau + 1.2, 2 * neurons + 1, device="cuda"
        ).requires_grad_()
        tau_value = storage[1::2]
        assert not tau_value.is_contiguous()
        parameters = (tau_value,)
        parameter_roots = (storage,)
    else:
        raise AssertionError(layout)
    return x, states, parameters, parameter_roots


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("layout", ("scalar", "broadcast", "strided"))
def test_exponential_tiny_step_matches_module_operation_graph(backend, layout):
    require_cuda_backend(backend)
    x, states, parameters, parameter_roots = _leaves(
        128, layout, large_initial_state=True
    )

    # At this FP32 scale, replacing the Module's expm1-based exponential-Euler
    # graph with ``psc * exp(-dt / tau)`` accumulates visible rounding error.
    # The long sequence also exercises history and reverse replay semantics.
    assert_module_sequence(
        backend,
        NeuronSpec("exponential", dt=1e-06),
        x,
        states,
        parameters,
        ("psc",),
        parameter_roots=parameter_roots,
    )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_exponential_infinite_tau_matches_zero_linear_limit(backend):
    require_cuda_backend(backend)
    x, states, parameters, parameter_roots = _leaves(16, "scalar", tau=float("inf"))

    # tau=inf makes both the derivative and its declared linear coefficient
    # zero. Decay leaves the existing PSC unchanged while delivered events are
    # still accumulated, and the tau VJP must retain the Module's finite limit.
    actual = assert_module_sequence(
        backend,
        NeuronSpec("exponential", dt=0.01),
        x,
        states,
        parameters,
        ("psc",),
        parameter_roots=parameter_roots,
    )
    assert all(torch.isfinite(value).all() for value in actual)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_exponential_nonzero_state_and_events_preserve_delivery_order(backend):
    require_cuda_backend(backend)
    x, states, parameters, parameter_roots = _leaves(5, "broadcast")
    event = torch.zeros_like(x)
    event[0] = 0.8
    event[2, 0] = -0.25
    event[4, 1] = 0.4
    x = _leaf(event)

    # The actual Module first decays the nonzero stored PSC, then adds this
    # step's event and emits the result. Pulses at multiple steps make an
    # accidental input delay or input-before-decay schedule observable.
    actual = assert_module_sequence(
        backend,
        NeuronSpec("exponential", dt=0.25),
        x,
        states,
        parameters,
        ("psc",),
        parameter_roots=parameter_roots,
    )
    torch.testing.assert_close(actual[0], actual[2], atol=0, rtol=0)
