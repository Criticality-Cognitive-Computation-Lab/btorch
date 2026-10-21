"""Sequence contracts for voltage-aware ConductancePSC execution."""

from __future__ import annotations

import copy

import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.synapse import ConductancePSC, ExponentialPSC


DT = 0.35
T, B, N_SOURCE, N_POST, N_DELAY = 7, 2, 2, 3, 3
DTYPE = torch.float32
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _values(device):
    generator = torch.Generator().manual_seed(2718)
    values = {
        "weight": torch.randn(
            N_POST, N_SOURCE * N_DELAY, generator=generator, dtype=DTYPE
        ),
        "psc": torch.randn(B, N_POST, generator=generator, dtype=DTYPE) * 0.2,
        "history": torch.randn(B, N_DELAY, N_SOURCE, generator=generator, dtype=DTYPE)
        * 0.1,
        "z": torch.randn(T, B, N_SOURCE, generator=generator, dtype=DTYPE) * 0.3,
        "v": -55.0 + torch.randn(T, 1, N_POST, generator=generator, dtype=DTYPE),
    }
    return {name: value.to(device) for name, value in values.items()}


def _build(values, *, differentiable=False):
    device = values["weight"].device
    linear = torch.nn.Linear(
        N_SOURCE * N_DELAY, N_POST, bias=False, dtype=DTYPE, device=device
    )
    # Declaring the delay-expanded input makes ConductancePSC own the ring and
    # exercises continuation of both physical state objects in one real Module.
    linear.n_delay = N_DELAY
    linear.n_pre = N_SOURCE
    with torch.no_grad():
        linear.weight.copy_(values["weight"])
    synapse = ConductancePSC(
        N_POST,
        tau_syn=torch.tensor(4.7, dtype=DTYPE, device=device),
        linear=linear,
        reversal_potential=torch.tensor(
            [0.0, -72.0, -10.0], dtype=DTYPE, device=device
        ),
        step_mode="m",
    )
    init_net_state(synapse, batch_size=B, dtype=DTYPE, device=device)
    psc = values["psc"].clone().requires_grad_(differentiable)
    history = values["history"].clone().requires_grad_(differentiable)
    synapse.psc = psc
    synapse.delayed_spike.history = history
    synapse.tau_syn.requires_grad_(differentiable)
    synapse.E.requires_grad_(differentiable)
    return synapse, (psc, history, linear.weight, synapse.tau_syn, synapse.E)


def _single_step_sequence(synapse, z_seq, v_seq):
    """Use the public single-step Module path as the sequence oracle."""
    return torch.stack(
        [synapse.single_step_forward(z_seq[t], v_seq[t]) for t in range(len(z_seq))]
    )


def _assert_final_state_equal(actual, expected):
    torch.testing.assert_close(actual.psc, expected.psc, rtol=0.0, atol=0.0)
    if actual.delayed_spike is None:
        assert expected.delayed_spike is None
        return
    torch.testing.assert_close(
        actual.delayed_spike.history,
        expected.delayed_spike.history,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_voltage_sequence_matches_single_step_and_chunk_continuation(device):
    """Whole and chunked calls preserve nonzero conductance and delay state."""
    values = _values(device)
    whole, _ = _build(values)
    chunked, _ = _build(values)
    oracle, _ = _build(values)
    voltage = values["v"].expand(T, B, N_POST)

    with torch.no_grad(), environ.context(dt=DT):
        whole_output = whole(values["z"], v_seq=voltage)
        oracle_output = _single_step_sequence(oracle, values["z"], voltage)
        split = 3
        chunked_output = torch.cat(
            (
                chunked(values["z"][:split], v_seq=voltage[:split]),
                chunked(values["z"][split:], v_seq=voltage[split:]),
            )
        )

    assert values["psc"].abs().sum() > 0
    assert values["history"].abs().sum() > 0
    torch.testing.assert_close(whole_output, oracle_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(chunked_output, oracle_output, rtol=0.0, atol=0.0)
    _assert_final_state_equal(whole, oracle)
    _assert_final_state_equal(chunked, oracle)


@pytest.mark.parametrize("device", DEVICES)
def test_voltage_sequence_vjp_matches_single_step_oracle(device):
    """Output and final-state losses reach every differentiable COBA input."""
    values = _values(device)
    candidate, candidate_state = _build(values, differentiable=True)
    oracle, oracle_state = _build(values, differentiable=True)
    z_candidate = values["z"].clone().requires_grad_()
    z_oracle = values["z"].clone().requires_grad_()
    # A singleton batch dimension is intentionally broadcast over both lanes;
    # its VJP must sum the lane contributions just like repeated single steps.
    v_candidate = values["v"].clone().requires_grad_()
    v_oracle = values["v"].clone().requires_grad_()

    with environ.context(dt=DT):
        actual = candidate(z_candidate, v_seq=v_candidate)
        expected = _single_step_sequence(oracle, z_oracle, v_oracle)
        generator = torch.Generator().manual_seed(419)
        output_cotangent = torch.randn(
            actual.shape, generator=generator, dtype=DTYPE
        ).to(device)
        psc_cotangent = torch.randn(
            candidate.psc.shape, generator=generator, dtype=DTYPE
        ).to(device)
        history_cotangent = torch.randn(
            candidate.delayed_spike.history.shape,
            generator=generator,
            dtype=DTYPE,
        ).to(device)
        actual_gradients = torch.autograd.grad(
            (actual, candidate.psc, candidate.delayed_spike.history),
            (z_candidate, v_candidate, *candidate_state),
            (output_cotangent, psc_cotangent, history_cotangent),
        )
        expected_gradients = torch.autograd.grad(
            (expected, oracle.psc, oracle.delayed_spike.history),
            (z_oracle, v_oracle, *oracle_state),
            (output_cotangent, psc_cotangent, history_cotangent),
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    _assert_final_state_equal(candidate, oracle)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        assert actual_gradient.abs().sum() > 0
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("device", DEVICES)
def test_voltage_sequence_time_broadcast_with_multidimensional_neurons(device):
    """A length-one voltage sequence broadcasts over time and batch axes."""
    generator = torch.Generator().manual_seed(812)
    linear = torch.nn.Linear(4, 4, bias=False, dtype=DTYPE, device=device)
    candidate = ConductancePSC(
        (2, 2),
        torch.tensor(3.0, dtype=DTYPE, device=device),
        linear,
        reversal_potential=torch.tensor(0.0, dtype=DTYPE, device=device),
        step_mode="m",
    )
    oracle = copy.deepcopy(candidate)
    init_net_state(candidate, batch_size=2, dtype=DTYPE, device=device)
    init_net_state(oracle, batch_size=2, dtype=DTYPE, device=device)
    z_seq = torch.randn(4, 2, 2, 2, generator=generator, dtype=DTYPE).to(device)
    voltage = torch.randn(1, 1, 2, 2, generator=generator, dtype=DTYPE).to(device)

    with torch.no_grad(), environ.context(dt=DT):
        actual = candidate(z_seq, v_seq=voltage)
        expected = _single_step_sequence(oracle, z_seq, voltage.expand(4, 2, 2, 2))

    assert actual.shape == (4, 2, 2, 2)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    _assert_final_state_equal(candidate, oracle)


@pytest.mark.parametrize("device", DEVICES)
def test_omitted_voltage_preserves_legacy_stateless_convolution(device):
    """The existing CUBA-compatible path still ignores and preserves state."""
    values = _values(device)
    candidate, _ = _build(values)
    oracle = copy.deepcopy(candidate)
    initial_psc = candidate.psc.clone()
    initial_history = candidate.delayed_spike.history.clone()

    with torch.no_grad(), environ.context(dt=DT):
        actual = candidate(values["z"], kernel_len=11)
        expected = ExponentialPSC.multi_step_forward(oracle, values["z"], kernel_len=11)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(candidate.psc, initial_psc, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        candidate.delayed_spike.history, initial_history, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("device", DEVICES)
def test_empty_voltage_sequence_preserves_memory_adjoint_and_current_dtype(device):
    """An empty chunk advances no memory and keeps the COBA output contract."""
    values = _values(device)
    candidate, leaves = _build(values, differentiable=True)
    oracle, _ = _build(values)
    initial_psc, initial_history = leaves[:2]
    voltage = torch.empty(
        0, 1, N_POST, dtype=torch.float64, device=device, requires_grad=True
    )

    with environ.context(dt=DT):
        output = candidate(values["z"][:0], v_seq=voltage)
        # The existing single-step Module establishes mixed-voltage dtype
        # promotion; no second implementation of conductance dynamics is used.
        reference = oracle.single_step_forward(
            values["z"][0],
            torch.full((B, N_POST), -55.0, dtype=voltage.dtype, device=voltage.device),
        )
        loss = (
            output.sum()
            + candidate.psc.square().sum()
            + candidate.delayed_spike.history.square().sum()
        )
        grad_psc, grad_history, grad_voltage = torch.autograd.grad(
            loss, (initial_psc, initial_history, voltage)
        )

    assert output.shape == (0, B, N_POST)
    assert output.dtype == reference.dtype
    torch.testing.assert_close(candidate.psc, initial_psc, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        candidate.delayed_spike.history, initial_history, rtol=0.0, atol=0.0
    )
    # State cotangents pass directly across an empty chunk, while its empty
    # voltage receives a well-defined empty gradient.
    torch.testing.assert_close(grad_psc, 2 * initial_psc)
    torch.testing.assert_close(grad_history, 2 * initial_history)
    torch.testing.assert_close(grad_voltage, torch.zeros_like(voltage))
