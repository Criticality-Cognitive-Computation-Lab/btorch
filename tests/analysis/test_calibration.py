"""Calibrate an actual recurrent model without consuming its state or VJP."""

import pytest
import torch

from btorch.analysis import compute_input_scale
from btorch.models import environ, functional
from btorch.models.neurons import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC


def test_rate_calibration_restores_trial_state_and_gradients():
    # Heterogeneous current makes population rate adjustable without changing
    # the circuit. A strided neuron axis also exercises the normal input API.
    network = RecurrentNN(
        LIF(4, tau=5.0, c_m=1.0, v_threshold=1.0, trainable_param={"tau"}),
        ExponentialPSC(4, 3.0, torch.nn.Linear(4, 4, bias=False)),
        update_state_names=("neuron.v", "synapse.psc"),
    )
    with torch.no_grad():
        network.synapse.linear.weight.fill_(0.002)
    functional.init_net_state(network, batch_size=1, device="cpu", dtype=torch.float32)
    network.neuron.v.requires_grad_()
    initial = functional.named_hidden_states(network)
    storage = torch.linspace(0.2, 0.8, 8, dtype=torch.float32).expand(250, 1, 8).clone()
    inputs = storage[..., ::2].requires_grad_()
    roots = (inputs, initial["neuron.v"], *network.parameters())

    def forward_vjp():
        functional.set_hidden_states(network, initial)
        spikes, records = network(inputs)
        loss = spikes.mean() + records["neuron.v"].square().mean()
        return torch.autograd.grad(loss, roots)

    with environ.context(dt=1.0):
        expected = forward_vjp()
        functional.set_hidden_states(network, initial)
        scale, measured = compute_input_scale(
            network, inputs, target_hz=20.0, dt_ms=1.0, tolerance_hz=1.0
        )
        spikes, records = network(inputs * scale)
        assert abs(spikes.mean().item() * 1000 - 20.0) <= 1.0
        assert abs(measured - 20.0) <= 1.0
        assert records.keys() == {"neuron.v", "synapse.psc"}
        actual = forward_vjp()
        for gradient, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(gradient, reference, rtol=0, atol=0)

        # An unreachable target must also hand back the unconsumed trial;
        # otherwise calibration would silently change a subsequent simulation.
        functional.set_hidden_states(network, initial)
        with pytest.raises(RuntimeError, match="Cannot calibrate"):
            compute_input_scale(
                network,
                torch.zeros_like(inputs),
                target_hz=20.0,
                dt_ms=1.0,
                max_trials=2,
            )
        for gradient, reference in zip(forward_vjp(), expected, strict=True):
            torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
