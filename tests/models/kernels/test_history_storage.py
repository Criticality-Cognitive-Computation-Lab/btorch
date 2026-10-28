"""Keep live charged memories independent of recorded trajectory storage."""

import copy

import torch
from torch import nn

from btorch.models import environ
from btorch.models.kernels import KernelOptions
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC


def _assert_one_step_storage(value):
    # A last-row view has the right logical shape but pins every prior timestep.
    assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def test_standalone_charged_memory_owns_storage_and_preserves_vjp():
    neuron = LIF(
        3,
        pre_spike_v=True,
        trainable_param={"tau"},
        backend=KernelOptions(neuron="torch"),
        step_mode="m",
    )
    neuron.init_state(batch_size=1, dtype=torch.float32)
    reference = copy.deepcopy(neuron)
    reference.backend = "torch"
    x = torch.linspace(0.1, 1.4, 21).reshape(7, 1, 3).requires_grad_()
    y = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        actual = neuron(x)
        expected = torch.stack([reference.single_step_forward(step) for step in y])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(neuron.v_pre_spike, reference.v_pre_spike)
    _assert_one_step_storage(neuron.v_pre_spike)
    # The live charged state can itself enter a loss. Detaching it to save
    # storage would pass output parity but lose this gradient contribution.
    (actual.sum() + neuron.v_pre_spike.square().sum()).backward()
    (expected.sum() + reference.v_pre_spike.square().sum()).backward()
    torch.testing.assert_close(x.grad, y.grad)
    torch.testing.assert_close(neuron.tau.grad, reference.tau.grad)


def test_recurrent_charged_record_and_live_memory_preserve_continuation_vjp(
    reference_backend,
):
    torch.manual_seed(19)
    model = RecurrentNN(
        LIF(3, pre_spike_v=True, trainable_param={"tau"}),
        AlphaPSC(3, 2.0, nn.Linear(3, 3, bias=False)),
        backend=reference_backend,
        chunk_size=None,
        update_state_names=("neuron.v", "neuron.v_pre_spike", "synapse.psc"),
    ).float()
    model.neuron.init_state(batch_size=1, dtype=torch.float32)
    model.synapse.init_state(batch_size=1, dtype=torch.float32)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = torch.linspace(0.1, 1.4, 27).reshape(9, 1, 3).requires_grad_()
    y = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        first, records = model(x[:6])
        _assert_one_step_storage(model.neuron.v_pre_spike)
        torch.testing.assert_close(
            model.neuron.v_pre_spike, records["neuron.v_pre_spike"][-1]
        )
        second, tail = model(x[6:])
        expected, expected_records = reference(y)
    actual = torch.cat((first, second))
    traces = {name: torch.cat((value, tail[name])) for name, value in records.items()}
    torch.testing.assert_close(actual, expected)
    for name, value in traces.items():
        torch.testing.assert_close(value, expected_records[name])
    loss = actual.sum() + sum(value.square().sum() for value in traces.values())
    oracle = expected.sum() + sum(
        value.square().sum() for value in expected_records.values()
    )
    loss = loss + model.neuron.v_pre_spike.square().sum()
    oracle = oracle + reference.neuron.v_pre_spike.square().sum()
    loss.backward()
    oracle.backward()
    torch.testing.assert_close(x.grad, y.grad)
    for actual_parameter, expected_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
