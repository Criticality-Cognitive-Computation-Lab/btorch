"""Validate recurrent AlphaPSC public-record gradient association."""

import copy

import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.kernels import KernelOptions
from btorch.models.neurons import GLIF3
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import ATan
from btorch.models.synapse import AlphaPSC
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


RECORDS = (
    "neuron.v",
    "neuron.Iasc",
    "neuron.refractory",
    "synapse.psc",
    "synapse.h",
)


def _model(backend: str, *, steps: int = 128, neurons: int = 64):
    from btorch.models.connection import SparseConnection

    device = torch.device("cuda")
    lane = torch.arange(neurons, device=device, dtype=torch.float32)
    indegree = 8
    post = torch.arange(neurons, device=device).repeat_interleave(indegree)
    offsets = torch.tensor([0, 1, 3, 7, 11, 17, 23, 31], device=device)
    pre = (post.reshape(neurons, indegree) + offsets).remainder(neurons).flatten()
    values = indegree**-0.5 * torch.sin(
        torch.arange(pre.numel(), device=device, dtype=torch.float32) * 0.17
    )
    connection = SparseConnection.from_edges(
        pre, post, neurons, neurons, values=values, bias=True
    )
    with torch.no_grad():
        connection.bias.copy_(0.02 * torch.cos(lane * 0.13))

    modes = torch.arange(2, device=device, dtype=torch.float32)
    neuron = GLIF3(
        neurons,
        v_threshold=-50.0 + 0.3 * torch.sin(lane * 0.07),
        v_reset=-70.0 + 0.2 * torch.cos(lane * 0.05),
        v_rest=-70.0,
        c_m=0.05 + 0.002 * (lane.remainder(5) / 5),
        tau=20.0 + lane.remainder(7) * 0.1,
        k=0.1 + 0.03 * ((lane[:, None] + modes) % 5),
        asc_amps=0.03 * torch.sin(lane[:, None] * 0.11 + modes),
        tau_ref=0.0,
        hard_reset=False,
        surrogate_function=ATan(alpha=2.0, spiking=True),
        trainable_param={"v_threshold", "c_m", "tau", "k", "asc_amps"},
        device=device,
        dtype=torch.float32,
    )
    model = RecurrentNN(
        neuron,
        AlphaPSC(neurons, 3.0, connection, g_max=1.0),
        backend=KernelOptions(
            neuron=backend, enable_fp_fusion=False, precise_division=True
        ),
        update_state_names=RECORDS,
    ).to(device)
    init_net_state(model, batch_size=1, device=device, dtype=torch.float32)

    phase = lane[None]
    model.neuron.v = (-63.0 + 9.0 * torch.sin(phase * 0.19)).requires_grad_()
    model.neuron.Iasc = (
        0.02 * torch.sin(phase[..., None] * 0.13 + modes)
    ).requires_grad_()
    model.synapse.psc = (0.01 * torch.cos(phase * 0.17)).requires_grad_()
    model.synapse.h = (0.01 * torch.sin(phase * 0.23)).requires_grad_()
    time = torch.arange(steps, device=device, dtype=torch.float32)[:, None, None]
    source = (
        0.45 + 0.35 * torch.sin(time * 0.31 + phase[None] * 0.07)
    ).requires_grad_()
    return model, source


def _run(model: RecurrentNN, source: torch.Tensor):
    output, records = model(source)
    final = {
        "neuron.v": model.neuron.v,
        "neuron.Iasc": model.neuron.Iasc,
        "synapse.psc": model.synapse.psc,
        "synapse.h": model.synapse.h,
    }
    # Public PSC records and final states are independent consumers of the
    # same recurrent boundary. Mixed-sign feedback makes their cotangents
    # partially cancel, exposing the FP32 association used by the real graph.
    loss = output.square().mean()
    loss = loss + 0.28 * records["synapse.psc"].square().mean()
    loss = loss + 0.21 * records["synapse.h"].square().mean()
    loss = loss + 0.07 * records["neuron.v"].square().mean()
    loss = loss + 0.11 * records["neuron.Iasc"].square().mean()
    loss = loss + 0.37 * final["synapse.psc"].square().mean()
    loss = loss + 0.29 * final["synapse.h"].square().mean()
    loss = loss + 0.13 * final["neuron.v"].square().mean()
    loss = loss + 0.17 * final["neuron.Iasc"].square().mean()
    return output, records, final, loss


def _leaves(model: RecurrentNN, source: torch.Tensor):
    return {
        "input": source,
        "initial.neuron.v": model.neuron.v,
        "initial.neuron.Iasc": model.neuron.Iasc,
        "initial.synapse.psc": model.synapse.psc,
        "initial.synapse.h": model.synapse.h,
        "sparse.weight": model.synapse.linear.weight.value,
    }


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_recurrent_alpha_psc_record_and_final_state_vjp_matches_module(backend):
    require_cuda_backend(backend)
    model, source = _model(backend)
    reference = copy.deepcopy(model)
    reference.backend = "torch"

    actual_source = source.detach().clone().requires_grad_()
    reference_source = source.detach().clone().requires_grad_()
    actual_leaves = _leaves(model, actual_source)
    reference_leaves = _leaves(reference, reference_source)
    with environ.context(dt=1.0):
        actual = _run(model, actual_source)
        expected = _run(reference, reference_source)

    torch.testing.assert_close(actual[0], expected[0], atol=3e-5, rtol=3e-5)
    for name in RECORDS:
        torch.testing.assert_close(
            actual[1][name], expected[1][name], atol=3e-5, rtol=3e-5
        )
    for name in actual[2]:
        torch.testing.assert_close(
            actual[2][name], expected[2][name], atol=3e-5, rtol=3e-5
        )
    assert 0 < int(actual[0].sum()) < actual[0].numel()

    gradients = torch.autograd.grad(actual[-1], tuple(actual_leaves.values()))
    expected_gradients = torch.autograd.grad(
        expected[-1], tuple(reference_leaves.values())
    )
    for name, value, trusted in zip(
        actual_leaves,
        gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(value, trusted, atol=8e-5, rtol=8e-4, msg=name)
