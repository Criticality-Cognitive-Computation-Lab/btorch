"""Validate model-scoped numerical defaults at public sequence boundaries."""

import copy
from dataclasses import replace

import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.kernels import KernelOptions
from btorch.models.neurons import ALIF, ELIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


def _cuda_policy_network(neuron_type, backend, *, cudagraph=False):
    neurons, batch = 4, 2
    neuron_kwargs = {
        "v_threshold": 0.62,
        "v_reset": -0.12,
        "c_m": 1.15,
        "g_leak": 0.72,
        "E_leak": -0.08,
        "E_k": -0.75,
        "tau_adapt": 3.8,
        "dg_k": 0.09,
    }
    if neuron_type is ELIF:
        neuron_kwargs.update(delta_T=0.16, v_T=0.31)
    connection = torch.nn.Linear(neurons, neurons)
    with torch.no_grad():
        connection.weight.copy_(
            torch.tensor(
                [
                    [0.08, -0.03, 0.02, 0.04],
                    [0.03, 0.09, -0.02, 0.01],
                    [-0.01, 0.04, 0.07, 0.03],
                    [0.02, -0.01, 0.05, 0.10],
                ]
            )
        )
        connection.bias.copy_(torch.tensor([-0.018, 0.012, -0.006, 0.021]))
    model = RecurrentNN(
        neuron_type(neurons, **neuron_kwargs),
        ExponentialPSC(neurons, 2.7, connection),
        backend=backend,
        update_state_names=("neuron.v", "neuron.g_k", "synapse.psc"),
        cudagraph=cudagraph,
        cudagraph_warmup=1,
    ).to(device="cuda", dtype=torch.float32)
    init_net_state(model, batch_size=batch, device="cuda", dtype=torch.float32)
    lane = torch.arange(batch * neurons, device="cuda").reshape(batch, neurons)
    model.neuron.v = -0.11 + 0.035 * lane
    model.neuron.g_k = 0.035 + 0.008 * lane
    model.synapse.psc = -0.04 + 0.012 * lane
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("neuron_type", (ALIF, ELIF))
@pytest.mark.parametrize("math_option", ("precise_division", "precise_exp"))
def test_cuda_capture_recaptures_when_effective_math_policy_changes(
    monkeypatch, backend, neuron_type, math_option
):
    """A policy switch must never replay a graph with stale math semantics.

    Each policy is called twice with different inputs. The first call
    captures (or recaptures), while the second must replay and still
    agree with an eager native execution. An actual Torch-module network
    supplies the independent scientific reference for every continued
    trajectory.
    """
    require_cuda_backend(backend)

    torch.manual_seed(918)
    graphed = _cuda_policy_network(neuron_type, backend, cudagraph=True)
    eager_native = copy.deepcopy(graphed)
    eager_native.cudagraph = False
    reference = copy.deepcopy(graphed)
    reference.backend = "torch"
    reference.cudagraph = False

    replay_calls = []
    original_replay = torch.cuda.CUDAGraph.replay

    def counted_replay(graph):
        replay_calls.append(graph)
        return original_replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted_replay)
    generator = torch.Generator(device="cuda").manual_seed(919)
    drives = [
        torch.rand((5, 2, 4), generator=generator, device="cuda") * 0.32 - 0.05
        for _ in range(6)
    ]
    # Both model shorthands promise accurate division. The eager twin uses an
    # independently specified policy, so a shared resolver bug cannot make two
    # identically wrong shorthand paths pass the exact native comparison.
    base_options = KernelOptions(neuron=backend, precise_division=True)
    policies = (
        (backend, replace(base_options, precise_division=False), backend)
        if math_option == "precise_division"
        else (base_options, replace(base_options, precise_exp=True), base_options)
    )

    with torch.inference_mode(), environ.context(dt=0.7):
        for phase, policy in enumerate(policies, start=1):
            graphed.backend = policy
            eager_native.backend = base_options if isinstance(policy, str) else policy
            # Flip only the selected policy while keeping the other math
            # choices fixed. The middle phase must recapture, and each second
            # call below must reuse that policy's graph on changed inputs.
            for replay_index in range(2):
                drive = drives[2 * (phase - 1) + replay_index]
                # The Torch module is an independent one-call oracle. Seed it
                # from the eager native pre-call state so earlier backend ULPs
                # cannot accumulate into this policy/replay comparison.
                reference.neuron.v = eager_native.neuron.v.clone()
                reference.neuron.g_k = eager_native.neuron.g_k.clone()
                reference.synapse.psc = eager_native.synapse.psc.clone()
                expected_native, native_records = eager_native(drive)
                expected_reference, reference_records = reference(drive)
                replay_before = len(replay_calls)
                actual, actual_records = graphed(drive)

                torch.testing.assert_close(actual, expected_native, atol=0, rtol=0)
                for name in actual_records:
                    torch.testing.assert_close(
                        actual_records[name], native_records[name], atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        actual_records[name],
                        reference_records[name],
                        atol=3e-5,
                        rtol=3e-5,
                    )
                torch.testing.assert_close(
                    actual, expected_reference, atol=3e-5, rtol=3e-5
                )
                assert graphed._cudagraph_runner.n_captures == phase
                if replay_index == 1:
                    assert len(replay_calls) > replay_before
