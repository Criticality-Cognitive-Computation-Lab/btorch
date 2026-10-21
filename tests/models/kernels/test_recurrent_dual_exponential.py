"""Validate full BPTT through independent dual-exponential recurrent
memories."""

import copy

import pytest
import torch

from btorch.models import environ
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.functional import init_net_state
from btorch.models.kernels import KernelOptions
from btorch.models.neurons import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import DualExponentialPSC
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


RECORDS = ("neuron.v", "synapse.psc", "synapse.g_rise", "synapse.g_decay")


def _build(connection_kind, backend):
    neurons, batch = 4, 2
    bias = torch.linspace(-0.015, 0.025, neurons)
    if connection_kind == "delayed_sparse":
        connection = SparseConnection.from_edges(
            torch.tensor([0, 1, 1, 2, 2, 3, 3, 0]),
            torch.tensor([1, 0, 2, 1, 3, 2, 0, 3]),
            neurons,
            neurons,
            Synapse(delay=torch.tensor([0, 2, 1, 0, 2, 1, 2, 0]), n_delay=3),
            values=torch.tensor([0.12, -0.08, 0.10, 0.07, -0.06, 0.13, 0.09, -0.04]),
            bias=bias,
        )
    else:
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
            connection.bias.copy_(bias)
    model = RecurrentNN(
        LIF(
            neurons,
            v_threshold=0.65,
            v_reset=-0.1,
            c_m=1.1,
            tau=4.0,
            trainable_param={"tau", "v_threshold"},
        ),
        DualExponentialPSC(neurons, 4.5, 1.7, connection, A=0.8),
        backend=KernelOptions(
            neuron=backend,
            connection="triton",
            enable_fp_fusion=False,
            precise_division=True,
        ),
        update_state_names=RECORDS,
    ).to(device="cuda", dtype=torch.float32)
    init_net_state(model, batch_size=batch, device="cuda", dtype=torch.float32)
    lane = torch.arange(batch * neurons, device="cuda").reshape(batch, neurons)
    model.neuron.v = (-0.06 + 0.02 * lane).requires_grad_()
    # These are independent incoming memories. In particular, psc_init affects
    # neuron step zero even though the first PSC update overwrites that memory.
    model.synapse.psc = (0.07 + 0.01 * lane).requires_grad_()
    model.synapse.g_rise = (-0.03 + 0.004 * lane).requires_grad_()
    model.synapse.g_decay = (0.09 - 0.003 * lane).requires_grad_()
    for name in ("tau_rise", "tau_decay", "a"):
        getattr(model.synapse, name).requires_grad_()
    if model.synapse.delayed_spike is not None:
        history = model.synapse.delayed_spike.history
        phase = torch.arange(history.numel(), device="cuda").reshape(history.shape)
        model.synapse.delayed_spike.history = (
            0.04 * torch.sin(phase * 0.3)
        ).requires_grad_()
    return model


def _roots(model, x, events):
    roots = {"input": x, "external_event": events}
    roots.update(dict(model.named_parameters()))
    roots["initial.v"] = model.neuron.v
    for name in ("psc", "g_rise", "g_decay"):
        roots[f"initial.{name}"] = getattr(model.synapse, name)
    for name in ("tau_rise", "tau_decay", "a"):
        roots[f"parameter.{name}"] = getattr(model.synapse, name)
    if model.synapse.delayed_spike is not None:
        roots["initial.delay_history"] = model.synapse.delayed_spike.history
    return roots


def _final(model):
    states = (
        model.neuron.v,
        model.synapse.psc,
        model.synapse.g_rise,
        model.synapse.g_decay,
    )
    if model.synapse.delayed_spike is not None:
        states += (model.synapse.delayed_spike.history,)
    return states


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("connection_kind", ("dense", "delayed_sparse"))
def test_two_forwards_preserve_full_bptt_through_all_dual_memories(
    backend, connection_kind
):
    """Second-call losses differentiate first-call input and incoming memories.

    Whole-sequence actual Modules are the oracle. Random losses touch
    only second-chunk outputs/records and final state, so matching the
    first chunk alone cannot satisfy the gradient checks. Sparse weight
    VJPs stay in the stored edge order, including heterogeneous delay-
    ring continuation.
    """
    require_cuda_backend(backend)
    torch.manual_seed(492)
    actual = _build(connection_kind, backend)
    expected = copy.deepcopy(actual)
    expected.backend = "torch"
    x = (0.25 + 0.3 * torch.rand(9, 2, 4, device="cuda")).requires_grad_()
    events = (0.04 * torch.randn_like(x)).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    reference_events = events.detach().clone().requires_grad_()
    roots = _roots(actual, x, events)
    reference_roots = _roots(expected, reference_x, reference_events)
    assert roots.keys() == reference_roots.keys()

    with environ.context(dt=0.3):
        first, first_records = actual(x[:4], events[:4])
        second, second_records = actual(x[4:], events[4:])
        whole, whole_records = expected(reference_x, reference_events)
    torch.testing.assert_close(torch.cat((first, second)), whole, atol=5e-6, rtol=5e-5)
    for name in RECORDS:
        torch.testing.assert_close(
            torch.cat((first_records[name], second_records[name])),
            whole_records[name],
            atol=5e-6,
            rtol=5e-5,
        )
    actual_values = (
        second,
        *(second_records[name] for name in RECORDS),
        *_final(actual),
    )
    reference_values = (
        whole[4:],
        *(whole_records[name][4:] for name in RECORDS),
        *_final(expected),
    )
    for value, reference in zip(_final(actual), _final(expected), strict=True):
        torch.testing.assert_close(value, reference, atol=5e-6, rtol=5e-5)
    generator = torch.Generator(device="cuda").manual_seed(930)
    cotangents = tuple(
        torch.randn(value.shape, device="cuda", generator=generator)
        for value in actual_values
    )
    gradients = torch.autograd.grad(
        actual_values, tuple(roots.values()), cotangents, allow_unused=True
    )
    reference_gradients = torch.autograd.grad(
        reference_values, tuple(reference_roots.values()), cotangents, allow_unused=True
    )
    for (name, root), gradient, reference in zip(
        roots.items(), gradients, reference_gradients, strict=True
    ):
        assert (gradient is None) == (reference is None), name
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert bool(gradient.abs().sum() > 0), name
        torch.testing.assert_close(gradient, reference, atol=5e-5, rtol=5e-4, msg=name)
        if name in ("input", "external_event"):
            assert bool(gradient[:4].abs().sum() > 0), name
        if root is getattr(actual.synapse.linear.weight, "value", None):
            assert gradient.numel() == actual.synapse.linear.nnz
