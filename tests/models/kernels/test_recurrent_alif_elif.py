"""Validate recurrent native ALIF and ELIF execution against their Modules."""

import copy
from itertools import product

import pytest
import torch

from btorch.models import environ
from btorch.models.connection import SparseConnection
from btorch.models.functional import init_net_state
from btorch.models.kernels import KernelOptions
from btorch.models.neurons import ALIF, ELIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC
from tests.models.kernels.helpers import (
    GPU_BACKENDS,
    NewEmptyShapes,
    require_cuda_backend,
    validation_cases,
)


RECORDS = ("neuron.v", "neuron.g_k", "synapse.psc")


def _connection(kind, neurons):
    bias = torch.tensor([-0.018, 0.012, -0.006, 0.021])
    if kind == "sparse":
        return SparseConnection.from_edges(
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
            torch.tensor([1, 3, 0, 2, 1, 3, 0, 2]),
            neurons,
            neurons,
            values=torch.tensor([0.11, -0.04, 0.07, 0.09, -0.05, 0.13, 0.06, 0.08]),
            bias=bias,
        )
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
    return connection


def _build(neuron_type, connection_kind, backend, *, records=RECORDS):
    neurons, batch = 4, 2
    common = {
        "v_threshold": 0.62,
        "v_reset": -0.12,
        "c_m": 1.15,
        "g_leak": 0.72,
        "E_leak": -0.08,
        "E_k": -0.75,
        "tau_adapt": 3.8,
        "dg_k": 0.09,
        "trainable_param": {
            "v_threshold",
            "v_reset",
            "c_m",
            "g_leak",
            "E_leak",
            "E_k",
            "tau_adapt",
            "dg_k",
        },
    }
    if neuron_type is ELIF:
        common.update(delta_T=0.16, v_T=0.31)
        common["trainable_param"] = common["trainable_param"] | {"delta_T", "v_T"}
    model = RecurrentNN(
        neuron_type(neurons, **common),
        ExponentialPSC(neurons, 2.7, _connection(connection_kind, neurons)),
        backend=(
            "torch"
            if backend == "torch"
            else KernelOptions(
                neuron=backend,
                connection="triton",
                enable_fp_fusion=False,
                precise_division=True,
            )
        ),
        update_state_names=records,
    ).to(device="cuda", dtype=torch.float32)
    init_net_state(model, batch_size=batch, device="cuda", dtype=torch.float32)
    lane = torch.arange(batch * neurons, device="cuda").reshape(batch, neurons)
    model.neuron.v = (-0.11 + 0.035 * lane).requires_grad_()
    model.neuron.g_k = (0.035 + 0.008 * lane).requires_grad_()
    model.synapse.psc = (-0.04 + 0.012 * lane).requires_grad_()
    model.synapse.tau_syn.requires_grad_()
    return model


def _roots(model, drive, events):
    roots = {"drive": drive, "events": events, **dict(model.named_parameters())}
    roots.update(
        {
            "initial.v": model.neuron.v,
            "initial.g_k": model.neuron.g_k,
            "initial.psc": model.synapse.psc,
            "parameter.tau_syn": model.synapse.tau_syn,
        }
    )
    return roots


def _final_states(model):
    return model.neuron.v, model.neuron.g_k, model.synapse.psc


def _run(model, drive, events, split):
    if split is None:
        output, records = model(drive, events)
    else:
        first, first_records = model(drive[:split], events[:split])
        second, second_records = model(drive[split:], events[split:])
        output = torch.cat((first, second))
        records = {
            name: torch.cat((first_records[name], second_records[name]))
            for name in RECORDS
        }
    return (output, *(records[name] for name in RECORDS), *_final_states(model))


# Select connection scheduling and precision-lowering features independently;
# all model/policy pairs are available only in explicit full validation.
@pytest.mark.parametrize(
    "backend,connection_kind,neuron_type,fp_policy",
    validation_cases(
        product(
            GPU_BACKENDS,
            ("dense", "sparse"),
            (ALIF, ELIF),
            ("default", "strict", "precise_exp"),
        ),
        selected=(
            ("triton", "dense", ALIF, "default"),
            ("triton", "sparse", ELIF, "default"),
            ("triton", "sparse", ALIF, "strict"),
            ("triton", "dense", ELIF, "precise_exp"),
            ("cupy", "dense", ALIF, "strict"),
            ("cupy", "sparse", ELIF, "precise_exp"),
            ("tilelang", "sparse", ALIF, "default"),
            ("tilelang", "dense", ELIF, "strict"),
        ),
    ),
)
def test_recurrent_whole_and_two_forward_bptt_match_actual_modules(
    backend, connection_kind, neuron_type, fp_policy
):
    """Whole and continued native calls preserve every scientific VJP.

    The Torch ``RecurrentNN`` uses the actual ALIF or ELIF and ExponentialPSC
    Modules as the oracle. Random cotangents cover spikes, every requested
    trajectory, and all final states. The second native execution splits the
    same sequence, so gradients reaching its first chunk also verify that
    ``v``, ``g_k``, and ``psc`` remain connected across public calls.
    """
    require_cuda_backend(backend)
    torch.manual_seed(735)
    initial = _build(neuron_type, connection_kind, "torch")
    whole = copy.deepcopy(initial)
    continued = copy.deepcopy(initial)
    reference = copy.deepcopy(initial)
    options = KernelOptions(neuron=backend, connection="triton")
    if fp_policy in ("strict", "precise_exp"):
        options = KernelOptions(
            neuron=backend,
            connection="triton",
            enable_fp_fusion=False,
            precise_division=True,
            precise_exp=fp_policy == "precise_exp",
        )
    whole.backend = options
    continued.backend = options

    drive = 0.52 + 0.72 * torch.rand(8, 2, 4, device="cuda")
    events = 0.07 * torch.randn_like(drive)
    inputs = []
    for model in (whole, continued, reference):
        model_drive = drive.detach().clone().requires_grad_()
        model_events = events.detach().clone().requires_grad_()
        inputs.append(
            (model_drive, model_events, _roots(model, model_drive, model_events))
        )

    with environ.context(dt=0.28):
        whole_values = _run(whole, inputs[0][0], inputs[0][1], None)
        continued_values = _run(continued, inputs[1][0], inputs[1][1], 3)
        reference_values = _run(reference, inputs[2][0], inputs[2][1], None)

    for values in (whole_values, continued_values):
        for actual, expected in zip(values, reference_values, strict=True):
            torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
    assert 0 < int(reference_values[0].sum()) < reference_values[0].numel()

    generator = torch.Generator(device="cuda").manual_seed(936)
    cotangents = []
    for index, value in enumerate(reference_values):
        cotangent = torch.randn(value.shape, device="cuda", generator=generator)
        if index < 1 + len(RECORDS):
            # Only the second chunk's temporal losses may seed reverse. Any
            # gradient on the first chunk must therefore cross the public-call
            # boundary through the carried recurrent states.
            cotangent[:3] = 0
        cotangents.append(cotangent)
    cotangents = tuple(cotangents)
    reference_gradients = torch.autograd.grad(
        reference_values,
        tuple(inputs[2][2].values()),
        cotangents,
        allow_unused=True,
    )
    for values, (_, _, roots) in zip(
        (whole_values, continued_values), inputs[:2], strict=True
    ):
        gradients = torch.autograd.grad(
            values, tuple(roots.values()), cotangents, allow_unused=True
        )
        for (name, root), gradient, expected in zip(
            roots.items(), gradients, reference_gradients, strict=True
        ):
            assert (gradient is None) == (expected is None), name
            assert gradient is not None and torch.isfinite(gradient).all(), name
            assert bool(gradient.abs().sum() > 0), name
            torch.testing.assert_close(
                gradient, expected, atol=5e-5, rtol=5e-4, msg=name
            )
            if values is continued_values and name in ("drive", "events"):
                assert bool(gradient[:3].abs().sum() > 0), name
            if connection_kind == "sparse" and name == "synapse.linear.weight.value":
                assert gradient.numel() == whole.synapse.linear.nnz


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("neuron_type", (ALIF, ELIF))
def test_recurrent_inference_retains_no_backward_only_history(
    backend, neuron_type, monkeypatch
):
    """Inference carries adaptive state without allocating reverse histories.

    Inputs come from a trainable encoder before inference is entered. This is
    intended deployment usage: native mode selection must still avoid an
    autograd tape and private time-by-state trajectories while two calls carry
    the final ``v``, ``g_k``, and ``psc`` states exactly like the Modules.
    """
    require_cuda_backend(backend)
    from btorch.models.kernels import recurrent_operator

    actual = _build(neuron_type, "dense", backend, records=())
    expected = copy.deepcopy(actual)
    expected.backend = "torch"
    encoder = torch.nn.Linear(4, 4, bias=False, device="cuda")
    sources = [encoder(torch.rand(5, 2, 4, device="cuda")) for _ in range(2)]
    assert all(
        source.requires_grad and source.grad_fn is not None for source in sources
    )

    allocations = NewEmptyShapes()
    saved = []
    trajectory = recurrent_operator._trajectory
    native_inference = recurrent_operator.inference
    native_training = recurrent_operator.training
    operator_calls = {"inference": 0, "training": 0}

    def instrumented_trajectory(*args, **kwargs):
        with allocations:
            return trajectory(*args, **kwargs)

    def counted_inference(*args, **kwargs):
        operator_calls["inference"] += 1
        return native_inference(*args, **kwargs)

    def counted_training(*args, **kwargs):
        operator_calls["training"] += 1
        return native_training(*args, **kwargs)

    monkeypatch.setattr(recurrent_operator, "_trajectory", instrumented_trajectory)
    monkeypatch.setattr(recurrent_operator, "inference", counted_inference)
    monkeypatch.setattr(recurrent_operator, "training", counted_training)
    final_voltages = []
    with (
        environ.context(dt=0.28),
        torch.autograd.graph.saved_tensors_hooks(
            lambda value: saved.append(value) or value, lambda value: value
        ),
        torch.inference_mode(),
    ):
        for source in sources:
            actual_output, actual_records = actual(source)
            expected_output, expected_records = expected(source)
            torch.testing.assert_close(
                actual_output, expected_output, atol=5e-6, rtol=5e-5
            )
            assert actual_records == expected_records == {}
            for actual_state, expected_state in zip(
                _final_states(actual), _final_states(expected), strict=True
            ):
                torch.testing.assert_close(
                    actual_state, expected_state, atol=5e-6, rtol=5e-5
                )
            final_voltages.append(actual.neuron.v.clone())

    temporal_allocations = [
        shape
        for shape in allocations.shapes
        if len(shape) >= 3 and shape[:3] == (5, 2, 4)
    ]
    assert temporal_allocations == []
    assert saved == []
    assert operator_calls == {"inference": 2, "training": 0}
    assert not torch.equal(final_voltages[0], final_voltages[1])
