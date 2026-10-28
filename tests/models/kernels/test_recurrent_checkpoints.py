"""Validate the first native recurrent physical-state checkpoint layout."""

import copy
import json
from contextlib import nullcontext

import pytest
import torch

from btorch.models import BPTT, environ
from btorch.models.connection import SparseConnection
from btorch.models.kernels import KernelOptions
from btorch.models.kernels.contracts import NeuronSpec
from btorch.models.kernels.operators import encode_spec
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import Sigmoid, Truncated
from btorch.models.synapse import ExponentialPSC
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


def _model(backend, device="cpu", requested=(), *, pre_spike_v=False):
    linear = torch.nn.Linear(4, 4, bias=True)
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [
                    [0.08, -0.03, 0.02, 0.01],
                    [0.01, 0.07, -0.02, 0.03],
                    [-0.02, 0.01, 0.09, 0.02],
                    [0.03, -0.01, 0.02, 0.08],
                ]
            )
        )
        linear.bias.copy_(torch.tensor([0.01, -0.02, 0.015, 0.0]))
    model = RecurrentNN(
        LIF(
            4,
            v_threshold=0.52,
            v_reset=-0.08,
            c_m=1.15,
            tau=7.0,
            hard_reset=True,
            trainable_param={"v_threshold", "v_reset", "c_m", "tau"},
            pre_spike_v=pre_spike_v,
        ),
        ExponentialPSC(4, 3.5, linear),
        backend=KernelOptions(neuron=backend),
        update_state_names=requested,
    ).to(device=device, dtype=torch.float32)
    model.neuron.init_state(batch_size=2, device=device, dtype=torch.float32)
    model.synapse.init_state(batch_size=2, device=device, dtype=torch.float32)
    with torch.no_grad():
        model.neuron.v.copy_(
            torch.tensor(
                [[0.12, -0.04, 0.31, 0.08], [0.22, 0.02, -0.06, 0.27]],
                device=device,
            )
        )
        model.synapse.psc.copy_(
            torch.tensor(
                [[0.03, -0.02, 0.05, 0.01], [-0.01, 0.04, 0.02, -0.03]],
                device=device,
            )
        )
    model.neuron.v.requires_grad_()
    model.synapse.psc.requires_grad_()
    # ExponentialPSC declares tau_syn as a buffer, but the tensor operator
    # contract still supports its VJP when a caller makes that tensor trainable.
    model.synapse.tau_syn.requires_grad_()
    return model


def _inputs(steps, device):
    generator = torch.Generator(device=device).manual_seed(1700 + steps)
    x = (
        torch.rand((steps, 2, 4), generator=generator, device=device) * 0.95 - 0.12
    ).requires_grad_()
    x_syn = (
        torch.rand((steps, 2, 4), generator=generator, device=device) * 0.18 - 0.04
    ).requires_grad_()
    return x, x_syn


def _gradient_inputs(model, x1, x_syn1, x2, x_syn2):
    return (
        x1,
        x_syn1,
        x2,
        x_syn2,
        model.neuron.v,
        model.synapse.psc,
        model.synapse.tau_syn,
        *model.parameters(),
    )


def _run_two(model, x1, x_syn1, x2, x_syn2):
    first_output, first_records = model(x1, x_syn1)
    first_states = (model.neuron.v, model.synapse.psc)
    second_output, second_records = model(x2, x_syn2)
    assert first_records == second_records == {}
    return (
        first_output,
        *first_states,
        second_output,
        model.neuron.v,
        model.synapse.psc,
    )


def _assert_two_forward_vjp(backend, steps, device):
    torch.manual_seed(1601)
    model = _model(backend, device)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x1, x_syn1 = _inputs(steps, device)
    x2, x_syn2 = _inputs(3, device)
    xr1 = x1.detach().clone().requires_grad_()
    xsr1 = x_syn1.detach().clone().requires_grad_()
    xr2 = x2.detach().clone().requires_grad_()
    xsr2 = x_syn2.detach().clone().requires_grad_()
    actual_inputs = _gradient_inputs(model, x1, x_syn1, x2, x_syn2)
    expected_inputs = _gradient_inputs(reference, xr1, xsr1, xr2, xsr2)

    with environ.context(dt=0.5):
        actual = _run_two(model, x1, x_syn1, x2, x_syn2)
        expected = _run_two(reference, xr1, xsr1, xr2, xsr2)

    # Spike rasters are the recurrent connection source, so event equality is
    # stricter than the continuous-state tolerance used below.
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)
    for value, target in zip(
        (*actual[1:3], *actual[4:]), (*expected[1:3], *expected[4:])
    ):
        torch.testing.assert_close(value, target, atol=5e-6, rtol=5e-5)

    generator = torch.Generator(device=device).manual_seed(1802 + steps)
    cotangents = tuple(
        torch.randn(value.shape, generator=generator, device=device) for value in actual
    )
    actual_gradients = torch.autograd.grad(
        actual, actual_inputs, cotangents, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected, expected_inputs, cotangents, allow_unused=True
    )
    for value, target, source in zip(
        actual_gradients, expected_gradients, actual_inputs
    ):
        value = torch.zeros_like(source) if value is None else value
        target = torch.zeros_like(source) if target is None else target
        torch.testing.assert_close(value, target, atol=7e-5, rtol=7e-4)


def _assert_values_and_vjp(
    model, reference, x, x_syn, xr, xsr, requested, final_names=(), saved=None
):
    actual_inputs = (
        x,
        x_syn,
        model.neuron.v,
        model.synapse.psc,
        model.synapse.tau_syn,
        *model.parameters(),
    )
    expected_inputs = (
        xr,
        xsr,
        reference.neuron.v,
        reference.synapse.psc,
        reference.synapse.tau_syn,
        *reference.parameters(),
    )
    with environ.context(dt=0.5):
        if saved is None:
            output, records = model(x, x_syn)
        else:
            with torch.autograd.graph.saved_tensors_hooks(
                lambda value: saved.append(value) or value, lambda value: value
            ):
                output, records = model(x, x_syn)
        expected_output, expected_records = reference(xr, xsr)
    actual = (
        output,
        *(records[name] for name in requested),
        *(getattr(model.neuron, name) for name in final_names),
        model.neuron.v,
        model.synapse.psc,
    )
    expected = (
        expected_output,
        *(expected_records[name] for name in requested),
        *(getattr(reference.neuron, name) for name in final_names),
        reference.neuron.v,
        reference.synapse.psc,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    for value, target in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(value, target, atol=5e-6, rtol=5e-5)

    generator = torch.Generator(device=x.device).manual_seed(1917 + len(requested))
    cotangents = tuple(
        torch.randn(value.shape, generator=generator, device=x.device)
        for value in actual
    )
    actual_gradients = torch.autograd.grad(
        actual, actual_inputs, cotangents, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected, expected_inputs, cotangents, allow_unused=True
    )
    for value, target, source in zip(
        actual_gradients, expected_gradients, actual_inputs
    ):
        value = torch.zeros_like(source) if value is None else value
        target = torch.zeros_like(source) if target is None else target
        torch.testing.assert_close(value, target, atol=7e-5, rtol=7e-4)
    return actual, records


@pytest.mark.parametrize("steps", [1, 16, 17, 33])
def test_cpu_reference_fixture_covers_checkpoint_boundaries(reference_backend, steps):
    """Keep the checkpoint parity fixture grounded in the actual Modules."""
    _assert_two_forward_vjp(reference_backend, steps, "cpu")


@pytest.mark.parametrize(
    "steps,expected_checkpoints", [(1, 0), (16, 0), (17, 1), (33, 2)]
)
def test_checkpoint_fake_contract_uses_staggered_block_storage(
    steps, expected_checkpoints
):
    """The private ABI scales by block count and is absent without its plan."""
    from btorch.models.kernels.recurrent_operator import _fake

    states = [torch.empty(2, 4), torch.empty(2, 4)]
    spec = {
        "neuron": encode_spec(NeuronSpec("lif", dt=0.5, hard_reset=True)),
        "synapse": encode_spec(NeuronSpec("exponential", dt=0.5)),
        "charged_names": [],
        "paired_checkpoint_interval": 16,
    }
    x = torch.empty(steps, 2, 4)
    outputs = _fake(
        x, x, states, [], [], json.dumps(spec), [], "triton", torch.tensor(0.5)
    )
    assert len(outputs) == 5
    assert [tuple(value.shape) for value in outputs[-2:]] == [
        (expected_checkpoints, 2, 4),
        (expected_checkpoints, 2, 4),
    ]

    inference_spec = dict(spec)
    del inference_spec["paired_checkpoint_interval"]
    inference_outputs = _fake(
        x,
        x,
        states,
        [],
        [],
        json.dumps(inference_spec),
        [],
        "triton",
        torch.tensor(0.5),
    )
    assert len(inference_outputs) == 3


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("steps", [1, 16, 17, 33])
def test_native_checkpointed_recurrence_matches_module_vjp(backend, steps):
    require_cuda_backend(backend)
    _assert_two_forward_vjp(backend, steps, "cuda")


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_native_checkpoint_saved_tensors_omit_full_physical_histories(backend):
    require_cuda_backend(backend)
    model = _model(backend, "cuda")
    x, x_syn = _inputs(33, "cuda")
    saved = []

    def pack(value):
        saved.append(value)
        return value

    # Inputs and the connection spike raster legitimately remain T-sized. The
    # membrane and PSC residuals are the two interval-sized tensors below.
    with (
        environ.context(dt=0.5),
        torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value),
    ):
        output, records = model(x, x_syn)
        assert records == {}
        loss = output.square().sum() + model.neuron.v.sum() + model.synapse.psc.sum()
    loss.backward()

    # The scalar loss saves the spike output once more, so count storage rather
    # than hook calls. No private T-sized physical history may own new storage.
    sequence_storages = {
        value.untyped_storage().data_ptr()
        for value in saved
        if tuple(value.shape) == (33, 2, 4)
    }
    permitted_sequence_storages = {
        value.untyped_storage().data_ptr() for value in (x, x_syn, output)
    }
    checkpoint_storages = {
        value.untyped_storage().data_ptr()
        for value in saved
        if tuple(value.shape) == (2, 2, 4)
    }
    assert sequence_storages <= permitted_sequence_storages
    assert len(checkpoint_storages) == 2


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize(
    "requested",
    [("neuron.v",), ("synapse.psc",), ("neuron.v", "synapse.psc")],
)
def test_public_physical_records_use_full_history_fallback(backend, requested):
    require_cuda_backend(backend)
    model = _model(backend, "cuda", requested)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x, x_syn = _inputs(17, "cuda")
    xr = x.detach().clone().requires_grad_()
    xsr = x_syn.detach().clone().requires_grad_()
    saved = []

    # A requested physical trajectory is public output. The implementation
    # deliberately falls back to the established full-history reverse path.
    _, records = _assert_values_and_vjp(
        model, reference, x, x_syn, xr, xsr, requested, saved=saved
    )

    assert set(records) == set(requested)
    assert all(records[name].shape == (17, 2, 4) for name in requested)
    assert all(tuple(value.shape) != (1, 2, 4) for value in saved)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_charged_record_keeps_checkpointed_physical_reverse(backend):
    require_cuda_backend(backend)
    requested = ("neuron.v_pre_spike",)
    model = _model(backend, "cuda", requested, pre_spike_v=True)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x, x_syn = _inputs(17, "cuda")
    xr = x.detach().clone().requires_grad_()
    xsr = x_syn.detach().clone().requires_grad_()
    saved = []

    actual, records = _assert_values_and_vjp(
        model,
        reference,
        x,
        x_syn,
        xr,
        xsr,
        requested,
        final_names=("v_pre_spike",),
        saved=saved,
    )

    torch.testing.assert_close(model.neuron.v_pre_spike, records[requested[0]][-1])
    permitted_sequence_storages = {
        value.untyped_storage().data_ptr()
        for value in (x, x_syn, actual[0], records[requested[0]])
    }
    sequence_storages = {
        value.untyped_storage().data_ptr()
        for value in saved
        if tuple(value.shape) == (17, 2, 4)
    }
    checkpoint_storages = {
        value.untyped_storage().data_ptr()
        for value in saved
        if tuple(value.shape) == (1, 2, 4)
    }
    assert sequence_storages <= permitted_sequence_storages
    assert len(checkpoint_storages) == 2


@pytest.mark.parametrize(
    "backend,amp",
    (("triton", False), ("cupy", False), ("tilelang", False), ("triton", True)),
)
def test_compiled_checkpoint_replays_completed_training_steps(
    backend, amp, monkeypatch
):
    require_cuda_backend(backend)
    model = _model(backend, "cuda")
    sparse = backend == "triton"
    if sparse:
        # The two existing Triton cases cover auto dense and explicit push
        # instead of multiplying the backend/AMP matrix. External synaptic input
        # is omitted below: its VJP requires the full presynaptic adjoint.
        linear = model.synapse.linear
        post, pre = linear.weight.detach().nonzero(as_tuple=True)
        model.synapse.linear = SparseConnection.from_edges(
            pre,
            post,
            4,
            4,
            values=linear.weight.detach()[post, pre].clone(),
            bias=linear.bias.detach(),
        )
        model.synapse.linear.weight.value.requires_grad_()
        model.neuron.surrogate_function = Truncated(Sigmoid(alpha=2.0), 0.4)
        model.gradient = BPTT(sparse=True)
        model.backend = KernelOptions(
            neuron="triton",
            connection="triton" if amp else "auto",
            connection_method="push" if amp else "auto",
        )
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    base_x, base_x_syn = _inputs(17, "cuda")
    if sparse:
        # A single sample exercises the joint list-driven weight/input VJP.
        # Keep sliced (noncontiguous) input storage while matching leaf states.
        base_x = base_x[:, :1].detach().requires_grad_()
        base_x_syn = base_x_syn[:, :1].detach().requires_grad_()
        for network in (model, reference):
            network.neuron.v = network.neuron.v[:1].detach().clone().requires_grad_()
            network.synapse.psc = (
                network.synapse.psc[:1].detach().clone().requires_grad_()
            )
    trials = (
        (base_x, base_x_syn),
        (
            (base_x.detach() * 0.71 + 0.09).requires_grad_(),
            (base_x_syn.detach() * -0.43 + 0.02).requires_grad_(),
        ),
        (
            (base_x.detach() * -0.37 + 0.31).requires_grad_(),
            (base_x_syn.detach() * 0.58 - 0.01).requires_grad_(),
        ),
        (
            (base_x.detach() * 0.43 + 0.12).requires_grad_(),
            (base_x_syn.detach() * 0.36 + 0.01).requires_grad_(),
        ),
    )
    external_v = model.neuron.v.detach().clone().requires_grad_()
    external_psc = model.synapse.psc.detach().clone().requires_grad_()
    reference_v = reference.neuron.v.detach().clone().requires_grad_()
    reference_psc = reference.synapse.psc.detach().clone().requires_grad_()
    carried_v = external_v.detach().clone()
    carried_psc = external_psc.detach().clone()
    reference_carried_v = reference_v.detach().clone()
    reference_carried_psc = reference_psc.detach().clone()

    replay_calls = []
    original_replay = torch.cuda.CUDAGraph.replay

    def counted_replay(graph):
        replay_calls.append(graph)
        return original_replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted_replay)

    def loss(value, synaptic_value):
        output, records = model(value, None if sparse else synaptic_value)
        assert records == {}
        return (
            output.square().sum()
            + model.neuron.v.square().sum()
            + model.synapse.psc.square().sum()
        )

    compiled_loss = torch.compile(loss, mode="reduce-overhead", fullgraph=True)
    replay_deltas = []
    for x, x_syn in trials:
        # Each iteration is one completed truncated-BPTT training step. Copy
        # only numerical state into stable leaf buffers before announcing the
        # next CUDA Graph Tree step; the previous graph has already completed
        # backward and no cross-step adjoint is intended here.
        with torch.no_grad():
            external_v.copy_(carried_v)
            external_psc.copy_(carried_psc)
            reference_v.copy_(reference_carried_v)
            reference_psc.copy_(reference_carried_psc)
        model.neuron.v = external_v
        model.synapse.psc = external_psc
        reference.neuron.v = reference_v
        reference.synapse.psc = reference_psc
        xr = x.detach().clone().requires_grad_()
        xsr = x_syn.detach().clone().requires_grad_()
        actual_inputs = (
            x,
            x_syn,
            model.neuron.v,
            model.synapse.psc,
            model.synapse.tau_syn,
            *model.parameters(),
            *((model.synapse.linear.weight.value,) if sparse else ()),
        )
        expected_inputs = (
            xr,
            xsr,
            reference.neuron.v,
            reference.synapse.psc,
            reference.synapse.tau_syn,
            *reference.parameters(),
            *((reference.synapse.linear.weight.value,) if sparse else ()),
        )
        before_replay = len(replay_calls)
        torch.compiler.cudagraph_mark_step_begin()
        # Recompute and reverse run after this autocast context exits. The AMP
        # case must replay the captured low-precision weight casts as well.
        with (
            environ.context(dt=0.5),
            torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext(),
        ):
            actual_loss = compiled_loss(x, x_syn)
            expected_output, expected_records = reference(xr, None if sparse else xsr)
            expected_loss = (
                expected_output.square().sum()
                + reference.neuron.v.square().sum()
                + reference.synapse.psc.square().sum()
            )
        assert expected_records == {}
        actual_gradients = torch.autograd.grad(
            actual_loss, actual_inputs, allow_unused=True
        )
        replay_deltas.append(len(replay_calls) - before_replay)
        expected_gradients = torch.autograd.grad(
            expected_loss, expected_inputs, allow_unused=True
        )
        torch.testing.assert_close(actual_loss, expected_loss, atol=5e-6, rtol=5e-5)
        torch.testing.assert_close(
            model.neuron.v, reference.neuron.v, atol=5e-6, rtol=5e-5
        )
        torch.testing.assert_close(
            model.synapse.psc, reference.synapse.psc, atol=5e-6, rtol=5e-5
        )
        for value, target, source in zip(
            actual_gradients, expected_gradients, actual_inputs
        ):
            value = torch.zeros_like(source) if value is None else value
            target = torch.zeros_like(source) if target is None else target
            # Only the new AMP variant admits BF16 GEMM reduction rounding;
            # existing FP32 cases keep their original VJP tolerances.
            atol, rtol = (2e-3, 4e-2) if amp else (7e-5, 7e-4)
            torch.testing.assert_close(value, target, atol=atol, rtol=rtol)
        carried_v = model.neuron.v.detach().clone()
        carried_psc = model.synapse.psc.detach().clone()
        reference_carried_v = reference.neuron.v.detach().clone()
        reference_carried_psc = reference.synapse.psc.detach().clone()
        if sparse:
            # Captured kernels must consume current parameters after an
            # optimizer-style update, never a stale prepared value snapshot.
            with torch.no_grad():
                model.synapse.linear.weight.value.mul_(0.99)
                reference.synapse.linear.weight.value.mul_(0.99)
        del actual_gradients, actual_inputs, actual_loss, source, target, value

    # Warmup/capture timing is Torch-owned and can differ with a warm compile
    # cache. Once settled, changed-input steps must replay both forward and
    # backward while preserving the numerical/gradient checks above.
    assert replay_deltas[-2:] == [2, 2]


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_frozen_eval_model_checkpoints_trainable_encoder(backend):
    require_cuda_backend(backend)
    model = _model(backend, "cuda")
    model.eval().requires_grad_(False)
    model.neuron.v.requires_grad_(False)
    model.synapse.psc.requires_grad_(False)
    model.synapse.tau_syn.requires_grad_(False)
    reference = copy.deepcopy(model)
    reference.backend = "torch"

    encoder = torch.nn.Linear(4, 4, bias=False, device="cuda")
    with torch.no_grad():
        encoder.weight.copy_(
            torch.tensor(
                [
                    [0.71, -0.08, 0.05, 0.02],
                    [0.03, 0.64, -0.04, 0.07],
                    [-0.06, 0.02, 0.76, 0.04],
                    [0.08, -0.03, 0.01, 0.69],
                ],
                device="cuda",
            )
        )
    reference_encoder = copy.deepcopy(encoder)
    raw, x_syn = _inputs(33, "cuda")
    raw = raw.detach()
    x_syn = x_syn.detach()
    x = encoder(raw)
    xr = reference_encoder(raw)
    assert x.grad_fn is not None and not x.is_leaf
    saved = []

    with environ.context(dt=0.5):
        with torch.autograd.graph.saved_tensors_hooks(
            lambda value: saved.append(value) or value, lambda value: value
        ):
            output, records = model(x, x_syn)
        expected_output, expected_records = reference(xr, x_syn)
    assert records == expected_records == {}
    torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
    torch.testing.assert_close(model.neuron.v, reference.neuron.v, atol=5e-6, rtol=5e-5)
    torch.testing.assert_close(
        model.synapse.psc, reference.synapse.psc, atol=5e-6, rtol=5e-5
    )

    generator = torch.Generator(device="cuda").manual_seed(2033)
    cotangents = tuple(
        torch.randn(value.shape, generator=generator, device="cuda")
        for value in (output, model.neuron.v, model.synapse.psc)
    )
    loss = sum(
        (value * cotangent).sum()
        for value, cotangent in zip(
            (output, model.neuron.v, model.synapse.psc), cotangents
        )
    )
    expected_loss = sum(
        (value * cotangent).sum()
        for value, cotangent in zip(
            (expected_output, reference.neuron.v, reference.synapse.psc), cotangents
        )
    )
    gradient = torch.autograd.grad(loss, encoder.weight)[0]
    expected_gradient = torch.autograd.grad(expected_loss, reference_encoder.weight)[0]
    torch.testing.assert_close(gradient, expected_gradient, atol=7e-5, rtol=7e-4)

    sequence_storages = {
        value.untyped_storage().data_ptr()
        for value in saved
        if tuple(value.shape) == (33, 2, 4)
    }
    permitted_sequence_storages = {
        value.untyped_storage().data_ptr() for value in (x, x_syn, output)
    }
    checkpoint_storages = {
        value.untyped_storage().data_ptr()
        for value in saved
        if tuple(value.shape) == (2, 2, 4)
    }
    assert sequence_storages <= permitted_sequence_storages
    assert len(checkpoint_storages) == 2
    assert all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_native_inference_has_no_checkpoint_outputs_or_autograd_tape(
    backend, monkeypatch
):
    require_cuda_backend(backend)
    from btorch.models.kernels import recurrent_operator

    model = _model(backend, "cuda")
    model.requires_grad_(False)
    model.neuron.v.requires_grad_(False)
    model.synapse.psc.requires_grad_(False)
    model.synapse.tau_syn.requires_grad_(False)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x, x_syn = _inputs(17, "cuda")
    x.requires_grad_(False)
    x_syn.requires_grad_(False)
    observed = []
    saved = []
    trajectory = recurrent_operator._trajectory

    def instrumented_trajectory(*args, **kwargs):
        result = trajectory(*args, **kwargs)
        spec = json.loads(args[5])
        observed.append(("paired_checkpoint_interval" in spec, len(result)))
        return result

    monkeypatch.setattr(recurrent_operator, "_trajectory", instrumented_trajectory)
    with (
        environ.context(dt=0.5),
        torch.autograd.graph.saved_tensors_hooks(
            lambda value: saved.append(value) or value, lambda value: value
        ),
        torch.inference_mode(),
    ):
        actual, records = model(x, x_syn)
        expected, expected_records = reference(x, x_syn)

    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
    torch.testing.assert_close(model.neuron.v, reference.neuron.v, atol=5e-6, rtol=5e-5)
    torch.testing.assert_close(
        model.synapse.psc, reference.synapse.psc, atol=5e-6, rtol=5e-5
    )
    assert records == expected_records == {}
    assert observed == [(False, 3)]
    assert saved == []


@pytest.mark.parametrize(
    "backend,dtype,execution",
    (
        ("triton", torch.float16, "eager"),
        ("triton", torch.bfloat16, "compile"),
        ("triton", torch.bfloat16, "offload"),
        ("cupy", torch.float16, "eager"),
        ("tilelang", torch.bfloat16, "eager"),
    ),
)
def test_autocast_checkpoint_training_matches_module(backend, dtype, execution):
    """AMP preserves physical state and scaled training across replay blocks.

    Use actual Torch Modules under the same autocast policy as the
    oracle. Backward deliberately runs outside autocast, as recommended
    for AMP; saved connection operands must retain the forward
    precision. The longer compiled case crosses two native checkpoint
    boundaries. Input is a sliced view, while master parameters and
    physical memories remain FP32.
    """
    require_cuda_backend(backend)
    torch.manual_seed(2134)
    model = _model(backend, "cuda")
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    if execution == "offload":
        # Exercise the existing chunk/checkpoint/offload composition under
        # autocast, including restoration during backward outside its context.
        model.chunk_size = 8
        model.grad_checkpoint = True
        model.cpu_offload = True
    steps = 33 if execution == "compile" else 17
    padded, events = _inputs(steps * 2, "cuda")
    x = padded.detach()[1::2].requires_grad_()
    events = events.detach()[1::2].requires_grad_()
    xr = x.detach().clone().requires_grad_()
    er = events.detach().clone().requires_grad_()
    roots = (
        x,
        events,
        model.neuron.v,
        model.synapse.psc,
        model.synapse.tau_syn,
        *model.parameters(),
    )
    reference_roots = (
        xr,
        er,
        reference.neuron.v,
        reference.synapse.psc,
        reference.synapse.tau_syn,
        *reference.parameters(),
    )
    execute = torch.compile(model, fullgraph=True) if execution == "compile" else model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=dtype == torch.float16, init_scale=128.0
    )
    reference_scaler = torch.amp.GradScaler(
        "cuda", enabled=dtype == torch.float16, init_scale=128.0
    )
    with environ.context(dt=0.5), torch.autocast("cuda", dtype=dtype):
        output, _ = execute(x, events)
        target, _ = reference(xr, er)
        actual = (output, model.neuron.v, model.synapse.psc)
        expected = (target, reference.neuron.v, reference.synapse.psc)
        loss = sum(value.square().mean() for value in actual)
        reference_loss = sum(value.square().mean() for value in expected)
    # AMP quantizes GEMM operands; native scalar equations still use FP32.
    # Keep the strict event comparison and allow only continuous rounding.
    torch.testing.assert_close(output.to(target.device), target, atol=0, rtol=0)
    for value, wanted in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(value, wanted, atol=2e-5, rtol=2e-4)
    scaler.scale(loss).backward()
    reference_scaler.scale(reference_loss).backward()
    scale = scaler.get_scale()
    # Compare all incoming-memory, input and master-parameter VJPs before
    # unscaling: GradScaler.unscale_ only operates on optimizer parameters.
    atol, rtol = (2e-4, 5e-3) if dtype == torch.float16 else (2e-3, 4e-2)
    for root, reference_root in zip(roots, reference_roots):
        assert root.grad is not None and reference_root.grad is not None
        torch.testing.assert_close(
            root.grad / scale, reference_root.grad / scale, atol=atol, rtol=rtol
        )
    parameter_values = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    reference_parameter_values = tuple(
        parameter.detach().clone() for parameter in reference.parameters()
    )
    scaler.step(optimizer)
    scaler.update()
    reference_scaler.step(reference_optimizer)
    reference_scaler.update()
    # Compare effective SGD updates in gradient units. Direct parameter
    # comparison is unstable when BF16 cancellation leaves a near-zero value.
    learning_rate = optimizer.param_groups[0]["lr"]
    for (
        parameter,
        wanted,
        initial,
        reference_initial,
    ) in zip(
        model.parameters(),
        reference.parameters(),
        parameter_values,
        reference_parameter_values,
        strict=True,
    ):
        update = (initial - parameter) / learning_rate
        reference_update = (reference_initial - wanted) / learning_rate
        torch.testing.assert_close(update, reference_update, atol=atol, rtol=rtol)


@torch.no_grad()
def test_autocast_cudagraph_reselects_precision_and_replays_changed_input():
    """Changing AMP policy must not replay kernels captured at another dtype.

    Keep input/master-weight dtype and shape fixed, switching only
    autocast. Revisit FP16 and disabled AMP with fresh data to exercise
    existing graph entries as well as first capture. Compare physical
    states, not just spikes.
    """
    from btorch.models.functional import named_hidden_states, set_hidden_states

    require_cuda_backend("triton")
    model = _model("triton", "cuda")
    model.cudagraph = True
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    reference.cudagraph = False
    initial = {
        name: value.detach().clone()
        for name, value in named_hidden_states(model).items()
    }
    x, events = _inputs(7, "cuda")
    for index, dtype in enumerate(
        (None, torch.float16, torch.bfloat16, torch.float16, None)
    ):
        for network in (model, reference):
            set_hidden_states(
                network, {name: value.clone() for name, value in initial.items()}
            )
        with (
            environ.context(dt=0.5),
            torch.autocast(
                "cuda", dtype=dtype or torch.float16, enabled=dtype is not None
            ),
        ):
            output, _ = model(x + index * 0.017, events)
            expected, _ = reference(x + index * 0.017, events)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            model.neuron.v, reference.neuron.v, atol=2e-5, rtol=2e-4
        )
        torch.testing.assert_close(
            model.synapse.psc, reference.synapse.psc, atol=2e-5, rtol=2e-4
        )
