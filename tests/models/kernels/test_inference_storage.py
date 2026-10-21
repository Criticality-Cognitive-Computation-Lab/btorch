"""Verify recurrent inference retains only public temporal records."""

import copy
from contextlib import nullcontext

import pytest
import torch

from tests.models.kernels.helpers import (
    GPU_BACKENDS,
    NewEmptyShapes,
    require_cuda_backend,
)


@pytest.mark.parametrize("context", [nullcontext, torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("neuron_model", ["lif", "izhikevich"])
@pytest.mark.parametrize("request_charged_history", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_recurrent_inference_keeps_only_requested_charged_trajectories(
    context, neuron_model, request_charged_history, compiled, monkeypatch
):
    from btorch.models import environ
    from btorch.models.kernels import KernelOptions
    from btorch.models.neurons.izhikevich import Izhikevich
    from btorch.models.neurons.lif import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import AlphaPSC

    torch.manual_seed(820)
    if neuron_model == "lif":
        neuron = LIF(4, v_threshold=0.45, pre_spike_v=True)
        charged_names = ("neuron.v_pre_spike",)
    else:
        neuron = Izhikevich(
            4,
            v_threshold=0.45,
            v_peak=0.8,
            pre_spike=True,
        )
        charged_names = ("neuron.v_pre_spike", "neuron.u_pre_spike")
    requested = charged_names[:1] if request_charged_history else ()
    model = RecurrentNN(
        neuron,
        AlphaPSC(4, 2.0, torch.nn.Identity()),
        backend=KernelOptions(),
        update_state_names=requested,
    ).float()
    model.neuron.init_state(batch_size=2, dtype=torch.float32)
    model.synapse.init_state(batch_size=2, dtype=torch.float32)
    model.requires_grad_(False)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = torch.rand(6, 2, 4) * 0.7
    if compiled:
        # Each parametrized case specializes record count and grad mode. Keep
        # those fullgraph checks from consuming the process-wide recompile cache
        # used by unrelated compiler tests in the same pytest worker.
        torch._dynamo.reset()
    runner = (
        model
        if not compiled
        else torch.compile(model, backend="aot_eager", fullgraph=True)
    )

    allocations = NewEmptyShapes()
    if not compiled:
        from btorch.models.kernels import recurrent_operator

        trajectory = recurrent_operator._trajectory

        def instrumented_trajectory(*args, **kwargs):
            with allocations:
                return trajectory(*args, **kwargs)

        monkeypatch.setattr(recurrent_operator, "_trajectory", instrumented_trajectory)
    with environ.context(dt=1.0), context():
        actual, records = runner(x)
        expected, expected_records = reference(x)

    torch.testing.assert_close(actual, expected)
    assert tuple(records) == tuple(expected_records) == requested
    for name in requested:
        torch.testing.assert_close(records[name], expected_records[name])
    torch.testing.assert_close(model.neuron.v_pre_spike, reference.neuron.v_pre_spike)
    if neuron_model == "izhikevich":
        torch.testing.assert_close(
            model.neuron.u_pre_spike, reference.neuron.u_pre_spike
        )
    if not request_charged_history:
        # The O(1) charged outputs cross the custom-op boundary as independent
        # tensors, not aliases of the input, spike output, or reset state.
        occupied = {
            x.untyped_storage().data_ptr(),
            actual.untyped_storage().data_ptr(),
            model.neuron.v.untyped_storage().data_ptr(),
        }
        assert model.neuron.v_pre_spike.untyped_storage().data_ptr() not in occupied
        if neuron_model == "izhikevich":
            assert (
                model.neuron.u_pre_spike.untyped_storage().data_ptr()
                != model.neuron.u.untyped_storage().data_ptr()
            )

    # The recurrent operator allocates one T-length ``new_empty`` tensor per
    # public record. Final pre-spike module state is an O(1) output when its
    # trajectory is not requested, so it must not add another T-length buffer.
    if not compiled:
        trajectory_shape = tuple(x.shape)
        assert allocations.shapes.count(trajectory_shape) == len(requested)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("request_history", [False, True])
def test_gpu_recurrent_inference_returns_final_charged_state_without_hidden_record(
    backend, request_history
):
    require_cuda_backend(backend)

    from btorch.models import environ
    from btorch.models.kernels import KernelOptions
    from btorch.models.neurons.lif import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import AlphaPSC

    torch.manual_seed(821)
    requested = ("neuron.v_pre_spike",) if request_history else ()
    model = RecurrentNN(
        LIF(4, v_threshold=0.45, pre_spike_v=True),
        AlphaPSC(4, 2.0, torch.nn.Identity()),
        backend=KernelOptions(neuron=backend),
        update_state_names=requested,
    ).cuda()
    model.neuron.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    model.synapse.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    model.requires_grad_(False)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = torch.rand(6, 2, 4, device="cuda") * 0.7

    with environ.context(dt=1.0), torch.inference_mode():
        actual, records = model(x)
        expected, expected_records = reference(x)

    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
    torch.testing.assert_close(
        model.neuron.v_pre_spike,
        reference.neuron.v_pre_spike,
        atol=5e-6,
        rtol=5e-5,
    )
    assert tuple(records) == requested
    if request_history:
        torch.testing.assert_close(
            records[requested[0]], expected_records[requested[0]], atol=5e-6, rtol=5e-5
        )
        torch.testing.assert_close(model.neuron.v_pre_spike, records[requested[0]][-1])


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("request_g_rise", [False, True])
def test_dual_exponential_native_inference_keeps_only_public_state_history(
    backend, request_g_rise, monkeypatch
):
    """Dual-exponential inference keeps final physics without reverse tape.

    The input pulse starts after timestep zero so continuation cannot
    pass by replaying only initialization. The second captured call
    changes the input while carrying every recurrent state from the
    first call.
    """
    require_cuda_backend(backend)

    from btorch.models import environ
    from btorch.models.functional import init_net_state
    from btorch.models.kernels import KernelOptions, recurrent_operator
    from btorch.models.neurons.lif import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import DualExponentialPSC

    T, B, N = 9, 2, 4
    requested = ("synapse.g_rise",) if request_g_rise else ()

    def build(*, native=True, cudagraph=False):
        model = RecurrentNN(
            LIF(
                N,
                v_threshold=0.45,
                v_reset=0.0,
                c_m=1.0,
                tau=2.0,
                hard_reset=True,
            ),
            DualExponentialPSC(N, 5.0, 2.0, torch.nn.Identity(), A=1.0),
            backend=KernelOptions(neuron=backend) if native else "torch",
            update_state_names=requested,
            cudagraph=cudagraph,
        ).to(device="cuda", dtype=torch.float32)
        init_net_state(model, batch_size=B, device="cuda", dtype=torch.float32)
        lane = torch.arange(B * N, device="cuda", dtype=torch.float32).reshape(B, N)
        model.neuron.v = -0.08 + 0.025 * lane
        model.synapse.psc = 0.08 + 0.01 * lane
        model.synapse.g_rise = -0.04 + 0.008 * lane
        model.synapse.g_decay = 0.11 - 0.006 * lane
        model.requires_grad_(False)
        return model

    def assert_run(actual, expected, actual_model, expected_model):
        actual_output, actual_records = actual
        expected_output, expected_records = expected
        torch.testing.assert_close(actual_output, expected_output, atol=5e-6, rtol=5e-5)
        assert tuple(actual_records) == tuple(expected_records) == requested
        for name in requested:
            torch.testing.assert_close(
                actual_records[name], expected_records[name], atol=5e-6, rtol=5e-5
            )
        for name in ("v",):
            torch.testing.assert_close(
                getattr(actual_model.neuron, name),
                getattr(expected_model.neuron, name),
                atol=5e-6,
                rtol=5e-5,
            )
        for name in ("psc", "g_rise", "g_decay"):
            torch.testing.assert_close(
                getattr(actual_model.synapse, name),
                getattr(expected_model.synapse, name),
                atol=5e-6,
                rtol=5e-5,
            )
        if request_g_rise:
            torch.testing.assert_close(
                actual_model.synapse.g_rise,
                actual_records["synapse.g_rise"][-1],
                atol=5e-6,
                rtol=5e-5,
            )

    x = torch.zeros(T, B, N, device="cuda", dtype=torch.float32)
    x[2] = torch.tensor([1.4, 1.1, 0.8, 1.3], device="cuda")
    x[5] = torch.tensor([0.2, 1.2, 0.9, 0.4], device="cuda")
    x[7] = 0.55
    actual_model, expected_model = build(), build(native=False)
    allocations = NewEmptyShapes()
    saved = []
    trajectory = recurrent_operator._trajectory

    def instrumented_trajectory(*args, **kwargs):
        with allocations:
            return trajectory(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(recurrent_operator, "_trajectory", instrumented_trajectory)
        with (
            environ.context(dt=1.0),
            torch.autograd.graph.saved_tensors_hooks(
                lambda value: saved.append(value) or value, lambda value: value
            ),
            torch.inference_mode(),
        ):
            actual = actual_model(x)
    with environ.context(dt=1.0), torch.inference_mode():
        expected = expected_model(x)
    assert_run(actual, expected, actual_model, expected_model)
    assert saved == []
    assert allocations.shapes.count(tuple(x.shape)) == len(requested)
    assert 0 < int(actual[0].sum()) < actual[0].numel()

    # A split call exercises registered state continuation independently of the
    # whole-sequence comparison above. No state is copied between chunks.
    actual_model, expected_model = build(), build(native=False)
    split = 4
    with environ.context(dt=1.0), torch.inference_mode():
        a1, ar1 = actual_model(x[:split])
        a2, ar2 = actual_model(x[split:])
        e1, er1 = expected_model(x[:split])
        e2, er2 = expected_model(x[split:])
        actual = (
            torch.cat((a1, a2)),
            {name: torch.cat((ar1[name], ar2[name])) for name in requested},
        )
        expected = (
            torch.cat((e1, e2)),
            {name: torch.cat((er1[name], er2[name])) for name in requested},
        )
    assert_run(actual, expected, actual_model, expected_model)

    # RecurrentNN's explicit inference graph captures once. The next call has
    # changed input values and continues from the captured call's final states;
    # matching the eager Module proves that replay consumed both updates.
    actual_model, expected_model = build(cudagraph=True), build(native=False)
    changed = x.roll(1, dims=0) * 0.7 + x * 0.15
    graph_saved = []
    with (
        environ.context(dt=1.0),
        torch.autograd.graph.saved_tensors_hooks(
            lambda value: graph_saved.append(value) or value, lambda value: value
        ),
        torch.inference_mode(),
    ):
        first = actual_model(x)
        first_expected = expected_model(x)
        assert_run(first, first_expected, actual_model, expected_model)
        second = actual_model(changed)
        second_expected = expected_model(changed)
        assert_run(second, second_expected, actual_model, expected_model)
    assert graph_saved == []
    assert not torch.equal(first[0], second[0])
    assert actual_model._cudagraph_runner.n_captures == 1


def _glif_alpha_model(backend, requested):
    from btorch.models.functional import init_net_state
    from btorch.models.kernels import KernelOptions
    from btorch.models.neurons import GLIF3
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import AlphaPSC

    neurons, batch = 4, 2
    connection = torch.nn.Linear(neurons, neurons, bias=False, device="cuda")
    with torch.no_grad():
        connection.weight.copy_(
            torch.tensor(
                [
                    [0.12, -0.03, 0.02, 0.01],
                    [0.01, 0.10, -0.02, 0.03],
                    [-0.02, 0.04, 0.11, 0.01],
                    [0.03, 0.01, -0.01, 0.09],
                ],
                device="cuda",
            )
        )
    neuron = GLIF3(
        neurons,
        v_threshold=0.52,
        v_reset=0.0,
        v_rest=0.0,
        c_m=1.0,
        tau=3.25,
        k=[0.13, 0.05],
        asc_amps=[-0.07, 0.025],
        tau_ref=0.0,
        hard_reset=False,
    )
    model = RecurrentNN(
        neuron,
        AlphaPSC(neurons, 2.4, connection),
        backend=KernelOptions(
            neuron=backend, enable_fp_fusion=False, precise_division=True
        ),
        update_state_names=requested,
    ).to(device="cuda", dtype=torch.float32)
    init_net_state(model, batch_size=batch, device="cuda", dtype=torch.float32)
    lane = torch.arange(batch * neurons, device="cuda", dtype=torch.float32).reshape(
        batch, neurons
    )
    model.neuron.v = 0.08 + 0.012 * lane
    model.neuron.Iasc = torch.stack(
        (0.015 * torch.sin(lane), -0.01 * torch.cos(lane)), dim=-1
    )
    model.synapse.psc = 0.02 * torch.cos(lane * 0.3)
    model.synapse.h = 0.01 * torch.sin(lane * 0.2)
    model.requires_grad_(False)
    return model


def _glif_alpha_sources(encoder):
    time = torch.arange(6, device="cuda", dtype=torch.float32)[:, None, None]
    lane = torch.arange(8, device="cuda", dtype=torch.float32).reshape(1, 2, 4)
    first = 0.35 + 0.18 * torch.sin(time * 0.41 + lane * 0.17)
    second = 0.27 + 0.21 * torch.cos(time * 0.29 + lane * 0.13)
    return encoder(first), encoder(second)


def _assert_glif_alpha_state_close(actual, expected):
    for owner, names in (
        ("neuron", ("v", "Iasc", "refractory")),
        ("synapse", ("psc", "h")),
    ):
        actual_owner = getattr(actual, owner)
        expected_owner = getattr(expected, owner)
        for name in names:
            torch.testing.assert_close(
                getattr(actual_owner, name),
                getattr(expected_owner, name),
                atol=3e-5,
                rtol=3e-5,
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native recurrent inference requires CUDA"
)
@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
@pytest.mark.parametrize("requested", [(), ("neuron.Iasc",)])
def test_glif_alpha_native_inference_keeps_only_requested_history(
    backend, mode, requested, monkeypatch
):
    """Frozen GLIF recurrence uses inference despite grad-carrying input
    provenance.

    A trainable encoder produces ordinary tensors before grad mode is
    disabled. Both inference contexts must avoid reverse storage, carry
    every hidden state across calls, and allocate a temporal GLIF mode
    tensor only when requested.
    """
    require_cuda_backend(backend)

    from btorch.models import environ
    from btorch.models.kernels import recurrent_operator

    actual = _glif_alpha_model(backend, requested)
    expected = copy.deepcopy(actual)
    expected.backend = "torch"
    encoder = torch.nn.Linear(4, 4, bias=False, device="cuda")
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(4, device="cuda") * 0.9 + 0.025)
    sources = _glif_alpha_sources(encoder)
    assert all(
        source.requires_grad and source.grad_fn is not None for source in sources
    )

    allocations = NewEmptyShapes()
    saved = []
    trajectory = recurrent_operator._trajectory

    def instrumented_trajectory(*args, **kwargs):
        with allocations:
            return trajectory(*args, **kwargs)

    monkeypatch.setattr(recurrent_operator, "_trajectory", instrumented_trajectory)
    context = getattr(torch, mode)
    final_voltages = []
    with (
        environ.context(dt=0.25),
        torch.autograd.graph.saved_tensors_hooks(
            lambda value: saved.append(value) or value, lambda value: value
        ),
        context(),
    ):
        for source in sources:
            actual_output, actual_records = actual(source)
            expected_output, expected_records = expected(source)
            torch.testing.assert_close(
                actual_output, expected_output, atol=3e-5, rtol=3e-5
            )
            assert tuple(actual_records) == tuple(expected_records) == requested
            for name in requested:
                torch.testing.assert_close(
                    actual_records[name],
                    expected_records[name],
                    atol=3e-5,
                    rtol=3e-5,
                )
            _assert_glif_alpha_state_close(actual, expected)
            final_voltages.append(actual.neuron.v.clone())

    expected_history_shapes = [(6, 2, 4, 2), (6, 2, 4, 2)] if requested else []
    temporal_allocations = [
        shape
        for shape in allocations.shapes
        if len(shape) >= 3 and shape[:3] == (6, 2, 4)
    ]
    assert temporal_allocations == expected_history_shapes
    assert saved == []
    assert not torch.equal(final_voltages[0], final_voltages[1])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native recurrent training requires CUDA"
)
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_frozen_glif_alpha_retains_encoder_vjp_and_all_final_states(backend):
    """Grad-enabled encoder input selects training for a frozen recurrent cell.

    Every physical final state participates in the loss. This proves
    that the upstream VJP crosses the native recurrence without making
    frozen GLIF or Alpha parameters trainable.
    """
    require_cuda_backend(backend)

    from btorch.models import environ

    actual = _glif_alpha_model(backend, ())
    expected = copy.deepcopy(actual)
    expected.backend = "torch"
    encoder = torch.nn.Linear(4, 4, bias=False, device="cuda")
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(4, device="cuda") * 0.88 + 0.03)
    expected_encoder = copy.deepcopy(encoder)
    raw = _glif_alpha_sources(torch.nn.Identity())[0].detach().requires_grad_()
    expected_raw = raw.detach().clone().requires_grad_()
    source = encoder(raw)
    expected_source = expected_encoder(expected_raw)

    with environ.context(dt=0.25):
        actual_output, actual_records = actual(source)
        expected_output, expected_records = expected(expected_source)
    assert actual_records == expected_records == {}
    torch.testing.assert_close(actual_output, expected_output, atol=3e-5, rtol=3e-5)
    _assert_glif_alpha_state_close(actual, expected)

    actual_finals = (
        actual.neuron.v,
        actual.neuron.Iasc,
        actual.neuron.refractory,
        actual.synapse.psc,
        actual.synapse.h,
    )
    expected_finals = (
        expected.neuron.v,
        expected.neuron.Iasc,
        expected.neuron.refractory,
        expected.synapse.psc,
        expected.synapse.h,
    )
    weights = (0.13, 0.17, 0.19, 0.23, 0.29)
    actual_loss = actual_output.square().mean() + sum(
        weight * value.square().mean()
        for weight, value in zip(weights, actual_finals, strict=True)
    )
    expected_loss = expected_output.square().mean() + sum(
        weight * value.square().mean()
        for weight, value in zip(weights, expected_finals, strict=True)
    )
    actual_gradients = torch.autograd.grad(actual_loss, (encoder.weight, raw))
    expected_gradients = torch.autograd.grad(
        expected_loss, (expected_encoder.weight, expected_raw)
    )
    for value, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(value, reference, atol=8e-5, rtol=8e-4)

    assert not actual.neuron.tau.requires_grad
    assert actual.neuron.tau.grad is None
    assert all(parameter.grad is None for parameter in actual.parameters())
