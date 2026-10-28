"""Validate native recurrent ALIF/ELIF with Torch-owned Poisson draws."""

import copy

import pytest
import torch

from btorch.models import environ
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.functional import init_net_state
from btorch.models.kernels import KernelOptions
from btorch.models.neurons import ALIF, ELIF, LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import PoissonRandomSpike
from btorch.models.synapse import ExponentialPSC
from tests.models.kernels.helpers import (
    GPU_BACKENDS,
    NewEmptyShapes,
    require_cuda_backend,
)


RECORDS = (
    "neuron.v",
    "neuron.g_k",
    "neuron.refractory",
    "synapse.psc",
)


def _connection(kind, neurons):
    bias = torch.tensor([-0.016, 0.011, -0.007, 0.019])
    if kind == "sparse":
        return SparseConnection.from_edges(
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
            torch.tensor([1, 3, 0, 2, 1, 3, 0, 2]),
            neurons,
            neurons,
            Synapse(delay=torch.tensor([0, 2, 1, 0, 2, 1, 2, 0]), n_delay=3),
            values=torch.tensor([0.10, -0.04, 0.07, 0.09, -0.05, 0.12, 0.06, 0.08]),
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


def _build(neuron_type, connection_kind, backend, *, spiking=True, records=RECORDS):
    neurons, batch = 4, 2
    trainable = {
        "v_threshold",
        "v_reset",
        "c_m",
        "g_leak",
        "E_leak",
        "E_k",
        "tau_adapt",
        "dg_k",
        "tau_ref",
    }
    kwargs = {
        "v_threshold": 0.72,
        "v_reset": -0.13,
        "c_m": 1.15,
        "g_leak": 0.68,
        "E_leak": -0.07,
        "E_k": -0.78,
        "tau_adapt": 3.9,
        "dg_k": 0.10,
        "tau_ref": 0.75,
        "hard_reset": True,
        "pre_spike_v": True,
        "surrogate_function": PoissonRandomSpike(
            spiking=spiking,
            tau=2.4,
            rho=0.72,
            leak=0.18,
            k=1.08,
            damping_factor=0.57,
        ),
        "trainable_param": trainable,
    }
    if neuron_type is ELIF:
        kwargs.update(delta_T=0.19, v_T=0.34)
        kwargs["trainable_param"] = trainable | {"delta_T", "v_T"}
    model = RecurrentNN(
        neuron_type(neurons, **kwargs),
        ExponentialPSC(neurons, 2.6, _connection(connection_kind, neurons)),
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
    model.neuron.v = (-0.10 + 0.032 * lane).requires_grad_()
    model.neuron.g_k = (0.025 + 0.007 * lane).requires_grad_()
    refractory = 0.05 * (lane % 3)
    refractory[0, 0] = 3.0
    model.neuron.refractory = refractory.requires_grad_()
    model.synapse.psc = (-0.035 + 0.010 * lane).requires_grad_()
    model.synapse.tau_syn.requires_grad_()
    if model.synapse.delayed_spike is not None:
        history = model.synapse.delayed_spike.history
        phase = torch.arange(history.numel(), device="cuda").reshape(history.shape)
        model.synapse.delayed_spike.history = (
            0.035 * torch.sin(phase * 0.4)
        ).requires_grad_()
    return model


def _roots(model, drive, events):
    roots = {"drive": drive, "events": events, **dict(model.named_parameters())}
    roots.update(
        {
            "initial.v": model.neuron.v,
            "initial.g_k": model.neuron.g_k,
            "initial.refractory": model.neuron.refractory,
            "initial.psc": model.synapse.psc,
            "parameter.tau_syn": model.synapse.tau_syn,
        }
    )
    if model.synapse.delayed_spike is not None:
        roots["initial.delay_history"] = model.synapse.delayed_spike.history
    return roots


def _final(model):
    result = (
        model.neuron.v,
        model.neuron.g_k,
        model.neuron.refractory,
        model.neuron.v_pre_spike,
        model.synapse.psc,
    )
    if model.synapse.delayed_spike is not None:
        result += (model.synapse.delayed_spike.history,)
    return result


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
    return (output, *(records[name] for name in RECORDS), *_final(model))


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("connection_kind", ("dense", "sparse"))
@pytest.mark.parametrize("neuron_type", (ALIF, ELIF))
def test_recurrent_poisson_whole_chunk_rng_and_full_bptt(
    backend, connection_kind, neuron_type
):
    """Native recurrence matches actual Modules, RNG position, and full BPTT.

    Temporal cotangents are zero before the split, so nonzero first-
    chunk input gradients can only arrive through the carried neuron,
    PSC, and delay states. Backward is run after the RNG snapshot to
    prove it reuses saved spikes and consumes no additional random
    numbers.
    """
    require_cuda_backend(backend)
    torch.manual_seed(834)
    initial = _build(neuron_type, connection_kind, "torch")
    whole = copy.deepcopy(initial)
    chunked = copy.deepcopy(initial)
    expected = copy.deepcopy(initial)
    options = KernelOptions(
        neuron=backend,
        connection="triton",
        enable_fp_fusion=False,
        precise_division=True,
    )
    whole.backend = options
    chunked.backend = options

    generator = torch.Generator().manual_seed(391)
    drive_host = 0.34 + 0.42 * torch.rand(9, 2, 4, generator=generator)
    events_host = 0.045 * torch.randn(9, 2, 4, generator=generator)
    inputs = []
    for model in (whole, chunked, expected):
        drive = drive_host.cuda().requires_grad_()
        events = events_host.cuda().requires_grad_()
        inputs.append((drive, events, _roots(model, drive, events)))

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(6721)
        initial_rng = torch.cuda.get_rng_state()
        with environ.context(dt=0.27):
            expected_values = _run(expected, inputs[2][0], inputs[2][1], None)
        expected_rng = torch.cuda.get_rng_state()

        torch.cuda.set_rng_state(initial_rng)
        with environ.context(dt=0.27):
            whole_values = _run(whole, inputs[0][0], inputs[0][1], None)
        whole_rng = torch.cuda.get_rng_state()

        torch.cuda.set_rng_state(initial_rng)
        with environ.context(dt=0.27):
            chunked_values = _run(chunked, inputs[1][0], inputs[1][1], 4)
        chunked_rng = torch.cuda.get_rng_state()

        for values in (whole_values, chunked_values):
            assert torch.equal(values[0], expected_values[0])
            for actual, reference in zip(values[1:], expected_values[1:], strict=True):
                torch.testing.assert_close(actual, reference, atol=6e-6, rtol=6e-5)
        assert expected_values[0].eq(0).any()
        assert expected_values[0].eq(1).any()
        assert torch.equal(whole_rng, expected_rng)
        assert torch.equal(chunked_rng, expected_rng)

        cotangent_generator = torch.Generator().manual_seed(993)
        cotangents = []
        for index, value in enumerate(expected_values):
            cotangent = torch.randn(value.shape, generator=cotangent_generator).cuda()
            if index < 1 + len(RECORDS):
                cotangent[:4] = 0
            cotangents.append(cotangent)
        cotangents = tuple(cotangents)
        rng_before_backward = torch.cuda.get_rng_state()
        expected_gradients = torch.autograd.grad(
            expected_values,
            tuple(inputs[2][2].values()),
            cotangents,
            allow_unused=True,
        )
        for values, (_, _, roots) in zip(
            (whole_values, chunked_values), inputs[:2], strict=True
        ):
            gradients = torch.autograd.grad(
                values, tuple(roots.values()), cotangents, allow_unused=True
            )
            for (name, root), gradient, reference in zip(
                roots.items(), gradients, expected_gradients, strict=True
            ):
                assert (gradient is None) == (reference is None), name
                assert gradient is not None and torch.isfinite(gradient).all(), name
                assert bool(gradient.abs().sum() > 0), name
                torch.testing.assert_close(
                    gradient, reference, atol=7e-5, rtol=7e-4, msg=name
                )
                if values is chunked_values and name in ("drive", "events"):
                    assert bool(gradient[:4].abs().sum() > 0), name
                if connection_kind == "sparse" and root is getattr(
                    chunked.synapse.linear.weight, "value", None
                ):
                    assert gradient.numel() == chunked.synapse.linear.nnz
        assert torch.equal(torch.cuda.get_rng_state(), rng_before_backward)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("neuron_type", (ALIF, ELIF))
def test_recurrent_nonspiking_poisson_is_deterministic(backend, neuron_type):
    """The analog Poisson primitive consumes no RNG and remains
    differentiable."""
    require_cuda_backend(backend)
    actual = _build(neuron_type, "dense", backend, spiking=False)
    expected = copy.deepcopy(actual)
    expected.backend = "torch"
    drive = torch.linspace(-0.15, 0.65, 48, device="cuda").reshape(6, 2, 4)
    events = torch.zeros_like(drive)
    actual_drive = drive.detach().requires_grad_()
    expected_drive = drive.detach().requires_grad_()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(193)
        before = torch.cuda.get_rng_state()
        with environ.context(dt=0.27):
            actual_values = _run(actual, actual_drive, events, None)
        after = torch.cuda.get_rng_state()
        with environ.context(dt=0.27):
            expected_values = _run(expected, expected_drive, events, None)
        assert torch.equal(torch.cuda.get_rng_state(), after)
        assert torch.equal(before, after)
    for value, reference in zip(actual_values, expected_values, strict=True):
        torch.testing.assert_close(value, reference, atol=6e-6, rtol=6e-5)
    cotangents = tuple(torch.ones_like(value) for value in actual_values)
    (gradient,) = torch.autograd.grad(actual_values, actual_drive, cotangents)
    (reference_gradient,) = torch.autograd.grad(
        expected_values, expected_drive, cotangents
    )
    torch.testing.assert_close(gradient, reference_gradient, atol=7e-5, rtol=7e-4)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_recurrent_poisson_inference_has_no_backward_history(backend, monkeypatch):
    """Inference advances RNG and state without saving uniforms or
    trajectories."""
    require_cuda_backend(backend)
    from btorch.models.kernels import recurrent_operator

    actual = _build(ALIF, "dense", backend, records=())
    expected = copy.deepcopy(actual)
    expected.backend = "torch"
    drive = torch.linspace(0.22, 0.78, 40, device="cuda").reshape(5, 2, 4)
    allocations = NewEmptyShapes()
    saved = []
    calls = {"inference": 0, "training": 0}
    native_inference = recurrent_operator.inference
    native_training = recurrent_operator.training
    trajectory = recurrent_operator._trajectory

    def instrumented_trajectory(*args, **kwargs):
        with allocations:
            return trajectory(*args, **kwargs)

    def counted_inference(*args, **kwargs):
        calls["inference"] += 1
        return native_inference(*args, **kwargs)

    def counted_training(*args, **kwargs):
        calls["training"] += 1
        return native_training(*args, **kwargs)

    monkeypatch.setattr(recurrent_operator, "_trajectory", instrumented_trajectory)
    monkeypatch.setattr(recurrent_operator, "inference", counted_inference)
    monkeypatch.setattr(recurrent_operator, "training", counted_training)
    with (
        torch.random.fork_rng(devices=[torch.cuda.current_device()]),
        torch.autograd.graph.saved_tensors_hooks(
            lambda value: saved.append(value) or value, lambda value: value
        ),
        torch.inference_mode(),
    ):
        torch.cuda.manual_seed(475)
        initial_rng = torch.cuda.get_rng_state()
        with environ.context(dt=0.27):
            expected_output, expected_records = expected(drive)
        expected_rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(initial_rng)
        with environ.context(dt=0.27):
            actual_output, actual_records = actual(drive)
        actual_rng = torch.cuda.get_rng_state()
    assert torch.equal(actual_output, expected_output)
    assert actual_records == expected_records == {}
    for value, reference in zip(_final(actual), _final(expected), strict=True):
        torch.testing.assert_close(value, reference, atol=6e-6, rtol=6e-5)
    assert torch.equal(actual_rng, expected_rng)
    assert saved == []
    assert calls == {"inference": 1, "training": 0}
    assert not any(shape[:3] == (5, 2, 4) for shape in allocations.shapes)


def test_recurrent_poisson_cudagraph_shapes_preserve_state_and_rng():
    """Match shape-specific split captures for stochastic durations."""
    require_cuda_backend("triton")
    eager = _build(ALIF, "dense", "triton")
    captured = copy.deepcopy(eager)
    captured.cudagraph = True
    drive = torch.linspace(0.22, 0.78, 413 * 2 * 4, device="cuda").reshape(413, 2, 4)
    events = torch.zeros_like(drive)
    sequences = (
        (drive[:13], events[:13]),
        (drive, events),
        (drive[:31], events[:31]),
    )

    with (
        torch.random.fork_rng(devices=[torch.cuda.current_device()]),
        torch.inference_mode(),
    ):
        torch.cuda.manual_seed(4751)
        initial_rng = torch.cuda.get_rng_state()
        expected_results = []
        with environ.context(dt=0.27):
            for current_drive, current_events in sequences:
                expected_results.append(eager(current_drive, current_events))
        expected_rng = torch.cuda.get_rng_state()

        torch.cuda.set_rng_state(initial_rng)
        actual_results = []
        with environ.context(dt=0.27):
            for current_drive, current_events in sequences:
                actual_results.append(captured(current_drive, current_events))
        actual_rng = torch.cuda.get_rng_state()

    for (actual, actual_records), (expected, expected_records) in zip(
        actual_results, expected_results, strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=6e-6, rtol=6e-5)
        assert actual_records.keys() == expected_records.keys()
        for name in RECORDS:
            torch.testing.assert_close(
                actual_records[name],
                expected_records[name],
                atol=6e-6,
                rtol=6e-5,
            )
    for actual_state, expected_state in zip(
        _final(captured), _final(eager), strict=True
    ):
        torch.testing.assert_close(actual_state, expected_state, atol=6e-6, rtol=6e-5)
    assert torch.equal(actual_rng, expected_rng)


@pytest.mark.parametrize("input_module", (torch.nn.Dropout(0.2), torch.nn.RReLU()))
@pytest.mark.parametrize("spiking", (False, True))
def test_stochastic_input_module_is_rejected_before_rng_or_callback(
    input_module, spiking
):
    """Ambiguous RNG interleaving fails before executing the input callback."""
    model = RecurrentNN(
        ALIF(4, surrogate_function=PoissonRandomSpike(spiking=spiking)),
        ExponentialPSC(4, 2.0, torch.nn.Linear(4, 4)),
        neuron_inp_module=input_module,
        backend="triton",
    )
    init_net_state(model, batch_size=2)
    x = torch.ones(3, 2, 4)
    before = torch.get_rng_state()
    with pytest.raises(NotImplementedError, match="stochastic input modules"):
        model(x)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("spiking", (False, True))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("target", ("input", "model"))
def test_recurrent_poisson_rejects_reduced_precision_before_rng(spiking, dtype, target):
    """The public recurrent path rejects unqualified precision before draws.

    This covers both sampled and analog Poisson modes: the latter consumes no
    RNG, but shares the same FP32 native scientific qualification.
    """
    model = RecurrentNN(
        ALIF(
            4,
            surrogate_function=PoissonRandomSpike(spiking=spiking),
            hard_reset=True,
        ),
        ExponentialPSC(4, 2.0, torch.nn.Linear(4, 4)),
        backend="triton",
    )
    model_dtype = dtype if target == "model" else torch.float32
    model.to(dtype=model_dtype)
    init_net_state(model, batch_size=2, dtype=model_dtype)
    input_dtype = dtype if target == "input" else torch.float32
    drive = torch.full((3, 2, 4), 0.4, dtype=input_dtype)

    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="requires FP32"):
        with environ.context(dt=0.25):
            model(drive)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_lif_poisson_t17_bptt_uses_saved_raster_without_checkpoint_replay(backend):
    """Spiking LIF bypasses deterministic replay and consumes no backward RNG.

    T=17 would select the block-16 LIF checkpoint plan for a
    deterministic surrogate. Matching the actual Module and preserving
    the generator through backward proves that stochastic LIF instead
    retains its authoritative sampled raster.
    """
    require_cuda_backend(backend)
    neurons, batch = 4, 2
    connection = torch.nn.Linear(neurons, neurons)
    with torch.no_grad():
        connection.weight.copy_(0.04 * torch.eye(neurons) + 0.012)
        connection.bias.copy_(torch.linspace(-0.01, 0.018, neurons))
    actual = RecurrentNN(
        LIF(
            neurons,
            v_threshold=0.78,
            v_reset=-0.12,
            c_m=1.1,
            tau=4.2,
            trainable_param={"v_threshold", "v_reset", "c_m", "tau"},
            surrogate_function=PoissonRandomSpike(
                tau=2.4,
                rho=0.72,
                leak=0.18,
                k=1.08,
                damping_factor=0.57,
            ),
            hard_reset=True,
            pre_spike_v=True,
        ),
        ExponentialPSC(neurons, 2.7, connection),
        backend=KernelOptions(neuron=backend),
        update_state_names=(),
    ).to(device="cuda", dtype=torch.float32)
    init_net_state(actual, batch_size=batch, device="cuda", dtype=torch.float32)
    lane = torch.arange(batch * neurons, device="cuda").reshape(batch, neurons)
    actual.neuron.v = (-0.08 + 0.025 * lane).requires_grad_()
    actual.synapse.psc = (-0.03 + 0.008 * lane).requires_grad_()
    actual.synapse.tau_syn.requires_grad_()
    expected = copy.deepcopy(actual)
    expected.backend = "torch"
    generator = torch.Generator().manual_seed(643)
    x = (0.31 + 0.27 * torch.rand(17, 2, 4, generator=generator)).cuda()
    events = (0.03 * torch.randn(17, 2, 4, generator=generator)).cuda()
    x = x.requires_grad_()
    events = events.requires_grad_()
    expected_x = x.detach().clone().requires_grad_()
    expected_events = events.detach().clone().requires_grad_()
    roots = (
        x,
        events,
        actual.neuron.v,
        actual.synapse.psc,
        actual.synapse.tau_syn,
        *actual.parameters(),
    )
    expected_roots = (
        expected_x,
        expected_events,
        expected.neuron.v,
        expected.synapse.psc,
        expected.synapse.tau_syn,
        *expected.parameters(),
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(3047)
        initial_rng = torch.cuda.get_rng_state()
        with environ.context(dt=0.27):
            output, records = actual(x, events)
        actual_rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(initial_rng)
        with environ.context(dt=0.27):
            expected_output, expected_records = expected(expected_x, expected_events)
        assert records == expected_records == {}
        assert torch.equal(actual_rng, torch.cuda.get_rng_state())
        assert torch.equal(output, expected_output)
        torch.testing.assert_close(
            actual.neuron.v, expected.neuron.v, atol=2e-5, rtol=2e-5
        )
        torch.testing.assert_close(
            actual.synapse.psc, expected.synapse.psc, atol=2e-5, rtol=2e-5
        )
        cotangent_generator = torch.Generator().manual_seed(487)
        cotangents = (
            torch.randn(output.shape, generator=cotangent_generator).cuda(),
            torch.randn(actual.neuron.v.shape, generator=cotangent_generator).cuda(),
            torch.randn(actual.synapse.psc.shape, generator=cotangent_generator).cuda(),
        )
        rng_before_backward = torch.cuda.get_rng_state()
        gradients = torch.autograd.grad(
            (output, actual.neuron.v, actual.synapse.psc),
            roots,
            cotangents,
            allow_unused=True,
        )
        references = torch.autograd.grad(
            (expected_output, expected.neuron.v, expected.synapse.psc),
            expected_roots,
            cotangents,
            allow_unused=True,
        )
        assert torch.equal(torch.cuda.get_rng_state(), rng_before_backward)
        for index, (gradient, reference) in enumerate(
            zip(gradients, references, strict=True)
        ):
            assert (gradient is None) == (reference is None), index
            if reference is not None:
                torch.testing.assert_close(
                    gradient, reference, atol=8e-5, rtol=8e-4, msg=f"root {index}"
                )
