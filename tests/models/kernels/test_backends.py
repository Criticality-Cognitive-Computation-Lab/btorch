"""Compare GPU sequence kernels with trusted eager autograd recurrences."""

import importlib.util

import pytest
import torch

from btorch.models.kernels.backends.engine import backward, forward
from btorch.models.kernels.contracts import CellSpec
from btorch.models.kernels.templates import get_template
from btorch.models.kernels.torch_backend import forward_step


BACKENDS = ("triton", "cupy", "tilelang")
MODELS = (
    "if",
    "lif",
    "glif3",
    "izhikevich",
    "exponential",
    "alpha",
    "alpha_billeh",
    "dual_exponential",
)


def _available(backend):
    if not torch.cuda.is_available() or importlib.util.find_spec(backend) is None:
        pytest.skip(f"{backend} CUDA runtime is unavailable")


def _example(spec, dtype=torch.float32):
    template = get_template(spec)
    torch.manual_seed(312)
    # Nonzero initial states exercise continuation, rather than testing only a
    # reset trajectory. Broadcast parameters exercise per-neuron reductions.
    x = (torch.rand(7, 2, 5, device="cuda", dtype=dtype) * 0.8).requires_grad_()
    defaults = {
        "v_threshold": 1.0,
        "v_reset": 0.0,
        "c_m": 1.2,
        "tau": 4.0,
        "v_rest": 0.1,
        "k": 0.2,
        "asc_amps": -0.1,
        "v_peak": 1.5,
        "a": 0.1,
        "b": 0.2,
        "d": 0.1,
        "tau_ref": 2.0,
        "tau_syn": 3.0,
        "g_max": 1.1,
        "syn_decay": 0.7,
        "tau_rise": 2.0,
        "tau_decay": 5.0,
    }
    states = tuple(
        (
            torch.rand((2, 5, 2) if name == "Iasc" else (2, 5), device="cuda") * 0.2
            if name != "refractory"
            else torch.zeros(2, 5, device="cuda")
        ).requires_grad_()
        for name in template.state_names
    )
    parameters = tuple(
        torch.full(
            (5, 2) if spec.model == "glif3" and name in ("k", "asc_amps") else (5,),
            defaults[name],
            device="cuda",
        ).requires_grad_()
        for name in template.parameter_names
    )
    return x, states, parameters


def _trusted(x, states, parameters, spec, names):
    outputs, records = [], {name: [] for name in names}
    for value in x.float().unbind(0):
        output, states, step_records = forward_step(value, states, parameters, spec)
        outputs.append(output)
        for name in names:
            records[name].append(step_records[name])
    return (
        torch.stack(outputs).to(x.dtype),
        states,
        tuple(torch.stack(records[name]) for name in names),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dt", (0.25, 1.0))
def test_sequence_outputs_and_all_float_gradients(backend, model, dt):
    _available(backend)
    if model == "alpha_billeh" and dt != 1.0:
        pytest.skip("AlphaPSCBilleh fixes its discrete timestep to one")
    # Different physical timesteps share generated source, but must change both
    # the recurrence and its VJP consistently with the Python Module golden.
    spec = CellSpec(
        model, dt=dt, hard_reset=True, refractory=model in ("if", "lif", "glif3")
    )
    template = get_template(spec)
    names = template.history_names
    x, states, parameters = _example(spec)
    expected = _trusted(x, states, parameters, spec, names)
    actual = forward(backend, x, states, parameters, spec, names)
    expected_flat = (expected[0], *expected[1], *expected[2])
    actual_flat = (actual[0], *actual[1], *actual[2])
    for result, reference in zip(actual_flat, expected_flat):
        torch.testing.assert_close(result, reference, atol=5e-6, rtol=5e-5)
    # An attached loss on every history and final state is valid research usage
    # (e.g. voltage regularization). Its cotangent must join the spike loss and
    # propagate backwards through the whole recurrent state chain.
    cotangents = tuple(torch.randn_like(value) for value in expected_flat)
    trusted_grads = torch.autograd.grad(
        expected_flat, (x, *states, *parameters), cotangents, allow_unused=True
    )
    n_state = len(states)
    actual_grads = backward(
        backend,
        x,
        states,
        parameters,
        spec,
        names,
        actual[2],
        cotangents[0],
        cotangents[1 : 1 + n_state],
        cotangents[1 + n_state :],
    )
    for result, reference, value in zip(
        (actual_grads[0], *actual_grads[1], *actual_grads[2]),
        trusted_grads,
        (x, *states, *parameters),
    ):
        # The overwritten dual-exponential PSC initial state does not
        # participate in the dynamics; its derivative is zero.
        if reference is None:
            reference = torch.zeros_like(value)
        torch.testing.assert_close(result, reference, atol=5e-5, rtol=5e-4)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "surrogate", ("sigmoid", "atan", "atan_approx", "erf", "superspike", "triangle")
)
@pytest.mark.parametrize("detach_reset", (False, True))
@pytest.mark.parametrize("refractory", (False, True))
def test_normalized_surrogate_and_detached_soft_reset(
    backend, surrogate, detach_reset, refractory
):
    _available(backend)
    spec = CellSpec(
        "lif",
        surrogate=surrogate,
        alpha=3.1,
        damping=0.7,
        detach_reset=detach_reset,
        refractory=refractory,
    )
    x, states, parameters = _example(spec)
    names = get_template(spec).state_names
    expected = _trusted(x, states, parameters, spec, names)
    actual = forward(backend, x, states, parameters, spec, names)
    cotangent = torch.linspace(-0.4, 0.7, x.numel(), device="cuda").reshape_as(x)
    # Refractory occupancy is also a valid state regularizer. Its cotangent
    # must pass through elapsed time and tau_ref, while detach_reset cuts
    # only the spike-dependent reset path. Testing the spike loss alone would
    # leave this state-adjoint branch entirely unexercised.
    final_cotangents = tuple(torch.randn_like(value) for value in expected[1])
    gradients = torch.autograd.grad(
        (expected[0], *expected[1]),
        (x, *states, *parameters),
        (cotangent, *final_cotangents),
    )
    reverse = backward(
        backend,
        x,
        states,
        parameters,
        spec,
        names,
        actual[2],
        cotangent,
        final_cotangents,
        tuple(None for _ in names),
    )
    # These widths/damping values intentionally differ from defaults: the
    # derivative is evaluated after normalization by threshold-reset gap.
    for result, reference in zip((reverse[0], *reverse[1], *reverse[2]), gradients):
        torch.testing.assert_close(result, reference, atol=4e-5, rtol=4e-4)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_low_precision_long_sequence_reports_first_spike_divergence(backend, dtype):
    _available(backend)
    spec = CellSpec("lif")
    x, states, parameters = _example(spec, dtype)
    x = x.detach().repeat(32, 1, 1).requires_grad_()
    expected = _trusted(x, states, parameters, spec, ("v",))
    actual = forward(backend, x, states, parameters, spec, ("v",))
    divergence = (actual[0] != expected[0]).nonzero()
    assert (
        divergence.numel() == 0
    ), f"First spike divergence [time,batch,neuron]: {divergence[0].tolist()}"
    torch.testing.assert_close(actual[1][0], expected[1][0], atol=4e-6, rtol=4e-5)
    assert actual[1][0].dtype == torch.float32
    # The external input/gradient stays in its advertised low precision while
    # the temporal state adjoint accumulates in FP32 until the boundary cast.
    cotangent = torch.ones_like(expected[0])
    trusted_gradient = torch.autograd.grad(expected[0], x, cotangent)[0]
    reverse = backward(
        backend,
        x,
        states,
        parameters,
        spec,
        ("v",),
        actual[2],
        cotangent,
        (None,),
        (None,),
        needs_input_grad=(True, False, False, False, False, False),
    )
    torch.testing.assert_close(
        reverse[0], trusted_gradient, atol=2e-5, rtol=2 * torch.finfo(dtype).eps
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_simulation_chunk_continuation_and_cuda_graph(backend):
    _available(backend)
    spec = CellSpec("alpha")
    x, states, parameters = _example(spec)
    with torch.no_grad():
        whole = forward(backend, x, states, parameters, spec, ())
        first = forward(backend, x[:3], states, parameters, spec, ())
        second = forward(backend, x[3:], first[1], parameters, spec, ())
        torch.testing.assert_close(
            torch.cat((first[0], second[0])), whole[0], atol=4e-6, rtol=4e-5
        )
        assert whole[2] == ()  # Inference has no unrequested backward history.
        # Compile and allocator warmup happen before capture. Replay must use
        # current device buffers, not a stale tensor copied through the host.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward(backend, x, states, parameters, spec, ())
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward(backend, x, states, parameters, spec, ())
        graph.replay()
        torch.testing.assert_close(captured[0], whole[0], atol=4e-6, rtol=4e-5)
        # A device timestep is invocation data, not a constexpr. The same
        # captured kernel must use its new value without regeneration/capture.
        dt = torch.ones((), device=x.device)
        runtime_spec = CellSpec("alpha", dt=dt)
        forward(backend, x, states, parameters, runtime_spec, ())
        runtime_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(runtime_graph):
            runtime_output = forward(backend, x, states, parameters, runtime_spec, ())
        dt.fill_(0.25)
        runtime_graph.replay()
        expected = _trusted(x, states, parameters, CellSpec("alpha", dt=0.25), ())
        torch.testing.assert_close(runtime_output[0], expected[0], atol=4e-6, rtol=4e-5)
        for actual_state, expected_state in zip(runtime_output[1], expected[1]):
            torch.testing.assert_close(
                actual_state, expected_state, atol=4e-6, rtol=4e-5
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fullgraph_trainable_encoder_through_frozen_cell(backend):
    _available(backend)
    from btorch.models.kernels import HistoryPlan, run_multi_step

    spec = CellSpec("alpha")
    x, states, parameters = _example(spec)
    states = tuple(value.detach() for value in states)
    parameters = tuple(value.detach() for value in parameters)

    # Freezing the cell does not remove its transpose/state reverse recurrence:
    # the encoder upstream still needs input gradients through every timestep.
    def network(value):
        encoded = value * 0.7 + 0.1
        result = run_multi_step(
            encoded,
            states,
            parameters,
            spec,
            HistoryPlan(requested=("h", "psc")),
            backend=backend,
        )
        return result[0].square().mean() + result[-2].square().mean()

    expected = network(x)
    expected_grad = torch.autograd.grad(expected, x)[0]
    compiled = torch.compile(network, fullgraph=True)
    actual = compiled(x)
    actual_grad = torch.autograd.grad(actual, x)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad, atol=5e-5, rtol=5e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_sequence_and_masked_parameter_gradients(backend):
    _available(backend)
    spec = CellSpec("alpha")
    x, states, parameters = _example(spec)
    template = get_template(spec)
    empty = forward(backend, x[:0], states, parameters, spec, template.state_names)
    assert empty[0].shape[0] == 0
    for result, initial in zip(empty[1], states):
        torch.testing.assert_close(result, initial)
        # Pure sequence operators cannot return input aliases, including when
        # no step executes; users may continue or checkpoint the returned state.
        assert result.data_ptr() != initial.data_ptr()
    actual = forward(backend, x, states, parameters, spec, template.state_names)
    grad = backward(
        backend,
        x,
        states,
        parameters,
        spec,
        template.state_names,
        actual[2],
        torch.ones_like(x),
        (None, None),
        (None, None),
        needs_input_grad=(True, False, False, False, False),
    )
    trusted = _trusted(x, states, parameters, spec, template.state_names)
    expected = torch.autograd.grad(trusted[0], x, torch.ones_like(x))[0]
    torch.testing.assert_close(grad[0], expected, atol=5e-5, rtol=5e-4)
    assert grad[1] == (None, None)
    assert grad[2] == (None, None)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("record_loss", (False, True))
@pytest.mark.parametrize("neuron_model", ("if", "lif", "glif3", "izhikevich"))
@pytest.mark.parametrize(
    "psc_model", ("exponential", "alpha", "alpha_billeh", "dual_exponential")
)
def test_fused_psc_then_neuron_receptor_order(
    backend, neuron_model, psc_model, record_loss
):
    _available(backend)
    from btorch.models.kernels.backends.fused import (
        fused_step_backward,
        fused_step_forward,
    )

    # Distinct invocation timesteps catch accidental sharing of the two
    # prefixed dt registers in the fused forward and reverse stages.
    neuron_spec, psc_spec = (
        CellSpec(neuron_model, dt=0.25),
        CellSpec(psc_model, dt=1.0 if psc_model == "alpha_billeh" else 0.5),
    )
    neuron_template, psc_template = get_template(neuron_spec), get_template(psc_spec)
    _, neuron_states, neuron_parameters = _example(neuron_spec)
    _, psc_states, psc_parameters = _example(psc_spec)
    receptors = 2
    # Connections keep their flat, receptor-fast topology. Two distinct PSC
    # states are advanced before their current is summed for the NEXT neuron;
    # confusing this order would shift the recurrent trajectory by one step.
    psc_states = tuple(value.repeat_interleave(receptors, -1) for value in psc_states)
    psc_parameters = tuple(
        value.repeat_interleave(receptors, -1) for value in psc_parameters
    )
    drive = (torch.randn(2, 10, device="cuda") * 0.2).requires_grad_()
    neuron_input = (torch.rand(2, 5, device="cuda") * 0.1).requires_grad_()
    current, expected_psc, psc_records = forward_step(
        drive, psc_states, psc_parameters, psc_spec
    )
    expected_spike, expected_neuron, neuron_records = forward_step(
        neuron_input + current.reshape(2, 5, receptors).sum(-1),
        neuron_states,
        neuron_parameters,
        neuron_spec,
    )
    actual = fused_step_forward(
        neuron_input,
        drive,
        neuron_states,
        psc_states,
        neuron_parameters,
        psc_parameters,
        neuron_spec,
        psc_spec,
        neuron_template.history_names,
        psc_template.history_names,
        backend=backend,
    )
    torch.testing.assert_close(actual[0], expected_spike)
    for result, expected in zip(actual[1], expected_neuron):
        torch.testing.assert_close(result, expected, atol=5e-6, rtol=5e-5)
    for result, expected in zip(actual[2], expected_psc):
        torch.testing.assert_close(result, expected, atol=5e-6, rtol=5e-5)
    for result, name in zip(actual[3], neuron_template.history_names):
        torch.testing.assert_close(
            result[0], neuron_records[name], atol=5e-6, rtol=5e-5
        )
    for result, name in zip(actual[4], psc_template.history_names):
        torch.testing.assert_close(result[0], psc_records[name], atol=5e-6, rtol=5e-5)
    # The paired reverse stage consumes state and record losses together, then
    # sends the neuron's current adjoint into every receptor's PSC output.
    expected_flat = (
        expected_spike,
        *expected_neuron,
        *expected_psc,
        *(
            neuron_records[name]
            for name in neuron_template.history_names
            if record_loss
        ),
        *(psc_records[name] for name in psc_template.history_names if record_loss),
    )
    cotangents = tuple(torch.randn_like(value) for value in expected_flat)
    nn, np, nh = (
        len(neuron_states),
        len(psc_states),
        len(neuron_template.history_names) if record_loss else 0,
    )
    if not record_loss:
        # A loss on the first neuron state only exercises missing spike and
        # unused final-state cotangents as well as missing record losses. The
        # eager oracle uses explicit zeros; the kernel receives None and must
        # generate zeros locally without allocating corresponding GPU tensors.
        cotangents = tuple(
            value if index == 1 else torch.zeros_like(value)
            for index, value in enumerate(cotangents)
        )
    inputs = (
        neuron_input,
        drive,
        *neuron_states,
        *psc_states,
        *neuron_parameters,
        *psc_parameters,
    )
    trusted = torch.autograd.grad(expected_flat, inputs, cotangents, allow_unused=True)
    actual_reverse = fused_step_backward(
        neuron_input,
        drive,
        neuron_states,
        psc_states,
        neuron_parameters,
        psc_parameters,
        neuron_spec,
        psc_spec,
        cotangents[0] if record_loss else None,
        cotangents[1 : 1 + nn]
        if record_loss
        else (cotangents[1], *((None,) * (nn - 1))),
        cotangents[1 + nn : 1 + nn + np] if record_loss else (None,) * np,
        # With no record loss, reverse must recompute locally without allocating
        # zero record cotangents. This uses the same state/parameter VJP oracle
        # as the record-loss case, including GLIF3 mode lanes and PSC receptors.
        grad_neuron_records=dict(
            zip(
                neuron_template.history_names,
                cotangents[1 + nn + np : 1 + nn + np + nh],
            )
        )
        if record_loss
        else None,
        grad_psc_records=dict(
            zip(psc_template.history_names, cotangents[1 + nn + np + nh :])
        )
        if record_loss
        else None,
        backend=backend,
    )
    reverse_flat = (
        *actual_reverse[:2],
        *actual_reverse[2],
        *actual_reverse[3],
        *actual_reverse[4],
        *actual_reverse[5],
    )
    for value, reference, initial in zip(reverse_flat, trusted, inputs):
        if reference is None:
            reference = torch.zeros_like(initial)
        torch.testing.assert_close(value, reference, atol=5e-5, rtol=5e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fused_merged_state_preserves_drive_receptor_layout(backend):
    _available(backend)
    from btorch.models.kernels.backends.fused import (
        fused_step_backward,
        fused_step_forward,
    )

    neuron_spec, psc_spec = CellSpec("lif"), CellSpec("alpha")
    _, neuron_states, neuron_parameters = _example(neuron_spec)
    _, psc_states, psc_parameters = _example(psc_spec)
    drive = (torch.randn(2, 15, device="cuda") * 0.3).requires_grad_()
    external = (torch.rand(2, 5, device="cuda") * 0.1).requires_grad_()
    # Compatible shared linear dynamics merge only their internal PSC state.
    # The connection still has three drive receptor columns, which the cell
    # sums before the single merged PSC update.
    merged_drive = drive.reshape(2, 5, 3).sum(-1)
    current, expected_psc, _ = forward_step(
        merged_drive, psc_states, psc_parameters, psc_spec
    )
    expected_spike, expected_neuron, _ = forward_step(
        external + current, neuron_states, neuron_parameters, neuron_spec
    )
    result = fused_step_forward(
        external,
        drive,
        neuron_states,
        psc_states,
        neuron_parameters,
        psc_parameters,
        neuron_spec,
        psc_spec,
        backend=backend,
    )
    torch.testing.assert_close(result[0], expected_spike)
    for actual, expected in zip(
        (*result[1], *result[2]), (*expected_neuron, *expected_psc)
    ):
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
    reverse = fused_step_backward(
        external,
        drive,
        neuron_states,
        psc_states,
        neuron_parameters,
        psc_parameters,
        neuron_spec,
        psc_spec,
        torch.ones_like(expected_spike),
        tuple(None for _ in neuron_states),
        tuple(None for _ in psc_states),
        needs_neuron_parameters=(False,) * len(neuron_parameters),
        needs_psc_parameters=(False,) * len(psc_parameters),
        backend=backend,
    )
    expected_gradients = torch.autograd.grad(
        expected_spike, (external, drive), torch.ones_like(expected_spike)
    )
    for actual, expected in zip(reverse[:2], expected_gradients):
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-4)
    assert reverse[4] == (None,) * len(neuron_parameters)
    assert reverse[5] == (None,) * len(psc_parameters)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpu_sparse_recurrent_heter_delay_and_ring_gradient(backend):
    _available(backend)
    import copy

    from btorch.models import environ
    from btorch.models.connection import SparseConnection, Synapse
    from btorch.models.kernels import KernelOptions
    from btorch.models.neurons import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import AlphaPSC, HeterSynapsePSC

    connection = SparseConnection.from_edges(
        torch.tensor([0, 0, 1, 2]),
        torch.tensor([1, 2, 0, 1]),
        3,
        3,
        Synapse(
            delay=torch.tensor([0, 2, 1, 2]),
            receptor=torch.tensor([0, 1, 0, 1]),
            n_delay=3,
            n_receptor=2,
        ),
        values=torch.tensor([0.3, -0.4, 0.2, 0.6], dtype=torch.float32),
    )
    synapse = HeterSynapsePSC(3, 2, None, connection, base_psc=AlphaPSC, tau_syn=2.0)
    model = RecurrentNN(
        LIF(3, v_threshold=0.7, trainable_param={"tau", "v_threshold"}),
        synapse,
        backend=KernelOptions(cell=backend, connection="triton"),
        update_state_names=("neuron.v", "synapse.base_psc.psc"),
    ).to(device="cuda", dtype=torch.float32)
    model.neuron.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    synapse.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    synapse.base_psc.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    # The initial delay ring is an incoming state, not a constant padding zero.
    # Its cotangent must survive sparse transpose and the paired cell pipeline.
    initial_ring = torch.randn_like(synapse.delayed_spike.history).requires_grad_()
    reference_ring = initial_ring.detach().clone().requires_grad_()
    synapse.delayed_spike.history = initial_ring
    reference.synapse.delayed_spike.history = reference_ring
    x = torch.randn(6, 2, 3, device="cuda").requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        actual, records = model(x)
        expected, expected_records = reference(reference_x)
    torch.testing.assert_close(actual, expected)
    for name in records:
        torch.testing.assert_close(
            records[name], expected_records[name], atol=5e-6, rtol=5e-5
        )
    actual_loss = actual.sum() + records["neuron.v"].square().sum()
    actual_loss = actual_loss + records["synapse.base_psc.psc"].square().sum()
    expected_loss = expected.sum() + expected_records["neuron.v"].square().sum()
    expected_loss = (
        expected_loss + expected_records["synapse.base_psc.psc"].square().sum()
    )
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=5e-5, rtol=5e-4)
    torch.testing.assert_close(
        initial_ring.grad, reference_ring.grad, atol=5e-5, rtol=5e-4
    )
    for (name, parameter), (_, trusted) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(
            parameter.grad, trusted.grad, atol=5e-5, rtol=5e-4, msg=name
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neuron_model", ("if", "lif", "glif3", "izhikevich"))
@pytest.mark.parametrize(
    "psc_model", ("exponential", "alpha", "alpha_billeh", "dual_exponential")
)
def test_recurrent_all_model_pairs_state_record_and_parameter_gradients(
    backend, neuron_model, psc_model
):
    _available(backend)
    import copy

    from btorch.models import environ
    from btorch.models.kernels import KernelOptions
    from btorch.models.neurons.glif import GLIF3
    from btorch.models.neurons.izhikevich import Izhikevich
    from btorch.models.neurons.lif import IF, LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import (
        AlphaPSC,
        AlphaPSCBilleh,
        DualExponentialPSC,
        ExponentialPSC,
    )

    torch.manual_seed(391)
    neuron_template = get_template(CellSpec(neuron_model))
    trainable = set(neuron_template.parameter_names)
    if neuron_model == "glif3":
        trainable.add("_v_rest")
        neuron = GLIF3(
            3,
            v_threshold=1.0,
            v_reset=0.0,
            v_rest=0.1,
            c_m=1.2,
            tau=4.0,
            k=[0.2, 0.3],
            asc_amps=[-0.2, -0.1],
            trainable_param=trainable,
        )
    elif neuron_model == "izhikevich":
        neuron = Izhikevich(
            3,
            v_threshold=1.0,
            v_reset=0.0,
            v_rest=0.1,
            v_peak=1.5,
            c_m=1.2,
            k=0.2,
            a=0.1,
            b=0.2,
            d=0.1,
            trainable_param=trainable,
        )
    else:
        neuron = (IF if neuron_model == "if" else LIF)(
            3,
            c_m=1.2,
            tau=4.0,
            trainable_param=trainable,
        )
    linear = torch.nn.Linear(3, 3, bias=True)
    linear.weight.data.mul_(0.1)
    linear.bias.data.zero_()
    with environ.context(dt=1.0):
        if psc_model == "dual_exponential":
            synapse = DualExponentialPSC(3, 5.0, 2.0, linear, A=1.0)
        else:
            synapse = {
                "alpha": AlphaPSC,
                "alpha_billeh": AlphaPSCBilleh,
                "exponential": ExponentialPSC,
            }[psc_model](3, 3.0, linear)
    psc_template = get_template(CellSpec(psc_model))
    state_names = (
        *("neuron." + name for name in neuron_template.state_names),
        *("synapse." + name for name in psc_template.state_names),
    )
    model = RecurrentNN(
        neuron,
        synapse,
        backend=KernelOptions(cell=backend),
        update_state_names=state_names,
    ).to(device="cuda", dtype=torch.float32)
    neuron.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    synapse.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    # Model parameters, initial states and PSC buffers can all be trainable in
    # a scientific fitting workflow. Record losses exercise their joint VJP.
    for name in neuron_template.state_names:
        getattr(neuron, name).requires_grad_()
    for name in psc_template.state_names:
        getattr(synapse, name).requires_grad_()
    for name in psc_template.parameter_names:
        getattr(synapse, name).requires_grad_()
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = (torch.rand(5, 2, 3, device="cuda") * 0.8).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    actual_inputs = (
        x,
        *tuple(model.parameters()),
        *(getattr(neuron, name) for name in neuron_template.state_names),
        *(getattr(synapse, name) for name in psc_template.state_names),
        *(getattr(synapse, name) for name in psc_template.parameter_names),
    )
    expected_inputs = (
        xr,
        *tuple(reference.parameters()),
        *(getattr(reference.neuron, name) for name in neuron_template.state_names),
        *(getattr(reference.synapse, name) for name in psc_template.state_names),
        *(getattr(reference.synapse, name) for name in psc_template.parameter_names),
    )
    with environ.context(dt=1.0):
        actual, records = model(x)
        expected, trusted_records = reference(xr)
    torch.testing.assert_close(actual, expected)
    for name in state_names:
        torch.testing.assert_close(
            records[name], trusted_records[name], atol=1e-5, rtol=1e-4
        )
    actual_loss = actual.sum() + sum(value.square().sum() for value in records.values())
    expected_loss = expected.sum() + sum(
        value.square().sum() for value in trusted_records.values()
    )
    gradients = torch.autograd.grad(actual_loss, actual_inputs, allow_unused=True)
    trusted_gradients = torch.autograd.grad(
        expected_loss, expected_inputs, allow_unused=True
    )
    for index, (value, trusted, initial) in enumerate(
        zip(gradients, trusted_gradients, actual_inputs)
    ):
        if trusted is None:
            trusted = torch.zeros_like(initial)
        if value is None:
            value = torch.zeros_like(initial)
        torch.testing.assert_close(
            value, trusted, atol=1e-4, rtol=1e-3, msg=f"input {index}"
        )
