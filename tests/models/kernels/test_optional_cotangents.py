"""Verify that sequence reverse passes preserve absent output cotangents."""

import inspect

import pytest
import torch

from btorch.models.kernels import CellSpec, HistoryPlan, KernelOptions
from btorch.models.kernels.operators import run_multi_step
from btorch.models.kernels.torch_backend import forward_step
from tests.models.kernels.helpers import GPU_BACKENDS, cell_inputs, require_cuda_backend


MODELS = (
    "if",
    "lif",
    "glif3",
    "izhikevich",
    "alpha",
    "exponential",
    "alpha_billeh",
    "dual_exponential",
)


def _bind_custom_op(operator, args, kwargs):
    """Bind a CustomOpDef call to the named implementation parameters."""
    return inspect.signature(operator._init_fn).bind(*args, **kwargs).arguments


def _inputs(model: str, n_step: int = 4):
    """Reuse the shared cell layouts with a time-major drive for cotangent
    tests."""
    spec, _, states, parameters = cell_inputs(model)
    x_seq = (torch.randn(n_step, 2, 3) * 0.3 + 0.4).requires_grad_()
    return spec, x_seq, states, parameters


def _clone_inputs(x_seq, states, parameters):
    return (
        x_seq.detach().clone().requires_grad_(x_seq.requires_grad),
        tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            for value in states
        ),
        tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            for value in parameters
        ),
    )


def _reference(x_seq, states, parameters, spec, requested=()):
    """Run ordinary differentiable steps as the independent gradient oracle."""
    current = states
    outputs = []
    records = {name: [] for name in requested}
    for x_step in x_seq:
        output, current, step_records = forward_step(x_step, current, parameters, spec)
        outputs.append(output)
        for name in requested:
            records[name].append(step_records[name])
    output_seq = torch.stack(outputs) if outputs else x_seq * 0
    histories = tuple(
        torch.stack(records[name]) if records[name] else x_seq.new_empty(x_seq.shape)
        for name in requested
    )
    return (output_seq, *current, *histories)


def _gradients(loss, inputs):
    return torch.autograd.grad(loss, inputs, allow_unused=True)


def _assert_gradients(actual, expected, inputs, *, atol=3e-6, rtol=3e-5):
    for value, reference, source in zip(actual, expected, inputs):
        torch.testing.assert_close(
            torch.zeros_like(source) if value is None else value,
            torch.zeros_like(source) if reference is None else reference,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("loss_kind", ("output", "final"))
def test_output_and_final_only_gradients_match_step_reference(model, loss_kind):
    # Final-only losses deliberately omit grad_output. Output-only losses omit
    # every public final-state cotangent, exercising both optional directions.
    torch.manual_seed(110)
    spec, x_seq, states, parameters = _inputs(model)
    ref_x, ref_states, ref_parameters = _clone_inputs(x_seq, states, parameters)
    actual = run_multi_step(x_seq, states, parameters, spec)
    expected = _reference(ref_x, ref_states, ref_parameters, spec)
    n_states = len(states)
    if loss_kind == "output":
        actual_loss = actual[0].sum()
        expected_loss = expected[0].sum()
    else:
        actual_loss = sum(value.square().sum() for value in actual[1 : 1 + n_states])
        expected_loss = sum(
            value.square().sum() for value in expected[1 : 1 + n_states]
        )
    actual_inputs = (x_seq, *states, *parameters)
    expected_inputs = (ref_x, *ref_states, *ref_parameters)
    _assert_gradients(
        _gradients(actual_loss, actual_inputs),
        _gradients(expected_loss, expected_inputs),
        actual_inputs,
    )


def test_partial_history_keeps_other_cotangents_absent(monkeypatch):
    # Izhikevich exposes two charged records. A loss on only v_charged must
    # leave the output, final states, v history, and u histories absent.
    from btorch.models.kernels import operators

    torch.manual_seed(111)
    spec, x_seq, states, parameters = _inputs("izhikevich")
    ref_x, ref_states, ref_parameters = _clone_inputs(x_seq, states, parameters)
    original_reverse = operators._reverse
    observed = {}

    def capture(*args, **kwargs):
        arguments = _bind_custom_op(original_reverse, args, kwargs)
        observed["grad_output"] = arguments["grad_output"]
        observed["grad_final"] = arguments["grad_final"]
        observed["grad_history"] = arguments["grad_history"]
        return original_reverse(*args, **kwargs)

    monkeypatch.setattr(operators, "_reverse", capture)
    requested = ("v", "v_charged")
    actual = run_multi_step(
        x_seq, states, parameters, spec, HistoryPlan(requested=requested)
    )
    expected = _reference(ref_x, ref_states, ref_parameters, spec, requested=requested)
    actual_loss = actual[-1].square().sum()
    expected_loss = expected[-1].square().sum()
    actual_inputs = (x_seq, *states, *parameters)
    expected_inputs = (ref_x, *ref_states, *ref_parameters)
    _assert_gradients(
        _gradients(actual_loss, actual_inputs),
        _gradients(expected_loss, expected_inputs),
        actual_inputs,
    )
    assert observed["grad_output"] is None
    assert all(value is None for value in observed["grad_final"])
    # Training saves v, v_charged, and the unrequested u state history.
    assert [value is not None for value in observed["grad_history"]] == [
        False,
        True,
        False,
    ]


def test_encoder_gradient_and_primary_psc_residual_stay_optional(monkeypatch):
    # The encoder leaf is outside the custom op while all cell tensors are
    # frozen. This verifies the missing output/history paths still propagate
    # the final-h loss back through the encoded input.
    from btorch.models.kernels import operators

    torch.manual_seed(112)
    spec, base_x, states, parameters = _inputs("alpha")
    raw = base_x.detach().clone().requires_grad_()
    ref_raw = raw.detach().clone().requires_grad_()
    states = tuple(value.detach() for value in states)
    parameters = tuple(value.detach() for value in parameters)
    ref_states = tuple(value.clone() for value in states)
    ref_parameters = tuple(value.clone() for value in parameters)
    original_reverse = operators._reverse
    observed = {}

    def capture(*args, **kwargs):
        arguments = _bind_custom_op(original_reverse, args, kwargs)
        observed["grad_output"] = arguments["grad_output"]
        observed["grad_final"] = arguments["grad_final"]
        observed["grad_history"] = arguments["grad_history"]
        return original_reverse(*args, **kwargs)

    monkeypatch.setattr(operators, "_reverse", capture)
    actual = run_multi_step(raw.sigmoid(), states, parameters, spec)
    expected = _reference(ref_raw.sigmoid(), ref_states, ref_parameters, spec)
    torch.testing.assert_close(
        torch.autograd.grad(actual[2].square().sum(), raw)[0],
        torch.autograd.grad(expected[2].square().sum(), ref_raw)[0],
        atol=3e-6,
        rtol=3e-5,
    )
    assert observed["grad_output"] is None
    assert [value is not None for value in observed["grad_final"]] == [False, True]
    # h is saved for recurrence and the primary psc output is appended as a
    # residual alias; neither has a direct history loss here.
    assert observed["grad_history"] == [None, None]


def test_fullgraph_final_only_backward_accepts_optional_cotangents():
    torch.manual_seed(113)
    spec, x_seq, states, parameters = _inputs("alpha")

    def final_h(inp, initial, cell_parameters):
        return run_multi_step(inp, initial, cell_parameters, spec)[2]

    eager = final_h(x_seq, states, parameters)
    eager_grad = torch.autograd.grad(
        eager.square().sum(), (x_seq, *states, *parameters)
    )
    compiled_x, compiled_states, compiled_parameters = _clone_inputs(
        x_seq, states, parameters
    )
    compiled = torch.compile(final_h, backend="aot_eager", fullgraph=True)
    compiled_result = compiled(compiled_x, compiled_states, compiled_parameters)
    compiled_grad = torch.autograd.grad(
        compiled_result.square().sum(),
        (compiled_x, *compiled_states, *compiled_parameters),
    )
    torch.testing.assert_close(compiled_result, eager)
    for value, reference in zip(compiled_grad, eager_grad):
        torch.testing.assert_close(value, reference, atol=3e-6, rtol=3e-5)


def test_empty_sequence_final_state_gradients():
    # T=0 bypasses the reverse loop: final states are the initial states and
    # optional output/history cotangents must not disturb that identity VJP.
    torch.manual_seed(114)
    spec, x_seq, states, parameters = _inputs("lif", n_step=0)
    result = run_multi_step(x_seq, states, parameters, spec)
    gradients = torch.autograd.grad(
        result[1].sum(), (x_seq, *states, *parameters), allow_unused=True
    )
    torch.testing.assert_close(gradients[0], torch.zeros_like(x_seq))
    torch.testing.assert_close(gradients[1], torch.ones_like(states[0]))
    for value, parameter in zip(gradients[2:], parameters):
        torch.testing.assert_close(value, torch.zeros_like(parameter))


def test_external_backend_receives_dense_cotangents():
    from btorch.models.kernels import register_cell_backend
    from btorch.models.kernels.torch_backend import (
        backward_sequence,
        forward_sequence,
    )

    seen = {}

    def backward(*args, **kwargs):
        seen["dense"] = (
            args[6] is not None
            and all(value is not None for value in args[7])
            and all(value is not None for value in args[8])
        )
        return backward_sequence(*args, **kwargs)

    register_cell_backend(
        "optional_cotangent_external",
        forward=forward_sequence,
        backward=backward,
        replace=True,
    )
    torch.manual_seed(115)
    spec, x_seq, states, parameters = _inputs("lif")
    result = run_multi_step(
        x_seq,
        states,
        parameters,
        spec,
        backend="optional_cotangent_external",
    )
    gradients = torch.autograd.grad(
        result[1].square().sum(), (x_seq, *states, *parameters)
    )
    assert seen["dense"]
    assert all(torch.isfinite(value).all() for value in gradients)


def test_outstanding_forwards_preserve_their_compilation_policies(monkeypatch):
    from btorch.models.kernels import operators

    torch.manual_seed(116)
    spec, false_x, false_states, false_parameters = _inputs("lif")
    true_x, true_states, true_parameters = _clone_inputs(
        false_x, false_states, false_parameters
    )
    ref_x, ref_states, ref_parameters = _clone_inputs(
        false_x, false_states, false_parameters
    )
    original_reverse = operators._reverse
    observed = []

    def capture(*args, **kwargs):
        arguments = _bind_custom_op(original_reverse, args, kwargs)
        observed.append((arguments["enable_fp_fusion"], arguments["precise_division"]))
        return original_reverse(*args, **kwargs)

    monkeypatch.setattr(operators, "_reverse", capture)
    false_result = run_multi_step(
        false_x,
        false_states,
        false_parameters,
        spec,
        backend=KernelOptions(
            neuron="torch", enable_fp_fusion=False, precise_division=True
        ),
    )
    true_result = run_multi_step(
        true_x,
        true_states,
        true_parameters,
        spec,
        backend=KernelOptions(neuron="torch", enable_fp_fusion=True),
    )
    reference = _reference(ref_x, ref_states, ref_parameters, spec)
    cotangent = torch.randn_like(false_result[1])
    true_inputs = (true_x, *true_states, *true_parameters)
    false_inputs = (false_x, *false_states, *false_parameters)
    reference_inputs = (ref_x, *ref_states, *ref_parameters)
    reference_gradients = _gradients((reference[1] * cotangent).sum(), reference_inputs)
    # Reverse in the opposite order from forward. Each saved autograd context
    # must retain its own compile policy rather than reading a later selection.
    _assert_gradients(
        _gradients((true_result[1] * cotangent).sum(), true_inputs),
        reference_gradients,
        true_inputs,
    )
    ref_x2, ref_states2, ref_parameters2 = _clone_inputs(
        false_x, false_states, false_parameters
    )
    reference2 = _reference(ref_x2, ref_states2, ref_parameters2, spec)
    reference_inputs2 = (ref_x2, *ref_states2, *ref_parameters2)
    _assert_gradients(
        _gradients((false_result[1] * cotangent).sum(), false_inputs),
        _gradients((reference2[1] * cotangent).sum(), reference_inputs2),
        false_inputs,
    )
    assert observed == [(True, False), (False, True)]


def test_external_backend_rejects_nondefault_compile_policy_before_dispatch():
    from btorch.models.kernels import register_cell_backend
    from btorch.models.kernels.torch_backend import (
        backward_sequence,
        forward_sequence,
    )

    called = []

    def forward(*args, **kwargs):
        called.append("forward")
        return forward_sequence(*args, **kwargs)

    register_cell_backend(
        "compile_policy_external",
        forward=forward,
        backward=backward_sequence,
        replace=True,
    )
    spec, x_seq, states, parameters = _inputs("lif")
    for options in (
        KernelOptions(neuron="compile_policy_external", enable_fp_fusion=False),
        KernelOptions(neuron="compile_policy_external", precise_division=True),
    ):
        with pytest.raises(NotImplementedError, match="built-in cell backends"):
            run_multi_step(
                x_seq,
                states,
                parameters,
                spec,
                backend=options,
            )
    assert called == []


def test_recurrent_capture_version_includes_compilation_policies():
    from btorch.models.neurons.lif import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import AlphaPSC

    model = RecurrentNN(
        LIF(3),
        AlphaPSC(3, 2.0, torch.nn.Identity()),
        backend=KernelOptions(
            neuron="torch", enable_fp_fusion=True, precise_division=False
        ),
    )
    default_version = model.capture_version
    model.backend = KernelOptions(
        neuron="torch", enable_fp_fusion=False, precise_division=True
    )
    precise_version = model.capture_version
    assert default_version != precise_version
    assert default_version[3:5] == (True, False)
    assert precise_version[3:5] == (False, True)


def _cuda_lif_inputs(n_step: int = 8):
    """Use scalar, neuron, strided, and batch-broadcast parameter layouts."""
    spec = CellSpec("lif", dt=0.5, hard_reset=True)
    x_seq = (torch.rand(n_step, 2, 5, device="cuda") * 0.9 + 0.15).requires_grad_()
    states = ((torch.rand(2, 5, device="cuda") * 0.3).requires_grad_(),)
    c_m_storage = torch.linspace(0.8, 1.6, 11, device="cuda")
    parameters = (
        torch.tensor(0.55, device="cuda").requires_grad_(),
        torch.linspace(-0.1, 0.05, 5, device="cuda").requires_grad_(),
        c_m_storage[1::2].detach().requires_grad_(),
        torch.tensor([[2.5], [4.0]], device="cuda").requires_grad_(),
    )
    return spec, x_seq, states, parameters


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("loss_kind", ("output", "final"))
def test_native_optional_output_and_final_gradients_with_broadcast_parameters(
    backend, loss_kind, monkeypatch
):
    require_cuda_backend(backend)
    from btorch.models.kernels import operators

    torch.manual_seed(210)
    spec, x_seq, states, parameters = _cuda_lif_inputs()
    ref_x, ref_states, ref_parameters = _clone_inputs(x_seq, states, parameters)
    expected = _reference(ref_x, ref_states, ref_parameters, spec)
    cotangent = torch.randn_like(expected[0] if loss_kind == "output" else expected[1])
    original_reverse = operators._reverse
    original_zeros_like = torch.zeros_like
    observed = {}
    zero_like_shapes = []

    def capture(*args, **kwargs):
        arguments = _bind_custom_op(original_reverse, args, kwargs)
        observed["grad_output"] = arguments["grad_output"]
        observed["grad_final"] = arguments["grad_final"]
        observed["grad_history"] = arguments["grad_history"]
        return original_reverse(*args, **kwargs)

    def record_zeros_like(value, *args, **kwargs):
        zero_like_shapes.append(tuple(value.shape))
        return original_zeros_like(value, *args, **kwargs)

    monkeypatch.setattr(operators, "_reverse", capture)
    monkeypatch.setattr(torch, "zeros_like", record_zeros_like)
    actual = run_multi_step(x_seq, states, parameters, spec, backend=backend)
    actual_target = actual[0] if loss_kind == "output" else actual[1]
    expected_target = expected[0] if loss_kind == "output" else expected[1]
    actual_inputs = (x_seq, *states, *parameters)
    expected_inputs = (ref_x, *ref_states, *ref_parameters)
    _assert_gradients(
        _gradients((actual_target * cotangent).sum(), actual_inputs),
        _gradients((expected_target * cotangent).sum(), expected_inputs),
        actual_inputs,
        atol=5e-5,
        rtol=5e-4,
    )
    # The old operator path allocated every absent output/final/history
    # cotangent with zeros_like before entering the native engine.
    assert zero_like_shapes == []
    if loss_kind == "output":
        assert observed["grad_output"] is not None
        assert observed["grad_final"] == [None]
    else:
        assert observed["grad_output"] is None
        assert observed["grad_final"][0] is not None
    assert all(value is None for value in observed["grad_history"])
    # The chosen drive crosses threshold but is not saturated; a silent or
    # always-spiking trajectory could hide a broken surrogate VJP.
    assert actual[0].amin() == 0 and actual[0].amax() == 1


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_native_partial_charged_record_loss(backend):
    require_cuda_backend(backend)
    torch.manual_seed(211)
    spec, x_seq, states, parameters = _inputs("izhikevich", n_step=7)
    x_seq = x_seq.detach().cuda().requires_grad_()
    states = tuple(value.detach().cuda().requires_grad_() for value in states)
    parameters = tuple(value.detach().cuda().requires_grad_() for value in parameters)
    ref_x, ref_states, ref_parameters = _clone_inputs(x_seq, states, parameters)
    requested = ("v", "v_charged")
    actual = run_multi_step(
        x_seq,
        states,
        parameters,
        spec,
        HistoryPlan(requested=requested),
        backend=backend,
    )
    expected = _reference(ref_x, ref_states, ref_parameters, spec, requested=requested)
    cotangent = torch.randn_like(actual[-1])
    actual_inputs = (x_seq, *states, *parameters)
    expected_inputs = (ref_x, *ref_states, *ref_parameters)
    _assert_gradients(
        _gradients((actual[-1] * cotangent).sum(), actual_inputs),
        _gradients((expected[-1] * cotangent).sum(), expected_inputs),
        actual_inputs,
        atol=5e-5,
        rtol=5e-4,
    )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_native_frozen_cell_propagates_final_loss_to_encoder(backend):
    require_cuda_backend(backend)
    torch.manual_seed(212)
    spec, base_x, states, parameters = _inputs("alpha", n_step=7)
    raw = base_x.detach().cuda().requires_grad_()
    ref_raw = raw.detach().clone().requires_grad_()
    states = tuple(value.detach().cuda() for value in states)
    parameters = tuple(value.detach().cuda() for value in parameters)
    ref_states = tuple(value.clone() for value in states)
    ref_parameters = tuple(value.clone() for value in parameters)
    actual = run_multi_step(raw.sigmoid(), states, parameters, spec, backend=backend)
    expected = _reference(ref_raw.sigmoid(), ref_states, ref_parameters, spec)
    cotangent = torch.randn_like(actual[2])
    torch.testing.assert_close(
        torch.autograd.grad((actual[2] * cotangent).sum(), raw)[0],
        torch.autograd.grad((expected[2] * cotangent).sum(), ref_raw)[0],
        atol=5e-5,
        rtol=5e-4,
    )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_native_empty_sequence_final_state_gradient(backend):
    require_cuda_backend(backend)
    torch.manual_seed(213)
    spec, x_seq, states, parameters = _cuda_lif_inputs(n_step=0)
    result = run_multi_step(x_seq, states, parameters, spec, backend=backend)
    cotangent = torch.randn_like(result[1])
    gradients = torch.autograd.grad(
        (result[1] * cotangent).sum(),
        (x_seq, *states, *parameters),
        allow_unused=True,
    )
    torch.testing.assert_close(gradients[0], torch.zeros_like(x_seq))
    torch.testing.assert_close(gradients[1], cotangent)
    for value, parameter in zip(gradients[2:], parameters):
        torch.testing.assert_close(value, torch.zeros_like(parameter))


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_native_compiled_final_only_backward_replays_in_cuda_graph(backend):
    require_cuda_backend(backend)
    torch.manual_seed(214)
    spec, base_x, states, parameters = _inputs("alpha", n_step=6)
    states = tuple(value.detach().cuda() for value in states)
    parameters = tuple(value.detach().cuda() for value in parameters)

    def final_h(value):
        return run_multi_step(
            value.sigmoid(), states, parameters, spec, backend=backend
        )[2]

    compiled = torch.compile(final_h, backend="eager", fullgraph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Create and warm the capture leaf on the capture stream. This keeps its
    # autograd metadata independent of eager work performed on the default
    # stream and avoids a forbidden cross-stream dependency during capture.
    with torch.cuda.stream(stream):
        static_x = base_x.detach().cuda().requires_grad_()
        cotangent = torch.randn_like(states[1])
        for _ in range(2):
            warm = compiled(static_x)
            torch.autograd.grad((warm * cotangent).sum(), static_x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = compiled(static_x)
        captured_gradient = torch.autograd.grad((captured * cotangent).sum(), static_x)[
            0
        ]
    graph.replay()

    reference_x = static_x.detach().clone().requires_grad_()
    reference = _reference(
        reference_x.sigmoid(),
        tuple(value.clone() for value in states),
        tuple(value.clone() for value in parameters),
        spec,
    )[2]
    expected_gradient = torch.autograd.grad((reference * cotangent).sum(), reference_x)[
        0
    ]
    torch.testing.assert_close(captured, reference, atol=5e-6, rtol=5e-5)
    torch.testing.assert_close(
        captured_gradient, expected_gradient, atol=5e-5, rtol=5e-4
    )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_native_disabled_fp_fusion_matches_step_reference(backend):
    require_cuda_backend(backend)
    torch.manual_seed(215)
    spec, x_seq, states, parameters = _inputs("izhikevich", n_step=9)
    x_seq = x_seq.detach().cuda().requires_grad_()
    states = tuple(value.detach().cuda().requires_grad_() for value in states)
    parameters = tuple(value.detach().cuda().requires_grad_() for value in parameters)
    ref_x, ref_states, ref_parameters = _clone_inputs(x_seq, states, parameters)
    actual = run_multi_step(
        x_seq,
        states,
        parameters,
        spec,
        HistoryPlan(requested=("v",)),
        backend=KernelOptions(neuron=backend, enable_fp_fusion=False),
    )
    expected = _reference(ref_x, ref_states, ref_parameters, spec, requested=("v",))
    cotangents = tuple(torch.randn_like(value) for value in actual)
    actual_inputs = (x_seq, *states, *parameters)
    expected_inputs = (ref_x, *ref_states, *ref_parameters)
    _assert_gradients(
        _gradients(
            sum((value * grad).sum() for value, grad in zip(actual, cotangents)),
            actual_inputs,
        ),
        _gradients(
            sum((value * grad).sum() for value, grad in zip(expected, cotangents)),
            expected_inputs,
        ),
        actual_inputs,
        atol=5e-5,
        rtol=5e-4,
    )
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=5e-6, rtol=5e-5)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_recurrent_compile_policy_fullgraph_and_capture_invalidation(backend):
    require_cuda_backend(backend)
    import copy

    from btorch.models import environ
    from btorch.models.neurons.lif import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import AlphaPSC

    torch.manual_seed(216)
    model = RecurrentNN(
        LIF(3, v_threshold=0.65, v_reset=0.0, c_m=1.1, tau=3.0),
        AlphaPSC(3, 2.0, torch.nn.Identity()),
        backend=KernelOptions(
            neuron=backend, enable_fp_fusion=False, precise_division=True
        ),
        update_state_names=("neuron.v",),
    ).cuda()
    model.neuron.init_state(batch_size=2, device="cuda", dtype=torch.float32)
    model.synapse.init_state(batch_size=2, device="cuda", dtype=torch.float32)
    model.requires_grad_(False)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = torch.rand(5, 2, 3, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()

    def network(value):
        spikes, records = model(value)
        return spikes.sum() + records["neuron.v"].square().sum()

    with environ.context(dt=1.0):
        actual_loss = torch.compile(network, backend="aot_eager", fullgraph=True)(x)
        expected_spikes, expected_records = reference(xr)
        expected_loss = (
            expected_spikes.sum() + expected_records["neuron.v"].square().sum()
        )
    torch.testing.assert_close(actual_loss, expected_loss, atol=5e-5, rtol=5e-4)
    torch.testing.assert_close(
        torch.autograd.grad(actual_loss, x)[0],
        torch.autograd.grad(expected_loss, xr)[0],
        atol=5e-5,
        rtol=5e-4,
    )

    for layer in (model, reference):
        layer.neuron.reset(batch_size=2, device="cuda", dtype=torch.float32)
        layer.synapse.reset(batch_size=2, device="cuda", dtype=torch.float32)
    model.backend = KernelOptions(
        neuron=backend, enable_fp_fusion=True, precise_division=False
    )
    model.cudagraph = True
    drive = x.detach()
    with torch.no_grad(), environ.context(dt=1.0):
        for _ in range(2):
            actual, records = model(drive)
            expected, expected_records = reference(drive)
            torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
            torch.testing.assert_close(
                records["neuron.v"],
                expected_records["neuron.v"],
                atol=5e-6,
                rtol=5e-5,
            )
        assert model._cudagraph_runner.n_captures == 1
        model.backend = KernelOptions(
            neuron=backend, enable_fp_fusion=False, precise_division=True
        )
        actual, records = model(drive)
        expected, expected_records = reference(drive)
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
        torch.testing.assert_close(
            records["neuron.v"],
            expected_records["neuron.v"],
            atol=5e-6,
            rtol=5e-5,
        )
    assert model._cudagraph_runner.n_captures == 2
