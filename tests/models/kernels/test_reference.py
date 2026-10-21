"""Exercise tensor adapters and operator gradients against existing Python
models."""

import pytest
import torch

from btorch.models.kernels.contracts import CellSpec, HistoryPlan
from btorch.models.kernels.templates import get_template
from btorch.models.kernels.torch_backend import forward_step
from tests.models.kernels.helpers import cell_inputs


@pytest.mark.parametrize(
    "model",
    [
        "if",
        "lif",
        "glif3",
        "izhikevich",
        "alpha",
        "exponential",
        "alpha_billeh",
        "dual_exponential",
    ],
)
def test_reference_matches_existing_single_step(model):
    # Existing stateful modules are an independent specification of update
    # ordering, especially the input-updated h and charged-voltage u stages.
    from btorch.models import environ
    from btorch.models.neurons.glif import GLIF3
    from btorch.models.neurons.izhikevich import Izhikevich
    from btorch.models.neurons.lif import IF, LIF
    from btorch.models.synapse import (
        AlphaPSC,
        AlphaPSCBilleh,
        DualExponentialPSC,
        ExponentialPSC,
    )

    torch.manual_seed(31)
    spec, x, states, parameters = cell_inputs(model)
    template = get_template(spec)
    values = dict(zip(template.parameter_names, parameters))
    cls = {
        "if": IF,
        "lif": LIF,
        "glif3": GLIF3,
        "izhikevich": Izhikevich,
        "alpha": AlphaPSC,
        "exponential": ExponentialPSC,
        "alpha_billeh": AlphaPSCBilleh,
        "dual_exponential": DualExponentialPSC,
    }[model]
    constructor = values.copy()
    if model in ("alpha", "exponential", "alpha_billeh", "dual_exponential"):
        constructor["linear"] = torch.nn.Identity()
        constructor.pop("syn_decay", None)
        constructor.pop("a", None)
    with environ.context(dt=spec.dt):
        module = cls(3, **constructor)
    module.init_state(batch_size=2)
    for name, value in zip(template.state_names, states):
        setattr(module, name, value)
    # Billeh caches its decay, and DualExponential computes normalization in
    # the constructor. Explicitly match those actually participating tensors.
    for name, value in values.items():
        if name in ("syn_decay", "a") and model in ("alpha_billeh", "dual_exponential"):
            setattr(module, name, value)
    with environ.context(dt=spec.dt):
        expected = module.single_step_forward(x)
    actual, final, _ = forward_step(x, states, parameters, spec)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    for name, value in zip(template.state_names, final):
        torch.testing.assert_close(value, getattr(module, name), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize(
    "model", ["lif", "glif3", "izhikevich", "alpha", "dual_exponential"]
)
@pytest.mark.parametrize("n_step", [0, 1, 7])
def test_sequence_operator_gradients_and_record_losses(model, n_step):
    from btorch.models.kernels.operators import run_multi_step

    torch.manual_seed(12)
    spec, x, states, parameters = cell_inputs(model)
    x_seq = torch.randn(n_step, *x.shape, requires_grad=True)
    template = get_template(spec)
    requested = template.history_names
    actual = run_multi_step(
        x_seq, states, parameters, spec, HistoryPlan(requested=requested)
    )
    current = states
    outputs, records = [], {name: [] for name in requested}
    for x_step in x_seq:
        out, current, step_records = forward_step(x_step, current, parameters, spec)
        outputs.append(out)
        for name in requested:
            records[name].append(step_records[name])
    expected_out = torch.stack(outputs) if outputs else x_seq * 0
    expected = (expected_out, *current)
    if n_step:
        expected = (*expected, *(torch.stack(records[name]) for name in requested))
    else:
        expected = (*expected, *actual[1 + len(states) :])
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
    cotangents = tuple(torch.randn_like(value) for value in actual)
    inputs = (x_seq, *states, *parameters)
    actual_grad = torch.autograd.grad(
        sum((value * grad).sum() for value, grad in zip(actual, cotangents)),
        inputs,
        allow_unused=True,
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(
        sum((value * grad).sum() for value, grad in zip(expected, cotangents)),
        inputs,
        allow_unused=True,
    )
    for value, reference, inp in zip(actual_grad, expected_grad, inputs):
        torch.testing.assert_close(
            torch.zeros_like(inp) if value is None else value,
            torch.zeros_like(inp) if reference is None else reference,
            atol=3e-6,
            rtol=3e-5,
        )


@pytest.mark.parametrize("compiler", ["eager", "aot_eager"])
def test_static_fullgraph_forward_backward_and_chunk_continuation(compiler):
    from btorch.models.kernels.operators import run_multi_step

    spec, x, states, parameters = cell_inputs("lif")
    x_seq = torch.randn(6, *x.shape, requires_grad=True)
    history = HistoryPlan(requested=("v", "v_charged"))

    def sequence(inp, initial, params):
        return run_multi_step(inp, initial, params, spec, history)

    # Eager backend still exercises Dynamo's fullgraph capture and fake/meta
    # registration; numerical Inductor/CUDA capture belongs to GPU tests.
    compiled = torch.compile(sequence, backend=compiler, fullgraph=True)
    actual = compiled(x_seq, states, parameters)
    expected = sequence(x_seq, states, parameters)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
    actual_grad = torch.autograd.grad(
        sum(value.sum() for value in actual), (x_seq, *states, *parameters)
    )
    expected_grad = torch.autograd.grad(
        sum(value.sum() for value in expected), (x_seq, *states, *parameters)
    )
    for value, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(value, reference)
    first = sequence(x_seq[:3], states, parameters)
    second = sequence(x_seq[3:], (first[1],), parameters)
    torch.testing.assert_close(torch.cat((first[0], second[0])), expected[0])
    torch.testing.assert_close(second[1], expected[1])
    # The first call's saved history must survive the next call. Different
    # losses on both chunks accumulate through the undetached boundary state.
    chunk_grad = torch.autograd.grad(
        first[0].sum() + second[0].sum() + second[1].sum(), x_seq
    )[0]
    fresh = sequence(x_seq, states, parameters)
    full_grad = torch.autograd.grad(fresh[0].sum() + fresh[1].sum(), x_seq)[0]
    torch.testing.assert_close(chunk_grad, full_grad)


@pytest.mark.parametrize("trainable", ["input", "state", "parameter", "none"])
@pytest.mark.parametrize("grad_mode", ["enabled", "no_grad", "inference"])
def test_dispatch_tracks_actual_inputs_and_retains_only_needed_history(
    trainable, grad_mode
):
    from contextlib import nullcontext

    from btorch.models.kernels import register_cell_backend
    from btorch.models.kernels.operators import run_multi_step
    from btorch.models.kernels.torch_backend import backward_sequence, forward_sequence

    requested_names = []

    def forward(x, initial, parameters, spec, history_names):
        requested_names.append(history_names)
        return forward_sequence(x, initial, parameters, spec, history_names)

    register_cell_backend(
        "reference_dispatch", forward=forward, backward=backward_sequence, replace=True
    )
    spec, x, states, parameters = cell_inputs("alpha")
    x_seq = torch.randn(3, *x.shape).requires_grad_(trainable == "input")
    states = tuple(
        value.detach().requires_grad_(trainable == "state") for value in states
    )
    parameters = tuple(
        value.detach().requires_grad_(trainable == "parameter") for value in parameters
    )
    context = {
        "enabled": nullcontext,
        "no_grad": torch.no_grad,
        "inference": torch.inference_mode,
    }[grad_mode]
    # The requested h history is available in inference too. Only training
    # needs psc temporal history, reusing the FP32 output storage instead of
    # allocating another sequence even when parameters require gradients.
    with context():
        result = run_multi_step(
            x_seq,
            states,
            parameters,
            spec,
            HistoryPlan(requested=("h",)),
            backend="reference_dispatch",
        )
    training = trainable != "none" and grad_mode == "enabled"
    assert result[0].requires_grad == training
    assert requested_names[-1] == ("h",)
    if training:
        inputs = [
            value for value in (x_seq, *states, *parameters) if value.requires_grad
        ]
        gradients = torch.autograd.grad(
            result[0].square().sum() + result[-1].square().sum(), inputs
        )
        assert all(torch.isfinite(value).all() for value in gradients)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_state_accumulation_and_multidimensional_batch(dtype):
    from btorch.models.kernels.operators import run_multi_step

    # FP32 states retain small currents over a long trajectory. Two batch axes
    # show that the pure tensor operator preserves axes without flattening them.
    x = torch.full((100, 2, 3, 4), 0.002, dtype=dtype, requires_grad=True)
    states = (torch.zeros(2, 3, 4, dtype=dtype, requires_grad=True),)
    params = (
        torch.full((4,), 3.0),
        torch.zeros(4),
        torch.ones(4),
        torch.full((4,), 20.0),
    )
    spec = CellSpec("lif")
    history = HistoryPlan(requested=("v",))
    result = run_multi_step(x, states, params, spec, history)
    reference = run_multi_step(
        x.detach().float().requires_grad_(),
        tuple(value.detach().float() for value in states),
        params,
        spec,
        history,
    )
    torch.testing.assert_close(result[0].float(), reference[0], rtol=0, atol=0)
    torch.testing.assert_close(result[1], reference[1], rtol=0, atol=0)
    torch.testing.assert_close(result[2], reference[2], rtol=0, atol=0)
    assert result[1].dtype == torch.float32
    gradients = torch.autograd.grad(result[0].sum() + result[1].sum(), (x, *states))
    assert gradients[0].dtype == dtype and all(
        torch.isfinite(value).all() for value in gradients
    )


def test_psc_output_history_and_autograd_share_one_storage():
    from btorch.models.kernels.operators import run_multi_step

    spec, x, states, parameters = cell_inputs("alpha")
    x_seq = torch.randn(5, *x.shape, requires_grad=True)
    saved = []
    # The same post-step PSC record participates in both an output loss and a
    # state-record loss. Neither path should create a second PSC residual.
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append(value) or value, lambda value: value
    ):
        result = run_multi_step(
            x_seq, states, parameters, spec, HistoryPlan(requested=("psc", "h"))
        )
    assert result[0].data_ptr() == result[-2].data_ptr()
    assert sum(value.data_ptr() == result[-2].data_ptr() for value in saved) == 1
    loss = result[0].sum() + result[-2].square().sum() + result[-1].sum()
    gradients = torch.autograd.grad(loss, (x_seq, *states, *parameters))
    assert all(torch.isfinite(value).all() for value in gradients)


@pytest.mark.parametrize("n_step", [0, 6])
def test_external_model_declares_history_shapes_and_reference_derivatives(n_step):
    from btorch.models.kernels import (
        ModelTemplate,
        register_model_template,
        run_multi_step,
    )

    # An external lab model supplies a trusted recurrence and an explicit VJP;
    # its named auxiliary record declares the state that supplies its shape.
    template = ModelTemplate(
        "lab_integrator",
        ("q",),
        ("decay",),
        ("q", "charged"),
        history_sources=(("charged", "q"),),
    )

    def step(x, states, parameters, spec):
        q = parameters[0] * states[0] + x
        return q, (q,), {"q": q, "charged": q}

    def reverse(x, states, parameters, spec, grad_output, grad_states, grad_records):
        grad_q = (
            grad_output
            + grad_states[0]
            + grad_records.get("q", 0)
            + grad_records.get("charged", 0)
        )
        return (
            grad_q,
            (grad_q * parameters[0],),
            ((grad_q * states[0]).sum_to_size(parameters[0].shape),),
        )

    register_model_template(
        template, forward_step=step, backward_step=reverse, replace=True
    )
    x = torch.randn(n_step, 2, 4, requires_grad=True)
    q_init = torch.randn(2, 4, requires_grad=True)
    decay = torch.tensor(0.8, requires_grad=True)
    actual = run_multi_step(
        x,
        (q_init,),
        (decay,),
        CellSpec(template.name),
        HistoryPlan(requested=("charged",)),
    )
    q, expected_steps = q_init, []
    for drive in x:
        q = decay * q + drive
        expected_steps.append(q)
    expected_output = torch.stack(expected_steps) if n_step else x * 0
    torch.testing.assert_close(actual[0], expected_output)
    torch.testing.assert_close(actual[1], q)
    assert actual[2].shape == x.shape
    loss = actual[0].sum() + actual[1].sum() + actual[2].sum()
    expected_loss = expected_output.sum() * 2 + q.sum()
    gradients = torch.autograd.grad(loss, (x, q_init, decay), allow_unused=True)
    expected_gradients = torch.autograd.grad(
        expected_loss, (x, q_init, decay), allow_unused=True
    )
    for value, reference, source in zip(
        gradients, expected_gradients, (x, q_init, decay)
    ):
        torch.testing.assert_close(
            torch.zeros_like(source) if value is None else value,
            torch.zeros_like(source) if reference is None else reference,
        )


@pytest.mark.parametrize("neuron_model", ["lif", "izhikevich"])
@pytest.mark.parametrize("compiler", [None, "aot_eager"])
def test_recurrent_charged_history_losses_and_multidimensional_states(
    neuron_model, compiler
):
    import copy

    from btorch.models import environ
    from btorch.models.kernels import KernelOptions
    from btorch.models.neurons.izhikevich import Izhikevich
    from btorch.models.neurons.lif import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import AlphaPSC

    # Charged-state records represent the instant before reset. Their losses
    # must enter the charge/adaptation stages, even when reset detaches spikes.
    shape = (2, 3)
    neuron_cls = LIF if neuron_model == "lif" else Izhikevich
    neuron = neuron_cls(
        shape,
        **({"pre_spike_v": True} if neuron_model == "lif" else {"pre_spike": True}),
        detach_reset=True,
        trainable_param={"c_m", "v_threshold"},
    )
    records = ["neuron.v_pre_spike", "synapse.psc"]
    if neuron_model == "izhikevich":
        records.append("neuron.u_pre_spike")
    model = RecurrentNN(
        neuron,
        AlphaPSC(shape, 2.0, torch.nn.Linear(6, 6, bias=False)),
        backend=KernelOptions(),
        update_state_names=tuple(records),
    ).float()
    model.neuron.init_state(batch_size=2, dtype=torch.float32)
    model.synapse.init_state(batch_size=2, dtype=torch.float32)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = torch.randn(5, 2, *shape, dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    runner = (
        model
        if compiler is None
        else torch.compile(model, backend=compiler, fullgraph=True)
    )
    with environ.context(dt=1.0):
        actual, actual_records = runner(x)
        expected, expected_records = reference(xr)
    torch.testing.assert_close(actual, expected)
    for name in records:
        torch.testing.assert_close(actual_records[name], expected_records[name])
    torch.testing.assert_close(model.neuron.v_pre_spike, reference.neuron.v_pre_spike)
    (
        actual.sum()
        + sum(value.square().sum() for value in actual_records.values())
        + model.neuron.v_pre_spike.square().sum()
    ).backward()
    (
        expected.sum()
        + sum(value.square().sum() for value in expected_records.values())
        + reference.neuron.v_pre_spike.square().sum()
    ).backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=3e-6, rtol=3e-5)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(reference.named_parameters())[name].grad,
            atol=3e-6,
            rtol=3e-5,
        )


@pytest.mark.parametrize("model", ["lif", "glif3", "alpha", "dual_exponential"])
def test_preallocated_reference_records_preserve_eager_loss_gradients(model):
    from btorch.models.kernels.torch_backend import forward_sequence

    # User losses can combine output, final state and records. Allocating the
    # reference history once must preserve every such eager autograd path,
    # including GLIF3's additional adaptation axis and PSC output/state overlap.
    torch.manual_seed(39)
    spec, drive, states, parameters = cell_inputs(model)
    x = drive.detach().float().unsqueeze(0).repeat(5, 1, 1).requires_grad_()
    states = tuple(value.detach().float().requires_grad_() for value in states)
    parameters = tuple(value.detach().float().requires_grad_() for value in parameters)
    names = get_template(spec).history_names
    output, final, histories = forward_sequence(x, states, parameters, spec, names)
    expected_states = states
    expected_outputs, expected_histories = [], {name: [] for name in names}
    for step in x:
        expected, expected_states, records = forward_step(
            step, expected_states, parameters, spec
        )
        expected_outputs.append(expected)
        for name in names:
            expected_histories[name].append(records[name])
    expected_values = (
        torch.stack(expected_outputs),
        *expected_states,
        *(torch.stack(expected_histories[name]) for name in names),
    )
    actual_values = (output, *final, *histories)
    cotangents = tuple(torch.randn_like(value) for value in actual_values)
    for actual, expected in zip(actual_values, expected_values):
        torch.testing.assert_close(actual, expected)
    inputs = (x, *states, *parameters)
    actual_gradients = torch.autograd.grad(
        sum((value * grad).sum() for value, grad in zip(actual_values, cotangents)),
        inputs,
        retain_graph=True,
        allow_unused=True,
    )
    expected_gradients = torch.autograd.grad(
        sum((value * grad).sum() for value, grad in zip(expected_values, cotangents)),
        inputs,
        allow_unused=True,
    )
    for actual, expected in zip(actual_gradients, expected_gradients):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)


def test_parameter_only_training_preserves_temporal_state_adjoint():
    from btorch.models.kernels.torch_backend import backward_sequence, forward_sequence

    # A frozen input and initial state do not make later states independent.
    # Training tau/gain still requires the full temporal state-adjoint chain,
    # but must not allocate or return an input-gradient sequence.
    spec, drive, initial, parameters = cell_inputs("alpha")
    x = drive.detach().float().unsqueeze(0).repeat(7, 1, 1)
    initial = tuple(value.detach().float() for value in initial)
    parameters = tuple(value.detach().float().requires_grad_() for value in parameters)
    names = get_template(spec).state_names
    output, final, histories = forward_sequence(x, initial, parameters, spec, names)
    grad_output = torch.randn_like(output)
    grad_final = tuple(torch.randn_like(value) for value in final)
    grad_history = tuple(torch.randn_like(value) for value in histories)
    loss = (
        (output * grad_output).sum()
        + sum((value * grad).sum() for value, grad in zip(final, grad_final))
        + sum((value * grad).sum() for value, grad in zip(histories, grad_history))
    )
    expected = torch.autograd.grad(loss, parameters)
    gx, gs, gp = backward_sequence(
        x,
        initial,
        parameters,
        spec,
        names,
        histories,
        grad_output,
        grad_final,
        grad_history,
        needs_input_grad=(
            False,
            *(False for _ in initial),
            *(True for _ in parameters),
        ),
    )
    assert gx is None and all(value is None for value in gs)
    for actual, reference in zip(gp, expected):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("registration", ["backend", "model"])
@pytest.mark.parametrize("compiler", [None, "aot_eager"])
def test_replacement_rejects_outstanding_backward(registration, compiler):
    from btorch.models.kernels import register_cell_backend, run_multi_step
    from btorch.models.kernels.templates import ModelTemplate, register_model_template
    from btorch.models.kernels.torch_backend import backward_sequence, forward_sequence

    # OOT developers may replace an implementation while iterating. An already
    # computed loss belongs to the old recurrence: applying a replacement VJP
    # silently would corrupt training. Check this at runtime in both eager and
    # compiled backward, even when the replacement keeps the same tensor ABI.
    suffix = f"{registration}_{compiler or 'eager'}"
    backend = f"replacement_{suffix}"
    register_cell_backend(
        backend, forward=forward_sequence, backward=backward_sequence, replace=True
    )
    if registration == "model":
        template = ModelTemplate(f"replacement_q_{suffix}", ("q",), (), ("q",))

        def forward_q(x, states, parameters, spec):
            q = states[0] + x
            return q, (q,), {"q": q}

        def backward_q(
            x,
            states,
            parameters,
            spec,
            grad_output,
            grad_states,
            grad_records,
            *,
            parameter_mask=None,
        ):
            grad_q = grad_output + grad_states[0] + grad_records.get("q", 0)
            return grad_q, (grad_q,), ()

        register_model_template(
            template, forward_step=forward_q, backward_step=backward_q, replace=True
        )
        spec, states, parameters = CellSpec(template.name), (torch.zeros(2, 3),), ()
    else:
        spec, _, states, parameters = cell_inputs("lif")
        states = tuple(value.detach().float() for value in states)
        parameters = tuple(value.detach().float() for value in parameters)

    def sequence(x):
        return run_multi_step(x, states, parameters, spec, backend=backend)[0]

    runner = (
        sequence
        if compiler is None
        else torch.compile(sequence, backend=compiler, fullgraph=True)
    )
    x = torch.ones(4, 2, 3, dtype=torch.float32, requires_grad=True)
    output = runner(x)
    if registration == "backend":
        register_cell_backend(
            backend, forward=forward_sequence, backward=backward_sequence, replace=True
        )
    else:
        register_model_template(
            template, forward_step=forward_q, backward_step=backward_q, replace=True
        )
    with pytest.raises(RuntimeError, match="registration changed after forward"):
        output.sum().backward()
    # A fresh forward belongs to the new revision and remains trainable.
    run_multi_step(x, states, parameters, spec, backend=backend)[0].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_higher_order_training_is_explicitly_rejected():
    from btorch.models.kernels import run_multi_step

    # Meta-learning/Hessian callers ask for a differentiable first gradient.
    # The handwritten VJP has no second-order contract; reject create_graph
    # immediately rather than silently returning a detached gradient and
    # producing an unrelated "does not require grad" error at the next step.
    spec, drive, states, parameters = cell_inputs("alpha")
    x = drive.detach().float().unsqueeze(0).repeat(4, 1, 1).requires_grad_()
    states = tuple(value.detach().float() for value in states)
    parameters = tuple(value.detach().float() for value in parameters)
    output = run_multi_step(x, states, parameters, spec)[0]
    with pytest.raises(NotImplementedError, match="first-order gradients only"):
        torch.autograd.grad(output.square().sum(), x, create_graph=True)
    # The same numerical recurrence continues to provide ordinary BPTT.
    output = run_multi_step(x, states, parameters, spec)[0]
    gradient = torch.autograd.grad(output.square().sum(), x)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_training_operator_registration_with_opcheck():
    from btorch.models.kernels.operators import _training, encode_spec

    # opcheck checks dispatcher schemas, fake shapes, autograd registration and
    # AOT behavior. Numerical derivative parity is tested separately: opcheck
    # must not be mistaken for a correctness check of the kernel equations.
    spec, x, states, parameters = cell_inputs("alpha")
    torch.library.opcheck(
        _training,
        (
            x.unsqueeze(0).contiguous(),
            list(states),
            list(parameters),
            encode_spec(spec),
            "h",
            "torch",
            torch.tensor(spec.dt, device=x.device),
        ),
    )


def test_smooth_psc_python_gradient_with_gradcheck():
    from btorch.models import environ
    from btorch.models.synapse import AlphaPSC

    # Finite differences are meaningful for this smooth PSC. Hard-threshold
    # spikes intentionally use surrogate derivatives, so neuron gradcheck is
    # mathematically invalid and cannot replace parity to Python autograd.
    module = AlphaPSC(2, tau_syn=4.0, linear=torch.nn.Identity()).double()
    values = tuple(
        torch.randn(1, 2, dtype=torch.double).requires_grad_() for _ in range(3)
    )
    tau = torch.full((2,), 4.0, dtype=torch.double, requires_grad=True)
    gain = torch.full((2,), 0.7, dtype=torch.double, requires_grad=True)

    def step(x, psc, h, tau_syn, g_max):
        module.psc, module.h = psc, h
        module.tau_syn, module.g_max = tau_syn, g_max
        with environ.context(dt=0.5):
            output = module.single_step_forward(x)
        return output, module.h

    assert torch.autograd.gradcheck(step, (*values, tau, gain))


def test_external_inference_only_model_rejects_training_before_execution():
    from btorch.models.kernels import (
        ModelTemplate,
        register_model_template,
        run_multi_step,
    )

    # An external recurrence can be useful without a supplied VJP. The API
    # must diagnose this at the sequence entry instead of failing in backward.
    def step(x, states, parameters, spec):
        q = states[0] + x
        return q, (q,), {"q": q}

    template = ModelTemplate("lab_inference_integrator", ("q",), (), ("q",))
    register_model_template(template, forward_step=step, replace=True)
    x = torch.ones(3, 2, 4, requires_grad=True)
    state = torch.zeros(2, 4)
    with torch.inference_mode():
        output, final = run_multi_step(x, (state,), (), CellSpec(template.name))
    torch.testing.assert_close(
        output, torch.arange(1, 4).reshape(3, 1, 1).expand_as(x).float()
    )
    torch.testing.assert_close(final, torch.full_like(state, 3))
    with pytest.raises(NotImplementedError, match="does not provide a backward_step"):
        run_multi_step(x, (state,), (), CellSpec(template.name))


@pytest.mark.parametrize(
    "extra_dims,sources",
    [((("q", 0), ("q", 1)), ()), ((), (("unknown_record", "q"),))],
)
def test_external_model_rejects_ambiguous_state_and_record_layouts(extra_dims, sources):
    from btorch.models.kernels import ModelTemplate, register_model_template

    # Conflicting state ranks and misspelled record names would otherwise be
    # silently lost by dict conversion and generate an incorrect buffer ABI.
    template = ModelTemplate(
        "lab_invalid_layout",
        ("q",),
        (),
        ("q",),
        state_extra_dims=extra_dims,
        history_sources=sources,
    )
    with pytest.raises(ValueError):
        register_model_template(template, forward_step=lambda *args: None, replace=True)


def test_external_template_without_reference_vjp_can_use_backend_vjp():
    from btorch.models.kernels import (
        ModelTemplate,
        register_cell_backend,
        register_model_template,
        run_multi_step,
    )
    from btorch.models.kernels.torch_backend import forward_sequence

    # The model author supplies only Python forward semantics; a separate OOT
    # backend can provide its own sequence algorithm and VJP independently.
    def step(x, states, parameters, spec):
        q = states[0] + x
        return q, (q,), {"q": q}

    def reverse(
        x,
        states,
        parameters,
        spec,
        names,
        histories,
        grad_output,
        grad_final,
        grad_histories,
        *,
        needs_input_grad=None,
    ):
        # A prefix-sum forward has a suffix-sum VJP. Final-state and recorded
        # state losses join that temporal adjoint, including the initial state.
        grad_steps = grad_output.clone()
        for name, gradient in zip(names, grad_histories):
            if name == "q":
                grad_steps = grad_steps + gradient
        grad_x = grad_steps.flip(0).cumsum(0).flip(0) + grad_final[0]
        grad_initial = grad_steps.sum(0) + grad_final[0]
        return grad_x, (grad_initial,), ()

    template = ModelTemplate("lab_backend_integrator", ("q",), (), ("q",))
    register_model_template(template, forward_step=step, replace=True)
    register_cell_backend(
        "lab_integrator_vjp",
        forward=forward_sequence,
        backward=reverse,
        replace=True,
    )
    x = torch.randn(4, 2, 3, requires_grad=True)
    state = torch.randn(2, 3, requires_grad=True)
    output, final, records = run_multi_step(
        x,
        (state,),
        (),
        CellSpec(template.name),
        HistoryPlan(requested=("q",)),
        backend="lab_integrator_vjp",
    )
    expected = x.cumsum(0) + state
    weights = torch.randn_like(output)
    loss = (output * weights).sum() + final.square().sum() + records.square().sum()
    expected_loss = (
        (expected * weights).sum()
        + expected[-1].square().sum()
        + expected.square().sum()
    )
    actual_gradients = torch.autograd.grad(loss, (x, state))
    expected_gradients = torch.autograd.grad(expected_loss, (x, state))
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(final, expected[-1])
    for actual, trusted in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, trusted)


@pytest.mark.parametrize("model", ["lif", "alpha"])
def test_sequence_backward_retains_each_invocation_timestep(model):
    from dataclasses import replace

    from btorch.models.kernels.golden import python_module_sequence
    from btorch.models.kernels.operators import run_multi_step

    # Two outstanding forwards must retain their own physical timestep, even
    # when backward runs only after the environment has moved to another dt.
    spec, x, states, parameters = cell_inputs(model)
    x_seq = x.unsqueeze(0).expand(4, *x.shape).detach().clone().requires_grad_()
    reference_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (x_seq, *states, *parameters)
    )
    reference_x = reference_inputs[0]
    reference_states = reference_inputs[1 : 1 + len(states)]
    reference_parameters = reference_inputs[1 + len(states) :]
    actual_losses, expected_losses = [], []
    for dt in (0.25, 1.5):
        invocation = replace(spec, dt=dt)
        actual = run_multi_step(x_seq, states, parameters, invocation)
        output, final, _ = python_module_sequence(
            reference_x, reference_states, reference_parameters, invocation, ()
        )
        for got, expected in zip(actual, (output, *final)):
            torch.testing.assert_close(got, expected)
        actual_losses.append(sum(value.square().sum() for value in actual))
        expected_losses.append(sum(value.square().sum() for value in (output, *final)))
    actual_gradients = torch.autograd.grad(
        sum(actual_losses), (x_seq, *states, *parameters)
    )
    expected_gradients = torch.autograd.grad(sum(expected_losses), reference_inputs)
    for got, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
def test_environment_rejects_invalid_physical_timestep(dt):
    from btorch.models import environ
    from btorch.models.kernels import KernelOptions
    from btorch.models.neurons.lif import LIF

    # Invalid timesteps must fail before executing a generated trajectory,
    # rather than yielding NaN states or silently running time backwards.
    neuron = LIF(3, backend=KernelOptions(), step_mode="m")
    neuron.init_state(batch_size=2)
    with environ.context(dt=dt), pytest.raises(ValueError, match="finite and positive"):
        neuron(torch.ones(2, 2, 3))
