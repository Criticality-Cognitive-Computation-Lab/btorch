"""Exercise the public external-kernel workflow and catch numerical
mistakes."""

import pytest
import torch

from btorch.models import environ
from btorch.models.kernels import (
    HistoryPlan,
    NeuronSpec,
    register_cell_backend,
    run_multi_step,
)
from btorch.models.kernels.testing import (
    ParityCase,
    ParityOptions,
    check_backend_parity,
)
from btorch.models.kernels.torch_backend import backward_sequence, forward_sequence
from btorch.models.neurons.lif import LIF


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["triton", "tilelang", "cupy"])
def test_public_parity_captures_training_gradients(backend):
    # External kernel authors must validate replayed VJPs as well as inference.
    # Nonzero initial states and record losses make all three gradient families
    # observable; the helper replays twice to detect persistent accumulation.
    pytest.importorskip(backend)
    case = _case()
    case = ParityCase(
        case.x_seq.cuda(),
        tuple(value.cuda() for value in case.states),
        tuple(value.cuda() for value in case.parameters),
        case.spec,
        case.history,
    )
    report = check_backend_parity(backend, [case], options=ParityOptions(capture=True))
    report.assert_close()
    assert any("capture_training/grad_v" in item.name for item in report.comparisons)


# A separately packaged implementation registers callbacks through the public
# API. Reusing the supplied reference callbacks is useful for developing its
# integration before replacing numerical execution with generated GPU code.
register_cell_backend(
    "oot_example_lif",
    forward=forward_sequence,
    backward=backward_sequence,
    supports=lambda spec, device: spec.model == "lif" and device.type == "cpu",
)


@pytest.fixture(autouse=True)
def timestep():
    # Use a scoped physical timestep so this executable extension example does
    # not mutate the environment defaults used by unrelated model tests.
    with environ.context(dt=1.0):
        yield


def _case():
    # Use nonzero initial voltage, heterogeneous capacitance and a voltage
    # record loss so state, parameter and charged-state gradients all matter.
    x_seq = torch.tensor(
        [[[0.6, 0.3]], [[0.7, 1.2]], [[0.2, 0.5]]], dtype=torch.float32
    )
    states = (torch.tensor([[0.15, 0.25]], dtype=torch.float32),)
    parameters = (
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor([0.9, 1.1], dtype=torch.float32),
        torch.tensor(4.0, dtype=torch.float32),
    )
    return ParityCase(
        x_seq,
        states,
        parameters,
        NeuronSpec("lif"),
        HistoryPlan(requested=("v", "v_charged")),
        name="charged_loss",
    )


def test_external_backend_public_module_training_matches_existing_lif():
    # Real user workflow: the normal neuron API selects a registered external
    # backend and preserves gradients through a trainable upstream encoder.
    options = dict(
        n_neuron=2,
        tau=4.0,
        c_m=[0.9, 1.1],
        trainable_param={"tau", "c_m", "v_threshold"},
        step_mode="m",
    )
    reference = LIF(**options).float()
    candidate = LIF(**options, backend="oot_example_lif").float()
    reference.init_state(batch_size=1, dtype=torch.float32)
    candidate.init_state(batch_size=1, dtype=torch.float32)
    encoder_weight = torch.tensor(0.8, requires_grad=True, dtype=torch.float32)
    source = _case().x_seq
    expected = reference(source * encoder_weight)
    expected_grad = torch.autograd.grad(
        expected.sum() + reference.v.sum(),
        (encoder_weight, reference.tau, reference.c_m, reference.v_threshold),
    )
    candidate.eval()  # Evaluation mode must still keep the encoder gradient.
    actual = candidate(source * encoder_weight)
    actual_grad = torch.autograd.grad(
        actual.sum() + candidate.v.sum(),
        (encoder_weight, candidate.tau, candidate.c_m, candidate.v_threshold),
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(candidate.v, reference.v)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value)


def test_public_parity_checks_state_record_and_parameter_gradients():
    # The external author can validate its callback independently of the Module
    # adapter. Compile/capture are separate opt-in checks; unavailable capture
    # on this CPU trajectory is reported explicitly rather than called a pass.
    report = check_backend_parity(
        "oot_example_lif", [_case()], options=ParityOptions(capture=True)
    )
    report.assert_close()
    assert any("grad_tau" in item.name for item in report.comparisons)
    assert any("v_charged" in item.name for item in report.comparisons)
    assert "charged_loss/capture: CUDA inputs required" in report.skipped
    assert all("CUDA inputs required" in skip for skip in report.skipped)


@pytest.mark.parametrize(
    ("fault", "gradient_name"),
    [("state", "grad_v"), ("parameter", "grad_c_m"), ("record", "grad_")],
)
def test_public_parity_catches_wrong_training_gradients(fault, gradient_name):
    def wrong_backward(*args, **kwargs):
        if fault == "record":
            # Losing cotangents from user records is invisible to spike-only
            # training, but breaks auxiliary voltage/charged-state objectives.
            args = (*args[:-1], tuple(torch.zeros_like(g) for g in args[-1]))
        gx, gs, gp = backward_sequence(*args, **kwargs)

        # Deliberately plausible forward with a broken temporal state adjoint:
        # spike-only parity would pass and initial-state optimization would fail.
        def zero_gradients(values):
            return tuple(None if g is None else torch.zeros_like(g) for g in values)

        return (
            gx,
            zero_gradients(gs) if fault == "state" else gs,
            (zero_gradients(gp) if fault == "parameter" else gp),
        )

    register_cell_backend(
        "oot_broken_state",
        forward=forward_sequence,
        backward=wrong_backward,
        replace=True,
    )
    report = check_backend_parity("oot_broken_state", [_case()])
    assert not report.passed
    with pytest.raises(AssertionError, match=gradient_name):
        report.assert_close()


def test_simulation_only_extension_fails_before_training_execution():
    # A user intentionally provides forward only for inference. It should work
    # under inference_mode and reject a trainable encoder at sequence entry,
    # instead of succeeding in forward then crashing much later in backward.
    register_cell_backend(
        "oot_inference",
        forward=forward_sequence,
        supports=lambda spec, device: spec.model == "lif",
        replace=True,
    )
    case = _case()
    with torch.inference_mode():
        actual = run_multi_step(
            case.x_seq,
            case.states,
            case.parameters,
            case.spec,
            backend="oot_inference",
        )
    expected = forward_sequence(case.x_seq, case.states, case.parameters, case.spec, ())
    torch.testing.assert_close(actual[0], expected[0])
    with pytest.raises(NotImplementedError, match="backward|inference"):
        run_multi_step(
            case.x_seq.requires_grad_(),
            case.states,
            case.parameters,
            case.spec,
            backend="oot_inference",
        )
    # A declared model restriction must not silently route an unsupported PSC
    # to another implementation just because one happens to be available.
    with pytest.raises(NotImplementedError, match="support"):
        run_multi_step(
            torch.ones(2, 1, 2, dtype=torch.float32),
            (torch.zeros(1, 2),),
            (torch.tensor(2.0),),
            NeuronSpec("exponential"),
            backend="oot_inference",
        )


def test_empty_external_trajectory_keeps_the_initial_state_derivative():
    # With no timesteps, the parameter path is unused while final state still
    # has an identity derivative. The parity helper must check that edge case
    # without trying to differentiate a nonexistent eager parameter graph.
    case = _case()
    empty = ParityCase(
        case.x_seq[:0],
        case.states,
        case.parameters,
        case.spec,
        case.history,
        name="empty",
    )
    report = check_backend_parity("oot_example_lif", [empty])
    report.assert_close()


def test_external_callback_cannot_mutate_the_callers_initial_state():
    def mutating_forward(x_seq, states, parameters, spec, names):
        result = forward_sequence(x_seq, states, parameters, spec, names)
        # Simulate a raw kernel writing its initial-state pointer after returning
        # otherwise correct results. This breaks continuation, checkpointing and
        # the functional compiler contract even when inference outputs agree.
        with torch.no_grad():
            states[0].add_(0.1)
        return result

    register_cell_backend(
        "oot_mutating",
        forward=mutating_forward,
        backward=backward_sequence,
        replace=True,
    )
    report = check_backend_parity("oot_mutating", [_case()])
    with pytest.raises(AssertionError, match="mutated input"):
        report.assert_close()


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
def test_public_parity_uses_original_python_models(model):
    # Validate actual multi-step trajectories against existing Python Modules,
    # including charged-state losses, multiple adaptation modes and every
    # participating parameter. This is the same golden used by OOT authors.
    from btorch.models.kernels.templates import get_template
    from tests.models.kernels.helpers import cell_inputs

    torch.manual_seed(41)
    spec, x, states, parameters = cell_inputs(model)
    template = get_template(spec)
    case = ParityCase(
        torch.stack((x, x * 0.4, x * 1.3)),
        states,
        parameters,
        spec,
        HistoryPlan(requested=template.history_names),
        name=model,
    )
    check_backend_parity("torch", [case]).assert_close()


def test_parity_checks_noncontiguous_input_and_record_losses_by_default():
    def faulty_forward(x_seq, states, parameters, spec, names):
        output, final, records = forward_sequence(
            x_seq, states, parameters, spec, names
        )
        # A generated implementation may accidentally treat every input pointer
        # as contiguous. The supplied trajectory is contiguous, so only the
        # automatic strided case reveals this defect.
        if not x_seq.is_contiguous():
            records = tuple(torch.zeros_like(value) for value in records)
        return output, final, records

    register_cell_backend(
        "oot_bad_strides",
        forward=faulty_forward,
        backward=backward_sequence,
        replace=True,
    )
    report = check_backend_parity("oot_bad_strides", [_case()])
    with pytest.raises(AssertionError, match="noncontiguous"):
        report.assert_close()


def test_parity_checks_chunk_state_gradient_and_two_live_graphs():
    def faulty_backward(*args, **kwargs):
        gx, gs, gp = backward_sequence(*args, **kwargs)
        # Dropping the initial-state adjoint leaves the visible forward correct
        # but cuts gradients across a chunk boundary. It must be reported under
        # continuation even when a full unchunked input-gradient test looks good.
        return (
            gx,
            tuple(None if value is None else torch.zeros_like(value) for value in gs),
            gp,
        )

    register_cell_backend(
        "oot_bad_continuation",
        forward=forward_sequence,
        backward=faulty_backward,
        replace=True,
    )
    report = check_backend_parity("oot_bad_continuation", [_case()])
    assert any(
        not item.passed and "/continuation/grad_x_seq" in item.name
        for item in report.comparisons
    )
    assert any("/two_forwards/" in item.name for item in report.comparisons)
    with pytest.raises(AssertionError, match="continuation/grad_x_seq"):
        report.assert_close()


def test_parity_checks_declared_charged_records_without_explicit_record_request():
    def faulty_forward(x_seq, states, parameters, spec, names):
        output, final, records = forward_sequence(
            x_seq, states, parameters, spec, names
        )
        # A backend can return correct spikes/final voltage while exposing an
        # incorrect charged voltage for auxiliary losses. The default validator
        # must request that declared record even when the supplied case does not.
        records = tuple(
            value + 0.25 if name == "v_charged" else value
            for name, value in zip(names, records)
        )
        return output, final, records

    register_cell_backend(
        "oot_bad_charged_record",
        forward=faulty_forward,
        backward=backward_sequence,
        replace=True,
    )
    base = _case()
    case = ParityCase(base.x_seq, base.states, base.parameters, base.spec)
    report = check_backend_parity("oot_bad_charged_record", [case])
    with pytest.raises(AssertionError, match="all_records/.*/v_charged"):
        report.assert_close()
