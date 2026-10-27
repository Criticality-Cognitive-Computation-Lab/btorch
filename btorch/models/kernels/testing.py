"""Validate external cell kernels against the existing btorch Python
Modules."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from .contracts import HistoryPlan, NeuronSpec
from .golden import python_module_sequence
from .operators import run_multi_step
from .templates import get_template


@dataclass(frozen=True)
class ParityCase:
    """Describe an independent numerical trajectory for kernel validation.

    Args:
        x_seq: Time-first drive tensor.
        states: Initial states in the declared model order.
        parameters: Floating parameters in the declared model order.
        spec: Model and scalar integration choices.
        history: Records included in the loss as well as output comparison.
        name: Human-readable scenario name.
        golden: Independent Python sequence callable for external models. Built-in
            models always default to the original btorch Torch Modules.
    """

    x_seq: Tensor
    states: tuple[Tensor, ...]
    parameters: tuple[Tensor, ...]
    spec: NeuronSpec
    history: HistoryPlan = HistoryPlan()
    name: str = "trajectory"
    golden: Callable | None = None


@dataclass(frozen=True)
class ParityOptions:
    """Configure numerical tolerances and additional execution checks.

    Args:
        atol: Absolute tolerance; None selects a dtype-dependent default.
        rtol: Relative tolerance; None selects a dtype-dependent default.
        check_gradients: Check every floating input, initial state and parameter.
        check_edge_cases: Also check empty, single-step and noncontiguous drives,
            plus losses on every record declared by the model template.
        check_continuation: Also check chunk continuation and two outstanding
            forwards followed by one combined backward.
        compile: Also test static fullgraph compilation and its gradients.
        capture: Also test warmed-up CUDA inference replay and, when
            check_gradients is enabled, captured forward/backward replay.
    """

    atol: float | None = None
    rtol: float | None = None
    check_gradients: bool = True
    check_edge_cases: bool = True
    check_continuation: bool = True
    compile: bool = False
    capture: bool = False


@dataclass(frozen=True)
class TensorParity:
    """Report an independently compared numerical output or derivative.

    Args:
        name: Scenario, execution mode and tensor name.
        passed: Whether shape, dtype and numerical values match.
        max_abs_error: Maximum absolute error, or infinity for a layout mismatch.
        max_rel_error: Maximum relative error with an absolute-tolerance floor.
        first_mismatch: Coordinates of the first mismatching value.
        first_spike_divergence: First divergent timestep for neuron spike output.
        detail: Layout or precision diagnostics.
    """

    name: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    first_mismatch: tuple[int, ...] | None = None
    first_spike_divergence: int | None = None
    detail: str = ""


@dataclass
class ParityReport:
    """Collect numerical evidence and explicit capability skips.

    Args:
        backend: Registered candidate backend name.
        comparisons: Output/state/history/gradient comparisons.
        skipped: Unavailable optional checks, never counted as passes.
        errors: Execution errors with scenario and mode names.
    """

    backend: str
    comparisons: list[TensorParity] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return whether numerical checks ran and no failure was observed."""
        return (
            bool(self.comparisons)
            and not self.errors
            and all(comparison.passed for comparison in self.comparisons)
        )

    def assert_close(self) -> None:
        """Raise an actionable assertion for incorrect or wholly untested
        kernels."""
        failures = [
            f"{item.name}: abs={item.max_abs_error:g}, rel={item.max_rel_error:g}, "
            f"first={item.first_mismatch}, first_spike={item.first_spike_divergence} "
            f"{item.detail}"
            for item in self.comparisons
            if not item.passed
        ]
        if not self.comparisons:
            failures.append("No numerical comparisons executed")
        failures.extend(self.errors)
        if failures:
            raise AssertionError(
                f"Backend {self.backend!r} parity failed:\n" + "\n".join(failures)
            )


def _compare(name, actual, expected, atol, rtol, *, spikes=False) -> TensorParity:
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or actual.device != expected.device
    ):
        return TensorParity(
            name,
            False,
            float("inf"),
            float("inf"),
            detail=(
                f"expected {tuple(expected.shape)} {expected.dtype} {expected.device}; "
                f"got {tuple(actual.shape)} {actual.dtype} {actual.device}"
            ),
        )
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if not actual.numel():
        return TensorParity(name, True, 0.0, 0.0)
    difference = (actual.double() - expected.double()).abs()
    close = (
        actual == expected
        if spikes or not (actual.is_floating_point() or actual.is_complex())
        else torch.isclose(actual, expected, atol=atol, rtol=rtol)
    )
    mismatch = torch.nonzero(~close, as_tuple=False)
    first = tuple(mismatch[0].tolist()) if mismatch.numel() else None
    relative = difference / expected.double().abs().clamp_min(max(atol, 1e-12))
    return TensorParity(
        name,
        first is None,
        difference.max().item(),
        relative.max().item(),
        first,
        first[0] if spikes and first is not None else None,
    )


def _inputs(case, gradients):
    values = (case.x_seq, *case.states, *case.parameters)
    mask = (gradients,) * len(values) if isinstance(gradients, bool) else gradients
    result = []
    for value, needed in zip(values, mask):
        # clone() can compact sliced views. Preserve the actual input strides so
        # indexing defects in an external kernel are observable.
        copied = torch.empty_strided(
            value.shape, value.stride(), dtype=value.dtype, device=value.device
        )
        copied.copy_(value.detach())
        result.append(copied.requires_grad_(needed))
    return tuple(result)


def _loss_weights(outputs):
    # Local seeded, nonperiodic VJP probes expose indexing defects without
    # consuming the caller's RNG. A probe is numerical evidence, not a proof
    # of the complete Jacobian.
    weights = []
    for index, value in enumerate(outputs):
        generator = torch.Generator(device=value.device).manual_seed(1729 + index)
        weights.append(
            torch.randn(
                value.shape,
                device=value.device,
                dtype=torch.float32,
                generator=generator,
            )
        )
    return tuple(weights)


def _loss(outputs, weights=None):
    if weights is None:
        weights = _loss_weights(outputs)
    terms = [
        (value.float() * weight).sum() / (index + 1)
        for index, (value, weight) in enumerate(zip(outputs, weights))
    ]
    return sum(terms)


def _gradients(outputs, inputs, weights=None):
    if not inputs:
        return ()
    loss = _loss(outputs, weights)
    # An empty trajectory or unused parameter has the zero mathematical
    # derivative even when the eager reference constructs no autograd graph.
    if not loss.requires_grad:
        return (None,) * len(inputs)
    return torch.autograd.grad(loss, inputs, allow_unused=True)


def check_backend_parity(
    backend: str, cases: list[ParityCase], *, options: ParityOptions = ParityOptions()
) -> ParityReport:
    """Compare external execution and gradients with the original btorch Python
    Modules.

    The reference uses ordinary autograd over the original Module recurrence,
    independently of the candidate's handwritten backward. Every floating input
    is tested with gradients enabled, including otherwise frozen initial state.
    Inference is checked independently under no_grad and inference_mode.

    Args:
        backend: Registered implementation to validate.
        cases: Representative user trajectories, parameter shapes and options.
        options: Tolerances and optional fullgraph/capture checks.

    Returns:
        Structured numerical report. Call ``assert_close`` to reject failures;
        inspect ``skipped`` separately for unavailable optional capabilities.
    """
    report = ParityReport(backend)
    original_cases = cases
    if options.check_edge_cases:
        cases = []
        for case in original_cases:
            cases.append(case)
            record_names = get_template(case.spec).history_names
            if case.history.requested != record_names:
                cases.append(
                    replace(
                        case,
                        history=HistoryPlan(requested=record_names),
                        name=f"{case.name}/all_records",
                    )
                )
            cases.append(replace(case, x_seq=case.x_seq[:0], name=f"{case.name}/empty"))
            if case.x_seq.shape[0]:
                cases.append(
                    replace(case, x_seq=case.x_seq[:1], name=f"{case.name}/singleton")
                )
            if case.x_seq.numel():
                storage = case.x_seq.new_empty((*case.x_seq.shape, 2))
                view = storage[..., 0]
                view.copy_(case.x_seq)
                cases.append(
                    replace(case, x_seq=view, name=f"{case.name}/noncontiguous")
                )
    for case in cases:
        if case.x_seq.is_cuda and not torch.cuda.is_available():
            report.skipped.append(f"{case.name}: CUDA unavailable")
            continue
        template = get_template(case.spec)
        names = ("output_seq", *template.state_names, *case.history.requested)
        atol = (
            options.atol
            if options.atol is not None
            else (3e-5 if case.x_seq.dtype == torch.float32 else 5e-3)
        )
        rtol = (
            options.rtol
            if options.rtol is not None
            else (3e-5 if case.x_seq.dtype == torch.float32 else 5e-3)
        )
        modes = [
            "all_grad",
            "input_grad",
            "state_grad",
            "parameter_grad",
            "frozen",
            "readout",
            "no_grad",
            "inference",
        ]
        if options.compile:
            modes.append("compile")
        for mode in modes:
            gradients = options.check_gradients and (
                mode
                in ("all_grad", "input_grad", "state_grad", "parameter_grad", "compile")
            )
            context = (
                torch.inference_mode()
                if mode == "inference"
                else (
                    torch.enable_grad()
                    if gradients or mode in ("frozen", "readout")
                    else torch.no_grad()
                )
            )
            try:
                with torch.inference_mode(False), context:
                    mask = (
                        gradients and mode in ("all_grad", "input_grad", "compile"),
                        *(
                            (
                                gradients
                                and mode in ("all_grad", "state_grad", "compile"),
                            )
                            * len(case.states)
                        ),
                        *(
                            (
                                gradients
                                and mode in ("all_grad", "parameter_grad", "compile"),
                            )
                            * len(case.parameters)
                        ),
                    )
                    reference_inputs = _inputs(case, mask)
                    candidate_inputs = _inputs(case, mask)
                    n_states = len(case.states)

                    def execute(inputs, reference=False):
                        x = inputs[0]
                        states = inputs[1 : 1 + n_states]
                        parameters = inputs[1 + n_states :]
                        if reference:
                            output, final, histories = (
                                case.golden or python_module_sequence
                            )(x, states, parameters, case.spec, case.history.requested)
                            return (output.to(x.dtype), *final, *histories)
                        return run_multi_step(
                            x,
                            states,
                            parameters,
                            case.spec,
                            case.history,
                            backend=backend,
                        )

                    expected = execute(reference_inputs, reference=True)
                    runner = (
                        torch.compile(execute, fullgraph=True)
                        if mode == "compile"
                        else execute
                    )
                    candidate_snapshot = tuple(
                        value.detach().clone() for value in candidate_inputs
                    )
                    actual = runner(candidate_inputs)
                    input_names = (
                        "x_seq",
                        *template.state_names,
                        *template.parameter_names,
                    )
                    for input_name, value, before in zip(
                        input_names, candidate_inputs, candidate_snapshot
                    ):
                        if not torch.equal(value.detach(), before):
                            report.errors.append(
                                f"{case.name}/{mode}: callback mutated "
                                f"input {input_name!r}"
                            )
                    for name, a, e in zip(names, actual, expected):
                        report.comparisons.append(
                            _compare(
                                f"{case.name}/{mode}/{name}",
                                a,
                                e,
                                atol,
                                rtol,
                                spikes=name == "output_seq"
                                and "v" in template.state_names,
                            )
                        )
                    if len(actual) != len(expected):
                        report.errors.append(
                            f"{case.name}/{mode}: output count mismatch"
                        )
                    if mode == "readout":
                        # Frozen dynamics should use inference even in grad mode,
                        # while a downstream trainable readout still gets gradients.
                        expected_weight = expected[0].new_tensor(
                            0.7, requires_grad=True
                        )
                        actual_weight = actual[0].new_tensor(0.7, requires_grad=True)
                        expected_readout_grad = torch.autograd.grad(
                            (expected[0] * expected_weight).square().sum(),
                            expected_weight,
                        )[0]
                        actual_readout_grad = torch.autograd.grad(
                            (actual[0] * actual_weight).square().sum(), actual_weight
                        )[0]
                        report.comparisons.append(
                            _compare(
                                f"{case.name}/{mode}/grad_readout",
                                actual_readout_grad,
                                expected_readout_grad,
                                atol,
                                rtol,
                            )
                        )
                    if gradients:
                        reference_active = tuple(
                            value
                            for value, needed in zip(reference_inputs, mask)
                            if needed
                        )
                        candidate_active = tuple(
                            value
                            for value, needed in zip(candidate_inputs, mask)
                            if needed
                        )
                        expected_grad = _gradients(expected, reference_active)
                        actual_grad = _gradients(actual, candidate_active)
                        grad_names = (
                            "x_seq",
                            *template.state_names,
                            *template.parameter_names,
                        )
                        active_names = tuple(
                            name for name, needed in zip(grad_names, mask) if needed
                        )
                        for name, a, e, source in zip(
                            active_names, actual_grad, expected_grad, reference_active
                        ):
                            a = torch.zeros_like(source) if a is None else a
                            e = torch.zeros_like(source) if e is None else e
                            report.comparisons.append(
                                _compare(
                                    f"{case.name}/{mode}/grad_{name}", a, e, atol, rtol
                                )
                            )
            except Exception as error:
                report.errors.append(
                    f"{case.name}/{mode}: {type(error).__name__}: {error}"
                )
        if options.capture:
            if not case.x_seq.is_cuda:
                report.skipped.append(f"{case.name}/capture: CUDA inputs required")
                continue
            try:
                with torch.no_grad():
                    inputs = _inputs(case, False)
                    states = inputs[1 : 1 + len(case.states)]
                    parameters = inputs[1 + len(case.states) :]

                    def replay_target():
                        return run_multi_step(
                            inputs[0],
                            states,
                            parameters,
                            case.spec,
                            case.history,
                            backend=backend,
                        )

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            replay_target()
                    torch.cuda.current_stream().wait_stream(stream)
                    expected_output, expected_final, expected_histories = (
                        case.golden or python_module_sequence
                    )(
                        inputs[0],
                        states,
                        parameters,
                        case.spec,
                        case.history.requested,
                    )
                    expected = (
                        expected_output.to(inputs[0].dtype),
                        *expected_final,
                        *expected_histories,
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = replay_target()
                    graph.replay()
                    torch.cuda.synchronize()
                    for name, a, e in zip(names, actual, expected):
                        report.comparisons.append(
                            _compare(
                                f"{case.name}/capture/{name}",
                                a,
                                e,
                                atol,
                                rtol,
                                spikes=name == "output_seq"
                                and "v" in template.state_names,
                            )
                        )
            except Exception as error:
                report.errors.append(
                    f"{case.name}/capture: {type(error).__name__}: {error}"
                )
            if options.check_gradients:
                try:
                    with torch.inference_mode(False), torch.enable_grad():
                        inputs = _inputs(case, True)
                        reference_inputs = _inputs(case, True)
                        n_states = len(case.states)
                        reference_output, reference_final, reference_history = (
                            case.golden or python_module_sequence
                        )(
                            reference_inputs[0],
                            reference_inputs[1 : 1 + n_states],
                            reference_inputs[1 + n_states :],
                            case.spec,
                            case.history.requested,
                        )
                        expected = (
                            reference_output.to(reference_inputs[0].dtype),
                            *reference_final,
                            *reference_history,
                        )
                        # CUDA generators cannot be constructed during capture.
                        # Reuse identical deterministic cotangents for the golden
                        # VJP and every warmup/captured candidate invocation.
                        weights = _loss_weights(expected)
                        expected_grad = _gradients(expected, reference_inputs, weights)

                        def training_target():
                            outputs = run_multi_step(
                                inputs[0],
                                inputs[1 : 1 + n_states],
                                inputs[1 + n_states :],
                                case.spec,
                                case.history,
                                backend=backend,
                            )
                            return outputs, _gradients(outputs, inputs, weights)

                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                training_target()
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual, actual_grad = training_target()
                        # A second replay catches accidental accumulation into
                        # persistent gradient buffers, not just capture success.
                        graph.replay()
                        graph.replay()
                        torch.cuda.synchronize()
                        grad_names = (
                            "x_seq",
                            *template.state_names,
                            *template.parameter_names,
                        )
                        for name, a, e in zip(names, actual, expected):
                            report.comparisons.append(
                                _compare(
                                    f"{case.name}/capture_training/{name}",
                                    a,
                                    e,
                                    atol,
                                    rtol,
                                    spikes=name == "output_seq"
                                    and "v" in template.state_names,
                                )
                            )
                        for name, a, e, source in zip(
                            grad_names, actual_grad, expected_grad, reference_inputs
                        ):
                            report.comparisons.append(
                                _compare(
                                    f"{case.name}/capture_training/grad_{name}",
                                    torch.zeros_like(source) if a is None else a,
                                    torch.zeros_like(source) if e is None else e,
                                    atol,
                                    rtol,
                                )
                            )
                except Exception as error:
                    report.errors.append(
                        f"{case.name}/capture_training: {type(error).__name__}: {error}"
                    )
    if options.check_continuation:
        for case in original_cases:
            _check_composed_calls(backend, case, options, report)
    return report


def _check_composed_calls(backend, case, options, report):
    """Check state continuation and backward ownership across multiple
    calls."""
    template = get_template(case.spec)
    names = ("output_seq", *template.state_names, *case.history.requested)
    grad_names = ("x_seq", *template.state_names, *template.parameter_names)
    atol = (
        options.atol
        if options.atol is not None
        else (3e-5 if case.x_seq.dtype == torch.float32 else 5e-3)
    )
    rtol = (
        options.rtol
        if options.rtol is not None
        else (3e-5 if case.x_seq.dtype == torch.float32 else 5e-3)
    )
    n_states = len(case.states)
    for scenario in ("continuation", "two_forwards"):
        try:
            with (
                torch.inference_mode(False),
                torch.set_grad_enabled(options.check_gradients),
            ):
                expected_inputs = _inputs(case, options.check_gradients)
                actual_inputs = _inputs(case, options.check_gradients)

                def execute(inputs, reference):
                    x, states, parameters = (
                        inputs[0],
                        inputs[1 : 1 + n_states],
                        inputs[1 + n_states :],
                    )

                    def run(drive, initial):
                        if reference:
                            output, final, records = (
                                case.golden or python_module_sequence
                            )(
                                drive,
                                initial,
                                parameters,
                                case.spec,
                                case.history.requested,
                            )
                            return (output, *final, *records)
                        return run_multi_step(
                            drive,
                            initial,
                            parameters,
                            case.spec,
                            case.history,
                            backend=backend,
                        )

                    if scenario == "two_forwards":
                        # Both graphs stay live. The second call uses a distinct
                        # trajectory to expose reused mutable backward buffers.
                        return (*run(x, states), *run(x.flip(0), states))
                    if reference:
                        return run(x, states)
                    split = x.shape[0] // 2
                    first = run(x[:split], states)
                    second = run(x[split:], first[1 : 1 + n_states])
                    return (
                        torch.cat((first[0], second[0])),
                        *second[1 : 1 + n_states],
                        *(
                            torch.cat((left, right))
                            for left, right in zip(
                                first[1 + n_states :], second[1 + n_states :]
                            )
                        ),
                    )

                expected = execute(expected_inputs, True)
                actual = execute(actual_inputs, False)
                output_names = (
                    names
                    if scenario == "continuation"
                    else tuple(
                        f"call{call}/{name}" for call in (1, 2) for name in names
                    )
                )
                for name, a, e in zip(output_names, actual, expected):
                    report.comparisons.append(
                        _compare(
                            f"{case.name}/{scenario}/{name}",
                            a,
                            e,
                            atol,
                            rtol,
                            spikes=name.endswith("output_seq")
                            and "v" in template.state_names,
                        )
                    )
                if options.check_gradients:
                    expected_grads = _gradients(expected, expected_inputs)
                    actual_grads = _gradients(actual, actual_inputs)
                    for name, a, e, source in zip(
                        grad_names, actual_grads, expected_grads, expected_inputs
                    ):
                        report.comparisons.append(
                            _compare(
                                f"{case.name}/{scenario}/grad_{name}",
                                torch.zeros_like(source) if a is None else a,
                                torch.zeros_like(source) if e is None else e,
                                atol,
                                rtol,
                            )
                        )
        except Exception as error:
            report.errors.append(
                f"{case.name}/{scenario}: {type(error).__name__}: {error}"
            )
