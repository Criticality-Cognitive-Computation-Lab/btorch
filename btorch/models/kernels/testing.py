"""Compare explicit kernel workloads with independent model executions."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor
from torch.utils._pytree import keystr, tree_flatten, tree_flatten_with_path

from .contracts import HistoryPlan, NeuronSpec
from .golden import python_module_sequence
from .operators import run_multi_step
from .templates import get_template


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
    difference = (
        actual.to(torch.complex128 if actual.is_complex() else torch.float64)
        - expected.to(torch.complex128 if expected.is_complex() else torch.float64)
    ).abs()
    close = (
        actual == expected
        if spikes or not (actual.is_floating_point() or actual.is_complex())
        else torch.isclose(actual, expected, atol=atol, rtol=rtol)
    )
    mismatch = torch.nonzero(~close, as_tuple=False)
    first = tuple(mismatch[0].tolist()) if mismatch.numel() else None
    relative = difference / expected.abs().to(torch.float64).clamp_min(max(atol, 1e-12))
    return TensorParity(
        name,
        first is None,
        difference.max().item(),
        relative.max().item(),
        first,
        first[0] if spikes and first is not None and first else None,
    )


def _loss_weights(outputs):
    """Create identical local cotangents without consuming the caller's RNG."""
    weights = []
    for index, value in enumerate(outputs):
        generator = torch.Generator(device=value.device).manual_seed(1729 + index)
        dtype = (
            value.dtype
            if value.is_floating_point() or value.is_complex()
            else torch.float32
        )
        weights.append(
            torch.randn(
                value.shape, device=value.device, dtype=dtype, generator=generator
            )
        )
    return tuple(weights)


@dataclass(frozen=True)
class ModelParityCase:
    """Describe a whole model or operator comparison.

    Callables must start from the supplied independent initial tensors on each
    invocation; factories or closures can reconstruct stateful Modules. No
    model equations or numerical-gradient semantics are inferred by this tool.

    Args:
        name: Scenario identifier.
        golden: Confirmed Python model execution.
        candidate: Native execution or binding with the same tensor contract.
        inputs: Inputs, initial states and parameters, including frozen tensors.
        input_names: Optional semantic names for input/state/parameter VJPs.
        exact_outputs: Pytree paths compared exactly, such as spike records.
        immutable_inputs: Reject a candidate mutating the supplied pure tensor
            ABI inputs; stateful model wrappers may leave this disabled.
        gradient_oracle: Optional callable ``(inputs, outputs, cotangents)``
            returning one gradient per input for alternative gradient semantics.
    """

    name: str
    golden: Callable
    candidate: Callable
    inputs: tuple[Tensor, ...]
    gradient_oracle: Callable | None = None
    input_names: tuple[str, ...] = ()
    exact_outputs: tuple[str, ...] = ()
    immutable_inputs: bool = False


@dataclass(frozen=True)
class ModelParityOptions:
    """Configure whole-operator numerical comparison.

    Args:
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        check_gradients: Compare VJPs of inputs marked requires_grad.
        gradient_atol: Optional VJP tolerance; None uses atol.
        gradient_rtol: Optional VJP tolerance; None uses rtol.
        modes: Verification autograd contexts, not a Module execution switch.
            Inference-only kernels can select inference and no_grad. A training
            context with frozen inputs does not prove training-kernel execution.
    """

    atol: float = 1e-5
    rtol: float = 1e-5
    check_gradients: bool = True
    gradient_atol: float | None = None
    gradient_rtol: float | None = None
    modes: tuple[str, ...] = ("training", "inference")


def _storage_bytes(value):
    storage = value.untyped_storage()
    raw = torch.empty(0, device=value.device, dtype=torch.uint8)
    return raw.set_(storage, 0, (storage.nbytes(),), (1,))


def _clone_inputs(inputs):
    """Isolate materialized strided inputs while retaining their storage
    graph."""
    storages, identities, copied = {}, {}, []
    with torch.no_grad():
        for value in inputs:
            if value.layout != torch.strided or value.device.type == "meta":
                raise ValueError("Parity inputs must be materialized strided tensors")
            if id(value) not in identities:
                storage = value.untyped_storage()
                key = (value.device, storage._cdata)
                if key not in storages:
                    # Clone bytes once, so mixed-dtype aliases retain storage too.
                    storages[key] = _storage_bytes(value).clone().untyped_storage()
                clone = torch.empty(0, device=value.device, dtype=value.dtype)
                clone.set_(
                    storages[key], value.storage_offset(), value.shape, value.stride()
                )
                clone = clone.detach().requires_grad_(value.requires_grad)
                identities[id(value)] = clone
            copied.append(identities[id(value)])
    return tuple(copied)


def _copy_inputs(destinations, sources):
    """Restore aliased or expanded input views through their backing stores."""
    copied = set()
    with torch.no_grad():
        for dst, src in zip(destinations, sources, strict=True):
            key = (dst.device, dst.untyped_storage()._cdata)
            if key not in copied:
                _storage_bytes(dst).copy_(_storage_bytes(src))
                copied.add(key)


def _outputs(result):
    values, structure = tree_flatten(result)
    if any(not isinstance(value, Tensor) for value in values):
        raise TypeError("Parity outputs must be a tensor pytree")
    return values, structure


def _vjp(inputs, outputs, cotangents):
    active = list(dict.fromkeys(value for value in inputs if value.requires_grad))
    selected = [
        (value, grad) for value, grad in zip(outputs, cotangents) if value.requires_grad
    ]
    if not active or not selected:
        return (None,) * len(inputs)
    gradients = torch.autograd.grad(
        [value for value, _ in selected],
        active,
        [grad for _, grad in selected],
        allow_unused=True,
    )
    lookup = {id(value): grad for value, grad in zip(active, gradients)}
    return tuple(lookup.get(id(value)) for value in inputs)


def check_model_parity(
    cases: list[ModelParityCase], *, options: ModelParityOptions = ModelParityOptions()
) -> ParityReport:
    """Compare inference, training outputs and declared input gradients.

    Output pytree leaves include final states and user records, so their VJPs
    also participate in backward checks. A local seeded nonperiodic cotangent
    samples one VJP, not the complete Jacobian. Supply separate cases for edge
    cases, continuation and alternative workloads; no time layout is assumed.

    Args:
        cases: Independent executions using the actual Python model as golden.
        options: Numerical tolerances and gradient-check selection.

    Returns:
        Existing btorch parity report, including explicit unavailable checks.
    """
    if not options.modes or any(
        mode not in ("inference", "no_grad", "training") for mode in options.modes
    ):
        raise ValueError("modes must select inference, no_grad or training contexts")
    report = ParityReport("native")
    for case in cases:
        if case.input_names and len(case.input_names) != len(case.inputs):
            raise ValueError("input_names must name every input tensor")
        for mode in options.modes:
            label = f"{case.name}/{mode}"
            context = {
                "inference": torch.inference_mode,
                "no_grad": torch.no_grad,
                "training": torch.enable_grad,
            }[mode]
            try:
                with context():
                    expected_inputs = _clone_inputs(case.inputs)
                    actual_inputs = _clone_inputs(case.inputs)
                    expected_result = case.golden(*expected_inputs)
                    expected, expected_tree = _outputs(expected_result)
                    named_leaves, _ = tree_flatten_with_path(expected_result)
                    output_names = tuple(keystr(path) for path, _ in named_leaves)
                    if set(case.exact_outputs).difference(output_names):
                        raise ValueError(
                            "exact_outputs must name existing output tensor paths"
                        )
                    snapshots = (
                        tuple(v.detach().clone() for v in actual_inputs)
                        if case.immutable_inputs
                        else ()
                    )
                    actual, actual_tree = _outputs(case.candidate(*actual_inputs))
                    for index, before in enumerate(snapshots):
                        if not torch.equal(actual_inputs[index].detach(), before):
                            name = (
                                case.input_names[index]
                                if case.input_names
                                else str(index)
                            )
                            raise RuntimeError(f"callback mutated input {name!r}")
                    if expected_tree != actual_tree:
                        raise ValueError("Candidate output pytree differs from golden")
                    for index, (value, reference) in enumerate(zip(actual, expected)):
                        report.comparisons.append(
                            _compare(
                                f"{label}/output/{output_names[index]}",
                                value,
                                reference,
                                options.atol,
                                options.rtol,
                                spikes=output_names[index] in case.exact_outputs,
                            )
                        )
                    if mode == "training" and not any(
                        value.requires_grad for value in case.inputs
                    ):
                        report.skipped.append(
                            f"{label}: frozen inputs may select inference; "
                            "not evidence of training-kernel execution"
                        )
                    if mode != "training" or not options.check_gradients:
                        continue
                    if not any(value.requires_grad for value in case.inputs):
                        report.skipped.append(
                            f"{label}/backward: no inputs request gradients"
                        )
                        continue
                    cotangents = tuple(
                        weight.to(value.dtype)
                        for value, weight in zip(expected, _loss_weights(expected))
                    )
                    expected_gradients = (
                        case.gradient_oracle(expected_inputs, expected, cotangents)
                        if case.gradient_oracle is not None
                        else _vjp(expected_inputs, expected, cotangents)
                    )
                    actual_gradients = _vjp(actual_inputs, actual, cotangents)
                    if len(expected_gradients) != len(case.inputs):
                        raise ValueError(
                            "Gradient oracle must return one gradient per input"
                        )
                    for index, (value, reference) in enumerate(
                        zip(actual_gradients, expected_gradients)
                    ):
                        if not case.inputs[index].requires_grad:
                            continue
                        input_name = (
                            case.input_names[index] if case.input_names else str(index)
                        )
                        name = f"{label}/backward/input/{input_name}"
                        # Torch materializes unused gradients as None, while a
                        # valid native backward may return explicit zeros. The
                        # numerical contract compares their mathematical VJP.
                        detail = (
                            f"candidate grad={value is not None}; "
                            f"oracle grad={reference is not None}"
                        )
                        value = (
                            torch.zeros_like(actual_inputs[index])
                            if value is None
                            else value
                        )
                        reference = (
                            torch.zeros_like(expected_inputs[index])
                            if reference is None
                            else reference
                        )
                        comparison = _compare(
                            name,
                            value,
                            reference,
                            options.atol
                            if options.gradient_atol is None
                            else options.gradient_atol,
                            options.rtol
                            if options.gradient_rtol is None
                            else options.gradient_rtol,
                        )
                        report.comparisons.append(
                            replace(comparison, detail=f"{detail}; {comparison.detail}")
                        )
            except Exception as error:
                report.errors.append(f"{label}: {type(error).__name__}: {error}")
    return report


@dataclass(frozen=True)
class ParityCase:
    """Declare one cell workload; stimuli and edge cases remain explicit.

    Args:
        x_seq: Time-first drive, preserving dtype and storage layout.
        states: Initial physical states in template order.
        parameters: Floating parameters in template order.
        spec: Declared model math.
        history: Records participating in output and full-VJP comparison.
        name: Physical scenario name.
        golden: Independent execution; defaults to original Torch Modules.
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
    """Select cell execution checks without expanding workload combinations.

    Args:
        atol: Absolute tolerance, or dtype-dependent default.
        rtol: Relative tolerance, or dtype-dependent default.
        check_gradients: Compare all drive, state and parameter VJPs.
        modes: Explicit inference/no_grad/training contexts.
        compile: Also compare one compiled training execution.
        capture: Also compare captured inference and full-VJP replay on CUDA.
    """

    atol: float | None = None
    rtol: float | None = None
    check_gradients: bool = True
    modes: tuple[str, ...] = ("training", "inference")
    compile: bool = False
    capture: bool = False


def check_backend_parity(
    backend: str, cases: list[ParityCase], *, options: ParityOptions = ParityOptions()
) -> ParityReport:
    """Adapt explicitly selected cell cases to the shared execution comparator.

    Empty/single-step drives, record selection, trainability and
    continuation are caller-owned cases. This function never
    manufactures a case matrix.
    """
    report = ParityReport(backend)
    for case in cases:
        template = get_template(case.spec)
        n_states = len(case.states)
        input_names = ("x_seq", *template.state_names, *template.parameter_names)
        output_names = (
            "output_seq",
            *("final/" + n for n in template.state_names),
            *("record/" + n for n in case.history.requested),
        )
        # Preserve storage aliases and strides before enabling the declared full VJP.
        values = _clone_inputs((case.x_seq, *case.states, *case.parameters))
        for value in values:
            value.requires_grad_(
                options.check_gradients
                and (value.is_floating_point() or value.is_complex())
            )

        def golden(*inputs):
            output, final, records = (case.golden or python_module_sequence)(
                inputs[0],
                inputs[1 : 1 + n_states],
                inputs[1 + n_states :],
                case.spec,
                case.history.requested,
            )
            return dict(zip(output_names, (output, *final, *records), strict=True))

        def candidate(*inputs):
            result = run_multi_step(
                inputs[0],
                inputs[1 : 1 + n_states],
                inputs[1 + n_states :],
                case.spec,
                case.history,
                backend=backend,
            )
            return dict(zip(output_names, result, strict=True))

        exact = ("['output_seq']",) if "v" in template.state_names else ()
        model_case = ModelParityCase(
            case.name,
            golden,
            candidate,
            values,
            input_names=input_names,
            exact_outputs=exact,
            immutable_inputs=True,
        )
        tolerance = {torch.float64: 1e-7, torch.float32: 3e-5}.get(
            case.x_seq.dtype, 5e-3
        )
        model_options = ModelParityOptions(
            atol=tolerance if options.atol is None else options.atol,
            rtol=tolerance if options.rtol is None else options.rtol,
            check_gradients=options.check_gradients,
            modes=options.modes,
        )
        result = check_model_parity([model_case], options=model_options)
        report.comparisons.extend(result.comparisons)
        report.errors.extend(result.errors)
        report.skipped.extend(result.skipped)
        if options.compile:
            compiled = replace(
                model_case,
                name=case.name + "/compile",
                candidate=torch.compile(candidate, fullgraph=True),
            )
            result = check_model_parity(
                [compiled], options=replace(model_options, modes=("training",))
            )
            report.comparisons.extend(result.comparisons)
            report.errors.extend(result.errors)
        if options.capture:
            if not case.x_seq.is_cuda:
                report.skipped.append(f"{case.name}/capture: CUDA inputs required")
            else:
                _check_capture(model_case, model_options, report)
    return report


def _check_capture(case, options, report):
    """Replay owned inference and VJP buffers without re-running a case
    matrix."""
    for training in (False, True) if options.check_gradients else (False,):
        mode = "capture_training" if training else "capture"
        try:
            with torch.inference_mode(False), torch.set_grad_enabled(training):
                inputs = _clone_inputs(case.inputs)
                if not training:
                    inputs = tuple(value.detach() for value in inputs)
                # Build cotangents before capture; local generator construction is
                # host work. The canonical comparator still owns the golden/VJP.
                sample, _ = _outputs(case.candidate(*inputs))
                cotangents = _loss_weights(sample)
                # Release the probe graph before warming on the capture stream;
                # otherwise its leaf accumulators remain bound to this stream.
                del sample

                def operation():
                    result = case.candidate(*inputs)
                    flat, _ = _outputs(result)
                    gradients = _vjp(inputs, flat, cotangents) if training else ()
                    return result, gradients

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    operation()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured, gradients = operation()

                def replay(*new_inputs):
                    _copy_inputs(inputs, new_inputs)
                    graph.replay()
                    graph.replay()  # Detect persistent gradient accumulation.
                    return captured

                # The captured forward has no eager graph in inference, and the
                # VJP is an explicit captured output in training. Compare both as
                # named tensor outputs through the same primitive.
                def with_vjp(fn, *new_inputs):
                    result = fn(*new_inputs)
                    flat, _ = _outputs(result)
                    grads = _vjp(new_inputs, flat, cotangents) if training else ()
                    return {
                        "forward": result,
                        "vjp": tuple(
                            torch.zeros_like(v) if g is None else g
                            for v, g in zip(new_inputs, grads)
                        ),
                    }

                def captured_result(*new_inputs):
                    result = replay(*new_inputs)
                    return {
                        "forward": result,
                        "vjp": tuple(
                            torch.zeros_like(v) if g is None else g
                            for v, g in zip(inputs, gradients)
                        ),
                    }

                check = ModelParityCase(
                    case.name + "/" + mode,
                    lambda *args: with_vjp(case.golden, *args),
                    captured_result,
                    case.inputs,
                    input_names=case.input_names,
                    exact_outputs=tuple(
                        "['forward']" + name for name in case.exact_outputs
                    ),
                )
                result = check_model_parity(
                    [check],
                    options=replace(
                        options,
                        check_gradients=False,
                        modes=("training",) if training else ("inference",),
                    ),
                )
                report.comparisons.extend(result.comparisons)
                report.errors.extend(result.errors)
        except Exception as error:
            report.errors.append(f"{case.name}/{mode}: {type(error).__name__}: {error}")
