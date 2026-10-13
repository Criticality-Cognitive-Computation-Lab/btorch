"""Bind native Torch operators and compare complete user model executions."""

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten

from .testing import ParityReport, _compare


@dataclass(frozen=True)
class KernelBinding:
    """Select inference or training without replacing Torch autograd.

    Args:
        inference: Callable returning output, final state and requested records.
            It must not allocate backward-only tape.
        training: Callable with native Torch autograd support. Its backward may
            use an entirely different algorithm. All participating parameters
            and initial states must be explicit tensor inputs.
    """

    inference: Callable
    training: Callable | None = None

    def __call__(self, *inputs):
        leaves, _ = tree_flatten(inputs)
        needs_backward = torch.is_grad_enabled() and any(
            isinstance(value, Tensor) and value.requires_grad for value in leaves
        )
        if needs_backward:
            if self.training is None:
                raise RuntimeError("This kernel binding has no training implementation")
            return self.training(*inputs)
        return self.inference(*inputs)


def load_kernel_package(module: str | Path) -> Mapping[str, KernelBinding]:
    """Import a trusted Python kernel package and return its declared bindings.

    A package exposes ``KERNEL_BINDINGS``, a nonempty mapping of names to
    :class:`KernelBinding`. Native operator registration remains the package's
    responsibility. A path may identify a Python file or package directory.
    Import executes Python code; use only sources you trust.

    Args:
        module: Importable module name or durable source path.

    Returns:
        The package's validated binding mapping.
    """
    source = Path(module)
    if source.exists():
        source = source.resolve()
        if source.is_dir():
            source = source / "__init__.py"
        name = "_btorch_oot_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, source)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot import kernel package {source}")
            package = importlib.util.module_from_spec(spec)
            sys.modules[name] = package
            try:
                spec.loader.exec_module(package)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        package = sys.modules[name]
    else:
        if isinstance(module, Path):
            raise FileNotFoundError(module)
        package = importlib.import_module(module)
    bindings = getattr(package, "KERNEL_BINDINGS", None)
    if (
        not isinstance(bindings, Mapping)
        or not bindings
        or any(
            not isinstance(name, str) or not isinstance(binding, KernelBinding)
            for name, binding in bindings.items()
        )
    ):
        raise ValueError(
            "Kernel package must declare a nonempty KERNEL_BINDINGS mapping"
        )
    return bindings


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
        gradient_oracle: Optional callable ``(inputs, outputs, cotangents)``
            returning one gradient per input for alternative gradient semantics.
    """

    name: str
    golden: Callable
    candidate: Callable
    inputs: tuple[Tensor, ...]
    gradient_oracle: Callable | None = None


@dataclass(frozen=True)
class ModelParityOptions:
    """Configure whole-operator numerical comparison.

    Args:
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        check_gradients: Compare VJPs of inputs marked requires_grad.
        modes: Verification autograd contexts, not a Module execution switch.
            Inference-only kernels can select inference and no_grad. A training
            context with frozen inputs does not prove training-kernel execution.
    """

    atol: float = 1e-5
    rtol: float = 1e-5
    check_gradients: bool = True
    modes: tuple[str, ...] = ("inference", "no_grad", "training")


def _clone_inputs(inputs):
    # Preserve strides, shared storage and repeated identities. Independent
    # golden/candidate storage prevents in-place state updates contaminating
    # the other execution. Overlapping views remain separate autograd leaves.
    storages, identities, copied = {}, {}, []
    with torch.no_grad():
        for value in inputs:
            if id(value) not in identities:
                key = (value.device, value.dtype, value.untyped_storage().data_ptr())
                if key not in storages:
                    size = value.untyped_storage().nbytes() // value.element_size()
                    storages[key] = torch.empty(
                        size, device=value.device, dtype=value.dtype
                    )
                    storages[key].copy_(value.detach().as_strided((size,), (1,), 0))
                clone = storages[key].as_strided(
                    value.shape, value.stride(), value.storage_offset()
                )
                clone = clone.detach().requires_grad_(value.requires_grad)
                identities[id(value)] = clone
            copied.append(identities[id(value)])
    return tuple(copied)


def _outputs(result):
    values, structure = tree_flatten(result)
    if any(not isinstance(value, Tensor) for value in values):
        raise TypeError("Parity outputs must be a tensor pytree")
    return values, structure


def _cotangents(outputs):
    # One reproducible nonperiodic VJP probe, not complete Jacobian validation.
    # Separate local generators avoid changing user stochastic model execution.
    values = []
    for index, value in enumerate(outputs):
        generator = torch.Generator(device=value.device).manual_seed(1729 + index)
        values.append(
            torch.randn(
                value.shape,
                device=value.device,
                dtype=torch.float32,
                generator=generator,
            ).to(value.dtype)
        )
    return tuple(values)


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
                    expected, expected_tree = _outputs(case.golden(*expected_inputs))
                    actual, actual_tree = _outputs(case.candidate(*actual_inputs))
                    if expected_tree != actual_tree:
                        raise ValueError("Candidate output pytree differs from golden")
                    for index, (value, reference) in enumerate(zip(actual, expected)):
                        report.comparisons.append(
                            _compare(
                                f"{label}/output/{index}",
                                value,
                                reference,
                                options.atol,
                                options.rtol,
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
                    cotangents = _cotangents(expected)
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
                        name = f"{label}/backward/input/{index}"
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
                            name, value, reference, options.atol, options.rtol
                        )
                        report.comparisons.append(replace(comparison, detail=detail))
            except Exception as error:
                report.errors.append(f"{label}: {type(error).__name__}: {error}")
    return report
