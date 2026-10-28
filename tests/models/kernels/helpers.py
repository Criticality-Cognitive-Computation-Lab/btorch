"""Shared numerical oracles and observations for native-kernel tests.

Keep model-specific drives, physical expectations, and edge cases in the
tests. These helpers only coordinate existing Module goldens and Torch
autograd.
"""

import importlib.util
from dataclasses import dataclass

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from btorch.models.kernels import HistoryPlan, KernelOptions, NeuronSpec
from btorch.models.kernels.golden import python_module_sequence
from btorch.models.kernels.operators import run_multi_step
from btorch.models.kernels.templates import get_template
from btorch.models.kernels.testing import (
    ModelParityCase,
    ModelParityOptions,
    check_model_parity,
)


GPU_BACKENDS = ("triton", "cupy", "tilelang")


def validation_cases(cases, *, selected):
    """Keep selected scenarios routine and reserve the rest for full
    validation.

    Cases describe the same numerical oracle. Selection belongs next to
    that oracle, where the physics or renderer feature motivating each
    case is clear. Full validation retains every original combination
    without duplicating tests.
    """
    cases = tuple(cases)
    selected = set(selected)
    missing = selected.difference(cases)
    if missing:
        raise ValueError(
            f"Selected validation cases are absent from the matrix: {missing}"
        )
    return [
        pytest.param(
            *case, marks=() if case in selected else pytest.mark.full_validation
        )
        for case in cases
    ]


def require_cuda_backend(backend: str) -> None:
    """Skip when the requested optional CUDA implementation is unavailable."""
    if not torch.cuda.is_available() or importlib.util.find_spec(backend) is None:
        pytest.skip(f"{backend} CUDA runtime is unavailable")


class NewEmptyShapes(TorchDispatchMode):
    """Observe sequence-workspace allocations without intercepting their
    values."""

    def __init__(self):
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.new_empty.default:
            self.shapes.append(tuple(result.shape))
        return result


@dataclass(frozen=True)
class SequenceParityOptions:
    """Separate forward and VJP tolerances for a sequence oracle."""

    forward_atol: float = 3e-5
    forward_rtol: float = 3e-5
    gradient_atol: float = 5e-5
    gradient_rtol: float = 5e-4


def assert_module_sequence(
    backend: str,
    spec: NeuronSpec,
    x: torch.Tensor,
    states: tuple[torch.Tensor, ...],
    parameters: tuple[torch.Tensor, ...],
    histories: tuple[str, ...],
    *,
    parameter_roots: tuple[torch.Tensor, ...] | None = None,
    options: SequenceParityOptions = SequenceParityOptions(),
) -> tuple[torch.Tensor, ...]:
    """Compare every output and its VJP with the actual stateful Module.

    Backing-storage roots may differ from the parameter views passed to
    the kernel. Identity deduplication also preserves shared-parameter
    cases.
    """
    roots = parameters if parameter_roots is None else parameter_roots
    root_indices = {id(value): index for index, value in enumerate(roots)}
    parameter_layouts = tuple(
        (
            root_indices[id(root)],
            tuple(parameter.shape),
            tuple(parameter.stride()),
            parameter.storage_offset(),
        )
        for parameter, root in zip(parameters, roots, strict=True)
    )

    def bind_parameters(root_values):
        return tuple(
            root_values[index].as_strided(shape, stride, offset)
            for index, shape, stride, offset in parameter_layouts
        )

    def golden(*inputs):
        parameter_values = bind_parameters(inputs[1 + len(states) :])
        output, final, records = python_module_sequence(
            inputs[0], inputs[1 : 1 + len(states)], parameter_values, spec, histories
        )
        return output, *final, *records

    actual_outputs = []

    def candidate(*inputs):
        parameter_values = bind_parameters(inputs[1 + len(states) :])
        actual = run_multi_step(
            inputs[0],
            inputs[1 : 1 + len(states)],
            parameter_values,
            spec,
            HistoryPlan(requested=histories),
            backend=KernelOptions(neuron=backend),
        )
        actual_outputs.append(actual)
        return actual

    template = get_template(spec)
    input_names = (
        "x",
        *template.state_names,
        *(f"parameter_root_{index}" for index in range(len(roots))),
    )
    report = check_model_parity(
        [
            ModelParityCase(
                "module_sequence",
                golden,
                candidate,
                (x, *states, *roots),
                input_names=input_names,
                immutable_inputs=True,
            )
        ],
        options=ModelParityOptions(
            atol=options.forward_atol,
            rtol=options.forward_rtol,
            gradient_atol=options.gradient_atol,
            gradient_rtol=options.gradient_rtol,
            modes=("training",),
        ),
    )
    report.assert_close()
    return actual_outputs[-1]
