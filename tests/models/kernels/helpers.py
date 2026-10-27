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
    """Tolerances and reproducible cotangent sampling for a sequence oracle."""

    forward_atol: float = 3e-5
    forward_rtol: float = 3e-5
    gradient_atol: float = 5e-5
    gradient_rtol: float = 5e-4
    seed: int = 741


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
    output, final, records = python_module_sequence(
        x, states, parameters, spec, histories
    )
    expected = (output, *final, *records)
    actual = run_multi_step(
        x,
        states,
        parameters,
        spec,
        HistoryPlan(requested=histories),
        backend=KernelOptions(neuron=backend),
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            result, reference, atol=options.forward_atol, rtol=options.forward_rtol
        )
    generator = torch.Generator(device=x.device).manual_seed(options.seed)
    cotangents = tuple(
        torch.randn(
            value.shape, generator=generator, device=value.device, dtype=value.dtype
        )
        for value in actual
    )
    roots = parameters if parameter_roots is None else parameter_roots
    inputs = tuple({id(value): value for value in (x, *states, *roots)}.values())
    gradients = torch.autograd.grad(actual, inputs, cotangents)
    references = torch.autograd.grad(expected, inputs, cotangents)
    for result, reference in zip(gradients, references, strict=True):
        torch.testing.assert_close(
            result, reference, atol=options.gradient_atol, rtol=options.gradient_rtol
        )
    return actual


def cell_inputs(model, refractory=False):
    """Build heterogeneous state/parameter leaves for the shared cell
    contracts."""
    spec = NeuronSpec(model, dt=1, refractory=refractory)
    template = get_template(spec)
    values = dict(
        v_threshold=1.0,
        v_reset=0.0,
        c_m=1.2,
        tau=3.0,
        v_rest=-0.2,
        k=0.1,
        asc_amps=0.2,
        v_peak=1.3,
        a=0.1,
        b=0.2,
        d=0.3,
        tau_syn=2.0,
        g_max=0.7,
        syn_decay=0.6,
        tau_rise=2.0,
        tau_decay=4.0,
        tau_ref=2.0,
    )
    x = (torch.randn(2, 3) * 0.4 + 0.6).requires_grad_()
    states = tuple(
        (
            torch.randn(2, 3, 2) * 0.2 if n == "Iasc" else torch.randn(2, 3) * 0.2
        ).requires_grad_()
        for n in template.state_names
    )
    if refractory:
        states = (
            *states[:-1],
            torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0]], requires_grad=True),
        )
    parameters = tuple(
        torch.full(
            (3, 2) if n in ("k", "asc_amps") and model == "glif3" else (3,), values[n]
        ).requires_grad_()
        for n in template.parameter_names
    )
    return spec, x, states, parameters
