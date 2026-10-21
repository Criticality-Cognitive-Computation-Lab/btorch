"""Validate selective primitive reverse outputs against actual Modules."""

import pytest
import torch

from btorch.models.kernels import CellSpec, KernelOptions, get_template
from btorch.models.kernels.backends import engine
from btorch.models.kernels.golden import python_module_sequence
from btorch.models.kernels.operators import run_multi_step
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


MODEL_SCOPES = (
    ("if", "encoder"),
    ("lif", "parameters"),
    ("glif3", "initial_state"),
    ("izhikevich", "encoder"),
    ("exponential", "parameters"),
    ("alpha", "initial_state"),
    ("alpha", "parameters"),
    ("alpha_billeh", "encoder"),
    ("dual_exponential", "initial_state"),
)


def _parameter(value, shape, layout, device):
    if layout == "scalar":
        return torch.tensor(value, device=device)
    if layout == "per_neuron":
        return torch.full(shape, value, device=device)
    storage = torch.full((shape[0] * 2, *shape[1:]), -7.0, device=device)
    storage[::2] = value
    return storage[::2]


def _values(spec, scope):
    torch.manual_seed(712)
    template = get_template(spec)
    batch, neurons, modes = 2, 4, 2
    device = torch.device("cuda")
    x = torch.rand(5, batch, neurons, device=device) * 0.35 + 0.55
    defaults = {
        "v_threshold": 0.8,
        "v_reset": -0.1,
        "c_m": 1.2,
        "tau": 3.0,
        "v_rest": 0.0,
        "k": 0.12,
        "asc_amps": 0.04,
        "v_peak": 1.4,
        "a": 0.1,
        "b": 0.2,
        "d": 0.15,
        "tau_syn": 2.5,
        "g_max": 0.7,
        "syn_decay": 0.6,
        "tau_rise": 2.0,
        "tau_decay": 4.5,
    }
    states = tuple(
        torch.rand(
            (batch, neurons, modes) if name == "Iasc" else (batch, neurons),
            device=device,
        )
        * 0.08
        for name in template.state_names
    )
    layouts = ("scalar", "per_neuron", "strided")
    parameters = []
    for index, name in enumerate(template.parameter_names):
        shape = (
            (neurons, modes)
            if spec.model == "glif3" and name in ("k", "asc_amps")
            else (neurons,)
        )
        parameters.append(
            _parameter(defaults[name], shape, layouts[index % len(layouts)], device)
        )

    state_root = {
        "glif3": 1,
        "alpha": 1,
        "dual_exponential": 1,
    }.get(spec.model, 0)
    x = x.detach().requires_grad_(scope == "encoder")
    states = tuple(
        value.detach().requires_grad_(scope == "initial_state" and i == state_root)
        for i, value in enumerate(states)
    )
    parameters = tuple(
        value.detach().requires_grad_(scope == "parameters") for value in parameters
    )
    return x, states, parameters


def _clone(values):
    return tuple(
        value.detach().clone().requires_grad_(value.requires_grad) for value in values
    )


def _loss(outputs):
    return sum(
        (index + 1) * value.square().mean() for index, value in enumerate(outputs)
    )


def _differentiable(x, states, parameters):
    return tuple(value for value in (x, *states, *parameters) if value.requires_grad)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize(("model", "scope"), MODEL_SCOPES)
def test_selective_reverse_outputs_match_actual_module(backend, model, scope):
    require_cuda_backend(backend)
    spec = CellSpec(
        model,
        dt=1.0,
        hard_reset=model in ("if", "lif", "glif3", "izhikevich"),
        surrogate="atan",
    )
    x, states, parameters = _values(spec, scope)
    ref_x = x.detach().clone().requires_grad_(x.requires_grad)
    ref_states = _clone(states)
    ref_parameters = _clone(parameters)

    actual = run_multi_step(
        x,
        states,
        parameters,
        spec,
        backend=KernelOptions(
            neuron=backend, enable_fp_fusion=False, precise_division=True
        ),
    )
    expected_output, expected_final, _ = python_module_sequence(
        ref_x, ref_states, ref_parameters, spec
    )
    expected = (expected_output, *expected_final)
    actual_inputs = _differentiable(x, states, parameters)
    expected_inputs = _differentiable(ref_x, ref_states, ref_parameters)
    actual_gradients = torch.autograd.grad(
        _loss(actual), actual_inputs, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        _loss(expected), expected_inputs, allow_unused=True
    )
    for actual_gradient, expected_gradient, value in zip(
        actual_gradients, expected_gradients, actual_inputs, strict=True
    ):
        torch.testing.assert_close(
            torch.zeros_like(value) if actual_gradient is None else actual_gradient,
            torch.zeros_like(value) if expected_gradient is None else expected_gradient,
            atol=5e-5,
            rtol=5e-4,
        )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("scope", ("encoder", "parameters", "initial_state"))
def test_selective_roots_omit_unrequested_reverse_allocations(
    backend, scope, monkeypatch
):
    require_cuda_backend(backend)
    spec = CellSpec("lif", dt=0.5, hard_reset=True, surrogate="atan")
    x, states, parameters = _values(spec, scope)
    observed = []
    original = engine._launch

    def capture(*args, **kwargs):
        if kwargs.get("backward", False):
            observed.append(set(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_launch", capture)
    result = run_multi_step(
        x,
        states,
        parameters,
        spec,
        backend=KernelOptions(neuron=backend, enable_fp_fusion=False),
    )
    torch.autograd.grad(
        _loss(result), _differentiable(x, states, parameters), allow_unused=True
    )
    assert len(observed) == 1
    buffers = observed[0]
    assert ("grad_x_seq" in buffers) == (scope == "encoder")
    assert ("grad_v_init" in buffers) == (scope == "initial_state")
    for name in get_template(spec).parameter_names:
        assert (("grad_" + name) in buffers) == (scope == "parameters")


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize(
    "model", tuple(dict.fromkeys(model for model, _ in MODEL_SCOPES))
)
def test_reverse_source_abi_tracks_requested_gradient_targets(backend, model):
    spec = CellSpec(model, hard_reset=True)
    template = get_template(spec)
    modes = 2 if model == "glif3" else 0
    no_outputs = engine.generate_source(
        backend,
        spec,
        8,
        5,
        (),
        modes=modes,
        backward=True,
        gradient_targets=(False, (False,) * len(template.state_names)),
    )
    input_only = engine.generate_source(
        backend,
        spec,
        8,
        5,
        (),
        modes=modes,
        backward=True,
        gradient_targets=(True, (False,) * len(template.state_names)),
    )
    selected_state = tuple(index == 0 for index in range(len(template.state_names)))
    state_only = engine.generate_source(
        backend,
        spec,
        8,
        5,
        (),
        modes=modes,
        backward=True,
        gradient_targets=(False, selected_state),
    )
    no_names = {name for name, _ in no_outputs[1]}
    input_names = {name for name, _ in input_only[1]}
    state_names = {name for name, _ in state_only[1]}
    assert "grad_x_seq" not in no_names
    assert not any(
        "grad_" + name + "_init" in no_names for name in template.state_names
    )
    assert input_names == no_names | {"grad_x_seq"}
    assert state_names == no_names | {"grad_" + template.state_names[0] + "_init"}
    assert len({no_outputs[0], input_only[0], state_only[0]}) == 3


def test_empty_sequence_returns_only_requested_identity_gradients(monkeypatch):
    spec = CellSpec("dual_exponential", dt=0.5)
    x = torch.empty(0, 2, 3)
    states = (
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3),
    )
    parameters = (
        torch.tensor(2.0),
        torch.tensor(4.0),
        torch.linspace(0.6, 0.8, 3),
    )
    expected_output, expected_final, _ = python_module_sequence(
        x, states, parameters, spec
    )
    assert expected_output.shape == x.shape
    for actual, expected in zip(expected_final, states, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    selected_state = 1
    selected_parameter = 2
    cotangents = tuple(torch.randn(2, 3) for _ in states)
    needs = (
        False,
        *(index == selected_state for index in range(len(states))),
        *(index == selected_parameter for index in range(len(parameters))),
    )
    empty_like_calls = []
    zeros_like_calls = []
    original_empty_like = torch.empty_like
    original_zeros_like = torch.zeros_like

    def record_empty_like(value, *args, **kwargs):
        empty_like_calls.append(value)
        return original_empty_like(value, *args, **kwargs)

    def record_zeros_like(value, *args, **kwargs):
        zeros_like_calls.append(value)
        return original_zeros_like(value, *args, **kwargs)

    monkeypatch.setattr(engine.torch, "empty_like", record_empty_like)
    monkeypatch.setattr(engine.torch, "zeros_like", record_zeros_like)
    grad_x, grad_states, grad_parameters = engine.backward(
        "triton",
        x,
        states,
        parameters,
        spec,
        (),
        (),
        None,
        cotangents,
        (),
        needs_input_grad=needs,
    )
    assert grad_x is None
    assert empty_like_calls == []
    assert zeros_like_calls == [parameters[selected_parameter]]
    for index, gradient in enumerate(grad_states):
        if index == selected_state:
            torch.testing.assert_close(
                gradient,
                cotangents[index].sum_to_size(states[index].shape),
                atol=0,
                rtol=0,
            )
        else:
            assert gradient is None
    for index, gradient in enumerate(grad_parameters):
        if index == selected_parameter:
            torch.testing.assert_close(
                gradient, torch.zeros_like(parameters[index]), atol=0, rtol=0
            )
        else:
            assert gradient is None
