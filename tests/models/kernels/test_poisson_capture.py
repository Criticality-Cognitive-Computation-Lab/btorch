"""Exercise native Poisson cells through compile and CUDA Graph replay."""

import copy
from unittest.mock import patch

import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.kernels import HistoryPlan, KernelOptions
from btorch.models.kernels.adapters import prepare_cell
from btorch.models.kernels.golden import python_module_sequence
from btorch.models.kernels.operators import poisson_uniforms, run_multi_step
from btorch.models.kernels.registry import resolve_cell_backend
from btorch.models.kernels.templates import get_template
from btorch.models.neurons import ALIF
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import PoissonRandomSpike
from btorch.models.synapse import ExponentialPSC
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


def _primitive_case(*, detach_reset):
    module = ALIF(
        4,
        v_threshold=torch.tensor([0.78, 0.86, 0.94, 1.02]),
        v_reset=-0.14,
        c_m=torch.tensor([1.0, 1.1, 1.2, 1.3]),
        g_leak=0.23,
        E_leak=-0.08,
        E_k=-1.0,
        tau_adapt=torch.tensor([4.4, 4.9, 5.4, 5.9]),
        dg_k=torch.tensor([0.08, 0.10, 0.12, 0.14]),
        tau_ref=0.75,
        trainable_param={
            "v_threshold",
            "v_reset",
            "c_m",
            "g_leak",
            "E_leak",
            "E_k",
            "tau_adapt",
            "dg_k",
            "tau_ref",
        },
        surrogate_function=PoissonRandomSpike(
            tau=2.4,
            rho=0.72,
            leak=0.18,
            k=1.08,
            damping_factor=0.57,
        ),
        hard_reset=True,
        detach_reset=detach_reset,
        pre_spike_v=True,
        device="cuda",
        dtype=torch.float32,
    )
    init_net_state(module, batch_size=2, device="cuda", dtype=torch.float32)
    lane = torch.arange(8, device="cuda").reshape(2, 4)
    module.v = (0.18 + 0.035 * lane).requires_grad_()
    module.g_k = (0.025 + 0.008 * lane).requires_grad_()
    refractory = 0.05 * (lane % 3)
    refractory[0, 0] = 3.0
    module.refractory = refractory.requires_grad_()
    generator = torch.Generator().manual_seed(817)
    x = (0.30 + 0.24 * torch.rand(6, 2, 4, generator=generator)).cuda()
    with environ.context(dt=0.25):
        spec, _, states, parameters = prepare_cell(module, x)
    return x, spec, states, parameters


def _clone_roots(x, states, parameters):
    return (
        x.detach().clone().requires_grad_(),
        *(
            value.detach().clone().requires_grad_(value.requires_grad)
            for value in states
        ),
        *(
            value.detach().clone().requires_grad_(value.requires_grad)
            for value in parameters
        ),
    )


def _golden_with_uniforms(x, states, parameters, spec, requested, uniforms):
    draws = iter(uniforms.unbind())

    def supplied_uniform(reference, *args, **kwargs):
        value = next(draws)
        assert value.shape == reference.shape
        return value

    with patch.object(torch, "rand_like", supplied_uniform):
        output, final, histories = python_module_sequence(
            x, states, parameters, spec, requested
        )
    with pytest.raises(StopIteration):
        next(draws)
    return (output, *final, *histories)


def test_opaque_uniform_boundary_preserves_per_step_cpu_draws():
    """The compiler boundary retains the ordinary Torch draw schedule."""
    x = torch.empty_strided((5, 2, 4), (16, 8, 2), dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(318)
        actual = poisson_uniforms(x)
        actual_rng = torch.get_rng_state()
        torch.manual_seed(318)
        reference = torch.empty(2, 4)
        expected = torch.stack(tuple(torch.rand_like(reference) for _ in range(5)))
        expected_rng = torch.get_rng_state()
    assert torch.equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)
    assert actual.dtype == torch.float32 and actual.is_contiguous()


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("detach_reset", (False, True))
def test_compiled_primitive_replays_changed_uniforms_and_all_vjps(
    backend, detach_reset, monkeypatch
):
    """Explicit uniforms make compiled replay and every root VJP reproducible.

    The golden remains the actual ALIF Module: ``rand_like`` is supplied with
    each public uniform in call order, while all voltage, reset, adaptation,
    refractory, and surrogate-gradient operations execute unchanged.
    """
    require_cuda_backend(backend)
    torch._dynamo.reset()
    base_x, spec, base_states, base_parameters = _primitive_case(
        detach_reset=detach_reset
    )
    resolve_cell_backend(backend, spec, base_x.device)
    requested = get_template(spec).history_names
    counts = len(base_states), len(base_parameters)
    replay_calls = []
    original_replay = torch.cuda.CUDAGraph.replay

    def counted_replay(graph):
        replay_calls.append(graph)
        return original_replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted_replay)

    def sequence(value, uniform, *roots):
        ns, np = counts
        return run_multi_step(
            value,
            tuple(roots[:ns]),
            tuple(roots[ns : ns + np]),
            spec,
            HistoryPlan(requested=requested),
            backend=KernelOptions(neuron=backend),
            uniform_seq=uniform,
        )

    compiled = torch.compile(sequence, mode="reduce-overhead", fullgraph=True)
    host_generator = torch.Generator().manual_seed(442)
    trial_x = (
        base_x,
        base_x * 0.67 + 0.09,
        base_x.flip(0) * 1.13 - 0.06,
    )
    trial_uniforms = tuple(
        torch.rand(base_x.shape, generator=host_generator).cuda() for _ in trial_x
    )
    replay_deltas = []
    spike_counts = []
    for value, uniforms in zip(trial_x, trial_uniforms, strict=True):
        roots = _clone_roots(value, base_states, base_parameters)
        expected_roots = _clone_roots(value, base_states, base_parameters)
        before_rng = torch.cuda.get_rng_state()
        before_replay = len(replay_calls)
        torch.compiler.cudagraph_mark_step_begin()
        actual = compiled(roots[0], uniforms, *roots[1:])
        expected = _golden_with_uniforms(
            expected_roots[0],
            expected_roots[1 : 1 + counts[0]],
            expected_roots[1 + counts[0] :],
            spec,
            requested,
            uniforms,
        )
        assert torch.equal(torch.cuda.get_rng_state(), before_rng)
        assert torch.equal(actual[0], expected[0])
        spike_counts.append(int(actual[0].sum()))
        for result, reference in zip(actual[1:], expected[1:], strict=True):
            torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)

        cotangent_generator = torch.Generator().manual_seed(913)
        cotangents = tuple(
            torch.randn(result.shape, generator=cotangent_generator).cuda()
            for result in actual
        )
        gradients = torch.autograd.grad(actual, roots, cotangents, allow_unused=True)
        references = torch.autograd.grad(
            expected, expected_roots, cotangents, allow_unused=True
        )
        replay_deltas.append(len(replay_calls) - before_replay)
        assert torch.equal(torch.cuda.get_rng_state(), before_rng)
        for index, (gradient, reference) in enumerate(
            zip(gradients, references, strict=True)
        ):
            assert (gradient is None) == (reference is None), index
            if reference is not None:
                torch.testing.assert_close(
                    gradient, reference, atol=4e-5, rtol=4e-5, msg=f"root {index}"
                )
    assert replay_deltas == [0, 2, 2]
    assert any(count > 0 for count in spike_counts)
    assert any(count < base_x.numel() for count in spike_counts)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_inductor_primitive_default_rng_replay_matches_module_vjp(backend, monkeypatch):
    """Inductor cannot merge the T Torch draws across the custom-op
    boundary."""
    require_cuda_backend(backend)
    torch._dynamo.reset()
    base_x, spec, base_states, base_parameters = _primitive_case(detach_reset=False)
    resolve_cell_backend(backend, spec, base_x.device)
    requested = get_template(spec).history_names
    counts = len(base_states), len(base_parameters)
    replay_calls = []
    original_replay = torch.cuda.CUDAGraph.replay

    def counted_replay(graph):
        replay_calls.append(graph)
        return original_replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted_replay)

    def sequence(value, *roots):
        ns, np = counts
        return run_multi_step(
            value,
            tuple(roots[:ns]),
            tuple(roots[ns : ns + np]),
            spec,
            HistoryPlan(requested=requested),
            backend=backend,
        )

    compiled = torch.compile(sequence, mode="reduce-overhead", fullgraph=True)
    trials = (
        base_x,
        base_x * 0.69 + 0.08,
        base_x.roll(1, 0) * 1.11 - 0.05,
    )
    replay_deltas = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(6251)
        for value in trials:
            roots = _clone_roots(value, base_states, base_parameters)
            expected_roots = _clone_roots(value, base_states, base_parameters)
            initial_rng = torch.cuda.get_rng_state()
            before_replay = len(replay_calls)
            torch.compiler.cudagraph_mark_step_begin()
            actual = compiled(roots[0], *roots[1:])
            replay_rng = torch.cuda.get_rng_state()

            torch.cuda.set_rng_state(initial_rng)
            expected_output, expected_final, expected_histories = (
                python_module_sequence(
                    expected_roots[0],
                    expected_roots[1 : 1 + counts[0]],
                    expected_roots[1 + counts[0] :],
                    spec,
                    requested,
                )
            )
            expected = (expected_output, *expected_final, *expected_histories)
            assert torch.equal(torch.cuda.get_rng_state(), replay_rng)
            assert torch.equal(actual[0], expected[0])
            for result, reference in zip(actual[1:], expected[1:], strict=True):
                torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)

            cotangent_generator = torch.Generator().manual_seed(997)
            cotangents = tuple(
                torch.randn(result.shape, generator=cotangent_generator).cuda()
                for result in actual
            )
            rng_before_backward = torch.cuda.get_rng_state()
            gradients = torch.autograd.grad(
                actual, roots, cotangents, allow_unused=True
            )
            references = torch.autograd.grad(
                expected, expected_roots, cotangents, allow_unused=True
            )
            replay_deltas.append(len(replay_calls) - before_replay)
            assert torch.equal(torch.cuda.get_rng_state(), rng_before_backward)
            for index, (gradient, reference) in enumerate(
                zip(gradients, references, strict=True)
            ):
                assert (gradient is None) == (reference is None), index
                if reference is not None:
                    torch.testing.assert_close(
                        gradient,
                        reference,
                        atol=4e-5,
                        rtol=4e-5,
                        msg=f"root {index}",
                    )
    assert replay_deltas == [0, 2, 2]


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_primitive_cuda_graph_default_rng_matches_module(backend):
    """A captured default-RNG call advances exactly like the actual Module."""
    require_cuda_backend(backend)
    base_x, spec, states, parameters = _primitive_case(detach_reset=False)
    requested = get_template(spec).history_names
    static_x = base_x.detach().clone()

    def sequence(value):
        return run_multi_step(
            value,
            tuple(state.detach() for state in states),
            tuple(parameter.detach() for parameter in parameters),
            spec,
            HistoryPlan(requested=requested),
            backend=backend,
        )

    compiled = torch.compile(sequence, backend="eager", fullgraph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(2):
            compiled(static_x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
        captured = compiled(static_x)

    trials = (base_x, base_x * 0.71 + 0.08, base_x.roll(2, 0) * 1.09 - 0.04)
    spike_counts = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(7103)
        for value in trials:
            static_x.copy_(value)
            initial_rng = torch.cuda.get_rng_state()
            graph.replay()
            replay_rng = torch.cuda.get_rng_state()
            replay = tuple(result.clone() for result in captured)

            torch.cuda.set_rng_state(initial_rng)
            expected_output, expected_final, expected_histories = (
                python_module_sequence(
                    value,
                    tuple(state.detach().clone() for state in states),
                    tuple(parameter.detach().clone() for parameter in parameters),
                    spec,
                    requested,
                )
            )
            expected = (expected_output, *expected_final, *expected_histories)
            expected_rng = torch.cuda.get_rng_state()
            assert torch.equal(replay_rng, expected_rng)
            assert torch.equal(replay[0], expected[0])
            spike_counts.append(int(replay[0].sum()))
            for result, reference in zip(replay[1:], expected[1:], strict=True):
                torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
    assert len(set(spike_counts)) > 1


def _recurrent_model(backend, *, detach_reset):
    neurons, batch = 4, 2
    neuron = ALIF(
        neurons,
        v_threshold=0.82,
        v_reset=-0.12,
        c_m=1.1,
        g_leak=0.65,
        E_leak=-0.07,
        E_k=-0.8,
        tau_adapt=4.0,
        dg_k=0.1,
        tau_ref=0.75,
        trainable_param={
            "v_threshold",
            "v_reset",
            "c_m",
            "g_leak",
            "E_leak",
            "E_k",
            "tau_adapt",
            "dg_k",
            "tau_ref",
        },
        surrogate_function=PoissonRandomSpike(
            tau=2.4,
            rho=0.72,
            leak=0.18,
            k=1.08,
            damping_factor=0.57,
        ),
        hard_reset=True,
        detach_reset=detach_reset,
        pre_spike_v=True,
    )
    connection = torch.nn.Linear(neurons, neurons)
    with torch.no_grad():
        connection.weight.copy_(0.04 * torch.eye(neurons) + 0.015)
        connection.bias.copy_(torch.linspace(-0.01, 0.02, neurons))
    model = RecurrentNN(
        neuron,
        ExponentialPSC(neurons, 2.7, connection),
        backend=("torch" if backend == "torch" else KernelOptions(neuron=backend)),
        update_state_names=(),
    ).to(device="cuda", dtype=torch.float32)
    init_net_state(model, batch_size=batch, device="cuda", dtype=torch.float32)
    lane = torch.arange(batch * neurons, device="cuda").reshape(batch, neurons)
    model.neuron.v = (-0.08 + 0.03 * lane).requires_grad_()
    model.neuron.g_k = (0.025 + 0.006 * lane).requires_grad_()
    refractory = 0.05 * (lane % 3)
    refractory[0, 0] = 3.0
    model.neuron.refractory = refractory.requires_grad_()
    model.synapse.psc = (-0.03 + 0.009 * lane).requires_grad_()
    model.synapse.tau_syn.requires_grad_()
    return model


def _model_roots(model, x, events):
    roots = [x, events, model.neuron.v, model.neuron.g_k]
    roots.extend((model.neuron.refractory, model.synapse.psc, model.synapse.tau_syn))
    seen = {id(value) for value in roots}
    for parameter in model.parameters():
        if id(parameter) not in seen:
            roots.append(parameter)
            seen.add(id(parameter))
    return tuple(roots)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("detach_reset", (False, True))
def test_compiled_recurrent_default_rng_replay_matches_module_vjp(
    backend, detach_reset, monkeypatch
):
    """Completed recurrent steps replay changed inputs, RNG, state, and
    VJPs."""
    require_cuda_backend(backend)
    torch._dynamo.reset()
    model = _recurrent_model(backend, detach_reset=detach_reset)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    host_generator = torch.Generator().manual_seed(505)
    base_x = (0.30 + 0.28 * torch.rand(6, 2, 4, generator=host_generator)).cuda()
    base_events = (0.035 * torch.randn(6, 2, 4, generator=host_generator)).cuda()
    # Built-in backends register lazily. Resolve both callbacks outside the
    # fullgraph region so a cold optional backend does not import Python code
    # while Dynamo traces the public recurrent call.
    with environ.context(dt=0.27):
        neuron_spec, *_ = prepare_cell(model.neuron, base_x)
        synapse_spec, *_ = prepare_cell(model.synapse, base_events)
    resolve_cell_backend(backend, neuron_spec, base_x.device)
    resolve_cell_backend(backend, synapse_spec, base_x.device)
    trials = (
        (base_x, base_events),
        (base_x * 0.73 + 0.07, base_events * -0.61 + 0.015),
        (base_x.flip(0) * 1.08 - 0.03, base_events.roll(1, 0) * 0.82),
        (base_x.roll(2, 0) * 0.64 + 0.12, base_events.flip(0) * -0.77),
    )
    external = tuple(
        value.detach().clone().requires_grad_()
        for value in (
            model.neuron.v,
            model.neuron.g_k,
            model.neuron.refractory,
            model.synapse.psc,
        )
    )
    expected_external = tuple(
        value.detach().clone().requires_grad_()
        for value in (
            reference.neuron.v,
            reference.neuron.g_k,
            reference.neuron.refractory,
            reference.synapse.psc,
        )
    )
    carried = tuple(value.detach().clone() for value in external)
    expected_carried = tuple(value.detach().clone() for value in expected_external)
    replay_calls = []
    original_replay = torch.cuda.CUDAGraph.replay

    def counted_replay(graph):
        replay_calls.append(graph)
        return original_replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted_replay)

    cotangent_generator = torch.Generator().manual_seed(731)
    output_cotangent = torch.randn(6, 2, 4, generator=cotangent_generator).cuda()
    state_cotangents = tuple(
        torch.randn(value.shape, generator=cotangent_generator).cuda()
        for value in external
    )

    def step(value, events):
        output, records = model(value, events)
        assert records == {}
        states = (
            model.neuron.v,
            model.neuron.g_k,
            model.neuron.refractory,
            model.synapse.psc,
        )
        loss = (output * output_cotangent).sum()
        loss = loss + sum(
            (state * cotangent).sum()
            for state, cotangent in zip(states, state_cotangents, strict=True)
        )
        return loss, output

    compiled = torch.compile(step, mode="reduce-overhead", fullgraph=True)
    replay_graphs = []
    spike_counts = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(2891)
        for x_value, event_value in trials:
            with torch.no_grad():
                for target, source in zip(external, carried, strict=True):
                    target.copy_(source)
                for target, source in zip(
                    expected_external, expected_carried, strict=True
                ):
                    target.copy_(source)
            (
                model.neuron.v,
                model.neuron.g_k,
                model.neuron.refractory,
                model.synapse.psc,
            ) = external
            (
                reference.neuron.v,
                reference.neuron.g_k,
                reference.neuron.refractory,
                reference.synapse.psc,
            ) = expected_external
            x = x_value.detach().clone().requires_grad_()
            events = event_value.detach().clone().requires_grad_()
            expected_x = x_value.detach().clone().requires_grad_()
            expected_events = event_value.detach().clone().requires_grad_()
            roots = _model_roots(model, x, events)
            expected_roots = _model_roots(reference, expected_x, expected_events)

            initial_rng = torch.cuda.get_rng_state()
            before_replay = len(replay_calls)
            torch.compiler.cudagraph_mark_step_begin()
            with environ.context(dt=0.27):
                loss, output = compiled(x, events)
            replay_rng = torch.cuda.get_rng_state()

            torch.cuda.set_rng_state(initial_rng)
            with environ.context(dt=0.27):
                expected_output, expected_records = reference(
                    expected_x, expected_events
                )
            assert expected_records == {}
            expected_states = (
                reference.neuron.v,
                reference.neuron.g_k,
                reference.neuron.refractory,
                reference.synapse.psc,
            )
            expected_loss = (expected_output * output_cotangent).sum()
            expected_loss = expected_loss + sum(
                (state * cotangent).sum()
                for state, cotangent in zip(
                    expected_states, state_cotangents, strict=True
                )
            )
            assert torch.equal(torch.cuda.get_rng_state(), replay_rng)
            assert torch.equal(output, expected_output)
            spike_counts.append(int(output.sum()))
            for state, expected_state in zip(
                (
                    model.neuron.v,
                    model.neuron.g_k,
                    model.neuron.refractory,
                    model.synapse.psc,
                ),
                expected_states,
                strict=True,
            ):
                torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=2e-5)

            rng_before_backward = torch.cuda.get_rng_state()
            gradients = torch.autograd.grad(loss, roots, allow_unused=True)
            expected_gradients = torch.autograd.grad(
                expected_loss, expected_roots, allow_unused=True
            )
            replay_graphs.append(tuple(replay_calls[before_replay:]))
            assert torch.equal(torch.cuda.get_rng_state(), rng_before_backward)
            for index, (gradient, expected_gradient) in enumerate(
                zip(gradients, expected_gradients, strict=True)
            ):
                assert (gradient is None) == (expected_gradient is None), index
                if expected_gradient is not None:
                    torch.testing.assert_close(
                        gradient,
                        expected_gradient,
                        atol=8e-5,
                        rtol=8e-4,
                        msg=f"root {index}",
                    )
            carried = tuple(
                value.detach().clone()
                for value in (
                    model.neuron.v,
                    model.neuron.g_k,
                    model.neuron.refractory,
                    model.synapse.psc,
                )
            )
            expected_carried = tuple(
                value.detach().clone() for value in expected_states
            )
            # A completed iteration intentionally truncates BPTT. The cloned
            # continuation values above are ordinary allocations; release all
            # graph-owned outputs and adjoints before marking the next CUDA
            # Graph Tree step so its private pool has no live user references.
            del (
                events,
                expected_events,
                expected_gradients,
                expected_loss,
                expected_output,
                expected_roots,
                expected_states,
                expected_x,
                gradient,
                gradients,
                loss,
                output,
                roots,
                state,
                expected_state,
                x,
            )
    # Inductor may spend either of the first two calls warming the graph pool.
    # The final two changed-input steps must replay the same captured forward
    # and backward graphs, which proves actual reuse without fixing warmup
    # timing to one Torch release.
    assert len(replay_graphs[-2]) == len(replay_graphs[-1]) == 2
    assert len({id(graph) for graph in replay_graphs[-1]}) == 2
    assert all(
        previous is current
        for previous, current in zip(replay_graphs[-2], replay_graphs[-1], strict=True)
    )
    assert any(count > 0 for count in spike_counts)
    assert any(count < base_x.numel() for count in spike_counts)
