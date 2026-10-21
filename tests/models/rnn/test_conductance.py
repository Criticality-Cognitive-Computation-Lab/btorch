"""Validate the public Torch recurrent conductance schedule with actual
Modules."""

import copy

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from btorch.models import environ
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.functional import init_net_state
from btorch.models.kernels import KernelOptions
from btorch.models.neurons.lif import IF, LIF
from btorch.models.rnn import ConductanceRecurrentNN, RecurrentNN
from btorch.models.synapse import ConductancePSC, ExponentialPSC


DT = 0.4


def _dense_network(*, checkpoint: bool) -> ConductanceRecurrentNN:
    neurons = 3
    neuron = LIF(
        neurons,
        v_threshold=0.7,
        v_reset=-0.25,
        c_m=1.3,
        tau=4.2,
        hard_reset=True,
        pre_spike_v=True,
    )
    recurrent = torch.nn.Linear(neurons, neurons, bias=True)
    neuron_input = torch.nn.Linear(neurons, neurons, bias=False)
    synapse_input = torch.nn.Linear(neurons, neurons, bias=False)
    with torch.no_grad():
        recurrent.weight.copy_(
            torch.tensor(
                [[0.16, -0.04, 0.03], [0.07, 0.11, -0.02], [-0.05, 0.06, 0.13]]
            )
        )
        recurrent.bias.copy_(torch.tensor([0.01, -0.02, 0.015]))
        neuron_input.weight.copy_(torch.diag(torch.tensor([1.1, 0.9, 1.05])))
        synapse_input.weight.copy_(torch.diag(torch.tensor([0.8, 1.2, 0.7])))
    psc = ConductancePSC(
        neurons,
        tau_syn=torch.tensor(3.4),
        linear=recurrent,
        reversal_potential=torch.tensor([1.2, -0.9, 0.6]),
    )
    network = ConductanceRecurrentNN(
        neuron,
        psc,
        neuron_inp_module=neuron_input,
        syn_inp_module=synapse_input,
        update_state_names=("neuron.v", "neuron.v_pre_spike", "synapse.psc"),
        unroll=2,
        chunk_size=4 if checkpoint else None,
        grad_checkpoint=checkpoint,
        backend="torch",
    )
    init_net_state(network, batch_size=2, dtype=torch.float32)
    return network


def _bind_trainable_state(network: ConductanceRecurrentNN):
    initial_v = torch.tensor(
        [[-0.18, 0.08, -0.11], [0.04, -0.16, 0.12]], requires_grad=True
    )
    initial_g = torch.tensor(
        [[0.07, -0.03, 0.05], [-0.02, 0.06, 0.04]], requires_grad=True
    )
    network.neuron.v = initial_v
    network.synapse.psc = initial_g
    network.synapse.tau_syn.requires_grad_()
    network.synapse.E.requires_grad_()
    return initial_v, initial_g


def _module_oracle(network, x, x_syn):
    """Run the same public neuron and synapse Modules with explicit
    scheduling."""
    spikes = []
    records = {"neuron.v": [], "neuron.v_pre_spike": [], "synapse.psc": []}
    for drive, extra_event in zip(x, x_syn, strict=True):
        drive = network.neuron_inp_module(drive)
        extra_event = network.syn_inp_module(extra_event)
        voltage = network.neuron.v
        current = network.synapse.current_charge(voltage)
        spike = network.neuron(drive + current)
        network.synapse(spike + extra_event, voltage)
        spikes.append(spike)
        records["neuron.v"].append(network.neuron.v)
        records["neuron.v_pre_spike"].append(network.neuron.v_pre_spike)
        records["synapse.psc"].append(network.synapse.psc)
    return torch.stack(spikes), {
        name: torch.stack(values) for name, values in records.items()
    }


def _roots(network, x, x_syn, initial_v, initial_g):
    return (
        x,
        x_syn,
        initial_v,
        initial_g,
        network.synapse.linear.weight,
        network.synapse.linear.bias,
        network.synapse.tau_syn,
        network.synapse.E,
        network.neuron_inp_module.weight,
        network.syn_inp_module.weight,
    )


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("split_at", [None, 6], ids=("whole", "two-calls"))
def test_dense_conductance_schedule_matches_actual_modules_and_full_bptt(
    checkpoint, split_at
):
    """Whole and continued calls preserve every input, state, and parameter
    VJP."""
    torch.manual_seed(31)
    candidate = _dense_network(checkpoint=checkpoint)
    reference = copy.deepcopy(candidate)
    candidate_initial = _bind_trainable_state(candidate)
    reference_initial = _bind_trainable_state(reference)
    x = (0.34 * torch.randn(11, 2, 3)).requires_grad_()
    x_syn = (0.09 * torch.randn(11, 2, 3)).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    x_syn_ref = x_syn.detach().clone().requires_grad_()

    with environ.context(dt=DT):
        if split_at is None:
            actual, actual_records = candidate(x, x_syn)
        else:
            # The second call continues live state and full BPTT from the first.
            first, first_records = candidate(x[:split_at], x_syn[:split_at])
            second, second_records = candidate(x[split_at:], x_syn[split_at:])
            actual = torch.cat((first, second))
            actual_records = {
                name: torch.cat((first_records[name], second_records[name]))
                for name in first_records
            }
        expected, expected_records = _module_oracle(reference, x_ref, x_syn_ref)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    for name in actual_records:
        torch.testing.assert_close(
            actual_records[name], expected_records[name], atol=2e-6, rtol=2e-6
        )

    generator = torch.Generator().manual_seed(92)
    actual_values = (
        actual,
        actual_records["neuron.v"],
        actual_records["neuron.v_pre_spike"],
        actual_records["synapse.psc"],
        candidate.neuron.v,
        candidate.synapse.psc,
    )
    expected_values = (
        expected,
        expected_records["neuron.v"],
        expected_records["neuron.v_pre_spike"],
        expected_records["synapse.psc"],
        reference.neuron.v,
        reference.synapse.psc,
    )
    cotangents = tuple(
        torch.randn(value.shape, generator=generator) for value in actual_values
    )
    actual_gradients = torch.autograd.grad(
        actual_values,
        _roots(candidate, x, x_syn, *candidate_initial),
        cotangents,
    )
    expected_gradients = torch.autograd.grad(
        expected_values,
        _roots(reference, x_ref, x_syn_ref, *reference_initial),
        cotangents,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=2e-5, rtol=2e-5
        )


@pytest.mark.parametrize("conductance", [False, True], ids=("base-rnn", "conductance"))
def test_default_surrogate_fullgraph_limitation_is_inherited(conductance):
    """Default btorch surrogates hit the same upstream custom-JVP
    limitation."""
    torch._dynamo.reset()
    if conductance:
        network = _dense_network(checkpoint=False)
    else:
        network = RecurrentNN(
            LIF(3),
            ExponentialPSC(3, 3.4, torch.nn.Linear(3, 3)),
            backend="torch",
        )
        init_net_state(network, batch_size=2, dtype=torch.float32)

    def scheduler(drive, extra_event):
        spikes, _ = network._run_unroll_block(drive, extra_event, loop_args=(0, 1))
        return torch.stack(spikes)

    compiled = torch.compile(scheduler, backend="aot_eager", fullgraph=True)
    with (
        environ.context(dt=DT),
        pytest.raises(torch._dynamo.exc.Unsupported, match="custom jvp"),
    ):
        compiled(torch.randn(3, 2, 3), torch.zeros(3, 2, 3))


def test_public_compile_two_calls_match_actual_modules_and_vjps():
    """Public compilation preserves default-spike state coupling and
    gradients."""
    torch._dynamo.reset()
    torch.manual_seed(47)
    candidate = _dense_network(checkpoint=False)
    reference = copy.deepcopy(candidate)
    candidate_initial = _bind_trainable_state(candidate)
    reference_initial = _bind_trainable_state(reference)
    x = (0.3 * torch.randn(9, 2, 3)).requires_grad_()
    x_syn = (0.08 * torch.randn(9, 2, 3)).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    x_syn_ref = x_syn.detach().clone().requires_grad_()

    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(candidate, backend=counter)
    with environ.context(dt=DT):
        first, first_records = compiled(x[:4], x_syn[:4])
        second, second_records = compiled(x[4:], x_syn[4:])
        actual = torch.cat((first, second))
        actual_records = {
            name: torch.cat((first_records[name], second_records[name]))
            for name in first_records
        }
        expected_first, expected_first_records = _module_oracle(
            reference, x_ref[:4], x_syn_ref[:4]
        )
        expected_second, expected_second_records = _module_oracle(
            reference, x_ref[4:], x_syn_ref[4:]
        )
        expected = torch.cat((expected_first, expected_second))
        expected_records = {
            name: torch.cat(
                (expected_first_records[name], expected_second_records[name])
            )
            for name in expected_first_records
        }

    assert counter.frame_count > 0

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    for name in actual_records:
        torch.testing.assert_close(
            actual_records[name], expected_records[name], atol=2e-6, rtol=2e-6
        )
    torch.testing.assert_close(candidate.neuron.v, reference.neuron.v)
    torch.testing.assert_close(candidate.synapse.psc, reference.synapse.psc)

    actual_values = (
        actual,
        *actual_records.values(),
        candidate.neuron.v,
        candidate.synapse.psc,
    )
    expected_values = (
        expected,
        *expected_records.values(),
        reference.neuron.v,
        reference.synapse.psc,
    )
    generator = torch.Generator().manual_seed(79)
    cotangents = tuple(
        torch.randn(value.shape, generator=generator) for value in actual_values
    )
    actual_gradients = torch.autograd.grad(
        actual_values,
        _roots(candidate, x, x_syn, *candidate_initial),
        cotangents,
    )
    expected_gradients = torch.autograd.grad(
        expected_values,
        _roots(reference, x_ref, x_syn_ref, *reference_initial),
        cotangents,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=2e-5, rtol=2e-5
        )


def _delayed_sparse_network() -> ConductanceRecurrentNN:
    connection = SparseConnection.from_edges(
        torch.tensor([0, 1, 2, 0]),
        torch.tensor([1, 2, 0, 2]),
        3,
        3,
        Synapse(delay=torch.tensor([2, 0, 1, 2]), n_delay=3),
        values=torch.tensor([0.7, -0.4, 0.3, 0.2]),
    )
    network = ConductanceRecurrentNN(
        IF(3, v_threshold=0.8, v_reset=0.0, c_m=1.0, pre_spike_v=True),
        ConductancePSC(3, 2.8, connection, reversal_potential=1.1),
        update_state_names=("neuron.v", "synapse.psc"),
        backend="torch",
        unroll=2,
    )
    init_net_state(network, batch_size=1, dtype=torch.float32)
    return network


def test_sparse_delay_state_continues_across_calls():
    """Edge delays and their history remain part of recurrent continuation."""
    whole = _delayed_sparse_network()
    chunked = copy.deepcopy(whole)
    x = torch.zeros(9, 1, 3)
    x[0, 0, 0] = 1.2
    x[4, 0, 1] = 0.95

    with torch.no_grad(), environ.context(dt=DT):
        expected, expected_records = whole(x)
        first, first_records = chunked(x[:3])
        second, second_records = chunked(x[3:])

    torch.testing.assert_close(torch.cat((first, second)), expected)
    for name in expected_records:
        torch.testing.assert_close(
            torch.cat((first_records[name], second_records[name])),
            expected_records[name],
        )
    torch.testing.assert_close(chunked.neuron.v, whole.neuron.v)
    torch.testing.assert_close(chunked.synapse.psc, whole.synapse.psc)
    torch.testing.assert_close(
        chunked.synapse.delayed_spike.history,
        whole.synapse.delayed_spike.history,
    )


class _CountingInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


class _DerivedConductance(ConductancePSC):
    pass


class _NoVoltageNeuron(torch.nn.Module):
    backend = "torch"

    def forward(self, value):
        return value


@pytest.mark.parametrize(
    "invalid",
    (
        "wrong_synapse",
        "subclass",
        "missing_voltage",
        "network_backend",
        "neuron_backend",
        "synapse_backend",
        "descendant_backend",
    ),
)
@pytest.mark.parametrize("route", ("single", "multi", "chunk"))
def test_invalid_conductance_schedule_rejects_before_callbacks_or_state(invalid, route):
    """Unsupported scientific or execution choices cannot partially advance."""
    neuron = (
        _NoVoltageNeuron()
        if invalid == "missing_voltage"
        else LIF(2, v_threshold=1.0, v_reset=0.0)
    )
    linear = torch.nn.Linear(2, 2, bias=False)
    if invalid == "wrong_synapse":
        psc = ExponentialPSC(2, 3.0, linear)
    elif invalid == "subclass":
        psc = _DerivedConductance(2, 3.0, linear, 0.0)
    else:
        psc = ConductancePSC(2, 3.0, linear, 0.0)
    callback = _CountingInput()
    if invalid == "descendant_backend":
        callback.backend = KernelOptions()
    network = ConductanceRecurrentNN(
        neuron,
        psc,
        neuron_inp_module=callback,
        backend=KernelOptions() if invalid == "network_backend" else "torch",
    )
    if invalid == "neuron_backend":
        neuron.backend = KernelOptions()
    if invalid == "synapse_backend":
        psc.backend = KernelOptions()
    init_net_state(network, batch_size=1, dtype=torch.float32)
    voltage = neuron.v.clone() if hasattr(neuron, "v") else None
    state = psc.psc.clone()
    x = torch.zeros(3, 1, 2)

    error = (
        TypeError
        if invalid in ("wrong_synapse", "subclass", "missing_voltage")
        else NotImplementedError
    )
    with pytest.raises(error):
        if route == "single":
            network.single_step_forward(x[0])
        elif route == "chunk":
            network._run_chunk_steps(x, loop_args=(0,), unroll_size=2)
        else:
            network(x)

    assert callback.calls == 0
    if voltage is not None:
        torch.testing.assert_close(neuron.v, voltage, rtol=0.0, atol=0.0)
    torch.testing.assert_close(psc.psc, state, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cudagraph capture requires CUDA"
)
@pytest.mark.parametrize(
    "trainable_tensor",
    ("synapse.E", "synapse.tau_syn", "neuron.v", "synapse.psc"),
)
def test_inference_cudagraph_rejects_trainable_buffers_before_state_advances(
    trainable_tensor,
):
    """Inference capture rejects every public conductance gradient root.

    Conductance parameters and initial states are buffers rather than ordinary
    ``Parameter`` objects. Freezing all parameters isolates that distinction:
    the selected buffer alone would make a grad-enabled replay unsafe.
    """
    network = _dense_network(checkpoint=False).to("cuda")
    network.cudagraph = True
    for parameter in network.parameters():
        parameter.requires_grad_(False)

    owner = network
    *parents, name = trainable_tensor.split(".")
    for parent in parents:
        owner = getattr(owner, parent)
    getattr(owner, name).requires_grad_()

    voltage = network.neuron.v.detach().clone()
    conductance = network.synapse.psc.detach().clone()
    drive = torch.randn(4, 2, 3, device="cuda")
    extra_event = torch.randn_like(drive)
    with environ.context(dt=DT), pytest.raises(RuntimeError, match="inference-only"):
        network(drive, extra_event)

    assert network._cudagraph_runner.n_captures == 0
    torch.testing.assert_close(network.neuron.v, voltage, rtol=0.0, atol=0.0)
    torch.testing.assert_close(network.synapse.psc, conductance, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cudagraph capture requires CUDA"
)
def test_inference_cudagraph_replays_changed_inputs_and_carries_state(monkeypatch):
    """Captured inference replays fresh inputs and live conductance state.

    The three calls intentionally continue one trajectory. Only membrane
    voltage is requested as a public record; conductance still advances
    and is checked as final module state without asking the loop to
    retain its full history.
    """
    torch.manual_seed(113)
    eager = _dense_network(checkpoint=False).to("cuda")
    graphed = copy.deepcopy(eager)
    graphed.cudagraph = True
    graphed.update_state_names = ("neuron.v",)
    eager.update_state_names = ("neuron.v",)
    initial_v = torch.tensor([[-0.17, 0.06, -0.09], [0.03, -0.14, 0.1]], device="cuda")
    initial_g = torch.tensor([[0.05, -0.02, 0.04], [-0.01, 0.07, 0.03]], device="cuda")
    eager.neuron.v = initial_v.clone()
    eager.synapse.psc = initial_g.clone()
    graphed.neuron.v = initial_v.clone()
    graphed.synapse.psc = initial_g.clone()
    drives = [
        0.25 * torch.randn(7, 2, 3, device="cuda") + offset
        for offset in (0.0, 0.08, -0.05)
    ]
    extra_events = [
        0.06 * torch.randn(7, 2, 3, device="cuda") + offset
        for offset in (0.0, -0.02, 0.03)
    ]

    replay_calls = []
    original_replay = torch.cuda.CUDAGraph.replay

    def counted_replay(graph):
        replay_calls.append(graph)
        return original_replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted_replay)
    saved = []
    replay_deltas = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    def compare_continued_step(drive, extra_event, dt):
        with (
            torch.inference_mode(),
            torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
            environ.context(dt=dt),
        ):
            expected, expected_records = eager(drive, extra_event)
            before_replay = len(replay_calls)
            actual, actual_records = graphed(drive, extra_event)
            replay_delta = len(replay_calls) - before_replay
            assert actual_records.keys() == {"neuron.v"}
            assert actual.grad_fn is None
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(
                actual_records["neuron.v"],
                expected_records["neuron.v"],
                atol=2e-6,
                rtol=2e-6,
            )
            torch.testing.assert_close(
                graphed.synapse.psc, eager.synapse.psc, atol=2e-6, rtol=2e-6
            )
        return replay_delta

    for drive, extra_event in zip(drives, extra_events, strict=True):
        replay_deltas.append(compare_continued_step(drive, extra_event, DT))

    assert graphed._cudagraph_runner.n_captures == 1

    def replace_buffer(path, update):
        eager_owner, graphed_owner = path(eager), path(graphed)
        eager_owner[0].__setattr__(
            eager_owner[1], update(getattr(eager_owner[0], eager_owner[1]))
        )
        graphed_owner[0].__setattr__(
            graphed_owner[1], update(getattr(graphed_owner[0], graphed_owner[1]))
        )

    replacement_cases = (
        (
            lambda model: (model.synapse, "E"),
            lambda value: value.detach().clone() + 0.07,
        ),
        (
            lambda model: (model.synapse, "tau_syn"),
            lambda value: value.detach().clone() * 1.08,
        ),
        (
            lambda model: (model.neuron, "tau"),
            lambda value: value.detach().clone() * 0.94,
        ),
    )
    for index, (path, update) in enumerate(replacement_cases, start=1):
        replace_buffer(path, update)
        drive = torch.randn_like(drives[0]) * 0.2 + 0.025 * index
        extra_event = torch.randn_like(extra_events[0]) * 0.04
        replay_deltas.append(compare_continued_step(drive, extra_event, DT))
        assert graphed._cudagraph_runner.n_captures == 1 + index

    # Parameter replacement changes an address wired into the graph and must
    # recapture, while an in-place value edit remains visible to plain replay.
    for model in (eager, graphed):
        weight = model.synapse.linear.weight.detach().clone() * 0.91
        model.synapse.linear.weight = torch.nn.Parameter(weight)
    replay_deltas.append(
        compare_continued_step(torch.randn_like(drives[0]), extra_events[0], DT)
    )
    assert graphed._cudagraph_runner.n_captures == 5

    with torch.no_grad():
        eager.synapse.E.add_(0.03)
        graphed.synapse.E.add_(0.03)
    replay_deltas.append(compare_continued_step(drives[0] - 0.04, extra_events[0], DT))
    assert graphed._cudagraph_runner.n_captures == 5

    # Python scalar dt is captured as a constant, but a tensor dt is read from
    # its stable address so an in-place update remains a replay.
    replay_deltas.append(compare_continued_step(drives[1], extra_events[1], DT * 0.75))
    assert graphed._cudagraph_runner.n_captures == 6
    dt_tensor = torch.tensor(DT, device="cuda")
    replay_deltas.append(compare_continued_step(drives[2], extra_events[2], dt_tensor))
    assert graphed._cudagraph_runner.n_captures == 7
    dt_tensor.mul_(0.8)
    replay_deltas.append(compare_continued_step(drives[0], extra_events[0], dt_tensor))
    assert graphed._cudagraph_runner.n_captures == 7

    # A CPU scalar is copied into captured kernel arguments. Keeping its address
    # stable is insufficient: changing its value must invalidate the graph.
    host_dt = torch.tensor(DT)
    replay_deltas.append(compare_continued_step(drives[1], extra_events[1], host_dt))
    assert graphed._cudagraph_runner.n_captures == 8
    host_dt.mul_(0.8)
    replay_deltas.append(compare_continued_step(drives[2], extra_events[2], host_dt))
    assert graphed._cudagraph_runner.n_captures == 9

    # CPU inference tensors do not expose a version counter. Their scalar value
    # is therefore the capture token, and changing it must recapture while the
    # continued output and hidden conductance state still match eager execution.
    with torch.inference_mode():
        inference_host_dt = torch.tensor(DT)
        replay_deltas.append(
            compare_continued_step(drives[0], extra_events[0], inference_host_dt)
        )
        assert graphed._cudagraph_runner.n_captures == 10
        inference_host_dt.mul_(0.8)
        replay_deltas.append(
            compare_continued_step(drives[1], extra_events[1], inference_host_dt)
        )
        assert graphed._cudagraph_runner.n_captures == 11

    assert not saved
    assert all(delta > 0 for delta in replay_deltas)
