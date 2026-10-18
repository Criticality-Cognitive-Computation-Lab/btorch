"""Exercise scientific diagnostics using the existing neuron/network
equations."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

from btorch.models.connection import SparseConnection, Synapse
from btorch.models.functional import init_net_state
from btorch.models.kernels.response import (
    ResponsePlotOptions,
    plot_response,
    restore_model,
    simulate_response,
    snapshot_model,
    validate_response,
)
from btorch.models.neurons.glif import GLIF3
from btorch.models.neurons.izhikevich import Izhikevich
from btorch.models.rnn import RecurrentNN, make_rnn
from btorch.models.synapse import AlphaPSC, HeterSynapsePSC
from btorch.utils.file import save_fig


def _network(n, *, disconnected=False):
    # The identical network builder is used for single, pair and network runs.
    # Each presynaptic neuron drives its next neighbor through two receptors;
    # distinct delays exercise the actual btorch ring, including its cursor.
    pre = torch.arange(n).repeat_interleave(2)
    post = (pre + 1) % n
    connection = SparseConnection.from_edges(
        pre,
        post,
        n,
        n,
        Synapse(
            delay=torch.tensor([0, 2]).repeat(n),
            receptor=torch.tensor([0, 1]).repeat(n),
            n_delay=3,
            n_receptor=2,
        ),
        values=torch.tensor([50.0, -10.0]).repeat(n) * (0 if disconnected else 1),
    )
    neuron = Izhikevich(
        n, v_threshold=-40.0, v_peak=30.0, hard_reset=True, step_mode="s"
    )
    synapse = HeterSynapsePSC(
        n, 2, None, connection, base_psc=AlphaPSC, tau_syn=2.0, use_circular_buffer=True
    )
    model = RecurrentNN(neuron, synapse)
    model.float()
    init_net_state(model, batch_size=1, dtype=torch.float32)
    return model


@pytest.mark.parametrize("n,level", [(1, "single"), (2, "pair"), (5, "network")])
def test_response_network_continuation_and_activity(n, level):
    # Capture nonzero delayed traffic, then prove restoration reproduces both
    # spikes and every memory (not only voltage). This catches a missing cursor
    # or receptor state even if binary spikes happen to remain identical.
    model = _network(n)
    original_modes = {
        name: module.step_mode
        for name, module in model.named_modules()
        if hasattr(module, "step_mode")
    }
    original_records = model.update_state_names
    stimulus = torch.zeros(140, 1, n)
    stimulus[10:100, 0, 0] = 350.0
    first = simulate_response(model, stimulus[:73])
    assert first.spikes.sum() > 0
    assert model.update_state_names == original_records
    assert {
        name: module.step_mode
        for name, module in model.named_modules()
        if hasattr(module, "step_mode")
    } == original_modes
    if n > 1:
        # The unforced target changes voltage because the same real recurrent
        # connection/synapse code propagates upstream events.
        control = simulate_response(_network(n, disconnected=True), stimulus[:73])
        assert not torch.equal(
            first.states["neuron.v"][:, 0, 1], control.states["neuron.v"][:, 0, 1]
        )
    snapshot = snapshot_model(model)
    cursor_name = "synapse.delayed_spike._cursor"
    assert snapshot.buffers[cursor_name].item() == 73 % 3
    second = simulate_response(model, stimulus[73:])
    restore_model(model, snapshot)
    assert dict(model.named_buffers())[cursor_name].item() == 73 % 3
    repeated = simulate_response(model, stimulus[73:])
    assert validate_response(second) == []
    torch.testing.assert_close(second.spikes, repeated.spikes)
    for name in second.states:
        torch.testing.assert_close(second.states[name], repeated.states[name])
    # Also compare split execution against one uninterrupted call, so that
    # snapshot equality cannot hide a consistently incorrect chunk boundary.
    uninterrupted = _network(n)
    complete = simulate_response(uninterrupted, stimulus)
    torch.testing.assert_close(
        complete.spikes, torch.cat([first.spikes, second.spikes])
    )
    for name in complete.states:
        torch.testing.assert_close(
            complete.states[name], torch.cat([first.states[name], second.states[name]])
        )
    figure = plot_response(complete, title=f"Izhikevich heter/delay {level}")
    save_fig(figure, name=f"response_izhikevich_{level}", file=__file__, suffix="png")
    plt.close(figure)


def test_glif_all_adaptation_channels_and_empty_input():
    # A step followed by silence demonstrates adaptation across both channels.
    # This explicit reference-only adapter collects single-step histories. It
    # intentionally does not validate the primitive continuous sequence kernel.
    # No replacement GLIF equation is used; the original Module is the model.
    neuron = GLIF3(
        1,
        v_threshold=-45.0,
        v_reset=-65.0,
        c_m=200.0,
        tau=20.0,
        k=[0.05, 0.01],
        asc_amps=[-50.0, -15.0],
        tau_ref=2.0,
    )
    neuron.float()
    neuron.init_state(batch_size=1, dtype=torch.float32)
    stimulus = torch.zeros(1000, 1, 1)
    stimulus[:500] = 250.0
    trace = simulate_response(
        neuron, stimulus, sequence=make_rnn(neuron).multi_step_forward
    )
    assert (
        validate_response(
            trace,
            expectations={
                "driven spikes": lambda response: bool(response.spikes[:500].sum() > 0),
                "silent after stimulus": lambda response: bool(
                    response.spikes[500:].sum() == 0
                ),
            },
        )
        == []
    )
    assert trace.states["Iasc"].shape[-1] == 2
    assert trace.states["Iasc"][-1].abs().max() < trace.states["Iasc"][499].abs().max()
    figure = plot_response(trace, title="GLIF3 two adaptation channels")
    # The adaptation panel must contain both channels, unlike first-channel
    # slicing that can conceal a receptor/adaptation implementation error.
    axes = [axis for axis in figure.axes if axis.get_ylabel().startswith("Iasc [")]
    assert len(axes) == 2
    for component, axis in enumerate(axes):
        # Every adaptation component is displayed in its own panel with the
        # original numerical trajectory, not summed or normalized for appearance.
        torch.testing.assert_close(
            torch.as_tensor(axis.lines[0].get_ydata()),
            trace.states["Iasc"][:, 0, 0, component],
        )
    voltage_axis = next(axis for axis in figure.axes if axis.get_ylabel() == "v")
    assert "v_threshold" in [line.get_label() for line in voltage_axis.lines]
    save_fig(figure, name="response_glif3", file=__file__, suffix="png")
    plt.close(figure)
    before = snapshot_model(neuron)
    empty = simulate_response(
        make_rnn(neuron), stimulus[:0], state_names=("rnn_cell.v",)
    )
    assert empty.spikes.shape[0] == 0
    torch.testing.assert_close(
        empty.final_state.hidden_states["rnn_cell.v"], before.hidden_states["v"]
    )
    # Deliberately poisoned scientific data produces a useful state/time index.
    trace.states["v"][12, 0, 0] = float("nan")
    assert "v: nonfinite at index=(12, 0, 0), time=12" in validate_response(trace)


def test_harness_model_preserves_adaptation_across_validation_scales():
    factory = (
        Path(__file__).resolve().parents[3]
        / "skills/btorch-kernel-gen/assets/templates/glif3_rsnn.py"
    )
    build_model = runpy.run_path(str(factory))["build_model"]
    responses = []
    for n in (1, 2, 8):
        model = build_model(n)
        stimulus = torch.zeros(140, 1, n)
        stimulus[20:100, 0, 0] = 300.0
        init_net_state(model, batch_size=1, dtype=torch.float32)
        responses.append(simulate_response(model, stimulus))
    # The chain has no feedback to the driven neuron. Its complete physical
    # response must be identical at every scale. This catches ambiguous length-N
    # parameter lists that silently change adaptation when N equals channel count.
    for response in responses[1:]:
        torch.testing.assert_close(
            response.spikes[:, 0, 0], responses[0].spikes[:, 0, 0]
        )
        for name in ("neuron.v", "neuron.Iasc"):
            torch.testing.assert_close(
                response.states[name][:, 0, 0], responses[0].states[name][:, 0, 0]
            )


def test_response_plot_batch_reference_and_delay_indices():
    # Distinct trials make wrong batch selection visible in the actual model
    # trajectories. Comparison overlays must preserve the same raw values.
    model = _network(2)
    init_net_state(model, batch_size=2, dtype=torch.float32)
    stimulus = torch.zeros(70, 2, 2)
    stimulus[10:50, 0, 0] = 350.0
    stimulus[20:60, 1, 0] = 500.0
    trace = simulate_response(model, stimulus)
    figure = plot_response(
        trace,
        reference=trace,
        options=ResponsePlotOptions(
            batch_index=1, time_unit="ms", units={"neuron.v": "mV"}
        ),
    )
    voltage = next(
        axis for axis in figure.axes if axis.get_ylabel() == "neuron.v\n(mV)"
    )
    for neuron in range(2):
        expected = trace.states["neuron.v"][:, 1, neuron]
        for line in (voltage.lines[neuron], voltage.lines[neuron + 2]):
            torch.testing.assert_close(torch.as_tensor(line.get_ydata()), expected)
    # Delay-ring [time,batch,delay,neuron] axes don't match the conventional
    # neuron-state prefix. Their full indexed values must survive; don't guess
    # a neuron/batch layout and discard part of the continuation history.
    name = "synapse.delayed_spike.history"
    delay_axis = next(
        axis for axis in figure.axes if axis.get_ylabel().startswith(name)
    )
    history = trace.states[name].reshape(70, -1)
    for index in range(history.shape[1]):
        torch.testing.assert_close(
            torch.as_tensor(delay_axis.lines[index].get_ydata()), history[:, index]
        )
    save_fig(figure, name="response_batch_reference", file=__file__, suffix="png")
    plt.close(figure)


def test_snapshot_restores_rng_parameters_and_extra_state():
    # Online weights and unregistered event metadata need continuation too.
    model = torch.nn.Linear(2, 2)
    model.cursor = 3
    snapshot = snapshot_model(model, extra_state_names=("cursor",))
    expected_random = torch.rand(3)
    with torch.no_grad():
        model.weight.add_(10)
    model.cursor = 10
    restore_model(model, snapshot)
    torch.testing.assert_close(model.weight, snapshot.state_dict["weight"])
    torch.testing.assert_close(torch.rand(3), expected_random)
    assert model.cursor == 3


def test_snapshot_preserves_memory_views_and_nonpersistent_buffers():
    # Real event metadata often has external aliases, and receptors can expose
    # views of a shared state. Restoring must keep these relationships as well
    # as numeric values, including buffers deliberately absent from state_dict.
    model = GLIF3(2)
    model.float()
    model.init_state(batch_size=1, dtype=torch.float32)
    model.register_memory("v_alias", model.v, model.v.shape)
    model.register_buffer("v_alias", model.v, persistent=False)
    model.v_slice = model.v[:, :1]
    model.register_buffer("cursor", torch.tensor(1), persistent=False)
    external_cursor = model.cursor
    snapshot = snapshot_model(model, extra_state_names=("v_slice",))
    with torch.no_grad():
        model.cursor.fill_(2)
        model.v.add_(7)
    restore_model(model, snapshot)
    assert model.cursor is external_cursor
    assert model.cursor.item() == 1
    assert model.v is model.v_alias
    # Updating through the restored slice must update the corresponding memory.
    with torch.no_grad():
        model.v_slice.add_(3)
    torch.testing.assert_close(model.v[:, :1], model.v_slice)
    torch.testing.assert_close(model.v[:, :1], snapshot.hidden_states["v"][:, :1] + 3)


def test_snapshot_keeps_parameter_alias_in_explicit_state():
    # An online mechanism can expose its learned weight as event metadata.
    # Restoring must not silently turn that tied Parameter into a plain Tensor.
    model = torch.nn.Linear(2, 2)
    model.event_metadata = {"weight": model.weight}
    snapshot = snapshot_model(model, extra_state_names=("event_metadata",))
    original_parameter = model.weight
    with torch.no_grad():
        model.weight.add_(5)
    model.event_metadata = {"weight": model.weight.detach().clone()}
    restore_model(model, snapshot)
    assert model.event_metadata["weight"] is model.weight is original_parameter
    torch.testing.assert_close(model.weight, snapshot.parameters["weight"])


def test_description_exposes_channels_without_mutating_response():
    import json

    from btorch.models.kernels.response import describe_response

    neuron = GLIF3(1, k=[0.05, 0.01], asc_amps=[-1.0, -2.0])
    neuron.float()
    neuron.init_state(batch_size=1, dtype=torch.float32)
    trace = simulate_response(
        neuron, torch.zeros(3, 1, 1), sequence=make_rnn(neuron).multi_step_forward
    )
    before = trace.states["Iasc"].clone()
    rng_before = torch.get_rng_state().clone()
    description = describe_response(trace, units={"dt": "ms", "v": "mV"})
    assert description["tensors"]["Iasc"]["shape"][-1] == 2
    assert description["tensors"]["Iasc"]["values"] == before.tolist()
    assert description["tensors"]["v"]["units"] == "mV"
    torch.testing.assert_close(trace.states["Iasc"], before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    trace.states["v"][1, 0, 0] = float("nan")
    bounded = describe_response(trace, max_values=1)
    assert "values" not in bounded["tensors"]["v"]
    assert bounded["tensors"]["v"]["first_nonfinite"] == [1, 0, 0]
    # Strict JSON serialization is useful for agent feedback; NaN cannot leak
    # through either complete small values or finite-only large summaries.
    json.dumps(describe_response(trace), allow_nan=False)


def test_primitive_sequence_operator_preserves_initial_state_and_continuation():
    from btorch.models.base import MemoryModule

    class SequenceAccumulator(MemoryModule):
        # A primitive sequence op has a distinct entry: falling back to single
        # steps would lose its state/output contract and must fail this test.
        def __init__(self):
            super().__init__()
            self.register_memory("q", 3.0, 1)
            self.init_state(batch_size=1, dtype=torch.float32)

        def single_step_forward(self, x):
            raise AssertionError("Diagnostic helper replaced the sequence operator")

        def multi_step_forward(self, x_seq):
            outputs = x_seq.cumsum(0) + self.q
            if x_seq.shape[0]:
                self.q = outputs[-1]
            return outputs

    model = SequenceAccumulator()
    stimulus = torch.tensor([1.0, 2.0, 4.0]).reshape(3, 1, 1)
    initial = snapshot_model(model)
    with pytest.raises(ValueError, match="explicit sequence adapter"):
        simulate_response(model, stimulus, state_names=("q",))
    torch.testing.assert_close(model.q, initial.hidden_states["q"])
    empty = simulate_response(model, stimulus[:0])
    assert empty.spikes.shape == (0, 1, 1)
    torch.testing.assert_close(model.q, initial.hidden_states["q"])
    first = simulate_response(model, stimulus[:2])
    assert first.states == {}
    torch.testing.assert_close(first.spikes.flatten(), torch.tensor([4.0, 6.0]))
    second = simulate_response(model, stimulus[2:])
    torch.testing.assert_close(second.spikes.flatten(), torch.tensor([10.0]))
    restore_model(model, initial)

    def recorded_sequence(x_seq):
        # The adapter requests history from the very same continuous operation,
        # rather than replaying a different step algorithm for diagnostic data.
        outputs = model.multi_step_forward(x_seq)
        return outputs, {"q": outputs}

    complete = simulate_response(model, stimulus, sequence=recorded_sequence)
    torch.testing.assert_close(
        complete.spikes, torch.cat((first.spikes, second.spikes))
    )
    torch.testing.assert_close(complete.states["q"], complete.spikes)
    torch.testing.assert_close(complete.final_state.hidden_states["q"], model.q)
