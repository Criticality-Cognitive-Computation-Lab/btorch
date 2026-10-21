"""Validate native ConductancePSC composition against the actual Module."""

from __future__ import annotations

import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.kernels import KernelOptions
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ConductancePSC
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


DT = 0.35
T, B, N_SOURCE, N_POST, N_DELAY = 7, 2, 3, 3, 3
FORWARD_KW = {"atol": 3e-4, "rtol": 5e-4}
GRADIENT_KW = {"atol": 2e-3, "rtol": 2e-3}


def _values(device: str = "cuda") -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1601)
    values = {
        "dense_weight": torch.randn(
            N_POST, N_SOURCE, generator=generator, dtype=torch.float32
        ),
        "delay_weight": torch.randn(
            N_POST, N_SOURCE * N_DELAY, generator=generator, dtype=torch.float32
        ),
        "sparse_weight": torch.tensor([0.4, -0.2, 0.7, 0.3, -0.5]),
        "psc": torch.randn(B, N_POST, generator=generator) * 0.3,
        "history": torch.randn(B, N_DELAY, N_SOURCE, generator=generator) * 0.2,
        "z": torch.randn(T, B, N_SOURCE, generator=generator) * 0.4,
        # The missing batch lane is intentional: the native composition must
        # insert it between time and neuron dimensions before applying COBA.
        "v": -55.0 + torch.randn(T, N_POST, generator=generator),
    }
    return {name: value.to(device) for name, value in values.items()}


def _linear(kind: str, values: dict[str, torch.Tensor]):
    device = values["z"].device
    if kind == "sparse":
        from btorch.models.connection import SparseConnection

        connection = SparseConnection.from_edges(
            torch.tensor([0, 1, 2, 0, 2], device=device),
            torch.tensor([0, 0, 1, 2, 2], device=device),
            N_SOURCE,
            N_POST,
            values=values["sparse_weight"],
        )
        return connection.to(device=device, dtype=torch.float32)

    delayed = kind == "dense_delay"
    n_input = N_SOURCE * N_DELAY if delayed else N_SOURCE
    linear = torch.nn.Linear(
        n_input, N_POST, bias=False, device=device, dtype=torch.float32
    )
    with torch.no_grad():
        linear.weight.copy_(
            values["delay_weight"] if delayed else values["dense_weight"]
        )
    if delayed:
        linear.n_delay = N_DELAY
        linear.n_pre = N_SOURCE
    return linear


def _build(
    backend: str | KernelOptions,
    values: dict[str, torch.Tensor],
    *,
    kind: str = "dense_delay",
    differentiable: bool = False,
) -> tuple[ConductancePSC, tuple[torch.Tensor, ...]]:
    device = values["z"].device
    linear = _linear(kind, values)
    module = ConductancePSC(
        N_POST,
        tau_syn=torch.tensor(4.7, device=device),
        linear=linear,
        reversal_potential=torch.tensor([0.0, -72.0, -10.0], device=device),
        step_mode="m",
        backend=backend,
    )
    init_net_state(module, batch_size=B, dtype=torch.float32, device=device)
    psc = values["psc"].clone().requires_grad_(differentiable)
    module.psc = psc
    roots: list[torch.Tensor] = [psc]
    if module.delayed_spike is not None:
        history = values["history"].clone().requires_grad_(differentiable)
        module.delayed_spike.history = history
        roots.append(history)
    module.tau_syn.requires_grad_(differentiable)
    module.E.requires_grad_(differentiable)
    roots.extend((*linear.parameters(), module.tau_syn, module.E))
    return module, tuple(roots)


def _step_sequence(
    module: ConductancePSC,
    z_seq: torch.Tensor,
    v_seq: torch.Tensor | None,
) -> torch.Tensor:
    """Use repeated public single steps as the only scientific oracle."""
    return torch.stack(
        [
            module.single_step_forward(z_seq[t], None if v_seq is None else v_seq[t])
            for t in range(len(z_seq))
        ]
    )


def _assert_state(actual: ConductancePSC, expected: ConductancePSC) -> None:
    torch.testing.assert_close(actual.psc, expected.psc, **FORWARD_KW)
    if actual.delayed_spike is None:
        assert expected.delayed_spike is None
    else:
        torch.testing.assert_close(
            actual.delayed_spike.history,
            expected.delayed_spike.history,
            **FORWARD_KW,
        )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("with_voltage", [False, True])
def test_native_single_whole_and_chunked_sequences_match_module(backend, with_voltage):
    """CUBA and COBA share exact state, delay, and event-order semantics."""
    require_cuda_backend(backend)
    values = _values()
    whole, _ = _build(backend, values)
    chunked, _ = _build(backend, values)
    stepped, _ = _build(backend, values)
    oracle, _ = _build("torch", values)
    voltage = values["v"] if with_voltage else None

    with torch.no_grad(), environ.context(dt=DT):
        whole_output = whole(values["z"], v_seq=voltage)
        split = 3
        chunked_output = torch.cat(
            (
                chunked(
                    values["z"][:split],
                    v_seq=None if voltage is None else voltage[:split],
                ),
                chunked(
                    values["z"][split:],
                    v_seq=None if voltage is None else voltage[split:],
                ),
            )
        )
        stepped_output = _step_sequence(stepped, values["z"], voltage)
        expected = _step_sequence(oracle, values["z"], voltage)

    assert values["psc"].abs().sum() > 0
    assert values["history"].abs().sum() > 0
    for output in (whole_output, chunked_output, stepped_output):
        torch.testing.assert_close(output, expected, **FORWARD_KW)
    _assert_state(whole, oracle)
    _assert_state(chunked, oracle)
    _assert_state(stepped, oracle)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("kind", ["dense_delay", "sparse"])
def test_native_sequence_full_vjp_matches_module(backend, kind):
    """COBA losses reach voltage, state, dynamics, and connection weights."""
    require_cuda_backend(backend)
    values = _values()
    candidate, candidate_roots = _build(backend, values, kind=kind, differentiable=True)
    oracle, oracle_roots = _build("torch", values, kind=kind, differentiable=True)
    z = values["z"].clone().requires_grad_()
    zr = values["z"].clone().requires_grad_()
    v = values["v"].clone().requires_grad_()
    vr = values["v"].clone().requires_grad_()

    with environ.context(dt=DT):
        actual = candidate(z, v_seq=v)
        expected = _step_sequence(oracle, zr, vr)
        generator = torch.Generator(device="cuda").manual_seed(1602)
        output_cotangent = torch.randn(actual.shape, generator=generator, device="cuda")
        psc_cotangent = torch.randn(
            candidate.psc.shape, generator=generator, device="cuda"
        )
        actual_results: tuple[torch.Tensor, ...] = (actual, candidate.psc)
        expected_results: tuple[torch.Tensor, ...] = (expected, oracle.psc)
        cotangents: tuple[torch.Tensor, ...] = (output_cotangent, psc_cotangent)
        if candidate.delayed_spike is not None:
            history_cotangent = torch.randn(
                candidate.delayed_spike.history.shape,
                generator=generator,
                device="cuda",
            )
            actual_results += (candidate.delayed_spike.history,)
            expected_results += (oracle.delayed_spike.history,)
            cotangents += (history_cotangent,)
        actual_gradients = torch.autograd.grad(
            actual_results, (z, v, *candidate_roots), cotangents
        )
        expected_gradients = torch.autograd.grad(
            expected_results, (zr, vr, *oracle_roots), cotangents
        )

    torch.testing.assert_close(actual, expected, **FORWARD_KW)
    _assert_state(candidate, oracle)
    for gradient, reference in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        torch.testing.assert_close(gradient, reference, **GRADIENT_KW)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_compiled_conductance_replays_changed_input_training_steps(
    backend, monkeypatch
):
    """CUDA Graph Trees replay COBA forward and backward with carried state."""
    require_cuda_backend(backend)
    values = _values()
    model, _ = _build(backend, values, kind="dense", differentiable=True)
    reference, _ = _build("torch", values, kind="dense", differentiable=True)
    trials = (
        (values["z"], values["v"]),
        (values["z"] * 0.61 + 0.07, values["v"] * 0.97 - 0.3),
        (values["z"] * -0.43 + 0.12, values["v"] * 1.02 + 0.4),
    )
    external_psc = model.psc.detach().clone().requires_grad_()
    reference_psc = reference.psc.detach().clone().requires_grad_()
    carried_psc = external_psc.detach().clone()
    reference_carried_psc = reference_psc.detach().clone()
    replay_calls = []
    original_replay = torch.cuda.CUDAGraph.replay

    def counted_replay(graph):
        replay_calls.append(graph)
        return original_replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted_replay)

    def loss(z_seq, v_seq):
        output = model(z_seq, v_seq=v_seq)
        return output.square().sum() + model.psc.square().sum()

    compiled_loss = torch.compile(loss, mode="reduce-overhead", fullgraph=True)
    replay_deltas = []
    for z_value, v_value in trials:
        with torch.no_grad():
            external_psc.copy_(carried_psc)
            reference_psc.copy_(reference_carried_psc)
        model.psc = external_psc
        reference.psc = reference_psc
        z = z_value.detach().clone().requires_grad_()
        v = v_value.detach().clone().requires_grad_()
        zr = z_value.detach().clone().requires_grad_()
        vr = v_value.detach().clone().requires_grad_()
        actual_inputs = (
            z,
            v,
            external_psc,
            model.linear.weight,
            model.tau_syn,
            model.E,
        )
        expected_inputs = (
            zr,
            vr,
            reference_psc,
            reference.linear.weight,
            reference.tau_syn,
            reference.E,
        )
        before_replay = len(replay_calls)
        torch.compiler.cudagraph_mark_step_begin()
        with environ.context(dt=DT):
            actual_loss = compiled_loss(z, v)
            expected_output = reference(zr, v_seq=vr)
            expected_loss = (
                expected_output.square().sum() + reference.psc.square().sum()
            )
        actual_gradients = torch.autograd.grad(actual_loss, actual_inputs)
        replay_deltas.append(len(replay_calls) - before_replay)
        expected_gradients = torch.autograd.grad(expected_loss, expected_inputs)

        torch.testing.assert_close(actual_loss, expected_loss, **FORWARD_KW)
        _assert_state(model, reference)
        for gradient, expected_gradient in zip(
            actual_gradients, expected_gradients, strict=True
        ):
            torch.testing.assert_close(gradient, expected_gradient, **GRADIENT_KW)
        carried_psc = model.psc.detach().clone()
        reference_carried_psc = reference.psc.detach().clone()

    assert replay_deltas == [0, 2, 2]


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_native_inference_and_empty_chunks_keep_only_public_state(backend):
    """Inference saves no tape, while empty CUBA and COBA chunks advance
    nothing."""
    require_cuda_backend(backend)
    values = _values()
    model, _ = _build(backend, values)
    reference, _ = _build("torch", values)
    model.requires_grad_(False)
    reference.requires_grad_(False)
    saved = []

    with (
        environ.context(dt=DT),
        torch.autograd.graph.saved_tensors_hooks(
            lambda value: saved.append(value) or value, lambda value: value
        ),
        torch.inference_mode(),
    ):
        actual = model(values["z"], v_seq=values["v"])
        expected = _step_sequence(reference, values["z"], values["v"])
        state = model.psc.clone()
        history = model.delayed_spike.history.clone()
        empty_cuba = model(values["z"][:0])
        empty_voltage = torch.empty(0, N_POST, device="cuda", dtype=torch.float64)
        empty_coba = model(values["z"][:0], v_seq=empty_voltage)

    torch.testing.assert_close(actual, expected, **FORWARD_KW)
    _assert_state(model, reference)
    assert actual.grad_fn is None and saved == []
    assert empty_cuba.shape == (0, B, N_POST)
    assert empty_coba.shape == (0, B, N_POST)
    assert empty_coba.dtype == torch.float64
    torch.testing.assert_close(model.psc, state, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.delayed_spike.history, history, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_conductance_native_adapter_remains_standalone_only(backend):
    """Exact opt-in must not silently enable subclasses or recurrent COBA."""
    values = _values("cpu")

    class DerivedConductance(ConductancePSC):
        pass

    with pytest.raises(NotImplementedError, match="No cell backend template"):
        DerivedConductance(
            N_POST,
            4.7,
            torch.nn.Linear(N_SOURCE, N_POST, bias=False),
            0.0,
            backend=backend,
        )

    synapse, _ = _build(backend, values, kind="dense")
    neuron = LIF(N_POST, step_mode="m")
    init_net_state(neuron, batch_size=B, dtype=torch.float32)
    network = RecurrentNN(neuron, synapse, backend=KernelOptions(neuron=backend))
    initial_v = neuron.v.clone()
    initial_psc = synapse.psc.clone()
    with (
        environ.context(dt=DT),
        pytest.raises(NotImplementedError, match="voltage-aware recurrent scheduling"),
    ):
        network(torch.zeros(T, B, N_POST))

    torch.testing.assert_close(neuron.v, initial_v, rtol=0.0, atol=0.0)
    torch.testing.assert_close(synapse.psc, initial_psc, rtol=0.0, atol=0.0)


class _CountingInput(torch.nn.Module):
    """Expose whether a rejected recurrent call reached user callbacks."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("native_owner", ["network", "synapse"])
@pytest.mark.parametrize(
    "route",
    [
        "default",
        "cudagraph",
        "save_grad_history",
        "grad_state_names",
        "single",
        "chunk",
    ],
)
def test_native_recurrent_conductance_routes_reject_before_state_advance(
    backend, native_owner, route
):
    """Every recurrent entry rejects COBA before callbacks or memories
    change."""
    values = _values("cpu")
    synapse_backend = backend if native_owner == "synapse" else "torch"
    network_backend = backend if native_owner == "network" else "torch"
    recurrent_synapse, _ = _build(synapse_backend, values)
    neuron = LIF(N_POST, step_mode="m")
    init_net_state(neuron, batch_size=B, dtype=torch.float32)
    callback = _CountingInput()
    options = {
        "cudagraph": route == "cudagraph",
        "save_grad_history": route == "save_grad_history",
        "grad_state_names": ["neuron.v"] if route == "grad_state_names" else None,
    }
    network = RecurrentNN(
        neuron,
        recurrent_synapse,
        neuron_inp_module=callback,
        backend=network_backend,
        **options,
    )
    initial_v = neuron.v.clone()
    initial_psc = recurrent_synapse.psc.clone()
    initial_history = recurrent_synapse.delayed_spike.history.clone()
    x = torch.zeros(T, B, N_POST)

    with (
        environ.context(dt=DT),
        pytest.raises(NotImplementedError, match="voltage-aware recurrent scheduling"),
    ):
        if route == "single":
            network.single_step_forward(x[0])
        elif route == "chunk":
            network._run_chunk_steps(x, loop_args=(0,), unroll_size=2)
        else:
            network(x)

    assert callback.calls == 0
    torch.testing.assert_close(neuron.v, initial_v, rtol=0.0, atol=0.0)
    torch.testing.assert_close(recurrent_synapse.psc, initial_psc, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        recurrent_synapse.delayed_spike.history,
        initial_history,
        rtol=0.0,
        atol=0.0,
    )


def test_torch_recurrent_conductance_keeps_legacy_cuba_step():
    """The native guard leaves the established all-Torch recurrent step
    intact."""
    values = _values("cpu")
    recurrent_synapse, _ = _build("torch", values)
    neuron = LIF(N_POST, step_mode="m")
    init_net_state(neuron, batch_size=B, dtype=torch.float32)
    network = RecurrentNN(neuron, recurrent_synapse, backend="torch")
    initial_psc = recurrent_synapse.psc.clone()

    with torch.no_grad(), environ.context(dt=DT):
        spikes, _ = network.single_step_forward(torch.zeros(B, N_POST))

    assert spikes.shape == (B, N_POST)
    assert not torch.equal(recurrent_synapse.psc, initial_psc)
