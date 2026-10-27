"""Verify public stateful adapters against the original step recurrence."""

import copy
from itertools import product

import pytest
import torch
from torch import nn

from btorch.models import environ
from btorch.models.kernels.options import KernelOptions
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC
from tests.models.kernels.helpers import (
    GPU_BACKENDS,
    require_cuda_backend,
    validation_cases,
)


def test_compiled_cell_reads_changed_environment_dt_without_retracing():
    # A timestep is an invocation input, including when the caller supplies a
    # Python scalar through environ. Compare continued membrane states against
    # the existing Python model: subthreshold spikes alone would hide a stale dt.
    cell = LIF(2, backend=KernelOptions(), step_mode="m")
    cell.init_state(batch_size=1, dtype=torch.float32)
    reference = copy.deepcopy(cell)
    reference.backend = "torch"
    graphs = []

    def compiler(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(cell, backend=compiler, fullgraph=True)
    x = torch.full((3, 1, 2), 0.2)
    with torch.no_grad():
        for dt in (0.25, 1.0, 0.25):
            with environ.context(dt=dt):
                actual = compiled(x)
                expected = reference(x)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(cell.v, reference.v)
    assert len(graphs) == 1


def test_psc_sequence_continues_state_and_input_gradients():
    # The old PSC multi-step convolution starts at rest. Explicit kernel
    # options request a stateful sequence instead; compare split calls with
    # genuine step execution, including an initial current loss.
    cell = AlphaPSC(3, 2.0, nn.Linear(3, 3, bias=False), backend=KernelOptions())
    cell.float()
    cell.init_state(batch_size=2, dtype=torch.float32)
    reference = copy.deepcopy(cell)
    reference.backend = "torch"
    x = torch.randn(7, 2, 3, requires_grad=True, dtype=torch.float32)
    x_reference = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        actual = torch.cat(
            (cell.multi_step_forward(x[:3]), cell.multi_step_forward(x[3:]))
        )
        expected = torch.stack(
            [reference.single_step_forward(step) for step in x_reference]
        )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cell.h, reference.h)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, x_reference.grad)
    torch.testing.assert_close(cell.linear.weight.grad, reference.linear.weight.grad)


def test_recurrent_eval_retains_encoder_and_continuation_gradients():
    # eval() must not suppress the graph. A frozen recurrent layer still
    # propagates gradients to its upstream input over both calls, and records
    # may contribute their own auxiliary loss.
    neuron = LIF(3, v_threshold=0.7, trainable_param={"tau"})
    synapse = AlphaPSC(3, 2.0, nn.Linear(3, 3, bias=False))
    model = RecurrentNN(
        neuron,
        synapse,
        backend=KernelOptions(),
        update_state_names=("neuron.v", "synapse.psc"),
    )
    model.float()
    neuron.init_state(batch_size=2, dtype=torch.float32)
    synapse.init_state(batch_size=2, dtype=torch.float32)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    model.eval()
    x = torch.randn(6, 2, 3, requires_grad=True, dtype=torch.float32)
    x_reference = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        first, first_records = model(x[:2])
        second, second_records = model(x[2:])
        actual = torch.cat((first, second))
        actual_v = torch.cat((first_records["neuron.v"], second_records["neuron.v"]))
        expected, expected_records = reference(x_reference)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_v, expected_records["neuron.v"])
    (actual.sum() + actual_v.square().sum()).backward()
    (expected.sum() + expected_records["neuron.v"].square().sum()).backward()
    torch.testing.assert_close(x.grad, x_reference.grad)
    torch.testing.assert_close(neuron.tau.grad, reference.neuron.tau.grad)
    torch.testing.assert_close(
        synapse.linear.weight.grad, reference.synapse.linear.weight.grad
    )


def test_recurrent_fullgraph_encoder_readout():
    # A real composed network verifies the opaque boundary includes the whole
    # sequence, while AOTAutograd retains the upstream encoder and readout.
    torch.manual_seed(104)
    rnn = RecurrentNN(
        LIF(3),
        AlphaPSC(3, 2.0, nn.Linear(3, 3, bias=False)),
        backend=KernelOptions(),
        update_state_names=("neuron.v",),
    )
    rnn.float()
    rnn.neuron.init_state(batch_size=2, dtype=torch.float32)
    rnn.synapse.init_state(batch_size=2, dtype=torch.float32)
    encoder, readout = nn.Linear(4, 3).float(), nn.Linear(3, 2).float()

    # Ensure at least one spike: a silent random encoder gives correctly
    # zero readout-weight gradients and makes this connectivity assertion flaky.
    with torch.no_grad():
        encoder.bias.fill_(8.0)
        rnn.synapse.linear.weight.zero_()

    def network(x):
        spikes, states = rnn(encoder(x))
        return readout(spikes).square().sum() + states["neuron.v"].square().sum()

    x = torch.randn(5, 2, 4, dtype=torch.float32, requires_grad=True)
    with environ.context(dt=1.0):
        loss = torch.compile(network, backend="aot_eager", fullgraph=True)(x)
        loss.backward()
    assert torch.isfinite(x.grad).all()
    assert encoder.weight.grad.abs().sum() > 0
    assert readout.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("cell_backend", ("torch", *GPU_BACKENDS))
@pytest.mark.parametrize("neuron_model", ["lif", "izhikevich"])
def test_sparse_heter_delay_preserves_initial_ring_adjoint(cell_backend, neuron_model):
    # Compare the complete sparse/receptor/delay pipeline to the existing
    # Python Module recurrence, rather than a second implementation of its
    # equations. Nonzero incoming states expose continuation errors; losses
    # on all states expose missing adjoints even when spikes happen to match.
    from btorch.models.connection import SparseConnection, Synapse
    from btorch.models.kernels.adapters import prepare_cell
    from btorch.models.neurons.izhikevich import Izhikevich
    from btorch.models.synapse import HeterSynapsePSC

    device = "cpu" if cell_backend == "torch" else "cuda"
    if device == "cuda":
        require_cuda_backend(cell_backend)
    torch.manual_seed(163)
    connection = SparseConnection.from_edges(
        torch.tensor([0, 0, 1, 2]),
        torch.tensor([1, 2, 0, 1]),
        3,
        3,
        Synapse(
            delay=torch.tensor([0, 2, 1, 2]),
            receptor=torch.tensor([0, 1, 0, 1]),
            n_delay=3,
            n_receptor=2,
        ),
        values=torch.tensor([0.3, -0.4, 0.2, 0.6], dtype=torch.float32),
    )
    synapse = HeterSynapsePSC(3, 2, None, connection, base_psc=AlphaPSC, tau_syn=2.0)
    neuron = (
        LIF(3, tau=2.0, v_threshold=0.7)
        if neuron_model == "lif"
        else Izhikevich(
            3,
            v_threshold=0.7,
            v_reset=0.0,
            c_m=1.0,
            v_rest=0.0,
            v_peak=1.0,
            k=0.1,
            a=0.03,
            b=0.1,
            d=0.05,
        )
    )
    neuron_names = ("v",) if neuron_model == "lif" else ("v", "u")
    record_names = tuple(f"neuron.{name}" for name in neuron_names) + (
        "synapse.base_psc.psc",
        "synapse.base_psc.h",
    )
    model = RecurrentNN(
        neuron,
        synapse,
        backend=KernelOptions(
            neuron=cell_backend, connection="torch" if device == "cpu" else "cusparse"
        ),
        update_state_names=record_names,
    ).to(device=device, dtype=torch.float32)
    neuron.init_state(batch_size=2, dtype=torch.float32, device=device)
    synapse.init_state(batch_size=2, dtype=torch.float32, device=device)
    synapse.base_psc.init_state(batch_size=2, dtype=torch.float32, device=device)
    x = torch.randn(6, 2, 3, dtype=torch.float32, device=device, requires_grad=True)
    state_owners = [(neuron, name) for name in neuron_names] + [
        (synapse.base_psc, "psc"),
        (synapse.base_psc, "h"),
        (synapse.delayed_spike, "history"),
    ]
    with environ.context(dt=1.0):
        # Buffers that actually enter the equations also deserve gradient
        # coverage. Integer routing metadata intentionally remains frozen.
        for cell in (neuron, synapse.base_psc):
            for parameter in prepare_cell(cell, x)[3]:
                parameter.requires_grad_()
        for owner, name in state_owners:
            setattr(
                owner,
                name,
                (torch.randn_like(getattr(owner, name)) * 0.1).requires_grad_(),
            )
        # HeterSynapsePSC caches the receptor sum separately; an incoming
        # Python state must keep that public current consistent with its PSC.
        initial = [getattr(owner, name) for owner, name in state_owners]
        reference = copy.deepcopy(model)
        reference.backend = "torch"
        reference_owners = [(reference.neuron, name) for name in neuron_names] + [
            (reference.synapse.base_psc, "psc"),
            (reference.synapse.base_psc, "h"),
            (reference.synapse.delayed_spike, "history"),
        ]
        initial_reference = [getattr(owner, name) for owner, name in reference_owners]
        synapse.psc = synapse.base_psc.psc.reshape(2, 3, 2).sum(-1)
        reference.synapse.psc = reference.synapse.base_psc.psc.reshape(2, 3, 2).sum(-1)
        parameters = (
            *prepare_cell(neuron, x)[3],
            *prepare_cell(synapse.base_psc, x)[3],
            connection.weight.value,
        )
        reference_parameters = (
            *prepare_cell(reference.neuron, x)[3],
            *prepare_cell(reference.synapse.base_psc, x)[3],
            reference.synapse.linear.weight.value,
        )
        xr = x.detach().clone().requires_grad_()
        actual, records = model(x)
        expected, expected_records = reference(xr)
        torch.testing.assert_close(actual, expected)
        for name in record_names:
            torch.testing.assert_close(
                records[name], expected_records[name], atol=2e-5, rtol=2e-5
            )
        loss = (
            actual.sum()
            + sum(record.square().sum() for record in records.values())
            + synapse.delayed_spike.history.square().sum()
        )
        expected_loss = (
            expected.sum()
            + sum(record.square().sum() for record in expected_records.values())
            + reference.synapse.delayed_spike.history.square().sum()
        )
        loss.backward()
        expected_loss.backward()
    for name, value, golden in zip(
        [
            "input",
            *[name for _, name in state_owners],
            *[f"parameter_{i}" for i in range(len(parameters))],
        ],
        [x, *initial, *parameters],
        [xr, *initial_reference, *reference_parameters],
    ):
        assert value.grad is not None, name
        torch.testing.assert_close(
            value.grad,
            golden.grad,
            atol=3e-5,
            rtol=3e-5,
            msg=lambda details: f"{name}: {details}",
        )


@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_chunks_keep_full_bptt_and_state_records(cpu_offload, device):
    # Checkpoint restoration must restore each chunk's incoming voltage and
    # PSC before recomputation; detaching these boundaries loses early input
    # gradients. Compare the final state loss and all trajectory records.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for device-to-host offload")
    model = RecurrentNN(
        LIF(3, trainable_param={"tau"}),
        AlphaPSC(3, 2.0, nn.Linear(3, 3, bias=False)),
        backend=KernelOptions(),
        chunk_size=2,
        unroll=2,
        grad_checkpoint=True,
        cpu_offload=cpu_offload,
        update_state_names=("neuron.v", "synapse.psc"),
    )
    model.to(device=device, dtype=torch.float32)
    model.neuron.init_state(batch_size=2, dtype=torch.float32, device=device)
    model.synapse.init_state(batch_size=2, dtype=torch.float32, device=device)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    reference.chunk_size = None
    reference.grad_checkpoint = False
    reference.cpu_offload = False
    x = torch.randn(6, 2, 3, dtype=torch.float32, device=device, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        actual, records = model(x)
        expected, expected_records = reference(x_reference)
        (
            actual.sum()
            + records["synapse.psc"].square().sum()
            + model.neuron.v.square().sum()
        ).backward()
        (
            expected.sum()
            + expected_records["synapse.psc"].square().sum()
            + reference.neuron.v.square().sum()
        ).backward()
    # Offloaded outputs still carry the copy adjoint back to the GPU sequence,
    # including the first chunk. Move only for comparison, not before the loss.
    torch.testing.assert_close(actual.to(device), expected)
    torch.testing.assert_close(
        records["neuron.v"].to(device), expected_records["neuron.v"]
    )
    torch.testing.assert_close(x.grad, x_reference.grad)
    torch.testing.assert_close(
        model.synapse.linear.weight.grad, reference.synapse.linear.weight.grad
    )


def test_merge_shared_dynamics_matches_receptors_and_checkpoint():
    # A scalar tau is genuinely shared by all receptor channels. Reducing the
    # linear PSC state therefore preserves the total current and its gradient,
    # while independent receptor currents are deliberately unrecoverable.
    import pytest

    from btorch.models.synapse import HeterSynapsePSC

    synapse = HeterSynapsePSC(
        3, 2, None, nn.Linear(3, 6, bias=False), base_psc=AlphaPSC, tau_syn=2.0
    )
    model = RecurrentNN(
        LIF(3),
        synapse,
        backend=KernelOptions(merge_synapse=True),
        update_state_names=("neuron.v", "synapse.psc"),
    )
    model.float()
    model.neuron.init_state(batch_size=2, dtype=torch.float32, persistent=True)
    synapse.init_state(batch_size=2, dtype=torch.float32, persistent=True)
    synapse.base_psc.init_state(batch_size=2, dtype=torch.float32, persistent=True)
    synapse.base_psc.tau_syn.requires_grad_()
    reference = copy.deepcopy(model)
    restored = copy.deepcopy(model)
    reference.backend = "torch"
    x = torch.randn(6, 2, 3, dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        actual, records = model(x[:3])
        expected, reference_records = reference(xr[:3])
        checkpoint = copy.deepcopy(model.state_dict())
        restored.load_state_dict(checkpoint)
        # A fresh receiver has never run a sequence, but its checkpoint already
        # contains merged dynamics. Switching modes here would interpret the
        # reduced state as independent receptor state and corrupt continuation.
        with pytest.raises(RuntimeError, match="Reset"):
            restored.backend = "torch"
        next_actual, _ = model(x[3:])
        next_expected, _ = reference(xr[3:])
        restored_output, _ = restored(x[3:].detach())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(records["synapse.psc"], reference_records["synapse.psc"])
    torch.testing.assert_close(next_actual, next_expected)
    torch.testing.assert_close(restored_output, next_actual)
    (actual.sum() + next_actual.sum() + model.neuron.v.square().sum()).backward()
    (
        expected.sum() + next_expected.sum() + reference.neuron.v.square().sum()
    ).backward()
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(
        synapse.base_psc.tau_syn.grad, reference.synapse.base_psc.tau_syn.grad
    )
    with pytest.raises(ValueError, match="independent receptor"):
        synapse.get_psc(0)
    with pytest.raises(RuntimeError, match="Reset"):
        model.backend = "torch"


def test_sparse_fullgraph_sequence_and_upstream_gradient():
    # Warm route preparation occurs before compilation. The compiled graph
    # then includes one whole-sequence op and preserves a trainable encoder's
    # adjoint even when the recurrent connection parameters are frozen.
    from btorch.models.connection import SparseConnection

    conn = SparseConnection.from_edges(
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 0]),
        3,
        3,
        values=torch.tensor([0.2, -0.3, 0.4], dtype=torch.float32),
    )
    rnn = RecurrentNN(
        LIF(3),
        AlphaPSC(3, 2.0, conn),
        backend=KernelOptions(connection="torch"),
        update_state_names=("neuron.v",),
    )
    rnn.float()
    rnn.neuron.init_state(batch_size=2, dtype=torch.float32)
    rnn.synapse.init_state(batch_size=2, dtype=torch.float32)
    rnn.requires_grad_(False)
    reference = copy.deepcopy(rnn)
    reference.backend = "torch"
    encoder = nn.Linear(4, 3).float()
    encoder_reference = copy.deepcopy(encoder)
    x = torch.randn(5, 2, 4, dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        # A no-grad warmup binds immutable routes; reset restores the empty
        # trajectory without invalidating the unchanged sparse topology.
        with torch.no_grad():
            rnn(encoder(x))
        rnn.neuron.reset(batch_size=2, dtype=torch.float32)
        rnn.synapse.reset(batch_size=2, dtype=torch.float32)

        def network(value):
            spikes, records = rnn(encoder(value))
            return spikes.sum() + records["neuron.v"].square().sum()

        actual = torch.compile(network, fullgraph=True, backend="aot_eager")(x)
        expected, records = reference(encoder_reference(xr))
        loss_reference = expected.sum() + records["neuron.v"].square().sum()
        actual.backward()
        loss_reference.backward()
    torch.testing.assert_close(actual, loss_reference)
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(encoder.weight.grad, encoder_reference.weight.grad)


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_sparse_fullgraph_training_and_cudagraph_replay(backend):
    # Verify the two supported execution boundaries in a complete sparse RNN:
    # compiled training retains encoder gradients, and the inference runner
    # replays a captured trajectory while advancing its recurrent state.
    from btorch.models.connection import SparseConnection

    require_cuda_backend(backend)
    conn = SparseConnection.from_edges(
        torch.tensor([0, 1, 2], device="cuda"),
        torch.tensor([1, 2, 0], device="cuda"),
        3,
        3,
        values=torch.tensor([0.2, -0.3, 0.4], dtype=torch.float32, device="cuda"),
    )
    rnn = RecurrentNN(
        LIF(3),
        AlphaPSC(3, 2.0, conn),
        backend=KernelOptions(neuron=backend, connection="cusparse"),
        update_state_names=("neuron.v",),
    )
    rnn.to(device="cuda", dtype=torch.float32)
    rnn.neuron.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    rnn.synapse.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    reference = copy.deepcopy(rnn)
    reference.backend = "torch"
    encoder = nn.Linear(4, 3).to(device="cuda", dtype=torch.float32)
    encoder_reference = copy.deepcopy(encoder)
    x = torch.randn(5, 2, 4, dtype=torch.float32, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        with torch.no_grad():
            rnn(encoder(x))
        rnn.neuron.reset(batch_size=2, dtype=torch.float32, device="cuda")
        rnn.synapse.reset(batch_size=2, dtype=torch.float32, device="cuda")

        def network(value):
            spikes, records = rnn(encoder(value))
            return spikes.sum() + records["neuron.v"].square().sum()

        actual = torch.compile(network, backend="aot_eager", fullgraph=True)(x)
        expected, records = reference(encoder_reference(xr))
        expected_loss = expected.sum() + records["neuron.v"].square().sum()
        actual.backward()
        expected_loss.backward()
        torch.testing.assert_close(actual, expected_loss, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(x.grad, xr.grad, atol=2e-5, rtol=2e-5)
        rnn.neuron.reset(batch_size=2, dtype=torch.float32, device="cuda")
        rnn.synapse.reset(batch_size=2, dtype=torch.float32, device="cuda")
        reference.neuron.reset(batch_size=2, dtype=torch.float32, device="cuda")
        reference.synapse.reset(batch_size=2, dtype=torch.float32, device="cuda")
        rnn.requires_grad_(False)
        reference.requires_grad_(False)
        rnn.cudagraph = True
        drive = encoder(x.detach()).detach()
        with torch.no_grad():
            for _ in range(2):
                captured, state = rnn(drive)
                reference_spikes, reference_state = reference(drive)
                torch.testing.assert_close(captured, reference_spikes)
                torch.testing.assert_close(
                    state["neuron.v"], reference_state["neuron.v"], atol=2e-5, rtol=2e-5
                )
        assert rnn._cudagraph_runner.n_captures == 1
        # Rewiring changes the topology while all tensor dimensions stay the
        # same. Replay must invalidate its old binding and match the updated
        # eager graph, rather than silently using the formerly captured edge.
        with torch.no_grad():
            for connection in (conn, reference.synapse.linear):
                connection.set_edges_(
                    torch.tensor([0], device="cuda"),
                    pre=torch.tensor([0], device="cuda"),
                    post=torch.tensor([2], device="cuda"),
                )
            captured, state = rnn(drive)
            reference_spikes, reference_state = reference(drive)
            torch.testing.assert_close(captured, reference_spikes)
            torch.testing.assert_close(
                state["neuron.v"], reference_state["neuron.v"], atol=2e-5, rtol=2e-5
            )
        assert rnn._cudagraph_runner.n_captures == 2
        # Backend changes preserve the numerical state but invalidate the
        # captured implementation even though topology and shapes are stable.
        rnn.backend = KernelOptions(neuron="torch", connection="cusparse")
        with torch.no_grad():
            captured, state = rnn(drive)
            reference_spikes, reference_state = reference(drive)
            torch.testing.assert_close(captured, reference_spikes)
            torch.testing.assert_close(
                state["neuron.v"], reference_state["neuron.v"], atol=2e-5, rtol=2e-5
            )
        assert rnn._cudagraph_runner.n_captures == 3


# Module adaptation needs scalar/mode/multi-state pairs, not every pairing
# of independent equations. Keep CPU operator and all native renderers covered.
@pytest.mark.parametrize(
    "cell_backend,neuron_model,psc_model",
    validation_cases(
        product(
            ("torch", *GPU_BACKENDS),
            ("if", "lif", "glif3", "izhikevich"),
            ("exponential", "alpha", "alpha_billeh", "dual_exponential"),
        ),
        selected=(
            ("torch", "if", "exponential"),
            ("torch", "glif3", "dual_exponential"),
            ("triton", "if", "exponential"),
            ("triton", "lif", "alpha"),
            ("triton", "glif3", "alpha_billeh"),
            ("triton", "izhikevich", "dual_exponential"),
            ("cupy", "glif3", "alpha"),
            ("tilelang", "izhikevich", "dual_exponential"),
        ),
    ),
)
def test_recurrent_model_pairs_all_float_and_initial_state_gradients(
    neuron_model, psc_model, cell_backend
):
    # Exercise the selected cell pair as an actual recurrent network. Losses
    # on every recorded state require all adjoints, rather than only the spike
    # surrogate path. Floating buffers are made trainable deliberately so PSC
    # time constants and gain coefficients participate in the same comparison.
    from btorch.models.kernels.adapters import prepare_cell
    from btorch.models.neurons.glif import GLIF3
    from btorch.models.neurons.izhikevich import Izhikevich
    from btorch.models.neurons.lif import IF
    from btorch.models.synapse import AlphaPSCBilleh, DualExponentialPSC, ExponentialPSC

    device = "cpu" if cell_backend == "torch" else "cuda"
    if device == "cuda":
        require_cuda_backend(cell_backend)
    common = dict(v_threshold=0.7, v_reset=0.0, c_m=1.0)
    neuron = {
        "if": lambda: IF(3, **common),
        "lif": lambda: LIF(3, tau=2.0, **common),
        "glif3": lambda: GLIF3(
            3, tau=2.0, v_rest=0.0, k=[0.2, 0.3], asc_amps=[-0.1, -0.2], **common
        ),
        "izhikevich": lambda: Izhikevich(
            3, v_rest=0.0, v_peak=1.0, k=0.1, a=0.03, b=0.1, d=0.05, **common
        ),
    }[neuron_model]()
    connection = nn.Linear(3, 3, bias=True)
    with environ.context(dt=1.0):
        psc = {
            "exponential": lambda: ExponentialPSC(3, 2.0, connection),
            "alpha": lambda: AlphaPSC(3, 2.0, connection),
            "alpha_billeh": lambda: AlphaPSCBilleh(3, 2.0, connection),
            "dual_exponential": lambda: DualExponentialPSC(3, 3.0, 1.0, connection),
        }[psc_model]()
    model = RecurrentNN(neuron, psc, backend=KernelOptions(neuron=cell_backend))
    model.to(device=device, dtype=torch.float32)
    neuron.init_state(batch_size=2, dtype=torch.float32, device=device)
    psc.init_state(batch_size=2, dtype=torch.float32, device=device)
    x = torch.randn(5, 2, 3, dtype=torch.float32, device=device, requires_grad=True)
    with environ.context(dt=1.0):
        _, neuron_names, _, neuron_parameters = prepare_cell(neuron, x)
        _, psc_names, _, psc_parameters = prepare_cell(psc, x)
        for parameter in (*neuron_parameters, *psc_parameters):
            parameter.requires_grad_()
        for cell, names in ((neuron, neuron_names), (psc, psc_names)):
            for name in names:
                getattr(cell, name).requires_grad_()
        reference = copy.deepcopy(model)
        reference.backend = "torch"
        xr = x.detach().clone().requires_grad_()
        # Keep references to the incoming leaf states before forward writes
        # final states back to each module.
        initial = [
            getattr(cell, name)
            for cell, names in ((neuron, neuron_names), (psc, psc_names))
            for name in names
        ]
        initial_reference = [
            getattr(cell, name)
            for cell, names in (
                (reference.neuron, neuron_names),
                (reference.synapse, psc_names),
            )
            for name in names
        ]
        _, _, _, reference_neuron_parameters = prepare_cell(reference.neuron, xr)
        _, _, _, reference_psc_parameters = prepare_cell(reference.synapse, xr)
        actual, records = model(x)
        expected, expected_records = reference(xr)
        (
            actual.sum() + sum(value.square().sum() for value in records.values())
        ).backward()
        (
            expected.sum()
            + sum(value.square().sum() for value in expected_records.values())
        ).backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, xr.grad, atol=2e-5, rtol=2e-5)
    for name, record in records.items():
        torch.testing.assert_close(record, expected_records[name], atol=2e-5, rtol=2e-5)
    for actual_state, reference_state in zip(initial, initial_reference):
        torch.testing.assert_close(
            actual_state.grad, reference_state.grad, atol=2e-5, rtol=2e-5
        )
    for actual_parameter, reference_parameter in zip(
        (*neuron_parameters, *psc_parameters, connection.weight, connection.bias),
        (
            *reference_neuron_parameters,
            *reference_psc_parameters,
            reference.synapse.linear.weight,
            reference.synapse.linear.bias,
        ),
    ):
        torch.testing.assert_close(
            actual_parameter.grad, reference_parameter.grad, atol=2e-5, rtol=2e-5
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_encoder_rnn_readout_preserves_fp32_states(dtype):
    # Parameters and external activations use low precision, but the recurrent
    # state and connection accumulation follow the same FP32 policy as the
    # reference. Casts remain differentiable back to the original leaf dtype.
    rnn = RecurrentNN(
        LIF(3, trainable_param={"tau"}),
        AlphaPSC(3, 2.0, nn.Linear(3, 3, bias=False)),
        backend=KernelOptions(),
        update_state_names=("neuron.v",),
    )
    rnn.to(dtype=dtype)
    rnn.neuron.init_state(batch_size=2, dtype=dtype)
    rnn.synapse.init_state(batch_size=2, dtype=dtype)
    reference = copy.deepcopy(rnn).float()
    reference.backend = "torch"
    encoder, readout = nn.Linear(4, 3).to(dtype=dtype), nn.Linear(3, 2).to(dtype=dtype)
    encoder_reference, readout_reference = (
        copy.deepcopy(encoder).float(),
        copy.deepcopy(readout).float(),
    )
    x = torch.randn(5, 2, 4, dtype=dtype, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    with environ.context(dt=1.0):
        actual, states = rnn(encoder(x))
        expected, reference_states = reference(encoder_reference(xr).to(dtype).float())
        loss = (
            readout(actual).float().square().sum() + states["neuron.v"].square().sum()
        )
        reference_loss = (
            readout_reference(expected).to(dtype).float().square().sum()
            + reference_states["neuron.v"].square().sum()
        )
        loss.backward()
        reference_loss.backward()
    torch.testing.assert_close(actual.float(), expected)
    torch.testing.assert_close(
        states["neuron.v"], reference_states["neuron.v"], atol=3e-2, rtol=3e-2
    )
    assert rnn.neuron.v.dtype == torch.float32
    assert rnn.synapse.psc.dtype == torch.float32
    torch.testing.assert_close(x.grad.float(), xr.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        rnn.synapse.linear.weight.grad.float(),
        reference.synapse.linear.weight.grad,
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        encoder.weight.grad.float(), encoder_reference.weight.grad, atol=3e-2, rtol=3e-2
    )


@pytest.mark.parametrize("cell_backend", ("torch", *GPU_BACKENDS))
def test_frozen_receptor_vectors_merge_fullgraph_and_capture(cell_backend):
    # Frozen parameters may differ between neurons but must agree across that
    # neuron's receptors. Compatibility is checked at construction/move/load,
    # before Dynamo tracing and before CUDA graph capture begins.
    from btorch.models.synapse import HeterSynapsePSC

    device = "cpu" if cell_backend == "torch" else "cuda"
    if device == "cuda":
        require_cuda_backend(cell_backend)
    tau = torch.tensor([2.0, 2.0, 3.0, 3.0, 4.0, 4.0], dtype=torch.float32)
    synapse = HeterSynapsePSC(
        3, 2, None, nn.Linear(3, 6, bias=False), base_psc=AlphaPSC, tau_syn=tau
    )
    model = RecurrentNN(
        LIF(3),
        synapse,
        backend=KernelOptions(neuron=cell_backend, merge_synapse=True),
        update_state_names=("neuron.v", "synapse.psc"),
    )
    model.to(device=device, dtype=torch.float32)
    model.neuron.init_state(batch_size=2, device=device, dtype=torch.float32)
    synapse.init_state(batch_size=2, device=device, dtype=torch.float32)
    synapse.base_psc.init_state(batch_size=2, device=device, dtype=torch.float32)
    model.requires_grad_(False)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    x = torch.randn(5, 2, 3, device=device, dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):

        def network(value):
            spikes, states = model(value)
            return (
                spikes.sum()
                + states["synapse.psc"].square().sum()
                + states["neuron.v"].square().sum()
            )

        actual_loss = torch.compile(network, backend="aot_eager", fullgraph=True)(x)
        expected, states = reference(xr)
        reference_loss = (
            expected.sum()
            + states["synapse.psc"].square().sum()
            + states["neuron.v"].square().sum()
        )
        actual_loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(actual_loss, reference_loss, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(x.grad, xr.grad, atol=2e-5, rtol=2e-5)
        if device == "cuda":
            for layer in (model, reference):
                layer.neuron.reset(batch_size=2, device=device, dtype=torch.float32)
                layer.synapse.reset(batch_size=2, device=device, dtype=torch.float32)
                layer.synapse.base_psc.reset(
                    batch_size=2, device=device, dtype=torch.float32
                )
            model.cudagraph = True
            with torch.no_grad():
                for _ in range(2):
                    actual, actual_states = model(x.detach())
                    expected, expected_states = reference(x.detach())
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(
                        actual_states["synapse.psc"],
                        expected_states["synapse.psc"],
                        atol=2e-5,
                        rtol=2e-5,
                    )
            assert model._cudagraph_runner.n_captures == 1


def test_external_backend_default_single_step_uses_registered_dynamics():
    # A plugin deliberately doubles the drive. If the default single-step
    # Module path silently falls back to Torch, both voltage and its upstream
    # derivative disagree with the independently stepped reference.
    from btorch.models.kernels import (
        NeuronSpec,
        register_cell_backend,
        resolve_cell_backend,
    )

    torch_backend = resolve_cell_backend("torch", NeuronSpec("lif"), "cpu")
    forward_sequence = torch_backend.forward
    backward_sequence = torch_backend.backward

    def forward(x_seq, states, parameters, spec, names):
        return forward_sequence(2 * x_seq, states, parameters, spec, names)

    def backward(
        x_seq,
        states,
        parameters,
        spec,
        names,
        histories,
        grad_output,
        grad_states,
        grad_histories,
        *,
        needs_input_grad,
    ):
        gx, gs, gp = backward_sequence(
            2 * x_seq,
            states,
            parameters,
            spec,
            names,
            histories,
            grad_output,
            grad_states,
            grad_histories,
            needs_input_grad=needs_input_grad,
        )
        return 2 * gx if gx is not None else None, gs, gp

    register_cell_backend(
        "integration_double_drive", forward=forward, backward=backward
    )
    node = LIF(2, backend="integration_double_drive", v_threshold=0.8)
    node.float()
    node.init_state(batch_size=1, dtype=torch.float32)
    reference = copy.deepcopy(node)
    reference.backend = "torch"
    x = torch.tensor([[0.3, 0.6]], dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        actual = node(x)
        expected = reference(2 * xr)
        (actual.sum() + node.v.square().sum()).backward()
        (expected.sum() + reference.v.square().sum()).backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(node.v, reference.v)
    torch.testing.assert_close(x.grad, xr.grad)


def test_merged_checkpoint_restores_unmerged_incoming_state():
    # The first checkpoint chunk receives full receptor states. Recomputing it
    # after later chunks must recognize that restored shape, even though the
    # Module's Python flag now describes merged states from the last forward.
    from btorch.models.synapse import HeterSynapsePSC

    synapse = HeterSynapsePSC(
        3, 2, None, nn.Linear(3, 6, bias=False), base_psc=AlphaPSC, tau_syn=2.0
    )
    model = RecurrentNN(
        LIF(3),
        synapse,
        backend=KernelOptions(merge_synapse=True),
        grad_checkpoint=True,
        unroll=2,
        update_state_names=("neuron.v", "synapse.psc"),
    )
    model.float()
    model.neuron.init_state(batch_size=2, dtype=torch.float32)
    synapse.init_state(batch_size=2, dtype=torch.float32)
    synapse.base_psc.init_state(batch_size=2, dtype=torch.float32)
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    reference.grad_checkpoint = False
    x = torch.randn(6, 2, 3, requires_grad=True, dtype=torch.float32)
    xr = x.detach().clone().requires_grad_()
    with environ.context(dt=1.0):
        actual, records = model(x)
        expected, expected_records = reference(xr)
        (
            actual.sum()
            + records["neuron.v"].square().sum()
            + records["synapse.psc"].square().sum()
        ).backward()
        (
            expected.sum()
            + expected_records["neuron.v"].square().sum()
            + expected_records["synapse.psc"].square().sum()
        ).backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, xr.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        synapse.linear.weight.grad,
        reference.synapse.linear.weight.grad,
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replaced_external_backend_invalidates_recurrent_capture():
    # External authors can replace a registered implementation while retaining
    # the same public name. Replay must execute the new equations, rather than
    # a previously captured callback. Compare against eager continuation and
    # inspect voltage so zero spikes cannot hide a stale implementation.
    from btorch.models.kernels import (
        NeuronSpec,
        register_cell_backend,
        resolve_cell_backend,
    )

    forward_sequence = resolve_cell_backend("torch", NeuronSpec("lif"), "cpu").forward

    name = "integration_capture_replacement"
    register_cell_backend(name, forward=forward_sequence, replace=True)
    model = RecurrentNN(
        LIF(3),
        AlphaPSC(3, 2.0, nn.Identity()),
        backend=KernelOptions(neuron=name),
        update_state_names=("neuron.v", "synapse.psc"),
    ).to(device="cuda", dtype=torch.float32)
    model.neuron.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    model.synapse.init_state(batch_size=2, dtype=torch.float32, device="cuda")
    model.requires_grad_(False)
    reference = copy.deepcopy(model)
    model.cudagraph = True
    x = torch.full((4, 2, 3), 0.1, device="cuda", dtype=torch.float32)
    with environ.context(dt=1.0), torch.no_grad():
        first, first_records = model(x)
        expected, expected_records = reference(x)
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(
            first_records["neuron.v"], expected_records["neuron.v"]
        )
        assert model._cudagraph_runner.n_captures == 1

        def doubled_drive(x_seq, states, parameters, spec, history_names):
            return forward_sequence(x_seq * 2, states, parameters, spec, history_names)

        register_cell_backend(name, forward=doubled_drive, replace=True)
        actual, records = model(x)
        expected, expected_records = reference(x)
        torch.testing.assert_close(actual, expected)
        for state_name in records:
            torch.testing.assert_close(
                records[state_name], expected_records[state_name]
            )
        assert model._cudagraph_runner.n_captures == 2


@pytest.mark.parametrize("compiled", [False, True])
def test_recurrent_refuses_replaced_backend_before_backward(compiled):
    # A pending loss belongs to the callbacks that produced its trajectory.
    # Replacing the registration must not silently use a new VJP, including
    # when AOTAutograd has compiled the backward custom operator already.
    from btorch.models.kernels import (
        NeuronSpec,
        register_cell_backend,
        resolve_cell_backend,
    )

    torch_backend = resolve_cell_backend("torch", NeuronSpec("lif"), "cpu")
    forward_sequence = torch_backend.forward
    backward_sequence = torch_backend.backward

    name = "integration_pending_replacement"
    register_cell_backend(
        name, forward=forward_sequence, backward=backward_sequence, replace=True
    )
    model = RecurrentNN(
        LIF(3),
        AlphaPSC(3, 2.0, nn.Identity()),
        backend=KernelOptions(neuron=name),
        update_state_names=("neuron.v",),
    ).float()
    model.neuron.init_state(batch_size=2, dtype=torch.float32)
    model.synapse.init_state(batch_size=2, dtype=torch.float32)

    def loss(value):
        spikes, records = model(value)
        return spikes.sum() + records["neuron.v"].square().sum()

    operation = (
        torch.compile(loss, backend="aot_eager", fullgraph=True) if compiled else loss
    )
    x = torch.full((4, 2, 3), 0.1, dtype=torch.float32, requires_grad=True)
    with environ.context(dt=1.0):
        pending = operation(x)
        register_cell_backend(
            name, forward=forward_sequence, backward=backward_sequence, replace=True
        )
        with pytest.raises(RuntimeError, match="registration changed"):
            pending.backward()
        # Replacement remains a supported development workflow once users
        # discard pending losses and create a fresh trajectory under new code.
        # Compiled callables must also be recreated after replacing callbacks;
        # use eager here so the new reverse contract is prepared at call time.
        model.neuron.reset(batch_size=2, dtype=torch.float32)
        model.synapse.reset(batch_size=2, dtype=torch.float32)
        loss(x).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_recurrent_rejects_higher_order_gradients_explicitly():
    # A handwritten first-order VJP is not a differentiable reverse model.
    # Reject create_graph at the first request instead of returning a silently
    # detached gradient, while preserving ordinary BPTT on a retained loss.
    model = RecurrentNN(
        LIF(3),
        AlphaPSC(3, 2.0, nn.Identity()),
        backend=KernelOptions(),
        update_state_names=("neuron.v",),
    ).float()
    model.neuron.init_state(batch_size=2, dtype=torch.float32)
    model.synapse.init_state(batch_size=2, dtype=torch.float32)
    x = torch.full((4, 2, 3), 0.1, dtype=torch.float32, requires_grad=True)
    with environ.context(dt=1.0):
        spikes, records = model(x)
        loss = spikes.sum() + records["neuron.v"].square().sum()
        with pytest.raises(NotImplementedError, match="first-order"):
            torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)
        gradient = torch.autograd.grad(loss, x)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
