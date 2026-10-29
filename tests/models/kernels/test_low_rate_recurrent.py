"""Compare low-rate Triton recurrence with the actual Torch network
schedule."""

import pytest
import torch

from btorch.models import BPTT, KernelOptions, environ
from btorch.models.connection import SparseConnection
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import Sigmoid, Truncated
from btorch.models.synapse import ExponentialPSC
from btorch.sparse import Hints
from btorch.sparse.runtime import kernels_triton_spikelist, spike_list
from tests.models.kernels.helpers import require_cuda_backend


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _network(backend, rate):
    connection = SparseConnection.from_edges(
        torch.tensor([0, 0, 1, 2, 3], device="cuda"),
        torch.tensor([1, 2, 2, 3, 0], device="cuda"),
        4,
        4,
        values=torch.tensor([0.1, -0.03, 0.12, 0.04, 0.08], device="cuda"),
        hints=Hints(firing_rate_hz=rate),
    )
    model = RecurrentNN(
        LIF(4, tau=2.0, v_threshold=0.5, v_reset=0.0, c_m=1.0),
        ExponentialPSC(4, 2.5, connection),
        backend=KernelOptions(
            neuron=backend, connection="triton" if rate is not None else "auto"
        )
        if isinstance(backend, str)
        else backend,
        update_state_names=("neuron.v", "synapse.psc"),
    ).to(device="cuda", dtype=torch.float32)
    model.neuron.init_state(batch_size=2, device="cuda", dtype=torch.float32)
    model.synapse.init_state(batch_size=2, device="cuda", dtype=torch.float32)
    return model


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_low_rate_recurrent_states_continuation_and_input_vjp(
    training, compiled, monkeypatch
):
    pytest.importorskip("triton")
    # Eager keeps the advisory-rate route; compiled execution explicitly asks
    # for push with no activity hint, through the public RSNN kernel options.
    native = _network(
        KernelOptions(neuron="triton", connection_method="push")
        if compiled
        else "triton",
        None if compiled else 4.0,
    )
    reference = _network("torch", None)
    if training:
        # The clipped derivative defines the mathematical oracle on both
        # sides; sparse BPTT changes execution, never the surrogate itself.
        for model in (native, reference):
            model.gradient = BPTT(sparse=True)
            model.neuron.surrogate_function = Truncated(Sigmoid(alpha=2.0), 0.4)
    executions = []
    original = spike_list._PreparedSpikeList._consume

    def observed(*args, **kwargs):
        executions.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(spike_list._PreparedSpikeList, "_consume", observed)
    # Train through the actual Inductor boundary; the inference variant keeps
    # a lightweight Dynamo route check alongside the CUDA-graph tests below.
    run = (
        torch.compile(
            native, backend="inductor" if training else "eager", fullgraph=True
        )
        if compiled
        else native
    )
    # Strong input deliberately produces more spikes than hinted. Changing
    # input between windows exercises both ongoing PSC memory and continuation.
    x = torch.linspace(0.4, 1.2, 13 * 2 * 4, device="cuda").reshape(13, 2, 4)
    x.requires_grad_(training)
    x_ref = x.detach().clone().requires_grad_(training)
    with torch.set_grad_enabled(training), environ.context(dt=0.25):
        output, records = run(x)
        expected, expected_records = reference(x_ref)
        if not compiled:
            assert native.synapse.linear._plan.algorithm == "push"
            assert native.synapse.linear._plan.backend == "triton"
        # A compiled RSNN binds its explicit options without mutating the
        # standalone connection's default plan. Observe the actual fanout below.
        # Verify execution, since a correct plan can still silently fall back
        # to pull when a compiled call has no prepared weight buffer.
        assert executions
        assert torch.count_nonzero(output) > 0
        torch.testing.assert_close(output, expected)
        for name in records:
            torch.testing.assert_close(
                records[name], expected_records[name], atol=2e-6, rtol=2e-5
            )
        if training:
            loss = output.sum() + sum(v.square().sum() for v in records.values())
            ref_loss = expected.sum() + sum(
                v.square().sum() for v in expected_records.values()
            )
            grad = torch.autograd.grad(loss, x)[0]
            ref_grad = torch.autograd.grad(ref_loss, x_ref)[0]
            torch.testing.assert_close(grad, ref_grad, atol=3e-5, rtol=3e-4)
        else:
            # A second call must start with the exact final physical state,
            # rather than resetting because the task queues were reused.
            followup, states = run(x.flip(0))
            ref_followup, ref_states = reference(x_ref.flip(0))
            torch.testing.assert_close(followup, ref_followup)
            for name in states:
                torch.testing.assert_close(
                    states[name], ref_states[name], atol=2e-6, rtol=2e-5
                )


@pytest.mark.parametrize("change", ["weights", "topology", "method"])
def test_spike_list_cudagraph_observes_connection_updates(change):
    """Replay uses refreshed source-major values and the current topology."""
    pytest.importorskip("triton")
    native = _network(KernelOptions(neuron="triton", connection_method="push"), None)
    native.cudagraph = True
    native._cudagraph_runner.warmup = 1
    reference = _network("torch", None)
    x = torch.full((7, 2, 4), 1.1, device="cuda")
    from btorch.models.functional import reset_net_state

    with torch.no_grad(), environ.context(dt=0.25):
        for phase in range(2):
            if phase:
                for model in (native, reference):
                    connection = model.synapse.linear
                    if change == "weights":
                        connection.weight.value.add_(0.25)
                    elif change == "method":
                        if model is native:
                            model.backend = KernelOptions(
                                neuron="triton", connection_method="pull"
                            )
                    else:
                        connection.set_edges_(
                            torch.tensor([0], device="cuda"),
                            pre=torch.tensor([3], device="cuda"),
                            post=torch.tensor([2], device="cuda"),
                        )
            for model in (native, reference):
                reset_net_state(model, batch_size=2)
            actual, records = native(x)
            expected, expected_records = reference(x)
            torch.testing.assert_close(actual, expected)
            for name in records:
                torch.testing.assert_close(
                    records[name], expected_records[name], atol=2e-6, rtol=2e-5
                )
            if phase == 0:
                original_captures = native._cudagraph_runner.n_captures
        if change == "weights":
            # A value update must refresh captured storage without recapture.
            assert native._cudagraph_runner.n_captures == original_captures
        else:
            # Rewiring or switching method changes the execution contract.
            assert native._cudagraph_runner.n_captures > original_captures


def test_compiled_recurrent_method_changes_and_runtime_precedence(monkeypatch):
    """Changing explicit model options must change the actually executed
    route."""
    pytest.importorskip("triton")
    from btorch.sparse.runtime import use_method

    model = _network(KernelOptions(neuron="triton", connection_method="push"), None)
    run = torch.compile(model, backend="eager", fullgraph=True)
    launches = []
    original = spike_list._PreparedSpikeList._consume

    def observed(*args, **kwargs):
        launches.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(spike_list._PreparedSpikeList, "_consume", observed)
    x = torch.full((3, 2, 4), 1.1, device="cuda")
    with torch.no_grad(), environ.context(dt=0.25):
        run(x)
        assert launches
        launches.clear()
        model.backend = KernelOptions(neuron="triton", connection_method="pull")
        run(x.flip(0))
        assert not launches
        # Scoped overrides outrank RSNN settings but do not modify the
        # connection's own policy. Exercise eager execution under this scope.
        with use_method("push"):
            run(x)
        assert launches
        assert model.synapse.linear.method == "auto"
        launches.clear()
        run(x)
        assert not launches


@pytest.mark.parametrize(
    "method,input_dtype,backend,amp",
    [
        ("pull", torch.float32, "triton", False),
        ("push", torch.float32, "triton", False),
        ("pull", torch.bfloat16, "triton", False),
        ("pull", torch.bfloat16, "cupy", False),
        ("pull", torch.bfloat16, "tilelang", False),
        ("pull", torch.bfloat16, "triton", True),
        ("push", torch.bfloat16, "triton", True),
    ],
)
def test_recurrent_workspaces_preserve_outstanding_training_windows(
    method, input_dtype, backend, amp
):
    """Keep an earlier window's outputs and autograd tape live through the
    next."""
    pytest.importorskip("triton")
    require_cuda_backend(backend)
    # Exercise cuSPARSE's AMP boundary for pull as well as Triton spike-list
    # push. Other table entries retain the native Triton pull coverage.
    native = _network(
        KernelOptions(
            neuron=backend,
            connection="cusparse" if amp and method == "pull" else "auto",
            connection_method=method,
        ),
        None,
    )
    reference = _network("torch", None)
    first = torch.linspace(0.4, 1.2, 7 * 2 * 4, device="cuda").reshape(7, 2, 4)
    second = torch.linspace(1.1, 0.3, 5 * 2 * 4, device="cuda").reshape(5, 2, 4)
    # One window has an external synaptic input; the other omits it. Both its
    # values and cotangent must survive the absence-specialized next call.
    synaptic = torch.linspace(-0.02, 0.05, first.numel(), device="cuda").reshape_as(
        first
    )

    def strided_view(value):
        # Reverse time/batch storage order and leave gaps between neurons.
        # The nonzero offset also catches bindings that forget a view's base.
        storage = value.new_empty(
            (value.shape[1], value.shape[0], 2 * value.shape[2] + 1)
        )
        view = storage.transpose(0, 1)[..., 1::2]
        view.copy_(value)
        return view

    if input_dtype != torch.float32:
        # First-stage inputs are user-owned views; later stages use private
        # outputs. This exercises the boundary layout change in a single run.
        voltage = torch.linspace(-0.08, 0.06, 8, device="cuda").reshape(2, 4)
        current = torch.linspace(0.01, 0.03, 8, device="cuda").reshape(2, 4)
        for model in (native, reference):
            model.neuron.v = strided_view(voltage.unsqueeze(0))[0]
            model.synapse.psc = strided_view(current.unsqueeze(0))[0]

    # The low-precision edge also exercises offset, transposed and sliced
    # input views. Model states remain FP32 and both tapes must stay live.
    inputs = tuple(
        (strided_view(value.to(input_dtype)) if input_dtype != torch.float32 else value)
        .detach()
        .requires_grad_()
        for value in (first, synaptic, second)
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_() for value in inputs
    )

    def windows(model, roots):
        initial, records = model(roots[0], roots[1])
        earlier = (initial, *records.values())
        snapshot = tuple(value.detach().clone() for value in earlier)
        continued, next_records = model(roots[2])
        # A shared global workspace can silently overwrite the first window's
        # records while the second window is physically correct. Check the old
        # tensors themselves before using both windows in the same backward.
        for value, saved in zip(earlier, snapshot):
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
        outputs = (*earlier, continued, *next_records.values())
        loss = sum(
            (index + 1) * value.square().sum() for index, value in enumerate(outputs)
        )
        return outputs, loss

    # Sparse AMP keeps operand promotion for both algorithms. Backward is
    # outside autocast, with both outstanding trajectories still alive.
    with (
        environ.context(dt=0.25),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp),
    ):
        outputs, loss = windows(native, inputs)
        expected, reference_loss = windows(reference, reference_inputs)
    gradients = torch.autograd.grad(loss, inputs)
    reference_gradients = torch.autograd.grad(reference_loss, reference_inputs)
    for value, oracle in zip(outputs, expected):
        torch.testing.assert_close(value, oracle, atol=2e-6, rtol=2e-5)
    for gradient, oracle in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, oracle, atol=3e-5, rtol=3e-4)


def test_direct_spike_list_checkpoint_and_offload_preserve_training(monkeypatch):
    """Recreate private producer queues during checkpointed chunk replay."""
    pytest.importorskip("triton")
    native = _network(KernelOptions(neuron="triton", connection_method="push"), None)
    reference = _network("torch", None)
    for model in (native, reference):
        model.gradient = BPTT(sparse=True)
        model.neuron.surrogate_function = Truncated(Sigmoid(alpha=2.0), 0.4)
        model.update_state_names = ()
        model.neuron.v.requires_grad_()
        model.synapse.psc.requires_grad_()
    # Seventeen-step chunks cross the native sixteen-step physical checkpoint
    # boundary. The outer checkpoint also replays each chunk with fresh queue
    # storage, while offloaded raster outputs must retain their copy adjoint.
    native.unroll = native.chunk_size = 17
    native.grad_checkpoint = native.cpu_offload = True
    x = torch.linspace(0.4, 1.2, 35 * 2 * 4, device="cuda").reshape(35, 2, 4)
    x.requires_grad_()
    xr = x.detach().clone().requires_grad_()
    roots = (x, native.neuron.v, native.synapse.psc, *native.parameters())
    reference_roots = (
        xr,
        reference.neuron.v,
        reference.synapse.psc,
        *reference.parameters(),
    )
    with environ.context(dt=0.25):
        # Only forward is constrained here: reverse recomputation may consume
        # dense saved sources. Forward must use the producer-emitted lists.
        with monkeypatch.context() as producer_check:
            producer_check.setattr(
                kernels_triton_spikelist,
                "_pack_spike_list_kernel",
                lambda *args, **kwargs: pytest.fail("Direct producer compacted spikes"),
            )
            output, records = native(x)
        expected, expected_records = reference(xr)
        assert records == expected_records == {}
        torch.testing.assert_close(output.to("cuda"), expected)
        actual = (output.to("cuda"), native.neuron.v, native.synapse.psc)
        wanted = (expected, reference.neuron.v, reference.synapse.psc)
        for value, oracle in zip(actual, wanted):
            torch.testing.assert_close(value, oracle, atol=2e-6, rtol=2e-5)
        gradients = torch.autograd.grad(sum(v.square().sum() for v in actual), roots)
        reference_gradients = torch.autograd.grad(
            sum(v.square().sum() for v in wanted), reference_roots
        )
    for gradient, oracle in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, oracle, atol=3e-5, rtol=3e-4)


def test_cupy_push_inference_then_training_matches_torch():
    require_cuda_backend("cupy")
    # The first inference trajectory lazily allocates the source-major weight
    # cache. Reuse the same network for training after resetting physical
    # memories: cached read layouts must remain valid autograd inputs.
    from btorch.models import functional

    native = _network("cupy", 4.0)
    reference = _network("torch", None)
    for model in (native, reference):
        model.synapse.linear.weight.value.requires_grad_(True)
    inputs = torch.linspace(0.4, 1.2, 13 * 2 * 4, device="cuda").reshape(13, 2, 4)
    with environ.context(dt=0.25):
        with torch.inference_mode():
            spikes, records = native(inputs)
            expected, expected_records = reference(inputs)
            torch.testing.assert_close(spikes, expected)
            for name in records:
                torch.testing.assert_close(
                    records[name], expected_records[name], atol=2e-6, rtol=2e-5
                )
        for model in (native, reference):
            functional.init_net_state(
                model, batch_size=2, device="cuda", dtype=torch.float32
            )
        x = inputs.clone().requires_grad_()
        x_ref = inputs.clone().requires_grad_()
        spikes, records = native(x)
        expected, expected_records = reference(x_ref)
        torch.testing.assert_close(spikes, expected)
        for name in records:
            torch.testing.assert_close(
                records[name], expected_records[name], atol=2e-6, rtol=2e-5
            )
        loss = spikes.sum() + sum(value.square().sum() for value in records.values())
        ref_loss = expected.sum() + sum(
            value.square().sum() for value in expected_records.values()
        )
        gradients = torch.autograd.grad(loss, (x, native.synapse.linear.weight.value))
        expected_gradients = torch.autograd.grad(
            ref_loss, (x_ref, reference.synapse.linear.weight.value)
        )
        # Compare both input and weight VJPs, rather than only checking that
        # training no longer raises when it saves the cached execution layout.
        for actual, expected in zip(gradients, expected_gradients):
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


def test_tilelang_neuron_and_connection_training_and_graph_replay(monkeypatch):
    """Use TileLang for both components without hidden Triton propagation."""
    from btorch.sparse.runtime import (
        kernels_tilelang,
        kernels_triton,
        kernels_triton_tasks,
    )

    require_cuda_backend("tilelang")
    native = _network(
        KernelOptions(
            neuron="tilelang", connection="tilelang", connection_method="push"
        ),
        None,
    )
    reference = _network(KernelOptions(neuron="torch", connection="torch"), None)
    for model in (native, reference):
        model.neuron.v.requires_grad_()
        model.synapse.psc.requires_grad_()
    # Checkpoint/offload operate on btorch's actual recurrent state, including
    # trainable connections, rather than a second hand-written time loop.
    native.chunk_size = native.unroll = 5
    native.grad_checkpoint = native.cpu_offload = True
    x = torch.linspace(0.4, 1.2, 13 * 2 * 4, device="cuda").reshape(13, 2, 4)
    x.requires_grad_()
    xr = x.detach().clone().requires_grad_()
    roots = (x, native.neuron.v, native.synapse.psc, *native.parameters())
    ref_roots = (xr, reference.neuron.v, reference.synapse.psc, *reference.parameters())

    def forbidden(*args, **kwargs):
        raise AssertionError("All-TileLang network attempted a Triton launch")

    monkeypatch.setattr(kernels_triton, "_launch", forbidden)
    monkeypatch.setattr(kernels_triton_tasks._Execution, "launch", forbidden)
    monkeypatch.setattr(kernels_triton_spikelist, "prepare_spike_list", forbidden)
    prepare = kernels_tilelang._Execution.prepare
    forward_only = True

    def prepare_without_packing(self, name, *args, **kwargs):
        submit = prepare(self, name, *args, **kwargs)
        if name != "pack_spikes":
            return submit

        def pack(arguments):
            # Forward/inference must consume neuron-emitted queues directly.
            # Reverse recomputes the terminal PSC drive from saved dense spikes,
            # which legitimately uses TileLang packing once.
            if forward_only:
                raise AssertionError("TileLang forward submitted standalone packing")
            submit(arguments)

        return pack

    monkeypatch.setattr(kernels_tilelang._Execution, "prepare", prepare_without_packing)
    with environ.context(dt=0.25):
        output, records = native(x)
        expected, ref_records = reference(xr)
        torch.testing.assert_close(output.to("cuda"), expected)
        for name in records:
            torch.testing.assert_close(
                records[name].to("cuda"), ref_records[name], atol=2e-6, rtol=2e-5
            )
        forward_only = False
        actual = torch.autograd.grad(
            output.square().sum() + sum(v.square().sum() for v in records.values()),
            roots,
        )
        oracle = torch.autograd.grad(
            expected.square().sum()
            + sum(v.square().sum() for v in ref_records.values()),
            ref_roots,
        )
        for a, b in zip(actual, oracle):
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
        # Reuse the same public model for inference with graph replay. The
        # shorter final split remains eager, while full chunks replay.
        native.grad_checkpoint = native.cpu_offload = False
        native.cudagraph = True
        forward_only = True
        from btorch.models.functional import reset_net_state

        reset_net_state(native)
        reset_net_state(reference)
        with torch.no_grad():
            result, states = native(x)
            golden, golden_states = reference(xr)
            torch.testing.assert_close(result, golden)
            for name in states:
                torch.testing.assert_close(
                    states[name], golden_states[name], atol=2e-6, rtol=2e-5
                )


def test_tilelang_neuron_emits_ragged_lists_and_clears_receptors():
    """Retain exact int64 prefixes and tail counts across dense/silent
    reuse."""
    from btorch.models.kernels.backends.engine import forward
    from btorch.models.kernels.contracts import NeuronSpec
    from btorch.models.kernels.torch_backend import forward_step
    from btorch.sparse.runtime.spike_list import SPIKE_BLOCK, SpikeList

    require_cuda_backend("tilelang")
    batch, neurons, receptors = 2, 259, 3
    count = batch * neurons
    groups = -(-count // SPIKE_BLOCK)
    capacity = groups * SPIKE_BLOCK
    # A sample boundary lies inside a producer block and the final block is
    # partial. Large degrees expose any float conversion in the prefix scan.
    degree = torch.arange(neurons, device="cuda", dtype=torch.int64) * 123456789 + 1
    queue = SpikeList(
        torch.full((capacity,), -1, device="cuda", dtype=torch.int32),
        torch.full((capacity,), -1, device="cuda", dtype=torch.int64),
        torch.full((groups,), -1, device="cuda", dtype=torch.int32),
        degree,
        torch.empty(batch, neurons * receptors, device="cuda"),
        neurons,
    )
    x = torch.zeros(1, batch, neurons * 2, device="cuda")[..., ::2]
    state = (torch.zeros(batch, neurons, device="cuda"),)
    parameters = tuple(torch.tensor(v, device="cuda") for v in (1.0, 0.0, 1.0, 2.0))
    spec = NeuronSpec("lif", dt=1.0, hard_reset=True)
    with torch.no_grad():
        for mode in ("sparse", "dense", "silent"):
            if mode == "sparse":
                x[:, :, (0, 127, 128, 258)] = 2
            else:
                x.fill_(2 if mode == "dense" else 0)
            queue.output.fill_(float("nan"))
            output, final, records = forward(
                "tilelang", x, state, parameters, spec, ("v",), spike_list=queue
            )
            expected, expected_final, _ = forward_step(x[0], state, parameters, spec)
            torch.testing.assert_close(output[0], expected)
            torch.testing.assert_close(final[0], expected_final[0])
            torch.testing.assert_close(records[0][0], expected_final[0])
            torch.testing.assert_close(queue.output, torch.zeros_like(queue.output))
            for group in range(groups):
                start = group * SPIKE_BLOCK
                selected = expected.flatten()[start : start + SPIKE_BLOCK].nonzero()
                selected = selected.flatten() + start
                size = selected.numel()
                assert queue.counts[group].item() == size
                torch.testing.assert_close(
                    queue.indices[start : start + size].long(), selected
                )
                torch.testing.assert_close(
                    queue.cumulative[start : start + size],
                    degree[selected % neurons].cumsum(0),
                )
