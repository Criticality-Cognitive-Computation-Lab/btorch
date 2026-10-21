"""Compare low-rate Triton recurrence with the actual Torch network
schedule."""

import pytest
import torch

from btorch.models import KernelOptions, environ
from btorch.models.connection import SparseConnection
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC
from btorch.sparse import Hints
from btorch.sparse.runtime import kernels_triton_spikelist


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
        backend=KernelOptions(neuron=backend) if isinstance(backend, str) else backend,
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
    executions = []
    original = kernels_triton_spikelist.spike_push_list

    def observed(*args, **kwargs):
        executions.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernels_triton_spikelist, "spike_push_list", observed)
    run = torch.compile(native, backend="eager", fullgraph=True) if compiled else native
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
    original = kernels_triton_spikelist.spike_push_list

    def observed(*args, **kwargs):
        launches.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernels_triton_spikelist, "spike_push_list", observed)
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
