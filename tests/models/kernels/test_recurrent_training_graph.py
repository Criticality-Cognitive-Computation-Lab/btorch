"""Preserve complete recurrent adjoints across reusable CUDA Graph chunks."""

import copy

import pytest
import torch

from btorch.models import environ, functional
from examples.rsnn_common import ExampleOptions, initialize, stimulus
from examples.rsnn_izhikevich import build_model
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_izhikevich_graph_bptt_keeps_each_chunk_tape(backend):
    """A loss after two replayed chunks must reach the original input/state.

    Compare all physical memories, input and trainable
    weights/parameters against independent Torch equations. A second
    update and fresh input expose stale transposed weights, overwritten
    tapes and capture-time cotangents. The short tail executes eagerly;
    warmup/capture never runs the sequence.
    """
    _check_izhikevich_graph_bptt(backend)


def test_izhikevich_graph_bptt_checkpoint_offload():
    """CPU tape restoration and checkpoint replay preserve chunk adjoints."""
    _check_izhikevich_graph_bptt("triton", checkpoint=True, offload=True)


def _check_izhikevich_graph_bptt(backend, *, checkpoint=False, offload=False):
    require_cuda_backend(backend)
    torch.manual_seed(42)
    model = build_model(
        ExampleOptions(
            neurons=32,
            density=0.125,
            device="cuda",
            neuron_backend=backend,
            sparse_backend="tilelang" if backend == "tilelang" else "cusparse",
            method="pull",
            cudagraph=True,
            chunk_size=8,
        )
    )
    initialize(model, "cuda")
    reference = copy.deepcopy(model)
    reference.backend = "torch"
    reference.cudagraph = False
    reference.chunk_size = None
    model.grad_checkpoint = checkpoint
    model.cpu_offload = offload
    initial = {
        k: v.detach().clone()
        for k, v in functional.named_hidden_states(model).items()
        if k != "synapse.psc"
    }
    # Include an initial volley so recurrent-weight adjoints are exercised even
    # in this short, deliberately bounded test, rather than comparing zeros.
    initial["neuron.v"].copy_(
        torch.linspace(-43.0, 25.0, 32, device="cuda").reshape(1, 32)
    )
    for trial in range(2):
        incoming = {k: v.clone().requires_grad_() for k, v in initial.items()}
        expected_incoming = {k: v.clone().requires_grad_() for k, v in initial.items()}
        functional.set_hidden_states(model, incoming)
        functional.set_hidden_states(reference, expected_incoming)
        # The wrapper PSC is the sum of receptor memories, not a second
        # independent physical current. Preserve that dependency in the
        # Python oracle so the first-step current adjoint reaches base_psc.
        for network in (model, reference):
            network.synapse.psc = network.synapse.base_psc.psc.unflatten(
                -1, (32, network.synapse.n_receptor)
            ).sum(-1)
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        # Sliced storage also verifies that graph staging does not reinterpret
        # a strided input as contiguous memory.
        storage = stimulus("izhikevich", 19, 64, "cuda").clone() * (
            0.664 + trial * 0.001
        )
        x = storage[..., ::2].detach().requires_grad_()
        xr = x.detach().clone().requires_grad_()
        with environ.context(dt=1.0):
            spikes, records = model(x)
            expected_spikes, expected_records = reference(xr)
            # Loss only on the final tail forces state adjoints to cross both
            # preceding chunks. Physical traces exercise record cotangents.
            loss = spikes[-3:].sum()
            expected_loss = expected_spikes[-3:].sum()
            for name, value in records.items():
                torch.testing.assert_close(
                    value.cpu(), expected_records[name].cpu(), atol=3e-5, rtol=3e-5
                )
                loss = loss + value[-3:].square().mean() * 1e-5
                expected_loss = (
                    expected_loss + expected_records[name][-3:].square().mean() * 1e-5
                )
            loss.backward()
            expected_loss.backward()
        torch.testing.assert_close(spikes.cpu(), expected_spikes.cpu())
        assert spikes.sum() > 0
        torch.testing.assert_close(x.grad, xr.grad, atol=2e-5, rtol=2e-4)
        assert x.grad[:8].abs().sum() > 0
        for name, value in incoming.items():
            expected = expected_incoming[name].grad
            if expected is not None:
                torch.testing.assert_close(value.grad, expected, atol=2e-5, rtol=2e-4)
        for (name, p), (_, q) in zip(
            model.named_parameters(), reference.named_parameters()
        ):
            if q.grad is not None:
                torch.testing.assert_close(
                    p.grad,
                    q.grad,
                    atol=2e-5,
                    rtol=2e-4,
                    msg=lambda msg: f"{name}: {msg}",
                )
                with torch.no_grad():
                    p.add_(p.grad, alpha=-1e-5)
                    q.add_(q.grad, alpha=-1e-5)
        graphs = model._training_graphs
        assert all(stage.inputs[0].shape[0] == 8 for stage in graphs.stages.values())
        if trial == 0:
            captures = graphs.n_captures
        else:
            # Changed data/optimizer values reuse graphs; the short tail never
            # creates a separate capture or a full-sequence warmup.
            assert graphs.n_captures == captures


def test_offloaded_trace_mutation_does_not_silently_change_bptt():
    """Sharing a CPU recording with a saved tape must retain mutation
    checks."""
    require_cuda_backend("triton")
    network = build_model(
        ExampleOptions(
            neurons=16,
            density=0.25,
            device="cuda",
            neuron_backend="triton",
            sparse_backend="triton",
            method="pull",
            cudagraph=True,
            chunk_size=8,
        )
    )
    initialize(network, "cuda")
    network.cpu_offload = True
    x = stimulus("izhikevich", 8, 16, "cuda").clone().requires_grad_()
    with environ.context(dt=1.0):
        spikes, records = network(x)
        # A single chunk returns its CPU recording directly. It shares the
        # numerical snapshot with backward, but never drops its transfer VJP.
        with torch.no_grad():
            records["neuron.v"].add_(1.0)
        with pytest.raises(RuntimeError, match="modified|inplace|view"):
            spikes.sum().backward()


def test_dense_graph_bptt_refreshes_autocast_weight_copies():
    """AMP cast allocations are values, not fixed graph parameter addresses."""
    from tests.models.kernels.test_recurrent_dual_exponential import _build

    require_cuda_backend("triton")
    model = _build("dense", "triton")
    reference = copy.deepcopy(model)
    model.cudagraph, reference.cudagraph = True, False
    model.chunk_size = reference.chunk_size = 8
    initial = {
        name: value.detach().clone()
        for name, value in functional.named_hidden_states(model).items()
    }
    # Re-enter BF16 after an FP32 run and a weight update. Replaying a capture
    # bound to the first transient AMP cast would give stale values/gradients.
    for amp in (True, False, True):
        for network in (model, reference):
            functional.set_hidden_states(
                network, {k: v.clone().requires_grad_() for k, v in initial.items()}
            )
            network.zero_grad(set_to_none=True)
        x = torch.full((8, 2, 4), 0.55, device="cuda", requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        with (
            environ.context(dt=0.5),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp),
        ):
            output, records = model(x)
            expected, expected_records = reference(xr)
            loss = output.sum() + sum(v.square().mean() for v in records.values())
            expected_loss = expected.sum() + sum(
                v.square().mean() for v in expected_records.values()
            )
        loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(x.grad, xr.grad)
        for p, q in zip(model.parameters(), reference.parameters()):
            if q.grad is not None:
                torch.testing.assert_close(p.grad, q.grad)
                with torch.no_grad():
                    p.add_(p.grad, alpha=-1e-4)
                    q.add_(q.grad, alpha=-1e-4)
