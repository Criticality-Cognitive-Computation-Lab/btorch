import platform

import pytest
import torch

from btorch.models import environ
from btorch.models.dlif import DBNN, DLIF
from btorch.models.functional import init_net_state, reset_net
from btorch.models.rnn import make_rnn
from tests.utils.compile import compile_or_skip


@pytest.fixture(autouse=True)
def _dt_context():
    with environ.context(dt=1.0):
        yield


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_dlif_single_step_shape_supports_compact_and_receptor_axes(batch_shape):
    # This verifies the DLIF convenience contract:
    # n_receptor=1 accepts both (..., n_neuron) and (..., n_neuron, 1).
    cell = DLIF(n_neuron=(2, 4), n_receptor=1)
    batch_size = batch_shape if batch_shape else None

    init_net_state(cell, batch_size=batch_size, dtype=torch.float32)

    x_compact = torch.randn(*batch_shape, 2, 4)
    y_compact = cell.single_step_forward(x_compact)
    assert y_compact.shape == (*batch_shape, 2, 4)

    reset_net(cell, batch_size=batch_size)
    x_receptor = torch.randn(*batch_shape, 2, 4, 1)
    y_receptor = cell.single_step_forward(x_receptor)
    assert y_receptor.shape == (*batch_shape, 2, 4)


def test_dbnn_single_step_shape_with_receptor_axis():
    cell = DBNN(n_neuron=5, n_receptor=3)

    init_net_state(cell, batch_size=4, dtype=torch.float32)
    x = torch.randn(4, 5, 3)
    y = cell.single_step_forward(x)

    assert y.shape == (4, 5)


def test_dbnn_reset_restores_deterministic_rollout():
    torch.manual_seed(11)
    T, B, N, R = 8, 2, 4, 3
    cell = DBNN(n_neuron=N, n_receptor=R)
    x_seq = torch.randn(T, B, N, R)

    init_net_state(cell, batch_size=B, dtype=x_seq.dtype)
    first = torch.stack([cell.single_step_forward(x_seq[t]) for t in range(T)], dim=0)

    # Without reset, hidden states continue and outputs should differ.
    continued = torch.stack(
        [cell.single_step_forward(x_seq[t]) for t in range(T)],
        dim=0,
    )
    assert not torch.allclose(first, continued)

    # After reset, the same input should produce the same rollout.
    reset_net(cell, batch_size=B)
    repeated = torch.stack(
        [cell.single_step_forward(x_seq[t]) for t in range(T)], dim=0
    )
    torch.testing.assert_close(first, repeated, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "cell_factory",
    [
        lambda: DLIF(n_neuron=6, n_receptor=3),
        lambda: DBNN(n_neuron=6, n_receptor=3),
    ],
)
def test_make_rnn_parity_with_manual_unroll(cell_factory):
    torch.manual_seed(21)
    T, B, N, R = 10, 2, 6, 3
    x_seq = torch.randn(T, B, N, R)

    # Use two copies to compare manual single-step unroll vs make_rnn unroll.
    cell_manual = cell_factory()
    cell_rnn = cell_factory()
    cell_rnn.load_state_dict(cell_manual.state_dict())

    init_net_state(cell_manual, batch_size=B, dtype=x_seq.dtype)
    manual = torch.stack(
        [cell_manual.single_step_forward(x_seq[t]) for t in range(T)],
        dim=0,
    )

    rnn = make_rnn(cell_rnn, unroll=4)
    init_net_state(rnn, batch_size=B, dtype=x_seq.dtype)
    wrapped, states = rnn(x_seq)

    torch.testing.assert_close(manual, wrapped, atol=1e-6, rtol=0.0)
    assert wrapped.shape == (T, B, N)
    assert "soma.v" in states


def test_dbnn_gradients_flow_through_input_soma_and_synapse():
    torch.manual_seed(7)
    T, B, N, R = 6, 2, 4, 3
    x_seq = torch.randn(T, B, N, R, requires_grad=True)

    # Make soma parameters trainable so the test can verify soma gradients.
    cell = DBNN(
        n_neuron=N,
        n_receptor=R,
        soma_kwargs={"trainable_param": {"tau", "v_threshold"}},
    )
    rnn = make_rnn(cell, unroll=3)

    init_net_state(rnn, batch_size=B, dtype=x_seq.dtype)
    out, _ = rnn(x_seq)
    out.sum().backward()

    assert x_seq.grad is not None
    assert cell.soma.tau.grad is not None
    assert cell.soma.v_threshold.grad is not None
    assert cell.synapse_module is not None
    assert cell.synapse_module.linear.weight.grad is not None


@pytest.mark.skipif(
    platform.system() != "Linux", reason="torch.compile only fully supported on Linux"
)
def test_dbnn_make_rnn_compile_parity():
    torch.manual_seed(33)
    T, B, N, R = 12, 2, 5, 2

    eager_cell = DBNN(n_neuron=N, n_receptor=R)
    compiled_cell = DBNN(n_neuron=N, n_receptor=R)
    compiled_cell.load_state_dict(eager_cell.state_dict())

    eager = make_rnn(eager_cell, unroll=4)
    compiled_base = make_rnn(compiled_cell, unroll=4)
    compiled = compile_or_skip(compiled_base)

    x_eager = torch.randn(T, B, N, R, requires_grad=True)
    x_compiled = x_eager.detach().clone().requires_grad_(True)

    init_net_state(eager, batch_size=B, dtype=x_eager.dtype)
    out_eager, _ = eager(x_eager)

    init_net_state(compiled, batch_size=B, dtype=x_compiled.dtype)
    out_compiled, _ = compiled(x_compiled)

    torch.testing.assert_close(out_eager, out_compiled, atol=1e-5, rtol=1e-5)

    out_eager.sum().backward()
    out_compiled.sum().backward()

    torch.testing.assert_close(x_eager.grad, x_compiled.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        eager.rnn_cell.synapse_module.linear.weight.grad,
        compiled_base.rnn_cell.synapse_module.linear.weight.grad,
        atol=1e-5,
        rtol=1e-5,
    )
