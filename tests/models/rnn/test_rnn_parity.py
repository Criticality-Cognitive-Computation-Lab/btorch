import pytest
import torch
from torch import nn

from btorch.models.functional import reset_net_state
from btorch.models.rnn import make_rnn
from tests.models.rnn.rnn_utils import (
    DTYPE,
    SimpleRNNCell,
    last_step_sum,
    native_forward,
)


@pytest.mark.parametrize("T", [1, 5, 8, 10])
@pytest.mark.parametrize("unroll", [1, 4, 8, False])
def test_rnn_native_parity(T, unroll):
    """Compare make_rnn with native PyTorch unrolling."""
    torch.manual_seed(42)
    batch_size, input_size, hidden_size = 2, 3, 4

    # Create btorch RNN
    rnn = make_rnn(SimpleRNNCell, unroll=unroll)(
        input_size=input_size, hidden_size=hidden_size
    )

    # Create native RNNCell and sync weights
    native_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
    native_cell.weight_ih.data = rnn.rnn_cell.W_x.data.clone()
    native_cell.weight_hh.data = rnn.rnn_cell.W_h.data.clone()
    native_cell.bias_ih.data = rnn.rnn_cell.b.data.clone()
    native_cell.bias_hh.data.zero_()

    x = torch.randn(T, batch_size, input_size, dtype=DTYPE, requires_grad=True)
    x_native = x.detach().clone().requires_grad_(True)

    # Forward pass
    reset_net_state(rnn, batch_size=batch_size)
    out, _ = rnn(x)

    out_native = native_forward(native_cell, x_native)

    assert torch.allclose(
        out, out_native, atol=1e-6
    ), f"Output mismatch at T={T}, unroll={unroll}"

    # Backward pass
    loss = last_step_sum(out)
    loss.backward()

    loss_native = last_step_sum(out_native)
    loss_native.backward()

    assert torch.allclose(
        x.grad, x_native.grad, atol=1e-6
    ), f"Input grad mismatch at T={T}, unroll={unroll}"
    assert torch.allclose(
        rnn.rnn_cell.W_x.grad, native_cell.weight_ih.grad, atol=1e-6
    ), "W_x grad mismatch"
    assert torch.allclose(
        rnn.rnn_cell.W_h.grad, native_cell.weight_hh.grad, atol=1e-6
    ), "W_h grad mismatch"
    assert torch.allclose(
        rnn.rnn_cell.b.grad, native_cell.bias_ih.grad, atol=1e-6
    ), "Bias grad mismatch"


def test_rnn_unaligned_chunks_preserve_state_and_gradients():
    """Unaligned chunks preserve records, state and full input/weight VJPs."""
    torch.manual_seed(42)
    reference = make_rnn(SimpleRNNCell, unroll=4, chunk_size=None)(3, 4)
    actual = make_rnn(SimpleRNNCell, unroll=4, chunk_size=6, grad_checkpoint=True)(3, 4)
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(10, 2, 3, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    reset_net_state(reference, batch_size=2)
    reset_net_state(actual, batch_size=2)
    expected = reference(x)
    observed = actual(y)
    torch.testing.assert_close(observed, expected)
    # Include both the trajectory and the live final memory in the loss, so
    # checkpoint replay cannot silently drop cross-chunk state dependencies.
    (expected[0].square().sum() + reference.rnn_cell.h.square().sum()).backward()
    (observed[0].square().sum() + actual.rnn_cell.h.square().sum()).backward()
    torch.testing.assert_close(y.grad, x.grad)
    for param, oracle in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, oracle.grad)


def test_rnn_remainder_blocks():
    """Test when T is not divisible by unroll or chunk_size."""
    torch.manual_seed(42)
    T, batch_size, input_size, hidden_size = 11, 2, 3, 4
    unroll, chunk_size = 4, 8

    rnn = make_rnn(SimpleRNNCell, unroll=unroll, chunk_size=chunk_size)(
        input_size=input_size, hidden_size=hidden_size
    )

    x = torch.randn(T, batch_size, input_size, requires_grad=True)
    out, _ = rnn(x)
    assert out.shape == (T, batch_size, hidden_size)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
