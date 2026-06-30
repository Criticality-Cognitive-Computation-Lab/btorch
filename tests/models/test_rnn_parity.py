import pytest
import torch
from torch import nn

from btorch.models.functional import reset_net_state
from btorch.models.rnn import make_rnn
from tests.models.rnn_utils import DTYPE, SimpleRNNCell, last_step_sum, native_forward


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


def test_rnn_chunk_size_guard():
    """Test that chunk_size must be a multiple of unroll."""

    rnn = make_rnn(SimpleRNNCell, unroll=4, chunk_size=6)(input_size=3, hidden_size=4)
    x = torch.randn(10, 2, 3)
    with pytest.raises(ValueError, match="must be a multiple of unroll"):
        rnn(x)


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
