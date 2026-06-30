import itertools

import pytest
import torch

from btorch.models.functional import reset_net_state
from btorch.models.rnn import make_rnn
from tests.models.rnn_utils import DTYPE, SimpleRNNCell, last_step_sum


# Define parameter ranges for exhaustive testing
PARAMS = {
    "unroll": [1, 4, False],
    "chunk_size": [None, 4, 8],
    "grad_checkpoint": [True, False],
    "cpu_offload": [True, False],
    "save_grad_history": [True, False],
}


def _generate_configs():
    keys = list(PARAMS.keys())
    values = list(PARAMS.values())
    configs = []

    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))

        # Validations and Filters
        # Filter invalid combinations that would raise errors or are redundant

        # chunk_size must be multiple of unroll if both present and not None/False
        u = config["unroll"]
        c = config["chunk_size"]
        if u is not False and c is not None:
            if c % u != 0:
                continue

        configs.append(config)
    return configs


CONFIGS = _generate_configs()


@pytest.mark.parametrize("config", CONFIGS)
def test_rnn_exhaustive_ops(config):
    """Exhaustive test for all valid configuration combinations."""
    torch.manual_seed(42)
    T, batch_size, input_size, hidden_size = 16, 2, 3, 4
    # Baseline: Simple unroll=1 (pure python loop equivalent mainly) or
    # unroll=T (pure compiled). Let's use unroll=1 as robust baseline.
    rnn_base = make_rnn(
        SimpleRNNCell, unroll=1, save_grad_history=True, grad_state_names=["h"]
    )(input_size=input_size, hidden_size=hidden_size)

    # Test RNN
    # Inject grad_state_names if saving history
    kwargs = config.copy()
    if kwargs["save_grad_history"]:
        kwargs["grad_state_names"] = ["h"]

    rnn_test = make_rnn(SimpleRNNCell, **kwargs)(
        input_size=input_size, hidden_size=hidden_size
    )

    # Sync weights
    with torch.no_grad():
        rnn_test.rnn_cell.W_x.copy_(rnn_base.rnn_cell.W_x)
        rnn_test.rnn_cell.W_h.copy_(rnn_base.rnn_cell.W_h)
        rnn_test.rnn_cell.b.copy_(rnn_base.rnn_cell.b)

    x_base = torch.randn(T, batch_size, input_size, dtype=DTYPE, requires_grad=True)
    x_test = x_base.detach().clone().requires_grad_(True)

    # Forward Pass
    reset_net_state(rnn_base, batch_size=batch_size)
    out_base, _ = rnn_base(x_base)

    reset_net_state(rnn_test, batch_size=batch_size)
    out_test, _ = rnn_test(x_test)

    # Move to CPU for comparison if offloaded
    if out_test.device != out_base.device:
        out_test_cmp = out_test.cpu()
        out_base_cmp = out_base.cpu()
    else:
        out_test_cmp = out_test
        out_base_cmp = out_base

    assert torch.allclose(
        out_base_cmp, out_test_cmp, atol=1e-5
    ), f"Output mismatch with config {config}"

    # Backward Pass
    loss_base = last_step_sum(out_base)
    loss_base.backward()

    loss_test = last_step_sum(out_test)
    loss_test.backward()

    assert torch.allclose(
        x_base.grad, x_test.grad, atol=1e-5
    ), f"Input grad mismatch with config {config}"

    # Check Weight Grads
    assert torch.allclose(
        rnn_base.rnn_cell.W_x.grad, rnn_test.rnn_cell.W_x.grad, atol=1e-5
    ), f"W_x grad mismatch with config {config}"

    # Check Grad History correctness
    if config["save_grad_history"]:
        hist_base = rnn_base.get_grad_history()["h"]
        hist_test = rnn_test.get_grad_history()["h"]

        # Ensure all time steps are captured
        assert len(hist_test) == T
        assert not any(
            h is None for h in hist_test
        ), f"Missing gradients in history for config {config}"

        # Verify values against baseline
        for t in range(T):
            g_base = hist_base[t]
            g_test = hist_test[t]
            if g_test.device != g_base.device:
                g_test = g_test.cpu()
                g_base = g_base.cpu()

            assert torch.allclose(
                g_base, g_test, atol=1e-5
            ), f"Grad history mismatch at t={t} for config {config}"


def test_rnn_edge_cases():
    """Test edge cases: T not divisible, T smaller than chunk."""
    torch.manual_seed(123)
    batch_size, input_size, hidden_size = 2, 3, 4

    # Case 1: T < unroll
    T = 3
    rnn = make_rnn(SimpleRNNCell, unroll=4)(input_size, hidden_size)
    x = torch.randn(T, batch_size, input_size)
    out, _ = rnn(x)
    assert out.shape == (T, batch_size, hidden_size)

    # Case 2: T < chunk_size, unroll divides T
    T = 8
    rnn = make_rnn(SimpleRNNCell, unroll=2, chunk_size=10)(input_size, hidden_size)
    x = torch.randn(T, batch_size, input_size)
    out, _ = rnn(x)
    assert out.shape == (T, batch_size, hidden_size)

    # Case 3: T remainder with chunking
    T = 10
    rnn = make_rnn(SimpleRNNCell, unroll=4, chunk_size=8)(input_size, hidden_size)
    # chunks: [0..8], [8..10]
    # chunk 1: 8 steps (2 unrolls of 4)
    # chunk 2: 2 steps (remainder, unroll=4 -> clipped to 2)
    x = torch.randn(T, batch_size, input_size)
    out, _ = rnn(x)
    assert out.shape == (T, batch_size, hidden_size)
