import platform

import pytest
import torch

from btorch.models.functional import reset_net_state
from btorch.models.rnn import make_rnn
from tests.models.rnn_utils import SimpleRNNCell


# Only run on Linux which supports torch.compile
pytestmark = pytest.mark.skipif(
    platform.system() != "Linux", reason="torch.compile only fully supported on Linux"
)


def test_rnn_compile_basic_parity():
    """Verify torch.compile produces identical results to eager mode."""
    torch.manual_seed(42)
    T, batch_size, input_size, hidden_size = 20, 2, 4, 8

    rnn_eager = make_rnn(SimpleRNNCell, unroll=4)(
        input_size=input_size, hidden_size=hidden_size
    )

    rnn_compiled = make_rnn(SimpleRNNCell, unroll=4)(
        input_size=input_size, hidden_size=hidden_size
    )
    rnn_compiled.rnn_cell.load_state_dict(rnn_eager.rnn_cell.state_dict())

    compiled_model = torch.compile(rnn_compiled)

    x = torch.randn(T, batch_size, input_size, requires_grad=True)

    # Forward parity
    reset_net_state(rnn_eager, batch_size=batch_size)
    out_eager, _ = rnn_eager(x)

    reset_net_state(rnn_compiled, batch_size=batch_size)
    out_compiled, _ = compiled_model(x)

    assert torch.allclose(out_eager, out_compiled, atol=1e-5)

    # Backward parity
    out_eager.sum().backward()
    out_compiled.sum().backward()

    assert torch.allclose(
        rnn_eager.rnn_cell.W_x.grad, rnn_compiled.rnn_cell.W_x.grad, atol=1e-5
    )


def test_rnn_compile_with_chunking():
    """Verify torch.compile works correctly with chunking and offloading."""
    torch.manual_seed(42)
    T, batch_size, input_size, hidden_size = 32, 2, 4, 8

    # Chunking + Offloading + Compile
    rnn = make_rnn(
        SimpleRNNCell, unroll=8, chunk_size=16, cpu_offload=True, grad_checkpoint=True
    )(input_size=input_size, hidden_size=hidden_size)

    compiled_model = torch.compile(rnn)

    x = torch.randn(T, batch_size, input_size, requires_grad=True)

    reset_net_state(rnn, batch_size=batch_size)
    out, _ = compiled_model(x)

    assert out.shape == (T, batch_size, hidden_size)
    assert out.device == torch.device("cpu")

    out.sum().backward()
    assert rnn.rnn_cell.W_x.grad is not None


def test_rnn_compile_captures():
    """Ensure unrolled chunks trigger only a few graph captures."""
    torch._dynamo.reset()

    T = 42
    unroll_size = 10
    batch_size, input_size, hidden_size = 1, 4, 8

    rnn = make_rnn(SimpleRNNCell, unroll=unroll_size)(
        input_size=input_size, hidden_size=hidden_size
    )

    graph_count = 0

    def count_backend(gm, example_inputs):
        nonlocal graph_count
        graph_count += 1
        return gm.forward

    compiled_model = torch.compile(rnn, backend=count_backend)

    x = torch.randn(T, batch_size, input_size)
    out, _ = compiled_model(x)

    # We expect:
    # 1. Outer Python loop (might trigger some captures depending on dynamo)
    # 2. Inner chunk of size 10 (compiled once)
    # 3. Inner chunk of size 2 (compiled once, remainder)
    # Total captures should be low (~3)
    assert 1 <= graph_count <= 12, f"Unexpected number of captures: {graph_count}"
