import platform

import pytest
import scipy.sparse
import torch
from torch import nn

from btorch.models.base import MemoryModule
from btorch.models.functional import reset_net_state
from btorch.models.linear import DenseConn, SparseConn, available_sparse_backends
from btorch.models.rnn import make_rnn
from tests.utils.compile import compile_or_skip


class SparseConnRNNCell(MemoryModule):
    """RNN cell with a dense input projection and optional sparse
    recurrence."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        W_x: torch.Tensor,
        W_h_sparse: scipy.sparse.sparray | None,
        W_h_dense: torch.Tensor | None,
        b: torch.Tensor,
        sparse_backend: str,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = DenseConn(
            input_size,
            hidden_size,
            weight=W_x,
            bias=None,
            device=W_x.device,
            dtype=W_x.dtype,
        )
        if W_h_sparse is None and W_h_dense is None:
            raise ValueError("Provide either W_h_sparse or W_h_dense.")
        if W_h_sparse is not None and W_h_dense is not None:
            raise ValueError("Provide only one of W_h_sparse or W_h_dense.")

        if W_h_sparse is not None:
            self.W_h = SparseConn(
                W_h_sparse,
                bias=None,
                enforce_dale=False,
                sparse_backend=sparse_backend,
            )
        else:
            self.W_h = DenseConn(
                hidden_size,
                hidden_size,
                weight=W_h_dense,
                bias=None,
                device=W_h_dense.device,
                dtype=W_h_dense.dtype,
            )
        self.b = nn.Parameter(b.clone())

        self.register_memory("h", torch.zeros(1), hidden_size)
        self.init_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single step forward.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, hidden_size)
        """
        recurrent = self.W_h(self.h)
        self.h = torch.tanh(self.W_x(x) + recurrent + self.b)
        return self.h


def native_rnncell_forward(cell: nn.RNNCell, x_in: torch.Tensor) -> torch.Tensor:
    """Unroll a native RNNCell over time for a dense reference."""
    h = torch.zeros(
        x_in.shape[1], cell.hidden_size, device=x_in.device, dtype=x_in.dtype
    )
    outputs = []
    for t in range(x_in.shape[0]):
        h = cell(x_in[t], h)
        outputs.append(h)
    return torch.stack(outputs, dim=0)


@pytest.mark.parametrize("backend", available_sparse_backends())
def test_checkpointed_sparseconn_matches_eager_dense(backend: str):
    """Checkpointed sparse recurrence matches native dense math and grads."""
    torch.manual_seed(42)

    T, batch_size, input_size, hidden_size = 6, 2, 4, 5

    # Dense input projection shared by sparse and dense cells.
    W_x_dense = torch.randn(input_size, hidden_size) * 0.1

    # Sparse recurrent weights exercise SparseConn in the recurrent path.
    W_h_dense = torch.eye(hidden_size) + 0.01 * torch.randn(hidden_size, hidden_size)
    mask_h = torch.rand_like(W_h_dense) > 0.5
    mask_h.fill_diagonal_(True)
    W_h_dense = W_h_dense * mask_h
    W_h_sparse = scipy.sparse.coo_array(W_h_dense.numpy())
    b = torch.zeros(hidden_size)

    # Sparse recurrence uses gradient checkpointing; dense baseline uses
    # torch.nn.RNNCell for native dense math.
    rnn_sparse = make_rnn(SparseConnRNNCell, grad_checkpoint=True, unroll=4)(
        input_size,
        hidden_size,
        W_x_dense,
        W_h_sparse,
        None,
        b,
        sparse_backend=backend,
    )
    native_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, bias=True)
    native_cell.weight_ih.data = W_x_dense.T.clone()
    native_cell.weight_hh.data = W_h_dense.T.clone()
    native_cell.bias_ih.data = b.clone()
    native_cell.bias_hh.data.zero_()

    # Use identical inputs so sparse and dense paths are comparable.
    x_native = torch.randn(T, batch_size, input_size, requires_grad=True)
    x_sparse = x_native.clone().detach().requires_grad_(True)

    reset_net_state(rnn_sparse, batch_size=batch_size)

    out_sparse, _ = rnn_sparse(x_sparse)
    out_native = native_rnncell_forward(native_cell, x_native)

    torch.testing.assert_close(out_sparse, out_native, atol=1e-6, rtol=0.0)

    loss_sparse = out_sparse.sum()
    loss_native = out_native.sum()

    loss_sparse.backward()
    loss_native.backward()

    # Gradients should exist and align between sparse and native dense paths.
    assert x_sparse.grad is not None
    assert x_native.grad is not None
    assert rnn_sparse.rnn_cell.W_x.weight.grad is not None
    assert native_cell.weight_ih.grad is not None
    assert rnn_sparse.rnn_cell.W_h.magnitude.grad is not None
    assert native_cell.weight_hh.grad is not None

    torch.testing.assert_close(x_sparse.grad, x_native.grad, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(
        rnn_sparse.rnn_cell.W_x.weight.grad,
        native_cell.weight_ih.grad,
        atol=1e-5,
        rtol=0.0,
    )

    # Map sparse recurrent grads to dense matrix positions for comparison.
    sparse_idx = rnn_sparse.rnn_cell.W_h.indices
    rows = sparse_idx[1]
    cols = sparse_idx[0]
    dense_w_h_grad = native_cell.weight_hh.grad.T
    sparse_grad_dense = torch.zeros_like(dense_w_h_grad)
    sparse_grad_dense[rows, cols] = rnn_sparse.rnn_cell.W_h.magnitude.grad

    torch.testing.assert_close(
        sparse_grad_dense[rows, cols],
        dense_w_h_grad[rows, cols],
        atol=1e-5,
        rtol=0.0,
    )


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
@pytest.mark.parametrize("backend", available_sparse_backends())
def test_sparse_rnn_compiled_matches_eager(backend: str):
    """Compiled sparse RNN matches eager outputs and gradients."""
    torch.manual_seed(42)

    T, batch_size, input_size, hidden_size = 6, 2, 4, 5

    W_x_dense = torch.randn(input_size, hidden_size) * 0.1
    W_h_dense = torch.eye(hidden_size) + 0.01 * torch.randn(hidden_size, hidden_size)
    mask_h = torch.rand_like(W_h_dense) > 0.5
    mask_h.fill_diagonal_(True)
    W_h_dense = W_h_dense * mask_h
    W_h_sparse = scipy.sparse.coo_array(W_h_dense.numpy())
    b = torch.zeros(hidden_size)

    rnn_eager = make_rnn(SparseConnRNNCell, grad_checkpoint=True, unroll=4)(
        input_size,
        hidden_size,
        W_x_dense,
        W_h_sparse,
        None,
        b,
        sparse_backend=backend,
    )
    rnn_compiled = make_rnn(SparseConnRNNCell, grad_checkpoint=True, unroll=4)(
        input_size,
        hidden_size,
        W_x_dense,
        W_h_sparse,
        None,
        b,
        sparse_backend=backend,
    )

    compiled = compile_or_skip(rnn_compiled)

    x = torch.randn(T, batch_size, input_size, requires_grad=True)
    x_compiled = x.clone().detach().requires_grad_(True)

    reset_net_state(rnn_eager, batch_size=batch_size)
    reset_net_state(compiled, batch_size=batch_size)

    out_eager, _ = rnn_eager(x)
    out_compiled, _ = compiled(x_compiled)

    torch.testing.assert_close(out_eager, out_compiled, atol=1e-6, rtol=0.0)

    loss_eager = out_eager.sum()
    loss_compiled = out_compiled.sum()

    loss_eager.backward()
    loss_compiled.backward()

    assert x.grad is not None
    assert x_compiled.grad is not None
    assert rnn_eager.rnn_cell.W_x.weight.grad is not None
    assert compiled.rnn_cell.W_x.weight.grad is not None
    assert rnn_eager.rnn_cell.W_h.magnitude.grad is not None
    assert compiled.rnn_cell.W_h.magnitude.grad is not None

    torch.testing.assert_close(x.grad, x_compiled.grad, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(
        rnn_eager.rnn_cell.W_x.weight.grad,
        compiled.rnn_cell.W_x.weight.grad,
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rnn_eager.rnn_cell.W_h.magnitude.grad,
        compiled.rnn_cell.W_h.magnitude.grad,
        atol=1e-5,
        rtol=0.0,
    )
