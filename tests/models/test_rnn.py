import platform

import pytest
import torch
from torch import nn

from btorch.models.base import MemoryModule
from btorch.models.functional import reset_net_state
from btorch.models.rnn import make_rnn
from tests.utils.compile import compile_or_skip


class SimpleRNNCell(MemoryModule):
    """Simple RNN cell: h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.W_h = nn.Parameter(
            torch.eye(hidden_size) + 0.02 * torch.diag(torch.rand(hidden_size)) - 0.01
        )
        self.b = nn.Parameter(torch.zeros(hidden_size))

        self.register_memory("h", torch.zeros(1), hidden_size)
        self.init_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single step forward.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, hidden_size)
        """
        self.h = torch.tanh(x @ self.W_x.t() + self.h @ self.W_h.t() + self.b)
        return self.h


def last_step_sum(out: torch.Tensor) -> torch.Tensor:
    """Loss helper: sum the final timestep output."""
    return out[-1].sum()


def native_forward(cell: nn.RNNCell, x_in: torch.Tensor) -> torch.Tensor:
    """Unroll a native RNNCell over time."""
    h = torch.zeros(
        x_in.shape[1], cell.hidden_size, device=x_in.device, dtype=x_in.dtype
    )
    outputs = []
    for t in range(x_in.shape[0]):
        h = cell(x_in[t], h)
        outputs.append(h)
    return torch.stack(outputs, dim=0)


class TestGradientCorrectness:
    """Test suite for gradient correctness in RNN implementations."""

    @pytest.fixture
    def rnn_cell(self):
        """Create a simple RNN cell for testing."""
        return SimpleRNNCell(input_size=4, hidden_size=8)

    @pytest.fixture
    def rnn_no_checkpoint(self, rnn_cell):
        """Create RNN wrapper without gradient checkpointing."""
        RNNClass = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)
        return RNNClass(input_size=4, hidden_size=8)

    @pytest.fixture
    def rnn_with_checkpoint(self, rnn_cell):
        """Create RNN wrapper with gradient checkpointing."""
        RNNClass = make_rnn(SimpleRNNCell, grad_checkpoint=True, unroll=4)
        return RNNClass(input_size=4, hidden_size=8)

    def test_single_step_gradients(self, rnn_no_checkpoint):
        """Test that gradients flow correctly through single step."""
        torch.manual_seed(42)

        # Create input
        x = torch.randn(2, 4, requires_grad=True)  # (batch, input_size)
        reset_net_state(rnn_no_checkpoint, batch_size=2)

        # Forward pass
        out, states = rnn_no_checkpoint.single_step_forward(x)

        # Compute loss and backward
        loss = out.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None, "Input should have gradients"
        assert (
            rnn_no_checkpoint.rnn_cell.W_x.grad is not None
        ), "W_x should have gradients"
        assert (
            rnn_no_checkpoint.rnn_cell.W_h.grad is not None
        ), "W_h should have gradients"
        assert rnn_no_checkpoint.rnn_cell.b.grad is not None, "b should have gradients"

        # Check that gradients are non-zero
        assert torch.abs(x.grad).sum() > 0, "Input gradients should be non-zero"
        assert torch.abs(rnn_no_checkpoint.rnn_cell.W_x.grad).sum() > 0

    def test_long_sequence_gradient_flow(self):
        """Test that gradients flow correctly through long sequences (20+
        steps)."""
        torch.manual_seed(42)

        rnn = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)(
            input_size=4, hidden_size=8
        )

        # Long sequence - 25 timesteps
        T = 100
        x = torch.randn(T, 2, 4, requires_grad=True)

        reset_net_state(rnn, batch_size=2)
        out, _ = rnn.multi_step_forward(x)
        loss = last_step_sum(out)
        loss.backward()

        # Check early timesteps have gradients (BPTT through time)
        early_grad_norm = x.grad[:5].norm().item()
        middle_grad_norm = x.grad[10:15].norm().item()
        late_grad_norm = x.grad[-5:].norm().item()

        assert early_grad_norm > 0, "Early timesteps should have gradients"
        assert middle_grad_norm > 0, "Middle timesteps should have gradients"
        assert late_grad_norm > 0, "Late timesteps should have gradients"

        # Early timesteps should have smaller gradients (decayed through time)
        # but not vanished completely
        assert (
            early_grad_norm < late_grad_norm * 100
        ), "Early gradients shouldn't be too much smaller (gradient vanishing)"

    def test_gradients_match_native_rnncell(self):
        """Compare gradients against a native PyTorch RNNCell
        implementation."""
        torch.manual_seed(42)

        rnn = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)(
            input_size=3, hidden_size=4
        )

        native_cell = nn.RNNCell(input_size=3, hidden_size=4, bias=True)
        native_cell.weight_ih.data = rnn.rnn_cell.W_x.data.clone()
        native_cell.weight_hh.data = rnn.rnn_cell.W_h.data.clone()
        native_cell.bias_ih.data = rnn.rnn_cell.b.data.clone()
        native_cell.bias_hh.data.zero_()

        T, batch_size, input_size = 5, 2, 3
        x = torch.randn(T, batch_size, input_size, requires_grad=True)
        x_native = x.detach().clone().requires_grad_(True)

        reset_net_state(rnn, batch_size=batch_size)
        out, _ = rnn.multi_step_forward(x)
        loss = last_step_sum(out)
        loss.backward()

        out_native = native_forward(native_cell, x_native)
        loss_native = last_step_sum(out_native)
        loss_native.backward()

        assert torch.allclose(out, out_native, atol=1e-6), "Outputs should match"
        assert torch.allclose(
            x.grad, x_native.grad, atol=1e-6
        ), "Input gradients should match"
        assert torch.allclose(
            rnn.rnn_cell.W_x.grad, native_cell.weight_ih.grad, atol=1e-6
        ), "W_x gradients should match"
        assert torch.allclose(
            rnn.rnn_cell.W_h.grad, native_cell.weight_hh.grad, atol=1e-6
        ), "W_h gradients should match"
        assert torch.allclose(
            rnn.rnn_cell.b.grad, native_cell.bias_ih.grad, atol=1e-6
        ), "Bias gradients should match"

    def test_multi_step_gradients_no_checkpoint(self, rnn_no_checkpoint):
        """Test gradients through multi-step forward without checkpointing."""
        torch.manual_seed(42)

        T, batch_size, input_size = 10, 2, 4
        x = torch.randn(T, batch_size, input_size, requires_grad=True)

        reset_net_state(rnn_no_checkpoint, batch_size=batch_size)
        # Forward pass
        out, states = rnn_no_checkpoint.multi_step_forward(x)

        # Compute loss
        loss = last_step_sum(out)
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input should have gradients"
        assert rnn_no_checkpoint.rnn_cell.W_x.grad is not None
        assert rnn_no_checkpoint.rnn_cell.W_h.grad is not None
        assert rnn_no_checkpoint.rnn_cell.b.grad is not None

        # Check non-zero gradients
        assert torch.abs(x.grad).sum() > 0, "Input gradients should be non-zero"
        assert torch.abs(rnn_no_checkpoint.rnn_cell.W_x.grad).sum() > 0

    def test_multi_step_gradients_with_checkpoint(self, rnn_with_checkpoint):
        """Test gradients through multi-step forward with checkpointing."""
        torch.manual_seed(42)

        T, batch_size, input_size = 10, 2, 4
        x = torch.randn(T, batch_size, input_size, requires_grad=True)
        reset_net_state(rnn_with_checkpoint, batch_size=batch_size)
        # Forward pass
        out, states = rnn_with_checkpoint.multi_step_forward(x)

        # Compute loss
        loss = last_step_sum(out)
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input should have gradients with checkpointing"
        assert rnn_with_checkpoint.rnn_cell.W_x.grad is not None
        assert rnn_with_checkpoint.rnn_cell.W_h.grad is not None
        assert rnn_with_checkpoint.rnn_cell.b.grad is not None

        # Check non-zero gradients
        assert torch.abs(x.grad).sum() > 0
        assert torch.abs(rnn_with_checkpoint.rnn_cell.W_x.grad).sum() > 0

    def test_gradient_equivalence_checkpoint_vs_no_checkpoint(self):
        """Test that checkpointed and non-checkpointed versions give same
        gradients."""
        torch.manual_seed(42)

        T, batch_size, input_size, hidden_size = 10, 2, 4, 8

        # Create two RNNs with same initialization
        rnn_no_chkpt = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)(
            input_size, hidden_size
        )
        rnn_chkpt = make_rnn(SimpleRNNCell, grad_checkpoint=True, unroll=4)(
            input_size, hidden_size
        )

        # Copy weights to ensure identical initialization
        rnn_chkpt.rnn_cell.W_x.data = rnn_no_chkpt.rnn_cell.W_x.data.clone()
        rnn_chkpt.rnn_cell.W_h.data = rnn_no_chkpt.rnn_cell.W_h.data.clone()
        rnn_chkpt.rnn_cell.b.data = rnn_no_chkpt.rnn_cell.b.data.clone()

        # Same input
        x = torch.randn(T, batch_size, input_size, requires_grad=True)
        x_chkpt = x.clone().detach().requires_grad_(True)

        # Forward passes
        reset_net_state(rnn_no_chkpt, batch_size=batch_size)
        out_no_chkpt, _ = rnn_no_chkpt.multi_step_forward(x)
        reset_net_state(rnn_chkpt, batch_size=batch_size)
        out_chkpt, _ = rnn_chkpt.multi_step_forward(x_chkpt)

        # Check outputs are identical
        assert torch.allclose(
            out_no_chkpt, out_chkpt, atol=1e-6
        ), "Outputs should be identical"

        # Backward passes
        loss_no_chkpt = out_no_chkpt.sum()
        loss_chkpt = out_chkpt.sum()

        loss_no_chkpt.backward()
        loss_chkpt.backward()

        # Check gradients are close (may have small numerical differences)
        assert torch.allclose(
            x.grad, x_chkpt.grad, atol=1e-5
        ), "Input gradients should match"
        assert torch.allclose(
            rnn_no_chkpt.rnn_cell.W_x.grad, rnn_chkpt.rnn_cell.W_x.grad, atol=1e-5
        ), "W_x gradients should match"
        assert torch.allclose(
            rnn_no_chkpt.rnn_cell.W_h.grad, rnn_chkpt.rnn_cell.W_h.grad, atol=1e-5
        ), "W_h gradients should match"

    @pytest.mark.skipif(
        not platform.system() == "Linux", reason="Only Linux supports torch.compile"
    )
    def test_compiled_checkpoint_vs_eager_checkpoint(self):
        """Test compiled checkpointed matches eager checkpointed."""
        torch.manual_seed(42)

        T, batch_size, input_size, hidden_size = 10, 2, 4, 8

        rnn_eager = make_rnn(SimpleRNNCell, grad_checkpoint=True, unroll=4)(
            input_size, hidden_size
        )
        rnn_compiled = make_rnn(SimpleRNNCell, grad_checkpoint=True, unroll=4)(
            input_size, hidden_size
        )

        rnn_compiled.rnn_cell.W_x.data = rnn_eager.rnn_cell.W_x.data.clone()
        rnn_compiled.rnn_cell.W_h.data = rnn_eager.rnn_cell.W_h.data.clone()
        rnn_compiled.rnn_cell.b.data = rnn_eager.rnn_cell.b.data.clone()

        compiled = compile_or_skip(rnn_compiled)

        x = torch.randn(T, batch_size, input_size, requires_grad=True)
        x_compiled = x.clone().detach().requires_grad_(True)

        reset_net_state(rnn_eager, batch_size=batch_size)
        reset_net_state(compiled, batch_size=batch_size)

        out_eager, _ = rnn_eager.multi_step_forward(x)
        out_compiled, _ = compiled.multi_step_forward(x_compiled)

        assert torch.allclose(
            out_eager, out_compiled, atol=1e-6
        ), "Outputs should be identical"

        loss_eager = out_eager.sum()
        loss_compiled = out_compiled.sum()

        loss_eager.backward()
        loss_compiled.backward()

        assert torch.allclose(
            x.grad, x_compiled.grad, atol=1e-5
        ), "Input gradients should match"
        assert torch.allclose(
            rnn_eager.rnn_cell.W_x.grad, compiled.rnn_cell.W_x.grad, atol=1e-5
        ), "W_x gradients should match"
        assert torch.allclose(
            rnn_eager.rnn_cell.W_h.grad, compiled.rnn_cell.W_h.grad, atol=1e-5
        ), "W_h gradients should match"

    def test_gradient_across_unroll_boundaries(self):
        """Test that gradients flow correctly across unroll boundaries."""
        torch.manual_seed(42)

        # Create RNN with unroll=4
        rnn = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)(
            input_size=4, hidden_size=8
        )

        # Use T=9 to have full blocks (0-3, 4-7) and remainder (8)
        T, batch_size = 9, 2
        x = torch.randn(T, batch_size, 4, requires_grad=True)

        # Forward and backward
        reset_net_state(rnn, batch_size=batch_size)
        out, _ = rnn.multi_step_forward(x)
        loss = out.sum()
        loss.backward()

        # Check that early timesteps have gradients (they influence later steps)
        early_grad = x.grad[0].abs().sum()
        late_grad = x.grad[-1].abs().sum()

        assert early_grad > 0, "Early timesteps should have gradients"
        assert late_grad > 0, "Late timesteps should have gradients"

    def test_gradient_magnitude_scaling(self):
        """Test that gradients don't explode or vanish across many
        timesteps."""
        torch.manual_seed(42)

        rnn = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)(
            input_size=4, hidden_size=8
        )

        T, batch_size = 20, 2
        x = torch.randn(T, batch_size, 4, requires_grad=True)

        reset_net_state(rnn, batch_size=batch_size)
        out, _ = rnn.multi_step_forward(x)
        loss = last_step_sum(out)
        loss.backward()

        # Check gradient magnitudes across time
        grad_norms = torch.stack([x.grad[t].norm() for t in range(T)])

        # Gradients should exist throughout
        assert (grad_norms > 0).all(), "All timesteps should have non-zero gradients"

        # Ratio of max to min shouldn't be too extreme (allowing for some variation)
        ratio = grad_norms.max() / (grad_norms.min() + 1e-8)
        assert ratio < 1e4, f"Gradient magnitude ratio too large: {ratio}"

    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly over multiple backward
        passes."""
        torch.manual_seed(42)

        rnn = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)(
            input_size=4, hidden_size=8
        )

        T, batch_size = 5, 2

        # First backward pass
        x1 = torch.randn(T, batch_size, 4, requires_grad=True)
        reset_net_state(rnn, batch_size=batch_size)
        out1, _ = rnn.multi_step_forward(x1)
        loss1 = out1[-1].sum()
        loss1.backward()

        grad_W_x_first = rnn.rnn_cell.W_x.grad.clone()

        # Second backward pass (accumulate)
        x2 = torch.randn(T, batch_size, 4, requires_grad=True)
        rnn.rnn_cell.h = rnn.rnn_cell.h.detach()
        out2, _ = rnn.multi_step_forward(x2)
        loss2 = out2[-1].sum()
        loss2.backward()

        grad_W_x_accumulated = rnn.rnn_cell.W_x.grad

        # Accumulated should be larger
        assert torch.abs(grad_W_x_accumulated).sum() > torch.abs(grad_W_x_first).sum()
