import pytest
import torch
from btorch.models.base import MemoryModule
from btorch.models.rnn import make_rnn
from torch import nn


class SimpleRNNCell(MemoryModule):
    """Simple RNN cell: h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_size))

        self.register_memory("h", None, hidden_size)
        self.init_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single step forward.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, hidden_size)
        """
        batch_size = x.shape[0]

        if self.h is None or self.h.shape[0] != batch_size:
            self.h = torch.zeros(
                batch_size, self.hidden_size, device=x.device, dtype=x.dtype
            )

        self.h = torch.tanh(x @ self.W_x.t() + self.h @ self.W_h.t() + self.b)
        return self.h


def compute_numerical_gradient(func, x, eps=1e-5):
    """Compute numerical gradient using finite differences.

    Args:
        func: Function that takes x and returns a scalar loss
        x: Input tensor
        eps: Perturbation size

    Returns:
        Numerical gradient tensor with same shape as x
    """
    grad = torch.zeros_like(x)
    x_flat = x.view(-1)

    for i in range(x_flat.numel()):
        # Perturb +eps
        x_flat[i] += eps
        loss_plus = func(x.clone())

        # Perturb -eps
        x_flat[i] -= 2 * eps
        loss_minus = func(x.clone())

        # Restore original value
        x_flat[i] += eps

        # Compute gradient
        grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)

    return grad


@pytest.mark.skip(reason="TODO")
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

    def test_numerical_gradient_single_timestep(self):
        """Verify analytical gradients match numerical gradients for single
        timestep."""
        torch.manual_seed(42)

        # Small dimensions for tractable numerical gradient computation
        rnn = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)(
            input_size=3, hidden_size=4
        )

        # Single timestep input
        x = torch.randn(1, 2, 3, requires_grad=True)  # (T=1, batch=2, input_size=3)

        # Function to compute loss given input
        def loss_fn(x_in):
            rnn.rnn_cell.reset()  # Reset state
            out, _ = rnn.multi_step_forward(x_in)
            return out.sum()

        # Analytical gradient
        loss = loss_fn(x)
        loss.backward()
        analytical_grad = x.grad.clone()

        # Numerical gradient
        x_numerical = x.detach().clone().requires_grad_(False)
        numerical_grad = compute_numerical_gradient(loss_fn, x_numerical, eps=1e-6)

        # Compare
        rel_error = (analytical_grad - numerical_grad).abs() / (
            numerical_grad.abs() + 1e-8
        )
        max_rel_error = rel_error.max().item()

        assert max_rel_error < 1e-3, f"Max relative error: {max_rel_error:.6f}"
        assert torch.allclose(
            analytical_grad, numerical_grad, rtol=1e-3, atol=1e-5
        ), "Analytical and numerical gradients should match"

    def test_numerical_gradient_short_sequence(self):
        """Verify gradients for short sequence (3 timesteps)."""
        torch.manual_seed(42)

        rnn = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)(
            input_size=3, hidden_size=4
        )

        # Short sequence
        x = torch.randn(3, 2, 3, requires_grad=True)  # (T=3, batch=2, input_size=3)

        def loss_fn(x_in):
            rnn.rnn_cell.init_state()
            out, _ = rnn.multi_step_forward(x_in)
            return out.sum()

        # Analytical gradient
        loss = loss_fn(x)
        loss.backward()
        analytical_grad = x.grad.clone()

        # Numerical gradient (only check first and last timestep for speed)
        x_numerical = x.detach().clone()
        numerical_grad = torch.zeros_like(x_numerical)

        for t in [0, -1]:  # Check first and last timestep
            for b in range(x.shape[1]):
                for i in range(x.shape[2]):
                    # Perturb
                    eps = 1e-5
                    x_numerical[t, b, i] += eps
                    loss_plus = loss_fn(x_numerical)

                    x_numerical[t, b, i] -= 2 * eps
                    loss_minus = loss_fn(x_numerical)

                    x_numerical[t, b, i] += eps  # Restore

                    numerical_grad[t, b, i] = (loss_plus - loss_minus) / (2 * eps)

        # Compare checked timesteps
        for t in [0, -1]:
            rel_error = (analytical_grad[t] - numerical_grad[t]).abs() / (
                numerical_grad[t].abs() + 1e-8
            )
            max_rel_error = rel_error.max().item()

            assert (
                max_rel_error < 1e-3
            ), f"Timestep {t}: Max relative error: {max_rel_error:.6f}"

    def test_gradient_correctness_with_checkpoint(self):
        """Verify checkpointed gradients match numerical gradients."""
        torch.manual_seed(42)

        rnn = make_rnn(SimpleRNNCell, grad_checkpoint=True, unroll=4)(
            input_size=3, hidden_size=4
        )

        x = torch.randn(3, 2, 3, requires_grad=True)

        def loss_fn(x_in):
            rnn.rnn_cell.init_state()
            out, _ = rnn.multi_step_forward(x_in)
            return out.sum()

        # Analytical gradient
        loss = loss_fn(x)
        loss.backward()
        analytical_grad = x.grad.clone()

        # Numerical gradient (sample a few points)
        x_numerical = x.detach().clone()
        eps = 1e-5

        # Check a few random points
        for _ in range(5):
            t = torch.randint(0, x.shape[0], (1,)).item()
            b = torch.randint(0, x.shape[1], (1,)).item()
            i = torch.randint(0, x.shape[2], (1,)).item()

            x_numerical[t, b, i] += eps
            loss_plus = loss_fn(x_numerical)

            x_numerical[t, b, i] -= 2 * eps
            loss_minus = loss_fn(x_numerical)

            x_numerical[t, b, i] += eps

            numerical_grad_val = (loss_plus - loss_minus) / (2 * eps)
            analytical_grad_val = analytical_grad[t, b, i].item()

            rel_error = abs(analytical_grad_val - numerical_grad_val) / (
                abs(numerical_grad_val) + 1e-8
            )

            assert rel_error < 1e-3, (
                f"Point ({t},{b},{i}): analytical={analytical_grad_val:.6f}, "
                f"numerical={numerical_grad_val:.6f}, rel_error={rel_error:.6f}"
            )

    def test_long_sequence_gradient_flow(self):
        """Test that gradients flow correctly through long sequences (20+
        steps)."""
        torch.manual_seed(42)

        rnn = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)(
            input_size=4, hidden_size=8
        )

        # Long sequence - 25 timesteps
        T = 25
        x = torch.randn(T, 2, 4, requires_grad=True)

        out, _ = rnn.multi_step_forward(x)
        loss = out.sum()
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

    def test_parameter_gradient_correctness(self):
        """Verify parameter gradients match numerical gradients."""
        torch.manual_seed(42)

        rnn = make_rnn(SimpleRNNCell, grad_checkpoint=False, unroll=4)(
            input_size=3, hidden_size=4
        )

        x = torch.randn(3, 2, 3)

        # Analytical gradient for W_x
        x_grad = x.clone().requires_grad_(True)
        out, _ = rnn.multi_step_forward(x_grad)
        loss = out.sum()
        loss.backward()

        analytical_W_x_grad = rnn.rnn_cell.W_x.grad.clone()

        # Numerical gradient for W_x (sample a few parameters)
        eps = 1e-5
        for _ in range(5):
            i = torch.randint(0, rnn.rnn_cell.W_x.shape[0], (1,)).item()
            j = torch.randint(0, rnn.rnn_cell.W_x.shape[1], (1,)).item()

            # Reset gradients and state
            rnn.zero_grad()
            rnn.rnn_cell.init_state()

            # Perturb +eps
            original_val = rnn.rnn_cell.W_x.data[i, j].item()
            rnn.rnn_cell.W_x.data[i, j] += eps
            out_plus, _ = rnn.multi_step_forward(x)
            loss_plus = out_plus.sum()

            # Perturb -eps
            rnn.rnn_cell.W_x.data[i, j] = original_val - eps
            rnn.rnn_cell.init_state()
            out_minus, _ = rnn.multi_step_forward(x)
            loss_minus = out_minus.sum()

            # Restore
            rnn.rnn_cell.W_x.data[i, j] = original_val

            numerical_grad = (loss_plus - loss_minus) / (2 * eps)
            analytical_val = analytical_W_x_grad[i, j].item()

            rel_error = abs(analytical_val - numerical_grad.item()) / (
                abs(numerical_grad.item()) + 1e-8
            )

            assert rel_error < 1e-3, (
                f"W_x[{i},{j}]: analytical={analytical_val:.6f}, "
                f"numerical={numerical_grad.item():.6f}, rel_error={rel_error:.6f}"
            )

    def test_multi_step_gradients_no_checkpoint(self, rnn_no_checkpoint):
        """Test gradients through multi-step forward without checkpointing."""
        torch.manual_seed(42)

        T, batch_size, input_size = 10, 2, 4
        x = torch.randn(T, batch_size, input_size, requires_grad=True)

        # Forward pass
        out, states = rnn_no_checkpoint.multi_step_forward(x)

        # Compute loss
        loss = out.sum()
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

        # Forward pass
        out, states = rnn_with_checkpoint.multi_step_forward(x)

        # Compute loss
        loss = out.sum()
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
        out_no_chkpt, _ = rnn_no_chkpt.multi_step_forward(x)
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

        out, _ = rnn.multi_step_forward(x)
        loss = out.sum()
        loss.backward()

        # Check gradient magnitudes across time
        grad_norms = torch.stack([x.grad[t].norm() for t in range(T)])

        # Gradients should exist throughout
        assert (grad_norms > 0).all(), "All timesteps should have non-zero gradients"

        # Ratio of max to min shouldn't be too extreme (allowing for some variation)
        ratio = grad_norms.max() / (grad_norms.min() + 1e-8)
        assert ratio < 1e4, f"Gradient magnitude ratio too large: {ratio}"

    def test_parameter_gradient_shapes(self, rnn_no_checkpoint):
        """Test that parameter gradients have correct shapes."""
        torch.manual_seed(42)

        T, batch_size = 5, 2
        x = torch.randn(T, batch_size, 4, requires_grad=True)

        out, _ = rnn_no_checkpoint.multi_step_forward(x)
        loss = out.sum()
        loss.backward()

        # Check gradient shapes match parameter shapes
        assert (
            rnn_no_checkpoint.rnn_cell.W_x.grad.shape
            == rnn_no_checkpoint.rnn_cell.W_x.shape
        )
        assert (
            rnn_no_checkpoint.rnn_cell.W_h.grad.shape
            == rnn_no_checkpoint.rnn_cell.W_h.shape
        )
        assert (
            rnn_no_checkpoint.rnn_cell.b.grad.shape
            == rnn_no_checkpoint.rnn_cell.b.shape
        )

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
        out1, _ = rnn.multi_step_forward(x1)
        loss1 = out1.sum()
        loss1.backward()

        grad_W_x_first = rnn.rnn_cell.W_x.grad.clone()

        # Second backward pass (accumulate)
        x2 = torch.randn(T, batch_size, 4, requires_grad=True)
        rnn.rnn_cell.h = rnn.rnn_cell.h.detach()
        out2, _ = rnn.multi_step_forward(x2)
        loss2 = out2.sum()
        loss2.backward()

        grad_W_x_accumulated = rnn.rnn_cell.W_x.grad

        # Accumulated should be larger
        assert torch.abs(grad_W_x_accumulated).sum() > torch.abs(grad_W_x_first).sum()
