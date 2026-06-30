import numpy as np
import pandas as pd
import torch

from btorch.models.conv import Conv1dSpatial


def create_test_neurons(n_neurons: int = 10) -> pd.DataFrame:
    """Create a simple test neurons DataFrame with spatial coordinates."""
    return pd.DataFrame(
        {
            "x": np.random.uniform(0, 10, n_neurons),
            "y": np.random.uniform(0, 10, n_neurons),
            "z": np.random.uniform(0, 10, n_neurons),
        }
    )


def test_conv1d_spatial_forward_correctness():
    """Test that Conv1dSpatial forward pass produces correct output shapes and
    values."""
    print("Testing Conv1dSpatial forward correctness...")

    # Test parameters
    batch_size = 3
    in_channels = 4
    out_channels = 6
    n_neurons = 8
    n_neighbor = 3
    include_self = True

    # Create test data
    neurons = create_test_neurons(n_neurons)

    # Create the layer
    layer = Conv1dSpatial(
        in_channels=in_channels,
        out_channels=out_channels,
        neurons=neurons,
        n_neighbor=n_neighbor,
        include_self=include_self,
        bias=True,
    )

    # Test with different input shapes
    test_cases = [
        # Shape: (batch_size, in_channels, n_neurons)
        (batch_size, in_channels, n_neurons),
        # Shape with extra leading dimension: (2, batch_size, in_channels, n_neurons)
        (2, batch_size, in_channels, n_neurons),
        # Single sample: (in_channels, n_neurons)
        (in_channels, n_neurons),
    ]

    for input_shape in test_cases:
        print(f"  Testing input shape: {input_shape}")

        # Create test input
        x = torch.randn(input_shape, requires_grad=True)

        # Forward pass
        output = layer(x)

        # Check output shape
        expected_output_shape = list(input_shape)
        expected_output_shape[-2] = (
            out_channels  # Replace in_channels with out_channels
        )

        assert output.shape == tuple(
            expected_output_shape
        ), f"Expected output shape {expected_output_shape}, got {output.shape}"

        # Check that output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"

        # Check that gradients can flow
        loss = output.sum()
        loss.backward()
        assert x.grad is not None, "No gradients computed for input"
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

        print(f"    ✓ Shape: {input_shape} -> {output.shape}")

        # Reset gradients for next test
        x.grad = None

    # Test specific values for a simple case
    print("  Testing specific values...")

    # Create a simple deterministic case
    torch.manual_seed(42)
    x_simple = torch.ones(1, in_channels, n_neurons, requires_grad=True)

    # Set layer weights to known values for predictable output
    with torch.no_grad():
        layer.weight.fill_(0.5)
        if layer.bias is not None:
            layer.bias.fill_(0.1)

    output_simple = layer(x_simple)

    # Since all inputs are 1 and all weights are 0.5, each output should be:
    # kernel_size * in_channels * 0.5 + 0.1 (bias)
    kernel_size = n_neighbor + (1 if include_self else 0)
    expected_value = kernel_size * in_channels * 0.5 + 0.1

    # Check that outputs are close to expected
    assert torch.allclose(
        output_simple, torch.full_like(output_simple, expected_value), atol=1e-5
    ), (
        f"Expected output values around {expected_value}, "
        f"got {output_simple.mean().item()}"
    )

    print("    ✓ Values are correct")
    print("✓ Forward correctness test passed!")


def test_conv1d_spatial_gradient_backprop():
    """Test that Conv1dSpatial properly computes gradients during
    backpropagation."""
    print("Testing Conv1dSpatial gradient backpropagation...")

    # Test parameters
    batch_size = 2
    in_channels = 3
    out_channels = 4
    n_neurons = 6
    n_neighbor = 2
    include_self = True

    # Create test data
    neurons = create_test_neurons(n_neurons)

    # Create the layer
    layer = Conv1dSpatial(
        in_channels=in_channels,
        out_channels=out_channels,
        neurons=neurons,
        n_neighbor=n_neighbor,
        include_self=include_self,
        bias=True,
    )

    # Test input
    x = torch.randn(batch_size, in_channels, n_neurons, requires_grad=True)

    print("  Testing gradient computation...")

    # Forward pass
    output = layer(x)

    # Compute loss (sum of squares)
    loss = (output**2).sum()

    # Backward pass
    loss.backward()

    # Check that all parameters have gradients
    assert layer.weight.grad is not None, "Weight gradients not computed"
    if layer.bias is not None:
        assert layer.bias.grad is not None, "Bias gradients not computed"
    assert x.grad is not None, "Input gradients not computed"

    # Check gradient shapes
    assert layer.weight.grad.shape == layer.weight.shape, (
        f"Weight gradient shape {layer.weight.grad.shape} "
        f"doesn't match weight shape {layer.weight.shape}"
    )

    if layer.bias is not None:
        assert layer.bias.grad.shape == layer.bias.shape, (
            f"Bias gradient shape {layer.bias.grad.shape} "
            f"doesn't match bias shape {layer.bias.shape}"
        )

    assert (
        x.grad.shape == x.shape
    ), f"Input gradient shape {x.grad.shape} doesn't match input shape {x.shape}"

    # Check that gradients are finite
    assert torch.isfinite(
        layer.weight.grad
    ).all(), "Weight gradients contain non-finite values"
    if layer.bias is not None:
        assert torch.isfinite(
            layer.bias.grad
        ).all(), "Bias gradients contain non-finite values"
    assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

    # Check that gradients are non-zero (layer should be learning)
    assert not torch.allclose(
        layer.weight.grad, torch.zeros_like(layer.weight.grad)
    ), "Weight gradients are all zero"
    assert not torch.allclose(
        x.grad, torch.zeros_like(x.grad)
    ), "Input gradients are all zero"

    print("    ✓ All gradients computed and finite")

    # Test gradient numerical stability with multiple backward passes
    print("  Testing gradient accumulation...")

    # Second forward/backward pass (gradients should accumulate)
    output2 = layer(x)
    loss2 = (output2**2).sum()
    loss2.backward()

    # Check that gradients accumulated (should be roughly 2x the original)
    assert not torch.allclose(
        layer.weight.grad, torch.zeros_like(layer.weight.grad)
    ), "Gradients disappeared after accumulation"

    print("    ✓ Gradient accumulation works")

    # Test gradient zeroing
    print("  Testing gradient zeroing...")

    layer.zero_grad()
    assert layer.weight.grad is None or torch.allclose(
        layer.weight.grad, torch.zeros_like(layer.weight.grad)
    ), "Weight gradients not zeroed"
    if layer.bias is not None:
        assert layer.bias.grad is None or torch.allclose(
            layer.bias.grad, torch.zeros_like(layer.bias.grad)
        ), "Bias gradients not zeroed"

    print("    ✓ Gradient zeroing works")
    print("✓ Gradient backpropagation test passed!")


if __name__ == "__main__":
    test_conv1d_spatial_forward_correctness()
    print()
    test_conv1d_spatial_gradient_backprop()
    print("\n🎉 All tests passed!")
