"""Tests for hex convolution with human inspection figures.

Demonstrates:
1. Hexagonal kernel shape visualization
2. Receptive field pattern (center-surround)
3. Comparison with rectangular convolution
4. Edge case: small vs large kernels
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from btorch.models.hex import Conv2dHex
from btorch.utils.file import save_fig
from btorch.utils.hex import disk, to_pixel


def test_hex_kernel_shape():
    """Visualize the hexagonal kernel mask.

    Shows how Conv2dHex constrains the receptive field to hex shape
    compared to standard rectangular convolution.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    kernel_sizes = [3, 5, 7]

    for i, k in enumerate(kernel_sizes):
        # Create hex conv layer
        conv = Conv2dHex(1, 1, kernel_size=k)

        # Get the mask
        mask = conv.mask[0, 0].numpy()

        # Visualize mask
        axes[0, i].imshow(mask, cmap="Blues", origin="lower")
        axes[0, i].set_title(f"Hex Kernel k={k}")
        axes[0, i].axis("off")

        # Overlay hex coordinates
        if k > 1:
            q, r = disk(k // 2)
            q = q - q.min()
            r = r - r.min()
            axes[0, i].scatter(q, r, c="red", s=50, marker="x")

        # Show effective receptive field
        axes[1, i].imshow(mask, cmap="viridis", origin="lower", alpha=0.5)

        # Add weight visualization
        weights = conv.weight[0, 0].detach().numpy()
        axes[1, i].imshow(weights, cmap="RdBu_r", origin="lower", alpha=0.7)
        axes[1, i].set_title(f"Initial Weights k={k}")
        axes[1, i].axis("off")

    plt.suptitle("Hexagonal Kernel Shapes", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "hex_kernel_shapes")
    plt.close()

    # Verify hex shape
    conv7 = Conv2dHex(1, 1, kernel_size=7)
    mask7 = conv7.mask[0, 0].numpy()

    # Count active positions
    n_active = mask7.sum()
    # For radius 3 disk: 1 + 3*3*4 = 37 hexes
    assert n_active == 37, f"Expected 37 active positions for k=7, got {n_active}"


def test_center_surround_rf():
    """Create and visualize a center-surround receptive field using Conv2dHex.

    Example use case: modeling retinal ganglion cell or LGN receptive fields.
    """
    # Create a Conv2dHex layer
    k = 7
    conv = Conv2dHex(1, 1, kernel_size=k)

    # Manually set weights to create center-surround pattern
    with torch.no_grad():
        weights = torch.zeros_like(conv.weight)

        # Get hex coordinates
        q, r = disk(k // 2)
        dist = np.sqrt(q**2 + r**2)

        # Center-surround weights (positive center, negative surround)
        for i, (qi, ri, di) in enumerate(zip(q, r, dist)):
            q_norm = qi - q.min()
            r_norm = ri - r.min()
            if di < 1.5:
                weights[0, 0, q_norm, r_norm] = 1.0  # center
            elif di < 3.0:
                weights[0, 0, q_norm, r_norm] = -0.3  # surround
            else:
                weights[0, 0, q_norm, r_norm] = 0.0  # outside

        conv.weight.copy_(weights)

    # Visualize the RF
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Weight map
    rf_weights = conv.weight[0, 0].detach().numpy()
    im1 = axes[0].imshow(rf_weights, cmap="RdBu_r", origin="lower", vmin=-0.5, vmax=1)
    axes[0].set_title("RF Weights (Cartesian Map)")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0])

    # Hex grid visualization
    x, y = to_pixel(q, r)
    rf_values = [
        conv.weight[0, 0, qi - q.min(), ri - r.min()].item() for qi, ri in zip(q, r)
    ]
    im2 = axes[1].scatter(
        x, y, c=rf_values, cmap="RdBu_r", s=200, vmin=-0.5, vmax=1, edgecolors="black"
    )
    axes[1].set_title("RF on Hex Grid")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    # Cross-section
    center_idx = np.argmin(np.sqrt(q**2 + r**2))
    center_x, center_y = x[center_idx], y[center_idx]
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    sort_idx = np.argsort(distances)

    axes[2].plot(distances[sort_idx], np.array(rf_values)[sort_idx], "o-")
    axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[2].set_xlabel("Distance from center")
    axes[2].set_ylabel("Weight")
    axes[2].set_title("RF Cross-Section")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Center-Surround Receptive Field (Conv2dHex)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "center_surround_rf")
    plt.close()


def test_conv_forward():
    """Test forward pass and compare hex vs rectangular convolution.

    Shows the difference in output patterns between hexagonal and
    standard rectangular convolution.
    """
    # Create input feature map
    size = 32
    x = torch.zeros(1, 1, size, size)
    # Create a point stimulus in center
    x[0, 0, size // 2, size // 2] = 1.0

    # Hex convolution
    conv_hex = Conv2dHex(1, 1, kernel_size=7, padding=3)
    with torch.no_grad():
        conv_hex.weight.fill_(0.1)  # uniform weights
        if conv_hex.bias is not None:
            conv_hex.bias.fill_(0.0)  # zero bias for testing

    out_hex = conv_hex(x)

    # Standard rectangular convolution (for comparison)
    conv_rect = torch.nn.Conv2d(1, 1, kernel_size=7, padding=3)
    with torch.no_grad():
        conv_rect.weight.fill_(0.1)

    out_rect = conv_rect(x)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Input
    axes[0, 0].imshow(x[0, 0].detach().numpy(), cmap="viridis")
    axes[0, 0].set_title("Input (Point Stimulus)")
    axes[0, 0].axis("off")

    # Hex output
    axes[0, 1].imshow(out_hex[0, 0].detach().numpy(), cmap="viridis")
    axes[0, 1].set_title("Hex Conv Output")
    axes[0, 1].axis("off")

    # Rect output
    axes[0, 2].imshow(out_rect[0, 0].detach().numpy(), cmap="viridis")
    axes[0, 2].set_title("Rect Conv Output")
    axes[0, 2].axis("off")

    # Difference
    diff = out_hex - out_rect
    im = axes[1, 0].imshow(diff[0, 0].detach().numpy(), cmap="RdBu_r")
    axes[1, 0].set_title("Difference (Hex - Rect)")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0])

    # Hex kernel
    axes[1, 1].imshow(conv_hex.mask[0, 0].detach().numpy(), cmap="Blues")
    axes[1, 1].set_title("Hex Kernel Mask")
    axes[1, 1].axis("off")

    # Cross-section comparison
    center = size // 2
    axes[1, 2].plot(out_hex[0, 0, center, :].detach().numpy(), label="Hex", linewidth=2)
    axes[1, 2].plot(
        out_rect[0, 0, center, :].detach().numpy(), label="Rect", linewidth=2
    )
    axes[1, 2].set_xlabel("Position")
    axes[1, 2].set_ylabel("Response")
    axes[1, 2].set_title("Cross-Section")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("Hexagonal vs Rectangular Convolution", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "conv_forward_comparison")
    plt.close()

    # Verify output has expected properties
    out_hex_np = out_hex[0, 0].detach().numpy()
    # Output should have non-zero values where kernel overlaps with stimulus
    assert out_hex_np.sum() > 0, "Output should have non-zero response"
    # Output shape should match input shape (with padding)
    assert out_hex_np.shape == (size, size), "Output shape should match input"


if __name__ == "__main__":
    test_hex_kernel_shape()
    test_center_surround_rf()
    test_conv_forward()
    print("Conv2dHex tests passed with figures saved")
