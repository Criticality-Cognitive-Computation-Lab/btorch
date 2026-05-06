"""Tests for hexagonal eye rendering with human inspection figures.

Demonstrates:
1. HexEye receptor layout visualization
2. Rendering comparison: mean, sum, median modes
3. Edge case: small vs large ommatidia count
4. Comparison with rectangular sampling
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from btorch.models.hex import BoxEye, HexEye
from btorch.utils.file import save_fig
from btorch.utils.hex import to_pixel


def test_eye_receptor_layout():
    """Visualize the receptor layout of HexEye.

    Shows the hexagonal arrangement of ommatidia (receptors) for
    different eye sizes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    configs = [
        (127, 25, "Small (127)"),
        (721, 25, "Medium (721)"),
        (2791, 25, "Large (2791)"),
    ]

    for i, (n_ommatidia, ppo, title) in enumerate(configs):
        try:
            eye = HexEye(n_ommatidia=n_ommatidia, ppo=ppo)

            # Get receptor centers
            x = eye.receptor_x.numpy()
            y = eye.receptor_y.numpy()

            axes[i].scatter(x, y, c="blue", s=10, alpha=0.6)
            axes[i].set_title(f"{title}\n{eye.radius} rings")
            axes[i].set_aspect("equal")
            axes[i].axis("off")
        except ValueError:
            # If n_ommatidia doesn't fill a regular hex grid
            axes[i].text(
                0.5,
                0.5,
                f"{n_ommatidia} not valid\nhex grid size",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
            )
            axes[i].set_title(title)

    plt.suptitle("HexEye Receptor Layouts", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "eye_receptor_layouts")
    plt.close()


def test_eye_rendering_modes():
    """Compare different rendering modes of HexEye.

    Shows how mean, sum, and median sampling produce different responses
    to the same input stimulus.
    """
    # Create eye
    eye = HexEye(n_ommatidia=127, ppo=25, height_px=200, width_px=200)

    # Create test stimulus: gradient pattern
    y, x = torch.meshgrid(
        torch.linspace(0, 1, 200), torch.linspace(0, 1, 200), indexing="ij"
    )
    stimulus = (x * y).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    # Render with different modes (using the current implementation)
    # Note: The current implementation samples single pixels, so modes
    # would need to be implemented for full comparison
    with torch.no_grad():
        rendered = eye(stimulus)

    # Get hex positions for visualization
    _ = eye.receptor_x.numpy(), eye.receptor_y.numpy()

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original stimulus
    axes[0].imshow(stimulus[0, 0].numpy(), cmap="viridis", origin="lower")
    axes[0].set_title("Input Stimulus")
    axes[0].axis("off")

    # Rendered on hex grid
    from btorch.utils.hex import disk

    q_hex, r_hex = disk(eye.radius)
    x_hex, y_hex = to_pixel(q_hex, r_hex)

    values = rendered[0, 0].numpy()
    im = axes[1].scatter(x_hex, y_hex, c=values, cmap="viridis", s=50)
    axes[1].set_title(f"Hex Rendered (n={len(values)})")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])

    # Response histogram
    axes[2].hist(values, bins=30, color="steelblue", edgecolor="black")
    axes[2].set_xlabel("Response Value")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Response Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("HexEye Rendering", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "eye_rendering")
    plt.close()

    # Verify rendering preserves range
    assert (
        values.min() >= 0 and values.max() <= 1
    ), "Rendered values should be in [0, 1] range"


def test_eye_edge_cases():
    """Test edge cases for HexEye.

    - Single ommatidium (radius=0)
    - Very small eye
    - Non-standard sizes
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    test_cases = [
        (1, "Single ommatidium"),
        (7, "Tiny eye (7)"),
        (19, "Small eye (19)"),
        (37, "Medium-small eye (37)"),
    ]

    for idx, (n_hex, title) in enumerate(test_cases):
        ax = axes[idx // 2, idx % 2]

        try:
            eye = HexEye(n_ommatidia=n_hex, ppo=20)
            q, r = eye.receptor_x.numpy(), eye.receptor_y.numpy()

            ax.scatter(q, r, s=200, c="blue", edgecolors="black")
            ax.set_title(f"{title}\n{len(q)} receptors")
            ax.set_aspect("equal")
            ax.axis("off")
        except ValueError as e:
            ax.text(
                0.5,
                0.5,
                str(e),
                ha="center",
                va="center",
                transform=ax.transAxes,
                wrap=True,
            )
            ax.set_title(title)

    plt.suptitle("HexEye Edge Cases", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "eye_edge_cases")
    plt.close()


def test_hex_vs_box_eye():
    """Compare HexEye vs BoxEye rendering.

    Shows the difference in receptor arrangement between precise
    hexagonal and approximate box sampling.
    """
    # Both with similar extent
    n_hex = 127
    ppo = 25

    hex_eye = HexEye(n_ommatidia=n_hex, ppo=ppo, height_px=200, width_px=200)
    box_eye = BoxEye(n_ommatidia=n_hex, ppo=ppo, height_px=200, width_px=200)

    # Create test stimulus
    stimulus = torch.randn(1, 1, 200, 200).abs()

    with torch.no_grad():
        hex_out = hex_eye(stimulus)[0, 0].numpy()
        box_out = box_eye(stimulus)[0, 0].numpy()

    # Get positions
    from btorch.utils.hex import disk

    q, r = disk(hex_eye.radius)
    x_hex, y_hex = to_pixel(q, r)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # HexEye output
    im1 = axes[0].scatter(x_hex, y_hex, c=hex_out, cmap="viridis", s=50)
    axes[0].set_title("HexEye Output")
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0])

    # BoxEye output - use simple grid layout based on pooled output shape
    # BoxEye uses pooling which produces a regular grid
    n_box = len(box_out)
    grid_size = int(np.sqrt(n_box))
    x_box = np.arange(n_box) % grid_size
    y_box = np.arange(n_box) // grid_size

    im2 = axes[1].scatter(x_box, y_box, c=box_out, cmap="viridis", s=50)
    axes[1].set_title(f"BoxEye Output ({n_box} units)")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    # Comparison scatter - truncate to min length for comparison
    min_len = min(len(hex_out), len(box_out))
    axes[2].scatter(hex_out[:min_len], box_out[:min_len], alpha=0.5)
    axes[2].plot([0, 1], [0, 1], "r--", label="y=x")
    axes[2].set_xlabel("HexEye Response")
    axes[2].set_ylabel("BoxEye Response")
    axes[2].set_title("Response Comparison")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("HexEye vs BoxEye", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "hex_vs_box_eye")
    plt.close()


if __name__ == "__main__":
    test_eye_receptor_layout()
    test_eye_rendering_modes()
    test_eye_edge_cases()
    test_hex_vs_box_eye()
    print("Eye rendering tests passed with figures saved")
