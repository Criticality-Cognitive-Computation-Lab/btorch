"""Tests for hex grid functionality with human inspection figures.

This test generates visualizations for:
1. FlyWire coordinate mapping verification (known data point)
2. Hex grid rotation symmetry (edge case: rotation invariance)
3. Distance from center pattern (radial gradient)
"""

import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import save_fig
from btorch.utils.hex import (
    HexGrid,
    axial_to_zigzag,
    disk,
    radius,
    ring,
    rotate,
    spiral,
    to_pixel,
    zigzag_to_axial,
)


def test_zigzag_coordinate_mapping():
    """Verify zigzag coordinate conversion with known FlyWire data point.

    FlyWire's x,y are zigzag coordinates. From FlyWire Codex: hex (-17,
    -8) in axial maps to (4, -25) in zigzag (x, y).
    """
    # Known data point from FlyWire Codex
    q, r = np.array([-17]), np.array([-8])
    x, y = axial_to_zigzag(q, r)

    # Verify forward conversion
    assert x[0] == 4, f"Expected x=4, got {x[0]}"
    assert y[0] == -25, f"Expected y=-25, got {y[0]}"

    # Verify inverse conversion
    q_back, r_back = zigzag_to_axial(x, y)
    assert q_back[0] == -17, f"Expected q=-17, got {q_back[0]}"
    assert r_back[0] == -8, f"Expected r=-8, got {r_back[0]}"

    # Create visualization showing the mapping
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Axial coordinates (standard hex grid)
    q_all, r_all = disk(20)
    x_px, y_px = to_pixel(q_all, r_all)

    axes[0].scatter(x_px, y_px, c="lightblue", s=20, alpha=0.5)
    q_px, r_px = to_pixel(q, r)
    axes[0].scatter(q_px, r_px, c="red", s=100, marker="x", linewidths=3)
    axes[0].set_xlabel("x (pixels)")
    axes[0].set_ylabel("y (pixels)")
    axes[0].set_title("Axial Coordinates (Pointy-Top)\n(-17, -8) highlighted")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Zigzag coordinates as raw values
    x_all, y_all = axial_to_zigzag(q_all, r_all)

    axes[1].scatter(x_all, y_all, c="lightblue", s=20, alpha=0.5)
    axes[1].scatter(x, y, c="red", s=100, marker="x", linewidths=3)
    axes[1].set_xlabel("x (zigzag)")
    axes[1].set_ylabel("y (zigzag)")
    axes[1].set_title("Zigzag Coordinates\n(4, -25) highlighted")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Convert zigzag back to pixel coordinates for proper hex visualization
    q_back_all, r_back_all = zigzag_to_axial(x_all, y_all)
    x_back_px, y_back_px = to_pixel(q_back_all, r_back_all)

    axes[2].scatter(x_back_px, y_back_px, c="lightblue", s=20, alpha=0.5)
    q_h_px, r_h_px = to_pixel(q, r)
    axes[2].scatter(q_h_px, r_h_px, c="red", s=100, marker="x", linewidths=3)
    axes[2].set_xlabel("x (pixels)")
    axes[2].set_ylabel("y (pixels)")
    axes[2].set_title("Zigzag → Axial → Pixels\n(Roundtrip verification)")
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "zigzag_coordinate_mapping", suffix="png")
    plt.close()


def test_hex_rotation_symmetry():
    """Visualize rotation symmetry - after 6 rotations of 60°, we return to start.

    This tests the edge case of rotational invariance and verifies
    that our rotation implementation is consistent.
    """
    # Create a marked hex grid
    grid = HexGrid(radius=5)
    q, r = grid.q.copy(), grid.r.copy()

    # Mark a specific hex for tracking
    marker_idx = np.where((q == 3) & (r == 0))[0][0]
    values = np.zeros(len(q))
    values[marker_idx] = 1.0

    # Apply 6 rotations of 60°
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for n in range(6):
        q_rot, r_rot = rotate(q, r, n)
        x, y = to_pixel(q_rot, r_rot)

        # The marked hex should cycle through positions
        ax = axes[n]
        ax.scatter(x, y, c=values, cmap="Blues", s=100, vmin=0, vmax=1)
        ax.set_title(f"Rotation {n}×60° = {n * 60}°")
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("Hex Grid Rotation Symmetry (6-fold)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "hex_rotation_symmetry", suffix="png")
    plt.close()

    # Verify 6 rotations returns to original
    q_final, r_final = rotate(q, r, 6)
    assert np.array_equal(q, q_final), "6 rotations should return to original"
    assert np.array_equal(r, r_final), "6 rotations should return to original"


def test_radial_distance_gradient():
    """Visualize radial distance from center - a common network pattern.

    This shows a Gaussian-like receptive field pattern which is
    a classic example use case for hexagonal grids in neural networks.
    """
    grid = HexGrid(radius=10)
    q, r = grid.q, grid.r

    # Compute distance from center
    dist = radius(q, r)

    # Create a center-surround receptive field
    sigma_center = 3.0
    sigma_surround = 6.0
    center = np.exp(-(dist**2) / (2 * sigma_center**2))
    surround = -0.5 * np.exp(-(dist**2) / (2 * sigma_surround**2))
    rf = center + surround

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Distance map
    x, y = to_pixel(q, r)
    im1 = axes[0].scatter(x, y, c=dist, cmap="viridis", s=50)
    axes[0].set_title("Distance from Center")
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0])

    # Center component
    im2 = axes[1].scatter(x, y, c=center, cmap="Reds", s=50)
    axes[1].set_title("Center (σ=3)")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    # Full center-surround RF
    im3 = axes[2].scatter(x, y, c=rf, cmap="RdBu_r", s=50, vmin=-0.5, vmax=1)
    axes[2].set_title("Center-Surround RF")
    axes[2].set_aspect("equal")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle("Radial Distance Patterns (Example Network RF)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "radial_distance_gradient", suffix="png")
    plt.close()

    # Verify max distance equals radius
    assert dist.max() == 10, "Max distance should equal grid radius"


def test_spiral_ordering():
    """Visualize spiral ordering - used for permutation-based transforms.

    Shows the ordering of hexes in spiral order (center, ring1, ring2, ...)
    which is essential for data augmentation and indexing operations.
    """
    radius = 4
    q, r = spiral(radius)
    x, y = to_pixel(q, r)

    # Color by index in spiral order
    indices = np.arange(len(q))

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(x, y, c=indices, cmap="tab20", s=200, edgecolors="black")

    # Annotate first few indices
    for i in range(min(19, len(q))):
        ax.annotate(str(i), (x[i], y[i]), ha="center", va="center", fontsize=8)

    ax.set_title(f"Spiral Ordering (radius={radius}, n={len(q)} hexes)")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.colorbar(scatter, ax=ax, label="Spiral index")

    save_fig(fig, "spiral_ordering", suffix="png")
    plt.close()

    # Verify spiral order: center first, then rings
    assert q[0] == 0 and r[0] == 0, "First hex should be center (0,0)"
    # Ring 1 should be indices 1-6
    q1, r1 = ring(1)
    assert len(q1) == 6, "Ring 1 should have 6 hexes"


if __name__ == "__main__":
    test_zigzag_coordinate_mapping()
    test_hex_rotation_symmetry()
    test_radial_distance_gradient()
    test_spiral_ordering()
    print("All hex grid tests passed with figures saved to fig/tests/")
