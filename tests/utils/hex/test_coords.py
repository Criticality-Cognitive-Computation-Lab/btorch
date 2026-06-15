"""Tests for hex grid primitives: disk, ring, spiral, distance, neighbors.

Generates figures for human inspection and runs assertion-heavy checks
on core coordinate math.
"""

import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import save_fig
from btorch.utils.hex import (
    HexGrid,
    disk,
    radius,
    rotate,
    to_pixel,
)
from btorch.utils.hex.coords import disk_count, rectangle, ring, spiral
from btorch.utils.hex.distance import distance, within_range
from btorch.utils.hex.neighbor import diagonal_neighbors, neighbors


# ---- disk / ring / spiral ----


class TestDiskRingSpiral:
    def test_disk_count_formula(self):
        for r in range(8):
            assert disk_count(r) == 1 + 3 * r * (r + 1)

    def test_disk_radius_zero(self):
        q, r = disk(0)
        assert len(q) == 1
        assert q[0] == 0 and r[0] == 0

    def test_disk_center_unique(self):
        q, r = disk(3)
        pairs = {(int(qi), int(ri)) for qi, ri in zip(q, r)}
        assert len(pairs) == len(q), "disk has duplicate coordinates"
        center_count = sum(1 for qi, ri in zip(q, r) if qi == 0 and ri == 0)
        assert center_count == 1

    def test_disk_with_offset(self):
        q, r = disk(2, center_q=5, center_r=-3)
        assert (5, -3) in set(zip(q.tolist(), r.tolist()))

    def test_ring_count(self):
        for rad in range(1, 6):
            q, r = ring(rad)
            assert len(q) == 6 * rad

    def test_ring_all_at_distance(self):
        for rad in [1, 3, 5]:
            q, r = ring(rad)
            d = radius(q, r)
            assert np.all(d == rad)

    def test_spiral_starts_at_center(self):
        q, r = spiral(5)
        assert q[0] == 0 and r[0] == 0

    def test_spiral_covers_disk(self):
        for rad in range(1, 5):
            q_sp, r_sp = spiral(rad)
            q_d, r_d = disk(rad)
            assert set(zip(q_sp.tolist(), r_sp.tolist())) == set(
                zip(q_d.tolist(), r_d.tolist())
            )

    def test_rectangle_shape(self):
        q, r = rectangle(4, 3, orientation="pointy")
        assert len(q) > 0
        assert len(q) == len(r)


# ---- distance ----


class TestDistance:
    def test_self_distance_zero(self):
        q, r = disk(3)
        d = distance(q, r, q, r)
        assert np.all(d == 0)

    def test_known_distances(self):
        origin_q = np.array([0])
        origin_r = np.array([0])
        targets = [
            (np.array([1]), np.array([0]), 1),
            (np.array([0]), np.array([1]), 1),
            (np.array([2]), np.array([0]), 2),
            (np.array([1]), np.array([1]), 2),
            (np.array([-1]), np.array([2]), 2),
        ]
        for tq, tr, expected in targets:
            d = distance(origin_q, origin_r, tq, tr)
            assert (
                d[0] == expected
            ), f"distance(0,0)->({tq[0]},{tr[0]}) = {d[0]}, expected {expected}"

    def test_radius_of_disk(self):
        q, r = disk(4)
        rad = radius(q, r)
        assert rad.max() == 4

    def test_within_range(self):
        q, r = disk(5)
        mask = within_range(q, r, 0, 0, 2)
        assert mask.sum() == disk_count(2)

    def test_distance_symmetry(self):
        q1, r1 = np.array([1, -2]), np.array([3, 0])
        q2, r2 = np.array([-1, 4]), np.array([2, -3])
        d12 = distance(q1, r1, q2, r2)
        d21 = distance(q2, r2, q1, r1)
        assert np.array_equal(d12, d21)


# ---- neighbors ----


class TestNeighbors:
    def test_neighbor_count(self):
        q, r = np.array([0]), np.array([0])
        qn, rn = neighbors(q, r)
        assert qn.shape == (6, 1)
        assert rn.shape == (6, 1)

    def test_all_neighbors_dist1(self):
        q, r = np.array([0]), np.array([0])
        qn, rn = neighbors(q, r)
        d = distance(q, r, qn.flatten(), rn.flatten())
        assert np.all(d == 1)

    def test_neighbor_no_duplicates(self):
        q, r = np.array([0]), np.array([0])
        qn, rn = neighbors(q, r)
        pairs = list(zip(qn.flatten().tolist(), rn.flatten().tolist()))
        assert len(pairs) == len(set(pairs))

    def test_diagonal_neighbors_dist2(self):
        q, r = np.array([0]), np.array([0])
        qd, rd = diagonal_neighbors(q, r)
        d = distance(q, r, qd.flatten(), rd.flatten())
        assert np.all(d == 2)


# ---- visualization ----


def test_zigzag_coordinate_mapping():
    """Verify zigzag coordinate conversion with known FlyWire data point.

    FlyWire's x,y are zigzag coordinates. From FlyWire Codex: hex (-17,
    -8) in axial maps to (4, -25) in zigzag (x, y).
    """
    from btorch.utils.hex import axial_to_zigzag, zigzag_to_axial

    q, r = np.array([-17]), np.array([-8])
    x, y = axial_to_zigzag(q, r)

    assert x[0] == 4, f"Expected x=4, got {x[0]}"
    assert y[0] == -25, f"Expected y=-25, got {y[0]}"

    q_back, r_back = zigzag_to_axial(x, y)
    assert q_back[0] == -17, f"Expected q=-17, got {q_back[0]}"
    assert r_back[0] == -8, f"Expected r=-8, got {r_back[0]}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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

    x_all, y_all = axial_to_zigzag(q_all, r_all)
    axes[1].scatter(x_all, y_all, c="lightblue", s=20, alpha=0.5)
    axes[1].scatter(x, y, c="red", s=100, marker="x", linewidths=3)
    axes[1].set_xlabel("x (zigzag)")
    axes[1].set_ylabel("y (zigzag)")
    axes[1].set_title("Zigzag Coordinates\n(4, -25) highlighted")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    q_back_all, r_back_all = zigzag_to_axial(x_all, y_all)
    x_back_px, y_back_px = to_pixel(q_back_all, r_back_all)
    axes[2].scatter(x_back_px, y_back_px, c="lightblue", s=20, alpha=0.5)
    axes[2].scatter(q_px, r_px, c="red", s=100, marker="x", linewidths=3)
    axes[2].set_xlabel("x (pixels)")
    axes[2].set_ylabel("y (pixels)")
    axes[2].set_title("Zigzag → Axial → Pixels\n(Roundtrip verification)")
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "zigzag_coordinate_mapping", suffix="png")
    plt.close()


def test_hex_rotation_symmetry():
    """Visualize rotation symmetry - after 6 rotations of 60deg, return to start.

    This tests the edge case of rotational invariance and verifies
    that our rotation implementation is consistent.
    """
    grid = HexGrid(radius=5)
    q, r = grid.q.copy(), grid.r.copy()

    marker_idx = np.where((q == 3) & (r == 0))[0][0]
    values = np.zeros(len(q))
    values[marker_idx] = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for n in range(6):
        q_rot, r_rot = rotate(q, r, n)
        x, y = to_pixel(q_rot, r_rot)
        axes[n].scatter(x, y, c=values, cmap="Blues", s=100, vmin=0, vmax=1)
        axes[n].set_title(f"Rotation {n}×60° = {n * 60}°")
        axes[n].set_aspect("equal")
        axes[n].axis("off")

    plt.suptitle("Hex Grid Rotation Symmetry (6-fold)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "hex_rotation_symmetry", suffix="png")
    plt.close()


def test_radial_distance_gradient():
    """Visualize radial distance from center — a common network pattern.

    Shows a Gaussian-like receptive field pattern which is a classic
    example use case for hexagonal grids in neural networks.
    """
    grid = HexGrid(radius=10)
    q, r = grid.q, grid.r
    dist = radius(q, r)

    sigma_center = 3.0
    sigma_surround = 6.0
    center = np.exp(-(dist**2) / (2 * sigma_center**2))
    surround = -0.5 * np.exp(-(dist**2) / (2 * sigma_surround**2))
    rf = center + surround

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x, y = to_pixel(q, r)

    im1 = axes[0].scatter(x, y, c=dist, cmap="viridis", s=50)
    axes[0].set_title("Distance from Center")
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].scatter(x, y, c=center, cmap="Reds", s=50)
    axes[1].set_title("Center (σ=3)")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].scatter(x, y, c=rf, cmap="RdBu_r", s=50, vmin=-0.5, vmax=1)
    axes[2].set_title("Center-Surround RF")
    axes[2].set_aspect("equal")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle("Radial Distance Patterns (Example Network RF)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "radial_distance_gradient", suffix="png")
    plt.close()
