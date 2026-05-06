"""Tests for coordinate range operations.

Demonstrates range intersection and union operations.
Reference: https://www.redblobgames.com/grids/hexagons/#range-intersection
"""

import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import save_fig
from btorch.utils.hex import range_intersection, range_union, ranges_intersect, to_pixel
from btorch.utils.hex.coords import disk


def test_range_intersection():
    """Test range intersection - hexes within ALL ranges."""
    # Two overlapping disks
    centers = [(0, 0), (3, 0)]
    radii = [2, 2]

    q, r = range_intersection(centers, radii)

    # The intersection should be non-empty
    assert len(q) > 0, "Intersection should not be empty"

    # All hexes in intersection should be within both ranges
    from btorch.utils.hex import distance

    for i in range(len(q)):
        for (cq, cr), rad in zip(centers, radii):
            dist = int(
                distance(
                    np.array([q[i]]), np.array([r[i]]), np.array([cq]), np.array([cr])
                )[0]
            )
            assert (
                dist <= rad
            ), f"Hex ({q[i]},{r[i]}) is not within range of ({cq},{cr})"


def test_range_union():
    """Test range union - hexes within ANY range."""
    # Two overlapping disks
    centers = [(0, 0), (5, 0)]
    radii = [2, 2]

    q, r = range_union(centers, radii)

    # The union should have more hexes than either individual disk
    q1, r1 = disk(2, 0, 0)
    assert len(q) > len(q1), "Union should be larger than individual disks"

    # Verify no duplicates
    coords = list(zip(q.tolist(), r.tolist()))
    assert len(coords) == len(set(coords)), "Union should not have duplicates"


def test_ranges_intersect():
    """Test the ranges_intersect function."""
    # Overlapping ranges
    assert ranges_intersect((0, 0), 2, (3, 0), 2) is True

    # Non-overlapping ranges
    assert ranges_intersect((0, 0), 1, (5, 0), 1) is False

    # Touching ranges (distance == sum of radii)
    assert ranges_intersect((0, 0), 2, (4, 0), 2) is True


def test_range_intersection_visualization():
    """Visualize range intersection."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Two overlapping disks
    centers = [(0, 0), (4, 0)]
    radii = [3, 3]

    # Get intersection
    q_int, r_int = range_intersection(centers, radii)

    # Background grid
    q_bg, r_bg = disk(6)
    x_bg, y_bg = to_pixel(q_bg, r_bg)

    for idx, (ax, (cq, cr), rad) in enumerate(zip(axes[:2], centers, radii)):
        ax.scatter(x_bg, y_bg, c="lightgray", s=50, alpha=0.3)

        # Full disk
        q_disk, r_disk = disk(rad, cq, cr)
        x_disk, y_disk = to_pixel(q_disk, r_disk)
        ax.scatter(x_disk, y_disk, c="lightblue", s=100, alpha=0.5)

        # Center
        x_c, y_c = to_pixel(np.array([cq]), np.array([cr]))
        ax.scatter(x_c, y_c, c="red", s=200, marker="*", zorder=5)

        ax.set_title(f"Disk {idx + 1}\nCenter ({cq},{cr}), Radius {rad}")
        ax.set_aspect("equal")
        ax.axis("off")

    # Intersection
    ax = axes[2]
    ax.scatter(x_bg, y_bg, c="lightgray", s=50, alpha=0.3)

    x_int, y_int = to_pixel(q_int, r_int)
    ax.scatter(x_int, y_int, c="green", s=150, edgecolors="black")

    # Both centers
    for cq, cr in centers:
        x_c, y_c = to_pixel(np.array([cq]), np.array([cr]))
        ax.scatter(x_c, y_c, c="red", s=200, marker="*", zorder=5)

    ax.set_title(f"Intersection\n({len(q_int)} hexes)")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.suptitle("Range Intersection (Hexes in ALL ranges)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "range_intersection", suffix="png")
    plt.close()


def test_range_union_visualization():
    """Visualize range union."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Two overlapping disks
    centers = [(0, 0), (6, 0)]
    radii = [3, 3]

    # Get union
    q_union, r_union = range_union(centers, radii)

    # Background grid
    q_bg, r_bg = disk(10)
    x_bg, y_bg = to_pixel(q_bg, r_bg)

    for idx, (ax, (cq, cr), rad) in enumerate(zip(axes[:2], centers, radii)):
        ax.scatter(x_bg, y_bg, c="lightgray", s=50, alpha=0.3)

        # Full disk
        q_disk, r_disk = disk(rad, cq, cr)
        x_disk, y_disk = to_pixel(q_disk, r_disk)
        ax.scatter(x_disk, y_disk, c="lightblue", s=100, alpha=0.5)

        # Center
        x_c, y_c = to_pixel(np.array([cq]), np.array([cr]))
        ax.scatter(x_c, y_c, c="red", s=200, marker="*", zorder=5)

        ax.set_title(f"Disk {idx + 1}\nCenter ({cq},{cr}), Radius {rad}")
        ax.set_aspect("equal")
        ax.axis("off")

    # Union
    ax = axes[2]
    ax.scatter(x_bg, y_bg, c="lightgray", s=50, alpha=0.3)

    x_union, y_union = to_pixel(q_union, r_union)
    ax.scatter(x_union, y_union, c="orange", s=100, alpha=0.7, edgecolors="black")

    # Both centers
    for cq, cr in centers:
        x_c, y_c = to_pixel(np.array([cq]), np.array([cr]))
        ax.scatter(x_c, y_c, c="red", s=200, marker="*", zorder=5)

    ax.set_title(f"Union\n({len(q_union)} hexes)")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.suptitle("Range Union (Hexes in ANY range)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "range_union", suffix="png")
    plt.close()


def test_multiple_range_intersection():
    """Test intersection of multiple ranges."""
    # Three disks with common intersection
    centers = [(0, 0), (2, 0), (1, 2)]
    radii = [3, 3, 3]

    q, r = range_intersection(centers, radii)

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 8))

    # Background
    q_bg, r_bg = disk(6)
    x_bg, y_bg = to_pixel(q_bg, r_bg)
    ax.scatter(x_bg, y_bg, c="lightgray", s=50, alpha=0.3)

    # Individual disks
    colors = ["lightblue", "lightgreen", "lightyellow"]
    for (cq, cr), rad, color in zip(centers, radii, colors):
        q_disk, r_disk = disk(rad, cq, cr)
        x_disk, y_disk = to_pixel(q_disk, r_disk)
        ax.scatter(x_disk, y_disk, c=color, s=100, alpha=0.4)

    # Intersection
    if len(q) > 0:
        x_int, y_int = to_pixel(q, r)
        ax.scatter(x_int, y_int, c="red", s=200, edgecolors="black", zorder=5)

    # Centers
    for cq, cr in centers:
        x_c, y_c = to_pixel(np.array([cq]), np.array([cr]))
        ax.scatter(x_c, y_c, c="black", s=100, marker="*", zorder=6)

    ax.set_title(f"Three-Way Intersection\n({len(q)} hexes in common)")
    ax.set_aspect("equal")
    ax.axis("off")

    save_fig(fig, "multiple_range_intersection", suffix="png")
    plt.close()


if __name__ == "__main__":
    test_range_intersection()
    test_range_union()
    test_ranges_intersect()
    test_range_intersection_visualization()
    test_range_union_visualization()
    test_multiple_range_intersection()
    print("Range operation tests passed with figures saved")
