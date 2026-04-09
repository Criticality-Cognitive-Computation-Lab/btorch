"""Tests for doubled coordinate systems.

Doubled coordinates make rectangular maps easier to work with.
Reference: https://www.redblobgames.com/grids/hexagons/#coordinates-doubled
"""

import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import save_fig
from btorch.utils.hex import (
    axial_to_doubleheight,
    axial_to_doublewidth,
    doubleheight_distance,
    doubleheight_to_axial,
    doubleheight_to_pixel,
    doublewidth_distance,
    doublewidth_to_axial,
    doublewidth_to_pixel,
    pixel_to_doubleheight,
    pixel_to_doublewidth,
    to_pixel,
)
from btorch.utils.hex.coords import rectangle


def test_doublewidth_roundtrip():
    """Test double-width coordinate roundtrip conversion."""
    q, r = rectangle(5, 5, orientation="pointy")

    # Convert to double-width and back
    col, row = axial_to_doublewidth(q, r)
    q_back, r_back = doublewidth_to_axial(col, row)

    assert np.array_equal(q, q_back), "double-width roundtrip failed for q"
    assert np.array_equal(r, r_back), "double-width roundtrip failed for r"


def test_doubleheight_roundtrip():
    """Test double-height coordinate roundtrip conversion."""
    q, r = rectangle(5, 5, orientation="flat")

    # Convert to double-height and back
    col, row = axial_to_doubleheight(q, r)
    q_back, r_back = doubleheight_to_axial(col, row)

    assert np.array_equal(q, q_back), "double-height roundtrip failed for q"
    assert np.array_equal(r, r_back), "double-height roundtrip failed for r"


def test_doublewidth_distance():
    """Test distance calculation in double-width coordinates."""
    from btorch.utils.hex import distance

    test_cases = [
        (0, 0, 2, 0, 1),  # Adjacent in double-width
        (0, 0, 0, 1, 1),  # Vertical neighbor
        (0, 0, 4, 0, 2),  # Two steps away
    ]

    for c1, r1, c2, r2, expected in test_cases:
        dist = doublewidth_distance(c1, r1, c2, r2)

        # Verify against axial distance
        q1, r1_axial = doublewidth_to_axial(np.array([c1]), np.array([r1]))
        q2, r2_axial = doublewidth_to_axial(np.array([c2]), np.array([r2]))
        axial_dist = int(distance(q1, r1_axial, q2, r2_axial)[0])

        assert (
            dist == axial_dist
        ), f"Distance mismatch: doublewidth={dist}, axial={axial_dist}"


def test_doubleheight_distance():
    """Test distance calculation in double-height coordinates."""
    from btorch.utils.hex import distance

    test_cases = [
        (0, 0, 0, 2, 1),  # Adjacent in double-height
        (0, 0, 1, 0, 1),  # Horizontal neighbor
        (0, 0, 0, 4, 2),  # Two steps away
    ]

    for c1, r1, c2, r2, expected in test_cases:
        dist = doubleheight_distance(c1, r1, c2, r2)

        # Verify against axial distance
        q1, r1_axial = doubleheight_to_axial(np.array([c1]), np.array([r1]))
        q2, r2_axial = doubleheight_to_axial(np.array([c2]), np.array([r2]))
        axial_dist = int(distance(q1, r1_axial, q2, r2_axial)[0])

        assert (
            dist == axial_dist
        ), f"Distance mismatch: doubleheight={dist}, axial={axial_dist}"


def test_doublewidth_visualization():
    """Visualize double-width coordinate system."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create hex grid
    q, r = rectangle(6, 5, orientation="pointy")
    x, y = to_pixel(q, r)

    # Double-width coordinates
    col, row = axial_to_doublewidth(q, r)

    # Left: axial coordinates
    ax = axes[0]
    ax.scatter(x, y, c="lightblue", s=400, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"({q[i]},{r[i]})", (x[i], y[i]), ha="center", va="center", fontsize=8
        )
    ax.set_title("Axial Coordinates")
    ax.set_aspect("equal")
    ax.axis("off")

    # Right: double-width coordinates
    ax = axes[1]
    ax.scatter(x, y, c="lightgreen", s=400, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"({col[i]},{row[i]})", (x[i], y[i]), ha="center", va="center", fontsize=8
        )
    ax.set_title("Double-Width Coordinates\n(col=2*q+r, row=r)")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.suptitle("Double-Width Coordinate System (Pointy-Top)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "doublewidth_coordinates", suffix="png")
    plt.close()


def test_doubleheight_visualization():
    """Visualize double-height coordinate system."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create hex grid (flat-top)
    from btorch.utils.hex.transform import to_pixel as _to_pixel

    q, r = rectangle(6, 5, orientation="flat")
    x, y = _to_pixel(q, r, orientation="flat")

    # Double-height coordinates
    col, row = axial_to_doubleheight(q, r)

    # Left: axial coordinates
    ax = axes[0]
    ax.scatter(x, y, c="lightblue", s=400, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"({q[i]},{r[i]})", (x[i], y[i]), ha="center", va="center", fontsize=8
        )
    ax.set_title("Axial Coordinates")
    ax.set_aspect("equal")
    ax.axis("off")

    # Right: double-height coordinates
    ax = axes[1]
    ax.scatter(x, y, c="lightcoral", s=400, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"({col[i]},{row[i]})", (x[i], y[i]), ha="center", va="center", fontsize=8
        )
    ax.set_title("Double-Height Coordinates\n(col=q, row=2*r+q)")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.suptitle("Double-Height Coordinate System (Flat-Top)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "doubleheight_coordinates", suffix="png")
    plt.close()


def test_doubled_pixel_conversion():
    """Test pixel conversion for doubled coordinates."""
    # Double-width pixel conversion
    col = np.array([0, 2, 4])
    row = np.array([0, 0, 1])
    size = 1.0

    x, y = doublewidth_to_pixel(col, row, size)

    # Verify by converting back
    col_back, row_back = pixel_to_doublewidth(x, y, size)
    assert np.allclose(col, col_back), "doublewidth pixel roundtrip failed"
    assert np.allclose(row, row_back), "doublewidth pixel roundtrip failed"

    # Double-height pixel conversion
    col = np.array([0, 1, 2])
    row = np.array([0, 2, 4])

    x, y = doubleheight_to_pixel(col, row, size)

    # Verify by converting back
    col_back, row_back = pixel_to_doubleheight(x, y, size)
    assert np.allclose(col, col_back), "doubleheight pixel roundtrip failed"
    assert np.allclose(row, row_back), "doubleheight pixel roundtrip failed"


if __name__ == "__main__":
    test_doublewidth_roundtrip()
    test_doubleheight_roundtrip()
    test_doublewidth_distance()
    test_doubleheight_distance()
    test_doublewidth_visualization()
    test_doubleheight_visualization()
    test_doubled_pixel_conversion()
    print("Doubled coordinate tests passed with figures saved")
