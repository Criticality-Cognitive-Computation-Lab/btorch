"""Tests for doubled coordinate systems with human inspection figures.

Doubled coordinates make rectangular maps easier to work with.
Reference: https://www.redblobgames.com/grids/hexagons/#coordinates-doubled
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from btorch.utils.file import save_fig
from btorch.utils.hex import (
    axial_to_doubleheight,
    axial_to_doublewidth,
    doubleheight_to_pixel,
    doublewidth_to_pixel,
    pixel_to_doubleheight,
    pixel_to_doublewidth,
    to_pixel,
)
from btorch.utils.hex.coords import rectangle
from btorch.utils.hex.doubled import doubleheight_to_axial, doublewidth_to_axial


@pytest.mark.parametrize(
    "to_doubled,from_doubled,orientation",
    [
        (axial_to_doublewidth, doublewidth_to_axial, "pointy"),
        (axial_to_doubleheight, doubleheight_to_axial, "flat"),
    ],
    ids=["doublewidth", "doubleheight"],
)
def test_doubled_roundtrip(to_doubled, from_doubled, orientation):
    """Both doubled conventions must roundtrip through axial."""
    q, r = rectangle(5, 5, orientation=orientation)
    col, row = to_doubled(q, r)
    q_b, r_b = from_doubled(col, row)
    assert np.array_equal(q, q_b) and np.array_equal(r, r_b)


def test_doublewidth_visualization():
    """Visualize double-width coordinate system.

    Double-width doubles the column index (col = 2*q + r) so that every
    hex sits in its own column — useful for array storage with pointy-
    top hexes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    q, r = rectangle(6, 5, orientation="pointy")
    x, y = to_pixel(q, r)
    col, row = axial_to_doublewidth(q, r)

    # Left: axial coordinates
    ax = axes[0]
    ax.scatter(x, y, c="lightblue", s=400, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"({q[i]},{r[i]})",
            (x[i], y[i]),
            ha="center",
            va="center",
            fontsize=8,
        )
    ax.set_title("Axial Coordinates")
    ax.set_aspect("equal")
    ax.axis("off")

    # Right: double-width coordinates
    ax = axes[1]
    ax.scatter(x, y, c="lightgreen", s=400, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"({col[i]},{row[i]})",
            (x[i], y[i]),
            ha="center",
            va="center",
            fontsize=8,
        )
    ax.set_title("Double-Width Coordinates\n(col=2*q+r, row=r)")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.suptitle("Double-Width Coordinate System (Pointy-Top)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "doublewidth_coordinates", suffix="png")
    plt.close()


def test_doubleheight_visualization():
    """Visualize double-height coordinate system.

    Double-height doubles the row index (row = 2*r + q) so that every
    hex sits in its own row — useful for array storage with flat-top
    hexes.
    """
    from btorch.utils.hex.transform import to_pixel as _to_pixel

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    q, r = rectangle(6, 5, orientation="flat")
    x, y = _to_pixel(q, r, orientation="flat")
    col, row = axial_to_doubleheight(q, r)

    # Left: axial coordinates
    ax = axes[0]
    ax.scatter(x, y, c="lightblue", s=400, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"({q[i]},{r[i]})",
            (x[i], y[i]),
            ha="center",
            va="center",
            fontsize=8,
        )
    ax.set_title("Axial Coordinates")
    ax.set_aspect("equal")
    ax.axis("off")

    # Right: double-height coordinates
    ax = axes[1]
    ax.scatter(x, y, c="lightcoral", s=400, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"({col[i]},{row[i]})",
            (x[i], y[i]),
            ha="center",
            va="center",
            fontsize=8,
        )
    ax.set_title("Double-Height Coordinates\n(col=q, row=2*r+q)")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.suptitle("Double-Height Coordinate System (Flat-Top)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "doubleheight_coordinates", suffix="png")
    plt.close()


def test_doubled_pixel_conversion():
    """Test pixel conversion for doubled coordinates.

    Verifies that doublewidth/doubleheight → pixel → back is lossless.
    """
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
