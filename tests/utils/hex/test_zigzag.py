"""Tests for zigzag offset coordinates.

Zigzag coordinates create clean vertical columns with alternating x values:
x = floor((r - q) / 2), y = q + r
"""

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import save_fig
from btorch.utils.hex import axial_to_zigzag, zigzag_to_axial
from btorch.utils.hex.coords import disk
from btorch.visualisation.hex import grid, scatter


def test_zigzag_roundtrip():
    """Axial -> zigzag -> axial must be exact for all hexes in a disk."""
    q, r = disk(5)

    x, y = axial_to_zigzag(q, r)
    q_back, r_back = zigzag_to_axial(x, y)

    assert np.array_equal(q, q_back), "Zigzag roundtrip failed for q"
    assert np.array_equal(r, r_back), "Zigzag roundtrip failed for r"


def test_zigzag_known_values():
    """Hand-checked values from FlyWire visual column coordinates."""
    q = np.array([-17, -17, -18, -18])
    r = np.array([-7, -8, -8, -9])

    x, y = axial_to_zigzag(q, r)

    expected_x = np.array([5, 4, 5, 4])
    expected_y = np.array([-24, -25, -26, -27])

    assert np.array_equal(x, expected_x), f"Expected x={expected_x}, got {x}"
    assert np.array_equal(y, expected_y), f"Expected y={expected_y}, got {y}"

    q_back, r_back = zigzag_to_axial(x, y)
    assert np.array_equal(q, q_back)
    assert np.array_equal(r, r_back)


def test_zigzag_column_pattern():
    """Zigzag must place adjacent hexes in alternating columns."""
    # Single vertical column in zigzag: same x, y differs by 1
    q = np.array([0, 0, 0, 0])
    r = np.array([0, 1, 2, 3])

    x, y = axial_to_zigzag(q, r)

    # x should alternate between two values (zigzag pattern)
    assert (
        len(np.unique(x)) <= 2
    ), "Zigzag should have at most 2 column values for a straight line"
    # y should increment by 1 each step
    assert np.all(
        np.diff(y) == 1
    ), "Zigzag y should increment by 1 for adjacent axial steps"


def test_zigzag_scatter_visualization():
    """Scatter plot with zigzag coordinates must produce clean hex columns."""
    q, r = disk(4)
    values = np.arange(len(q))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    scatter(
        q,
        r,
        values,
        coord_format="zigzag",
        size=1.0,
        orientation="pointy",
        ax=axes[0],
        title="Zigzag (vertex compass)",
        compass="vertex",
    )

    scatter(
        q,
        r,
        values,
        coord_format="zigzag",
        size=1.0,
        orientation="pointy",
        ax=axes[1],
        title="Zigzag (edge compass)",
        compass="edge",
    )

    plt.tight_layout()
    save_fig(fig, "zigzag_scatter_visualization", suffix="png")
    plt.close()


def test_zigzag_grid_visualization():
    """Grid plot with zigzag coordinates and annotations."""
    fig, ax = plt.subplots(figsize=(8, 8))

    grid(
        radius=3,
        coord_format="zigzag",
        orientation="pointy",
        annotate=True,
        ax=ax,
        title="Zigzag grid with annotations",
        compass="vertex",
    )

    save_fig(fig, "zigzag_grid_visualization", suffix="png")
    plt.close()


if __name__ == "__main__":
    test_zigzag_roundtrip()
    test_zigzag_known_values()
    test_zigzag_column_pattern()
    test_zigzag_scatter_visualization()
    test_zigzag_grid_visualization()
    print("Zigzag tests passed")
