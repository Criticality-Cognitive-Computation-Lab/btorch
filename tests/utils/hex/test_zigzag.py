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
    assert np.array_equal(q, q_back)
    assert np.array_equal(r, r_back)


def test_zigzag_known_flywire_values():
    """Hand-checked values from FlyWire visual column coordinates."""
    q = np.array([-17, -17, -18, -18])
    r = np.array([-7, -8, -8, -9])
    x, y = axial_to_zigzag(q, r)
    assert np.array_equal(x, [5, 4, 5, 4])
    assert np.array_equal(y, [-24, -25, -26, -27])


def test_zigzag_column_pattern():
    """Zigzag must place adjacent hexes in alternating columns.

    A vertical line in axial (same q, incrementing r) should produce at
    most 2 distinct x values and y incrementing by 1.
    """
    q = np.array([0, 0, 0, 0])
    r = np.array([0, 1, 2, 3])
    x, y = axial_to_zigzag(q, r)
    assert len(np.unique(x)) <= 2
    assert np.all(np.diff(y) == 1)


def test_zigzag_scatter_visualization():
    """Scatter plot with zigzag coordinates must produce clean hex columns.

    Demonstrates how zigzag encoding arranges hexes into vertical
    columns, which is the natural storage layout for FlyWire connectome
    data.
    """
    q, r = disk(4)
    # Use spiral index as values so we can visually trace ordering
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
        show_compass="vertex",
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
        show_compass="edge",
    )

    plt.tight_layout()
    save_fig(fig, "zigzag_scatter_visualization", suffix="png")
    plt.close()


def test_zigzag_grid_visualization():
    """Grid plot with zigzag coordinates and annotations.

    Useful for understanding how axial (q,r) maps to zigzag (x,y)
    columns — each hex shows both representations.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    grid(
        radius=3,
        coord_format="zigzag",
        orientation="pointy",
        annotate=True,
        ax=ax,
        title="Zigzag grid with annotations",
        show_compass="vertex",
    )

    save_fig(fig, "zigzag_grid_visualization", suffix="png")
    plt.close()
