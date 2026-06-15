"""Tests for offset coordinate systems with human inspection figures.

Offset coordinates are commonly used for storing hex grids in 2D arrays.
Demonstrates odd-r, even-r, odd-q, even-q conversions.
Reference: https://www.redblobgames.com/grids/hexagons/#coordinates-offset
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from btorch.utils.file import save_fig
from btorch.utils.hex import (
    axial_to_even_q,
    axial_to_even_r,
    axial_to_odd_q,
    axial_to_odd_r,
    to_pixel,
)
from btorch.utils.hex.coords import rectangle
from btorch.utils.hex.offset import (
    even_q_to_axial,
    even_r_to_axial,
    odd_q_to_axial,
    odd_r_to_axial,
)


@pytest.mark.parametrize(
    "to_axial,from_axial,orientation",
    [
        (axial_to_odd_r, odd_r_to_axial, "pointy"),
        (axial_to_even_r, even_r_to_axial, "pointy"),
        (axial_to_odd_q, odd_q_to_axial, "flat"),
        (axial_to_even_q, even_q_to_axial, "flat"),
    ],
    ids=["odd_r", "even_r", "odd_q", "even_q"],
)
def test_offset_roundtrip(to_axial, from_axial, orientation):
    """All 4 offset conventions must roundtrip through axial."""
    q, r = rectangle(5, 5, orientation=orientation)
    col, row = to_axial(q, r)
    q_b, r_b = from_axial(col, row)
    assert np.array_equal(q, q_b) and np.array_equal(r, r_b)


def test_offset_visualization():
    """Visualize offset coordinate systems.

    Shows how the same hex grid maps to different (col, row) values
    depending on the offset convention used. Pointy-top hexes use
    odd-r/even-r; flat-top hexes use odd-q/even-q.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Pointy-top hexes (odd-r, even-r)
    q, r = rectangle(6, 5, orientation="pointy")
    x, y = to_pixel(q, r)

    # Odd-r offset
    ax = axes[0, 0]
    col, row = axial_to_odd_r(q, r)
    ax.scatter(x, y, c=col, cmap="tab20", s=300, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"c{col[i]}\nr{row[i]}",
            (x[i], y[i]),
            ha="center",
            va="center",
            fontsize=7,
        )
    ax.set_title("Odd-r Offset (pointy-top)")
    ax.set_aspect("equal")
    ax.axis("off")

    # Even-r offset
    ax = axes[0, 1]
    col, row = axial_to_even_r(q, r)
    ax.scatter(x, y, c=col, cmap="tab20", s=300, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"c{col[i]}\nr{row[i]}",
            (x[i], y[i]),
            ha="center",
            va="center",
            fontsize=7,
        )
    ax.set_title("Even-r Offset (pointy-top)")
    ax.set_aspect("equal")
    ax.axis("off")

    # Flat-top hexes (odd-q, even-q)
    q, r = rectangle(6, 5, orientation="flat")
    # Re-orient for flat-top visualization
    from btorch.utils.hex.transform import to_pixel as _to_pixel

    x, y = _to_pixel(q, r, orientation="flat")

    # Odd-q offset
    ax = axes[1, 0]
    col, row = axial_to_odd_q(q, r)
    ax.scatter(x, y, c=row, cmap="tab20", s=300, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"c{col[i]}\nr{row[i]}",
            (x[i], y[i]),
            ha="center",
            va="center",
            fontsize=7,
        )
    ax.set_title("Odd-q Offset (flat-top)")
    ax.set_aspect("equal")
    ax.axis("off")

    # Even-q offset
    ax = axes[1, 1]
    col, row = axial_to_even_q(q, r)
    ax.scatter(x, y, c=row, cmap="tab20", s=300, edgecolors="black")
    for i in range(len(q)):
        ax.annotate(
            f"c{col[i]}\nr{row[i]}",
            (x[i], y[i]),
            ha="center",
            va="center",
            fontsize=7,
        )
    ax.set_title("Even-q Offset (flat-top)")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.suptitle("Offset Coordinate Systems", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "offset_coordinate_systems", suffix="png")
    plt.close()


def test_offset_to_axial_neighbors():
    """Verify that offset coordinates can use axial neighbors via conversion.

    The key property: converting offset→axial→neighbors→offset should
    give the same neighbor set as staying in axial the whole time.
    """
    from btorch.utils.hex import neighbors as axial_neighbors
    from btorch.utils.hex.offset import odd_r_to_axial

    # Test a few hexes - convert offset -> axial -> neighbors -> offset
    test_coords_axial = [(0, 0), (1, 0), (0, 1), (2, -1)]

    for q, r in test_coords_axial:
        # Get axial neighbors directly
        q_axial = np.array([q])
        r_axial = np.array([r])
        qn_axial, rn_axial = axial_neighbors(q_axial, r_axial)

        # Convert to odd-r, then back via neighbors
        col, row = axial_to_odd_r(q_axial, r_axial)
        q_back, r_back = odd_r_to_axial(col, row)

        # Verify roundtrip
        assert q_back[0] == q and r_back[0] == r, "Offset roundtrip failed"

        # Get neighbors in axial space
        qn_from_offset, rn_from_offset = axial_neighbors(q_back, r_back)

        # Should match direct axial neighbors
        axial_set = set(zip(qn_axial.flatten().tolist(), rn_axial.flatten().tolist()))
        offset_set = set(
            zip(qn_from_offset.flatten().tolist(), rn_from_offset.flatten().tolist())
        )

        assert axial_set == offset_set, (
            f"Neighbor mismatch at ({q}, {r}): "
            f"axial={axial_set}, via_offset={offset_set}"
        )
