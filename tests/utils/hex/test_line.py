"""Tests for line drawing on hex grids.

Demonstrates the line drawing algorithm using cube lerp + round.
Reference: https://www.redblobgames.com/grids/hexagons/#line-drawing
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from btorch.utils.file import save_fig
from btorch.utils.hex import line, line_n, to_pixel
from btorch.utils.hex.coords import disk
from btorch.utils.hex.distance import distance


@pytest.mark.parametrize(
    "q1,r1,q2,r2",
    [
        (0, 0, 1, 0),
        (0, 0, 3, -2),
        (-2, 3, 2, -1),
    ],
)
def test_line_distance_property(q1, r1, q2, r2):
    """Line length must equal distance + 1 (including both endpoints)."""
    ql, rl = line(np.array([q1]), np.array([r1]), np.array([q2]), np.array([r2]))
    d = int(distance(np.array([q1]), np.array([r1]), np.array([q2]), np.array([r2]))[0])
    assert len(ql) == d + 1


def test_line_to_self():
    q, r = line(np.array([2]), np.array([3]), np.array([2]), np.array([3]))
    assert len(q) == 1


def test_line_endpoints_present():
    q, r = line(np.array([0]), np.array([0]), np.array([2]), np.array([2]))
    coords = set(zip(q.tolist(), r.tolist()))
    assert (0, 0) in coords and (2, 2) in coords


def test_line_n_length():
    q, r = line_n(np.array([0]), np.array([0]), np.array([3]), np.array([0]), 3)
    assert len(q) == 4


def test_line_visualization():
    """Visualize lines between various points in a hex grid.

    Shows lines along all 6 cardinal directions to verify the cube lerp
    + round algorithm produces straight paths.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Get background grid
    q_bg, r_bg = disk(5)
    x_bg, y_bg = to_pixel(q_bg, r_bg)

    # Test lines to different destinations
    test_cases = [
        (0, 0, 4, 0, "Horizontal E"),
        (0, 0, 0, 4, "SE diagonal"),
        (0, 0, -4, 4, "SW diagonal"),
        (0, 0, -4, 0, "Horizontal W"),
        (0, 0, 0, -4, "NW diagonal"),
        (0, 0, 4, -4, "NE diagonal"),
    ]

    for ax, (q1, r1, q2, r2, title) in zip(axes, test_cases):
        # Draw line
        q_line, r_line = line(
            np.array([q1]), np.array([r1]), np.array([q2]), np.array([r2])
        )
        x_line, y_line = to_pixel(q_line, r_line)

        # Background grid
        ax.scatter(x_bg, y_bg, c="lightgray", s=50, alpha=0.5)

        # Line points
        ax.scatter(
            x_line,
            y_line,
            c=range(len(q_line)),
            cmap="viridis",
            s=200,
            edgecolors="black",
        )

        # Connect line
        ax.plot(x_line, y_line, "b-", linewidth=2, alpha=0.5)

        # Mark start and end
        ax.scatter(x_line[0], y_line[0], c="green", s=100, marker="o", zorder=5)
        ax.scatter(x_line[-1], y_line[-1], c="red", s=100, marker="s", zorder=5)

        ax.set_title(f"{title}\n({len(q_line)} hexes)")
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("Hex Grid Line Drawing (Cube Lerp + Round)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "line_drawing_examples", suffix="png")
    plt.close()


def test_line_n_visualization():
    """Visualize line drawing with fixed number of steps.

    Shows how line_n interpolates between two endpoints with increasing
    resolution — useful for understanding sub-hex interpolation.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    q_bg, r_bg = disk(6)
    x_bg, y_bg = to_pixel(q_bg, r_bg)

    for n, ax in enumerate(axes):
        n_steps = n + 1  # 1 to 6 steps
        q_line, r_line = line_n(
            np.array([0]),
            np.array([0]),
            np.array([5]),
            np.array([3]),
            n=n_steps,
        )
        x_line, y_line = to_pixel(q_line, r_line)

        ax.scatter(x_bg, y_bg, c="lightgray", s=50, alpha=0.3)
        ax.scatter(x_line, y_line, c="blue", s=200, edgecolors="black")
        ax.plot(x_line, y_line, "b-", linewidth=2, alpha=0.5)

        ax.set_title(f"n={n_steps} steps ({len(q_line)} points)")
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("Line Drawing with Fixed Step Count", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "line_n_fixed_steps", suffix="png")
    plt.close()
