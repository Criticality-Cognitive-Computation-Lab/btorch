"""Tests for line drawing on hex grids.

Demonstrates the line drawing algorithm using cube lerp + round.
Reference: https://www.redblobgames.com/grids/hexagons/#line-drawing
"""

import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import save_fig
from btorch.utils.hex import line, line_n, to_pixel
from btorch.utils.hex.coords import disk


def test_line_basic():
    """Test basic line drawing between two points."""
    # Line from (0, 0) to (3, 3)
    q, r = line(np.array([0]), np.array([0]), np.array([3]), np.array([3]))

    # Distance is 3, so we expect 4 points including endpoints
    from btorch.utils.hex import distance

    dist = int(distance(np.array([0]), np.array([0]), np.array([3]), np.array([3]))[0])
    assert len(q) == dist + 1, f"Expected {dist + 1} points, got {len(q)}"

    # Verify start and end points
    assert q[0] == 0 and r[0] == 0, "Line should start at (0, 0)"
    assert q[-1] == 3 and r[-1] == 3, "Line should end at (3, 3)"


def test_line_visualization():
    """Visualize lines between various points in a hex grid."""
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


def test_line_n_fixed_steps():
    """Test line drawing with fixed number of steps."""
    # Draw line with exactly 5 points
    q, r = line_n(
        np.array([0]),
        np.array([0]),
        np.array([5]),
        np.array([0]),
        n=4,  # 5 points including endpoints
    )

    assert len(q) == 5, f"Expected 5 points, got {len(q)}"

    # Visualize different step counts
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    q_bg, r_bg = disk(6)
    x_bg, y_bg = to_pixel(q_bg, r_bg)

    for n, ax in enumerate(axes):
        n_steps = n + 1  # 1 to 6 steps
        q_line, r_line = line_n(
            np.array([0]), np.array([0]), np.array([5]), np.array([3]), n=n_steps
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


def test_line_distance_property():
    """Verify that line length equals distance + 1."""
    from btorch.utils.hex import distance

    # Test various distances
    test_cases = [
        (0, 0, 1, 0),
        (0, 0, 2, 1),
        (0, 0, 3, -2),
        (1, -1, 4, 2),
        (-2, 3, 2, -1),
    ]

    for q1, r1, q2, r2 in test_cases:
        q_line, r_line = line(
            np.array([q1]), np.array([r1]), np.array([q2]), np.array([r2])
        )
        dist = int(
            distance(np.array([q1]), np.array([r1]), np.array([q2]), np.array([r2]))[0]
        )

        # Line should have distance + 1 points (including endpoints)
        assert len(q_line) == dist + 1, (
            f"Line from ({q1},{r1}) to ({q2},{r2}): "
            f"expected {dist + 1} points, got {len(q_line)}"
        )


if __name__ == "__main__":
    test_line_basic()
    test_line_visualization()
    test_line_n_fixed_steps()
    test_line_distance_property()
    print("Line drawing tests passed with figures saved")
