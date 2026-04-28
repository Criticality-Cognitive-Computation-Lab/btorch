"""Neighbor operations.

Reference: https://www.redblobgames.com/grids/hexagons/#neighbors
"""

import numpy as np
from numpy.typing import NDArray


# Direction vectors (axial, pointy-top): NE, E, SE, SW, W, NW
DIRECTIONS: NDArray = np.array(
    [
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
    ]
)

# Diagonal directions (6 diagonals between the 6 cardinal directions)
DIAGONALS: NDArray = np.array(
    [
        [2, -1],  # NE-E (between NE and E)
        [1, -2],  # E-SE (between E and SE)
        [-1, -1],  # SE-SW (between SE and SW)
        [-2, 1],  # SW-W (between SW and W)
        [-1, 2],  # W-NW (between W and NW)
        [1, 1],  # NW-NE (between NW and NE)
    ]
)


def neighbor(q: NDArray, r: NDArray, direction: int) -> tuple[NDArray, NDArray]:
    """Get neighbor in direction (0-5).

    Directions (axial, pointy-top):
    0: NE, 1: E, 2: SE, 3: SW, 4: W, 5: NW

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.
        direction: Direction index (0-5).

    Returns:
        Neighbor coordinates (q, r).
    """
    dq, dr = DIRECTIONS[direction % 6]
    return q + dq, r + dr


def neighbors(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Get all 6 neighbors.

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (q_neighbors, r_neighbors) arrays with shape (6, len(q)).
    """
    qn = q[None, :] + DIRECTIONS[:, 0:1]
    rn = r[None, :] + DIRECTIONS[:, 1:2]
    return qn, rn


def diagonal_neighbor(
    q: NDArray, r: NDArray, direction: int
) -> tuple[NDArray, NDArray]:
    """Get diagonal neighbor in direction (0-5).

    Diagonal directions are between the 6 cardinal directions:
    0: NE-E, 1: E-SE, 2: SE-SW, 3: SW-W, 4: W-NW, 5: NW-NE

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.
        direction: Direction index (0-5).

    Returns:
        Diagonal neighbor coordinates (q, r).
    """
    dq, dr = DIAGONALS[direction % 6]
    return q + dq, r + dr


def diagonal_neighbors(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Get all 6 diagonal neighbors.

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (q_neighbors, r_neighbors) arrays with shape (6, len(q)).
    """
    qn = q[None, :] + DIAGONALS[:, 0:1]
    rn = r[None, :] + DIAGONALS[:, 1:2]
    return qn, rn


def all_neighbors(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Get all 12 neighbors (6 cardinal + 6 diagonal).

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (q_neighbors, r_neighbors) arrays with shape (12, len(q)).
    """
    cardinal_q = q[None, :] + DIRECTIONS[:, 0:1]
    cardinal_r = r[None, :] + DIRECTIONS[:, 1:2]
    diagonal_q = q[None, :] + DIAGONALS[:, 0:1]
    diagonal_r = r[None, :] + DIAGONALS[:, 1:2]

    qn = np.vstack([cardinal_q, diagonal_q])
    rn = np.vstack([cardinal_r, diagonal_r])
    return qn, rn
