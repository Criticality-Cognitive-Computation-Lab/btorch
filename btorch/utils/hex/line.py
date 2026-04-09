"""Line drawing on hex grids.

Algorithm: cube lerp + round, then deduplicate.
Reference: https://www.redblobgames.com/grids/hexagons/#line-drawing
"""

import numpy as np
from numpy.typing import NDArray


def line(q1: NDArray, r1: NDArray, q2: NDArray, r2: NDArray) -> tuple[NDArray, NDArray]:
    """Draw line from (q1,r1) to (q2,r2) using cube lerp + round.

    Uses Red Blob Games algorithm:
    1. Convert to cube coordinates
    2. Lerp between endpoints in cube space
    3. Round each interpolated point to nearest hex
    4. Remove duplicates

    Args:
        q1: Start q coordinate(s).
        r1: Start r coordinate(s).
        q2: End q coordinate(s).
        r2: End r coordinate(s).

    Returns:
        Tuple of (q, r) arrays for hexes on the line.

    Example:
        >>> q, r = line(np.array([0]), np.array([0]), np.array([3]), np.array([3]))
        >>> len(q)  # Distance is 3, so 4 points including endpoints
        4
    """
    # Ensure scalars are arrays
    q1 = np.atleast_1d(q1)
    r1 = np.atleast_1d(r1)
    q2 = np.atleast_1d(q2)
    r2 = np.atleast_1d(r2)

    # Calculate distance for number of steps
    from .distance import distance

    dist = int(distance(q1, r1, q2, r2)[0])
    n_steps = dist + 1

    if n_steps <= 1:
        return q1.copy(), r1.copy()

    # Convert to cube
    s1 = -q1 - r1
    s2 = -q2 - r2

    # Lerp and round
    q_points, r_points = [], []
    for i in range(n_steps):
        t = i / dist if dist > 0 else 0
        qf = q1[0] * (1 - t) + q2[0] * t
        rf = r1[0] * (1 - t) + r2[0] * t
        sf = s1[0] * (1 - t) + s2[0] * t

        # Round to nearest hex
        rq, rr, _ = _cube_round(qf, rf, sf)
        q_points.append(rq)
        r_points.append(rr)

    # Remove duplicates
    result_q, result_r = _deduplicate(np.array(q_points), np.array(r_points))

    return result_q, result_r


def line_n(
    q1: NDArray, r1: NDArray, q2: NDArray, r2: NDArray, n: int
) -> tuple[NDArray, NDArray]:
    """Line with exactly n+1 points including endpoints.

    Args:
        q1: Start q coordinate(s).
        r1: Start r coordinate(s).
        q2: End q coordinate(s).
        r2: End r coordinate(s).
        n: Number of steps (results in n+1 points).

    Returns:
        Tuple of (q, r) arrays for hexes on the line.
    """
    q1 = np.atleast_1d(q1)
    r1 = np.atleast_1d(r1)
    q2 = np.atleast_1d(q2)
    r2 = np.atleast_1d(r2)

    if n <= 0:
        return q1.copy(), r1.copy()

    # Convert to cube
    s1 = -q1 - r1
    s2 = -q2 - r2

    # Lerp with exactly n+1 points
    q_points, r_points = [], []
    for i in range(n + 1):
        t = i / n if n > 0 else 0
        qf = q1[0] * (1 - t) + q2[0] * t
        rf = r1[0] * (1 - t) + r2[0] * t
        sf = s1[0] * (1 - t) + s2[0] * t

        rq, rr, _ = _cube_round(qf, rf, sf)
        q_points.append(rq)
        r_points.append(rr)

    result_q, result_r = _deduplicate(np.array(q_points), np.array(r_points))

    return result_q, result_r


def _cube_round(x: float, y: float, z: float) -> tuple[int, int, int]:
    """Round cube coordinates to nearest hex.

    Args:
        x: Fractional cube x (q) coordinate.
        y: Fractional cube y (r) coordinate.
        z: Fractional cube z (s) coordinate.

    Returns:
        Rounded cube coordinates (rx, ry, rz).
    """
    rx = round(x)
    ry = round(y)
    rz = round(z)

    x_diff = abs(rx - x)
    y_diff = abs(ry - y)
    z_diff = abs(rz - z)

    # Adjust based on which coordinate had largest rounding error
    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return int(rx), int(ry), int(rz)


def _deduplicate(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Remove duplicate consecutive coordinates.

    Args:
        q: Q coordinates.
        r: R coordinates.

    Returns:
        Tuple of deduplicated (q, r) arrays.
    """
    if len(q) == 0:
        return q, r

    # Find indices where coordinates change
    mask = np.ones(len(q), dtype=bool)
    mask[1:] = (q[1:] != q[:-1]) | (r[1:] != r[:-1])

    return q[mask], r[mask]
