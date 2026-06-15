"""Line drawing on hex grids.

Algorithm: cube lerp + round, then deduplicate.
Reference: https://www.redblobgames.com/grids/hexagons/#line-drawing
"""

import numpy as np

from .transform import round_axial


def line(
    q1: np.ndarray,
    r1: np.ndarray,
    q2: np.ndarray,
    r2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    q1 = np.atleast_1d(q1)
    r1 = np.atleast_1d(r1)
    q2 = np.atleast_1d(q2)
    r2 = np.atleast_1d(r2)

    from .distance import distance

    dist = int(distance(q1, r1, q2, r2)[0])
    n_steps = dist + 1

    if n_steps <= 1:
        return q1.copy(), r1.copy()

    t = np.linspace(0, 1, n_steps)
    qf = q1[0] * (1 - t) + q2[0] * t
    rf = r1[0] * (1 - t) + r2[0] * t

    rq, rr = round_axial(qf, rf)

    return _deduplicate(rq, rr)


def line_n(
    q1: np.ndarray, r1: np.ndarray, q2: np.ndarray, r2: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
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

    t = np.linspace(0, 1, n + 1)
    qf = q1[0] * (1 - t) + q2[0] * t
    rf = r1[0] * (1 - t) + r2[0] * t

    rq, rr = round_axial(qf, rf)

    return _deduplicate(rq, rr)


def _deduplicate(q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate consecutive coordinates.

    Args:
        q: Q coordinates.
        r: R coordinates.

    Returns:
        Tuple of deduplicated (q, r) arrays.
    """
    if len(q) == 0:
        return q, r

    mask = np.ones(len(q), dtype=bool)
    mask[1:] = (q[1:] != q[:-1]) | (r[1:] != r[:-1])

    return q[mask], r[mask]
