"""Distance calculations.

Reference: https://www.redblobgames.com/grids/hexagons/#distances
"""

import numpy as np


def distance(
    q1: np.ndarray,
    r1: np.ndarray,
    q2: np.ndarray,
    r2: np.ndarray,
) -> np.ndarray:
    """Hex distance. Formula: max(|dq|, |dr|, |ds|) where ds = -(dq+dr).

    Args:
        q1: First set q coordinates.
        r1: First set r coordinates.
        q2: Second set q coordinates.
        r2: Second set r coordinates.

    Returns:
        Distance between coordinates.
    """
    dq = q1 - q2
    dr = r1 - r2
    ds = -(dq + dr)
    return np.maximum(np.maximum(np.abs(dq), np.abs(dr)), np.abs(ds))


def radius(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Distance from origin.

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Distance from origin for each coordinate.
    """
    return distance(q, r, 0, 0)


def within_range(
    q: np.ndarray, r: np.ndarray, center_q: int, center_r: int, n: int
) -> np.ndarray:
    """Boolean mask for coords within n steps of center.

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.
        center_q: Center q coordinate.
        center_r: Center r coordinate.
        n: Maximum distance from center.

    Returns:
        Boolean mask for coordinates within range.
    """
    return distance(q, r, center_q, center_r) <= n


def mask(q: np.ndarray, r: np.ndarray, max_radius: int) -> np.ndarray:
    """Boolean mask for coords within max_radius of origin.

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.
        max_radius: Maximum distance from origin.

    Returns:
        Boolean mask for coordinates within max_radius.
    """
    return radius(q, r) <= max_radius
