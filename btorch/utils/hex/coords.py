"""Hex coordinate generation.

Axial: (q, r) with s = -q - r implicit
Reference: https://www.redblobgames.com/grids/hexagons/

Code adapted from flyvis (MIT License) and Hexy (MIT License).
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def ring(radius: int, center_q: int = 0, center_r: int = 0) -> tuple[NDArray, NDArray]:
    """Hexes at exactly `radius` from center (6*radius hexes, or 1 if
    radius=0).

    Args:
        radius: Distance from center.
        center_q: Center q coordinate.
        center_r: Center r coordinate.

    Returns:
        Tuple of (q, r) arrays for hex coordinates on the ring.
    """
    if radius == 0:
        return np.array([center_q]), np.array([center_r])

    q, r = [], []
    # Start at radius steps in the -r direction from center
    q0, r0 = center_q, center_r - radius

    # Walk around the ring in 6 directions
    for dq, dr in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]:
        for _ in range(radius):
            q.append(q0)
            r.append(r0)
            q0 += dq
            r0 += dr

    return np.array(q), np.array(r)


def disk(radius: int, center_q: int = 0, center_r: int = 0) -> tuple[NDArray, NDArray]:
    """All hexes within `radius` of center. Count: 1 + 3*radius*(radius+1).

    Args:
        radius: Maximum distance from center.
        center_q: Center q coordinate.
        center_r: Center r coordinate.

    Returns:
        Tuple of (q, r) arrays for hex coordinates in the disk.

    See Also:
        https://www.redblobgames.com/grids/hexagons/#range-coordinate
    """
    q, r = [], []
    for dq in range(-radius, radius + 1):
        for dr in range(max(-radius, -radius - dq), min(radius, radius - dq) + 1):
            q.append(center_q + dq)
            r.append(center_r + dr)
    return np.array(q), np.array(r)


def spiral(
    radius: int, center_q: int = 0, center_r: int = 0
) -> tuple[NDArray, NDArray]:
    """Hexes in spiral order: center, ring1, ring2, ..., ring_radius.

    This ordering is stable under rotation (permutes by ring).

    Args:
        radius: Maximum distance from center.
        center_q: Center q coordinate.
        center_r: Center r coordinate.

    Returns:
        Tuple of (q, r) arrays in spiral order.
    """
    q_all, r_all = [center_q], [center_r]

    for r in range(1, radius + 1):
        qr, rr = ring(r, center_q, center_r)
        q_all.extend(qr.tolist())
        r_all.extend(rr.tolist())

    return np.array(q_all), np.array(r_all)


def rectangle(
    width: int, height: int, orientation: Literal["pointy", "flat"] = "pointy"
) -> tuple[NDArray, NDArray]:
    """Rectangular region in axial coords (slanted edges).

    Args:
        width: Width of rectangle.
        height: Height of rectangle.
        orientation: Hexagon orientation ("pointy" or "flat").

    Returns:
        Tuple of (q, r) arrays for hex coordinates in rectangle.
    """
    q, r = [], []
    for row in range(height):
        for col in range(width):
            if orientation == "pointy":
                # Offset odd rows
                q.append(col - (row // 2))
                r.append(row)
            else:  # flat
                # Offset odd columns
                q.append(col)
                r.append(row - (col // 2))
    return np.array(q), np.array(r)


def disk_count(radius: int) -> int:
    """Number of hexes in disk: 1 + 3*radius*(radius+1).

    Args:
        radius: Disk radius.

    Returns:
        Number of hexes in the disk.
    """
    return 1 + 3 * radius * (radius + 1)


def disk_radius(count: int) -> int:
    """Inverse of disk_count. Radius for given count.

    Args:
        count: Number of hexes.

    Returns:
        Radius of disk containing approximately count hexes.

    Note:
        Returns floor of exact value.
    """
    # Solve: count = 1 + 3*r*(r+1)
    # => 3*r^2 + 3*r + (1 - count) = 0
    # => r = (-3 + sqrt(9 - 12*(1-count))) / 6
    # => r = (-3 + sqrt(12*count - 3)) / 6
    if count < 1:
        return 0
    return int((-3 + np.sqrt(12 * count - 3)) / 6)
