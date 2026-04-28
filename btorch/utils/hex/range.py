"""Coordinate range operations.

Provides intersection and union operations for hexagonal ranges (disks).

Reference: https://www.redblobgames.com/grids/hexagons/#range-intersection
"""

import numpy as np
from numpy.typing import NDArray


def range_intersection(
    centers: list[tuple[int, int]], radii: list[int]
) -> tuple[NDArray, NDArray]:
    """Hexes that are within ALL of the given ranges.

    Uses the algebraic approach from Red Blob Games:
    - Each range is: center-N <= q <= center+N, etc.
    - Intersection is: max(lows) <= q <= min(highs)

    Args:
        centers: List of (q, r) center coordinates.
        radii: List of radii (one per center).

    Returns:
        Tuple of (q, r) arrays for hexes in the intersection.

    Example:
        >>> # Two overlapping disks
        >>> centers = [(0, 0), (3, 0)]
        >>> radii = [2, 2]
        >>> q, r = range_intersection(centers, radii)
        >>> len(q)  # Hexes within both disks
        3
    """
    if len(centers) != len(radii):
        raise ValueError("centers and radii must have same length")

    if not centers:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Convert to cube for range calculation
    centers_cube = []
    for q, r in centers:
        s = -q - r
        centers_cube.append((q, r, s))

    # Find intersection bounds in cube coordinates
    q_min = max(c[0] - rad for c, rad in zip(centers_cube, radii))
    q_max = min(c[0] + rad for c, rad in zip(centers_cube, radii))
    r_min = max(c[1] - rad for c, rad in zip(centers_cube, radii))
    r_max = min(c[1] + rad for c, rad in zip(centers_cube, radii))
    s_min = max(c[2] - rad for c, rad in zip(centers_cube, radii))
    s_max = min(c[2] + rad for c, rad in zip(centers_cube, radii))

    # Generate hexes within intersection
    q_list, r_list = [], []
    for q in range(q_min, q_max + 1):
        for r in range(r_min, r_max + 1):
            s = -q - r
            if s_min <= s <= s_max:
                q_list.append(q)
                r_list.append(r)

    return np.array(q_list, dtype=int), np.array(r_list, dtype=int)


def range_union(
    centers: list[tuple[int, int]], radii: list[int]
) -> tuple[NDArray, NDArray]:
    """Hexes that are within ANY of the given ranges.

    Args:
        centers: List of (q, r) center coordinates.
        radii: List of radii (one per center).

    Returns:
        Tuple of (q, r) arrays for hexes in the union.

    Example:
        >>> # Two overlapping disks
        >>> centers = [(0, 0), (5, 0)]
        >>> radii = [2, 2]
        >>> q, r = range_union(centers, radii)
        >>> len(q)  # Hexes in either disk (overlap counted once)
        19
    """
    if len(centers) != len(radii):
        raise ValueError("centers and radii must have same length")

    if not centers:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Collect all hexes from each range
    all_q, all_r = [], []
    from .coords import disk

    for (q, r), rad in zip(centers, radii):
        dq, dr = disk(rad, q, r)
        all_q.extend(dq.tolist())
        all_r.extend(dr.tolist())

    # Remove duplicates using unique
    all_coords = np.column_stack([all_q, all_r])
    unique_coords = np.unique(all_coords, axis=0)

    if len(unique_coords) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    return unique_coords[:, 0], unique_coords[:, 1]


def ranges_intersect(
    center1: tuple[int, int], radius1: int, center2: tuple[int, int], radius2: int
) -> bool:
    """Check if two hex ranges intersect.

    Args:
        center1: First center (q, r).
        radius1: First radius.
        center2: Second center (q, r).
        radius2: Second radius.

    Returns:
        True if the ranges intersect, False otherwise.

    Example:
        >>> ranges_intersect((0, 0), 2, (3, 0), 2)
        True
        >>> ranges_intersect((0, 0), 1, (5, 0), 1)
        False
    """
    from .distance import distance

    q1, r1 = center1
    q2, r2 = center2
    dist = distance(np.array([q1]), np.array([r1]), np.array([q2]), np.array([r2]))[0]

    return bool(dist <= radius1 + radius2)
