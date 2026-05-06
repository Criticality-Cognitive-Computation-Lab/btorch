"""Doubled coordinate systems.

Doubled coordinates make rectangular maps easier to work with.
- Double-width: for pointy-top hexes (cols alternate between hexes)
- Double-height: for flat-top hexes (rows alternate between hexes)

Reference: https://www.redblobgames.com/grids/hexagons/#coordinates-doubled
"""

import numpy as np
from numpy.typing import NDArray


def axial_to_doublewidth(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Convert axial to double-width coordinates.

    Formula: col = 2*q + r, row = r

    Double-width is useful for pointy-top hexes in rectangular maps.
    Every other column is used (odd columns are skipped).

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (col, row) double-width coordinates.

    Example:
        >>> col, row = axial_to_doublewidth(np.array([0, 1]), np.array([0, 1]))
        >>> col
        array([0, 3])
    """
    q = np.asarray(q)
    r = np.asarray(r)
    col = 2 * q + r
    row = r
    return col, row


def doublewidth_to_axial(col: NDArray, row: NDArray) -> tuple[NDArray, NDArray]:
    """Convert double-width to axial coordinates.

    Formula: q = (col - row) / 2, r = row

    Args:
        col: Double-width column coordinates.
        row: Double-width row coordinates.

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    col = np.asarray(col)
    row = np.asarray(row)
    q = (col - row) // 2
    r = row
    return q, r


def axial_to_doubleheight(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Convert axial to double-height coordinates.

    Formula: col = q, row = 2*r + q

    Double-height is useful for flat-top hexes in rectangular maps.
    Every other row is used (odd rows are skipped).

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (col, row) double-height coordinates.

    Example:
        >>> col, row = axial_to_doubleheight(np.array([0, 1]), np.array([0, 1]))
        >>> row
        array([0, 3])
    """
    q = np.asarray(q)
    r = np.asarray(r)
    col = q
    row = 2 * r + q
    return col, row


def doubleheight_to_axial(col: NDArray, row: NDArray) -> tuple[NDArray, NDArray]:
    """Convert double-height to axial coordinates.

    Formula: q = col, r = (row - col) / 2

    Args:
        col: Double-height column coordinates.
        row: Double-height row coordinates.

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    col = np.asarray(col)
    row = np.asarray(row)
    q = col
    r = (row - col) // 2
    return q, r


def doublewidth_distance(c1: int, r1: int, c2: int, r2: int) -> int:
    """Distance in double-width coordinates (direct formula).

    Formula: drow + max(0, (dcol-drow)/2) where drow = |r2-r1|,
    dcol = |c2-c1|

    Args:
        c1: First column coordinate.
        r1: First row coordinate.
        c2: Second column coordinate.
        r2: Second row coordinate.

    Returns:
        Hex distance between the two coordinates.

    Example:
        >>> doublewidth_distance(0, 0, 2, 0)
        1
    """
    dcol = abs(c2 - c1)
    drow = abs(r2 - r1)
    return drow + max(0, (dcol - drow) // 2)


def doubleheight_distance(c1: int, r1: int, c2: int, r2: int) -> int:
    """Distance in double-height coordinates (direct formula).

    Formula: dcol + max(0, (drow-dcol)/2) where dcol = |c2-c1|,
    drow = |r2-r1|

    Args:
        c1: First column coordinate.
        r1: First row coordinate.
        c2: Second column coordinate.
        r2: Second row coordinate.

    Returns:
        Hex distance between the two coordinates.

    Example:
        >>> doubleheight_distance(0, 0, 0, 2)
        1
    """
    dcol = abs(c2 - c1)
    drow = abs(r2 - r1)
    return dcol + max(0, (drow - dcol) // 2)


def doublewidth_to_pixel(
    col: NDArray, row: NDArray, size: float = 1.0
) -> tuple[NDArray, NDArray]:
    """Convert double-width to pixel coordinates.

    Formula: x = sqrt(3)/2 * size * col, y = 3/2 * size * row

    Args:
        col: Double-width column coordinates.
        row: Double-width row coordinates.
        size: Hexagon size (distance from center to corner).

    Returns:
        Tuple of (x, y) pixel coordinates.
    """
    col = np.asarray(col)
    row = np.asarray(row)
    x = np.sqrt(3) / 2 * size * col
    y = 3.0 / 2 * size * row
    return x, y


def doubleheight_to_pixel(
    col: NDArray, row: NDArray, size: float = 1.0
) -> tuple[NDArray, NDArray]:
    """Convert double-height to pixel coordinates.

    Formula: x = 3/2 * size * col, y = sqrt(3)/2 * size * row

    Args:
        col: Double-height column coordinates.
        row: Double-height row coordinates.
        size: Hexagon size (distance from center to corner).

    Returns:
        Tuple of (x, y) pixel coordinates.
    """
    col = np.asarray(col)
    row = np.asarray(row)
    x = 3.0 / 2 * size * col
    y = np.sqrt(3) / 2 * size * row
    return x, y


def pixel_to_doublewidth(
    x: NDArray, y: NDArray, size: float = 1.0
) -> tuple[NDArray, NDArray]:
    """Convert pixel to double-width coordinates.

    Args:
        x: Pixel x coordinates.
        y: Pixel y coordinates.
        size: Hexagon size.

    Returns:
        Tuple of (col, row) double-width coordinates.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    col = (2 * x) / (np.sqrt(3) * size)
    row = (2 * y) / (3 * size)
    return col, row


def pixel_to_doubleheight(
    x: NDArray, y: NDArray, size: float = 1.0
) -> tuple[NDArray, NDArray]:
    """Convert pixel to double-height coordinates.

    Args:
        x: Pixel x coordinates.
        y: Pixel y coordinates.
        size: Hexagon size.

    Returns:
        Tuple of (col, row) double-height coordinates.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    col = (2 * x) / (3 * size)
    row = (2 * y) / (np.sqrt(3) * size)
    return col, row
