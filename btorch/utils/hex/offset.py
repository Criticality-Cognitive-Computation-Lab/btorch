"""Offset coordinate systems for rectangular map storage.

Offset coordinates are commonly used for storing hex grids in 2D arrays.
They add an offset to either rows or columns based on parity (odd/even).

Reference: https://www.redblobgames.com/grids/hexagons/#coordinates-offset
"""

import numpy as np
from numpy.typing import NDArray


# Pointy-top hexes (rows are horizontal, offset varies by row)
def axial_to_odd_r(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Convert axial to odd-r offset coordinates.

    Formula: col = q + (r - (r & 1)) / 2, row = r

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (col, row) offset coordinates.

    Example:
        >>> col, row = axial_to_odd_r(np.array([0, 1]), np.array([0, 1]))
        >>> col
        array([0, 1])
    """
    q = np.asarray(q)
    r = np.asarray(r)
    col = q + (r - (r & 1)) // 2
    row = r
    return col, row


def odd_r_to_axial(col: NDArray, row: NDArray) -> tuple[NDArray, NDArray]:
    """Convert odd-r offset to axial coordinates.

    Formula: q = col - (row - (row & 1)) / 2, r = row

    Args:
        col: Column coordinates.
        row: Row coordinates.

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    col = np.asarray(col)
    row = np.asarray(row)
    q = col - (row - (row & 1)) // 2
    r = row
    return q, r


def axial_to_even_r(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Convert axial to even-r offset coordinates.

    Formula: col = q + (r + (r & 1)) / 2, row = r

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (col, row) offset coordinates.
    """
    q = np.asarray(q)
    r = np.asarray(r)
    col = q + (r + (r & 1)) // 2
    row = r
    return col, row


def even_r_to_axial(col: NDArray, row: NDArray) -> tuple[NDArray, NDArray]:
    """Convert even-r offset to axial coordinates.

    Formula: q = col - (row + (row & 1)) / 2, r = row

    Args:
        col: Column coordinates.
        row: Row coordinates.

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    col = np.asarray(col)
    row = np.asarray(row)
    q = col - (row + (row & 1)) // 2
    r = row
    return q, r


# Flat-top hexes (columns are vertical, offset varies by column)
def axial_to_odd_q(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Convert axial to odd-q offset coordinates.

    Formula: col = q, row = r + (q - (q & 1)) / 2

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (col, row) offset coordinates.
    """
    q = np.asarray(q)
    r = np.asarray(r)
    col = q
    row = r + (q - (q & 1)) // 2
    return col, row


def odd_q_to_axial(col: NDArray, row: NDArray) -> tuple[NDArray, NDArray]:
    """Convert odd-q offset to axial coordinates.

    Formula: q = col, r = row - (col - (col & 1)) / 2

    Args:
        col: Column coordinates.
        row: Row coordinates.

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    col = np.asarray(col)
    row = np.asarray(row)
    q = col
    r = row - (col - (col & 1)) // 2
    return q, r


def axial_to_even_q(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Convert axial to even-q offset coordinates.

    Formula: col = q, row = r + (q + (q & 1)) / 2

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (col, row) offset coordinates.
    """
    q = np.asarray(q)
    r = np.asarray(r)
    col = q
    row = r + (q + (q & 1)) // 2
    return col, row


def even_q_to_axial(col: NDArray, row: NDArray) -> tuple[NDArray, NDArray]:
    """Convert even-q offset to axial coordinates.

    Formula: q = col, r = row - (col + (col & 1)) / 2

    Args:
        col: Column coordinates.
        row: Row coordinates.

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    col = np.asarray(col)
    row = np.asarray(row)
    q = col
    r = row - (col + (col & 1)) // 2
    return q, r


def axial_to_zigzag(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Convert axial to zigzag offset coordinates (column-skipped layout).

    This is the display-friendly version where x alternates cleanly
    between columns, creating the classic hex zigzag pattern:

        col 0   col 1   col 0   col 1
          ●       ●       ●       ●
        ●       ●       ●       ●

    Formula: x = floor((r - q) / 2), y = q + r

    The reverse is exact and bidirectional with integers.

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (x, y) zigzag offset coordinates.

    Example:
        >>> axial_to_zigzag(np.array([0, 1, -1]), np.array([0, 0, 0]))
        (array([0, 0, 0]), array([0, 1, -1]))

        >>> axial_to_zigzag(np.array([0, 1, 0, -1]), np.array([0, -1, 1, 0]))
        (array([0, -1, 0, 0]), array([0, 0, 1, -1]))
    """
    q = np.asarray(q)
    r = np.asarray(r)
    x = np.floor((r - q) / 2).astype(int)
    y = q + r
    return x, y


def zigzag_to_axial(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Convert zigzag offset coordinates back to axial.

    Formula: q = floor((y - 2*x) / 2), r = y - q

    Args:
        x: Zigzag column coordinates.
        y: Zigzag row coordinates.

    Returns:
        Tuple of (q, r) axial coordinates.

    Example:
        >>> zigzag_to_axial(np.array([0, 0, 0]), np.array([0, 1, -1]))
        (array([0, 1, -1]), array([0, 0, 0]))
    """
    x = np.asarray(x)
    y = np.asarray(y)
    q = np.floor((y - 2 * x) / 2).astype(int)
    r = y - q
    return q, r


# Note: For offset neighbor calculations, convert to axial coordinates first:
#   q, r = odd_r_to_axial(col, row)
#   qn, rn = neighbors(q, r)  # Get neighbors in axial
#   cn, rn = axial_to_odd_r(qn, rn)  # Convert back to offset
# This approach is simpler and less error-prone than using separate
# neighbor tables for each offset type.
