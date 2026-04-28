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


def zigzag_to_pixel(
    x: NDArray, y: NDArray, size: float = 1.0
) -> tuple[NDArray, NDArray]:
    """Convert zigzag display coordinates to flat-top pixel space.

    FlyWire website-style lattice display is a staggered-column layout where
    every second column is vertically shifted by half a cell.

    Formula:
        pixel_x = 1.5 * size * x
        pixel_y = sqrt(3) * size * (y + 0.5 * (x mod 2))
    """
    x = np.asarray(x)
    y = np.asarray(y)
    parity = np.mod(x, 2)
    pixel_x = 1.5 * size * x
    pixel_y = np.sqrt(3.0) * size * (y + 0.5 * parity)
    return pixel_x.astype(float), pixel_y.astype(float)


def flywire_xy_to_pixel(
    x_zigzag: NDArray,
    y_zigzag: NDArray,
    size: float = 1.0,
    rotation_deg: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Convert FlyWire saved-page display indices `(x,y)` to pixel positions.

    Saved HTML/CSS shows the retina grid is rendered on DOM rows with:
    - row pitch of 24 px
    - tile step of 75 px
    - half-row indentation of 35 px on alternating rows

    Values are normalized here relative to `size=1.0`.
    """
    x_zigzag = np.asarray(x_zigzag)
    y_zigzag = np.asarray(y_zigzag)

    css_hex_width = 45.0
    css_full_step_x = 75.0
    css_half_row_shift_x = 35.0
    css_row_step_y = 24.0
    css_radius = css_hex_width / 2.0

    full_step_x = (css_full_step_x / css_radius) * size
    half_row_shift_x = (css_half_row_shift_x / css_radius) * size
    row_step_y = (css_row_step_y / css_radius) * size

    parity = np.mod(y_zigzag, 2)
    x = -(full_step_x * x_zigzag + half_row_shift_x * parity)
    y = -row_step_y * y_zigzag

    if rotation_deg == 0.0:
        return x.astype(float), y.astype(float)

    theta = np.deg2rad(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = cos_t * x - sin_t * y
    yr = sin_t * x + cos_t * y
    return xr.astype(float), yr.astype(float)


def flywire_to_pixel(
    p: NDArray,
    q: NDArray,
    size: float = 1.0,
    rotation_deg: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Convert FlyWire visual-column coords `(p,q)` to pixel coordinates.

    FlyWire visual columns map uses axial `(p,q)` data coordinates but displays
    them on DOM rows using derived zigzag indices:

    - `x = floor((q - p) / 2)`
    - `y = p + q`

    The saved HTML/CSS page then lays out those rows with an alternating
    half-row indentation and fixed row pitch. This helper mirrors that display
    logic so btorch plots can match the website layout.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    x_zigzag, y_zigzag = axial_to_zigzag(p, q)
    return flywire_xy_to_pixel(
        x_zigzag,
        y_zigzag,
        size=size,
        rotation_deg=rotation_deg,
    )


# Note: For offset neighbor calculations, convert to axial coordinates first:
#   q, r = odd_r_to_axial(col, row)
#   qn, rn = neighbors(q, r)  # Get neighbors in axial
#   cn, rn = axial_to_odd_r(qn, rn)  # Convert back to offset
# This approach is simpler and less error-prone than using separate
# neighbor tables for each offset type.
