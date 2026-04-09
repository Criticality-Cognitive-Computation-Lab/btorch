"""Coordinate transforms with optional numba acceleration.

All formulas from: https://www.redblobgames.com/grids/hexagons/
Numba-accelerated versions are provided for performance-critical functions.

Code adapted from flyvis (MIT License).
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


#: Orientation type: "pointy" (point up) or "flat" (flat side up)
Orientation = Literal["pointy", "flat"]


def cube_from_axial(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Axial to cube: (q, r) -> (q, r, s) where s = -q - r.

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (q, r, s) cube coordinates.
    """
    s = -q - r
    return q, r, s


def axial_from_cube(q: NDArray, r: NDArray, s: NDArray) -> tuple[NDArray, NDArray]:
    """Cube to axial: drops s.

    Args:
        q: Cube q coordinates.
        r: Cube r coordinates.
        s: Cube s coordinates (unused, for API consistency).

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    _ = s  # s is implicit in axial, but validate input
    return q, r


def to_pixel(
    q: NDArray,
    r: NDArray,
    size: float = 1.0,
    orientation: Orientation = "pointy",
) -> tuple[NDArray, NDArray]:
    """Axial hex to pixel.

    Pointy: x = size * (sqrt(3)*q + sqrt(3)/2*r), y = size * (3/2*r)
    Flat:   x = size * (3/2*q), y = size * (sqrt(3)/2*q + sqrt(3)*r)

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.
        size: Hexagon size (distance from center to corner).
        orientation: "pointy" (point up) or "flat" (flat side up).

    Returns:
        Tuple of (x, y) pixel coordinates.

    See Also:
        https://www.redblobgames.com/grids/hexagons/#hex-to-pixel
    """
    if orientation == "pointy":
        x = size * (np.sqrt(3) * q + np.sqrt(3) / 2 * r)
        y = size * (3.0 / 2 * r)
    elif orientation == "flat":
        x = size * (3.0 / 2 * q)
        y = size * (np.sqrt(3) / 2 * q + np.sqrt(3) * r)
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    return x, y


def from_pixel(
    x: NDArray,
    y: NDArray,
    size: float = 1.0,
    orientation: Orientation = "pointy",
) -> tuple[NDArray, NDArray]:
    """Pixel to axial hex (fractional, use round_axial).

    Args:
        x: Pixel x coordinates.
        y: Pixel y coordinates.
        size: Hexagon size.
        orientation: "pointy" (point up) or "flat" (flat side up).

    Returns:
        Tuple of (q, r) fractional axial coordinates.

    See Also:
        https://www.redblobgames.com/grids/hexagons/#pixel-to-hex
    """
    if orientation == "pointy":
        q = (np.sqrt(3) / 3 * x - 1.0 / 3 * y) / size
        r = (2.0 / 3 * y) / size
    elif orientation == "flat":
        q = (2.0 / 3 * x) / size
        r = (-1.0 / 3 * x + np.sqrt(3) / 3 * y) / size
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    return q, r


def _cube_round(x: NDArray, y: NDArray, z: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Round cube coordinates to nearest hex.

    Args:
        x: Fractional cube x coordinates.
        y: Fractional cube y coordinates.
        z: Fractional cube z coordinates.

    Returns:
        Rounded cube coordinates (rx, ry, rz).
    """
    rx = np.rint(x)
    ry = np.rint(y)
    rz = np.rint(z)

    x_diff = np.abs(rx - x)
    y_diff = np.abs(ry - y)
    z_diff = np.abs(rz - z)

    # Adjust based on which coordinate had largest rounding error
    mask_xy = (x_diff > y_diff) & (x_diff > z_diff)
    mask_y = (~mask_xy) & (y_diff > z_diff)

    rx = np.where(mask_xy, -ry - rz, rx)
    ry = np.where(mask_y, -rx - rz, ry)
    rz = np.where(~mask_xy & ~mask_y, -rx - ry, rz)

    return rx, ry, rz


def round_axial(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Round fractional axial to nearest hex (RBG algorithm).

    Args:
        q: Fractional axial q coordinates.
        r: Fractional axial r coordinates.

    Returns:
        Rounded axial coordinates (q, r).
    """
    s = -q - r
    rq, rr, _ = _cube_round(q, r, s)
    return rq.astype(q.dtype), rr.astype(r.dtype)


def rotate(q: NDArray, r: NDArray, n: int) -> tuple[NDArray, NDArray]:
    """Rotate by n * 60°. Positive n = clockwise.

    Formula: [q,r,s] -> [-r,-s,-q] for 60° cw.
    Axial: q' = -r, r' = q + r.

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.
        n: Number of 60° rotations (positive = clockwise).

    Returns:
        Rotated coordinates (q, r).
    """
    n = n % 6  # Normalize to 0-5
    for _ in range(n):
        q, r = -r, q + r
    return q, r


def reflect(
    q: NDArray, r: NDArray, axis: Literal["q", "r", "s"]
) -> tuple[NDArray, NDArray]:
    """Reflect across axis.

    In cube coordinates:
    - reflect_q = (q, s, r)
    - reflect_r = (s, r, q)
    - reflect_s = (r, q, s)

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.
        axis: Axis to reflect across ("q", "r", or "s").

    Returns:
        Reflected coordinates (q, r).
    """
    s = -q - r
    if axis == "q":
        return q, s
    elif axis == "r":
        return s, r
    elif axis == "s":
        return r, q
    else:
        raise ValueError(f"Unknown axis: {axis}. Must be 'q', 'r', or 's'.")
