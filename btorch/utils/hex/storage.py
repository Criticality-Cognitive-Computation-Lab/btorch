"""Data alignment and indexing with optional numba acceleration.

Code adapted from flyvis (MIT License).
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ...config import HAS_NUMBA, njit


def align(
    q_target: NDArray,
    r_target: NDArray,
    q_source: NDArray,
    r_source: NDArray,
    values: NDArray,
    fill: float = np.nan,
    use_numba: bool = True,
) -> NDArray:
    """Align source values to target coordinates. Missing -> fill.

    Uses numba-accelerated implementation if available and use_numba=True.

    Args:
        q_target: Target q coordinates.
        r_target: Target r coordinates.
        q_source: Source q coordinates.
        r_source: Source r coordinates.
        values: Source values to align.
        fill: Fill value for missing coordinates.
        use_numba: Whether to use numba acceleration if available.

    Returns:
        Aligned values with same length as target coordinates.
    """
    # Numba implementation only supports 1D values or 2D values (features, n_hexes)
    # Also fall back to numpy for higher dimensional arrays or when dtype promotion
    # is needed (fill=nan with integer dtype)
    needs_dtype_promotion = np.isnan(fill) and not np.issubdtype(
        values.dtype, np.floating
    )
    can_use_numba = (
        use_numba and HAS_NUMBA and values.ndim <= 2 and not needs_dtype_promotion
    )
    if can_use_numba:
        return _align_numba(q_target, r_target, q_source, r_source, values, fill)
    else:
        return _align_numpy(q_target, r_target, q_source, r_source, values, fill)


def _align_numpy(
    q_target: np.ndarray,
    r_target: np.ndarray,
    q_source: np.ndarray,
    r_source: np.ndarray,
    values: np.ndarray,
    fill: float,
) -> np.ndarray:
    """NumPy implementation of alignment."""
    # Create lookup from (q, r) to index
    source_lookup = {
        (int(q), int(r)): i for i, (q, r) in enumerate(zip(q_source, r_source))
    }

    # Build result array
    result_shape = values.shape[:-1] + (len(q_target),)

    # Handle fill=nan with integer dtype: promote to float
    result_dtype = values.dtype
    if np.isnan(fill) and not np.issubdtype(values.dtype, np.floating):
        result_dtype = np.float64

    result = np.full(result_shape, fill, dtype=result_dtype)

    for i, (qt, rt) in enumerate(zip(q_target, r_target)):
        key = (int(qt), int(rt))
        if key in source_lookup:
            result[..., i] = values[..., source_lookup[key]]

    return result


@njit(cache=True)
def _align_numba(
    q_target: np.ndarray,
    r_target: np.ndarray,
    q_source: np.ndarray,
    r_source: np.ndarray,
    values: np.ndarray,
    fill: float,
) -> np.ndarray:
    """Numba-accelerated alignment implementation."""
    n_target = len(q_target)
    n_source = len(q_source)

    # Build hash map from source coordinates to indices
    # Use dictionary for coordinate lookup
    coord_to_idx = {}
    for i in range(n_source):
        coord_to_idx[(int(q_source[i]), int(r_source[i]))] = i

    # Allocate output
    if values.ndim > 1:
        result = np.full((values.shape[0], n_target), fill, dtype=values.dtype)
        for i in range(n_target):
            key = (int(q_target[i]), int(r_target[i]))
            if key in coord_to_idx:
                idx = coord_to_idx[key]
                for j in range(values.shape[0]):
                    result[j, i] = values[j, idx]
    else:
        result = np.full(n_target, fill, dtype=values.dtype)
        for i in range(n_target):
            key = (int(q_target[i]), int(r_target[i]))
            if key in coord_to_idx:
                result[i] = values[coord_to_idx[key]]

    return result


def permute(radius: int, n_rot: int) -> NDArray:
    """Permutation index for rotating spiral-ordered data by n_rot*60°.

    Args:
        radius: Grid radius.
        n_rot: Number of 60° rotations (positive = clockwise).

    Returns:
        Permutation indices for rotating data.
    """
    from .coords import spiral
    from .transform import rotate

    # Get spiral-ordered coordinates
    q, r = spiral(radius)

    # Rotate coordinates
    qr, rr = rotate(q, r, n_rot)

    # Find permutation: for each target, find source index
    # Create lookup
    coord_to_idx = {(int(q[i]), int(r[i])): i for i in range(len(q))}

    # Build permutation
    perm = np.zeros(len(q), dtype=np.int64)
    for i, (qt, rt) in enumerate(zip(qr, rr)):
        key = (int(qt), int(rt))
        if key in coord_to_idx:
            perm[i] = coord_to_idx[key]
        else:
            # This shouldn't happen for spiral ordering
            perm[i] = i

    return perm


def reflect_index(radius: int, axis: Literal["q", "r", "s"]) -> NDArray:
    """Permutation index for reflecting spiral-ordered data.

    Args:
        radius: Grid radius.
        axis: Axis to reflect across ("q", "r", or "s").

    Returns:
        Permutation indices for reflecting data.
    """
    from .coords import spiral
    from .transform import reflect

    # Get spiral-ordered coordinates
    q, r = spiral(radius)

    # Reflect coordinates
    qr, rr = reflect(q, r, axis)

    # Find permutation
    coord_to_idx = {(int(q[i]), int(r[i])): i for i in range(len(q))}

    perm = np.zeros(len(q), dtype=np.int64)
    for i, (qt, rt) in enumerate(zip(qr, rr)):
        key = (int(qt), int(rt))
        if key in coord_to_idx:
            perm[i] = coord_to_idx[key]
        else:
            perm[i] = i

    return perm


# Map storage layout functions
# Reference: https://www.redblobgames.com/grids/hexagons/#map-storage


def axial_to_rect_index(
    q: NDArray, r: NDArray, orientation: str = "pointy"
) -> tuple[NDArray, NDArray]:
    """Convert axial to rectangular array indices.

    For pointy-top: store at array[r][q + floor(r/2)]
    For flat-top: store at array[q][r + floor(q/2)]

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.
        orientation: "pointy" or "flat".

    Returns:
        Tuple of (row, col) array indices.

    Example:
        >>> row, col = axial_to_rect_index(np.array([0, 1]), np.array([0, 1]))
        >>> row
        array([0, 1])
    """
    q = np.asarray(q)
    r = np.asarray(r)

    if orientation == "pointy":
        row = r
        col = q + np.floor(r / 2).astype(int)
    elif orientation == "flat":
        row = q
        col = r + np.floor(q / 2).astype(int)
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    return row, col


def rect_index_to_axial(
    row: NDArray, col: NDArray, orientation: str = "pointy"
) -> tuple[NDArray, NDArray]:
    """Convert rectangular array indices back to axial.

    Args:
        row: Array row indices.
        col: Array column indices.
        orientation: "pointy" or "flat".

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    row = np.asarray(row)
    col = np.asarray(col)

    if orientation == "pointy":
        r = row
        q = col - np.floor(row / 2).astype(int)
    elif orientation == "flat":
        q = row
        r = col - np.floor(row / 2).astype(int)
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    return q, r


def axial_to_hex_index(q: NDArray, r: NDArray, radius: int) -> tuple[NDArray, NDArray]:
    """Convert axial to array indices for hexagon-shaped map.

    Row r (relative to center) has size 2*N+1 - abs(N-r) columns.
    Store at array[r + N][q - max(0, N-r) + N]

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.
        radius: Map radius (N).

    Returns:
        Tuple of (row, col) array indices.
    """
    q = np.asarray(q)
    r = np.asarray(r)

    row = r + radius
    col = q - np.maximum(0, radius - r) + radius

    return row, col


def hex_index_to_axial(
    row: NDArray, col: NDArray, radius: int
) -> tuple[NDArray, NDArray]:
    """Convert array indices back to axial for hexagon-shaped map.

    Args:
        row: Array row indices.
        col: Array column indices.
        radius: Map radius (N).

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    row = np.asarray(row)
    col = np.asarray(col)

    r = row - radius
    q = col + np.maximum(0, radius - r) - radius

    return q, r


def axial_to_triangle_index(
    q: NDArray, r: NDArray, size: int, pointing: str = "down"
) -> tuple[NDArray, NDArray]:
    """Convert axial to array indices for triangle-shaped map.

    Down-pointing: store at array[r][q], row r has size N+1-r
    Up-pointing: store at array[r][q - N+1+r], row r has size 1+r

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.
        size: Triangle size (N).
        pointing: "down" or "up".

    Returns:
        Tuple of (row, col) array indices.
    """
    q = np.asarray(q)
    r = np.asarray(r)

    if pointing == "down":
        row = r
        col = q
    elif pointing == "up":
        row = r
        col = q - size + 1 + r
    else:
        raise ValueError(f"Unknown pointing: {pointing}")

    return row, col


def triangle_index_to_axial(
    row: NDArray, col: NDArray, size: int, pointing: str = "down"
) -> tuple[NDArray, NDArray]:
    """Convert array indices back to axial for triangle-shaped map.

    Args:
        row: Array row indices.
        col: Array column indices.
        size: Triangle size (N).
        pointing: "down" or "up".

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    row = np.asarray(row)
    col = np.asarray(col)

    if pointing == "down":
        r = row
        q = col
    elif pointing == "up":
        r = row
        q = col + size - 1 - r
    else:
        raise ValueError(f"Unknown pointing: {pointing}")

    return q, r


def axial_to_rhombus_index(q: NDArray, r: NDArray) -> tuple[NDArray, NDArray]:
    """Convert axial to array indices for rhombus-shaped map.

    Rhombus maps store axial directly: array[r][q]

    Args:
        q: Axial q coordinates.
        r: Axial r coordinates.

    Returns:
        Tuple of (row, col) array indices.
    """
    q = np.asarray(q)
    r = np.asarray(r)
    return r, q


def rhombus_index_to_axial(row: NDArray, col: NDArray) -> tuple[NDArray, NDArray]:
    """Convert array indices back to axial for rhombus-shaped map.

    Args:
        row: Array row indices.
        col: Array column indices.

    Returns:
        Tuple of (q, r) axial coordinates.
    """
    row = np.asarray(row)
    col = np.asarray(col)
    return col, row
