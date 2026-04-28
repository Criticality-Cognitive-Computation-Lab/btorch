"""Struct-of-arrays hex data structures.

Replaces Hexal/HexArray/HexLattice with performant, numpy-compatible
types. All operations use functional API internally for consistency.

Code adapted from flyvis (MIT License).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class HexCoords:
    """Struct-of-arrays for hex coordinates (q, r).

    This is the coordinate-only type. Use HexData for coords + values.
    All methods delegate to functional API for consistency.

    Args:
        q: Axial q coordinates, shape (n_hexes,)
        r: Axial r coordinates, shape (n_hexes,)

    Example:
        >>> coords = HexCoords.from_disk(radius=3)
        >>> coords_q, coords_r = coords.q, coords.r
        >>> px, py = coords.to_pixel()
        >>> neighbors = coords.neighbors()  # HexCoords with 6x coords
    """

    q: NDArray
    r: NDArray

    def __post_init__(self):
        """Validate shapes match."""
        if len(self.q) != len(self.r):
            raise ValueError(
                f"q and r must have same length, got {len(self.q)} and {len(self.r)}"
            )

    def __len__(self) -> int:
        return len(self.q)

    def __eq__(self, other: HexCoords) -> NDArray:
        """Boolean mask of matching coordinates."""
        return (self.q == other.q) & (self.r == other.r)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """Iterate over (q, r) pairs."""
        return iter(zip(self.q, self.r))

    # ---- Construction methods ----

    @classmethod
    def from_disk(cls, radius: int, center_q: int = 0, center_r: int = 0) -> HexCoords:
        """Create from disk of given radius."""
        from .coords import disk

        q, r = disk(radius, center_q, center_r)
        return cls(q, r)

    @classmethod
    def from_ring(cls, radius: int, center_q: int = 0, center_r: int = 0) -> HexCoords:
        """Create from ring of given radius."""
        from .coords import ring

        q, r = ring(radius, center_q, center_r)
        return cls(q, r)

    @classmethod
    def from_spiral(
        cls, radius: int, center_q: int = 0, center_r: int = 0
    ) -> HexCoords:
        """Create in spiral order (center, ring1, ring2...)."""
        from .coords import spiral

        q, r = spiral(radius, center_q, center_r)
        return cls(q, r)

    @classmethod
    def from_zigzag(cls, x: NDArray, y: NDArray) -> HexCoords:
        """Create from zigzag (x, y) coordinates."""
        from .offset import zigzag_to_axial

        q, r = zigzag_to_axial(x, y)
        return cls(q, r)

    # ---- Coordinate transforms ----

    def to_pixel(
        self, size: float = 1.0, orientation: str = "pointy"
    ) -> tuple[NDArray, NDArray]:
        """Convert to pixel coordinates."""
        from .transform import to_pixel

        return to_pixel(self.q, self.r, size, orientation)

    def to_zigzag(self) -> tuple[NDArray, NDArray]:
        """Convert to zigzag (x, y) coordinates."""
        from .offset import axial_to_zigzag

        return axial_to_zigzag(self.q, self.r)

    # ---- Geometric operations ----

    def distance(self, other: Optional[HexCoords] = None) -> Union[NDArray, int]:
        """Distance to other coords, or from origin if other is None."""
        from .distance import distance, radius

        if other is None:
            return radius(self.q, self.r)
        return distance(self.q, self.r, other.q, other.r)

    @property
    def extent(self) -> int:
        """Maximum distance from origin."""
        return int(np.max(self.distance()))

    def neighbors(self) -> HexCoords:
        """Get 6 neighbors for each coordinate.

        Returns HexCoords with shape (6 * n_hexes,).
        """
        from .neighbor import neighbors

        qn, rn = neighbors(self.q, self.r)
        # Flatten to (6 * n,)
        return HexCoords(qn.flatten(), rn.flatten())

    def rotate(self, n: int) -> HexCoords:
        """Rotate by n * 60 degrees."""
        from .transform import rotate

        q, r = rotate(self.q, self.r, n)
        return HexCoords(q, r)

    def reflect(self, axis: str) -> HexCoords:
        """Reflect across axis ('q', 'r', or 's')."""
        from .transform import reflect

        q, r = reflect(self.q, self.r, axis)
        return HexCoords(q, r)

    def within_range(self, center: HexCoords, n: int) -> NDArray:
        """Boolean mask for coords within n steps of center."""
        from .distance import within_range

        return within_range(self.q, self.r, int(center.q[0]), int(center.r[0]), n)

    # ---- Masking and filtering ----

    def mask(self, condition: NDArray) -> HexCoords:
        """Filter by boolean mask."""
        return HexCoords(self.q[condition], self.r[condition])

    def sort(self) -> HexCoords:
        """Sort by q then r."""
        idx = np.lexsort((self.r, self.q))
        return HexCoords(self.q[idx], self.r[idx])


@dataclass
class HexData:
    """Struct-of-arrays for hex coordinates with associated values.

    This is the primary user-facing type for hex grid data.
    Separates coordinates (coords) from data (values).

    Args:
        coords: HexCoords instance
        values: Data values, shape (n_hexes,) or (n_hexes, n_features)

    Example:
        >>> coords = HexCoords.from_disk(radius=3)
        >>> data = HexData(coords, np.random.randn(len(coords)))
        >>> data_q = data.q  # Access coords
        >>> data_vals = data.values  # Access values
    """

    coords: HexCoords
    values: NDArray

    def __post_init__(self):
        """Validate values shape matches coords."""
        if len(self.values) != len(self.coords):
            raise ValueError(
                f"values length {len(self.values)} must match "
                f"coords length {len(self.coords)}"
            )

    def __len__(self) -> int:
        return len(self.coords)

    @property
    def q(self) -> NDArray:
        return self.coords.q

    @property
    def r(self) -> NDArray:
        return self.coords.r

    # ---- Construction ----

    @classmethod
    def from_arrays(cls, q: NDArray, r: NDArray, values: NDArray) -> HexData:
        """Construct from separate arrays."""
        return cls(HexCoords(q, r), values)

    # ---- Coordinate operations (delegated to coords) ----

    def to_pixel(
        self, size: float = 1.0, orientation: str = "pointy"
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Convert to pixel coordinates with values."""
        x, y = self.coords.to_pixel(size, orientation)
        return x, y, self.values

    def rotate(self, n: int) -> HexData:
        """Rotate coordinates, preserving values."""
        return HexData(self.coords.rotate(n), self.values)

    def reflect(self, axis: str) -> HexData:
        """Reflect coordinates, preserving values."""
        return HexData(self.coords.reflect(axis), self.values)

    # ---- Value operations ----

    def mask(self, condition: NDArray) -> HexData:
        """Filter by boolean mask."""
        return HexData(self.coords.mask(condition), self.values[condition])

    def where_value(self, value: float, rtol: float = 0, atol: float = 0) -> NDArray:
        """Boolean mask where values match (supports np.nan)."""
        return np.isclose(self.values, value, rtol=rtol, atol=atol, equal_nan=True)

    def fill(self, value: float) -> HexData:
        """Return new HexData with all values set to value."""
        return HexData(self.coords, np.full_like(self.values, value))

    def sort(self) -> HexData:
        """Sort by coordinates."""
        idx = np.lexsort((self.coords.r, self.coords.q))
        return HexData(
            HexCoords(self.coords.q[idx], self.coords.r[idx]), self.values[idx]
        )

    # ---- Visualization ----

    def plot(self, coord_format: str = "axial", **kwargs):
        """Quick plot of values."""
        from ...visualisation.hex.static import scatter

        if coord_format == "axial":
            return scatter(self.q, self.r, self.values, coord_format="axial", **kwargs)
        elif coord_format == "zigzag":
            x, y = self.coords.to_zigzag()
            return scatter(x, y, self.values, coord_format="pixel", **kwargs)
        else:
            raise ValueError(f"Unknown coord_format: {coord_format}")


class HexGrid:
    """Regular hexagonal grid with extent (replaces HexLattice).

    This is a specialized HexData for regular hexagonal grids where
    coordinates form a complete disk of given radius.

    Args:
        radius: Grid radius (extent)
        values: Optional initial values
        center_q, center_r: Center coordinates

    Example:
        >>> grid = HexGrid(radius=5)
        >>> grid.circle(radius=3)  # Get circle of coords
        >>> grid.hull  # Outer ring
    """

    def __init__(
        self,
        radius: int,
        values: Optional[NDArray] = None,
        center_q: int = 0,
        center_r: int = 0,
    ):
        self.radius = radius
        self.center = HexCoords(np.array([center_q]), np.array([center_r]))

        # Generate coordinates
        from .coords import disk

        q, r = disk(radius, center_q, center_r)
        coords = HexCoords(q, r)

        # Initialize values
        if values is None:
            values = np.full(len(coords), np.nan)
        elif len(values) != len(coords):
            raise ValueError(f"values length must match grid size {len(coords)}")

        self._data = HexData(coords, values)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def q(self) -> NDArray:
        return self._data.q

    @property
    def r(self) -> NDArray:
        return self._data.r

    @property
    def values(self) -> NDArray:
        return self._data.values

    @values.setter
    def values(self, v: NDArray):
        self._data.values = v

    @property
    def coords(self) -> HexCoords:
        return self._data.coords

    @property
    def data(self) -> HexData:
        """Access as HexData."""
        return self._data

    @property
    def extent(self) -> int:
        """Maximum distance from center."""
        from .distance import distance

        distances = distance(self.q, self.r, self.center.q[0], self.center.r[0])
        return int(np.max(distances))

    @property
    def hull(self) -> HexData:
        """Outer ring of the grid."""
        return self.circle(radius=self.radius)

    def circle(self, radius: Optional[int] = None) -> HexData:
        """Get circle of given radius from center.

        Returns HexData with values=1 on circle, others filtered out.
        """
        from .distance import distance

        r = radius if radius is not None else self.radius
        distances = distance(self.q, self.r, self.center.q[0], self.center.r[0])
        mask = distances == r
        values = np.where(mask, 1, np.nan)
        return HexData(self.coords.mask(mask), values[mask])

    def filled_circle(self, radius: int) -> HexData:
        """Get filled circle of given radius from center."""
        from .distance import distance

        distances = distance(self.q, self.r, self.center.q[0], self.center.r[0])
        mask = distances <= radius
        values = np.where(mask, 1, np.nan)
        return HexData(self.coords.mask(mask), values[mask])

    def line(self, angle: float) -> HexData:
        """Get line through center at given angle.

        Args:
            angle: Angle in radians

        Returns HexData with line coordinates and values=1.
        """
        from .coords import ring
        from .transform import round_axial

        # Find line span across the grid
        # Get distant hull points at target angle
        distant_q, distant_r = ring(2 * self.radius)
        distant_coords = HexCoords(distant_q, distant_r)

        # Calculate angles to find matching direction
        px, py = distant_coords.to_pixel()
        angles = np.arctan2(py, px)

        # Find closest to target angle (modulo pi for line)
        angle_diff = np.abs((angles - angle + np.pi) % np.pi - np.pi / 2)
        sorted_idx = np.argsort(angle_diff)

        # Get span points
        span_q = distant_q[sorted_idx[:2]]
        span_r = distant_r[sorted_idx[:2]]

        # Interpolate line between them
        from .distance import distance as hex_distance

        d = hex_distance(span_q[0:1], span_r[0:1], span_q[1:2], span_r[1:2])[0]
        line_q, line_r = [], []
        for i in range(int(d) + 1):
            t = i / d if d > 0 else 0
            q = span_q[0] + (span_q[1] - span_q[0]) * t
            r = span_r[0] + (span_r[1] - span_r[0]) * t
            # Round to nearest hex
            rq, rr = round_axial(np.array([q]), np.array([r]))
            line_q.append(rq[0])
            line_r.append(rr[0])

        line_coords = HexCoords(np.array(line_q), np.array(line_r))
        values = np.ones(len(line_coords))
        return HexData(line_coords, values)

    def valid_neighbors(self) -> tuple[tuple[int, ...], ...]:
        """Get valid neighbor indices for each hex in grid.

        Returns tuple of tuples, where each inner tuple contains indices
        of valid neighbors within the grid.
        """
        from .neighbor import neighbors

        qn, rn = neighbors(self.q, self.r)  # (6, n)

        result = []
        for i in range(len(self)):
            neighbor_indices = []
            for j in range(6):
                # Find if neighbor j of i exists in grid
                nq, nr = qn[j, i], rn[j, i]
                matches = (self.q == nq) & (self.r == nr)
                if matches.any():
                    neighbor_indices.append(np.where(matches)[0][0])
            result.append(tuple(neighbor_indices))
        return tuple(result)

    def to_pixel(
        self, size: float = 1.0, orientation: str = "pointy"
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Convert to pixel coordinates with values."""
        return self._data.to_pixel(size, orientation)

    def plot(self, **kwargs):
        """Plot the grid values."""
        return self._data.plot(**kwargs)


# Maintain backward compatibility alias
HexLattice = HexGrid
