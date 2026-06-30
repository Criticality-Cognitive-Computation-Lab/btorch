"""Struct-of-arrays hex data structures.

Provides performant, numpy-compatible types for hex grid data.

- :class:`HexCoords` — coordinate-only container (q, r).
- :class:`HexData` — coordinates with associated scalar values.
- :class:`HexGrid` — pre-built circular grid with neighbor topology.

All operations delegate to the functional API for consistency.

Code adapted from flyvis (MIT License).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


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

    q: np.ndarray
    r: np.ndarray

    def __post_init__(self) -> None:
        """Validate shapes match."""
        if len(self.q) != len(self.r):
            raise ValueError(
                f"q and r must have same length, got {len(self.q)} and {len(self.r)}"
            )

    def __len__(self) -> int:
        return len(self.q)

    def __eq__(self, other: object) -> bool:
        """Check equality of all coordinates."""
        if not isinstance(other, HexCoords):
            return NotImplemented
        return bool(np.array_equal(self.q, other.q) and np.array_equal(self.r, other.r))

    def is_equal_elementwise(self, other: HexCoords) -> np.ndarray:
        """Boolean mask of per-element matching coordinates."""
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
    def from_zigzag(cls, x: np.ndarray, y: np.ndarray) -> HexCoords:
        """Create from zigzag (x, y) coordinates."""
        from .offset import zigzag_to_axial

        q, r = zigzag_to_axial(x, y)
        return cls(q, r)

    @classmethod
    def from_odd_r(cls, col: np.ndarray, row: np.ndarray) -> HexCoords:
        """Create from odd-r offset coordinates."""
        from .offset import odd_r_to_axial

        q, r = odd_r_to_axial(col, row)
        return cls(q, r)

    @classmethod
    def from_even_r(cls, col: np.ndarray, row: np.ndarray) -> HexCoords:
        """Create from even-r offset coordinates."""
        from .offset import even_r_to_axial

        q, r = even_r_to_axial(col, row)
        return cls(q, r)

    @classmethod
    def from_odd_q(cls, col: np.ndarray, row: np.ndarray) -> HexCoords:
        """Create from odd-q offset coordinates."""
        from .offset import odd_q_to_axial

        q, r = odd_q_to_axial(col, row)
        return cls(q, r)

    @classmethod
    def from_even_q(cls, col: np.ndarray, row: np.ndarray) -> HexCoords:
        """Create from even-q offset coordinates."""
        from .offset import even_q_to_axial

        q, r = even_q_to_axial(col, row)
        return cls(q, r)

    @classmethod
    def from_doublewidth(cls, col: np.ndarray, row: np.ndarray) -> HexCoords:
        """Create from double-width coordinates."""
        from .doubled import doublewidth_to_axial

        q, r = doublewidth_to_axial(col, row)
        return cls(q, r)

    @classmethod
    def from_doubleheight(cls, col: np.ndarray, row: np.ndarray) -> HexCoords:
        """Create from double-height coordinates."""
        from .doubled import doubleheight_to_axial

        q, r = doubleheight_to_axial(col, row)
        return cls(q, r)

    @classmethod
    def from_cube(cls, q: np.ndarray, r: np.ndarray, s: np.ndarray) -> HexCoords:
        """Create from cube coordinates (validates q+r+s=0)."""
        q, r, s = np.asarray(q), np.asarray(r), np.asarray(s)
        if not np.allclose(q + r + s, 0):
            raise ValueError("Cube coordinates must satisfy q + r + s = 0")
        return cls(q, r)

    @classmethod
    def from_pixel(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        size: float = 1.0,
        orientation: str = "pointy",
    ) -> HexCoords:
        """Create from pixel coordinates (rounds to nearest hex)."""
        from .transform import from_pixel, round_axial

        fq, fr = from_pixel(x, y, size=size, orientation=orientation)
        q, r = round_axial(fq, fr)
        return cls(q, r)

    # ---- Coordinate transforms ----

    def to_pixel(
        self, size: float = 1.0, orientation: str = "pointy"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert to pixel coordinates."""
        from .transform import to_pixel

        return to_pixel(self.q, self.r, size, orientation)

    def to_zigzag(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to zigzag (x, y) coordinates."""
        from .offset import axial_to_zigzag

        return axial_to_zigzag(self.q, self.r)

    def to_odd_r(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to odd-r offset coordinates (col, row)."""
        from .offset import axial_to_odd_r

        return axial_to_odd_r(self.q, self.r)

    def to_even_r(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to even-r offset coordinates (col, row)."""
        from .offset import axial_to_even_r

        return axial_to_even_r(self.q, self.r)

    def to_odd_q(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to odd-q offset coordinates (col, row)."""
        from .offset import axial_to_odd_q

        return axial_to_odd_q(self.q, self.r)

    def to_even_q(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to even-q offset coordinates (col, row)."""
        from .offset import axial_to_even_q

        return axial_to_even_q(self.q, self.r)

    def to_doublewidth(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to double-width coordinates (col, row)."""
        from .doubled import axial_to_doublewidth

        return axial_to_doublewidth(self.q, self.r)

    def to_doubleheight(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to double-height coordinates (col, row)."""
        from .doubled import axial_to_doubleheight

        return axial_to_doubleheight(self.q, self.r)

    def to_cube(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to cube coordinates (q, r, s) where s = -q - r."""
        from .transform import cube_from_axial

        return cube_from_axial(self.q, self.r)

    # ---- Geometric operations ----

    def distance(self, other: HexCoords | None = None) -> np.ndarray | int:
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

    def within_range(self, center: HexCoords, n: int) -> np.ndarray:
        """Boolean mask for coords within n steps of center."""
        from .distance import within_range

        return within_range(self.q, self.r, int(center.q[0]), int(center.r[0]), n)

    # ---- Masking and filtering ----

    def mask(self, condition: np.ndarray) -> HexCoords:
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
    values: np.ndarray

    def __post_init__(self) -> None:
        """Validate values shape matches coords."""
        if len(self.values) != len(self.coords):
            raise ValueError(
                f"values length {len(self.values)} must match "
                f"coords length {len(self.coords)}"
            )

    def __len__(self) -> int:
        return len(self.coords)

    @property
    def q(self) -> np.ndarray:
        return self.coords.q

    @property
    def r(self) -> np.ndarray:
        return self.coords.r

    # ---- Construction ----

    @classmethod
    def from_arrays(cls, q: np.ndarray, r: np.ndarray, values: np.ndarray) -> HexData:
        """Construct from separate arrays."""
        return cls(HexCoords(q, r), values)

    # ---- Coordinate operations (delegated to coords) ----

    def to_pixel(
        self, size: float = 1.0, orientation: str = "pointy"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def mask(self, condition: np.ndarray) -> HexData:
        """Filter by boolean mask."""
        return HexData(self.coords.mask(condition), self.values[condition])

    def where_value(self, value: float, rtol: float = 0, atol: float = 0) -> np.ndarray:
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


class HexGrid:
    """Regular hexagonal grid with extent.

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
        values: np.ndarray | None = None,
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
    def q(self) -> np.ndarray:
        return self._data.q

    @property
    def r(self) -> np.ndarray:
        return self._data.r

    @property
    def values(self) -> np.ndarray:
        return self._data.values

    @values.setter
    def values(self, v: np.ndarray):
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

    def circle(self, radius: int | None = None) -> HexData:
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
        coord_to_idx = {(int(self.q[i]), int(self.r[i])): i for i in range(len(self))}

        result = []
        for i in range(len(self)):
            neighbor_indices = []
            for j in range(6):
                key = (int(qn[j, i]), int(rn[j, i]))
                if key in coord_to_idx:
                    neighbor_indices.append(coord_to_idx[key])
            result.append(tuple(neighbor_indices))
        return tuple(result)

    def to_pixel(
        self, size: float = 1.0, orientation: str = "pointy"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to pixel coordinates with values."""
        return self._data.to_pixel(size, orientation)
