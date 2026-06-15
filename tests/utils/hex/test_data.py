"""Tests for HexCoords, HexData, HexGrid data structures."""

import numpy as np
import pytest

from btorch.utils.hex.coords import disk_count
from btorch.utils.hex.data import HexCoords, HexData, HexGrid


COORDS = (np.array([0, 3, -2, 5]), np.array([0, 1, -3, 2]))


# ---- HexCoords ----


class TestHexCoords:
    def test_from_disk(self):
        c = HexCoords.from_disk(radius=3)
        assert len(c) == disk_count(3)
        assert c.q.dtype == np.int64 or c.q.dtype == np.int32

    def test_from_ring(self):
        c = HexCoords.from_ring(radius=2)
        assert len(c) == 12

    def test_from_spiral(self):
        c = HexCoords.from_spiral(radius=3)
        assert len(c) == disk_count(3)
        assert c.q[0] == 0 and c.r[0] == 0

    def test_len(self):
        c = HexCoords(np.array([0, 1, 2]), np.array([0, 0, 0]))
        assert len(c) == 3

    def test_sort(self):
        c = HexCoords(np.array([3, 0, 1]), np.array([0, 0, 0]))
        s = c.sort()
        assert np.array_equal(s.q, [0, 1, 3])

    def test_mask(self):
        c = HexCoords(np.array([0, 1, 2, 3]), np.array([0, 0, 0, 0]))
        m = c.mask(np.array([True, False, True, False]))
        assert len(m) == 2
        assert np.array_equal(m.q, [0, 2])

    def test_distance_from_origin(self):
        c = HexCoords(np.array([2, 0, -1]), np.array([0, 2, 1]))
        d = c.distance()
        assert np.array_equal(d, [2, 2, 1])

    def test_distance_to_other(self):
        a = HexCoords(np.array([0]), np.array([0]))
        b = HexCoords(np.array([3]), np.array([0]))
        assert a.distance(b)[0] == 3

    def test_extent(self):
        c = HexCoords.from_disk(radius=5)
        assert c.extent == 5

    def test_neighbors(self):
        c = HexCoords(np.array([0]), np.array([0]))
        n = c.neighbors()
        assert len(n) == 6

    def test_rotate_6_times_identity(self):
        c = HexCoords(np.array([1, 0]), np.array([0, 1]))
        r = c.rotate(6)
        assert np.array_equal(r.q, c.q) and np.array_equal(r.r, c.r)

    def test_reflect_preserves_length(self):
        c = HexCoords(np.array([1, 2]), np.array([0, -1]))
        for axis in ["q", "r", "s"]:
            r = c.reflect(axis)
            assert len(r) == len(c)

    def test_eq_returns_bool(self):
        a = HexCoords(np.array([0, 1]), np.array([0, 1]))
        b = HexCoords(np.array([0, 1]), np.array([0, 1]))
        c = HexCoords(np.array([0, 2]), np.array([0, 1]))
        assert a == b
        assert a != c

    def test_eq_different_lengths(self):
        a = HexCoords(np.array([0, 1]), np.array([0, 1]))
        b = HexCoords(np.array([0]), np.array([0]))
        assert a != b

    def test_is_equal_elementwise(self):
        a = HexCoords(np.array([0, 1, 2]), np.array([0, 1, 2]))
        b = HexCoords(np.array([0, 2, 2]), np.array([0, 1, 3]))
        mask = a.is_equal_elementwise(b)
        assert mask[0] and not mask[1] and not mask[2]


@pytest.mark.parametrize(
    "from_fn,to_fn",
    [
        (HexCoords.from_odd_r, HexCoords.to_odd_r),
        (HexCoords.from_even_r, HexCoords.to_even_r),
        (HexCoords.from_odd_q, HexCoords.to_odd_q),
        (HexCoords.from_even_q, HexCoords.to_even_q),
        (HexCoords.from_doublewidth, HexCoords.to_doublewidth),
        (HexCoords.from_doubleheight, HexCoords.to_doubleheight),
        (HexCoords.from_cube, HexCoords.to_cube),
        (HexCoords.from_zigzag, HexCoords.to_zigzag),
    ],
    ids=[
        "odd_r",
        "even_r",
        "odd_q",
        "even_q",
        "doublewidth",
        "doubleheight",
        "cube",
        "zigzag",
    ],
)
def test_hexcoords_roundtrip(from_fn, to_fn):
    """All coordinate format roundtrips on HexCoords."""
    q, r = COORDS
    c = HexCoords(q, r)
    result = from_fn(*to_fn(c))
    assert np.array_equal(result.q, q) and np.array_equal(result.r, r)


def test_hexcoords_pixel_roundtrip():
    """Pixel roundtrip recovers exact hex coords for integer positions."""
    c_orig = HexCoords.from_disk(radius=3)
    x, y = c_orig.to_pixel(size=1.0)
    c_back = HexCoords.from_pixel(x, y, size=1.0)
    assert np.array_equal(c_orig.q, c_back.q)
    assert np.array_equal(c_orig.r, c_back.r)


# ---- HexData ----


class TestHexData:
    def test_construction(self):
        coords = HexCoords.from_disk(radius=2)
        vals = np.arange(len(coords), dtype=float)
        data = HexData(coords, vals)
        assert len(data) == len(coords)

    def test_from_arrays(self):
        data = HexData.from_arrays(
            np.array([0, 1]), np.array([0, 0]), np.array([10.0, 20.0])
        )
        assert data.values[0] == 10.0

    def test_length_mismatch_raises(self):
        coords = HexCoords(np.array([0, 1]), np.array([0, 0]))
        with pytest.raises(ValueError, match="values length"):
            HexData(coords, np.array([1.0]))

    def test_mask(self):
        data = HexData.from_arrays(
            np.array([0, 1, 2]), np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0])
        )
        masked = data.mask(np.array([True, False, True]))
        assert len(masked) == 2
        assert np.array_equal(masked.values, [1.0, 3.0])

    def test_where_value(self):
        data = HexData.from_arrays(
            np.array([0, 1, 2]), np.array([0, 0, 0]), np.array([1.0, np.nan, 3.0])
        )
        assert data.where_value(np.nan).sum() == 1
        assert data.where_value(1.0).sum() == 1

    def test_fill(self):
        data = HexData.from_arrays(
            np.array([0, 1]), np.array([0, 0]), np.array([1.0, 2.0])
        )
        filled = data.fill(99.0)
        assert np.all(filled.values == 99.0)

    def test_sort(self):
        data = HexData.from_arrays(
            np.array([2, 0, 1]), np.array([0, 0, 0]), np.array([20.0, 10.0, 15.0])
        )
        s = data.sort()
        assert np.array_equal(s.q, [0, 1, 2])
        assert np.array_equal(s.values, [10.0, 15.0, 20.0])

    def test_to_pixel(self):
        data = HexData.from_arrays(
            np.array([0, 1]), np.array([0, 0]), np.array([1.0, 2.0])
        )
        x, y, vals = data.to_pixel()
        assert len(x) == 2

    def test_rotate_6_times_identity(self):
        data = HexData.from_arrays(
            np.array([0, 1]), np.array([0, 0]), np.array([10.0, 20.0])
        )
        rot = data.rotate(6)
        assert np.array_equal(rot.values, data.values)


# ---- HexGrid ----


class TestHexGrid:
    def test_construction(self):
        grid = HexGrid(radius=3)
        assert len(grid) == disk_count(3)

    def test_q_r_access(self):
        grid = HexGrid(radius=2)
        assert len(grid.q) == len(grid.r) == len(grid)

    def test_values_default_nan(self):
        grid = HexGrid(radius=2)
        assert np.all(np.isnan(grid.values))

    def test_values_setter(self):
        grid = HexGrid(radius=2)
        grid.values = np.ones(len(grid))
        assert np.all(grid.values == 1.0)

    def test_extent(self):
        grid = HexGrid(radius=5)
        assert grid.extent == 5

    def test_hull(self):
        grid = HexGrid(radius=3)
        hull = grid.hull
        assert len(hull) == 6 * 3

    def test_circle(self):
        grid = HexGrid(radius=4)
        c = grid.circle(radius=2)
        assert len(c) == 6 * 2

    def test_filled_circle(self):
        grid = HexGrid(radius=4)
        fc = grid.filled_circle(radius=2)
        assert len(fc) == disk_count(2)

    def test_valid_neighbors(self):
        """Every hex should have 1-6 valid neighbor indices."""
        grid = HexGrid(radius=3)
        vn = grid.valid_neighbors()
        assert len(vn) == len(grid)
        for i, n_indices in enumerate(vn):
            assert isinstance(n_indices, tuple)
            for j in n_indices:
                assert 0 <= j < len(grid)

    def test_valid_neighbors_center_has_6(self):
        grid = HexGrid(radius=3)
        vn = grid.valid_neighbors()
        center_idx = np.where((grid.q == 0) & (grid.r == 0))[0][0]
        assert len(vn[center_idx]) == 6

    def test_valid_neighbors_edge_has_fewer(self):
        grid = HexGrid(radius=3)
        vn = grid.valid_neighbors()
        assert any(len(n) < 6 for n in vn), "edge hexes should have < 6 neighbors"

    def test_line(self):
        grid = HexGrid(radius=5)
        line_data = grid.line(0.0)
        assert len(line_data) > 0
        assert np.all(line_data.values == 1.0)

    def test_custom_center(self):
        grid = HexGrid(radius=2, center_q=3, center_r=-1)
        assert (3, -1) in set(zip(grid.q.tolist(), grid.r.tolist()))

    def test_to_pixel(self):
        grid = HexGrid(radius=2)
        x, y, vals = grid.to_pixel()
        assert len(x) == len(grid)
