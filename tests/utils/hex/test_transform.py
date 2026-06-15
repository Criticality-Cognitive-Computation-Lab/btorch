"""Tests for hex coordinate transforms: rotate, reflect, round, pixel conversion."""

import numpy as np

from btorch.utils.hex.coords import disk
from btorch.utils.hex.distance import distance, radius
from btorch.utils.hex.neighbor import neighbors
from btorch.utils.hex.transform import (
    cube_from_axial,
    from_pixel,
    reflect,
    rotate,
    round_axial,
    to_pixel,
)


class TestRoundAxial:
    def test_integer_passthrough(self):
        q, r = np.array([2.0, -1.0, 0.0]), np.array([3.0, 1.0, -2.0])
        rq, rr = round_axial(q, r)
        assert np.array_equal(rq, q.astype(int))
        assert np.array_equal(rr, r.astype(int))

    def test_fractional_preserves_cube_constraint(self):
        q = np.array([0.4, 0.6, -0.1])
        r = np.array([0.3, 0.3, 0.9])
        rq, rr = round_axial(q, r)
        s = -rq - rr
        assert np.allclose(rq + rr + s, 0), "cube constraint violated after round"


class TestRotate:
    def test_6_rotations_identity(self):
        q, r = disk(5)
        q6, r6 = rotate(q, r, 6)
        assert np.array_equal(q, q6)
        assert np.array_equal(r, r6)

    def test_preserves_distances(self):
        """Every rotation must preserve distances from origin."""
        q, r = disk(4)
        d_orig = radius(q, r)
        for n in range(1, 6):
            qr, rr = rotate(q, r, n)
            d_rot = distance(qr, rr, 0, 0)
            assert np.array_equal(d_orig, d_rot), f"distances changed at rotation {n}"


class TestReflect:
    def test_self_inverse(self):
        """Reflecting twice across the same axis returns to original."""
        q, r = disk(4)
        for axis in ["q", "r", "s"]:
            qr, rr = reflect(q, r, axis)
            q2, r2 = reflect(qr, rr, axis)
            assert np.array_equal(q, q2), f"reflect {axis} not self-inverse"
            assert np.array_equal(r, r2), f"reflect {axis} not self-inverse"


class TestToPixel:
    def test_origin(self):
        x, y = to_pixel(np.array([0]), np.array([0]))
        assert x[0] == 0.0 and y[0] == 0.0

    def test_flat_vs_pointy_differ(self):
        q = np.array([1])
        r = np.array([0])
        xp, yp = to_pixel(q, r, orientation="pointy")
        xf, yf = to_pixel(q, r, orientation="flat")
        assert not np.allclose(xp, xf), "flat and pointy should differ"

    def test_neighbors_produce_distinct_positions(self):
        q, r = neighbors(np.array([0]), np.array([0]))
        x, y = to_pixel(q.flatten(), r.flatten())
        dists = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        np.fill_diagonal(dists, np.inf)
        assert dists.min() > 0


class TestFromPixel:
    def test_roundtrip(self):
        """to_pixel → from_pixel → round_axial should recover original
        coords."""
        q, r = disk(3)
        x, y = to_pixel(q, r, size=1.5)
        fq, fr = from_pixel(x, y, size=1.5)
        qr, rr = round_axial(fq, fr)
        assert np.array_equal(q, qr)
        assert np.array_equal(r, rr)


class TestCubeFromAxial:
    def test_cube_constraint(self):
        q, r = np.array([0, 3, -2, 5]), np.array([0, 1, -3, 2])
        qc, rc, sc = cube_from_axial(q, r)
        assert np.allclose(qc + rc + sc, 0), "cube constraint q+r+s=0"
