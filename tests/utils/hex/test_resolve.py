"""Tests for the resolve_hex coord→pixel pipeline."""

import numpy as np
import pytest

from btorch.utils.hex.offset import axial_to_odd_r, axial_to_zigzag
from btorch.utils.hex.resolve import hex_symbol_for, resolve_hex


class TestResolveHex:
    @pytest.mark.parametrize(
        "fmt,c1,c2",
        [
            ("axial", np.array([0, 1, -1]), np.array([0, 0, 0])),
            ("odd_r", *axial_to_odd_r(np.array([0, 1, -1]), np.array([0, 0, 0]))),
            ("zigzag", *axial_to_zigzag(np.array([0, 1, -1]), np.array([0, 0, 0]))),
            ("doublewidth", np.array([0, 3, -1]), np.array([0, 0, 0])),
            ("pixel", np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
        ],
        ids=["axial", "odd_r", "zigzag", "doublewidth", "pixel"],
    )
    def test_coord_format(self, fmt, c1, c2):
        q, r, x, y = resolve_hex(c1, c2, coord_format=fmt, layout="pointy")
        assert len(x) == 3
        if fmt == "pixel":
            assert np.allclose(x, c1) and np.all(np.isnan(q))

    @pytest.mark.parametrize(
        "layout",
        [
            "pointy",
            "flat",
            "flywire",
        ],
    )
    def test_layout(self, layout):
        q = np.array([0, 1, -1])
        r = np.array([0, 0, 0])
        _, _, x, y = resolve_hex(q, r, coord_format="axial", layout=layout)
        assert len(x) == 3

    def test_unknown_coord_format_raises(self):
        with pytest.raises(ValueError, match="Unknown coord_format"):
            resolve_hex(np.array([0]), np.array([0]), coord_format="bad")

    def test_unknown_layout_raises(self):
        with pytest.raises(ValueError, match="Unknown layout"):
            resolve_hex(
                np.array([0]),
                np.array([0]),
                coord_format="axial",
                layout="bad",
            )

    @pytest.mark.parametrize(
        "layout,expected",
        [
            ("pointy", 14),
            ("flat", 15),
            ("flywire", 15),
            ("unknown", 14),
        ],
    )
    def test_hex_symbol_for(self, layout, expected):
        assert hex_symbol_for(layout) == expected
