"""Tests for hex augmentation transforms."""

import numpy as np
import torch

from btorch.datasets.transforms import (
    HexReflection,
    HexRotation,
    reflect_hex,
    rotate_hex,
)
from btorch.utils.hex.coords import disk_count


def test_rotate_hex_six_rotations_is_identity():
    """Rotating by 6 * 60 degrees returns the original data."""
    radius = 2
    x = np.arange(disk_count(radius))
    np.testing.assert_array_equal(rotate_hex(x, radius, 6), x)
    # A single rotation must move data (radius >= 1 has non-central hexes).
    assert not np.array_equal(rotate_hex(x, radius, 1), x)


def test_reflect_hex_is_involution_and_torch_compatible():
    """Reflecting twice restores data; torch tensors are supported."""
    radius = 2
    x = torch.arange(disk_count(radius)).float()
    once = reflect_hex(x, radius, "q")
    assert isinstance(once, torch.Tensor)
    assert torch.equal(reflect_hex(once, radius, "q"), x)


def test_transform_classes_probability():
    """P=0 never transforms, p=1 with a fixed rotation always does."""
    radius = 2
    x = np.arange(disk_count(radius))
    np.testing.assert_array_equal(HexRotation(radius, n=1, p=0.0)(x), x)
    np.testing.assert_array_equal(
        HexRotation(radius, n=1, p=1.0)(x), rotate_hex(x, radius, 1)
    )
    np.testing.assert_array_equal(
        HexReflection(radius, axis="r", p=1.0)(x), reflect_hex(x, radius, "r")
    )
