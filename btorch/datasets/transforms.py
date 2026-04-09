"""Hex data augmentation transforms.

Code adapted from flyvis (MIT License).
"""

from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ..utils.hex.storage import permute, reflect_index


class HexRotation:
    """Rotate hex data by n*60°.

    Args:
        radius: Grid radius
        n: Number of 60° rotations (None = random). Positive = cw.
        p: Probability of applying
    """

    def __init__(self, radius: int, n: int | None = None, p: float = 0.5):
        self.radius = radius
        self.n = n
        self.p = p

    def __call__(self, x: torch.Tensor | NDArray) -> torch.Tensor | NDArray:
        """Apply rotation."""
        if np.random.random() > self.p:
            return x

        n = self.n if self.n is not None else np.random.randint(0, 6)
        return rotate_hex(x, self.radius, n)


class HexReflection:
    """Reflect hex data across axis.

    Args:
        radius: Grid radius
        axis: "q", "r", or "s" (None = random)
        p: Probability of applying
    """

    def __init__(
        self, radius: int, axis: Literal["q", "r", "s"] | None = None, p: float = 0.5
    ):
        self.radius = radius
        self.axis = axis
        self.p = p

    def __call__(self, x: torch.Tensor | NDArray) -> torch.Tensor | NDArray:
        """Apply reflection."""
        if np.random.random() > self.p:
            return x

        axis = self.axis
        if axis is None:
            axis = np.random.choice(["q", "r", "s"])
        return reflect_hex(x, self.radius, axis)


def rotate_hex(
    x: torch.Tensor | NDArray, radius: int, n: int
) -> torch.Tensor | NDArray:
    """Rotate data by n*60°.

    Args:
        x: Input data, shape (..., n_hexes) where n_hexes matches disk_count(radius)
        radius: Grid radius
        n: Number of 60° rotations (positive = clockwise)

    Returns:
        Rotated data with same shape as input
    """
    # Get permutation index
    perm = permute(radius, n)

    # Apply permutation
    if isinstance(x, torch.Tensor):
        return x[..., perm]
    else:
        return x[..., perm]


def reflect_hex(
    x: torch.Tensor | NDArray, radius: int, axis: Literal["q", "r", "s"]
) -> torch.Tensor | NDArray:
    """Reflect data across axis.

    Args:
        x: Input data, shape (..., n_hexes)
        radius: Grid radius
        axis: Axis to reflect across ("q", "r", or "s")

    Returns:
        Reflected data with same shape as input
    """
    # Get permutation index
    perm = reflect_index(radius, axis)

    # Apply permutation
    if isinstance(x, torch.Tensor):
        return x[..., perm]
    else:
        return x[..., perm]
