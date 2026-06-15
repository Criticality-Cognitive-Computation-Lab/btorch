"""Hex data augmentation transforms.

Code adapted from flyvis (MIT License).
"""

from typing import Literal

import numpy as np
import torch

from ..utils.hex.storage import permute, reflect_index


_rng = np.random.default_rng()


class HexRotation:
    """Rotate hex data by n*60 degrees.

    Applies a random or fixed rotation to data stored on a hexagonal
    grid. Useful for data augmentation during training.

    Args:
        radius: Grid radius. Determines the number of hexes via
            ``disk_count(radius)``.
        n: Number of 60-degree rotations. ``None`` picks a random
            rotation each call. Positive values rotate clockwise.
        p: Probability of applying the rotation on each call.

    Examples:
        Fixed 120-degree rotation:

        >>> augment = HexRotation(radius=5, n=2)
        >>> rotated = augment(data)

        Random rotation with 50% chance:

        >>> augment = HexRotation(radius=5, p=0.5)
    """

    def __init__(self, radius: int, n: int | None = None, p: float = 0.5):
        self.radius = radius
        self.n = n
        self.p = p

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Apply rotation."""
        if _rng.random() > self.p:
            return x

        n = self.n if self.n is not None else _rng.integers(0, 6)
        return rotate_hex(x, self.radius, n)


class HexReflection:
    """Reflect hex data across a symmetry axis.

    Applies a random or fixed reflection to data stored on a hexagonal
    grid. Useful for enforcing bilateral symmetry or data augmentation.

    Args:
        radius: Grid radius. Determines the number of hexes via
            ``disk_count(radius)``.
        axis: Symmetry axis to reflect across: ``"q"``, ``"r"``, or
            ``"s"``. ``None`` picks a random axis each call.
        p: Probability of applying the reflection on each call.

    Examples:
        Fixed reflection across q-axis:

        >>> augment = HexReflection(radius=5, axis="q")
        >>> reflected = augment(data)

        Random reflection with 50% chance:

        >>> augment = HexReflection(radius=5, p=0.5)
    """

    def __init__(
        self, radius: int, axis: Literal["q", "r", "s"] | None = None, p: float = 0.5
    ):
        self.radius = radius
        self.axis = axis
        self.p = p

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Apply reflection."""
        if _rng.random() > self.p:
            return x

        axis = self.axis
        if axis is None:
            axis = _rng.choice(["q", "r", "s"])
        return reflect_hex(x, self.radius, axis)


def rotate_hex(
    x: torch.Tensor | np.ndarray, radius: int, n: int
) -> torch.Tensor | np.ndarray:
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
    x: torch.Tensor | np.ndarray, radius: int, axis: Literal["q", "r", "s"]
) -> torch.Tensor | np.ndarray:
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
