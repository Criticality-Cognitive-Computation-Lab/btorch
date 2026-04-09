"""Hexagonal model components subpackage.

Convolution layers and differentiable rendering.

Code adapted from flyvis (MIT License).
"""

from .conv import Conv2dHex, Conv2dHexSpace
from .eye import BoxEye, HexEye


__all__ = [
    "Conv2dHex",
    "Conv2dHexSpace",
    "HexEye",
    "BoxEye",
]
