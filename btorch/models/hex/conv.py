"""Hexagonal convolution layer.

Provides :class:`Conv2dHex`, which constrains standard 2D convolution
kernels to a hexagonal receptive field via a binary mask applied once
at initialisation.

Code adapted from flyvis (MIT License).
"""

from typing import Any

import torch
import torch.nn as nn

from ...utils.hex.coords import disk
from ...utils.hex.storage import axial_to_rect_index


class Conv2dHex(nn.Conv2d):
    """2D Convolution with hexagonally-shaped filters (in cartesian map
    storage).

    Applies a hexagonal mask to the convolution kernel, constraining
    the receptive field to a hexagonal shape on the input.

    Reference to map storage:
    https://www.redblobgames.com/grids/hexagons/#map-storage

    Note:
        kernel_size must be odd!

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of hexagonal kernel (must be odd)
        stride: Stride for convolution (default: 1)
        padding: Padding for convolution (default: 0)
        storage: Rectangular storage orientation. "rect_pointy" (default)
            or "rect_flat". Determines how axial disk coords map to
            array indices for the mask.
        **kwargs: Additional keyword arguments for Conv2d.

    Attributes:
        mask: Mask for hexagonal convolution.

    Example:
        >>> conv = Conv2dHex(16, 32, kernel_size=7)
        >>> x = torch.randn(1, 16, 64, 64)
        >>> out = conv(x)  # Output has hexagonal receptive fields
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        storage: str = "rect_pointy",
        **kwargs: Any,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

        if not kernel_size % 2:
            raise ValueError(f"{kernel_size} is even. Must be odd.")
        if kernel_size > 1:
            orientation = storage.removeprefix("rect_")
            q, r = disk(kernel_size // 2)
            row, col = axial_to_rect_index(q, r, orientation=orientation)
            row = (row - row.min()).astype(int)
            col = (col - col.min()).astype(int)
            mask = torch.zeros_like(self.weight)
            mask[..., row, col] = 1
            self.register_buffer("mask", mask, persistent=False)
            self.filter_to_hex()
            self._filter_to_hex = True
        else:
            self._filter_to_hex = False

    def filter_to_hex(self) -> torch.Tensor:
        """Apply hexagonal filter to weights."""
        return self.weight.data.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Conv2dHex layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after hexagonal convolution.
        """
        if self._filter_to_hex:
            self.filter_to_hex()
        return super().forward(x)
