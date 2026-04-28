"""Hexagonal convolution layer.

Adapted from flyvis.task.decoder.Conv2dHexSpace

Code adapted from flyvis (MIT License).
"""

import torch
import torch.nn as nn

from ...utils.hex.coords import disk


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
        **kwargs: Additional keyword arguments for Conv2d.

    Attributes:
        mask: Mask for hexagonal convolution.
        _filter_to_hex: Whether to apply hexagonal filter.

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
        **kwargs,
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
            q, r = disk(kernel_size // 2)
            q -= q.min()
            r -= r.min()
            mask = torch.zeros_like(self.weight)
            mask[..., q, r] = 1
            self.register_buffer("mask", mask, persistent=False)
            self.weight.data.mul_(self.mask)
            self._filter_to_hex = True
        else:
            self._filter_to_hex = False

    def filter_to_hex(self):
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


class Conv2dHexSpace(Conv2dHex):
    """Convolution with regularly, hexagonally shaped filters.

    This is an alias for Conv2dHex for compatibility with flyvis naming.

    Reference to map storage:
    https://www.redblobgames.com/grids/hexagons/#map-storage

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding added to input.
        **kwargs: Additional keyword arguments for Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )
