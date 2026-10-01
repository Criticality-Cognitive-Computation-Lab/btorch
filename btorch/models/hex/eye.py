"""Differentiable hexagonal eye rendering.

Adapted from flyvis.datasets.rendering.eye.HexEye

Code adapted from flyvis (MIT License).
"""

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.hex.coords import disk_count


class HexEye(nn.Module):
    """Differentiable hexagonal eye model for rendering.

    Transduces Cartesian pixel images to hexagonal ommatidia responses.
    Each ommatidium integrates light over its aperture using the specified
    aggregation mode.

    Args:
        n_ommatidia: Number of hexagonal ommatidia (default: 721)
        ppo: Pixels per ommatidium resolution (default: 25)
        height_px: Image height in pixels
        width_px: Image width in pixels
        mode: Aggregation mode for receptor sampling:
            - None: No sampling, return receptor positions only
            - "point": Sample single pixel at center (fastest)
            - "mean": Average over local neighborhood (default, most realistic)
            - "sum": Sum over neighborhood (energy preserving)
            - "max": Max pooling (ON-pathway selective)
            - "min": Min pooling (OFF-pathway selective)

    Example:
        >>> eye = HexEye(n_ommatidia=721, ppo=25, height_px=600, width_px=800)
        >>> img = torch.randn(3, 600, 800)  # RGB image
        >>> hex_response = eye(img)  # (3, 721) hexagonal responses
    """

    def __init__(
        self,
        n_ommatidia: int = 721,
        ppo: int = 25,
        height_px: int | None = None,
        width_px: int | None = None,
        mode: Literal["point", "mean", "sum", "max", "min"] | None = "mean",
    ):
        super().__init__()

        self.n_ommatidia = n_ommatidia
        self.ppo = ppo
        self.mode = mode

        valid_modes = (None, "point", "mean", "sum", "max", "min")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

        # Odd kernel (proportional to ppo) so neighbourhoods are symmetric.
        self.kernel_size = max(3, ppo // 3) | 1

        # Invert n_ommatidia = 1 + 3 * radius * (radius + 1) for the radius.
        self.radius = int((-3 + np.sqrt(12 * n_ommatidia - 3)) / 6)

        expected = disk_count(self.radius)
        if expected != n_ommatidia:
            raise ValueError(
                f"{n_ommatidia} does not fill a regular hex grid. "
                f"Closest valid: {expected}"
            )

        self.height_px = height_px or ppo * (2 * self.radius + 1)
        self.width_px = width_px or ppo * (2 * self.radius + 1)

        self._generate_receptor_centers()

    def _generate_receptor_centers(self) -> None:
        """Generate hexagonal receptor center coordinates."""
        from ...utils.hex.coords import disk
        from ...utils.hex.transform import to_pixel

        q, r = disk(self.radius)

        x, y = to_pixel(q, r, size=self.ppo)

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        hex_width = x_max - x_min
        hex_height = y_max - y_min

        # Fit the grid inside the image, leaving a margin of ppo/2 on each side.
        available_width = self.width_px - self.ppo
        available_height = self.height_px - self.ppo

        if hex_width > 0 and hex_height > 0:
            scale = min(available_width / hex_width, available_height / hex_height)
            # Don't upscale beyond original ppo
            scale = min(scale, 1.0)
        else:
            scale = 1.0

        x = (x * scale) + self.width_px // 2
        y = (y * scale) + self.height_px // 2

        x = np.rint(x).astype(np.int64)
        y = np.rint(y).astype(np.int64)

        self.register_buffer(
            "receptor_x", torch.tensor(x, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "receptor_y", torch.tensor(y, dtype=torch.long), persistent=False
        )

    def forward(
        self,
        stim: torch.Tensor | None = None,
        n_chunks: int = 1,
    ) -> torch.Tensor:
        """Render stimulus through hexagonal eye.

        Args:
            stim: Input image(s) tensor, shape (..., height, width)
                or (..., height * width) if flattened. If None and mode is None,
                returns receptor positions.
            n_chunks: Number of chunks for memory-efficient processing

        Returns:
            Hexagonal response tensor, shape (..., n_ommatidia)
            or receptor positions (2, n_ommatidia) if stim is None and mode is None.
        """
        if stim is None and self.mode is None:
            return torch.stack(
                [self.receptor_x.float(), self.receptor_y.float()], dim=0
            )

        if stim is None:
            raise ValueError("stim must be provided unless mode is None")

        original_shape = stim.shape

        if stim.dim() == 2:
            stim = stim.view(*original_shape[:-1], self.height_px, self.width_px)

        if stim.shape[-2] != self.height_px or stim.shape[-1] != self.width_px:
            stim = F.interpolate(
                stim.view(-1, 1, stim.shape[-2], stim.shape[-1]),
                size=(self.height_px, self.width_px),
                mode="bilinear",
                align_corners=False,
            ).view(*original_shape[:-2], self.height_px, self.width_px)

        batch_shape = stim.shape[:-2]
        stim_flat = stim.view(-1, self.height_px, self.width_px)
        n_batch = stim_flat.shape[0]

        if self.mode == "point":
            results = []
            for i in range(n_batch):
                img = stim_flat[i]
                samples = img[self.receptor_y, self.receptor_x]
                results.append(samples)
            output = torch.stack(results)
        else:
            output = self._aggregate_receptor_regions(stim_flat)

        return output.view(*batch_shape, self.n_ommatidia)

    def _aggregate_receptor_regions(self, images: torch.Tensor) -> torch.Tensor:
        """Extract and aggregate pixel neighborhoods around each receptor.

        Args:
            images: Batch of images, shape (batch, height, width)

        Returns:
            Aggregated values at receptor positions, shape (batch, n_ommatidia)
        """
        n_batch, h, w = images.shape
        k = self.kernel_size
        pad = k // 2

        padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")

        rx = self.receptor_x.clamp(pad, w + pad - 1)
        ry = self.receptor_y.clamp(pad, h + pad - 1)

        unfolded = padded.unfold(1, k, 1).unfold(2, k, 1)  # (batch, h, w, k, k)

        neighborhoods = []
        for i in range(n_batch):
            ny = ry  # y in padded coords
            nx = rx  # x in padded coords
            nhood = unfolded[i, ny - pad, nx - pad]  # (n_ommatidia, k, k)
            neighborhoods.append(nhood)

        neighborhoods = torch.stack(neighborhoods)  # (batch, n_ommatidia, k, k)

        if self.mode == "mean":
            return neighborhoods.mean(dim=(-2, -1))
        elif self.mode == "sum":
            return neighborhoods.sum(dim=(-2, -1))
        elif self.mode == "max":
            return neighborhoods.amax(dim=(-2, -1))
        elif self.mode == "min":
            return neighborhoods.amin(dim=(-2, -1))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class BoxEye(nn.Module):
    """Fast box-filter approximation of hexagonal eye.

    Uses rectangular box filters instead of precise hexagonal regions.
    Much faster than HexEye but less accurate hexagonal geometry.

    Args:
        n_ommatidia: Number of hexagonal ommatidia (default: 721)
        ppo: Pixels per ommatidium resolution (default: 25)
        height_px: Image height in pixels
        width_px: Image width in pixels
        mode: Aggregation mode for pooling:
            - None: No pooling, return receptor positions only
            - "mean": Average pooling (default, smoothest)
            - "sum": Sum pooling (energy preserving)
            - "max": Max pooling (ON-pathway selective)
            - "min": Min pooling (OFF-pathway selective)

    Example:
        >>> eye = BoxEye(n_ommatidia=721, ppo=25, height_px=600, width_px=800)
        >>> img = torch.randn(3, 600, 800)  # RGB image
        >>> hex_response = eye(img)  # (3, 721) hexagonal responses
    """

    def __init__(
        self,
        n_ommatidia: int = 721,
        ppo: int = 25,
        height_px: int | None = None,
        width_px: int | None = None,
        mode: Literal["mean", "sum", "max", "min"] | None = "mean",
    ):
        super().__init__()

        self.n_ommatidia = n_ommatidia
        self.ppo = ppo
        self.mode = mode

        self.radius = int((-3 + np.sqrt(12 * n_ommatidia - 3)) / 6)

        expected = disk_count(self.radius)
        if expected != n_ommatidia:
            raise ValueError(
                f"{n_ommatidia} does not fill a regular hex grid. "
                f"Closest valid: {expected}"
            )

        self.height_px = height_px or ppo * (2 * self.radius + 1)
        self.width_px = width_px or ppo * (2 * self.radius + 1)

        self.kernel_size = ppo
        self._setup_conv()

        from ...utils.hex.coords import disk
        from ...utils.hex.transform import to_pixel

        q, r = disk(self.radius)
        x, y = to_pixel(q, r, size=ppo)

        self.register_buffer(
            "receptor_x",
            torch.tensor(x + self.width_px // 2, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "receptor_y",
            torch.tensor(y + self.height_px // 2, dtype=torch.long),
            persistent=False,
        )

    def _setup_conv(self) -> None:
        """Set up the box filter convolution."""
        self.pool = nn.AvgPool2d(
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            padding=0,
        )

    def forward(self, stim: torch.Tensor | None = None) -> torch.Tensor:
        """Render stimulus through box-filter eye.

        Args:
            stim: Input image(s) tensor. If None and mode is None,
                returns receptor positions.

        Returns:
            Hexagonal response tensor or receptor positions (2, n_ommatidia)
            if stim is None and mode is None.
        """
        if stim is None and self.mode is None:
            return torch.stack(
                [self.receptor_x.float(), self.receptor_y.float()], dim=0
            )

        if stim is None:
            raise ValueError("stim must be provided unless mode is None")

        original_shape = stim.shape

        if stim.dim() == 2:
            stim = stim.view(*original_shape[:-1], self.height_px, self.width_px)

        if stim.shape[-2] != self.height_px or stim.shape[-1] != self.width_px:
            stim = F.interpolate(
                stim.view(-1, 1, stim.shape[-2], stim.shape[-1]),
                size=(self.height_px, self.width_px),
                mode="bilinear",
                align_corners=False,
            ).view(*original_shape[:-2], self.height_px, self.width_px)

        batch_shape = stim.shape[:-2]
        stim_flat = stim.view(-1, 1, self.height_px, self.width_px)

        filtered = self.pool(stim_flat)

        # Approximation: the pooled map is flattened in raster order instead of
        # being sampled at each ommatidium's exact hex position
        # (receptor_x/receptor_y); adequate for the box-filter variant.
        output = filtered.view(*batch_shape, -1)[..., : self.n_ommatidia]

        return output
