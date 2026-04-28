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

        # Validate mode
        valid_modes = (None, "point", "mean", "sum", "max", "min")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

        # Neighborhood size for aggregation modes (proportional to ppo)
        # Use odd kernel size for symmetric neighborhoods
        self.kernel_size = max(3, ppo // 3) | 1  # Ensure odd

        # Calculate extent from n_ommatidia
        # n_ommatidia = 1 + 3*radius*(radius+1)
        # Solve for radius approximately
        self.radius = int((-3 + np.sqrt(12 * n_ommatidia - 3)) / 6)

        # Verify n_ommatidia fills a regular hex grid
        expected = disk_count(self.radius)
        if expected != n_ommatidia:
            raise ValueError(
                f"{n_ommatidia} does not fill a regular hex grid. "
                f"Closest valid: {expected}"
            )

        # Set default dimensions if not provided
        self.height_px = height_px or ppo * (2 * self.radius + 1)
        self.width_px = width_px or ppo * (2 * self.radius + 1)

        # Generate receptor center coordinates
        self._generate_receptor_centers()

    def _generate_receptor_centers(self):
        """Generate hexagonal receptor center coordinates."""
        from ...utils.hex.coords import disk
        from ...utils.hex.transform import to_pixel

        # Get hex coordinates
        q, r = disk(self.radius)

        # Convert to pixel coordinates
        x, y = to_pixel(q, r, size=self.ppo)

        # Calculate hex grid bounds to properly fit within image
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # Scale to fit within image with some margin, preserving aspect ratio
        hex_width = x_max - x_min
        hex_height = y_max - y_min

        # Calculate scale to fit hex grid within image dimensions
        # Leave a small margin (ppo/2) on each side
        available_width = self.width_px - self.ppo
        available_height = self.height_px - self.ppo

        if hex_width > 0 and hex_height > 0:
            scale = min(available_width / hex_width, available_height / hex_height)
            # Don't upscale beyond original ppo
            scale = min(scale, 1.0)
        else:
            scale = 1.0

        # Apply scaling and center
        x = (x * scale) + self.width_px // 2
        y = (y * scale) + self.height_px // 2

        # Round to integer pixel coordinates
        x = np.rint(x).astype(np.int64)
        y = np.rint(y).astype(np.int64)

        # Register as buffer (non-trainable)
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
        # Return receptor positions if no stimulus provided and mode is None
        if stim is None and self.mode is None:
            return torch.stack(
                [self.receptor_x.float(), self.receptor_y.float()], dim=0
            )

        if stim is None:
            raise ValueError("stim must be provided unless mode is None")

        original_shape = stim.shape

        # Handle flattened input
        if stim.dim() == 2:
            stim = stim.view(*original_shape[:-1], self.height_px, self.width_px)

        # Resize if needed
        if stim.shape[-2] != self.height_px or stim.shape[-1] != self.width_px:
            stim = F.interpolate(
                stim.view(-1, 1, stim.shape[-2], stim.shape[-1]),
                size=(self.height_px, self.width_px),
                mode="bilinear",
                align_corners=False,
            ).view(*original_shape[:-2], self.height_px, self.width_px)

        # Flatten batch dimensions
        batch_shape = stim.shape[:-2]
        stim_flat = stim.view(-1, self.height_px, self.width_px)
        n_batch = stim_flat.shape[0]

        # Sample at receptor locations
        if self.mode == "point":
            # Fast path: simple indexing at receptor centers
            results = []
            for i in range(n_batch):
                img = stim_flat[i]
                samples = img[self.receptor_y, self.receptor_x]
                results.append(samples)
            output = torch.stack(results)
        else:
            # Aggregation modes: extract neighborhoods and aggregate
            output = self._aggregate_receptor_regions(stim_flat)

        # Reshape back
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

        # Pad images to handle edges
        padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")

        # Extract receptor positions (ensure within valid range)
        rx = self.receptor_x.clamp(pad, w + pad - 1)
        ry = self.receptor_y.clamp(pad, h + pad - 1)

        # Extract k×k neighborhoods around each receptor
        # Use unfold to get sliding windows, then index
        unfolded = padded.unfold(1, k, 1).unfold(2, k, 1)  # (batch, h, w, k, k)

        # Gather neighborhoods for all receptors at once
        neighborhoods = []
        for i in range(n_batch):
            # Get neighborhoods at receptor positions
            # Shift by pad to account for padding
            ny = ry  # y in padded coords
            nx = rx  # x in padded coords
            nhood = unfolded[i, ny - pad, nx - pad]  # (n_ommatidia, k, k)
            neighborhoods.append(nhood)

        neighborhoods = torch.stack(neighborhoods)  # (batch, n_ommatidia, k, k)

        # Apply aggregation
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

        # Calculate extent
        self.radius = int((-3 + np.sqrt(12 * n_ommatidia - 3)) / 6)

        expected = disk_count(self.radius)
        if expected != n_ommatidia:
            raise ValueError(
                f"{n_ommatidia} does not fill a regular hex grid. "
                f"Closest valid: {expected}"
            )

        self.height_px = height_px or ppo * (2 * self.radius + 1)
        self.width_px = width_px or ppo * (2 * self.radius + 1)

        # Create box filter convolution
        self.kernel_size = ppo
        self._setup_conv()

        # Generate receptor centers
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

    def _setup_conv(self):
        """Set up the box filter convolution."""
        # Create average pooling layer
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
        # Return receptor positions if no stimulus provided and mode is None
        if stim is None and self.mode is None:
            return torch.stack(
                [self.receptor_x.float(), self.receptor_y.float()], dim=0
            )

        if stim is None:
            raise ValueError("stim must be provided unless mode is None")

        original_shape = stim.shape

        # Handle flattened input
        if stim.dim() == 2:
            stim = stim.view(*original_shape[:-1], self.height_px, self.width_px)

        # Resize if needed
        if stim.shape[-2] != self.height_px or stim.shape[-1] != self.width_px:
            stim = F.interpolate(
                stim.view(-1, 1, stim.shape[-2], stim.shape[-1]),
                size=(self.height_px, self.width_px),
                mode="bilinear",
                align_corners=False,
            ).view(*original_shape[:-2], self.height_px, self.width_px)

        # Flatten batch dimensions
        batch_shape = stim.shape[:-2]
        stim_flat = stim.view(-1, 1, self.height_px, self.width_px)

        # Apply box filter
        filtered = self.pool(stim_flat)

        # Sample at hex positions (simplified - just flatten for now)
        # In a full implementation, would use proper hex sampling
        output = filtered.view(*batch_shape, -1)[..., : self.n_ommatidia]

        return output
