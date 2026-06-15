"""Hexagonal grid animations using matplotlib.

Adapted from flyvis.analysis.animations.
Supports multiple coordinate formats: axial (q,r), zigzag (x,y), pixel (px,py).

Code adapted from flyvis (MIT License).
"""

from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes

from ...utils.hex.resolve import resolve_hex


class HexScatter:
    """Animated hex scatter plot for time-series data.

    Args:
        values: Time-series data, shape (n_frames, n_hexes)
        c1, c2: Coordinates (interpreted based on coord_format)
        coord_format: "axial" (q,r), "zigzag" (x,y), or "pixel" (px,py)
        figsize: Figure size
        cmap: Colormap
        vmin, vmax: Color limits
        interval: Animation interval in milliseconds

    Example:
        >>> anim = HexScatter(data, q, r, coord_format="axial")
        >>> anim.save("output.mp4")
    """

    def __init__(
        self,
        values: np.ndarray,
        c1: np.ndarray,
        c2: np.ndarray,
        coord_format: Literal["axial", "zigzag", "pixel"] = "axial",
        layout: str | None = None,
        figsize: tuple[float, float] = (6, 6),
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        interval: int = 200,
        ax: Optional[Axes] = None,
    ):
        self.values = values
        self.c1 = c1
        self.c2 = c2
        self.coord_format = coord_format
        self.n_frames = values.shape[0]

        effective = layout or ("flat" if coord_format == "zigzag" else "pointy")
        _, _, self.x, self.y = resolve_hex(
            c1,
            c2,
            coord_format=coord_format,
            layout=effective,
        )

        # Create figure
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
            self.fig = ax.figure

        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.interval = interval

        # Initialize scatter
        self._init_plot()

    def _init_plot(self) -> None:
        """Initialize the plot."""
        self.sc = self.ax.scatter(
            self.x,
            self.y,
            c=self.values[0],
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            s=100,
        )
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        plt.colorbar(self.sc, ax=self.ax)

    def update(self, frame: int) -> None:
        """Update animation for given frame."""
        self.sc.set_array(self.values[frame])

    def _create_animation(self) -> FuncAnimation:
        """Create the animation object."""
        return FuncAnimation(
            self.fig,
            self.update,
            frames=self.n_frames,
            interval=self.interval,
            blit=False,
        )

    def save(self, path: str, **kwargs: Any) -> None:
        """Save animation to file."""
        anim = self._create_animation()
        anim.save(path, **kwargs)
        plt.close(self.fig)

    def show(self) -> None:
        """Display animation."""
        self._create_animation()
        plt.show()


class HexQuiver:
    """Animated vector field on hex grid.

    Args:
        flow: Flow data, shape (n_frames, 2, n_hexes) where
            [:, 0, :] = dc1, [:, 1, :] = dc2
        c1, c2: Coordinates (interpreted based on coord_format)
        coord_format: "axial" (q,r), "zigzag" (x,y), or "pixel" (px,py)
        scale: Vector scale factor
        cwheel: Show colorwheel for direction encoding

    Example:
        >>> anim = HexQuiver(flow_data, q, r, coord_format="axial")
        >>> anim.save("flow.mp4")
    """

    def __init__(
        self,
        flow: np.ndarray,
        c1: np.ndarray,
        c2: np.ndarray,
        coord_format: Literal["axial", "zigzag", "pixel"] = "axial",
        layout: str | None = None,
        figsize: tuple[float, float] = (6, 6),
        scale: float = 1.0,
        cmap: str = "hsv",
        cwheel: bool = True,
        interval: int = 200,
        ax: Optional[Axes] = None,
    ):
        self.flow = flow
        self.c1 = c1
        self.c2 = c2
        self.coord_format = coord_format
        self.n_frames = flow.shape[0]
        self.scale = scale

        effective = layout or ("flat" if coord_format == "zigzag" else "pointy")
        _, _, self.x, self.y = resolve_hex(
            c1,
            c2,
            coord_format=coord_format,
            layout=effective,
        )

        # Create figure
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
            self.fig = ax.figure

        self.cmap = cmap
        self.cwheel = cwheel
        self.interval = interval

        # Initialize quiver
        self._init_plot()

    def _init_plot(self) -> None:
        """Initialize the plot."""
        # Calculate initial flow
        u = self.flow[0, 0] * self.scale
        v = self.flow[0, 1] * self.scale

        # Calculate angles for coloring
        angles = np.arctan2(v, u)

        self.quiver = self.ax.quiver(
            self.x,
            self.y,
            u,
            v,
            angles,
            cmap=self.cmap,
            scale_units="xy",
            angles="xy",
            scale=1,
        )
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def update(self, frame: int) -> None:
        """Update animation for given frame."""
        u = self.flow[frame, 0] * self.scale
        v = self.flow[frame, 1] * self.scale
        angles = np.arctan2(v, u)

        self.quiver.set_UVC(u, v, angles)

    def _create_animation(self) -> FuncAnimation:
        """Create the animation object."""
        return FuncAnimation(
            self.fig,
            self.update,
            frames=self.n_frames,
            interval=self.interval,
            blit=False,
        )

    def save(self, path: str, **kwargs: Any) -> None:
        """Save animation to file."""
        anim = self._create_animation()
        anim.save(path, **kwargs)
        plt.close(self.fig)

    def show(self) -> None:
        """Display animation."""
        self._create_animation()
        plt.show()
