"""Hexagonal grid animations using matplotlib.

Adapted from flyvis.analysis.animations.
Supports multiple coordinate formats: axial (q,r), zigzag (x,y), pixel (px,py).

Code adapted from flyvis (MIT License).
"""

from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ...utils.hex.transform import to_pixel


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
        values: NDArray,
        c1: NDArray,
        c2: NDArray,
        coord_format: Literal["axial", "zigzag", "pixel"] = "axial",
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

        # Convert to pixel coordinates
        if coord_format == "axial":
            self.x, self.y = to_pixel(c1, c2)
        elif coord_format == "zigzag":
            from ...utils.hex.offset import zigzag_to_axial

            qz, rz = zigzag_to_axial(c1, c2)
            self.x, self.y = to_pixel(qz, rz)
        elif coord_format == "pixel":
            self.x, self.y = c1, c2
        else:
            raise ValueError(f"Unknown coord_format: {coord_format}")

        # Create figure
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
            self.fig = ax.figure

        self.vmin = vmin if vmin is not None else np.nanmin(values)
        self.vmax = vmax if vmax is not None else np.nanmax(values)

        self.cmap = plt.get_cmap(cmap)
        self.interval = interval

        # Initialize scatter
        self._init_plot()

    def _init_plot(self):
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

    def save(self, path: str, **kwargs) -> None:
        """Save animation to file."""
        anim = self._create_animation()
        anim.save(path, **kwargs)
        plt.close(self.fig)

    def show(self) -> None:
        """Display animation."""
        self._create_animation()
        plt.show()


class HexFlow:
    """Animated vector field on hex grid.

    Args:
        flow: Flow data, shape (n_frames, 2, n_hexes) where
            [:, 0, :] = dc1, [:, 1, :] = dc2
        c1, c2: Coordinates (interpreted based on coord_format)
        coord_format: "axial" (q,r), "zigzag" (x,y), or "pixel" (px,py)
        scale: Vector scale factor
        cwheel: Show colorwheel for direction encoding

    Example:
        >>> anim = HexFlow(flow_data, q, r, coord_format="axial")
        >>> anim.save("flow.mp4")
    """

    def __init__(
        self,
        flow: NDArray,
        c1: NDArray,
        c2: NDArray,
        coord_format: Literal["axial", "zigzag", "pixel"] = "axial",
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

        # Convert to pixel coordinates
        if coord_format == "axial":
            self.x, self.y = to_pixel(c1, c2)
        elif coord_format == "zigzag":
            from ...utils.hex.offset import zigzag_to_axial

            qz, rz = zigzag_to_axial(c1, c2)
            self.x, self.y = to_pixel(qz, rz)
        elif coord_format == "pixel":
            self.x, self.y = c1, c2
        else:
            raise ValueError(f"Unknown coord_format: {coord_format}")

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

    def _init_plot(self):
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

    def save(self, path: str, **kwargs) -> None:
        """Save animation to file."""
        anim = self._create_animation()
        anim.save(path, **kwargs)
        plt.close(self.fig)

    def show(self) -> None:
        """Display animation."""
        self._create_animation()
        plt.show()


class HexAnimationCollector:
    """Collect and compose multiple hex animations.

    Allows synchronizing multiple hex visualizations (e.g., scatter +
    flow) in a single figure.
    """

    def __init__(self, animations: List[HexScatter | HexFlow]):
        self.animations = animations
        self.fig = animations[0].fig if animations else None

    def save(self, path: str, **kwargs) -> None:
        """Save composed animation."""
        # For simplicity, just save the first animation
        # Full implementation would synchronize all animations
        if self.animations:
            self.animations[0].save(path, **kwargs)
