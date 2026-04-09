"""Receptive field visualization for hexagonal grids.

Adapted from flyvis.analysis.visualization.plots.
Supports multiple coordinate formats: axial (q,r), zigzag (x,y).

Code adapted from flyvis (MIT License).
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ...utils.hex.transform import to_pixel
from .static import scatter


def kernel(
    c1: NDArray,
    c2: NDArray,
    values: NDArray,
    coord_format: Literal["axial", "zigzag"] = "axial",
    figsize: tuple[float, float] = (1, 1),
    cmap: str = "seismic",
    vmin: float | None = None,
    vmax: float | None = None,
    midpoint: float | None = None,
    annotate: bool = False,
    annotate_coords: bool = False,
    edgecolor: str | None = None,
    fontsize: int = 5,
    cbar: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot receptive field kernel on hexagonal lattice.

    Args:
        c1, c2: Coordinates
        values: Kernel values
        coord_format: Coordinate system
        figsize: Figure size
        cmap: Colormap
        vmin, vmax: Value limits
        midpoint: Center value for colormap
        annotate: Whether to annotate values
        annotate_coords: Whether to annotate coordinates
        edgecolor: Hexagon edge color
        fontsize: Font size for annotations
        cbar: Whether to show colorbar
        ax: Optional axes

    Returns:
        Figure and axes objects
    """
    # Convert coordinates to pixel space for annotations
    if coord_format == "axial":
        x, y = to_pixel(c1, c2)
    elif coord_format == "zigzag":
        from ...utils.hex.offset import zigzag_to_axial

        qz, rz = zigzag_to_axial(c1, c2)
        x, y = to_pixel(qz, rz)
    else:
        raise ValueError(f"Unknown coord_format: {coord_format}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine value range
    vmin = vmin if vmin is not None else np.nanmin(values)
    vmax = vmax if vmax is not None else np.nanmax(values)

    if midpoint is not None:
        # Symmetric colormap around midpoint
        max_abs = max(abs(vmax - midpoint), abs(vmin - midpoint))
        vmin = midpoint - max_abs
        vmax = midpoint + max_abs

    # Create scatter plot
    scatter(
        c1,
        c2,
        values,
        coord_format=coord_format,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    # Add annotations
    if annotate or annotate_coords:
        for xi, yi, vi, qi, ri in zip(x, y, values, c1, c2):
            if annotate:
                ax.text(
                    xi, yi, f"{vi:.2f}", ha="center", va="center", fontsize=fontsize
                )
            elif annotate_coords:
                ax.text(
                    xi,
                    yi,
                    f"({int(qi)},{int(ri)})",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )

    ax.set_aspect("equal")
    ax.axis("off")

    if cbar:
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        plt.colorbar(sm, ax=ax)

    return fig, ax


def strf(
    time: NDArray,
    rf: NDArray,
    c1: NDArray,
    c2: NDArray,
    coord_format: Literal["axial", "zigzag"] = "axial",
    figsize: tuple[float, float] = (5, 1),
    fontsize: int = 6,
    hlines: bool = True,
    vlines: bool = True,
) -> tuple[Figure, NDArray]:
    """Plot Spatio-Temporal Receptive Field (STRF).

    Args:
        time: Time points, shape (n_time,)
        rf: Receptive field data, shape (n_time, n_hexes)
        c1, c2: Coordinates
        coord_format: Coordinate system
        figsize: Figure size
        fontsize: Font size
        hlines: Whether to draw horizontal lines
        vlines: Whether to draw vertical lines

    Returns:
        Figure and axes array
    """
    n_time = len(time)
    n_cols = min(n_time, 10)
    n_rows = (n_time + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    vmax = np.abs(rf).max()
    vmin = -vmax

    for i, (t, ax) in enumerate(zip(time, axes)):
        kernel(
            c1,
            c2,
            rf[i],
            coord_format=coord_format,
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.set_title(f"t={t:.2f}", fontsize=fontsize)

    # Hide unused axes
    for ax in axes[n_time:]:
        ax.axis("off")

    plt.tight_layout()
    return fig, axes


class ReceptiveFieldViewer:
    """Interactive receptive field viewer for connectome analysis."""

    def __init__(
        self,
        edges: pd.DataFrame,
        target_type: str,
        coord_format: Literal["axial", "zigzag"] = "axial",
        c1_col: str = "u",
        c2_col: str = "v",
        weight_col: str = "weight",
    ):
        """Initialize RF viewer.

        Args:
            edges: Edge dataframe with connectivity
            target_type: Target cell type
            coord_format: Coordinate system
            c1_col: Column name for first coordinate
            c2_col: Column name for second coordinate
            weight_col: Column name for edge weights
        """
        self.edges = edges
        self.target_type = target_type
        self.coord_format = coord_format
        self.c1_col = c1_col
        self.c2_col = c2_col
        self.weight_col = weight_col

    def plot(
        self,
        source: str,
        max_extent: int | None = None,
        **kwargs,
    ) -> Figure:
        """Plot receptive field from source to target.

        Args:
            source: Source cell type
            max_extent: Maximum extent to plot
            **kwargs: Additional arguments for kernel()

        Returns:
            Figure object
        """
        # Filter edges
        mask = self.edges["target_type"] == self.target_type
        if source:
            mask = mask & (self.edges["source_type"] == source)

        filtered = self.edges[mask]

        if len(filtered) == 0:
            raise ValueError(f"No edges found for {source} -> {self.target_type}")

        # Get coordinates and weights
        c1 = filtered[self.c1_col].to_numpy()
        c2 = filtered[self.c2_col].to_numpy()
        values = filtered[self.weight_col].to_numpy()

        # Limit extent if requested
        if max_extent is not None:
            from ...utils.hex.distance import radius

            r = radius(c1, c2)
            mask = r <= max_extent
            c1, c2, values = c1[mask], c2[mask], values[mask]

        fig, _ = kernel(c1, c2, values, coord_format=self.coord_format, **kwargs)
        return fig
