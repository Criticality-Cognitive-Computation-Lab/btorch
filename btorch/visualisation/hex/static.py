"""Static hex plots using matplotlib.

Supports multiple coordinate formats: axial (q,r), zigzag (x,y), pixel (px,py).
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from numpy.typing import NDArray

from ...utils.hex.transform import to_pixel


def _hex_axis_directions(
    size: float = 1.0, orientation: str = "pointy", alignment: str = "vertex"
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute 3-axis reference directions for drawing.

    Args:
        size: Arrow length in pixel units.
        orientation: "pointy" or "flat".
        alignment: "vertex" (axes through hex vertices) or
            "edge" (axes parallel to hex edges).

    Returns:
        Three (dx, dy) direction vectors for q, r, s axes.
    """
    # Compute natural pixel directions for +q, +r, +s
    q_dir = np.array(
        to_pixel(np.array([1.0]), np.array([0.0]), size=size, orientation=orientation)
    )
    r_dir = np.array(
        to_pixel(np.array([0.0]), np.array([1.0]), size=size, orientation=orientation)
    )
    s_dir = np.array(
        to_pixel(np.array([-1.0]), np.array([1.0]), size=size, orientation=orientation)
    )

    # Stack as (3, 2) array
    dirs = np.stack([q_dir, r_dir, s_dir], axis=0).squeeze()

    if alignment == "edge":
        # Rotate all directions by 30° (π/6)
        angle = np.pi / 6
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        dirs = dirs @ rot.T

    return dirs[0], dirs[1], dirs[2]


def draw_hex_axes(
    ax: plt.Axes,
    origin: tuple[float, float] = (0.0, 0.0),
    size: float = 1.0,
    orientation: str = "pointy",
    alignment: str = "vertex",
    colors: tuple[str, str, str] = ("red", "green", "blue"),
    labels: tuple[str, str, str] = ("q", "r", "s"),
    linewidth: float = 1.5,
    head_width: float = 0.3,
) -> None:
    """Draw 3-axis reference arrows on a hex grid plot.

    Args:
        ax: Matplotlib axes.
        origin: (x, y) position for the axis origin.
        size: Length of each arrow in pixel units.
        orientation: "pointy" or "flat".
        alignment: "vertex" (axes through hex vertices) or
            "edge" (axes parallel to hex edges).
        colors: Colors for q, r, s arrows.
        labels: Labels for q, r, s arrows.
        linewidth: Arrow line width.
        head_width: Arrow head width.
    """
    q_dir, r_dir, s_dir = _hex_axis_directions(
        size=size, orientation=orientation, alignment=alignment
    )

    ox, oy = origin
    for (dx, dy), color, label in zip([q_dir, r_dir, s_dir], colors, labels):
        ax.arrow(
            ox,
            oy,
            dx,
            dy,
            head_width=head_width,
            head_length=head_width * 0.6,
            fc=color,
            ec=color,
            linewidth=linewidth,
            length_includes_head=True,
            zorder=1000,
        )
        # Label near arrow tip
        ax.text(
            ox + dx * 1.15,
            oy + dy * 1.15,
            label,
            color=color,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=1001,
        )


def hex_compass(
    ax: plt.Axes,
    loc: str = "lower left",
    size: float = 0.15,
    alignment: str = "vertex",
    colors: tuple[str, str, str] = ("#66c2a5", "#fc8d62", "#8da0cb"),
    label_color: str = "#333333",
    bg_color: str = "white",
    alpha: float = 0.9,
    border_color: str = "#cccccc",
    border_width: float = 0.8,
) -> plt.Axes:
    """Draw a hex-direction compass rose as an inset on the axes.

    This creates a small map-style compass showing all six axial
    directions (+q, -q, +r, -r, +s, -s) radiating from a central circle,
    similar to the style used by Red Blob Games.

    Args:
        ax: Main matplotlib axes.
        loc: Compass location, e.g. "lower left", "upper right".
        size: Compass size as fraction of the main axes (0-1).
        alignment: "vertex" (axes through hex vertices) or
            "edge" (axes parallel to hex edges).
        colors: Colors for q, r, s axis pairs.
        label_color: Text color for direction labels.
        bg_color: Background fill color for the compass box.
        alpha: Background transparency.
        border_color: Border color for the compass box.
        border_width: Border line width.

    Returns:
        The inset axes containing the compass.
    """
    from matplotlib.patches import Circle, FancyBboxPatch

    # Create inset axes
    inset = ax.inset_axes((0, 0, size, size), transform=ax.transAxes)

    # Position based on loc string
    pos_map = {
        "upper left": (0.02, 0.98 - size),
        "upper right": (0.98 - size, 0.98 - size),
        "lower left": (0.02, 0.02),
        "lower right": (0.98 - size, 0.02),
    }
    x0, y0 = pos_map.get(loc, pos_map["lower left"])
    inset.set_position((x0, y0, size, size))

    # Set up the coordinate system
    inset.set_xlim(-1.4, 1.4)
    inset.set_ylim(-1.4, 1.4)
    inset.set_aspect("equal")
    inset.axis("off")

    # Draw background box with border
    bbox = FancyBboxPatch(
        (-1.35, -1.35),
        2.7,
        2.7,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=border_width,
        alpha=alpha,
        zorder=0,
    )
    inset.add_patch(bbox)

    # Compute 6 directions
    q_dir = np.array([1.0, 0.0])
    r_dir = np.array([0.5, np.sqrt(3) / 2])
    s_dir = np.array([-0.5, np.sqrt(3) / 2])

    if alignment == "edge":
        angle = np.pi / 6
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        q_dir = rot @ q_dir
        r_dir = rot @ r_dir
        s_dir = rot @ s_dir

    directions = [
        (q_dir, "+q", colors[0]),
        (-q_dir, "-q", colors[0]),
        (r_dir, "+r", colors[1]),
        (-r_dir, "-r", colors[1]),
        (s_dir, "+s", colors[2]),
        (-s_dir, "-s", colors[2]),
    ]

    arrow_len = 0.85
    label_offset = 1.15

    for vec, label, color in directions:
        dx, dy = vec * arrow_len
        lx, ly = vec * label_offset

        inset.annotate(
            "",
            xy=(dx, dy),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.5,
                mutation_scale=10,
            ),
            zorder=2,
        )

        # Label
        inset.text(
            lx,
            ly,
            label,
            color=label_color,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=3,
        )

    # Central circle
    circle = Circle(
        (0, 0), 0.15, facecolor="white", edgecolor="#888888", linewidth=1, zorder=4
    )
    inset.add_patch(circle)

    return inset


def scatter(
    c1: NDArray,
    c2: NDArray,
    values: NDArray,
    coord_format: Literal["axial", "zigzag", "flywire", "pixel"] = "axial",
    size: float = 1.0,
    orientation: str = "pointy",
    rotation_deg: float = 0.0,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    edgecolor: str | None = None,
    edgewidth: float = 0.5,
    alpha: float = 1.0,
    figsize: tuple[float, float] = (6, 6),
    ax: plt.Axes | None = None,
    title: str | None = None,
    axes_alignment: str | None = None,
    compass: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Colored hex scatter.

    Args:
        c1, c2: Coordinates (interpreted based on coord_format)
        values: Color values for each hex
        coord_format: Which coordinate system to use
        size: Hexagon size (for axial/zigzag conversion to pixel)
        orientation: "pointy" or "flat"
        rotation_deg: Optional global display rotation in degrees
        cmap: Matplotlib colormap name
        vmin, vmax: Color limits
        edgecolor: Hexagon edge color
        edgewidth: Hexagon edge width
        alpha: Transparency (0-1)
        figsize: Figure size
        ax: Optional axes to plot on
        title: Plot title
        axes_alignment: If "vertex" or "edge", draw q/r/s reference axes
        compass: If "vertex" or "edge", draw a hex compass rose inset

    Returns:
        Figure and axes objects
    """
    # Convert coordinates to pixel space
    # Track effective orientation for hexagon patch drawing
    effective_orientation = orientation
    if coord_format == "axial":
        x, y = to_pixel(c1, c2, size=size, orientation=orientation)
    elif coord_format == "zigzag":
        from ...utils.hex.offset import zigzag_to_pixel

        x, y = zigzag_to_pixel(c1, c2, size=size)
    elif coord_format == "flywire":
        from ...utils.hex.offset import flywire_to_pixel

        x, y = flywire_to_pixel(c1, c2, size=size, rotation_deg=rotation_deg)
        effective_orientation = "flat"
    elif coord_format == "pixel":
        x, y = c1, c2
    else:
        raise ValueError(f"Unknown coord_format: {coord_format}")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Normalize values for coloring
    vmin = vmin if vmin is not None else np.nanmin(values)
    vmax = vmax if vmax is not None else np.nanmax(values)

    # Create hexagon patches
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    for xi, yi, vi in zip(x, y, values):
        if np.isnan(vi):
            color = "white"
        else:
            color = cmap_obj(norm(vi))

        # Determine hexagon orientation based on effective layout
        if effective_orientation == "pointy":
            hex_orientation = 0  # pointy top
        else:
            hex_orientation = np.pi / 6  # flat top

        hex_patch = RegularPolygon(
            (xi, yi),
            numVertices=6,
            radius=size * 0.95,  # slight gap between hexes
            orientation=hex_orientation,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=edgewidth,
            alpha=alpha,
        )
        ax.add_patch(hex_patch)

    # Set limits and aspect
    ax.set_aspect("equal")
    ax.set_xlim(x.min() - size, x.max() + size)
    ax.set_ylim(y.min() - size, y.max() + size)

    if title:
        ax.set_title(title)

    if axes_alignment in ("vertex", "edge"):
        draw_hex_axes(
            ax,
            origin=(x.min() + size, y.min() + size),
            size=size * 2,
            orientation=orientation,
            alignment=axes_alignment,
        )

    if compass in ("vertex", "edge"):
        hex_compass(ax, alignment=compass, loc="lower left")

    ax.axis("off")

    return fig, ax


def flow(
    c1: NDArray,
    c2: NDArray,
    dc1: NDArray,
    dc2: NDArray,
    coord_format: Literal["axial", "zigzag", "pixel"] = "axial",
    size: float = 1.0,
    orientation: str = "pointy",
    scale: float = 1.0,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (6, 6),
    ax: plt.Axes | None = None,
    title: str | None = None,
    axes_alignment: str | None = None,
    compass: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Vector field on hex grid.

    Args:
        c1, c2: Coordinates
        dc1, dc2: Vector components
        coord_format: Coordinate system
        size: Hexagon size
        orientation: "pointy" or "flat"
        scale: Vector scale factor
        cmap: Colormap for vector magnitude
        figsize: Figure size
        ax: Optional axes
        title: Plot title
        axes_alignment: If "vertex" or "edge", draw q/r/s reference axes
        compass: If "vertex" or "edge", draw a hex compass rose inset

    Returns:
        Figure and axes objects
    """
    # Convert coordinates to pixel space
    if coord_format == "axial":
        x, y = to_pixel(c1, c2, size=size, orientation=orientation)
        # Convert flow vectors too
        from ...utils.hex.transform import to_pixel as hex_to_pixel

        # Calculate pixel displacement for unit vectors
        dx_pixel, dy_pixel = hex_to_pixel(
            c1 + dc1 * scale, c2 + dc2 * scale, size=size, orientation=orientation
        )
        dx = dx_pixel - x
        dy = dy_pixel - y
    elif coord_format == "zigzag":
        from ...utils.hex.offset import zigzag_to_axial

        qz, rz = zigzag_to_axial(c1, c2)
        x, y = to_pixel(qz, rz, size=size, orientation=orientation)
        dx, dy = dc1 * scale, dc2 * scale
    elif coord_format == "pixel":
        x, y = c1, c2
        dx, dy = dc1 * scale, dc2 * scale
    else:
        raise ValueError(f"Unknown coord_format: {coord_format}")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Calculate magnitudes for coloring
    magnitudes = np.sqrt(dx**2 + dy**2)
    vmin, vmax = np.nanmin(magnitudes), np.nanmax(magnitudes)

    # Plot quiver
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(magnitudes))

    ax.quiver(x, y, dx, dy, color=colors, scale_units="xy", angles="xy", scale=1)

    ax.set_aspect("equal")
    ax.set_xlim(x.min() - size * 2, x.max() + size * 2)
    ax.set_ylim(y.min() - size * 2, y.max() + size * 2)

    if title:
        ax.set_title(title)

    if axes_alignment in ("vertex", "edge"):
        draw_hex_axes(
            ax,
            origin=(x.min() + size, y.min() + size),
            size=size * 2,
            orientation=orientation,
            alignment=axes_alignment,
        )

    if compass in ("vertex", "edge"):
        hex_compass(ax, alignment=compass, loc="lower left")

    ax.axis("off")

    return fig, ax


def grid(
    radius: int,
    coord_format: Literal["axial", "zigzag"] = "axial",
    annotate: bool = False,
    size: float = 1.0,
    orientation: Literal["pointy", "flat"] = "pointy",
    figsize: tuple[float, float] = (6, 6),
    ax: plt.Axes | None = None,
    title: str | None = None,
    axes_alignment: str | None = None,
    compass: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Coordinate grid with optional labels.

    Args:
        radius: Grid radius
        coord_format: Coordinate format
        annotate: Whether to annotate coordinates
        size: Hexagon size
        orientation: "pointy" or "flat"
        figsize: Figure size
        ax: Optional axes to plot on
        title: Plot title
        axes_alignment: If "vertex" or "edge", draw q/r/s reference axes
        compass: If "vertex" or "edge", draw a hex compass rose inset

    Returns:
        Figure and axes objects
    """
    from ...utils.hex.coords import disk

    q, r = disk(radius)

    # Convert to pixel space
    if coord_format == "axial":
        x, y = to_pixel(q, r, size=size, orientation=orientation)
        labels = [f"({qi},{ri})" for qi, ri in zip(q, r)]
    elif coord_format == "zigzag":
        from ...utils.hex.offset import axial_to_zigzag, zigzag_to_pixel

        zx, zy = axial_to_zigzag(q, r)
        x, y = zigzag_to_pixel(zx, zy, size=size)
        labels = [f"({zi[0]},{zi[1]})" for zi in zip(zx, zy)]
    else:
        raise ValueError(f"Unknown coord_format: {coord_format}")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Draw hexagons
    hex_orientation = 0 if orientation == "pointy" else np.pi / 6

    for xi, yi, label in zip(x, y, labels):
        hex_patch = RegularPolygon(
            (xi, yi),
            numVertices=6,
            radius=size * 0.95,
            orientation=hex_orientation,
            facecolor="white",
            edgecolor="gray",
            linewidth=0.5,
        )
        ax.add_patch(hex_patch)

        if annotate:
            ax.text(xi, yi, label, ha="center", va="center", fontsize=6)

    ax.set_aspect("equal")
    ax.set_xlim(x.min() - size, x.max() + size)
    ax.set_ylim(y.min() - size, y.max() + size)

    if title:
        ax.set_title(title)

    if axes_alignment in ("vertex", "edge"):
        draw_hex_axes(
            ax,
            origin=(x.min() + size, y.min() + size),
            size=size * 2,
            orientation=orientation,
            alignment=axes_alignment,
        )

    if compass in ("vertex", "edge"):
        hex_compass(ax, alignment=compass, loc="lower left")

    ax.axis("off")

    return fig, ax


def scatter_from_index(
    df,
    size: float = 1.0,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    edgecolor: str | None = None,
    edgewidth: float = 0.5,
    alpha: float = 1.0,
    figsize: tuple[float, float] = (6, 6),
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Hex scatter from DataFrame with string "x,y" indices.

    Parses indices formatted as "-12,34" (double-width coordinates)
    commonly used in connectome datasets.

    Args:
        df: pd.Series or pd.DataFrame with "x,y" string indices
        size: Hexagon size
        cmap: Matplotlib colormap name
        vmin, vmax: Color limits
        edgecolor: Hexagon edge color
        edgewidth: Hexagon edge width
        alpha: Transparency (0-1)
        figsize: Figure size
        ax: Optional axes to plot on
        title: Plot title

    Returns:
        Figure and axes objects

    Example:
        >>> import pandas as pd
        >>> df = pd.Series([0.5, 0.3, 0.8], index=["0,0", "1,0", "0,2"])
        >>> fig, ax = scatter_from_index(df)
    """
    import pandas as pd

    from ...utils.hex.doubled import doublewidth_to_axial

    # Handle DataFrame - use first column or flatten
    if isinstance(df, pd.DataFrame):
        if len(df.columns) == 1:
            series = df.iloc[:, 0]
        else:
            # Use mean across columns for visualization
            series = df.mean(axis=1)
    else:
        series = df

    # Parse "x,y" indices into double-width coordinates
    coords = [tuple(map(float, idx.split(","))) for idx in series.index]
    x_vals, y_vals = zip(*coords)

    # Convert double-width to axial
    q, r = doublewidth_to_axial(np.array(x_vals), np.array(y_vals))

    # Call main scatter function
    return scatter(
        q,
        r,
        series.values,
        coord_format="axial",
        size=size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor=edgecolor,
        edgewidth=edgewidth,
        alpha=alpha,
        figsize=figsize,
        ax=ax,
        title=title,
    )


def looming_stimulus(
    start_coords: list[str], all_coords: list[str], n_time: int = 4
) -> list[list[str]]:
    """Generate expanding stimulus pattern on hex grid.

    Creates a sequence of expanding hex regions starting from start_coords,
    simulating a looming object approaching the retina.

    Args:
        start_coords: List of "x,y" strings for starting hexes
        all_coords: List of all valid "x,y" hex coordinates in grid
        n_time: Number of time steps for expansion

    Returns:
        List of lists, where each inner list contains "x,y" coordinates
        stimulated at that time step

    Example:
        >>> all_coords = ["0,0", "1,0", "0,2", "-1,0"]
        >>> looming_stimulus(["0,0"], all_coords, n_time=3)
        [['0,0'], ['0,0', '1,0', '-1,0', '0,2', '0,-2'], [...]]
    """
    # Parse all coordinates to get grid bounds
    coords = [tuple(map(float, c.split(","))) for c in all_coords]
    x_vals, y_vals = zip(*coords)

    # Build coordinate lookup
    x_sorted = sorted(set(x_vals))
    x_to_rank = {x: i for i, x in enumerate(x_sorted)}
    rank_to_x = {i: x for x, i in x_to_rank.items()}

    y_sorted = sorted(set(y_vals))
    y_to_rank = {y: i for i, y in enumerate(y_sorted)}
    rank_to_y = {i: y for y, i in y_to_rank.items()}

    # Parse start coordinates
    start = [tuple(map(float, c.split(","))) for c in start_coords]

    # Generate expanding stimulus
    stimulus = [start]
    current = start.copy()

    for _ in range(n_time):
        next_hexes = current.copy()
        for x, y in current:
            # Hexes above and below (y ± 2 in double-width)
            if y_to_rank[y] + 2 in rank_to_y:
                next_hexes.append((x, rank_to_y[y_to_rank[y] + 2]))
            if y_to_rank[y] - 2 in rank_to_y:
                next_hexes.append((x, rank_to_y[y_to_rank[y] - 2]))
            # Hexes to the left (x + 1, y ± 1)
            if x_to_rank[x] + 1 in rank_to_x:
                if y_to_rank[y] + 1 in rank_to_y:
                    next_hexes.append(
                        (rank_to_x[x_to_rank[x] + 1], rank_to_y[y_to_rank[y] + 1])
                    )
                if y_to_rank[y] - 1 in rank_to_y:
                    next_hexes.append(
                        (rank_to_x[x_to_rank[x] + 1], rank_to_y[y_to_rank[y] - 1])
                    )
            # Hexes to the right (x - 1, y ± 1)
            if x_to_rank[x] - 1 in rank_to_x:
                if y_to_rank[y] + 1 in rank_to_y:
                    next_hexes.append(
                        (rank_to_x[x_to_rank[x] - 1], rank_to_y[y_to_rank[y] + 1])
                    )
                if y_to_rank[y] - 1 in rank_to_y:
                    next_hexes.append(
                        (rank_to_x[x_to_rank[x] - 1], rank_to_y[y_to_rank[y] - 1])
                    )

        # Remove duplicates and store
        current = list(set(next_hexes))
        stimulus.append(current)

    # Format back to strings
    result = []
    for hex_list in stimulus:
        formatted = []
        for x, y in hex_list:
            # Format to remove .0 for integers
            x_str = str(int(x)) if x == int(x) else str(x)
            y_str = str(int(y)) if y == int(y) else str(y)
            formatted.append(f"{x_str},{y_str}")
        result.append(formatted)

    return result
