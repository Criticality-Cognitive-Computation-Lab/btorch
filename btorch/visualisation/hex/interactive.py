"""Interactive hex heatmaps using Plotly.

Hex polygons are drawn as SVG path shapes in data coordinates — no pixel
math, no marker-size estimation.  A single invisible scatter trace at
hex centres provides hover, colour bar, and optional text labels. Plotly
handles all scaling (zoom, resize, export).

Coordinate conversion utilities adapted from flyvis (MIT License) by
Yijie Yin (yijieyin).

Supports axial (q,r), zigzag (x,y), pixel (px,py), and connectome-style
string indices ("x,y" double-width coordinates).
"""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from ...utils.hex.offset import (
    axial_to_zigzag,
    flywire_to_pixel,
    zigzag_to_axial,
    zigzag_to_pixel,
)
from ...utils.hex.transform import to_pixel as hex_to_pixel


_POINTS_PER_INCH = 72
_MM_PER_INCH = 25.4

_DEFAULT_STYLE: dict = {
    "font_type": "arial",
    "linecolor": "black",
    "papercolor": "rgba(255,255,255,255)",
}

_DEFAULT_SIZING: dict = {
    "fig_width": 260,
    "fig_height": 220,
    "fig_margin": 0,
    "fsize_ticks_pt": 20,
    "fsize_title_pt": 20,
    "ticklen": 15,
    "tickwidth": 5,
    "axislinewidth": 3,
    "hex_line_color": "lightgrey",
    "hex_line_width": 0.9,
    "cbar_thickness": 20,
    "cbar_len": 0.75,
}

_DEFAULT_COLORSCALE = [[0, "rgb(255, 255, 255)"], [1, "rgb(0, 20, 200)"]]


# -- internal helpers ------------------------------------------------------


def _merge_style(user: dict | None) -> dict:
    """Merge user style overrides with defaults."""
    merged = _DEFAULT_STYLE.copy()
    if user is not None:
        merged.update(user)
    return merged


def _merge_sizing(user: dict | None) -> dict:
    """Merge user sizing overrides with defaults."""
    merged = _DEFAULT_SIZING.copy()
    if user is not None:
        merged.update(user)
    return merged


def _compute_pixel_dims(sizing: dict, dpi: int) -> tuple[float, float, float, float]:
    """Return (area_width, area_height, fsize_ticks_px, fsize_title_px)."""
    px_per_mm = dpi / _MM_PER_INCH
    w = (sizing["fig_width"] - sizing["fig_margin"]) * px_per_mm
    h = (sizing["fig_height"] - sizing["fig_margin"]) * px_per_mm
    f_ticks = sizing["fsize_ticks_pt"] / _POINTS_PER_INCH * dpi
    f_title = sizing["fsize_title_pt"] / _POINTS_PER_INCH * dpi
    return w, h, f_ticks, f_title


def _resolve_coords(
    c1: np.ndarray,
    c2: np.ndarray,
    coord_format: str,
    orientation: str = "flat",
    layout: str | None = None,
    rotation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert coordinates to pixel space and derive axial/zigzag.

    Returns:
        (q, r, zigzag_x, zigzag_y, pixel_x, pixel_y)
    """
    if coord_format in ("zigzag", "flywire"):
        coord_format = "zigzag"

    if coord_format == "axial":
        q = np.asarray(c1, dtype=float)
        r = np.asarray(c2, dtype=float)
        zx, zy = axial_to_zigzag(np.rint(q).astype(int), np.rint(r).astype(int))
        if layout == "flywire":
            x, y = flywire_to_pixel(q, r, rotation_deg=rotation_deg)
        else:
            x, y = hex_to_pixel(q, r, orientation=orientation)
        return q, r, zx.astype(float), zy.astype(float), x, y

    if coord_format == "zigzag":
        zx = np.asarray(c1, dtype=float)
        zy = np.asarray(c2, dtype=float)
        q, r = zigzag_to_axial(np.rint(zx).astype(int), np.rint(zy).astype(int))
        x, y = zigzag_to_pixel(zx, zy)
        return q.astype(float), r.astype(float), zx, zy, x, y

    if coord_format == "pixel":
        x = np.asarray(c1, dtype=float)
        y = np.asarray(c2, dtype=float)
        nan = np.full_like(x, np.nan, dtype=float)
        return nan, nan, nan, nan, x, y

    raise ValueError(f"Unknown coord_format: {coord_format}")


def _colorbar_dict(style: dict, sizing: dict, f_ticks: float, f_title: float) -> dict:
    """Build the Plotly colorbar configuration dict."""
    return {
        "orientation": "v",
        "outlinecolor": style["linecolor"],
        "outlinewidth": sizing["axislinewidth"],
        "thickness": sizing["cbar_thickness"],
        "len": sizing["cbar_len"],
        "ticklen": sizing["ticklen"],
        "tickwidth": sizing["tickwidth"],
        "tickcolor": style["linecolor"],
        "tickfont": {
            "size": f_ticks,
            "family": style["font_type"],
            "color": style["linecolor"],
        },
        "tickformat": ".5f",
        "title": {
            "font": {
                "family": style["font_type"],
                "size": f_title,
                "color": style["linecolor"],
            },
            "side": "right",
        },
    }


def _slider_config() -> dict:
    """Default slider layout for animated plots."""
    return {
        "active": 0,
        "currentvalue": {
            "font": {"size": 16},
            "visible": True,
            "xanchor": "right",
        },
        "pad": {"b": 10, "t": 0},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }


# -- hex polygon helpers ---------------------------------------------------


def _hex_vertices(
    cx: float, cy: float, radius: float, orientation: str
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 6 hexagon vertices in data coordinates.

    Args:
        cx, cy: Centre of the hexagon.
        radius: Circumradius (distance from centre to vertex).
            Matches ``to_pixel(size=radius)``.
        orientation: ``"flat"`` or ``"pointy"``.

    Returns:
        (vx, vy) arrays of length 6.
    """
    if orientation == "pointy":
        angles = np.arange(6) * np.pi / 3 + np.pi / 6
    else:
        angles = np.arange(6) * np.pi / 3
    return cx + radius * np.cos(angles), cy + radius * np.sin(angles)


def _hex_shape(
    cx: float,
    cy: float,
    radius: float,
    fillcolor: str,
    line_color: str,
    line_width: float,
    orientation: str,
) -> dict:
    """Build a Plotly layout-shape dict for a single hexagon."""
    vx, vy = _hex_vertices(cx, cy, radius, orientation)
    path = "M " + " L ".join(f"{x:.6f},{y:.6f}" for x, y in zip(vx, vy)) + " Z"
    return dict(
        type="path",
        path=path,
        fillcolor=fillcolor,
        line=dict(color=line_color, width=line_width),
    )


def _values_to_colors(
    values: np.ndarray, colorscale: list | str, vmin: float, vmax: float
) -> list[str]:
    """Map values to colour strings via colorscale interpolation."""
    if vmax == vmin:
        t = np.full(len(values), 0.5)
    else:
        t = np.clip((np.asarray(values, dtype=float) - vmin) / (vmax - vmin), 0, 1)
    return sample_colorscale(colorscale, t.tolist())


def _hide_axes(
    fig: go.Figure,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> None:
    """Remove axes, grid, and ticks; enforce equal aspect ratio."""
    no_show = dict(showgrid=False, showticklabels=False, showline=False, visible=False)
    fig.update_xaxes(**no_show)
    fig.update_yaxes(**no_show, scaleanchor="x", scaleratio=1)
    if x_range is not None:
        fig.update_xaxes(range=[x_range[0], x_range[1]])
    if y_range is not None:
        fig.update_yaxes(range=[y_range[0], y_range[1]])


# -- public API ------------------------------------------------------------


def heatmap(
    df: pd.DataFrame,
    dataset: pd.DataFrame,
    style: dict | None = None,
    sizing: dict | None = None,
    dpi: int = 72,
    custom_colorscale: list | None = None,
    title: str | None = None,
    colorbar: bool = True,
    value_name: str = "value",
    coord_format: Literal["axial", "zigzag", "flywire", "pixel"] = "axial",
    orientation: Literal["pointy", "flat"] = "flat",
    layout: str | None = None,
    rotation_deg: float = 0.0,
    include_flywire_hover: bool = True,
) -> go.Figure:
    """Generate an interactive hexagonal heatmap.

    Hex polygons are drawn as SVG path shapes in data coordinates — no
    pixel math, no marker-size estimation.  A single invisible scatter
    trace at hex centres provides hover, colour bar, and optional text
    labels.  Plotly handles all scaling (zoom, resize, export).

    Args:
        df: DataFrame with ``'p'`` and ``'q'`` columns plus one or more
            value columns.  Single value column produces a static heatmap;
            multiple columns produce an animated heatmap with a slider.
        dataset: Reference hex grid (``'p'``, ``'q'`` columns).  Background
            hexes are drawn white for spatial context.
        style: Styling overrides — ``font_type``, ``linecolor``,
            ``papercolor``.
        sizing: Size overrides — ``fig_width``, ``fig_height``,
            ``hex_line_color``, ``hex_line_width``, ``cbar_thickness``,
            ``cbar_len``, etc.
        dpi: Dots per inch for pixel dimension calculations.
        custom_colorscale: Plotly colorscale (list of [pos, colour] pairs
            or a named string like ``"Viridis"``).
        title: Optional plot title.
        colorbar: Whether to show colour bar (default ``True``).
        value_name: Label for values in hover tooltip.
        coord_format: How to interpret ``'p'``/``'q'`` columns —
            ``"axial"``, ``"zigzag"``, ``"flywire"``, or ``"pixel"``.
        orientation: ``"flat"`` or ``"pointy"`` hex orientation.
        layout: Set to ``"flywire"`` to use the flywire pixel projection.
        rotation_deg: Rotation for flywire layout.
        include_flywire_hover: Include axial and zigzag coordinates in
            hover tooltip (default ``True``).

    Returns:
        Plotly Figure with hexagonal heatmap.

    Example:
        >>> fig = heatmap(data_df, background_df)
        >>> fig.show()
    """
    sty = _merge_style(style)
    sz = _merge_sizing(sizing)
    area_w, area_h, f_ticks, f_title = _compute_pixel_dims(sz, dpi)
    colorscale = custom_colorscale or _DEFAULT_COLORSCALE

    global_min = min(0, float(df.values.min()))
    global_max = float(df.values.max())

    # Background grid
    bg = dataset.drop_duplicates(subset=["p", "q"])[["p", "q"]].astype(float)
    _, _, _, _, bg_x, bg_y = _resolve_coords(
        bg.p.to_numpy(),
        bg.q.to_numpy(),
        coord_format,
        orientation,
        layout=layout,
        rotation_deg=rotation_deg,
    )

    # Data hexes
    df_c = df.copy()
    dq, dr, dzx, dzy, dx, dy = _resolve_coords(
        df_c.p.to_numpy(),
        df_c.q.to_numpy(),
        coord_format,
        orientation,
        layout=layout,
        rotation_deg=rotation_deg,
    )

    value_cols = [c for c in df_c.columns if c not in ("p", "q")]
    hex_pad = 1.2  # hex circumradius (1.0) + margin

    # -- builders ----------------------------------------------------------

    def _build_shapes(series: pd.Series) -> list[dict]:
        """Layout shapes for background + data hexes."""
        data_colors = _values_to_colors(
            series.values, colorscale, global_min, global_max
        )
        shapes = [
            _hex_shape(
                cx,
                cy,
                1.0,
                "white",
                sz["hex_line_color"],
                sz["hex_line_width"],
                orientation,
            )
            for cx, cy in zip(bg_x, bg_y)
        ]
        shapes.extend(
            _hex_shape(
                cx,
                cy,
                1.0,
                col,
                sz["hex_line_color"],
                sz["hex_line_width"],
                orientation,
            )
            for cx, cy, col in zip(dx, dy, data_colors)
        )
        return shapes

    def _build_scatter(series: pd.Series, show_cbar: bool) -> go.Scatter:
        """Invisible scatter at hex centres for hover + colorbar."""
        vals = series.values
        if include_flywire_hover:
            customdata = np.stack([dq, dr, dzx, dzy, vals], axis=-1)
            hovertemplate = (
                "p,q = %{customdata[0]:.0f},%{customdata[1]:.0f}<br>"
                "x,y = %{customdata[2]:.0f},%{customdata[3]:.0f}<br>"
                + value_name
                + ": %{customdata[4]:.4f}<extra></extra>"
            )
        else:
            customdata = np.stack([dx, dy, vals], axis=-1)
            hovertemplate = (
                "x: %{customdata[0]:.2f}<br>y: %{customdata[1]:.2f}<br>"
                + value_name
                + ": %{customdata[2]:.4f}<extra></extra>"
            )
        marker: dict = dict(
            color=vals,
            colorscale=colorscale,
            cmin=global_min,
            cmax=global_max,
            size=0,
        )
        if show_cbar:
            marker["showscale"] = True
            marker["colorbar"] = _colorbar_dict(sty, sz, f_ticks, f_title)
        return go.Scatter(
            x=dx,
            y=dy,
            mode="markers",
            marker=marker,
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=False,
        )

    x_range = (float(bg_x.min()) - hex_pad, float(bg_x.max()) + hex_pad)
    y_range = (float(bg_y.min()) - hex_pad, float(bg_y.max()) + hex_pad)

    # -- static or animated ------------------------------------------------

    if len(value_cols) <= 1:
        series = df_c[value_cols[0]] if value_cols else df_c.iloc[:, 0]
        fig = go.Figure(
            data=[_build_scatter(series, colorbar)],
            layout=go.Layout(shapes=_build_shapes(series)),
        )
        fig.update_layout(
            autosize=False,
            height=area_h,
            width=area_w,
            margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
            paper_bgcolor=sty["papercolor"],
            plot_bgcolor=sty["papercolor"],
        )
        _hide_axes(fig, x_range=x_range, y_range=y_range)
    else:
        slider_h = 100
        fig = go.Figure()
        fig.update_layout(
            autosize=False,
            height=area_h + slider_h,
            width=area_w,
            margin={"l": 0, "r": 0, "b": slider_h, "t": 0, "pad": 0},
            paper_bgcolor=sty["papercolor"],
            plot_bgcolor=sty["papercolor"],
            sliders=[_slider_config()],
        )
        _hide_axes(fig, x_range=x_range, y_range=y_range)

        frames: list[go.Frame] = []
        slider_steps: list[dict] = []
        for i, col_name in enumerate(value_cols):
            series = df_c[col_name]
            frame = go.Frame(
                data=[_build_scatter(series, colorbar)],
                layout=go.Layout(shapes=_build_shapes(series)),
                name=str(i),
            )
            frames.append(frame)
            slider_steps.append(
                {
                    "args": [
                        [str(i)],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                        },
                    ],
                    "label": col_name,
                    "method": "animate",
                }
            )
            if i == 0:
                fig.add_trace(_build_scatter(series, colorbar))
                fig.layout.shapes = _build_shapes(series)

        fig.layout.sliders[0].steps = slider_steps  # type: ignore
        fig.frames = frames

    if title:
        fig.update_layout(title=dict(text=title))

    return fig


# -- overlay utilities (Plotly equivalents of static.py) ------------------


def draw_axes(
    fig: go.Figure,
    origin: tuple[float, float] = (0.0, 0.0),
    size: float = 1.0,
    orientation: str = "pointy",
    alignment: str = "vertex",
    colors: tuple[str, str, str] = ("red", "green", "blue"),
    labels: tuple[str, str, str] = ("q", "r", "s"),
) -> None:
    """Draw 3-axis reference arrows on a Plotly hex figure.

    Adds arrow annotations for the q, r, s hex axes.  Uses
    :func:`~btorch.utils.hex.transform.to_pixel` for direction
    computation so arrows always match the hex layout.

    Args:
        fig: Plotly figure to annotate.
        origin: ``(x, y)`` data-space position for the axis origin.
        size: Arrow length in data units.
        orientation: ``"pointy"`` or ``"flat"``.
        alignment: ``"vertex"`` (axes through hex vertices) or
            ``"edge"`` (axes parallel to hex edges).
        colors: Colours for q, r, s arrows.
        labels: Text labels for q, r, s arrows.
    """
    from ...utils.hex.transform import to_pixel

    q_dir = np.array(
        to_pixel(np.array([1.0]), np.array([0.0]), size=size, orientation=orientation)
    ).squeeze()
    r_dir = np.array(
        to_pixel(np.array([0.0]), np.array([1.0]), size=size, orientation=orientation)
    ).squeeze()
    s_dir = np.array(
        to_pixel(np.array([-1.0]), np.array([1.0]), size=size, orientation=orientation)
    ).squeeze()

    dirs = [q_dir, r_dir, s_dir]
    if alignment == "edge":
        angle = np.pi / 6
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        dirs = [rot @ d for d in dirs]

    ox, oy = origin
    for (dx, dy), color, label in zip(dirs, colors, labels):
        fig.add_annotation(
            x=ox + dx,
            y=oy + dy,
            ax=ox,
            ay=oy,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=color,
        )
        fig.add_annotation(
            x=ox + dx * 1.15,
            y=oy + dy * 1.15,
            xref="x",
            yref="y",
            text=label,
            showarrow=False,
            font=dict(color=color, size=12, family="bold"),
        )


def compass(
    fig: go.Figure,
    loc: str = "lower left",
    size: float = 1.0,
    orientation: str = "pointy",
    alignment: str = "vertex",
    colors: tuple[str, str, str] = ("#66c2a5", "#fc8d62", "#8da0cb"),
    label_color: str = "#333333",
    bg_color: str = "white",
    border_color: str = "#cccccc",
) -> None:
    """Draw a hex-direction compass rose on a Plotly figure.

    Creates a small secondary axis pair (``xaxis2``/``yaxis2``) with
    ``scaleanchor`` so the compass is always circular regardless of
    figure dimensions.  Draws six direction arrows (+q, -q, +r, -r,
    +s, -s) radiating from a central circle, matching the style of
    :func:`~btorch.visualisation.hex.static.compass`.

    Args:
        fig: Plotly figure to annotate.
        loc: Corner — ``"lower left"``, ``"lower right"``,
            ``"upper left"``, or ``"upper right"``.
        size: Compass radius in secondary-axis data units.
        orientation: ``"pointy"`` or ``"flat"``.
        alignment: ``"vertex"`` or ``"edge"``.
        colors: Colours for q, r, s axis pairs.
        label_color: Text colour.
        bg_color: Background fill colour.
        border_color: Border colour.
    """
    # Secondary axis domain in the chosen corner
    span = 0.18
    domain_map = {
        "lower left": ([0, span], [0, span]),
        "lower right": ([1 - span, 1], [0, span]),
        "upper left": ([0, span], [1 - span, 1]),
        "upper right": ([1 - span, 1], [1 - span, 1]),
    }
    xdom, ydom = domain_map.get(loc, domain_map["lower left"])

    fig.update_layout(
        xaxis2=dict(
            domain=xdom,
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
            fixedrange=True,
            range=[-1.5, 1.5],
        ),
        yaxis2=dict(
            domain=ydom,
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
            fixedrange=True,
            range=[-1.5, 1.5],
            scaleanchor="x2",
            scaleratio=1,
        ),
    )

    # Base directions (pointy-top)
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

    arrow_len = size * 0.85
    label_offset = size * 1.15

    # Background circle on secondary axes
    fig.add_shape(
        type="circle",
        xref="x2",
        yref="y2",
        x0=-size,
        y0=-size,
        x1=size,
        y1=size,
        fillcolor=bg_color,
        line=dict(color=border_color, width=1),
        opacity=0.9,
    )

    for vec, label, color in directions:
        dx, dy = vec * arrow_len
        lx, ly = vec * label_offset
        # Arrow shaft
        fig.add_shape(
            type="line",
            xref="x2",
            yref="y2",
            x0=0,
            y0=0,
            x1=dx,
            y1=dy,
            line=dict(color=color, width=1.5),
        )
        # Arrowhead triangle
        hw = arrow_len * 0.25
        perp = np.array([-vec[1], vec[0]]) * hw
        bx, by = dx, dy
        lx1, ly1 = bx - vec[0] * hw + perp[0], by - vec[1] * hw + perp[1]
        lx2, ly2 = bx - vec[0] * hw - perp[0], by - vec[1] * hw - perp[1]
        fig.add_shape(
            type="path",
            xref="x2",
            yref="y2",
            path=f"M {bx:.6f},{by:.6f} L {lx1:.6f},{ly1:.6f} L {lx2:.6f},{ly2:.6f} Z",
            fillcolor=color,
            line=dict(color=color, width=0.5),
        )
        fig.add_annotation(
            xref="x2",
            yref="y2",
            x=lx,
            y=ly,
            text=label,
            showarrow=False,
            font=dict(color=label_color, size=8, family="bold"),
        )

    # Central dot
    dot_r = size * 0.12
    fig.add_shape(
        type="circle",
        xref="x2",
        yref="y2",
        x0=-dot_r,
        y0=-dot_r,
        x1=dot_r,
        y1=dot_r,
        fillcolor="white",
        line=dict(color="#888888", width=1),
    )


def grid(
    radius: int,
    coord_format: str = "axial",
    orientation: str = "pointy",
    annotate: bool = False,
    style: dict | None = None,
    sizing: dict | None = None,
    dpi: int = 72,
    title: str | None = None,
    show_axes: bool = False,
    show_compass: bool | str = False,
) -> go.Figure:
    """Interactive coordinate grid with optional labels.

    Args:
        radius: Grid radius (number of hex rings).
        coord_format: ``"axial"`` or ``"zigzag"``.
        orientation: ``"pointy"`` or ``"flat"``.
        annotate: Show ``(q, r)`` labels at hex centres.
        style: Style overrides.
        sizing: Size overrides.
        dpi: Dots per inch.
        title: Optional plot title.
        show_axes: Draw q/r/s reference axes.
        show_compass: ``"vertex"``, ``"edge"``, or ``False``.

    Returns:
        Plotly Figure with hex grid.
    """
    from ...utils.hex.coords import disk
    from ...utils.hex.offset import axial_to_zigzag

    q, r = disk(radius)
    if coord_format == "axial":
        labels = [f"({qi},{ri})" for qi, ri in zip(q, r)]
        _, _, _, _, x, y = _resolve_coords(q, r, "axial", orientation)
    elif coord_format == "zigzag":
        zx, zy = axial_to_zigzag(q, r)
        labels = [f"({zi},{zyi})" for zi, zyi in zip(zx, zy)]
        _, _, _, _, x, y = _resolve_coords(zx, zy, "zigzag", orientation)
    else:
        raise ValueError(f"Unknown coord_format: {coord_format}")

    sty = _merge_style(style)
    sz = _merge_sizing(sizing)
    area_w, area_h, *_ = _compute_pixel_dims(sz, dpi)

    # White hex shapes
    shapes = [
        _hex_shape(
            xi,
            yi,
            1.0,
            "white",
            "gray",
            0.5,
            orientation,
        )
        for xi, yi in zip(x, y)
    ]

    # Invisible scatter for hover
    customdata = np.stack([x, y], axis=-1)
    scatter = go.Scatter(
        x=x,
        y=y,
        mode="markers+text" if annotate else "markers",
        marker=dict(size=0),
        text=labels if annotate else None,
        textposition="middle center",
        textfont=dict(size=8),
        customdata=customdata,
        hovertemplate=(
            "x: %{customdata[0]:.2f}<br>y: %{customdata[1]:.2f}" "<extra></extra>"
        ),
        showlegend=False,
    )

    hex_pad = 1.2
    x_range = (float(x.min()) - hex_pad, float(x.max()) + hex_pad)
    y_range = (float(y.min()) - hex_pad, float(y.max()) + hex_pad)

    fig = go.Figure(
        data=[scatter],
        layout=go.Layout(shapes=shapes),
    )
    fig.update_layout(
        autosize=False,
        height=area_h,
        width=area_w,
        margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
        paper_bgcolor=sty["papercolor"],
        plot_bgcolor=sty["papercolor"],
    )
    _hide_axes(fig, x_range=x_range, y_range=y_range)

    if show_axes:
        draw_axes(
            fig,
            origin=(float(x.min()) + 1, float(y.min()) + 1),
            size=2.0,
            orientation=orientation,
        )
    if show_compass:
        align = show_compass if isinstance(show_compass, str) else "vertex"
        compass(fig, alignment=align)
    if title:
        fig.update_layout(title=dict(text=title))

    return fig


def quiver(
    c1: np.ndarray,
    c2: np.ndarray,
    dc1: np.ndarray,
    dc2: np.ndarray,
    coord_format: str = "axial",
    orientation: str = "pointy",
    scale: float = 1.0,
    custom_colorscale: list | None = None,
    style: dict | None = None,
    sizing: dict | None = None,
    dpi: int = 72,
    title: str | None = None,
    show_axes: bool = False,
    show_compass: bool | str = False,
) -> go.Figure:
    """Interactive quiver (vector field) on hex grid.

    Args:
        c1, c2: Hex coordinates.
        dc1, dc2: Vector components in hex coordinates.
        coord_format: ``"axial"``, ``"zigzag"``, or ``"pixel"``.
        orientation: ``"pointy"`` or ``"flat"``.
        scale: Vector scale factor.
        custom_colorscale: Plotly colourscale for magnitude.
        style: Style overrides.
        sizing: Size overrides.
        dpi: Dots per inch.
        title: Optional plot title.
        show_axes: Draw q/r/s reference axes.
        show_compass: ``"vertex"``, ``"edge"``, or ``False``.

    Returns:
        Plotly Figure with quiver plot.
    """

    _, _, _, _, x, y = _resolve_coords(c1, c2, coord_format, orientation)

    if coord_format == "pixel":
        tx, ty = c1 + dc1 * scale, c2 + dc2 * scale
    else:
        _, _, _, _, tx, ty = _resolve_coords(
            c1 + dc1 * scale,
            c2 + dc2 * scale,
            coord_format,
            orientation,
        )
    dx = tx - x
    dy = ty - y

    magnitudes = np.sqrt(dx**2 + dy**2)
    colorscale = custom_colorscale or _DEFAULT_COLORSCALE
    sty = _merge_style(style)
    sz = _merge_sizing(sizing)
    area_w, area_h, f_ticks, f_title = _compute_pixel_dims(sz, dpi)

    # Scatter at hex centres (invisible, for hover)
    scatter = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            color=magnitudes,
            colorscale=colorscale,
            size=0,
            showscale=True,
            colorbar=_colorbar_dict(sty, sz, f_ticks, f_title),
        ),
        customdata=np.stack([x, y, magnitudes], axis=-1),
        hovertemplate="x: %{customdata[0]:.2f}<br>y: %{customdata[1]:.2f}"
        "<br>|v|: %{customdata[2]:.4f}<extra></extra>",
        showlegend=False,
    )

    # Arrow annotations for each vector
    fig = go.Figure(data=[scatter])
    for xi, yi, dxi, dyi in zip(x, y, dx, dy):
        if abs(dxi) < 1e-9 and abs(dyi) < 1e-9:
            continue
        fig.add_annotation(
            x=xi + dxi,
            y=yi + dyi,
            ax=xi,
            ay=yi,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="gray",
        )

    hex_pad = 2.0
    x_range = (float(x.min()) - hex_pad, float(x.max()) + hex_pad)
    y_range = (float(y.min()) - hex_pad, float(y.max()) + hex_pad)

    fig.update_layout(
        autosize=False,
        height=area_h,
        width=area_w,
        margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
        paper_bgcolor=sty["papercolor"],
        plot_bgcolor=sty["papercolor"],
    )
    _hide_axes(fig, x_range=x_range, y_range=y_range)

    if show_axes:
        draw_axes(
            fig,
            origin=(float(x.min()) + 1, float(y.min()) + 1),
            size=2.0,
            orientation=orientation,
        )
    if show_compass:
        align = show_compass if isinstance(show_compass, str) else "vertex"
        compass(fig, alignment=align)
    if title:
        fig.update_layout(title=dict(text=title))

    return fig
