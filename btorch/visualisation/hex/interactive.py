"""Interactive hex heatmaps using Plotly.

Moved from visualisation/hexmap.py.
Supports multiple coordinate formats: axial (q,r), zigzag (x,y), pixel (px,py),
and connectome-style string indices ("x,y" double-width coordinates).
"""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...utils.hex.doubled import doublewidth_to_pixel
from ...utils.hex.offset import (
    axial_to_zigzag,
    flywire_to_pixel,
    zigzag_to_axial,
    zigzag_to_pixel,
)
from ...utils.hex.transform import to_pixel as hex_to_pixel


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
    rotation_deg: float = 0.0,
    include_flywire_hover: bool = True,
) -> go.Figure:
    """Generate an interactive hexagonal heatmap.

    Visualizes data on a hexagonal grid layout. Single-column data produces
    a static heatmap; multi-column DataFrames produce an animated heatmap
    with a slider to navigate through timepoints or conditions.

    Args:
        df: DataFrame to visualize. Must have 'p' and 'q' columns representing
            hex grid coordinates. Single-column DataFrames produce static plots,
            multi-column DataFrames produce animated plots (one frame per column).
        dataset: Reference dataset defining the full hex grid background.
            Used to render empty hexagons for spatial context.
        style: Styling options dict with keys:
            - "font_type": Font family (default: "arial")
            - "markerlinecolor": Marker line color
            - "linecolor": Axis/line color (default: "black")
            - "papercolor": Background color (default: "rgba(255,255,255,255)")
        sizing: Size configuration dict with keys:
            - "fig_width", "fig_height": Figure dimensions in mm
            - "markersize": Hexagon marker size (default: 16)
            - "cbar_thickness", "cbar_len": Colorbar dimensions
        dpi: Dots per inch for pixel calculations (default: 72).
        custom_colorscale: Custom Plotly colorscale. Default is white-to-blue.
        title: Optional plot title.
        colorbar: Whether to show colorbar (default: True).
                coord_format: Coordinate interpretation for `p`,`q` columns:
                        - "axial": `p`,`q` are axial q,r
                        - "zigzag": `p`,`q` are zigzag x,y
                        - "flywire": `p`,`q` are FlyWire axial coordinates rendered with
                            the saved-page DOM-equivalent layout
                        - "pixel": `p`,`q` are pixel x,y
        orientation: Pixel projection orientation for hex coordinates.
        rotation_deg: Optional global display rotation in degrees for
            `coord_format="flywire"`.
        value_name: Name for value in hover tooltip (default: "value").
        include_flywire_hover: Whether to include axial and zigzag coordinates
            in hover information when available.

    Returns:
        Plotly Figure with hexagonal heatmap. Static for single-column input,
        animated with slider for multi-column DataFrames.

    Example:
        >>> # Static heatmap
        >>> fig = heatmap(data_series, background_dataset)
        >>> fig.show()
        >>>
        >>> # Animated heatmap with timepoints
        >>> fig = heatmap(timepoint_df, background_dataset)
        >>> fig.write_html("animated_hexmap.html")
    """

    def coords_to_pixel(
        c1: np.ndarray,
        c2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if coord_format == "axial":
            q_arr = np.asarray(c1, dtype=float)
            r_arr = np.asarray(c2, dtype=float)
            x_arr, y_arr = hex_to_pixel(q_arr, r_arr, orientation=orientation)
            zx_arr, zy_arr = axial_to_zigzag(
                np.rint(q_arr).astype(int), np.rint(r_arr).astype(int)
            )
            return (
                q_arr,
                r_arr,
                zx_arr.astype(float),
                zy_arr.astype(float),
                x_arr,
                y_arr,
            )

        if coord_format == "zigzag":
            zx_arr = np.asarray(c1, dtype=float)
            zy_arr = np.asarray(c2, dtype=float)
            q_arr, r_arr = zigzag_to_axial(
                np.rint(zx_arr).astype(int), np.rint(zy_arr).astype(int)
            )
            x_arr, y_arr = zigzag_to_pixel(zx_arr, zy_arr)
            return (
                q_arr.astype(float),
                r_arr.astype(float),
                zx_arr,
                zy_arr,
                x_arr,
                y_arr,
            )

        if coord_format == "flywire":
            q_arr = np.asarray(c1, dtype=float)
            r_arr = np.asarray(c2, dtype=float)
            zx_arr, zy_arr = axial_to_zigzag(
                np.rint(q_arr).astype(int), np.rint(r_arr).astype(int)
            )
            x_arr, y_arr = flywire_to_pixel(
                q_arr,
                r_arr,
                rotation_deg=rotation_deg,
            )
            return (
                q_arr.astype(float),
                r_arr.astype(float),
                zx_arr,
                zy_arr,
                x_arr,
                y_arr,
            )

        if coord_format == "pixel":
            x_arr = np.asarray(c1, dtype=float)
            y_arr = np.asarray(c2, dtype=float)
            nan_arr = np.full_like(x_arr, np.nan, dtype=float)
            return nan_arr, nan_arr, nan_arr, nan_arr, x_arr, y_arr

        raise ValueError(f"Unknown coord_format: {coord_format}")

    def bg_hex():
        return go.Scatter(
            x=background_hex["x"],
            y=background_hex["y"],
            mode="markers",
            marker_symbol=symbol_number,
            marker={
                "size": sizing["markersize"],
                "color": "white",
                "line": {
                    "width": sizing["markerlinewidth"],
                    "color": "lightgrey",
                },
            },
            showlegend=False,
        )

    def data_hex(aseries):
        marker_config = {
            "cmin": global_min,
            "cmax": global_max,
            "size": sizing["markersize"],
            "color": aseries.values,
            "line": {
                "width": sizing["markerlinewidth"],
                "color": "lightgrey",
            },
            "colorscale": custom_colorscale,
        }
        if colorbar:
            marker_config["colorbar"] = {
                "orientation": "v",
                "outlinecolor": style["linecolor"],
                "outlinewidth": sizing["axislinewidth"],
                "thickness": sizing["cbar_thickness"],
                "len": sizing["cbar_len"],
                "tickmode": "array",
                "ticklen": sizing["ticklen"],
                "tickwidth": sizing["tickwidth"],
                "tickcolor": style["linecolor"],
                "tickfont": {
                    "size": fsize_ticks_px,
                    "family": style["font_type"],
                    "color": style["linecolor"],
                },
                "tickformat": ".5f",
                "title": {
                    "font": {
                        "family": style["font_type"],
                        "size": fsize_title_px,
                        "color": style["linecolor"],
                    },
                    "side": "right",
                },
            }
        if include_flywire_hover:
            customdata = np.stack(
                [q_vals, r_vals, zigzag_x_vals, zigzag_y_vals, aseries.values],
                axis=-1,
            )
            hovertemplate = (
                "p,q = %{customdata[0]:.0f},%{customdata[1]:.0f}<br>"
                "x,y = %{customdata[2]:.0f},%{customdata[3]:.0f}<br>"
                + value_name
                + ": %{customdata[4]:.4f}<extra></extra>"
            )
        else:
            customdata = np.stack([x_vals, y_vals, aseries.values], axis=-1)
            hovertemplate = (
                "x: %{customdata[0]:.2f}<br>y: %{customdata[1]:.2f}<br>"
                + value_name
                + ": %{customdata[2]:.4f}<extra></extra>"
            )

        goscatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker_symbol=symbol_number,
            marker=marker_config,
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=False,
        )
        return goscatter

    default_style = {
        "font_type": "arial",
        "markerlinecolor": "rgba(0,0,0,0)",
        "linecolor": "black",
        "papercolor": "rgba(255,255,255,255)",
    }

    default_sizing = {
        "fig_width": 260,
        "fig_height": 220,
        "fig_margin": 0,
        "fsize_ticks_pt": 20,
        "fsize_title_pt": 20,
        "markersize": None,
        "markersize_scale": 1.0,
        "ticklen": 15,
        "tickwidth": 5,
        "axislinewidth": 3,
        "markerlinewidth": 0.9,
        "cbar_thickness": 20,
        "cbar_len": 0.75,
    }

    if style is not None:
        default_style.update(style)
    style = default_style

    if sizing is not None:
        default_sizing.update(sizing)
    sizing = default_sizing

    POINTS_PER_INCH = 72
    MM_PER_INCH = 25.4

    pixelsperinch = dpi
    pixelspermm = pixelsperinch / MM_PER_INCH

    if custom_colorscale is None:
        custom_colorscale = [[0, "rgb(255, 255, 255)"], [1, "rgb(0, 20, 200)"]]

    area_width = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    area_height = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm

    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1 / POINTS_PER_INCH) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1 / POINTS_PER_INCH) * pixelsperinch

    global_min = min(0, df.values.min())
    global_max = df.values.max()

    symbol_number = 15

    background_hex = dataset.copy()
    background_hex = background_hex.drop_duplicates(subset=["p", "q"])[
        ["p", "q"]
    ].astype(float)
    (
        bg_q,
        bg_r,
        bg_zigzag_x,
        bg_zigzag_y,
        x,
        y,
    ) = coords_to_pixel(
        background_hex.p.to_numpy(),
        background_hex.q.to_numpy(),
    )
    background_hex["q_axial"] = bg_q
    background_hex["r_axial"] = bg_r
    background_hex["x_zigzag"] = bg_zigzag_x
    background_hex["y_zigzag"] = bg_zigzag_y
    background_hex["x"], background_hex["y"] = x, y

    if sizing["markersize"] is None:
        points = np.stack(
            [background_hex["x"].to_numpy(), background_hex["y"].to_numpy()],
            axis=1,
        )
        if len(points) > 1:
            deltas = points[:, None, :] - points[None, :, :]
            dists = np.sqrt(np.sum(deltas * deltas, axis=-1))
            np.fill_diagonal(dists, np.inf)
            positive = dists[np.isfinite(dists) & (dists > 1e-9)]
            min_dist = float(np.min(positive)) if positive.size > 0 else 1.0
        else:
            min_dist = 1.0

        x_span = float(np.ptp(background_hex["x"].to_numpy()))
        y_span = float(np.ptp(background_hex["y"].to_numpy()))
        px_per_unit_x = area_width / max(x_span, 1e-9)
        px_per_unit_y = area_height / max(y_span, 1e-9)
        px_per_unit = min(px_per_unit_x, px_per_unit_y)
        auto_markersize = 0.92 * min_dist * px_per_unit
        sizing["markersize"] = max(2.0, auto_markersize) * float(
            sizing["markersize_scale"]
        )

    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        height=area_height,
        width=area_width,
        margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
        paper_bgcolor=style["papercolor"],
        plot_bgcolor=style["papercolor"],
    )
    fig.update_xaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        showline=False,
        visible=False,
        scaleanchor="x",
        scaleratio=1,
    )

    df = df.copy()
    (
        df_q,
        df_r,
        df_zigzag_x,
        df_zigzag_y,
        df_x,
        df_y,
    ) = coords_to_pixel(df.p.to_numpy(), df.q.to_numpy())
    df["q_axial"] = df_q
    df["r_axial"] = df_r
    df["x_zigzag"] = df_zigzag_x
    df["y_zigzag"] = df_zigzag_y
    df["x"], df["y"] = df_x, df_y
    x_vals, y_vals = df.x, df.y
    q_vals = df["q_axial"].to_numpy(dtype=float)
    r_vals = df["r_axial"].to_numpy(dtype=float)
    zigzag_x_vals = df["x_zigzag"].to_numpy(dtype=float)
    zigzag_y_vals = df["y_zigzag"].to_numpy(dtype=float)
    df = df.drop(
        columns=[
            "p",
            "q",
            "x",
            "y",
            "q_axial",
            "r_axial",
            "x_zigzag",
            "y_zigzag",
        ]
    )

    if len(df.columns) == 1:
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        fig.add_trace(bg_hex())
        fig.add_trace(data_hex(df))

    elif isinstance(df, pd.DataFrame):
        slider_height = 100
        area_height += slider_height

        frames = []
        slider_steps = []

        fig.update_layout(
            autosize=False,
            height=area_height,
            width=area_width,
            margin={
                "l": 0,
                "r": 0,
                "b": slider_height,
                "t": 0,
                "pad": 0,
            },
            paper_bgcolor=style["papercolor"],
            plot_bgcolor=style["papercolor"],
            sliders=[
                {
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
            ],
        )

        for i, col_name in enumerate(df.columns):
            series = df[col_name]
            frame_data = [
                bg_hex(),
                data_hex(series),
            ]

            frames.append(go.Frame(data=frame_data, name=str(i)))

            slider_steps.append(
                {
                    "args": [
                        [str(i)],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                    ],
                    "label": col_name,
                    "method": "animate",
                }
            )

            if i == 0:
                fig.add_traces(frame_data)

        fig.layout.sliders[0].steps = slider_steps  # type: ignore
        fig.frames = frames

        fig.update_xaxes(
            showgrid=False, showticklabels=False, showline=False, visible=False
        )
        fig.update_yaxes(
            showgrid=False, showticklabels=False, showline=False, visible=False
        )

    else:
        raise ValueError("df must be a pd.Series or pd.DataFrame")

    return fig


def heatmap_from_index(
    df: pd.Series | pd.DataFrame,
    style: dict | None = None,
    sizing: dict | None = None,
    dpi: int = 72,
    custom_colorscale: list | None = None,
    global_min: float | None = None,
    global_max: float | None = None,
    colorbar: bool = True,
    title: str | None = None,
    value_name: str = "value",
) -> go.Figure:
    """Generate interactive hexagonal heatmap from string "x,y" indices.

    Connectome-style visualization where coordinates are stored as string
    indices like "-12,34" (double-width coordinates). Supports both static
    (single column/Series) and animated (multi-column DataFrame) visualizations.

    Args:
        df: Data to visualize. Series or single-column DataFrame for static
            plots; multi-column DataFrame produces animated plots with slider.
        style: Styling options dict with keys:
            - "font_type": Font family (default: "arial")
            - "linecolor": Axis/line color (default: "black")
            - "papercolor": Background color (default: "white")
        sizing: Size configuration dict with keys:
            - "fig_width", "fig_height": Figure dimensions in mm
            - "markersize": Hexagon marker size
            - "cbar_thickness", "cbar_len": Colorbar dimensions
        dpi: Dots per inch for pixel calculations.
        custom_colorscale: Custom Plotly colorscale. Default is white-to-blue.
        global_min: Global minimum for color scale. Uses 0 or data min if None.
        global_max: Global maximum for color scale. Uses data max if None.
        colorbar: Whether to show colorbar.
        title: Optional plot title.
        value_name: Name displayed in hover tooltip for values.

    Returns:
        Plotly Figure with hexagonal heatmap.

    Example:
        >>> # Create fake connectome data
        >>> coords = [f"{x},{y}" for x, y in [(0, 0), (1, 0), (0, 2), (-1, 0)]]
        >>> values = pd.Series([0.5, 0.3, 0.8, 0.2], index=coords)
        >>> fig = heatmap_from_index(values)
        >>> fig.show()
    """
    # Parse "x,y" indices to get all coordinates
    all_coords = [tuple(map(float, idx.split(","))) for idx in df.index]
    all_x, all_y = zip(*all_coords)

    # Convert double-width to pixel for visualization
    bg_x, bg_y = doublewidth_to_pixel(np.array(all_x), np.array(all_y), size=1.0)

    # Determine data range
    if isinstance(df, pd.DataFrame):
        vals = df.values
    else:
        vals = df.values.reshape(-1, 1)

    if global_min is None:
        global_min = min(0, vals.min())
    if global_max is None:
        global_max = vals.max()

    # Default styling
    default_style = {
        "font_type": "arial",
        "linecolor": "black",
        "papercolor": "rgba(255,255,255,255)",
    }
    if style is not None:
        default_style.update(style)
    style = default_style

    markersize = 18
    default_sizing = {
        "fig_width": 260 if colorbar else 206,
        "fig_height": 220,
        "fig_margin": 0,
        "fsize_ticks_pt": 20,
        "fsize_title_pt": 20,
        "fsize_plot_title_pt": 24,
        "title_margin": 50,
        "markersize": markersize,
        "ticklen": 15,
        "tickwidth": 5,
        "axislinewidth": 3,
        "markerlinewidth": 0.5,
        "cbar_thickness": 20,
        "cbar_len": 0.75,
    }
    if sizing is not None:
        default_sizing.update(sizing)
    sizing = default_sizing

    # Unit conversions
    POINTS_PER_INCH = 72
    MM_PER_INCH = 25.4
    pixelsperinch = dpi
    pixelspermm = pixelsperinch / MM_PER_INCH

    area_width = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    area_height = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm
    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1 / POINTS_PER_INCH) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1 / POINTS_PER_INCH) * pixelsperinch
    fsize_plot_title_px = (
        sizing["fsize_plot_title_pt"] * (1 / POINTS_PER_INCH) * pixelsperinch
    )

    if custom_colorscale is None:
        custom_colorscale = [[0, "rgb(255, 255, 255)"], [1, "rgb(0, 20, 200)"]]

    symbol_number = 15  # hexagon marker

    # Create background hex trace
    def bg_hex():
        return go.Scatter(
            x=bg_x,
            y=bg_y,
            mode="markers",
            marker_symbol=symbol_number,
            marker={
                "size": sizing["markersize"],
                "color": "white",
                "line": {
                    "width": sizing["markerlinewidth"],
                    "color": "lightgrey",
                },
            },
            showlegend=False,
            hoverinfo="skip",
        )

    # Create data hex trace
    def data_hex(series: pd.Series):
        # Get coordinates for this series
        coords = [tuple(map(float, idx.split(","))) for idx in series.index]
        xs, ys = zip(*coords)
        x_vals, y_vals = doublewidth_to_pixel(np.array(xs), np.array(ys), size=1.0)

        marker_config = {
            "cmin": global_min,
            "cmax": global_max,
            "size": sizing["markersize"],
            "color": series.values,
            "line": {
                "width": sizing["markerlinewidth"],
                "color": "lightgrey",
            },
            "colorscale": custom_colorscale,
        }
        if colorbar:
            marker_config["colorbar"] = {
                "orientation": "v",
                "outlinecolor": style["linecolor"],
                "outlinewidth": sizing["axislinewidth"],
                "thickness": sizing["cbar_thickness"],
                "len": sizing["cbar_len"],
                "tickmode": "array",
                "ticklen": sizing["ticklen"],
                "tickwidth": sizing["tickwidth"],
                "tickcolor": style["linecolor"],
                "tickfont": {
                    "size": fsize_ticks_px,
                    "family": style["font_type"],
                    "color": style["linecolor"],
                },
                "tickformat": ".5f",
                "title": {
                    "font": {
                        "family": style["font_type"],
                        "size": fsize_title_px,
                        "color": style["linecolor"],
                    },
                    "side": "right",
                },
            }

        return go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker_symbol=symbol_number,
            marker=marker_config,
            customdata=np.stack([x_vals, y_vals, series.values], axis=-1),
            hovertemplate=(
                f"x: %{{customdata[0]:.2f}}<br>y: %{{customdata[1]:.2f}}<br>"
                f"{value_name}: %{{customdata[2]:.4f}}<extra></extra>"
            ),
            showlegend=False,
        )

    # Handle Series vs DataFrame
    if isinstance(df, pd.Series):
        series = df
    elif isinstance(df, pd.DataFrame) and len(df.columns) == 1:
        series = df.iloc[:, 0]
    else:
        series = None

    # Build figure
    top_margin = sizing["title_margin"] if title else 0
    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        height=area_height + top_margin,
        width=area_width,
        margin={"l": 0, "r": 0, "b": 0, "t": top_margin, "pad": 0},
        paper_bgcolor=style["papercolor"],
        plot_bgcolor=style["papercolor"],
        title=(
            dict(
                text=title,
                x=0.5,
                xanchor="center",
                font=dict(size=fsize_plot_title_px, family=style["font_type"]),
            )
            if title
            else None
        ),
    )
    fig.update_xaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )
    fig.update_yaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )

    if series is not None:
        # Static plot
        fig.add_trace(bg_hex())
        fig.add_trace(data_hex(series))
    else:
        # Animated plot with slider
        slider_height = 100
        area_height += slider_height

        fig.update_layout(
            autosize=False,
            height=area_height + top_margin,
            width=area_width,
            margin={"l": 0, "r": 0, "b": slider_height, "t": top_margin, "pad": 0},
            sliders=[
                {
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
            ],
        )

        frames = []
        slider_steps = []

        for i, col_name in enumerate(df.columns):
            series = df[col_name]
            frame_data = [bg_hex(), data_hex(series)]
            frames.append(go.Frame(data=frame_data, name=str(i)))
            slider_steps.append(
                {
                    "args": [
                        [str(i)],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                    ],
                    "label": col_name,
                    "method": "animate",
                }
            )
            if i == 0:
                fig.add_traces(frame_data)

        fig.layout.sliders[0].steps = slider_steps  # type: ignore
        fig.frames = frames

    return fig


def mollweide_projection(
    data: pd.Series | pd.DataFrame,
    fig_size: tuple = (900, 700),
    custom_colorscale: list | str | None = None,
    global_min: float | None = None,
    global_max: float | None = None,
    marker_size: int = 8,
    value_name: str = "value",
    colorbar: bool = True,
) -> go.Figure:
    """Mollweide projection of hex data onto sphere.

    Projects hexagonal coordinates onto a sphere using Mollweide projection,
    commonly used for mapping the visual field or spherical surfaces.

    Args:
        data: Data with "x,y" string indices representing hex coordinates.
        fig_size: Figure size in pixels (width, height).
        custom_colorscale: Plotly colorscale (list or named colorscale).
        global_min: Global minimum for color scale.
        global_max: Global maximum for color scale.
        marker_size: Size of scatter markers.
        value_name: Name for value in hover tooltip.
        colorbar: Whether to show colorbar.

    Returns:
        Plotly Figure with Mollweide projection.

    Example:
        >>> coords = [f"{x},{y}" for x, y in [(0, 0), (1, 0), (0, 2)]]
        >>> values = pd.Series([0.5, 0.3, 0.8], index=coords)
        >>> fig = mollweide_projection(values)
    """

    def cart2sph(xyz: np.ndarray) -> np.ndarray:
        """Convert Cartesian to spherical coordinates."""
        r = np.sqrt((xyz**2).sum(1))
        theta = np.arccos(xyz[:, 2] / (r + 1e-10))
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi[phi < 0] = phi[phi < 0] + 2 * np.pi
        return np.stack((r, theta, phi), axis=1)

    def sph2mollweide(thetaphi: np.ndarray) -> np.ndarray:
        """Spherical to Mollweide projection."""
        azim = thetaphi[:, 1]
        azim[azim > np.pi] = azim[azim > np.pi] - 2 * np.pi
        elev = np.pi / 2 - thetaphi[:, 0]

        N = len(azim)
        xy = np.zeros((N, 2))
        for i in range(N):
            theta = np.arcsin(np.clip(2 * elev[i] / np.pi, -1, 1))
            if np.abs(np.abs(theta) - np.pi / 2) < 0.001:
                xy[i] = [
                    2 * np.sqrt(2) / np.pi * azim[i] * np.cos(theta),
                    np.sqrt(2) * np.sin(theta),
                ]
            else:
                # Newton-Raphson iteration
                dtheta = 1
                while dtheta > 1e-3:
                    theta_new = theta - (
                        2 * theta + np.sin(2 * theta) - np.pi * np.sin(elev[i])
                    ) / (2 + 2 * np.cos(2 * theta))
                    dtheta = np.abs(theta_new - theta)
                    theta = theta_new
                xy[i] = [
                    2 * np.sqrt(2) / np.pi * azim[i] * np.cos(theta),
                    np.sqrt(2) * np.sin(theta),
                ]
        return xy

    # Generate fake spherical coordinates from hex indices
    coords = [tuple(map(float, idx.split(","))) for idx in data.index]
    hex_x, hex_y = zip(*coords)

    # Convert to spherical (simplified mapping)
    # Scale hex coordinates to spherical angles
    hex_x = np.array(hex_x)
    hex_y = np.array(hex_y)

    # Normalize to unit sphere
    theta = np.pi * (hex_y - hex_y.min()) / (hex_y.max() - hex_y.min() + 1e-10)
    phi = 2 * np.pi * (hex_x - hex_x.min()) / (hex_x.max() - hex_x.min() + 1e-10)

    # Convert to Cartesian then back to spherical for proper Mollweide
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    xyz = np.stack([x, y, z], axis=1)

    rtp = cart2sph(xyz)
    xy = sph2mollweide(rtp[:, 1:3])

    # Get values
    if isinstance(data, pd.DataFrame):
        vals = data.values[:, 0] if len(data.columns) > 0 else data.values.flatten()
    else:
        vals = data.values

    if global_min is None:
        global_min = min(0, vals.min())
    if global_max is None:
        global_max = vals.max()

    if custom_colorscale is None:
        custom_colorscale = [[0, "rgb(255, 255, 255)"], [1, "rgb(0, 20, 200)"]]

    # Create figure
    fig = go.Figure()

    # Add data points
    scatter = go.Scatter(
        x=xy[:, 0],
        y=xy[:, 1],
        mode="markers",
        marker=dict(
            color=vals,
            colorscale=custom_colorscale,
            cmin=global_min,
            cmax=global_max,
            size=marker_size,
            colorbar=dict(title=dict(text=value_name, side="right"))
            if colorbar
            else None,
        ),
        customdata=np.stack([xy[:, 0], xy[:, 1], vals], axis=-1),
        hovertemplate=f"x: %{{customdata[0]:.2f}}<br>y: %{{customdata[1]:.2f}}"
        f"<br>{value_name}: %{{customdata[2]:.4f}}<extra></extra>",
        showlegend=False,
    )
    fig.add_trace(scatter)

    # Update layout
    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        xaxis=dict(
            range=[-np.pi, np.pi],
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            showticklabels=False,
            showline=False,
            visible=False,
        ),
        yaxis=dict(
            range=[-np.pi / 2, np.pi / 2],
            showgrid=False,
            showticklabels=False,
            showline=False,
            visible=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=50),
    )

    return fig
