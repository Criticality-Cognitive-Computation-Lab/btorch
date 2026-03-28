"""Hexagonal heatmap visualization using Plotly.

Provides interactive hexagonal grid plots for spatial data
visualization, commonly used for brain region mapping with hexagonal
tiling.
"""

import pandas as pd
import plotly.graph_objects as go

from ..utils.hex_utils import hex_to_pixel


def hex_heatmap(
    df: pd.Series | pd.DataFrame,
    dataset: pd.DataFrame,
    style: dict | None = None,
    sizing: dict | None = None,
    dpi: int = 72,
    custom_colorscale: list | None = None,
) -> go.Figure:
    """Generate an interactive hexagonal heatmap.

    Visualizes data on a hexagonal grid layout. Single-column data produces
    a static heatmap; multi-column DataFrames produce an animated heatmap
    with a slider to navigate through timepoints or conditions.

    Args:
        df: Data to visualize. Single Series for static plot, or DataFrame
            with multiple columns for animated plot (one frame per column).
            Must have 'p' and 'q' columns representing hex grid coordinates.
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

    Returns:
        Plotly Figure with hexagonal heatmap. Static for Series input,
        animated with slider for DataFrame input.

    Raises:
        ValueError: If `df` is not a Series or DataFrame.

    Example:
        >>> # Static heatmap
        >>> fig = hex_heatmap(data_series, background_dataset)
        >>> fig.show()
        >>>
        >>> # Animated heatmap with timepoints
        >>> fig = hex_heatmap(timepoint_df, background_dataset)
        >>> fig.write_html("animated_hexmap.html")
    """

    def bg_hex():
        goscatter = go.Scatter(
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
        return goscatter

    def data_hex(aseries):
        goscatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker_symbol=symbol_number,
            marker={
                "cmin": global_min,
                "cmax": global_max,
                "size": sizing["markersize"],
                "color": aseries.values,
                "line": {
                    "width": sizing["markerlinewidth"],
                    "color": "lightgrey",
                },
                "colorbar": {
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
                },
                "colorscale": custom_colorscale,
            },
            showlegend=False,
        )
        return goscatter

    default_style = {
        "font_type": "arial",
        "markerlinecolor": "rgba(0,0,0,0)",
        "linecolor": "black",
        "papercolor": "rgba(255,255,255,255)",
    }

    markersize = 16

    default_sizing = {
        "fig_width": 260,
        "fig_height": 220,
        "fig_margin": 0,
        "fsize_ticks_pt": 20,
        "fsize_title_pt": 20,
        "markersize": markersize,
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

    background_hex = dataset
    background_hex = background_hex.drop_duplicates(subset=["p", "q"])[
        ["p", "q"]
    ].astype(float)
    x, y = hex_to_pixel(background_hex.p, background_hex.q, mode="flat")
    background_hex["x"], background_hex["y"] = x, y

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

    df["x"], df["y"] = hex_to_pixel(df.p, df.q, mode="flat")
    x_vals, y_vals = df.x, df.y
    df = df.drop(columns=["p", "q", "x", "y"])

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

        fig.layout.sliders[0].steps = slider_steps
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
