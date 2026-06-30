"""Tests for interactive hex heatmaps (Plotly).

Intended usage examples and edge cases.  Each test saves HTML + PNG to
fig_path() for manual visual inspection.  Coordinate conversions are
tested separately in test_static.py.

Interactive hex heatmap tests. Coordinate conversion utilities adapted
from flyvis (MIT License) by Yijie Yin (yijieyin).
"""

import matplotlib


matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from btorch.utils.file import fig_path
from btorch.utils.hex.coords import disk
from btorch.visualisation.hex.interactive import (
    _compute_pixel_dims,
    _hex_shape,
    _hex_vertices,
    _hide_axes,
    _merge_sizing,
    _merge_style,
    _resolve_coords,
    _values_to_colors,
    compass,
    draw_axes,
    grid,
    heatmap,
    quiver,
)


OUT = fig_path()


def _save(fig, name):
    """Save Plotly figure as HTML + PNG for manual inspection."""
    fig.write_html((OUT / f"{name}.html").as_posix())
    fig.write_image((OUT / f"{name}.png").as_posix())


def _axial_disk(radius=3):
    """Return (values_df, dataset_df) for a hex disk in axial coords."""
    q, r = disk(radius)
    dataset = pd.DataFrame({"p": q, "q": r})
    rng = np.random.default_rng(42)
    values = dataset.copy()
    values["val"] = rng.random(len(q))
    return values, dataset


# -- hex geometry ----------------------------------------------------------


def test_hex_vertices_flat():
    """Flat-top hex: first vertex points right (angle=0)."""
    vx, vy = _hex_vertices(0.0, 0.0, 1.0, "flat")
    assert len(vx) == 6
    np.testing.assert_allclose(vx[0], 1.0, atol=1e-10)
    np.testing.assert_allclose(vy[0], 0.0, atol=1e-10)
    # Adjacent vertex at 60°
    np.testing.assert_allclose(vx[1], 0.5, atol=1e-10)
    np.testing.assert_allclose(vy[1], np.sqrt(3) / 2, atol=1e-10)


def test_hex_vertices_pointy():
    """Pointy-top hex: first vertex at 30°."""
    vx, vy = _hex_vertices(0.0, 0.0, 1.0, "pointy")
    np.testing.assert_allclose(vx[0], np.cos(np.pi / 6), atol=1e-10)
    np.testing.assert_allclose(vy[0], np.sin(np.pi / 6), atol=1e-10)


def test_hex_vertices_tiling():
    """Adjacent flat-top hexes (centres 1.5, √3/2 apart) share an edge.

    The circumradius is 1.0, so the distance between centres of two
    adjacent flat-top hexes is √3.  The vertex at 30° on hex 0 should
    coincide with the vertex at 150° on hex 1.
    """
    v0x, v0y = _hex_vertices(0.0, 0.0, 1.0, "flat")
    # Second hex centre for flat-top: (1.5, √3/2)
    v1x, v1y = _hex_vertices(1.5, np.sqrt(3) / 2, 1.0, "flat")
    # Vertex 1 of hex 0 (at 60°) should equal vertex 3 of hex 1 (at 180°)
    np.testing.assert_allclose(v0x[1], v1x[3], atol=1e-10)
    np.testing.assert_allclose(v0y[1], v1y[3], atol=1e-10)


# -- colour mapping --------------------------------------------------------


def test_values_to_colors_interpolation():
    """Values map to interpolated colours, not just endpoints."""
    cs = [[0, "rgb(255,255,255)"], [1, "rgb(0,0,255)"]]
    colors = _values_to_colors(np.array([0.0, 0.5, 1.0]), cs, 0.0, 1.0)
    assert len(colors) == 3
    assert all(isinstance(c, str) for c in colors)
    # Midpoint should not equal either endpoint
    assert colors[1] != colors[0]
    assert colors[1] != colors[2]


def test_values_to_colors_constant():
    """Constant values map to the midpoint colour (t=0.5)."""
    cs = [[0, "rgb(0,0,0)"], [1, "rgb(255,255,255)"]]
    colors = _values_to_colors(np.array([5.0, 5.0]), cs, 0.0, 10.0)
    # Both should map to the same midpoint colour
    assert colors[0] == colors[1]


# -- heatmap: static rendering ---------------------------------------------


def test_heatmap_static_disk():
    """Static heatmap of a hex disk — primary usage.

    Verifies: single invisible scatter trace, shape per hex (bg + data),
    colour bar present, axes hidden with equal aspect.
    """
    values, dataset = _axial_disk(radius=3)
    fig = heatmap(values.copy(), dataset)
    _save(fig, "heatmap_static_disk")

    # One invisible scatter trace for hover + colorbar
    assert len(fig.data) == 1
    assert fig.data[0].marker.size == 0
    assert fig.data[0].marker.showscale is True

    # Shapes = bg hexes (white) + data hexes (coloured)
    n_bg = len(dataset.drop_duplicates(subset=["p", "q"]))
    assert len(fig.layout.shapes) == n_bg + len(values)
    # Background shapes are white
    assert fig.layout.shapes[0].fillcolor == "white"
    # Data shapes are coloured
    assert fig.layout.shapes[n_bg].fillcolor != "white"


def test_heatmap_pointy_orientation():
    """Pointy-top orientation produces different vertex geometry.

    Same data as flat-top but with orientation="pointy" — the SVG paths
    should differ (rotated 30°).
    """
    values, dataset = _axial_disk(radius=2)
    fig_flat = heatmap(values.copy(), dataset, orientation="flat")
    fig_pointy = heatmap(values.copy(), dataset, orientation="pointy")
    _save(fig_pointy, "heatmap_pointy")

    # The SVG paths differ because vertices are rotated
    assert fig_flat.layout.shapes[0].path != fig_pointy.layout.shapes[0].path


def test_heatmap_colorbar_off():
    """Colorbar=False hides the colour bar."""
    values, dataset = _axial_disk(radius=2)
    fig = heatmap(values.copy(), dataset, colorbar=False)
    _save(fig, "heatmap_no_colorbar")
    assert not fig.data[0].marker.showscale


def test_heatmap_custom_colorscale():
    """Custom RGB colorscale is passed through to scatter marker."""
    values, dataset = _axial_disk(radius=2)
    cs = [[0, "rgb(255,0,0)"], [1, "rgb(0,0,255)"]]
    fig = heatmap(values.copy(), dataset, custom_colorscale=cs)
    _save(fig, "heatmap_custom_colors")
    assert list(fig.data[0].marker.colorscale) == [
        (0, "rgb(255,0,0)"),
        (1, "rgb(0,0,255)"),
    ]


def test_heatmap_hover_flywire():
    """Default hover includes p,q and x,y flywire coordinates."""
    values, dataset = _axial_disk(radius=2)
    fig = heatmap(values.copy(), dataset)
    tmpl = fig.data[0].hovertemplate
    assert "p,q" in tmpl
    assert "x,y" in tmpl


def test_heatmap_hover_simple():
    """include_flywire_hover=False shows only pixel x,y."""
    values, dataset = _axial_disk(radius=2)
    fig = heatmap(values.copy(), dataset, include_flywire_hover=False)
    tmpl = fig.data[0].hovertemplate
    assert "p,q" not in tmpl
    assert "x:" in tmpl


def test_heatmap_single_hex():
    """Edge case: single hex cell still renders correctly."""
    dataset = pd.DataFrame({"p": [0], "q": [0]})
    values = pd.DataFrame({"p": [0], "q": [0], "val": [0.5]})
    fig = heatmap(values, dataset)
    _save(fig, "heatmap_single_hex")
    assert len(fig.layout.shapes) == 2  # 1 bg + 1 data


# -- heatmap: animation ----------------------------------------------------


def test_heatmap_animated():
    """Multi-column DataFrame produces frames + slider.

    Each frame has its own set of layout shapes (the data hex colours
    change per frame, background stays white).
    """
    values, dataset = _axial_disk(radius=2)
    values["t1"] = np.random.default_rng(0).random(len(values))
    values["t2"] = np.random.default_rng(1).random(len(values))
    fig = heatmap(values.copy(), dataset)
    _save(fig, "heatmap_animated")

    assert len(fig.frames) == 3
    assert len(fig.layout.sliders[0].steps) == 3

    # Slider labels match column names
    labels = [s.label for s in fig.layout.sliders[0].steps]
    assert labels == ["val", "t1", "t2"]

    # Each frame has its own shapes
    for frame in fig.frames:
        assert frame.layout.shapes is not None
        assert len(frame.layout.shapes) > 0


# -- overlay utilities -----------------------------------------------------


def test_draw_axes_annotations():
    """draw_axes adds 6 annotations (3 arrows + 3 labels)."""
    values, dataset = _axial_disk(radius=2)
    fig = heatmap(values.copy(), dataset)
    n_before = len(fig.layout.annotations)
    draw_axes(fig, origin=(0.0, 0.0), size=2.0, orientation="flat")
    assert len(fig.layout.annotations) == n_before + 6  # 3 arrows + 3 labels


def test_grid_with_labels():
    """Grid with annotate=True shows coordinate labels."""
    fig = grid(radius=2, annotate=True)
    _save(fig, "grid_annotated")
    assert len(fig.layout.shapes) > 0
    # Text mode should be "markers+text"
    assert fig.data[0].mode == "markers+text"


def test_grid_with_axes_and_compass():
    """Grid with show_axes and show_compass overlays."""
    fig = grid(radius=2, show_axes=True, show_compass="vertex")
    _save(fig, "grid_axes_compass")
    # Should have annotations from draw_axes + compass
    assert len(fig.layout.annotations) > 0


def test_quiver_basic():
    """Quiver plot with rotational flow field."""
    from btorch.utils.hex.coords import disk

    q, r = disk(3)
    angle = np.arctan2(r.astype(float), q.astype(float) + 1e-10)
    mag = np.sqrt(q.astype(float) ** 2 + r.astype(float) ** 2) + 0.5
    dq = (-np.sin(angle) * mag * 0.3).astype(float)
    dr = (np.cos(angle) * mag * 0.3).astype(float)
    fig = quiver(q, r, dq, dr, coord_format="axial", scale=2.0)
    _save(fig, "quiver_rotational")
    # Should have arrow annotations
    assert len(fig.layout.annotations) > 0
    # Should have colorbar
    assert fig.data[0].marker.showscale is True


def test_full_combo():
    """Full combo: grid + scatter points + quiver + axes + compass.

    Matches the static test_full_combo: hex grid with coloured scatter
    points at centres, quiver vectors, q/r/s axes, and compass rose.
    """
    from btorch.utils.hex.coords import disk

    q, r = disk(3)
    values = np.sqrt(q.astype(float) ** 2 + r.astype(float) ** 2)
    angle = np.arctan2(r.astype(float), q.astype(float) + 1e-10)
    dq = (-np.sin(angle) * 0.4).astype(float)
    dr = (np.cos(angle) * 0.4).astype(float)

    # Build figure: white hex grid + coloured scatter + quiver arrows
    _, _, _, _, x, y = _resolve_coords(q, r, "axial", "pointy")
    _, _, _, _, tx, ty = _resolve_coords(q + dq * 2, r + dr * 2, "axial", "pointy")

    bg_shapes = [
        _hex_shape(xi, yi, 1.0, "white", "gray", 0.5, "pointy") for xi, yi in zip(x, y)
    ]

    fig = go.Figure(layout=go.Layout(shapes=bg_shapes))

    # Scatter points coloured by distance
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                color=values,
                colorscale="Viridis",
                size=8,
                line=dict(color="black", width=0.5),
            ),
            showlegend=False,
        )
    )

    # Quiver arrows
    for xi, yi, dxi, dyi in zip(x, y, tx - x, ty - y):
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
            arrowcolor="red",
            opacity=0.7,
        )

    sty = _merge_style(None)
    sz = _merge_sizing(None)
    area_w, area_h, *_ = _compute_pixel_dims(sz, 72)
    hex_pad = 2.0
    fig.update_layout(
        autosize=False,
        height=area_h,
        width=area_w,
        margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
        paper_bgcolor=sty["papercolor"],
        plot_bgcolor=sty["papercolor"],
    )
    _hide_axes(
        fig,
        x_range=(float(x.min()) - hex_pad, float(x.max()) + hex_pad),
        y_range=(float(y.min()) - hex_pad, float(y.max()) + hex_pad),
    )

    # Axes overlay (lower-left, above grid)
    draw_axes(
        fig,
        origin=(float(x.min()) + 1.5, float(y.min()) + 1.5),
        size=2.5,
        orientation="pointy",
    )

    # Compass rose (lower-left, floats on top via pixel coords)
    compass(fig, loc="lower left", alignment="vertex")

    _save(fig, "full_combo_interactive")
    assert len(fig.layout.shapes) > 0
    assert len(fig.layout.annotations) > 0
