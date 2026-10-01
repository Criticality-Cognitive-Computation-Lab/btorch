"""Tests for hex visualization with human inspection figures.

Demonstrates:
1. Scatter plot with different coordinate formats
2. Quiver (vector field) visualization
3. Grid visualization with annotations
4. Coordinate system comparison (axial vs FlyWire)
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import RegularPolygon

from btorch.utils.file import save_fig
from btorch.utils.hex import disk, to_pixel
from btorch.utils.hex.offset import axial_to_zigzag
from btorch.visualisation.hex import grid, quiver, scatter
from btorch.visualisation.hex.static import (
    _resolve_to_pixel,
    compass,
    draw_axes,
    looming_stimulus,
)


def test_scatter_coordinate_formats():
    """Compare scatter visualization in different coordinate formats.

    Shows the same data in axial, zigzag, and pixel coordinates to
    verify coordinate transformations are consistent.
    """
    q, r = disk(5)
    values = np.sin(q * np.pi / 3) * np.cos(r * np.pi / 3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial coordinates (converted to pixel for display)
    scatter(q, r, values, coord_format="axial", ax=axes[0], cmap="RdBu_r")
    axes[0].set_title("Axial Coordinates (q,r)")

    # Zigzag coordinates: must convert axial → zigzag first
    zx, zy = axial_to_zigzag(q, r)
    scatter(zx, zy, values, coord_format="zigzag", ax=axes[1], cmap="RdBu_r")
    axes[1].set_title("Zigzag Coordinates (x,y)")

    # Pixel coordinates (direct)
    px, py = to_pixel(q, r, size=1.0)
    scatter(px, py, values, coord_format="pixel", ax=axes[2], cmap="RdBu_r")
    axes[2].set_title("Pixel Coordinates")

    plt.suptitle("Scatter: Same Data, Different Coordinate Systems", fontsize=14)
    plt.tight_layout()
    # Each panel draws one RegularPolygon per hex (disk(5) has 3*5*6+1 = 91).
    assert len(q) == 91
    for ax in axes:
        assert len(ax.patches) == 91
    # The three coordinate formats describe the same lattice, so the drawn hex
    # centres must coincide once the axial input is converted to pixel space.
    centres = [
        np.array(sorted(map(tuple, np.round([p.xy for p in ax.patches], 6))))
        for ax in axes
    ]
    # axial and pixel panels are built from identical geometry
    np.testing.assert_allclose(centres[0], centres[2], atol=1e-6)
    # zigzag panel covers the same number of distinct centres
    assert len(np.unique(centres[1], axis=0)) == 91
    save_fig(fig, "scatter_coordinate_formats")
    plt.close()


def test_flow_field_visualization():
    """Visualize flow fields on hex grid - example: rotational flow.

    Demonstrates optic flow or motion field visualization,
    common in fly vision research.
    """
    q, r = disk(4)

    # Create rotational flow around center
    # Flow direction is perpendicular to position vector
    # In hex coordinates, this is approximated
    angle = np.arctan2(r, q + 1e-10)
    magnitude = np.sqrt(q**2 + r**2) + 0.5

    # Flow components (in axial, these are approximations)
    dq = -np.sin(angle) * magnitude * 0.3
    dr = np.cos(angle) * magnitude * 0.3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Quiver plot
    ret = quiver(
        q, r, dq, dr, coord_format="axial", ax=axes[0], scale=2.0, cmap="viridis"
    )
    # quiver returns (fig, ax) and adds exactly one Quiver artist with one arrow
    # per input hex.
    assert ret[1] is axes[0]
    assert len(axes[0].collections) == 1
    assert axes[0].collections[0].U.shape == (len(q),)
    axes[0].set_title("Rotational Flow Field")

    # Overlay on scalar field
    vorticity = magnitude  # proxy for visualization
    scatter(q, r, vorticity, coord_format="axial", ax=axes[1], cmap="Blues", alpha=0.5)
    # Add flow arrows
    x, y = to_pixel(q, r)
    dx, dy = to_pixel(q + dq, r + dr)
    axes[1].quiver(x, y, dx - x, dy - y, scale=10, color="red", width=0.005)
    axes[1].set_title("Flow Overlay on Scalar Field")
    # scalar hexes (patches) plus the overlaid red quiver (collection)
    assert len(axes[1].patches) == len(q)
    assert len(axes[1].collections) == 1

    plt.suptitle("Flow Field Visualization", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "flow_field_visualization")
    plt.close()


def test_grid_annotation():
    """Visualize hex grid with coordinate annotations.

    Useful for understanding hex coordinate systems and debugging
    coordinate transformations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Small grid with annotations
    fig_ret, ax_ret = grid(2, coord_format="axial", annotate=True, ax=axes[0])
    # radius-2 grid has 19 hexes, each with one patch and one text label
    assert ax_ret is axes[0] and fig_ret is fig
    assert len(axes[0].patches) == 19
    assert len(axes[0].texts) == 19
    # axial labels are "(q,r)" strings; the centre hex must be present
    assert "(0,0)" in [t.get_text() for t in axes[0].texts]
    axes[0].set_title("Axial Coordinates (q,r)")

    # Same in zigzag coordinates
    grid(2, coord_format="zigzag", annotate=True, ax=axes[1], orientation="pointy")
    axes[1].set_title("Zigzag Coordinates (x,y)")
    assert len(axes[1].patches) == 19
    assert len(axes[1].texts) == 19

    plt.suptitle("Hex Grid Coordinate Systems", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "grid_annotation")
    plt.close()


def test_receptive_field_visualization():
    """Visualize center-surround receptive field on hex grid.

    Example use case: modeling retinal ganglion cell or
    early visual system receptive fields.
    """
    from btorch.utils.hex import radius

    q, r = disk(8)
    dist = radius(q, r)

    # Difference of Gaussians (DoG) - classic RF model
    center = np.exp(-(dist**2) / (2 * 2.5**2))
    surround = 0.6 * np.exp(-(dist**2) / (2 * 5.0**2))
    rf = center - surround

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Center component
    scatter(q, r, center, coord_format="axial", ax=axes[0], cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title("Center (Excitatory)")

    # Surround component
    scatter(
        q, r, surround, coord_format="axial", ax=axes[1], cmap="Blues", vmin=0, vmax=0.6
    )
    axes[1].set_title("Surround (Inhibitory)")

    # Full RF
    scatter(
        q, r, rf, coord_format="axial", ax=axes[2], cmap="RdBu_r", vmin=-0.6, vmax=1
    )
    axes[2].set_title("Full Receptive Field (DoG)")

    plt.suptitle("Center-Surround Receptive Field Model", fontsize=14)
    plt.tight_layout()
    # DoG receptive field: positive at the centre, negative far in the surround
    assert rf[dist == 0][0] == pytest.approx(0.4)  # 1 - 0.6
    assert rf.min() < 0 < rf.max()
    assert all(len(ax.patches) == len(q) for ax in axes)
    # vmin/vmax are honoured: the centre hex (value 1 == vmax) gets the top
    # colour of the "Reds" map, the far corner (value ~0 == vmin) the lightest.
    face = {tuple(np.round(p.xy, 6)): p.get_facecolor() for p in axes[0].patches}
    far = max(face, key=lambda k: np.hypot(*k))
    assert face[(0.0, 0.0)][:3] == pytest.approx(plt.get_cmap("Reds")(1.0)[:3])
    assert sum(face[far][:3]) > sum(face[(0.0, 0.0)][:3])
    save_fig(fig, "receptive_field_visualization")
    plt.close()


def test_orientation_map():
    """Visualize orientation preference map on hex grid.

    Example use case: modeling orientation-selective neurons
    in early visual cortex.
    """
    q, r = disk(10)

    # Create orientation map (pinwheel pattern)
    angle = np.arctan2(r, q + 0.1)  # angle from center
    orientation = (np.sin(angle * 3) + 1) / 2  # map to [0, 1]

    # Add spatial frequency component
    sf = np.sin(np.sqrt(q**2 + r**2) * np.pi / 3)
    combined = orientation * (1 + 0.3 * sf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter(
        q, r, orientation, coord_format="axial", ax=axes[0], cmap="hsv", vmin=0, vmax=1
    )
    axes[0].set_title("Orientation Preference")

    scatter(
        q,
        r,
        combined,
        coord_format="axial",
        ax=axes[1],
        cmap="twilight",
        vmin=0,
        vmax=1.3,
    )
    axes[1].set_title("Orientation + Spatial Frequency")

    plt.suptitle("Orientation Maps on Hex Grid", fontsize=14)
    plt.tight_layout()
    assert orientation.min() >= 0 and orientation.max() <= 1
    assert len(axes[0].patches) == len(q) == len(axes[1].patches)
    save_fig(fig, "orientation_map")
    plt.close()


def test_looming_stimulus():
    """Test looming stimulus generation on hex grid.

    Creates an expanding stimulus pattern starting from center,
    simulating a looming object approaching the retina.
    """
    # Create a hex grid and get all coordinates as strings
    q, r = disk(8)
    from btorch.utils.hex.doubled import axial_to_doublewidth

    x, y = axial_to_doublewidth(q, r)
    all_coords = [f"{int(xi)},{int(yi)}" for xi, yi in zip(x, y)]

    # Start from center
    start_coords = ["0,0"]

    # Generate looming stimulus over time
    stim_sequence = looming_stimulus(start_coords, all_coords, n_time=5)

    # Visualize each time step
    fig, axes = plt.subplots(1, len(stim_sequence), figsize=(15, 3))

    for t, stim_coords in enumerate(stim_sequence):
        # Create binary mask for stimulated hexes
        values = np.array(
            [
                1.0 if f"{int(xi)},{int(yi)}" in stim_coords else 0.0
                for xi, yi in zip(x, y)
            ]
        )

        scatter(
            q,
            r,
            values,
            coord_format="axial",
            ax=axes[t],
            cmap="Reds",
            vmin=0,
            vmax=1,
            edgecolor="gray",
        )
        axes[t].set_title(f"t={t}")

    plt.suptitle("Looming Stimulus Expansion", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "looming_stimulus")
    plt.close()

    # Verify stimulus expands
    assert len(stim_sequence) == 5 + 1  # initial frame + n_time steps
    # the centre stays stimulated throughout, and all coords are valid hexes
    assert all("0,0" in s for s in stim_sequence)
    assert all(set(s) <= set(all_coords) for s in stim_sequence)
    # the first step already grows beyond the single start hex
    assert len(stim_sequence[1]) > 1
    sizes = [len(s) for s in stim_sequence]
    assert sizes == sorted(sizes), "Stimulus should expand monotonically"
    assert sizes[0] == 1, "Stimulus should start with single hex"


def test_axes_overlay():
    """Overlay q/r/s axis arrows directly on a hex plot."""
    q, r = disk(3)
    x, y = to_pixel(q, r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, align, title in [
        (axes[0], "vertex", "Vertex-aligned axes"),
        (axes[1], "edge", "Edge-aligned axes"),
    ]:
        ax.scatter(x, y, c="lightgray", s=100)
        draw_axes(
            ax,
            origin=(x.min(), y.min()),
            size=2.0,
            orientation="pointy",
            alignment=align,
        )
        ax.set_aspect("equal")
        ax.set_title(title)

    plt.tight_layout()
    # draw_axes adds 3 labelled axes (q, r, s) as text annotations
    for ax in axes:
        assert len(ax.texts) == 3
    save_fig(fig, "axes_overlay")
    plt.close()


def test_compass_inset():
    """Hex compass rose inset in corner of a plot."""
    q, r = disk(4)
    x, y = to_pixel(q, r)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c=range(len(q)), cmap="viridis", s=200)
    ax.set_aspect("equal")

    returned = compass(ax, alignment="vertex", loc="lower left")

    # the compass lives in an inset axes in the lower-left corner
    assert returned is not None and len(ax.child_axes) == 1
    bbox = ax.child_axes[0].get_position()
    assert bbox.x0 < 0.5 and bbox.y0 < 0.5  # lower left of the parent axes
    save_fig(fig, "compass_inset")
    plt.close()


def test_compass_edge_alignment():
    """Hex compass with edge-aligned axes on flat-top grid."""
    q, r = disk(4)
    x, y = to_pixel(q, r, orientation="flat")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c=range(len(q)), cmap="plasma", s=200)
    ax.set_aspect("equal")

    compass(ax, alignment="edge", loc="upper right")

    assert len(ax.child_axes) == 1
    bbox = ax.child_axes[0].get_position()
    assert bbox.x0 > 0.5 and bbox.y0 > 0.5  # upper right of the parent axes
    save_fig(fig, "compass_edge_alignment")
    plt.close()


def test_grid_with_compass():
    """Grid visualization with hex compass rose inset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    grid(
        radius=3,
        coord_format="axial",
        orientation="pointy",
        annotate=True,
        ax=axes[0],
        title="Axial with vertex compass",
        show_compass="vertex",
    )

    grid(
        radius=3,
        coord_format="axial",
        orientation="pointy",
        annotate=True,
        ax=axes[1],
        title="Axial with edge compass",
        show_compass="edge",
    )

    plt.tight_layout()
    # radius-3 grid: 37 hexes; each panel gets exactly one compass inset
    for ax in axes:
        assert len(ax.patches) == 37
        assert len(ax.child_axes) == 1
    assert axes[0].get_title() == "Axial with vertex compass"
    save_fig(fig, "grid_with_compass")
    plt.close()


def test_full_combo():
    """Full combo: grid + scatter points + quiver vectors + axes + compass.

    Exercises all static hex visualisation primitives on a single figure
    to verify they compose correctly.
    """
    q, r = disk(3)
    x, y = to_pixel(q, r, orientation="pointy")

    # Values for scatter coloring
    values = np.sqrt(q.astype(float) ** 2 + r.astype(float) ** 2)

    # Rotational flow vectors
    angle = np.arctan2(r.astype(float), q.astype(float) + 1e-10)
    dq = -np.sin(angle) * 0.4
    dr = np.cos(angle) * 0.4

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Grid (white hexes with gray edges)
    for xi, yi in zip(x, y):
        hex_patch = RegularPolygon(
            (xi, yi),
            numVertices=6,
            radius=0.95,
            orientation=0,
            facecolor="white",
            edgecolor="gray",
            linewidth=0.5,
        )
        ax.add_patch(hex_patch)

    # 2. Scatter points at hex centres, colored by distance
    ax.scatter(
        x,
        y,
        c=values,
        cmap="viridis",
        s=60,
        zorder=5,
        edgecolors="black",
        linewidths=0.5,
    )

    # 3. Quiver vectors
    (tx, ty), _ = _resolve_to_pixel(
        q + dq * 2,
        r + dr * 2,
        "axial",
        size=1.0,
        orientation="pointy",
    )
    dx, dy = tx - x, ty - y
    ax.quiver(
        x,
        y,
        dx,
        dy,
        scale_units="xy",
        angles="xy",
        scale=1,
        color="red",
        alpha=0.7,
        zorder=6,
    )

    # 4. Axes overlay
    draw_axes(
        ax,
        origin=(x.min() + 1, y.min() + 1),
        size=2.0,
        orientation="pointy",
        alignment="vertex",
    )

    # 5. Compass
    compass(ax, alignment="vertex", loc="lower left")

    ax.set_aspect("equal")
    ax.set_xlim(x.min() - 1.5, x.max() + 1.5)
    ax.set_ylim(y.min() - 1.5, y.max() + 1.5)
    ax.set_title("Full combo: grid + scatter + quiver + axes + compass")
    ax.axis("off")

    # 37 hex patches + 3 axis arrows from draw_axes; 1 scatter + 1 quiver
    # collection; the compass inset adds a child axes
    assert len(ax.patches) == 37 + 3
    assert len(ax.collections) == 2
    assert len(ax.child_axes) == 1
    # _resolve_to_pixel of unit axial steps matches pointy-top geometry:
    # (1, 0) -> (sqrt(3), 0), (0, 1) -> (sqrt(3)/2, 1.5)
    (px, py), _ = _resolve_to_pixel(
        np.array([1, 0]), np.array([0, 1]), "axial", size=1.0, orientation="pointy"
    )
    np.testing.assert_allclose(px, [np.sqrt(3), np.sqrt(3) / 2], atol=1e-9)
    np.testing.assert_allclose(py, [0.0, 1.5], atol=1e-9)
    save_fig(fig, "full_combo")
    plt.close()
