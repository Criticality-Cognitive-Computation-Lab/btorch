"""Tests for hex visualization with human inspection figures.

Demonstrates:
1. Scatter plot with different coordinate formats
2. Flow field visualization (motion/optic flow)
3. Grid visualization with annotations
4. Coordinate system comparison (axial vs FlyWire)
"""

import matplotlib.pyplot as plt
import numpy as np

from btorch.utils.file import save_fig
from btorch.utils.hex import disk, to_pixel
from btorch.visualisation.hex import flow, grid, scatter
from btorch.visualisation.hex.static import (
    draw_hex_axes,
    hex_compass,
    looming_stimulus,
    scatter_from_index,
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

    # Zigzag coordinates
    scatter(q, r, values, coord_format="zigzag", ax=axes[1], cmap="RdBu_r")
    axes[1].set_title("Zigzag Coordinates (x,y)")

    # Pixel coordinates (direct)
    px, py = to_pixel(q, r, size=1.0)
    scatter(px, py, values, coord_format="pixel", ax=axes[2], cmap="RdBu_r")
    axes[2].set_title("Pixel Coordinates")

    plt.suptitle("Scatter: Same Data, Different Coordinate Systems", fontsize=14)
    plt.tight_layout()
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
    flow(q, r, dq, dr, coord_format="axial", ax=axes[0], scale=2.0, cmap="viridis")
    axes[0].set_title("Rotational Flow Field")

    # Overlay on scalar field
    vorticity = magnitude  # proxy for visualization
    scatter(q, r, vorticity, coord_format="axial", ax=axes[1], cmap="Blues", alpha=0.5)
    # Add flow arrows
    x, y = to_pixel(q, r)
    dx, dy = to_pixel(q + dq, r + dr)
    axes[1].quiver(x, y, dx - x, dy - y, scale=10, color="red", width=0.005)
    axes[1].set_title("Flow Overlay on Scalar Field")

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
    grid(2, coord_format="axial", annotate=True, ax=axes[0])
    axes[0].set_title("Axial Coordinates (q,r)")

    # Same in zigzag coordinates
    grid(2, coord_format="zigzag", annotate=True, ax=axes[1], orientation="pointy")
    axes[1].set_title("Zigzag Coordinates (x,y)")

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
    save_fig(fig, "orientation_map")
    plt.close()


def test_scatter_from_index():
    """Test scatter visualization from DataFrame with string "x,y" indices.

    Demonstrates connectome-style hex visualization where coordinates
    are stored as string indices like "-12,34" (double-width format).
    """
    import pandas as pd

    from btorch.utils.hex.doubled import axial_to_doublewidth

    # Create hex grid in axial coordinates
    q, r = disk(5)
    values = np.sin(q * np.pi / 3) * np.cos(r * np.pi / 3)

    # Convert to double-width coordinates (used by connectome datasets)
    x, y = axial_to_doublewidth(q, r)

    # Create DataFrame with "x,y" string indices (connectome format)
    index = [f"{int(xi)},{int(yi)}" for xi, yi in zip(x, y)]
    df = pd.Series(values, index=index)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Using scatter_from_index with DataFrame
    scatter_from_index(df, ax=axes[0], cmap="RdBu_r")
    axes[0].set_title("From DataFrame (string indices)")

    # Compare with regular scatter using axial coords
    scatter(q, r, values, coord_format="axial", ax=axes[1], cmap="RdBu_r")
    axes[1].set_title("From arrays (axial coords)")

    plt.suptitle("Scatter from Index vs Arrays", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "scatter_from_index")
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
        draw_hex_axes(
            ax,
            origin=(x.min(), y.min()),
            size=2.0,
            orientation="pointy",
            alignment=align,
        )
        ax.set_aspect("equal")
        ax.set_title(title)

    plt.tight_layout()
    save_fig(fig, "axes_overlay")
    plt.close()


def test_compass_inset():
    """Hex compass rose inset in corner of a plot."""
    q, r = disk(4)
    x, y = to_pixel(q, r)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c=range(len(q)), cmap="viridis", s=200)
    ax.set_aspect("equal")

    hex_compass(ax, alignment="vertex", loc="lower left")

    save_fig(fig, "compass_inset")
    plt.close()


def test_compass_edge_alignment():
    """Hex compass with edge-aligned axes on flat-top grid."""
    q, r = disk(4)
    x, y = to_pixel(q, r, orientation="flat")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c=range(len(q)), cmap="plasma", s=200)
    ax.set_aspect("equal")

    hex_compass(ax, alignment="edge", loc="upper right")

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
        compass="vertex",
    )

    grid(
        radius=3,
        coord_format="axial",
        orientation="pointy",
        annotate=True,
        ax=axes[1],
        title="Axial with edge compass",
        compass="edge",
    )

    plt.tight_layout()
    save_fig(fig, "grid_with_compass")
    plt.close()
