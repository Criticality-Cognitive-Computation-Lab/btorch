"""Tests for hexagonal heatmap visualization."""

import numpy as np
import pandas as pd

from btorch.utils.file import fig_path
from btorch.visualisation.hex.interactive import (
    heatmap,
    heatmap_from_index,
    mollweide_projection,
)


def test_hex_heatmap_static():
    """Static hexmap for single timepoint - common for brain region mapping."""
    # Hex grid representing brain regions (e.g., fly visual system columns)
    dataset = pd.DataFrame(
        {
            "p": [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4],
            "q": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        }
    )
    values = pd.DataFrame(
        {
            "p": [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4],
            "q": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "firing_rate": [0.1, 0.5, 0.9, 0.2, 0.6, 0.3, 0.8, 0.4, 0.7, 0.5, 0.2, 0.6],
        }
    )

    fig = heatmap(values.copy(), dataset)

    output_path = fig_path(__file__) / "hexmap_static.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path.as_posix())


def test_hex_heatmap_timeseries():
    """Animated hexmap for time series - common for temporal activity patterns."""
    # Small hex grid with activity across multiple timepoints
    dataset = pd.DataFrame(
        {
            "p": [0, 1, 2, 0, 1, 2],
            "q": [0, 0, 0, 1, 1, 1],
        }
    )
    values = pd.DataFrame(
        {
            "p": [0, 1, 2, 0, 1, 2],
            "q": [0, 0, 0, 1, 1, 1],
            "t_0ms": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "t_50ms": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "t_100ms": [0.3, 0.6, 0.2, 0.5, 0.1, 0.4],
        }
    )

    fig = heatmap(values.copy(), dataset)

    assert len(fig.frames) == 3
    assert len(fig.layout.sliders[0].steps) == 3

    output_path = fig_path(__file__) / "hexmap_timeseries.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path.as_posix())


def test_yijieyin_style_connectome_visualization():
    """Reproduce YijieYin connectome_interpreter visualization with fake data.

    Tests heatmap_from_index and mollweide_projection with connectome-
    style double-width hex coordinates (string "x,y" indices).
    """
    # Create fake hex grid (similar to Nern2024 optic lobe columns)
    # Double-width coordinates: col = 2*q + r, row = r
    np.random.seed(42)
    coords = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:  # hex disk constraint
                x = 2 * q + r  # double-width column
                y = r  # double-width row
                coords.append(f"{int(x)},{int(y)}")

    # Create fake activity data (static)
    values_static = pd.Series(np.random.rand(len(coords)), index=coords)

    # Test static heatmap_from_index
    fig_static = heatmap_from_index(
        values_static,
        title="Fake Connectome - Static Activity",
        value_name="firing_rate",
    )
    output_path = fig_path(__file__) / "yijieyin_static.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_static.write_html(output_path.as_posix())

    # Create animated data (multiple timepoints)
    timepoints = ["t_0", "t_1", "t_2", "t_3"]
    data_dict = {}
    for t in timepoints:
        data_dict[t] = np.random.rand(len(coords))
    values_animated = pd.DataFrame(data_dict, index=coords)

    # Test animated heatmap_from_index
    fig_animated = heatmap_from_index(
        values_animated,
        title="Fake Connectome - Temporal Activity",
        value_name="response",
    )
    assert len(fig_animated.frames) == len(timepoints)
    output_path = fig_path(__file__) / "yijieyin_animated.html"
    fig_animated.write_html(output_path.as_posix())

    # Test mollweide projection
    fig_moll = mollweide_projection(
        values_static,
        value_name="activity",
    )
    output_path = fig_path(__file__) / "yijieyin_mollweide.html"
    fig_moll.write_html(output_path.as_posix())

    print("YijieYin-style connectome visualization tests passed")
