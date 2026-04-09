"""Hexagonal visualization subpackage.

Supports multiple coordinate formats: axial (q,r), FlyWire (x,y).
"""

# Interactive (Plotly)
# Animations (Matplotlib)
from .animate import HexAnimationCollector, HexFlow, HexScatter
from .interactive import heatmap, heatmap_from_index, mollweide_projection

# Receptive field (Matplotlib)
from .receptive_field import ReceptiveFieldViewer, kernel, strf

# Static (Matplotlib)
from .static import (
    draw_hex_axes,
    flow,
    grid,
    hex_compass,
    looming_stimulus,
    scatter,
    scatter_from_index,
)


__all__ = [
    # Interactive
    "heatmap",
    "heatmap_from_index",
    "mollweide_projection",
    # Static
    "scatter",
    "scatter_from_index",
    "flow",
    "grid",
    "looming_stimulus",
    "draw_hex_axes",
    "hex_compass",
    # Animation
    "HexScatter",
    "HexFlow",
    "HexAnimationCollector",
    # Receptive field
    "kernel",
    "strf",
    "ReceptiveFieldViewer",
]
