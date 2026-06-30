"""Hexagonal visualization subpackage.

Supports multiple coordinate formats: axial (q,r), FlyWire (x,y).
"""

# Interactive (Plotly)
# Animations (Matplotlib)
from .animate import HexQuiver, HexScatter
from .interactive import heatmap

# Receptive field (Matplotlib)
from .receptive_field import ReceptiveFieldViewer, kernel, strf

# Static (Matplotlib)
from .static import (
    compass,
    draw_axes,
    grid,
    looming_stimulus,
    quiver,
    scatter,
)


__all__ = [
    # Interactive
    "heatmap",
    # Static
    "scatter",
    "quiver",
    "grid",
    "looming_stimulus",
    "draw_axes",
    "compass",
    # Animation
    "HexScatter",
    "HexQuiver",
    # Receptive field
    "kernel",
    "strf",
    "ReceptiveFieldViewer",
]
