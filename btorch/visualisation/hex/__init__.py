"""Hexagonal visualization subpackage.

Supports multiple coordinate formats: axial (q,r), FlyWire (x,y).
"""

from .animate import HexQuiver, HexScatter
from .interactive import heatmap
from .options import HexColorMap, HexGeometry, HexPatchStyle, HexReference
from .receptive_field import ReceptiveFieldViewer, kernel, strf
from .static import (
    compass,
    draw_axes,
    grid,
    looming_stimulus,
    quiver,
    scatter,
)


__all__ = [
    "HexColorMap",
    "HexGeometry",
    "HexPatchStyle",
    "HexReference",
    "heatmap",
    "scatter",
    "quiver",
    "grid",
    "looming_stimulus",
    "draw_axes",
    "compass",
    "HexScatter",
    "HexQuiver",
    "kernel",
    "strf",
    "ReceptiveFieldViewer",
]
