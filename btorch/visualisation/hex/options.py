"""Option dataclasses for the static hex plots.

These group the keyword options of :func:`~btorch.visualisation.hex.scatter` and
:func:`~btorch.visualisation.hex.quiver` into frozen, reusable objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class HexGeometry:
    """How hex coordinates are laid out in pixel space.

    Attributes:
        layout: Explicit pixel layout (e.g. ``"flat"``); defaults to ``"flat"`` for
            zigzag/flywire coordinates and to ``orientation`` otherwise.
        size: Hexagon size used to convert coordinates to pixels.
        orientation: ``"pointy"`` or ``"flat"`` top.
        rotation_deg: Global display rotation in degrees (only applied by
            the ``"flywire"`` layout).
    """

    layout: str | None = None
    size: float = 1.0
    orientation: Literal["pointy", "flat"] = "pointy"
    rotation_deg: float = 0.0


@dataclass(frozen=True, kw_only=True)
class HexColorMap:
    """Colour mapping of the plotted values (or vector magnitudes).

    Attributes:
        cmap: Matplotlib colormap name.
        vmin: Lower colour limit; defaults to the data minimum.
        vmax: Upper colour limit; defaults to the data maximum.
    """

    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None


@dataclass(frozen=True, kw_only=True)
class HexPatchStyle:
    """Appearance of the hexagon patches.

    Attributes:
        edgecolor: Hexagon edge colour (``None`` for matplotlib's default).
        edgewidth: Hexagon edge width.
        alpha: Transparency (0-1).
    """

    edgecolor: str | None = None
    edgewidth: float = 0.5
    alpha: float = 1.0


@dataclass(frozen=True, kw_only=True)
class HexReference:
    """Orientation aids drawn on top of the plot.

    Attributes:
        axes_alignment: If ``"vertex"`` or ``"edge"``, draw q/r/s reference axes.
        show_compass: If ``"vertex"`` or ``"edge"``, draw a hex compass rose inset.
    """

    axes_alignment: Literal["vertex", "edge"] | None = None
    show_compass: Literal["vertex", "edge"] | None = None
