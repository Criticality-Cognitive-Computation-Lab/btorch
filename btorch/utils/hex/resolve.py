"""Shared coordinate-to-pixel pipeline for hex visualisation.

All visualisation functions (static, interactive, animate) share this
module for coordinate conversion. The entry point is :func:`resolve_hex`,
which converts any supported hex input to axial ``(q, r)`` and pixel
``(x, y)`` in a single call.

Coordinate formats are dispatched via :data:`_TO_AXIAL`, and screen
layouts via :data:`_TO_PIXEL`. Both are plain dicts mapping string
keys to converter functions.
"""

import numpy as np

from .doubled import doubleheight_to_axial, doublewidth_to_axial
from .offset import (
    even_q_to_axial,
    even_r_to_axial,
    flywire_to_pixel,
    odd_q_to_axial,
    odd_r_to_axial,
    zigzag_to_axial,
    zigzag_to_pixel,
)
from .transform import axial_from_cube, to_pixel


_TO_AXIAL = {
    "axial": lambda c1, c2: (
        np.asarray(c1, dtype=float),
        np.asarray(c2, dtype=float),
    ),
    "odd_r": odd_r_to_axial,
    "even_r": even_r_to_axial,
    "odd_q": odd_q_to_axial,
    "even_q": even_q_to_axial,
    "doublewidth": doublewidth_to_axial,
    "doubleheight": doubleheight_to_axial,
    "zigzag": zigzag_to_axial,
}


def _to_pixel_pointy(q, r, size=1.0, **_kw):
    return to_pixel(q, r, size=size, orientation="pointy")


def _to_pixel_flat(q, r, size=1.0, **_kw):
    return to_pixel(q, r, size=size, orientation="flat")


def _to_pixel_flywire(q, r, size=1.0, rotation_deg=0.0, **_kw):
    return flywire_to_pixel(q, r, size=size, rotation_deg=rotation_deg)


_TO_PIXEL = {
    "pointy": _to_pixel_pointy,
    "flat": _to_pixel_flat,
    "flywire": _to_pixel_flywire,
}

_HEX_SYMBOLS = {
    "pointy": 14,
    "flat": 15,
    "flywire": 15,
}


def resolve_hex(
    c1,
    c2,
    coord_format="axial",
    layout="pointy",
    size=1.0,
    **layout_kw,
):
    """Convert any hex input to axial and pixel coordinates.

    Single entry point used by all visualisation functions. Handles
    the two-stage pipeline: (1) convert input to axial ``(q, r)``,
    (2) project axial to screen ``(x, y)``.

    Args:
        c1: First coordinate. Meaning depends on ``coord_format``:
            axial q, zigzag x, pixel x, doubled col, etc.
        c2: Second coordinate. Meaning depends on ``coord_format``.
        coord_format: How to interpret ``c1, c2``. One of:
            ``"axial"``, ``"odd_r"``, ``"even_r"``, ``"odd_q"``,
            ``"even_q"``, ``"doublewidth"``, ``"doubleheight"``,
            ``"zigzag"`` (or ``"flywire"``, alias), ``"cube"``,
            ``"pixel"``.
        layout: Screen projection. One of:
            ``"pointy"``, ``"flat"``, ``"flywire"``, ``"pixel"``.
        size: Hexagon size (center-to-corner distance) for pixel
            projection.
        **layout_kw: Extra args forwarded to the layout projector
            (e.g. ``rotation_deg`` for flywire).

    Returns:
        ``(q, r, x, y)`` as ``np.ndarray``.  ``q`` and ``r`` are
        ``np.nan`` when ``coord_format="pixel"``.

    Raises:
        ValueError: If ``coord_format`` or ``layout`` is not recognised.

    Examples:
        Axial input, pointy-top layout:

        >>> q, r, x, y = resolve_hex(
        ...     np.array([0, 1]), np.array([0, 0]),
        ...     coord_format="axial", layout="pointy",
        ... )

        Zigzag input (FlyWire saved-page data):

        >>> q, r, x, y = resolve_hex(
        ...     zx, zy, coord_format="zigzag", layout="flat",
        ... )

    References:
        Red Blob Games — Hexagonal Grids:
        https://www.redblobgames.com/grids/hexagons/
    """
    if coord_format in ("zigzag", "flywire"):
        coord_format = "zigzag"

    if coord_format == "pixel":
        x = np.asarray(c1, dtype=float)
        y = np.asarray(c2, dtype=float)
        return np.full_like(x, np.nan), np.full_like(y, np.nan), x, y

    if coord_format == "zigzag":
        zx = np.asarray(c1, dtype=float)
        zy = np.asarray(c2, dtype=float)
        q, r = zigzag_to_axial(np.rint(zx).astype(int), np.rint(zy).astype(int))
        q = np.asarray(q, dtype=float)
        r = np.asarray(r, dtype=float)
        if layout == "pixel":
            return q, r, zx, zy
        x, y = zigzag_to_pixel(zx, zy, size=size)
        return q, r, np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    if coord_format == "cube":
        q, r = axial_from_cube(
            np.asarray(c1),
            np.asarray(c2),
            np.asarray(layout_kw.pop("c3", -c1 - c2)),
        )
        q, r = np.asarray(q, dtype=float), np.asarray(r, dtype=float)
    else:
        converter = _TO_AXIAL.get(coord_format)
        if converter is None:
            raise ValueError(f"Unknown coord_format: {coord_format}")
        q, r = converter(c1, c2)
        q, r = np.asarray(q, dtype=float), np.asarray(r, dtype=float)

    if layout == "pixel":
        return q, r, q, r

    projector = _TO_PIXEL.get(layout)
    if projector is None:
        raise ValueError(f"Unknown layout: {layout}")
    x, y = projector(q, r, size=size, **layout_kw)
    return q, r, np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def hex_symbol_for(layout: str) -> int:
    """Return the Plotly hex marker symbol for a given layout.

    Args:
        layout: One of ``"pointy"``, ``"flat"``, ``"flywire"``.

    Returns:
        Plotly marker symbol int: ``14`` (pointy-top hexagon) or
        ``15`` (flat-top hexagon).
    """
    return _HEX_SYMBOLS.get(layout, 14)
