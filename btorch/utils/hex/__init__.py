"""Hexagonal grid utilities (Red Blob Games algorithms).

https://www.redblobgames.com/grids/hexagons/
Primary: axial (q, r) with s = -q-r implicit.

Provides both functional API (for performance) and object-oriented
struct-of-arrays types (for convenience).

Code adapted from flyvis (MIT License) and Hexy (MIT License).
"""

# Coords
from .coords import disk, disk_count, disk_radius, rectangle, ring, spiral

# Data structures (struct-of-arrays)
from .data import HexCoords, HexData, HexGrid, HexLattice

# Distance
from .distance import distance, mask, radius, within_range

# Doubled coordinates
from .doubled import (
    axial_to_doubleheight,
    axial_to_doublewidth,
    doubleheight_distance,
    doubleheight_to_axial,
    doubleheight_to_pixel,
    doublewidth_distance,
    doublewidth_to_axial,
    doublewidth_to_pixel,
    pixel_to_doubleheight,
    pixel_to_doublewidth,
)

# Line drawing
from .line import line, line_n

# Neighbor
from .neighbor import (
    DIAGONALS,
    DIRECTIONS,
    all_neighbors,
    diagonal_neighbor,
    diagonal_neighbors,
    neighbor,
    neighbors,
)

# Offset coordinates
from .offset import (
    axial_to_even_q,
    axial_to_even_r,
    axial_to_odd_q,
    axial_to_odd_r,
    axial_to_zigzag,
    even_q_to_axial,
    even_r_to_axial,
    odd_q_to_axial,
    odd_r_to_axial,
    zigzag_to_axial,
)

# Range operations
from .range import range_intersection, range_union, ranges_intersect

# Storage
from .storage import (
    align,
    axial_to_hex_index,
    axial_to_rect_index,
    axial_to_rhombus_index,
    axial_to_triangle_index,
    hex_index_to_axial,
    permute,
    rect_index_to_axial,
    reflect_index,
    rhombus_index_to_axial,
    triangle_index_to_axial,
)

# Transform
from .transform import (
    Orientation,
    axial_from_cube,
    cube_from_axial,
    from_pixel,
    reflect,
    rotate,
    round_axial,
    to_pixel,
)


__all__ = [
    # Coords
    "ring",
    "disk",
    "spiral",
    "rectangle",
    "disk_count",
    "disk_radius",
    # Transform
    "cube_from_axial",
    "axial_from_cube",
    "to_pixel",
    "from_pixel",
    "round_axial",
    "rotate",
    "reflect",
    "Orientation",
    # Distance
    "distance",
    "radius",
    "within_range",
    "mask",
    # Neighbor
    "DIRECTIONS",
    "DIAGONALS",
    "neighbor",
    "neighbors",
    "diagonal_neighbor",
    "diagonal_neighbors",
    "all_neighbors",
    # Storage
    "align",
    "permute",
    "reflect_index",
    "axial_to_rect_index",
    "rect_index_to_axial",
    "axial_to_hex_index",
    "hex_index_to_axial",
    "axial_to_triangle_index",
    "triangle_index_to_axial",
    "axial_to_rhombus_index",
    "rhombus_index_to_axial",
    # Line
    "line",
    "line_n",
    # Offset
    "axial_to_odd_r",
    "odd_r_to_axial",
    "axial_to_even_r",
    "even_r_to_axial",
    "axial_to_odd_q",
    "odd_q_to_axial",
    "axial_to_even_q",
    "even_q_to_axial",
    "axial_to_zigzag",
    "zigzag_to_axial",
    # Doubled
    "axial_to_doublewidth",
    "doublewidth_to_axial",
    "axial_to_doubleheight",
    "doubleheight_to_axial",
    "doublewidth_distance",
    "doubleheight_distance",
    "doublewidth_to_pixel",
    "doubleheight_to_pixel",
    "pixel_to_doublewidth",
    "pixel_to_doubleheight",
    # Range
    "range_intersection",
    "range_union",
    "ranges_intersect",
    # Data structures
    "HexCoords",
    "HexData",
    "HexGrid",
    "HexLattice",
]
