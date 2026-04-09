# Hexagonal Grids

`btorch.utils.hex` provides a complete toolkit for hexagonal grid operations based on [Red Blob Games](https://www.redblobgames.com/grids/hexagons/). All algorithms are implemented as pure NumPy functions with optional Numba acceleration, plus object-oriented wrappers for convenience.

## Table of Contents

1. [Hex Orientations](#hex-orientations)
2. [Coordinate Systems](#coordinate-systems)
3. [Hexagonal Convolution](#hexagonal-convolution)
4. [Neighbors and Geometry](#neighbors-and-geometry)
5. [Coordinate Generation](#coordinate-generation)
6. [Distance and Lines](#distance-and-lines)
7. [Storage Layouts](#storage-layouts)
8. [Visualization](#visualization)
9. [Data Structures](#data-structures)
10. [Common Pitfalls](#common-pitfalls)

---

## Hex Orientations

There are exactly two ways to orient a regular hexagon on a 2D grid:

| Orientation | Description | Width | Height |
|-------------|-------------|-------|--------|
| **Pointy-top** | Point facing up (+y) | `√3 * size` | `2 * size` |
| **Flat-top** | Flat side facing up | `2 * size` | `√3 * size` |

The `size` parameter is the distance from the center to any corner (circumradius).

All btorch functions accept `orientation="pointy"` or `orientation="flat"`. The default is **pointy-top**, which is the standard for Red Blob Games and most hex grid applications.

**Critical rule**: Axial/cube coordinates are independent of orientation. `(q=1, r=0)` is the same hex regardless of whether you render it pointy or flat. Orientation only affects **pixel placement**.

```python
from btorch.utils.hex import to_pixel
import numpy as np

q, r = np.array([1]), np.array([0])

# Same axial coord, different pixel positions:
xp, yp = to_pixel(q, r, size=1.0, orientation="pointy")  # (1.732, 0.0)
xf, yf = to_pixel(q, r, size=1.0, orientation="flat")     # (1.500, 0.866)
```

---

## Coordinate Systems

### Axial / Cube (Primary)

**Axial** `(q, r)` is the primary coordinate system in btorch. The third cube coordinate `s` is implicit:

```
q + r + s = 0    →    s = -q - r
```

All algorithms internally use axial coordinates. Conversions:

```python
from btorch.utils.hex import cube_from_axial, axial_from_cube

q, r, s = cube_from_axial(np.array([1]), np.array([0]))     # (1, 0, -1)
q, r = axial_from_cube(np.array([1]), np.array([0]), np.array([-1]))  # (1, 0)
```

### Offset Coordinates

Offset coordinates add a row/column offset for storing hex grids in 2D arrays. There are four variants:

| Variant | Hex Orientation | Offset Applied To | Formula (to offset) |
|---------|-----------------|-------------------|---------------------|
| **odd-r** | Pointy-top | Odd rows | `col = q + (r - (r&1)) / 2` |
| **even-r** | Pointy-top | Even rows | `col = q + (r + (r&1)) / 2` |
| **odd-q** | Flat-top | Odd columns | `row = r + (q - (q&1)) / 2` |
| **even-q** | Flat-top | Even columns | `row = r + (q + (q&1)) / 2` |

```python
from btorch.utils.hex import axial_to_odd_r, odd_r_to_axial

q = np.array([0, 1, 2, 0, 1, 2])
r = np.array([0, 0, 0, 1, 1, 1])

col, row = axial_to_odd_r(q, r)
# col: [0, 1, 2, 0, 1, 2]
# row: [0, 0, 0, 1, 1, 1]

q_back, r_back = odd_r_to_axial(col, row)  # roundtrip exact
```

**Note**: For neighbor lookups in offset coordinates, the recommended approach is:
1. Convert offset → axial
2. Use `neighbors()` in axial space
3. Convert axial → offset

### Doubled Coordinates

Doubled coordinates make rectangular maps easier by skipping every other row/column:

| Variant | Hex Orientation | Formula |
|---------|-----------------|---------|
| **Double-width** | Pointy-top | `col = 2*q + r`, `row = r` |
| **Double-height** | Flat-top | `col = q`, `row = 2*r + q` |

```python
from btorch.utils.hex import axial_to_doublewidth, doublewidth_to_axial

q = np.array([0, 1, 0, -1, 0, 1])
r = np.array([0, 0, 1, 0, -1, -1])

col, row = axial_to_doublewidth(q, r)
# col: [0, 2, 1, -2, -1, 1]
# row: [0, 0, 1, 0, -1, -1]

q_back, r_back = doublewidth_to_axial(col, row)  # roundtrip exact
```

Distance can be computed directly in doubled coords without converting:

```python
from btorch.utils.hex import doublewidth_distance, doubleheight_distance

doublewidth_distance(0, 0, 4, 0)   # 2
doubleheight_distance(0, 0, 0, 4)  # 2
```

### Zigzag Offset Coordinates

Zigzag coordinates are a display-friendly offset layout for visual columns:

```
x = floor((r - q) / 2),  y = q + r
```

This is **not** a direct pixel coordinate system. When visualizing, convert back to axial and then use `to_pixel(..., orientation="pointy")`.

```python
from btorch.utils.hex import axial_to_zigzag, zigzag_to_axial
import numpy as np

q = np.array([-17])
r = np.array([-8])

# Forward: zigzag display coordinates
x, y = axial_to_zigzag(q, r)  # (4, -25)

# Reverse: exact integer roundtrip
q_back, r_back = zigzag_to_axial(x, y)  # (-17, -8)
```

**FlyWire coordinates**: FlyWire's visual column coordinates `(x, y)` are zigzag coordinates. FlyWire's `(p, q)` are axial `(q, r)`. Use `axial_to_zigzag` / `zigzag_to_axial` when working with FlyWire data.

**Important**: Zigzag coordinates are purely for display/layout. They create the classic hex zigzag where adjacent cells alternate between two columns. The reverse conversion is exact because the rounding rule is consistent.

### Flyvis Coordinate System

Btorch's hex utilities are adapted from the **flyvis** codebase. Flyvis uses its own conventions which differ slightly from standard Red Blob Games:

| Property | Flyvis | Btorch (RBG) |
|----------|--------|--------------|
| Coordinate names | `u`, `v` | `q`, `r` |
| Default pixel orientation | `mode="default"` (custom) | `orientation="pointy"` |
| `Hexal` / `HexArray` / `HexLattice` | OOP classes | `HexCoords` / `HexData` / `HexGrid` |
| `extent` | Radius of hex disk | `radius` |
| Disk size formula | `1 + 3*extent*(extent+1)` | Same |

**Flyvis default pixel transform** (`mode="default"`):
```python
# Flyvis hex_to_pixel with mode="default"
x = 3/2 * v
y = -sqrt(3) * (u + v/2)
```

This is **not** the same as either pointy-top or flat-top. It's a rotated/skewed convention used historically in flyvis. **Btorch uses standard Red Blob Games pointy-top by default**, which is more intuitive and widely documented.

When porting code from flyvis to btorch:
- Replace `u, v` with `q, r`
- Replace `hex_to_pixel(u, v, mode="default")` with `to_pixel(q, r, orientation="pointy")`
- Replace `HexLattice(extent=...)` with `HexGrid(radius=...)`
- Replace `HexArray` with `HexCoords` or `HexData`

---

## Hexagonal Convolution

`btorch.models.hex` provides `Conv2dHex` (and alias `Conv2dHexSpace`) for hexagonal convolutions. This is adapted from `flyvis.task.decoder.Conv2dHexSpace`.

### How it works

Standard PyTorch `Conv2d` operates on rectangular kernels. `Conv2dHex` applies a **hexagonal mask** to the kernel weights, constraining the receptive field to a hexagon shape. The underlying convolution is still a standard 2D convolution on a Cartesian feature map—only the **active weights** are hexagonally arranged.

```
Rectangular kernel (7x7)          Hexagonal mask (radius=3)
+--+--+--+--+--+--+--+           +--+--+--+--+--+--+--+
|  |  |  |  |  |  |  |           |  |  |  |  |  |  |  |
+--+--+--+--+--+--+--+           +--+--+--+--+--+--+--+
|  |  |  |  |  |  |  |           |  |  |  #  #  |  |  |
+--+--+--+--+--+--+--+     →     +--+--+--+--+--+--+--+
|  |  |  |  |  |  |  |           |  |  #  #  #  #  |  |
+--+--+--+--+--+--+--+           +--+--+--+--+--+--+--+
|  |  |  |  |  |  |  |           |  #  #  #  #  #  #  |
+--+--+--+--+--+--+--+           +--+--+--+--+--+--+--+
...                               ...
```

The mask is generated using `disk(kernel_size // 2)` and shifted to start at `(0, 0)`. For `kernel_size=7`, the mask has 37 active positions (a disk of radius 3).

### Usage

```python
from btorch.models.hex import Conv2dHex

# kernel_size must be odd
conv = Conv2dHex(in_channels=16, out_channels=32, kernel_size=7)
x = torch.randn(1, 16, 64, 64)
out = conv(x)  # Output has hexagonal receptive fields
```

### Key properties

- `kernel_size` **must be odd**. Even sizes raise `ValueError`.
- The mask is applied on every forward pass via `filter_to_hex()`, zeroing out non-hexagonal weights.
- Internally it's still a `Conv2d` on a square tensor; the hex constraint only affects which kernel weights are trainable.

### Comparison with standard Conv2d

```python
import torch

# Hex convolution
conv_hex = Conv2dHex(1, 1, kernel_size=7, padding=3)

# Standard rectangular convolution
conv_rect = torch.nn.Conv2d(1, 1, kernel_size=7, padding=3)

# Same input
x = torch.zeros(1, 1, 32, 32)
x[0, 0, 16, 16] = 1.0

out_hex = conv_hex(x)
out_rect = conv_rect(x)

# out_hex has hexagonal response pattern
# out_rect has rectangular response pattern
```

### Center-surround receptive field example

```python
import numpy as np
import torch
from btorch.models.hex import Conv2dHex
from btorch.utils.hex import disk

k = 7
conv = Conv2dHex(1, 1, kernel_size=k)

q, r = disk(k // 2)
dist = np.sqrt(q**2 + r**2)

with torch.no_grad():
    weights = torch.zeros_like(conv.weight)
    for qi, ri, di in zip(q, r, dist):
        qn = qi - q.min()
        rn = ri - r.min()
        if di < 1.5:
            weights[0, 0, qn, rn] = 1.0   # center
        elif di < 3.0:
            weights[0, 0, qn, rn] = -0.3  # surround
    conv.weight.copy_(weights)
```

---

## Neighbors and Geometry

### Cardinal Neighbors

There are 6 cardinal directions. The `DIRECTIONS` array uses pointy-top semantics:

| Index | Direction | `(dq, dr)` |
|-------|-----------|------------|
| 0 | NE | `(1, 0)` |
| 1 | E | `(1, -1)` |
| 2 | SE | `(0, -1)` |
| 3 | SW | `(-1, 0)` |
| 4 | W | `(-1, 1)` |
| 5 | NW | `(0, 1)` |

```python
from btorch.utils.hex import neighbor, neighbors

# Single neighbor
nq, nr = neighbor(np.array([0]), np.array([0]), direction=0)  # NE → (1, 0)

# All 6 neighbors (shape: (6, n_hexes))
qn, rn = neighbors(np.array([0, 1]), np.array([0, 0]))
```

### Diagonal Neighbors

There are also 6 diagonal directions (between cardinals):

```python
from btorch.utils.hex import diagonal_neighbor, diagonal_neighbors, all_neighbors

qn, rn = diagonal_neighbors(np.array([0]), np.array([0]))
# Returns 6 diagonals: [(2,-1), (1,-2), (-1,-1), (-2,1), (-1,2), (1,1)]

# All 12 neighbors (6 cardinal + 6 diagonal)
qn, rn = all_neighbors(np.array([0]), np.array([0]))  # shape (12, n_hexes)
```

---

## Coordinate Generation

```python
from btorch.utils.hex import disk, ring, spiral, rectangle

# All hexes within radius 2 of center (19 hexes)
q, r = disk(radius=2)

# Hexes exactly at radius 2 (12 hexes)
q, r = ring(radius=2)

# Spiral order: center, ring1, ring2, ...
q, r = spiral(radius=2)
# Order: [(0,0), (0,-1), (1,-1), (1,0), (0,1), (-1,1), (-1,0), (0,-2), ...]

# Rectangular region (rhombus in axial coords)
q, r = rectangle(width=3, height=2, orientation="pointy")
```

Disk count formula: `1 + 3*r*(r+1)`

```python
from btorch.utils.hex import disk_count, disk_radius

disk_count(3)   # 37
disk_radius(37) # 3
```

---

## Distance and Lines

### Distance

Hex distance uses the cube metric: `max(|dq|, |dr|, |ds|)`

```python
from btorch.utils.hex import distance, radius, within_range, mask

d = distance(np.array([0]), np.array([0]), np.array([3]), np.array([3]))  # 6
r = radius(np.array([3]), np.array([3]))  # 6

# Boolean masks
within_range(q, r, center_q=0, center_r=0, n=2)
mask(q, r, max_radius=2)
```

### Line Drawing

Line drawing uses cube linear interpolation + rounding:

```python
from btorch.utils.hex import line, line_n

# Standard line: N = distance + 1 points
q, r = line(np.array([0]), np.array([0]), np.array([3]), np.array([0]))
# [(0,0), (1,0), (2,0), (3,0)]

# Line with exactly n+1 points
q, r = line_n(np.array([0]), np.array([0]), np.array([3]), np.array([3]), n=6)
```

### Range Operations

```python
from btorch.utils.hex import range_intersection, range_union, ranges_intersect

# Hexes within ALL given ranges
q, r = range_intersection([(0, 0), (3, 0)], [2, 2])

# Hexes within ANY given range (deduplicated)
q, r = range_union([(0, 0), (5, 0)], [2, 2])

# Simple overlap test
ranges_intersect((0, 0), 2, (3, 0), 2)  # True
```

---

## Storage Layouts

Map storage functions convert axial coordinates to array indices for different map shapes:

### Rectangular Maps

```python
from btorch.utils.hex import axial_to_rect_index, rect_index_to_axial

q = np.array([0, 1, 2, -1])
r = np.array([0, 0, 1, 0])

# Pointy-top: array[r][q + floor(r/2)]
row, col = axial_to_rect_index(q, r, orientation="pointy")
# row: [0, 0, 1, 0], col: [0, 1, 2, -1]

q_back, r_back = rect_index_to_axial(row, col, orientation="pointy")
```

### Hexagon-shaped Maps

```python
from btorch.utils.hex import axial_to_hex_index, hex_index_to_axial

q = np.array([0, 1, -1, 0])
r = np.array([0, 0, 0, 1])

# array[r + N][q - max(0, N-r) + N]
row, col = axial_to_hex_index(q, r, radius=1)
# row: [1, 1, 1, 2], col: [1, 2, 0, 1]
```

### Triangle Maps

```python
from btorch.utils.hex import axial_to_triangle_index, triangle_index_to_axial

# Down-pointing: array[r][q]
row, col = axial_to_triangle_index(q, r, size=3, pointing="down")

# Up-pointing: array[r][q - N+1+r]
row, col = axial_to_triangle_index(q, r, size=3, pointing="up")
```

### Rhombus Maps

```python
from btorch.utils.hex import axial_to_rhombus_index, rhombus_index_to_axial

# Direct mapping: array[r][q]
row, col = axial_to_rhombus_index(q, r)
```

---

## Storage and Indexing

### Alignment

Align values from one coordinate set to another:

```python
from btorch.utils.hex import align

q_src = np.array([0, 1, -1])
r_src = np.array([0, 0, 0])
vals = np.array([10.0, 20.0, 30.0])

q_tgt = np.array([1, 0, -1, 2])
r_tgt = np.array([0, 0, 0, 0])

aligned = align(q_tgt, r_tgt, q_src, r_src, vals, fill=-1.0)
# [20.0, 10.0, 30.0, -1.0]
```

### Permutations

For rotating or reflecting spiral-ordered data:

```python
from btorch.utils.hex import permute, reflect_index

# Rotation permutation (clockwise 60°)
perm = permute(radius=2, n_rot=1)

# Reflection permutations
perm_q = reflect_index(radius=2, axis="q")
perm_r = reflect_index(radius=2, axis="r")
perm_s = reflect_index(radius=2, axis="s")
```

---

## Visualization

### Coordinate Format Rules

The visualization layer (`btorch.visualisation.hex`) supports three `coord_format` values:

| Format | Meaning | How it's rendered |
|--------|---------|-------------------|
| `"axial"` | `(q, r)` axial coords | Converted to pixels via `to_pixel(..., orientation)` |
| `"zigzag"` | Zigzag `(x, y)` display coords | Plotted directly as pixel positions |
| `"pixel"` | Direct `(x, y)` pixel coords | Plotted as-is |

```python
from btorch.visualisation.hex.static import scatter

# Axial with pointy-top (default)
scatter(q, r, values, coord_format="axial", orientation="pointy")

# Axial with flat-top
scatter(q, r, values, coord_format="axial", orientation="flat")

# Zigzag (creates clean vertical columns with alternating x)
scatter(q, r, values, coord_format="zigzag")
```

### Receptive Fields

```python
from btorch.visualisation.hex.receptive_field import kernel, strf

# Single receptive field
fig, ax = kernel(q, r, values, coord_format="axial")

# Spatio-temporal receptive field
fig, axes = strf(time, rf, q, r, coord_format="axial")
```

---

## Data Structures

### HexCoords

Coordinate-only struct-of-arrays:

```python
from btorch.utils.hex import HexCoords

coords = HexCoords.from_disk(radius=3)
px, py = coords.to_pixel(size=1.0, orientation="pointy")
neighbors = coords.neighbors()  # HexCoords with 6*n_hexes entries
rotated = coords.rotate(n=1)
```

### HexData

Coordinates + values:

```python
from btorch.utils.hex import HexData

data = HexData(coords, np.random.randn(len(coords)))
filtered = data.mask(data.values > 0)
fig = data.plot(coord_format="axial")
```

### HexGrid

Regular disk-shaped grid:

```python
from btorch.utils.hex import HexGrid

grid = HexGrid(radius=5)
grid.values = np.random.randn(len(grid))

circle = grid.circle(radius=3)      # outer ring
filled = grid.filled_circle(radius=3)  # disk
```

---

## Common Pitfalls

### 1. Zigzag is for Display Only

Zigzag coordinates `(x, y)` are designed for screen layout, not geometric computation. Always convert back to axial for distance, rotation, or neighbor calculations:

```python
q, r = zigzag_to_axial(x, y)
dist = distance(q, r, q2, r2)
```

### 2. Offset Coordinates are for Storage, Not Math

Offset coordinates make 2D array storage convenient but break geometric operations. Always convert to axial for distance, rotation, or neighbor calculations.

### 3. Orientation Only Affects Pixels

`(q=1, r=0)` is the same hex in both pointy and flat orientations. The orientation parameter only changes where it appears on screen.

### 4. Double-Width vs Double-Height

- **Double-width** → use with **pointy-top** hexes
- **Double-height** → use with **flat-top** hexes

Using the wrong pairing will produce non-contiguous grids.

### 5. Spiral Order is Required for Permutations

`permute()` and `reflect_index()` assume spiral ordering. They will produce incorrect results if used with arbitrary coordinate orderings.

---

## Reference

All formulas follow [Red Blob Games: Hexagonal Grids](https://www.redblobgames.com/grids/hexagons/).
