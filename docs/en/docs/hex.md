# Hexagonal Grid Utilities

btorch provides a comprehensive hex grid toolkit for neuromorphic visual system modelling.
All algorithms follow [Red Blob Games](https://www.redblobgames.com/grids/hexagons/) conventions.

**Primary coordinate system:** axial `(q, r)` with `s = -q - r` implicit.

## Coordinate Systems

btorch supports multiple hex coordinate systems. All convert to/from axial internally.

| System | Module | Use Case |
|--------|--------|----------|
| **Axial** `(q, r)` | `transform` | Primary. All internal math. |
| **Cube** `(q, r, s)` | `transform` | Distance, line drawing (s = -q-r). |
| **Odd-r / Even-r** | `offset` | Row-based array storage. |
| **Odd-q / Even-q** | `offset` | Column-based array storage. |
| **Zigzag** `(x, y)` | `offset` | FlyWire connectome data. |
| **Double-width / Double-height** | `doubled` | Rectangular array maps. |
| **Pixel** `(x, y)` | `transform` | Screen-space plotting. |

### Converting Between Systems

Use the functional API directly:

```python
from btorch.utils.hex import (
    axial_to_odd_r, odd_r_to_axial,
    axial_to_zigzag, zigzag_to_axial,
    axial_to_doublewidth, doublewidth_to_axial,
    to_pixel, from_pixel,
)

# Axial -> odd-r offset -> back
q, r = np.array([1, -2]), np.array([0, 3])
col, row = axial_to_odd_r(q, r)
q_back, r_back = odd_r_to_axial(col, row)

# Axial -> pixel for plotting
x, y = to_pixel(q, r, size=1.0, orientation="pointy")
```

Or use the `HexCoords` OOP wrapper:

```python
from btorch.utils.hex import HexCoords

c = HexCoords.from_disk(radius=5)
col, row = c.to_odd_r()
c_back = HexCoords.from_odd_r(col, row)
assert c == c_back
```

### Using resolve_hex

The `resolve_hex` function is the single entry point for the
coord → pixel pipeline used by all visualisation functions:

```python
from btorch.utils.hex import resolve_hex

# Any coord format → (q, r, x, y)
q, r, x, y = resolve_hex(c1, c2, coord_format="axial", layout="pointy")

# FlyWire zigzag → screen coords
q, r, x, y = resolve_hex(zx, zy, coord_format="zigzag", layout="flat")
```

Supported `coord_format` values:
`"axial"`, `"odd_r"`, `"even_r"`, `"odd_q"`, `"even_q"`,
`"doublewidth"`, `"doubleheight"`, `"zigzag"` (or `"flywire"`, alias), `"pixel"`.

Supported `layout` values:
`"pointy"`, `"flat"`, `"flywire"`, `"pixel"`.

## Data Structures

### HexCoords

Struct-of-arrays for hex coordinates `(q, r)`:

```python
from btorch.utils.hex import HexCoords

# Create from various sources
c = HexCoords.from_disk(radius=5)
c = HexCoords.from_ring(radius=3)
c = HexCoords.from_spiral(radius=4)
c = HexCoords(q_array, r_array)

# Geometric operations
n = c.neighbors()       # 6 neighbors per hex
d = c.distance()        # distance from origin
rotated = c.rotate(1)   # rotate 60°
reflected = c.reflect("q")
```

### HexData

Coordinates with associated values:

```python
from btorch.utils.hex import HexData

coords = HexCoords.from_disk(radius=5)
values = np.random.randn(len(coords))
data = HexData(coords, values)

# Filtering, sorting, plotting
masked = data.mask(mask_array)
sorted_data = data.sort()
x, y, v = data.to_pixel()
```

### HexGrid

Pre-built circular grid with convenience methods:

```python
from btorch.utils.hex import HexGrid

grid = HexGrid(radius=10)
grid.values = np.random.randn(len(grid))

# Neighbor topology
vn = grid.valid_neighbors()  # list of neighbor index tuples
hull = grid.hull             # boundary ring

# Subset selection
circle = grid.circle(radius=3)
filled = grid.filled_circle(radius=3)
```

## Generating Coordinate Sets

```python
from btorch.utils.hex import disk, ring, spiral, rectangle

q, r = disk(radius=5)           # filled disk
q, r = ring(radius=3)           # boundary ring only
q, r = spiral(radius=4)         # spiral order (center, ring1, ...)
q, r = rectangle(8, 6)          # rectangular patch
```

## Distance and Neighbors

```python
from btorch.utils.hex import distance, radius, within_range, neighbors

d = distance(q1, r1, q2, r2)   # hex distance between point sets
r = radius(q, r)                # distance from origin
mask = within_range(q, r, 0, 0, radius=3)  # boolean mask

qn, rn = neighbors(q, r)       # 6 neighbors per hex, shape (6, n)
```

## Storage (Array Indexing)

Convert between axial coords and flat array indices for rectangular storage:

```python
from btorch.utils.hex import (
    axial_to_rect_index, rect_index_to_axial,
    axial_to_hex_index, hex_index_to_axial,
    align, permute, reflect_index,
)

# Axial -> rectangular array index
row, col = axial_to_rect_index(q, r, orientation="pointy")
q_back, r_back = rect_index_to_axial(row, col, orientation="pointy")

# Align data between grids of different sizes
aligned = align(q_tgt, r_tgt, q_src, r_src, values_src, fill=np.nan)

# Rotation/reflection permutation for data augmentation
perm = permute(radius=5, rotation=1)
rotated_values = values[perm]
```

## Visualization

### Static (Matplotlib)

```python
from btorch.visualisation.hex import scatter, flow, grid

scatter(q, r, values, coord_format="axial", cmap="viridis")
flow(q, r, dq, dr, coord_format="axial")
grid(radius=3, annotate=True)
```

### Interactive (Plotly)

```python
from btorch.visualisation.hex import heatmap, heatmap_from_index

heatmap(df, dataset, coord_format="axial", orientation="pointy")
heatmap_from_index(df, title="Connectome Activity")
```

### Animation

```python
from btorch.visualisation.hex.animate import HexScatter, HexFlow

anim = HexScatter(values, q, r, coord_format="axial")
anim.save("activity.mp4", writer="ffmpeg", fps=5)
```

### Hex Convolution

```python
from btorch.models.hex import Conv2dHex

conv = Conv2dHex(
    in_channels=16, out_channels=32,
    kernel_size=7, storage="rect_pointy",
)
out = conv(x)  # hexagonal receptive field mask
```

## References

- [Red Blob Games: Hexagonal Grids](https://www.redblobgames.com/grids/hexagons/)
- Code adapted from flyvis (MIT License) and Hexy (MIT License).
