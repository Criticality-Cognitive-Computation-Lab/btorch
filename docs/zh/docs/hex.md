# 六边形网格工具

btorch 提供了一套完整的六边形网格工具包，用于神经形态视觉系统建模。
所有算法遵循 [Red Blob Games](https://www.redblobgames.com/grids/hexagons/) 的约定。

**主坐标系：** 轴向坐标 `(q, r)`，其中 `s = -q - r` 为隐式值。

## 坐标系

btorch 支持多种六边形坐标系。所有坐标均可通过内部函数与轴向坐标相互转换。

| 坐标系 | 模块 | 用途 |
|--------|------|------|
| **轴向坐标** `(q, r)` | `transform` | 主坐标系，所有内部计算均使用此格式。 |
| **立方坐标** `(q, r, s)` | `transform` | 距离计算、直线绘制（s = -q-r）。 |
| **Odd-r / Even-r** | `offset` | 基于行的数组存储。 |
| **Odd-q / Even-q** | `offset` | 基于列的数组存储。 |
| **Zigzag** `(x, y)` | `offset` | FlyWire 连接组数据。 |
| **Double-width / Double-height** | `doubled` | 矩形数组映射。 |
| **像素坐标** `(x, y)` | `transform` | 屏幕绘图。 |

### 坐标系转换

使用函数式 API 直接转换：

```python
from btorch.utils.hex import (
    axial_to_odd_r, odd_r_to_axial,
    axial_to_zigzag, zigzag_to_axial,
    axial_to_doublewidth, doublewidth_to_axial,
    to_pixel, from_pixel,
)

# 轴向 → odd-r 偏移 → 反向转换
q, r = np.array([1, -2]), np.array([0, 3])
col, row = axial_to_odd_r(q, r)
q_back, r_back = odd_r_to_axial(col, row)

# 轴向 → 像素坐标（用于绘图）
x, y = to_pixel(q, r, size=1.0, orientation="pointy")
```

或使用 `HexCoords` 面向对象封装：

```python
from btorch.utils.hex import HexCoords

c = HexCoords.from_disk(radius=5)
col, row = c.to_odd_r()
c_back = HexCoords.from_odd_r(col, row)
assert c == c_back
```

### 使用 resolve_hex

`resolve_hex` 是所有可视化函数共享的坐标 → 像素转换入口：

```python
from btorch.utils.hex import resolve_hex

# 任意坐标格式 → (q, r, x, y)
q, r, x, y = resolve_hex(c1, c2, coord_format="axial", layout="pointy")

# FlyWire zigzag → 屏幕坐标
q, r, x, y = resolve_hex(zx, zy, coord_format="zigzag", layout="flat")
```

支持的 `coord_format` 值：
`"axial"`, `"odd_r"`, `"even_r"`, `"odd_q"`, `"even_q"`,
`"doublewidth"`, `"doubleheight"`, `"zigzag"`（或 `"flywire"`，别名）, `"pixel"`。

支持的 `layout` 值：
`"pointy"`, `"flat"`, `"flywire"`, `"pixel"`。

## 数据结构

### HexCoords

基于结构体数组（struct-of-arrays）的六边形坐标 `(q, r)`：

```python
from btorch.utils.hex import HexCoords

# 从各种来源创建
c = HexCoords.from_disk(radius=5)
c = HexCoords.from_ring(radius=3)
c = HexCoords.from_spiral(radius=4)
c = HexCoords(q_array, r_array)

# 几何运算
n = c.neighbors()       # 每个六边形的 6 个邻居
d = c.distance()        # 到原点的距离
rotated = c.rotate(1)   # 旋转 60°
reflected = c.reflect("q")
```

### HexData

带关联值的坐标数据：

```python
from btorch.utils.hex import HexData

coords = HexCoords.from_disk(radius=5)
values = np.random.randn(len(coords))
data = HexData(coords, values)

# 过滤、排序、绘图
masked = data.mask(mask_array)
sorted_data = data.sort()
x, y, v = data.to_pixel()
```

### HexGrid

预构建的圆形网格，提供便捷方法：

```python
from btorch.utils.hex import HexGrid

grid = HexGrid(radius=10)
grid.values = np.random.randn(len(grid))

# 邻居拓扑
vn = grid.valid_neighbors()  # 邻居索引元组列表
hull = grid.hull             # 边界环

# 子集选择
circle = grid.circle(radius=3)
filled = grid.filled_circle(radius=3)
```

## 生成坐标集

```python
from btorch.utils.hex import disk, ring, spiral, rectangle

q, r = disk(radius=5)           # 填充圆盘
q, r = ring(radius=3)           # 仅边界环
q, r = spiral(radius=4)         # 螺旋顺序（中心，第1环，第2环...）
q, r = rectangle(8, 6)          # 矩形区域
```

## 距离与邻居

```python
from btorch.utils.hex import distance, radius, within_range, neighbors

d = distance(q1, r1, q2, r2)   # 点集之间的六边形距离
r = radius(q, r)                # 到原点的距离
mask = within_range(q, r, 0, 0, radius=3)  # 布尔掩码

qn, rn = neighbors(q, r)       # 每个六边形的 6 个邻居，形状 (6, n)
```

## 存储（数组索引）

在轴向坐标和矩形存储的平坦数组索引之间转换：

```python
from btorch.utils.hex import (
    axial_to_rect_index, rect_index_to_axial,
    axial_to_hex_index, hex_index_to_axial,
    align, permute, reflect_index,
)

# 轴向 → 矩形数组索引
row, col = axial_to_rect_index(q, r, orientation="pointy")
q_back, r_back = rect_index_to_axial(row, col, orientation="pointy")

# 在不同大小的网格之间对齐数据
aligned = align(q_tgt, r_tgt, q_src, r_src, values_src, fill=np.nan)

# 旋转/反射排列用于数据增强
perm = permute(radius=5, rotation=1)
rotated_values = values[perm]
```

## 可视化

### 静态图（Matplotlib）

```python
from btorch.visualisation.hex import scatter, flow, grid

scatter(q, r, values, coord_format="axial", cmap="viridis")
flow(q, r, dq, dr, coord_format="axial")
grid(radius=3, annotate=True)
```

### 交互式图（Plotly）

```python
from btorch.visualisation.hex import heatmap, heatmap_from_index

heatmap(df, dataset, coord_format="axial", orientation="pointy")
heatmap_from_index(df, title="连接组活动")
```

### 动画

```python
from btorch.visualisation.hex.animate import HexScatter, HexFlow

anim = HexScatter(values, q, r, coord_format="axial")
anim.save("activity.mp4", writer="ffmpeg", fps=5)
```

### 六边形卷积

```python
from btorch.models.hex import Conv2dHex

conv = Conv2dHex(
    in_channels=16, out_channels=32,
    kernel_size=7, storage="rect_pointy",
)
out = conv(x)  # 六边形感受野掩码
```

## 参考资料

- [Red Blob Games — 六边形网格](https://www.redblobgames.com/grids/hexagons/)
- 代码改编自 flyvis（MIT 许可证）和 Hexy（MIT 许可证）。
