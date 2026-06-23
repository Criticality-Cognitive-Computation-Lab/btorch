# Sparse Architecture Design

## Status

This document describes the target architecture for `btorch.sparse`. It is a
design specification, not a description of every currently implemented API.

## Goals

- Provide a persistent, non-ephemeral sparse matrix type that carries its own
  operations, format cache, and connectivity metadata.
- Register sparse types as PyTree nodes so they compose with `torch.compile`,
  `torch.func.grad`, and `torch.func.vmap`.
- Let concrete format subtypes (`CSR`, `BSR`, `COO`, `ELL`, …) enforce
  structure-specific invariants while sharing a common operation interface.
- Bridge sparse matrices into PyTorch's training infrastructure through a thin
  `nn.Module` wrapper analogous to `TensorDictParams`.
- Allow backends to dispatch directly on the concrete subtype without a prior
  conversion to COO.
- Represent fixed fan-in, fan-out, and other connectivity properties as
  validated fields, not unchecked hints.
- Represent heterogeneous synapse attributes in a format-independent,
  domain-aligned container.
- Make format selection explicit and observable; provide a format-agnostic
  class as a convenience, not a requirement.
- Retain a complete native fallback without routing normal execution through
  COO.

## Non-goals

- Making `SparseTensorBase` or its subtypes inherit from `nn.Module` directly.
- A universal constructor that infers both format and connectivity semantics.
- One class for every combination of format, property, and attribute.
- Automatic conversion to COO before backend dispatch.
- Hiding expensive format conversion inside an apparently cheap property or
  method call.
- Dynamic connectivity resampling during ordinary forward calls.
- Requiring every format to expose a flat `values` tensor.
- Making format selection invisible to users who need to reason about kernel
  compatibility or memory layout.

## Architecture Overview

```text
SparseTensorBase        (abstract base — interface only, no storage)
├── SparseTensor        (concrete — auto-selects format, wraps inner)
├── COO                 (concrete — enforces COO storage)
├── CSR                 (concrete — enforces CSR storage)
├── CSC                 (concrete — enforces CSC storage)
├── ELL                 (concrete — enforces ELL storage)
├── BSR                 (concrete — enforces BSR storage)
├── SemiStructured24    (concrete — enforces packed 2:4 storage)
└── ProceduralSparse    (concrete — no stored edges)

SparseParam  (SparseTensorBase subclass AND nn.Module)
├── exposes trainable tensors as nn.Parameters
├── exposes structural tensors as nn.Buffers
├── delegates all ops to the wrapped SparseTensorBase instance
└── analogous to TensorDictParams

SparseLinear  (thin nn.Module convenience layer)
├── holds a SparseParam submodule
├── owns bias
└── implements forward()

Backend dispatch  (internal)
└── dispatches on concrete format subtype (CSR, BSR, ELL, …), never on
    SparseTensor or SparseParam
```

`SparseTensorBase` and its subtypes are plain Python classes, not `nn.Module`
subclasses. `SparseParam` is the bridge that makes a sparse matrix visible to
the optimizer, `state_dict`, and device movement, following the same pattern as
`TensorDictParams` in the TensorDict library.

---

## SparseTensorBase: Abstract Interface

`SparseTensorBase` is the common interface for all sparse matrix types. It is a
pure abstract class — it owns no tensors and defines no storage layout. Every
concrete subtype provides its own storage and registers itself as a PyTree node
independently.

```python
class SparseTensorBase:
    shape: tuple[int, int]
    properties: SparseProperties
    attributes: EdgeAttributes
    constraint: "Constraint | None"          # Python attribute, not in PyTree
    parameterization: "Parameterization | None"  # Python attribute, not in PyTree
    _cache: dict                             # mutable, not part of the PyTree

    # --- matrix operations (abstract) ---
    def mm(self, x: torch.Tensor) -> torch.Tensor: ...
    def event_mm(self, events: EventRepresentation) -> torch.Tensor: ...

    # --- format conversion (abstract) ---
    def to_coo(self) -> "COO": ...
    def to_csr(self) -> "CSR": ...
    def to_csc(self) -> "CSC": ...
    def to_ell(self) -> "ELL": ...
    def to_bsr(self, block_shape: tuple[int, int]) -> "BSR": ...

    # --- prepared format access ---
    def prepare_as(self, format: type, **kwargs) -> None: ...
    def as_bsr(self) -> "BSR": ...   # requires prior prepare_as(BSR, ...)

    # --- inspection (abstract) ---
    def logical_edges(self) -> "LogicalEdges": ...
    def nnz(self) -> int: ...

    # --- value access ---
    def effective_values(self) -> torch.Tensor:
        """Return the live weight values for forward computation.
        When a Parameterization is set, derives values from compact parameters.
        When no Parameterization is set, returns self.values directly."""
        if self.parameterization is not None:
            return self.parameterization.effective_values()
        return self.values

    # --- constraint and parameterization binding ---
    def with_constraint(self, c: "Constraint") -> "SparseTensorBase": ...
    def with_parameterization(self, p: "Parameterization") -> "SparseTensorBase": ...
```

`constraint` and `parameterization` are plain Python attributes on every
`SparseTensorBase` instance. Neither is part of the PyTree, so `tree_map` and
`tree_unflatten` never touch them. They represent two complementary strategies
for enforcing invariants on trainable values — see the Constraints and
Parameterization sections for details.

Backends call `W.effective_values()` rather than `W.values` directly. This
ensures they see the correct live weights whether or not a parameterization is
active.

`_cache` is a plain mutable Python dict. It is not part of the PyTree:
`tree_unflatten` always produces a new instance with `_cache = {}`. Any method
that mutates trainable values must clear the cache:

```python
@_erase_cache
def _replace_values(self, new_values: torch.Tensor) -> "SparseTensorBase": ...
```

`_new_unsafe` skips validation and is used exclusively for PyTree unflatten and
internal reconstruction. All public constructors run complete validation.

### Property propagation

Operations that produce a new `SparseTensorBase` follow these rules for
`SparseProperties`:

- Operations that modify only values (projection, dtype cast) may forward
  structural properties unchanged.
- Operations that reorder edges (sorting, permutation) must revalidate or drop
  properties that depend on order.
- Operations that change topology must revalidate all properties.
- `_new_unsafe` does not validate properties. The calling operation is
  responsible for passing correct properties.

---

## SparseTensor: Auto-selecting Concrete Type

`SparseTensor` is the default concrete type for users who do not want to commit
to a specific storage format. It wraps an inner concrete subtype selected by
heuristics at construction time and delegates all operations to it.

```python
class SparseTensor(SparseTensorBase):
    _inner: CSR | ELL | BSR | COO | ...

    @classmethod
    def from_edges(
        cls,
        sources: torch.Tensor,
        destinations: torch.Tensor,
        values: torch.Tensor,
        shape: tuple[int, int],
        properties: SparseProperties | None = None,
        attributes: EdgeAttributes | None = None,
    ) -> "SparseTensor":
        inner = _select_format(sources, destinations, values, shape, properties)
        return cls._new_unsafe(inner)

    @property
    def selected_format(self) -> type:
        return type(self._inner)

    def mm(self, x):         return self._inner.mm(x)
    def to_csr(self):        return self._inner.to_csr()
    def prepare_as(self, fmt, **kw): self._inner.prepare_as(fmt, **kw)
    def as_bsr(self):        return self._inner.as_bsr()
    def logical_edges(self): return self._inner.logical_edges()
    def nnz(self):           return self._inner.nnz()
    # ... all other ops delegate to self._inner
```

### Why a class rather than a factory returning a concrete type

With a simple factory that returns `CSR` or `ELL` depending on heuristics,
`type(W)` would vary with the heuristic outcome. Code cannot distinguish
"format was auto-selected" from "user enforced CSR." `SparseTensor` as a class
makes the distinction type-visible:

```python
W1 = SparseTensor.from_edges(...)     # format auto-selected
W2 = CSR(row_ptr, col_indices, ...)   # format enforced by caller

isinstance(W1, SparseTensor)          # True — auto-selected
isinstance(W2, CSR)                   # True — enforced
isinstance(W1, SparseTensorBase)      # True
isinstance(W2, SparseTensorBase)      # True
```

Backend functions that require a guaranteed layout can reject `SparseTensor`
explicitly at the type level:

```python
def triton_bsr_mm(W: BSR, x: torch.Tensor) -> torch.Tensor:
    # SparseTensor does not satisfy this annotation
    ...
```

### Format selection heuristics

| Condition | Selected inner format |
|---|---|
| `max_fan_out` declared and validated | `ELL` |
| `max_fan_in` declared and validated | `CSC` |
| `block_shape` provided in kwargs | `BSR` |
| General case | `CSR` |

The selected format is observable via `W.selected_format`. It is fixed at
construction and does not change. Changing format requires explicit conversion.

### PyTree registration

`SparseTensor`'s PyTree flatten places the inner type in the static context so
that different inner formats produce different `TreeSpec` values:

```python
def _sparsetensor_flatten(W: SparseTensor):
    inner_leaves, inner_context = pytree_flatten(W._inner)
    leaves  = inner_leaves
    context = (type(W._inner), inner_context, W.constraint)
    return leaves, context

def _sparsetensor_unflatten(context, leaves) -> SparseTensor:
    inner_type, inner_context, constraint = context
    inner = pytree_unflatten(inner_type, inner_context, leaves)
    return SparseTensor._new_unsafe(inner, constraint)
```

`torch.compile` produces separate compiled graphs for CSR-backed and
ELL-backed `SparseTensor` because their `TreeSpec` values differ. For static
connectivity (the common case in neuromorphic simulation) the format is fixed
after construction, so no recompilation occurs during training.

### Dispatch

`SparseTensor` never appears as a dispatch key. `SparseTensor.mm(x)` calls
`self._inner.mm(x)`, and dispatch sees `type(self._inner)` — `CSR`, `ELL`,
etc. The dispatch registry is entirely unaware of `SparseTensor`.

---

## Concrete Format Types

Each format is a `SparseTensorBase` subclass. Subtypes own the tensors natural
to their encoding and are independently registered as PyTree nodes. There is no
universal required field such as `values` or `canonical_coo()`.

### COO

```python
class COO(SparseTensorBase):
    source: torch.Tensor       # (nnz,) int64
    destination: torch.Tensor  # (nnz,) int64
    values: torch.Tensor       # (nnz,) trainable
```

COO is the natural result of edge-list construction. It is an interoperability
and fallback format, not the primary execution format. See COO Policy.

### CSR

```python
class CSR(SparseTensorBase):
    row_ptr: torch.Tensor      # (n_rows + 1,) int64
    col_indices: torch.Tensor  # (nnz,) int64
    values: torch.Tensor       # (nnz,) trainable
```

CSR is the default format for general `sparse_mm`. Row-major access makes it
natural for fixed fan-out connectivity.

### CSC

```python
class CSC(SparseTensorBase):
    col_ptr: torch.Tensor      # (n_cols + 1,) int64
    row_indices: torch.Tensor  # (nnz,) int64
    values: torch.Tensor       # (nnz,) trainable
```

CSC is natural for fixed fan-in connectivity and column-major kernels.

### ELL

```python
class ELL(SparseTensorBase):
    col_indices: torch.Tensor  # (n_rows, width) int64
    values: torch.Tensor       # (n_rows, width) trainable
    width: int                 # number of entries per row
```

ELL enforces uniform row width. It is the natural format when `max_fan_out`
is declared and validated.

### BSR

```python
class BSR(SparseTensorBase):
    row_ptr: torch.Tensor           # (n_block_rows + 1,) int64
    block_col_indices: torch.Tensor # (n_blocks,) int64
    blocks: torch.Tensor            # (n_blocks, block_rows, block_cols) trainable
    block_shape: tuple[int, int]
```

BSR stores trainable blocks rather than one scalar per structural entry. The
parameter domain and the storage unit domain do not coincide. `block_shape` is
part of the PyTree context, not a tensor leaf.

### SemiStructured24

```python
class SemiStructured24(SparseTensorBase):
    packed_values: torch.Tensor    # hardware-packed representation
    packed_metadata: torch.Tensor  # position metadata coupled to packed_values
```

2:4 structured sparsity for tensor-core kernels. Values and metadata are
coupled and cannot be separated. Requires hardware support (Ampere+) and
specific dtype constraints. Training semantics differ from inference semantics;
see backend documentation.

### ProceduralSparse

```python
class ProceduralSparse(SparseTensorBase):
    parameters: torch.Tensor   # compact generative parameters
    rule: ConnectivityRule     # deterministic generation function (non-tensor)
```

Procedural representations generate edge indices from compact parameters on
demand. They do not store an explicit edge list. Ordinary forward calls must
not silently resample connectivity; dynamic resampling is a separate explicit
operation.

### PyTree registration per subtype

Each concrete subtype is registered independently:

```python
register_pytree_node(
    CSR,
    flatten_fn=_csr_flatten,
    unflatten_fn=_csr_unflatten,
    flatten_with_keys_fn=_csr_flatten_with_keys,
)
```

The flatten contract separates **trainable tensors** (PyTree leaves) from
**structural tensors and metadata** (PyTree context):

```python
def _csr_flatten(W: CSR):
    leaves  = [W.values]
    context = (W.row_ptr, W.col_indices,
               W.shape, W.properties, W.attributes, W.constraint)
    return leaves, context

def _csr_unflatten(context, leaves) -> CSR:
    row_ptr, col_indices, shape, properties, attributes, constraint = context
    return CSR._new_unsafe(row_ptr, col_indices, leaves[0],
                           shape, properties, attributes, constraint)
    # _cache is always empty on reconstructed instances
```

Structural tensors (`row_ptr`, `col_indices`) live in the PyTree context, not
as leaves. `tree_map` applies only to trainable tensors. Device movement is
handled by `SparseParam._apply`, not by PyTree traversal.

### Why each format owns its native tensors

A mandatory structure/value split is not general:

- BSR stores trainable blocks, not one scalar per structural entry.
- Semi-structured formats couple packed values with position metadata.
- Quantized formats associate packed values with per-block scales.
- Procedural formats have no edge list at all.

PyTree handles this because each subtype defines its own tensor leaves and
static context independently.

---

## SparseProperties: Validated Connectivity

```python
@dataclass(frozen=True)
class SparseProperties:
    max_fan_out: int | None = None   # every source connects to exactly this many targets
    max_fan_in: int | None = None    # every target receives from exactly this many sources
    sorted_indices: bool = False       # column indices are sorted within each row
    unique_edges: bool = True          # no duplicate source-destination pairs
```

Properties are **validated against the physical representation** at
construction time. They are never accepted as unchecked hints.

- `CSR` may declare and validate `max_fan_out`.
- `CSC` may declare and validate `max_fan_in`.
- `ELL` always has a natural `max_fan_out` equal to `width`.
- `COO` may declare either property after counting indices.

Backends may use properties to refine dispatch. A property alone does not make
a format suitable for a kernel; the backend must support the format's native
tensors or perform an explicit conversion.

---

## Logical Domains and Attributes

Sparse formats may use several alignment domains:

```text
logical edge domain    one item per source-destination contribution
storage unit domain    one item per stored scalar, block, or packed unit
parameter domain       one item per independent trainable parameter
source domain          one item per presynaptic source
destination domain     one item per postsynaptic target
```

For CSR these domains coincide. For BSR, the parameter domain is the block and
the logical edge domain is the scalar within the block. For grouped
parameterizations they diverge further.

Edge attributes must declare their alignment domain:

```python
@dataclass(frozen=True)
class EdgeAttributes:
    labels: dict[str, torch.Tensor] = field(default_factory=dict)
```

Every tensor in `labels` must contain one item per stored edge (logical edge
domain) or carry a separately documented indexing contract. Receptor type,
channel assignment, delay bin, constraint group, and other heterogeneous
synapse metadata all follow this rule and live in `labels` by key.

Any `SparseTensorBase` subtype may carry attributes:

```python
CSR(..., attributes=EdgeAttributes(labels={"receptor": receptor_ids}))
ELL(..., attributes=EdgeAttributes(labels={"channel": channel_ids}))
```

### LogicalEdges

`LogicalEdges` is a flat topology view for constraint preparation, format
conversion, and debugging. It is not the backend execution contract.

```python
@dataclass(frozen=True)
class LogicalEdges:
    source: torch.Tensor           # (n_logical_edges,) int64
    destination: torch.Tensor      # (n_logical_edges,) int64
    storage_index: torch.Tensor    # index into the format's storage unit
    local_index: torch.Tensor | None = None  # position within a block (BSR)
```

For CSR, `storage_index` addresses a scalar value. For BSR, it identifies a
block and `local_index` identifies a scalar within that block. Procedural or
packed formats may reject `logical_edges()` or return a materialized
approximation.

---

## Constraints

There are two complementary strategies for enforcing invariants on trainable
values. They differ in when and how the invariant is satisfied:

| | **Projection** (Constraint) | **Reparameterization** (Parameterization) |
|---|---|---|
| Parameters | one per edge (`nnz`) | one per group (`n_groups ≤ nnz`) |
| Values | stored directly as `nn.Parameter` | derived at forward time |
| Enforcement | post-optimizer correction step | structural — always exactly satisfied |
| Dale's law | clamp values after step | clamp magnitudes, never violates |
| When to use | full per-edge freedom needed | biological groups share one weight |

This section covers the **Projection** strategy. The Parameterization section
covers reparameterization.

A `Constraint` is a structural description of an invariant that trainable
values must satisfy. It lives as a plain Python attribute on every
`SparseTensorBase` instance. It is not part of the PyTree — not a trainable
tensor, not a structural tensor — so `tree_map` and `tree_unflatten` never
touch it.

```python
W = CSR(row_ptr, col_indices, values, shape)
W = W.with_constraint(GroupedWeights(group_ids=group_ids))
```

`with_constraint` returns a new instance of the same concrete type with the
constraint attached and `_cache = {}`. The constraint does not affect values at
this point; it only guides future preprocessing and projection.

### What a Constraint is not

- Not a gradient-based regularization. Regularization is applied in the loss
  function. A constraint is a hard invariant enforced after the optimizer step.
- Not a structural (topology) constraint. Fixed degree, unique edges, and no
  self-connections are `SparseProperties` validated at construction time.
- Not a forward-pass operation. Constraints are applied post-optimizer via
  `SparseParam.constrain()`, not during forward.
- Not a reparameterization. If the invariant should always be satisfied exactly
  and the number of independent parameters should be reduced, use
  `Parameterization` instead.

### Built-in constraint types

```python
@dataclass(frozen=True)
class NonNegative:
    """All trainable values must be >= 0. Projection: clamp to zero."""

@dataclass(frozen=True)
class GroupedWeights:
    """Edges in the same group share one scalar parameter.
    group_ids[i] is the group index for edge i (logical edge domain).
    Projection: replace each edge's value with its group's mean."""
    group_ids: torch.Tensor   # (nnz,) int64

@dataclass(frozen=True)
class Symmetric:
    """W[i,j] = W[j,i]. Requires reciprocal edges to exist.
    Projection: average each pair of reciprocal edge values."""

@dataclass(frozen=True)
class Bounded:
    """Clamp trainable values to [min_val, max_val].
    Projection: torch.clamp."""
    min_val: float
    max_val: float

@dataclass(frozen=True)
class BlockUnitNorm:
    """Each named group of values has unit L2 norm.
    block_ids[i] is the norm group for edge i.
    Projection: divide each group by its L2 norm."""
    block_ids: torch.Tensor   # (nnz,) int64
```

### Constraint interaction with preprocessing

`prepare_as` and layout algorithms in `to_bsr` may inspect `W.constraint` to
produce a more efficient physical layout. For example, when `GroupedWeights` is
set, a BSR layout algorithm can place edges from the same group into the same
block, reducing the projection's gather-scatter cost:

```python
W = CSR(row_ptr, col_indices, values, shape)
W = W.with_constraint(GroupedWeights(group_ids=group_ids))

# Layout considers group boundaries:
W.prepare_as(BSR, block_shape=(16, 16))

# Projection operates on the constraint-aware layout:
W_param = SparseParam(W)
W_param.constrain()
```

---

## PreparedProjection

A `PreparedProjection` is the compiled form of a `Constraint` — physical index
tensors that implement the projection in terms of the specific format's storage
layout. It is derived from the constraint and the topology but contains no
trainable values.

### Storage

`PreparedProjection` is stored in `SparseTensorBase._cache["projection"]`,
following the same pattern as format caches:

```python
_cache = {
    "projection": GroupProjection(parameter_index, group_scale, ...),
    (BSR, (16, 16)): BSRStructure(row_ptr, block_col_indices, ...),
}
```

It is not registered as an `nn.Module` buffer. It lives in `_cache` and is
invalidated whenever `_cache` is cleared.

### Cache invalidation

The `PreparedProjection` is invalidated in two situations:

1. **Device movement.** `SparseParam._apply` creates a new `SparseTensorBase`
   instance via `_rebuild`, which starts with `_cache = {}`. The next call to
   `constrain(W)` recomputes the projection on the new device.
2. **Constraint change.** `W.with_constraint(new_constraint)` returns a new
   instance with `_cache = {}`. The old projection is discarded.

### Built-in PreparedProjection types

```python
@dataclass(frozen=True)
class GroupProjection:
    """Prepared form of GroupedWeights."""
    parameter_index: torch.Tensor  # (nnz,) int64 — edge → group index
    group_scale: torch.Tensor      # (n_groups,) float — normalization per group
    nonneg_mask: torch.Tensor | None = None  # optional NonNegative flag

@dataclass(frozen=True)
class NonNegativeProjection:
    """Prepared form of NonNegative. No tensors; projection is ReLU."""

@dataclass(frozen=True)
class SymmetricProjection:
    """Prepared form of Symmetric."""
    edge_pair_index: torch.Tensor  # (nnz,) int64 — each edge → its reciprocal index
    # project: values = (values + values[edge_pair_index]) / 2

@dataclass(frozen=True)
class BoundedProjection:
    """Prepared form of Bounded."""
    min_val: float
    max_val: float
```

### Functional interface

```python
def constrain(W: SparseTensorBase) -> SparseTensorBase:
    """Apply constraint projection. Lazy: computes and caches PreparedProjection
    on first call. Subsequent calls within the same SparseParam lifetime reuse
    the cache."""
    if W.constraint is None:
        return W
    if "projection" not in W._cache:
        W._cache["projection"] = prepare_projection(W.constraint, W.logical_edges())
    return project(W, W._cache["projection"])
```

`prepare_projection(constraint, logical_edges)` compiles the constraint into a
`PreparedProjection` using the topology in `logical_edges`. For `GroupedWeights`
this involves grouping edges by `group_ids`, computing group sizes for
normalization, and building a scatter-reduce index. The result is immutable and
device-local.

`project(W, prepared)` applies the prepared projection to `W`'s trainable
tensors and returns a new `SparseTensorBase` instance sharing structural
tensors. It does not assume a universal `values` field; each subtype defines
its own projection support.

### Training interface on SparseParam

`SparseParam.constrain()` is the training-loop entry point. It calls the
functional `constrain(W)` and copies the projected values back in place so that
the optimizer's parameter tensors are updated without reallocating them:

```python
@torch.no_grad()
def constrain(self) -> None:
    projected = constrain(self._W)      # functional, populates cache
    self._copy_trainable_leaves_(projected)
```

This is called after each optimizer step, not inside forward:

```python
optimizer.step()
W_param.constrain()   # correct any invariant violations introduced by the step
```

---

## Parameterization

A `Parameterization` replaces the direct per-edge `values` tensor with a compact
set of trainable parameters and a deterministic rule that derives effective values
at forward time. Unlike projection, the invariant is always exactly satisfied by
construction — there is no post-step correction step.

```python
class Parameterization:
    def effective_values(self) -> torch.Tensor: ...
    def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]: ...
    def _fixed_tensors(self) -> list[tuple[str, torch.Tensor]]: ...
```

`parameterization` is a Python attribute on `SparseTensorBase`, set via
`with_parameterization`. It is not part of the PyTree.

```python
W = CSR(row_ptr, col_indices, initial_weight, shape)
W = W.with_parameterization(GroupedMagnitude(
    initial_weight=initial_weight,
    group_index=group_index,
    magnitude=torch.ones(n_groups),
    dale=True,
))
```

### GroupedMagnitude

The standard parameterization for biologically structured connections: connections
of the same cell-type pair share one learnable magnitude scalar. Initial
connectivity weights (e.g. synapse counts from the connectome) are fixed buffers;
only the per-group scalar is trained.

```python
@dataclass
class GroupedMagnitude:
    """values[i] = initial_weight[i] * magnitude[group_index[i]]

    initial_weight : (nnz,)     fixed buffer from connectome, never trained
    group_index    : (nnz,)     int64, structural — maps each edge to its group
    magnitude      : (n_groups,) the only trainable parameter
    dale           : bool        if True, enforce magnitude >= 0 by clamping
    """
    initial_weight: torch.Tensor
    group_index: torch.Tensor
    magnitude: torch.Tensor
    dale: bool = False

    def effective_values(self) -> torch.Tensor:
        return self.initial_weight * self.magnitude[self.group_index]

    def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("magnitude", self.magnitude)]

    def _fixed_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("initial_weight", self.initial_weight),
                ("group_index", self.group_index)]
```

The number of trainable parameters is `n_groups`, which is typically orders of
magnitude smaller than `nnz` when biological cell-type groups are large.

### Interaction with SparseParam

When a parameterization is present, `SparseParam` exposes its trainable tensors
to the optimizer instead of `values`:

```python
class SparseParam(SparseTensorBase, nn.Module):
    def __init__(self, sparse: SparseTensorBase) -> None:
        nn.Module.__init__(self)
        self._W = sparse
        if sparse.parameterization is not None:
            p = sparse.parameterization
            for name, tensor in p._trainable_tensors():
                self._parameters[name] = nn.Parameter(tensor)
            for name, tensor in p._fixed_tensors():
                self._buffers[name] = tensor
        else:
            for name, tensor in sparse._trainable_tensors():
                self._parameters[name] = nn.Parameter(tensor)
        for name, tensor in sparse._structural_tensors():
            self._buffers[name] = tensor
```

The optimizer receives `magnitude` (shape `(n_groups,)`) rather than `values`
(shape `(nnz,)`). The effective weights are computed at forward time via
`W.effective_values()` inside the backend kernel.

### Dale's law with GroupedMagnitude

When `dale=True`, `SparseParam.constrain()` clamps the magnitude parameter
directly, without any index operations:

```python
@torch.no_grad()
def constrain(self) -> None:
    p = self._W.parameterization
    if p is not None and getattr(p, "dale", False):
        self._parameters["magnitude"].clamp_(min=0)
    elif self._W.constraint is not None:
        projected = constrain(self._W)
        self._copy_trainable_leaves_(projected)
```

This is faster than per-edge projection and exactly enforces Dale's law: a
non-negative magnitude multiplied by an initial weight of fixed sign always
produces a weight of the same sign.

### Construction from heterogeneous connectivity

```python
conn_mat, constraint_mat, receptor_idx = make_hetersynapse_constrained_conn(
    neurons, connections, ...)

group_index = extract_group_index(constraint_mat)   # (nnz,) int64, 0-based
initial_weight = torch.tensor(conn_mat.data, dtype=torch.float32)

W = CSR.from_edges(sources, destinations, initial_weight, shape)
W = W.with_parameterization(GroupedMagnitude(
    initial_weight=initial_weight,
    group_index=group_index,
    magnitude=torch.ones(group_index.max() + 1),
    dale=True,
))
W_param = SparseParam(W)
# optimizer sees: W_param._parameters["magnitude"]  shape (n_groups,)
# buffers:        W_param._buffers["initial_weight"] shape (nnz,)   fixed
#                 W_param._buffers["group_index"]    shape (nnz,)   structural
```

---

## Prepared Format Conversion

When a target format is needed every forward pass, `prepare_as` precomputes and
caches the structural layout at setup time and assembles the target format
cheaply in each forward call.

A structural cache record holds precomputed layout without trainable weights:

```python
@dataclass(frozen=True)
class BSRStructure:
    row_ptr: torch.Tensor
    block_col_indices: torch.Tensor
    block_shape: tuple[int, int]
    value_gather_index: torch.Tensor  # (n_blocks, br, bc) → flat index into values
    shape: tuple[int, int]
    properties: SparseProperties
```

```python
def prepare_as(self, format: type, **kwargs) -> None:
    """Precompute structural layout for the target format.

    Considers self.constraint when computing layout (e.g. group-aware
    block placement for BSR). Stores result in self._cache.
    Call before torch.compile or the training loop.
    """
    key = (format, frozenset(kwargs.items()))
    if key in self._cache:
        return
    self._cache[key] = _compute_structure(self, format, self.constraint, **kwargs)
```

```python
def as_bsr(self) -> BSR:
    """Assemble a live BSR from cached structure and current values.

    The returned BSR is ephemeral. Only the value gather is performed per
    call. torch.compile traces the gather as a tensor operation with no
    graph break; gradients flow through the gather index.
    """
    key = (BSR, ...)
    if key not in self._cache:
        raise RuntimeError("call prepare_as(BSR, block_shape=...) first")
    s = self._cache[key]
    blocks = self.effective_values()[s.value_gather_index]
    return BSR._new_unsafe(
        row_ptr=s.row_ptr,
        block_col_indices=s.block_col_indices,
        blocks=blocks,
        shape=self.shape,
        block_shape=s.block_shape,
        properties=s.properties,
        attributes=self.attributes,
        constraint=self.constraint,
    )
```

Format caches and `PreparedProjection` caches are cleared together when
`_cache` is reset on device movement. Recomputation on next use is inexpensive:
pure index arithmetic on already-moved structural tensors.

---

## SparseParam: nn.Module Bridge

`SparseParam` bridges any `SparseTensorBase` into PyTorch's training
infrastructure. It follows the same pattern as `TensorDictParams` in the
TensorDict library: it inherits from both `SparseTensorBase` and `nn.Module`,
making it simultaneously a valid sparse matrix and an `nn.Module` whose
parameters and buffers are visible to the optimizer, `state_dict`, and device
movement.

```python
class SparseParam(SparseTensorBase, nn.Module):
    def __init__(self, sparse: SparseTensorBase) -> None:
        nn.Module.__init__(self)
        self._W = sparse   # persistent — cache lives here across forward calls
        if sparse.parameterization is not None:
            p = sparse.parameterization
            for name, tensor in p._trainable_tensors():
                self._parameters[name] = nn.Parameter(tensor)
            for name, tensor in p._fixed_tensors():
                self._buffers[name] = tensor
        else:
            for name, tensor in sparse._trainable_tensors():
                self._parameters[name] = nn.Parameter(tensor)
        for name, tensor in sparse._structural_tensors():
            self._buffers[name] = tensor

    def sparse(self) -> SparseTensorBase:
        return self._W
```

`SparseParam` stores `self._W` as a persistent reference. The cache on
`self._W` survives between forward calls. All `SparseTensorBase` operations
delegate to `self._W`:

```python
    def mm(self, x: torch.Tensor) -> torch.Tensor:
        return sparse_mm(self._W, x)

    def prepare_as(self, format: type, **kwargs) -> None:
        self._W.prepare_as(format, **kwargs)

    def as_bsr(self) -> BSR:
        return self._W.as_bsr()
```

`SparseParam` does not implement `forward()`. Execution belongs in the neural
layer.

Each `SparseTensorBase` subtype implements the tensor classification methods
used when no parameterization is present:

```python
# on CSR:
def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]:
    return [("values", self.values)]

def _structural_tensors(self) -> list[tuple[str, torch.Tensor]]:
    return [("row_ptr", self.row_ptr), ("col_indices", self.col_indices)]
```

When a `Parameterization` is attached, `SparseParam` uses the parameterization's
`_trainable_tensors()` and `_fixed_tensors()` instead. The sparse format's own
`_trainable_tensors()` is not called in that case.

### Device movement and cache invalidation

`SparseParam._apply` is overridden to rebuild `self._W` from the moved
parameter and buffer tensors after device movement:

```python
    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)   # moves parameters and buffers
        self._W = self._W._rebuild(self._parameters, self._buffers)
        # _rebuild constructs a new instance from moved tensors with _cache = {}
        return result
```

`_rebuild` creates a new `SparseTensorBase` instance from moved tensors with an
empty cache. Format caches and `PreparedProjection` are recomputed lazily on
first use after device movement. This is acceptable because device movement is
infrequent and recomputation is cheap.

### Constraint application after optimizer step

```python
    @torch.no_grad()
    def constrain(self) -> None:
        p = self._W.parameterization
        if p is not None and getattr(p, "dale", False):
            # Reparameterization: clamp the compact magnitude parameter directly.
            self._parameters["magnitude"].clamp_(min=0)
        elif self._W.constraint is not None:
            # Projection: correct per-edge values post-optimizer.
            projected = constrain(self._W)      # lazy: populates cache on first call
            self._copy_trainable_leaves_(projected)
```

`_copy_trainable_leaves_` copies projected values back into the existing
`nn.Parameter` tensors in place. The optimizer continues to hold references to
the same parameter objects; no reallocation occurs.

The two branches are mutually exclusive: a `SparseTensorBase` carries either a
`Parameterization` or a `Constraint`, not both. If Dale's law is needed with
full per-edge parameters, use `NonNegative` as the constraint.

---

## SparseLinear: Neural Layer

`SparseLinear` is a thin convenience layer. It holds a `SparseParam` submodule
and adds bias and shape validation. It is not an architectural requirement; any
`nn.Module` may hold a `SparseParam` directly.

```python
class SparseLinear(nn.Module):
    def __init__(
        self,
        W: SparseTensorBase,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.W = SparseParam(W)
        self.bias = nn.Parameter(torch.zeros(W.shape[1])) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.W.mm(x)
        return out if self.bias is None else out + self.bias
```

Neural wrappers for event-based or channel-aware execution follow the same
pattern: hold a `SparseParam`, call the appropriate operation.

---

## Format Construction and Conversion

### Format-agnostic construction

Construct a `SparseTensor` (auto-selecting) from edge data:

```python
W = SparseTensor.from_edges(
    sources,
    destinations,
    values,
    shape=(N_pre, N_post),
    properties=SparseProperties(max_fan_out=32),
)
print(W.selected_format)   # ELL, because max_fan_out was declared
```

The returned type is always `SparseTensor`. The inner format is observable via
`selected_format`. Format is fixed at construction.

### Explicit construction

Construct a concrete enforced type directly:

```python
W = CSR(row_ptr, col_indices, values, shape, properties, attributes)
W = BSR(row_ptr, block_col_indices, blocks, shape, block_shape, properties, attributes)
```

Direct construction is preferred when data is already in the target layout or
when a specific format is required throughout the object's lifetime.

### Functional conversion

Pure conversion functions consume a `SparseTensorBase` and return a new
instance of the target subtype. They pay the full conversion cost on every call
and are appropriate for one-time setup, debugging, and serialization:

```python
W_csr: CSR = to_csr(W)
W_bsr: BSR = to_bsr(W, block_shape=(16, 16))
W_coo: COO = to_coo(W)
```

Conversion uses a `ConversionPlan` — an immutable record of the permutation or
mapping tensors needed to reorder values and attributes consistently:

```python
plan: ConversionPlan = prepare_conversion(W, CSR)
W_csr: CSR = apply_conversion(plan, W)
```

Conversions that merge duplicate edges require an explicit reduction policy.
Conversions between different logical domains (e.g. BSR-to-CSR expanding blocks
into scalar edges) require richer mappings than a permutation.

### Module-level prepared conversion

When a target format is needed every forward pass, use `prepare_as` on the
`SparseTensorBase` instance (accessed through `SparseParam` or directly). The
structural computation is done once; each forward call assembles the target
format from cached structure plus live parameter values via a cheap index
gather. `torch.compile` traces the index gather as a tensor operation with no
graph break.

| Scenario | Use |
|---|---|
| One-time setup, inspection, serialization | `to_*(W)` |
| Every forward pass | `prepare_as` + `as_*()` |
| Constraint preparation | `prepare_projection` + `PreparedProjection` |
| Backend fallback | `to_coo(W)` with a logged warning |

---

## Backend Dispatch

Dispatch selects a kernel implementation from the concrete subtype and
operation context:

```python
implementation = registry.resolve(
    operation=operation,          # sparse_mm, event_sparse_mm, …
    format=type(matrix),          # CSR, BSR, ELL, … (never SparseTensor)
    properties=matrix.properties, # max_fan_out, sorted_indices, …
    attributes=matrix.attributes, # labels present
    device=input.device.type,     # cpu, cuda, …
    dtype=input.dtype,
    requires_grad=requires_grad,
)
```

`SparseTensor` and `SparseParam` are never dispatch targets. Dispatch always
sees the concrete format type: `CSR`, `ELL`, `BSR`, etc.

Priority order:

1. Exact kernel for this format and operation.
2. Property-specialised kernel for this format (e.g. fixed-fan-out CSR).
3. Explicit cached conversion to another supported format.
4. Native Python fallback.

When two kernels match at the same priority level, the one registered with
higher specificity (more dispatch dimensions matched) wins. If specificity is
equal, registration order determines priority and a warning is emitted.

The selected implementation receives the concrete `SparseTensorBase` subtype
directly. It must not call `to_coo()` or `canonical_coo()` unless it is the
explicit COO fallback path.

An explicitly requested backend that is unavailable raises a focused error.
`backend="auto"` falls back according to the priority order above.

---

## COO Policy

COO is an interoperability and fallback format. It is not the universal
backend contract.

The pattern:

```python
source, destination, values = matrix.canonical_coo()
```

must not appear in any backend. It discards format information and prevents
kernels from exploiting row pointers, sorted indices, and fixed-degree
properties.

Explicit COO conversion:

```python
W_coo: COO = to_coo(W)
```

is appropriate for:

- Debugging and topology inspection.
- Serialization interchange.
- The COO fallback backend path (must log a warning).
- Format conversion as an intermediate step.

Conversion results may be cached when connectivity is static. Caches store
derived structural plans, never independent trainable weight copies.

---

## Procedural Connectivity

`ProceduralSparse` contains compact generative parameters and a deterministic
generation rule rather than a stored edge list.

It participates in dispatch as a concrete `SparseTensorBase` subtype. A backend
may execute it directly or require explicit materialization:

```python
W_csr: CSR = materialize(W_procedural, format=CSR)
```

Ordinary forward calls must not silently resample connectivity. Dynamic
resampling is a separate explicit operation with separate reproducibility
semantics.

---

## Required Tests

- PyTree flatten/unflatten round-trip for each concrete subtype.
- `flatten_with_keys_fn` produces correct keyed paths.
- `tree_map` applies only to trainable tensors; structural tensors and
  `constraint` are unchanged.
- `_new_unsafe` bypasses validation; public constructors reject invalid inputs.
- Property validation: correct properties accepted, incorrect properties
  rejected, for COO, CSR, CSC, ELL.
- Equivalent numeric output across COO, CSR, CSC, ELL for the same connectivity.
- `SparseTensor.from_edges` selects correct inner format for each heuristic
  branch; `selected_format` reports the inner type accurately.
- `isinstance` checks: `SparseTensor` is not `CSR`; `CSR` is `SparseTensorBase`;
  backend dispatch does not see `SparseTensor` as a dispatch key.
- `SparseParam` exposes correct parameters and buffers; `state_dict` contains
  exactly the trainable leaves.
- `SparseParam.sparse()` returns `self._W`: cache is accessible across calls.
- Device movement via `SparseParam.to(device)` moves parameters and buffers;
  `self._W._cache` is cleared; `_rebuild` produces a correctly-typed instance.
- `prepare_as` populates `_cache`; `as_*()` assembles a correct live format;
  gradients flow through the value gather index.
- Numeric equivalence between direct format and `prepare_as`-assembled format.
- Gradient correctness: `values` gradient is non-zero; structural tensor
  gradients are absent.
- Functional conversion `to_*()`: numeric equivalence, attribute alignment,
  `ConversionPlan` round-trip.
- `with_constraint()` attaches constraint without modifying values; returns a
  new instance with empty `_cache`.
- `constrain(W)` with no constraint is a no-op and returns the same instance.
- `constrain(W)` on first call computes and caches `PreparedProjection`;
  subsequent calls reuse the cache.
- `PreparedProjection` is cleared after device movement; recomputed on next
  `constrain(W)` call on the correct device.
- `GroupProjection`: projected values satisfy grouped-weights invariant.
- `NonNegativeProjection`: projected values are all >= 0.
- `SymmetricProjection`: projected values satisfy W[i,j] = W[j,i].
- `SparseParam.constrain()` copies projected values into existing parameter
  tensors without reallocating them (optimizer references remain valid).
- Constraint-aware layout: `prepare_as(BSR)` with `GroupedWeights` places
  same-group edges in the same block where possible.
- BSR: execution without flattening blocks; `prepare_as(BSR)` and `as_bsr()`.
- Channel semantics via `EdgeAttributes.labels` across at least two subtypes.
- Event semantics across at least two compatible subtypes.
- Value and attribute alignment after conversion.
- Alignment across logical, storage, and parameter domains.
- Duplicate-edge conversion with explicit reduction policy.
- Backend dispatch: correct kernel selected; deterministic fallback; explicit
  unavailable backend raises; `SparseTensor` and `SparseParam` are not valid
  dispatch targets.
- Functional `project()` followed by `SparseParam._copy_trainable_leaves_()`.
- `Parameterization`: `effective_values()` returns correct derived values before
  and after optimizer step.
- `GroupedMagnitude`: optimizer updates `magnitude`; `effective_values()` reflects
  the update without any explicit copy.
- `GroupedMagnitude` with `dale=True`: `SparseParam.constrain()` clamps `magnitude`
  in place; `effective_values()` returns non-negative values after constrain.
- `SparseParam` with parameterization exposes `magnitude` as `nn.Parameter` and
  `initial_weight`, `group_index` as buffers; `values` is not in `_parameters`.
- `state_dict` contains only `magnitude`, not `initial_weight` or `values`.
- Device movement moves `magnitude` and `initial_weight` correctly; `effective_values()`
  returns tensors on the correct device after `.to(device)`.
- `constrain()` is a no-op when neither constraint nor parameterization is set.
- Projection and parameterization are mutually exclusive on one instance; mixing
  them raises at construction time.
- `effective_values()` traces correctly under `torch.compile` for both direct and
  parameterized cases.
- `torch.compile`, `torch.export`, and `torch.func` behavior for CSR and ELL.
- `SparseTensor` PyTree: CSR-backed and ELL-backed instances have different
  `TreeSpec` values; `torch.compile` does not conflate them.
