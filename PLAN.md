# Task: Prototype and Implement a PyTorch-First Sparse / Connectivity System for btorch

You are working on the existing **btorch** codebase, a PyTorch-based computational neuroscience / recurrent SNN framework.

Your task is to **audit the current sparse-connection implementation, design a concrete migration path, and implement a working prototype of the new sparse/connectivity subsystem**.

Do not treat this as a greenfield rewrite. Inspect the existing repository first, preserve existing behavior where reasonable, add compatibility shims where necessary, and migrate incrementally.

The final design must feel natural to users familiar with:

- PyTorch / `torch.nn`
- `torch.compile`
- SciPy sparse
- NEST-style network connectivity
- SNN frameworks such as GeNN / SpikingJelly

while supporting requirements that ordinary sparse-matrix libraries do not cover:

- recurrent SNN connectivity
- heterogeneous receptors
- grouped/constraint weights
- delays
- batched trajectories
- batches of different networks
- sparse spike execution
- dynamic topology / Deep Rewiring
- procedural/implicit connectivity
- future multi-GPU/offload/runtime optimization

The goal is **not** to implement every possible optimization now. The goal is to establish correct, extensible public APIs and a small but real runtime that can later support more backends and planners without changing model code.

---

# 0. First: audit the existing btorch code

Before editing code, inspect the repository and identify the current implementation and tests around at least:

- `SparseConnection`
- `StructuredConnection`
- `ImplicitConnection`
- `ConstrainedWeight`
- `Synapse`
- `make_hetersynapse_conn`
- `make_hetersynapse_constraint`
- `make_hetersynapse_constrained_conn`
- sparse backend selection (`native`, `torch_sparse`)
- constraint/group construction
- Dale-law enforcement
- receptor/delay expansion
- serialization / `state_dict`
- `.to()` / `_apply()`
- current tests and examples using these APIs

Current implementation characteristics that must be verified against the actual code:

1. `SparseConnection` accepts btorch sparse objects plus supported Torch,
   SciPy, adjacency, and edge-list conversion paths.
2. Connection orientation is explicit as `pre_post` or `post_pre`; conversion
   must not silently transpose or lose stored edge order.
3. Forward preserves leading sample dimensions and uses the runtime planner,
   registered operators, and reconstructible execution representations.
4. `ConstrainedWeight` represents grouped learnable magnitudes using base
   values, group ids, and group scales.
5. Heterosynapse adapters support the legacy expanded representation and the
   semantic receptor/delay representation.
6. Runtime selection distinguishes representation, algorithm, and backend.
7. Runtime caches, prepared task layouts, and CUDA Graph captures are not
   canonical checkpoint state.

Verify all of these before changing anything.

Produce a short implementation note describing:

- current behavior
- current orientation conventions
- current tests
- current pain points
- which code can be reused
- which code needs replacement

Do not begin with a broad rewrite.

---

# 1. Hard design principles

The following are architectural invariants.

## 1.1 PyTorch owns the compilation lifecycle

Users must write:

```python
model = Model(...).cuda()
model.train()

y = model(x)
loss.backward()

model = torch.compile(model)
```

Do NOT introduce public APIs such as:

```python
model.compile()
conn.compile()
net.compile()
plan = conn.compile(...)
```

Execution planning is an internal runtime detail.

---

## 1.2 Separate modeling semantics, numerical representation, and execution

These are different concepts.

### Modeling

```text
Projection
├── ConnectionRule
└── Synapse
```

### Numerical objects

```text
LinearOperator
├── Sparse
│   ├── COO
│   ├── CSR
│   ├── CSC
│   ├── BSR
│   └── future CSF
├── StructuredOperator
├── ImplicitOperator
└── CompositeOperator
```

### Connection realization

```text
Connection
├── SparseConnection
├── StructuredConnection
├── ImplicitConnection
└── HybridConnection
```

### Runtime

```text
Planner
RepresentationCache
BackendRegistry
KernelCache
```

Do not merge these layers.

In particular:

```text
Projection != Sparse
ConnectionRule != Sparse
Sparse != CSR
Procedural != Sparse
Sparse format != execution algorithm
Execution algorithm != backend
```

---

# 2. Sparse public API

Implement a generic sparse numerical API independent of neuroscience connectivity.

The desired user experience should be close to SciPy sparse and PyTorch.

Example:

```python
from btorch import sparse

A = sparse.csr(
    crow_indices,
    col_indices,
    values,
    shape=(M, N),
)

y = A @ x
```

Generic format-agnostic sparse:

```python
A = sparse.from_edges(
    rows,
    cols,
    values,
    shape=(M, N),
)

y = A @ x

A_csr = A.tocsr()
A_coo = A.tocoo()
```

Core generic properties:

```python
A.shape
A.dtype
A.device
A.nnz
A.ndim

A.batch_shape
A.sparse_shape
A.dense_shape

A.batch_dim()
A.sparse_dim()
A.dense_dim()
```

---

# 3. Format-agnostic and format-specific objects

A generic:

```python
A: Sparse
```

must not expose format-specific data such as:

```python
A.indptr
```

unless that representation is explicitly requested.

Format-specific objects should expose familiar APIs.

For CSR:

```python
csr.indptr
csr.indices
csr.data
```

and preferably aliases compatible with PyTorch terminology:

```python
csr.crow_indices()
csr.col_indices()
csr.values()
```

COO should expose:

```python
coo.indices()
coo.values()
```

or convenience:

```python
coo.row
coo.col
coo.data
```

where appropriate.

Avoid inventing unnecessarily different terminology.

---

# 4. Conversion versus representation access

Distinguish:

```python
A.tocsr()
```

from:

```python
A.as_csr()
```

Desired semantics:

### `tocsr()`

May create/materialize/convert a CSR representation.

### `as_csr()`

Must not perform an expensive conversion.

It returns an existing compatible representation or raises a clear error.

Likewise consider:

```python
tocoo()
as_coo()

tocsc()
as_csc()

tobsr(...)
as_bsr(...)
```

This distinction is useful for expert users and runtime code.

---

# 5. Torch sparse and SciPy sparse interop

This is a required first-class feature.

Create one central conversion layer.

Recommended API:

```python
A = sparse.asarray(obj)
```

plus explicit constructors:

```python
A = Sparse.from_torch(torch_sparse_tensor)
A = Sparse.from_scipy(scipy_sparse_array)
```

and module-level aliases:

```python
sparse.from_torch(...)
sparse.from_scipy(...)
```

Also implement export where practical:

```python
A.to_torch()
A.to_scipy()
```

Optional format argument:

```python
A.to_torch(layout="csr")
A.to_scipy(format="csr")
```

## 5.1 Supported PyTorch layouts

At minimum recognize:

- `torch.sparse_coo`
- `torch.sparse_csr`
- `torch.sparse_csc`
- `torch.sparse_bsr`
- `torch.sparse_bsc`

Preserve format when btorch supports the format.

Do not silently densify.

Do not silently transpose.

Preserve shape semantics exactly.

For unsupported sparse layouts, either:

1. convert through COO when lossless and reasonable, or
2. raise a descriptive error.

Do not detach values unnecessarily.

If the PyTorch sparse tensor's values require gradients, conversion must preserve an appropriate autograd relationship where possible.

---

## 5.2 Supported SciPy inputs

Support both:

- modern `scipy.sparse.sparray`
- legacy `scipy.sparse.spmatrix`

At minimum understand:

- COO
- CSR
- CSC
- BSR

For:

- LIL
- DOK
- DIA

it is acceptable initially to normalize through COO/CSR.

Prefer SciPy sparse-array semantics internally over legacy matrix semantics.

SciPy input is CPU-origin data; device transfer should be explicit or controlled by:

```python
Sparse.from_scipy(
    A,
    device=...,
    dtype=...,
)
```

---

## 5.3 One converter used everywhere

Do NOT separately implement:

```text
SparseConnection.from_scipy
Projection.FromSparse.from_scipy
ConstrainedWeight.from_scipy
...
```

with duplicated conversion logic.

Instead:

```python
def as_sparse(x, ...):
    ...
```

and all higher-level APIs reuse it.

For example:

```python
SparseConnection(A)
```

may accept:

```text
btorch Sparse
torch sparse Tensor
SciPy sparse array/matrix
```

and internally call:

```python
as_sparse(A)
```

Similarly:

```python
FromSparse(A)
```

must use the same path.

---

# 6. Matrix orientation: solve this explicitly

This is important because current btorch connectivity and normal sparse-linear-algebra conventions are not necessarily identical.

Generic `Sparse` must obey normal mathematical matrix semantics:

```python
A.shape == (M, N)

y = A @ x
```

means:

```text
x last dimension = N
y last dimension = M
```

`from_torch()` and `from_scipy()` must never silently transpose.

The btorch connection convention is:

```text
pre_post:
    rows = pre-synaptic neurons
    columns = post-synaptic neurons

post_pre:
    rows = post-synaptic neurons
    columns = pre-synaptic neurons
```

`from_adjacency()` and `FromSparse` use `pre_post` by default. A connection
stores this orientation explicitly and `to_sparse()` defaults to the
orientation from which the connection was built, so conversion round-trips.

For `pre_post`, propagation is mathematically equivalent to:

```text
x @ connectivity
```

Do not silently change existing model behavior.

Define this boundary explicitly.

The explicit interface is:

```python
SparseConnection.from_adjacency(
    A,
    orientation="pre_post",
)
```

and also accepts `orientation="post_pre"`. Reject every other string.

The generic numerical sparse API continues to follow standard matrix
convention. Connection orientation is an adapter concern at the boundary
between adjacency semantics and numerical matrix semantics.

The implementation must contain tests preventing accidental double transpose or orientation inversion.

Document this clearly.

---

# 7. N-D sparse tensor model

Do not design the sparse core as 2-D-only.

The logical shape model should support:

```text
[*batch, *sparse, *dense]
```

where:

- `batch_shape`: independent sparse objects / batches
- `sparse_shape`: dimensions represented sparsely
- `dense_shape`: tensor payload associated with each stored sparse entry

Example:

```text
shape        = [8, 1000, 2000, 4]
batch_shape  = [8]
sparse_shape = [1000, 2000]
dense_shape  = [4]
```

Interpretation:

> 8 sparse matrices; each stored `(i,j)` entry contains a vector of length 4.

Do not confuse dense-value axes with BSR block dimensions.

---

# 8. Format dimensionality

Initial intended semantics:

### COO

Supports arbitrary sparse dimensionality.

```text
sparse_dim >= 1
```

### CSR / CSC

Exactly two logical sparse dimensions.

```text
sparse_dim == 2
```

May also have:

```text
*batch + [M,N] + *dense
```

### BSR / BSC

Exactly two logical sparse dimensions.

Block size is storage metadata, not extra logical axes.

### Future CSF

Reserve architecture space for arbitrary higher-order compressed sparse tensors.

Do not implement a full CSF compiler unless necessary for the prototype.

---

# 9. Sparse operations

Primary common API:

```python
A @ x
```

and:

```python
sparse.matmul(A, x)
```

For higher-order sparse tensors, provide or reserve:

```python
sparse.einsum(...)
```

For example:

```python
Y = sparse.einsum(
    "bijkd,bk->bijd",
    A,
    x,
)
```

Internally it is fine to canonicalize to a generic contraction representation.

Do not expose APIs centered around:

```text
spmv()
spmm()
spmspv()
spgemm()
```

as the main user abstraction.

These are lowerings.

Possible lowering:

```text
Sparse × dense vector → SpMV
Sparse × multiple RHS → SpMM
Sparse × sparse signal → sparse push / SpMSpV-like path
Sparse × Sparse → lazy composition or SpGEMM
```

---

# 10. Do not overimplement general sparse compilation in phase 1

We want architecture compatible with Scorch/TACO-like sparse tensor compilation in the future.

We do NOT need a complete TACO clone now.

For the prototype:

- implement COO
- implement CSR
- implement conversion between them
- implement basic matmul
- establish N-D metadata
- scaffold higher-order COO
- make `einsum` either limited or explicitly experimental

Keep extension points clean.

---

# 11. Connection modeling layer

Above numerical sparse, define neuroscience connectivity semantics.

Desired conceptual API:

```python
proj = Projection(
    pre=E,
    post=I,
    rule=FixedIndegree(1000),
    synapse=Synapse(
        weight=...,
        delay=...,
        receptor=...,
        plasticity=...,
    ),
)
```

The central concepts are:

```text
Projection
ConnectionRule
Synapse
```

A `ConnectionRule` expresses model semantics, not sparse format.

Examples:

```python
OneToOne()
AllToAll()
FixedIndegree(k)
FixedOutdegree(k)
PairwiseBernoulli(p)
DistanceDependent(...)
FromEdges(src, dst)
FromSparse(A)
```

---

# 12. Connection rule does not determine execution representation

For example:

```python
FixedIndegree(100)
```

could later be realized as:

```text
generated once → SparseConnection(CSR)
```

or:

```text
runtime regenerated → ImplicitConnection
```

Likewise:

```python
AllToAll()
```

may become a structured operator instead of an explicit dense matrix.

Do not make:

```text
ConnectionRule IS-A Sparse
```

---

# 13. Connection realization

Internally allow:

```text
Connection
├── SparseConnection
├── StructuredConnection
├── ImplicitConnection
└── HybridConnection
```

`SparseConnection` may approximately contain:

```python
class SparseConnection(nn.Module):
    matrix: Sparse
    synapse: Synapse
```

The normal model API remains:

```python
current = conn(spikes)
```

while generic numerical API remains:

```python
y = A @ x
```

These should not be conflated.

---

# 14. Synapse semantics

A synapse can describe:

```python
Synapse(
    weight=...,
    delay=...,
    receptor=...,
    plasticity=...,
)
```

Do not create combinatorial subclasses such as:

```text
SparseConstrainedHeterDelayedPlasticConn
```

Instead make weight/routing/plasticity orthogonal semantics.

---

# 15. Weight representations

Support a conceptual hierarchy such as:

```text
constant
per-edge
grouped/factorized
kernel-shared
procedural
```

Examples:

```python
Synapse(weight=1.0)
```

```python
Synapse(weight=nn.Parameter(...))
```

Grouped constraint:

```python
Synapse(
    weight=ConstrainedWeight(
        base=base_weight,
        group=group_id,
        scale=group_scale,
    )
)
```

Mathematical semantics:

```text
w[e] = base[e] * scale[group[e]]
```

Possibly later:

```python
KernelWeight(...)
ProceduralWeight(...)
```

---

# 16. Grouped constraints belong in `ConstrainedWeight`

The existing grouped-weight behavior must remain available.

Do not throw away the current functionality.

Keep the underlying semantics as:

```text
SparseConnection
+
ConstrainedWeight
```

Preserve:

- group magnitude parameters
- initial weights
- Dale-law behavior
- group introspection where useful
- checkpoint compatibility where practical

Construction-time Pandas logic may remain temporarily for legacy conversion, but Pandas must not appear in execution or compiled hot paths.

---

# 17. Receptor / heterogeneous synapse semantics

Current btorch utilities physically expand dimensions to encode receptor channels.

Do not make that the only semantic representation going forward.

Preferred semantic representation:

```text
edge:
    src
    dst
    weight
    receptor
```

or:

```python
Synapse(
    weight=...,
    receptor=receptor_id,
)
```

Runtime may lower this into:

```text
expanded (dst,receptor) sparse matrix
separate matrix per receptor
custom routed kernel
```

depending on backend.

Keep the existing physical-expansion helpers working as compatibility utilities, but regard them as one possible lowering / legacy representation rather than the fundamental model.

Do not introduce a public ``HeterogeneousLinear`` class. A dense numerical
layer remains an ordinary PyTorch-compatible ``Linear`` with weight shape
``[out_features, in_features]``. Receptor, delay, group, and Dale-law meaning
belongs to ``Synapse`` and the semantic edge table owned by
``SparseConnection``. A backend may select a dense internal lowering for a
topologically dense projection without changing that modeling interface.

---

# 18. Delay semantics

Same principle:

```python
Synapse(
    delay=delay
)
```

Do not permanently define delayed connectivity by multiplying matrix dimensions.

Possible backend representations include:

```text
delay buckets
ring buffers
event queues
expanded sparse dimensions
fused routing
```

but these belong below the modeling API.

---

# 19. Structure and values are not always separable

Do not enforce the architectural invariant:

```text
topology structure
+
independent values
```

for every sparse representation.

Examples where they may be coupled:

- COO canonicalization and duplicate reduction
- BSR block payload
- packed graph representations
- bitmask/boolean adjacency
- procedural edge-and-weight generation
- symmetric half-storage
- quantized packed edge structures

Internally support the fact that representation transforms may require value transforms.

However, do NOT expose a complicated `SEPARATE / ALIGNED / FUSED` taxonomy to ordinary users.

This belongs in internal storage/capability metadata.

---

# 20. Representation transforms must preserve value correspondence

If COO → CSR sorts or coalesces edges, all edge-aligned metadata must be transformed consistently:

```text
weight
group_id
receptor
delay
plasticity state
edge ID
```

Design an internal transform/permutation mapping where needed.

For duplicate reduction, backward semantics must be correct.

Do not let each metadata array independently guess the permutation.

---

# 21. Batched connection semantics

Distinguish:

```text
network batch
```

from:

```text
training/sample/trajectory batch
```

They are not the same.

---

# 22. Network batch

Example:

```text
G different networks
```

Logical connection:

```python
A.shape == (G, M, N)
A.batch_shape == (G,)
```

Support at least two important cases.

## 22.1 Shared topology, different weights

```text
crow:    [M + 1]
col:     [E]

values:  [G, E]
```

This is extremely important for:

- ensembles
- parameter sweeps
- replicas with shared structure
- multiple independently trained weights

Do not duplicate connectivity indices G times unnecessarily.

It should remain logically a batched sparse tensor, not a new mathematical type.

---

## 22.2 Different topology per network

Different graphs may have:

```text
nnz_0 != nnz_1 != nnz_2 ...
```

Do not require padding to equal `max_nnz`.

Support or design for packed/ragged physical storage:

```text
network_ptr
row pointers / offsets
col[sum(nnz_g)]
values[sum(nnz_g)]
```

The public interface should remain simple.

Possible public construction:

```python
A = sparse.stack([A0, A1, A2])
```

Do not expose implementation names such as:

```text
RaggedPackedBatchedCSR
```

unless needed as an internal class.

For the first prototype, it is acceptable to implement:

1. shared-pattern batched CSR fully
2. different-pattern batching through a generic batched COO or grouped fallback

and leave optimized ragged CSR for a later stage.

---

# 23. Sample batch and network batch broadcasting

Normal sample batch:

```text
A:       [M, N]
spikes:  [B, N]
out:     [B, M]
```

Network batch:

```text
A:       [G, M, N]
spikes:  [G, B, N]
out:     [G, B, M]
```

Follow PyTorch broadcasting rules wherever possible.

Do NOT make:

```python
conn(spikes[B, N])
```

implicitly mean Cartesian product over all G networks.

If the same B samples should run on all G networks, the user should be able to write standard broadcast-style code such as:

```python
spikes = spikes.unsqueeze(0)
```

and rely on normal broadcasting.

No hidden Cartesian-product semantics.

---

# 24. Training-generated batched spikes

This is a critical correctness constraint.

For surrogate-gradient SNN training:

```python
spike = neuron(v)
```

the forward spike may be 0/1 and highly sparse.

However, the gradient with respect to spike/membrane state may be dense.

Therefore:

> sparse active indices are an execution representation, not necessarily the logical autograd object.

Default training-generated spike should remain an ordinary PyTorch Tensor:

```python
spike.shape == (..., N)
```

The runtime may internally derive:

```text
active_indices
batch_ptr
bitset
```

for faster forward propagation.

A possible compiled path:

```python
active_idx, ptr = torch.ops.btorch.pack_spikes(spike)

out = torch.ops.btorch.sparse_spike_propagate(
    structure,
    weight,
    spike,          # logical/autograd input
    active_idx,     # execution aid
    ptr,
)
```

The backward op must still be capable of returning dense:

```text
grad_spike[..., N]
```

so surrogate gradients are correct.

Do not design training around an indices-only logical spike object.

---

# 25. Genuine sparse/event input

Externally supplied events may truly be sparse:

```text
event-camera data
precomputed spikes
event-driven simulator events
```

These may use:

```python
SparseSpike(...)
```

or generic sparse COO where appropriate.

It is fine to distinguish:

```text
training-generated spike Tensor
```

from:

```text
genuine event/sparse input
```

Do not force every spike in the framework into a custom `SpikeTensor` class.

---

# 26. Dynamic topology / Deep Rewiring

Dynamic structural training must not be implemented as a new sparse matrix format.

Treat it as a training/update policy.

Preferred conceptual API:

```python
conn = SparseConnection(...)

rewire = HardDeepR(
    conn,
    target_nnz=K,
    ...
)

rewire.attach(optimizer)
```

Normal user training remains:

```python
loss.backward()
optimizer.step()
```

The rewiring controller performs its structural update around/post optimizer step.

---

# 27. Hard Deep R

Hard rewiring should favor a fixed number of active edge slots:

```text
src   [K]
dst   [K]
theta [K]
```

Rewiring updates edge identities in-place:

```text
slot k:
(old_src, old_dst)
→
(new_src, new_dst)
```

while preserving tensor shapes.

Advantages:

- stable parameter shapes
- stable optimizer-state shapes
- friendlier to `torch.compile`
- friendlier to CUDA Graph
- no allocation required solely because an edge changes identity

When topology changes:

```text
topology_version += 1
```

invalidate structural execution caches such as:

- sorted CSR
- transpose CSR
- BSR conversion
- structural properties
- backend preprocessing dependent on pattern

but do not force Dynamo graph recompilation merely because indices changed.

Implement at least a minimal fixed-slot hard rewiring prototype if practical.

---

# 28. Soft Deep R

Do not pretend exact soft rewiring has the same memory model as hard sparse rewiring.

Exact soft rewiring retains dormant latent variables that can reactivate.

Represent this explicitly.

Possible model:

```text
candidate latent parameters
+
active mask
```

If the full candidate universe is dense, acknowledge the corresponding memory cost.

If implementing a scalable approximation with only a dormant candidate pool, give it a distinct name such as:

```text
SampledSoftRewire
```

and document that it is not exact soft Deep R.

Soft Deep R is lower priority than the hard fixed-slot prototype.

---

# 29. Procedural connectivity

Keep two separate concepts.

## 29.1 Construction-time connection rule

Example:

```python
FixedIndegree(100)
```

may generate explicit edges once.

Result:

```text
SparseConnection
```

## 29.2 Runtime implicit/procedural operator

An operator may never materialize its full matrix:

```python
A = ImplicitOperator(
    shape=(N, N),
    matvec=...,
)
```

This may be sparse, dense, structured, or analytically defined.

Therefore:

```text
procedural != sparse
```

Do not place every procedural object under `Sparse`.

---

# 30. Implicit operator capabilities

An implicit operator may optionally support capabilities such as:

```text
matvec
matmat
enumerate_edges
materialize_sparse
transpose_apply
```

Do not require all of them.

For example:

```text
apply-only operator
```

need not implement:

```python
tocsr()
```

while an enumerable procedural sparse operator could.

---

# 31. Runtime planner is hidden

The user must not choose traversal algorithms in normal model definitions.

Do not expose as primary public APIs:

```text
source-driven
destination-driven
push
pull
word-packed
SpMV
SpMM
SpMSpV
```

These are runtime choices.

Planner inputs may include:

```text
operation
input representation
input density
batch/RHS shape
sparse properties
training/eval
device
available cached representations
limited user hints
```

Planner may choose:

```text
representation
algorithm
backend
```

Use the following runtime vocabulary consistently:

```text
PlanningContext   immutable facts available to the planner
RouteKey          representation / algorithm / backend identity
RouteSpec         one complete registered execution route
Plan              the selected route and the context identity it depends on
PlanningDefaults  conservative fallback cost assumptions
TaskStrategy      a backend-private implementation choice
```

These are implementation types, not ordinary model-building APIs. They may
use normal Python names and be kept out of the public package surface through
module placement and ``__all__``; do not rely on a proliferation of leading
underscore names to communicate the boundary.

The planner must run when a connection is built or when a relevant planning
fact changes. It must not run inside every recurrent timestep.

---

# 32. Representation, algorithm, backend are separate

Example:

```text
representation:
    pre_post CSR

algorithm:
    sparse push

backend:
    custom CUDA
```

or:

```text
representation:
    post_pre CSR

algorithm:
    dense multi-RHS propagation

backend:
    aten
```

Do not encode all combinations in one giant enum.

`RouteKey` identifies one combination without collapsing the dimensions:

```python
@dataclass(frozen=True)
class RouteKey:
    operation: str
    representation: str
    algorithm: str
    backend: str
```

Examples:

```text
propagate / pre_post CSR / push / triton
propagate / post_pre CSR / pull / aten
```

Backend-private choices are not additional public planning dimensions. For
example, direct and hash aggregation may both be used by one Triton task-push
execution. They are `TaskStrategy` choices inside that route, not separate
algorithms or route variants.

The Triton task-push route selects direct/hash aggregation while preparing its
task layout. Both strategies share one route and operator contract. The chosen
per-task layout, workspace requirements, and strategy/allocation version are
part of `PreparedState` and therefore participate in the prepared-cache key and
CUDA Graph capture identity.

Name a backend by the implementation registered with btorch. An ATen CUDA CSR
route remains backend `aten` even when ATen internally calls cuSPARSE. Use
backend `cusparse` only if btorch has a separately registered direct cuSPARSE
implementation.

---

# 33. Minimal runtime components

Avoid premature framework complexity.

Start with only:

```text
RepresentationCache
Planner
BackendRegistry
KernelCache
```

No public `ExecutionPlan`.

No public `Traversal`.

No public `ConnectionRepresentation`.

Internal objects are fine where useful.

The four runtime modules form deep modules with small interfaces:

```text
RepresentationCache
    owns reconstructible execution representations

Planner
    constructs a PlanningContext, filters routes, and selects a Plan

BackendRegistry
    owns kernel implementations, complete RouteSpec registrations,
    backend overrides, and a monotonic generation

KernelCache
    owns compiled kernels, prepared route state, workspaces, and task layouts
```

`RouteSpec`, `RouteKey`, `PlanningContext`, and `Plan` are records used inside
these modules. They do not introduce additional runtime subsystems.

`RouteSpec` is the atomic planning registration unit:

```python
@dataclass(frozen=True)
class RouteSpec:
    key: RouteKey
    supports: Callable[[PlanningContext], SupportResult]
    estimate_cost: Callable[[PlanningContext], CostEstimate]
    prepare: Callable[..., PreparedState]
    operator: RegisteredOperator
```

```python
@dataclass(frozen=True)
class RegisteredOperator:
    op: Callable[..., Tensor]
    supports_backward: bool
    supports_compile: bool
    supports_capture: bool
```

`SupportResult` must account for dtype/device, value batching, training,
backward, deterministic execution, `torch.compile`, and CUDA Graph capture as
required by the `PlanningContext`. An unsupported training or backward route
is filtered during planning, not discovered after execution begins.

`RegisteredOperator` is one complete execution contract. For a training
`torch.library` route, registration validates the forward implementation,
fake/meta implementation, and autograd formula as one unit. A route owns that operator,
its cost model, preparation, workspace requirements, and invalidation behavior
together. Do not assemble a plan by independently selecting unrelated
forward, backward, and prepare kernels.

Keep the existing low-level kernel interface for compatibility:

```python
registry.register(kernel, backend, fn, ...)
```

Add complete planner candidates through:

```python
registry.register_route(route_spec)
```

A legacy kernel registration does not automatically become a planner
candidate. It participates through a complete `RouteSpec`, an explicit legacy
adapter, or the temporary legacy fallback path.

---

# 34. Runtime caches and versions

At minimum distinguish:

```text
topology_version
value_version
routing_version
```

Potentially later:

```text
property_version
placement_version
```

The registry must also maintain:

```text
registry_generation
```

Registering, replacing, or removing a route, or changing route availability,
global defaults, or calibration increments the process-wide
`registry_generation`. A context-local backend override does not mutate that
global generation; it has a separate context-local `override_revision` and
override snapshot. Both identities participate in planning and capture
validation. Duplicate route registration should be an error unless replacement
is explicit. Re-registering the same implementation may be idempotent.

Each `RouteKey` also has a monotonic `route_revision` stored with its registry
entry. Initial registration creates the first revision. Explicit replacement
creates a new revision. Availability changes leave the executable revision
unchanged but increment `registry_generation`. A `Plan` binds one exact route
revision; prepared caches and captures include that revision in their keys.

Typical SGD update:

```text
value_version++
```

must not invalidate topology representations.

Rewiring:

```text
topology_version++
```

must invalidate structure-dependent representations.

Changing receptor/delay routing may increment:

```text
routing_version
```

`Plan` records only immutable planning identity:

```python
@dataclass(frozen=True)
class Plan:
    route: RouteKey
    route_revision: int
    registry_generation: int
    backend_override: str | None
    override_revision: int
    problem_signature: Hashable
    reason: str
```

Do not store device pointers, live tensors, Python backend objects, workspace,
or prepared task layouts in `Plan`. Prepared state belongs in `KernelCache`
and is keyed by the route and registry generation plus topology, routing,
device type and index, dtype, shape and batch signature, deterministic mode,
compile/capture requirements, and allocation identities on which it depends.

A plan becomes stale when any planning fact in its signature changes,
including topology, routing, device, dtype, deterministic requirements,
compile/capture requirements, active backend override, or registry generation.
An optimizer update that preserves shape and allocation does not by itself
require replanning or recapture. Replacing a Parameter, buffer, workspace, or
prepared cache allocation does require dependent prepared state and captures
to be invalidated even when shape and dtype are unchanged.

Preparation follows route selection:

```text
construct PlanningContext
    → enumerate RouteSpec candidates
    → check availability
    → supports(context)
    → estimate_cost(context)
    → select Plan
    → prepare selected route
    → execute prepared route
```

`supports()` and `estimate_cost()` must be pure and cheap. They must not
allocate workspaces, compile kernels, synchronize a device, mutate caches, or
inspect data-dependent spike contents, including by lazily initializing CUDA
or Triton. `prepare()` runs only for the selected route and may construct
reconstructible state outside CUDA Graph capture. It must publish prepared
state atomically: a failed or partial preparation must not leave an older or
incomplete cache entry marked valid.

Execution receives the selected `Plan` and uses only the `RouteSpec` revision
bound to that plan. It must not independently call low-level registry
resolution for forward, backward, or preparation. Replacing an implementation
creates a new route revision and registry generation; old plans can no longer
resolve to the replacement accidentally.

The CUDA Graph owner includes the `Plan` identity and prepared-state allocation
identity in its capture version. Before replay it compares the current module
version against that capture version. A mismatch recaptures outside the graph
or raises a clear error; it never reuses the stale capture.

---

# 35. `torch.compile` / Dynamo compatibility

This is a hard requirement.

Do not rely on native PyTorch sparse Tensor objects as the sole hot-path runtime IR.

Do not put a Python planner inside every forward call.

Do not require arbitrary Python `ExecutionPlan` objects as graph inputs.

`PlanningContext`, `RouteSpec`, and `Plan` remain outside the compiled tensor
graph. The selected route must lower to stable tensor/custom-op calls before
Dynamo traces the hot execution path.

Forward and backward must use the same route contract captured at trace time.
Neither autograd nor a compiled graph may re-query the mutable registry and
select a different backend during backward. If a planning dependency changes,
invalidate and rebuild the compiled/captured execution or reject the change;
do not silently alter an already compiled execution path.

The preferred compiled boundary is:

```text
Python modeling / Sparse objects
          ↓
raw Tensor buffers / Parameters
          ↓
torch.library registered op
          ↓
Tensor output
```

Example:

```python
torch.ops.btorch.csr_propagate(
    crow,
    col,
    values,
    x,
)
```

or:

```python
torch.ops.btorch.constraint_propagate(
    crow,
    col,
    base_weight,
    group_id,
    group_scale,
    x,
)
```

---

# 36. Custom-op requirements

Any important custom op intended for training should provide:

- CPU reference implementation where practical
- CUDA implementation eventually
- fake/meta implementation
- autograd registration
- clear alias/mutation schema
- tests with `torch.library.opcheck`
- gradient checks where meaningful

Support:

```python
torch.compile(module, fullgraph=True)
```

for the main connection modules as an acceptance target.

Do not use a `torch.Tensor` subclass as the core Sparse design unless there is an exceptionally strong demonstrated reason.

A normal semantic Python wrapper plus Tensor-backed connection modules is preferred.

---

# 37. Direct `Sparse @ x` compile behavior

Investigate separately whether:

```python
A = sparse.csr(...)
f = torch.compile(lambda x: A @ x)
```

can be made robust when `A` is closed over.

This is desirable but secondary to:

```python
torch.compile(SparseConnection(...))
```

which is the hard requirement.

Do not sacrifice the clean architecture solely to make arbitrary custom Sparse objects valid Dynamo graph inputs.

Report the result of this experiment.

---

# 38. `.to()`, dtype, and `state_dict()`

Respect normal PyTorch module behavior.

Persistent model state may include:

- trainable weights
- constraint scales
- canonical topology buffers when applicable
- canonical routing metadata
- procedural seed/config if this defines model state

Execution caches must not be serialized by default:

- temporary transposed CSR
- backend descriptors
- autotune result
- temporary workspace
- packed spike buffers
- backend-specific plans
- route specifications and selected plans
- registry generation and backend override state
- calibration data

Semantic state != execution cache.

Fix or remove the current fragile pattern where a cached sparse tensor can become stale relative to model buffers after device moves or checkpoint loads.

Prefer canonical raw Tensor buffers plus reconstructible caches.

Loading a checkpoint on another machine must rebuild planning and prepared
state from the current registry, device, dtype, and runtime requirements.
After `load_state_dict()`, invalidate `Plan`, prepared state, and CUDA Graph
captures. Canonical topology, `pre_post`/`post_pre` orientation, receptor,
delay, and routing metadata remain semantic model state; runtime version
counters and selections are reconstructed rather than restored verbatim.

---

# 39. Native sparse and `torch_sparse`

Do not make `torch_sparse` a required architectural dependency.

Current compatibility may be retained temporarily.

The new core should work without it.

Recommended direction:

```text
backend="auto"
```

internally considers:

- ATen/native reference path
- custom registered op
- optional `torch_sparse` legacy path
- future cuSPARSE/custom CUDA/Triton

Do not expose a growing list of backend names throughout model constructors.

Keep backend override an advanced/runtime option.

`use_backend(name)` is a planner candidate filter for expert testing and
debugging. It is not permission to re-resolve each low-level kernel
independently after a route has been selected. Entering or leaving an override
must invalidate affected plans and CUDA Graph captures, or be rejected while
an incompatible capture is active.

Backend overrides must be context-local and safely nest across exceptions.
They must not leak across unrelated threads, tasks, or concurrent model
executions. Implement `use_backend()` with a `ContextVar` token and include
the override snapshot and context-local `override_revision` in `Plan`. Entering
and leaving an override advances only the local revision; it does not mutate
the process-wide `registry_generation`. If a platform cannot provide that
isolation, concurrent override use must be rejected explicitly rather than
falling back to mutable process-global state.

A selected route must resolve preparation and execution from the same
`RouteSpec`. Dynamic fallback to a different route is permitted only before
capture, after a clearly recoverable availability or preparation failure.
CUDA Graph capture and replay must never switch routes dynamically.

After capture, the selected route, prepared state, workspace, and captured
Tensor addresses remain fixed for the lifetime of that capture. Registry or
override changes cannot modify what an existing graph executes.

---

# 40. Preserve Dale-law semantics

Existing:

```text
enforce_dale
```

behavior must remain available.

Refactor unsafe `.data` mutation where reasonable toward:

```python
with torch.no_grad():
    ...
```

or an explicitly designed constraint mechanism.

Do not silently change learned-weight semantics.

Add tests ensuring signs behave identically to current btorch expectations.

---

# 41. Initial migration architecture

A reasonable package layout is approximately:

```text
btorch/
│
├── sparse/
│   ├── __init__.py
│   ├── base.py
│   ├── coo.py
│   ├── csr.py
│   ├── conversion.py
│   ├── ops.py
│   └── properties.py
│
├── models/
│   └── connection/
│       ├── base.py
│       ├── sparse.py
│       ├── weight.py
│       ├── synapse.py
│       ├── rule.py
│       └── projection.py
│
├── optim/
│   └── rewire.py
│
└── sparse/
    └── runtime/
        ├── planner.py
        ├── cache.py
        ├── backend.py
        └── ops.py
```

Do not force this exact layout if it conflicts badly with the existing repository.

Prefer minimal integration with existing package organization.

---

# 42. Proposed public APIs to prototype

## 42.1 Sparse creation

```python
A = sparse.coo(
    indices,
    values,
    shape=(M, N),
)
```

```python
A = sparse.csr(
    crow,
    col,
    values,
    shape=(M, N),
)
```

```python
A = sparse.from_edges(
    src,
    dst,
    values,
    shape=(M, N),
)
```

---

## 42.2 External sparse conversion

```python
A = sparse.asarray(torch_sparse_tensor)
A = sparse.asarray(scipy_sparse)
```

```python
A = sparse.from_torch(torch_sparse_tensor)
A = sparse.from_scipy(scipy_sparse)
```

```python
torch_A = A.to_torch()
scipy_A = A.to_scipy()
```

---

## 42.3 Format conversion

```python
csr = A.tocsr()
coo = A.tocoo()
```

No-copy access:

```python
csr = A.as_csr()
```

---

## 42.4 Numerical operation

```python
y = A @ x
```

Eventually:

```python
y = sparse.einsum("...", A, x)
```

---

## 42.5 Sparse connection

```python
conn = SparseConnection(
    A,
    synapse=Synapse(...),
)
```

The canonical connection name is `SparseConnection`. Do not introduce a
second abbreviated connection class or compatibility alias for abandoned
intermediate designs.

---

## 42.6 Constraint weight

```python
weight = ConstrainedWeight(
    base=base_weight,
    group=group_id,
    scale=scale,
)
```

---

## 42.7 NEST-like modeling

```python
proj = Projection(
    pre=E,
    post=I,
    rule=FixedIndegree(1000),
    synapse=Synapse(
        weight=...,
        delay=...,
        receptor=...,
    ),
)
```

This layer can be implemented after core sparse/connection functionality if needed.

---

# 43. Compatibility with existing heterosynapse helpers

Do not delete the existing data-preparation workflows abruptly.

Existing code may rely on:

```python
make_hetersynapse_conn(...)
make_hetersynapse_constraint(...)
make_hetersynapse_constrained_conn(...)
stack_hetersynapse(...)
```

Keep them initially.

Add adapters that translate their outputs into the new semantic objects.

Then add a cleaner future path that stores:

```text
src
dst
weight
receptor
delay
constraint_group
```

without mandatory physical expansion.

Tests should verify both old and new pathways produce equivalent currents for small reference networks.

---

# 44. Properties and hints

Keep this minimal.

Correctness/structural properties may include:

```text
sorted
unique
symmetric
structurally_symmetric
triangular
fixed_pattern
shared_pattern
```

Hints may include only a few performance expectations:

```text
expected_calls
expected_density
expected_batch
```

Do not expose dozens of backend tuning knobs.

Properties must not be confused with hints.

`Hints` contains workload expectations only. It must not name a backend,
algorithm, vendor library, task strategy, launch configuration, workspace
size, or route-specific crossover threshold.

Backend-specific interpretation belongs in `RouteSpec.supports()` and
`RouteSpec.estimate_cost()`. For example, the same
`expected_density=0.01` may favor different routes on different devices or
graph structures without changing the public hint.

Connections should support replacing several hints atomically so the planner
runs once:

```python
conn.update_hints(
    expected_density=0.01,
    expected_batch=32,
    expected_calls=1000,
)
```

Assigning a complete validated `Hints` value may also replan once. Mutating
individual fields of a frozen `Hints` object is not supported.

## 44.1 Route costs and defaults

Capability is a hard constraint. Cost ranks only routes that satisfy the
planning context.

Every candidate should estimate an absolute cost in a shared unit or a value
with equivalent ordering semantics. Do not give each backend an independent
boolean threshold and expect pairwise crossovers to remain consistent when a
new candidate is added.

Approximate amortized cost as:

```text
prepare cost / expected calls
    + steady-state execution cost
    + capture cost / expected calls
    + optional memory penalty
```

`PlanningDefaults` provides conservative fallback assumptions. Kernel-specific
cost knowledge belongs beside the route registration in its kernel module.
Machine-specific measured calibration may later override cost coefficients,
but calibration is runtime data, not model state.

Use a minimal common estimate contract:

```python
@dataclass(frozen=True)
class CostEstimate:
    execution_ns: float
    preparation_ns: float = 0.0
    capture_ns: float = 0.0
    workspace_bytes: int = 0
    confidence: float = 0.0
```

Time fields are ordering estimates in nanoseconds, not benchmark promises.
`confidence` is diagnostic and never converts an unsupported route into a
candidate.

`PlanningDefaults` may fill only missing cost assumptions such as launch,
preparation, capture, and memory penalties. It cannot provide route identity,
capabilities, preparation, operator execution, or autograd behavior.

`PreparedState` is backend-defined reconstructible runtime state. Its cache
entry records the complete preparation key and an allocation/version identity.
Publish it only after successful preparation, reuse it while the key stays
valid, and discard rather than serialize it.

The historical CUDA `2%` push threshold is not a global planner rule. If it is
retained during migration, it is an initial parameter of the relevant route
cost model. Its meaning depends on graph shape, degree distribution, batch,
dtype, training mode, determinism, and CUDA Graph usage.

Avoid a generic mutable inheritance hierarchy such as:

```text
global → device → backend → kernel → variant
```

because identity, capabilities, cost coefficients, and preparation contracts
have different merge semantics. Resolve each `RouteSpec` into a complete,
immutable descriptor at registration. Global defaults may fill missing cost
assumptions, but must not synthesize a route by merging unrelated pieces.

---

# 45. Shared-pattern batched sparse

Explicitly prototype the common case:

```text
same topology
different edge values
```

Example:

```python
A = sparse.csr(
    crow,
    col,
    values,                # [G, E]
    shape=(G, M, N),
)
```

or another clean constructor if shape ambiguity requires it.

Test:

```text
G independently weighted networks
B training samples each
```

with input:

```text
[G, B, N]
```

and output:

```text
[G, B, M]
```

Compare against dense reference.

---

# 46. Different-pattern batched sparse

Provide a simple correct reference implementation.

Possible initial strategy:

```text
stack to batched COO
```

or grouped loop hidden below the API.

Do not prematurely optimize.

The architecture must not assume all batch elements have equal `nnz`.

Add a test with deliberately different edge counts.

---

# 47. Autograd requirements

Test gradients for:

1. per-edge values
2. dense input
3. constrained group scales
4. batched shared-pattern values
5. compiled connection path

For:

```text
w[e] = base[e] * scale[group[e]]
```

check gradient numerically against dense/reference implementation.

Topology tensors:

```text
indices
crow
group IDs
receptor IDs
delay IDs
```

do not receive ordinary differentiable gradients unless a future specialized estimator explicitly implements this.

---

# 48. Sparse topology mutation

Do not perform arbitrary variable-length tensor reallocations inside compiled forward.

Topology updates occur outside the normal forward graph.

For rewiring:

```text
optimizer/update boundary
```

is preferable.

Runtime structures can be lazily rebuilt after topology version changes.

---

# 49. Prototype phases

Implement in staged commits.

## Phase 0 — Audit and baseline

- inspect repository
- document current sparse behavior
- run existing tests
- add missing regression tests for current behavior before refactoring
- benchmark current `native` and `torch_sparse` paths on small/medium matrices

No major code changes yet.

---

## Phase 1 — Sparse core and interop

Implement:

- `Sparse`
- `COO`
- `CSR`
- `from_edges`
- `asarray`
- `from_torch`
- `from_scipy`
- `to_torch`
- `to_scipy`
- `tocoo`
- `tocsr`
- standard `A @ x`
- shape/batch/sparse/dense metadata

Focus on correctness and API.

---

## Phase 2 — Migrate existing sparse connection

Continue consolidating `SparseConnection` onto the new sparse core.

Requirements:

- preserve documented `SparseConnection` behavior
- remove repeated SciPy-only conversion logic
- preserve leading batch dimensions
- fix stale sparse-cache issues
- avoid reconstructing unnecessary objects each forward where possible
- maintain `.to()` and `state_dict()` behavior
- compile with `torch.compile`

Compatibility adapters are acceptable only for public behavior that still
exists in the supported package, not for already-removed prototype class names.

---

## Phase 3 — Constraint weight refactor

Implement:

```text
ConstrainedWeight
```

and use it through `SparseConnection`.

Make group mapping edge-aligned and construction-time.

Support:

```text
training dynamic group scales
```

without reconstructing semantic sparse objects unnecessarily.

Add gradient tests.

---

## Phase 4 — Receptor/delay semantic representation

Create a non-expanded semantic representation for:

```text
receptor
delay
```

Preserve old expanded pathway as a lowering/reference implementation.

Do NOT immediately write a sophisticated CUDA routed kernel.

A reference implementation is sufficient.

---

## Phase 5 — Batched connectivity

Implement:

- shared topology + batched values
- different topology reference fallback
- normal sample batch
- network batch
- broadcasting semantics

Test all combinations.

---

## Phase 6 — Compile-safe runtime operations

Introduce `torch.library` registered ops where useful.

Target:

```python
torch.compile(conn, fullgraph=True)
```

for:

- ordinary sparse connection
- constrained connection
- batched shared-pattern sparse connection

Provide fake/meta + autograd support.

---

## Phase 7 — Sparse spike execution prototype

Keep logical training spike as dense Tensor.

Prototype:

```text
pack_spikes
+
sparse propagation
+
dense backward
```

Compare gradients against dense propagation.

Do not introduce complicated user-facing spike types prematurely.

---

## Phase 8 — Hard rewiring prototype

Implement fixed-slot topology mutation.

Integrate with optimizer update.

Test:

- number of active edges stays fixed
- optimizer state for rewired slots is handled correctly
- forward graph does not change shape
- topology cache invalidation works
- `torch.compile` still runs after repeated rewiring

---

## Phase 9 — Experimental N-D sparse

Add:

- N-D COO metadata
- simple higher-order operations / scaffold
- initial `sparse.einsum` experiment if feasible

Do not build a full sparse compiler.

Document what future Scorch/TACO-inspired work would be required.

---

# 50. Tests and acceptance criteria

Create a serious test matrix.

## Conversion tests

Round-trip:

```text
SciPy COO → btorch → SciPy
SciPy CSR → btorch → SciPy
SciPy CSC → btorch → SciPy
SciPy BSR → btorch → SciPy

Torch COO → btorch → Torch
Torch CSR → btorch → Torch
Torch CSC → btorch → Torch
Torch BSR → btorch → Torch
Torch BSC → btorch → Torch
```

Verify:

- shape
- values
- indices
- duplicates/canonicalization semantics
- dtype
- device where relevant

---

## Numerical tests

Compare sparse implementation with dense reference for:

```text
A @ x
A @ X
leading batch dims
network batch
sample batch
network + sample batch
```

Use random sparse matrices including:

- empty rows
- duplicate COO entries
- unsorted COO
- zero-size cases where supported

---

## Gradient tests

Check:

```text
grad input
grad values
grad constraint scale
batched values
```

against dense reference / gradcheck.

---

## `torch.compile` tests

At minimum:

```python
compiled = torch.compile(conn, fullgraph=True)
```

and verify forward/backward.

Test repeated calls.

Test `.to()` before compile.

Test checkpoint save/load followed by compile.

Test that planning occurs before tracing and is not repeated in each recurrent
timestep.

Test every registered training route for:

```text
fake/meta implementation
torch.library.opcheck
forward compilation
backward compilation
fullgraph execution where supported
```

Routes that intentionally do not support backward must reject training inputs
through `supports()` or a clear pre-execution error; they must not silently
detach or fall through to an unrelated backward implementation.

---

## Planner and registry tests

Verify:

- capability filtering happens before cost estimation
- unsupported routes are never selected regardless of their estimated cost
- `supports()` and `estimate_cost()` do not allocate, synchronize, or mutate
- preparation comes from the selected `RouteSpec`
- forward and backward come from one compatible route contract
- legacy kernel registrations remain usable but are not accidental candidates
- duplicate route registration requires explicit replacement
- registry generation invalidates cached plans
- backend override entry and exit invalidate affected plans
- changing several hints causes one replan
- optimizer value updates do not invalidate topology preparation
- optimizer value updates preserving allocation do not force recapture
- replacing a Parameter/buffer/workspace allocation does force recapture
- topology, routing, device, and dtype changes do invalidate dependent state
- device index, batch shape, deterministic mode, and compile/capture
  requirements participate in cache identity
- failed preparation cannot expose partial state or preserve a stale valid flag

---

## CUDA Graph tests

Verify eager, first capture, and replay separately.

Test:

- preparation completes before capture
- cold capture either prepares safely beforehand or raises a clear error
- replay performs no Python planning, allocation, route lookup, or host sync
- unchanged topology captures once and replays repeatedly
- weight updates preserving allocation do not force recapture
- rewiring, route changes, registry generation, workspace reallocation, device,
  and dtype changes do force recapture
- `use_backend()` cannot leave a captured graph executing a stale route
- nested and exceptional `use_backend()` scopes restore the previous context
- concurrent override use is isolated or rejected explicitly
- capture/replay never dynamically falls back to another route
- forward and backward use the same selected route despite registry changes

For recurrent networks, instrument planning and preparation counts. Sparse
representation and static task preparation should occur once per stable
connection state, not once per timestep.

---

## State serialization tests

Ensure:

```text
canonical model state survives save/load
execution cache does not need serialization
```

Loading a checkpoint must not leave stale cached sparse objects.

---

## Legacy compatibility tests

Existing:

```text
SparseConnection
ConstrainedWeight
heterosynapse utility
```

tests should continue to pass or have explicit migration updates.

---

## Rewiring tests

Hard Deep R:

- edge count remains K
- rewired coordinates actually change
- optimizer state reset/transfer policy is explicit
- topology version changes
- old cached representation is not reused incorrectly

---

# 51. Performance sanity checks

Do not over-optimize yet, but measure:

1. current `SparseConnection`
2. new eager implementation
3. new `torch.compile` implementation
4. dense reference
5. optional `torch_sparse`
6. each eligible registered route in isolation
7. eager and CUDA Graph execution separately

Workloads:

```text
small debug
~4k-neuron recurrent network
larger sparse recurrent synthetic graph
batch=1
batch=32
low spike density
high spike density
representative degree distributions
training and inference where supported
```

Report:

- forward latency
- forward+backward latency
- peak GPU memory where convenient
- conversion/preprocessing cost separately from steady-state execution
- route preparation and CUDA Graph capture cost
- selected route and planning reason
- actual rather than only expected activity density

Compare the Triton task-push route against the ATen CUDA CSR route without
calling the latter a separate cuSPARSE backend. If a direct cuSPARSE route is
added later, benchmark and name it separately.

Measure crossover behavior across density rather than validating one global
threshold. Record enough workload facts to explain why a route wins:

```text
shape
nnz
degree distribution
batch/RHS shape
dtype
training/eval
deterministic mode
CUDA Graph enabled/disabled
```

Do not claim optimization without measurements.

---

# 52. Avoid these design mistakes

Do NOT:

- create a second public compile lifecycle
- put traversal choice into `Projection`
- make every procedural object sparse
- model receptor/delay only via permanent dimension expansion
- require one value per edge
- require structure/value separation for every representation
- make training-generated sparse spikes indices-only logical objects
- require equal `nnz` across network batches
- create combinatorial connection subclasses
- make `torch_sparse` mandatory
- make a `torch.Tensor` subclass the core API without strong evidence
- put Pandas/SciPy manipulation inside recurrent forward
- mutate topology inside Dynamo hot path
- make backend descriptors part of `state_dict`
- silently transpose Torch/SciPy input during conversion
- silently densify unsupported sparse input
- expose SpMV/SpMM/push/pull as the primary modeling API
- treat a density threshold as a universal planner policy
- put backend-specific tuning fields into `Hints`
- register forward, backward, and preparation as independently selectable
  pieces of one execution route
- call task direct/hash aggregation separate public routes or algorithms
- call an ATen implementation `cusparse` merely because ATen may use cuSPARSE
- run planning, calibration, compilation, or workspace allocation inside a
  recurrent timestep or CUDA Graph capture
- let `use_backend()` silently change low-level kernels beneath a cached plan
- merge arbitrary global/device/backend/kernel profiles into a synthetic route
- serialize `PlanningContext`, `RouteSpec`, `Plan`, prepared state, or
  calibration data as model state

---

# 53. Things that may remain internal

These concepts are useful internally but should not become ordinary public model APIs:

```text
PlanningContext
RouteKey
RouteSpec
Plan
PlanningDefaults
CostEstimate
TaskStrategy
Traversal
push/pull
source-driven/destination-driven
SpMV/SpMM/SpMSpV
representation permutation
structure/value coupling
cuSPARSE descriptor
autotune plan
workspace
backend capability table
```

Use the normal class names above in their implementation modules and control
the supported import surface with module organization and `__all__`. Internal
does not require every type name to begin with an underscore.

Developer/debug APIs may expose them through something like:

```python
sparse.explain(conn, example_input)
```

but model code should not depend on them.

---

# 54. Optional `explain()` debugging API

A useful expert API would be:

```python
print(sparse.explain(conn, spikes))
```

Example output:

```text
Input:
    shape = [32, 134013]
    representation = dense
    estimated spike density = 0.8%

Connection:
    logical shape = [134013, 134013]
    nnz = ...
    canonical format = CSR
    fixed pattern = True

Selected runtime:
    cached representation = pre_post CSR
    operation = propagate
    algorithm = push
    backend = triton
    registry generation = ...
    reason = estimated lowest amortized cost

Prepared state:
    topology version = ...
    routing version = ...
    CUDA Graph compatible = True

Constraint:
    fused

Receptor:
    expanded reference lowering

Alternatives:
    post_pre CSR / pull / aten
        supported = True
        estimated cost = ...
    pre_post CSR / push / triton
        selected
```

Backend-private task strategies may appear in an optional diagnostic detail
section, but they must not be presented as user-selectable algorithms.

This is preferable to making execution plans part of normal API.

Do not prioritize this ahead of correctness.

---

# 55. Design for future distributed/offload support without implementing it now

Do not bake assumptions that:

```text
all topology lives on one GPU
all neuron state lives on one GPU
```

Eventually connectivity may be source- or destination-sharded.

However, do NOT implement distributed sparse runtime in this prototype.

Just keep:

```text
storage representation
logical sparse object
connection semantics
```

separate enough that placement can be added later.

Persistent global state may later use DTensor while hot kernels operate on local tensors.

---

# 56. Deliverables

At the end, provide:

## A. Architecture note

Short document describing:

- current implementation
- target architecture
- final public APIs
- orientation conventions
- batching semantics
- conversion semantics
- training spike semantics
- dynamic topology semantics
- torch.compile boundary

---

## B. Working code

Not pseudocode only.

At minimum implement phases 1–3 robustly, and as much of later phases as practical.

Prefer several reviewable commits rather than one giant patch.

---

## C. Migration table

Example:

```text
current / legacy surface       target ownership
-----------------------------------------------------------
SparseConnection               SparseConnection + Sparse runtime
ConstrainedWeight              grouped weight semantics
SciPy/Torch conversion paths   sparse.asarray(...) adapters
make_hetersynapse_conn         legacy lowering / semantic adapter
backend-specific execution     Planner + BackendRegistry
```

Use actual repository names after audit.

---

## D. Test results

Report:

- existing test status
- new tests
- eager correctness
- gradient correctness
- torch.compile correctness
- conversion round-trips
- batching
- rewiring if implemented

---

## E. Known limitations

Be explicit.

Examples:

```text
N-D COO only partially implemented
CSF not implemented
different-pattern batched CSR uses reference fallback
no CUDA custom sparse-push kernel yet
soft Deep R not implemented
receptor routing still uses expanded reference lowering
```

Do not hide missing pieces behind abstractions.

---

# 57. Decision priority

When making a tradeoff, use this order:

1. mathematical correctness
2. PyTorch/autograd/torch.compile composability
3. simple and familiar public API
4. compatibility with existing btorch models
5. clean separation of semantics and execution
6. extensibility
7. performance
8. advanced compiler sophistication

Do not introduce large compiler abstractions before the basic APIs are proven.

---

# 58. Final architectural invariants

Treat these as design tests.

### Invariant 1

```text
Projection != Sparse
```

### Invariant 2

```text
Sparse != sparse format
```

### Invariant 3

```text
ConnectionRule != execution representation
```

### Invariant 4

```text
Procedural != Sparse
```

### Invariant 5

```text
Synapse metadata != matrix format
```

### Invariant 6

```text
network batch != sample batch
```

### Invariant 7

```text
logical training spike != sparse execution indices
```

### Invariant 8

```text
topology mutation != sparse format
```

### Invariant 9

```text
format != algorithm != backend
```

### Invariant 10

```text
torch.compile(model)
```

is the only ordinary user-facing compilation lifecycle.

### Invariant 11

```text
one Plan → one complete RouteSpec
```

Preparation, execution, and backward compatibility are resolved atomically;
they are not independently selected low-level kernels.

### Invariant 12

```text
Hints = workload expectations
```

Backend choice, algorithm choice, task strategy, and tuning parameters remain
runtime implementation concerns.

### Invariant 13

```text
planning and preparation happen outside the recurrent timestep
```

Stable connections reuse their selected route and prepared state until a
versioned planning dependency changes.

### Invariant 14

```text
CUDA Graph replay does not plan, allocate, synchronize, or switch routes
```

Every change that invalidates the selected route or prepared allocation must
invalidate the capture or be rejected explicitly.

---

# 59. Start by doing this

Do not immediately implement the entire design.

First return an audit containing:

1. exact relevant btorch files/classes/functions
2. current orientation and shape conventions
3. current sparse conversion path
4. current autograd path
5. current `torch.compile` failure/success points
6. current checkpoint/device-move risks
7. existing tests
8. a concrete file-by-file migration plan
9. which parts of this specification should be implemented in the first patch
10. any contradiction between this specification and current btorch public behavior

Then begin the first implementation patch.

When there is a conflict between architectural elegance and preserving important current behavior, preserve behavior first and introduce a clear deprecation/migration layer rather than silently breaking user code.
