# Sparse Connectivity

This guide covers the two packages that replace the former sparse layers of
`btorch.models.linear`:

| Package | Purpose | Needed for |
|---|---|---|
| `btorch.sparse` | Sparse arrays with a SciPy/PyTorch-like API: create, convert, multiply. | Any numerical work with sparse matrices. |
| `btorch.models.connection` | `SparseConnection`, an `nn.Module` that maps pre-synaptic activity to post-synaptic input; `Synapse`, the description of weights, receptors and delays; `Projection` and the connection rules, which build a connection from two populations; `HardDeepR` rewiring. | Building models. |
| `btorch.sparse.runtime` | Execution layer: registered operators, kernel backends, planner. | Nothing in model code; see [Runtime](#runtime-advanced). |

If you are porting code that used `SparseConn` or `SparseConstrainedConn`, go
to [Migration](#migration-from-sparseconn).

All Python blocks on this page run in order as one script.

## `btorch.sparse` in five minutes

A `Sparse` array stores only some of its entries. Three storage formats exist:
`COO` (one coordinate tuple per entry), `CSR` (compressed rows) and `CSC`
(compressed columns). `Sparse` is the common interface; it is a plain Python
object that holds tensors, not a `torch.Tensor` subclass.

### Create

```python
import torch
from btorch import sparse

# Entries A[0, 2] = 3 and A[1, 0] = 5 of a 2 x 3 matrix.
A = sparse.from_edges(
    rows=torch.tensor([0, 1]),
    cols=torch.tensor([2, 0]),
    values=torch.tensor([3.0, 5.0]),
    shape=(2, 3),
)
print(A)  # COO(shape=(2, 3), nnz=2, dtype=torch.float32)

# The same matrix through the format-specific constructors.
A_coo = sparse.coo(torch.tensor([[0, 1], [2, 0]]), torch.tensor([3.0, 5.0]), shape=(2, 3))
A_csr = sparse.csr(
    torch.tensor([0, 1, 2]), torch.tensor([2, 0]), torch.tensor([3.0, 5.0]), shape=(2, 3)
)

# From the non-zero entries of a dense tensor (never done implicitly).
A_dense = sparse.from_dense(torch.tensor([[0.0, 0.0, 3.0], [5.0, 0.0, 0.0]]))
```

`nnz` is the number of stored entries. Duplicate coordinates are allowed in
COO and are summed by every operation, as in SciPy and PyTorch.

### Multiply

Products follow standard linear algebra: a matrix of shape `(M, N)` maps a
length-`N` vector to a length-`M` vector.

```python
x = torch.tensor([1.0, 0.0, 2.0])
print(A @ x)  # tensor([6., 5.])

# A @ X follows torch.matmul: X is [..., N, K] and the result is [..., M, K].
X = torch.rand(3, 4)
assert (A @ X).shape == (2, 4)

# matvec applies A along the LAST axis of a stack of vectors [..., N].
batch = torch.rand(8, 3)
assert A.matvec(batch).shape == (8, 2)

# rmatvec applies the transpose without building it: [..., M] -> [..., N].
assert A.rmatvec(torch.rand(8, 2)).shape == (8, 3)
v = torch.tensor([1.0, 2.0])
assert torch.equal(v @ A, A.rmatvec(v))  # tensor([10., 0., 3.])
```

Use `matvec` for activity tensors such as `[time, batch, n_neuron]`; use `@`
when you want `torch.matmul` shape rules. `sparse.matmul`, `sparse.matvec` and
`sparse.rmatvec` are the function forms. Products are differentiable with
respect to both the stored values and the dense operand. The product of two
sparse arrays is not implemented.

### Convert: `tocsr()` versus `as_csr()`

- `tocoo()`, `tocsr()`, `tocsc()` **convert**. They may sort entries and sum
  duplicates, and they return `self` when the array already has that format.
- `as_coo()`, `as_csr()`, `as_csc()` **assert**. They return `self` if the
  array is stored in that format and raise `TypeError` otherwise. They never
  allocate.

Format-specific fields such as `indptr` only exist on the concrete class, so
obtain it first with one of the two. `coalesce()` keeps the format and returns
an array with sorted indices in which duplicate coordinates are summed; it
exists for `COO`, `CSR` and `CSC`.

```python
csr = A.tocsr()
print(csr.indptr, csr.indices, csr.data)  # tensor([0, 1, 2]) tensor([2, 0]) tensor([3., 5.])
assert csr.as_csr() is csr

try:
    A.as_csr()  # A is stored as COO
except TypeError as err:
    print(err)

# Transposing CSR gives CSC over the same arrays (no copy).
assert A.tocsr().T.format == "csc"
assert torch.equal(A.to_dense(), A.tocsc().to_dense())
```

Both naming schemes are available: SciPy names (`indptr`, `indices`, `data`,
`row`, `col`) and PyTorch names (`crow_indices()`, `col_indices()`,
`values()`).

### SciPy and PyTorch interop

`sparse.asarray` (alias `sparse.as_sparse`) accepts a btorch `Sparse`, a
PyTorch sparse tensor or a SciPy sparse array/matrix. Conversions keep the
format where possible and never transpose or densify. A dense tensor is
rejected; use `sparse.from_dense`.

```python
import scipy.sparse

S = scipy.sparse.random(5, 4, density=0.4, format="csr", random_state=0)

B = sparse.asarray(S)  # same as sparse.from_scipy(S)
assert B.format == "csr" and B.shape == (5, 4)
assert sparse.asarray(S, format="coo", dtype=torch.float32).format == "coo"

# Back to SciPy (CPU, detached) and to PyTorch (keeps autograd history).
assert (B.to_scipy() != S).nnz == 0
T = B.to_torch()  # torch.sparse_csr tensor
assert T.layout == torch.sparse_csr
assert sparse.from_torch(T).shape == (5, 4)
```

SciPy data is `float64`; pass `dtype=` to cast. `to_scipy` only accepts
unbatched matrices. `to_torch(layout="bsr", blocksize=...)` and
`to_scipy(format="bsr")` export block storage, which is not a native btorch
format.

### Shapes and batches

The logical shape of a sparse array is `[*batch, *sparse, *dense]`:

| Group | Meaning | Accessors |
|---|---|---|
| batch | Independent sparse arrays, for example an ensemble of networks. | `batch_shape`, `batch_dim()` |
| sparse | The dimensions stored sparsely (two for a matrix). | `sparse_shape`, `sparse_dim()` |
| dense | A dense payload attached to every stored entry. | `dense_shape`, `dense_dim()` |

A batch comes in two kinds.

```python
# 1. Shared pattern: one set of indices, values [G, nnz]. Works for COO, CSR, CSC.
shared = B.with_values(torch.rand(4, B.nnz))
assert shared.shape == (4, 5, 4) and shared.batch_shape == (4,)

# 2. Different patterns: sparse.stack builds a batched COO without padding.
S2 = scipy.sparse.random(5, 4, density=0.2, format="csr", random_state=1)
ragged = sparse.stack([sparse.asarray(S), sparse.asarray(S2)])
assert ragged.shape == (2, 5, 4) and ragged.format == "coo"

# Batch dimensions align with the LEADING dimensions of the input of matvec.
x = torch.rand(4, 7, 4, dtype=torch.float64)  # [G, samples, N]
assert shared.matvec(x).shape == (4, 7, 5)
# A leading dimension of size 1 broadcasts the same samples to every member.
assert shared.matvec(x[:1]).shape == (4, 7, 5)
```

Batch members are never combined implicitly with input samples: an input with
fewer leading dimensions than the array has batch dimensions raises a
`ValueError`. `sparse.stack` returns a shared-pattern batch when all members
have identical indices.

### `sparse.einsum` (experimental)

`sparse.einsum(subscripts, A, *operands)` contracts one sparse array with dense
tensors. The result equals `torch.einsum(subscripts, A.to_dense(), *operands)`,
but `A.to_dense()` is never built: time and memory are proportional to the
number of stored entries. It is the tool for arrays with more than two sparse
dimensions, which `@` does not handle.

```python
# The 2 x 3 matrix A from above: a batched matrix-vector product.
xb = torch.tensor([[1.0, 0.0, 2.0]])
assert torch.equal(sparse.einsum("mn,bn->bm", A, xb), torch.tensor([[6.0, 5.0]]))
assert sparse.einsum("mn->", A) == 8.0  # no dense operand: sum of the entries

# A third-order sparse array (one row of indices per dimension) contracted
# with two dense tensors.
T3 = sparse.coo(
    torch.tensor([[0, 1, 1], [2, 0, 2], [1, 1, 0]]),
    torch.tensor([1.0, 2.0, 3.0]),
    shape=(2, 3, 2),
)
u3, v3 = torch.rand(4, 3), torch.rand(4, 2)
out3 = sparse.einsum("ijk,bj,bk->bi", T3, u3, v3)
assert torch.allclose(out3, torch.einsum("ijk,bj,bk->bi", T3.to_dense(), u3, v3))
```

The function is experimental and its interface may change. What is supported:

- Exactly one sparse operand, and it comes first. It may have any format and
  any number of batch, sparse and dense dimensions; its subscript has one
  letter per dimension of `A.shape`.
- Any number of dense operands (including none).
- A dense output, in any order of the letters, with or without `->`.

What raises instead: an ellipsis (`...`), a second sparse operand, and a letter
used with two different sizes (size-1 dimensions do not broadcast). A sparse
output is not available. The reasons and the cost model are in
[Sparse einsum: scope and what a compiler would add](../design/sparse_einsum_notes.md).

### Linear operators

A *linear operator* is anything that maps a vector of length `N` to a vector
of length `M` linearly. A sparse array is one way to do that. The module
`btorch.sparse.operator` holds the others: operators that compute the product
from a formula and never store a matrix. They are not `Sparse` arrays: they
have no `nnz`, no format and no indices.

| Operator | Matrix | Cost of a product |
|---|---|---|
| `ConstantOperator((M, N), value)` | every entry equals `value` (all-to-all) | `O(M + N)` |
| `DiagonalOperator(d)` | `diag(d)` (one-to-one) | `O(N)` |
| `LowRankOperator(U, V)` | `U @ V.T` with `U [M, r]`, `V [N, r]` | `O((M + N) r)` |
| `ImplicitOperator((M, N), matvec=f)` | whatever the callable `f` computes | that of `f` |

All of them derive from `LinearOperator` and share the product interface of a
sparse array: `op.matvec(x)` applies the operator along the last axis of
`x [..., N]`, `op @ x` follows `torch.matmul`, and `op.rmatvec(x)` applies the
transpose. Operators combine lazily, with each other and with sparse arrays:
`+` and `-` give a sum, `@` a composition, multiplication or division by a
number a scaled operator, and `.T` the transpose. No matrix is formed by any
of these.

```python
from btorch.sparse.operator import (
    ConstantOperator,
    DiagonalOperator,
    ImplicitOperator,
    LowRankOperator,
)

ones = ConstantOperator((4, 4), 1.0)
eye = DiagonalOperator(torch.ones(4))
low = LowRankOperator(torch.rand(4, 2), torch.rand(4, 2))
# Only matvec is given: a circular shift.
shift = ImplicitOperator((4, 4), matvec=lambda t: t.roll(1, -1))

# All-to-all without the diagonal, plus a low-rank part and a sparse matrix.
ring_edges = sparse.from_edges(
    torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([1.0, 2.0]), (4, 4)
)
op = 0.5 * (ones - eye) + low + ring_edges
dense_op = 0.5 * (torch.ones(4, 4) - torch.eye(4)) + low.to_dense() + ring_edges.to_dense()

xo = torch.rand(8, 4)
assert torch.allclose(op.matvec(xo), xo @ dense_op.T)
assert torch.allclose(op @ xo[0], dense_op @ xo[0])
assert torch.allclose((ones @ ring_edges).matvec(xo), xo @ (torch.ones(4, 4) @ ring_edges.to_dense()).T)
```

Not every operator can do everything. A *capability* is one of `"matvec"`,
`"matmat"`, `"rmatvec"`, `"enumerate_edges"` and `"materialize_sparse"`;
`op.supports(name)` tells whether it is available, and using an unavailable
one raises `NotImplementedError` naming it. A sum or product supports a
capability only if all of its parts do.

```python
# Entries of a structured operator can be listed on request ...
assert (ones - eye).supports("materialize_sparse")
assert (ones - eye).tocsr().shape == (4, 4)
# ... but a low-rank matrix is dense, and a product would need sparse x sparse.
assert not low.supports("materialize_sparse") and not op.supports("materialize_sparse")
assert op.to_dense().shape == (4, 4)  # explicit, never implicit

# An ImplicitOperator has exactly the capabilities whose callable was given.
assert shift.supports("matvec") and not shift.supports("rmatvec")
try:
    shift.T
except NotImplementedError as err:
    print(err)
```

In a model an operator is wrapped in a connection, so that it is called like
every other connection (`current = conn(spikes)`, any leading dimensions).
`OperatorConnection(op)` does that for any object with a `shape` and a
`matvec`; `StructuredConnection` (closed-form operators) and
`ImplicitConnection` (procedural operators) are the two named kinds, and they
behave identically. `HybridConnection([...])` adds the outputs of several
connections between the same two populations.

```python
from btorch.models.connection import (
    HybridConnection,
    ImplicitConnection,
    SparseConnection,
    StructuredConnection,
)

inhibition = StructuredConnection(-0.2 * (ones - eye))  # global, no self-inhibition
ring = ImplicitConnection(shift)
local = SparseConnection(ring_edges)

hybrid = HybridConnection([local, inhibition, ring])
assert (hybrid.n_pre, hybrid.n_post) == (4, 4)
reference = xo @ (ring_edges.to_dense() - 0.2 * (torch.ones(4, 4) - torch.eye(4))).T
assert torch.allclose(hybrid(xo), reference + xo.roll(1, -1), atol=1e-6)
assert torch.allclose(torch.compile(hybrid, fullgraph=True)(xo), hybrid(xo), atol=1e-6)
```

Three points to keep in mind:

- **Inside compiled code call `op.matvec(x)`, not `op @ x`.** The Dynamo
  tracer of PyTorch 2.11 does not dispatch `@` to a user-defined object, so
  `op @ x` and `A @ x` fail under `torch.compile(..., fullgraph=True)`. The
  connections above call `matvec`. For a sparse array, `A.matvec(x)` and the
  function forms `sparse.matvec(A, x)` / `sparse.matmul(A, x)` also work in
  compiled code; the function forms accept sparse arrays only, not operators.
- **Operators are plain objects, not modules.** A tensor held by an operator
  (`d`, `U`, `V`, a tensor `value`) is not a buffer or parameter of the
  connection that wraps it: `connection.to(device)` does not move it, and it
  does not appear in `parameters()` or in the `state_dict`. Create the tensors
  on the target device, and register a learnable tensor on your own module:

```python
from torch import nn


class GlobalInhibition(nn.Module):
    """All-to-all inhibition with one learnable gain."""

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.gain = nn.Parameter(torch.tensor(-0.2))

    def forward(self, spikes):
        # Building the operator is free; it follows the device of the gain.
        return ConstantOperator((self.n, self.n), self.gain).matvec(spikes)


gain_module = GlobalInhibition(4)
gain_module(xo).sum().backward()
assert gain_module.gain.grad is not None
```

- **Rules can produce operators.** A [`Projection`](#connection-rules-and-projection)
  with an all-to-all or one-to-one rule and one fixed weight builds a
  `StructuredConnection` by itself.

## Orientation conventions

Two conventions for a connectivity matrix are in use, and they are transposes
of each other. "Source" (pre-synaptic) neurons send spikes; "target"
(post-synaptic) neurons receive current.

| Name | Shape | Entry `[i, j]` | Product | Where it is used |
|---|---|---|---|---|
| operator, `"dst_src"` | `(n_post, n_pre)` | weight from source `j` to target `i` | `current = A @ spikes` | `btorch.sparse`, the `SparseConnection(...)` constructor, `conn.to_sparse()` |
| adjacency, `"src_dst"` | `(n_pre, n_post)` | weight from source `i` to target `j` | `current = spikes @ W` | connectome matrices, `btorch.connectome.connection`, `SparseConnection.from_adjacency(...)` (default) |

Rules:

- `btorch.sparse` never transposes. What you put in is what is multiplied.
- `SparseConnection.from_adjacency(W)` is the only place where a transpose
  happens, and only for `orientation="src_dst"` (the default).
- Whatever the orientation of the input, `conn(x)` maps `[..., in_features]`
  to `[..., out_features]`.

```python
from btorch.models.connection import SparseConnection

# W[i, j]: weight from source i to target j (3 sources, 2 targets).
W = scipy.sparse.coo_array(([0.5, -1.0], ([0, 2], [1, 0])), shape=(3, 2))
spikes = torch.tensor([[1.0, 0.0, 1.0]])

conn = SparseConnection.from_adjacency(W)  # orientation="src_dst"
assert (conn.n_pre, conn.n_post) == (3, 2)
assert torch.allclose(conn(spikes), spikes @ torch.tensor(W.toarray(), dtype=torch.float32))

# The same connection from the operator A = W.T, shape (n_post, n_pre).
conn_op = SparseConnection(W.T)
conn_op2 = SparseConnection.from_adjacency(W.T, orientation="dst_src")
assert torch.equal(conn_op(spikes), conn(spikes))
assert torch.equal(conn_op2(spikes), conn(spikes))
```

## `SparseConnection` and `Synapse`

`SparseConnection` stores an explicit list of edges. A `Synapse` describes
what the edges do: their weight, the receptor channel they act on and their
transmission delay. The three are independent of each other.

Three constructors exist:

| Constructor | Input |
|---|---|
| `SparseConnection(A, synapse)` | operator `(n_post, n_pre)` |
| `SparseConnection.from_adjacency(W, synapse, orientation="src_dst")` | adjacency matrix |
| `SparseConnection.from_edges(pre, post, n_pre, n_post, synapse, values=None)` | edge list; weights default to 1 |

All accept `bias=` (a `[out_features]` tensor that becomes a parameter),
`hints=`, `device=` and `dtype=`. The matrix may be a btorch `Sparse`, a
PyTorch sparse tensor or a SciPy sparse array. Weights from SciPy input and
from integer matrices use the PyTorch default dtype. Duplicate edges are
summed.

Per-edge arrays inside a `Synapse` (weights, receptors, delays, group ids) are
aligned with the stored entries of the matrix you pass, in the order you pass
them. The connection reorders them together with the edges.

### Weights

| `Synapse(weight=...)` | Result | Trainable |
|---|---|---|
| `None` (default) | `EdgeWeight` initialised from the matrix values | yes |
| a number | `ConstantWeight`: the same weight on every edge | no |
| a `[n_edge]` tensor | `EdgeWeight` with these values | no |
| an `nn.Parameter` | `EdgeWeight` with these values | follows `requires_grad` |
| a `Weight` module | used as given, for example `ConstrainedWeight` | module-defined |

The weight module is `conn.weight`; calling it returns the effective weight of
every edge.

```python
from torch import nn
from btorch.models.connection import ConstrainedWeight, Synapse

rng_matrix = scipy.sparse.random(6, 6, density=0.4, format="csr", random_state=0)
rng_matrix.data -= 0.5  # mixed signs

per_edge = SparseConnection.from_adjacency(rng_matrix)  # trainable per-edge
assert isinstance(per_edge.weight.value, nn.Parameter)

constant = SparseConnection.from_adjacency(rng_matrix, Synapse(weight=0.1))
assert not list(constant.parameters())

fixed = SparseConnection.from_adjacency(
    rng_matrix, Synapse(weight=torch.ones(rng_matrix.nnz))
)
assert not list(fixed.parameters())
```

`ConstrainedWeight` ties edges into groups. Each edge has a fixed `base`
weight and each group one learnable `scale`; the effective weight of edge `e`
is `base[e] * scale[group[e]]`, so the edges of a group keep their relative
strengths.

```python
import numpy as np

# Group ids as a matrix with the same orientation and pattern as the
# connection matrix. Ids in a matrix are 1-based because 0 means "no entry".
group_matrix = rng_matrix.copy()
group_matrix.data = np.random.default_rng(0).integers(1, 4, rng_matrix.nnz).astype(float)

tied = SparseConnection.from_adjacency(
    rng_matrix, Synapse(weight=ConstrainedWeight(group=group_matrix))
)
assert tied.weight.scale.shape == (3,)  # the only trainable tensor
print(tied.weight.group_info())  # DataFrame: group_id, num_connections, scale

tied.weight.set_scale(0, 2.0)  # double every edge of group 0
by_group = tied.weight.weights_by_group()  # {group id: effective weights}

# Alternatively pass 0-based ids as a [n_edge] tensor aligned with the entries.
ids = torch.from_numpy(group_matrix.data).long() - 1
tied2 = SparseConnection.from_adjacency(rng_matrix, Synapse(weight=ConstrainedWeight(group=ids)))
assert torch.equal(tied2.weight.group, tied.weight.group)
```

### Dale's law

Dale's law states that all outgoing synapses of a neuron have the same sign.
btorch enforces the per-edge form: an edge keeps the sign it had at
construction.

- `Synapse(dale=True)` with per-edge weights stores the reference sign in
  `conn.weight.sign`. `constrain()` sets weights that crossed zero to zero.
- `ConstrainedWeight(group=..., dale=True)` keeps every scale non-negative.
- `Synapse(weight=module, dale=True)` with a `Weight` module switches the
  `dale` flag of that module on, so
  `Synapse(weight=ConstrainedWeight(group=...), dale=True)` is the same as the
  previous line. A module without a `dale` flag (`ConstantWeight`) raises a
  `ValueError`. With `dale=False` the module is used as given.

The projection onto the constraint is not part of the forward pass. Call
`btorch.models.constrain.constrain_net(model)` after every optimizer step; it
calls `constrain()` on every weight module of the model. For a single
connection, `conn.constrain()` does the same. `dale` defaults to `False`.

```python
from btorch.models.constrain import constrain_net

dale = SparseConnection.from_adjacency(rng_matrix, Synapse(dale=True))
optimizer = torch.optim.SGD(dale.parameters(), lr=10.0)

loss = dale(torch.rand(4, 6)).sum()
loss.backward()
optimizer.step()
constrain_net(dale)  # weights that changed sign are now 0

w, s = dale.weight.value, dale.weight.sign
assert bool((w * s >= 0).all())
```

### Receptors and delays

A receptor is an input channel of the target neuron (for example AMPA and
GABA); a delay is the number of time steps a spike takes to arrive. Both are
attributes of an edge:

```python
pre = torch.tensor([0, 1, 2, 2])
post = torch.tensor([1, 2, 0, 1])
synapse = Synapse(
    receptor=torch.tensor([0, 1, 0, 1]),  # channel on the target, per edge
    delay=torch.tensor([0, 2, 1, 0]),  # time steps, per edge
    n_receptor=2,
    n_delay=3,
)
routed = SparseConnection.from_edges(pre, post, n_pre=3, n_post=3, synapse=synapse)
assert routed.in_features == 3 * 3  # n_pre * n_delay
assert routed.out_features == 3 * 2  # n_post * n_receptor
print(routed.edge_table())  # dict with pre, post, weight, receptor, delay
```

An int instead of a tensor assigns the same receptor or delay to all edges.
`n_receptor` and `n_delay` default to the largest id plus one. The connection
stores the edge list with its attributes. Its input and output are flat
layouts derived from them:

| Side | Size | Index of (neuron, attribute) | Produced / consumed by |
|---|---|---|---|
| input | `n_pre * n_delay` | `pre * n_delay + delay` | `SpikeHistory.get_flattened(n_delay)` |
| output | `n_post * n_receptor` | `post * n_receptor + receptor` | `HeterSynapsePSC`; or `out.reshape(..., n_post, n_receptor)` |

Delay `d` reads the spikes from `d` steps ago, so delay 0 is the current step.

```python
from btorch.models.history import SpikeHistory

history = SpikeHistory(n_neuron=3, max_delay_steps=3)
history.init_state(batch_size=1)
for _ in range(4):
    history.update((torch.rand(1, 3) < 0.5).float())

current = routed(history.get_flattened(3))  # [1, n_post * n_receptor]
per_receptor = current.reshape(1, 3, 2)  # [batch, n_post, n_receptor]
```

The helpers in `btorch.connectome.connection` (`make_hetersynapse_conn`,
`expand_conn_for_delays`, `stack_hetersynapse`, ...) are unchanged. They
return a physically expanded adjacency matrix of shape
`(n_pre * n_delay, n_post * n_receptor)` that uses exactly these layouts. Two
ways to use such a matrix give the same result:

- `SparseConnection.from_adjacency(expanded)` treats it as an ordinary matrix
  between `n_pre * n_delay` sources and `n_post * n_receptor` targets.
- `SparseConnection.from_hetersynapse(expanded, n_receptor=..., n_delay=...)`
  decodes the expansion into per-edge `receptor` and `delay` attributes, so
  `conn.n_pre`, `conn.n_post`, `conn.edge_table()` and the checkpoint describe
  neurons and not expanded rows and columns.

```python
import pandas as pd
from btorch.connectome.connection import make_hetersynapse_conn
from btorch.models import environ, functional
from btorch.models.synapse import AlphaPSC, HeterSynapsePSC

neurons = pd.DataFrame({"simple_id": range(4), "EI": ["E", "E", "I", "I"]})
edges = pd.DataFrame(
    {
        "pre_simple_id": [0, 1, 2, 3, 0],
        "post_simple_id": [1, 2, 3, 0, 2],
        "syn_count": [3, 1, 2, 4, 1],
        "delay_steps": [0, 1, 2, 0, 1],
    }
)
expanded, receptor_idx = make_hetersynapse_conn(
    neurons, edges, receptor_type_col="EI", delay_col="delay_steps", n_delay_bins=3
)
n_receptor = len(receptor_idx)  # (pre type, post type) pairs: 4
assert expanded.shape == (4 * 3, 4 * n_receptor)

hetero = SparseConnection.from_hetersynapse(expanded, n_receptor=n_receptor, n_delay=3)
flat = SparseConnection.from_adjacency(expanded)
assert (hetero.n_pre, hetero.n_post) == (4, 4) and (flat.n_pre, flat.n_post) == (12, 16)
z = torch.rand(2, 12)
assert torch.allclose(hetero(z), flat(z))

# HeterSynapsePSC buffers the spike history itself and sums over receptors.
with environ.context(dt=1.0):
    psc = HeterSynapsePSC(
        n_neuron=4,
        n_receptor=n_receptor,
        receptor_type_index=receptor_idx,
        linear=hetero,
        base_psc=AlphaPSC,
        tau_syn=5.0,
        max_delay_steps=3,
    )
    functional.init_net_state(psc, batch_size=2)
    out = psc(torch.ones(2, 4))  # [batch, n_neuron]
assert out.shape == (2, 4)
```

A `ConstrainedWeight` group matrix passed to `from_hetersynapse` uses the same
expanded layout as the connection matrix (this is what
`make_hetersynapse_constrained_conn` returns).

## Connection rules and `Projection`

Instead of a matrix, a connection can be specified the way NEST does it: two
populations, a *rule* that says which pairs of neurons are connected, and a
`Synapse` that says what the connections do. `Projection` combines the three
and builds the connection.

A rule returns an edge list `(pre, post)` in population-local indices and
nothing else. Two terms from NEST: an *autapse* is a connection of a neuron
onto itself, a *multapse* is a pair that is connected more than once.

| Rule | Which pairs are connected |
|---|---|
| `OneToOne()` | source `i` to target `i`; needs equal population sizes |
| `AllToAll(allow_autapses=True)` | every source to every target |
| `FixedIndegree(k, allow_autapses=True, allow_multapses=True)` | every target receives exactly `k` connections from random sources |
| `FixedOutdegree(k, allow_autapses=True, allow_multapses=True)` | every source makes exactly `k` connections onto random targets |
| `PairwiseBernoulli(p, allow_autapses=True)` | every pair independently with probability `p` |
| `DistanceDependent(pre_pos, post_pos, probability, max_distance=None)` | every pair independently with probability `probability(distance)` |
| `FromEdges(pre, post, values=None)` | the given edge list |
| `FromSparse(A, orientation="post_pre")` | the stored entries of a sparse matrix; `"post_pre"` is the operator `(n_post, n_pre)`, `"pre_post"` the adjacency `(n_pre, n_post)` |

`allow_autapses=False` is only defined when a population projects onto itself
(`n_pre == n_post`). All randomness comes from the `generator` argument, so a
seeded `torch.Generator` reproduces the edges.

```python
from btorch.models.connection import (
    AllToAll,
    FixedIndegree,
    FixedOutdegree,
    FromSparse,
    Projection,
    Synapse,
)
from btorch.models.neurons.lif import LIF

exc, inh = LIF(n_neuron=80), LIF(n_neuron=20)

# Every inhibitory neuron receives 10 excitatory inputs, two steps delayed.
e_to_i = Projection(
    pre=exc,
    post=inh,
    rule=FixedIndegree(10),
    synapse=Synapse(weight=0.1, delay=2),
    generator=torch.Generator().manual_seed(0),
)
assert (e_to_i.n_pre, e_to_i.n_post) == (80, 20)
assert (e_to_i.in_features, e_to_i.out_features) == (80 * 3, 20)  # delays 0..2

# Mutual inhibition without self-connections.
i_to_i = Projection(inh, inh, AllToAll(allow_autapses=False), Synapse(weight=-0.5))
assert i_to_i(torch.ones(20)).tolist() == [-0.5 * 19] * 20
```

`pre` and `post` are neuron counts or modules that know their size (`size` or
`n_neuron`, as every btorch neuron model has); only the size is read. A
projection is an `nn.Module` and is called like a connection; the connection it
built is `proj.connection`.

**Synapse arrays align with the rule's edges.** Entry `e` of a per-edge
weight, delay, receptor or group array belongs to edge `e` of
`rule.edges(n_pre, n_post, generator=...)`. To build such arrays, ask the rule
for its edges with a generator of the same seed as the one given to the
projection:

```python
out_rule = FixedOutdegree(3, allow_multapses=False)
rule_pre, rule_post = out_rule.edges(80, 20, generator=torch.Generator().manual_seed(1))

# Even sources act on receptor 0, odd sources on receptor 1, with other weights.
by_source = Projection(
    80,
    20,
    out_rule,
    Synapse(weight=0.1 * (1 + rule_pre % 2), receptor=rule_pre % 2),
    generator=torch.Generator().manual_seed(1),
)
table = by_source.connection.edge_table()
assert torch.equal(table["receptor"], table["pre"] % 2)
assert torch.allclose(table["weight"], 0.1 * (1 + table["pre"] % 2))
```

Without a synapse, `FromEdges` and `FromSparse` use the values they carry as
trainable per-edge weights, and all other rules start from trainable unit
weights. A weight given in the synapse replaces the values of the rule.

```python
# The 3 x 2 adjacency W of the orientation section (rows are sources).
from_matrix = Projection(3, 2, FromSparse(sparse.asarray(W, dtype=torch.float32), "pre_post"))
assert torch.allclose(from_matrix(spikes), conn(spikes))
assert isinstance(from_matrix.connection.weight.value, nn.Parameter)
```

**Multapses are merged.** Parallel edges that agree on source, target,
receptor and delay become one stored edge. Their weights add: per-edge weights
are summed, and a fixed scalar weight counts with the multiplicity (a pair
connected three times carries three times the weight). Edges between the same
pair with different receptors or delays stay separate. Merged edges must
belong to the same `ConstrainedWeight` group; otherwise a `ValueError` is
raised.

**Rules never decide the execution format.** A rule is not a sparse array and
has no format, algorithm or backend; the same rule object can be used for
several projections. How a projection is realised follows from the synapse:

- If the rule has a closed form (`OneToOne`, `AllToAll`, `PairwiseBernoulli`
  with `p == 1`) and the synapse is one fixed number without receptor and
  delay, the projection builds a `StructuredConnection` over a
  [linear operator](#linear-operators). No edge is stored, and the
  `state_dict` is empty.
- In every other case (trainable or per-edge weights, receptors, delays, any
  other rule) it builds a `SparseConnection` with explicit edges.
  `realization="sparse"` forces this.

```python
assert isinstance(i_to_i.connection, StructuredConnection)
assert isinstance(e_to_i.connection, SparseConnection)
explicit = Projection(
    inh, inh, AllToAll(allow_autapses=False), Synapse(weight=-0.5), realization="sparse"
)
assert explicit.connection.nnz == 20 * 19
xi = torch.rand(5, 20)
assert torch.allclose(explicit(xi), i_to_i(xi), atol=1e-5)
```

A random rule draws new edges every time a projection is constructed. The
edges are part of the `state_dict` of a sparse projection, so
`load_state_dict` restores the saved network. It requires the same number of
stored edges, as described under [Checkpoints](#checkpoints): that holds for
`FixedIndegree` and `FixedOutdegree` with `allow_multapses=False` and for the
deterministic rules. With multapses, `PairwiseBernoulli` or
`DistanceDependent` the number of stored edges is random, so construct the
receiving projection with the same generator seed as the saved one.

## Structural plasticity

*Structural plasticity* changes which neurons are connected during training.
`HardDeepR` implements hard Deep Rewiring (Bellec et al., 2018) for a
`SparseConnection` with trainable per-edge weights.

The connection keeps a fixed number of *edge slots* (`conn.nnz`). Slot `k` has
a weight `w_k` and a reference sign `s_k`. After an optimizer step, a slot is
*dormant* if its weight reached or crossed zero relative to that sign
(`s_k * w_k <= 0`). The connection of a dormant slot is removed and the slot
is reused for a new connection at a random position that is currently
unconnected, with a small initial weight (`init`). The number of connections
therefore never changes, no tensor changes shape and the weight parameter is
never replaced, so the optimizer and compiled graphs stay valid.

```python
from btorch.models.connection import HardDeepR, HardDeepROptions

plastic_matrix = scipy.sparse.random(30, 30, density=0.1, format="csr", random_state=0)
plastic_matrix.data -= 0.5  # mixed signs
plastic = SparseConnection.from_adjacency(plastic_matrix, Synapse(dale=True))

# 1. The controller, 2. torch.compile, 3. attach to the optimizer.
rewire = HardDeepR(
    plastic, HardDeepROptions(l1=1e-3), generator=torch.Generator().manual_seed(0)
)
plastic_fast = torch.compile(plastic, fullgraph=True)
plastic_opt = torch.optim.Adam(plastic.parameters(), lr=0.05)
rewire.attach(plastic_opt)

n_slot, weight_param = plastic.nnz, plastic.weight.value
edges_before = plastic.indices.clone()
xp, target = torch.rand(16, 30), torch.rand(16, 30)
for _ in range(20):
    plastic_opt.zero_grad()
    ((plastic_fast(xp) - target) ** 2).mean().backward()
    plastic_opt.step()  # gradient step, then the rewiring update

assert rewire.n_rewired > 0 and plastic.topology_version > 0
assert not torch.equal(plastic.indices, edges_before)  # edges moved ...
assert plastic.nnz == n_slot and plastic.weight.value is weight_param  # ... slots did not
assert torch.allclose(plastic_fast(xp), plastic(xp), atol=1e-5)  # no stale graph
```

- **`attach(optimizer)`** registers a hook that runs after every
  `optimizer.step()`. It applies the optional L1 shrink (`l1`) and random walk
  (`noise`) of Deep R to the weights and then, on every `every`-th step,
  rewires the dormant slots. The shrink and the random walk are skipped on
  steps where the weights received no gradient. `rewire.step()` runs the
  structural update by hand; `rewire.detach()` removes the hook.
- **Signs.** With `Synapse(dale=True)` the reference sign is the `sign` buffer
  of the weight, and a new connection takes the sign of its source neuron.
  Without Dale's law the controller tracks the sign each weight had when its
  slot was last activated, and a new connection gets a random sign.
- **Optimizer state.** A rewired slot is a new connection, so its entries in
  the optimizer's per-parameter state need a policy (`optimizer_state`).
  `"neutral"` (the default) zeroes first-moment state (Adam's `exp_avg`, SGD
  momentum) and sets second-moment state (`exp_avg_sq`, RMSprop's
  `square_avg`) to the mean over the other slots, so a new connection takes
  steps of the same size as established ones. `"reset"` zeroes everything;
  because Adam's step count is global, the first step of a reset slot is then
  about 2.5 times larger than normal (10 times for RMSprop). `"keep"` leaves
  the state of the removed connection in place.
- **When no free position is left.** On a small or nearly full layer there
  may be fewer admissible unconnected positions than dormant slots. The
  update then places as many as it can; the remaining slots stay where they
  are with weight exactly zero and are retried at later updates.
  `rewire.n_unplaced` reports how many are waiting, and a `RuntimeWarning`
  is emitted once per controller. A position vacated in an update becomes
  available at the next one.
- **Construct the controller before `torch.compile`.** The constructor calls
  `conn.enable_rewiring()`, which makes the traced forward pass independent of
  the order of the edges, so rewiring never changes the traced graph.
- **`conn.topology_version`** is incremented by every update that moves at
  least one edge (and by `load_state_dict`). The execution layouts are rebuilt
  at that moment, outside the forward pass. `rewire.n_rewired` counts the
  rewired slots. The low-level call is
  `conn.set_edges_(slots, post, pre)`.
- **What is persisted.** The edges, weights and Dale signs are in the
  `state_dict` of the connection. The controller has its own
  `state_dict()` / `load_state_dict()` (counters, tracked signs, waiting
  slots and the state of its random generator). To resume a run, load the
  connection, then the optimizer, then the controller; training then
  continues exactly as an uninterrupted run would.

Other options (`candidate=` to restrict where new connections may appear,
`allow_autapses`, `sign`, `max_tries`) are described in the docstring of
`HardDeepROptions`. Weights other than a trainable, unbatched `EdgeWeight`
(`ConstantWeight`, `ConstrainedWeight`, a network batch) are refused.

Soft Deep R is not implemented. In soft Deep R a dormant connection keeps its
parameter and can reactivate at its old position, which needs one parameter
for every possible connection, that is dense `n_post * n_pre` memory. This is
what a sparse connection avoids.

## Network batch versus sample batch

Two kinds of leading dimension occur in the input of a connection:

- A **sample batch** is a set of inputs for one network: time steps, trials,
  mini-batch entries. A connection accepts any number of them:
  `[..., in_features]`.
- A **network batch** is a set of `G` networks (an ensemble or a parameter
  sweep). It exists when the weights have shape `[G, n_edge]`, or when the
  matrix is a `sparse.stack` of different patterns. The input is then
  `[G, ..., in_features]`: network dimensions come first.

```python
# Four networks that share the pattern of rng_matrix but not the weights.
ensemble_matrix = sparse.asarray(rng_matrix, dtype=torch.float32)
ensemble_matrix = ensemble_matrix.with_values(torch.randn(4, rng_matrix.nnz))
ensemble = SparseConnection.from_adjacency(ensemble_matrix)
assert ensemble.batch_shape == (4,)

x = torch.rand(4, 10, 32, 6)  # [G, time, batch, n_pre]
assert ensemble(x).shape == (4, 10, 32, 6)

# The same samples for every network: a leading dimension of size 1 broadcasts.
shared_x = torch.rand(10, 32, 6)
assert ensemble(shared_x.unsqueeze(0)).shape == (4, 10, 32, 6)

try:
    ensemble(torch.rand(6))  # no network dimension
except ValueError as err:
    print(err)
```

A network dimension is never inferred from, or merged with, a sample
dimension. With `ConstrainedWeight`, batch the scales (`scale=` of shape
`[G, n_group]`), not the base weights. Receptors and delays are not supported
for a batch of different patterns.

For a batch of different patterns, `conn.edge_table()` and `conn.to_sparse()`
report network coordinates: the table has an additional `network` entry with
the batch index of every edge, `pre` and `post` are neuron ids within that
network, and `to_sparse()` returns a batched COO of shape
`[*batch, n_post, n_pre]`. (The stored `indices` buffer instead addresses the
block-diagonal operator over all networks that is used for execution.)

```python
ragged_conn = SparseConnection(ragged.to(torch.float32))  # two 5 x 4 operators
ragged_table = ragged_conn.edge_table()
assert set(ragged_table["network"].flatten().tolist()) == {0, 1}
assert int(ragged_table["post"].max()) < 5 and int(ragged_table["pre"].max()) < 4
assert ragged_conn.to_sparse().shape == (2, 5, 4)
assert ragged_conn(torch.rand(2, 7, 4)).shape == (2, 7, 5)
```

## `torch.compile`

There is no separate preparation step and no backend to choose: compile the
connection, or the model that contains it, with `torch.compile`.
`fullgraph=True` works on CPU and GPU and does not need `torch_sparse`. The
compiled graph contains one registered operator per connection call; the
sparse layout is not visible to the tracer.

```python
compiled = torch.compile(per_edge, fullgraph=True)
x = torch.rand(5, 6)
assert torch.allclose(compiled(x), per_edge(x), atol=1e-6)
```

Gradients with respect to the weights and the input are available in compiled
and uncompiled mode. The input stays a dense tensor, so surrogate-gradient
training needs no change.

## Checkpoints

The `state_dict` of a connection contains the edge list and everything needed
to reproduce the weights. Execution layouts (CSR pointers, permutations) are
derived, never saved, and rebuilt after `load_state_dict`.

| Key | Content | Present |
|---|---|---|
| `indices` | `[2, n_edge]`: target (row 0) and source (row 1) neuron of every edge | always |
| `receptor`, `delay` | `[n_edge]` ids | when the synapse sets them |
| `bias` | `[out_features]` | when `bias=` is given |
| `weight.value` | per-edge weights | `EdgeWeight`, `ConstantWeight` (scalar) |
| `weight.multiplicity` | `[n_edge]` number of parallel input edges merged into each edge | `ConstantWeight` |
| `weight.sign` | reference signs for Dale's law | `EdgeWeight` with `dale=True` |
| `weight.scale`, `weight.group`, `weight.base` | scales, group ids, base weights | `ConstrainedWeight` |

```python
print(list(dale.state_dict()))  # ['indices', 'weight.value', 'weight.sign']
print(list(tied.state_dict()))  # ['indices', 'weight.scale', 'weight.group', 'weight.base']
print(list(routed.state_dict()))  # ['indices', 'receptor', 'delay', 'weight.value']

# A module with the same number of edges, but other edges, weights and signs.
other = rng_matrix.copy()
other.indices = np.random.default_rng(1).permutation(other.indices)
other.data = -other.data
restored = SparseConnection.from_adjacency(other, Synapse(dale=True))
restored.load_state_dict(dale.state_dict())

x = torch.rand(3, 6)
assert torch.equal(restored(x), dale(x))
assert torch.equal(restored.weight.sign, dale.weight.sign)
```

To load a checkpoint, construct the connection with the same options and the
same number of edges, then call `load_state_dict`. The edges, Dale signs and
group structure all come from the checkpoint, not from the matrix used to
construct the module.

Population sizes and the numbers of receptors and delays are constructor
arguments and are not stored. A checkpoint whose `indices`, `receptor` or
`delay` address a neuron or channel outside the module is rejected by
`load_state_dict` with an error, and that tensor is not copied into the
module.

## Performance hints and `explain`

`Hints` describe how a connection will be used. They never change results,
only how the product is executed.

```python
from btorch.sparse import Hints

hinted = SparseConnection.from_adjacency(rng_matrix, hints=Hints(expected_density=0.001))
x = (torch.rand(8, 6) < 0.001).float()
print(hinted.explain(x))
```

`expected_density` is the expected fraction of non-zero entries in the input
(the spike probability per neuron and step). Without it the connection always
uses the "pull" algorithm, which visits every edge. With an expected density
at or below the planner's limit for the device it uses "adaptive-push", which
visits only the outgoing edges of active sources. The limits are measured
crossover points and differ between CPU and GPU; they are listed in
`btorch.sparse.runtime.planner.push_max_density`. A network batch of weights
always uses pull.

`conn.explain(x)` (also available as `sparse.explain(conn, x)`) returns a text
report: the measured density of `x`, the edge count, the chosen algorithm with
the reason, the kernel backend and the weight module. It exists for
`SparseConnection`; `x` is optional. `conn.set_hints(Hints(...))` changes the
hints of an existing connection. `conn.to_sparse()` returns the current effective operator as a
`Sparse` (`"dst_src"` by default, `"src_dst"` for the adjacency orientation).

`Hints` also has `expected_calls` and `expected_batch`; the planner does not
use them yet.

## Runtime (advanced)

`btorch.sparse.runtime` holds the registered `torch.library` operators
(`csr_propagate`, `spike_propagate`), the kernel backend registry
(`registry`), the planner and the representation cache. Model code does not
import it.

A *kernel* is a low-level routine such as the CSR product; a *backend* is one
implementation of it. The runtime picks the highest-priority available
backend. The default backend `"aten"` uses only PyTorch. If `torch_sparse` is
installed it is registered as an additional, lower-priority backend for the
CSR product, and can be selected for a block of code:

```python
from btorch.sparse import runtime

print(runtime.registry.available("csr_matvec", "cpu"))  # ['aten'] or ['aten', 'torch_sparse']

with runtime.use_backend("aten"):
    y = per_edge(torch.rand(5, 6))
```

`use_backend` is a process-wide override meant for debugging and benchmarks.
A backend that does not implement a kernel is skipped for that kernel. A name
that no kernel knows raises a `ValueError` listing the registered backends.

### Triton backend (CUDA)

On a GPU, the kernels of the destination-driven ("pull") path have a second
implementation written in [Triton](https://github.com/triton-lang/triton),
registered as backend `"triton"` for the device type `"cuda"`:

| Kernel | Role |
|---|---|
| `"csr_matvec"` | the destination-driven ("pull") product of the forward pass |
| `"edge_grad"` | the gradient with respect to the edge weights in the backward pass |

The backend is available when the `triton` package can be imported and CUDA is
present. It then has a higher priority than `"aten"`, so it is the default on
CUDA and nothing has to be selected. The Triton kernels handle `float32`
data; for other dtypes and for cases they do not cover they call the ATen
kernel themselves. `"aten"` remains the fallback when Triton is missing, the
default on CPU, and the reference the Triton kernels are tested against.

```python
print(runtime.registry.available("csr_matvec", "cuda"))  # best first

if torch.cuda.is_available():
    gpu_conn = SparseConnection.from_adjacency(rng_matrix).to("cuda")
    print(sparse.explain(gpu_conn))  # "backend = triton" when Triton is installed

    xg = torch.rand(5, 6, device="cuda")
    y_default = gpu_conn(xg)
    with runtime.use_backend("aten"):  # force the reference kernels
        assert runtime.registry.name("csr_matvec", "cuda") == "aten"
        y_aten = gpu_conn(xg)
    assert torch.allclose(y_default, y_aten, atol=1e-5)
```

The backend is looked up at every call, so `use_backend` affects existing
connections. The backend line of `conn.explain()` is the one
recorded when the connection was last planned (at construction, by
`.to(device)` and by `set_hints`); inside a `use_backend` block,
`runtime.registry.name(kernel, device)` gives the backend in effect.

<!-- TODO(sparse): push backend -->

## Migration from `SparseConn`

`SparseConn`, `SparseConstrainedConn`, `BaseSparseConn`, `SparseBackend`,
`available_sparse_backends` and the `sparse_backend=` argument were removed
from `btorch.models.linear` without a compatibility layer. `DenseConn` is
unchanged.

| Old | New |
|---|---|
| `SparseConn(conn, bias=b, enforce_dale=E)` (default `enforce_dale=True`) | `SparseConnection.from_adjacency(conn, Synapse(dale=E), bias=b)` (default `dale=False`) |
| `sparse_backend=`, `available_sparse_backends()` | removed; the runtime chooses. Advanced: `btorch.sparse.runtime.use_backend(...)` |
| `layer.magnitude` / `layer.initial_sign` / `get_sparse_matrix()` | `conn.weight.value` / `conn.weight.sign` (now persistent) / `conn.to_sparse("src_dst")` |
| `SparseConstrainedConn(conn, constraint, enforce_dale=E)` | `SparseConnection.from_adjacency(conn, Synapse(weight=ConstrainedWeight(group=constraint, dale=E)))` |
| `.magnitude` / `.initial_weight` / `._constraint_scatter_indices` | `conn.weight.scale` / `.base` / `.group` |
| `get_group_info` / `set_group_magnitude` / `get_weights_by_group` | `conn.weight.group_info` / `set_scale` / `weights_by_group` |
| `SparseConstrainedConn.from_hetersynapse(conn, constraint, receptor_idx)`, `constraint_info`, `persist_initial_weight` | `SparseConnection.from_adjacency(...)` (expanded layout) or `SparseConnection.from_hetersynapse(conn, Synapse(weight=ConstrainedWeight(group=constraint)), n_receptor=..., n_delay=...)` (semantic per-edge receptor/delay); the rest removed |
| `state_dict` keys `magnitude`, `indices` | `indices`, `weight.value` (+ `weight.sign`) or `weight.scale`/`weight.group`/`weight.base`, plus `receptor`/`delay` |

Points that need attention when porting:

- **Dale's law is opt-in.** `SparseConn` enforced it by default; a plain
  `SparseConnection.from_adjacency(conn)` does not. Pass `Synapse(dale=True)`
  to keep the old behaviour.
- **Checkpoints are not compatible.** The `state_dict` keys changed (last row
  of the table), so a checkpoint written by the old layers does not load into
  a `SparseConnection`. There is no converter.
- **`torch_sparse` is optional.** It is no longer required for
  `torch.compile(..., fullgraph=True)`.
- **Receptor index tables are not stored on the connection.** Keep the
  `receptor_type_index` DataFrame returned by the connectome helpers next to
  the model, as `HeterSynapsePSC` already requires.

```python
# Before:
#   from btorch.models.linear import SparseConn
#   layer = SparseConn(conn_matrix, enforce_dale=True, sparse_backend="native")
#   w = layer.get_sparse_matrix()
# After:
layer = SparseConnection.from_adjacency(rng_matrix, Synapse(dale=True))
w = layer.to_sparse("src_dst").to_scipy()
assert w.shape == rng_matrix.shape
```

See also the design note
[Sparse and connectivity subsystem](../design/sparse_connectivity.md).
