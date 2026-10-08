# Sparse Connectivity

This guide covers the packages that replace the former sparse layers of
`btorch.models.linear`:

| Package | Purpose | Needed for |
|---|---|---|
| `btorch.models.connection` | `SparseConnection`, an `nn.Module` that maps pre-synaptic activity to post-synaptic input; `Synapse`, the description of weights, receptors and delays; `Projection` and the connection rules, which build a connection from two populations; `HardDeepR` rewiring. | Building models. [Part 1](#part-1-building-models). |
| `btorch.sparse` | Sparse arrays with a SciPy/PyTorch-like API: create, convert, multiply. Matrix-free linear operators. | Numerical work with sparse matrices. [Part 2](#part-2-numerical-api-btorchsparse). |
| `btorch.sparse.runtime` | Execution layer: registered operators, kernel backends, planner. | Nothing in model code. [Part 3](#runtime-advanced). |

If you are porting code that used `SparseConn` or `SparseConstrainedConn`, go
to [Migration](#migration-from-sparseconn).

All Python blocks on this page run in order as one script.

## Part 1: building models

### Orientation: which way round?

A *source* (pre-synaptic) neuron sends spikes; a *target* (post-synaptic)
neuron receives current. A connectivity matrix can be written in two ways that
are transposes of each other, and every function in this guide that takes or
returns a matrix names the one it means with one of two strings:

| String | Shape | Entry `[i, j]` | Product | Where it comes from |
|---|---|---|---|---|
| `"pre_post"` | `(n_pre, n_post)` | weight from source `i` to target `j` | `current = spikes @ W` | connectome tables, `btorch.connectome.connection` |
| `"post_pre"` | `(n_post, n_pre)` | weight from source `j` to target `i` | `current = A @ spikes` | linear algebra, `btorch.sparse` |

A `"pre_post"` matrix is called an *adjacency matrix* below and a `"post_pre"`
matrix an *operator*. Whatever the orientation of the input, a connection is
called the same way: `conn(x)` maps `[..., in_features]` to
`[..., out_features]`.

```python
import scipy.sparse
import torch
from torch import nn

from btorch import sparse
from btorch.models.connection import SparseConnection, Synapse

# W[i, j]: weight from source i to target j (3 sources, 2 targets).
W = scipy.sparse.coo_array(([0.5, -1.0], ([0, 2], [1, 0])), shape=(3, 2))
spikes = torch.tensor([[1.0, 0.0, 1.0]])

conn = SparseConnection.from_adjacency(W)  # orientation="pre_post" is the default
assert (conn.n_pre, conn.n_post) == (3, 2)
assert torch.allclose(conn(spikes), spikes @ torch.tensor(W.toarray(), dtype=torch.float32))

# The same connection from the operator A = W.T, shape (n_post, n_pre).
conn_op = SparseConnection(W.T)
conn_op2 = SparseConnection.from_adjacency(W.T, orientation="post_pre")
assert torch.equal(conn_op(spikes), conn(spikes))
assert torch.equal(conn_op2(spikes), conn(spikes))
```

The table below lists every entry point and accessor that has an orientation.
"Fixed" means there is no argument to change it.

| Entry point or accessor | What it takes or returns | Orientation |
|---|---|---|
| `SparseConnection(A, synapse)` | matrix `A` of shape `(n_post, n_pre)` | fixed `"post_pre"` |
| `SparseConnection.from_adjacency(W, synapse, orientation=...)` | matrix `W` | default `"pre_post"`: `(n_pre, n_post)`; `"post_pre"` on request |
| `SparseConnection.from_edges(pre, post, n_pre, n_post, synapse)` | two index tensors and two sizes | source first, then target |
| `SparseConnection.from_hetersynapse(M, ...)` | expanded matrix `(n_pre * n_delay, n_post * n_receptor)` | fixed `"pre_post"` |
| `FromSparse(A, orientation=...)` (rule) | matrix `A` | default `"pre_post"`; `"post_pre"` on request |
| `FromEdges(pre, post, values)` (rule), `rule.edges(n_pre, n_post)` | index tensors; `edges` returns `(pre, post)` | source first, then target |
| `Projection(pre, post, rule, synapse)` | two populations | source first, then target |
| `ConstrainedWeight(group=G)` with a matrix `G` | group-id matrix | the orientation of the matrix the connection is built from |
| `conn.to_sparse(orientation=None)` | effective weight matrix | default `conn.orientation`, the orientation the connection was built from |
| `conn.orientation` | `"pre_post"` or `"post_pre"` | `"post_pre"` after `SparseConnection(A)`; the argument after `from_adjacency`; `"pre_post"` after `from_edges`, `from_hetersynapse` and for the connection of a `Projection` |
| `conn.indices` | `[2, n_edge]` buffer | row 0 is the **target**, row 1 the **source**: `[post, pre]` |
| `conn.pre`, `conn.post` | `[n_edge]` source and target of every edge | named, no order to remember |
| `conn.edge_table()` | dict with the keys `pre`, `post`, `weight`, ... | named |
| `conn.find_edges(pre, post)` | edge slots between the given neurons | source first, then target |
| `conn.set_edges_(slots, *, pre, post)` | new source and target of edge slots | keyword-only |
| `HardDeepROptions(candidate=f)` | `f(pre, post)` returning a boolean mask | source first, then target |
| `sparse.from_edges(rows, cols, values, shape)` | numerical sparse array, no neuron semantics | used as a product, `rows` are the outputs (targets) and `cols` the inputs (sources) |
| `A @ x`, `A.matvec(x)` on a `Sparse` | product | `A` is `(M, N)` and maps length `N` to length `M`: `"post_pre"` |
| `Linear(in_features, out_features, weight=W)` | dense `W` of shape `(out_features, in_features)` | standard PyTorch `nn.Linear`; its `mask` has the same shape as `weight` |

Three rules summarise the table:

- `btorch.sparse` never transposes. What you put in is what is multiplied.
- In `btorch.models.connection`, every function that takes two index tensors
  or two populations takes the source first. Matrices default to
  `"pre_post"`, with one exception: the bare constructor `SparseConnection(A)`
  takes the operator.
- The stored buffer `conn.indices` is `[post, pre]`. Prefer `conn.pre` and
  `conn.post`.

!!! warning "A wrong orientation of a square matrix is not an error"
    For a recurrent connection `n_pre == n_post`, so a matrix in the wrong
    orientation has a valid shape. The connection is built without complaint
    and computes with the transposed network. Check one known edge:
    `conn.find_edges(pre=i, post=j)` must be non-empty for a pair you know is
    connected from `i` to `j`.

    The two product forms of a sparse array differ in the same silent way.
    `A @ X` follows `torch.matmul` and reads the vectors along the
    second-to-last axis of `X`; `A.matvec(X)` reads them along the last axis.
    For a square `A` and a square batch `X` both run and return different
    numbers.

```python
# Edge 0 -> 1 with weight 0.5, edge 2 -> 0 with weight -1.0.
assert conn.find_edges(pre=0, post=1).numel() == 1
assert conn.find_edges(pre=1, post=0).numel() == 0

# A square operator and a square batch: both products run, the numbers differ.
A_square = sparse.from_edges(
    rows=torch.tensor([0, 1]), cols=torch.tensor([1, 2]), values=torch.tensor([1.0, 2.0]), shape=(3, 3)
)
X_square = torch.rand(3, 3)
assert torch.allclose(A_square @ X_square, A_square.to_dense() @ X_square)  # columns are vectors
assert torch.allclose(A_square.matvec(X_square), X_square @ A_square.to_dense().T)  # rows are vectors
assert not torch.allclose(A_square @ X_square, A_square.matvec(X_square))
```

Use `matvec` (or a connection) for activity tensors such as
`[time, batch, n_neuron]`.

### `SparseConnection`

`SparseConnection` stores an explicit list of edges. Three constructors exist:

| Constructor | Input |
|---|---|
| `SparseConnection(A, synapse)` | operator `(n_post, n_pre)` |
| `SparseConnection.from_adjacency(W, synapse, orientation="pre_post")` | adjacency matrix, or the operator with `orientation="post_pre"` |
| `SparseConnection.from_edges(pre, post, n_pre, n_post, synapse, values=None)` | edge list; weights default to 1; ids outside the populations raise `ValueError` |

A matrix may be a btorch `Sparse`, a PyTorch sparse tensor or a SciPy sparse
array. All constructors accept:

- `bias=`: `True` for a zero-initialised bias of shape `[out_features]`, or a
  tensor of that shape as the initial value (a wrong shape raises
  `ValueError`). The bias is a parameter.
- `hints=`: see [Performance hints](#performance-hints-and-explain).
- `device=` and `dtype=`. Without `dtype`, weights from btorch and PyTorch
  input keep their dtype; weights from SciPy input and from integer matrices
  use the PyTorch default dtype.

An *edge slot* is one stored edge. The connection keeps the slots sorted by
target and then source, and merges duplicate entries of the input into one
slot whose weight is their sum. The following members describe the slots:

```python
print(conn.nnz)  # number of edge slots: 2
print(conn.pre, conn.post)  # tensor([2, 0]) tensor([0, 1])
print(conn.edge_table())  # {'pre': ..., 'post': ..., 'weight': ...}; the weight is a detached snapshot
print(conn.orientation)  # 'pre_post': how this connection was built

# to_sparse() returns the effective weights in the orientation of the input ...
assert conn.to_sparse().shape == (3, 2)
assert torch.equal(conn.to_sparse().to_dense(), torch.tensor(W.toarray(), dtype=torch.float32))
# ... or in the orientation you ask for.
assert conn.to_sparse("post_pre").shape == (2, 3)
assert conn_op.to_sparse().shape == (2, 3)  # built from the operator

with_bias = SparseConnection.from_adjacency(W, bias=True)
assert with_bias.bias.shape == (2,) and not with_bias.bias.any()
```

Each call to `to_sparse()` snapshots the current effective weights while
preserving their autograd history, so gradients flow through
them to the weight parameters.

### `Synapse` and weights

A `Synapse` describes what the edges of a connection do: their weight, the
receptor channel they act on and their transmission delay. The three are
independent of each other.

Per-edge arrays inside a `Synapse` (weights, receptors, delays, group ids) are
aligned with the edges as you pass them: the stored entries of the matrix in
their stored order, or the edge list. The connection reorders them together
with the edges.

| `Synapse(weight=...)` | Weight module | Trainable |
|---|---|---|
| `None` (default) | `EdgeWeight` initialised from the matrix values | yes |
| a number | `ConstantWeight`: the same weight on every edge | no |
| a `[n_edge]` tensor | `EdgeWeight` with these values | no |
| an `nn.Parameter` of shape `[n_edge]` | `EdgeWeight` with these values | follows `requires_grad` |
| a callable `f(n_edge) -> Tensor` | `EdgeWeight` holding `f(n_edge)`, called once the edges are known | yes |
| a `torch.distributions.Distribution` | `EdgeWeight` holding `dist.sample((n_edge,))` | yes |
| a `Weight` module | that module, for example `ConstrainedWeight` | module-defined |

A *weight module* is an `nn.Module` that owns everything trainable about the
strength of the edges. It is `conn.weight`. Calling it returns the *effective*
weight of every edge slot, a tensor of shape `[n_edge]` aligned with
`conn.pre` and `conn.post`. The tensors it stores are its attributes:

| Expression | Meaning |
|---|---|
| `conn.weight` | the weight module (`EdgeWeight`, `ConstantWeight`, `ConstrainedWeight`) |
| `conn.weight()` | effective weight of every edge slot, `[n_edge]` (`[G, n_edge]` for a [network batch](#network-batch-versus-sample-batch)) |
| `conn.weight.shape`, `.dtype`, `.device` | those of `conn.weight()` |
| `conn.weight.value` | `EdgeWeight`: the stored per-edge weights, an `nn.Parameter` if trainable and a buffer otherwise. `ConstantWeight`: the scalar |
| `conn.weight.sign` | `EdgeWeight` with Dale's law: the reference sign of every edge |
| `conn.weight.scale`, `.base`, `.group` | `ConstrainedWeight`: learnable scale per group, fixed weight per edge, group id per edge |

#### Recipes

```python
rng_matrix = scipy.sparse.random(6, 6, density=0.4, format="csr", random_state=0)
rng_matrix.data -= 0.5  # mixed signs
n_edge = rng_matrix.nnz

# 1. Matrix values as trainable per-edge weights (the default).
per_edge = SparseConnection.from_adjacency(rng_matrix)
assert isinstance(per_edge.weight.value, nn.Parameter)

# 2. One fixed number on every edge: nothing to train.
constant = SparseConnection.from_adjacency(rng_matrix, Synapse(weight=0.1))
assert not list(constant.parameters())

# 3. A tensor: fixed per-edge weights (a buffer), aligned with the stored entries.
fixed = SparseConnection.from_adjacency(rng_matrix, Synapse(weight=torch.ones(n_edge)))
assert not list(fixed.parameters())

# 4. A callable or a distribution: trainable weights drawn once the edges are known.
drawn = SparseConnection.from_adjacency(rng_matrix, Synapse(weight=lambda n: 0.1 * torch.randn(n)))
sampled = SparseConnection.from_adjacency(
    rng_matrix, Synapse(weight=torch.distributions.Uniform(0.0, 0.2))
)
assert drawn.weight.shape == sampled.weight.shape == (n_edge,)
assert isinstance(sampled.weight.value, nn.Parameter)
```

A callable receives the number of edges as supplied, before duplicates are
merged, and its result is aligned with them. A distribution is sampled with
the global PyTorch random generator.

**Optimizer and `nn.init`.** The trainable tensors are ordinary parameters of
the connection, so `conn.parameters()` (or `model.parameters()`) is all an
optimizer needs. To address the weights alone use `conn.weight.value` (or
`conn.weight.scale` for a `ConstrainedWeight`). In-place initialisers work on
the same tensors:

```python
optimizer = torch.optim.Adam([per_edge.weight.value], lr=1e-3)  # same as per_edge.parameters()
nn.init.normal_(per_edge.weight.value, std=0.1)
```

With Dale's law (below) the reference signs are taken when the connection is
built. Initialise such weights through a callable or a distribution instead
of `nn.init`, or the new values are projected back onto the old signs.

**Passing your own `nn.Parameter`.** The connection stores edges sorted by
target and then source. If the parameter you pass is already in that order and
no duplicates had to be merged, the connection adopts it: `conn.weight.value`
is the same object, and an optimizer built from your parameter trains the
connection. Otherwise the values are copied into a new parameter and a
`UserWarning` says so; an optimizer must then use `conn.weight.value`.

```python
import warnings

# Adopted: a CSR operator is already sorted by target, then source.
operator_csr = rng_matrix.T.tocsr()
mine = nn.Parameter(torch.rand(n_edge))
adopted = SparseConnection(operator_csr, Synapse(weight=mine))
assert adopted.weight.value is mine

# Copied: the stored order of a CSR adjacency matrix is source, then target.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    copied = SparseConnection.from_adjacency(rng_matrix, Synapse(weight=mine))
assert copied.weight.value is not mine
assert any("could not be adopted" in str(w.message) for w in caught)
```

The order is kept by `SparseConnection(A)` with a canonical CSR operator, by
`from_adjacency(W)` with a canonical CSC adjacency matrix, and by `from_edges`
with edges sorted by `(post, pre)`. The edges of a random rule in a
[`Projection`](#connection-rules-and-projection) are generally not in that
order. Code that does not control the order should always read the parameter
back from `conn.weight.value`.

**Passing a `Weight` module.** The module you pass *is* `conn.weight`. It
holds the parameters of that one connection, so it can be used once; building
a second connection with it raises a `RuntimeError`. A `Synapse` with any
other kind of weight can be reused for several connections, and each gets its
own weight module.

```python
from btorch.models.connection import EdgeWeight

module = EdgeWeight(torch.rand(n_edge))
shared_synapse = Synapse(weight=module)
first = SparseConnection.from_adjacency(rng_matrix, shared_synapse)
assert first.weight is module
try:
    SparseConnection.from_adjacency(rng_matrix, shared_synapse)
except RuntimeError as err:
    print(err)  # This weight module is already bound to a connection. ...
```

`Synapse(plasticity=...)` is reserved for online plasticity rules. Any value
other than `None` raises `NotImplementedError`.

#### `ConstrainedWeight`: tied weights

`ConstrainedWeight` ties edges into groups. Each edge has a fixed `base`
weight and each group one learnable `scale`; the effective weight of edge `e`
is `base[e] * scale[group[e]]`, so the edges of a group keep their relative
strengths.

```python
import numpy as np

from btorch.models.connection import ConstrainedWeight

# Group ids as a matrix with the same orientation and pattern as the
# connection matrix. Ids in a matrix are 1-based because 0 means "no entry".
group_matrix = rng_matrix.copy()
group_matrix.data = np.random.default_rng(0).integers(1, 4, n_edge).astype(float)

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

#### Dale's law

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
- `Synapse(weight=<number>, dale=True)` raises a `ValueError`: the sign of one
  fixed number cannot change, so there is nothing to enforce.

The projection onto the constraint is not part of the forward pass. Call
`btorch.models.constrain.constrain_net(model)` after every optimizer step; it
calls `constrain()` on every weight module of the model. For a single
connection, `conn.constrain()` does the same. `dale` defaults to `False`.

```python
from btorch.models.constrain import constrain_net

dale = SparseConnection.from_adjacency(rng_matrix, Synapse(dale=True))
dale_opt = torch.optim.SGD(dale.parameters(), lr=10.0)

loss = dale(torch.rand(4, 6)).sum()
loss.backward()
dale_opt.step()
constrain_net(dale)  # weights that changed sign are now 0

w, s = dale.weight.value, dale.weight.sign
assert bool((w * s >= 0).all())
```

#### Receptors and delays

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

| Side | Size | Index of (neuron, attribute) |
|---|---|---|
| input | `n_pre * n_delay` | `pre * n_delay + delay` |
| output | `n_post * n_receptor` | `post * n_receptor + receptor` |

Delay `d` reads the spikes from `d` steps ago, so delay 0 is the current step.
The input layout is the one `SpikeHistory.get_flattened(n_delay)` produces; the
output can be reshaped to `[..., n_post, n_receptor]`.

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
- `SparseConnection.from_hetersynapse(expanded, n_receptor=None, n_delay=1,
  receptor_type_index=None)` decodes the expansion into per-edge `receptor`
  and `delay` attributes, so `conn.n_pre`, `conn.n_post`, `conn.edge_table()`
  and the checkpoint describe neurons and not expanded rows and columns. Pass
  either `n_receptor` or the receptor index table returned by the helper as
  `receptor_type_index`; only the length of the table is used, and the two
  must agree if both are given.

```python
import pandas as pd

from btorch.connectome.connection import make_hetersynapse_conn

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

hetero = SparseConnection.from_hetersynapse(expanded, receptor_type_index=receptor_idx, n_delay=3)
flat = SparseConnection.from_adjacency(expanded)
assert (hetero.n_pre, hetero.n_post) == (4, 4) and (flat.n_pre, flat.n_post) == (12, 16)
assert (hetero.n_receptor, hetero.n_delay) == (n_receptor, 3)
z = torch.rand(2, 12)
assert torch.allclose(hetero(z), flat(z))
assert hetero.to_sparse().shape == expanded.shape  # the expanded "pre_post" layout
```

The connection does not store the receptor index table. Keep it next to the
model to map channel ids to receptor names. A `ConstrainedWeight` group matrix
passed to `from_hetersynapse` uses the same expanded layout as the connection
matrix (this is what `make_hetersynapse_constrained_conn` returns).

### Connection rules and `Projection`

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
| `FromSparse(A, orientation="pre_post")` | the stored entries of a sparse matrix; `"pre_post"` is the adjacency `(n_pre, n_post)`, `"post_pre"` the operator `(n_post, n_pre)` |

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
    StructuredConnection,
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
assert (e_to_i.n_delay, e_to_i.n_receptor) == (3, 1)

# Mutual inhibition without self-connections.
i_to_i = Projection(inh, inh, AllToAll(allow_autapses=False), Synapse(weight=-0.5))
assert i_to_i(torch.ones(20)).tolist() == [-0.5 * 19] * 20
```

`pre` and `post` are neuron counts or modules that know their size (`size` or
`n_neuron`, as every btorch neuron model has); only the size is read. A
projection is an `nn.Module` and is called like a connection. The connection
it built is `proj.connection`; `proj.n_delay`, `proj.n_receptor`,
`proj.weight`, `proj.edge_table()` and `proj.to_sparse()` forward to it. The
last three need explicit edges and raise `AttributeError` for a projection
that was realised without them (see below).

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
table = by_source.edge_table()
assert torch.equal(table["receptor"], table["pre"] % 2)
assert torch.allclose(table["weight"], 0.1 * (1 + table["pre"] % 2))
```

A weight that does not depend on the individual edge needs no such alignment:
pass a number, a callable or a distribution.

Without a synapse, `FromEdges` and `FromSparse` use the values they carry as
trainable per-edge weights, and all other rules start from trainable unit
weights. A weight given in the synapse replaces the values of the rule.

```python
# The 3 x 2 adjacency W of the orientation section (rows are sources).
from_matrix = Projection(3, 2, FromSparse(sparse.asarray(W, dtype=torch.float32)))
assert torch.allclose(from_matrix(spikes), conn(spikes))
assert isinstance(from_matrix.weight.value, nn.Parameter)
assert from_matrix.to_sparse().shape == (3, 2)
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
  with `p == 1`) and the synapse is one fixed number without receptor, delay
  and Dale's law, the projection builds a `StructuredConnection` over a
  [linear operator](#linear-operators). No edge is stored.
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

### Wiring into neuron, PSC and RNN modules

A connection is the `linear` argument of a post-synaptic current (PSC) module
from `btorch.models.synapse`. The PSC module filters the output of the
connection in time; `RecurrentNN` feeds the spikes of a neuron module back
through the PSC module and unrolls the loop over time steps.

The example below builds a recurrent network of 80 excitatory (E) and 20
inhibitory (I) neurons from four projections. One neuron module holds both
populations, E first. A small module splits the spike vector by population,
applies the four projections and concatenates the two target populations
again. Weights are drawn from a distribution (excitatory) and a callable
(inhibitory), and Dale's law keeps their signs during training.

```python
from btorch.models import environ, functional
from btorch.models.connection import PairwiseBernoulli
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC

torch.manual_seed(0)  # seeds the weight draws; the generator below seeds the edges
n_exc, n_inh = 80, 20
n_all = n_exc + n_inh

excitatory = Synapse(weight=torch.distributions.Uniform(0.0, 0.2), dale=True)
inhibitory = Synapse(weight=lambda n: -0.8 * torch.rand(n), dale=True)


class EIRecurrence(nn.Module):
    """Recurrent input of an E/I network: [..., n_exc + n_inh] -> the same."""

    def __init__(self, n_exc, n_inh, generator):
        super().__init__()
        self.n_exc = n_exc
        in_exc = FixedIndegree(8, allow_multapses=False)
        self.ee = Projection(
            n_exc,
            n_exc,
            FixedIndegree(8, allow_autapses=False, allow_multapses=False),
            excitatory,
            generator=generator,
        )
        self.ei = Projection(n_exc, n_inh, in_exc, excitatory, generator=generator)
        self.ie = Projection(n_inh, n_exc, PairwiseBernoulli(0.25), inhibitory, generator=generator)
        self.ii = Projection(
            n_inh,
            n_inh,
            PairwiseBernoulli(0.25, allow_autapses=False),
            inhibitory,
            generator=generator,
        )

    def forward(self, spikes):
        z_exc, z_inh = spikes[..., : self.n_exc], spikes[..., self.n_exc :]
        to_exc = self.ee(z_exc) + self.ie(z_inh)
        to_inh = self.ei(z_exc) + self.ii(z_inh)
        return torch.cat([to_exc, to_inh], dim=-1)


recurrence = EIRecurrence(n_exc, n_inh, torch.Generator().manual_seed(0))
net = RecurrentNN(
    neuron=LIF(n_neuron=n_all, v_threshold=1.0, v_reset=0.0, tau=20.0, tau_ref=2.0, step_mode="s"),
    synapse=ExponentialPSC(n_all, tau_syn=5.0, linear=recurrence, step_mode="s"),
    step_mode="m",  # the module consumes a whole [time, batch, n_all] sequence
    update_state_names=("neuron.v",),
)
functional.init_net_state(net, batch_size=4, dtype=torch.float32)

# The four weight tensors are the only parameters of this network.
assert len(list(net.parameters())) == 4
ei_opt = torch.optim.Adam(net.parameters(), lr=1e-2)
drive = 0.08 + 0.05 * torch.randn(50, 4, n_all)  # external input [time, batch, n_all]

with environ.context(dt=1.0):
    for _ in range(3):
        functional.reset_net(net, batch_size=4)
        net_spikes, states = net(drive)  # [50, 4, 100], {"neuron.v": [50, 4, 100]}
        rate_loss = (net_spikes.mean(dim=0) - 0.05).pow(2).mean()  # target rate per step
        ei_opt.zero_grad()
        rate_loss.backward()  # surrogate gradient through spikes, PSC and connections
        ei_opt.step()
        constrain_net(net)  # Dale's law: project the weights after every step

assert net_spikes.shape == states["neuron.v"].shape == (50, 4, n_all)
for name in ("ee", "ei", "ie", "ii"):
    projection = getattr(recurrence, name)
    assert projection.weight.value.grad is not None
assert bool((recurrence.ee.weight.value >= 0).all()) and bool((recurrence.ie.weight.value <= 0).all())
```

The same `Synapse` object describes two projections each: a `Synapse` with a
distribution or a callable creates a new weight module per connection. The
synapses of this example have no delay.

A connection with receptor channels has `n_post * n_receptor` outputs.
`HeterSynapsePSC` takes such a connection, keeps one PSC state per channel and
sums the channels of every neuron. It needs the receptor index table of the
connectome helper:

```python
from btorch.models.synapse import AlphaPSC, HeterSynapsePSC

# The neurons and edges of the receptor section, without the delay column.
by_type, type_idx = make_hetersynapse_conn(
    neurons, edges.drop(columns="delay_steps"), receptor_type_col="EI"
)
typed = SparseConnection.from_hetersynapse(by_type, receptor_type_index=type_idx)
assert (typed.out_features, typed.n_receptor) == (4 * 4, 4)

with environ.context(dt=1.0):
    receptor_psc = HeterSynapsePSC(
        n_neuron=4,
        n_receptor=typed.n_receptor,
        receptor_type_index=type_idx,
        linear=typed,
        base_psc=AlphaPSC,
        tau_syn=5.0,
    )
    functional.init_net_state(receptor_psc, batch_size=2)
    summed = receptor_psc(torch.ones(2, 4))  # [batch, n_neuron]
assert summed.shape == (2, 4)
```

<!-- TODO(delays): PSC + delayed connection -->

### Network batch versus sample batch

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
ensemble_matrix = ensemble_matrix.with_values(torch.randn(4, n_edge))
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
network, and `to_sparse()` returns a batched COO. The stored `indices` buffer
(and with it `conn.pre` and `conn.post`) instead addresses the block-diagonal
operator over all networks that is used for execution.

```python
S_a = scipy.sparse.random(5, 4, density=0.4, format="csr", random_state=0)
S_b = scipy.sparse.random(5, 4, density=0.2, format="csr", random_state=1)
ragged = sparse.stack([sparse.asarray(S_a, dtype=torch.float32), sparse.asarray(S_b, dtype=torch.float32)])

ragged_conn = SparseConnection(ragged)  # two 5 x 4 operators
ragged_table = ragged_conn.edge_table()
assert set(ragged_table["network"].flatten().tolist()) == {0, 1}
assert int(ragged_table["post"].max()) < 5 and int(ragged_table["pre"].max()) < 4
assert ragged_conn.to_sparse().shape == (2, 5, 4)
assert ragged_conn(torch.rand(2, 7, 4)).shape == (2, 7, 5)
```

### `torch.compile`

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

#### CUDA graphs and `reduce-overhead`

A *CUDA graph* records the GPU kernel launches of a piece of code once and
replays them later without running the Python code again. Replay reads the
contents of the tensors it recorded, but everything decided in Python at
capture time is frozen: which kernels run, with which launch sizes, and on
which device buffers.

For a connection this means:

- **Weight updates are safe.** The weights are read at replay time. An
  optimizer step, `constrain()` or an in-place edit does not invalidate a
  graph.
- **A graph captured around a connection is invalid after** `set_edges_`
  (and so after a `HardDeepR` update that moved an edge), after a
  `load_state_dict` that brings other edges, after `.to(...)` (also
  `.cuda()`, `.float()`, ...) and after a `set_hints` that changes the
  planned algorithm. Replaying it then computes with the old wiring or the
  old execution path, without an error.
- **`conn.capture_version`** is a hashable value that changes on each of
  these events (on every `load_state_dict`, whether or not the edges differ)
  and does not change on weight updates. Compare it before a replay and
  capture again if it differs.
  `btorch.models.cudagraph.capture_versions(model)` collects it for every
  connection of a model.
- **`conn.capture_incompatibility()`** returns `None`, or the reason why the
  connection cannot be captured at all. Today there is one reason: a
  connection with a density hint that is executed by the reference backend
  (algorithm `"adaptive-push"`, see
  [Performance hints](#performance-hints-and-explain)) packs the spikes on
  the host, which a graph cannot record.

`conn.value_version` is the counterpart for weights: it is derived from the
in-place version counters of the weight tensors, so it changes whenever a
weight tensor is written in place (an optimizer step, a `constrain()` that
clamps, a checkpoint load).

```python
from btorch.models.cudagraph import capture_incompatibilities, capture_versions

graph_conn = SparseConnection.from_adjacency(rng_matrix)
captured_at = capture_versions(graph_conn)  # (graph_conn.capture_version,)
weights_at = graph_conn.value_version

with torch.no_grad():
    graph_conn.weight.value.mul_(0.5)  # a weight update ...
assert capture_versions(graph_conn) == captured_at  # ... keeps a captured graph valid
assert graph_conn.value_version != weights_at

slot = torch.tensor([0])
graph_conn.set_edges_(slot, pre=torch.tensor([1]), post=torch.tensor([4]))  # rewiring ...
assert capture_versions(graph_conn) != captured_at  # ... does not

assert capture_incompatibilities(graph_conn) == {}  # no density hint: can be captured
```

`RecurrentNN(cudagraph=True)` checks these versions on every call and
captures again by itself when one of them changed, so rewiring and loading a
checkpoint between calls need no extra code there; it raises with the reason
when a connection reports a capture incompatibility.
<!-- verify: cudagraph re-capture -->

`torch.compile(mode="reduce-overhead")` uses CUDA graphs internally and has no
such check. Two rules follow:

- **Do not combine `mode="reduce-overhead"` with rewiring or with loading a
  different connectivity pattern.** After `set_edges_`, a `HardDeepR` update
  or a `load_state_dict` with other edges, the compiled module keeps replaying
  the graph of the old pattern. Use the default compile mode for such models,
  or `RecurrentNN(cudagraph=True)` for inference.
- **Call the connection once in eager mode before compiling in that mode**, on
  its final device and with its final hints. The first call lets the backend
  finish its one-time setup outside any capture.

When you drive `torch.cuda.CUDAGraph` by hand, do the same and keep the
versions next to the graph:

```python
if torch.cuda.is_available():
    cuda_conn = SparseConnection.from_adjacency(rng_matrix).to("cuda")
    static_x = torch.rand(8, 6, device="cuda")
    assert not capture_incompatibilities(cuda_conn)

    def capture():
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            for _ in range(3):  # warm-up in eager mode, outside the capture
                cuda_conn(static_x)
            with torch.cuda.graph(graph):
                static_y = cuda_conn(static_x)
        return graph, static_y, capture_versions(cuda_conn)

    graph, static_y, versions = capture()
    static_x.copy_(torch.rand(8, 6))  # new input, same buffer
    graph.replay()
    with torch.no_grad():
        assert torch.allclose(static_y, cuda_conn(static_x), atol=1e-6)

    cuda_conn.set_edges_(torch.tensor([0]), pre=torch.tensor([1]), post=torch.tensor([4]))
    if capture_versions(cuda_conn) != versions:  # the old graph is stale
        graph, static_y, versions = capture()
    graph.replay()
    with torch.no_grad():
        assert torch.allclose(static_y, cuda_conn(static_x), atol=1e-6)
```

### Checkpoints

The `state_dict` of a connection contains the edge list, the sizes it was
built with and everything needed to reproduce the weights. Execution layouts
(CSR pointers, permutations) are derived, never saved, and rebuilt after
`load_state_dict`.

| Key | Content | Present |
|---|---|---|
| `indices` | `[2, n_edge]`: target (row 0) and source (row 1) neuron of every edge | always |
| `layout` | `[n_post, n_pre, n_receptor, n_delay]`, followed by the batch shape for a batch of different patterns | always |
| `receptor`, `delay` | `[n_edge]` ids | when the synapse sets them |
| `bias` | `[out_features]` | when `bias=` is given |
| `weight.value` | per-edge weights | `EdgeWeight`, `ConstantWeight` (scalar) |
| `weight.multiplicity` | `[n_edge]` number of parallel input edges merged into each edge | `ConstantWeight` |
| `weight.sign` | reference signs for Dale's law | `EdgeWeight` with `dale=True` |
| `weight.scale`, `weight.group`, `weight.base` | scales, group ids, base weights | `ConstrainedWeight` |

```python
print(list(dale.state_dict()))  # ['indices', 'layout', 'weight.value', 'weight.sign']
print(list(tied.state_dict()))  # ['indices', 'layout', 'weight.scale', 'weight.group', 'weight.base']
print(list(routed.state_dict()))  # ['indices', 'receptor', 'delay', 'layout', 'weight.value']

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

To load a checkpoint, construct the connection with the same options, the
same population sizes and the same number of edges, then call
`load_state_dict`. The edges, Dale signs and group structure all come from the
checkpoint, not from the matrix used to construct the module.

`load_state_dict` validates what it is given, and raises a `RuntimeError`
without copying anything from that checkpoint into the connection when a check
fails. The checks also apply with `strict=False`:

- The population sizes and the numbers of receptors and delays in `layout`
  must equal those of the module.
- `indices`, `receptor` and `delay` must address neurons and channels inside
  the module.
- A checkpoint that has `indices` but no `layout` was written by the removed
  `SparseConn` / `SparseConstrainedConn` layers. It is refused with a message
  that says how to migrate: rebuild the connection from its matrix with
  `SparseConnection.from_adjacency` and copy the weights.

```python
wider = SparseConnection.from_adjacency(
    scipy.sparse.random(7, 6, density=0.4, format="csr", random_state=0)
)
try:
    wider.load_state_dict(per_edge.state_dict())  # 6 sources into a module with 7
except RuntimeError as err:
    print(err)

legacy = {"indices": per_edge.indices.clone(), "magnitude": torch.rand(n_edge)}
try:
    per_edge.load_state_dict(legacy, strict=False)
except RuntimeError as err:
    print(err)
```

### Performance hints and `explain`

`Hints` describe how a connection will be used. They do not change what is
computed, only how the product is executed (see the note on rounding below).

```python
from btorch.sparse import Hints

hinted = SparseConnection.from_adjacency(rng_matrix, hints=Hints(expected_density=0.001))
x = (torch.rand(8, 6) < 0.001).float()
print(hinted.explain(x))  # on CPU: "algorithm = adaptive-push (expected_density 0.001 <= 0.002)"
```

`expected_density` is the expected fraction of non-zero entries in the input
(the spike probability per neuron and step), a number in `[0, 1]`. The
*planner* chooses one of three algorithms from it when the connection is
built, moved to a device, or given new hints:

| Algorithm | When it is planned | What it does |
|---|---|---|
| `"pull"` | no density hint; a hint above the limit of the device; a network batch of weights | Visits every edge: each target sums over its incoming edges. Deterministic. |
| `"push"` | a hint at or below the limit, on a backend that compacts the spikes on the device (Triton on CUDA) | Visits only the outgoing edges of active sources, in one launch without host synchronisation. Used for every call, whatever the density of an individual input. |
| `"adaptive-push"` | a hint at or below the limit, on the reference backend (CPU, or CUDA without Triton) | Packs the active sources on the host and visits only their outgoing edges. Each call whose input is denser than the limit falls back to pull. |

The limits are measured crossover points: 0.2 % on CPU and 2 % on CUDA. They
are stored in `btorch.sparse.runtime.planner.push_max_density`.

Points to know before setting a hint:

- **`"push"` is not bitwise reproducible.** Several sources add into the same
  target concurrently, so the order of the floating-point additions varies
  between runs. Results agree with pull to about `4e-7` relative error. Under
  `torch.use_deterministic_algorithms(True)` the planner plans `"pull"`
  instead; the setting is read when the plan is made, so set it before the
  connection is built or moved.
- **`"push"` takes the hint at its word.** It is used for every input, also
  for one that is far denser than the hint; it stays correct and becomes slower
  than pull.
- **`"push"` hides non-finite weights of silent sources.** Pull multiplies
  every weight with its input, so an `inf` or `nan` weight reaches the output
  even when its source is silent (`inf * 0` is `nan`). Push never visits the
  edges of a silent source. A diverged weight can therefore go unnoticed
  until its source fires.
- **`"adaptive-push"` cannot be captured in a CUDA graph**, because it
  synchronises with the host to pack the spikes (see
  [CUDA graphs](#cuda-graphs-and-reduce-overhead)).
- The gradient is the same deterministic one for all three algorithms.

`conn.explain(x)` (also available as `sparse.explain(conn, x)`) returns a text
report: the measured density of `x`, the edge count, the chosen algorithm with
the reason, the kernel backend and the weight module. `x` is optional.
`conn.set_hints(Hints(...))` changes the hints of an existing connection and
plans again.

`Hints` also has `expected_calls` and `expected_batch` (positive integers);
the planner does not use them yet. A value outside its range raises a
`ValueError` when the `Hints` object is created.

### Structural plasticity

*Structural plasticity* changes which neurons are connected during training.
`HardDeepR` implements hard Deep Rewiring (Bellec et al., 2018) for a
`SparseConnection` with trainable per-edge weights.

The connection keeps a fixed number of edge slots (`conn.nnz`). Slot `k` has
a weight `w_k` and a reference sign `s_k`. After an optimizer step, a slot is
*dormant* if its weight reached or crossed zero relative to that sign
(`s_k * w_k <= 0`). The connection of a dormant slot is removed and the slot
is reused for a new connection at a random position that is currently
unconnected, with a small initial weight (`init`). The number of connections
therefore never changes, no tensor changes shape and the weight parameter is
never replaced, so the optimizer and modules compiled with the default mode of
`torch.compile` stay valid.

```python
from btorch.models.connection import HardDeepR, HardDeepROptions

plastic_matrix = scipy.sparse.random(30, 30, density=0.1, format="csr", random_state=0)
plastic_matrix.data -= 0.5  # mixed signs
plastic = SparseConnection.from_adjacency(plastic_matrix, Synapse(dale=True))

# 1. The controller, 2. torch.compile, 3. attach to the optimizer.
rewire = HardDeepR(
    plastic,
    HardDeepROptions(l1=1e-3, candidate=lambda pre, post: pre < 20),  # new edges from sources 0..19 only
    generator=torch.Generator().manual_seed(0),
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
moved = (plastic.indices != edges_before).any(dim=0)
assert bool(moved.any()) and bool((plastic.pre[moved] < 20).all())  # edges moved, as allowed ...
assert plastic.nnz == n_slot and plastic.weight.value is weight_param  # ... slots did not
assert torch.allclose(plastic_fast(xp), plastic(xp), atol=1e-5)  # no stale graph
```

- **What can be rewired.** `HardDeepR` takes a `SparseConnection`, or a
  `Projection` that was realised as one. The weight must be a trainable,
  unbatched `EdgeWeight`; `ConstantWeight`, `ConstrainedWeight`, fixed weights
  and a network batch are refused.
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
- **Where new connections may appear.** `candidate=f` restricts the positions:
  `f(pre, post)` receives two integer tensors of equal shape, the source and
  the target of each proposed connection, and returns a boolean mask.
  `allow_autapses=False` excludes `pre == post`.
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
  the order of the edges, so rewiring never changes the traced graph. This
  holds for the default compile mode. `mode="reduce-overhead"` must not be
  combined with rewiring, and a hand-captured CUDA graph has to be captured
  again after an update that moved an edge; see
  [CUDA graphs](#cuda-graphs-and-reduce-overhead).
- **`conn.topology_version`** is incremented by every update that moves at
  least one edge (and by `load_state_dict`). The execution layouts are rebuilt
  at that moment, outside the forward pass. `rewire.n_rewired` counts the
  rewired slots.
- **What is persisted.** The edges, weights and Dale signs are in the
  `state_dict` of the connection. The controller has its own
  `state_dict()` / `load_state_dict()` (counters, tracked signs, waiting
  slots and the state of its random generator). To resume a run, load the
  connection, then the optimizer, then the controller; training then
  continues exactly as an uninterrupted run would.

The remaining options (`sign`, `max_tries`, `every`, `init`) are described in
the docstring of `HardDeepROptions`.

The low-level call behind the controller is
`conn.set_edges_(slots, *, pre, post, receptor=None, delay=None)`. The index
arguments are keyword-only. It changes which edge each listed slot stands
for, in place, and rebuilds the execution layouts. Call it outside the
forward pass, and call `conn.enable_rewiring()` before compiling a connection
you rewire by hand.

```python
manual = SparseConnection.from_adjacency(plastic_matrix)
manual.enable_rewiring()
free_slots = manual.find_edges(pre=manual.pre[:2], post=manual.post[:2])[:2]
manual.set_edges_(free_slots, pre=torch.tensor([3, 4]), post=torch.tensor([5, 6]))
assert manual.find_edges(pre=3, post=5).numel() >= 1
assert manual.topology_version == 1
```

Two slots may describe the same pair after such a call; their weights then
add.

Soft Deep R is not implemented. In soft Deep R a dormant connection keeps its
parameter and can reactivate at its old position, which needs one parameter
for every possible connection, that is dense `n_post * n_pre` memory. This is
what a sparse connection avoids.

## Part 2: numerical API (`btorch.sparse`)

`btorch.sparse` is a sparse-array library without neuron semantics. Model
code needs it only to prepare matrices; a `SparseConnection` accepts SciPy and
PyTorch sparse objects directly.

A `Sparse` array stores only some of its entries. Three storage formats exist:
`COO` (one coordinate tuple per entry), `CSR` (compressed rows) and `CSC`
(compressed columns). `Sparse` is the common interface; it is a plain Python
object that holds tensors, not a `torch.Tensor` subclass.

### Create

```python
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
when you want `torch.matmul` shape rules. For a square matrix the two accept
the same input and return different results (see the
[warning](#orientation-which-way-round) in Part 1). `sparse.matmul`,
`sparse.matvec` and `sparse.rmatvec` are the function forms; they accept
sparse arrays and [linear operators](#linear-operators). Products are
differentiable with respect to both the stored values and the dense operand.
The product of two sparse arrays is not implemented.

These products run through a reference implementation (a gather and an
`index_add` over the stored entries). The kernel backends of
[Part 3](#runtime-advanced) are reached only through a `SparseConnection`.

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

Two details of PyTorch COO tensors:

- A PyTorch COO tensor has no notion of a batch. A tensor of shape
  `(G, M, N)` is read as an array with three sparse dimensions unless
  `batch_dim=1` says that it is `G` matrices. `sparse.from_torch` and
  `sparse.as_sparse` both take `batch_dim`.
- An uncoalesced COO tensor that does not require grad keeps its entries as
  stored, in their order and with duplicates, so arrays aligned with the
  entries stay aligned. An uncoalesced tensor that requires grad is coalesced
  on import, because PyTorch exposes its values to autograd only then.

```python
stacked = torch.sparse_coo_tensor(
    torch.tensor([[0, 1], [0, 1], [1, 0]]), torch.tensor([1.0, 2.0]), (2, 2, 2)
)
assert sparse.from_torch(stacked).sparse_dim() == 3
as_batch = sparse.from_torch(stacked, batch_dim=1)
assert as_batch.batch_shape == (2,) and as_batch.sparse_dim() == 2

uncoalesced = torch.sparse_coo_tensor(
    torch.tensor([[1, 0, 1], [0, 1, 0]]), torch.tensor([1.0, 2.0, 3.0]), (2, 2)
)
kept = sparse.from_torch(uncoalesced)
assert kept.nnz == 3 and kept.row.tolist() == [1, 0, 1]  # order and duplicate kept
```

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
stacked_patterns = sparse.stack([sparse.asarray(S), sparse.asarray(S2)])
assert stacked_patterns.shape == (2, 5, 4) and stacked_patterns.format == "coo"

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
transpose. The function forms `sparse.matvec(op, x)`, `sparse.matmul(op, x)`
and `sparse.rmatvec(op, x)` accept operators as well. Operators combine
lazily, with each other and with sparse arrays: `+` and `-` give a sum, `@` a
composition, multiplication or division by a number a scaled operator, and
`.T` the transpose. No matrix is formed by any of these.

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
assert torch.allclose(sparse.matvec(op, xo), op.matvec(xo))
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

#### Operators as connections

In a model an operator is wrapped in a connection, so that it is called like
every other connection (`current = conn(spikes)`, any leading dimensions).
`OperatorConnection(op)` does that for any object with a `shape` and a
`matvec`; `StructuredConnection` (closed-form operators) and
`ImplicitConnection` (procedural operators) are the two named kinds, and they
behave identically. `HybridConnection([...])` adds the outputs of several
connections between the same two populations.

```python
from btorch.models.connection import HybridConnection, ImplicitConnection

inhibition = StructuredConnection(-0.2 * (ones - eye))  # global, no self-inhibition
ring = ImplicitConnection(shift)
local = SparseConnection(ring_edges)

hybrid = HybridConnection([local, inhibition, ring])
assert (hybrid.n_pre, hybrid.n_post) == (4, 4)
reference = xo @ (ring_edges.to_dense() - 0.2 * (torch.ones(4, 4) - torch.eye(4))).T
assert torch.allclose(hybrid(xo), reference + xo.roll(1, -1), atol=1e-6)
assert torch.allclose(torch.compile(hybrid, fullgraph=True)(xo), hybrid(xo), atol=1e-6)
```

An operator itself is a plain object, not a module. The connection that wraps
it registers the tensors the operator holds (`d`, `U`, `V`, a tensor `value`,
and those of the parts of a composite operator): an `nn.Parameter` becomes a
parameter of the connection, every other tensor a buffer. They therefore
follow `connection.to(...)`, appear in `parameters()` and are saved in the
`state_dict`. A learnable operator needs no extra module:

```python
gain = nn.Parameter(torch.tensor(-0.2))
global_inhibition = StructuredConnection(ConstantOperator((4, 4), gain))

assert [name for name, _ in global_inhibition.named_parameters()] == ["operator_0_value"]
assert list(global_inhibition.state_dict()) == ["operator_0_value"]
global_inhibition(xo).sum().backward()
assert gain.grad is not None
```

Two points to keep in mind:

- **Inside compiled code call `op.matvec(x)`, not `op @ x`.** The Dynamo
  tracer of PyTorch 2.11 does not dispatch `@` to a user-defined object, so
  `op @ x` and `A @ x` fail under `torch.compile(..., fullgraph=True)`. The
  connections above call `matvec`. `A.matvec(x)` and the function forms
  `sparse.matvec` / `sparse.matmul` also work in compiled code.
- **Rules can produce operators.** A [`Projection`](#connection-rules-and-projection)
  with an all-to-all or one-to-one rule and one fixed weight builds a
  `StructuredConnection` by itself.

## Runtime (advanced)

`btorch.sparse.runtime` holds the registered `torch.library` operators
(`csr_propagate`, `spike_propagate`), the kernel backend registry
(`registry`), the planner and the representation cache. Model code does not
import it. Its public names are `registry`, `use_backend`, `planner`,
`Planner`, `BackendRegistry`, `KernelCache`, `RepresentationCache`,
`propagate`, `csr_propagate`, `spike_propagate` and `pack_spikes`; the kernel
modules are internal.

A *kernel* is a low-level routine such as the CSR product; a *backend* is one
implementation of it. The runtime picks the highest-priority available
backend. The default backend `"aten"` uses only PyTorch and is the reference.
If `torch_sparse` is installed it is registered as an additional,
lower-priority backend for the CSR product, and can be selected for a block of
code:

```python
from btorch.sparse import runtime

print(runtime.registry.available("csr_matvec", "cpu"))  # ['aten'] or ['aten', 'torch_sparse']

with runtime.use_backend("aten"):
    y = per_edge(torch.rand(5, 6))
```

`use_backend` is a process-wide override meant for debugging and benchmarks.
A backend that does not implement a kernel is skipped for that kernel. A name
that no kernel knows raises a `ValueError` listing the registered backends.

Only a `SparseConnection` reaches these kernels. A product with a `Sparse`
array (`A @ x`, `A.matvec(x)`) always uses the reference implementation of
`btorch.sparse`.

### Triton backend (CUDA)

On a GPU, the kernels have a second implementation written in
[Triton](https://github.com/triton-lang/triton), registered as backend
`"triton"` for the device type `"cuda"`:

| Kernel | Role |
|---|---|
| `"csr_matvec"` | the destination-driven ("pull") product of the forward pass |
| `"edge_grad"` | the gradient with respect to the edge weights in the backward pass |
| `"spike_push_dense"` | the source-driven ("push") product that compacts the spikes on the device |

The backend is available when the `triton` package can be imported and CUDA is
present. It then has a higher priority than `"aten"`, so it is the default on
CUDA and nothing has to be selected. The Triton kernels handle `float32` data
only. For every other dtype they call the ATen kernel themselves, without a
message; `conn.explain()` still reports `triton` in that case. `"aten"`
remains the fallback when Triton is missing, the default on CPU, and the
reference the Triton kernels are tested against.

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

The connection keeps its prepared route until planning inputs change.
Entering or leaving `use_backend` invalidates that route, so the next call
replans an existing connection; `conn.explain()` reports the backend that
would run under the current override.

With the Triton push kernel a density-hinted connection is planned as
`"push"`; without it (CPU, or CUDA without Triton) as `"adaptive-push"`. The
differences between the two are listed under
[Performance hints](#performance-hints-and-explain).

## Migration from `SparseConn`

`SparseConn`, `SparseConstrainedConn`, `BaseSparseConn`, `SparseBackend`,
`available_sparse_backends` and the `sparse_backend=` argument were removed
from `btorch.models.linear` without a compatibility layer. `Linear` is the
its interface.

| Old | New |
|---|---|
| `SparseConn(conn, bias=b, enforce_dale=E)` (default `enforce_dale=True`) | `SparseConnection.from_adjacency(conn, Synapse(dale=E), bias=b)` (default `dale=False`) |
| `sparse_backend=`, `available_sparse_backends()` | removed; the runtime chooses. Advanced: `btorch.sparse.runtime.use_backend(...)` |
| `layer.magnitude` / `layer.initial_sign` / `get_sparse_matrix()` | `conn.weight.value` / `conn.weight.sign` (now persistent) / `conn.to_sparse()` (the orientation of the input; `conn.to_sparse("pre_post")` to be explicit) |
| `SparseConstrainedConn(conn, constraint, enforce_dale=E)` | `SparseConnection.from_adjacency(conn, Synapse(weight=ConstrainedWeight(group=constraint, dale=E)))` |
| `.magnitude` / `.initial_weight` / `._constraint_scatter_indices` | `conn.weight.scale` / `.base` / `.group` |
| `get_group_info` / `set_group_magnitude` / `get_weights_by_group` | `conn.weight.group_info` / `set_scale` / `weights_by_group` |
| `SparseConstrainedConn.from_hetersynapse(conn, constraint, receptor_idx)`, `constraint_info`, `persist_initial_weight` | `SparseConnection.from_adjacency(...)` (expanded layout) or `SparseConnection.from_hetersynapse(conn, Synapse(weight=ConstrainedWeight(group=constraint)), receptor_type_index=receptor_idx, n_delay=...)` (semantic per-edge receptor/delay); the rest removed |
| `state_dict` keys `magnitude`, `indices` | `indices`, `layout`, `weight.value` (+ `weight.sign`) or `weight.scale`/`weight.group`/`weight.base`, plus `receptor`/`delay` |

Points that need attention when porting:

- **Dale's law is opt-in.** `SparseConn` enforced it by default; a plain
  `SparseConnection.from_adjacency(conn)` does not. Pass `Synapse(dale=True)`
  to keep the old behaviour.
- **Checkpoints are not compatible.** The `state_dict` keys changed (last row
  of the table). `load_state_dict` refuses a checkpoint written by the old
  layers with a message, also with `strict=False`. There is no converter:
  rebuild the connection from its matrix and copy the weights.
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
w = layer.to_sparse().to_scipy()  # "pre_post", like rng_matrix
assert w.shape == rng_matrix.shape
```

See also the design note
[Sparse and connectivity subsystem](../design/sparse_connectivity.md).
