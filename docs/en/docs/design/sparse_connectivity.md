# Sparse and connectivity subsystem

Design note for the PyTorch-first sparse / connectivity system. Part 1 is the
audit of the implementation this work started from (kept as a record; the
classes it describes no longer exist). Parts 2 onwards describe what replaced
it. For usage see the [sparse connectivity guide](../guides/sparse_connectivity.md).

## 1. Audit of the starting point

### 1.1 Relevant code

| Path | Names |
|---|---|
| `btorch/models/linear.py` | `BaseSparseConn`, `SparseConn`, `SparseConstrainedConn`, `DenseConn`, `SparseBackend`, `_resolve_sparse_backend`, `available_sparse_backends` |
| `btorch/models/constrain.py` | `HasConstraint`, `constrain_net` (runs `constrain()` under `torch.no_grad`) |
| `btorch/connectome/connection.py` | `make_sparse_mat`, `make_constraint_by_neuron_type`, `make_hetersynapse_conn`, `make_hetersynapse_constraint`, `make_hetersynapse_constrained_conn`, `stack_hetersynapse`, `expand_conn_for_delays` |
| `btorch/models/synapse.py` | `BasePSC` and subclasses call `self.linear(z)`; `HeterSynapsePSC`, `DelayedPSC`, `BilinearMixingSynapse` consume the physically expanded receptor / delay axes |
| `btorch/models/history.py` | `SpikeHistory` produces the `[..., n_neuron * n_delay_bins]` input of a delay-expanded matrix |

Tests: `tests/models/test_linear.py`, `tests/models/test_hetersynapse_conn.py`,
`tests/connectome/` (`test_hetersynapse_conn_modes.py`, `test_delay_expansion.py`,
`test_connection_validation.py`), `tests/models/rnn/test_sparse_rnn.py`,
`tests/models/test_hetero_rnn.py`, `tests/models/test_cudagraph_layers.py`,
`tests/models/test_mem_load_save.py`, `tests/models/test_numpy_parity.py`,
`tests/test_pipeline_e2e.py`. Baseline before any change: 131 passed, 2 xfailed
for these files.

### 1.2 Behaviour, checked against the code

All eight characteristics listed in the task specification hold:

1. `BaseSparseConn` takes a SciPy sparse array (legacy `spmatrix` also works
   because only `.tocoo()`, `.T` and `.sum_duplicates()` are used). A torch
   sparse tensor is rejected with an `AttributeError`.
2. The matrix is converted to COO, **transposed**, and `sum_duplicates()` is
   applied, which also sorts. The `indices` buffer therefore holds
   `[dst, src]`, sorted by destination then source.
3. `forward` flattens all leading dimensions to one batch axis and computes
   `x @ A` as `(Aᵀ @ xᵀ)ᵀ`; a 1-D input is supported.
4. `native` rebuilds a `torch.sparse_coo_tensor` from `indices` and the current
   effective values on every call and uses `torch.sparse.mm`.
5. `torch_sparse` calls `torch_sparse.spmm(indices, values, n_dst, n_src, xᵀ)`.
6. `SparseConstrainedConn` stores `initial_weight[e]`, a group index per edge
   (`_constraint_scatter_indices`) and one `magnitude` per group; the effective
   weight is `initial_weight[e] * magnitude[group[e]]`.
7. The edge-to-group map is built at construction by a pandas merge on
   `(row, col)`; group ids in the constraint matrix are 1-based.
8. `make_hetersynapse_conn` expands columns to `post * n_receptor + receptor`
   and, with `delay_col`, rows to `pre * n_delay_bins + delay_bin`.

### 1.3 Orientation and shape conventions

- User-facing connectivity matrices are **`(n_src, n_dst)`**: rows are source
  (pre-synaptic) neurons, columns are destinations. The layer computes
  `y = x @ A` with `x[..., n_src]`, `y[..., n_dst]`.
- Internally the layer stores the transpose, so the stored COO is the standard
  linear-algebra operator `M = Aᵀ` of shape `(n_dst, n_src)` with `y = M @ x`,
  sorted row-major. The stored order is exactly CSR order of `M`.
- `BaseSparseConn.shape` is `(in_features, out_features)`, i.e. the user
  orientation, although its docstring says `(num_dst, num_src)`.
- `get_sparse_matrix()` swaps the index rows back and returns the user
  orientation `(n_src, n_dst)`. It passes `is_coalesced=True` although the
  swapped indices are no longer sorted.
- `DenseConn(weight=W)` takes `W` as `(in, out)` and stores `W.T` in
  `nn.Linear.weight`.

### 1.4 Autograd path

`magnitude` → (gather through the group index for the constrained layer) →
values of a freshly built sparse tensor → `torch.sparse.mm` / `torch_sparse.spmm`
→ output. Gradients reach both the values and the dense input. `constrain()`
mutates `magnitude.data` (Dale's law) and is called from `constrain_net` under
`torch.no_grad()`.

### 1.5 `torch.compile`

Measured with torch 2.11:

| Backend | `torch.compile(conn)` | `torch.compile(conn, fullgraph=True)` |
|---|---|---|
| `native` | runs, but Dynamo graph-breaks on `torch.sparse_coo_tensor` | **fails**: `Unsupported: Attempted to wrap sparse Tensor` |
| `torch_sparse` | runs | runs (one graph, no breaks) |

The existing test only passes for `native` because `compile_or_skip` does not
request `fullgraph`.

### 1.6 Checkpoint and device-move risks

- `SparseConn.initial_sign` is a non-persistent buffer derived from the
  construction-time matrix. Loading a checkpoint into a layer built from a
  different matrix keeps the old signs; the next `constrain()` zeroes every
  weight whose sign differs. Reproduced: loading weights `[-1, 2]` into a layer
  built from `[1, -2]` gives `[0, -0]` after one `constrain()`.
- `SparseConstrainedConn.initial_weight` is non-persistent unless
  `persist_initial_weight=True`, and `_constraint_scatter_indices` is never
  persistent. A checkpoint therefore only reproduces the layer if it is loaded
  into a layer built from the same matrices. `constraint_info` (a DataFrame) is
  a plain attribute and is not saved.
- `forward` silently casts the effective values to the input's device and
  dtype on every call (`float64` input with `float32` weights yields `float64`).
- The stale cached sparse tensor the specification mentions has already been
  removed (the COO tensor is rebuilt per call); the regression test
  `test_sparse_conn_state_dict_roundtrip_loads_new_pattern` covers it. The
  price is a COO construction, and inside `torch.sparse.mm` a COO→CSR
  conversion, on every forward.
- `torch.sparse.mm` backward with respect to the sparse values does not scale:
  at 100k neurons and 10M edges the `native` backend fails with an
  out-of-memory error asking for 37 GiB (`N × N` floats).

### 1.7 Pain points

- SciPy-only input; no torch sparse input; conversion logic lives in the layer.
- The `native` path is not `fullgraph`-compilable and rebuilds a sparse tensor
  per call; `torch_sparse` is the de-facto required backend for large graphs.
- Backend names appear in every constructor (`sparse_backend=`).
- Grouped weights are a separate matrix class rather than a weight
  parameterisation; Dale's law is implemented twice.
- Receptors and delays exist only as physically expanded matrix dimensions.
- No network batch (ensembles), no shared-pattern batched values.
- No sparse-spike execution path; cost is independent of spike density.

### 1.8 Baseline measurements

`benchmarks/sparse_conn/bench_sparse_conn.py`, random recurrent graphs, median
of 20 calls. RTX 5090 (shared with another job; note the ~2.2 ms floor on every
cell, which is launch/synchronisation latency, not kernel time):

| Workload | Implementation | B | forward (ms) | forward+backward (ms) | peak (MB) |
|---|---|---:|---:|---:|---:|
| 4,096 neurons, 0.4M edges | `native` | 32 | 2.3 | 11.7 | 98 |
| | `torch_sparse` | 32 | 4.6 | 5.1 | 226 |
| | dense | 32 | 2.3 | 2.4 | 146 |
| 100k neurons, 10M edges | `native` | 1 / 32 | out of memory | out of memory | – |
| | `torch_sparse` | 1 | 4.7 | 5.0 | 360 |
| | `torch_sparse` | 32 | 19.1 | 58.2 | 5,138 |
| 1M neurons, 50M edges | `torch_sparse` | 1 | 6.0 | 8–12 | 1,738 |
| | `torch_sparse` | 32 | 159 | 300 | 25,699 |

Latency does not depend on spike density (1% and 50% give the same numbers).

### 1.9 Prior kernel work consulted

These are references for kernel algorithms and measurements only.

- **Task-packed Triton SpMSpV** (PR 70, `_sparse_triton.py`): source-major
  blocked layout, on-device pack → persistent workers with float atomics, a
  per-worker hash table for duplicate destinations. Forward only in Triton;
  backward is dense `index_add_` with a `[B, E]` temporary. 8.6× over the
  per-call COO path at 0.1% activity, break-even near 10%. Its preprocessing is
  a Python loop over sources.
- **Pull CSR kernels** (megakernel / `glif_net`): one warp per row,
  deterministic, bandwidth-equal to cuSPARSE at large N; the gains at small N
  come from fusing with the neuron update. One batched Triton `triton_op`
  exists (`btorch::csr_spmv`).
- **Push kernels** (`connectome_dataset`, Praxist): per-active-source atomic
  scatter over source-major CSR. On the 5090, push beats pull below 30–60%
  spike density on connectomes up to 140k neurons, by 14–27× at 1%. All
  measurements are single-vector; no batched push kernel exists, and no kernel
  has a custom backward.
- Sort-based, hash-based (VDHA) and tensor-core SpMM kernels never won in the
  connectome regime and are not planned.

Consequences for this design: the runtime keeps a destination-major CSR (pull,
batched SpMM, deterministic) and a source-major CSR of the same edges (push,
and the transpose product for the input gradient) linked by an edge
permutation, so one value array serves both. Gradients with respect to edge
values are a gather-multiply (SDDMM) and never build an `N × N` intermediate.

### 1.10 Contradictions with the specification or earlier decisions

- **Compatibility.** The specification asks to preserve behaviour through
  legacy wrappers and deprecation paths. The project decision is the opposite:
  no backward compatibility. `SparseConn`, `SparseConstrainedConn`,
  `BaseSparseConn` and the `sparse_backend=` argument are removed and every
  caller moves to `SparseConnection`. Numerical behaviour (orientation,
  duplicate summing, Dale's law, grouped weights) is preserved and pinned by
  the contract tests; names and `state_dict` keys are not.
- **Package layout.** The specification suggests `btorch/nn/connection` and
  `btorch/sparse_runtime`. The repository has no `btorch.nn`; modules live in
  `btorch.models`. The new code is placed in `btorch/sparse/` (numerical),
  `btorch/sparse/runtime/` (planner, caches, registered ops) and
  `btorch/models/connection/` (connection semantics).
- **Name clash.** `btorch.models.synapse.Synapse` is an existing protocol for
  PSC modules. The new synapse description is
  `btorch.models.connection.Synapse`.
- **`connection_conversion.py`** is referenced by an earlier decision but does
  not exist on this branch; nothing here depends on it.

## 2. Architecture

Three layers, kept separate on purpose:

```text
btorch.models.connection      modelling semantics
    Projection, ConnectionRule, Synapse, Weight
    Connection: SparseConnection | StructuredConnection | ImplicitConnection | HybridConnection
btorch.sparse                 numerical objects
    Sparse: COO | CSR | CSC          LinearOperator: Structured | Implicit | Composite
btorch.sparse.runtime         execution
    Planner, RepresentationCache, BackendRegistry, KernelCache, registered ops
```

- A `ConnectionRule` says which pairs are connected. It is not a sparse array
  and does not choose a storage format.
- A `Synapse` says what the edges do (weight, receptor, delay, plasticity).
  These are orthogonal edge attributes, not matrix formats and not subclasses.
- A `Sparse` is a format-agnostic numerical array; `COO`, `CSR`, `CSC` are
  representations of it. Procedural operators are `LinearOperator`s, not
  `Sparse`.
- The runtime picks representation, algorithm and backend independently.
  None of them appears in a model definition.

`torch.compile(model)` is the only compilation step a user performs. There is
no `conn.compile()` and no public execution-plan object.

## 3. Numerical API (`btorch.sparse`)

- Shape model `[*batch, *sparse, *dense]` with `batch_shape`, `sparse_shape`,
  `dense_shape`. Stored values are `[*value_batch, nnz, *dense]`.
- `COO` supports any number of sparse dimensions; `CSR` / `CSC` exactly two.
  BSR / BSC are read and written through COO and are not native formats.
- `tocsr()` / `tocoo()` / `tocsc()` convert; `as_csr()` / `as_coo()` /
  `as_csc()` return the object only if it already has that format and raise
  otherwise.
- `as_sparse` (alias `asarray`) is the single conversion entry point for
  btorch, PyTorch and SciPy sparse objects. Everything that accepts "a sparse
  matrix" calls it. Conversions never transpose and never densify; dense input
  is rejected.
- `A @ x` follows `torch.matmul`. `matvec(A, x)` applies `A` along the last
  axis of `x [..., N]`. Both are standard orientation: `A.shape == (M, N)`
  maps length-`N` to length-`M`.
- Representation changes that reorder or merge entries return an `EdgeMap`.
  Every edge-aligned array (weights, group ids, receptors, delays) is moved
  with the same map: weights are summed over merged duplicates, categorical
  metadata must agree or the merge is refused.

### Batches

Two kinds of batch exist and are not the same thing.

| | Meaning | Where it lives |
|---|---|---|
| Sample batch | independent inputs / trajectories of one network | leading dimensions of `x` |
| Network batch | different networks | `batch_shape` of the sparse array |

A network batch with a **shared pattern** stores the indices once and values
`[G, nnz]`. A batch of **different patterns** (`sparse.stack`) is a COO whose
first index row is the network id; members may have different `nnz` and
nothing is padded. A connection executes it as one block-diagonal operator.

The network dimensions of `x` are its leading dimensions and broadcast against
`batch_shape`: `A [G, M, N]` with `x [G, B, N]` gives `[G, B, M]`, and
`x.unsqueeze(0)` shares `B` samples across all networks. An input without the
network dimensions is an error; networks are never combined with samples
implicitly.

## 4. Orientation

| Object | Shape | Product |
|---|---|---|
| `btorch.sparse` arrays, `SparseConnection(A)` | `(n_post, n_pre)` | `y = A @ x` |
| Connectome matrices, hetersynapse helpers, `from_adjacency(W)` | `(n_pre, n_post)` | `y = x @ W` |

`SparseConnection.from_adjacency(W, orientation="src_dst")` is the only place
a transpose is expressed. The canonical edge buffer `indices` holds
`[post, pre]` per edge.

## 5. Connections

`SparseConnection` state:

- Persistent (in the `state_dict`): `indices [2, E]`, optional `receptor [E]`
  and `delay [E]`, and the weight module's state (`weight.value` and, with
  Dale's law, `weight.sign`; or `weight.base`, `weight.group`, `weight.scale`).
- Derived (non-persistent buffers of `conn.cache`): destination-major CSR
  (`crow`, `col`, `perm`) and source-major CSR (`t_crow`, `t_col`, `t_perm`) of
  the same edges. They are rebuilt from the persistent buffers after
  `load_state_dict` and after rewiring, into the existing tensors when the
  number of edges is unchanged. Nothing derived can outlive the state it was
  derived from.

Weights are modules returning one effective value per edge: `EdgeWeight`
(per edge), `ConstantWeight`, `ConstrainedWeight` (`w[e] = base[e] *
scale[group[e]]`). Dale's law is a constraint of the weight module, applied by
`constrain_net` under `torch.no_grad()`; its reference sign is persistent.

Receptors and delays are per-edge attributes. The executed operator is the
expanded reference lowering `(n_post * n_receptor, n_pre * n_delay)`, which is
the layout `HeterSynapsePSC` and `SpikeHistory` already use; the lowering is
derived state, the semantic edge list is the model. The hetersynapse helpers
are unchanged and `SparseConnection.from_hetersynapse` decodes their expanded
matrices into edge attributes.

## 6. Spikes during training

The spike tensor produced by a neuron model stays an ordinary dense tensor
`[..., N]`. Its forward values are sparse, its surrogate gradient is not, so
the logical object and the autograd object are dense. Sparse execution is
derived from it inside the operator: `spike_propagate` packs the non-zero
entries, visits only the out-edges of active sources, and returns the same
dense gradient as the destination-driven product, including for silent
neurons. Genuinely sparse input (event data) can be passed as a `Sparse`
array to `SparseConnection.propagate_events`.

## 7. `torch.compile` boundary

```text
SparseConnection.forward            Python, traced by Dynamo
    weight()                        ordinary tensor ops
    torch.ops.btorch.csr_propagate  one opaque node: raw buffers in, tensor out
        kernel from BackendRegistry (ATen reference, Triton, ...)
```

The registered operators (`btorch::csr_propagate`, `btorch::spike_propagate`,
and the non-differentiable building blocks `btorch::csr_matvec`,
`btorch::csr_edge_grad`) have fake implementations and an explicit backward:
the value gradient is a sampled product over the edges, the input gradient a
destination-driven product over the source-major CSR. No dense `M x N`
intermediate exists in either direction. PyTorch sparse tensors never reach
the tracer, and `forward` contains no planning: the plan is fixed when the
connection is built or its hints change.

## 8. Dynamic topology

Structural plasticity is an update policy, not a format. A connection has a
fixed number of edge slots; `set_edges_` changes which edge a slot represents
without changing any tensor shape or parameter identity, increments
`topology_version`, and rebuilds the derived layouts in place. Value updates
do not touch topology-derived state. The version counters
(`topology_version`, `routing_version`, `value_version`) are bookkeeping for
caches and are not read in `forward`, so rewiring does not retrace a compiled
module.
