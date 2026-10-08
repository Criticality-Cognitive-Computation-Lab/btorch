# Sparse and connectivity subsystem

Design note for the PyTorch-first sparse / connectivity system. Part 1 is the
audit of the implementation this work started from (kept as a record; the
classes it describes no longer exist). Parts 2 onwards describe what replaced
it. For usage see the [sparse connectivity guide](../guides/sparse_connectivity.md).

## 1. Audit of the starting point

### 1.1 Relevant code

| Path | Names |
|---|---|
| `btorch/models/linear.py` | `Linear` |
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
- `Linear(weight=W)` follows PyTorch directly: `W` has shape `(out, in)` and
  is stored unchanged in `nn.Linear.weight`.

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

The legacy layers were measured before the refactor and are kept as a frozen
copy in `benchmarks/sparse_conn/_legacy_baseline.py`, so they can be timed
side by side with the new path. Two properties of the old implementation
matter for the comparison:

- Latency did not depend on spike density (1% and 50% gave the same numbers).
- The `native` backend could not train at scale: at 100k neurons and 10M
  edges its backward failed with an out-of-memory error asking for 37 GiB,
  and `torch_sparse` needed 5.1 GB of peak memory at batch 32.

Current numbers for both paths are in
`benchmarks/sparse_conn/RESULTS.md`.

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
- A `Synapse` says what the edges do (weight, receptor, delay). These are
  orthogonal edge attributes, not matrix formats and not subclasses. The
  `plasticity` field is reserved: any value other than `None` raises
  `NotImplementedError`.
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
  maps length-`N` to length-`M`. For a square `A` and a square batch the two
  accept the same input and return different results. The function forms
  `matvec`, `matmul` and `rmatvec` accept linear operators as well as sparse
  arrays.
- `from_torch(t, batch_dim=k)` and `as_sparse(t, batch_dim=k)` read the first
  `k` sparse dimensions of a PyTorch COO tensor as batch coordinates (PyTorch
  COO has no notion of a batch). An uncoalesced COO tensor that does not
  require grad keeps its stored order and duplicates; one that requires grad
  is coalesced, because PyTorch exposes its values to autograd only then.
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

Two strings name the two orientations everywhere: `"post_pre"` (the operator)
and `"pre_post"` (the adjacency matrix of connectome tables).

| Orientation | Shape | Product | Objects |
|---|---|---|---|
| `"post_pre"` | `(n_post, n_pre)` | `y = A @ x` | `btorch.sparse` arrays, `SparseConnection(A)`, `from_adjacency(A, orientation="post_pre")`, `FromSparse(A, orientation="post_pre")` |
| `"pre_post"` | `(n_pre, n_post)` | `y = x @ W` | connectome matrices, hetersynapse helpers, `from_adjacency(W)` and `FromSparse(W)` (their default), `from_hetersynapse` |

- `SparseConnection.from_adjacency` and `FromSparse` are the places where a
  transpose is expressed, and both default to `"pre_post"`. The bare
  constructor `SparseConnection(A)` takes the operator.
- Functions that take index tensors or populations take the source first:
  `from_edges(pre, post, ...)`, `FromEdges(pre, post)`, `find_edges(pre,
  post)`, `Projection(pre, post, ...)`, `candidate(pre, post)`.
  `set_edges_(slots, *, pre, post)` is keyword-only.
- The canonical edge buffer `indices` holds `[post, pre]` per edge;
  `conn.pre` and `conn.post` are the named accessors.
- A connection remembers the orientation it was built from in
  `conn.orientation`; `conn.to_sparse(orientation=None)` defaults to it and
  validates its argument.
- For a square matrix a wrong orientation is a valid input. Nothing can detect
  it.

## 5. Connections

`SparseConnection` state:

- Persistent (in the `state_dict`): `indices [2, E]`, `layout`
  (`[n_post, n_pre, n_receptor, n_delay]`, followed by the batch shape of a
  batch of different patterns), optional `receptor [E]` and `delay [E]`, an
  optional `bias [out_features]`, and the weight module's state
  (`weight.value` and, with Dale's law, `weight.sign`; or `weight.base`,
  `weight.group`, `weight.scale`). `load_state_dict` compares `layout` with
  the module, checks that the ids lie inside the populations, and refuses a
  checkpoint with `indices` but without `layout` (written by the removed
  layers) with a migration message. These checks raise also with
  `strict=False`.
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

Weights are bound by identity, not by copy:

- A `Weight` module passed in `Synapse(weight=...)` is `conn.weight` itself.
  It holds the parameters of one connection; binding it a second time raises.
- An `nn.Parameter` is adopted as `conn.weight.value` (same object) when the
  connection keeps the edge order, that is when the input entries are already
  sorted by target and then source and nothing is merged. Otherwise the values
  are copied into a new parameter and a warning is emitted.
- A callable `f(n_edge)` or a `torch.distributions.Distribution` creates
  trainable per-edge weights once the edges are known.
- `Synapse(weight=<number>, dale=True)` raises: one fixed number cannot change
  sign.
- Every weight module exposes `shape`, `dtype` and `device` of its effective
  weights. `edge_table()["weight"]` is a detached snapshot.

`bias=True` creates a zero-initialised `[out_features]` bias; a tensor is
validated against that shape at construction.

Receptors and delays are per-edge attributes. The executed operator is the
expanded reference lowering `(n_post * n_receptor, n_pre * n_delay)`, which is
the layout `HeterSynapsePSC` and `SpikeHistory` already use; the lowering is
derived state, the semantic edge list is the model. The hetersynapse helpers
are unchanged and `SparseConnection.from_hetersynapse(matrix, synapse,
n_receptor=None, n_delay=1, receptor_type_index=None)` decodes their expanded
matrices into edge attributes; the number of receptor channels is given
directly or as the receptor index table of the helper. `from_edges` validates
that the ids lie inside the populations.

`Projection` forwards `n_delay`, `n_receptor`, `weight`, `edge_table()` and
`to_sparse()` to the connection it built.

## 6. Spikes during training

The spike tensor produced by a neuron model stays an ordinary dense tensor
`[..., N]`. Its forward values are sparse, its surrogate gradient is not, so
the logical object and the autograd object are dense. Sparse execution is
derived from it inside the operator: `spike_propagate` finds the non-zero
entries (on the device with the Triton backend, on the host with the
reference backend), visits only the out-edges of active sources, and returns
the same dense gradient as the destination-driven product, including for
silent neurons. Genuinely sparse input (event data) can be passed as a `Sparse`
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
do not touch topology-derived state. `HardDeepR` is the provided policy: it
runs from an optimizer post-step hook, samples new positions uniformly among
unconnected pairs without building a dense mask, applies an explicit policy to
the optimizer state of rewired slots, never raises when a layer is full
(unplaced slots wait at zero weight) and has its own checkpoint state. It
accepts a `SparseConnection` or a `Projection` realised as one, and calls
`candidate(pre, post)` to restrict new positions.

`set_edges_(slots, *, pre, post, receptor=None, delay=None)` takes its index
arguments by keyword only.

Four counters describe what changed. None of them is read in `forward`, so
rewiring does not retrace a module compiled with the default mode.

| Counter | Changes when | Consumer |
|---|---|---|
| `topology_version` | `set_edges_`, `load_state_dict` | derived layouts |
| `routing_version` | `set_edges_` with receptors or delays, `load_state_dict` | derived layouts |
| `value_version` | a weight tensor is written in place (optimizer step, `constrain`, checkpoint load); derived from the tensors' in-place version counters | none yet |
| `capture_version` | a CUDA graph captured around the connection became invalid: `set_edges_`, `load_state_dict`, `.to()`, a `set_hints` that changes the plan, a reallocation of backend layouts. Not on weight updates. | `CudaGraphRunner` |

### CUDA graphs

A CUDA graph replays recorded kernel launches and does not run Python again,
so the wiring, the execution path and the device buffers of a connection are
frozen at capture time. Weights are read at replay time and may change.

- `conn.capture_version` changes whenever a captured graph is no longer valid
  (table above). `conn.capture_incompatibility()` returns the reason why a
  connection cannot be captured at all, or `None`; the one reason today is
  the host-side packing of `"adaptive-push"`.
- `btorch.models.cudagraph` defines the protocol for any submodule:
  `capture_versions(module)` and `capture_incompatibilities(module)` collect
  the two over a module tree.
- `RecurrentNN(cudagraph=True)` reads the versions on every call and captures
  again when one changed, and refuses a module that reports an
  incompatibility.
  <!-- verify: cudagraph re-capture -->
- `torch.compile(mode="reduce-overhead")` uses CUDA graphs internally and has
  no such hook. It must not be combined with rewiring or with loading a
  different pattern, and a connection should be called once in eager mode
  before it is compiled in that mode.

## 9. Runtime backends

| Kernel | ATen (reference, all devices) | Triton (CUDA) |
|---|---|---|
| `csr_matvec` (pull, also the transposed product of the input gradient) | `torch.sparse` CSR product; gather / `index_add` for dtypes without one | row-block tiles over `(rows, samples)`, hub rows cut into segments, one writer per output: deterministic |
| `edge_grad` (value gradient) | chunked gather-multiply | one launch over entry blocks, no `[B, E]` temporary |
| `spike_push` (packed events) | expand active sources into out-edges, `index_add` | thread per delivered edge with a binary search over the prefix sum of active degrees, float atomics |
| `spike_push_dense` (device compaction) | – | fixed grid of 64-source tiles, silent tiles exit after one load; long out-edge lists as static chunks; no host sync |

The Triton kernels reuse ideas from the earlier experiments listed in 1.9
(row-block and CSR-vector pull, nnz-balanced handling of hubs, tiled and
per-edge atomic push). Sort-based and hash-based push variants were not ported
because they never won in the measured regime. Push results depend on the
arrival order of float atomics (relative error up to about `4e-7`); pull
results are bitwise reproducible.

The planner chooses one of three algorithms when a connection is built,
moved or given new hints. Without an `expected_density` hint, with a hint
above the measured crossover (2% on CUDA, 0.2% on CPU), and for a
shared-pattern batch of weights it plans `"pull"`. With a hint at or below
the crossover it plans a source-driven algorithm, and which one depends on
the backend:

| Algorithm | Backend | Behaviour |
|---|---|---|
| `"pull"` | all | destination-driven product over every edge; deterministic |
| `"push"` | device-compacting (`spike_push_dense`, Triton on CUDA) | taken for every call once planned, whatever the density of an individual input; one launch, no host synchronisation; float atomics, so not bitwise reproducible; can be captured in a CUDA graph |
| `"adaptive-push"` | reference (`spike_push`, ATen) | packs the active sources on the host; each call falls back to pull when its input is denser than the limit; cannot be captured in a CUDA graph |

Two consequences of `"push"` differ from `"pull"`. Under
`torch.use_deterministic_algorithms(True)` the planner plans `"pull"` instead
of `"push"` (the flag is read when the plan is made). And `"push"` never
visits the edges of a silent source, so a non-finite weight on such an edge
does not reach the output, whereas `"pull"` propagates it (`inf * 0`).

`Hints` validates its fields: `expected_density` must lie in `[0, 1]`, and the
reserved `expected_calls` and `expected_batch` must be positive integers.

The optional `torch_sparse` backend remains selectable with
`runtime.use_backend("torch_sparse")` and nothing depends on it.
`btorch.sparse.runtime.__all__` lists the registry, planner, cache and
operator names only; the kernel modules (`ops`, `kernels_aten`,
`kernels_triton`, `kernels_triton_push`) are internal. Importing
`kernels_aten` no longer installs process-wide warning filters.

Eager and compiled execution take different routes to the same kernels:
compiled code calls the registered operators, eager code a plain
`autograd.Function`, because the operator dispatch costs tens of microseconds
per call and dominates for networks of a few thousand neurons.

### Measured behaviour

Full tables are in `benchmarks/sparse_conn/RESULTS.md`, together with the
recorded state of the GPU. The numbers below are from an RTX 5090 that had no
other compute process when the run was launched, synthetic graphs with
uniform in-degree, medians per call.

| Neurons (edges) | Batch | Forward (ms) | Forward + backward (ms) | Legacy `native` forward / forward + backward |
|---|---:|---:|---:|---|
| 4,096 (0.4M) | 1 | 0.016 | 0.17 | 0.068 / 1.4 |
| 4,096 (0.4M) | 32 | 0.033 | 0.19-0.23 | 0.093 / 0.55 |
| 100,000 (10M) | 1 | 0.105 | 0.42 | 0.235 / cannot run |
| 100,000 (10M) | 32 | 0.28 | 1.03 | 3.04 / cannot run |
| 1,000,000 (50M) | 1 | 0.54 | 2.4 | 1.39 / cannot run |
| 1,000,000 (50M) | 32 | 2.1 | 9.8 | cannot run |

- A dense layer is still faster at 4,096 neurons (0.13-0.14 ms forward +
  backward).
- With a density hint at 1 % spikes, the push path brings the forward to
  0.030 ms at 100,000 neurons and 0.031 ms at one million (batch 1), and to
  0.10 ms and 1.29 ms at batch 32. It makes no useful difference at 4,096
  neurons.
- `torch.compile` of a single connection equals eager from 100,000 neurons
  up and is slower below (0.052 against 0.016 ms forward at 4,096 neurons):
  the registered-operator boundary costs more per call than the eager
  autograd node.
- In a 200-step recurrent network of 4,096 neurons the connection is not the
  bottleneck: the new path, dense and legacy are within 0.23-0.29 ms per
  inference step. At 100,000 neurons and batch 32 a step takes 0.47 ms
  (inference) and 1.37 ms (training) against 3.26 ms and "cannot run".
- The comparison against the legacy `torch_sparse` path exists only from a
  shared GPU: at 100,000 neurons and batch 32, 1.06 ms and 0.76 GB against
  17.4 ms and 5.1 GB for forward + backward.

## 10. Migration

| Removed | Replacement |
|---|---|
| `BaseSparseConn` | `SparseConnection` over `btorch.sparse.Sparse` |
| `SparseConn(conn, bias=b, enforce_dale=E)` | `SparseConnection.from_adjacency(conn, Synapse(dale=E), bias=b)`; Dale's law is now off by default |
| `SparseConstrainedConn(conn, constraint, enforce_dale=E)` | `SparseConnection.from_adjacency(conn, Synapse(weight=ConstrainedWeight(group=constraint, dale=E)))` |
| `SparseConstrainedConn.from_hetersynapse`, `constraint_info`, `persist_initial_weight` | `SparseConnection.from_hetersynapse(conn, synapse, n_receptor=, n_delay=, receptor_type_index=)`; base weights and groups are always persistent |
| SciPy-only constructor input | anything `btorch.sparse.as_sparse` accepts (btorch, PyTorch, SciPy) |
| `magnitude`, `initial_sign`, `initial_weight`, `_constraint_scatter_indices` | `weight.value`, `weight.sign`, `weight.base`, `weight.group` (`weight.scale` for group scales) |
| `get_sparse_matrix()` | `to_sparse()` after `from_adjacency` (the orientation of the input), or `to_sparse("pre_post")` explicitly |
| `get_group_info`, `set_group_magnitude`, `get_weights_by_group` | `weight.group_info`, `weight.set_scale`, `weight.weights_by_group` |
| `sparse_backend=`, `available_sparse_backends()`, `SparseBackend` | internal backend selection; `btorch.sparse.runtime.use_backend(...)` for experts |
| `make_hetersynapse_conn`, `make_hetersynapse_constraint`, `make_hetersynapse_constrained_conn`, `stack_hetersynapse`, `expand_conn_for_delays` | unchanged; their expanded matrices are one lowering, decoded by `from_hetersynapse` |
| `state_dict` keys `magnitude`, `indices` | `indices`, `layout`, `weight.*` (and `receptor`, `delay`); old checkpoints are refused with a migration message, also with `strict=False` |

`Linear` follows the standard PyTorch interface; its `constrain()` does not write through
`.data`.

Names that changed during the development of this subsystem (no aliases
exist):

| Earlier | Now |
|---|---|
| `orientation="src_dst"` / `"dst_src"` | `"pre_post"` / `"post_pre"` |
| `FromSparse(A)` defaulting to `"post_pre"` | defaults to `"pre_post"`, like `from_adjacency` |
| `conn.to_sparse()` defaulting to the operator | defaults to `conn.orientation` |
| `set_edges_(slots, post, pre)` | `set_edges_(slots, *, pre, post, receptor=None, delay=None)` |
| `candidate(post, pre)` | `candidate(pre, post)` |
| weights passed in a `Synapse` were copied | a `Weight` module is `conn.weight`; an `nn.Parameter` is adopted when the edge order is kept |
| planner algorithm `"adaptive-push"` on every backend | `"push"` on a device-compacting backend, `"adaptive-push"` on the reference backend |

## 11. Known limitations

- **N-D sparse.** COO carries N-D metadata and `sparse.einsum` handles one
  sparse operand against dense operands with a dense output. There is no
  CSF format, no sparse-sparse contraction, no sparse output and no ellipsis.
  See [the einsum notes](sparse_einsum_notes.md) for what a Scorch/TACO-style
  compiler would add.
- **BSR / BSC** are imported and exported through COO; they are not native
  formats and there is no block kernel.
- **Different-pattern network batches** are a reference fallback: stored as
  batched COO and executed as one block-diagonal operator. There is no ragged
  CSR, and they cannot be combined with a shared-pattern value batch, with
  receptor/delay routing, or with rewiring.
- **Receptors and delays** execute through the expanded reference lowering.
  There is no routed kernel, delay ring buffer or event queue.
- **Source-driven propagation** is opt-in through a density hint and is not
  used for value-batched weights. `"push"` (Triton on CUDA) is not bitwise
  deterministic, is used for every input once planned even when that input is
  dense, and does not propagate a non-finite weight of a silent source.
  `"adaptive-push"` (CPU, and CUDA without Triton) packs spikes on the host,
  which synchronises and cannot be captured in a CUDA graph.
- **CUDA graphs.** A graph captured around a connection is invalid after
  `set_edges_`, `load_state_dict` with different edges, `.to()` and a
  `set_hints` that changes the plan; replaying it computes with the old
  structure and raises nothing. `conn.capture_version` reports these events
  and `RecurrentNN(cudagraph=True)` acts on it. `torch.compile(mode=
  "reduce-overhead")` has no such hook and must not be combined with rewiring
  or with loading a different pattern. Modules compiled with the default mode
  are unaffected.
- **Triton kernels are `float32` only.** For any other dtype they call the
  reference backend themselves, without a message, and `explain()` still
  names `triton`.
- **A shared-pattern batch on the reference backend** is executed as a Python
  loop over the batch members.
- **`A @ x` and `A.matvec(x)` on a `Sparse`** use the gather reference path of
  `btorch.sparse` and never a backend kernel. Only `SparseConnection` reaches
  the kernels.
- **`torch.func` transforms** (`vmap`, `func.grad`, forward-mode
  differentiation) are not supported through a connection.
- **Index memory.** All index buffers are `int64` and both CSR layouts are
  always built, which costs 52 bytes per edge on CPU in addition to the
  weights.
- **Import time.** Importing `btorch.models` also imports `torch._dynamo`
  through the registration of the custom operators, which takes roughly 2 s.
- **Kernel inputs are trusted.** The registered operators check the shapes of
  what a connection passes them, not arbitrary hand-built buffers: an index
  permutation or column index outside its range is undefined behaviour on
  the Triton backend where the ATen reference raises.
- **Sparse event input** (`propagate_events`) uses the ATen kernel directly:
  it is not behind a registered operator, has no CUDA kernel, and does not
  support network batches.
- **Double backward** through the propagation operators is not supported and
  raises.
- **Complex dtypes** are rejected by the propagation operators.
- **`A @ x` inside `torch.compile`** fails in PyTorch 2.11 because Dynamo
  does not dispatch `@` to user objects. `A.matvec(x)` and
  `sparse.matmul(A, x)` compile, as does every connection module.
- **Checkpoints** require the same number of edge slots as the module they
  are loaded into. A stochastic rule with a different seed can therefore
  produce a module that refuses the checkpoint; build with the same seed or
  with a rule of fixed edge count.
- **Soft Deep R** is not implemented. Exact soft rewiring keeps a latent
  parameter for every candidate edge, which is dense memory; a pooled
  approximation would need a distinct name.
- **`HardDeepR`** supports only an unbatched, trainable `EdgeWeight`.
  `ConstantWeight`, `ConstrainedWeight`, fixed weights and network batches
  are refused.
- **Parameter adoption depends on the edge order.** An `nn.Parameter` passed
  as weight stays the trained object only when the input entries are already
  sorted by target and then source. A CSR adjacency matrix and the edges of a
  random rule are not, so the common case is a copy with a warning.
- **`Synapse(plasticity=...)`** is reserved; any value other than `None`
  raises `NotImplementedError`.
- **`value_version`** is derived from the in-place version counters of the
  weight tensors, so it changes on optimizer steps, but nothing consumes it
  yet.
- **Hints** `expected_calls` and `expected_batch` are accepted and not yet
  used by the planner.
- **Distributed / offloaded execution** is not implemented. Storage,
  the logical sparse object and connection semantics are separate layers so
  that placement can be added below the connection.

## 12. Test status

`pytest tests/sparse tests/models tests/connectome tests/test_pipeline_e2e.py`
passes with four expected failures, all the Dynamo `@` limitation listed
above. What the new tests cover:

| Area | Files |
|---|---|
| Conversion round trips (SciPy COO/CSR/CSC/BSR/LIL/DOK/DIA, torch COO/CSR/CSC/BSR/BSC), `as_sparse`, `EdgeMap` | `tests/sparse/test_conversion.py`, `test_as_sparse.py`, `test_edge_map.py` |
| Products against dense references, batches, gradients, closures under `torch.compile` | `tests/sparse/test_matmul.py`, `test_batch.py`, `test_grad.py` |
| `einsum`, linear operators | `tests/sparse/test_einsum.py`, `test_operator.py` |
| Registered operators (`opcheck`, `gradcheck`), caches, backends, eager/compiled parity | `tests/sparse/test_runtime_ops.py`, `test_runtime_propagate.py` |
| Triton kernels against the ATen reference | `tests/sparse/test_kernels_triton_pull.py`, `test_kernels_triton_push.py` |
| `SparseConnection`: construction, orientation, weights, Dale's law, gradients, `fullgraph` compile, checkpoints, batches, receptor/delay routing against the hetersynapse helpers | `tests/models/connection/test_sparse_connection*.py`, `test_connection_*.py` |
| Rules, `Projection`, `HardDeepR` | `tests/models/connection/test_rule.py`, `test_projection.py`, `test_rewire.py` |
| Regression tests for both code reviews | `tests/models/connection/test_review_regressions.py` |

The Triton tests run only where CUDA and Triton are available; they pass on
an RTX 5090 and on a small local GPU with the current kernels. Tests that
need the optional `torch_sparse` package are skipped where it is missing.
