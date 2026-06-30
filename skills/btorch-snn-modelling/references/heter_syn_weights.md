# Group-Aware Weight Stacking for Heterogeneous Synapses

When weights or delays are generated with different rules per receptor group, build per-group sparse matrices first, then stack them into the single matrix that btorch expects.

## Decision Table

| Situation | Tool / Mode |
|-----------|-------------|
| Have a dict of per-receptor-group sparse matrices | `stack_hetersynapse(conn_mats, receptor_type_index)` |
| Need to map generated per-post-neuron weights onto an existing sparse structure | `map_weight_to_conn(weights, conn_mat, mode=...)` |
| No ordering preference | `mode="random"` |
| Preserve strength topology (largest weights to strongest original positions) | `mode="large_to_large"` |

## Stacking Per-Receptor Matrices

```python
from btorch.connectome.connection import stack_hetersynapse

# conn_mats is an OrderedDict: {("E", "E"): spmat_ee, ("E", "I"): spmat_ei, ...}
conn_stacked, new_receptor_idx = stack_hetersynapse(
    conn_mats=conn_mats,
    receptor_type_index=receptor_idx,
)
```

`stack_hetersynapse` produces a single sparse matrix with expanded columns, matching the format expected by `SparseConn` and `HeterSynapsePSC`.

## Mapping Group-Specific Weights Back to Sparse Connectivity

```python
from network_generator.algorithms.weight_gen import map_weight_to_conn

# weights can be a flat array or a per-post-neuron list of arrays
conn_with_weights = map_weight_to_conn(
    weight=weights,
    conn_mat=conn_mat,
    mode="large_to_large",  # or "random"
    rng=42,
)
```

### Assignment Modes

- **`random`**: Shuffles the generated weights randomly before assigning them to the non-zero edges of `conn_mat`. Use this when the source distribution is what matters, not which specific edge gets which weight.
- **`large_to_large`**: Sorts the existing non-zero values in `conn_mat` by magnitude (descending), sorts the generated weights by magnitude, and assigns the largest generated weight to the position that originally had the largest value. Use this when you want to preserve the topological strength structure of a reference connectivity pattern.

## DataFrame → Sparse → Stack Workflow

A typical pipeline looks like this:

1. **Prepare per-group connections** as DataFrames (one row per edge with `pre_simple_id`, `post_simple_id`, `syn_count`).
2. **Convert each group** to a sparse matrix (e.g., via `make_sparse_mat` or `make_hetersynapse_conn(return_dict=True)`).
3. **Apply group-specific weight rules** to each sparse matrix (or generate weights separately and map them with `map_weight_to_conn`).
4. **Stack** the per-group matrices with `stack_hetersynapse` (or use `make_hetersynapse_conn` directly if you started from a single DataFrame).
5. **Pass the stacked matrix** to `SparseConn` or `SparseConstrainedConn.from_hetersynapse`.

## Common Pitfalls

1. **Double expansion** — `stack_hetersynapse` and `make_hetersynapse_conn` both produce column-expanded matrices. Do not apply column expansion twice. If you already used `make_hetersynapse_conn`, pass the resulting matrix directly to `SparseConn`; do not stack again.
2. **Misaligned weight lists** — When using `map_weight_to_conn` with a list of per-post-neuron arrays, the length of the list must match the number of post-synaptic columns (`conn_mat.shape[1]`). Each array's length must match the in-degree of that column.
3. **Forgetting `receptor_type_index`** — `stack_hetersynapse` requires the `receptor_type_index` DataFrame to know the ordering of receptor groups. Keep it alongside the stacked matrix so `HeterSynapsePSC` can resolve receptor indices correctly.
