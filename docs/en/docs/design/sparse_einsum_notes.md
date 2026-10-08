# Sparse einsum: scope and what a compiler would add

Status: **experimental**. `btorch.sparse.einsum.einsum(subscripts, A, *operands)`
contracts exactly one sparse array (the first operand) with zero or more dense
tensors and returns a dense tensor equal to
`torch.einsum(subscripts, A.to_dense(), *operands)`.

## What is implemented

- One letter per logical axis of `A.shape == (*batch, *sparse, *dense)`.
  Letters on COO coordinate axes are *indexed*; letters on axes of `values`
  (shared-pattern batch, dense payload) are *non-indexed*.
- Any output order, implicit output, summed-out letters, several dense
  operands, diagonals over repeated sparse letters. CSR/CSC go via `tocoo()`.
- Raises for: ellipsis, a second sparse operand, a letter repeated across an
  indexed and a non-indexed axis, size mismatches (no size-1 broadcasting).

## Lowering and cost model

With `E` stored entries:

1. *Filter* entries whose repeated coordinates disagree (diagonals only).
2. *Gather* every dense operand along the indexed letters it shares with `A`:
   `[E, *other axes]`.
3. *Contract* values and gathered operands with one `torch.einsum` over the
   entry axis, giving contributions `[E, *non-indexed output letters]`.
4. *Scatter* with `index_add` into the linearised surviving indexed letters,
   then reshape and permute. Skipped if no indexed letter survives.

Time and temporary memory are `O(E * P)`, where `P` is the product of the sizes
of the non-indexed letters involved; the peak temporaries are the gathered
operands and the contributions (plus pairwise intermediates inside
`torch.einsum`). Nothing scales with `prod(sparse_shape)` except the dense
output. Example: `"ijk,bk,bj->bi"` with `B = 8` peaks at about `4 * 8 * E`
elements (two gathered operands, one intermediate, the contributions).

The lowering is one fixed loop nest: iterate the entries of `A`, look up
everything else by random access. This is optimal only because every other
operand and the output are dense.

## What a Scorch/TACO-style compiler would have to add

- **Per-level format abstraction.** Describe each dimension by a level format
  (dense, compressed, singleton, hashed, ...) with `locate`/`iterate`/`append`
  capabilities; COO, CSR, CSC, DCSR and CSF become compositions of levels in a
  chosen mode order. Today COO is the only N-D format and the mode order is
  irrelevant because entries are visited flat.
- **Iteration graph and co-iteration.** For sparse × sparse the loop over an
  index variable must merge several sparse levels. This needs an iteration
  graph (a legal variable order consistent with every operand's level order,
  else a transposition/sort) and merge lattices per variable: intersection for
  multiplication, union for addition, with locate-based lookup when a level
  supports random access.
- **Workspaces.** When the result of an inner loop is scattered into a sparse
  or out-of-order target (SpGEMM row accumulation, sparse outputs with a
  non-concordant order), a dense or hashed temporary must be inserted and the
  expression split around it.
- **Sparse outputs.** Assembling a compressed result needs a *symbolic* phase
  (count or bound the entries, build pointers/coordinates) and a *numeric*
  phase (fill values), or append-capable levels. Autograd must then be defined
  on the stored values with a fixed pattern.
- **Scheduling.** Loop order, contraction order, splitting/fusion and
  parallelism; here delegated to `torch.einsum` over the entry axis. TACO emits
  scalar loops, so a PyTorch backend must also express merges with sorts,
  `searchsorted` and segment reductions (or custom kernels) to stay
  differentiable and fast on GPU.
