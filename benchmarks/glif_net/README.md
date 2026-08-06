# GLIF3 recurrent SNN kernels

Reference kernels for a GLIF3 spiking network with a recurrent connection,
across four GPU backends — **Triton**, **Warp**, **CuPy**, and **TileLang** —
plus eager PyTorch. Each backend covers the single step, the neuron-only
multistep, and the recurrent multistep (dense all-to-all **and** 5 % scale-free
sparse), for both **inference** and **training**. TileLang additionally provides
a **persistent** fused sparse kernel that runs the whole recurrence in one
cooperative launch (`T.sync_grid()` barrier per step), keeping neuron state
resident and fusing the CSR-vector SpMV with the neuron update in a single
launch — which stock Triton/Warp/CuPy cannot express (see below).

The GLIF3 update per step (exponential-Euler membrane, heaviside spike with an
ATan surrogate gradient, `M` after-spike currents):

```
v_inf  = v_rest + tau * (x + sum I_asc) / c_m
v'     = v_inf + (v - v_inf) * exp(-dt / tau)
spike  = heaviside(v' - v_th) * not_refrac
v_post = v' - (v' - v_reset) * spike      (hard reset)
       = v' - (v_th - v_reset) * spike    (soft reset)
I_asc' = I_asc * exp(-k dt) + asc_amps * spike
```

with recurrent input `x_in[t] = x[t] + bias + weight @ spike[t-1]`.

## Interface

Every backend exposes one `GLIF3StepOps` object (`glif_common.py`) with a
uniform surface:

| method | purpose |
|---|---|
| `step(...)` | single GLIF3 step (the autograd training primitive) |
| `multistep_fused(...)` | neuron-only multistep |
| `dense_multistep_fused(...)` | neuron + dense all-to-all recurrent multistep |
| `sparse_multistep_fused(...)` | neuron + 5 % scale-free sparse recurrent multistep |

The multistep entries route on gradient need: `torch.is_grad_enabled()` and any
input requiring grad selects the training path; otherwise a lean forward-only
kernel runs with **no autograd bookkeeping**. Training uses explicit forward and
backward kernels, except Warp, which uses `wp.Tape` autodiff.

```python
from benchmarks.glif_net.glif_triton import glif3_step_triton

with torch.no_grad():                      # inference: no graph is built
    spike_seq, v_seq, v_out, I_out = glif3_step_triton.dense_multistep_fused(
        x_seq=x_seq, weight=W, bias=b, v=v0, Iasc=Iasc0, params=params,
        not_refrac=not_refrac, dt=1.0, M=M, hard_reset=False,
    )
```

## Dense multistep: one barrier per step

The recurrent term `weight @ spike[t-1]` couples every neuron, so all spikes of
step `t` must be visible before step `t+1` — a grid-wide barrier each step. The
backends realize it differently (inference):

- **Triton** — `fused_matmul=True` runs a **fused per-step kernel**: each block
  computes its neuron rows' recurrent gemv `W[rows,:] @ spike[t-1]` with a
  **pipelined column loop** (`tl.range(num_stages=…)` overlaps the weight-tile
  streaming with the accumulate) fused with the neuron update, one launch per
  step. The launch boundary is the (proper, sanctioned) cross-step barrier — no
  hand-rolled grid sync. `fused_matmul=False` uses **cuBLAS** for the matmul plus
  a neuron kernel per step.
- **Warp** — the matmul is delegated to **cuBLAS** and the neuron dynamics run in
  a Warp kernel **per step**.
- **CuPy** — one **cooperative launch** runs the whole loop with an in-kernel
  **coalesced block-per-row** matmul and `cg::this_grid().sync()` between steps
  (the only backend with a real device-wide grid-sync primitive).

Training (all backends) composes the single-step autograd op with a per-step
`torch.mv`, so gradients for `weight` / `bias` / `x` / initial state / `asc_amps`
fall out of the single-step backward.

## Performance (RTX 5090, sm_120)

**Dense inference, `N = 32768`, `T = 64`, `M = 2`.** The kernel re-reads the
`N×N` weight (4.3 GB) every step, so it is HBM-bound; the floor is
`T·N²·4 / peak ≈ 150 ms` at ~1.8 TB/s. All backends now saturate it and produce
the same spikes.

| backend / path | time | effective BW |
|---|---:|---:|
| Triton — fused per-step (pipelined gemv) | 162 ms | 1666 GB/s |
| Triton — cuBLAS per-step | 162 ms | 1674 GB/s |
| Warp — cuBLAS per-step | 162 ms | 1671 GB/s |
| CuPy — cooperative fused (1 launch) | 164 ms | 1651 GB/s |

**Latency across all paths, `N = 8192`, `T = 32`, `M = 2`** (ms, forward for
inference; forward+backward for training):

| backend | neuron infer | neuron train | dense infer | dense train |
|---|---:|---:|---:|---:|
| Triton | 0.06 | 0.45 | 5.2 | 33 |
| Warp | 0.26 | 1.3 | 5.4 | 38 |
| CuPy | 0.06 | 0.46 | 5.5 | 33 |
| TileLang | **0.04** | **0.36** | **4.8** | 33 |
| `torch.compile` (reduce-overhead) | — | — | 5.9 | 93 |

TileLang is the **fastest** backend for neuron inference, neuron training, and
dense inference, and ties on dense training. Its scalar one-thread-per-neuron
kernels fuse the whole GLIF3 step in registers, and the neuron multistep runs
the T-step loop *inside* one kernel (a runtime `T.serial` loop, not unrolled) —
for training a single **fused BPTT** kernel walks the T steps in reverse carrying
the reverse-time adjoints in registers, one launch instead of T.

Neuron-multistep inference is a single fused kernel (µs-scale); dense training is
dominated by the per-step autograd graph. Dense inference is close across
backends and converges to the HBM ceiling as `N` grows (table above).

### vs `torch.compile(mode="reduce-overhead")`

As a general-purpose baseline, the canonical eager btorch recurrent nets —
`GLIFDenseNet` (`GLIF3` neuron + `nn.Linear`) and `GLIFSparseNet` (a 5%
scale-free **cuSPARSE** CSR SpMV, `torch.sparse.mm` on a CSR tensor, *not*
torch's slower native COO) — compiled with `mode="reduce-overhead"` (Inductor +
CUDA-graph). Overlaid on both sweeps as pink squares. `N=8192`, `T=32` (ms):

| workload | custom kernel | torch.compile | ratio |
|---|---:|---:|---:|
| dense inference | 5.2 (triton) | 5.9 | ~1.1× |
| dense training | 33 (triton) | 93 | ~2.8× |
| sparse inference | 0.86 (triton) / 0.51 (cupy) | 6.2 | ~7–12× |
| sparse training | 9.9 (triton) | 727 | ~73× |

- **Inference.** Dense is HBM-bound on re-reading the weight, so Inductor's
  fused per-step launch matches the kernels once bandwidth saturates. Sparse is
  ~8–13× slower: the compiled path calls cuSPARSE `csrmv` for the SpMV but does
  **not** fuse it with the neuron update, and at these sizes the per-step
  launches (not the SpMV) dominate.
- **Training is far slower — and it is mostly *not* compute.** See below.

#### Why training is slow (profiler + Nsight Compute)

Two independent overheads, both absent from the fused kernels:

1. **Unfused neuron update = a swarm of launch-bound kernels.** `ncu` on the
   eager dense-training pass shows the GLIF3 update decomposed into dozens of
   tiny elementwise kernels per step (`add`/`mul`/`div`/`neg`/`reciprocal`/
   `expm1`/…), each ~2.5 µs at **<0.1% SM and <2% DRAM utilisation** — pure
   launch overhead — and autograd roughly doubles the count (one adjoint kernel
   per forward op). Inductor fuses most of these (so `default`-mode compile
   beats eager), but the custom kernels fuse the *entire* step — neuron dynamics
   and the recurrent gemv/SpMV — into **one** kernel per step.
2. **`reduce-overhead` re-captures the CUDA graph every iteration.** The torch
   profiler shows only ~133 ms of actual GPU kernel time for a `T=128` dense
   training iter, yet wall-clock is ~680 ms — a ~5× gap. The stateful net's
   `reset_net_state` reallocates the neuron state each iteration, which
   invalidates the captured graph, so reduce-overhead **re-records** it every
   call instead of replaying (plus ~28% of the GPU time is `Memcpy DtoD` from
   graph I/O copies). So the training numbers are dominated by graph re-capture,
   not arithmetic; a loop that reset state in place (letting the graph replay)
   would recover most of the gap.

Sparse training is pathological (`T=256`: ~25 s) because the cuSPARSE CSR SpMV
**graph-breaks** under dynamo — reduce-overhead cannot form one graph and
re-captures fragmented pieces every step (`default`-mode compile, without CUDA
graphs, is ~6× faster there). The neuron-only m-step is left kernel-only (it
doesn't trace under reduce-overhead, and is launch-overhead-bound anyway).

## Sparse recurrent connection (scale-free SpMV)

A dense recurrent weight makes inference HBM-bound on re-reading the whole `N×N`
matrix. Real cortical connectivity is sparse and **scale-free** — a power-law
degree distribution where a few hub neurons carry most of the edges. With the
weight at **5 % nonzeros** the recurrent term becomes a sparse matrix–vector
product (SpMV), and `scale_free_csr` (`glif_common.py`) builds the CSR once (the
one-time preprocess; no runtime coalesce). `.sparse_multistep_fused(...)` takes
that `SparseWeight` in place of the dense weight.

All three backends use the classic **CSR-vector** schedule — one cooperative
*group per row* reduces `lin_i = Σ_p val[p]·spike[t-1][col[p]]` over the row's
nonzeros, so the scale-free load imbalance (a few hub rows with many nonzeros)
is spread across lanes rather than stalling one thread:

- **Triton** — one *warp per row* (`num_warps=1`, program-per-row), fused with
  the neuron update, one launch per step.
- **CuPy** — one *warp per row* inside a single **cooperative launch** with a
  `cg::this_grid().sync()` between steps.
- **Warp** — one *tile per row* (`_TILE_LANES` threads), reduced with the
  cooperative `wp.tile_sum` (Warp's intended tile idiom, not a hand-rolled
  atomic or a thread-per-row scalar kernel).

Training shares a memory-efficient custom SpMV autograd op (`glif_common._SparseSpmv`):
both forward and backward are O(nnz) scatter/gathers, so **no dense `(N, N)`
gradient is ever materialized**. (`torch.sparse.mm`'s backward does materialize
it and OOMs at `N = 2**15` — ~27 GB vs ~1.6 GB here.)

**Four tables — `T = 32`, `M = 2`, 5 % scale-free, RTX 5090** (ms; forward for
inference, forward+backward for training). These are slices of the full
`bench_glif_full.py` sweep; `plot_glif_kernels.py` renders the full 8-point N
and T sweeps as `glif_{dense,sparse}_sweep.{png,pdf}` (inference/training rows,
N-sweep/T-sweep columns; recurrent solid, neuron-only baseline dashed).

*Neuron multistep — inference*
| backend | N=8192 | N=16384 | N=32768 | N=65536 |
|---|---:|---:|---:|---:|
| Triton | 0.06 | 0.06 | 0.06 | 0.06 |
| Warp | 0.26 | 0.27 | 0.26 | 0.27 |
| CuPy | 0.06 | 0.06 | 0.06 | 0.06 |
| TileLang | 0.04 | 0.04 | 0.04 | 0.04 |

*Neuron multistep — training*
| backend | N=8192 | N=16384 | N=32768 | N=65536 |
|---|---:|---:|---:|---:|
| Triton | 0.45 | 0.45 | 0.45 | 0.45 |
| Warp | 1.32 | 1.28 | 1.29 | 1.34 |
| CuPy | 0.46 | 0.47 | 0.47 | 0.48 |
| TileLang | 0.36 | 0.36 | 0.36 | 0.37 |

*Sparse recurrent multistep — inference*
| backend | N=8192 | N=16384 | N=32768 | N=65536 |
|---|---:|---:|---:|---:|
| Triton (CSR-vector) | 0.86 | 1.58 | 8.77 | 34.4 |
| Warp (CSR-vector) | 6.49 | 6.47 | 9.18 | 36.0 |
| CuPy (CSR-vector) | 0.51 | 1.23 | 8.23 | 35.1 |
| TileLang (CSR-vector, persistent) | 1.0 | 2.9 | 17.8 | 62.7 |

*Sparse recurrent multistep — training*
| backend | N=8192 | N=16384 | N=32768 | N=65536 |
|---|---:|---:|---:|---:|
| Triton | 9.9 | 31.0 | 121 | 470 |
| Warp | 44.8 | 109 | 261 | 709 |
| CuPy | 10.3 | 30.9 | 121 | 470 |
| TileLang | 8.1 | 31.6 | 122 | 474 |

The neuron-only multistep is a single fused kernel whose runtime is flat in `N`
(µs-scale, memory-bandwidth-trivial); the sparse recurrent term is what grows
the cost, becoming HBM-bound on streaming the CSR `val`+`col` arrays once `nnz`
exceeds L2 (the knee near `N=2**15`). Warp's per-launch overhead dominates its
sparse inference until then, so its curve is flat-then-rising rather than
monotone. At the same `N`, dense recurrent (table above) costs ~10× the sparse
SpMV for inference — it re-reads the full `N×N` weight — and OOMs at `N=65536`
where the `N×N` weight (16 GB, ×2 with its gradient) no longer fits.

### TileLang persistent fused sparse kernel

The stock backends realize the sparse recurrent inference as **T separate kernel
launches** — each launch is the cross-step barrier, and the neuron state
(`v`, `Iasc`) round-trips global memory across every boundary. TileLang expresses
what they cannot: **one cooperative launch for the whole recurrence**
(`_sparse_persistent` in `glif_tilelang.py`). The grid stays resident; each
timestep does a CSR-vector SpMV (same warp-per-row schedule as the siblings —
`rows` rows/block × one warp each, reduced with `T.reduce_sum`), fuses the GLIF3
update, double-buffers the spike vector in a grid-global scratch, and a single
`T.sync_grid()` separates steps. No relaunch, and only the spike vector (not the
full neuron state) crosses the barrier.

It is **correct** (matches the eager reference and the sibling kernels). The
**raw CSR-vector SpMV is competitive** — a single full-grid SpMV is ~24 µs,
matching cuSPARSE (~22 µs). The cooperative grid is sized the TileLang-sanctioned
way: `driver.get_num_sms()`, one block per SM, exactly as every persistent
example and `T.PersistentTileScheduler`'s default. TileLang exposes **no**
occupancy API and no CUfunction handle, so it can't do what CuPy does — CuPy
sizes its cooperative grid with `occupancyMaxActiveBlocksPerMultiprocessor(...)
* numSMs` on the raw CUfunction (that product is exactly the cooperative-launch
limit). In TileLang's one-block-per-SM model you add parallelism with a *bigger*
block: one 1024-thread CTA per SM, 32 rows × one warp (`_SP_ROWS`/`_SP_LANES`).

The remaining ~2× vs CuPy at large N was **isolated by ablation**: zeroing each
component and re-timing (`N=8192`, full = 0.99 ms) attributes **0.91 ms (92%) to
the CSR gather loop itself**, 0.02 ms to the per-step `sync_grid`, and 0.04 ms to
the neuron update + persistent bookkeeping (at `N=32768` the gather is 99%). So
the gap is the gather `acc += val[p] * s[col[p]]`, and every other suspect is
minor. Three hypotheses about *why the gather is slower than CuPy's* were tested
and ruled out:
- **Not occupancy** — ncu: TileLang 67% achieved warp occupancy vs CuPy's *lower*
  ~17%, both near-zero DRAM (the random gather is latency-bound). More occupied,
  yet slower.
- **Not the reduction** — `T.reduce_sum(dim=1)` lowers to a `tl::AllReduce` with a
  block `NamedBarrier`, but rewriting it as `T.warp_reduce_sum` (CuPy's exact
  `__shfl_down_sync` idiom) is bit-exact and the *same* time (0.99 ms).
- **Not loop scheduling** — unrolling the gather (`T.unroll(unroll_factor=2/4/8)`,
  to give nvcc a window to overlap the independent loads) does **not** help
  (0.99 → 1.00 ms); the `col[p]`→`s[col[p]]` load chain isn't reorderable enough.
What remains is the **generated scalar-gather code quality** — TileLang's per-step
gather (~28 µs) vs CuPy's hand-written pointer gather (~15 µs, faster than even
cuSPARSE's 22 µs by keeping the CSR arrays + spike vector hot in L2 across steps).
Matching it would need tighter gather codegen than the accessible TileLang knobs
produce; it is a real codegen gap, not a missing primitive or a config mistake.
Net: TileLang sparse inference **wins at small N** (`N=512`: **0.09 ms** vs
Triton 0.87, CuPy 0.14), ties around `N=8192` (1.0 vs 0.85/0.48), and is ~2×
behind at large N (`N=65536`: 63 ms vs Triton/CuPy ~34). Sparse **training**
(composing the TileLang single-step through the shared autograd path) is the
**fastest** backend at `N=8192` (8.1 ms vs Triton 10.0, CuPy 10.5), tying at
larger N where the shared SpMV backward dominates.

### Roofline

SpMV streams the CSR arrays `val`+`col` (8 B/nonzero, ~320 MB at `N=2**15`) while
the gathered spike vector `s` (128 KB) fits L2, so arithmetic intensity is
`2·nnz / 8·nnz ≈ 0.25 FLOP/B` — deep in the memory-bound regime. Achieved DRAM
bandwidth (analytical, cross-checked with Nsight Compute) vs the cuSPARSE
`csrmv_v3` reference, at `N=2**15`, `T=32` (RTX 5090, HBM peak 1792 GB/s):

| backend | DRAM BW | % of HBM peak |
|---|---:|---:|
| Triton (CSR-vector) | ~1220 GB/s | ~68 % |
| CuPy (CSR-vector) | ~1300 GB/s | ~73 % |
| Warp (CSR-vector, `wp.tile_sum`) | ~1150 GB/s | ~64 % |
| cuSPARSE (SpMV only) | ~1260 GB/s | ~70 % |

## Running

```bash
# correctness (all backends, inference + training)
pytest benchmarks/glif_net/test_glif_kernels.py

# quick tables (neuron + sparse, three N; prints to stdout)
python -m benchmarks.glif_net.bench_glif_sparse

# full sweep — {neuron, dense, sparse} x {inference, training} x 3 backends,
# 8 N points and 8 T points -> JSON. Heavy; run on a GPU node:
sbatch benchmarks/glif_net/bench_glif.slurm
#   ... or directly:
python -m benchmarks.glif_net.bench_glif_full \
    --out benchmarks/glif_net/bench_glif_full_results.json

# Nature-style figures from the sweep JSON (dense and sparse in separate 2x2
# panels: rows = inference/training, cols = N-sweep/T-sweep; solid = recurrent,
# dashed = neuron-only baseline). Writes glif_{dense,sparse}_sweep.{png,pdf}:
python -m benchmarks.glif_net.plot_glif_kernels \
    --results benchmarks/glif_net/bench_glif_full_results.json
```
