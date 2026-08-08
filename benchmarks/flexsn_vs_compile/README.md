# FlexSN vs `torch.compile`

Does spikingjelly's FlexSN (a Triton kernel generated from a single-step
neuron function) actually beat generic `torch.compile`? A spiking neuron is a
chain of elementwise ops, which should be easy for inductor to fuse. These
scripts test that hypothesis on a "complicated LIF" neuron with adaptive and
fixed thresholds, dual reset, and an ATan surrogate.

All backends live in `kernels.py` and are verified numerically identical
(forward outputs and input gradients) before any timing:

- eager Python time loop (reference),
- `torch.compile` of an **unrolled** Python loop,
- `torch.compile` of the `scan` higher-order op,
- FlexSN (Triton backend),
- a hand-written CuPy `RawKernel` (forward + analytical BPTT).

Timing uses `btorch.utils.bench.do_bench` (mean +/- std, CUDA-event timed, L2
flushed between reps); the CUDA-graph variants use triton's `do_bench_cudagraph`.

## Usage

```bash
# Figure 1: short-T bars (inference | training), inductor vs FlexSN vs CuPy
python benchmarks/flexsn_vs_compile/bench_short_t.py --N 1048576 --T 2 4 8 16

# Figure 2: T sweep up to 1024, FlexSN vs compile / CUDA-graph strategies + CuPy
python benchmarks/flexsn_vs_compile/bench_sweep.py --N 32768 --max-flexsn-T 1024

# correctness gates
python benchmarks/flexsn_vs_compile/test_cupy.py     # CuPy backend
python benchmarks/flexsn_vs_compile/test_flexsn.py   # FlexSN, incl. init-state grads

# render both figures from the JSON results
python benchmarks/flexsn_vs_compile/plot.py
```

Long runs are best on SLURM (`short_t.sbatch`, `sweep.sbatch`). The sweep
appends every result to JSON as it goes and supports `--append` to resume a
partial run. Figures/JSON land under `fig/.../flexsn_vs_compile/` via `fig_path`.

## What the data shows (RTX 5090, torch 2.11)

- For short `T`, the unrolled `torch.compile` path fuses the whole sequence into
  one register-resident kernel and matches FlexSN/CuPy at inference -- the
  "SNN is just elementwise ops" intuition holds.
- The `scan` lowering emits one kernel launch **per timestep** (a host-side
  while-loop); capturing it in a CUDA graph removes that overhead.
- The hand-written **CuPy float4** kernel is the best all-rounder: fastest
  training at every `T`, competitive inference, ~0.4 s one-time compile, and it
  runs at any `T`/`N`.
- With the FlexSN fix below, FlexSN is fastest at **inference** across the range
  and second (behind CuPy) at training, with a flat ~1.7-1.9 s compile.

## Notes / findings

**FlexSN slow-compile was a template bug, not `tl.static_range` per se.**
FlexSN was locked into `tl.static_range` (full unroll -> ~quadratic Triton
compile: 2154 s for T=64 training, infeasible beyond). It *couldn't* use
`tl.range` because its backward `grad_init_state_store` referenced the loop
variable `t` **after** the loop -- valid only when unrolling makes `t` a
constant (`0` after the reverse loop). Under `tl.range`, `t` is the terminal
value `-1`, so `offsets=(-1, ...)` wrote one row before the buffer (a wild
`Invalid __global__ write`, caught by `compute-sanitizer`). The forward's
`final_state_store` correctly used `offsets=(0, ...)` -- the inconsistency was
the tell. Two-line fix in `spikingjelly/.../flexsn/template.py`:
`static_range -> tl.range` and `grad_init` store `offsets=(t,...) -> (0,...)`,
`shape=(T,NCL) -> (1,NCL)`. Verified correct to ~1e-6 vs eager including
`grad_v0`/`grad_rho0` (`test_flexsn.py`); compile drops ~1270x. Worth
upstreaming to spikingjelly.

**The T=8 inductor-inference dip (T=8 slower than T=16) is a fusion split, not
noise.** Directly profiled at N=1M: at T<=8 inductor emits *two* kernels -- a
recurrence kernel (66 us) plus a separate `stack` output-materialization kernel
(162 us) that reloads per-step intermediates and writes `[T,N]` with masks. At
T=16 it fuses both into one kernel (190 us). So T=8 (229 us total) > T=16
(190 us) despite fewer steps. It's inductor's fusion heuristic flipping between
"split" and "fused" as the unrolled graph grows (and it fragments into many
kernels again past ~T=32, hence the T=32+ cliff).

---

# PDL vs CUDA-graphs vs persistent kernel (sparse RSNN multistep)

`bench_pdl.py` is a separate study on the **same launch-overhead question**, but
for a workload that is a *hard* sequential recurrence: a GLIF3 recurrent spiking
network (RSNN) with a **5% scale-free recurrent connection** `W`, stepped for
`T=64`. Step `t` needs every spike of `t-1` (`x_in = x[t] + bias + W @ s[t-1]`),
so the launch boundary *is* the cross-step barrier — one CSR-vector SpMV + GLIF3
update kernel per timestep. It asks: what actually removes that per-step launch
cost, and when does it matter?

Four strategies, each in **Triton**, **CuPy**, and **TileLang** (`bench_pdl.py`):

- `plain` — `T` ordinary per-step launches (baseline).
- `pdl` — **Programmatic Dependent Launch** (sm_90+): step `t+1` starts in the
  drain shadow of `t`, `griddepcontrol.wait` gating only the spike dependency.
  Triton `launch_pdl=True` + `gdc_wait`/`gdc_launch_dependents`; TileLang
  `pdl_sync`/`pdl_trigger`; CuPy inline-PTX `griddepcontrol` + `cuLaunchKernelEx`
  with `PROGRAMMATIC_STREAM_SERIALIZATION` (via `cuda-python`), including an
  **L2-prefetch prologue** that streams this row's `val`/`col` before the wait.
- `cudagraph` — the `T` plain launches captured once and replayed.
- `persistent` — one cooperative launch runs the whole `T`-loop with `grid.sync`
  between steps (CuPy `cg::this_grid().sync`, TileLang `T.sync_grid`; reused from
  `glif_net`). No relaunch, but the grid is capped at SM co-residency. *Triton /
  Warp cannot express this* — only the cooperative backends have a `persistent`
  cell, which is itself the point.

## What the data shows (RTX 5090, T=64, speedup vs each DSL's `plain`)

| N | DSL | plain (ms) | PDL | cudagraph | persistent |
|------:|----------|----:|-----:|-----:|-----:|
| 512 | triton | 1.435 | 0.99x | **11.1x** | — |
| 512 | cupy | 0.621 | 0.62x | 3.70x | 2.25x |
| 512 | tilelang | 1.274 | 1.01x | 9.71x | **8.77x** |
| 2048 | triton | 1.449 | 1.02x | 4.62x | — |
| 2048 | cupy | 0.621 | 0.62x | 2.91x | 1.31x |
| 2048 | tilelang | 1.279 | 1.01x | 5.07x | 4.89x |
| 8192 | triton | 1.431 | 1.01x | 1.56x | — |
| 8192 | cupy | 0.639 | 0.63x | 1.01x | 0.64x |
| 8192 | tilelang | 1.281 | 1.02x | 1.60x | 1.21x |
| 24576 | triton | 9.557 | 1.01x | 1.00x | — |
| 24576 | cupy | 8.352 | 0.98x | 1.01x | 0.89x |
| 24576 | tilelang | 9.198 | 1.01x | 1.01x | 0.80x |

**The whole story is a launch-bound → bandwidth-bound crossover at ~N=8192.**

- **CUDA graphs** are the only broadly-winning trick, and only while launches
  dominate: **9–11x at N=512**, decaying monotonically to **~1.0x at N=24576**.
  Removing `T` host launches (~5 µs each) is worth a lot when a step is ~20 µs
  and nothing when it is ~150 µs.
- **Persistent** is strong at small N (**8.8x**, TileLang, N=512) but **inverts
  at scale — 0.80–0.89x at N=24576.** The cooperative launch pins the grid to
  one block/SM, so at large N it under-subscribes the memory system that the
  full-grid per-step launches saturate. It trades relaunch cost for occupancy,
  and past the crossover that is a losing trade.
- **PDL is ~1.0x for the whole sweep** (Triton/TileLang 0.98–1.02x). At small N
  the step is *launch*-bound, and PDL hides launch *latency* but not the launch
  *cost* graphs remove — so graphs win and PDL doesn't. At large N the step is
  *bandwidth*-bound: the previous kernel already saturates HBM, so there is no
  spare bandwidth for the next grid to prefetch into (the L2-prefetch prologue
  has nothing to overlap). PDL only helps kernels bounded by inter-launch
  *latency* with spare memory/compute to overlap — this sparse SpMV is bounded by
  neither at the same time.
- **CuPy PDL is 0.62x** — not a PDL effect but the launch *mechanism*: the
  `cuda-python` `cuLaunchKernelEx` host path is heavier than CuPy's native
  `RawKernel.__call__`, and at small N that extra host cost per launch dominates.
  The same kernel via the native launcher (`plain`) is 1.6x faster.

**Bottom line for the N=24576 sparse RSNN:** it is memory-bandwidth-bound, so
none of PDL / graphs / persistent helps (0.80–1.01x); the per-step launch is
already optimal and the win, if any, must come from the *kernel* (bandwidth), not
the *launch*. The launch-side tricks pay off only in the launch-bound regime
(N ≲ 4096), where CUDA graphs are the robust choice and a persistent fused kernel
is competitive on the cooperative backends.

## Numerical note

Backends match the torch (cuSPARSE) reference **exactly through T=16** at
N=24576; at T=64 the per-step Triton and the CuPy-persistent schedules differ by
~2000/24576 spikes (`n_mismatch`). This is float **summation-order** sensitivity,
not a bug: the ~1200-term hub-row dot products reduce in a different order than
cuSPARSE, flipping borderline threshold crossings that then compound over the
64-step recurrence (CuPy per-step even goes 0→2→0 mismatches as T grows). The
harness reports the spike-mismatch *count*, not max-error, for exactly this
reason.

## Usage

```bash
# sweep launch-bound -> GPU-bound (default N = 512 2048 8192 24576), T=64
python -m benchmarks.flexsn_vs_compile.bench_pdl --N 512 2048 8192 24576 --T 64
# or a single size; heavy — prefer SLURM (dedicated GPU = clean timing)
sbatch benchmarks/flexsn_vs_compile/pdl.sbatch
```
