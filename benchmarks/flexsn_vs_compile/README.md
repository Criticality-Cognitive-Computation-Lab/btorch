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
