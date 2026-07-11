# Persistent-kernel investigation — thought history

A narrative of how this work unfolded, including dead-ends and corrections, so the
reasoning (not just the conclusions in `report.md`) is recoverable. Chronological.

## 0. Starting point

Task: understand the new CUDA persistent-kernel backend for recurrent LIF+ExponentialPSC
SNN dynamics, compare it against CUDA graphs, and explain why it wasn't fast. The kernel
is one `cudaLaunchCooperativeKernel` that loops all T timesteps internally with `grid.sync()`
barriers, implementing an event-driven "prespan" fan-out (spike list as an atomic work queue).

## 1. Building a fair comparison

Built `benchmark_rsnn_cudagraph_compare.py` with providers sharing one RSNN step function:
eager, `torch.compile(reduce-overhead)`, manual CUDA graph (native CSR), and persistent.
Later added `eager_prespan` and `cudagraph_prespan` — the *same* prespan algorithm the
persistent kernel implements, but as ordinary Triton kernel launches captured in a graph —
for a true apples-to-apples "cooperative kernel vs same-algorithm-in-a-graph" comparison.

Two CUDA-graph capture bugs found and fixed while building this:
- `CSR.mm`'s `repeat_interleave` isn't stream-capture-safe (output size resolved by a sync).
  Fix: precompute the per-edge row index outside the graph, feed a capture-safe mm.
- Capture-only tensors (`static_v0`, `static_psc0`, `row`) were reclaimed after `_build()`
  returned; the graph still wrote those addresses on replay → silent corruption (replay #1
  fine, #2 wrong, #3 OOB assert). Fix: keep them alive in the returned state dict.

## 2. First "why is it slow" pass — a wrong conclusion, then caught

`nsys` on a T=1 case: the kernel ran ~28µs but the wrapper cost ~1.4ms. First hypothesis:
redundant `.item()` syncs (delay check duplicated in Python+C++, event-offset readback done
unconditionally). Deduplicated them (Fix A).

**The number barely moved (1.46ms → 1.37ms).** That mismatch — a fix that should have helped
by nsys's own API summary but didn't move wall-clock — was the signal. Isolated by calling
the raw op directly (bypassing all Python validation): still ~1.26ms. So the cost was inside
the op but invisible to nsys's CUDA API trace.

**Real cause:** `cooperative_grid_dim()` re-ran `cudaGetDeviceProperties` +
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on *every* forward call — synchronous
CPU-blocking driver calls (not stream ops, so not in nsys's memcpy/sync/launch summary),
~1.1–1.2ms. Cached it (Fix B). T=1 floor: 1.37ms → 0.23ms.

**Lesson recorded:** when a fix doesn't move the number you predicted, isolate before writing
it up — don't assume the fix's category was the mechanism.

## 3. Chunked CUDA graphs — answering "do we need to run 1000 steps first?"

Full-capture graphs must run all T steps once at capture. Question: avoid that for long/
streaming rollouts while still recording full history? Yes — capture one short window, replay
it `T/chunk` times. Two design points: (a) carry `v`/`psc` in place across replays (captured
region copies its own output state back to its input buffers → RNN-style hidden state at fixed
addresses); (b) write per-chunk spikes into a full-length buffer with an *eager* copy between
replays (not captured), so it can index dynamically. Recording all T doesn't need capturing
all T. Measured ~3–5% overhead. The persistent kernel sidesteps this entirely — single
cooperative launch, fast from call #1 for any T.

## 4. Enhancing the kernel — the real per-step gap

Read the fan-out phase (Task 3). Three per-step pathologies, all inside one kernel so
invisible to nsys:
1. **`work_counter` atomic storm:** every ~160K grid threads `atomicAdd` one global int
   though only ~330 neurons spike/step (~99.8% wasted).
2. **Underutilization:** only ~330 threads do fan-out work.
3. **Serial edge walk:** one thread per neuron — brutal for the production graph's skew
   (P99 ~1911, max ~4165), and the trailing `grid.sync()` waits for the slowest.

`plan.md` confirmed the production fanout is highly skewed → chose warp-level **work-stealing**
(dynamic balancing) over a static grid-stride split. Change B: one warp per neuron, lane 0
claims via one atomicAdd (32× fewer), 32 lanes walk edges in parallel. Spotted a free Change A:
fold `psc *= decay` into the LIF loop (each thread owns its cell) → removes a grid pass + a
`grid.sync()` (5→4 barriers).

Result: ~1.4× across the sweep; at T=128 beats both cudagraph variants, at T=256 beats prespan.
Small-T still floor-limited (launch overhead, not fan-out).

### Benchmarking discipline learned the hard way

First enhanced sweep was ruined by editing the `.cu` while a sweep was running — the JIT loader
hashes sources and tried to rebuild mid-run, failed (ninja didn't inherit `CPATH`), and
clobbered the `.so`. Redone cleanly: benchmark baseline first (committed kernel), then apply
the enhancement, rebuild, benchmark — two back-to-back sweeps on the same idle GPU.
(Also: this is a shared 4×4090 node; watched `nvidia-smi` and pinned an idle GPU throughout.)

## 5. Correctness — a second self-correction

Claimed the persistent kernel's large-T diff was "1-spike near-threshold nondeterminism."
Pushed to verify rigorously:
- Chunked match is **non-trivial**: 736k spikes at T=64, ramping across chunks; a
  reset-state-each-chunk control diverges by 546k spikes → the test discriminates.
- CUDA graphs match the dense reference **exactly** at every T (they reuse the dense sum order).
- Persistent: exact at T≤64; at T≥128 it differs by **10** spikes (not 1) out of ~2M.
- Determinism check (6 runs, identical input): **byte-identical every run.** So it is
  **deterministic**, not nondeterminism — a float accumulation-order difference (event
  `atomicAdd` order vs dense matmul) flipping ~10 exactly-at-threshold cells.
- Baseline vs enhanced: the mismatching set is the **exact same 10 `(t,b,n)` cells** →
  enhanced output is bit-identical to baseline. My change altered nothing about correctness.

Corrected the report and the commit message accordingly. (Earlier drafts — and the prior
report — called this "nondeterminism"; that was wrong.)

## 6. Course correction — revert the rebalance, and the fair comparison

Feedback: don't touch work balance; the warp-per-neuron fan-out (Change B) rebalanced work,
so it was **reverted** — the fan-out is back to the original one-thread-per-neuron scheme.
Change A (folded PSC decay) was kept, but with B gone its measured gain is marginal (~1–3%;
the A-only kernel ≈ the original). So the micro-optimizations don't move the needle much.

The real question resurfaced: is a fair comparison even being made? The old set mixed
*algorithm* and *dispatch* — and it's worth being precise, because only ONE pair is truly
same-code:
- `persistent_prespan_cuda` (cooperative) and `cudagraph_prespan_cuda` (graph): **identical**
  hand-CUDA phase code (the stepped op is literally the cooperative kernel's 4 grid.sync()
  phases split into 4 launches). This is the controlled, dispatch-only comparison.
- `cudagraph_prespan` (Triton): same prespan *idea* but a different work distribution (2D
  event×edge tiling over a padded CSR layout) — not same code.
- `cudagraph_native_sparse`: a different algorithm entirely (dense scatter over all edges).

So to isolate dispatch, added `cudagraph_prespan_cuda` via a new
`persistent_snn_forward_stepped` op. (A cuSPARSE-in-the-kernel idea was dropped — cuSPARSE is
host-only, can't be called from a cooperative kernel.)

**Result:** same-code, dispatch-only — the cooperative kernel is **~2.8–3.8× slower** than the
graph of per-step kernels, and `cudagraph_prespan_cuda` is the fastest provider overall. So
the persistent kernel's slowness is the **cooperative single-launch design** (occupancy-capped
co-resident grid + grid.sync() cost), not the prespan algorithm. Takeaway: prefer a graph of
ordinary event-driven kernels; the cooperative kernel buys launch-overhead avoidance that a
CUDA graph already provides, while giving up occupancy and cheap barriers.

Naming: cooperative = `persistent_prespan_cuda`, graph-of-same-code = `cudagraph_prespan_cuda`
(prefix = dispatch, suffix = implementation; parallel to the Triton `cudagraph_prespan`).

## Open item (out of scope, unfixed)

The ~10-spike deterministic divergence from the *dense* reference at T≥128 is a property of
the event-driven summation order, present in baseline and enhanced alike, within `plan.md`'s
<1e-4 state tolerance. Worth a separate look only if exact dense-equivalence on the binary
spike readout is required.

## Environment quirks (for reproducers)

- JIT loader assumes a system CUDA layout; conda env needs
  `CPATH=$CONDA_PREFIX/targets/x86_64-linux/include` and `LIBRARY_PATH=$CONDA_PREFIX/lib`.
- `rm -rf /tmp/btorch_extensions/btorch_persistent_snn_plain` to force a rebuild after `.cu` edits.
- `ncu` unusable here (`ERR_NVGPUCTRPERM`, no passwordless sudo); all profiling via `nsys` +
  targeted microbenchmarks.
- Commits need `-c commit.gpgsign=false` (no GPG secret key in this env) and must run through
  the `ml-py312` env so `pre-commit` (ruff/docformatter) is on PATH.
