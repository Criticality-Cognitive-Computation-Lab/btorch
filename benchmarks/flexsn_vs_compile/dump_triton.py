"""Dump the Triton code inductor generates for the unrolled short-T neuron,
plus an annotated reading of the result.

The script re-execs itself with TORCH_LOGS=output_code to capture the exact
Python + Triton module inductor compiles for the unrolled time loop
(inference and training), writing them under the resolved figure directory.
Run: python benchmark/flexsn_vs_compile/dump_triton.py --T 4 --N 32768

----

# How inductor compiles the unrolled short-T neuron

Annotated reading of `inf_T4_N32768.py` and `train_T4_N32768.py` (the raw
`output_code` inductor emits for `torch.compile(eager_loop, dynamic=False)` at
`T=4, N=32768`). The point: a *generic* compiler, given an unrolled Python
time loop, produces essentially the same kernel a hand-written spiking-neuron
Triton template would — the recurrence runs entirely in registers, and nothing
touches DRAM between timesteps.

Reference: the neuron per step is
`h = 0.9v + x; s1 = sg(h-(ρ+1)); s2 = sg(h-1); ρ = 0.8ρ + s1;
yy = σ(y); v = h(1-s1)·yy + (h-s2)(1-yy)`, surrogate `sg` = straight-through
ATan (`soft + (spike-soft).detach()`).

---

## Inference: 2 kernels

`call()` launches exactly two kernels:

| kernel | grid (`xnumel`) | role |
|--------|-----------------|------|
| `..._sigmoid_sub_0` | **32768** = N | sequential recurrence, state in registers |
| `..._stack_sub_1` | **131072** = T·N | write the stacked `[T,N]` spike outputs |

Inductor **split by iteration space**, which is exactly the right call:

- the recurrence is *sequential in T but parallel in N* → run it with N threads,
  each looping the 4 steps internally (state stays in registers);
- the output tensors are *`[T,N]` and fully parallel* → a separate T·N-wide
  pointwise kernel writes them, **recomputing** the cheap surrogate from a few
  saved scalars instead of round-tripping every spike through DRAM.

### Kernel 0 — the register-resident scan (cleaned, step 0 shown in full)

The raw code is 130 `tmpNN =` lines; here is step 0 with real names. Pointers:
`in_ptr0=v0, in_ptr1=x_seq, in_ptr2=rho0, in_ptr3=y_seq`.

```python
@triton.jit
def recurrence(v0, x_seq, rho0, y_seq, *out, xnumel, XBLOCK: tl.constexpr):
    x0 = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)   # one lane per neuron

    # --- all inputs loaded ONCE, up front (4 steps of x and y) ---
    v    = tl.load(v0   + x0)                 # tmp0
    rho  = tl.load(rho0 + x0)                 # tmp5
    x_t0 = tl.load(x_seq + x0)                # tmp3   (x[0])
    y_t0 = tl.load(y_seq + x0)                # tmp21  (y[0])
    x_t1 = tl.load(x_seq + (32768  + x0))     # tmp37  (x[1])  ← T dim is just
    x_t2 = tl.load(x_seq + (65536  + x0))     # tmp71  (x[2])    a compile-time
    x_t3 = tl.load(x_seq + (98304  + x0))     # tmp100 (x[3])    address offset
    # ... y[1..3] likewise ...

    # ================= step t=0 (tmp1..tmp41) =================
    h   = v * 0.9 + x_t0                      # tmp4   h = βv + x
    a1  = h - (rho + 1.0)                     # tmp8   adaptive-threshold arg
    soft= libdevice.atan(a1 * pi) * (1/pi)    # tmp13  surrogate soft value
    hard= (a1 >= 0.0).to(tl.float32)          # tmp16  heaviside spike
    s1  = soft + (hard - soft)                # tmp18  straight-through == hard
    v1  = h * (1.0 - s1)                       # tmp20  hard-reset branch
    yy  = tl.sigmoid(y_t0)                     # tmp22  modulation
    a2  = h - 1.0                              # tmp24  fixed-threshold arg
    s2  = ...atan(a2*pi)...                    # tmp31  (same surrogate on a2)
    v2  = h - s2                               # tmp32  soft-reset branch
    v   = v1 * yy + v2 * (1.0 - yy)            # tmp35  new membrane  ← in register
    rho = rho * 0.8 + s1                       # tmp41  new adaptation ← in register

    # ================= step t=1 reuses v, rho directly =================
    h   = v * 0.9 + x_t1                       # tmp38  NO store/load between steps!
    # ... identical body, tmp42..tmp70 ...
    # ================= steps t=2, t=3 likewise (tmp71..tmp130) =========

    # only the FINAL states and a few per-step scalars are written to DRAM:
    tl.store(out_final_v   + x0, v)            # buf8  -> returned as final v
    tl.store(out_final_rho + x0, rho)          # buf9  -> returned as final rho
    tl.store(out_scalars_k + x0, ...)          # buf0..5: h / s / yy reused by K1
```

The critical line is `h = v * 0.9 + x_t1` at the top of step 1: the updated
`v` and `rho` from step 0 are **SSA values in registers**, consumed directly by
step 1. Across all 4 steps the state is never spilled — total DRAM traffic for
the recurrence is `2·T` reads (x, y) + a handful of writes, i.e. *memory-bound
at the theoretical floor*. This is the identical dataflow to FlexSN's
`tl.static_range(T)` template, reached here purely by dynamo unrolling +
inductor's pointwise fusion.

### Kernel 1 — parallel output write

`xnumel = 131072 = 4·32768`, so one lane per `(t, neuron)` output element. Each
lane figures out which timestep it belongs to (`x0 < 32768`, `< 65536`, …) with
`tl.where` masks and recomputes that step's spike from kernel 0's saved
scalars. Recompute-from-scalars is cheaper than having kernel 0 write all
`2·T·N` spikes and reloading them — a classic materialize-vs-recompute tradeoff
inductor makes automatically.

---

## Training: 2 forward + 3 backward kernels

`train_T4_N32768.py` has two `call()` graphs:

**Forward (2 kernels)** — identical shape to inference: the register-resident
recurrence (K0, N threads) + stacked-output write (K1, T·N threads). K0
additionally returns `primals_*` (the inputs) as saved-for-backward.

**Backward (3 kernels):**

| kernel | grid | role |
|--------|------|------|
| `..._div_ge_..._0` | 32768 = N | reverse-time BPTT, grad-state in registers |
| `..._select_backward_..._1` | 131072 | per-timestep grad of stacked outputs |
| `..._add_mul_select_backward_2` | 131072 | final accumulation → `grad_x`, `grad_y` |

The first backward kernel is the important one: it is the reverse recurrence
done the same way as the forward — **N threads, T steps unrolled in registers,
grad-state (`grad_v`, `grad_rho`) threaded through registers**, no per-step
launches. Contrast the `scan` lowering, whose backward materializes
`aten.flip` copies of every saved sequence and runs a second host-driven
while-loop. That difference is exactly why unrolled training is ~2× faster than
scan at short T.

---

## The optimizations inductor achieved, in one list

1. **Whole-sequence fusion.** The 4-step Python loop + surrogate + dual reset —
   ~130 aten ops — collapse into a *single* compute kernel (K0). No per-op or
   per-step kernel launches.
2. **Register-resident recurrent state.** `v` and `ρ` live in registers across
   all timesteps; zero intermediate DRAM. DRAM traffic = `2·T` input reads +
   final states, the memory-bound floor for this neuron.
3. **Iteration-space splitting.** Sequential recurrence (N-wide) and parallel
   output materialization (T·N-wide) become separate kernels with the right
   grid each, instead of one ill-shaped kernel.
4. **Materialize-vs-recompute.** Cheap surrogate is recomputed in the output
   kernel rather than stored/reloaded, cutting write+read of `2·T·N` spikes.
5. **Constant folding + vectorization.** β, γ, `1/π`, thresholds are baked in as
   `tl.full` immediates; `XBLOCK` and `num_warps` are autotuned per shape;
   `tt.divisibility 16` hints let Triton emit vectorized 128-bit loads.
6. **Register-resident reverse-time BPTT.** The backward is the forward pattern
   mirrored — no `aten.flip`, no host loop — which the `scan` path cannot do.

**Caveat (why T doesn't scale):** every one of these depends on the loop being
*unrolled at compile time*, so K0's register footprint grows with T. That is
the source of the non-monotonic timing (T=8 ≈ 12.3 µs > T=16 ≈ 10.3 µs — the
autotuner lands on different occupancy per shape) and the cliff at T=32
(≈261 µs, register spilling) and OOM/fragmentation past a few hundred steps.
Unrolling is what makes short-T fast *and* what makes long-T infeasible.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def _child() -> None:
    """Run one workload; inductor prints output_code to stderr (captured)."""
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kernels import eager_loop

    T = int(os.environ["DUMP_T"])
    N = int(os.environ["DUMP_N"])
    mode = os.environ["DUMP_MODE"]
    dev = "cuda"
    x = torch.randn(T, N, device=dev)
    y = torch.randn(T, N, device=dev)
    v0 = torch.zeros(N, device=dev)
    rho0 = torch.zeros(N, device=dev)
    f = torch.compile(eager_loop, fullgraph=True, dynamic=False)

    if mode == "inf":
        with torch.no_grad():
            f(x, y, v0, rho0)
    else:
        xg = x.requires_grad_(True)
        yg = y.requires_grad_(True)
        s1, s2, _, _ = f(xg, yg, v0, rho0)
        (s1.sum() + s2.sum()).backward()


def _extract_output_code(log: str) -> str:
    """Strip the log prefix, keeping only the generated-module source lines."""
    out = []
    for line in log.splitlines():
        if "[__output_code]" in line:
            out.append(line.split("[__output_code]", 1)[1].lstrip())
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--N", type=int, default=32768)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    if os.environ.get("DUMP_CHILD") == "1":
        _child()
        return

    from btorch.utils.file import fig_path

    outdir = args.outdir or (fig_path() / "generated_triton")
    outdir.mkdir(parents=True, exist_ok=True)

    for mode in ("inf", "train"):
        env = dict(os.environ,
                   DUMP_CHILD="1", DUMP_MODE=mode,
                   DUMP_T=str(args.T), DUMP_N=str(args.N),
                   TORCH_LOGS="output_code")
        proc = subprocess.run([sys.executable, __file__], env=env,
                              capture_output=True, text=True)
        code = _extract_output_code(proc.stderr + proc.stdout)
        if not code.strip():
            print(f"[{mode}] no output_code captured; stderr tail:\n"
                  f"{proc.stderr[-800:]}")
            continue
        dst = outdir / f"{mode}_T{args.T}_N{args.N}.py"
        dst.write_text(code)
        n_kernels = code.count("async_compile.triton(")
        print(f"[{mode}] wrote {dst}  ({n_kernels} triton kernel(s))")


if __name__ == "__main__":
    main()
