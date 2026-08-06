"""Full GLIF3 kernel benchmark: sweep neuron count N and sequence length T.

Measures every ``{backend} x {kind} x {mode}`` combination:

- kind : ``neuron`` (neuron-only multistep), ``dense`` (dense recurrent
         multistep), ``sparse`` (5% scale-free recurrent multistep)
- mode : ``inference`` (forward, no grad) or ``training`` (forward + backward)
- backend : ``triton`` / ``warp`` / ``cupy`` / ``tilelang``, plus a
            ``torch.compile`` (mode ``reduce-overhead``) baseline on the
            canonical eager btorch net (``*/compile`` rows, dense + sparse)

Two sweeps (8 points each): N at fixed T, and T at a medium N. Results (min over
several timed windows to reject shared-GPU contention; ``None`` on out-of-memory)
are written to JSON for ``plot_glif_kernels.py``.
This is the heavy full benchmark (``bench_glif_sparse.py`` is the quick one); run
it on a GPU node via ``bench_glif.slurm``::

    python -m benchmarks.glif_net.bench_glif_full --out results.json

Results (RTX 5090, M=2, 5% scale-free sparse; ms per pass). ``inference`` is
forward only, ``training`` is forward+backward. ``OOM`` marks out of memory; for
the ``*/compile`` rows a blank cell is OOM, a reduce-overhead trace failure
(dense training T<16), or not measured (sparse training T>=256, where CUDA-graph
re-capture makes it impractically slow to time)::

    inference | N sweep (T=32)
           kind/backend      512    1024    2048    4096    8192   16384   32768   65536
          neuron/triton     0.06    0.06    0.06    0.06    0.06    0.06    0.06    0.06
            neuron/warp     0.29    0.29    0.29    0.29    0.29    0.26    0.25    0.25
            neuron/cupy     0.06    0.06    0.06    0.06    0.06    0.06    0.06    0.06
        neuron/tilelang     0.04    0.04    0.04    0.04    0.04    0.04    0.04    0.04
           dense/triton     1.56    1.38    1.37    1.37    5.23   20.56   80.93     OOM
             dense/warp     5.35    5.36    5.35    5.37    5.37   20.03   80.58     OOM
             dense/cupy     0.15    0.21    0.35    0.75    5.52   20.89   80.91     OOM
         dense/tilelang     1.67    1.69    1.67    1.68    4.80   19.96   80.60     OOM
          dense/compile     1.89    1.65    1.89    1.77    5.86   20.94   81.54     OOM
          sparse/triton     0.87    0.86    0.86    0.86    0.85    1.54    8.86   33.94
            sparse/warp     6.52    6.51    6.49    6.50    6.50    6.50    9.00   35.64
            sparse/cupy     0.14    0.15    0.25    0.34    0.48    1.18    8.28   33.91
        sparse/tilelang     0.09    0.13    0.24    0.47    1.01    2.86   17.81   62.68
         sparse/compile     6.24    6.27    6.23    6.36    6.24    6.23    9.57     OOM

    training | N sweep (T=32)
           kind/backend      512    1024    2048    4096    8192   16384   32768   65536
          neuron/triton     0.42    0.42    0.41    0.42    0.45    0.46    0.43    0.46
            neuron/warp     1.27    1.31    1.22    1.26    1.23    1.24    1.24    1.25
            neuron/cupy     0.43    0.43    0.43    0.44    0.44    0.44    0.44    0.44
        neuron/tilelang     0.35    0.35    0.35    0.37    0.36    0.36    0.36    0.37
           dense/triton     7.71    7.78    7.71    8.25   32.65   127.9   495.7     OOM
             dense/warp    33.06   34.46   34.55   34.51   36.99   129.0   497.0     OOM
             dense/cupy     7.51    7.62    7.51    8.28   32.67   127.9   495.7     OOM
         dense/tilelang     6.51    6.67    6.62    8.20   32.63   128.0   495.8     OOM
          dense/compile    59.54   59.50   63.41   65.54   92.61   195.1   608.1     OOM
          sparse/triton     9.97   11.90   11.83   11.86   10.02   31.59   121.5   473.0
            sparse/warp    46.94   47.54   48.04   48.09   48.85   107.9   266.3   731.7
            sparse/cupy     9.63   10.25   10.66   10.62   10.46   31.62   121.6   473.1
        sparse/tilelang     8.14    8.18    8.16    8.15    8.11   31.60   121.6   474.3
         sparse/compile    741.3   687.0   608.9   618.0   726.7  1253.0  3075.0     OOM

    inference | T sweep (N=8192)
           kind/backend        4       8      16      32      64     128     256     512
          neuron/triton     0.06    0.06    0.06    0.06    0.06    0.06    0.06    0.12
            neuron/warp     0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25
            neuron/cupy     0.06    0.06    0.06    0.06    0.06    0.08    0.16    0.31
        neuron/tilelang     0.04    0.04    0.04    0.04    0.04    0.04    0.05    0.09
           dense/triton     0.66    1.31    2.62    5.23   10.44   20.88   41.68   83.29
             dense/warp     0.74    1.41    2.73    5.34   10.63   21.16   42.18   87.05
             dense/cupy     0.55    1.26    2.68    5.51   11.16   22.48   45.06   90.34
         dense/tilelang     0.60    1.19    2.37    4.77    9.56   19.38   40.15   84.29
          dense/compile     1.71    2.43    3.73    5.84   11.59   20.55   41.39   84.99
          sparse/triton     0.15    0.26    0.46    0.86    1.67    3.26    6.50   12.95
            sparse/warp     0.89    1.69    3.28    6.47   12.89   25.72   51.06   102.1
            sparse/cupy     0.07    0.12    0.23    0.48    0.97    1.97    3.93    7.77
        sparse/tilelang     0.13    0.25    0.52    1.02    2.08    4.10    8.12   15.78
         sparse/compile     2.00    2.61    3.82    6.37   11.16   20.76   39.92   78.22

    training | T sweep (N=8192)
           kind/backend        4       8      16      32      64     128     256     512
          neuron/triton     0.44    0.42    0.43    0.43    0.43    0.43    0.43    0.95
            neuron/warp     1.22    1.22    1.23    1.24    1.33    1.24    1.25    1.46
            neuron/cupy     0.43    0.43    0.44    0.46    0.46    0.47    0.49    1.13
        neuron/tilelang     0.35    0.35    0.36    0.37    0.36    0.45    0.45    0.56
           dense/triton     3.54    7.72   15.99   32.67   66.02   132.9   268.3   545.5
             dense/warp     4.47    9.03   19.50   38.17   73.60   151.1   342.1   752.2
             dense/cupy     3.54    7.69   15.99   32.68   66.04   132.9   268.4   545.6
         dense/tilelang     3.52    7.68   15.99   32.63   66.01   132.7   267.9   543.8
          dense/compile      OOM     OOM   44.24   92.93   225.7   631.4  2009.3  7111.2
          sparse/triton     1.52    2.69    4.97    9.33   20.15   38.98   73.67   149.2
            sparse/warp     5.78   11.09   23.08   44.68   87.67   169.8   345.2   734.8
            sparse/cupy     1.58    2.80    5.13    9.65   22.33   43.05   76.07   146.5
        sparse/tilelang     1.33    2.36    4.34    8.16   16.53   33.77   63.83   128.8
         sparse/compile    58.85   121.7   281.6   726.1  2168.3  7113.2     OOM     OOM
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from benchmarks.glif_net.glif_common import scale_free_csr

DEVICE = torch.device("cuda")
M, DT, ALPHA, DENSITY = 2, 1.0, 2.0, 0.05

N_SWEEP = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
N_SWEEP_T = 32                      # fixed T for the N sweep
T_SWEEP = [4, 8, 16, 32, 64, 128, 256, 512]
T_SWEEP_N = 8192                    # fixed (medium) N for the T sweep

KINDS = ["neuron", "dense", "sparse"]
MODES = ["inference", "training"]
BACKENDS = ["triton", "warp", "cupy", "tilelang"]


def _load_backends() -> dict:
    from benchmarks.glif_net.glif_triton import glif3_step_triton
    from benchmarks.glif_net.glif_warp import glif3_step_warp
    from benchmarks.glif_net.glif_cupy import glif3_step_cupy
    from benchmarks.glif_net.glif_tilelang import glif3_step_tilelang
    return {"triton": glif3_step_triton, "warp": glif3_step_warp,
            "cupy": glif3_step_cupy, "tilelang": glif3_step_tilelang}


def _inputs(N, T, kind, grad, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = 0.5 + 0.6 * torch.randn(T, N, generator=g, device=DEVICE)
    bias = 0.02 * torch.randn(N, generator=g, device=DEVICE)
    asc = 0.05 * torch.randn(N * M, generator=g, device=DEVICE)
    params = dict(
        v_th=torch.full((N,), -50.0, device=DEVICE), v_reset=torch.full((N,), -70.0, device=DEVICE),
        v_rest=torch.full((N,), -70.0, device=DEVICE), c_m=torch.full((N,), 0.05, device=DEVICE),
        tau=torch.full((N,), 20.0, device=DEVICE),
        k=0.1 + 0.2 * torch.rand(N * M, generator=g, device=DEVICE), asc_amps=asc)
    v = torch.full((N,), -65.0, device=DEVICE)
    Iasc = torch.zeros(N * M, device=DEVICE)
    nr = torch.ones(N, device=DEVICE)
    weight = None
    if kind == "dense":
        weight = torch.randn(N, N, generator=g, device=DEVICE) / N**0.5
    elif kind == "sparse":
        weight = scale_free_csr(N, DENSITY, DEVICE, seed=seed)
    if grad:
        leaves = [x, bias, asc]
        if kind == "dense":
            leaves.append(weight)
        elif kind == "sparse":
            leaves.append(weight.val)
        for t in leaves:
            t.requires_grad_(True)
    return x, bias, asc, params, v, Iasc, nr, weight


def _call(step, kind, x, bias, params, v, Iasc, nr, weight):
    common = dict(params=params, not_refrac=nr, dt=DT, M=M, hard_reset=False, alpha=ALPHA)
    if kind == "neuron":
        return step.multistep_fused(x_seq=x, v=v.clone(), Iasc=Iasc.clone(), **common)
    if kind == "dense":
        return step.dense_multistep_fused(
            x_seq=x, weight=weight, bias=bias, v=v.clone(), Iasc=Iasc.clone(), **common)
    return step.sparse_multistep_fused(
        x_seq=x, weight=weight, bias=bias, v=v.clone(), Iasc=Iasc.clone(), **common)


def _time_ms(fn, grad, iters, windows=5):
    """Min-of-N wall time per call. Warms up generously and auto-calibrates the
    rep count so the fixed per-measurement overhead (host dispatch + the trailing
    ``cuda.synchronize``) is amortized — with only a couple of warmups and ~10
    reps, sub-0.1 ms kernels (TileLang's especially, whose per-call dispatch
    transient is larger than Triton's) read several× too slow; batching reps to a
    ~50 ms window fixes it. Then time ``windows`` separate batches and take the
    **minimum**: on a shared GPU, transient contention from another process only
    ever inflates a window, so the min rejects those bursts (best-of-N)."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for _ in range(25):
            fn()
        torch.cuda.synchronize()
        # calibrate: per-call time from a batch (no per-call sync), then pick reps
        t0 = time.time()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        per_call = (time.time() - t0) / 20
        reps = min(1000, max(iters, int(0.05 / max(per_call, 1e-6))))
        best = float("inf")
        for _ in range(windows):
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            best = min(best, (time.time() - t0) / reps)
    return best * 1000.0


def measure(step, kind, mode, N, T):
    """Median ms for one cell, or None if it runs out of memory. The whole cell —
    input allocation included — is guarded, since building the (N, N) dense weight
    at large N is itself an OOM risk."""
    grad = mode == "training"
    data = None
    try:
        data = _inputs(N, T, kind, grad)
        x, bias, asc, params, v, Iasc, nr, weight = data
        leaves = [x, bias, asc] + ([weight] if kind == "dense" else
                                   [weight.val] if kind == "sparse" else [])

        def forward():
            return _call(step, kind, x, bias, params, v, Iasc, nr, weight)

        def train():
            for leaf in leaves:
                leaf.grad = None
            out = forward()
            sum(o.sum() for o in out).backward()

        return _time_ms(train if grad else forward, grad, iters=5 if grad else 10)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if "memory" not in str(exc).lower():
            raise
        return None
    finally:
        del data
        torch.cuda.empty_cache()


def compile_ms(kind, mode, N, T):
    """``torch.compile(mode="reduce-overhead")`` on the canonical eager btorch
    recurrent net, as a general-purpose baseline for the fused kernels:

    - ``dense``  -> ``GLIFDenseNet``  (GLIF3 neuron + ``nn.Linear`` recurrence),
    - ``sparse`` -> ``GLIFSparseNet`` (5% scale-free **cuSPARSE** CSR SpMV).

    Both are chunk-unrolled so compile cost stays flat in T. Median ms, or None
    on OOM / trace failure. Neuron-only stays kernel-only (its m-step doesn't
    trace under reduce-overhead, and it is launch-overhead-bound anyway)."""
    from benchmarks.glif_net.glif_common import (
        make_inputs, build_neuron, GLIFDenseNet, GLIFSparseNet)
    from btorch.models import environ
    from btorch.models.functional import init_net_state, reset_net_state
    grad = mode == "training"
    model = None
    try:
        weight, bias, x_seq, params = make_inputs(T, N, DEVICE, require_grad=grad)
        neuron = build_neuron("torch_eager", N, params, grad)
        if kind == "dense":
            model = GLIFDenseNet(N, neuron, unroll=8)
            init_net_state(model, device=DEVICE, dtype=torch.float32)
            model.linear.weight.data.copy_(weight)
            model.linear.bias.data.copy_(bias)
        else:
            W = scale_free_csr(N, DENSITY, DEVICE, seed=0)
            model = GLIFSparseNet(N, neuron, W, bias, unroll=8)
            init_net_state(model, device=DEVICE, dtype=torch.float32)
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

        def fn():
            torch.compiler.cudagraph_mark_step_begin()
            reset_net_state(model)
            with environ.context(dt=DT):
                if grad:
                    spike_seq, _ = model(x_seq)
                    spike_seq.sum().backward()
                else:
                    with torch.no_grad():
                        model(x_seq)

        # first calls compile + capture the CUDA graph; time steady-state replay.
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        iters = 5 if grad else 10
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000.0
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return None                      # OOM or a compile/trace failure -> N/A
    finally:
        del model
        torch.cuda.empty_cache()
        torch._dynamo.reset()


def run() -> dict:
    backends = _load_backends()
    out = {"meta": {"M": M, "dt": DT, "density": DENSITY,
                    "n_sweep": {"N": N_SWEEP, "T": N_SWEEP_T},
                    "t_sweep": {"T": T_SWEEP, "N": T_SWEEP_N}},
           "results": {}}
    for kind in KINDS:
        for mode in MODES:
            for backend, step in backends.items():
                key = f"{kind}/{mode}/{backend}"
                n_row = [measure(step, kind, mode, N, N_SWEEP_T) for N in N_SWEEP]
                t_row = [measure(step, kind, mode, T_SWEEP_N, T) for T in T_SWEEP]
                out["results"][key] = {"N": n_row, "T": t_row}
                print(f"{key:28} N-sweep {n_row}", flush=True)
                print(f"{key:28} T-sweep {t_row}", flush=True)
    # torch.compile(reduce-overhead) baseline, dense + sparse recurrent.
    for kind in ("dense", "sparse"):
        for mode in MODES:
            key = f"{kind}/{mode}/torch_compile"
            n_row = [compile_ms(kind, mode, N, N_SWEEP_T) for N in N_SWEEP]
            t_row = [compile_ms(kind, mode, T_SWEEP_N, T) for T in T_SWEEP]
            out["results"][key] = {"N": n_row, "T": t_row}
            print(f"{key:28} N-sweep {n_row}", flush=True)
            print(f"{key:28} T-sweep {t_row}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_glif_full_results.json")
    args = ap.parse_args()
    results = run()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
