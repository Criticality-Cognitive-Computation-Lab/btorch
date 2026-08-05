"""Full GLIF3 kernel benchmark: sweep neuron count N and sequence length T.

Measures every ``{backend} x {kind} x {mode}`` combination:

- kind : ``neuron`` (neuron-only multistep), ``dense`` (dense recurrent
         multistep), ``sparse`` (5% scale-free recurrent multistep)
- mode : ``inference`` (forward, no grad) or ``training`` (forward + backward)
- backend : ``triton`` / ``warp`` / ``cupy`` / ``tilelang``, plus a
            ``torch.compile`` (mode ``reduce-overhead``) baseline on the
            canonical eager btorch net (``*/compile`` rows, dense + sparse)

Two sweeps (8 points each): N at fixed T, and T at a medium N. Results (median
ms, ``None`` on out-of-memory) are written to JSON for ``plot_glif_kernels.py``.
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
            neuron/warp     0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25
            neuron/cupy     0.06    0.06    0.06    0.06    0.06    0.06    0.06    0.06
        neuron/tilelang     0.04    0.04    0.04    0.04    0.04    0.04    0.04    0.04
           dense/triton     1.57    1.55    1.62    1.56    5.21   20.41   80.94     OOM
             dense/warp     5.43    5.46    5.45    5.46    5.49   20.02   80.60     OOM
             dense/cupy     0.15    0.21    0.35    0.75    5.51   20.88   80.91     OOM
         dense/tilelang     1.65    1.65    1.66    1.67    4.76   19.87   80.42     OOM
          dense/compile     1.89    1.65    1.89    1.77    5.86   20.94   81.54     OOM
          sparse/triton     1.01    1.00    1.00    1.00    1.00    1.61    8.86   34.03
            sparse/warp     6.67    6.66    6.68    6.67    6.67    6.67    9.08   35.66
            sparse/cupy     0.14    0.15    0.25    0.32    0.49    1.26    8.32   34.11
        sparse/tilelang     0.09    0.13    0.21    0.46    1.14    3.97   23.81   83.10
         sparse/compile     6.24    6.27    6.23    6.36    6.24    6.23    9.57     OOM

    training | N sweep (T=32)
           kind/backend      512    1024    2048    4096    8192   16384   32768   65536
          neuron/triton     0.42    0.45    0.44    0.45    0.45    0.43    0.46    0.44
            neuron/warp     1.28    1.29    1.29    1.30    1.38    1.31    1.32    1.32
            neuron/cupy     0.44    0.44    0.45    0.45    0.45    0.45    0.46    0.45
        neuron/tilelang     4.16    4.15    4.26    4.23    4.20    4.26    4.23    4.23
           dense/triton     8.17    7.76    7.88    8.32   32.68   128.0   495.8     OOM
             dense/warp    33.86   35.31   34.20   34.69   37.50   129.1   496.8     OOM
             dense/cupy     7.47    7.63    7.54    8.43   32.69   128.0   495.8     OOM
         dense/tilelang     6.62    6.71    6.50    8.43   32.65   127.9   495.7     OOM
          dense/compile    59.54   59.50   63.41   65.54   92.61   195.1   608.1     OOM
          sparse/triton     9.93   10.10   10.19   10.28   10.11   31.54   122.2   473.2
            sparse/warp    43.32   43.45   45.22   44.98   66.69   107.5   265.6   732.2
            sparse/cupy    11.39   12.46   12.20   12.99   13.37   31.54   122.2   473.5
        sparse/tilelang     8.84    8.79    9.14    8.86    8.88   31.55   122.2   472.9
         sparse/compile    741.3   687.0   608.9   618.0   726.7  1253.0  3075.0     OOM

    inference | T sweep (N=8192)
           kind/backend        4       8      16      32      64     128     256     512
          neuron/triton     0.06    0.06    0.06    0.06    0.06    0.06    0.06    0.12
            neuron/warp     0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25
            neuron/cupy     0.06    0.06    0.06    0.06    0.06    0.08    0.16    0.31
        neuron/tilelang     0.04    0.04    0.04    0.04    0.04    0.04    0.05    0.09
           dense/triton     0.66    1.31    2.61    5.21   10.42   20.78   41.54   83.07
             dense/warp     0.76    1.43    2.78    5.49   10.91   21.65   43.27   87.04
             dense/cupy     0.55    1.26    2.68    5.51   11.18   22.47   45.04   90.38
         dense/tilelang     0.60    1.19    2.38    4.76    9.58   19.40   40.08   84.38
          dense/compile     1.71    2.43    3.73    5.84   11.59   20.55   41.39   84.99
          sparse/triton     0.15    0.26    0.46    0.87    1.69    3.30    6.62   13.07
            sparse/warp     0.91    1.74    3.38    6.68   13.23   26.32   52.97   105.8
            sparse/cupy     0.07    0.12    0.24    0.49    1.00    2.00    3.98    7.81
        sparse/tilelang     0.15    0.29    0.57    1.15    2.28    4.59    9.19   18.33
         sparse/compile     2.00    2.61    3.82    6.37   11.16   20.76   39.92   78.22

    training | T sweep (N=8192)
           kind/backend        4       8      16      32      64     128     256     512
          neuron/triton     0.42    0.43    0.45    0.44    0.44    0.44    0.44    1.00
            neuron/warp     1.30    1.29    1.29    1.30    1.30    1.30    1.32    1.47
            neuron/cupy     0.45    0.44    0.45    0.46    0.45    0.46    0.49    1.12
        neuron/tilelang     0.78    1.29    2.31    4.24    8.45   17.18   32.68   63.77
           dense/triton     3.55    7.72   16.02   32.67   66.05   133.1   268.2   540.4
             dense/warp     4.51    8.95   18.67   37.20   73.35   151.9   360.7   769.8
             dense/cupy     3.55    7.75   16.03   32.70   66.08   133.1   268.3   540.7
         dense/tilelang     3.54    7.70   16.02   32.66   65.97   133.0   268.0   539.9
          dense/compile      OOM     OOM   44.24   92.93   225.7   631.4  2009.3  7111.2
          sparse/triton     1.52    2.69    5.01    9.38   20.58   40.24   74.51   142.1
            sparse/warp     5.80   11.15   44.10   44.48   85.21   164.5   343.9   716.1
            sparse/cupy     1.70    2.99    5.52   10.83   23.08   43.71   81.30   156.7
        sparse/tilelang     1.34    2.39    4.39    8.23   17.16   33.67   64.89   148.5
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


def _time_ms(fn, grad, iters):
    """Median-ish wall time per call. Warms up generously and auto-calibrates the
    rep count so the fixed per-measurement overhead (host dispatch + the single
    trailing ``cuda.synchronize``) is amortized. With only a couple of warmups
    and ~10 reps, sub-0.1 ms kernels — TileLang's especially, whose per-call
    dispatch transient is larger than Triton's — read several× too slow; batching
    reps to a ~50 ms window fixes it for every backend."""
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
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000.0


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
