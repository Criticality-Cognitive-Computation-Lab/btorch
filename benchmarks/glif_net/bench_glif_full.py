"""Full GLIF3 kernel benchmark: sweep neuron count N and sequence length T.

Measures every ``{backend} x {kind} x {mode}`` combination:

- kind : ``neuron`` (neuron-only multistep), ``dense`` (dense recurrent
         multistep), ``sparse`` (5% scale-free recurrent multistep)
- mode : ``inference`` (forward, no grad) or ``training`` (forward + backward)
- backend : ``triton`` / ``warp`` / ``cupy``, plus a ``torch.compile`` (mode
            ``reduce-overhead``) baseline on the canonical eager btorch net —
            dense recurrent only (see ``compile_dense_ms``)

Two sweeps (8 points each): N at fixed T, and T at a medium N. Results (median
ms, ``None`` on out-of-memory) are written to JSON for ``plot_glif_kernels.py``.
This is the heavy full benchmark (``bench_glif_sparse.py`` is the quick one); run
it on a GPU node via ``bench_glif.slurm``::

    python -m benchmarks.glif_net.bench_glif_full --out results.json

Results (RTX 5090, M=2, 5% scale-free sparse; ms per pass). ``inference`` is
forward only, ``training`` is forward+backward. ``OOM`` marks out of memory; for
the ``*/compile`` rows (torch.compile reduce-overhead, dense + sparse recurrent)
a blank cell is OOM, a reduce-overhead trace failure (dense training T<16), or
not measured (sparse training T>=256, where CUDA-graph re-capture makes it
impractically slow to time)::

    inference | N sweep (T=32)
           kind/backend      512    1024    2048    4096    8192   16384   32768   65536
          neuron/triton     0.07    0.06    0.06    0.06    0.06    0.06    0.06    0.06
            neuron/warp     0.29    0.26    0.26    0.27    0.26    0.27    0.26    0.27
            neuron/cupy     0.07    0.06    0.06    0.06    0.06    0.06    0.06    0.06
           dense/triton     1.33    1.33    1.33    1.34    5.23   20.49   80.84     OOM
             dense/warp     5.36    5.43    5.34    5.35    5.39   19.96   80.45     OOM
             dense/cupy     0.16    0.22    0.35    0.75    5.50   20.81   80.81     OOM
          dense/compile     1.89    1.65    1.89    1.77    5.86   20.94   81.54     OOM
          sparse/triton     0.87    0.86    0.86    0.86    0.86    1.58    8.77   34.36
            sparse/warp     6.50    6.47    6.51    6.51    6.49    6.47    9.18   36.03
            sparse/cupy     0.15    0.16    0.25    0.33    0.51    1.23    8.23   35.14
         sparse/compile     6.24    6.27    6.23    6.36    6.24    6.23    9.57     OOM

    training | N sweep (T=32)
           kind/backend      512    1024    2048    4096    8192   16384   32768   65536
          neuron/triton     0.73    0.45    0.44    0.44    0.45    0.45    0.45    0.45
            neuron/warp     1.36    1.27    1.28    1.28    1.32    1.28    1.29    1.34
            neuron/cupy     0.49    0.48    0.46    0.49    0.46    0.47    0.47    0.48
           dense/triton     7.65    7.74    7.63    8.27   32.62   127.7   495.0     OOM
             dense/warp    33.81   33.84   66.29   35.53   37.77   128.9   496.1     OOM
             dense/cupy     7.87    7.94    7.87    8.51   32.62   127.6   495.0     OOM
          dense/compile    59.54   59.50   63.41   65.54   92.61   195.1   608.1     OOM
          sparse/triton     9.96    9.93    9.88    9.97    9.93   31.00   120.7   470.4
            sparse/warp    42.77   42.84   74.73   45.06   44.82   109.3   261.3   708.9
            sparse/cupy    10.83   10.21   10.19   10.24   10.29   30.90   120.8   470.4
         sparse/compile    741.3   687.0   608.9   618.0   726.7  1253.0  3075.0     OOM

    inference | T sweep (N=8192)
           kind/backend        4       8      16      32      64     128     256     512
          neuron/triton     0.06    0.06    0.06    0.06    0.06    0.06    0.06    0.12
            neuron/warp     0.26    0.26    0.27    0.26    0.26    0.26    0.26    0.29
            neuron/cupy     0.06    0.06    0.06    0.06    0.06    0.08    0.16    0.31
           dense/triton     0.67    1.32    2.62    5.22   10.42   20.82   41.72   83.20
             dense/warp     0.73    1.40    2.72    5.45   10.67   21.26   42.40   86.33
             dense/cupy     0.56    1.26    2.69    5.49   11.13   22.43   44.93   90.14
          dense/compile     1.71    2.43    3.73    5.84   11.59   20.55   41.39   84.99
          sparse/triton     0.16    0.26    0.46    0.86    1.68    3.34    6.50   13.04
            sparse/warp     0.89    1.69    3.29    6.48   12.83   25.57   51.24   102.2
            sparse/cupy     0.09    0.13    0.25    0.51    1.01    1.97    3.85    7.78
         sparse/compile     2.00    2.61    3.82    6.37   11.16   20.76   39.92   78.22

    training | T sweep (N=8192)
           kind/backend        4       8      16      32      64     128     256     512
          neuron/triton     0.47    0.44    0.44    0.45    0.44    0.46    0.47    0.98
            neuron/warp     1.32    1.27    1.28    1.28    1.32    1.28    1.35    1.64
            neuron/cupy     0.47    0.47    0.46    0.48    0.47    0.50    0.54    1.14
           dense/triton     3.53    7.68   15.99   32.61   65.86   132.8   267.7   539.9
             dense/warp     4.91    9.62   19.43   37.96   73.26   150.9   367.9   776.4
             dense/cupy     3.54    7.68   15.99   32.63   65.88   132.7   267.7   540.0
          dense/compile      OOM     OOM   44.24   92.93   225.7   631.4  2009.3  7111.2
          sparse/triton     1.65    2.86    5.22    9.99   21.82   40.95   76.69   148.6
            sparse/warp     6.23   11.92   24.34   45.37   116.5   190.2   355.3   752.7
            sparse/cupy     1.76    3.06    5.57   10.95   22.93   42.86   81.80   157.3
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
BACKENDS = ["triton", "warp", "cupy"]


def _load_backends() -> dict:
    from benchmarks.glif_net.glif_triton import glif3_step_triton
    from benchmarks.glif_net.glif_warp import glif3_step_warp
    from benchmarks.glif_net.glif_cupy import glif3_step_cupy
    return {"triton": glif3_step_triton, "warp": glif3_step_warp, "cupy": glif3_step_cupy}


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
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


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
