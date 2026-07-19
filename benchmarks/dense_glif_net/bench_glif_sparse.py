"""Four benchmark tables for the GLIF3 kernels:

    {neuron multistep, sparse recurrent multistep} x {inference, training}

with the recurrent connection a 5% scale-free sparse SpMV (``scale_free_csr``).
Rows are the backend, columns the neuron count N; times are ms (forward for
inference, forward+backward for training). Run with::

    python -m benchmarks.dense_glif_net.bench_glif_sparse

Results (T=32, M=2, 5% scale-free, RTX 5090; ms)::

    neuron multistep | inference           neuron multistep | training (fwd+bwd)
     backend | 8192 | 16384 | 32768         backend | 8192 | 16384 | 32768
      triton | 0.07 |  0.06 |  0.06          triton | 0.61 |  0.44 |  0.45
        warp | 0.28 |  0.28 |  0.27            warp | 1.34 |  1.35 |  1.28
        cupy | 0.07 |  0.06 |  0.07            cupy | 0.48 |  0.47 |  0.45

    sparse recurrent | inference          sparse recurrent | training (fwd+bwd)
     backend | 8192 | 16384 | 32768         backend |  8192 | 16384 | 32768
      triton | 0.86 |  1.49 |  8.23          triton | 11.03 | 29.96 | 115.86
        warp | 6.63 |  6.58 |  8.92            warp | 45.54 |102.53 | 252.50
        cupy | 0.47 |  1.30 |  9.02            cupy |  9.99 | 29.91 | 115.86
"""

from __future__ import annotations

import time

import torch

from benchmarks.dense_glif_net.glif_common import scale_free_csr

DEVICE = torch.device("cuda")
T, M, DT, ALPHA, DENSITY = 32, 2, 1.0, 2.0, 0.05
NEURON_COUNTS = (8192, 16384, 32768)


def _load_backends() -> dict:
    from benchmarks.dense_glif_net.glif_triton import glif3_step_triton
    from benchmarks.dense_glif_net.glif_warp import glif3_step_warp
    from benchmarks.dense_glif_net.glif_cupy import glif3_step_cupy
    return {"triton": glif3_step_triton, "warp": glif3_step_warp, "cupy": glif3_step_cupy}


def _inputs(N: int, sparse: bool, require_grad: bool, seed: int):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = 0.5 + 0.6 * torch.randn(T, N, generator=gen, device=DEVICE)
    bias = 0.02 * torch.randn(N, generator=gen, device=DEVICE)
    asc = 0.05 * torch.randn(N * M, generator=gen, device=DEVICE)
    params = dict(
        v_th=torch.full((N,), -50.0, device=DEVICE), v_reset=torch.full((N,), -70.0, device=DEVICE),
        v_rest=torch.full((N,), -70.0, device=DEVICE), c_m=torch.full((N,), 0.05, device=DEVICE),
        tau=torch.full((N,), 20.0, device=DEVICE),
        k=0.1 + 0.2 * torch.rand(N * M, generator=gen, device=DEVICE), asc_amps=asc)
    v = torch.full((N,), -65.0, device=DEVICE)
    Iasc = torch.zeros(N * M, device=DEVICE)
    not_refrac = torch.ones(N, device=DEVICE)
    weight = scale_free_csr(N, DENSITY, DEVICE, seed=seed) if sparse else None
    if require_grad:
        for leaf in (x, bias, asc) + ((weight.val,) if sparse else ()):
            leaf.requires_grad_(True)
    return x, bias, asc, params, v, Iasc, not_refrac, weight


def _run(step, sparse: bool, x, bias, params, v, Iasc, not_refrac, weight):
    common = dict(params=params, not_refrac=not_refrac, dt=DT, M=M, hard_reset=False, alpha=ALPHA)
    if sparse:
        return step.sparse_multistep_fused(
            x_seq=x, weight=weight, bias=bias, v=v.clone(), Iasc=Iasc.clone(), **common)
    return step.multistep_fused(x_seq=x, v=v.clone(), Iasc=Iasc.clone(), **common)


def _time(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


def _measure(step, sparse: bool, train: bool, N: int) -> float:
    x, bias, asc, params, v, Iasc, not_refrac, weight = _inputs(N, sparse, train, seed=N)
    leaves = [x, bias, asc] + ([weight.val] if sparse else [])

    def forward():
        return _run(step, sparse, x, bias, params, v, Iasc, not_refrac, weight)

    def train_step():
        for leaf in leaves:
            leaf.grad = None
        out = forward()
        sum(o.sum() for o in out).backward()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        return _time(train_step if train else lambda: forward(), iters=5 if train else 10)


def _table(backends: dict, sparse: bool, train: bool) -> None:
    kind = "sparse recurrent" if sparse else "neuron"
    mode = "training (fwd+bwd)" if train else "inference"
    print(f"\n{kind} multistep | {mode} (ms)")
    print(f"{'backend':>8} | " + " | ".join(f"N={N:>6}" for N in NEURON_COUNTS))
    for name, step in backends.items():
        cells = []
        for N in NEURON_COUNTS:
            try:
                cells.append(f"{_measure(step, sparse, train, N):8.2f}")
            except RuntimeError as exc:
                cells.append(" OOM/err" if "memory" in str(exc) else "     err")
            torch.cuda.empty_cache()
        print(f"{name:>8} | " + " | ".join(f"{c:>8}" for c in cells), flush=True)


def main() -> None:
    print(f"T={T} M={M}  sparse density={DENSITY} (scale-free)")
    backends = _load_backends()
    for sparse in (False, True):
        for train in (False, True):
            _table(backends, sparse, train)


if __name__ == "__main__":
    main()
