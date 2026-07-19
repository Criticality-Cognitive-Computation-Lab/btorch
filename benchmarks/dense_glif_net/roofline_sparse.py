"""Roofline analysis of the fused sparse-SpMV inference kernels.

SpMV has very low arithmetic intensity, so it is memory-bound; the question is
whether the kernels reach HBM bandwidth or lose it to the random ``s``-gather and
the scale-free load imbalance. Two ways to answer, both here:

1. Analytical roofline (default, no privileges): time each backend's sparse
   inference and divide the modeled HBM traffic by the time. The traffic is
   dominated by streaming the CSR arrays ``val`` + ``col`` (8 B / nonzero, which
   at N=2**15 is ~320 MB and exceeds L2); the gathered spike vector ``s``
   (N*4 B) fits L2, and the neuron state is a rounding error. So

       HBM bytes / step ~= 8 * nnz ,   FLOP / step = 2 * nnz ,   AI ~= 0.25 FLOP/B.

2. Measured, via Nsight Compute (needs elevated perf counters). Print the exact
   command with ``--ncu``; run it yourself:

       sudo -E /usr/local/cuda-12.8/bin/ncu --target-processes all \
         --kernel-name regex:sparse --launch-skip 3 --launch-count 1 \
         --metrics dram__bytes.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
         <python> -m benchmarks.dense_glif_net.roofline_sparse workload <backend> 32768

Reproduce::

    python -m benchmarks.dense_glif_net.roofline_sparse            # analytical roofline
    python -m benchmarks.dense_glif_net.roofline_sparse --ncu      # print the ncu command

Results (N=32768, T=32, M=2, 5% scale-free; nnz=40,079,209; AI=0.25 FLOP/B;
RTX 5090, HBM peak 1792 GB/s)::

                  backend |   time |  DRAM BW | % HBM peak | GFLOP/s
      triton (CSR-vector) |  8.42ms | 1219GB/s |       68%  |    305
        cupy (CSR-vector) |  7.86ms | 1305GB/s |       73%  |    326
        warp (CSR-vector) |  8.92ms | 1150GB/s |       64%  |    288
     cuSPARSE (SpMV only) |  8.14ms | 1260GB/s |       70%  |    315

ncu-measured (cuSPARSE csrmv_v3_kernel, isolated): dram throughput 56.6% of
sustained peak, sm throughput 15.6%, 325 us / launch.
"""

from __future__ import annotations

import sys
import time

import torch

from benchmarks.dense_glif_net.glif_common import scale_free_csr

DEVICE = torch.device("cuda")
N_DEFAULT, T, M, DENSITY = 32768, 32, 2, 0.05
# RTX 5090 GDDR7: 512-bit bus x 28 Gbps = 1792 GB/s; ~105 TFLOP/s fp32.
PEAK_BW_GBS = 1792.0
PEAK_FP32_GFLOPS = 105_000.0


def _inputs(N: int, seed: int = 0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W = scale_free_csr(N, DENSITY, DEVICE, seed=seed)
    x = 0.5 + 0.6 * torch.randn(T, N, generator=g, device=DEVICE)
    bias = 0.02 * torch.randn(N, generator=g, device=DEVICE)
    params = dict(
        v_th=torch.full((N,), -50.0, device=DEVICE), v_reset=torch.full((N,), -70.0, device=DEVICE),
        v_rest=torch.full((N,), -70.0, device=DEVICE), c_m=torch.full((N,), 0.05, device=DEVICE),
        tau=torch.full((N,), 20.0, device=DEVICE),
        k=0.1 + 0.2 * torch.rand(N * M, generator=g, device=DEVICE),
        asc_amps=0.05 * torch.randn(N * M, generator=g, device=DEVICE))
    return (W, x, bias, params, torch.full((N,), -65.0, device=DEVICE),
            torch.zeros(N * M, device=DEVICE), torch.ones(N, device=DEVICE))


def _run(step, W, x, bias, params, v, Iasc, nr):
    return step.sparse_multistep_fused(
        x_seq=x, weight=W, bias=bias, v=v.clone(), Iasc=Iasc.clone(),
        params=params, not_refrac=nr, dt=1.0, M=M, hard_reset=False, alpha=2.0)


def _time_ms(step, data, iters=10):
    fn = lambda: _run(step, *data)
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


def _cusparse_spmv_ms(W, N, iters=20):
    """Reference: T cuSPARSE CSR SpMVs (torch.sparse.mm), no neuron update."""
    Wcsr = torch.sparse_csr_tensor(W.crow, W.col, W.val, (N, N))
    s = (torch.rand(N, device=DEVICE) > 0.5).float()
    with torch.no_grad():
        for _ in range(3):
            torch.sparse.mm(Wcsr, s.view(N, 1))
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            for _ in range(T):
                torch.sparse.mm(Wcsr, s.view(N, 1))
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


def roofline(N: int) -> None:
    from benchmarks.dense_glif_net.glif_triton import glif3_step_triton
    from benchmarks.dense_glif_net.glif_warp import glif3_step_warp
    from benchmarks.dense_glif_net.glif_cupy import glif3_step_cupy
    backends = {"triton (CSR-vector)": glif3_step_triton,
                "cupy (CSR-vector)": glif3_step_cupy,
                "warp (CSR-vector)": glif3_step_warp}

    data = _inputs(N)
    nnz = data[0].val.numel()
    hbm_bytes = T * nnz * 8.0          # val + col streamed every step
    flop = T * nnz * 2.0              # one multiply-add per nonzero
    ai = flop / hbm_bytes            # ~0.25 FLOP/B -> deep in the memory-bound regime
    mem_ceiling = ai * PEAK_BW_GBS   # GFLOP/s attainable at this AI (memory roof)

    print(f"Sparse SpMV roofline  |  N={N}  T={T}  nnz={nnz:,} (density {nnz/N/N:.3f})")
    print(f"HBM peak {PEAK_BW_GBS:.0f} GB/s | fp32 peak {PEAK_FP32_GFLOPS/1e3:.0f} TFLOP/s | "
          f"AI {ai:.2f} FLOP/B -> memory ceiling {mem_ceiling:.0f} GFLOP/s\n")
    print(f"{'backend':>20} | {'time':>8} | {'DRAM BW':>10} | {'% HBM peak':>10} | {'GFLOP/s':>8}")
    for name, step in backends.items():
        ms = _time_ms(step, data)
        gbs = hbm_bytes / (ms / 1e3) / 1e9
        gflops = flop / (ms / 1e3) / 1e9
        print(f"{name:>20} | {ms:7.2f}ms | {gbs:7.0f}GB/s | {100*gbs/PEAK_BW_GBS:9.0f}% | {gflops:8.0f}")
    ms = _cusparse_spmv_ms(data[0], N)   # vendor reference, SpMV only (no neuron)
    gbs = hbm_bytes / (ms / 1e3) / 1e9
    print(f"{'cuSPARSE (SpMV only)':>20} | {ms:7.2f}ms | {gbs:7.0f}GB/s | "
          f"{100*gbs/PEAK_BW_GBS:9.0f}% | {flop/(ms/1e3)/1e9:8.0f}")


def workload(backend: str, N: int) -> None:
    data = _inputs(N)
    if backend == "cusparse":
        W = data[0]
        Wcsr = torch.sparse_csr_tensor(W.crow, W.col, W.val, (N, N))
        s = (torch.rand(N, device=DEVICE) > 0.5).float()
        with torch.no_grad():
            for _ in range(T):
                torch.sparse.mm(Wcsr, s.view(N, 1))
    else:
        step = getattr(__import__(f"benchmarks.dense_glif_net.glif_{backend}",
                                  fromlist=["x"]), f"glif3_step_{backend}")
        with torch.no_grad():
            _run(step, *data)
    torch.cuda.synchronize()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "workload":
        workload(sys.argv[2], int(sys.argv[3]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--ncu":
        print("sudo -E /usr/local/cuda-12.8/bin/ncu --target-processes all "
              "--kernel-name regex:sparse --launch-skip 3 --launch-count 1 --metrics "
              "dram__bytes.sum,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,"
              "sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum "
              f"{sys.executable} -m benchmarks.dense_glif_net.roofline_sparse "
              "workload triton 32768")
    else:
        roofline(N_DEFAULT)
