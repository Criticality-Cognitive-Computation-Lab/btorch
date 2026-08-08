"""PDL vs CUDA-graph vs persistent kernel for a sparse recurrent (RSNN) multistep.

A GLIF3 spiking network with a 5% scale-free recurrent connection is a *sequential*
T-step recurrence: step ``t`` needs all spikes of ``t-1`` through ``x_in = x[t] +
bias + W @ s[t-1]``. The natural GPU realization is **one kernel launch per step**
(the launch boundary is the cross-step barrier). This script measures four ways to
run that T-step chain, in each of **Triton**, **CuPy**, and **TileLang**:

- ``plain``      — T ordinary per-step launches (baseline).
- ``pdl``        — T per-step launches with **Programmatic Dependent Launch**:
                   step ``t+1`` begins (streaming the step-invariant CSR weight)
                   while step ``t`` drains; ``griddepcontrol.wait`` gates only the
                   small spike dependency and ``launch_dependents`` releases the next
                   grid. Triton: ``launch_pdl=True`` + ``tl...cuda.gdc_{wait,launch_
                   dependents}``. TileLang: ``pdl_sync``/``pdl_trigger`` (host attr
                   auto-set). CuPy: inline PTX ``griddepcontrol`` + ``cuLaunchKernelEx``
                   with ``PROGRAMMATIC_STREAM_SERIALIZATION`` via ``cuda-python``.
- ``cudagraph``  — the T plain launches captured once into a ``CUDAGraph`` and
                   replayed (removes the host-side per-launch cost).
- ``persistent`` — a single cooperative launch running the whole T loop on-device
                   with ``grid.sync`` between steps (the fused sparse persistent
                   kernels from ``glif_net``: CuPy ``cg::this_grid().sync``, TileLang
                   ``T.sync_grid``). No relaunch and no per-step state round-trip.

PDL only overlaps the *GPU-side* inter-kernel gap (opportunistically); CUDA graphs
remove the *host-side* launch cost; a persistent kernel removes relaunch entirely
but caps the grid at SM co-residency. Results (best-of-windows ms, contention
rejected) are written to JSON.

Run (heavy — use SLURM ``pdl.sbatch``)::

    # sweep launch-bound -> GPU-bound (default), or a single size
    python -m benchmarks.flexsn_vs_compile.bench_pdl --N 512 2048 8192 24576 --T 64
    python -m benchmarks.flexsn_vs_compile.bench_pdl --N 24576 --T 64

Results (RTX 5090, T=64, speedup vs each DSL's per-step ``plain`` launch) — a
launch-bound -> bandwidth-bound crossover at ~N=8192. Full table + analysis in
``README.md``::

    N       DSL        plain    PDL   cudagraph  persistent
    512     tilelang   1.27ms   1.01x   9.71x      8.77x
    512     triton     1.44ms   0.99x  11.10x       --
    8192    tilelang   1.28ms   1.02x   1.60x      1.21x
    24576   tilelang   9.20ms   1.01x   1.01x      0.80x   <- bandwidth-bound
    24576   cupy       8.35ms   0.98x   1.01x      0.89x

CUDA graphs are the only broad win, and only while launches dominate (9-11x at
N=512 -> ~1.0x at N=24576). Persistent is strong small (8.8x) but *inverts* at
scale (0.80x, grid-occupancy starvation). PDL is ~1.0x throughout: it hides
launch latency, not the launch cost graphs remove, and the bandwidth-bound step
leaves nothing to overlap. (CuPy PDL 0.62x is the heavier cuLaunchKernelEx host
path, not PDL itself.)
"""

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from benchmarks.glif_net.glif_common import scale_free_csr

DT, ALPHA, M, DENSITY, HARD_RESET = 1.0, 2.0, 2, 0.05, False


# ---------------------------------------------------------------------------
# Shared inputs + a torch reference (numerical gate for every backend/strategy)
# ---------------------------------------------------------------------------
def make_inputs(N, T, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = scale_free_csr(N, DENSITY, "cuda", seed=seed)
    x = 0.5 + 0.6 * torch.randn(T, N, generator=g, device="cuda")
    bias = 0.02 * torch.randn(N, generator=g, device="cuda")
    p = dict(
        v_th=torch.full((N,), -50.0, device="cuda"), v_reset=torch.full((N,), -70.0, device="cuda"),
        v_rest=torch.full((N,), -70.0, device="cuda"), c_m=torch.full((N,), 0.05, device="cuda"),
        tau=torch.full((N,), 20.0, device="cuda"),
        k=0.1 + 0.2 * torch.rand(N * M, generator=g, device="cuda"),
        asc_amps=0.05 * torch.randn(N * M, generator=g, device="cuda"))
    v0 = torch.full((N,), -65.0, device="cuda")
    I0 = torch.zeros(N * M, device="cuda")
    nr = torch.ones(N, device="cuda")
    return W, x, bias, p, v0, I0, nr


def reference_spikes(W, x, bias, p, v0, I0, nr, T, N):
    """Eager GLIF3 sparse recurrent multistep — returns the final spike vector."""
    Wcsr = torch.sparse_csr_tensor(W.crow, W.col, W.val, (N, N))
    v = v0.clone(); I = I0.clone().view(N, M); s = torch.zeros(N, device="cuda")
    kk = p["k"].view(N, M); aa = p["asc_amps"].view(N, M)
    with torch.no_grad():
        for t in range(T):
            lin = torch.sparse.mm(Wcsr, s.view(N, 1)).view(N)
            x_in = x[t] + bias + lin
            v_inf = p["v_rest"] + p["tau"] * (x_in + I.sum(1)) / p["c_m"]
            vp = v_inf + (v - v_inf) * torch.exp(-DT / p["tau"])
            s = (vp >= p["v_th"]).float() * nr
            v = vp - (p["v_th"] - p["v_reset"]) * s
            I = I * torch.exp(-kk * DT) + aa * s.unsqueeze(1)
    return s


def bench(fn, warm=10, reps=30, windows=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(windows):
        t0 = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.time() - t0) / reps)
    return best * 1e3


# ===========================================================================
# TileLang: per-step (plain / PDL) + cudagraph + persistent (reused)
# ===========================================================================
def tilelang_runners(W, x, bias, p, v0, I0, nr, T, N):
    import functools

    import tilelang
    import tilelang.language as T_
    from tilelang.cuda.language.pdl import pdl_sync, pdl_trigger

    nnz = int(W.col.numel())

    @functools.lru_cache(maxsize=None)
    def step(pdl, rows=8, lanes=32):
        @T_.prim_func
        def main(crow: T_.Tensor((N + 1,), "int32"), col: T_.Tensor((nnz,), "int32"),
                 val: T_.Tensor((nnz,), "float32"), s_prev: T_.Tensor((N,), "float32"),
                 x_t: T_.Tensor((N,), "float32"), bias_: T_.Tensor((N,), "float32"),
                 v: T_.Tensor((N,), "float32"), Iasc: T_.Tensor((N * M,), "float32"),
                 v_th: T_.Tensor((N,), "float32"), v_reset: T_.Tensor((N,), "float32"),
                 v_rest: T_.Tensor((N,), "float32"), c_m: T_.Tensor((N,), "float32"),
                 tau: T_.Tensor((N,), "float32"), k: T_.Tensor((N * M,), "float32"),
                 asc: T_.Tensor((N * M,), "float32"), nr_: T_.Tensor((N,), "float32"),
                 dt: T_.float32, s_out: T_.Tensor((N,), "float32"),
                 v_out: T_.Tensor((N,), "float32"), I_out: T_.Tensor((N * M,), "float32")):
            with T_.Kernel(T_.ceildiv(N, rows), threads=rows * lanes) as bx:
                if pdl:
                    pdl_sync()
                acc = T_.alloc_fragment((rows, lanes), "float32")
                T_.clear(acc)
                for r, lane in T_.Parallel(rows, lanes):
                    row = bx * rows + r
                    if row < N:
                        for pp in T_.serial(crow[row] + lane, crow[row + 1], lanes):
                            acc[r, lane] += val[pp] * s_prev[col[pp]]
                lin = T_.alloc_fragment((rows,), "float32")
                T_.reduce_sum(acc, lin, dim=1)
                for r in T_.Parallel(rows):
                    row = bx * rows + r
                    if row < N:
                        isum = T_.alloc_local((1,), "float32"); isum[0] = T_.float32(0)
                        for m in range(M):
                            isum[0] += Iasc[row * M + m]
                        x_in = x_t[row] + bias_[row] + lin[r]
                        v_inf = v_rest[row] + tau[row] * (x_in + isum[0]) / c_m[row]
                        vp = v_inf + (v[row] - v_inf) * T_.exp(-dt / tau[row])
                        sp = T_.alloc_local((1,), "float32"); sp[0] = T_.float32(0)
                        if vp >= v_th[row]:
                            sp[0] = nr_[row]
                        v_out[row] = vp - (v_th[row] - v_reset[row]) * sp[0]
                        s_out[row] = sp[0]
                        for m in range(M):
                            j = row * M + m
                            I_out[j] = Iasc[j] * T_.exp(-k[j] * dt) + asc[j] * sp[0]
                if pdl:
                    pdl_trigger()
        pc = {tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True}
        return tilelang.compile(main, out_idx=[17, 18, 19], pass_configs=pc)

    crow, colc, val = W.crow.int().contiguous(), W.col.int().contiguous(), W.val.float().contiguous()
    fp = [t.float().contiguous() for t in (p["v_th"], p["v_reset"], p["v_rest"], p["c_m"],
                                           p["tau"], p["k"], p["asc_amps"])]

    def make_step_runner(pdl):
        kern = step(pdl)

        def run():
            s = torch.zeros(N, device="cuda"); v = v0.clone(); I = I0.clone()
            for t in range(T):
                s, v, I = kern(crow, colc, val, s, x[t], bias, v, I, *fp[:5], fp[5], fp[6], nr, DT)
            return s
        return run

    def run_cudagraph():
        kern = step(False)
        s0 = torch.zeros(N, device="cuda"); v0c = v0.clone(); I0c = I0.clone()
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            s, v, I = s0.clone(), v0c.clone(), I0c.clone()
            for t in range(T):
                s, v, I = kern(crow, colc, val, s, x[t], bias, v, I, *fp[:5], fp[5], fp[6], nr, DT)
        torch.cuda.current_stream().wait_stream(side)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            s, v, I = s0, v0c, I0c
            for t in range(T):
                s, v, I = kern(crow, colc, val, s, x[t], bias, v, I, *fp[:5], fp[5], fp[6], nr, DT)

        def replay():
            gr.replay()
        # keep every capture tensor alive — else torch.cuda.empty_cache() between
        # strategies reclaims the graph pool's blocks and replay illegal-accesses.
        replay._keep = (gr, s0, v0c, I0c, s, v, I)
        return replay

    from benchmarks.glif_net.glif_tilelang import _persistent_grid, _sparse_persistent

    def run_persistent():
        kern = _sparse_persistent(T, N, M, nnz, HARD_RESET, _persistent_grid())
        s_seq, _, _, _ = kern(crow, colc, val, x, bias, v0.clone(), I0.clone(),
                              *fp[:5], fp[5], fp[6], nr, DT)
        return s_seq[-1]

    return {"plain": make_step_runner(False), "pdl": make_step_runner(True),
            "cudagraph": run_cudagraph(), "persistent": run_persistent}


# ===========================================================================
# Triton: per-step (plain / PDL via launch_pdl) + cudagraph
# ===========================================================================
def triton_runners(W, x, bias, p, v, I0, nr, T, N):
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    @triton.jit
    def sstep(crow, col, val, s_prev, x_t, bias, v_ptr, I_ptr,
              v_th, v_reset, v_rest, c_m, tau, k, asc, nr_, s_out, v_out, I_out,
              N, dt, M: tl.constexpr, BK: tl.constexpr, USE_PDL: tl.constexpr):
        row = tl.program_id(0)
        # crow is step-invariant — the row's nnz range can be read before the
        # PDL wait. s_prev/v/I (written by the previous step) are read after it.
        start = tl.load(crow + row); end = tl.load(crow + row + 1)
        if USE_PDL:
            gdc_wait()
        # CSR-vector reduction: BK-wide masked vector loads, reduced with tl.sum
        # (same schedule as glif_net's glif3_sparse_step_kernel — hub rows with
        # 10-100x the mean nnz don't serialize into scalar dependent loads).
        acc = tl.zeros((BK,), dtype=tl.float32)
        for p0 in range(start, end, BK):
            offs = p0 + tl.arange(0, BK)
            m = offs < end
            cols = tl.load(col + offs, mask=m, other=0)
            vals = tl.load(val + offs, mask=m, other=0.0)
            sv = tl.load(s_prev + cols, mask=m, other=0.0)
            acc += vals * sv
        lin = tl.sum(acc)
        isum = 0.0
        for mm in range(M):
            isum += tl.load(I_ptr + row * M + mm)
        tauv = tl.load(tau + row)
        x_in = tl.load(x_t + row) + tl.load(bias + row) + lin
        v_inf = tl.load(v_rest + row) + tauv * (x_in + isum) / tl.load(c_m + row)
        vp = v_inf + (tl.load(v_ptr + row) - v_inf) * tl.exp(-dt / tauv)
        vth = tl.load(v_th + row)
        sp = tl.where(vp >= vth, tl.load(nr_ + row), 0.0)
        tl.store(v_out + row, vp - (vth - tl.load(v_reset + row)) * sp)
        tl.store(s_out + row, sp)
        for mm in range(M):
            j = row * M + mm
            tl.store(I_out + j, tl.load(I_ptr + j) * tl.exp(-tl.load(k + j) * dt)
                     + tl.load(asc + j) * sp)
        if USE_PDL:
            gdc_launch_dependents()

    crow, colc, val = W.crow.int().contiguous(), W.col.int().contiguous(), W.val.float().contiguous()
    args_static = (p["v_th"], p["v_reset"], p["v_rest"], p["c_m"], p["tau"], p["k"], p["asc_amps"])

    def _loop(s, vv, I, so, vo, Io, use_pdl):
        for t in range(T):
            sstep[(N,)](crow, colc, val, s, x[t], bias, vv, I, *args_static, nr, so, vo, Io,
                        N, DT, M=M, BK=128, USE_PDL=use_pdl, launch_pdl=use_pdl, num_warps=1)
            s, so = so, s; vv, vo = vo, vv; I, Io = Io, I
        return s

    def make_run(use_pdl):
        def run():
            s = torch.zeros(N, device="cuda"); vv = v.clone(); I = I0.clone()
            so = torch.empty_like(s); vo = torch.empty_like(vv); Io = torch.empty_like(I)
            return _loop(s, vv, I, so, vo, Io, use_pdl)
        return run

    def run_cudagraph():
        s = torch.zeros(N, device="cuda"); vv = v.clone(); I = I0.clone()
        so = torch.empty_like(s); vo = torch.empty_like(vv); Io = torch.empty_like(I)
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _loop(s.clone(), vv.clone(), I.clone(), so.clone(), vo.clone(), Io.clone(), False)
        torch.cuda.current_stream().wait_stream(side)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            _loop(s, vv, I, so, vo, Io, False)

        def replay():
            gr.replay()
        replay._keep = (gr, s, vv, I, so, vo, Io)  # see tilelang note
        return replay

    return {"plain": make_run(False), "pdl": make_run(True), "cudagraph": run_cudagraph()}


# ===========================================================================
# CuPy: per-step (plain / PDL via cuLaunchKernelEx) + persistent (reused)
# ===========================================================================
_CUPY_SRC = r'''
extern "C" __global__ void sparse_step(
    const int* crow, const int* col, const float* val, const float* s_prev,
    const float* x_t, const float* bias, const float* v, const float* Iasc,
    const float* v_th, const float* v_reset, const float* v_rest, const float* c_m,
    const float* tau, const float* k, const float* asc, const float* nr,
    float* s_out, float* v_out, float* I_out, int N, int M, float dt, int USE_PDL) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int row = warp;
    int start = (warp < N) ? crow[row] : 0;
    int end   = (warp < N) ? crow[row + 1] : 0;
    // PDL input prefetch: warm this row's step-invariant CSR stream (val/col, the
    // 171 MB/step bulk) into L2 in the shadow of the previous step, *before*
    // waiting on the spike dependency. s_prev[col[p]] is an indirect gather and
    // cannot be prefetched — only the direct val/col stream. crow/val/col do not
    // depend on the previous step, so this is safe to issue pre-wait.
    if (USE_PDL) {
        for (int p = start + lane; p < end; p += 32) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(val + p));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(col + p));
        }
        asm volatile("griddepcontrol.wait;");
    }
    if (warp < N) {
        float acc = 0.0f;
        for (int p = start + lane; p < end; p += 32) acc += val[p] * s_prev[col[p]];
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) {
            float isum = 0.0f;
            for (int m = 0; m < M; ++m) isum += Iasc[row*M+m];
            float x_in = x_t[row] + bias[row] + acc;
            float v_inf = v_rest[row] + tau[row]*(x_in+isum)/c_m[row];
            float vp = v_inf + (v[row]-v_inf)*expf(-dt/tau[row]);
            float sp = (vp >= v_th[row]) ? nr[row] : 0.0f;
            v_out[row] = vp - (v_th[row]-v_reset[row])*sp;
            s_out[row] = sp;
            for (int m = 0; m < M; ++m)
                I_out[row*M+m] = Iasc[row*M+m]*expf(-k[row*M+m]*dt) + asc[row*M+m]*sp;
        }
    }
    if (USE_PDL) asm volatile("griddepcontrol.launch_dependents;");
}
'''


def cupy_runners(W, x, bias, p, v0, I0, nr, T, N):
    import cupy as cp
    from cuda.bindings import driver as drv

    mod = cp.RawModule(code=_CUPY_SRC, options=("-arch=sm_90",))
    kern = mod.get_function("sparse_step")
    THREADS = 256
    grid = (N * 32 + THREADS - 1) // THREADS

    def _cp(t):  # torch cuda -> cupy view (zero copy)
        return cp.asarray(t)

    crow, colc, val = _cp(W.crow.int()), _cp(W.col.int()), _cp(W.val.float())
    xc = _cp(x.contiguous()); biasc = _cp(bias); nrc = _cp(nr)
    pv = {kk: _cp(p[kk].contiguous()) for kk in p}
    v0c, I0c = _cp(v0), _cp(I0)
    # Launch on *torch's* current stream (not cupy's default) so torch.cuda.graph
    # can capture the RawKernel launches (glif_cupy._current_stream does the same).
    torch_stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    stream_ptr = torch_stream.ptr

    def ptr(a):
        return int(a.data.ptr)

    def _args(s, xt, v, I, so, vo, Io):
        vals = (ptr(crow), ptr(colc), ptr(val), ptr(s), ptr(xt), ptr(biasc), ptr(v), ptr(I),
                ptr(pv["v_th"]), ptr(pv["v_reset"]), ptr(pv["v_rest"]), ptr(pv["c_m"]),
                ptr(pv["tau"]), ptr(pv["k"]), ptr(pv["asc_amps"]), ptr(nrc),
                ptr(so), ptr(vo), ptr(Io), N, M, DT, 1)
        types = (ctypes.c_void_p,) * 19 + (ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int)
        return (vals, types)

    def launch_pdl(s, xt, v, I, so, vo, Io):
        cfg = drv.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = THREADS, 1, 1
        cfg.sharedMemBytes = 0
        cfg.hStream = drv.CUstream(stream_ptr)
        attr = drv.CUlaunchAttribute()
        attr.id = drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        attr.value.programmaticStreamSerializationAllowed = 1
        cfg.attrs = [attr]; cfg.numAttrs = 1
        err = drv.cuLaunchKernelEx(cfg, drv.CUfunction(kern.kernel.ptr),
                                   _args(s, xt, v, I, so, vo, Io), 0)
        code = err[0] if isinstance(err, tuple) else err
        if code != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernelEx failed: {code}")

    def launch_plain(s, xt, v, I, so, vo, Io):
        kern((grid,), (THREADS,), (crow, colc, val, s, xt, biasc, v, I,
            pv["v_th"], pv["v_reset"], pv["v_rest"], pv["c_m"], pv["tau"], pv["k"],
            pv["asc_amps"], nrc, so, vo, Io, cp.int32(N), cp.int32(M),
            cp.float32(DT), cp.int32(0)))

    def _loop(s, v, I, so, vo, Io, use_pdl):
        for t in range(T):
            (launch_pdl if use_pdl else launch_plain)(s, xc[t], v, I, so, vo, Io)
            s, so = so, s; v, vo = vo, v; I, Io = Io, I
        return s

    def make_run(use_pdl):
        def run():
            with torch_stream:
                s = cp.zeros(N, cp.float32); v = v0c.copy(); I = I0c.copy()
                so = cp.empty_like(s); vo = cp.empty_like(v); Io = cp.empty_like(I)
                s = _loop(s, v, I, so, vo, Io, use_pdl)
            return torch.as_tensor(s, device="cuda")
        return run

    def run_cudagraph():
        s = cp.zeros(N, cp.float32); v = v0c.copy(); I = I0c.copy()
        so = cp.empty_like(s); vo = cp.empty_like(v); Io = cp.empty_like(I)
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), cp.cuda.ExternalStream(side.cuda_stream):
            _loop(s.copy(), v.copy(), I.copy(), so.copy(), vo.copy(), Io.copy(), False)
        torch.cuda.current_stream().wait_stream(side)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            # inside capture, torch's current stream is the capture stream — bind
            # cupy's launches to it so they are recorded into the graph.
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                _loop(s, v, I, so, vo, Io, False)

        def replay():
            gr.replay()
        replay._keep = (gr, s, v, I, so, vo, Io)  # see tilelang note
        return replay

    from benchmarks.glif_net.glif_cupy import glif3_sparse_multistep_fused_cupy

    def run_persistent():
        with torch.no_grad():
            s_seq, _, _, _ = glif3_sparse_multistep_fused_cupy(
                x, W, bias, v0.clone(), I0.clone().view(N, M), p, nr, DT, M,
                hard_reset=HARD_RESET, alpha=ALPHA)
        return s_seq[-1]

    # cudagraph last: a CUDA fault there can't be caught cleanly (context poison),
    # so run it after the cells whose results we want to keep.
    return {"plain": make_run(False), "pdl": make_run(True),
            "persistent": run_persistent, "cudagraph": run_cudagraph()}


# ===========================================================================
BUILDERS = {"triton": triton_runners, "cupy": cupy_runners, "tilelang": tilelang_runners}


def run_one_dsl(dsl, N, T, W, x, bias, p, v0, I0, nr, ref):
    """Build + bench a single DSL's strategies. Isolated so a CUDA fault in one
    backend cannot poison the others (the orchestrator runs each in a subprocess)."""
    out = {}
    try:
        runners = BUILDERS[dsl](W, x, bias, p, v0, I0, nr, T, N)
    except Exception as e:  # noqa: BLE001
        print(f"[{dsl}] build failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return out
    for strat, fn in runners.items():
        try:
            o = fn()
            if o is None:  # cudagraph replay returns nothing to validate
                err, nmis = float("nan"), -1
            else:
                o = torch.as_tensor(o, device="cuda").float()
                err = (o - ref).abs().max().item()
                # spikes are 0/1; float summation order can flip a borderline
                # crossing, so report the *count* of differing spikes, not max.
                nmis = int((o != ref).sum().item())
            ms = bench(fn)
            out[f"{dsl}/{strat}"] = {"ms": ms, "spike_err": err, "n_mismatch": nmis}
            print(f"{dsl:9}/{strat:11}: {ms:8.3f} ms   spike_err={err:.1e}  mismatch={nmis}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            out[f"{dsl}/{strat}"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"{dsl:9}/{strat:11}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
        torch.cuda.empty_cache()
    return out


def _print_speedups(res):
    for dsl in BUILDERS:
        base = res.get(f"{dsl}/plain", {}).get("ms")
        if base is None:
            continue
        row = [f"{dsl}: plain {base:.2f}ms"]
        for strat in ("pdl", "cudagraph", "persistent"):
            r = res.get(f"{dsl}/{strat}", {})
            if "ms" in r:
                row.append(f"{strat} {base / r['ms']:.2f}x")
        print("  " + "  ".join(row), flush=True)


def run(N_list, T, out_path):
    """Orchestrator: for each N, run each DSL in its own subprocess. A CUDA
    illegal-access in any single backend poisons the whole context irrecoverably,
    so isolate them (a fault loses only its DSL, not the sweep)."""
    sweeps = []
    for N in N_list:
        print(f"\n########## N={N} T={T} ##########", flush=True)
        merged = {}
        for dsl in BUILDERS:
            part = f"{out_path}.{dsl}.part"
            Path(part).unlink(missing_ok=True)
            cmd = [sys.executable, "-m", "benchmarks.flexsn_vs_compile.bench_pdl",
                   "--N", str(N), "--T", str(T), "--only", dsl, "--out", part]
            print(f"\n--- subprocess: {dsl} (N={N}) ---", flush=True)
            subprocess.run(cmd, check=False)
            if Path(part).exists():
                merged.update(json.loads(Path(part).read_text()))
                Path(part).unlink(missing_ok=True)
        sweeps.append({"N": N, "results": merged})
        print(f"\n=== N={N}: speedup vs that DSL's plain per-step launch ===", flush=True)
        _print_speedups(merged)

    out = {"meta": {"T": T, "M": M, "density": DENSITY, "N_list": list(N_list)}, "sweeps": sweeps}
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)


def run_only(dsl, N, T, out_path):
    """Worker: build + bench one DSL, write its partial results (subprocess entry)."""
    W, x, bias, p, v0, I0, nr = make_inputs(N, T)
    ref = reference_spikes(W, x, bias, p, v0, I0, nr, T, N)
    out = run_one_dsl(dsl, N, T, W, x, bias, p, v0, I0, nr, ref)
    Path(out_path).write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[512, 2048, 8192, 24576],
                    help="neuron count(s); a list sweeps the launch-bound -> GPU-bound crossover")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--only", choices=list(BUILDERS), default=None,
                    help="internal: run a single DSL at a single N in this process (subprocess worker)")
    ap.add_argument("--out", default="benchmarks/flexsn_vs_compile/bench_pdl_results.json")
    args = ap.parse_args()
    if args.only:
        run_only(args.only, args.N[0], args.T, args.out)
    else:
        run(args.N, args.T, args.out)


if __name__ == "__main__":
    main()
