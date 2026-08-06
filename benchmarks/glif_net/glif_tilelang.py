"""TileLang GLIF3 kernels: single-step, neuron / dense / sparse multistep.

TileLang (``import tilelang.language as T``) lets a single **persistent** kernel
keep the neuron state resident across timesteps and fuse the recurrent
matrix-vector product with the neuron update — the sparse recurrent inference
path is a persistent kernel that fuses the CSR SpMV with the GLIF3 update
(``glif3_sparse_persistent``), which stock Triton/Warp/CuPy cannot express (no
grid-resident state carried across a fused per-step barrier).

GLIF3 update per step (exact exp-only analytic form, so only ``T.exp`` is needed
— no ``expm1``; ``T.exp`` is the precise intrinsic, not an ``ex2.approx``):

    v_inf   = v_rest + tau * (x + sum I_asc) / c_m
    v'      = v_inf + (v - v_inf) * exp(-dt / tau)
    spike   = heaviside(v' - v_th) * not_refrac
    v_post  = v' - (v' - v_reset) * spike     (hard reset)
            = v' - (v_th - v_reset) * spike   (soft reset)
    I_asc'  = I_asc * exp(-k dt) + asc_amps * spike

Design:
- inference is a lean forward-only kernel (no saved intermediates, no autograd);
- training composes the single-step autograd op (TileLang forward + backward
  kernels wrapped in ``torch.autograd.Function``) with the shared
  ``*_multistep_autograd`` helpers, so ``weight`` / ``bias`` / ``x`` / initial
  state / ``asc_amps`` gradients all fall out of the single-step backward.
  Pure inference therefore never pays for any training bookkeeping.

Conventions (flattened, contiguous, fp32 on CUDA): ``v`` / ``x`` / ``not_refrac``
are ``(B,)``; ``Iasc`` / ``k`` / ``asc_amps`` are ``(B*M,)`` with base ``i*M``.

Performance (measured on an *unloaded* GPU with adequate warmup — see the README;
a busy GPU or a 2-warmup/10-iter loop badly mis-reports these µs-scale kernels):
- neuron inference, neuron training, and dense inference are the **fastest** of
  all backends, and dense training ties. The neuron multistep loops T inside one
  kernel (a runtime ``T.serial`` loop, not unrolled); training is a single
  **fused BPTT** kernel that walks T in reverse carrying the adjoints in
  registers — one launch instead of T composed single-step backwards.
- the persistent sparse kernel's raw CSR-vector SpMV matches cuSPARSE; the fused
  kernel is competitive at small N and ~2x behind CuPy at large N. The grid is
  sized the sanctioned way (one block/SM, ``driver.get_num_sms()``, as in every
  persistent example / ``T.PersistentTileScheduler`` default). TileLang has no
  occupancy API and doesn't surface a CUfunction handle, so it can't size the
  cooperative grid via ``occupancyMaxActiveBlocksPerMultiprocessor(func)*numSMs``
  the way CuPy does; parallelism comes from a bigger block (one 1024-thread CTA/SM,
  32 rows x one warp, ``_SP_ROWS``/``_SP_LANES``). The residual gap is **not
  isolated**, and two hypotheses were tested + ruled out: not occupancy (ncu: 67%
  here vs CuPy's lower ~17%, both latency-bound), and not the reduction (rewriting
  ``T.reduce_sum`` as ``T.warp_reduce_sum`` — the ``__shfl_down_sync`` idiom CuPy
  uses — is bit-exact and the *same* time). The raw SpMV matches cuSPARSE, so the
  loss is in the fused persistent structure (per-step ``sync_grid``, grid-stride,
  the lane-0 neuron update, or gather codegen) — the primitives to match CuPy
  exist, but the straightforward levers don't close it.
The dense path still uses cuBLAS ``addmv`` + a neuron kernel per step (not an
in-kernel gemv); its training composes the single-step op through the shared
autograd helper, hidden behind the matmul backward.
"""

import functools

import torch
import tilelang
import tilelang.language as T

from benchmarks.glif_net.glif_common import (
    GLIF3StepOps,
    SparseWeight,
    dense_multistep_autograd,
    sparse_multistep_autograd,
)

_BLOCK = 256


def _as_fp32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float32 and t.is_contiguous() else t.float().contiguous()


# ----------------------------------------------------------------------------
# Single-step kernels (forward + backward), one thread per neuron.
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _step_forward(B: int, M: int, hard_reset: bool, block: int = _BLOCK):
    N = B * M
    HR = 1 if hard_reset else 0

    @T.prim_func
    def main(
        v: T.Tensor((B,), "float32"), Iasc: T.Tensor((N,), "float32"),
        x: T.Tensor((B,), "float32"),
        v_th: T.Tensor((B,), "float32"), v_reset: T.Tensor((B,), "float32"),
        v_rest: T.Tensor((B,), "float32"), c_m: T.Tensor((B,), "float32"),
        tau: T.Tensor((B,), "float32"), k: T.Tensor((N,), "float32"),
        asc_amps: T.Tensor((N,), "float32"), nr: T.Tensor((B,), "float32"),
        dt: T.float32,
        v_out: T.Tensor((B,), "float32"), I_out: T.Tensor((N,), "float32"),
        s_out: T.Tensor((B,), "float32"),
    ):
        with T.Kernel(T.ceildiv(B, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < B:
                    isum = T.alloc_local((1,), "float32")
                    isum[0] = T.float32(0)
                    for m in range(M):
                        isum[0] += Iasc[idx * M + m]
                    v_inf = v_rest[idx] + tau[idx] * (x[idx] + isum[0]) / c_m[idx]
                    vp = v_inf + (v[idx] - v_inf) * T.exp(-dt / tau[idx])
                    sp = T.alloc_local((1,), "float32")
                    sp[0] = T.float32(0)
                    if vp >= v_th[idx]:
                        sp[0] = nr[idx]
                    if HR == 1:
                        v_out[idx] = vp - (vp - v_reset[idx]) * sp[0]
                    else:
                        v_out[idx] = vp - (v_th[idx] - v_reset[idx]) * sp[0]
                    s_out[idx] = sp[0]
                    for m in range(M):
                        j = idx * M + m
                        I_out[j] = Iasc[j] * T.exp(-k[j] * dt) + asc_amps[j] * sp[0]

    return tilelang.compile(main, out_idx=[12, 13, 14])


@functools.lru_cache(maxsize=None)
def _step_backward(B: int, M: int, hard_reset: bool, block: int = _BLOCK):
    N = B * M
    HR = 1 if hard_reset else 0

    @T.prim_func
    def main(
        v: T.Tensor((B,), "float32"), Iasc: T.Tensor((N,), "float32"),
        x: T.Tensor((B,), "float32"),
        v_th: T.Tensor((B,), "float32"), v_reset: T.Tensor((B,), "float32"),
        v_rest: T.Tensor((B,), "float32"), c_m: T.Tensor((B,), "float32"),
        tau: T.Tensor((B,), "float32"), k: T.Tensor((N,), "float32"),
        asc_amps: T.Tensor((N,), "float32"), nr: T.Tensor((B,), "float32"),
        spike: T.Tensor((B,), "float32"), dv_out: T.Tensor((B,), "float32"),
        dI_out: T.Tensor((N,), "float32"), ds_out: T.Tensor((B,), "float32"),
        dt: T.float32, alpha: T.float32,
        dv: T.Tensor((B,), "float32"), dI: T.Tensor((N,), "float32"),
        dx: T.Tensor((B,), "float32"), dasc: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(B, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < B:
                    isum = T.alloc_local((1,), "float32")
                    isum[0] = T.float32(0)
                    for m in range(M):
                        isum[0] += Iasc[idx * M + m]
                    decay = T.exp(-dt / tau[idx])
                    v_inf = v_rest[idx] + tau[idx] * (x[idx] + isum[0]) / c_m[idx]
                    vp = v_inf + (v[idx] - v_inf) * decay
                    denom = v_th[idx] - v_reset[idx]
                    u = (vp - v_th[idx]) / denom
                    ds_dvp = nr[idx] / (1.0 + (alpha * u) * (alpha * u)) / denom
                    dvprime = T.alloc_local((1,), "float32")
                    ds_from_v = T.alloc_local((1,), "float32")
                    if HR == 1:
                        dvprime[0] = dv_out[idx] * (1.0 - spike[idx])
                        ds_from_v[0] = dv_out[idx] * (-(vp - v_reset[idx]))
                    else:
                        dvprime[0] = dv_out[idx]
                        ds_from_v[0] = dv_out[idx] * (-(v_th[idx] - v_reset[idx]))
                    dIs = T.alloc_local((1,), "float32")
                    dIs[0] = T.float32(0)
                    for m in range(M):
                        dIs[0] += dI_out[idx * M + m] * asc_amps[idx * M + m]
                    ds_total = ds_out[idx] + ds_from_v[0] + dIs[0]
                    dvprime[0] = dvprime[0] + ds_total * ds_dvp
                    dv_inf = dvprime[0] * (1.0 - decay)
                    dI_common = dv_inf * (tau[idx] / c_m[idx])
                    dv[idx] = dvprime[0] * decay
                    dx[idx] = dI_common
                    for m in range(M):
                        j = idx * M + m
                        dI[j] = dI_out[j] * T.exp(-k[j] * dt) + dI_common
                        dasc[j] = dI_out[j] * spike[idx]

    return tilelang.compile(main, out_idx=[17, 18, 19, 20])


class _GLIF3StepTileLang(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, Iasc, x, v_th, v_reset, v_rest, c_m, tau, k, asc_amps,
                not_refrac, dt, M, hard_reset, alpha):
        B = v.numel()
        args = [_as_fp32(t) for t in (v, Iasc, x, v_th, v_reset, v_rest, c_m, tau,
                                      k, asc_amps, not_refrac)]
        v_out, I_out, s_out = _step_forward(B, M, hard_reset)(*args, float(dt))
        ctx.save_for_backward(*args, s_out)
        ctx.dt, ctx.M, ctx.hard_reset, ctx.alpha = float(dt), M, hard_reset, float(alpha)
        return v_out, I_out, s_out

    @staticmethod
    def backward(ctx, dv_out, dI_out, ds_out):
        saved = ctx.saved_tensors
        (v, Iasc, x, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, nr, s_out) = saved
        B = v.numel()
        # An unused output (e.g. a spike-only loss that ignores v_out / I_out)
        # arrives as None; substitute zeros like the Triton/Warp/CuPy backends.
        dv_out = dv_out if dv_out is not None else torch.zeros_like(v)
        dI_out = dI_out if dI_out is not None else torch.zeros_like(Iasc)
        ds_out = ds_out if ds_out is not None else torch.zeros_like(v)
        dv, dI, dx, dasc = _step_backward(B, ctx.M, ctx.hard_reset)(
            v, Iasc, x, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, nr,
            s_out, _as_fp32(dv_out), _as_fp32(dI_out), _as_fp32(ds_out),
            ctx.dt, ctx.alpha)
        return (dv, dI, dx, None, None, None, None, None, None, dasc,
                None, None, None, None, None)


def _glif3_step_tilelang(v, Iasc, x, params, not_refrac, dt, M,
                         hard_reset=False, alpha=2.0):
    """Single-step GLIF3 update (autograd) — the training primitive."""
    return _GLIF3StepTileLang.apply(
        v, Iasc, x, params["v_th"], params["v_reset"], params["v_rest"],
        params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
        float(dt), int(M), bool(hard_reset), float(alpha))


# ----------------------------------------------------------------------------
# Neuron multistep — persistent, one thread per neuron carrying state across T.
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _neuron_multistep(T_steps: int, B: int, M: int, hard_reset: bool, block: int = _BLOCK):
    N = B * M
    HR = 1 if hard_reset else 0

    @T.prim_func
    def main(
        x_seq: T.Tensor((T_steps, B), "float32"),
        v0: T.Tensor((B,), "float32"), I0: T.Tensor((N,), "float32"),
        v_th: T.Tensor((B,), "float32"), v_reset: T.Tensor((B,), "float32"),
        v_rest: T.Tensor((B,), "float32"), c_m: T.Tensor((B,), "float32"),
        tau: T.Tensor((B,), "float32"), k: T.Tensor((N,), "float32"),
        asc_amps: T.Tensor((N,), "float32"), nr: T.Tensor((B,), "float32"),
        dt: T.float32,
        s_seq: T.Tensor((T_steps, B), "float32"),
        v_seq: T.Tensor((T_steps, B), "float32"),
        v_out: T.Tensor((B,), "float32"), I_out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(B, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < B:
                    vcur = T.alloc_local((1,), "float32")
                    Icur = T.alloc_local((M,), "float32")
                    vcur[0] = v0[idx]
                    for m in range(M):
                        Icur[m] = I0[idx * M + m]
                    for t in T.serial(0, T_steps):   # runtime loop, not unrolled
                        isum = T.alloc_local((1,), "float32")
                        isum[0] = T.float32(0)
                        for m in range(M):
                            isum[0] += Icur[m]
                        v_inf = v_rest[idx] + tau[idx] * (x_seq[t, idx] + isum[0]) / c_m[idx]
                        vp = v_inf + (vcur[0] - v_inf) * T.exp(-dt / tau[idx])
                        sp = T.alloc_local((1,), "float32")
                        sp[0] = T.float32(0)
                        if vp >= v_th[idx]:
                            sp[0] = nr[idx]
                        if HR == 1:
                            vcur[0] = vp - (vp - v_reset[idx]) * sp[0]
                        else:
                            vcur[0] = vp - (v_th[idx] - v_reset[idx]) * sp[0]
                        for m in range(M):
                            j = idx * M + m
                            Icur[m] = Icur[m] * T.exp(-k[j] * dt) + asc_amps[j] * sp[0]
                        s_seq[t, idx] = sp[0]
                        v_seq[t, idx] = vcur[0]
                    v_out[idx] = vcur[0]
                    for m in range(M):
                        I_out[idx * M + m] = Icur[m]

    return tilelang.compile(main, out_idx=[12, 13, 14, 15])


@functools.lru_cache(maxsize=None)
def _neuron_multistep_train_fwd(T_steps: int, B: int, M: int, hard_reset: bool,
                                block: int = _BLOCK):
    """Training forward: like the inference multistep but also saves the pre-step
    ``I_seq`` the backward needs. Runtime ``T.serial`` loop — not unrolled."""
    N = B * M
    HR = 1 if hard_reset else 0

    @T.prim_func
    def main(
        x_seq: T.Tensor((T_steps, B), "float32"),
        v0: T.Tensor((B,), "float32"), I0: T.Tensor((N,), "float32"),
        v_th: T.Tensor((B,), "float32"), v_reset: T.Tensor((B,), "float32"),
        v_rest: T.Tensor((B,), "float32"), c_m: T.Tensor((B,), "float32"),
        tau: T.Tensor((B,), "float32"), k: T.Tensor((N,), "float32"),
        asc_amps: T.Tensor((N,), "float32"), nr: T.Tensor((B,), "float32"),
        dt: T.float32,
        s_seq: T.Tensor((T_steps, B), "float32"),
        v_seq: T.Tensor((T_steps, B), "float32"),
        I_seq: T.Tensor((T_steps, N), "float32"),
        v_out: T.Tensor((B,), "float32"), I_out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(B, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < B:
                    vcur = T.alloc_local((1,), "float32"); vcur[0] = v0[idx]
                    Icur = T.alloc_local((M,), "float32")
                    for m in range(M):
                        Icur[m] = I0[idx * M + m]
                    for t in T.serial(0, T_steps):
                        for m in range(M):
                            I_seq[t, idx * M + m] = Icur[m]   # pre-step I_asc
                        isum = T.alloc_local((1,), "float32"); isum[0] = T.float32(0)
                        for m in range(M):
                            isum[0] += Icur[m]
                        v_inf = v_rest[idx] + tau[idx] * (x_seq[t, idx] + isum[0]) / c_m[idx]
                        vp = v_inf + (vcur[0] - v_inf) * T.exp(-dt / tau[idx])
                        sp = T.alloc_local((1,), "float32"); sp[0] = T.float32(0)
                        if vp >= v_th[idx]:
                            sp[0] = nr[idx]
                        if HR == 1:
                            vcur[0] = vp - (vp - v_reset[idx]) * sp[0]
                        else:
                            vcur[0] = vp - (v_th[idx] - v_reset[idx]) * sp[0]
                        for m in range(M):
                            j = idx * M + m
                            Icur[m] = Icur[m] * T.exp(-k[j] * dt) + asc_amps[j] * sp[0]
                        s_seq[t, idx] = sp[0]; v_seq[t, idx] = vcur[0]
                    v_out[idx] = vcur[0]
                    for m in range(M):
                        I_out[idx * M + m] = Icur[m]

    return tilelang.compile(main, out_idx=[12, 13, 14, 15, 16])


@functools.lru_cache(maxsize=None)
def _neuron_multistep_train_bwd(T_steps: int, B: int, M: int, hard_reset: bool,
                                block: int = _BLOCK):
    """Fused BPTT backward: ONE kernel walks the T steps in reverse (runtime
    ``T.serial``, not unrolled), carrying the reverse-time adjoints (dv_post, dI,
    dasc) in registers — the same math as the single-step backward composed T
    times, but a single launch instead of T."""
    N = B * M
    HR = 1 if hard_reset else 0

    @T.prim_func
    def main(
        x_seq: T.Tensor((T_steps, B), "float32"), v0: T.Tensor((B,), "float32"),
        v_seq: T.Tensor((T_steps, B), "float32"), s_seq: T.Tensor((T_steps, B), "float32"),
        I_seq: T.Tensor((T_steps, N), "float32"),
        v_th: T.Tensor((B,), "float32"), v_reset: T.Tensor((B,), "float32"),
        v_rest: T.Tensor((B,), "float32"), c_m: T.Tensor((B,), "float32"),
        tau: T.Tensor((B,), "float32"), k: T.Tensor((N,), "float32"),
        asc_amps: T.Tensor((N,), "float32"), nr: T.Tensor((B,), "float32"),
        ds_seq: T.Tensor((T_steps, B), "float32"), dv_seq: T.Tensor((T_steps, B), "float32"),
        dv_out: T.Tensor((B,), "float32"), dI_out: T.Tensor((N,), "float32"),
        dt: T.float32, alpha: T.float32,
        dv0: T.Tensor((B,), "float32"), dI0: T.Tensor((N,), "float32"),
        dx: T.Tensor((T_steps, B), "float32"), dasc: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(B, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < B:
                    decay = T.exp(-dt / tau[idx])
                    dv_post = T.alloc_local((1,), "float32"); dv_post[0] = dv_out[idx]
                    dI = T.alloc_local((M,), "float32")
                    dA = T.alloc_local((M,), "float32")
                    for m in range(M):
                        dI[m] = dI_out[idx * M + m]; dA[m] = T.float32(0)
                    for ti in T.serial(0, T_steps):
                        t = T_steps - 1 - ti
                        dv_post[0] += dv_seq[t, idx]
                        sp = s_seq[t, idx]
                        vpre = T.alloc_local((1,), "float32")
                        if t == 0:
                            vpre[0] = v0[idx]
                        else:
                            vpre[0] = v_seq[t - 1, idx]
                        isum = T.alloc_local((1,), "float32"); isum[0] = T.float32(0)
                        for m in range(M):
                            isum[0] += I_seq[t, idx * M + m]
                        v_inf = v_rest[idx] + tau[idx] * (x_seq[t, idx] + isum[0]) / c_m[idx]
                        vprime = v_inf + (vpre[0] - v_inf) * decay
                        denom = v_th[idx] - v_reset[idx]
                        u = (vprime - v_th[idx]) / denom
                        ds_dvp = nr[idx] / (1.0 + (alpha * u) * (alpha * u)) / denom
                        dvp = T.alloc_local((1,), "float32")
                        dsv = T.alloc_local((1,), "float32")
                        if HR == 1:
                            dvp[0] = dv_post[0] * (1.0 - sp)
                            dsv[0] = dv_post[0] * (-(vprime - v_reset[idx]))
                        else:
                            dvp[0] = dv_post[0]
                            dsv[0] = dv_post[0] * (-(v_th[idx] - v_reset[idx]))
                        dIs = T.alloc_local((1,), "float32"); dIs[0] = T.float32(0)
                        for m in range(M):
                            dIs[0] += dI[m] * asc_amps[idx * M + m]
                        ds_total = ds_seq[t, idx] + dsv[0] + dIs[0]
                        dvp[0] = dvp[0] + ds_total * ds_dvp
                        dv_inf = dvp[0] * (1.0 - decay)
                        dI_common = dv_inf * (tau[idx] / c_m[idx])
                        dx[t, idx] = dI_common
                        for m in range(M):
                            j = idx * M + m
                            dA[m] += dI[m] * sp
                            dI[m] = dI[m] * T.exp(-k[j] * dt) + dI_common
                        dv_post[0] = dvp[0] * decay
                    dv0[idx] = dv_post[0]
                    for m in range(M):
                        dI0[idx * M + m] = dI[m]; dasc[idx * M + m] = dA[m]

    return tilelang.compile(main, out_idx=[19, 20, 21, 22])


class _GLIF3NeuronMultiStepTileLang(torch.autograd.Function):
    """Neuron-only multistep with a single fused BPTT backward kernel."""

    @staticmethod
    def forward(ctx, x_seq, v0, I0, v_th, v_reset, v_rest, c_m, tau, k, asc_amps,
                nr, dt, M, hard_reset, alpha):
        Tn, B = x_seq.shape
        args = [_as_fp32(t) for t in (x_seq, v0, I0, v_th, v_reset, v_rest, c_m,
                                      tau, k, asc_amps, nr)]
        s_seq, v_seq, I_seq, v_out, I_out = _neuron_multistep_train_fwd(
            Tn, B, M, hard_reset)(*args, float(dt))
        ctx.save_for_backward(args[0], args[1], v_seq, s_seq, I_seq, *args[3:11])
        ctx.dt, ctx.M, ctx.hard_reset, ctx.alpha = float(dt), M, hard_reset, float(alpha)
        return s_seq, v_seq, v_out, I_out

    @staticmethod
    def backward(ctx, ds_seq, dv_seq, dv_out, dI_out):
        (x_seq, v0, v_seq, s_seq, I_seq, v_th, v_reset, v_rest, c_m, tau, k,
         asc_amps, nr) = ctx.saved_tensors
        Tn, B = x_seq.shape
        N = B * ctx.M
        z = torch.zeros
        ds_seq = _as_fp32(ds_seq) if ds_seq is not None else z(Tn, B, device=x_seq.device)
        dv_seq = _as_fp32(dv_seq) if dv_seq is not None else z(Tn, B, device=x_seq.device)
        dv_out = _as_fp32(dv_out) if dv_out is not None else z(B, device=x_seq.device)
        dI_out = _as_fp32(dI_out) if dI_out is not None else z(N, device=x_seq.device)
        dv0, dI0, dx, dasc = _neuron_multistep_train_bwd(Tn, B, ctx.M, ctx.hard_reset)(
            x_seq, v0, v_seq, s_seq, I_seq, v_th, v_reset, v_rest, c_m, tau, k,
            asc_amps, nr, ds_seq, dv_seq, dv_out, dI_out, ctx.dt, ctx.alpha)
        return (dx, dv0, dI0, None, None, None, None, None, None, dasc,
                None, None, None, None, None)


def _neuron_multistep_autograd(x_seq, v, Iasc, params, not_refrac, dt, M,
                               hard_reset, alpha):
    """Neuron-only multistep training: a single fused forward + fused BPTT
    backward (both loop over T internally, not Python-unrolled)."""
    Tn, B = x_seq.shape
    s_seq, v_seq, v_out, I_out = _GLIF3NeuronMultiStepTileLang.apply(
        x_seq, v, Iasc.reshape(-1), params["v_th"], params["v_reset"],
        params["v_rest"], params["c_m"], params["tau"], params["k"],
        params["asc_amps"], not_refrac, float(dt), int(M), bool(hard_reset),
        float(alpha))
    return s_seq, v_seq, v_out, I_out.view(B, M)


def glif3_multistep_fused_tilelang(x_seq, v, Iasc, params, not_refrac, dt, M,
                                   hard_reset=False, alpha=2.0):
    """Neuron-only multistep: lean forward-only kernel for inference, a fused
    forward + single fused BPTT kernel for training."""
    need_grad = torch.is_grad_enabled() and (
        x_seq.requires_grad or v.requires_grad or Iasc.requires_grad
        or params["asc_amps"].requires_grad)
    if need_grad:
        return _neuron_multistep_autograd(
            x_seq, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha)
    Tn, B = x_seq.shape
    fp = [_as_fp32(t) for t in (x_seq, v, Iasc.reshape(-1), params["v_th"],
                                params["v_reset"], params["v_rest"], params["c_m"],
                                params["tau"], params["k"], params["asc_amps"],
                                not_refrac)]
    s_seq, v_seq, v_out, I_out = _neuron_multistep(Tn, B, M, hard_reset)(*fp, float(dt))
    return s_seq, v_seq, v_out, I_out.view(B, M)


# ----------------------------------------------------------------------------
# Dense recurrent multistep — cuBLAS gemv + TileLang neuron step per timestep.
# ----------------------------------------------------------------------------
def glif3_dense_multistep_fused_tilelang(x_seq, weight, bias, v, Iasc, params,
                                         not_refrac, dt, M, hard_reset=False, alpha=2.0,
                                         fused_matmul=True):
    # fused_matmul is accepted for interface parity with the Triton backend; the
    # TileLang dense path always uses a cuBLAS gemv + the TileLang neuron step.
    need_grad = torch.is_grad_enabled() and (
        weight.requires_grad or bias.requires_grad or x_seq.requires_grad
        or v.requires_grad or Iasc.requires_grad or params["asc_amps"].requires_grad)
    if need_grad:
        return dense_multistep_autograd(
            _glif3_step_tilelang, x_seq, weight, bias, v, Iasc, params,
            not_refrac, dt, M, hard_reset, alpha)
    Tn, B = x_seq.shape
    fwd = _step_forward(B, M, hard_reset)
    fp = {key: _as_fp32(params[key]) for key in params}
    v_th, v_reset, v_rest = fp["v_th"], fp["v_reset"], fp["v_rest"]
    c_m, tau, k, asc = fp["c_m"], fp["tau"], fp["k"], fp["asc_amps"]
    nr = _as_fp32(not_refrac)
    v_cur = _as_fp32(v).clone()
    I_cur = _as_fp32(Iasc.reshape(-1)).clone()
    s_prev = torch.zeros(B, device=v.device, dtype=torch.float32)
    W = _as_fp32(weight)
    b = _as_fp32(bias)
    s_seq = torch.empty((Tn, B), device=v.device, dtype=torch.float32)
    v_seq = torch.empty((Tn, B), device=v.device, dtype=torch.float32)
    with torch.no_grad():
        for t in range(Tn):
            x_in = torch.addmv(x_seq[t] + b, W, s_prev)
            v_cur, I_cur, s_prev = fwd(v_cur, I_cur, x_in, v_th, v_reset, v_rest,
                                       c_m, tau, k, asc, nr, float(dt))
            s_seq[t] = s_prev
            v_seq[t] = v_cur
    return s_seq, v_seq, v_cur, I_cur.view(B, M)


# ----------------------------------------------------------------------------
# Sparse recurrent multistep — TileLang CSR SpMV fused with the neuron update.
# ----------------------------------------------------------------------------

# CSR-vector tile: one warp per row (lanes=32, cheap single-warp shuffle reduce,
# same as CuPy/Triton), packed to the 1024-thread block max (rows=32) so one CTA
# processes 32 rows — the way to get parallelism in TileLang's one-block-per-SM
# cooperative model is a bigger block, not more blocks (see _persistent_grid).
_SP_ROWS, _SP_LANES = 32, 32


@functools.lru_cache(maxsize=None)
def _persistent_grid() -> int:
    """CTAs for the cooperative persistent kernel = one block per SM, the pattern
    every TileLang persistent example uses (``driver.get_num_sms()`` in
    ``example_gemm_persistent`` / ``example_mla_decode_persistent``, and the
    default of ``T.PersistentTileScheduler``). A cooperative launch needs all CTAs
    co-resident; TileLang has no occupancy API and does no grid clamping, so it
    would pass an over-provisioned grid straight to ``cudaLaunchCooperativeKernel``
    and hit ``CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_MANY_BLOCKS``. num_SMs blocks is
    always ≤ the cooperative max; throughput comes from a bigger block."""
    from tilelang.carver.arch import driver
    return driver.get_num_sms()


@functools.lru_cache(maxsize=None)
def _sparse_persistent(T_steps: int, B: int, M: int, nnz: int, hard_reset: bool,
                       blocks: int, rows: int = _SP_ROWS, lanes: int = _SP_LANES):
    """Persistent fused sparse multistep: ONE cooperative launch runs the whole
    T-step recurrence. The neuron state (v, Iasc) is resident in grid-global
    scratch; each step does a **CSR-vector** SpMV (``rows`` rows per block,
    ``lanes`` lanes striding each row's nonzeros, reduced over lanes — the same
    warp-per-row schedule as the Triton/CuPy/Warp kernels), fuses the GLIF3
    update, double-buffers the spike vector, and a ``T.sync_grid()`` barrier
    separates steps. No per-step kernel relaunch and no state round-trip through
    a launch boundary — the fusion the stock backends cannot express."""
    N = B * M
    HR = 1 if hard_reset else 0

    @T.prim_func
    def main(
        crow: T.Tensor((B + 1,), "int32"), col: T.Tensor((nnz,), "int32"),
        val: T.Tensor((nnz,), "float32"),
        x_seq: T.Tensor((T_steps, B), "float32"), bias: T.Tensor((B,), "float32"),
        v0: T.Tensor((B,), "float32"), I0: T.Tensor((N,), "float32"),
        v_th: T.Tensor((B,), "float32"), v_reset: T.Tensor((B,), "float32"),
        v_rest: T.Tensor((B,), "float32"), c_m: T.Tensor((B,), "float32"),
        tau: T.Tensor((B,), "float32"), k: T.Tensor((N,), "float32"),
        asc_amps: T.Tensor((N,), "float32"), nr: T.Tensor((B,), "float32"),
        dt: T.float32,
        s_seq: T.Tensor((T_steps, B), "float32"),
        v_seq: T.Tensor((T_steps, B), "float32"),
        v_out: T.Tensor((B,), "float32"), I_out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(blocks, threads=rows * lanes) as pid:
            sbuf = T.alloc_global((2, B), "float32")   # spike double buffer
            vbuf = T.alloc_global((B,), "float32")     # resident membrane state
            Ibuf = T.alloc_global((N,), "float32")     # resident after-spike currents
            for tile in T.Persistent([T.ceildiv(B, rows)], blocks, pid):
                for r, lane in T.Parallel(rows, lanes):
                    row = tile * rows + r
                    if row < B:
                        if lane == 0:
                            vbuf[row] = v0[row]
                            sbuf[0, row] = T.float32(0)
                            for m in range(M):
                                Ibuf[row * M + m] = I0[row * M + m]
            T.sync_grid()
            for t in T.serial(0, T_steps):   # runtime loop, not unrolled
                cur = t % 2
                nxt = (t + 1) % 2
                for tile in T.Persistent([T.ceildiv(B, rows)], blocks, pid):
                    acc = T.alloc_fragment((rows, lanes), "float32")
                    T.clear(acc)
                    for r, lane in T.Parallel(rows, lanes):
                        row = tile * rows + r
                        if row < B:
                            for p in T.serial(crow[row] + lane, crow[row + 1], lanes):
                                acc[r, lane] += val[p] * sbuf[cur, col[p]]
                    lin = T.alloc_fragment((rows,), "float32")
                    T.reduce_sum(acc, lin, dim=1)
                    for r in T.Parallel(rows):
                        row = tile * rows + r
                        if row < B:
                            isum = T.alloc_local((1,), "float32")
                            isum[0] = T.float32(0)
                            for m in range(M):
                                isum[0] += Ibuf[row * M + m]
                            x_in = x_seq[t, row] + bias[row] + lin[r]
                            v_inf = v_rest[row] + tau[row] * (x_in + isum[0]) / c_m[row]
                            vp = v_inf + (vbuf[row] - v_inf) * T.exp(-dt / tau[row])
                            sp = T.alloc_local((1,), "float32")
                            sp[0] = T.float32(0)
                            if vp >= v_th[row]:
                                sp[0] = nr[row]
                            if HR == 1:
                                vbuf[row] = vp - (vp - v_reset[row]) * sp[0]
                            else:
                                vbuf[row] = vp - (v_th[row] - v_reset[row]) * sp[0]
                            for m in range(M):
                                j = row * M + m
                                Ibuf[j] = Ibuf[j] * T.exp(-k[j] * dt) + asc_amps[j] * sp[0]
                            sbuf[nxt, row] = sp[0]
                            s_seq[t, row] = sp[0]
                            v_seq[t, row] = vbuf[row]
                T.sync_grid()
            for tile in T.Persistent([T.ceildiv(B, rows)], blocks, pid):
                for r, lane in T.Parallel(rows, lanes):
                    row = tile * rows + r
                    if row < B and lane == 0:
                        v_out[row] = vbuf[row]
                        for m in range(M):
                            I_out[row * M + m] = Ibuf[row * M + m]

    return tilelang.compile(
        main, out_idx=[16, 17, 18, 19],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})


def glif3_sparse_multistep_fused_tilelang(x_seq, weight, bias, v, Iasc, params,
                                          not_refrac, dt, M, hard_reset=False, alpha=2.0):
    need_grad = torch.is_grad_enabled() and (
        weight.val.requires_grad or bias.requires_grad or x_seq.requires_grad
        or v.requires_grad or Iasc.requires_grad or params["asc_amps"].requires_grad)
    if need_grad:
        return sparse_multistep_autograd(
            _glif3_step_tilelang, x_seq, weight, bias, v, Iasc, params,
            not_refrac, dt, M, hard_reset, alpha)
    Tn, B = x_seq.shape
    kern = _sparse_persistent(Tn, B, M, int(weight.col.numel()), hard_reset,
                              _persistent_grid())
    fp = {key: _as_fp32(params[key]) for key in params}
    with torch.no_grad():
        s_seq, v_seq, v_out, I_out = kern(
            weight.crow.int().contiguous(), weight.col.int().contiguous(),
            _as_fp32(weight.val), _as_fp32(x_seq), _as_fp32(bias),
            _as_fp32(v), _as_fp32(Iasc.reshape(-1)),
            fp["v_th"], fp["v_reset"], fp["v_rest"], fp["c_m"], fp["tau"],
            fp["k"], fp["asc_amps"], _as_fp32(not_refrac), float(dt))
    return s_seq, v_seq, v_out, I_out.view(B, M)


glif3_step_tilelang = GLIF3StepOps(
    step=_glif3_step_tilelang,
    multistep_fused=glif3_multistep_fused_tilelang,
    dense_multistep_fused=glif3_dense_multistep_fused_tilelang,
    sparse_multistep_fused=glif3_sparse_multistep_fused_tilelang,
)
