"""Warp GLIF3 kernels: single-step, neuron multistep, and dense (neuron +
recurrent connection) multistep.

The GLIF3 update per step is

    v_inf   = v_rest + tau * (x + sum I_asc) / c_m
    v'      = v_inf + (v - v_inf) * exp(-dt / tau)
    spike   = heaviside(v' - v_th) * not_refrac
    v_post  = v' - (v' - v_reset) * spike   (hard reset)
            = v' - (v_th - v_reset) * spike (soft reset)
    I_asc'  = I_asc * exp(-k dt) + asc_amps * spike

The device dynamics are factored into ``@wp.func`` helpers (``neuronal_charge``
/ ``neuronal_fire`` / ``neuronal_reset`` / ``asc_sum`` / ``asc_adapt``)
mirroring the eager ``btorch.models.neurons.glif.GLIF3`` methods, so each kernel
reads as a composition of those steps.

Autodiff is Warp-native: there are no hand-written backward kernels. Training
uses ``wp.Tape`` and the surrogate spike gradient is supplied as a custom
``@wp.func_grad`` on ``neuronal_fire``. Two facts from Warp's differentiability
guide shape the kernels:

- *Static loops* (compile-time bounds) are unrolled and differentiate cleanly.
  The ``M`` after-spike-current modes are carried as a Warp **vector** whose
  length is ``M``; the per-mode loops (``asc_sum`` / ``asc_adapt``) then have a
  compile-time bound and unroll. One generic (``dtype=Any``) kernel specializes
  per ``M`` automatically -- no per-``M`` factory.
- *Dynamic loops* (a runtime ``for t in range(T)``) are **not** replayed in the
  backward pass, so loop-carried state read from registers gives wrong
  gradients. The training multistep kernel therefore keeps the per-step state in
  arrays (``v_seq`` / ``I_seq``) and reads the previous step back from them, which
  the guide documents as the fix. The inference multistep kernel carries state
  in registers (forward only, no autodiff) and is correspondingly leaner.

Organization:
- single-step ``glif3_step_warp`` (autograd, the training primitive)
- neuron multistep ``glif3_multistep_fused_warp`` (autograd for training,
  a lean forward-only path for inference)
- dense multistep ``glif3_dense_multistep_fused_warp`` (inference or training;
  the recurrent matmul is delegated to cuBLAS and the neuron dynamics run in a
  Warp kernel per step)
- sparse multistep ``glif3_sparse_multistep_fused_warp`` (scale-free recurrent
  connection; inference is a CSR-vector SpMV via ``wp.tile_sum`` then the neuron
  kernel, training differentiates the SpMV on the ``wp.Tape`` too)

Conventions (contiguous, fp32 on CUDA):
- ``v``, ``x``, ``not_refrac``: shape ``(B,)``
- ``Iasc``, ``k``, ``asc_amps``: shape ``(B, M)`` (per-neuron ASC vector of length M)
- surrogate gradient is btorch's exact ``ATan`` (damping = 1)
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp
from jaxtyping import Float

from benchmarks.dense_glif_net.glif_common import (
    GLIF3StepOps,
    SparseWeight,
    dense_multistep_autograd,
    sparse_multistep_autograd,
)


_WARP_INITIALIZED = False


# ----------------------------------------------------------------------------
# Device helpers: one @wp.func per step of the GLIF3 update.
#
# The ASC helpers (asc_sum / asc_adapt) are generic over the per-neuron ASC
# vector type: ``i_asc.length`` is the compile-time mode count M, so the loops
# unroll and Warp specializes each kernel per M.
# ----------------------------------------------------------------------------
@wp.func
def expm1(x: float) -> float:
    """exp(x) - 1 (Warp has no wp.expm1; accurate enough over our arg range)."""
    return wp.exp(x) - 1.0


@wp.func
def exp_euler(x: float, derivative: float, linear: float, dt: float) -> float:
    """One exponential-Euler step of dx/dt = A*x + B, given the derivative f(x)
    at x and the linear term A: x_next = x + expm1(dt*A)/A * f(x). Mirrors
    btorch.models.ode.exp_euler_step."""
    return x + expm1(dt * linear) / linear * derivative


@wp.func
def neuronal_charge(
    v: float, x: float, i_sum: float,
    v_rest: float, c_m: float, tau: float, dt: float
) -> float:
    """Exponential-Euler membrane update -> v' (pre-spike voltage):
    dv/dt = -(v - v_rest)/tau + (x + sum I_asc)/c_m, linear term A = -1/tau."""
    dv = -(v - v_rest) / tau + (x + i_sum) / c_m
    return exp_euler(v, dv, -1.0 / tau, dt)


@wp.func
def neuronal_fire(
    v_prime: float, v_th: float, v_reset: float, mask: float, alpha: float
) -> float:
    """Hard-threshold spike (heaviside), gated by the refractory mask.

    The forward value ignores ``v_reset`` / ``alpha``; they parameterize the
    custom surrogate gradient below (Warp requires the grad function to share
    the forward signature)."""
    return wp.where(v_prime >= v_th, 1.0, 0.0) * mask


@wp.func_grad(neuronal_fire)
def neuronal_fire_grad(
    v_prime: float, v_th: float, v_reset: float, mask: float, alpha: float,
    adj: float,
):
    """Surrogate spike gradient d spike / d v' (btorch ATan: 1 / (1 + (alpha*u)^2),
    damping = 1, with u = (v' - v_th) / (v_th - v_reset)). Only v' carries an
    adjoint; the threshold/reset/mask are treated as constants."""
    denom = v_th - v_reset
    u = (v_prime - v_th) / denom
    wp.adjoint[v_prime] += adj * mask / (1.0 + (alpha * u) * (alpha * u)) / denom


@wp.func
def neuronal_reset(
    v_prime: float, spike: float, v_th: float, v_reset: float, hard_reset: int
) -> float:
    """Membrane reset after a spike (hard or soft)."""
    if hard_reset == 1:
        return v_prime - (v_prime - v_reset) * spike
    return v_prime - (v_th - v_reset) * spike


@wp.func
def neuronal_adaptation(
    i_asc: float, k: float, asc_amp: float, spike: float, dt: float
) -> float:
    """After-spike current for one mode: exp-Euler decay (dI/dt = -k*I) + jump.
    Scalar form used by the flat dense inference kernels."""
    return exp_euler(i_asc, -k * i_asc, -k, dt) + asc_amp * spike


@wp.func
def asc_sum(i_asc: Any):
    """Sum of the M after-spike currents (compile-time unrolled over the vector)."""
    s = float(0.0)
    for m in range(i_asc.length):
        s += i_asc[m]
    return s


@wp.func
def asc_adapt(i_asc: Any, k: Any, asc_amp: Any, spike: float, dt: float):
    """Per-mode after-spike-current update -> new ASC vector. Built from a fresh
    (zeroed) vector rather than aliasing ``i_asc`` so each component is assigned
    exactly once, which Warp autodiff requires."""
    out = i_asc * 0.0
    for m in range(i_asc.length):
        out[m] = exp_euler(i_asc[m], -k[m] * i_asc[m], -k[m], dt) + asc_amp[m] * spike
    return out


# ----------------------------------------------------------------------------
# Single-step kernel (generic over the ASC vector type = M)
# ----------------------------------------------------------------------------
@wp.kernel
def glif3_step_forward_kernel(
    v: wp.array(dtype=wp.float32),
    Iasc: wp.array(dtype=Any),
    x: wp.array(dtype=wp.float32),
    v_out: wp.array(dtype=wp.float32),
    Iasc_out: wp.array(dtype=Any),
    spike_out: wp.array(dtype=wp.float32),
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=Any),
    asc_amps: wp.array(dtype=Any),
    not_refrac: wp.array(dtype=wp.float32),
    dt: float,
    hard_reset: int,
    alpha: float,
):
    i = wp.tid()
    i_asc = Iasc[i]
    v_prime = neuronal_charge(
        v[i], x[i], asc_sum(i_asc), v_rest[i], c_m[i], tau[i], dt)
    spike = neuronal_fire(v_prime, v_th[i], v_reset[i], not_refrac[i], alpha)
    v_out[i] = neuronal_reset(v_prime, spike, v_th[i], v_reset[i], hard_reset)
    Iasc_out[i] = asc_adapt(i_asc, k[i], asc_amps[i], spike, dt)
    spike_out[i] = spike


# ----------------------------------------------------------------------------
# Neuron multistep kernels (generic over the ASC vector type = M)
# ----------------------------------------------------------------------------
@wp.kernel
def glif3_neuron_multistep_kernel(
    x_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v0: wp.array(dtype=wp.float32),       # (B,)
    I0: wp.array(dtype=Any),              # (B,) ASC vectors
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=Any),
    asc_amps: wp.array(dtype=Any),
    not_refrac: wp.array(dtype=wp.float32),
    dt: float,
    T: int,
    hard_reset: int,
    alpha: float,
    s_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    I_seq: wp.array2d(dtype=Any),         # (T, B) ASC vectors
):
    # Training path: the runtime `for t in range(T)` loop is a dynamic loop, so
    # the previous step's state is read back from v_seq / I_seq (not registers)
    # to keep the backward pass exact under wp.Tape.
    i = wp.tid()
    for t in range(T):
        if t == 0:
            v_prev = v0[i]
            i_asc = I0[i]
        else:
            v_prev = v_seq[t - 1, i]
            i_asc = I_seq[t - 1, i]

        v_prime = neuronal_charge(
            v_prev, x_seq[t, i], asc_sum(i_asc), v_rest[i], c_m[i], tau[i], dt)
        spike = neuronal_fire(v_prime, v_th[i], v_reset[i], not_refrac[i], alpha)
        v_seq[t, i] = neuronal_reset(v_prime, spike, v_th[i], v_reset[i], hard_reset)
        I_seq[t, i] = asc_adapt(i_asc, k[i], asc_amps[i], spike, dt)
        s_seq[t, i] = spike


@wp.kernel(enable_backward=False)
def glif3_neuron_multistep_infer_kernel(
    x_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v0: wp.array(dtype=wp.float32),       # (B,)
    I0: wp.array(dtype=Any),              # (B,) ASC vectors
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=Any),
    asc_amps: wp.array(dtype=Any),
    not_refrac: wp.array(dtype=wp.float32),
    dt: float,
    T: int,
    hard_reset: int,
    alpha: float,
    s_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v_out: wp.array(dtype=wp.float32),    # (B,)
    I_out: wp.array(dtype=Any),           # (B,) ASC vectors
):
    # Inference path: forward only, so per-step state is carried in registers.
    i = wp.tid()
    v_i = v0[i]
    i_asc = I0[i]
    for t in range(T):
        v_prime = neuronal_charge(
            v_i, x_seq[t, i], asc_sum(i_asc), v_rest[i], c_m[i], tau[i], dt)
        spike = neuronal_fire(v_prime, v_th[i], v_reset[i], not_refrac[i], alpha)
        v_i = neuronal_reset(v_prime, spike, v_th[i], v_reset[i], hard_reset)
        i_asc = asc_adapt(i_asc, k[i], asc_amps[i], spike, dt)
        s_seq[t, i] = spike
        v_seq[t, i] = v_i
    v_out[i] = v_i
    I_out[i] = i_asc


# ----------------------------------------------------------------------------
# Dense multistep kernels (inference; flat ASC, forward only)
# ----------------------------------------------------------------------------
@wp.kernel
def glif3_dense_step_kernel(
    x_in: wp.array(dtype=wp.float32),  # (B,) already includes bias + recurrent term
    v: wp.array(dtype=wp.float32),     # (B,)
    Iasc: wp.array(dtype=wp.float32),  # (B*M,)
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),
    asc_amps: wp.array(dtype=wp.float32),
    not_refrac: wp.array(dtype=wp.float32),
    s_out: wp.array(dtype=wp.float32),  # (B,)
    v_out: wp.array(dtype=wp.float32),  # (B,)
    M: int,
    dt: float,
    hard_reset: int,
    alpha: float,
):
    i = wp.tid()
    base = i * M

    i_sum = float(0.0)
    for m in range(M):
        i_sum += Iasc[base + m]

    v_prime = neuronal_charge(v[i], x_in[i], i_sum, v_rest[i], c_m[i], tau[i], dt)
    spike = neuronal_fire(v_prime, v_th[i], v_reset[i], not_refrac[i], alpha)
    v_post = neuronal_reset(v_prime, spike, v_th[i], v_reset[i], hard_reset)

    for m in range(M):
        Iasc[base + m] = neuronal_adaptation(
            Iasc[base + m], k[base + m], asc_amps[base + m], spike, dt)
    v[i] = v_post
    s_out[i] = spike
    v_out[i] = v_post


_TILE_LANES = wp.constant(64)  # threads per row (block) for the tiled SpMV


@wp.kernel(enable_backward=False)
def glif3_spmv_tiled_kernel(
    crow: wp.array(dtype=wp.int32),
    col: wp.array(dtype=wp.int32),
    val: wp.array(dtype=wp.float32),
    s_prev: wp.array(dtype=wp.float32),
    lin: wp.array(dtype=wp.float32),  # (N,)
):
    # CSR-vector with a cooperative reduction. Launched as one block of
    # _TILE_LANES threads per row (wp.launch, block_dim=_TILE_LANES). Each thread
    # strides the row's nonzeros for its partial; wp.tile(acc) gathers the block's
    # per-thread partials into a tile and wp.tile_sum reduces them in a single
    # shuffle-based cooperative reduction (no atomics).
    tid = wp.tid()                 # global thread id
    row = tid // _TILE_LANES
    lane = tid % _TILE_LANES
    acc = float(0.0)
    p = crow[row] + lane
    end = crow[row + 1]
    while p < end:
        acc += val[p] * s_prev[col[p]]
        p += _TILE_LANES
    total = wp.tile_sum(wp.tile(acc))
    wp.tile_store(lin, total, offset=(row,))


@wp.kernel
def glif3_spmv_grad_kernel(
    crow: wp.array(dtype=wp.int32),
    col: wp.array(dtype=wp.int32),
    val: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    lin: wp.array(dtype=wp.float32),
):
    # Differentiable one-thread-per-row SpMV for training: Warp's tape gets the
    # adjoints of ``val`` and ``s`` automatically (the read ``s[col[p]]``
    # back-propagates as an atomic scatter into ``s.grad``), so no hand-written
    # backward. Used only in the (graph-bound) training path.
    i = wp.tid()
    acc = float(0.0)
    for p in range(crow[i], crow[i + 1]):
        acc += val[p] * s[col[p]]
    lin[i] = acc


# ----------------------------------------------------------------------------
# Python-side helpers
# ----------------------------------------------------------------------------
def _as_fp32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor.contiguous()


def _check_shapes(
    B: int,
    M: int,
    v: torch.Tensor,
    Iasc: torch.Tensor,
    x: torch.Tensor,
    params: dict,
    not_refrac: torch.Tensor,
) -> None:
    if v.ndim != 1 or x.ndim != 1 or not_refrac.ndim != 1:
        raise ValueError("v, x, not_refrac must be 1D tensors.")
    if v.numel() != x.numel() or v.numel() != not_refrac.numel():
        raise ValueError("v, x, not_refrac must have the same length.")
    if Iasc.numel() != B * M:
        raise ValueError("Iasc must have B*M elements.")
    for key in ("v_th", "v_reset", "v_rest", "c_m", "tau"):
        if params[key].numel() != B:
            raise ValueError(f"params['{key}'] must have shape (B,).")
    for key in ("k", "asc_amps"):
        if params[key].numel() != B * M:
            raise ValueError(f"params['{key}'] must have shape (B*M,).")


def _ensure_warp_init() -> None:
    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True


def _vec_type(M: int):
    """Warp vector type holding one neuron's M after-spike currents. Warp caches
    these, so repeated calls with the same M return the same type."""
    return wp.types.vector(length=int(M), dtype=wp.float32)


def _wp(tensor: torch.Tensor) -> wp.array:
    """Wrap a contiguous fp32 tensor as a scalar Warp array (no gradient)."""
    return wp.from_torch(tensor, dtype=wp.float32, requires_grad=False)


def _wp_int(tensor: torch.Tensor) -> wp.array:
    """Wrap a contiguous int32 tensor as a Warp array (CSR index arrays)."""
    return wp.from_torch(tensor, dtype=wp.int32, requires_grad=False)


def _wp_grad(tensor: torch.Tensor) -> wp.array:
    """Scalar Warp array that accumulates a gradient (a differentiable leaf)."""
    return wp.from_torch(
        tensor.detach().contiguous(), dtype=wp.float32, requires_grad=True)


def _wp_vec(tensor: torch.Tensor, M: int, requires_grad: bool) -> wp.array:
    """Wrap ``(..., M)`` fp32 data as a Warp array of length-M vectors, so the
    trailing mode axis becomes the vector components."""
    data = tensor.detach().reshape(-1, int(M)).contiguous() if tensor.ndim == 1 \
        else tensor.detach().contiguous()
    return wp.from_torch(data, dtype=_vec_type(M), requires_grad=requires_grad)


def _wp_vec2d(tensor: torch.Tensor, M: int, requires_grad: bool) -> wp.array:
    """Wrap ``(T, B, M)`` fp32 data as a 2D ``(T, B)`` Warp array of length-M
    vectors."""
    return wp.from_torch(
        tensor.detach().contiguous(), dtype=_vec_type(M), requires_grad=requires_grad)


# ----------------------------------------------------------------------------
# Single-step (training primitive) -- Warp autodiff via wp.Tape
# ----------------------------------------------------------------------------
class GLIF3StepWarp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v,
        Iasc,
        x,
        v_th,
        v_reset,
        v_rest,
        c_m,
        tau,
        k,
        asc_amps,
        not_refrac,
        dt: float,
        M: int,
        hard_reset: bool,
        alpha: float,
    ):
        if not (v.is_cuda and Iasc.is_cuda and x.is_cuda):
            raise RuntimeError("GLIF3StepWarp requires CUDA tensors.")
        _ensure_warp_init()
        M = int(M)

        v, Iasc, x = _as_fp32(v), _as_fp32(Iasc), _as_fp32(x)
        v_th, v_reset, v_rest = _as_fp32(v_th), _as_fp32(v_reset), _as_fp32(v_rest)
        c_m, tau, k = _as_fp32(c_m), _as_fp32(tau), _as_fp32(k)
        asc_amps, not_refrac = _as_fp32(asc_amps), _as_fp32(not_refrac)

        B = v.numel()
        params = {
            "v_th": v_th, "v_reset": v_reset, "v_rest": v_rest,
            "c_m": c_m, "tau": tau, "k": k, "asc_amps": asc_amps,
        }
        _check_shapes(B, M, v, Iasc, x, params, not_refrac)

        # Differentiable inputs carry a Warp gradient; the rest are constants.
        v_wp = _wp_grad(v)
        I_wp = _wp_vec(Iasc, M, requires_grad=True)
        x_wp = _wp_grad(x)
        asc_wp = _wp_vec(asc_amps, M, requires_grad=True)

        v_out = torch.empty_like(v)
        I_out = torch.empty((B, M), device=v.device, dtype=v.dtype)
        s_out = torch.empty_like(v)
        v_out_wp = _wp_grad(v_out)
        I_out_wp = _wp_vec(I_out, M, requires_grad=True)
        s_out_wp = _wp_grad(s_out)

        const = [
            _wp(v_th), _wp(v_reset), _wp(v_rest), _wp(c_m), _wp(tau),
            _wp_vec(k, M, requires_grad=False), asc_wp, _wp(not_refrac),
        ]
        tape = wp.Tape()
        with tape:
            wp.launch(
                glif3_step_forward_kernel,
                dim=B,
                inputs=[
                    v_wp, I_wp, x_wp, v_out_wp, I_out_wp, s_out_wp, *const,
                    float(dt), int(1 if hard_reset else 0), float(alpha),
                ],
                device="cuda",
            )

        ctx.tape = tape
        ctx.grad_inputs = (v_wp, I_wp, x_wp, asc_wp)
        ctx.grad_outputs = (v_out_wp, I_out_wp, s_out_wp)
        ctx.M, ctx.Iasc_was_flat = M, Iasc.ndim == 1
        ctx.asc_was_flat = asc_amps.ndim == 1
        return v_out, I_out.reshape(-1) if Iasc.ndim == 1 else I_out, s_out

    @staticmethod
    def backward(ctx, dv_out, dI_out, ds_out):
        v_wp, I_wp, x_wp, asc_wp = ctx.grad_inputs
        v_out_wp, I_out_wp, s_out_wp = ctx.grad_outputs
        M = ctx.M

        for arr in ctx.grad_inputs:
            arr.grad.zero_()
        _seed(v_out_wp, dv_out)
        _seed(I_out_wp, dI_out)
        _seed(s_out_wp, ds_out)
        ctx.tape.backward()

        dv = wp.to_torch(v_wp.grad).clone()
        dI = wp.to_torch(I_wp.grad).clone()
        dx = wp.to_torch(x_wp.grad).clone()
        dasc = wp.to_torch(asc_wp.grad).clone()
        ctx.tape.zero()

        if ctx.Iasc_was_flat:
            dI = dI.reshape(-1)
        if ctx.asc_was_flat:
            dasc = dasc.reshape(-1)
        return (dv, dI, dx, None, None, None, None, None, None, dasc,
                None, None, None, None, None)


def _seed(arr: wp.array, grad: torch.Tensor | None) -> None:
    """Seed a Warp output array's gradient in place (zero if None)."""
    dst = wp.to_torch(arr.grad)
    if grad is None:
        dst.zero_()
    else:
        dst.copy_(grad.reshape(dst.shape).to(dtype=dst.dtype))


def _glif3_step_warp(
    v: Float[torch.Tensor, " B"],
    Iasc: Float[torch.Tensor, " B M"],
    x: Float[torch.Tensor, " B"],
    params: dict,
    not_refrac: Float[torch.Tensor, " B"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-step GLIF3 update (autograd) using Warp."""
    return GLIF3StepWarp.apply(
        v, Iasc, x,
        params["v_th"], params["v_reset"], params["v_rest"],
        params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
        float(dt), int(M), bool(hard_reset), float(alpha),
    )


# ----------------------------------------------------------------------------
# Neuron multistep
# ----------------------------------------------------------------------------
class GLIF3NeuronMultiStepWarp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq,
        v,
        Iasc,
        v_th,
        v_reset,
        v_rest,
        c_m,
        tau,
        k,
        asc_amps,
        not_refrac,
        dt: float,
        M: int,
        hard_reset: bool,
        alpha: float,
    ):
        if not x_seq.is_cuda:
            raise RuntimeError("Neuron multistep Warp requires CUDA tensors.")
        _ensure_warp_init()
        M = int(M)

        x_seq, v, Iasc = _as_fp32(x_seq), _as_fp32(v), _as_fp32(Iasc)
        v_th, v_reset, v_rest = _as_fp32(v_th), _as_fp32(v_reset), _as_fp32(v_rest)
        c_m, tau, k = _as_fp32(c_m), _as_fp32(tau), _as_fp32(k)
        asc_amps, not_refrac = _as_fp32(asc_amps), _as_fp32(not_refrac)
        T, B = x_seq.shape

        x_wp = _wp_grad(x_seq)
        v0_wp = _wp_grad(v)
        I0_wp = _wp_vec(Iasc, M, requires_grad=True)
        asc_wp = _wp_vec(asc_amps, M, requires_grad=True)

        # Per-step state lives in arrays so the dynamic T-loop differentiates.
        s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
        v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
        I_seq = torch.empty((T, B, M), device=v.device, dtype=v.dtype)
        s_wp = _wp_grad(s_seq)
        v_wp = _wp_grad(v_seq)
        I_wp = _wp_vec2d(I_seq, M, requires_grad=True)

        const = [
            _wp(v_th), _wp(v_reset), _wp(v_rest), _wp(c_m), _wp(tau),
            _wp_vec(k, M, requires_grad=False), asc_wp, _wp(not_refrac),
        ]
        tape = wp.Tape()
        with tape:
            wp.launch(
                glif3_neuron_multistep_kernel,
                dim=B,
                inputs=[
                    x_wp, v0_wp, I0_wp, *const,
                    float(dt), int(T), int(1 if hard_reset else 0), float(alpha),
                    s_wp, v_wp, I_wp,
                ],
                block_dim=256,
                device="cuda",
            )

        ctx.tape = tape
        ctx.grad_inputs = (x_wp, v0_wp, I0_wp, asc_wp)
        ctx.grad_state = (s_wp, v_wp, I_wp)
        ctx.T, ctx.B, ctx.M = T, B, M
        ctx.Iasc_was_flat = Iasc.ndim == 1
        ctx.asc_was_flat = asc_amps.ndim == 1

        v_out = v_seq[T - 1]
        I_out = I_seq[T - 1]
        return s_seq, v_seq, v_out, I_out.reshape(-1) if Iasc.ndim == 1 else I_out

    @staticmethod
    def backward(ctx, ds_seq, dv_seq, dv_out, dI_out):
        x_wp, v0_wp, I0_wp, asc_wp = ctx.grad_inputs
        s_wp, v_wp, I_wp = ctx.grad_state
        T, B, M = ctx.T, ctx.B, ctx.M

        for arr in ctx.grad_inputs:
            arr.grad.zero_()
        # v_seq and its last-step alias v_out both feed gradients into v_seq;
        # I_seq only carries a gradient at its last step (I_out).
        _seed(s_wp, ds_seq)
        _seed(v_wp, dv_seq)
        v_grad = wp.to_torch(v_wp.grad)
        if dv_out is not None:
            v_grad[T - 1] += dv_out.to(dtype=v_grad.dtype)
        i_grad = wp.to_torch(I_wp.grad)
        i_grad.zero_()
        if dI_out is not None:
            i_grad[T - 1] += dI_out.reshape(B, M).to(dtype=i_grad.dtype)
        ctx.tape.backward()

        dx = wp.to_torch(x_wp.grad).clone()
        dv0 = wp.to_torch(v0_wp.grad).clone()
        dI0 = wp.to_torch(I0_wp.grad).clone()
        dasc = wp.to_torch(asc_wp.grad).clone()
        ctx.tape.zero()

        if ctx.Iasc_was_flat:
            dI0 = dI0.reshape(-1)
        if ctx.asc_was_flat:
            dasc = dasc.reshape(-1)
        return (dx, dv0, dI0, None, None, None, None, None, None, dasc,
                None, None, None, None, None)


def _neuron_multistep_infer(
    x_seq, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha
):
    """Forward-only neuron multistep (inference). Returns (s_seq, v_seq, v_out, I_out)."""
    T, B = x_seq.shape
    v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    v_out = torch.empty_like(v)
    I_out = torch.empty((B, M), device=v.device, dtype=v.dtype)

    const = [
        _wp(params["v_th"]), _wp(params["v_reset"]), _wp(params["v_rest"]),
        _wp(params["c_m"]), _wp(params["tau"]),
        _wp_vec(params["k"], M, requires_grad=False),
        _wp_vec(params["asc_amps"], M, requires_grad=False), _wp(not_refrac),
    ]
    wp.launch(
        glif3_neuron_multistep_infer_kernel,
        dim=B,
        inputs=[
            _wp(x_seq), _wp(v), _wp_vec(Iasc, M, requires_grad=False), *const,
            float(dt), int(T), int(1 if hard_reset else 0), float(alpha),
            _wp(s_seq), _wp(v_seq), _wp(v_out), _wp_vec(I_out, M, requires_grad=False),
        ],
        block_dim=256,
        device="cuda",
    )
    return s_seq, v_seq, v_out, I_out


def glif3_multistep_fused_warp(
    x_seq: Float[torch.Tensor, " T B"],
    v: Float[torch.Tensor, " B"],
    Iasc: Float[torch.Tensor, " B M"],
    params: dict,
    not_refrac: Float[torch.Tensor, " B"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Neuron-only multistep. Uses a lean forward-only path for inference and
    the Warp-autodiff path (array-carried per-step state) when gradients are
    required."""
    need_grad = torch.is_grad_enabled() and (
        x_seq.requires_grad or v.requires_grad or Iasc.requires_grad
        or params["asc_amps"].requires_grad
    )
    if need_grad:
        return GLIF3NeuronMultiStepWarp.apply(
            x_seq, v, Iasc,
            params["v_th"], params["v_reset"], params["v_rest"],
            params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
            float(dt), int(M), bool(hard_reset), float(alpha),
        )

    if not x_seq.is_cuda:
        raise RuntimeError("Neuron multistep Warp requires CUDA tensors.")
    _ensure_warp_init()
    B = v.numel()
    fp32 = {key: _as_fp32(params[key]) for key in params}
    s_seq, v_seq, v_out, I_out = _neuron_multistep_infer(
        _as_fp32(x_seq), _as_fp32(v), _as_fp32(Iasc),
        fp32, _as_fp32(not_refrac), float(dt), int(M), bool(hard_reset), float(alpha),
    )
    return s_seq, v_seq, v_out, I_out.reshape(-1) if Iasc.ndim == 1 else I_out


# ----------------------------------------------------------------------------
# Dense (neuron + recurrent connection) multistep
# ----------------------------------------------------------------------------


def _dense_multistep_matmul(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha
):
    """cuBLAS matmul for the recurrent term, Warp neuron kernel for the dynamics:
    one launch pair per timestep."""
    T, B = x_seq.shape
    v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    step_v = torch.empty_like(v)
    step_s = torch.empty_like(v)
    s_prev = torch.zeros((B,), device=v.device, dtype=v.dtype)

    hr = int(1 if hard_reset else 0)
    for t in range(T):
        # lin_i = sum_j weight[i, j] * s_prev[j]  (cuBLAS matrix-vector).
        x_in = torch.addmv(x_seq[t] + bias, weight, s_prev).contiguous()
        wp.launch(
            glif3_dense_step_kernel,
            dim=B,
            inputs=[
                _wp(x_in), _wp(v), _wp(Iasc),
                _wp(params["v_th"]), _wp(params["v_reset"]), _wp(params["v_rest"]),
                _wp(params["c_m"]), _wp(params["tau"]),
                _wp(params["k"]), _wp(params["asc_amps"]), _wp(not_refrac),
                _wp(step_s), _wp(step_v),
                int(M), float(dt), hr, float(alpha),
            ],
            block_dim=256,
            device="cuda",
        )
        s_seq[t] = step_s
        v_seq[t] = step_v
        s_prev = step_s.clone()

    return s_seq, v_seq, v.clone(), Iasc.clone()


def glif3_dense_multistep_fused_warp(
    x_seq: Float[torch.Tensor, " T B"],
    weight: torch.Tensor,
    bias: torch.Tensor,
    v: Float[torch.Tensor, " B"],
    Iasc: Float[torch.Tensor, " B M"],
    params: dict,
    not_refrac: Float[torch.Tensor, " B"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
    fused_matmul: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense (neuron + recurrent connection) multistep (inference or training).

    Warp runs the neuron dynamics; the recurrent matmul is delegated to cuBLAS.
    Training composes the single-step Warp autograd primitive with a per-step
    ``torch.mv`` (``dense_multistep_autograd``), so gradients for ``weight`` /
    ``bias`` / ``x`` / initial state / ``asc_amps`` flow through the tape-based
    single step. Inference runs a per-step cuBLAS matmul plus one Warp neuron
    kernel. ``fused_matmul`` is accepted for interface parity but Warp always
    uses this per-step matmul.
    """
    B = v.numel()
    need_grad = torch.is_grad_enabled() and (
        weight.requires_grad or bias.requires_grad or x_seq.requires_grad
        or v.requires_grad or Iasc.requires_grad or params["asc_amps"].requires_grad
    )
    if need_grad:
        return dense_multistep_autograd(
            _glif3_step_warp, x_seq, weight, bias, v, Iasc, params, not_refrac,
            float(dt), int(M), bool(hard_reset), float(alpha),
        )
    if not x_seq.is_cuda:
        raise RuntimeError("Dense multistep Warp requires CUDA tensors.")
    _ensure_warp_init()
    if weight.shape != (B, B):
        raise ValueError("weight must have shape (B, B).")
    if bias.shape != (B,):
        raise ValueError("bias must have shape (B,).")

    x_seq, weight, bias = _as_fp32(x_seq), _as_fp32(weight), _as_fp32(bias)
    v, Iasc = _as_fp32(v), _as_fp32(Iasc.reshape(-1))
    fp32 = {key: _as_fp32(params[key]) for key in params}
    not_refrac = _as_fp32(not_refrac)

    s_seq, v_seq, v_out, I_out = _dense_multistep_matmul(
        x_seq, weight, bias, v, Iasc, fp32, not_refrac,
        float(dt), int(M), bool(hard_reset), float(alpha),
    )
    return s_seq, v_seq, v_out, I_out.view(B, int(M))


class _WarpSparseSpmv(torch.autograd.Function):
    """lin = W @ s via a differentiable Warp kernel on a ``wp.Tape`` — Warp gets
    the adjoints of the values and ``s`` with no hand-written backward. This is
    the Warp-backend replacement for the shared torch ``_SparseSpmv``, so Warp
    sparse training is tape-based end to end (neuron step + SpMV both on tape)."""

    @staticmethod
    def forward(ctx, crow, col, N, val, s):
        _ensure_warp_init()
        lin = torch.empty(N, device=s.device, dtype=s.dtype)
        val_wp, s_wp, lin_wp = _wp_grad(val), _wp_grad(s), _wp_grad(lin)
        tape = wp.Tape()
        with tape:
            wp.launch(glif3_spmv_grad_kernel, dim=N,
                      inputs=[_wp_int(crow), _wp_int(col), val_wp, s_wp, lin_wp],
                      device="cuda")
        ctx.tape = tape
        ctx.wp_arrays = (val_wp, s_wp, lin_wp)
        return wp.to_torch(lin_wp)

    @staticmethod
    def backward(ctx, glin):
        val_wp, s_wp, lin_wp = ctx.wp_arrays
        _seed(lin_wp, glin)
        ctx.tape.backward()
        return (None, None, None,
                wp.to_torch(val_wp.grad).clone(), wp.to_torch(s_wp.grad).clone())


def _warp_sparse_spmv(weight, s):
    return _WarpSparseSpmv.apply(weight.crow, weight.col, weight.N, weight.val, s)


def _sparse_multistep_fused(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha
):
    """Per-step sparse path: a CSR-vector SpMV (a block of ``_TILE_LANES`` threads
    per row, cooperative ``wp.tile_sum`` reduction -> ``lin``) followed by the
    neuron step kernel. Two launches per timestep; does not mutate v / Iasc."""
    T, N = x_seq.shape
    v_work = v.clone()
    I_work = Iasc.clone()
    v_seq = torch.empty((T, N), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, N), device=v.device, dtype=v.dtype)
    step_v = torch.empty_like(v)
    step_s = torch.empty_like(v)
    lin = torch.empty_like(v)
    s_prev = torch.zeros((N,), device=v.device, dtype=v.dtype)

    crow, col, val = _wp_int(weight.crow), _wp_int(weight.col), _wp(weight.val)
    hr = int(1 if hard_reset else 0)
    for t in range(T):
        wp.launch(
            glif3_spmv_tiled_kernel, dim=N * int(_TILE_LANES),
            inputs=[crow, col, val, _wp(s_prev), _wp(lin)],
            block_dim=int(_TILE_LANES), device="cuda",
        )
        x_in = (x_seq[t] + bias + lin).contiguous()
        wp.launch(
            glif3_dense_step_kernel, dim=N,
            inputs=[
                _wp(x_in), _wp(v_work), _wp(I_work),
                _wp(params["v_th"]), _wp(params["v_reset"]), _wp(params["v_rest"]),
                _wp(params["c_m"]), _wp(params["tau"]),
                _wp(params["k"]), _wp(params["asc_amps"]), _wp(not_refrac),
                _wp(step_s), _wp(step_v),
                int(M), float(dt), hr, float(alpha),
            ],
            block_dim=256, device="cuda",
        )
        s_seq[t] = step_s
        v_seq[t] = step_v
        s_prev = step_s.clone()

    return s_seq, v_seq, v_work, I_work


def glif3_sparse_multistep_fused_warp(
    x_seq: Float[torch.Tensor, " T N"],
    weight: SparseWeight,
    bias: torch.Tensor,
    v: Float[torch.Tensor, " N"],
    Iasc: Float[torch.Tensor, " N M"],
    params: dict,
    not_refrac: Float[torch.Tensor, " N"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse (neuron + scale-free recurrent connection) multistep.

    Inference runs a load-balanced CSR-vector SpMV (``_SPMV_LANES`` threads per
    row) then the neuron step kernel, per timestep. Training composes the
    single-step Warp autograd op with a COO ``torch.sparse`` SpMV each step
    (``sparse_multistep_autograd``).
    """
    N = v.numel()
    need_grad = torch.is_grad_enabled() and (
        weight.val.requires_grad or bias.requires_grad or x_seq.requires_grad
        or v.requires_grad or Iasc.requires_grad or params["asc_amps"].requires_grad
    )
    if need_grad:
        return sparse_multistep_autograd(
            _glif3_step_warp, x_seq, weight, bias, v, Iasc, params, not_refrac,
            float(dt), int(M), bool(hard_reset), float(alpha),
            spmv=_warp_sparse_spmv,  # SpMV gradient via wp.Tape (no manual backward)
        )
    if not x_seq.is_cuda:
        raise RuntimeError("Fused sparse multistep Warp requires CUDA tensors.")
    _ensure_warp_init()
    x_seq, bias = _as_fp32(x_seq), _as_fp32(bias)
    v, Iasc = _as_fp32(v), _as_fp32(Iasc.reshape(-1))
    fp32 = {key: _as_fp32(params[key]) for key in params}
    not_refrac = _as_fp32(not_refrac)

    s_seq, v_seq, v_out, I_out = _sparse_multistep_fused(
        x_seq, weight, bias, v, Iasc, fp32, not_refrac,
        float(dt), int(M), bool(hard_reset), float(alpha),
    )
    return s_seq, v_seq, v_out, I_out.view(N, int(M))


glif3_step_warp = GLIF3StepOps(
    step=_glif3_step_warp,
    multistep_fused=glif3_multistep_fused_warp,
    dense_multistep_fused=glif3_dense_multistep_fused_warp,
    sparse_multistep_fused=glif3_sparse_multistep_fused_warp,
)
