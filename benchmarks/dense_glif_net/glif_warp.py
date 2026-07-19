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
- dense multistep ``glif3_dense_multistep_fused_warp`` (inference or training; the
  recurrent matmul is either fused into the single kernel or delegated to
  cuBLAS via ``fused_matmul=False``)

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
    dense_multistep_autograd,
)
# The recurrent tile_matmul kernel lives in its own Warp module so it compiles
# lazily (with the right block_dim) instead of during the main module's backward
# pass. See glif_warp_tiles for why.
from benchmarks.dense_glif_net.glif_warp_tiles import (
    TILE_DIM,
    TILE_THREADS,
    glif3_recur_matmul_kernel,
)


_WARP_INITIALIZED = False
_TILES_AVAILABLE: bool | None = None


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


@wp.kernel(enable_backward=False)
def glif3_dense_multistep_forward_kernel(
    x_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    w: wp.array2d(dtype=wp.float32),      # (B, B)
    b: wp.array(dtype=wp.float32),        # (B,)
    v: wp.array(dtype=wp.float32),        # (B,)
    Iasc: wp.array(dtype=wp.float32),     # (B*M,)
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),
    asc_amps: wp.array(dtype=wp.float32),
    not_refrac: wp.array(dtype=wp.float32),
    s_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v_out: wp.array(dtype=wp.float32),    # (B,)
    I_out: wp.array(dtype=wp.float32),    # (B*M,)
    T: int,
    B: int,
    M: int,
    dt: float,
    hard_reset: int,
    alpha: float,
):
    # Single-thread fully-fused fallback: one thread walks the whole (t, i)
    # recurrence so the recurrent matmul stays fused with no cross-block sync.
    if wp.tid() != 0:
        return
    for t in range(T):
        for i in range(B):
            base = i * M
            i_sum = float(0.0)
            for m in range(M):
                i_sum += Iasc[base + m]

            lin = float(0.0)
            for j in range(B):
                s_prev = float(0.0)
                if t > 0:
                    s_prev = s_seq[t - 1, j]
                lin += w[i, j] * s_prev

            x_in = x_seq[t, i] + b[i] + lin
            v_prime = neuronal_charge(
                v[i], x_in, i_sum, v_rest[i], c_m[i], tau[i], dt)
            spike = neuronal_fire(v_prime, v_th[i], v_reset[i], not_refrac[i], alpha)
            v_post = neuronal_reset(v_prime, spike, v_th[i], v_reset[i], hard_reset)

            for m in range(M):
                Iasc[base + m] = neuronal_adaptation(
                    Iasc[base + m], k[base + m], asc_amps[base + m], spike, dt)
            v[i] = v_post
            s_seq[t, i] = spike
            v_seq[t, i] = v_post

    for i in range(B):
        v_out[i] = v[i]
        base = i * M
        for m in range(M):
            I_out[base + m] = Iasc[base + m]


# ----------------------------------------------------------------------------
# Fused-tile dense kernels: the recurrent matmul is a block-cooperative
# wp.tile_matmul (glif3_recur_matmul_kernel, imported from glif_warp_tiles), and
# the neuron dynamics run one-thread-per-neuron below. They are split into two
# kernels launched per timestep (the launch boundary is the cross-neuron barrier
# the recurrent term needs, and a tiled kernel has no per-thread lane for scalar
# work). Under a single wp.Tape the two-kernel chain differentiates end to end;
# forward-only launches serve inference.
# ----------------------------------------------------------------------------
@wp.kernel
def glif3_dense_dynamics_kernel(
    t: int,
    x_seq: wp.array2d(dtype=wp.float32),    # (T, B)
    lin: wp.array2d(dtype=wp.float32),      # (B, 1) recurrent input this step
    bias: wp.array(dtype=wp.float32),       # (B,)
    v_prev: wp.array(dtype=wp.float32),     # (B,)
    I_prev: wp.array(dtype=Any),            # (B,) ASC vectors
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
    s_out: wp.array2d(dtype=wp.float32),    # (B, 1)
    v_out: wp.array(dtype=wp.float32),      # (B,)
    I_out: wp.array(dtype=Any),             # (B,) ASC vectors
):
    n = wp.tid()
    i_asc = I_prev[n]
    x_in = x_seq[t, n] + bias[n] + lin[n, 0]
    v_prime = neuronal_charge(
        v_prev[n], x_in, asc_sum(i_asc), v_rest[n], c_m[n], tau[n], dt)
    spike = neuronal_fire(v_prime, v_th[n], v_reset[n], not_refrac[n], alpha)
    v_out[n] = neuronal_reset(v_prime, spike, v_th[n], v_reset[n], hard_reset)
    I_out[n] = asc_adapt(i_asc, k[n], asc_amps[n], spike, dt)
    s_out[n, 0] = spike


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
def _tiles_available() -> bool:
    """Whether wp.tile_matmul compiles+runs here (probed once via a tiny launch).
    Tile ops need Warp built against CUDA 12.6.3+ (MathDx); older toolchains
    raise at compile time, so we fall back rather than crash."""
    global _TILES_AVAILABLE
    if _TILES_AVAILABLE is None:
        _ensure_warp_init()
        n = int(TILE_DIM)
        try:
            wp.launch_tiled(
                glif3_recur_matmul_kernel,
                dim=1,
                inputs=[
                    wp.zeros((n, n), dtype=wp.float32, device="cuda"),
                    wp.zeros((n, 1), dtype=wp.float32, device="cuda"),
                    n,
                    wp.zeros((n, 1), dtype=wp.float32, device="cuda"),
                ],
                block_dim=TILE_THREADS,
                device="cuda",
            )
            wp.synchronize()
            _TILES_AVAILABLE = True
        except Exception:
            _TILES_AVAILABLE = False
    return _TILES_AVAILABLE


def _tile_eligible(B: int) -> bool:
    """The fused-tile path needs B to tile evenly and tile ops to be available."""
    return B % int(TILE_DIM) == 0 and _tiles_available()


def _dense_recur_const(params, not_refrac, M):
    """Constant (non-differentiable) neuron params as Warp arrays for the dense
    dynamics kernel."""
    return [
        _wp(params["v_th"]), _wp(params["v_reset"]), _wp(params["v_rest"]),
        _wp(params["c_m"]), _wp(params["tau"]),
        _wp_vec(params["k"], M, requires_grad=False),
        _wp_vec(params["asc_amps"], M, requires_grad=False), _wp(not_refrac),
    ]


def _dense_multistep_tile_infer(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha
):
    """Forward-only fused-tile dense multistep: per step, one tile_matmul kernel
    (W @ s_prev) then one neuron-dynamics kernel. State ping-pongs between two
    buffers; the per-step launch boundary is the recurrent cross-neuron barrier."""
    T, B = x_seq.shape
    dev, dtp = v.device, v.dtype
    s_seq = torch.empty((T, B), device=dev, dtype=dtp)
    v_seq = torch.empty((T, B), device=dev, dtype=dtp)

    x_wp, W_wp, bias_wp = _wp(x_seq), _wp(weight), _wp(bias)
    const = _dense_recur_const(params, not_refrac, M)
    hr = int(1 if hard_reset else 0)

    s_prev = torch.zeros((B, 1), device=dev, dtype=dtp)
    lin = torch.empty((B, 1), device=dev, dtype=dtp)
    v_cur, I_cur = v.clone(), Iasc.reshape(B, M).clone()
    s_nxt = torch.empty((B, 1), device=dev, dtype=dtp)
    v_nxt = torch.empty((B,), device=dev, dtype=dtp)
    I_nxt = torch.empty((B, M), device=dev, dtype=dtp)

    for t in range(T):
        wp.launch_tiled(
            glif3_recur_matmul_kernel, dim=B // int(TILE_DIM),
            inputs=[W_wp, _wp(s_prev), int(B), _wp(lin)],
            block_dim=TILE_THREADS, device="cuda",
        )
        wp.launch(
            glif3_dense_dynamics_kernel, dim=B,
            inputs=[
                int(t), x_wp, _wp(lin), bias_wp,
                _wp(v_cur), _wp_vec(I_cur, M, requires_grad=False), *const,
                float(dt), hr, float(alpha),
                _wp(s_nxt), _wp(v_nxt), _wp_vec(I_nxt, M, requires_grad=False),
            ],
            device="cuda",
        )
        s_seq[t], v_seq[t] = s_nxt[:, 0], v_nxt
        s_prev, s_nxt = s_nxt, s_prev
        v_cur, v_nxt = v_nxt, v_cur
        I_cur, I_nxt = I_nxt, I_cur
    return s_seq, v_seq, v_cur, I_cur


def _dense_multistep_singlethread(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha
):
    """Fully-fused fallback for when tiles are unavailable or B does not tile
    evenly: one thread walks the whole (t, i) recurrence in a single launch."""
    T, B = x_seq.shape
    v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    v_out = torch.empty_like(v)
    I_out = torch.empty_like(Iasc)
    wp.launch(
        glif3_dense_multistep_forward_kernel, dim=1,
        inputs=[
            _wp(x_seq), _wp(weight), _wp(bias), _wp(v), _wp(Iasc),
            _wp(params["v_th"]), _wp(params["v_reset"]), _wp(params["v_rest"]),
            _wp(params["c_m"]), _wp(params["tau"]),
            _wp(params["k"]), _wp(params["asc_amps"]), _wp(not_refrac),
            _wp(s_seq), _wp(v_seq), _wp(v_out), _wp(I_out),
            int(T), int(B), int(M), float(dt),
            int(1 if hard_reset else 0), float(alpha),
        ],
        block_dim=1, device="cuda",
    )
    return s_seq, v_seq, v_out, I_out


class GLIF3DenseMultiStepWarp(torch.autograd.Function):
    """Fused-tile dense multistep training. Runs the two-kernel (tile_matmul +
    dynamics) recurrence for T steps under one wp.Tape, with per-step state kept
    in arrays so the dynamic loop differentiates. Gradients for weight / bias /
    x_seq / initial state / asc_amps all fall out of tape.backward()."""

    @staticmethod
    def forward(
        ctx, x_seq, weight, bias, v, Iasc,
        v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
        dt, M, hard_reset, alpha,
    ):
        if not x_seq.is_cuda:
            raise RuntimeError("Dense multistep Warp requires CUDA tensors.")
        _ensure_warp_init()
        M = int(M)
        x_seq, weight, bias = _as_fp32(x_seq), _as_fp32(weight), _as_fp32(bias)
        v = _as_fp32(v)
        v_th, v_reset, v_rest = _as_fp32(v_th), _as_fp32(v_reset), _as_fp32(v_rest)
        c_m, tau, k = _as_fp32(c_m), _as_fp32(tau), _as_fp32(k)
        asc_amps, not_refrac = _as_fp32(asc_amps), _as_fp32(not_refrac)
        T, B = x_seq.shape
        dev, dtp = v.device, v.dtype

        x_wp, W_wp, b_wp = _wp_grad(x_seq), _wp_grad(weight), _wp_grad(bias)
        asc_wp = _wp_vec(asc_amps, M, requires_grad=True)
        const = [
            _wp(v_th), _wp(v_reset), _wp(v_rest), _wp(c_m), _wp(tau),
            _wp_vec(k, M, requires_grad=False), asc_wp, _wp(not_refrac),
        ]

        # Per-step state in arrays (array-carried loop => differentiable).
        s_list = [_wp_grad(torch.zeros((B, 1), device=dev, dtype=dtp))]
        v_list = [_wp_grad(v)]
        I_list = [_wp_vec(Iasc, M, requires_grad=True)]
        lin_list, s_torch, v_torch = [], [], []
        hr = int(1 if hard_reset else 0)

        tape = wp.Tape()
        with tape:
            for t in range(T):
                lin_wp = _wp_grad(torch.empty((B, 1), device=dev, dtype=dtp))
                lin_list.append(lin_wp)
                wp.launch_tiled(
                    glif3_recur_matmul_kernel, dim=B // int(TILE_DIM),
                    inputs=[W_wp, s_list[t], int(B), lin_wp],
                    block_dim=TILE_THREADS, device="cuda",
                )
                s_t = torch.empty((B, 1), device=dev, dtype=dtp)
                v_t = torch.empty((B,), device=dev, dtype=dtp)
                I_t = torch.empty((B, M), device=dev, dtype=dtp)
                s_wp, v_wp = _wp_grad(s_t), _wp_grad(v_t)
                I_wp = _wp_vec(I_t, M, requires_grad=True)
                wp.launch(
                    glif3_dense_dynamics_kernel, dim=B,
                    inputs=[
                        int(t), x_wp, lin_wp, b_wp, v_list[t], I_list[t], *const,
                        float(dt), hr, float(alpha), s_wp, v_wp, I_wp,
                    ],
                    device="cuda",
                )
                s_list.append(s_wp); v_list.append(v_wp); I_list.append(I_wp)
                s_torch.append(s_t); v_torch.append(v_t)

        ctx.tape = tape
        ctx.grad_inputs = (x_wp, W_wp, b_wp, v_list[0], I_list[0], asc_wp)
        ctx.keep = (s_list, v_list, I_list, lin_list, const)  # keep arrays alive
        ctx.T, ctx.B, ctx.M = T, B, M
        ctx.Iasc_was_flat = Iasc.ndim == 1
        ctx.asc_was_flat = asc_amps.ndim == 1

        s_seq = torch.stack([s.view(B) for s in s_torch])
        v_seq = torch.stack(v_torch)
        v_out = v_torch[-1]
        I_out = wp.to_torch(I_list[T]).clone()
        return s_seq, v_seq, v_out, I_out.reshape(-1) if Iasc.ndim == 1 else I_out

    @staticmethod
    def backward(ctx, ds_seq, dv_seq, dv_out, dI_out):
        x_wp, W_wp, b_wp, v0_wp, I0_wp, asc_wp = ctx.grad_inputs
        s_list, v_list, I_list, _lin, _const = ctx.keep
        T, B, M = ctx.T, ctx.B, ctx.M

        for t in range(1, T + 1):
            _seed(s_list[t], ds_seq[t - 1] if ds_seq is not None else None)
            _seed(v_list[t], dv_seq[t - 1] if dv_seq is not None else None)
        v_grad = wp.to_torch(v_list[T].grad)
        if dv_out is not None:
            v_grad += dv_out.to(dtype=v_grad.dtype)
        i_grad = wp.to_torch(I_list[T].grad)
        if dI_out is not None:
            i_grad += dI_out.reshape(B, M).to(dtype=i_grad.dtype)
        ctx.tape.backward()

        dx = wp.to_torch(x_wp.grad).clone()
        dW = wp.to_torch(W_wp.grad).clone()
        db = wp.to_torch(b_wp.grad).clone()
        dv0 = wp.to_torch(v0_wp.grad).clone()
        dI0 = wp.to_torch(I0_wp.grad).clone()
        dasc = wp.to_torch(asc_wp.grad).clone()

        if ctx.Iasc_was_flat:
            dI0 = dI0.reshape(-1)
        if ctx.asc_was_flat:
            dasc = dasc.reshape(-1)
        return (dx, dW, db, dv0, dI0, None, None, None, None, None, None, dasc,
                None, None, None, None, None)


def _dense_multistep_tile_autograd(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha
):
    return GLIF3DenseMultiStepWarp.apply(
        x_seq, weight, bias, v, Iasc,
        params["v_th"], params["v_reset"], params["v_rest"],
        params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
        float(dt), int(M), bool(hard_reset), float(alpha),
    )


def _dense_multistep_matmul(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha
):
    """Non-fused path: cuBLAS matmul for the recurrent term, neuron step kernel
    for the dynamics. One launch pair per timestep."""
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

    With ``fused_matmul=True`` the recurrent matmul is a Warp ``tile_matmul``
    kernel fused (per step) with the neuron dynamics -- for training the whole
    T-step recurrence runs under one ``wp.Tape``. With ``fused_matmul=False`` the
    matmul is delegated to cuBLAS: training composes the single-step Warp
    autograd primitive per step (``dense_multistep_autograd``); inference runs a
    per-step cuBLAS matmul plus neuron kernel. When B does not tile evenly or
    tile ops are unavailable, the fused path degrades gracefully (composition for
    training, single-thread fused kernel for inference).
    """
    B = v.numel()
    need_grad = torch.is_grad_enabled() and (
        weight.requires_grad or bias.requires_grad or x_seq.requires_grad
        or v.requires_grad or Iasc.requires_grad or params["asc_amps"].requires_grad
    )
    if need_grad:
        if fused_matmul and x_seq.is_cuda and _tile_eligible(B):
            return _dense_multistep_tile_autograd(
                x_seq, weight, bias, v, Iasc, params, not_refrac,
                float(dt), int(M), bool(hard_reset), float(alpha),
            )
        return dense_multistep_autograd(
            _glif3_step_warp, x_seq, weight, bias, v, Iasc, params, not_refrac,
            float(dt), int(M), bool(hard_reset), float(alpha),
        )
    if not x_seq.is_cuda:
        raise RuntimeError("Fused dense multistep Warp requires CUDA tensors.")
    _ensure_warp_init()
    if weight.shape != (B, B):
        raise ValueError("weight must have shape (B, B).")
    if bias.shape != (B,):
        raise ValueError("bias must have shape (B,).")

    x_seq, weight, bias = _as_fp32(x_seq), _as_fp32(weight), _as_fp32(bias)
    v, Iasc = _as_fp32(v), _as_fp32(Iasc.reshape(-1))
    fp32 = {key: _as_fp32(params[key]) for key in params}
    not_refrac = _as_fp32(not_refrac)

    if not fused_matmul:
        path = _dense_multistep_matmul
    elif _tile_eligible(B):
        path = _dense_multistep_tile_infer
    else:
        path = _dense_multistep_singlethread
    s_seq, v_seq, v_out, I_out = path(
        x_seq, weight, bias, v, Iasc, fp32, not_refrac,
        float(dt), int(M), bool(hard_reset), float(alpha),
    )
    return s_seq, v_seq, v_out, I_out.view(B, int(M))


glif3_step_warp = GLIF3StepOps(
    step=_glif3_step_warp,
    multistep_fused=glif3_multistep_fused_warp,
    dense_multistep_fused=glif3_dense_multistep_fused_warp,
)


class GLIF3Warp(torch.nn.Module):
    """Thin module wrapper exposing a ``step`` API."""

    def __init__(self, M: int, hard_reset: bool = False, alpha: float = 2.0):
        super().__init__()
        self.M = int(M)
        self.hard_reset = bool(hard_reset)
        self.alpha = float(alpha)

    def step(
        self,
        v: Float[torch.Tensor, " B"],
        Iasc: Float[torch.Tensor, " B M"],
        x: Float[torch.Tensor, " B"],
        params: dict,
        not_refrac: Float[torch.Tensor, " B"],
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _glif3_step_warp(
            v=v, Iasc=Iasc, x=x, params=params, not_refrac=not_refrac,
            dt=float(dt), M=self.M, hard_reset=self.hard_reset, alpha=self.alpha,
        )
