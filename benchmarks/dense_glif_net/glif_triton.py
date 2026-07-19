"""Triton GLIF3 kernels: single-step, neuron multistep, and dense (neuron +
recurrent connection) multistep.

The GLIF3 update per step is

    v_inf   = v_rest + tau * (x + sum I_asc) / c_m
    v'      = v_inf + (v - v_inf) * exp(-dt / tau)
    spike   = heaviside(v' - v_th) * not_refrac
    v_post  = v' - (v' - v_reset) * spike   (hard reset)
            = v' - (v_th - v_reset) * spike (soft reset)
    I_asc'  = I_asc * exp(-k dt) + asc_amps * spike

The device dynamics are factored into ``@triton.jit`` helpers
(``neuronal_charge`` / ``neuronal_fire`` / ``neuronal_reset`` /
``neuronal_adaptation`` / ``surrogate_grad``) mirroring the eager
``btorch.models.neurons.glif.GLIF3`` methods, so each kernel reads as a
composition of those steps.

Organization:
- single-step ``glif3_step_triton`` (autograd, the training primitive)
- neuron multistep ``glif3_multistep_fused_triton`` (autograd for training,
  a lean forward-only path for inference)
- dense multistep ``glif3_dense_multistep_fused_triton`` (inference or training;
  the recurrent matmul is fused into the neuron kernel via a pipelined in-kernel
  gemv, or delegated to cuBLAS via ``fused_matmul=False``)

Implementation notes:
- memory is addressed with ``tl.make_block_ptr`` + ``boundary_check`` via the
  ``load_*`` / ``store_*`` helpers. The ASC mode axis is loaded as a power-of-two
  ``(BLOCK, MPAD)`` tile from the real ``(B, M)`` tensor (Triton block shapes
  must be powers of two); ``boundary_check`` zero-pads the overhang. Padded modes
  hold zero ASC state (contributing nothing to ``i_sum``) and are refilled with a
  nonzero rate ``k`` at the load site so ``exp_euler`` never divides by zero.
- surrogate gradient is btorch's exact ``ATan`` (damping = 1).

Conventions (flattened, contiguous, fp32 on CUDA):
- ``v``, ``x``, ``not_refrac``: shape ``(B,)``
- ``Iasc``, ``k``, ``asc_amps``: shape ``(B*M,)`` with base ``i*M``
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from jaxtyping import Float
from triton.language.extra import libdevice

from benchmarks.dense_glif_net.glif_common import (
    GLIF3StepOps,
    SparseWeight,
    dense_multistep_autograd,
    sparse_multistep_autograd,
)


# Use libdevice for exp/expm1: Triton has no native expm1, and its native
# tl.exp is a low-precision ex2.approx. The stiff tau/c_m ~ 400 dynamics amplify
# that error past the 1e-4 gradient tolerance (verified: swapping in tl.exp
# fails the multistep tests), so accurate libdevice math is required.
@triton.jit
def precise_exp(x):
    """Accurate exp (libdevice)."""
    return libdevice.exp(x)


@triton.jit
def precise_expm1(x):
    """Accurate exp(x) - 1 (libdevice)."""
    return libdevice.expm1(x)


# ----------------------------------------------------------------------------
# Device helpers: one @triton.jit function per step of the GLIF3 update.
# All operate elementwise on Triton tensors (broadcasting the mode axis).
# ----------------------------------------------------------------------------
@triton.jit
def exp_euler(x, derivative, linear, dt):
    """One exponential-Euler step of dx/dt = A*x + B, given the derivative f(x)
    at x and the linear term A: x_next = x + expm1(dt*A)/A * f(x). Mirrors
    btorch.models.ode.exp_euler_step. Callers keep A != 0 (padded ASC lanes are
    loaded with a nonzero rate) so no zero-division guard is needed."""
    return x + precise_expm1(dt * linear) / linear * derivative


@triton.jit
def neuronal_charge(v, x, i_sum, v_rest, c_m, tau, dt):
    """Exponential-Euler membrane update -> v' (pre-spike voltage):
    dv/dt = -(v - v_rest)/tau + (x + sum I_asc)/c_m, linear term A = -1/tau."""
    dv = -(v - v_rest) / tau + (x + i_sum) / c_m
    return exp_euler(v, dv, -1.0 / tau, dt)


@triton.jit
def neuronal_fire(v_prime, v_th, mask):
    """Hard-threshold spike (heaviside), gated by the refractory mask."""
    return (v_prime >= v_th).to(tl.float32) * mask


@triton.jit
def neuronal_reset(v_prime, spike, v_th, v_reset, hard_reset: tl.constexpr):
    """Membrane reset after a spike (hard or soft)."""
    if hard_reset:
        return v_prime - (v_prime - v_reset) * spike
    return v_prime - (v_th - v_reset) * spike


@triton.jit
def neuronal_adaptation(i_asc, k, asc_amp, spike, dt):
    """After-spike current: exp-Euler decay (dI/dt = -k*I) + spike jump (per mode)."""
    return exp_euler(i_asc, -k * i_asc, -k, dt) + asc_amp * spike


@triton.jit
def surrogate_grad(v_prime, v_th, v_reset, mask, alpha):
    """Surrogate gradient d spike / d v' (btorch ATan: 1 / (1 + (alpha*u)^2),
    damping = 1, with u = (v' - v_th) / (v_th - v_reset))."""
    denom = v_th - v_reset
    u = (v_prime - v_th) / denom
    ds_du = mask / (1.0 + (alpha * u) * (alpha * u))
    return ds_du / denom


# ----------------------------------------------------------------------------
# Block-pointer load/store helpers. Every access uses tl.make_block_ptr with the
# real logical shape and boundary_check, so the (B, M) mode axis is loaded as a
# power-of-two (BLOCK, MPAD) tile with the overhang zero-padded. Row helpers pull
# a single time step out of a (T, B[, M]) sequence; the size-1 time dim is
# reshaped away (Triton block loads keep it). Padding is always zero, which is
# correct for every field except the ASC rate k (see the load-site fixups).
# ----------------------------------------------------------------------------
@triton.jit
def load_vec(base, n, off, BLOCK: tl.constexpr):
    """Load a (BLOCK,) tile of a length-n vector at element offset off."""
    ptr = tl.make_block_ptr(base, shape=(n,), strides=(1,), offsets=(off,),
                            block_shape=(BLOCK,), order=(0,))
    return tl.load(ptr, boundary_check=(0,), padding_option="zero")


@triton.jit
def store_vec(base, n, off, value, BLOCK: tl.constexpr):
    ptr = tl.make_block_ptr(base, shape=(n,), strides=(1,), offsets=(off,),
                            block_shape=(BLOCK,), order=(0,))
    tl.store(ptr, value, boundary_check=(0,))


@triton.jit
def load_modes(base, B, M: tl.constexpr, off, BLOCK: tl.constexpr, MPAD: tl.constexpr):
    """Load a (BLOCK, MPAD) tile of a (B, M) tensor; padded modes read zero."""
    ptr = tl.make_block_ptr(base, shape=(B, M), strides=(M, 1), offsets=(off, 0),
                            block_shape=(BLOCK, MPAD), order=(1, 0))
    return tl.load(ptr, boundary_check=(0, 1), padding_option="zero")


@triton.jit
def store_modes(base, B, M: tl.constexpr, off, value, BLOCK: tl.constexpr,
                MPAD: tl.constexpr):
    ptr = tl.make_block_ptr(base, shape=(B, M), strides=(M, 1), offsets=(off, 0),
                            block_shape=(BLOCK, MPAD), order=(1, 0))
    tl.store(ptr, value, boundary_check=(0, 1))


@triton.jit
def load_row(base, T, B, t, off, BLOCK: tl.constexpr):
    """Load time step t (a (BLOCK,) row) of a (T, B) sequence."""
    ptr = tl.make_block_ptr(base, shape=(T, B), strides=(B, 1), offsets=(t, off),
                            block_shape=(1, BLOCK), order=(1, 0))
    row = tl.load(ptr, boundary_check=(1,), padding_option="zero")
    return tl.reshape(row, (BLOCK,))


@triton.jit
def store_row(base, T, B, t, off, value, BLOCK: tl.constexpr):
    ptr = tl.make_block_ptr(base, shape=(T, B), strides=(B, 1), offsets=(t, off),
                            block_shape=(1, BLOCK), order=(1, 0))
    tl.store(ptr, tl.reshape(value, (1, BLOCK)), boundary_check=(1,))


@triton.jit
def load_row_modes(base, T, B, M: tl.constexpr, t, off, BLOCK: tl.constexpr,
                   MPAD: tl.constexpr):
    """Load time step t (a (BLOCK, MPAD) tile) of a (T, B, M) sequence."""
    ptr = tl.make_block_ptr(base, shape=(T, B, M), strides=(B * M, M, 1),
                            offsets=(t, off, 0), block_shape=(1, BLOCK, MPAD),
                            order=(2, 1, 0))
    tile = tl.load(ptr, boundary_check=(1, 2), padding_option="zero")
    return tl.reshape(tile, (BLOCK, MPAD))


@triton.jit
def store_row_modes(base, T, B, M: tl.constexpr, t, off, value, BLOCK: tl.constexpr,
                    MPAD: tl.constexpr):
    ptr = tl.make_block_ptr(base, shape=(T, B, M), strides=(B * M, M, 1),
                            offsets=(t, off, 0), block_shape=(1, BLOCK, MPAD),
                            order=(2, 1, 0))
    tl.store(ptr, tl.reshape(value, (1, BLOCK, MPAD)), boundary_check=(1, 2))


# ----------------------------------------------------------------------------
# Single-step kernels
# ----------------------------------------------------------------------------
@triton.jit
def glif3_step_forward_kernel(
    v_ptr, I_ptr, x_ptr,
    v_th_ptr, v_reset_ptr, v_rest_ptr, c_m_ptr, tau_ptr, k_ptr, asc_ptr, mask_ptr,
    v_out_ptr, I_out_ptr, s_out_ptr,
    B, dt, alpha,
    M: tl.constexpr, MPAD: tl.constexpr, hard_reset: tl.constexpr, BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK
    mode_valid = tl.arange(0, MPAD) < M

    v = load_vec(v_ptr, B, off, BLOCK)
    x = load_vec(x_ptr, B, off, BLOCK)
    v_th = load_vec(v_th_ptr, B, off, BLOCK)
    v_reset = load_vec(v_reset_ptr, B, off, BLOCK)
    v_rest = load_vec(v_rest_ptr, B, off, BLOCK)
    c_m = load_vec(c_m_ptr, B, off, BLOCK)
    tau = load_vec(tau_ptr, B, off, BLOCK)
    mask = load_vec(mask_ptr, B, off, BLOCK)
    iasc = load_modes(I_ptr, B, M, off, BLOCK, MPAD)
    # Padded modes load rate 0; refill with a nonzero rate so exp_euler's 1/A
    # term stays finite (the padded ASC state is 0, so the update is still 0).
    k = tl.where(mode_valid[None, :], load_modes(k_ptr, B, M, off, BLOCK, MPAD), 1.0)
    asc = load_modes(asc_ptr, B, M, off, BLOCK, MPAD)

    v_prime = neuronal_charge(v, x, tl.sum(iasc, axis=1), v_rest, c_m, tau, dt)
    spike = neuronal_fire(v_prime, v_th, mask)
    v_post = neuronal_reset(v_prime, spike, v_th, v_reset, hard_reset)
    I_post = neuronal_adaptation(iasc, k, asc, spike[:, None], dt)

    store_modes(I_out_ptr, B, M, off, I_post, BLOCK, MPAD)
    store_vec(v_out_ptr, B, off, v_post, BLOCK)
    store_vec(s_out_ptr, B, off, spike, BLOCK)


@triton.jit
def glif3_step_backward_kernel(
    v_ptr, I_ptr, x_ptr,
    v_th_ptr, v_reset_ptr, v_rest_ptr, c_m_ptr, tau_ptr, k_ptr, asc_ptr, mask_ptr,
    s_ptr, dv_out_ptr, dI_out_ptr, ds_out_ptr,
    dv_ptr, dI_ptr, dx_ptr, dasc_ptr,
    B, dt, alpha,
    M: tl.constexpr, MPAD: tl.constexpr, hard_reset: tl.constexpr, BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK
    mode_valid = tl.arange(0, MPAD) < M

    v = load_vec(v_ptr, B, off, BLOCK)
    x = load_vec(x_ptr, B, off, BLOCK)
    v_th = load_vec(v_th_ptr, B, off, BLOCK)
    v_reset = load_vec(v_reset_ptr, B, off, BLOCK)
    v_rest = load_vec(v_rest_ptr, B, off, BLOCK)
    c_m = load_vec(c_m_ptr, B, off, BLOCK)
    tau = load_vec(tau_ptr, B, off, BLOCK)
    mask = load_vec(mask_ptr, B, off, BLOCK)
    iasc = load_modes(I_ptr, B, M, off, BLOCK, MPAD)
    k = tl.where(mode_valid[None, :], load_modes(k_ptr, B, M, off, BLOCK, MPAD), 1.0)
    asc = load_modes(asc_ptr, B, M, off, BLOCK, MPAD)

    spike = load_vec(s_ptr, B, off, BLOCK)
    dv_post = load_vec(dv_out_ptr, B, off, BLOCK)
    dI_post = load_modes(dI_out_ptr, B, M, off, BLOCK, MPAD)
    ds_out = load_vec(ds_out_ptr, B, off, BLOCK)

    decay = precise_exp(-dt / tau)
    v_prime = neuronal_charge(v, x, tl.sum(iasc, axis=1), v_rest, c_m, tau, dt)
    ds_dvprime = surrogate_grad(v_prime, v_th, v_reset, mask, alpha)

    if hard_reset:
        dvprime = dv_post * (1.0 - spike)
        ds_from_v = dv_post * (-(v_prime - v_reset))
    else:
        dvprime = dv_post
        ds_from_v = dv_post * (-(v_th - v_reset))

    dI_s_sum = tl.sum(dI_post * asc, axis=1)
    dasc = dI_post * spike[:, None]
    ds_total = ds_out + ds_from_v + dI_s_sum
    dvprime = dvprime + ds_total * ds_dvprime

    dv_inf = dvprime * (1.0 - decay)
    dI_common = dv_inf * (tau / c_m)
    dI = dI_post * precise_exp(-k * dt) + dI_common[:, None]

    store_vec(dv_ptr, B, off, dvprime * decay, BLOCK)
    store_vec(dx_ptr, B, off, dI_common, BLOCK)
    store_modes(dI_ptr, B, M, off, dI, BLOCK, MPAD)
    store_modes(dasc_ptr, B, M, off, dasc, BLOCK, MPAD)


# ----------------------------------------------------------------------------
# Neuron multistep kernels
# ----------------------------------------------------------------------------
@triton.jit
def glif3_neuron_multistep_forward_kernel(
    x_ptr,  # (T*B,)
    v_ptr, I_ptr,
    v_th_ptr, v_reset_ptr, v_rest_ptr, c_m_ptr, tau_ptr, k_ptr, asc_ptr, mask_ptr,
    s_seq_ptr, v_seq_ptr, v_out_ptr, I_out_ptr, I_seq_ptr,
    B, dt, alpha, T,
    M: tl.constexpr, MPAD: tl.constexpr, hard_reset: tl.constexpr,
    save_iseq: tl.constexpr, BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK
    mode_valid = tl.arange(0, MPAD) < M

    v_th = load_vec(v_th_ptr, B, off, BLOCK)
    v_reset = load_vec(v_reset_ptr, B, off, BLOCK)
    v_rest = load_vec(v_rest_ptr, B, off, BLOCK)
    c_m = load_vec(c_m_ptr, B, off, BLOCK)
    tau = load_vec(tau_ptr, B, off, BLOCK)
    mask = load_vec(mask_ptr, B, off, BLOCK)
    k = tl.where(mode_valid[None, :], load_modes(k_ptr, B, M, off, BLOCK, MPAD), 1.0)
    asc = load_modes(asc_ptr, B, M, off, BLOCK, MPAD)

    v = load_vec(v_ptr, B, off, BLOCK)
    iasc = load_modes(I_ptr, B, M, off, BLOCK, MPAD)

    for t in tl.range(0, T):
        x = load_row(x_ptr, T, B, t, off, BLOCK)
        # Save the pre-step I_asc so backward reads exact values (training only).
        if save_iseq:
            store_row_modes(I_seq_ptr, T, B, M, t, off, iasc, BLOCK, MPAD)
        v_prime = neuronal_charge(v, x, tl.sum(iasc, axis=1), v_rest, c_m, tau, dt)
        spike = neuronal_fire(v_prime, v_th, mask)
        v = neuronal_reset(v_prime, spike, v_th, v_reset, hard_reset)
        iasc = neuronal_adaptation(iasc, k, asc, spike[:, None], dt)
        store_row(s_seq_ptr, T, B, t, off, spike, BLOCK)
        store_row(v_seq_ptr, T, B, t, off, v, BLOCK)

    store_vec(v_out_ptr, B, off, v, BLOCK)
    store_modes(I_out_ptr, B, M, off, iasc, BLOCK, MPAD)


@triton.jit
def glif3_neuron_multistep_backward_kernel(
    x_ptr, v0_ptr, v_seq_ptr, s_seq_ptr, I_seq_ptr,
    v_th_ptr, v_reset_ptr, v_rest_ptr, c_m_ptr, tau_ptr, k_ptr, asc_ptr, mask_ptr,
    ds_seq_ptr, dv_seq_ptr, dv_out_ptr, dI_out_ptr,
    dv0_ptr, dI0_ptr, dx_ptr, dasc_ptr,
    B, dt, alpha, T,
    M: tl.constexpr, MPAD: tl.constexpr, hard_reset: tl.constexpr, BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK
    mode_valid = tl.arange(0, MPAD) < M

    v_th = load_vec(v_th_ptr, B, off, BLOCK)
    v_reset = load_vec(v_reset_ptr, B, off, BLOCK)
    v_rest = load_vec(v_rest_ptr, B, off, BLOCK)
    c_m = load_vec(c_m_ptr, B, off, BLOCK)
    tau = load_vec(tau_ptr, B, off, BLOCK)
    mask = load_vec(mask_ptr, B, off, BLOCK)
    k = tl.where(mode_valid[None, :], load_modes(k_ptr, B, M, off, BLOCK, MPAD), 1.0)
    asc = load_modes(asc_ptr, B, M, off, BLOCK, MPAD)
    v0 = load_vec(v0_ptr, B, off, BLOCK)

    decay = precise_exp(-dt / tau)
    b = precise_exp(-k * dt)
    tau_over_c = tau / c_m

    # Seed reverse-time state with the upstream gradient on the final I_asc.
    dI = load_modes(dI_out_ptr, B, M, off, BLOCK, MPAD)
    dv_post = load_vec(dv_out_ptr, B, off, BLOCK)
    dasc = tl.zeros([BLOCK, MPAD], dtype=tl.float32)

    for ti in tl.range(0, T):
        t = T - 1 - ti
        dv_post = dv_post + load_row(dv_seq_ptr, T, B, t, off, BLOCK)
        spike = load_row(s_seq_ptr, T, B, t, off, BLOCK)
        ds_t = load_row(ds_seq_ptr, T, B, t, off, BLOCK)
        x = load_row(x_ptr, T, B, t, off, BLOCK)

        prev = tl.maximum(t - 1, 0)
        v_pre = tl.where(t == 0, v0, load_row(v_seq_ptr, T, B, prev, off, BLOCK))

        # Exact pre-step I_asc(t) saved by the forward pass.
        I_t = load_row_modes(I_seq_ptr, T, B, M, t, off, BLOCK, MPAD)
        v_prime = neuronal_charge(
            v_pre, x, tl.sum(I_t, axis=1), v_rest, c_m, tau, dt)
        ds_dvprime = surrogate_grad(v_prime, v_th, v_reset, mask, alpha)

        if hard_reset:
            dvprime = dv_post * (1.0 - spike)
            ds_from_v = dv_post * (-(v_prime - v_reset))
        else:
            dvprime = dv_post
            ds_from_v = dv_post * (-(v_th - v_reset))

        dI_s_sum = tl.sum(dI * asc, axis=1)
        ds_total = ds_t + ds_from_v + dI_s_sum
        dvprime = dvprime + ds_total * ds_dvprime

        dv_inf = dvprime * (1.0 - decay)
        dI_common = dv_inf * tau_over_c
        store_row(dx_ptr, T, B, t, off, dI_common, BLOCK)

        dasc = dasc + dI * spike[:, None]
        dI = dI * b + dI_common[:, None]
        dv_post = dvprime * decay

    store_vec(dv0_ptr, B, off, dv_post, BLOCK)
    store_modes(dI0_ptr, B, M, off, dI, BLOCK, MPAD)
    store_modes(dasc_ptr, B, M, off, dasc, BLOCK, MPAD)


# ----------------------------------------------------------------------------
# Dense multistep kernels (inference)
# ----------------------------------------------------------------------------
@triton.jit
def glif3_neuron_update(x_in, v, iasc, k, asc, v_th, v_reset, v_rest, c_m, tau,
                        mask, dt, hard_reset: tl.constexpr):
    """GLIF3 neuron update for a tile: from total input ``x_in`` and previous
    state ``(v, iasc)`` return ``(spike, v_post, I_post)``. ``iasc`` / ``k`` /
    ``asc`` are ``(BLOCK, MPAD)`` mode tiles; the rest are ``(BLOCK,)``."""
    v_prime = neuronal_charge(v, x_in, tl.sum(iasc, axis=1), v_rest, c_m, tau, dt)
    spike = neuronal_fire(v_prime, v_th, mask)
    v_post = neuronal_reset(v_prime, spike, v_th, v_reset, hard_reset)
    I_post = neuronal_adaptation(iasc, k, asc, spike[:, None], dt)
    return spike, v_post, I_post


@triton.jit
def glif3_dense_step_kernel(
    x_in_ptr,  # (B,) already includes bias + recurrent term
    v_ptr, I_ptr,
    v_th_ptr, v_reset_ptr, v_rest_ptr, c_m_ptr, tau_ptr, k_ptr, asc_ptr, mask_ptr,
    s_out_ptr, v_out_ptr,
    B, dt, alpha,
    M: tl.constexpr, MPAD: tl.constexpr, hard_reset: tl.constexpr, BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK
    mode_valid = tl.arange(0, MPAD) < M

    x = load_vec(x_in_ptr, B, off, BLOCK)
    v = load_vec(v_ptr, B, off, BLOCK)
    v_th = load_vec(v_th_ptr, B, off, BLOCK)
    v_reset = load_vec(v_reset_ptr, B, off, BLOCK)
    v_rest = load_vec(v_rest_ptr, B, off, BLOCK)
    c_m = load_vec(c_m_ptr, B, off, BLOCK)
    tau = load_vec(tau_ptr, B, off, BLOCK)
    mask = load_vec(mask_ptr, B, off, BLOCK)
    iasc = load_modes(I_ptr, B, M, off, BLOCK, MPAD)
    k = tl.where(mode_valid[None, :], load_modes(k_ptr, B, M, off, BLOCK, MPAD), 1.0)
    asc = load_modes(asc_ptr, B, M, off, BLOCK, MPAD)

    spike, v_post, I_post = glif3_neuron_update(
        x, v, iasc, k, asc, v_th, v_reset, v_rest, c_m, tau, mask, dt, hard_reset)

    store_vec(v_ptr, B, off, v_post, BLOCK)
    store_modes(I_ptr, B, M, off, I_post, BLOCK, MPAD)
    store_vec(s_out_ptr, B, off, spike, BLOCK)
    store_vec(v_out_ptr, B, off, v_post, BLOCK)


# Fused single-step dense kernel: the recurrent gemv ``lin = W[rows,:] @ s_prev``
# is computed in-kernel with a pipelined column loop (``tl.range`` num_stages
# overlaps the weight-tile streaming with the accumulate) and fused with the
# neuron update. One launch per timestep; the launch boundary is the cross-step
# barrier (every block reads the whole previous-step spike vector).
_DENSE_STEP_CONFIGS = [
    triton.Config({"BM": bm, "BK": bk, "NUM_STAGES": ns}, num_warps=nw)
    for bm in (8, 16, 32)
    for bk in (256, 512)
    for nw in (4, 8)
    for ns in (2, 3)
]


@triton.autotune(configs=_DENSE_STEP_CONFIGS, key=["B"],
                 restore_value=["v_ptr", "I_ptr"])
@triton.jit
def glif3_dense_multistep_step_kernel(
    w_ptr, x_ptr, bias_ptr,  # (B*B,), x_seq[t] (B,), (B,)
    v_ptr, I_ptr,
    v_th_ptr, v_reset_ptr, v_rest_ptr, c_m_ptr, tau_ptr, k_ptr, asc_ptr, mask_ptr,
    s_prev_ptr, s_out_ptr, v_out_ptr,  # spike[t-1], spike[t], v[t]
    B, dt, alpha,
    M: tl.constexpr, MPAD: tl.constexpr, hard_reset: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    off = tl.program_id(0) * BM
    mode_valid = tl.arange(0, MPAD) < M

    # Recurrent term lin[i] = sum_j W[i, j] * s_prev[j], columns pipelined.
    lin = tl.zeros([BM], dtype=tl.float32)
    for k0 in tl.range(0, B, BK, num_stages=NUM_STAGES):
        w = tl.make_block_ptr(w_ptr, shape=(B, B), strides=(B, 1), offsets=(off, k0),
                              block_shape=(BM, BK), order=(1, 0))
        w = tl.load(w, boundary_check=(0, 1), padding_option="zero")
        s = load_vec(s_prev_ptr, B, k0, BK)
        lin += tl.sum(w * s[None, :], axis=1)

    x_in = load_vec(x_ptr, B, off, BM) + load_vec(bias_ptr, B, off, BM) + lin
    v = load_vec(v_ptr, B, off, BM)
    v_th = load_vec(v_th_ptr, B, off, BM)
    v_reset = load_vec(v_reset_ptr, B, off, BM)
    v_rest = load_vec(v_rest_ptr, B, off, BM)
    c_m = load_vec(c_m_ptr, B, off, BM)
    tau = load_vec(tau_ptr, B, off, BM)
    mask = load_vec(mask_ptr, B, off, BM)
    iasc = load_modes(I_ptr, B, M, off, BM, MPAD)
    k = tl.where(mode_valid[None, :], load_modes(k_ptr, B, M, off, BM, MPAD), 1.0)
    asc = load_modes(asc_ptr, B, M, off, BM, MPAD)

    spike, v_post, I_post = glif3_neuron_update(
        x_in, v, iasc, k, asc, v_th, v_reset, v_rest, c_m, tau, mask, dt, hard_reset)

    store_vec(v_ptr, B, off, v_post, BM)
    store_modes(I_ptr, B, M, off, I_post, BM, MPAD)
    store_vec(s_out_ptr, B, off, spike, BM)
    store_vec(v_out_ptr, B, off, v_post, BM)


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
        raise ValueError("Iasc must have shape (B*M,) flattened.")
    for key in ("v_th", "v_reset", "v_rest", "c_m", "tau"):
        if params[key].numel() != B:
            raise ValueError(f"params['{key}'] must have shape (B,).")
    for key in ("k", "asc_amps"):
        if params[key].numel() != B * M:
            raise ValueError(f"params['{key}'] must have shape (B*M,).")


def _mpad(M: int) -> int:
    return triton.next_power_of_2(int(M))


# ----------------------------------------------------------------------------
# Single-step (training primitive)
# ----------------------------------------------------------------------------
class GLIF3StepTriton(torch.autograd.Function):
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
        block: int = 256,
    ):
        if not (v.is_cuda and Iasc.is_cuda and x.is_cuda):
            raise RuntimeError("GLIF3StepTriton requires CUDA tensors.")

        v, Iasc, x = _as_fp32(v), _as_fp32(Iasc), _as_fp32(x)
        v_th, v_reset, v_rest = _as_fp32(v_th), _as_fp32(v_reset), _as_fp32(v_rest)
        c_m, tau, k = _as_fp32(c_m), _as_fp32(tau), _as_fp32(k)
        asc_amps, not_refrac = _as_fp32(asc_amps), _as_fp32(not_refrac)

        B, M = v.numel(), int(M)
        params = {
            "v_th": v_th, "v_reset": v_reset, "v_rest": v_rest,
            "c_m": c_m, "tau": tau, "k": k, "asc_amps": asc_amps,
        }
        _check_shapes(B, M, v, Iasc, x, params, not_refrac)

        v_out = torch.empty_like(v)
        I_out = torch.empty_like(Iasc)
        s_out = torch.empty_like(v)

        glif3_step_forward_kernel[(triton.cdiv(B, block),)](
            v, Iasc, x,
            v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
            v_out, I_out, s_out,
            B, float(dt), float(alpha),
            M=M, MPAD=_mpad(M), hard_reset=bool(hard_reset), BLOCK=int(block),
        )

        ctx.save_for_backward(
            v, Iasc, x, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac, s_out
        )
        ctx.dt, ctx.M = float(dt), M
        ctx.hard_reset = bool(hard_reset)
        ctx.alpha, ctx.block = float(alpha), int(block)
        return v_out, I_out, s_out

    @staticmethod
    def backward(ctx, dv_out, dI_out, ds_out):
        (v, Iasc, x, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
         s_out) = ctx.saved_tensors
        B, M, block = v.numel(), ctx.M, ctx.block

        dv_out = _as_fp32(dv_out if dv_out is not None else torch.zeros_like(v))
        dI_out = _as_fp32(dI_out if dI_out is not None else torch.zeros_like(Iasc))
        ds_out = _as_fp32(ds_out if ds_out is not None else torch.zeros_like(v))

        dv = torch.empty_like(v)
        dI = torch.empty_like(Iasc)
        dx = torch.empty_like(x)
        dasc = torch.empty_like(asc_amps)

        glif3_step_backward_kernel[(triton.cdiv(B, block),)](
            v, Iasc, x,
            v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
            s_out, dv_out, dI_out, ds_out,
            dv, dI, dx, dasc,
            B, float(ctx.dt), float(ctx.alpha),
            M=M, MPAD=_mpad(M), hard_reset=ctx.hard_reset, BLOCK=int(block),
        )

        return (dv, dI, dx, None, None, None, None, None, None, dasc,
                None, None, None, None, None, None)


def _glif3_step_triton(
    v: Float[torch.Tensor, " B"],
    Iasc: Float[torch.Tensor, " B M"],
    x: Float[torch.Tensor, " B"],
    params: dict,
    not_refrac: Float[torch.Tensor, " B"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-step GLIF3 update (autograd) using Triton."""
    return GLIF3StepTriton.apply(
        v, Iasc, x,
        params["v_th"], params["v_reset"], params["v_rest"],
        params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
        float(dt), int(M), bool(hard_reset), float(alpha), int(block),
    )


# ----------------------------------------------------------------------------
# Neuron multistep
# ----------------------------------------------------------------------------
def _neuron_multistep_forward(
    x_seq, v_work, I_work, params, not_refrac, dt, M, hard_reset, alpha, block,
    save_iseq=False,
):
    """Launch the forward neuron-multistep kernel.

    Returns ``(s_seq, v_seq, v_out, I_out, I_seq)``, where ``I_seq`` holds the
    per-step pre-spike I_asc (shape ``(T, B*M)``) only when ``save_iseq`` is set
    (training); otherwise it is ``None``. ``v_work`` / ``I_work`` are updated in
    place; callers that must keep the originals (training) should pass clones.
    """
    T, B = x_seq.shape
    v_seq = torch.empty((T, B), device=v_work.device, dtype=v_work.dtype)
    s_seq = torch.empty((T, B), device=v_work.device, dtype=v_work.dtype)
    v_out = torch.empty_like(v_work)
    I_out = torch.empty_like(I_work)
    if save_iseq:
        I_seq = torch.empty((T, B * int(M)), device=v_work.device, dtype=v_work.dtype)
    else:
        I_seq = s_seq  # dummy pointer; never written when save_iseq is False

    glif3_neuron_multistep_forward_kernel[(triton.cdiv(B, block),)](
        x_seq.reshape(-1), v_work, I_work,
        params["v_th"], params["v_reset"], params["v_rest"],
        params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
        s_seq, v_seq, v_out, I_out, I_seq,
        B, float(dt), float(alpha), T,
        M=int(M), MPAD=_mpad(M), hard_reset=bool(hard_reset),
        save_iseq=bool(save_iseq), BLOCK=int(block),
    )
    return s_seq, v_seq, v_out, I_out, (I_seq if save_iseq else None)


class GLIF3NeuronMultiStepTriton(torch.autograd.Function):
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
        block: int = 256,
    ):
        if not x_seq.is_cuda:
            raise RuntimeError("Neuron multistep Triton requires CUDA tensors.")

        x_seq, v, Iasc = _as_fp32(x_seq), _as_fp32(v), _as_fp32(Iasc)
        v_th, v_reset, v_rest = _as_fp32(v_th), _as_fp32(v_reset), _as_fp32(v_rest)
        c_m, tau, k = _as_fp32(c_m), _as_fp32(tau), _as_fp32(k)
        asc_amps, not_refrac = _as_fp32(asc_amps), _as_fp32(not_refrac)

        B, M = x_seq.shape[1], int(M)
        ctx.Iasc_was_flat = Iasc.ndim == 1
        ctx.asc_was_flat = asc_amps.ndim == 1
        if Iasc.numel() != B * M:
            raise ValueError("Iasc must have B*M elements for multistep.")
        I_flat = Iasc.reshape(-1)

        params = {
            "v_th": v_th, "v_reset": v_reset, "v_rest": v_rest,
            "c_m": c_m, "tau": tau, "k": k, "asc_amps": asc_amps,
        }
        s_seq, v_seq, v_out, I_out, I_seq = _neuron_multistep_forward(
            x_seq, v.clone(), I_flat.clone(), params, not_refrac,
            dt, M, hard_reset, alpha, int(block), save_iseq=True,
        )

        ctx.save_for_backward(
            x_seq, v, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
            s_seq, v_seq, I_seq,
        )
        ctx.dt, ctx.M = float(dt), M
        ctx.hard_reset = bool(hard_reset)
        ctx.alpha, ctx.block = float(alpha), int(block)
        return s_seq, v_seq, v_out, I_out.view(B, M)

    @staticmethod
    def backward(ctx, ds_seq, dv_seq, dv_out, dI_out):
        (x_seq, v0, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
         s_seq, v_seq, I_seq) = ctx.saved_tensors
        T, B, M, block = *x_seq.shape, ctx.M, ctx.block

        ds_seq = _as_fp32(ds_seq if ds_seq is not None else torch.zeros_like(s_seq))
        dv_seq = _as_fp32(dv_seq if dv_seq is not None else torch.zeros_like(v_seq))
        dv_out = _as_fp32(dv_out if dv_out is not None else torch.zeros_like(v0))
        dI_out = _as_fp32(
            dI_out if dI_out is not None else torch.zeros((B * M,), device=v0.device)
        )

        dv0 = torch.empty_like(v0)
        dI0 = torch.empty((B * M,), device=v0.device, dtype=v0.dtype)
        dasc = torch.empty((B * M,), device=v0.device, dtype=v0.dtype)
        dx_seq = torch.empty_like(x_seq)

        glif3_neuron_multistep_backward_kernel[(triton.cdiv(B, block),)](
            x_seq.reshape(-1), v0, v_seq.reshape(-1), s_seq.reshape(-1),
            I_seq.reshape(-1),
            v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
            ds_seq.reshape(-1), dv_seq.reshape(-1), dv_out, dI_out.reshape(-1),
            dv0, dI0, dx_seq.reshape(-1), dasc,
            B, float(ctx.dt), float(ctx.alpha), T,
            M=M, MPAD=_mpad(M), hard_reset=ctx.hard_reset, BLOCK=int(block),
        )

        dI0_out = dI0.reshape(-1) if ctx.Iasc_was_flat else dI0.view(B, M)
        dasc_out = dasc.reshape(-1) if ctx.asc_was_flat else dasc.view(B, M)
        return (dx_seq, dv0, dI0_out, None, None, None, None, None, None, dasc_out,
                None, None, None, None, None, None)


def glif3_multistep_fused_triton(
    x_seq: Float[torch.Tensor, " T B"],
    v: Float[torch.Tensor, " B"],
    Iasc: Float[torch.Tensor, " B M"],
    params: dict,
    not_refrac: Float[torch.Tensor, " B"],
    dt: float,
    M: int,
    hard_reset: bool = False,
    alpha: float = 2.0,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Neuron-only multistep. Uses a lean forward-only path for inference and
    the autograd path (saving intermediates) when gradients are required."""
    need_grad = torch.is_grad_enabled() and (
        x_seq.requires_grad or v.requires_grad or Iasc.requires_grad
        or params["asc_amps"].requires_grad
    )
    if need_grad:
        return GLIF3NeuronMultiStepTriton.apply(
            x_seq, v, Iasc,
            params["v_th"], params["v_reset"], params["v_rest"],
            params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
            float(dt), int(M), bool(hard_reset), float(alpha), int(block),
        )

    if not x_seq.is_cuda:
        raise RuntimeError("Neuron multistep Triton requires CUDA tensors.")
    B = v.numel()
    fp32 = {key: _as_fp32(params[key]) for key in params}
    s_seq, v_seq, v_out, I_out, _ = _neuron_multistep_forward(
        _as_fp32(x_seq), _as_fp32(v), _as_fp32(Iasc.reshape(-1)),
        fp32, _as_fp32(not_refrac), float(dt), int(M), bool(hard_reset),
        float(alpha), int(block),
    )
    return s_seq, v_seq, v_out, I_out.view(B, int(M))


# ----------------------------------------------------------------------------
# Dense (neuron + recurrent connection) multistep — inference only
def _dense_multistep_matmul(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha, block
):
    """Non-fused path: cuBLAS matmul for the recurrent term, neuron step kernel
    for the dynamics. One launch pair per timestep."""
    T, B = x_seq.shape
    v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    step_v = torch.empty_like(v)
    step_s = torch.empty_like(v)
    s_prev = torch.zeros((B,), device=v.device, dtype=v.dtype)
    grid = (triton.cdiv(B, block),)

    for t in range(T):
        # lin_i = sum_j weight[i, j] * s_prev[j]  (cuBLAS matrix-vector).
        x_in = torch.addmv(x_seq[t] + bias, weight, s_prev).contiguous()
        glif3_dense_step_kernel[grid](
            x_in, v, Iasc,
            params["v_th"], params["v_reset"], params["v_rest"],
            params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
            step_s, step_v,
            B, float(dt), float(alpha),
            M=int(M), MPAD=_mpad(M), hard_reset=bool(hard_reset), BLOCK=int(block),
        )
        s_seq[t] = step_s
        v_seq[t] = step_v
        s_prev = step_s.clone()

    return s_seq, v_seq, v.clone(), Iasc.clone()


def _dense_multistep_fused(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha
):
    """Fused per-step path: one launch per timestep of the fused kernel that
    computes the recurrent gemv in-kernel (pipelined) and the neuron update. The
    launch boundary is the cross-step barrier. Does not mutate v / Iasc."""
    T, B = x_seq.shape
    v_work = v.clone()
    I_work = Iasc.clone()
    v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    zeros = torch.zeros((B,), device=v.device, dtype=v.dtype)
    grid = lambda meta: (triton.cdiv(B, meta["BM"]),)

    weight_flat = weight.reshape(-1)
    for t in range(T):
        s_prev = zeros if t == 0 else s_seq[t - 1]
        glif3_dense_multistep_step_kernel[grid](
            weight_flat, x_seq[t], bias, v_work, I_work,
            params["v_th"], params["v_reset"], params["v_rest"],
            params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
            s_prev, s_seq[t], v_seq[t],
            B, float(dt), float(alpha),
            M=int(M), MPAD=_mpad(M), hard_reset=bool(hard_reset),
        )
    return s_seq, v_seq, v_work, I_work


def glif3_dense_multistep_fused_triton(
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
    block: int = 256,
    fused_matmul: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense (neuron + recurrent connection) multistep.

    Routes on gradient need: when gradients are required it runs the autograd
    path (single-step op + torch matmul per step); otherwise a lean forward-only
    kernel. In the forward-only case ``fused_matmul=True`` fuses the recurrent
    gemv (pipelined in-kernel) with the neuron update, one launch per step; the
    launch boundary is the cross-step barrier. ``fused_matmul=False`` delegates
    the matmul to cuBLAS with a neuron kernel per step.
    """
    need_grad = torch.is_grad_enabled() and (
        weight.requires_grad or bias.requires_grad or x_seq.requires_grad
        or v.requires_grad or Iasc.requires_grad or params["asc_amps"].requires_grad
    )
    if need_grad:
        return dense_multistep_autograd(
            _glif3_step_triton, x_seq, weight, bias, v, Iasc, params, not_refrac,
            float(dt), int(M), bool(hard_reset), float(alpha),
        )
    if not x_seq.is_cuda:
        raise RuntimeError("Fused dense multistep Triton requires CUDA tensors.")
    B = v.numel()
    if weight.shape != (B, B):
        raise ValueError("weight must have shape (B, B).")
    if bias.shape != (B,):
        raise ValueError("bias must have shape (B,).")

    x_seq, weight, bias = _as_fp32(x_seq), _as_fp32(weight), _as_fp32(bias)
    v, Iasc = _as_fp32(v), _as_fp32(Iasc.reshape(-1))
    fp32 = {key: _as_fp32(params[key]) for key in params}
    not_refrac = _as_fp32(not_refrac)

    path = _dense_multistep_fused if fused_matmul else _dense_multistep_matmul
    args = (x_seq, weight, bias, v, Iasc, fp32, not_refrac,
            float(dt), int(M), bool(hard_reset), float(alpha))
    if fused_matmul:
        s_seq, v_seq, v_out, I_out = path(*args)
    else:
        s_seq, v_seq, v_out, I_out = path(*args, int(block))
    return s_seq, v_seq, v_out, I_out.view(B, int(M))


# ----------------------------------------------------------------------------
# Sparse (scale-free recurrent) multistep — fused CSR-vector SpMV + neuron update.
# One program (a single warp) per neuron row is the classic CSR-vector schedule:
# the warp reduces lin_i = sum_p val[p] * s_prev[col[p]] over the row's nonzeros
# (gathered), which tolerates the scale-free load imbalance since the scheduler
# overlaps short and long rows. One launch per step; the launch boundary is the
# cross-step barrier.
# ----------------------------------------------------------------------------
@triton.jit
def glif3_sparse_step_kernel(
    crow_ptr, col_ptr, val_ptr,  # CSR
    x_ptr, bias_ptr, v_ptr, I_ptr,
    v_th_ptr, v_reset_ptr, v_rest_ptr, c_m_ptr, tau_ptr, k_ptr, asc_ptr, mask_ptr,
    s_prev_ptr, s_out_ptr, v_out_ptr,
    N, dt, alpha,
    M: tl.constexpr, hard_reset: tl.constexpr, BK: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(crow_ptr + row)
    end = tl.load(crow_ptr + row + 1)

    # CSR-vector reduction of this row's nonzeros against the previous spikes.
    acc = tl.zeros((BK,), dtype=tl.float32)
    for p in range(start, end, BK):
        offs = p + tl.arange(0, BK)
        m = offs < end
        cols = tl.load(col_ptr + offs, mask=m, other=0)
        vals = tl.load(val_ptr + offs, mask=m, other=0.0)
        sv = tl.load(s_prev_ptr + cols, mask=m, other=0.0)
        acc += vals * sv
    lin = tl.sum(acc)

    # GLIF3 update for this single neuron row.
    i_sum = 0.0
    for mm in range(M):
        i_sum += tl.load(I_ptr + row * M + mm)
    x_in = tl.load(x_ptr + row) + tl.load(bias_ptr + row) + lin
    v_prime = neuronal_charge(
        tl.load(v_ptr + row), x_in, i_sum, tl.load(v_rest_ptr + row),
        tl.load(c_m_ptr + row), tl.load(tau_ptr + row), dt)
    spike = neuronal_fire(v_prime, tl.load(v_th_ptr + row), tl.load(mask_ptr + row))
    v_post = neuronal_reset(v_prime, spike, tl.load(v_th_ptr + row),
                            tl.load(v_reset_ptr + row), hard_reset)
    for mm in range(M):
        i_asc = tl.load(I_ptr + row * M + mm)
        tl.store(I_ptr + row * M + mm, exp_euler(
            i_asc, -tl.load(k_ptr + row * M + mm) * i_asc,
            -tl.load(k_ptr + row * M + mm), dt) + tl.load(asc_ptr + row * M + mm) * spike)
    tl.store(v_ptr + row, v_post)
    tl.store(s_out_ptr + row, spike)
    tl.store(v_out_ptr + row, v_post)


def _sparse_multistep_fused(
    x_seq, weight, bias, v, Iasc, params, not_refrac, dt, M, hard_reset, alpha
):
    """Fused per-step sparse path: one launch per timestep of the CSR-vector SpMV
    kernel fused with the neuron update. Does not mutate v / Iasc."""
    T, N = x_seq.shape
    v_work = v.clone()
    I_work = Iasc.clone()
    v_seq = torch.empty((T, N), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, N), device=v.device, dtype=v.dtype)
    zeros = torch.zeros((N,), device=v.device, dtype=v.dtype)

    for t in range(T):
        s_prev = zeros if t == 0 else s_seq[t - 1]
        glif3_sparse_step_kernel[(N,)](
            weight.crow, weight.col, weight.val,
            x_seq[t], bias, v_work, I_work,
            params["v_th"], params["v_reset"], params["v_rest"],
            params["c_m"], params["tau"], params["k"], params["asc_amps"], not_refrac,
            s_prev, s_seq[t], v_seq[t],
            N, float(dt), float(alpha),
            M=int(M), hard_reset=bool(hard_reset), BK=128, num_warps=1,
        )
    return s_seq, v_seq, v_work, I_work


def glif3_sparse_multistep_fused_triton(
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

    Inference fuses a CSR-vector SpMV with the neuron update, one launch per step.
    Training composes the single-step op with a COO ``torch.sparse`` SpMV each
    step (``sparse_multistep_autograd``).
    """
    need_grad = torch.is_grad_enabled() and (
        weight.val.requires_grad or bias.requires_grad or x_seq.requires_grad
        or v.requires_grad or Iasc.requires_grad or params["asc_amps"].requires_grad
    )
    if need_grad:
        return sparse_multistep_autograd(
            _glif3_step_triton, x_seq, weight, bias, v, Iasc, params, not_refrac,
            float(dt), int(M), bool(hard_reset), float(alpha),
        )
    if not x_seq.is_cuda:
        raise RuntimeError("Fused sparse multistep Triton requires CUDA tensors.")
    N = v.numel()
    x_seq, bias = _as_fp32(x_seq), _as_fp32(bias)
    v, Iasc = _as_fp32(v), _as_fp32(Iasc.reshape(-1))
    fp32 = {key: _as_fp32(params[key]) for key in params}
    not_refrac = _as_fp32(not_refrac)

    s_seq, v_seq, v_out, I_out = _sparse_multistep_fused(
        x_seq, weight, bias, v, Iasc, fp32, not_refrac,
        float(dt), int(M), bool(hard_reset), float(alpha),
    )
    return s_seq, v_seq, v_out, I_out.view(N, int(M))


glif3_step_triton = GLIF3StepOps(
    step=_glif3_step_triton,
    multistep_fused=glif3_multistep_fused_triton,
    dense_multistep_fused=glif3_dense_multistep_fused_triton,
    sparse_multistep_fused=glif3_sparse_multistep_fused_triton,
)
