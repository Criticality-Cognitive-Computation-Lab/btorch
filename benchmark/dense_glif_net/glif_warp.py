from __future__ import annotations

import math
import os

import torch
import warp as wp
from jaxtyping import Float


_PI = math.pi
_ATAN_SCALE = 0.5 * math.pi
_WARP_INITIALIZED = False
_WARP_ENABLE_TILES = (
    os.getenv("BTORCH_WARP_TILES", "1") == "1"
    and hasattr(wp, "tile_matmul")
    and hasattr(wp, "tile_load")
)


@wp.func
def surrogate_atan(u: float, alpha: float) -> float:
    # Match btorch.models.surrogate.ATan primitive:
    # s = 0.5 + atan(0.5*pi*alpha*u)/pi
    scale = _ATAN_SCALE * alpha
    return 0.5 + math.atan(scale * u) / _PI


@wp.func_grad(surrogate_atan)
def surrogate_atan_grad(u: float, alpha: float, adj: float):
    # d/du [atan(scale*u)/pi] with scale=0.5*pi*alpha
    scale = _ATAN_SCALE * alpha
    grad = (0.5 * alpha) * (1.0 / (1.0 + (scale * u) * (scale * u)))
    wp.adjoint[u] += adj * grad


@wp.kernel
def glif3_step_kernel(
    v: wp.array(dtype=wp.float32),
    Iasc: wp.array(dtype=wp.float32),  # flattened as [B*M]
    x: wp.array(dtype=wp.float32),
    v_out: wp.array(dtype=wp.float32),
    Iasc_out: wp.array(dtype=wp.float32),
    spike_out: wp.array(dtype=wp.float32),
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),  # flattened [B*M]
    asc_amps: wp.array(dtype=wp.float32),  # flattened [B*M]
    not_refrac: wp.array(dtype=wp.float32),  # stop-grad mask
    dt: float,
    M: int,
    hard_reset: int,
    alpha: float,
):
    i = wp.tid()  # 0..B-1

    v_i = v[i]

    # sum Iasc modes
    I_sum = float(0.0)
    base = i * M
    for m in range(M):
        I_sum += Iasc[base + m]

    # exp-Euler for v: v_next = v_inf + (v - v_inf)*a, a = exp(-dt/tau)
    tau_i = tau[i]
    a = wp.exp(-dt / tau_i)

    v_inf = v_rest[i] + tau_i * (x[i] + I_sum) / c_m[i]
    v_next = v_inf + (v_i - v_inf) * a

    # exp-Euler for Iasc modes: I_next = I * exp(-k dt)
    for m in range(M):
        km = k[base + m]
        Iasc_out[base + m] = Iasc[base + m] * wp.exp(-km * dt)

    # hard spike (forward); surrogate derivative is used in backward
    denom = v_th[i] - v_reset[i]
    u = (v_next - v_th[i]) / denom
    s = wp.where(u >= 0.0, 1.0, 0.0)

    # refractory gating (treat as constant mask)
    s = s * not_refrac[i]

    # reset
    if hard_reset == 1:
        v_post = v_next - (v_next - v_reset[i]) * s
    else:
        v_post = v_next - (v_th[i] - v_reset[i]) * s

    # after-spike current jump
    for m in range(M):
        Iasc_out[base + m] = Iasc_out[base + m] + asc_amps[base + m] * s

    v_out[i] = v_post
    spike_out[i] = s


@wp.kernel(enable_backward=False)
def glif3_bwd_kernel(
    v: wp.array(dtype=wp.float32),
    Iasc: wp.array(dtype=wp.float32),  # flattened as [B*M]
    x: wp.array(dtype=wp.float32),
    dv_out: wp.array(dtype=wp.float32),
    dI_out: wp.array(dtype=wp.float32),
    ds_out: wp.array(dtype=wp.float32),
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),  # flattened [B*M]
    asc_amps: wp.array(dtype=wp.float32),  # flattened [B*M]
    not_refrac: wp.array(dtype=wp.float32),
    dt: float,
    M: int,
    hard_reset: int,
    alpha: float,
    dv: wp.array(dtype=wp.float32),
    dI: wp.array(dtype=wp.float32),
    dx: wp.array(dtype=wp.float32),
    dasc: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    base = i * M

    v_i = v[i]
    x_i = x[i]
    v_th_i = v_th[i]
    v_reset_i = v_reset[i]
    v_rest_i = v_rest[i]
    c_m_i = c_m[i]
    tau_i = tau[i]
    mask_i = not_refrac[i]

    I_sum = float(0.0)
    for m in range(M):
        I_sum += Iasc[base + m]

    a = wp.exp(-dt / tau_i)
    v_inf = v_rest_i + tau_i * (x_i + I_sum) / c_m_i
    v_prime = v_inf + (v_i - v_inf) * a

    denom = v_th_i - v_reset_i
    u = (v_prime - v_th_i) / denom
    scale = _ATAN_SCALE * alpha
    s = wp.where(u >= 0.0, 1.0, 0.0) * mask_i

    ds_du = mask_i * (0.5 * alpha) / (1.0 + (scale * u) * (scale * u))
    ds_dvprime = ds_du / denom

    dvprime = dv_out[i]
    if hard_reset:
        dvprime = dv_out[i] * (1.0 - s)

    ds_from_v = dv_out[i] * (-(v_th_i - v_reset_i))
    if hard_reset:
        ds_from_v = dv_out[i] * (-(v_prime - v_reset_i))

    dI_s_sum = float(0.0)
    for m in range(M):
        dI_post_m = dI_out[base + m]
        dI_s_sum += dI_post_m * asc_amps[base + m]
        dasc[base + m] = dI_post_m * s

    ds_total = ds_out[i] + ds_from_v + dI_s_sum
    dvprime = dvprime + ds_total * ds_dvprime

    dv_i = dvprime * a
    dv_inf = dvprime * (1.0 - a)

    dx[i] = dv_inf * (tau_i / c_m_i)
    dI_common = dv_inf * (tau_i / c_m_i)

    for m in range(M):
        k_m = k[base + m]
        b = wp.exp(-k_m * dt)
        dI[base + m] = dI_out[base + m] * b + dI_common

    dv[i] = dv_i


def _as_fp32(tensor: torch.Tensor, name: str) -> torch.Tensor:
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


def _to_wp(tensor: torch.Tensor, *, requires_grad: bool) -> wp.array:
    return wp.from_torch(tensor, dtype=wp.float32, requires_grad=requires_grad)


def _ensure_warp_init() -> None:
    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True


class GLIF3StepWarp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v: torch.Tensor,  # (B,)
        Iasc: torch.Tensor,  # (B*M,) flattened
        x: torch.Tensor,  # (B,)
        v_th: torch.Tensor,  # (B,)
        v_reset: torch.Tensor,  # (B,)
        v_rest: torch.Tensor,  # (B,)
        c_m: torch.Tensor,  # (B,)
        tau: torch.Tensor,  # (B,)
        k: torch.Tensor,  # (B*M,)
        asc_amps: torch.Tensor,  # (B*M,)
        not_refrac: torch.Tensor,  # (B,)
        dt: float,
        M: int,
        hard_reset: bool,
        alpha: float,
    ):
        if not (v.is_cuda and Iasc.is_cuda and x.is_cuda):
            raise RuntimeError("GLIF3StepWarp requires CUDA tensors.")
        _ensure_warp_init()
        v = _as_fp32(v, "v")
        Iasc = _as_fp32(Iasc, "Iasc")
        x = _as_fp32(x, "x")
        v_th = _as_fp32(v_th, "v_th")
        v_reset = _as_fp32(v_reset, "v_reset")
        v_rest = _as_fp32(v_rest, "v_rest")
        c_m = _as_fp32(c_m, "c_m")
        tau = _as_fp32(tau, "tau")
        k = _as_fp32(k, "k")
        asc_amps = _as_fp32(asc_amps, "asc_amps")
        not_refrac = _as_fp32(not_refrac, "not_refrac")

        # Outputs
        B = v.numel()
        params = {
            "v_th": v_th,
            "v_reset": v_reset,
            "v_rest": v_rest,
            "c_m": c_m,
            "tau": tau,
            "k": k,
            "asc_amps": asc_amps,
        }
        _check_shapes(B, int(M), v, Iasc, x, params, not_refrac)
        v_out = torch.empty_like(v)
        I_out = torch.empty_like(Iasc)
        s_out = torch.empty_like(v)

        # Launch warp forward (no tape here)
        # Note: forward does NOT need warp requires_grad
        v_wp = _to_wp(v, requires_grad=False)
        I_wp = _to_wp(Iasc, requires_grad=False)
        x_wp = _to_wp(x, requires_grad=False)

        v_out_wp = _to_wp(v_out, requires_grad=False)
        I_out_wp = _to_wp(I_out, requires_grad=False)
        s_out_wp = _to_wp(s_out, requires_grad=False)

        v_th_wp = _to_wp(v_th, requires_grad=False)
        v_reset_wp = _to_wp(v_reset, requires_grad=False)
        v_rest_wp = _to_wp(v_rest, requires_grad=False)

        c_m_wp = _to_wp(c_m, requires_grad=False)
        tau_wp = _to_wp(tau, requires_grad=False)
        k_wp = _to_wp(k, requires_grad=False)
        asc_wp = _to_wp(asc_amps, requires_grad=False)

        mask_wp = _to_wp(not_refrac, requires_grad=False)

        wp.launch(
            glif3_step_kernel,
            dim=B,
            inputs=[
                v_wp,
                I_wp,
                x_wp,
                v_out_wp,
                I_out_wp,
                s_out_wp,
                v_th_wp,
                v_reset_wp,
                v_rest_wp,
                c_m_wp,
                tau_wp,
                k_wp,
                asc_wp,
                mask_wp,
                float(dt),
                int(M),
                int(1 if hard_reset else 0),
                float(alpha),
            ],
            device="cuda",
        )

        # Save everything needed to replay forward in backward
        ctx.save_for_backward(
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
        )
        ctx.dt = float(dt)
        ctx.M = int(M)
        ctx.hard_reset = bool(hard_reset)
        ctx.alpha = float(alpha)

        return v_out, I_out, s_out

    @staticmethod
    def backward(ctx, dv_out, dI_out, ds_out):
        (v, Iasc, x, v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac) = (
            ctx.saved_tensors
        )

        dt = ctx.dt
        M = ctx.M
        hard_reset = ctx.hard_reset
        alpha = ctx.alpha

        B = v.numel()

        dv_out = _as_fp32(
            dv_out if dv_out is not None else torch.zeros_like(v), "dv_out"
        )
        dI_out = _as_fp32(
            dI_out if dI_out is not None else torch.zeros_like(Iasc), "dI_out"
        )
        ds_out = _as_fp32(
            ds_out if ds_out is not None else torch.zeros_like(v), "ds_out"
        )

        dv = torch.empty_like(v)
        dI = torch.empty_like(Iasc)
        dx = torch.empty_like(x)
        dasc = torch.empty_like(asc_amps)

        _ensure_warp_init()
        wp.launch(
            glif3_bwd_kernel,
            dim=B,
            inputs=[
                _to_wp(_as_fp32(v, "v"), requires_grad=False),
                _to_wp(_as_fp32(Iasc, "Iasc"), requires_grad=False),
                _to_wp(_as_fp32(x, "x"), requires_grad=False),
                _to_wp(dv_out, requires_grad=False),
                _to_wp(dI_out, requires_grad=False),
                _to_wp(ds_out, requires_grad=False),
                _to_wp(_as_fp32(v_th, "v_th"), requires_grad=False),
                _to_wp(_as_fp32(v_reset, "v_reset"), requires_grad=False),
                _to_wp(_as_fp32(v_rest, "v_rest"), requires_grad=False),
                _to_wp(_as_fp32(c_m, "c_m"), requires_grad=False),
                _to_wp(_as_fp32(tau, "tau"), requires_grad=False),
                _to_wp(_as_fp32(k, "k"), requires_grad=False),
                _to_wp(_as_fp32(asc_amps, "asc_amps"), requires_grad=False),
                _to_wp(_as_fp32(not_refrac, "not_refrac"), requires_grad=False),
                float(dt),
                int(M),
                int(1 if hard_reset else 0),
                float(alpha),
            ],
            outputs=[
                _to_wp(dv, requires_grad=False),
                _to_wp(dI, requires_grad=False),
                _to_wp(dx, requires_grad=False),
                _to_wp(dasc, requires_grad=False),
            ],
            block_dim=256,
            device="cuda",
        )

        # Return grads matching forward signature:
        return (
            dv,
            dI,
            dx,
            None,
            None,
            None,  # v_th, v_reset, v_rest
            None,
            None,
            None,
            dasc,  # c_m, tau, k, asc_amps
            None,  # not_refrac
            None,
            None,
            None,
            None,  # dt, M, hard_reset, alpha
        )


def glif3_step_warp(
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
    """Single-step GLIF3 update using Warp."""
    return GLIF3StepWarp.apply(
        v,
        Iasc,
        x,
        params["v_th"],
        params["v_reset"],
        params["v_rest"],
        params["c_m"],
        params["tau"],
        params["k"],
        params["asc_amps"],
        not_refrac,
        float(dt),
        int(M),
        bool(hard_reset),
        float(alpha),
    )


@wp.kernel
def glif3_dense_fwd_kernel(
    x_seq: wp.array(dtype=wp.float32),  # (T*B,)
    w: wp.array(dtype=wp.float32),  # (B*B,)
    b: wp.array(dtype=wp.float32),  # (B,)
    v: wp.array(dtype=wp.float32),  # (B,)
    Iasc: wp.array(dtype=wp.float32),  # (B*M,)
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),  # (B*M,)
    asc_amps: wp.array(dtype=wp.float32),  # (B*M,)
    not_refrac: wp.array(dtype=wp.float32),
    s_seq: wp.array(dtype=wp.float32),  # (T*B,)
    v_seq: wp.array(dtype=wp.float32),  # (T*B,)
    v_out: wp.array(dtype=wp.float32),  # (B,)
    I_out: wp.array(dtype=wp.float32),  # (B*M,)
    T: int,
    B: int,
    M: int,
    dt: float,
    hard_reset: int,
    alpha: float,
):
    if wp.tid() != 0:
        return

    for t in range(T):
        for i in range(B):
            I_sum = float(0.0)
            base = i * M
            for m in range(M):
                I_sum += Iasc[base + m]

            lin = float(0.0)
            w_base = i * B
            for j in range(B):
                s_prev = float(0.0)
                if t > 0:
                    s_prev = s_seq[(t - 1) * B + j]
                lin += w[w_base + j] * s_prev

            x = x_seq[t * B + i] + b[i] + lin

            tau_i = tau[i]
            a = wp.exp(-dt / tau_i)
            v_inf = v_rest[i] + tau_i * (x + I_sum) / c_m[i]
            v_prime = v_inf + (v[i] - v_inf) * a

            denom = v_th[i] - v_reset[i]
            u = (v_prime - v_th[i]) / denom
            s = wp.where(u >= 0.0, 1.0, 0.0) * not_refrac[i]

            if hard_reset == 1:
                v_post = v_prime - (v_prime - v_reset[i]) * s
            else:
                v_post = v_prime - (v_th[i] - v_reset[i]) * s

            for m in range(M):
                km = k[base + m]
                I_dec = Iasc[base + m] * wp.exp(-km * dt)
                I_post = I_dec + asc_amps[base + m] * s
                Iasc[base + m] = I_post
                I_out[base + m] = I_post

            v[i] = v_post
            s_seq[t * B + i] = s
            v_seq[t * B + i] = v_post

    for i in range(B):
        v_out[i] = v[i]


@wp.kernel
def glif3_multistep_fwd_kernel(
    x_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v: wp.array(dtype=wp.float32),  # (B,)
    Iasc: wp.array(dtype=wp.float32),  # (B*M,)
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),  # (B*M,)
    asc_amps: wp.array(dtype=wp.float32),  # (B*M,)
    not_refrac: wp.array(dtype=wp.float32),
    s_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v_out: wp.array(dtype=wp.float32),  # (B,)
    I_out: wp.array(dtype=wp.float32),  # (B*M,)
    T: int,
    B: int,
    M: int,
    dt: float,
    hard_reset: int,
    alpha: float,
):
    i = wp.tid()
    if i >= B:
        return

    base = i * M
    v_i = v[i]

    for t in range(T):
        I_sum = float(0.0)
        for m in range(M):
            I_sum += Iasc[base + m]

        x = x_seq[t, i]
        tau_i = tau[i]
        a = wp.exp(-dt / tau_i)
        v_inf = v_rest[i] + tau_i * (x + I_sum) / c_m[i]
        v_prime = v_inf + (v_i - v_inf) * a

        denom = v_th[i] - v_reset[i]
        u = (v_prime - v_th[i]) / denom
        s = wp.where(u >= 0.0, 1.0, 0.0) * not_refrac[i]

        if hard_reset == 1:
            v_post = v_prime - (v_prime - v_reset[i]) * s
        else:
            v_post = v_prime - (v_th[i] - v_reset[i]) * s

        for m in range(M):
            km = k[base + m]
            I_dec = Iasc[base + m] * wp.exp(-km * dt)
            I_post = I_dec + asc_amps[base + m] * s
            Iasc[base + m] = I_post
            I_out[base + m] = I_post

        v_i = v_post
        s_seq[t, i] = s
        v_seq[t, i] = v_post

    v[i] = v_i
    v_out[i] = v_i


@wp.kernel(enable_backward=False)
def glif3_multistep_bwd_kernel(
    x_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v0: wp.array(dtype=wp.float32),  # (B,)
    v_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    s_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),  # (B*M,)
    asc_amps: wp.array(dtype=wp.float32),  # (B*M,)
    not_refrac: wp.array(dtype=wp.float32),
    ds_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    dv_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    dv_out: wp.array(dtype=wp.float32),  # (B,)
    T: int,
    B: int,
    M: int,
    dt: float,
    hard_reset: int,
    alpha: float,
    I_post: wp.array(dtype=wp.float32),  # (B*M,) working buffer
    dI: wp.array(dtype=wp.float32),  # (B*M,) working buffer
    dv0: wp.array(dtype=wp.float32),  # (B,)
    dx_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    dasc: wp.array(dtype=wp.float32),  # (B*M,)
):
    i = wp.tid()
    if i >= B:
        return

    base = i * M
    v0_i = v0[i]
    v_th_i = v_th[i]
    v_reset_i = v_reset[i]
    v_rest_i = v_rest[i]
    c_m_i = c_m[i]
    tau_i = tau[i]
    mask_i = not_refrac[i]

    a = wp.exp(-dt / tau_i)
    denom = v_th_i - v_reset_i
    tau_over_c = tau_i / c_m_i
    scale = _ATAN_SCALE * alpha

    dv_post = dv_out[i]

    for t in range(T - 1, -1, -1):
        dv_post = dv_post + dv_seq[t, i]
        s_t = s_seq[t, i]
        ds_t = ds_seq[t, i]

        v_pre = v0_i
        if t > 0:
            v_pre = v_seq[t - 1, i]

        I_sum = float(0.0)
        for m in range(M):
            k_m = k[base + m]
            b = wp.exp(-k_m * dt)
            I_pre = (I_post[base + m] - asc_amps[base + m] * s_t) / b
            I_post[base + m] = I_pre
            I_sum += I_pre

        x_t = x_seq[t, i]
        v_inf = v_rest_i + tau_i * (x_t + I_sum) / c_m_i
        v_prime = v_inf + (v_pre - v_inf) * a

        u = (v_prime - v_th_i) / denom
        ds_du = mask_i * (0.5 * alpha) / (1.0 + (scale * u) * (scale * u))
        ds_dvprime = ds_du / denom

        dvprime = dv_post
        ds_from_v = dv_post * (-(v_th_i - v_reset_i))
        if hard_reset == 1:
            dvprime = dv_post * (1.0 - s_t)
            ds_from_v = dv_post * (-(v_prime - v_reset_i))

        dI_s_sum = float(0.0)
        for m in range(M):
            dI_s_sum += dI[base + m] * asc_amps[base + m]

        ds_total = ds_t + ds_from_v + dI_s_sum
        dvprime = dvprime + ds_total * ds_dvprime

        dv_pre = dvprime * a
        dv_inf = dvprime * (1.0 - a)
        dx_seq[t, i] = dv_inf * tau_over_c
        dI_common = dv_inf * tau_over_c

        for m in range(M):
            k_m = k[base + m]
            b = wp.exp(-k_m * dt)
            dI_old = dI[base + m]
            dasc[base + m] = dasc[base + m] + dI_old * s_t
            dI[base + m] = dI_old * b + dI_common

        dv_post = dv_pre

    dv0[i] = dv_post


@wp.kernel
def glif3_dense_lin_kernel(
    w: wp.array2d(dtype=wp.float32),  # (B, B)
    s_prev: wp.array(dtype=wp.float32),  # (B,)
    lin: wp.array(dtype=wp.float32),  # (B,)
    B: int,
):
    i = wp.tid()
    if i >= B:
        return
    acc = float(0.0)
    for j in range(B):
        acc += w[i, j] * s_prev[j]
    lin[i] = acc


@wp.kernel
def glif3_dense_step_kernel(
    x_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    b: wp.array(dtype=wp.float32),  # (B,)
    lin: wp.array(dtype=wp.float32),  # (B,)
    v: wp.array(dtype=wp.float32),  # (B,)
    Iasc: wp.array(dtype=wp.float32),  # (B*M,)
    v_th: wp.array(dtype=wp.float32),
    v_reset: wp.array(dtype=wp.float32),
    v_rest: wp.array(dtype=wp.float32),
    c_m: wp.array(dtype=wp.float32),
    tau: wp.array(dtype=wp.float32),
    k: wp.array(dtype=wp.float32),  # (B*M,)
    asc_amps: wp.array(dtype=wp.float32),  # (B*M,)
    not_refrac: wp.array(dtype=wp.float32),
    s_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    v_seq: wp.array2d(dtype=wp.float32),  # (T, B)
    t: int,
    B: int,
    M: int,
    dt: float,
    hard_reset: int,
    alpha: float,
):
    i = wp.tid()
    if i >= B:
        return

    base = i * M
    I_sum = float(0.0)
    for m in range(M):
        I_sum += Iasc[base + m]

    x = x_seq[t, i] + b[i] + lin[i]
    tau_i = tau[i]
    a = wp.exp(-dt / tau_i)
    v_inf = v_rest[i] + tau_i * (x + I_sum) / c_m[i]
    v_prime = v_inf + (v[i] - v_inf) * a

    denom = v_th[i] - v_reset[i]
    u = (v_prime - v_th[i]) / denom
    s = wp.where(u >= 0.0, 1.0, 0.0) * not_refrac[i]

    if hard_reset == 1:
        v_post = v_prime - (v_prime - v_reset[i]) * s
    else:
        v_post = v_prime - (v_th[i] - v_reset[i]) * s

    for m in range(M):
        km = k[base + m]
        I_dec = Iasc[base + m] * wp.exp(-km * dt)
        I_post = I_dec + asc_amps[base + m] * s
        Iasc[base + m] = I_post

    v[i] = v_post
    s_seq[t, i] = s
    v_seq[t, i] = v_post


TILE_M = 16
TILE_K = 16
glif3_dense_fwd_tile_kernel = None

if _WARP_ENABLE_TILES:
    TILE_M = wp.constant(16)
    TILE_K = wp.constant(16)

    @wp.kernel
    def glif3_dense_fwd_tile_kernel(
        x_seq: wp.array2d(dtype=wp.float32),  # (T, B)
        w: wp.array2d(dtype=wp.float32),  # (B, B)
        b: wp.array(dtype=wp.float32),  # (B,)
        v: wp.array(dtype=wp.float32),  # (B,)
        Iasc: wp.array(dtype=wp.float32),  # (B*M,)
        v_th: wp.array(dtype=wp.float32),
        v_reset: wp.array(dtype=wp.float32),
        v_rest: wp.array(dtype=wp.float32),
        c_m: wp.array(dtype=wp.float32),
        tau: wp.array(dtype=wp.float32),
        k: wp.array(dtype=wp.float32),  # (B*M,)
        asc_amps: wp.array(dtype=wp.float32),  # (B*M,)
        not_refrac: wp.array(dtype=wp.float32),
        s_seq: wp.array2d(dtype=wp.float32),  # (T, B)
        v_seq: wp.array2d(dtype=wp.float32),  # (T, B)
        v_out: wp.array(dtype=wp.float32),  # (B,)
        I_out: wp.array(dtype=wp.float32),  # (B*M,)
        T: int,
        B: int,
        M: int,
        dt: float,
        hard_reset: int,
        alpha: float,
    ):
        tile_i = wp.tid()
        row = tile_i * TILE_M
        if row >= B:
            return

        for t in range(T):
            lin = wp.tile_zeros(shape=(TILE_M, 1), dtype=wp.float32)
            for k0 in range(0, B, TILE_K):
                w_tile = wp.tile_load(w, shape=(TILE_M, TILE_K), offset=(row, k0))
                if t == 0:
                    s_tile = wp.tile_zeros(shape=(TILE_K, 1), dtype=wp.float32)
                else:
                    s_tile = wp.tile_load(s_seq, shape=(TILE_K, 1), offset=(t - 1, k0))
                wp.tile_matmul(w_tile, s_tile, lin)

            for r in range(TILE_M):
                idx = row + r
                if idx >= B:
                    break
                base = idx * M
                I_sum = float(0.0)
                for m in range(M):
                    I_sum += Iasc[base + m]

                x = x_seq[t, idx] + b[idx] + lin[r, 0]
                tau_i = tau[idx]
                a = wp.exp(-dt / tau_i)
                v_inf = v_rest[idx] + tau_i * (x + I_sum) / c_m[idx]
                v_prime = v_inf + (v[idx] - v_inf) * a

                denom = v_th[idx] - v_reset[idx]
                u = (v_prime - v_th[idx]) / denom
                s = wp.where(u >= 0.0, 1.0, 0.0) * not_refrac[idx]

                if hard_reset == 1:
                    v_post = v_prime - (v_prime - v_reset[idx]) * s
                else:
                    v_post = v_prime - (v_th[idx] - v_reset[idx]) * s

                for m in range(M):
                    km = k[base + m]
                    I_dec = Iasc[base + m] * wp.exp(-km * dt)
                    I_post = I_dec + asc_amps[base + m] * s
                    Iasc[base + m] = I_post
                    I_out[base + m] = I_post

                v[idx] = v_post
                s_seq[t, idx] = s
                v_seq[t, idx] = v_post

        for r in range(TILE_M):
            idx = row + r
            if idx >= B:
                break
            v_out[idx] = v[idx]


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.is_grad_enabled():
        raise RuntimeError("Fused multistep Warp is forward-only.")
    if not x_seq.is_cuda:
        raise RuntimeError("Fused multistep Warp requires CUDA tensors.")

    _ensure_warp_init()
    T, B = x_seq.shape
    use_tiles = _WARP_ENABLE_TILES and glif3_dense_fwd_tile_kernel is not None
    x_seq = _as_fp32(x_seq, "x_seq")
    weight = _as_fp32(weight, "weight")
    bias = _as_fp32(bias, "bias")
    v = _as_fp32(v, "v")
    Iasc = _as_fp32(Iasc.reshape(-1), "Iasc")
    v_th = _as_fp32(params["v_th"], "v_th")
    v_reset = _as_fp32(params["v_reset"], "v_reset")
    v_rest = _as_fp32(params["v_rest"], "v_rest")
    c_m = _as_fp32(params["c_m"], "c_m")
    tau = _as_fp32(params["tau"], "tau")
    k = _as_fp32(params["k"], "k")
    asc_amps = _as_fp32(params["asc_amps"], "asc_amps")
    not_refrac = _as_fp32(not_refrac, "not_refrac")

    v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)

    v_out = torch.empty_like(v)
    I_out = torch.empty_like(Iasc)

    if use_tiles and B % int(TILE_M) == 0:
        wp.launch_tiled(
            glif3_dense_fwd_tile_kernel,
            dim=B // int(TILE_M),
            inputs=[
                _to_wp(x_seq, requires_grad=False),
                _to_wp(weight, requires_grad=False),
                _to_wp(bias, requires_grad=False),
                _to_wp(v, requires_grad=False),
                _to_wp(Iasc, requires_grad=False),
                _to_wp(v_th, requires_grad=False),
                _to_wp(v_reset, requires_grad=False),
                _to_wp(v_rest, requires_grad=False),
                _to_wp(c_m, requires_grad=False),
                _to_wp(tau, requires_grad=False),
                _to_wp(k, requires_grad=False),
                _to_wp(asc_amps, requires_grad=False),
                _to_wp(not_refrac, requires_grad=False),
                _to_wp(s_seq, requires_grad=False),
                _to_wp(v_seq, requires_grad=False),
                _to_wp(v_out, requires_grad=False),
                _to_wp(I_out, requires_grad=False),
                int(T),
                int(B),
                int(M),
                float(dt),
                int(1 if hard_reset else 0),
                float(alpha),
            ],
            block_dim=int(TILE_M),
            device="cuda",
        )
    else:
        lin = torch.empty((B,), device=v.device, dtype=v.dtype)
        v_out = torch.empty_like(v)
        I_out = torch.empty_like(Iasc)
        s_prev = torch.zeros((B,), device=v.device, dtype=v.dtype)

        w_wp = _to_wp(weight, requires_grad=False)
        x_wp = _to_wp(x_seq, requires_grad=False)
        b_wp = _to_wp(bias, requires_grad=False)
        v_wp = _to_wp(v, requires_grad=False)
        I_wp = _to_wp(Iasc, requires_grad=False)
        v_th_wp = _to_wp(v_th, requires_grad=False)
        v_reset_wp = _to_wp(v_reset, requires_grad=False)
        v_rest_wp = _to_wp(v_rest, requires_grad=False)
        c_m_wp = _to_wp(c_m, requires_grad=False)
        tau_wp = _to_wp(tau, requires_grad=False)
        k_wp = _to_wp(k, requires_grad=False)
        asc_wp = _to_wp(asc_amps, requires_grad=False)
        mask_wp = _to_wp(not_refrac, requires_grad=False)
        s_seq_wp = _to_wp(s_seq, requires_grad=False)
        v_seq_wp = _to_wp(v_seq, requires_grad=False)
        lin_wp = _to_wp(lin, requires_grad=False)
        s_prev_wp = _to_wp(s_prev, requires_grad=False)

        for t in range(T):
            if t > 0:
                s_prev = s_seq[t - 1]
                s_prev_wp = _to_wp(s_prev, requires_grad=False)
            wp.launch(
                glif3_dense_lin_kernel,
                dim=B,
                inputs=[w_wp, s_prev_wp, lin_wp, int(B)],
                block_dim=256,
                device="cuda",
            )
            wp.launch(
                glif3_dense_step_kernel,
                dim=B,
                inputs=[
                    x_wp,
                    b_wp,
                    lin_wp,
                    v_wp,
                    I_wp,
                    v_th_wp,
                    v_reset_wp,
                    v_rest_wp,
                    c_m_wp,
                    tau_wp,
                    k_wp,
                    asc_wp,
                    mask_wp,
                    s_seq_wp,
                    v_seq_wp,
                    int(t),
                    int(B),
                    int(M),
                    float(dt),
                    int(1 if hard_reset else 0),
                    float(alpha),
                ],
                block_dim=256,
                device="cuda",
            )

        v_out.copy_(v)
        I_out.copy_(Iasc)

    return s_seq, v_seq, v_out, I_out.view(B, M)


glif3_step_warp.dense_multistep_fused = glif3_dense_multistep_fused_warp


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
    return GLIF3MultiStepWarp.apply(
        x_seq,
        v,
        Iasc,
        params["v_th"],
        params["v_reset"],
        params["v_rest"],
        params["c_m"],
        params["tau"],
        params["k"],
        params["asc_amps"],
        not_refrac,
        float(dt),
        int(M),
        bool(hard_reset),
        float(alpha),
    )


class GLIF3MultiStepWarp(torch.autograd.Function):
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
            raise RuntimeError("Fused multistep Warp requires CUDA tensors.")

        _ensure_warp_init()
        x_seq = _as_fp32(x_seq, "x_seq")
        v = _as_fp32(v, "v")
        Iasc = _as_fp32(Iasc, "Iasc")
        v_th = _as_fp32(v_th, "v_th")
        v_reset = _as_fp32(v_reset, "v_reset")
        v_rest = _as_fp32(v_rest, "v_rest")
        c_m = _as_fp32(c_m, "c_m")
        tau = _as_fp32(tau, "tau")
        k = _as_fp32(k, "k")
        asc_amps = _as_fp32(asc_amps, "asc_amps")
        not_refrac = _as_fp32(not_refrac, "not_refrac")

        T, B = x_seq.shape
        ctx.Iasc_was_flat = Iasc.ndim == 1
        ctx.asc_was_flat = asc_amps.ndim == 1
        if Iasc.ndim == 1:
            if Iasc.numel() != B * int(M):
                raise ValueError("Iasc must have shape (B, M) for multistep.")
            Iasc = Iasc.view(B, int(M))
        Iasc_flat = Iasc.reshape(-1) if Iasc.ndim == 2 else Iasc
        if v.requires_grad or Iasc.requires_grad or x_seq.requires_grad:
            v_work = v.clone()
            I_work = Iasc_flat.clone()
        else:
            v_work = v
            I_work = Iasc_flat

        v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
        s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
        v_out = torch.empty_like(v)
        I_out = torch.empty_like(I_work)

        wp.launch(
            glif3_multistep_fwd_kernel,
            dim=B,
            inputs=[
                _to_wp(x_seq, requires_grad=False),
                _to_wp(v_work, requires_grad=False),
                _to_wp(I_work, requires_grad=False),
                _to_wp(v_th, requires_grad=False),
                _to_wp(v_reset, requires_grad=False),
                _to_wp(v_rest, requires_grad=False),
                _to_wp(c_m, requires_grad=False),
                _to_wp(tau, requires_grad=False),
                _to_wp(k, requires_grad=False),
                _to_wp(asc_amps, requires_grad=False),
                _to_wp(not_refrac, requires_grad=False),
                _to_wp(s_seq, requires_grad=False),
                _to_wp(v_seq, requires_grad=False),
                _to_wp(v_out, requires_grad=False),
                _to_wp(I_out, requires_grad=False),
                int(T),
                int(B),
                int(M),
                float(dt),
                int(1 if hard_reset else 0),
                float(alpha),
            ],
            block_dim=256,
            device="cuda",
        )

        ctx.save_for_backward(
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
            s_seq,
            v_seq,
            I_out,
        )
        ctx.dt = float(dt)
        ctx.M = int(M)
        ctx.hard_reset = bool(hard_reset)
        ctx.alpha = float(alpha)
        return s_seq, v_seq, v_out, I_out.view(B, M)

    @staticmethod
    def backward(ctx, ds_seq, dv_seq, dv_out, dI_out):
        (
            x_seq,
            v0,
            Iasc0,
            v_th,
            v_reset,
            v_rest,
            c_m,
            tau,
            k,
            asc_amps,
            not_refrac,
            s_seq,
            v_seq,
            I_out,
        ) = ctx.saved_tensors

        T, B = x_seq.shape
        M = ctx.M

        if ds_seq is None:
            ds_seq = torch.zeros_like(s_seq)
        if dv_seq is None:
            dv_seq = torch.zeros_like(v_seq)
        if dv_out is None:
            dv_out = torch.zeros_like(v0)
        if dI_out is None:
            dI_out = torch.zeros_like(I_out)

        dv0 = torch.empty_like(v0)
        dx_seq = torch.empty_like(x_seq)
        dI0 = dI_out.reshape(-1).clone()
        dasc = torch.zeros_like(dI0)
        I_post = I_out.clone()

        _ensure_warp_init()
        wp.launch(
            glif3_multistep_bwd_kernel,
            dim=B,
            inputs=[
                _to_wp(_as_fp32(x_seq, "x_seq"), requires_grad=False),
                _to_wp(_as_fp32(v0, "v"), requires_grad=False),
                _to_wp(_as_fp32(v_seq, "v_seq"), requires_grad=False),
                _to_wp(_as_fp32(s_seq, "s_seq"), requires_grad=False),
                _to_wp(_as_fp32(v_th, "v_th"), requires_grad=False),
                _to_wp(_as_fp32(v_reset, "v_reset"), requires_grad=False),
                _to_wp(_as_fp32(v_rest, "v_rest"), requires_grad=False),
                _to_wp(_as_fp32(c_m, "c_m"), requires_grad=False),
                _to_wp(_as_fp32(tau, "tau"), requires_grad=False),
                _to_wp(_as_fp32(k, "k"), requires_grad=False),
                _to_wp(_as_fp32(asc_amps, "asc_amps"), requires_grad=False),
                _to_wp(_as_fp32(not_refrac, "not_refrac"), requires_grad=False),
                _to_wp(_as_fp32(ds_seq, "ds_seq"), requires_grad=False),
                _to_wp(_as_fp32(dv_seq, "dv_seq"), requires_grad=False),
                _to_wp(_as_fp32(dv_out, "dv_out"), requires_grad=False),
                int(T),
                int(B),
                int(M),
                float(ctx.dt),
                int(1 if ctx.hard_reset else 0),
                float(ctx.alpha),
            ],
            outputs=[
                _to_wp(_as_fp32(I_post, "I_post"), requires_grad=False),
                _to_wp(_as_fp32(dI0, "dI0"), requires_grad=False),
                _to_wp(_as_fp32(dv0, "dv0"), requires_grad=False),
                _to_wp(_as_fp32(dx_seq, "dx_seq"), requires_grad=False),
                _to_wp(_as_fp32(dasc, "dasc"), requires_grad=False),
            ],
            block_dim=256,
            device="cuda",
        )

        if not ctx.Iasc_was_flat:
            dI0 = dI0.view(B, M)
        if not ctx.asc_was_flat:
            dasc = dasc.view(B, M)

        return (
            dx_seq,
            dv0,
            dI0,
            None,
            None,
            None,
            None,
            None,
            None,
            dasc,
            None,
            None,
            None,
            None,
            None,
        )


glif3_step_warp.multistep_fused = glif3_multistep_fused_warp


class GLIF3Warp(torch.nn.Module):
    """Example wrapper mirroring a GLIF3Triton-style "step" API.

    You control state (v, Iasc) outside or store internally, like your
    existing BaseNode.
    """

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
        return glif3_step_warp(
            v=v,
            Iasc=Iasc,
            x=x,
            params=params,
            not_refrac=not_refrac,
            dt=dt,
            M=self.M,
            hard_reset=self.hard_reset,
            alpha=self.alpha,
        )
