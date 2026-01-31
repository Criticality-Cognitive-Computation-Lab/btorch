from __future__ import annotations

import math

import torch
import warp as wp
from jaxtyping import Float


_PI = math.pi
_ATAN_SCALE = 0.5 * math.pi
_WARP_INITIALIZED = False


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


# --------------------------
# Your Warp kernel must exist
# --------------------------
# glif3_step_kernel(v, Iasc, x, v_out, Iasc_out, spike_out,
#                  v_th, v_reset, v_rest, c_m, tau, k, asc_amps, not_refrac,
#                  dt, M, hard_reset, alpha)


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

        # Manual analytic backward to match GLIF3 reference behavior.
        B = v.numel()

        dv_out = _as_fp32(dv_out, "dv_out") if dv_out is not None else None
        dI_out = _as_fp32(dI_out, "dI_out") if dI_out is not None else None
        ds_out = _as_fp32(ds_out, "ds_out") if ds_out is not None else None

        if dv_out is None:
            dv_out = torch.zeros_like(v)
        if dI_out is None:
            dI_out = torch.zeros_like(Iasc)
        if ds_out is None:
            ds_out = torch.zeros_like(v)

        Iasc_view = Iasc.view(B, M)
        k_view = k.view(B, M)
        asc_view = asc_amps.view(B, M)
        dI_out_view = dI_out.view(B, M)

        I_sum = Iasc_view.sum(dim=-1)
        a = torch.exp(-dt / tau)
        v_inf = v_rest + tau * (x + I_sum) / c_m
        v_prime = v_inf + (v - v_inf) * a

        denom = v_th - v_reset
        u = (v_prime - v_th) / denom
        scale = _ATAN_SCALE * alpha
        s = (u >= 0).to(v.dtype) * not_refrac

        ds_du = not_refrac * (0.5 * alpha) / (1.0 + (scale * u) * (scale * u))
        ds_dvprime = ds_du / denom

        if hard_reset:
            dvprime = dv_out * (1.0 - s)
            ds_from_v = dv_out * (-(v_prime - v_reset))
        else:
            dvprime = dv_out
            ds_from_v = dv_out * (-(v_th - v_reset))

        dI_s_sum = (dI_out_view * asc_view).sum(dim=-1)
        ds_total = ds_out + ds_from_v + dI_s_sum
        dvprime = dvprime + ds_total * ds_dvprime

        dv = dvprime * a
        dv_inf = dvprime * (1.0 - a)

        dx = dv_inf * (tau / c_m)
        dI_common = dv_inf * (tau / c_m)

        I_dec = Iasc_view * torch.exp(-k_view * dt)
        dI = dI_out_view * torch.exp(-k_view * dt) + dI_common.unsqueeze(-1)

        d_c_m = dv_inf * (-(tau * (x + I_sum) / (c_m * c_m)))
        d_tau = dv_inf * ((x + I_sum) / c_m)
        d_tau = d_tau + dvprime * (v - v_inf) * a * (dt / (tau * tau))

        dk = dI_out_view * (-dt) * I_dec
        dasc = dI_out_view * s.unsqueeze(-1)

        dI = dI.reshape(-1)
        dk = dk.reshape(-1)
        dasc = dasc.reshape(-1)

        dc_m = d_c_m if c_m.requires_grad else None
        dtau = d_tau if tau.requires_grad else None
        dk = dk if k.requires_grad else None
        dasc = dasc if asc_amps.requires_grad else None

        # Return grads matching forward signature:
        return (
            dv,
            dI,
            dx,
            None,
            None,
            None,  # v_th, v_reset, v_rest
            dc_m,
            dtau,
            dk,
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
    x_seq = _as_fp32(x_seq.reshape(-1), "x_seq")
    weight = _as_fp32(weight.reshape(-1), "weight")
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

    wp.launch(
        glif3_dense_fwd_kernel,
        dim=1,
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
            _to_wp(s_seq.reshape(-1), requires_grad=False),
            _to_wp(v_seq.reshape(-1), requires_grad=False),
            _to_wp(v_out, requires_grad=False),
            _to_wp(I_out, requires_grad=False),
            int(T),
            int(B),
            int(M),
            float(dt),
            int(1 if hard_reset else 0),
            float(alpha),
        ],
        device="cuda",
    )

    return s_seq, v_seq, v_out, I_out.view(B, M)


glif3_step_warp.dense_multistep_fused = glif3_dense_multistep_fused_warp


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
