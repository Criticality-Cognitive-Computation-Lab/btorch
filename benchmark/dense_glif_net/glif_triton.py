"""
GLIF3Triton: fused 1-step GLIF3 update (v, Iasc) + surrogate spike + reset
(+ Iasc jump) with a Triton forward + custom backward.

Highlights:
- grads w.r.t. v, Iasc, x, asc_amps
- refractory mask treated as constant (stop-grad), matching `.detach()` usage
- ATan surrogate uses a lightweight atan approximation (for older Triton)
- ds_out supported so spike losses backpropagate through the surrogate

Assumptions:
- v, x: shape (B,) contiguous
- Iasc: shape (B*M,) contiguous with base = i*M
- params["k"], params["asc_amps"]: shape (B*M,)
- M is small (<=4 typical); we unroll loop with tl.static_range.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from jaxtyping import Float


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


_ATAN_APPROX_A = 0.28


# ----------------------------
# Triton forward
# ----------------------------
@triton.jit
def glif3_fwd(
    v_ptr,
    I_ptr,
    x_ptr,
    v_th_ptr,
    v_reset_ptr,
    v_rest_ptr,
    c_m_ptr,
    tau_ptr,
    k_ptr,
    asc_ptr,
    mask_ptr,
    v_out_ptr,
    I_out_ptr,
    s_out_ptr,
    B: tl.constexpr,
    dt: tl.constexpr,
    M: tl.constexpr,
    hard_reset: tl.constexpr,
    alpha: tl.constexpr,
    approx_a: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    inb = i < B

    v = tl.load(v_ptr + i, mask=inb, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + i, mask=inb, other=0.0).to(tl.float32)

    v_th = tl.load(v_th_ptr + i, mask=inb, other=0.0).to(tl.float32)
    v_reset = tl.load(v_reset_ptr + i, mask=inb, other=0.0).to(tl.float32)
    v_rest = tl.load(v_rest_ptr + i, mask=inb, other=0.0).to(tl.float32)

    c_m = tl.load(c_m_ptr + i, mask=inb, other=1.0).to(tl.float32)
    tau = tl.load(tau_ptr + i, mask=inb, other=1.0).to(tl.float32)

    mask = tl.load(mask_ptr + i, mask=inb, other=1.0).to(tl.float32)

    a = libdevice.exp(-dt / tau)

    # sum Iasc over modes
    I_sum = tl.zeros([BLOCK], dtype=tl.float32)
    base = i * M
    for m in tl.static_range(0, M):
        I_m = tl.load(I_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
        I_sum += I_m

    v_inf = v_rest + tau * (x + I_sum) / c_m
    v_prime = v_inf + (v - v_inf) * a

    s = ((v_prime - v_th) > 0) * mask

    # reset
    if hard_reset:
        v_post = v_prime - (v_prime - v_reset) * s
    else:
        v_post = v_prime - (v_th - v_reset) * s

    # Iasc decay + jump
    for m in tl.static_range(0, M):
        I_m = tl.load(I_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
        k_m = tl.load(k_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
        b = libdevice.exp(-k_m * dt)
        I_dec = I_m * b

        asc_m = tl.load(asc_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
        I_post = I_dec + asc_m * s

        tl.store(I_out_ptr + base + m, I_post.to(tl.float32), mask=inb)

    tl.store(v_out_ptr + i, v_post.to(tl.float32), mask=inb)
    tl.store(s_out_ptr + i, s.to(tl.float32), mask=inb)


# ----------------------------
# Triton fused dense multistep (single-program, forward-only)
# ----------------------------
@triton.jit
def glif3_dense_fwd(
    x_ptr,
    w_ptr,
    b_ptr,
    v_ptr,
    I_ptr,
    v_th_ptr,
    v_reset_ptr,
    v_rest_ptr,
    c_m_ptr,
    tau_ptr,
    k_ptr,
    asc_ptr,
    mask_ptr,
    s_seq_ptr,
    v_seq_ptr,
    v_out_ptr,
    I_out_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    dt: tl.constexpr,
    M: tl.constexpr,
    hard_reset: tl.constexpr,
    alpha: tl.constexpr,
    approx_a: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    inb = i < B

    v = tl.load(v_ptr + i, mask=inb, other=0.0).to(tl.float32)
    v_th = tl.load(v_th_ptr + i, mask=inb, other=0.0).to(tl.float32)
    v_reset = tl.load(v_reset_ptr + i, mask=inb, other=0.0).to(tl.float32)
    v_rest = tl.load(v_rest_ptr + i, mask=inb, other=0.0).to(tl.float32)
    c_m = tl.load(c_m_ptr + i, mask=inb, other=1.0).to(tl.float32)
    tau = tl.load(tau_ptr + i, mask=inb, other=1.0).to(tl.float32)
    mask = tl.load(mask_ptr + i, mask=inb, other=1.0).to(tl.float32)

    for t in tl.static_range(0, T):
        x = tl.load(x_ptr + t * B + i, mask=inb, other=0.0).to(tl.float32)
        bias = tl.load(b_ptr + i, mask=inb, other=0.0).to(tl.float32)

        lin = tl.zeros([BLOCK], dtype=tl.float32)
        for k in tl.static_range(0, B, 32):
            k_idx = k + tl.arange(0, 32)
            in_k = k_idx < B
            w_ptrs = (i[:, None] * B) + k_idx[None, :]
            w = tl.load(
                w_ptr + w_ptrs,
                mask=inb[:, None] & in_k[None, :],
                other=0.0,
            ).to(tl.float32)
            s_prev = tl.where(
                t == 0,
                tl.zeros([32], dtype=tl.float32),
                tl.load(
                    s_seq_ptr + (t - 1) * B + k_idx,
                    mask=in_k,
                    other=0.0,
                ).to(tl.float32),
            )
            lin += tl.sum(w * s_prev[None, :], axis=1)
        x_in = x + bias + lin

        I_sum = tl.zeros([BLOCK], dtype=tl.float32)
        base = i * M
        for m in tl.static_range(0, M):
            I_m = tl.load(I_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
            I_sum += I_m

        a = libdevice.exp(-dt / tau)
        v_inf = v_rest + tau * (x_in + I_sum) / c_m
        v_prime = v_inf + (v - v_inf) * a

        s = ((v_prime - v_th) > 0) * mask
        if hard_reset:
            v_post = v_prime - (v_prime - v_reset) * s
        else:
            v_post = v_prime - (v_th - v_reset) * s

        for m in tl.static_range(0, M):
            I_m = tl.load(I_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
            k_m = tl.load(k_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
            b = libdevice.exp(-k_m * dt)
            I_dec = I_m * b
            asc_m = tl.load(asc_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
            I_post = I_dec + asc_m * s
            tl.store(I_ptr + base + m, I_post.to(tl.float32), mask=inb)
            tl.store(I_out_ptr + base + m, I_post.to(tl.float32), mask=inb)

        v = v_post
        tl.store(v_seq_ptr + t * B + i, v_post.to(tl.float32), mask=inb)
        tl.store(s_seq_ptr + t * B + i, s.to(tl.float32), mask=inb)

    tl.store(v_out_ptr + i, v.to(tl.float32), mask=inb)


# ----------------------------
# Triton backward (minimal)
# grads: v, Iasc, x, asc_amps
# ----------------------------
@triton.jit
def glif3_bwd(
    v_ptr,
    I_ptr,
    x_ptr,
    v_th_ptr,
    v_reset_ptr,
    v_rest_ptr,
    c_m_ptr,
    tau_ptr,
    k_ptr,
    asc_ptr,
    mask_ptr,
    s_ptr,  # saved spike (B,)
    ds_ptr,  # upstream ds_out (B,)
    dv_post_ptr,  # upstream dv_out (B,)
    dI_post_ptr,  # upstream dI_out (B*M,)
    dv_ptr,
    dI_ptr,
    dx_ptr,
    dasc_ptr,
    B: tl.constexpr,
    dt: tl.constexpr,
    M: tl.constexpr,
    hard_reset: tl.constexpr,
    alpha: tl.constexpr,
    approx_a: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    inb = i < B

    v = tl.load(v_ptr + i, mask=inb, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + i, mask=inb, other=0.0).to(tl.float32)

    v_th = tl.load(v_th_ptr + i, mask=inb, other=0.0).to(tl.float32)
    v_reset = tl.load(v_reset_ptr + i, mask=inb, other=0.0).to(tl.float32)
    v_rest = tl.load(v_rest_ptr + i, mask=inb, other=0.0).to(tl.float32)

    c_m = tl.load(c_m_ptr + i, mask=inb, other=1.0).to(tl.float32)
    tau = tl.load(tau_ptr + i, mask=inb, other=1.0).to(tl.float32)

    mask = tl.load(mask_ptr + i, mask=inb, other=1.0).to(tl.float32)

    s = tl.load(s_ptr + i, mask=inb, other=0.0).to(tl.float32)
    dv_post = tl.load(dv_post_ptr + i, mask=inb, other=0.0).to(tl.float32)

    a = libdevice.exp(-dt / tau)

    # recompute v_prime and u
    I_sum = tl.zeros([BLOCK], dtype=tl.float32)
    base = i * M
    for m in tl.static_range(0, M):
        I_m = tl.load(I_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
        I_sum += I_m

    v_inf = v_rest + tau * (x + I_sum) / c_m
    v_prime = v_inf + (v - v_inf) * a

    denom = v_th - v_reset
    u = (v_prime - v_th) / denom

    # ds/dvprime for atan surrogate (matches approximation above)
    pi_inv = 0.3183098861837907  # 1 / pi
    scale = 1.5707963267948966 * alpha  # 0.5 * pi
    z = scale * u
    denom_z = 1.0 + approx_a * z * z
    f_prime = (1.0 - approx_a * z * z) / (denom_z * denom_z)
    ds_du = mask * (scale * pi_inv) * f_prime
    ds_dvprime = ds_du * (1.0 / denom)

    # Backprop through reset:
    # - dvprime uses s as constant
    # - ds_from_v captures v_post's dependence on s
    if hard_reset:
        dvprime = dv_post * (1.0 - s)
        ds_from_v = dv_post * (-(v_prime - v_reset))
    else:
        dvprime = dv_post
        ds_from_v = dv_post * (-(v_th - v_reset))

    # ds gradient from Iasc jump: I_post = I_dec + asc*s
    dI_s_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for m in tl.static_range(0, M):
        dI_post_m = tl.load(dI_post_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
        asc_m = tl.load(asc_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
        dI_s_sum += dI_post_m * asc_m
        # grad wrt asc_amps: dL/dasc = dI_post * s
        tl.store(dasc_ptr + base + m, (dI_post_m * s).to(tl.float32), mask=inb)

    ds_out = tl.load(ds_ptr + i, mask=inb, other=0.0).to(tl.float32)
    ds_total = ds_from_v + dI_s_sum + ds_out
    dvprime = dvprime + ds_total * ds_dvprime

    # Backprop v_prime = a*v + (1-a)*v_inf
    dv = dvprime * a
    dv_inf = dvprime * (1.0 - a)

    # v_inf = v_rest + tau*(x + I_sum)/c_m
    dx = dv_inf * (tau / c_m)
    dI_common = dv_inf * (tau / c_m)

    # Iasc decay + add dI_common
    for m in tl.static_range(0, M):
        I_m = tl.load(I_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
        k_m = tl.load(k_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
        b = libdevice.exp(-k_m * dt)

        dI_post_m = tl.load(dI_post_ptr + base + m, mask=inb, other=0.0).to(tl.float32)
        dI_m = dI_post_m * b + dI_common
        tl.store(dI_ptr + base + m, dI_m.to(tl.float32), mask=inb)

    tl.store(dv_ptr + i, dv.to(tl.float32), mask=inb)
    tl.store(dx_ptr + i, dx.to(tl.float32), mask=inb)


# ----------------------------
# Autograd wrapper
# ----------------------------
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
        v = _as_fp32(v, "v")
        Iasc = _as_fp32(Iasc, "Iasc")
        x = _as_fp32(x, "x")

        # params
        v_th = _as_fp32(v_th, "v_th")
        v_reset = _as_fp32(v_reset, "v_reset")
        v_rest = _as_fp32(v_rest, "v_rest")
        c_m = _as_fp32(c_m, "c_m")
        tau = _as_fp32(tau, "tau")
        k = _as_fp32(k, "k")
        asc_amps = _as_fp32(asc_amps, "asc_amps")
        not_refrac = _as_fp32(not_refrac, "not_refrac")

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

        grid = (triton.cdiv(B, block),)
        glif3_fwd[grid](
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
            v_out,
            I_out,
            s_out,
            B=B,
            dt=float(dt),
            M=int(M),
            hard_reset=1 if hard_reset else 0,
            alpha=float(alpha),
            approx_a=_ATAN_APPROX_A,
            BLOCK=block,
            num_warps=4,
        )

        # Save for backward (save s_out; recompute v_prime internally)
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
            s_out,
        )
        ctx.dt = float(dt)
        ctx.M = int(M)
        ctx.hard_reset = bool(hard_reset)
        ctx.alpha = float(alpha)
        ctx.block = int(block)
        return v_out, I_out, s_out

    @staticmethod
    def backward(ctx, dv_out, dI_out, ds_out):
        (
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
            s_out,
        ) = ctx.saved_tensors

        B = v.numel()
        M = ctx.M
        block = ctx.block

        dv_out = _as_fp32(
            dv_out if dv_out is not None else torch.zeros_like(v),
            "dv_out",
        )
        dI_out = _as_fp32(
            dI_out if dI_out is not None else torch.zeros_like(Iasc),
            "dI_out",
        )
        ds_out = _as_fp32(
            ds_out if ds_out is not None else torch.zeros_like(v),
            "ds_out",
        )

        dv = torch.empty_like(v)
        dI = torch.empty_like(Iasc)
        dx = torch.empty_like(x)
        dasc = torch.empty_like(asc_amps)

        grid = (triton.cdiv(B, block),)
        glif3_bwd[grid](
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
            s_out,
            ds_out,
            dv_out,
            dI_out,
            dv,
            dI,
            dx,
            dasc,
            B=B,
            dt=ctx.dt,
            M=M,
            hard_reset=1 if ctx.hard_reset else 0,
            alpha=ctx.alpha,
            approx_a=_ATAN_APPROX_A,
            BLOCK=block,
            num_warps=4,
        )

        # Return grads matching forward args:
        return (
            dv,
            dI,
            dx,
            None,
            None,
            None,  # v_th, v_reset, v_rest
            None,
            None,  # c_m, tau (not implemented here)
            None,  # k (not implemented)
            dasc,  # asc_amps
            None,  # not_refrac
            None,
            None,
            None,  # dt, M, hard_reset
            None,  # alpha
            None,  # block
        )


def glif3_step_triton(
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
    """Single-step GLIF3 update using Triton."""
    return GLIF3StepTriton.apply(
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
        int(block),
    )


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
    use_torch_matmul: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.is_grad_enabled():
        raise RuntimeError("Fused multistep Triton is forward-only.")
    if not x_seq.is_cuda:
        raise RuntimeError("Fused multistep Triton requires CUDA tensors.")
    B = v.numel()
    if weight.shape[0] != B or weight.shape[1] != B:
        raise ValueError("weight must have shape (B, B).")
    if bias.shape[0] != B:
        raise ValueError("bias must have shape (B,).")

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

    T = x_seq.shape[0]
    v_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T, B), device=v.device, dtype=v.dtype)

    if use_torch_matmul:
        x_seq = _as_fp32(x_seq, "x_seq")
        weight = _as_fp32(weight, "weight")
        bias = _as_fp32(bias, "bias")
        s_prev = torch.zeros_like(v)
        for t in range(T):
            lin = torch.matmul(weight, s_prev)
            x_t = x_seq[t] + bias + lin
            v, Iasc, s_prev = glif3_step_triton(
                v=v,
                Iasc=Iasc,
                x=x_t,
                params={
                    "v_th": v_th,
                    "v_reset": v_reset,
                    "v_rest": v_rest,
                    "c_m": c_m,
                    "tau": tau,
                    "k": k,
                    "asc_amps": asc_amps,
                },
                not_refrac=not_refrac,
                dt=dt,
                M=M,
                hard_reset=hard_reset,
                alpha=alpha,
                block=block,
            )
            s_seq[t] = s_prev
            v_seq[t] = v
        return s_seq, v_seq, v, Iasc.view(B, M)

    if B > block:
        raise RuntimeError("Fused multistep Triton requires B <= block.")

    x_seq = _as_fp32(x_seq, "x_seq")
    weight = _as_fp32(weight, "weight")
    bias = _as_fp32(bias, "bias")
    v_out = torch.empty_like(v)
    I_out = torch.empty_like(Iasc)

    glif3_dense_fwd[(1,)](
        x_seq,
        weight,
        bias,
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
        v_out,
        I_out,
        B=B,
        T=T,
        dt=float(dt),
        M=int(M),
        hard_reset=1 if hard_reset else 0,
        alpha=float(alpha),
        approx_a=_ATAN_APPROX_A,
        BLOCK=int(block),
        num_warps=4,
    )

    return s_seq, v_seq, v_out, I_out.view(B, M)


glif3_step_triton.dense_multistep_fused = glif3_dense_multistep_fused_triton


class GLIF3Triton(torch.nn.Module):
    """Thin module wrapper, analogous to GLIF3Warp.

    You can:
      - keep v/Iasc as external state (recommended for functional style), or
      - store buffers in the module (easy to add).
    """

    def __init__(
        self, M: int, hard_reset: bool = False, alpha: float = 2.0, block: int = 256
    ):
        super().__init__()
        self.M = int(M)
        self.hard_reset = bool(hard_reset)
        self.alpha = float(alpha)
        self.block = int(block)

    def step(
        self,
        v: Float[torch.Tensor, " B"],
        Iasc: Float[torch.Tensor, " B M"],
        x: Float[torch.Tensor, " B"],
        params: dict,
        not_refrac: Float[torch.Tensor, " B"],
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return glif3_step_triton(
            v=v,
            Iasc=Iasc,
            x=x,
            params=params,
            not_refrac=not_refrac,
            dt=float(dt),
            M=self.M,
            hard_reset=self.hard_reset,
            alpha=self.alpha,
            block=self.block,
        )
