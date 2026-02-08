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
    B,
    dt,
    alpha,
    M: tl.constexpr,
    hard_reset: tl.constexpr,
    approx_a: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK
    i = offs_b + tl.arange(0, BLOCK)
    inb = i < B

    v_ptrs = tl.make_block_ptr(
        base=v_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    x_ptrs = tl.make_block_ptr(
        base=x_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_th_ptrs = tl.make_block_ptr(
        base=v_th_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_reset_ptrs = tl.make_block_ptr(
        base=v_reset_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_rest_ptrs = tl.make_block_ptr(
        base=v_rest_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    c_m_ptrs = tl.make_block_ptr(
        base=c_m_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tau_ptrs = tl.make_block_ptr(
        base=tau_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    mask_ptrs = tl.make_block_ptr(
        base=mask_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )

    v = tl.load(v_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    x = tl.load(x_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)

    v_th = tl.load(v_th_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    v_reset = tl.load(v_reset_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )
    v_rest = tl.load(v_rest_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )

    c_m = tl.load(c_m_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    c_m = tl.where(inb, c_m, 1.0)
    tau = tl.load(tau_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    tau = tl.where(inb, tau, 1.0)

    mask = tl.load(mask_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    mask = tl.where(inb, mask, 1.0)

    a = tl.exp(-dt / tau)

    # sum Iasc over modes
    I_ptrs = tl.make_block_ptr(
        base=I_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    asc_ptrs = tl.make_block_ptr(
        base=asc_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )

    I_block = tl.load(I_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    I_sum = tl.sum(I_block, axis=1)

    v_inf = v_rest + tau * (x + I_sum) / c_m
    v_prime = v_inf + (v - v_inf) * a

    s = ((v_prime - v_th) > 0) * mask

    # reset
    if hard_reset:
        v_post = v_prime - (v_prime - v_reset) * s
    else:
        v_post = v_prime - (v_th - v_reset) * s

    # Iasc decay + jump
    k_block = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    asc_block = tl.load(asc_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    b = tl.exp(-k_block * dt)
    I_dec = I_block * b
    I_post = I_dec + asc_block * s[:, None]

    I_out_ptrs = tl.make_block_ptr(
        base=I_out_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    tl.store(I_out_ptrs, I_post.to(tl.float32), boundary_check=(0, 1))

    v_out_ptrs = tl.make_block_ptr(
        base=v_out_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    s_out_ptrs = tl.make_block_ptr(
        base=s_out_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tl.store(v_out_ptrs, v_post.to(tl.float32), boundary_check=(0,))
    tl.store(s_out_ptrs, s.to(tl.float32), boundary_check=(0,))


# ----------------------------
# Triton fused dense multistep (single-program, forward-only)
# ----------------------------
@triton.jit
def glif3_dense_fwd(
    x_ptr,
    w_ptr,
    b_ptr,
    s_prev_ptr,
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
    B,
    dt,
    alpha,
    T_valid,
    T: tl.constexpr,
    M: tl.constexpr,
    hard_reset: tl.constexpr,
    approx_a: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    inb = i < B

    v_ptrs = tl.make_block_ptr(
        base=v_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_th_ptrs = tl.make_block_ptr(
        base=v_th_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_reset_ptrs = tl.make_block_ptr(
        base=v_reset_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_rest_ptrs = tl.make_block_ptr(
        base=v_rest_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    c_m_ptrs = tl.make_block_ptr(
        base=c_m_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tau_ptrs = tl.make_block_ptr(
        base=tau_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    mask_ptrs = tl.make_block_ptr(
        base=mask_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    b_ptrs = tl.make_block_ptr(
        base=b_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    s_prev_ptrs = tl.make_block_ptr(
        base=s_prev_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )

    v = tl.load(v_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    v_th = tl.load(v_th_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    v_reset = tl.load(v_reset_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )
    v_rest = tl.load(v_rest_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )
    c_m = tl.load(c_m_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    c_m = tl.where(inb, c_m, 1.0)
    tau = tl.load(tau_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    tau = tl.where(inb, tau, 1.0)
    mask = tl.load(mask_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    mask = tl.where(inb, mask, 1.0)

    bias = tl.load(b_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    s_prev_vec = tl.load(s_prev_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )

    k_idx = tl.arange(0, BLOCK)
    in_k = k_idx < B

    w_ptrs = tl.make_block_ptr(
        base=w_ptr,
        shape=(B, B),
        strides=(B, 1),
        offsets=(0, 0),
        block_shape=(BLOCK, BLOCK),
        order=(1, 0),
    )
    I_ptrs = tl.make_block_ptr(
        base=I_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(0, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    I_out_ptrs = tl.make_block_ptr(
        base=I_out_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(0, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(0, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    asc_ptrs = tl.make_block_ptr(
        base=asc_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(0, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )

    I_block = tl.load(I_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    k_block = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    asc_block = tl.load(asc_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )

    for t in tl.static_range(0, T):
        do = t < T_valid
        x_ptrs = tl.make_block_ptr(
            base=x_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, 0),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        x = x[0, :]

        w = tl.load(w_ptrs, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        s_prev = tl.where(in_k, s_prev_vec, 0.0)
        lin = tl.sum(w * s_prev[None, :], axis=1)
        x_in = x + bias + lin

        I_sum = tl.sum(I_block, axis=1)

        a = tl.exp(-dt / tau)
        v_inf = v_rest + tau * (x_in + I_sum) / c_m
        v_prime = v_inf + (v - v_inf) * a

        s = ((v_prime - v_th) > 0) * mask
        s = tl.where(do, s, 0.0)
        if hard_reset:
            v_post = v_prime - (v_prime - v_reset) * s
        else:
            v_post = v_prime - (v_th - v_reset) * s

        b = tl.exp(-k_block * dt)
        I_dec = I_block * b
        I_post = I_dec + asc_block * s[:, None]
        I_post = tl.where(do, I_post, I_block)
        I_block = I_post
        tl.store(I_ptrs, I_post.to(tl.float32), boundary_check=(0, 1))
        tl.store(I_out_ptrs, I_post.to(tl.float32), boundary_check=(0, 1))

        v = tl.where(do, v_post, v)
        s_prev_vec = tl.where(do, s, s_prev_vec)
        v_store = tl.where(do, v_post, v)

        v_seq_ptrs = tl.make_block_ptr(
            base=v_seq_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, 0),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        s_seq_ptrs = tl.make_block_ptr(
            base=s_seq_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, 0),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        tl.store(v_seq_ptrs, v_store[None, :].to(tl.float32), boundary_check=(0, 1))
        tl.store(s_seq_ptrs, s[None, :].to(tl.float32), boundary_check=(0, 1))

    v_out_ptrs = tl.make_block_ptr(
        base=v_out_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tl.store(v_out_ptrs, v.to(tl.float32), boundary_check=(0,))


# ----------------------------
# Triton fused multistep only (single-program, forward-only)
# ----------------------------
@triton.jit
def glif3_multistep_fwd(
    x_ptr,
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
    B,
    dt,
    alpha,
    T_valid,
    T: tl.constexpr,
    M: tl.constexpr,
    hard_reset: tl.constexpr,
    approx_a: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK
    i = offs_b + tl.arange(0, BLOCK)
    inb = i < B

    v_ptrs = tl.make_block_ptr(
        base=v_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_th_ptrs = tl.make_block_ptr(
        base=v_th_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_reset_ptrs = tl.make_block_ptr(
        base=v_reset_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_rest_ptrs = tl.make_block_ptr(
        base=v_rest_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    c_m_ptrs = tl.make_block_ptr(
        base=c_m_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tau_ptrs = tl.make_block_ptr(
        base=tau_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    mask_ptrs = tl.make_block_ptr(
        base=mask_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )

    v = tl.load(v_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    v_th = tl.load(v_th_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    v_reset = tl.load(v_reset_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )
    v_rest = tl.load(v_rest_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )
    c_m = tl.load(c_m_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    c_m = tl.where(inb, c_m, 1.0)
    tau = tl.load(tau_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    tau = tl.where(inb, tau, 1.0)
    mask = tl.load(mask_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    mask = tl.where(inb, mask, 1.0)

    I_ptrs = tl.make_block_ptr(
        base=I_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    I_out_ptrs = tl.make_block_ptr(
        base=I_out_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    asc_ptrs = tl.make_block_ptr(
        base=asc_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )

    I_block = tl.load(I_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    k_block = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    asc_block = tl.load(asc_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )

    for t in tl.static_range(0, T):
        do = t < T_valid
        x_ptrs = tl.make_block_ptr(
            base=x_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, offs_b),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        x = x[0, :]

        I_sum = tl.sum(I_block, axis=1)

        a = tl.exp(-dt / tau)
        v_inf = v_rest + tau * (x + I_sum) / c_m
        v_prime = v_inf + (v - v_inf) * a

        s = ((v_prime - v_th) > 0) * mask
        s = tl.where(do, s, 0.0)
        if hard_reset:
            v_post = v_prime - (v_prime - v_reset) * s
        else:
            v_post = v_prime - (v_th - v_reset) * s

        b = tl.exp(-k_block * dt)
        I_dec = I_block * b
        I_post = I_dec + asc_block * s[:, None]
        I_post = tl.where(do, I_post, I_block)
        I_block = I_post
        tl.store(I_ptrs, I_post.to(tl.float32), boundary_check=(0, 1))
        tl.store(I_out_ptrs, I_post.to(tl.float32), boundary_check=(0, 1))

        v = tl.where(do, v_post, v)
        v_store = tl.where(do, v_post, v)
        v_seq_ptrs = tl.make_block_ptr(
            base=v_seq_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, offs_b),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        s_seq_ptrs = tl.make_block_ptr(
            base=s_seq_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, offs_b),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        tl.store(v_seq_ptrs, v_store[None, :].to(tl.float32), boundary_check=(0, 1))
        tl.store(s_seq_ptrs, s[None, :].to(tl.float32), boundary_check=(0, 1))

    v_out_ptrs = tl.make_block_ptr(
        base=v_out_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tl.store(v_out_ptrs, v.to(tl.float32), boundary_check=(0,))


# ----------------------------
# Triton fused multistep backward
# ----------------------------
@triton.jit
def glif3_multistep_bwd(
    x_ptr,
    v0_ptr,
    v_seq_ptr,
    s_seq_ptr,
    I_out_ptr,
    v_th_ptr,
    v_reset_ptr,
    v_rest_ptr,
    c_m_ptr,
    tau_ptr,
    k_ptr,
    asc_ptr,
    mask_ptr,
    ds_ptr,
    dv_seq_ptr,
    dv_out_ptr,
    dI_out_ptr,
    dv0_ptr,
    dI0_ptr,
    dx_ptr,
    dasc_ptr,
    B,
    dt,
    alpha,
    T_valid,
    T: tl.constexpr,
    M: tl.constexpr,
    hard_reset: tl.constexpr,
    approx_a: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK
    i = offs_b + tl.arange(0, BLOCK)
    inb = i < B

    v0_ptrs = tl.make_block_ptr(
        base=v0_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_th_ptrs = tl.make_block_ptr(
        base=v_th_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_reset_ptrs = tl.make_block_ptr(
        base=v_reset_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_rest_ptrs = tl.make_block_ptr(
        base=v_rest_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    c_m_ptrs = tl.make_block_ptr(
        base=c_m_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tau_ptrs = tl.make_block_ptr(
        base=tau_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    mask_ptrs = tl.make_block_ptr(
        base=mask_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )

    v0 = tl.load(v0_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    v_th = tl.load(v_th_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    v_reset = tl.load(v_reset_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )
    v_rest = tl.load(v_rest_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )
    c_m = tl.load(c_m_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    c_m = tl.where(inb, c_m, 1.0)
    tau = tl.load(tau_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    tau = tl.where(inb, tau, 1.0)
    mask = tl.load(mask_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    mask = tl.where(inb, mask, 1.0)

    a = tl.exp(-dt / tau)
    denom = v_th - v_reset
    tau_over_c = tau / c_m
    scale = 1.5707963267948966 * alpha

    k_ptrs = tl.make_block_ptr(
        base=k_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    asc_ptrs = tl.make_block_ptr(
        base=asc_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    I_post_ptrs = tl.make_block_ptr(
        base=I_out_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    dI_post_ptrs = tl.make_block_ptr(
        base=dI_out_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    dI0_ptrs = tl.make_block_ptr(
        base=dI0_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    dasc_ptrs = tl.make_block_ptr(
        base=dasc_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )

    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b = tl.exp(-k * dt)
    asc = tl.load(asc_ptrs, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    I_post = tl.load(I_post_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    dI_post = tl.load(dI_post_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    dasc_acc = tl.zeros([BLOCK, M], dtype=tl.float32)

    dv_out_ptrs = tl.make_block_ptr(
        base=dv_out_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    dv_post = tl.load(dv_out_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )

    for t in tl.static_range(T - 1, -1, -1):
        do = t < T_valid
        do_f = tl.where(do, 1.0, 0.0)
        ds_ptrs = tl.make_block_ptr(
            base=ds_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, offs_b),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        dv_seq_ptrs = tl.make_block_ptr(
            base=dv_seq_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, offs_b),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        s_seq_ptrs = tl.make_block_ptr(
            base=s_seq_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, offs_b),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        ds_t = tl.load(ds_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        ds_t = ds_t * do_f
        dv_t = tl.load(dv_seq_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        dv_t = dv_t * do_f
        s_t = tl.load(s_seq_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        s_t = s_t * do_f
        ds_t = ds_t[0, :]
        dv_t = dv_t[0, :]
        s_t = s_t[0, :]
        dv_post = dv_post + dv_t

        if t == 0:
            v_pre = v0
        else:
            v_seq_ptrs = tl.make_block_ptr(
                base=v_seq_ptr,
                shape=(T, B),
                strides=(B, 1),
                offsets=(t - 1, offs_b),
                block_shape=(1, BLOCK),
                order=(1, 0),
            )
            v_pre = tl.load(
                v_seq_ptrs, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
            v_pre = v_pre[0, :]

        x_ptrs = tl.make_block_ptr(
            base=x_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, offs_b),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        x_t = tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        x_t = x_t[0, :]
        x_t = x_t * do_f

        I_pre = (I_post - asc * s_t[:, None]) / b
        I_sum = tl.sum(I_pre, axis=1)

        v_inf = v_rest + tau_over_c * (x_t + I_sum)
        v_prime = v_inf + (v_pre - v_inf) * a

        u = (v_prime - v_th) / denom
        z = scale * u
        denom_z = 1.0 + approx_a * z * z
        f_prime = (1.0 - approx_a * z * z) / (denom_z * denom_z)
        ds_du = mask * (0.5 * alpha) * f_prime
        ds_dvprime = ds_du / denom

        if hard_reset:
            dvprime = dv_post * (1.0 - s_t)
            ds_from_v = dv_post * (-(v_prime - v_reset))
        else:
            dvprime = dv_post
            ds_from_v = dv_post * (-(v_th - v_reset))

        dI_s_sum = tl.sum(dI_post * asc, axis=1)
        ds_total = ds_t + ds_from_v + dI_s_sum
        dvprime = dvprime + ds_total * ds_dvprime

        dv_pre = dvprime * a
        dv_inf = dvprime * (1.0 - a)

        dx = dv_inf * tau_over_c
        dx_ptrs = tl.make_block_ptr(
            base=dx_ptr,
            shape=(T, B),
            strides=(B, 1),
            offsets=(t, offs_b),
            block_shape=(1, BLOCK),
            order=(1, 0),
        )
        tl.store(
            dx_ptrs,
            (dx * do_f)[None, :].to(tl.float32),
            boundary_check=(0, 1),
        )

        dI_common = dv_inf * tau_over_c
        dasc_acc += dI_post * s_t[:, None]
        dI_post_new = dI_post * b + dI_common[:, None]
        dI_post = tl.where(do, dI_post_new, dI_post)
        I_post = tl.where(do, I_pre, I_post)

        dv_post = tl.where(do, dv_pre, dv_post)

    dv0_ptrs = tl.make_block_ptr(
        base=dv0_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tl.store(dv0_ptrs, dv_post.to(tl.float32), boundary_check=(0,))
    tl.store(dI0_ptrs, dI_post.to(tl.float32), boundary_check=(0, 1))
    tl.store(dasc_ptrs, dasc_acc.to(tl.float32), boundary_check=(0, 1))


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
    B,
    dt,
    alpha,
    M: tl.constexpr,
    hard_reset: tl.constexpr,
    approx_a: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK
    i = offs_b + tl.arange(0, BLOCK)
    inb = i < B

    v_ptrs = tl.make_block_ptr(
        base=v_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    x_ptrs = tl.make_block_ptr(
        base=x_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_th_ptrs = tl.make_block_ptr(
        base=v_th_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_reset_ptrs = tl.make_block_ptr(
        base=v_reset_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    v_rest_ptrs = tl.make_block_ptr(
        base=v_rest_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    c_m_ptrs = tl.make_block_ptr(
        base=c_m_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tau_ptrs = tl.make_block_ptr(
        base=tau_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    mask_ptrs = tl.make_block_ptr(
        base=mask_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    s_ptrs = tl.make_block_ptr(
        base=s_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    ds_ptrs = tl.make_block_ptr(
        base=ds_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    dv_post_ptrs = tl.make_block_ptr(
        base=dv_post_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )

    v = tl.load(v_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    x = tl.load(x_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)

    v_th = tl.load(v_th_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    v_reset = tl.load(v_reset_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )
    v_rest = tl.load(v_rest_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )

    c_m = tl.load(c_m_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    c_m = tl.where(inb, c_m, 1.0)
    tau = tl.load(tau_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    tau = tl.where(inb, tau, 1.0)

    mask = tl.load(mask_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    mask = tl.where(inb, mask, 1.0)

    s = tl.load(s_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    dv_post = tl.load(dv_post_ptrs, boundary_check=(0,), padding_option="zero").to(
        tl.float32
    )

    a = tl.exp(-dt / tau)

    # recompute v_prime and u
    I_ptrs = tl.make_block_ptr(
        base=I_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    asc_ptrs = tl.make_block_ptr(
        base=asc_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    dI_post_ptrs = tl.make_block_ptr(
        base=dI_post_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    dI_ptrs = tl.make_block_ptr(
        base=dI_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )
    dasc_ptrs = tl.make_block_ptr(
        base=dasc_ptr,
        shape=(B, M),
        strides=(M, 1),
        offsets=(offs_b, 0),
        block_shape=(BLOCK, M),
        order=(1, 0),
    )

    I_block = tl.load(I_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    I_sum = tl.sum(I_block, axis=1)

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
    dI_post = tl.load(dI_post_ptrs, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    asc = tl.load(asc_ptrs, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dI_s_sum = tl.sum(dI_post * asc, axis=1)
    # grad wrt asc_amps: dL/dasc = dI_post * s
    dasc = dI_post * s[:, None]
    tl.store(dasc_ptrs, dasc.to(tl.float32), boundary_check=(0, 1))

    ds_out = tl.load(ds_ptrs, boundary_check=(0,), padding_option="zero").to(tl.float32)
    ds_total = ds_from_v + dI_s_sum + ds_out
    dvprime = dvprime + ds_total * ds_dvprime

    # Backprop v_prime = a*v + (1-a)*v_inf
    dv = dvprime * a
    dv_inf = dvprime * (1.0 - a)

    # v_inf = v_rest + tau*(x + I_sum)/c_m
    dx = dv_inf * (tau / c_m)
    dI_common = dv_inf * (tau / c_m)

    # Iasc decay + add dI_common
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b = tl.exp(-k * dt)
    dI = dI_post * b + dI_common[:, None]
    tl.store(dI_ptrs, dI.to(tl.float32), boundary_check=(0, 1))

    dv_ptrs = tl.make_block_ptr(
        base=dv_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    dx_ptrs = tl.make_block_ptr(
        base=dx_ptr,
        shape=(B,),
        strides=(1,),
        offsets=(offs_b,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tl.store(dv_ptrs, dv.to(tl.float32), boundary_check=(0,))
    tl.store(dx_ptrs, dx.to(tl.float32), boundary_check=(0,))


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
            B,
            float(dt),
            float(alpha),
            M=int(M),
            hard_reset=1 if hard_reset else 0,
            approx_a=_ATAN_APPROX_A,
            BLOCK=block,
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
            B,
            float(ctx.dt),
            float(ctx.alpha),
            M=M,
            hard_reset=1 if ctx.hard_reset else 0,
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
    unroll: int = 16,
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

    T_total = x_seq.shape[0]
    unroll = int(unroll)
    if unroll <= 0:
        raise ValueError("unroll must be a positive integer.")
    v_seq = torch.empty((T_total, B), device=v.device, dtype=v.dtype)
    s_seq = torch.empty((T_total, B), device=v.device, dtype=v.dtype)

    if B > block:
        raise RuntimeError("Fused multistep Triton requires B <= block.")

    x_seq = _as_fp32(x_seq, "x_seq")
    weight = _as_fp32(weight, "weight")
    bias = _as_fp32(bias, "bias")
    v_out = torch.empty_like(v)
    I_out = torch.empty_like(Iasc)
    s_prev = torch.zeros((B,), device=v.device, dtype=v.dtype)
    v_work = v
    I_work = Iasc

    num_chunks = (T_total + unroll - 1) // unroll
    for chunk in range(num_chunks):
        t0 = chunk * unroll
        t1 = min(t0 + unroll, T_total)
        t_valid = t1 - t0

        x_chunk = torch.zeros((unroll, B), device=v.device, dtype=v.dtype)
        x_chunk[:t_valid] = x_seq[t0:t1]
        s_chunk = torch.empty((unroll, B), device=v.device, dtype=v.dtype)
        v_chunk = torch.empty((unroll, B), device=v.device, dtype=v.dtype)

        glif3_dense_fwd[(1,)](
            x_chunk,
            weight,
            bias,
            s_prev,
            v_work,
            I_work,
            v_th,
            v_reset,
            v_rest,
            c_m,
            tau,
            k,
            asc_amps,
            not_refrac,
            s_chunk,
            v_chunk,
            v_out,
            I_out,
            B,
            float(dt),
            float(alpha),
            t_valid,
            T=unroll,
            M=int(M),
            hard_reset=1 if hard_reset else 0,
            approx_a=_ATAN_APPROX_A,
            BLOCK=int(block),
            num_warps=4,
        )

        s_seq[t0:t1] = s_chunk[:t_valid]
        v_seq[t0:t1] = v_chunk[:t_valid]
        if t_valid > 0:
            s_prev = s_chunk[t_valid - 1].contiguous()
        v_work = v_out

    return s_seq, v_seq, v_out, I_out.view(B, M)


glif3_step_triton.dense_multistep_fused = glif3_dense_multistep_fused_triton


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
    unroll: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return GLIF3MultiStepTriton.apply(
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
        int(block),
        int(unroll),
    )


class GLIF3MultiStepTriton(torch.autograd.Function):
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
        unroll: int = 16,
    ):
        if not x_seq.is_cuda:
            raise RuntimeError("Fused multistep Triton requires CUDA tensors.")

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

        T_total, B = x_seq.shape
        unroll = int(unroll)
        if unroll <= 0:
            raise ValueError("unroll must be a positive integer.")

        ctx.Iasc_was_flat = Iasc.ndim == 1
        ctx.asc_was_flat = asc_amps.ndim == 1
        if Iasc.ndim == 1 and Iasc.numel() != B * int(M):
            raise ValueError("Iasc must have shape (B, M) for multistep.")
        Iasc_flat = Iasc.reshape(-1) if Iasc.ndim == 2 else Iasc

        if v.requires_grad or Iasc.requires_grad or x_seq.requires_grad:
            v_work = v.clone()
            I_work = Iasc_flat.clone()
        else:
            v_work = v
            I_work = Iasc_flat

        v_seq = torch.empty((T_total, B), device=v.device, dtype=v.dtype)
        s_seq = torch.empty((T_total, B), device=v.device, dtype=v.dtype)

        num_chunks = (T_total + unroll - 1) // unroll
        I_out_chunks = torch.empty(
            (num_chunks, B * int(M)), device=v.device, dtype=v.dtype
        )

        grid = (triton.cdiv(B, int(block)),)
        for chunk in range(num_chunks):
            t0 = chunk * unroll
            t1 = min(t0 + unroll, T_total)
            t_valid = t1 - t0

            x_chunk = torch.zeros((unroll, B), device=v.device, dtype=v.dtype)
            x_chunk[:t_valid] = x_seq[t0:t1]
            s_chunk = torch.empty((unroll, B), device=v.device, dtype=v.dtype)
            v_chunk = torch.empty((unroll, B), device=v.device, dtype=v.dtype)
            v_out = torch.empty_like(v_work)
            I_out = I_out_chunks[chunk]

            glif3_multistep_fwd[grid](
                x_chunk,
                v_work,
                I_work,
                v_th,
                v_reset,
                v_rest,
                c_m,
                tau,
                k,
                asc_amps,
                not_refrac,
                s_chunk,
                v_chunk,
                v_out,
                I_out,
                B,
                float(dt),
                float(alpha),
                t_valid,
                T=unroll,
                M=int(M),
                hard_reset=1 if hard_reset else 0,
                approx_a=_ATAN_APPROX_A,
                BLOCK=int(block),
                num_warps=4,
            )

            s_seq[t0:t1] = s_chunk[:t_valid]
            v_seq[t0:t1] = v_chunk[:t_valid]
            v_work = v_out

        ctx.save_for_backward(
            x_seq,
            v,
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
            I_out_chunks,
        )
        ctx.dt = float(dt)
        ctx.M = int(M)
        ctx.hard_reset = bool(hard_reset)
        ctx.alpha = float(alpha)
        ctx.block = int(block)
        ctx.unroll = int(unroll)
        v_out_final = v_work
        I_out_final = I_out_chunks[-1]
        return s_seq, v_seq, v_out_final, I_out_final.view(B, M)

    @staticmethod
    def backward(ctx, ds_seq, dv_seq, dv_out, dI_out):
        (
            x_seq,
            v0,
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
            I_out_chunks,
        ) = ctx.saved_tensors

        T_total, B = x_seq.shape
        M = ctx.M
        block = ctx.block
        unroll = ctx.unroll

        if ds_seq is None:
            ds_seq = torch.zeros_like(s_seq)
        if dv_seq is None:
            dv_seq = torch.zeros_like(v_seq)
        if dv_out is None:
            dv_out = torch.zeros_like(v0)
        if dI_out is None:
            dI_out = torch.zeros_like(I_out_chunks[-1])

        ds_seq = _as_fp32(ds_seq, "ds_seq")
        dv_seq = _as_fp32(dv_seq, "dv_seq")
        dv_out = _as_fp32(dv_out, "dv_out")
        dI_out = _as_fp32(dI_out, "dI_out")

        dx_seq = torch.zeros_like(x_seq)
        d_asc_acc = torch.zeros((B * M,), device=v0.device, dtype=v0.dtype)
        dI_next = dI_out.reshape(-1)
        dv_next = dv_out

        num_chunks = (T_total + unroll - 1) // unroll
        grid = (triton.cdiv(B, int(block)),)
        for chunk in range(num_chunks - 1, -1, -1):
            t0 = chunk * unroll
            t1 = min(t0 + unroll, T_total)
            t_valid = t1 - t0

            x_chunk = torch.zeros((unroll, B), device=v0.device, dtype=v0.dtype)
            x_chunk[:t_valid] = x_seq[t0:t1]
            s_chunk = torch.zeros((unroll, B), device=v0.device, dtype=v0.dtype)
            v_chunk = torch.zeros((unroll, B), device=v0.device, dtype=v0.dtype)
            ds_chunk = torch.zeros((unroll, B), device=v0.device, dtype=v0.dtype)
            dv_chunk = torch.zeros((unroll, B), device=v0.device, dtype=v0.dtype)
            s_chunk[:t_valid] = s_seq[t0:t1]
            v_chunk[:t_valid] = v_seq[t0:t1]
            ds_chunk[:t_valid] = ds_seq[t0:t1]
            dv_chunk[:t_valid] = dv_seq[t0:t1]

            if t0 == 0:
                v0_chunk = v0
            else:
                v0_chunk = v_seq[t0 - 1].contiguous()

            I_out = I_out_chunks[chunk]
            dv0 = torch.empty_like(v0)
            dI0 = torch.empty_like(I_out)
            d_asc = torch.empty_like(I_out)
            dx_chunk = torch.empty((unroll, B), device=v0.device, dtype=v0.dtype)

            glif3_multistep_bwd[grid](
                x_chunk,
                v0_chunk,
                v_chunk,
                s_chunk,
                I_out,
                v_th,
                v_reset,
                v_rest,
                c_m,
                tau,
                k,
                asc_amps,
                not_refrac,
                ds_chunk,
                dv_chunk,
                dv_next,
                dI_next,
                dv0,
                dI0,
                dx_chunk,
                d_asc,
                B,
                float(ctx.dt),
                float(ctx.alpha),
                t_valid,
                T=unroll,
                M=int(M),
                hard_reset=1 if ctx.hard_reset else 0,
                approx_a=_ATAN_APPROX_A,
                BLOCK=int(block),
                num_warps=4,
            )

            dx_seq[t0:t1] = dx_chunk[:t_valid]
            d_asc_acc += d_asc
            dv_next = dv0
            dI_next = dI0

        if not ctx.Iasc_was_flat:
            dI_next = dI_next.view(B, M)
        if not ctx.asc_was_flat:
            d_asc_acc = d_asc_acc.view(B, M)

        return (
            dx_seq,
            dv_next,
            dI_next,
            None,
            None,
            None,
            None,
            None,
            None,
            d_asc_acc,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


glif3_step_triton.multistep_fused = glif3_multistep_fused_triton


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
