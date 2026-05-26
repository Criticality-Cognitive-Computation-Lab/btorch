from __future__ import annotations

import torch
import triton

from .lif_kernels import lif_single_step_fwd_kernel


def triton_lif_single_step(
    x: torch.Tensor,
    v: torch.Tensor,
    v_threshold: torch.Tensor,
    v_reset: torch.Tensor,
    c_m: torch.Tensor,
    tau: torch.Tensor,
    *,
    dt: float,
    hard_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one fused LIF update on flattened CUDA tensors."""
    if not x.is_cuda:
        raise ValueError("triton_lif_single_step requires CUDA tensors.")
    if x.ndim != 1:
        raise ValueError("x must be flattened to shape (numel,).")
    if x.shape != v.shape:
        raise ValueError("x and v must have identical flattened shapes.")

    x = x.contiguous()
    v = v.contiguous()
    v_threshold = v_threshold.contiguous()
    v_reset = v_reset.contiguous()
    c_m = c_m.contiguous()
    tau = tau.contiguous()

    spikes = torch.empty_like(x)
    v_out = torch.empty_like(v)

    numel = x.numel()
    block_size = 256
    grid = (triton.cdiv(numel, block_size),)

    lif_single_step_fwd_kernel[grid](
        x,
        spikes,
        v,
        v_threshold,
        v_reset,
        c_m,
        tau,
        v_out,
        numel,
        dt,
        BLOCK_SIZE=block_size,
        HARD_RESET=hard_reset,
        num_warps=4,
    )
    return spikes, v_out
