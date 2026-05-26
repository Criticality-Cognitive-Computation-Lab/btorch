from __future__ import annotations

import torch
import triton

from .event_kernels import post_span_forward_kernel, pre_span_forward_kernel


def _validate_event_inputs(
    spike: torch.Tensor,
    row_length: torch.Tensor,
    ind: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not spike.is_cuda:
        raise ValueError("Event sparse Triton kernels require CUDA tensors.")
    if spike.ndim != 2:
        raise ValueError("spike must have shape (batch_size, n_pre).")
    if row_length.ndim != 1:
        raise ValueError("row_length must be 1D.")
    if ind.ndim != 2 or weight.ndim != 2:
        raise ValueError("ind and weight must be 2D tensors.")
    if ind.shape != weight.shape:
        raise ValueError("ind and weight must have identical shapes.")
    if row_length.shape[0] != ind.shape[0]:
        raise ValueError("row_length length must match the number of rows in ind.")
    if spike.dtype != weight.dtype:
        weight = weight.to(dtype=spike.dtype)
    if row_length.device != spike.device:
        row_length = row_length.to(device=spike.device)
    if ind.device != spike.device:
        ind = ind.to(device=spike.device)
    if weight.device != spike.device:
        weight = weight.to(device=spike.device)

    spike = spike.contiguous()
    row_length = row_length.contiguous()
    ind = ind.contiguous()
    weight = weight.contiguous()
    return spike, row_length, ind, weight


def pre_span_spmm(
    spike: torch.Tensor,
    row_length: torch.Tensor,
    ind: torch.Tensor,
    weight: torch.Tensor,
    *,
    size_m: int,
    is_bool_float: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a presynaptic-span sparse event propagation kernel."""
    spike, row_length, ind, weight = _validate_event_inputs(
        spike, row_length, ind, weight
    )
    batch_size, n_pre = spike.shape
    row_stride = ind.shape[1]
    if row_length.shape[0] != n_pre:
        raise ValueError("pre-span row_length must have one row per presynaptic neuron.")

    if out is None:
        out = torch.zeros(
            (batch_size, size_m), device=spike.device, dtype=spike.dtype
        )
    else:
        if out.shape != (batch_size, size_m):
            raise ValueError("out must have shape (batch_size, size_m).")
        if out.device != spike.device:
            raise ValueError("out must be on the same device as spike.")
        if out.dtype != spike.dtype:
            raise ValueError("out must have the same dtype as spike.")
        out = out.contiguous()
        out.zero_()

    grid = (batch_size, n_pre)
    pre_span_forward_kernel[grid](
        spike,
        row_length,
        ind,
        weight,
        out,
        spike.stride(0),
        spike.stride(1),
        ind.stride(0),
        ind.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        n_pre,
        row_stride,
        IS_BOOL_FLOAT=is_bool_float,
        ROW_STRIDE=row_stride,
        num_warps=1,
    )
    return out


def post_span_spmm(
    spike: torch.Tensor,
    row_length: torch.Tensor,
    ind: torch.Tensor,
    weight: torch.Tensor,
    *,
    size_m: int,
    is_bool_float: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a postsynaptic-span sparse event propagation kernel."""
    spike, row_length, ind, weight = _validate_event_inputs(
        spike, row_length, ind, weight
    )
    batch_size, n_pre = spike.shape
    row_stride = ind.shape[1]
    if row_length.shape[0] != n_pre:
        raise ValueError(
            "post-span row_length must have one row per presynaptic neuron."
        )

    if out is None:
        out = torch.zeros(
            (batch_size, size_m), device=spike.device, dtype=spike.dtype
        )
    else:
        if out.shape != (batch_size, size_m):
            raise ValueError("out must have shape (batch_size, size_m).")
        if out.device != spike.device:
            raise ValueError("out must be on the same device as spike.")
        if out.dtype != spike.dtype:
            raise ValueError("out must have the same dtype as spike.")
        out = out.contiguous()
        out.zero_()

    block_slot = min(max(triton.next_power_of_2(row_stride), 1), 256)
    grid = (batch_size, triton.cdiv(row_stride, block_slot))
    post_span_forward_kernel[grid](
        spike,
        row_length,
        ind,
        weight,
        out,
        spike.stride(0),
        spike.stride(1),
        ind.stride(0),
        ind.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        n_pre,
        row_stride,
        IS_BOOL_FLOAT=is_bool_float,
        BLOCK_SLOT=block_slot,
        num_warps=1,
    )
    return out
