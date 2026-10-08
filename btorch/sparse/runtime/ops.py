"""Registered custom operators: the ``torch.compile`` boundary.

Model code hands raw tensor buffers to these operators; everything behind
them (representation, algorithm, backend) is opaque to Dynamo and to autograd,
which see one node with an explicit backward. The operators therefore compile
with ``fullgraph=True`` and never expose PyTorch sparse tensors to the tracer.

All operators share one layout: the operator ``A`` (``M x N``) as
destination-major CSR ``(crow, col)`` with ``values [*vb, E]``, plus the
source-major CSR of the same edges ``(t_crow, t_col, t_perm)`` used by the
backward and by source-driven kernels. Inputs are dense ``x [*vb, *sample,
N]``; the logical spike tensor stays dense so surrogate gradients are exact.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.library import custom_op

from . import kernels_aten, kernels_triton
from .backend import registry


registry.register("csr_matvec", "aten", kernels_aten.csr_matvec)
registry.register("edge_grad", "aten", kernels_aten.edge_grad)
registry.register("spike_push", "aten", kernels_aten.spike_push)
registry.register(
    "csr_matvec",
    "torch_sparse",
    kernels_aten.csr_matvec_torch_sparse,
    priority=-1,
    available=kernels_aten.torch_sparse_available,
)


# Optional CUDA backends; each checks its own availability lazily.
kernels_triton.register(registry)


def _out_shape(values: Tensor, x: Tensor, n_out: int) -> tuple[int, ...]:
    n_vb = values.ndim - 1
    batch = torch.broadcast_shapes(values.shape[:-1], x.shape[:n_vb])
    return (*batch, *x.shape[n_vb:-1], n_out)


def _kernel(name: str, x: Tensor):
    return registry.resolve(name, x.device.type)


def _no_autocast(x: Tensor):
    """Kernels run in the dtypes they are given, independent of ambient AMP.

    Otherwise an autocast region would change the output dtype of the
    eager kernel but not of its traced (fake) counterpart.
    """
    return torch.autocast(device_type=x.device.type, enabled=False)


def _check_real(values: Tensor, x: Tensor) -> None:
    if values.is_complex() or x.is_complex():
        raise TypeError("Sparse propagation operators support real dtypes only.")


def _fake_out(values: Tensor, x: Tensor, n_out: int) -> Tensor:
    _check_real(values, x)
    dtype = torch.promote_types(values.dtype, x.dtype)
    return x.new_empty(_out_shape(values, x, n_out), dtype=dtype)


# --------------------------------------------------------------- raw kernels
# Not differentiable themselves; they are the building blocks of the backward.


@custom_op("btorch::csr_matvec", mutates_args=())
def csr_matvec(crow: Tensor, col: Tensor, values: Tensor, x: Tensor) -> Tensor:
    """``y[..., m] = sum_{k in row m} values[..., k] * x[..., col[k]]``."""
    with _no_autocast(x):
        return _kernel("csr_matvec", x)(crow, col, values, x)


@csr_matvec.register_fake
def _(crow, col, values, x):
    return _fake_out(values, x, crow.shape[0] - 1)


@custom_op("btorch::csr_edge_grad", mutates_args=())
def csr_edge_grad(
    crow: Tensor, col: Tensor, grad: Tensor, x: Tensor, n_vb: int
) -> Tensor:
    """Per-entry gradient ``sum_samples grad[..., row[k]] * x[...,
    col[k]]``."""
    with _no_autocast(x):
        return _kernel("edge_grad", x)(crow, col, grad, x, n_vb)


@csr_edge_grad.register_fake
def _(crow, col, grad, x, n_vb):
    return grad.new_empty(*grad.shape[:n_vb], col.shape[0])


# ------------------------------------------------------ differentiable ops


def _setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs[:7])


def _backward(ctx, grad: Tensor):
    crow, col, values, x, t_crow, t_col, t_perm = ctx.saved_tensors
    grad_values = grad_x = None
    if ctx.needs_input_grad[2]:
        n_vb = values.ndim - 1
        grad_values = (
            csr_edge_grad(crow, col, grad, x, n_vb)
            .sum_to_size(values.shape)
            .to(values.dtype)
        )
    if ctx.needs_input_grad[3]:
        # Transposed product as a pull over the source-major CSR: no atomics,
        # deterministic, and always a dense gradient w.r.t. the input.
        grad_x = (
            csr_matvec(t_crow, t_col, values.index_select(-1, t_perm), grad)
            .sum_to_size(x.shape)
            .to(x.dtype)
        )
    return (None, None, grad_values, grad_x, None, None, None) + (None,) * (
        len(ctx.needs_input_grad) - 7
    )


@custom_op("btorch::csr_propagate", mutates_args=())
def csr_propagate(
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
) -> Tensor:
    """Destination-driven sparse propagation ``y = A @ x`` along the last axis.

    Args:
        crow: ``[M + 1]`` row pointers of the destination-major CSR.
        col: ``[E]`` source index of each entry.
        values: ``[*vb, E]`` entry values in CSR order (differentiable).
        x: ``[*vb, *sample, N]`` dense input (differentiable).
        t_crow: ``[N + 1]`` pointers of the source-major CSR.
        t_col: ``[E]`` destination index of each source-major entry.
        t_perm: ``[E]`` CSR position of each source-major entry.

    Returns:
        ``[*vb, *sample, M]``.
    """
    _check_real(values, x)
    with _no_autocast(x):
        return _kernel("csr_matvec", x)(crow, col, values, x)


@csr_propagate.register_fake
def _(crow, col, values, x, t_crow, t_col, t_perm):
    return _fake_out(values, x, crow.shape[0] - 1)


csr_propagate.register_autograd(_backward, setup_context=_setup_context)


@custom_op("btorch::spike_propagate", mutates_args=())
def spike_propagate(
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    max_density: float,
) -> Tensor:
    """Sparse propagation that exploits sparse activity in ``x``.

    Mathematically identical to :func:`csr_propagate`. The forward packs the
    non-zero entries of the dense input and visits only the out-edges of
    active sources when the input density is at most ``max_density``;
    otherwise it falls back to the destination-driven product. The backward
    is the same dense one, so gradients w.r.t. ``x`` are exact for every
    entry, including the silent ones.

    Args:
        crow: ``[M + 1]`` row pointers of the destination-major CSR.
        col: ``[E]`` source index of each entry.
        values: ``[E]`` entry values in CSR order (differentiable).
        x: ``[*sample, N]`` dense input (differentiable).
        t_crow: ``[N + 1]`` pointers of the source-major CSR.
        t_col: ``[E]`` destination index of each source-major entry.
        t_perm: ``[E]`` CSR position of each source-major entry.
        max_density: Largest non-zero fraction of ``x`` handled by the
            source-driven path.

    Returns:
        ``[*sample, M]``.
    """
    n_out = crow.shape[0] - 1
    _check_real(values, x)
    with _no_autocast(x):
        if values.ndim == 1 and x.numel() > 0:
            flat = x.reshape(-1, x.shape[-1])
            active_idx, ptr = kernels_aten.pack_spikes(flat)
            if active_idx.shape[0] <= max_density * flat.numel():
                push = _kernel("spike_push", x)
                out = push(t_crow, t_col, t_perm, values, flat, active_idx, ptr, n_out)
                dtype = torch.promote_types(values.dtype, x.dtype)
                return out.to(dtype).reshape(*x.shape[:-1], n_out)
        return _kernel("csr_matvec", x)(crow, col, values, x)


@spike_propagate.register_fake
def _(crow, col, values, x, t_crow, t_col, t_perm, max_density):
    return _fake_out(values, x, crow.shape[0] - 1)


spike_propagate.register_autograd(_backward, setup_context=_setup_context)


def pack_spikes(x: Tensor) -> tuple[Tensor, Tensor]:
    """Pack the non-zero entries of a dense spike tensor ``[B, N]``.

    The packed indices are an execution aid only: the dense tensor remains
    the logical (and differentiable) spike representation.

    Returns:
        ``(active_idx, ptr)``; sample ``b`` owns
        ``active_idx[ptr[b]:ptr[b + 1]]``.
    """
    return kernels_aten.pack_spikes(x)
