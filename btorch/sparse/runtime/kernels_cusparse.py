"""Prepare invocation-owned CSR products through PyTorch's public CUDA API."""

import math

import torch

from ...utils.kernel import tensor_layout
from . import kernels_torch
from .activity import GradientList


def prepare_matvec(crow, col, values, x):
    """Bind CSR tensors and reusable dense storage for one invocation.

    Values remain fixed within this invocation. A subsequent preparation reads
    current optimizer values and topology. Sample and value batches preserve
    the same broadcasting and promoted precision as the ordinary CSR product.
    Public ``out=`` products retain this precision under autocast.
    """
    if not values.is_cuda or not x.is_cuda:
        raise ValueError("The cusparse backend requires CUDA tensors")
    n_vb = values.ndim - 1
    if x.ndim <= n_vb:
        raise ValueError("Sparse input is missing its neuron dimension")
    dtype = torch.promote_types(values.dtype, x.dtype)
    if dtype not in kernels_torch._FLOAT_DTYPES:
        raise ValueError("The cusparse backend requires floating-point products")
    batch = torch.broadcast_shapes(values.shape[:-1], x.shape[:n_vb])
    sample = x.shape[n_vb:-1]
    members, samples = math.prod(batch), math.prod(sample)
    n_in, n_out, n_edge = x.shape[-1], crow.numel() - 1, values.shape[-1]
    weights = values.to(dtype).expand(*batch, n_edge).reshape(members, n_edge)
    matrices = tuple(
        torch.sparse_csr_tensor(
            crow,
            col,
            weights[index].contiguous(),
            size=(n_out, n_in),
            check_invariants=False,
        )
        for index in range(members)
    )
    result = torch.empty((*batch, *sample, n_out), device=x.device, dtype=dtype)
    minor = (
        result.view(members, n_out, samples)
        if samples == 1
        else torch.empty((members, n_out, samples), device=x.device, dtype=dtype)
    )
    source = (
        x.expand(*batch, *sample, n_in).reshape(members, samples, n_in).transpose(1, 2)
    )
    rhs = (
        None
        if source.is_contiguous() and source.dtype == dtype
        else torch.empty((members, n_in, samples), device=x.device, dtype=dtype)
    )
    layout = tensor_layout(x)

    def forward(input):
        if tensor_layout(input) != layout:
            raise ValueError(
                "Sparse input layout/dtype changed; prepare a new execution"
            )
        operand = (
            input.expand(*batch, *sample, n_in)
            .reshape(members, samples, n_in)
            .transpose(1, 2)
        )
        if rhs is not None:
            rhs.copy_(operand)
            operand = rhs
        for index, matrix in enumerate(matrices):
            torch.mm(matrix, operand[index], out=minor[index])
        if samples != 1:
            result.view(members, samples, n_out).copy_(minor.transpose(1, 2))
        return result

    return forward


def prepare_backward(
    crow,
    col,
    t_crow,
    t_col,
    t_perm,
    values,
    x,
    grad,
    need_values=True,
    need_x=True,
    *,
    input_mask=None,
    grad_values=None,
):
    """Prepare transposed CSR once, retaining exact stored-edge VJPs."""
    input_backward = (
        prepare_matvec(t_crow, t_col, values.index_select(-1, t_perm), grad)
        if need_x
        else None
    )
    input_layout = tensor_layout(x)

    def backward(input, gradient, mask=None):
        if tensor_layout(input) != input_layout:
            raise ValueError(
                "Sparse input layout/dtype changed; prepare a new execution"
            )
        value_gradient = (
            kernels_torch.edge_grad(crow, col, gradient, input, values.ndim - 1)
            if need_values
            else None
        )
        if grad_values is not None and value_gradient is not None:
            grad_values.add_(value_gradient.sum_to_size(grad_values.shape))
            value_gradient = None
        grad_x = input_backward(gradient) if input_backward is not None else None
        activity = input_mask if mask is None else mask
        if activity is not None and grad_x is not None:
            activity = activity.mask if isinstance(activity, GradientList) else activity
            grad_x.mul_(activity.reshape(grad_x.shape))
        return value_gradient, grad_x

    return backward


def backward(crow, col, t_crow, t_col, t_perm, values, x, grad, need_values, need_x):
    """Compute a one-off VJP without allocating recurrent workspaces.

    Repeated recurrent execution uses :func:`prepare_backward` instead, so
    its CSR tensors and transposed values are built only once.
    """
    return (
        kernels_torch.edge_grad(crow, col, grad, x, values.ndim - 1)
        if need_values
        else None,
        kernels_torch.csr_matvec_cusparse(
            t_crow, t_col, values.index_select(-1, t_perm), grad
        )
        if need_x
        else None,
    )
