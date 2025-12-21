"""Tools for Hessian-based sloppiness (parameter sensitivity) analysis."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import nn


Tensor = torch.Tensor


def _normalize_params(params: Union[nn.Module, Iterable[Tensor]]) -> Sequence[Tensor]:
    if isinstance(params, nn.Module):
        params = list(params.parameters())
    else:
        params = list(params)
    params = [p for p in params if p.requires_grad]
    if not params:
        raise ValueError("No parameters with requires_grad=True were provided.")
    return params


def _flatten_tensors(tensors: Sequence[Tensor]) -> Tensor:
    if not tensors:
        return torch.tensor([])
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten_vector(vec: Tensor, params: Sequence[Tensor]) -> Sequence[Tensor]:
    if vec.dim() != 1:
        raise ValueError("Vector input must be 1D.")
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]
    if sum(sizes) != vec.numel():
        raise ValueError("Vector size does not match parameter sizes.")
    chunks = vec.split(sizes)
    return [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]


def hessian_vector_product(
    loss: Tensor,
    params: Union[nn.Module, Iterable[Tensor]],
    vec: Union[Tensor, Sequence[Tensor]],
    *,
    retain_graph: bool = False,
    return_list: bool = False,
) -> Union[Tensor, Sequence[Tensor]]:
    """Compute the Hessian-vector product for a scalar loss."""
    params = _normalize_params(params)
    if isinstance(vec, (list, tuple)):
        vec_list = vec
        vec_flat = _flatten_tensors(vec_list)
    else:
        vec_flat = vec
        vec_list = _unflatten_vector(vec_flat, params)

    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    hvp = torch.autograd.grad(
        grads,
        params,
        grad_outputs=vec_list,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    hvp = [torch.zeros_like(p) if g is None else g for g, p in zip(hvp, params)]

    if return_list:
        return hvp
    return _flatten_tensors(hvp)


def hessian_matrix(
    loss_fn: Callable[[], Tensor],
    params: Union[nn.Module, Iterable[Tensor]],
) -> Tensor:
    """Compute the full Hessian matrix (expensive; intended for small
    models)."""
    params = _normalize_params(params)
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
    flat_grads = _flatten_tensors(grads)
    n_params = flat_grads.numel()

    cols = []
    for i in range(n_params):
        grad_i = torch.autograd.grad(
            flat_grads[i], params, retain_graph=True, allow_unused=True
        )
        grad_i = [
            torch.zeros_like(p) if g is None else g for g, p in zip(grad_i, params)
        ]
        cols.append(_flatten_tensors(grad_i))
    return torch.stack(cols, dim=1)


def _matmat(matvec: Callable[[Tensor], Tensor], mat: Tensor) -> Tensor:
    cols = [matvec(mat[:, i]) for i in range(mat.shape[1])]
    return torch.stack(cols, dim=1)


def _randomized_symmetric_eig(
    matvec: Callable[[Tensor], Tensor],
    n_params: int,
    k: int,
    *,
    oversample: int = 5,
    n_iter: int = 2,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Randomized eigendecomposition for symmetric operators."""
    sample_dim = min(n_params, k + oversample)
    omega = torch.randn(n_params, sample_dim, device=device, dtype=dtype)
    y = _matmat(matvec, omega)

    for _ in range(max(n_iter, 0)):
        y = _matmat(matvec, y)
        y, _ = torch.linalg.qr(y, mode="reduced")

    q, _ = torch.linalg.qr(y, mode="reduced")
    aq = _matmat(matvec, q)
    b = q.T @ aq

    evals, evecs_small = torch.linalg.eigh(b)
    evals, evecs_small = evals.flip(0), evecs_small.flip(1)
    evecs = q @ evecs_small

    return evals[:k], evecs[:, :k]


def sloppiness_spectrum_from_hessian(
    hessian: Tensor,
    *,
    k: Optional[int] = None,
    return_vectors: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Eigen-spectrum from an explicit Hessian matrix."""
    if hessian.dim() != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be a square matrix.")
    evals, evecs = torch.linalg.eigh(hessian)
    evals, evecs = evals.flip(0), evecs.flip(1)

    if k is not None:
        evals = evals[:k]
        evecs = evecs[:, :k]

    if return_vectors:
        return evals, evecs
    return evals


def sloppiness_spectrum_from_matvec(
    matvec: Callable[[Tensor], Tensor],
    n_params: int,
    *,
    k: int = 10,
    oversample: int = 5,
    n_iter: int = 2,
    return_vectors: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Eigen-spectrum from a Hessian-vector product function."""
    k = min(k, n_params)
    evals, evecs = _randomized_symmetric_eig(
        matvec,
        n_params,
        k,
        oversample=oversample,
        n_iter=n_iter,
        device=device,
        dtype=dtype,
    )
    if return_vectors:
        return evals, evecs
    return evals


def sloppiness_spectrum(
    loss_fn: Callable[[], Tensor],
    params: Union[nn.Module, Iterable[Tensor]],
    *,
    k: int = 10,
    method: str = "randomized",
    oversample: int = 5,
    n_iter: int = 2,
    return_vectors: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute the top Hessian eigenvalues (and vectors) for sloppiness
    analysis."""
    params = _normalize_params(params)
    n_params = sum(p.numel() for p in params)
    k = min(k, n_params)

    if method == "exact":
        hess = hessian_matrix(loss_fn, params)
        return sloppiness_spectrum_from_hessian(
            hess, k=k, return_vectors=return_vectors
        )
    if method != "randomized":
        raise ValueError("method must be 'randomized' or 'exact'.")

    def matvec(vec: Tensor) -> Tensor:
        loss = loss_fn()
        return hessian_vector_product(loss, params, vec, retain_graph=False)

    device = params[0].device
    dtype = params[0].dtype
    return sloppiness_spectrum_from_matvec(
        matvec,
        n_params,
        k=k,
        oversample=oversample,
        n_iter=n_iter,
        return_vectors=return_vectors,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "hessian_vector_product",
    "hessian_matrix",
    "sloppiness_spectrum",
    "sloppiness_spectrum_from_hessian",
    "sloppiness_spectrum_from_matvec",
]
