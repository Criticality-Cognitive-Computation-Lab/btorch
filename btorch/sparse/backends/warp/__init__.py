from __future__ import annotations

import torch

from btorch.sparse.errors import BackendUnavailableError, UnsupportedCapabilityError


def is_available() -> bool:
    try:
        import warp  # noqa: F401
    except ImportError:
        return False
    return True


def sparse_mm(
    matrix,
    x: torch.Tensor,
) -> torch.Tensor:
    if not is_available():
        raise BackendUnavailableError("Warp is not installed.")
    values = matrix.effective_values()
    shape = matrix.shape
    if not x.is_cuda or values.dtype != torch.float32:
        raise UnsupportedCapabilityError(
            "The Warp sparse backend currently requires CUDA float32 tensors."
        )
    from .coo import coo_spmm_warp

    coo = matrix.to_coo()
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, shape[0])
    indices_t = torch.stack([coo.col, coo.row])
    out = coo_spmm_warp(indices_t, values, x_2d.T, size_m=shape[1]).T
    return out.reshape(*leading_shape, shape[1])
