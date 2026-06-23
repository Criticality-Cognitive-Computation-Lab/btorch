from __future__ import annotations

import torch

from btorch.sparse.errors import BackendUnavailableError, UnsupportedCapabilityError


def is_available() -> bool:
    try:
        import torch_sparse  # noqa: F401
    except ImportError:
        return False
    return True


def sparse_mm(
    matrix,
    x: torch.Tensor,
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        raise UnsupportedCapabilityError(
            "torch_sparse is not selected inside torch.compile graphs."
        )
    try:
        from torch_sparse import spmm
    except ImportError as exc:
        raise BackendUnavailableError(
            "torch_sparse is not installed. Install Btorch's 'sparse' extra "
            "using a wheel matching the active PyTorch and CUDA versions."
        ) from exc

    coo = matrix.to_coo()
    values = matrix.effective_values()
    shape = matrix.shape
    if x.shape[-1] != shape[0]:
        raise ValueError(f"Expected input size {shape[0]}, received {x.shape[-1]}.")
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, shape[0])
    indices_t = torch.stack([coo.col, coo.row])
    out = spmm(indices_t, values, shape[1], shape[0], x_2d.T).T
    return out.reshape(*leading_shape, shape[1])
