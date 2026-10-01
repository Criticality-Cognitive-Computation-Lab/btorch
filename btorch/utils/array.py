"""Array conversion helpers shared across btorch."""

from typing import Any

import numpy as np
import scipy.sparse as sp
import torch


def to_numpy(data: Any, *, strict: bool = False, name: str = "data") -> np.ndarray:
    """Convert a tensor or array-like to a dense NumPy array.

    Torch tensors are detached and moved to CPU; anything else goes through
    :func:`numpy.asarray`.

    Args:
        data: ``torch.Tensor``, ``np.ndarray`` or array-like.
        strict: Accept only ``torch.Tensor`` and ``np.ndarray`` and raise
            :class:`TypeError` for anything else (lists, scalars, ...).
        name: Argument name used in the ``strict`` error message.

    Returns:
        NumPy array (a view of the input when no conversion is needed).

    Raises:
        TypeError: If ``strict`` and ``data`` is not a tensor or ndarray.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if strict and not isinstance(data, np.ndarray):
        raise TypeError(f"`{name}` must be a numpy array or torch tensor.")
    return np.asarray(data)


def to_numpy_or_sparse(
    data: Any,
) -> np.ndarray | np.generic | sp.spmatrix | sp.sparray:
    """Convert to a dense NumPy array, keeping sparse data sparse.

    Unlike :func:`to_numpy`, SciPy sparse inputs pass through unchanged and
    sparse torch tensors become SciPy COO arrays instead of failing, and NumPy
    scalars are returned as they are. Objects exposing ``.numpy()`` (other
    tensor-likes) are converted through it.

    Args:
        data: ``torch.Tensor`` (dense or sparse), NumPy/SciPy array, scalar or
            array-like.

    Returns:
        Dense ``np.ndarray``, a NumPy scalar, or a SciPy sparse object.
    """
    if sp.issparse(data):
        return data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu()
        if data.is_sparse:
            coo = data.to_sparse_coo().coalesce()
            return sp.coo_array(
                (coo.values().numpy(), tuple(coo.indices().numpy())),
                shape=tuple(coo.shape),
            )
        return data.numpy()
    if isinstance(data, (np.ndarray, np.generic)):
        return data
    if hasattr(data, "numpy"):
        return data.numpy()
    return np.asarray(data)
