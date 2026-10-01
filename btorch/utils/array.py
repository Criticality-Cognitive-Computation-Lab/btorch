"""Array conversion helpers shared across btorch."""

from typing import Any

import numpy as np
import torch


def to_numpy(data: Any) -> np.ndarray:
    """Convert a tensor or array-like to a NumPy array.

    Torch tensors are detached and moved to CPU; anything else goes through
    :func:`numpy.asarray`.

    Args:
        data: ``torch.Tensor``, ``np.ndarray`` or array-like.

    Returns:
        NumPy array (a view of the input when no conversion is needed).
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)
