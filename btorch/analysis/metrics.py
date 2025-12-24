from typing import Literal

import numpy as np


def indices_to_mask(indices: np.ndarray, shape=None, array=None) -> np.ndarray:
    """Convert an array of indices to a boolean mask.

    For multi-dimensional masks, a 1D index array is treated as
    flattened indices. Provide a tuple of index arrays for per-axis
    indexing.
    """
    assert not (shape is None and array is None)
    mask = (
        np.zeros(shape, dtype=bool)
        if shape is not None
        else np.zeros_like(array, dtype=bool)
    )
    indices_arr = np.asarray(indices)
    if mask.ndim > 1 and indices_arr.ndim == 1 and not isinstance(indices, tuple):
        mask.flat[indices_arr] = True
    else:
        mask[indices] = True
    return mask


def select_on_metric(
    metrics: np.ndarray,
    num: int | None = None,
    mode: Literal["topk", "any"] = "topk",
    ret_indices: bool = False,
):
    """Select neurons based on a metric array."""
    if mode == "topk":
        assert num is not None
        ret = np.argpartition(metrics, -num)[-num:]
    elif mode == "any":
        ret = metrics.nonzero()[0]
        if num is not None and len(ret) > num:
            ret = np.random.choice(ret, num, replace=False)
    else:
        raise ValueError(f"Unsupported mode {mode}")

    if ret_indices:
        return ret, indices_to_mask(ret, array=metrics)
    else:
        return ret
