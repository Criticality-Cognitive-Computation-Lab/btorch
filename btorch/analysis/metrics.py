from typing import Literal

import numpy as np


def indices_to_mask(
    indices: np.ndarray | tuple[np.ndarray, ...],
    shape: tuple[int, ...] | None = None,
    array: np.ndarray | None = None,
) -> np.ndarray:
    """Convert indices to a boolean mask.

    For multi-dimensional masks, a 1D index array is treated as
    flattened indices. Provide a tuple of index arrays for per-axis
    indexing (e.g. the output of :func:`numpy.nonzero`).

    Args:
        indices: Index array (flat indices for ``ndim > 1`` masks) or a tuple
            of per-axis index arrays.
        shape: Shape of the mask. Takes precedence over ``array``.
        array: Array whose shape is used when ``shape`` is not given.

    Returns:
        Boolean array that is ``True`` at ``indices``.

    Raises:
        ValueError: If neither ``shape`` nor ``array`` is given.
    """
    if shape is None and array is None:
        raise ValueError("Either `shape` or `array` must be provided")
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
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Select neurons based on a metric array.

    Args:
        metrics: 1D per-neuron metric.
        num: Number of neurons to select (required for ``"topk"``; an optional
            cap, sampled at random, for ``"any"``).
        mode: ``"topk"`` selects the ``num`` largest entries, ``"any"`` the
            non-zero entries.
        ret_indices: If True, return ``(indices, mask)`` instead of indices.

    Returns:
        Selected neuron indices, or ``(indices, mask)`` if ``ret_indices``.

    Raises:
        ValueError: If ``mode`` is unsupported or ``num`` is missing for
            ``"topk"``.
    """
    if mode == "topk":
        if num is None:
            raise ValueError("`num` must be provided when mode is 'topk'")
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
