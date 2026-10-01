from collections.abc import Sequence
from numbers import Number
from typing import Any

import numpy as np
import torch

from ..types import TensorLike


@torch.no_grad()
def scale_state_(
    states: dict[str, Any],
    *,
    scale: TensorLike | None = None,
    zeropoint: TensorLike | None = None,
    unscale: bool = False,
    store: bool = False,
) -> tuple[TensorLike, TensorLike]:
    """Scale or unscale a state dictionary in-place.

    Maps ``v``, ``v_threshold`` and ``v_reset`` to ``(x - zeropoint) / scale``
    and ``Iasc``, ``psc`` and ``asc_amps`` to ``x / scale`` (inverted when
    ``unscale=True``). Scaling makes parameter magnitudes comparable with the
    learning rate; it is applied explicitly to a states dict, not hooked into
    modules.

    Args:
        states: Dictionary of state tensors to scale.
        scale: Scaling factor. If None, inferred from
            ``states["v_threshold"] - states["v_reset"]``.
        zeropoint: Zero point for scaling. If None, inferred from
            ``states["v_reset"]``.
        unscale: If True, apply unscaling instead of scaling.
        store: If True, store ``scale``, ``zeropoint``, and ``scaled``
            flag into ``states``.

    Returns:
        Tuple of ``(scale, zeropoint)`` used.

    Raises:
        ValueError: If required keys for inference are missing.
    """
    if scale is None:
        if "scale" in states:
            scale = states["scale"]
        elif ("v_threshold" in states) and ("v_reset" in states):
            scale = states["v_threshold"] - states["v_reset"]
    if scale is None:
        raise ValueError("Must specify either scale or v_threshold and v_reset")

    if zeropoint is None:
        if "zeropoint" in states:
            zeropoint = states["zeropoint"]
        elif "v_reset" in states:
            zeropoint = states["v_reset"]
        else:
            raise ValueError("Must specify either zeropoint or v_reset")

    if not unscale:
        if "scaled" in states and states["scaled"]:
            return scale, zeropoint
    else:
        if "scaled" in states and not states["scaled"]:
            return scale, zeropoint

    def _scale(v, zero=True):
        if zero:
            return (v - zeropoint) / scale
        else:
            return v / scale

    def _unscale(v, zero=True):
        if zero:
            return v * scale + zeropoint
        else:
            return v * scale

    fn = _unscale if unscale else _scale

    if "v" in states:
        states["v"] = fn(states["v"])

    if "Iasc" in states:
        states["Iasc"] = fn(states["Iasc"], zero=False)

    if "v_threshold" in states:
        states["v_threshold"] = fn(states["v_threshold"])

    if "v_reset" in states:
        states["v_reset"] = fn(states["v_reset"])

    if "psc" in states:
        states["psc"] = fn(states["psc"], zero=False)

    if "asc_amps" in states:
        # very annoying to write code for both torch and numpy
        scale = scale if isinstance(scale, Number) else scale[..., None]
        asc_amps = states["asc_amps"]
        if isinstance(asc_amps, Number):
            raise TypeError("asc_amps must be an array or sequence, not a number")
        asc_amps = np.array(asc_amps) if isinstance(asc_amps, Sequence) else asc_amps
        if unscale:
            states["asc_amps"] = asc_amps * scale
        else:
            states["asc_amps"] = asc_amps / scale

    if store:
        states["zeropoint"] = zeropoint
        states["scale"] = scale
        if unscale:
            states["scaled"] = False
        else:
            states["scaled"] = True

    return scale, zeropoint
