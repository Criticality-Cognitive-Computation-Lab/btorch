from typing import Literal

import numpy as np
import torch


def suggest_skip_timestep(data: np.ndarray | torch.Tensor) -> int:
    """Suggest a burn-in period based on trace length."""
    skip_timestep = data.shape[0] // 8
    if skip_timestep < 100:
        skip_timestep = 0
    if skip_timestep > 1000:
        skip_timestep = 1000
    return skip_timestep


def voltage_overshoot(
    V: np.ndarray | torch.Tensor,
    mode: Literal["std", "mse_threshold", "threshold_resting"] = "threshold_resting",
    skip_timestep: int | None = None,
    **params: float,
) -> np.ndarray | torch.Tensor:
    """Quantify voltage stability/overshoot in different ways.

    Raises:
        ValueError: If ``mode`` is not one of ``"std"``, ``"mse_threshold"`` or
            ``"threshold_resting"``.
        KeyError: If a required threshold parameter (``V_th``, and ``V_reset``
            for ``"threshold_resting"``) is missing from ``params``.
    """
    is_numpy = isinstance(V, np.ndarray)
    V = V.astype(np.float32) if is_numpy else V.to(torch.float32)

    if skip_timestep is None:
        skip_timestep = suggest_skip_timestep(V)

    V_slice = V[skip_timestep:]
    if mode == "std":
        return V_slice.std(0)

    elif mode == "mse_threshold":
        V_th = params["V_th"]
        return (
            (V_slice - V_th).pow(2).mean(0)
            if not is_numpy
            else np.mean((V_slice - V_th) ** 2, axis=0)
        )

    elif mode == "threshold_resting":
        V_th = params["V_th"]
        V_reset = params["V_reset"]
        n_scale = params.get("n_scale", 3)

        scale = V_th - V_reset

        upper = V_th + n_scale * scale
        lower = V_reset - n_scale * scale

        mask = (V_slice > upper) | (V_slice < lower)
        return mask.astype(np.float32).mean(0) if is_numpy else mask.float().mean(0)

    else:
        raise ValueError(f"Unsupported mode: {mode}")
