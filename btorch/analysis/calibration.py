"""Calibrate external drive using the actual stateful recurrent model."""

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from btorch.models.rnn import RecurrentNN


@torch.no_grad()
def compute_input_scale(
    network: "RecurrentNN",
    inputs: Tensor,
    *,
    target_hz: float,
    dt_ms: float,
    tolerance_hz: float = 0.25,
    max_trials: int = 24,
) -> tuple[float, float]:
    """Find a drive scale giving the requested finite-trial population rate.

    Evaluate the actual network from the same initialized memories on every
    trial. Restore the caller's memories, recording policy and CUDA Graph flag
    afterwards. Parameters and input storage are unchanged. Calibration runs
    eagerly without state histories; perform it before compilation or capture.

    This bracketed search assumes activity generally increases with input
    scale. It is a heuristic for a finite input window, not a per-neuron or
    stationary-rate guarantee; recurrent feedback can prevent convergence.

    Args:
        network: Initialized recurrent network accepting time-major input.
        inputs: Calibration current, shaped ``[time, batch, *neuron]``.
        target_hz: Desired mean spike rate across the supplied trial.
        dt_ms: Simulation timestep in milliseconds.
        tolerance_hz: Maximum accepted absolute rate difference.
        max_trials: Maximum number of bisection trials after bracketing.

    Returns:
        Input scale and measured population rate in Hz.

    Raises:
        ValueError: If the window, rate or search settings are invalid.
        RuntimeError: If no tested scale reaches the requested tolerance.
    """
    from btorch.models import environ, functional

    if (
        inputs.ndim < 3
        or not inputs.numel()
        or not all(math.isfinite(v) and v > 0 for v in (target_hz, dt_ms, tolerance_hz))
        or target_hz > 1000 / dt_ms
        or max_trials < 1
    ):
        raise ValueError(
            "Nonempty time-major input and positive finite search settings required"
        )
    original = functional.named_hidden_states(network)
    recording, graph = network.update_state_names, network.cudagraph
    network.update_state_names, network.cudagraph = (), False

    def measure(scale):
        functional.set_hidden_states(
            network,
            {
                name: value.clone() if isinstance(value, Tensor) else value
                for name, value in original.items()
            },
        )
        spikes, _ = network(inputs * scale)
        return spikes.mean().item() * 1000 / dt_ms

    try:
        with environ.context(dt=dt_ms):
            low, high = 0.0, 1.0
            rate = measure(high)
            while rate < target_hz and high < 64:
                low, high = high, high * 2
                rate = measure(high)
            best_scale, best_rate = high, rate
            for _ in range(max_trials):
                if abs(best_rate - target_hz) <= tolerance_hz:
                    break
                scale = (low + high) / 2
                rate = measure(scale)
                if abs(rate - target_hz) < abs(best_rate - target_hz):
                    best_scale, best_rate = scale, rate
                if rate < target_hz:
                    low = scale
                else:
                    high = scale
            if (
                not math.isfinite(best_rate)
                or abs(best_rate - target_hz) > tolerance_hz
            ):
                raise RuntimeError(
                    f"Cannot calibrate to {target_hz:g} Hz within {tolerance_hz:g} Hz "
                    f"(closest {best_rate:.3f} Hz); "
                    "use a longer window or revise the drive"
                )
            return best_scale, best_rate
    finally:
        functional.set_hidden_states(network, original)
        network.update_state_names, network.cudagraph = recording, graph
