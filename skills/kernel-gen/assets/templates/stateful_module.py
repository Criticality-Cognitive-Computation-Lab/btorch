"""Start a custom stateful Torch model from btorch MemoryModule.

This cumulative-drive toy illustrates memory/parameter ownership and a
continuous sequence entrypoint. Replace its dynamics with the user's
model; it is not an alternative GLIF model.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from btorch.models import environ
from btorch.models.base import MemoryModule


class CumulativeDrive(MemoryModule):
    r"""Integrate a drive in a registered state.

    .. math::
        q_{t+1} = q_t + \Delta t\,g\,x_t

    Args:
        n_neuron: Number of independent channels.
        gain: Initial differentiable drive gain.
        device: Parameter device; initialize memories with init_state.
        dtype: Floating parameter and state dtype.
    """

    def __init__(
        self,
        n_neuron: int,
        gain: float = 1.0,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(gain, device=device, dtype=dtype))
        self.register_memory("q", 0.0, n_neuron)

    def single_step_forward(
        self, x: Float[Tensor, "batch neuron"]
    ) -> Float[Tensor, "batch neuron"]:
        self.q = self.q + environ.get("dt") * self.gain * x
        return self.q

    def multi_step_forward(
        self, x_seq: Float[Tensor, "time batch neuron"]
    ) -> Float[Tensor, "time batch neuron"]:
        """Integrate a complete sequence from the current initial state.

        A generated sequence operator belongs at this boundary. Network
        scheduling is a separate responsibility of RecurrentNN; wrapping
        this module in a single-step loop would bypass this
        implementation.
        """
        if x_seq.shape[0] == 0:
            return x_seq.clone()
        q_seq = self.q.unsqueeze(0) + (environ.get("dt") * self.gain * x_seq).cumsum(0)
        self.q = q_seq[-1]
        return q_seq
