import torch
from torch import nn

from btorch.models.base import MemoryModule


DTYPE = torch.float32


class SimpleRNNCell(MemoryModule):
    """Simple RNN cell: h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)"""

    def __init__(self, input_size: int, hidden_size: int, dtype=None):
        super().__init__()
        if dtype is None:
            dtype = DTYPE
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dtype) * 0.1)
        self.W_h = nn.Parameter(
            torch.eye(hidden_size, dtype=dtype)
            + 0.02 * torch.diag(torch.rand(hidden_size, dtype=dtype))
            - 0.01
        )
        self.b = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

        self.register_memory("h", torch.zeros(1, dtype=dtype), hidden_size)
        self.init_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.h = torch.tanh(x @ self.W_x.t() + self.h @ self.W_h.t() + self.b)
        return self.h


def last_step_sum(out: torch.Tensor) -> torch.Tensor:
    return out[-1].sum()


def native_forward(cell: nn.RNNCell, x_in: torch.Tensor) -> torch.Tensor:
    h = torch.zeros(x_in.shape[1], cell.hidden_size, device=x_in.device, dtype=DTYPE)
    outputs = []
    for t in range(x_in.shape[0]):
        h = cell(x_in[t], h)
        outputs.append(h)
    return torch.stack(outputs, dim=0)
