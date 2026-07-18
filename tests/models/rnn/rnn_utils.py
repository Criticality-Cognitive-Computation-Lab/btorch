import pytest
import torch
from torch import nn

from btorch.models.base import MemoryModule


try:  # scan is a prototype HOP -- absent on older torch, and torch is unpinned.
    from torch._higher_order_ops.scan import scan
except ImportError:  # pragma: no cover - depends on the installed torch
    scan = None

HAS_SCAN = scan is not None

# Shared by tests/ and benchmarks/: feature-detected rather than version-gated,
# since the HOP's import path is the thing that actually has to exist.
requires_scan = pytest.mark.skipif(
    not HAS_SCAN, reason="torch._higher_order_ops.scan unavailable (torch too old)"
)

DTYPE = torch.float32


class ScanRNN(nn.Module):
    """Run a :class:`SimpleRNNCell`'s recurrence as a single ``scan`` HOP.

    Prototype for replacing the unrolled loop in ``btorch/models/rnn.py``:
    ``scan`` traces the step once and keeps T a runtime dim, so ``torch.compile``
    stays O(1) in sequence length instead of unrolling T steps. Wraps an existing
    cell so weights are shared -> parity can be checked directly against
    ``make_rnn(cell)`` (btorch's own loop).

    The step math is inlined statelessly (rather than calling ``cell.forward``,
    which mutates ``cell.h``) because ``scan`` requires a pure, non-mutating
    combine_fn. Reading the cell's parameters is fine -- dynamo lifts them into
    the scan HOP's ``additional_inputs``.
    """

    def __init__(self, cell: "SimpleRNNCell"):
        super().__init__()
        self.cell = cell

    def forward(self, x, h0=None):  # x: (T, B, input_size) -> (T, B, hidden_size)
        if scan is None:  # constructing is fine; only running needs the HOP
            raise RuntimeError(
                "ScanRNN needs torch._higher_order_ops.scan, which this torch "
                "does not have. Gate call sites with rnn_utils.requires_scan."
            )
        cell = self.cell
        if h0 is None:
            h0 = torch.zeros(
                x.shape[1], cell.hidden_size, dtype=x.dtype, device=x.device
            )

        def combine(carry, x_t):
            h = torch.tanh(x_t @ cell.W_x.t() + carry @ cell.W_h.t() + cell.b)
            # scan forbids the emitted output aliasing any input/carry -> clone.
            return h, h.clone()

        _, ys = scan(combine, h0, x)
        return ys


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
