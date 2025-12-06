import torch
from btorch.models.base import MemoryModule
from btorch.models.rnn import make_rnn
from matplotlib import pyplot as plt
from torch import nn
from torchviz import make_dot

from tests.utils.file import fig_path


torch.manual_seed(233)


class SimpleRNNCell(MemoryModule):
    """Simple RNN cell: h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))

        self.register_memory("h", None, hidden_size)
        self.init_state()
        self.h = None
        # self.init_state(dtype=torch.float32)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.h is None:
            self.h = torch.zeros(
                x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype
            )
        self.h = self.h @ self.W_h.T + x @ self.W_x.T + self.b
        return self.h


# Build the RNN with unroll support + grad tracking
rnn = make_rnn(
    SimpleRNNCell,
    grad_checkpoint=False,
    unroll=False,
    save_grad_history=True,
    grad_state_names=["h"],
    update_state_names=["h"],
    allow_buffer=True,
)(input_size=3, hidden_size=4)

# Create input sequence T × B × D
T, B, D = 20, 2, 3
x = 100 * torch.ones(T, B, D, requires_grad=True)

########################################
# 1) Draw NETWORK STRUCTURE using torchview
########################################
# model_graph = draw_graph(
#     rnn,
#     input_size=(T, B, D),
#     expand_nested=True,
#     save_graph=True,
#     directory=str(fig_path()),
#     filename="rnn_torchview",
# )
# print("Saved model structure graph:", str(fig_path() / "rnn_torchview.png"))

########################################
# 2) Draw GRAD GRAPH using torchviz
########################################
rnn.reset()
out, states = rnn.multi_step_forward(x)
loss = out.sum()
loss.backward()

grad_graph = make_dot(loss, params=dict(rnn.named_parameters()))
grad_graph.render(fig_path() / "rnn_grad_graph", format="png")
print("Saved backward grad graph:", fig_path() / "rnn_grad_graph.png")

grad_history = rnn.get_grad_history()
rnn.clear_grad_history()

########################################
# 3) Plot gradient magnitude of hidden state h_t
########################################
# grad_history["h"] is a list of T tensors, each (B, hidden)
# we take L2 norm per time step
magnitudes = []
for t, g in enumerate(grad_history["h"]):
    if g is None:
        magnitudes.append(0.0)
    else:
        magnitudes.append(g.norm().item())

plt.figure(figsize=(6, 4))
plt.plot(magnitudes, marker="o")
plt.title("Gradient Magnitude of h Across Time")
plt.xlabel("Time step")
plt.ylabel("||grad h_t||")
plt.grid(True)
plt.tight_layout()

save_path = fig_path() / "grad_h_timeseries.png"
plt.savefig(save_path)
plt.close()
print("Saved grad magnitude plot:", save_path)
