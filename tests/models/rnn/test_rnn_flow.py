import torch
from matplotlib import pyplot as plt

from btorch.models.functional import reset_net_state
from btorch.models.rnn import make_rnn
from btorch.monitor import grad
from btorch.utils.file import save_fig
from tests.models.rnn.rnn_utils import SimpleRNNCell, last_step_sum


def test_rnn_gradient_flow():
    """Visualize gradient flow through hidden states."""
    torch.manual_seed(233)
    T, batch_size, input_size, hidden_size = 50, 2, 4, 8

    # Enable grad history saving
    rnn = make_rnn(SimpleRNNCell, unroll=4, update_state_names={"h_grad": grad("h")})(
        input_size=input_size, hidden_size=hidden_size
    )

    x = torch.randn(T, batch_size, input_size, requires_grad=True)

    reset_net_state(rnn, batch_size=batch_size)
    out, _ = rnn(x)

    # Backward from the LAST step only to see how it flows back
    # Or backward from a sum of all steps
    loss = last_step_sum(out)
    loss.backward()

    h_grads = rnn.get_records()["h_grad"]

    assert len(h_grads) == T

    grad_norms = [g.norm().item() if g is not None else 0.0 for g in h_grads]
    h_norms = [out[t].norm().item() for t in range(T)]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grad_norms, marker="o", markersize=3, label="||grad h_t||")
    ax.plot(h_norms, marker="s", markersize=3, label="||h_t||", alpha=0.5)
    ax.set_title("RNN Gradient Flow (Gradient of Final Loss w.r.t Hidden States)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("L2 Norm")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()

    save_fig(fig, name="rnn_gradient_flow")
    plt.close(fig)

    # Check that gradients actually exist and flow back
    # Since we backward from the last step, we expect gradients to decay backwards
    assert grad_norms[-1] > 0
    assert grad_norms[0] > 0  # Should still have some gradient at the beginning


# NOTE: checkpointed-vs-eager gradient parity is pinned by
# tests/models/rnn/test_rnn_ops.py::test_rnn_exhaustive_ops[grad_checkpoint=True,
# record_grad=True]; a plotting-only duplicate lived here and was removed.
