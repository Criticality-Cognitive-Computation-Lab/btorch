import torch
from matplotlib import pyplot as plt

from btorch.models.functional import reset_net_state
from btorch.models.rnn import make_rnn
from btorch.utils.file import save_fig
from tests.models.rnn_utils import SimpleRNNCell, last_step_sum


def test_rnn_gradient_flow():
    """Visualize gradient flow through hidden states."""
    torch.manual_seed(233)
    T, batch_size, input_size, hidden_size = 50, 2, 4, 8

    # Enable grad history saving
    rnn = make_rnn(
        SimpleRNNCell, unroll=4, save_grad_history=True, grad_state_names=["h"]
    )(input_size=input_size, hidden_size=hidden_size)

    x = torch.randn(T, batch_size, input_size, requires_grad=True)

    reset_net_state(rnn, batch_size=batch_size)
    out, _ = rnn(x)

    # Backward from the LAST step only to see how it flows back
    # Or backward from a sum of all steps
    loss = last_step_sum(out)
    loss.backward()

    grad_history = rnn.get_grad_history()
    h_grads = grad_history.get("h", [])

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


def test_rnn_flow_checkpointing_comparison():
    """Compare gradient flow with and without checkpointing."""
    torch.manual_seed(42)
    T = 20

    # Create shared input and RNNs with synced weights
    x = torch.randn(T, 2, 3, requires_grad=True)

    configs = [
        {"name": "No CP", "grad_checkpoint": False},
        {"name": "With CP", "grad_checkpoint": True},
    ]

    # Create all RNNs first
    rnns = []
    for cfg in configs:
        rnn = make_rnn(
            SimpleRNNCell,
            unroll=4,
            save_grad_history=True,
            grad_checkpoint=cfg["grad_checkpoint"],
        )(input_size=3, hidden_size=4)
        rnns.append(rnn)

    # Sync weights from first RNN to all others
    for i in range(1, len(rnns)):
        rnns[i].rnn_cell.W_x.data.copy_(rnns[0].rnn_cell.W_x.data)
        rnns[i].rnn_cell.W_h.data.copy_(rnns[0].rnn_cell.W_h.data)
        rnns[i].rnn_cell.b.data.copy_(rnns[0].rnn_cell.b.data)

    fig, ax = plt.subplots(figsize=(10, 5))

    grad_histories = {}
    for cfg, rnn in zip(configs, rnns):
        x_copy = x.detach().clone().requires_grad_(True)
        reset_net_state(rnn, batch_size=2)
        out, _ = rnn(x_copy)
        out.sum().backward()

        grad_norms = [g.norm().item() for g in rnn.get_grad_history()["h"]]
        grad_histories[cfg["name"]] = rnn.get_grad_history()["h"]
        ax.plot(grad_norms, label=cfg["name"], marker="x")

    ax.set_title("Gradient Flow Comparison: Checkpointing vs Eager")
    ax.set_xlabel("Time step")
    ax.set_ylabel("||grad h_t||")
    ax.legend()
    ax.grid(True)

    save_fig(fig, name="rnn_flow_checkpoint_comparison")
    plt.close(fig)

    # Assert that gradient histories match between checkpointed and non-checkpointed
    # They SHOULD match because checkpoint is mathematically equivalent
    for t in range(T):
        g_no_cp = grad_histories["No CP"][t]
        g_with_cp = grad_histories["With CP"][t]
        assert torch.allclose(g_no_cp, g_with_cp, atol=1e-5), (
            f"Gradient mismatch at t={t}: "
            f"||no_cp||={g_no_cp.norm().item():.4f}, "
            f"||with_cp||={g_with_cp.norm().item():.4f}"
        )
