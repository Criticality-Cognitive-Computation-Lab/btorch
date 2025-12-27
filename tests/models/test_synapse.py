import platform

import matplotlib.pyplot as plt
import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.synapse import ExponentialPSC
from btorch.utils.file import save_fig
from tests.utils.compile import compile_or_skip


def _expected_exponential_psc(
    z_seq: torch.Tensor, dt: float, tau_syn: float, latency_steps: int
) -> torch.Tensor:
    # ExponentialPSC uses psc_{t+1} = psc_t * exp(-dt / tau_syn) + spike_{t-l}.
    decay = torch.exp(torch.tensor(-dt / tau_syn, dtype=z_seq.dtype))
    psc = torch.zeros_like(z_seq[0])
    expected = []

    for t in range(z_seq.shape[0]):
        if t >= latency_steps:
            spike = z_seq[t - latency_steps]
        else:
            spike = torch.zeros_like(psc)
        psc = psc * decay + spike
        expected.append(psc.clone())

    return torch.stack(expected, dim=0)


def _plot_exponential_psc(
    out: torch.Tensor, expected: torch.Tensor, dt: float, name: str
) -> None:
    time = torch.arange(out.shape[0], dtype=out.dtype) * dt
    out_np = out.detach().cpu().numpy()
    expected_np = expected.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    for idx in range(out_np.shape[1]):
        (line,) = ax.plot(time, out_np[:, idx], label=f"psc[{idx}]")
        ax.plot(
            time,
            expected_np[:, idx],
            linestyle="--",
            alpha=0.7,
            color=line.get_color(),
        )
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("PSC")
    ax.set_title("ExponentialPSC latency response")
    ax.legend(loc="best", ncols=2, fontsize=9)
    fig.tight_layout()
    save_fig(fig, name)
    plt.close(fig)


def test_exponential_psc_latency_matches_manual():
    # Use identity weights to isolate delay + exponential decay behavior.
    dt = 1.0
    tau_syn = 2.0
    latency = 3.0
    n_neuron = 3

    z_seq = torch.tensor(
        [
            [1.0, 1.0, 0.5],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )

    linear = torch.nn.Linear(n_neuron, n_neuron, bias=False)
    torch.nn.init.eye_(linear.weight)

    with environ.context(dt=dt):
        synapse = ExponentialPSC(
            n_neuron=n_neuron,
            tau_syn=tau_syn,
            linear=linear,
            latency=latency,
            step_mode="m",
        )
        init_net_state(synapse, dtype=torch.float32)
        out = synapse(z_seq)

    expected = _expected_exponential_psc(
        z_seq,
        dt=dt,
        tau_syn=tau_syn,
        latency_steps=round(latency / dt),
    )

    # _plot_exponential_psc(out, expected, dt=dt, name="exponential_psc_latency")

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_exponential_psc_latency_grad_matches_compile():
    # Compare eager vs compiled results and gradients under identical state.
    torch.manual_seed(123)

    dt = 1.0
    tau_syn = 3.0
    latency = 3.0
    n_neuron = 4
    steps = 6

    z_seq = torch.randn(steps, n_neuron, requires_grad=True)
    z_seq_compiled = z_seq.clone().detach().requires_grad_(True)

    with environ.context(dt=dt):
        eager = ExponentialPSC(
            n_neuron=n_neuron,
            tau_syn=tau_syn,
            linear=torch.nn.Linear(n_neuron, n_neuron, bias=False),
            latency=latency,
            step_mode="m",
        )
        compiled = ExponentialPSC(
            n_neuron=n_neuron,
            tau_syn=tau_syn,
            linear=torch.nn.Linear(n_neuron, n_neuron, bias=False),
            latency=latency,
            step_mode="m",
        )

    compiled.linear.weight.data.copy_(eager.linear.weight.data)

    compiled = compile_or_skip(compiled)

    with environ.context(dt=dt):
        init_net_state(eager)
        init_net_state(compiled)

        out_eager = eager(z_seq)
        out_compiled = compiled(z_seq_compiled)

    torch.testing.assert_close(out_eager, out_compiled, atol=1e-6, rtol=0.0)

    loss_eager = out_eager.sum()
    loss_compiled = out_compiled.sum()

    loss_eager.backward()
    loss_compiled.backward()

    torch.testing.assert_close(z_seq.grad, z_seq_compiled.grad, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(
        eager.linear.weight.grad,
        compiled.linear.weight.grad,
        atol=1e-5,
        rtol=0.0,
    )
