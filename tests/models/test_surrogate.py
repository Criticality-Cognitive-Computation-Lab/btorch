import matplotlib.pyplot as plt
import pytest
import torch

from btorch.models.surrogate import ATan, ATanApprox, Erf, Sigmoid, Triangle, atan
from btorch.utils.file import save_fig


def test_gradient_is_damped():
    damping_factor = 0.5

    x = torch.linspace(-2, 2, steps=9, requires_grad=True)
    spike_fn_damped = ATan(alpha=2.0, damping_factor=damping_factor)
    y_damped = spike_fn_damped(x)
    y_damped.sum().backward()
    grad_damped = x.grad.clone()

    x = torch.linspace(-2, 2, steps=9, requires_grad=True)
    spike_fn_orig = ATan(alpha=2.0, damping_factor=1.0)
    y_orig = spike_fn_orig(x)
    y_orig.sum().backward()
    grad_orig = x.grad.clone()

    ratio = grad_damped / (grad_orig + 1e-12)
    mean_ratio = ratio.abs().mean().item()

    assert torch.allclose(
        torch.tensor(mean_ratio),
        torch.tensor(damping_factor),
        atol=0.05,
    ), f"Expected gradient ratio ≈ {damping_factor}, got {mean_ratio}"


def test_functional_gradient_is_damped():
    damping_factor = 0.5

    x = torch.linspace(-2, 2, steps=9, requires_grad=True)
    y_damped = atan(x, alpha=2.0, damping_factor=damping_factor)
    y_damped.sum().backward()
    grad_damped = x.grad.clone()

    x = torch.linspace(-2, 2, steps=9, requires_grad=True)
    y_orig = atan(x, alpha=2.0, damping_factor=1.0)
    y_orig.sum().backward()
    grad_orig = x.grad.clone()

    ratio = grad_damped / (grad_orig + 1e-12)
    mean_ratio = ratio.abs().mean().item()

    assert torch.allclose(
        torch.tensor(mean_ratio),
        torch.tensor(damping_factor),
        atol=0.05,
    ), f"Expected gradient ratio ≈ {damping_factor}, got {mean_ratio}"


def _spikingjelly_surrogate():
    pytest.importorskip("spikingjelly")
    from spikingjelly.activation_based import surrogate as sj_surrogate

    return sj_surrogate


@pytest.mark.parametrize(
    ("name", "btorch_cls", "kwargs"),
    [
        ("ATan", ATan, {"alpha": 2.0}),
        ("ATanApprox", ATanApprox, {"alpha": 2.0}),
        ("Sigmoid", Sigmoid, {"alpha": 2.0}),
        ("Erf", Erf, {"alpha": 2.0}),
    ],
)
def test_matches_spikingjelly_when_no_damping(name, btorch_cls, kwargs):
    sj_surrogate = _spikingjelly_surrogate()
    if not hasattr(sj_surrogate, name):
        pytest.skip(f"spikingjelly.surrogate.{name} not available")
    sj_cls = getattr(sj_surrogate, name)

    x = torch.linspace(-3, 3, steps=12)
    x_sj = x.clone().detach()

    y = btorch_cls(damping_factor=1.0, spiking=False, **kwargs)(x)
    y_sj = sj_cls(spiking=False, **kwargs)(x_sj)
    assert torch.allclose(y, y_sj, atol=1e-6)

    x = torch.linspace(-3, 3, steps=12, requires_grad=True)
    x_sj = x.clone().detach().requires_grad_(True)
    y = btorch_cls(damping_factor=1.0, spiking=True, **kwargs)(x)
    y_sj = sj_cls(spiking=True, **kwargs)(x_sj)
    y.sum().backward()
    y_sj.sum().backward()
    assert torch.allclose(x.grad, x_sj.grad, atol=1e-6)


def test_plot_surrogate_comparison():
    """Compare all surrogate functions on the same figure.

    This test plots both the forward pass (spike function) and the
    backward pass (surrogate gradient) for all available surrogate
    types, making it easy to visualize their differences and
    characteristics.
    """
    # Generate input range for visualization
    x = torch.linspace(-5, 5, steps=200)
    x_grad = torch.linspace(-5, 5, steps=200, requires_grad=True)

    # Define all surrogate types to compare
    surrogates = [
        ("ATan", ATan, {"alpha": 2.0}),
        ("ATanApprox", ATanApprox, {"alpha": 2.0}),
        ("Sigmoid", Sigmoid, {"alpha": 2.0}),
        ("Erf", Erf, {"alpha": 2.0}),
        ("Triangle", Triangle, {"alpha": 1.0}),
    ]

    # Create figure with two subplots: forward pass and gradient
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Surrogate Function Comparison", fontsize=16)

    colors = ["blue", "red", "green", "purple", "orange"]

    for (name, surrogate_cls, kwargs), color in zip(surrogates, colors):
        # Forward pass (spike function) - spiking=False shows raw output
        spike_fn = surrogate_cls(damping_factor=1.0, spiking=False, **kwargs)
        y = spike_fn(x)
        ax1.plot(x.numpy(), y.numpy(), label=name, color=color, linewidth=2)

        # Backward pass (surrogate gradient)
        x_g = x_grad.clone().detach().requires_grad_(True)
        spike_fn_grad = surrogate_cls(damping_factor=1.0, spiking=True, **kwargs)
        y_g = spike_fn_grad(x_g)
        y_g.sum().backward()
        grad = x_g.grad.clone()
        ax2.plot(
            x_grad.detach().numpy(),
            grad.numpy(),
            label=name,
            color=color,
            linewidth=2,
        )

    # Format forward pass plot
    ax1.set_xlabel("Input (x)", fontsize=12)
    ax1.set_ylabel("Output", fontsize=12)
    ax1.set_title("Forward Pass (Spike Function)", fontsize=14)
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.axhline(y=1, color="k", linestyle="--", alpha=0.3)
    ax1.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Format gradient plot
    ax2.set_xlabel("Input (x)", fontsize=12)
    ax2.set_ylabel("Gradient", fontsize=12)
    ax2.set_title("Backward Pass (Surrogate Gradient)", fontsize=14)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "surrogate_comparison")
