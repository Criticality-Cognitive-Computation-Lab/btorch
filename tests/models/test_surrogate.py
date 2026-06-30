import matplotlib.pyplot as plt
import pytest
import torch

from btorch.models.surrogate import (
    ATan,
    ATanApprox,
    Erf,
    Sigmoid,
    SuperSpike,
    Triangle,
    atan,
)
from btorch.utils.file import save_fig


# All surrogates use alpha=2 as canonical default (HWHM = 1/alpha = 0.5).
_DEFAULT_ALPHA = 2.0

# (name, class, kwargs, internal_constant_label)
# internal_constant_label describes the k factor baked in for HWHM=1/alpha
_SURROGATES = [
    ("ATan", ATan, {"alpha": _DEFAULT_ALPHA}, "k=1 (Cauchy)"),
    ("ATanApprox", ATanApprox, {"alpha": _DEFAULT_ALPHA}, "k≈1 (rational)"),
    ("Sigmoid", Sigmoid, {"alpha": _DEFAULT_ALPHA}, "k=2ln(√2+1)≈1.76"),
    ("Erf", Erf, {"alpha": _DEFAULT_ALPHA}, "k=√ln2≈0.83"),
    ("Triangle", Triangle, {"alpha": _DEFAULT_ALPHA}, "k=1/2"),
    ("SuperSpike", SuperSpike, {"alpha": _DEFAULT_ALPHA}, "k=√2-1≈0.41"),
]


def test_gradient_is_damped():
    damping_factor = 0.5

    x = torch.linspace(-2, 2, steps=9, requires_grad=True)
    spike_fn_damped = ATan(alpha=_DEFAULT_ALPHA, damping_factor=damping_factor)
    y_damped = spike_fn_damped(x)
    y_damped.sum().backward()
    grad_damped = x.grad.clone()

    x = torch.linspace(-2, 2, steps=9, requires_grad=True)
    spike_fn_orig = ATan(alpha=_DEFAULT_ALPHA, damping_factor=1.0)
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
    y_damped = atan(x, alpha=_DEFAULT_ALPHA, damping_factor=damping_factor)
    y_damped.sum().backward()
    grad_damped = x.grad.clone()

    x = torch.linspace(-2, 2, steps=9, requires_grad=True)
    y_orig = atan(x, alpha=_DEFAULT_ALPHA, damping_factor=1.0)
    y_orig.sum().backward()
    grad_orig = x.grad.clone()

    ratio = grad_damped / (grad_orig + 1e-12)
    mean_ratio = ratio.abs().mean().item()

    assert torch.allclose(
        torch.tensor(mean_ratio),
        torch.tensor(damping_factor),
        atol=0.05,
    ), f"Expected gradient ratio ≈ {damping_factor}, got {mean_ratio}"


@pytest.mark.parametrize(("name", "btorch_cls", "kwargs", "k_label"), _SURROGATES)
def test_unit_gradient_at_threshold(name, btorch_cls, kwargs, k_label):
    """G(v=0, damping=1) must equal 1.0 for every surrogate at any alpha."""
    x = torch.tensor(0.0, requires_grad=True)
    fn = btorch_cls(damping_factor=1.0, spiking=True, **kwargs)
    fn(x).backward()
    assert (
        abs(x.grad.item() - 1.0) < 1e-5
    ), f"{name} grad at threshold = {x.grad.item():.6f}, expected 1.0"


@pytest.mark.parametrize(("name", "btorch_cls", "kwargs", "k_label"), _SURROGATES)
def test_consistent_hwhm(name, btorch_cls, kwargs, k_label):
    """HWHM must equal 1/alpha for every surrogate (alpha-consistent
    parametrisation).

    ATanApprox uses a rational approximation so its HWHM is only
    approximately 1/alpha (~8% error); a looser tolerance is applied for
    it.
    """
    alpha = kwargs["alpha"]
    expected_hw = 1.0 / alpha
    x = torch.linspace(0, 3.0 / alpha, steps=10000, requires_grad=True)
    fn = btorch_cls(damping_factor=1.0, spiking=True, **kwargs)
    fn(x).sum().backward()
    grad = x.grad.detach()
    mask = grad >= 0.5
    hw = x.detach()[mask].max().item() if mask.any() else 0.0
    atol = 0.06 if name == "ATanApprox" else 0.02
    assert (
        abs(hw - expected_hw) < atol
    ), f"{name} HWHM = {hw:.4f}, expected {expected_hw:.4f}"


def test_plot_surrogate_comparison():
    """Compare all surrogate gradients; annotate HWHM on the gradient
    subplot."""
    x = torch.linspace(-4, 4, steps=400)
    x_grad = torch.linspace(-4, 4, steps=400, requires_grad=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    hw_label = f"HWHM=1/alpha={1 / _DEFAULT_ALPHA:.2f}"
    fig.suptitle(
        f"Surrogate comparison — alpha={_DEFAULT_ALPHA}, {hw_label}",
        fontsize=14,
    )

    colors = ["blue", "red", "green", "purple", "orange", "brown"]

    for (name, surrogate_cls, kwargs, k_label), color in zip(_SURROGATES, colors):
        alpha = kwargs["alpha"]
        legend_label = f"{name} (α={alpha}, {k_label})"
        spike_fn = surrogate_cls(damping_factor=1.0, spiking=False, **kwargs)
        y = spike_fn(x)
        ax1.plot(
            x.numpy(), y.detach().numpy(), label=legend_label, color=color, linewidth=2
        )

        x_g = x_grad.clone().detach().requires_grad_(True)
        spike_fn_grad = surrogate_cls(damping_factor=1.0, spiking=True, **kwargs)
        spike_fn_grad(x_g).sum().backward()
        grad = x_g.grad.clone()
        ax2.plot(
            x_grad.detach().numpy(),
            grad.numpy(),
            label=legend_label,
            color=color,
            linewidth=2,
        )

    # Annotate HWHM on gradient plot
    hw = 1.0 / _DEFAULT_ALPHA
    ax2.axhline(y=0.5, color="k", linestyle=":", linewidth=1, label="half-max (0.5)")
    ax2.axvline(x=hw, color="k", linestyle=":", linewidth=1)
    ax2.axvline(x=-hw, color="k", linestyle=":", linewidth=1)
    ax2.annotate(
        f"HWHM={hw:.2f}",
        xy=(hw, 0.52),
        fontsize=9,
        ha="left",
    )

    ax1.set_xlabel("v")
    ax1.set_ylabel("Output")
    ax1.set_title("Primitive (spiking=False)")
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.axhline(y=1, color="k", linestyle="--", alpha=0.3)
    ax1.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("v")
    ax2.set_ylabel("Gradient")
    ax2.set_title("Surrogate gradient (backward)")
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "surrogate_comparison")
