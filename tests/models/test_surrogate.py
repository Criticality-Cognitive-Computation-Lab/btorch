import torch

from btorch.models.surrogate import ATan


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
