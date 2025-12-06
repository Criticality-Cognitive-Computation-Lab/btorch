import torch
from btorch.models.surrogate import add_damping
from spikingjelly.activation_based import surrogate


def test_gradient_is_damped():
    damping_factor = 0.5
    # Create damped surrogate
    ATanDamped = add_damping(damping_factor)(surrogate.ATan)

    # --- Compute gradient for damped ---
    x = torch.linspace(-2, 2, steps=9, requires_grad=True)
    spike_fn_damped = ATanDamped(alpha=2.0)
    y_damped = spike_fn_damped(x)
    y_damped.sum().backward()
    grad_damped = x.grad.clone()

    # --- Compute gradient for original ---
    x = torch.linspace(-2, 2, steps=9, requires_grad=True)
    spike_fn_orig = surrogate.ATan(alpha=2.0)
    y_orig = spike_fn_orig(x)
    y_orig.sum().backward()
    grad_orig = x.grad.clone()

    # --- Check ratio ---
    ratio = grad_damped / (grad_orig + 1e-12)
    mean_ratio = ratio.abs().mean().item()

    # The mean ratio should be close to the damping factor
    assert torch.allclose(
        mean_ratio * torch.ones_like(torch.tensor(mean_ratio)),
        torch.tensor(damping_factor),
        atol=0.05,
    ), f"Expected gradient ratio ≈ {damping_factor}, got {mean_ratio}"
