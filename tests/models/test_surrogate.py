import pytest
import torch

from btorch.models.surrogate import ATan, atan, Erf, Sigmoid


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
