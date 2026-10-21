"""Exercise eager surrogate behavior after removing the TorchScript shim."""

import pytest
import torch
from torch.func import vjp, vmap

from btorch.models.surrogate import (
    ATan,
    ATanApprox,
    Erf,
    PoissonRandomSpike,
    Sigmoid,
    SuperSpike,
    Triangle,
)


_SURROGATES = [
    pytest.param(ATan, {"alpha": 2.0}, id="atan"),
    pytest.param(ATanApprox, {"alpha": 2.0}, id="atan-approx"),
    pytest.param(Sigmoid, {"alpha": 2.0}, id="sigmoid"),
    pytest.param(Erf, {"alpha": 2.0}, id="erf"),
    pytest.param(Triangle, {"alpha": 2.0}, id="triangle"),
    pytest.param(SuperSpike, {"alpha": 2.0}, id="superspike"),
    pytest.param(
        PoissonRandomSpike,
        {"spiking": False, "k": 1.0, "leak": 0.25},
        id="poisson-deterministic",
    ),
]


@pytest.mark.parametrize(("surrogate", "kwargs"), _SURROGATES)
def test_surrogate_repeated_vjp_is_stable(surrogate, kwargs):
    """Every eager surrogate supports repeated VJP evaluation."""
    module = surrogate(damping_factor=0.5, **kwargs)
    x = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64)
    cotangent = torch.linspace(0.5, 1.5, 9, dtype=torch.float64)

    primal, vjp_fn = vjp(lambda value: module(value), x)
    first, second = vjp_fn(cotangent)[0], vjp_fn(cotangent)[0]
    expected = module.derivative(x, cotangent, module.damping_factor)

    if module.spiking:
        torch.testing.assert_close(primal, (x >= 0).to(x))
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert torch.equal(first, second)


def test_atan_vmap_gradient_matches_pointwise_vjp():
    """The eager custom autograd function remains compatible with vmap."""
    module = ATan(alpha=2.0)
    values = torch.linspace(-1.0, 1.0, 13, dtype=torch.float64)

    def scalar_value(value):
        return module(value)

    got = vmap(torch.func.grad(scalar_value))(values)
    expected = 1.0 / (1.0 + (2.0 * values) ** 2)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize(("surrogate", "kwargs"), _SURROGATES)
def test_surrogate_torch_compile_eager_backend_preserves_backward(surrogate, kwargs):
    """Torch compile's eager backend preserves each surrogate's VJP."""
    module = surrogate(damping_factor=0.75, **kwargs)

    def function(value):
        return module(value)

    compiled = torch.compile(function, backend="eager")
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64, requires_grad=True)
    first = compiled(x)
    second = compiled(x)
    if module.spiking:
        expected = (x >= 0).to(x)
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, expected)
    else:
        torch.testing.assert_close(first, module.primitive(x))
        torch.testing.assert_close(second, module.primitive(x))

    gradient = torch.autograd.grad(first.sum(), x, retain_graph=True)[0]
    expected_gradient = module.derivative(x, torch.ones_like(x), module.damping_factor)
    torch.testing.assert_close(gradient, expected_gradient)
