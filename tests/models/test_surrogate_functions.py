"""Dedicated tests for the Erf, Triangle, SuperSpike and PoissonRandomSpike
surrogates.

Every surrogate has the same contract (see
:class:`btorch.models.surrogate.SurrogateFunctionBase`):

* ``spiking=True``: forward is the Heaviside step (``x >= 0``), backward is the
  surrogate derivative ``damping * g(x) * grad_output``.
* ``spiking=False``: forward is the smooth *primitive* function; the backward
  still uses the surrogate derivative.
* The custom ``autograd.Function`` supports ``torch.func`` transforms
  (``jvp``, ``vjp``, ``vmap``).

The analytic formulas below are written out independently of the library code,
so the tests double as a specification of each surrogate.
"""

import math

import pytest
import torch
from torch.func import grad, jvp, vjp, vmap

from btorch.models.surrogate import (
    Erf,
    PoissonRandomSpike,
    SuperSpike,
    Triangle,
    erf,
    poisson_random_spike,
    superspike,
    triangle,
)


# Evaluation grid in float64 so finite differences are accurate.
X = torch.linspace(-2.0, 2.0, 41, dtype=torch.float64)
ALPHA = 2.0


# name -> (constructor, analytic derivative g(x; alpha), analytic primitive,
#          ratio d(primitive)/dx / g(x))
# g is normalised to g(0) = 1, so the primitive's slope is a constant multiple.
def _erf_g(x, a):
    return 2.0 ** (-((a * x) ** 2))


def _erf_prim(x, a):
    k = math.sqrt(math.log(2.0))
    return 0.5 * torch.erfc(-k * a * x)


def _tri_g(x, a):
    return (1.0 - (a * x / 2.0).abs()).clamp(min=0.0)


def _tri_prim(x, a):
    # Integral of the triangle pulse (height a/2, half-width 2/a): a piecewise
    # quadratic CDF going 0 -> 1.
    u = a * x / 2.0
    return torch.where(
        u < -1,
        torch.zeros_like(x),
        torch.where(
            u < 0,
            0.5 * (1 + u) ** 2,
            torch.where(u < 1, 1 - 0.5 * (1 - u) ** 2, torch.ones_like(x)),
        ),
    )


def _ss_g(x, a):
    k = math.sqrt(2.0) - 1.0
    return 1.0 / (1.0 + k * a * x.abs()) ** 2


def _ss_prim(x, a):
    k = math.sqrt(2.0) - 1.0
    return torch.where(x >= 0, 1.0 - 0.5 / (1.0 + k * a * x), 0.5 / (1.0 - k * a * x))


_CASES = {
    "erf": (
        Erf,
        _erf_g,
        _erf_prim,
        lambda a: math.sqrt(math.log(2.0)) * a / math.sqrt(math.pi),
    ),
    "triangle": (Triangle, _tri_g, _tri_prim, lambda a: a / 2.0),
    "superspike": (
        SuperSpike,
        _ss_g,
        _ss_prim,
        lambda a: 0.5 * (math.sqrt(2.0) - 1.0) * a,
    ),
}


@pytest.fixture(params=list(_CASES))
def case(request):
    cls, g, prim, ratio = _CASES[request.param]
    return request.param, cls, g, prim, ratio


def _backward_grad(fn, x):
    """Return d sum(fn(x)) / dx, i.e. the surrogate derivative at ``x``."""
    x = x.clone().requires_grad_(True)
    fn(x).sum().backward()
    return x.grad


def test_forward_is_heaviside(case):
    """With ``spiking=True`` the forward pass is ``(x >= 0)`` for every
    surrogate, including exactly 1 at x == 0."""
    _, cls, *_ = case
    x = torch.tensor([-1.0, -1e-3, 0.0, 1e-3, 1.0])
    out = cls(alpha=ALPHA, spiking=True)(x)
    assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))
    assert out.dtype == x.dtype


def test_derivative_matches_analytic_formula(case):
    """The backward pass equals ``g(x)`` of the documented formula."""
    _, cls, g, *_ = case
    got = _backward_grad(cls(alpha=ALPHA), X)
    torch.testing.assert_close(got, g(X, ALPHA))
    # The surrogate peaks at the threshold with value 1 (damping = 1).
    assert got[len(X) // 2].item() == pytest.approx(1.0)


def test_derivative_is_linear_in_grad_output(case):
    """``grad_output`` multiplies the surrogate derivative (chain rule)."""
    _, cls, g, *_ = case
    x = X.clone().requires_grad_(True)
    w = torch.linspace(0.5, 1.5, len(X), dtype=torch.float64)
    (cls(alpha=ALPHA)(x) * w).sum().backward()
    torch.testing.assert_close(x.grad, w * g(X, ALPHA))


def test_damping_scales_gradient(case):
    """``damping_factor`` rescales the gradient and does not change the forward
    values."""
    _, cls, g, *_ = case
    base = _backward_grad(cls(alpha=ALPHA, damping_factor=1.0), X)
    damped = _backward_grad(cls(alpha=ALPHA, damping_factor=0.25), X)
    torch.testing.assert_close(damped, 0.25 * base)
    torch.testing.assert_close(
        cls(alpha=ALPHA, damping_factor=0.25)(X), cls(alpha=ALPHA)(X)
    )


def test_primitive_values_and_slope(case):
    """``spiking=False`` returns the smooth primitive.

    Its values match the closed form, it rises monotonically from 0 to 1
    with value 0.5 at the threshold, and its slope is a constant
    multiple of the (peak-normalised) surrogate derivative.
    """
    _, cls, g, prim, ratio = case
    fn = cls(alpha=ALPHA, spiking=False)
    out = fn(X)
    torch.testing.assert_close(out, prim(X, ALPHA))
    assert out[len(X) // 2].item() == pytest.approx(0.5)
    assert (out[1:] >= out[:-1]).all()
    assert out.min() >= 0.0 and out.max() <= 1.0

    # d(primitive)/dx computed by autograd through the primitive itself
    # (``module.primitive`` bypasses the surrogate backward).
    x = X.clone().requires_grad_(True)
    fn.primitive(x).sum().backward()
    torch.testing.assert_close(x.grad, ratio(ALPHA) * g(X, ALPHA))

    # ...whereas the module's backward still uses the surrogate derivative.
    torch.testing.assert_close(_backward_grad(fn, X), g(X, ALPHA))


def test_functional_wrappers_match_modules():
    """The lower-case functions are thin wrappers around the modules."""
    for fn, cls in [(erf, Erf), (triangle, Triangle), (superspike, SuperSpike)]:
        torch.testing.assert_close(
            fn(X, alpha=3.0, damping_factor=0.5, spiking=False),
            cls(alpha=3.0, damping_factor=0.5, spiking=False)(X),
        )


def test_erf_variance_alias():
    """``variance`` is an alternative to ``alpha``: alpha =
    1/sqrt(variance)."""
    m = Erf(variance=0.25)
    assert m.alpha == pytest.approx(2.0)


def test_triangle_has_compact_support():
    """The triangle surrogate is exactly zero beyond |x| = 2 / alpha."""
    g_out = _backward_grad(Triangle(alpha=ALPHA), torch.tensor([-1.5, -1.0, 1.0, 1.5]))
    assert g_out[0] == 0.0 and g_out[-1] == 0.0
    assert g_out[1] == 0.0 and g_out[2] == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_is_finite(case, dtype):
    """Gradients are finite for all inputs, also far from the threshold and in
    float32 (no 0 * inf or division-by-zero in the formulas)."""
    _, cls, *_ = case
    x = torch.linspace(-50.0, 50.0, 1001, dtype=dtype)
    for spiking in (True, False):
        g = _backward_grad(cls(alpha=ALPHA, spiking=spiking), x)
        assert torch.isfinite(g).all()
        assert g.dtype == dtype


# --- torch.func transforms -------------------------------------------------


def test_jvp_matches_derivative(case):
    """Forward-mode: jvp(fn)(x; t) = g(x) * t (output primal is Heaviside)."""
    _, cls, g, *_ = case
    fn = cls(alpha=ALPHA, damping_factor=0.5)
    t = torch.linspace(-1.0, 1.0, len(X), dtype=torch.float64)
    primal, tangent = jvp(fn, (X,), (t,))
    assert torch.equal(primal, (X >= 0).to(X))
    torch.testing.assert_close(tangent, 0.5 * g(X, ALPHA) * t)


def test_vjp_matches_derivative(case):
    """Reverse-mode through ``torch.func.vjp`` agrees with ``.backward()``."""
    _, cls, g, *_ = case
    fn = cls(alpha=ALPHA)
    cot = torch.linspace(0.5, 2.0, len(X), dtype=torch.float64)
    _, vjp_fn = vjp(fn, X)
    (got,) = vjp_fn(cot)
    torch.testing.assert_close(got, cot * g(X, ALPHA))


def test_vmap_over_per_sample_grad(case):
    """``vmap(grad(f))`` gives per-sample derivatives in one call.

    ``generate_vmap_rule`` makes the autograd Function batchable.
    """
    _, cls, g, *_ = case
    fn = cls(alpha=ALPHA)

    def scalar_fn(x):
        return fn(x)

    per_sample = vmap(grad(scalar_fn))(X)
    torch.testing.assert_close(per_sample, g(X, ALPHA))

    # vmap of the forward alone also works and equals the eager result.
    torch.testing.assert_close(vmap(fn)(X.view(-1, 1)).view(-1), fn(X))


# --- PoissonRandomSpike ----------------------------------------------------


def test_poisson_non_spiking_is_leaky_relu_primitive():
    """``spiking=False``: forward is ``k * x`` for x >= 0 and ``leak * x``
    otherwise; the gradient is the piecewise-constant ``k`` / ``leak``."""
    m = PoissonRandomSpike(spiking=False, k=1.5, leak=0.1)
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    y = m(x)
    torch.testing.assert_close(y, torch.tensor([-0.2, -0.05, 0.0, 0.75, 3.0]))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.1, 0.1, 1.5, 1.5, 1.5]))


def test_poisson_spiking_forward_is_stochastic_binary():
    """``spiking=True`` draws Bernoulli spikes with probability ``exp(rho * x)
    / tau`` (clipped at 1): saturated inputs are deterministic, others have the
    right mean rate."""
    torch.manual_seed(0)
    m = PoissonRandomSpike(spiking=True, tau=1.0, rho=1.0)
    # exp(10) > 1: always spikes; exp(-50) ~ 2e-22: never.
    assert m(torch.full((1000,), 10.0)).eq(1.0).all()
    assert m(torch.full((1000,), -50.0)).eq(0.0).all()
    # Rate 0.3 at x = ln(0.3).
    x = torch.full((200_000,), math.log(0.3))
    out = m(x)
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    assert out.mean().item() == pytest.approx(0.3, abs=0.01)


def test_poisson_gradient_damping_and_derivative_method():
    """The backward uses the piecewise-constant gradient scaled by damping and
    ``derivative`` agrees with it (same signature as the base class)."""
    m = PoissonRandomSpike(spiking=True, k=2.0, leak=0.2, damping_factor=0.5)
    x = torch.tensor([-1.0, 1.0], requires_grad=True)
    m(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.1, 1.0]))

    ones = torch.ones(2)
    torch.testing.assert_close(m.derivative(x.detach(), ones, m.damping_factor), x.grad)
    torch.testing.assert_close(
        m.derivative(x.detach(), 3 * ones), 3 * torch.tensor([0.2, 2.0])
    )


def test_poisson_torch_func_transforms():
    """Jvp / vjp / vmap work through the Poisson spike function."""
    m = PoissonRandomSpike(spiking=False, k=1.0, leak=0.25, damping_factor=0.5)
    x = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float64)
    expected = 0.5 * torch.tensor([0.25, 0.25, 1.0, 1.0], dtype=torch.float64)

    t = torch.ones_like(x)
    _, tangent = jvp(m, (x,), (t,))
    torch.testing.assert_close(tangent, expected)

    _, vjp_fn = vjp(m, x)
    (cot_grad,) = vjp_fn(torch.ones_like(x))
    torch.testing.assert_close(cot_grad, expected)

    torch.testing.assert_close(vmap(grad(lambda v: m(v)))(x), expected)


def test_poisson_functional_wrapper():
    x = torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(
        poisson_random_spike(x, spiking=False, leak=0.5, k=2.0),
        PoissonRandomSpike(spiking=False, leak=0.5, k=2.0)(x),
    )
