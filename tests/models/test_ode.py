"""Tests for ``btorch.models.ode`` integrators.

Both integrators advance ``dx/dt = f(x)`` by one step ``dt``:

* :func:`euler_step`     ``x + dt * f(x)`` (first order)
* :func:`exp_euler_step` ``x + (exp(dt*A) - 1)/A * f(x)`` where ``A = df/dx``
  is the linear coefficient (exact for linear ODEs).

The tests check them against closed-form solutions.
"""

import math

import pytest
import torch

from btorch.models.ode import euler_step, exp_euler_step


TAU = 10.0


def _decay(x):
    """Dx/dt = -x / TAU, solution x(t) = x0 * exp(-t / TAU)."""
    return -x / TAU


def _decay_with_linear(x):
    """Same ODE, returning ``(derivative, linear)`` explicitly."""
    return -x / TAU, torch.full_like(x, -1.0 / TAU)


def _integrate(step, x0, dt, n_steps, **kw):
    x = x0
    for _ in range(n_steps):
        x = step(**kw, x=x, dt=dt)
    return x


def test_exp_euler_is_exact_for_linear_ode_even_with_large_dt():
    """Exponential Euler reproduces exp decay exactly, independent of dt."""
    x0 = torch.tensor([1.0, -3.0, 0.5], dtype=torch.float64)
    for dt in (0.1, 1.0, 50.0):  # dt >> TAU must still be exact
        x1 = exp_euler_step(_decay_with_linear, x0, dt=dt)
        expected = x0 * math.exp(-dt / TAU)
        torch.testing.assert_close(x1, expected, rtol=1e-12, atol=1e-12)


def test_exp_euler_vjp_fallback_matches_explicit_linear():
    """If ``f`` returns only a derivative, the linear term is recovered by
    autograd (vjp) and gives the same result as passing it explicitly."""
    x0 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    fallback = exp_euler_step(_decay, x0, dt=2.0)
    explicit = exp_euler_step(_decay_with_linear, x0, dt=2.0)
    kwarg = exp_euler_step(_decay, x0, dt=2.0, linear=torch.full_like(x0, -1 / TAU))
    torch.testing.assert_close(fallback, explicit)
    torch.testing.assert_close(kwarg, explicit)


def test_exp_euler_extra_args_are_passed_through():
    """Extra positional args reach ``f``; only the first is integrated.

    dx/dt = -x/TAU + I has fixed point x* = TAU * I, and exp-Euler with the
    exact linear coefficient lands on the analytic solution
    ``x* + (x0 - x*) exp(-dt/TAU)``.
    """

    def f(x, current):
        return -x / TAU + current, torch.full_like(x, -1.0 / TAU)

    x0 = torch.zeros(2, dtype=torch.float64)
    current = torch.tensor([0.5, 1.0], dtype=torch.float64)
    dt = 3.0
    x1 = exp_euler_step(f, x0, current, dt=dt)
    x_star = TAU * current
    torch.testing.assert_close(x1, x_star + (x0 - x_star) * math.exp(-dt / TAU))

    # same thing through the vjp fallback (derivative-only f)
    x1_fb = exp_euler_step(lambda x, i: -x / TAU + i, x0, current, dt=dt)
    torch.testing.assert_close(x1_fb, x1)


def test_euler_and_exp_euler_converge_to_analytic_solution():
    """Explicit Euler error shrinks ~O(dt); exp Euler stays at round-off."""
    x0 = torch.tensor([1.0], dtype=torch.float64)
    t_end = 20.0
    exact = x0 * math.exp(-t_end / TAU)

    errs = []
    for dt in (2.0, 1.0, 0.5):
        n = int(t_end / dt)
        x_e = _integrate(lambda x, dt: euler_step(_decay, x, dt=dt), x0, dt, n)
        errs.append(float((x_e - exact).abs()))
        x_x = _integrate(
            lambda x, dt: exp_euler_step(_decay_with_linear, x, dt=dt), x0, dt, n
        )
        assert float((x_x - exact).abs()) < 1e-12

    # halving dt roughly halves the first-order error
    assert errs[0] > errs[1] > errs[2]
    assert errs[1] / errs[2] == pytest.approx(2.0, rel=0.15)


def test_euler_step_single_step_value():
    """X + dt * f(x) with f(x) = -x/TAU: 1 + 2 * (-0.1) = 0.8."""
    out = euler_step(_decay, torch.tensor(1.0), dt=2.0)
    assert out.item() == pytest.approx(0.8)


def test_euler_step_accepts_tuple_returning_f():
    """``(derivative, linear)`` tuples are accepted; ``linear`` is ignored."""
    out = euler_step(_decay_with_linear, torch.tensor(1.0), dt=2.0)
    assert out.item() == pytest.approx(0.8)


def test_exp_euler_rejects_malformed_tuple():
    """A tuple that is not (derivative, linear) raises ValueError."""
    with pytest.raises(ValueError, match="derivative, linear"):
        exp_euler_step(lambda x: (x, x, x), torch.ones(2))


def test_exp_euler_is_differentiable():
    """Gradients flow through a step: d x1/d x0 = exp(dt * A)."""
    x0 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    x1 = exp_euler_step(_decay_with_linear, x0, dt=5.0)
    x1.backward()
    assert x0.grad.item() == pytest.approx(math.exp(-5.0 / TAU))


def test_exp_euler_zero_linear_term():
    """In the limit A -> 0 exp-Euler must reduce to explicit Euler.

    For dx/dt = 2 (constant drive, A = 0) the exact step is x + 2 dt.
    """

    def f(x):
        return torch.full_like(x, 2.0), torch.zeros_like(x)

    out = exp_euler_step(f, torch.tensor([1.0]), dt=0.5)
    assert torch.isfinite(out).all()
    assert out.item() == pytest.approx(2.0)
