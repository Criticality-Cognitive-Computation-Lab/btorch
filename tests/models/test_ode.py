"""Tests for ``btorch.models.ode`` integrators.

Both integrators advance ``dx/dt = f(x)`` by one step ``dt``:

* :func:`euler_step`     ``x + dt * f(x)`` (first order)
* :func:`exp_euler_step` ``x + (exp(dt*A) - 1)/A * f(x)`` where ``A = df/dx``
  is the linear coefficient (exact for linear ODEs).

The tests check them against closed-form solutions.
"""

import math
from decimal import Decimal, localcontext

import pytest
import torch

from btorch.models.ode import euler_step, exp_euler_step


TAU = 10.0


@pytest.mark.parametrize("explicit_linear", (False, True))
def test_exp_euler_zero_linear_coefficient_has_continuous_ode_gradient(
    explicit_linear,
):
    """Differentiate the exact affine-ODE solution through zero decay.

    For dx/dt = a*x + b, exponential Euler is exact. At a=0 its value is
    x + dt*b, but the sensitivity to a is dt*x + dt**2*b/2. Selecting a
    constant multiplier at that one point incorrectly drops the second
    term. This checks a mathematical limit, not agreement with a second
    integrator.
    """
    initial = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)
    rate = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    drive = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    dt = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

    def affine_ode(value, coefficient, forcing):
        derivative = coefficient * value + forcing
        return (derivative, coefficient) if explicit_linear else derivative

    actual = exp_euler_step(affine_ode, initial, rate, drive, dt=dt)
    gradients = torch.autograd.grad(actual, (initial, rate, drive, dt))
    torch.testing.assert_close(actual, initial + dt * drive)
    expected = (
        torch.ones_like(initial),
        dt * initial + 0.5 * dt.square() * drive,
        dt,
        drive,
    )
    for value, reference in zip(gradients, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_exp_euler_affine_solution_and_broadcast_vjp_near_zero(dtype):
    """Compare the affine ODE with a high-precision closed-form solution.

    Independent initial states share a per-neuron rate, scalar forcing
    and scalar timestep. This exercises the parameter reductions used
    when fitting a population, including rates on both sides of zero and
    outside the series region. Decimal arithmetic evaluates the analytic
    exponential, not another copy of the numerical integrator.
    """
    initial = torch.tensor(
        [[1.25, -0.7, 0.1, 0.8, -0.2], [-0.5, 0.4, 1.2, -0.3, 0.6]],
        dtype=dtype,
        requires_grad=True,
    )
    rate = torch.tensor([-1e-6, 0.0, 1e-6, -2.0, 2.0], dtype=dtype, requires_grad=True)
    drive = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    dt = torch.tensor(0.25, dtype=dtype, requires_grad=True)
    cotangent = torch.tensor(
        [[0.3, -0.4, 0.5, 1.0, -0.7], [-0.6, 0.8, -0.2, 0.4, 0.9]], dtype=dtype
    )

    def affine_ode(value, coefficient, forcing):
        return coefficient * value + forcing, coefficient

    actual = exp_euler_step(affine_ode, initial, rate, drive, dt=dt)
    gradients = torch.autograd.grad(actual, (initial, rate, drive, dt), cotangent)
    coefficients = []
    with localcontext() as context:
        context.prec = 60
        time = Decimal(str(float(dt.detach())))
        for value in rate.detach():
            coefficient = Decimal(str(float(value)))
            exponential = (time * coefficient).exp()
            if coefficient:
                phi = (exponential - 1) / coefficient
                phi_derivative = (
                    time * coefficient * exponential - exponential + 1
                ) / (coefficient * coefficient)
            else:
                phi, phi_derivative = time, time * time / 2
            coefficients.append((float(exponential), float(phi), float(phi_derivative)))
    exponential, phi, phi_derivative = torch.tensor(coefficients, dtype=dtype).unbind(
        -1
    )
    expected = initial * exponential + drive * phi
    expected_gradients = (
        cotangent * exponential,
        (cotangent * (dt * exponential * initial + drive * phi_derivative)).sum(0),
        (cotangent * phi).sum(),
        (cotangent * exponential * (rate * initial + drive)).sum(),
    )
    tolerance = 3e-6 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for value, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(value, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "dtype,tiny", ((torch.float32, 1e-40), (torch.float64, 1e-310))
)
def test_exp_euler_subnormal_rate_keeps_unused_quotient_backward_finite(dtype, tiny):
    """Keep the continuous affine sensitivity even for subnormal rates.

    Masking only the final quotient value is insufficient: its inactive
    backward can evaluate zero divided by an underflowed denominator and
    contaminate the selected polynomial's otherwise finite gradient.
    """
    initial = torch.tensor(1.25, dtype=dtype)
    rate = torch.tensor(tiny, dtype=dtype, requires_grad=True)
    result = exp_euler_step(lambda x: (rate * x + 0.7, rate), initial, dt=0.25)
    (gradient,) = torch.autograd.grad(result, (rate,))
    torch.testing.assert_close(gradient, torch.tensor(0.334375, dtype=dtype))


@pytest.mark.parametrize("rate_value", (0.0, 1e-5, -2.0))
def test_exp_euler_affine_parameter_second_derivatives(rate_value):
    """Support second-order fitting through zero and ordinary decay rates.

    The solver is exact for this affine ODE, so differentiating its
    parameter VJP should agree with finite differences on both sides of
    the series implementation. No spike surrogate or event discontinuity
    is involved.
    """
    roots = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (1.25, rate_value, 0.7, 0.25)
    )

    def solve(initial, rate, drive, dt):
        return exp_euler_step(
            lambda value: (rate * value + drive, rate), initial, dt=dt
        )

    assert torch.autograd.gradcheck(solve, roots)
    assert torch.autograd.gradgradcheck(solve, roots)


@pytest.mark.parametrize(
    "dtype,rate_value", ((torch.float32, -1e10), (torch.float64, -1e100))
)
def test_exp_euler_stiff_decay_keeps_unused_series_backward_finite(dtype, rate_value):
    """Fit a stiff relaxation rate without overflowing an inactive series.

    Starting from zero with unit forcing gives the exact value -1/a and
    sensitivity 1/a**2 after the transient underflows. The far-field
    quotient is finite, even though evaluating a sixth-degree polynomial
    at dt*a would overflow. Its unselected branch must not poison this
    gradient.
    """
    initial = torch.tensor(0.0, dtype=dtype)
    rate = torch.tensor(rate_value, dtype=dtype, requires_grad=True)
    result = exp_euler_step(lambda x: (rate * x + 1.0, rate), initial, dt=0.25)
    (gradient,) = torch.autograd.grad(result, (rate,))
    torch.testing.assert_close(result, -rate.reciprocal(), atol=0, rtol=1e-6)
    torch.testing.assert_close(gradient, rate.reciprocal().square(), atol=0, rtol=1e-6)


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
