from collections.abc import Callable

import torch
from torch import Tensor
from torch.func import vjp


_EXP_EULER_SERIES_CUTOFF = 1.0e-2


def _split_derivative_linear(out):
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError("Expected (derivative, linear) from ODE function.")
        return out
    return out, None


def _derivative_only(out):
    return out[0] if isinstance(out, tuple) else out


def _exp_euler_coefficient(linear: Tensor, dt: float | Tensor) -> tuple[Tensor, Tensor]:
    """Return the exponential-Euler multiplier and its expm1 numerator.

    The quotient ``expm1(dt * linear) / linear`` is well-conditioned as a
    mathematical function near zero, but its direct reverse subtraction is
    not. Evaluate the equivalent ``dt * exprel(dt * linear)`` series in that
    region so autograd sees the continuous derivative, including
    ``d phi / d linear = dt**2 / 2`` at zero. The direct quotient remains the
    far-from-zero graph. At the dimensionless cutoff |dt * linear| = 0.01,
    the degree-six exprel series has absolute remainder below 2.5e-19,
    smaller than FP64 rounding. Mask its inactive denominator as well as its
    output so subnormal rates cannot create NaNs in autograd.
    """
    z = dt * linear
    use_series = torch.abs(z) <= _EXP_EULER_SERIES_CUTOFF
    safe_linear = torch.where(use_series, torch.ones_like(linear), linear)
    delta = torch.expm1(z)
    ratio = delta / safe_linear
    # Both torch.where branches participate in autograd. Keep the inactive
    # polynomial bounded for stiff decay, just as its quotient is protected
    # near zero; otherwise an overflowed polynomial can contribute 0 * inf.
    series_z = torch.where(use_series, z, torch.zeros_like(z))
    exprel = 1.0 + series_z * (
        0.5
        + series_z
        * (
            1.0 / 6.0
            + series_z
            * (
                1.0 / 24.0
                + series_z
                * (1.0 / 120.0 + series_z * (1.0 / 720.0 + series_z / 5040.0))
            )
        )
    )
    series = dt * exprel
    phi = torch.where(use_series, series, ratio)
    return phi, delta


def exp_euler_step(
    f: Callable,
    *args: Tensor,
    dt: float | Tensor = 1.0,
    linear: Tensor | None = None,
) -> Tensor:
    """One integration step applying the exponential Euler method.

    .. math::
        \frac{dx}{dt} = f(x) = Ax + B

    where :math:`A` is the linear term and :math:`f(x)` is the derivative.

    The update rule is:

    .. math::
        x_{n+1} = x_n + \frac{e^{dt A} - 1}{A} f(x_n) \\
        &= e^{dt A}x_n + \frac{e^{dt A} - 1}{A} B
    """
    out = f(*args)
    derivative, linear_from_f = _split_derivative_linear(out)
    if linear is None:
        linear = linear_from_f
    if linear is None:
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "torch.compile cannot use vjp fallback here; return "
                "(derivative, linear) from f(*args) or pass linear=."
            )
        if len(args) > 1:
            _f = lambda x: _derivative_only(f(x, *args[1:]))
        else:
            _f = lambda x: _derivative_only(f(x))
        derivative, linear_f = vjp(_f, args[0])
        linear = linear_f(torch.ones_like(derivative))[0]
    # (e^{dt A} - 1) / A -> dt as A -> 0; the safe denominator keeps the
    # unused branch (and its gradient) finite.
    phi, _ = _exp_euler_coefficient(linear, dt)
    return args[0] + phi * derivative


def euler_step(f: Callable, *args: Tensor, dt: float | Tensor = 1.0) -> Tensor:
    derivative = _derivative_only(f(*args))
    return args[0] + dt * derivative
