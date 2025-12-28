from collections.abc import Callable

import torch
from torch import Tensor
from torch.func import vjp


def _split_derivative_linear(out):
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError("Expected (derivative, linear) from ODE function.")
        return out
    return out, None


def _derivative_only(out):
    return out[0] if isinstance(out, tuple) else out


def exp_euler_step(f: Callable, *args, dt=1.0, linear: Tensor | None = None):
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
    return args[0] + torch.expm1(dt * linear) / linear * derivative


def euler_step(f: Callable, *args, dt=1.0):
    derivative = _derivative_only(f(*args))
    return args[0] + dt * derivative
