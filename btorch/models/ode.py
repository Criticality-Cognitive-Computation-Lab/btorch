from typing import Callable

import torch
from torch import Tensor
from torch.func import vjp


def exp_euler_step_auto(f: Callable, *args, dt=1.0):
    """One integation step applying exponential euler method.

    dx/dt = f(x), f(x) = Ax + B, where A is linear term and f is
    derivative
    """
    if len(args) > 1:
        _f = lambda x: f(x, *args[1:])
    else:
        _f = f
    derivative, linear_f = vjp(_f, args[0])
    linear = linear_f(torch.ones_like(derivative))[0]
    return args[0] + torch.expm1(dt * linear) / linear * derivative


def exp_euler_step(linear: Tensor, f: Callable, *args, dt=1.0):
    """Faster if you know the linear term A in dV/dt = aV + g.

    No difference after going through jit or compilation.
    """
    derivative = f(*args)
    return args[0] + torch.expm1(dt * linear) / linear * derivative


def euler_step(f: Callable, *args, dt=1.0):
    return args[0] + dt * f(*args)
