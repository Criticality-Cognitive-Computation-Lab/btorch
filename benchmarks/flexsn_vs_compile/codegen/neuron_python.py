"""The "complicated LIF" neuron used by the FlexSN-vs-torch.compile benchmark.

This is the *reference* neuron the generated kernels implement. It is extracted
from ``benchmarks/flexsn_vs_compile/kernels.py`` (the eager/torch reference;
the CuPy, FlexSN and compiled backends all match it numerically).

Per-step dynamics:

    h   = beta * v + x
    s1  = sg(h - (rho + 1))          # adaptive-threshold spike
    s2  = sg(h - 1)                  # fixed-threshold spike
    rho = gamma * rho + s1           # threshold adaptation
    yy  = sigmoid(y)                 # modulation factor
    v   = (h * (1 - s1)) * yy + (h - s2) * (1 - yy)

``sg`` is the straight-through ATan surrogate: Heaviside forward, ATan-shaped
backward (identical to ``btorch.models.surrogate.ATan(alpha=2)``).
"""

import math

import torch


BETA = 0.9
GAMMA = 0.8
ALPHA = 2.0  # ATan alpha


def atan_sg(x: torch.Tensor) -> torch.Tensor:
    # Heaviside forward, ATan-shaped backward. Written as a pure-torch
    # straight-through (not an autograd.Function) so it stays traceable through
    # torch.compile + the scan HOP.
    spike = (x >= 0.0).to(x)
    soft = 0.5 + torch.atan(0.5 * math.pi * ALPHA * x) / math.pi
    return soft + (spike - soft).detach()


def core_step(x, y, v, rho, spike_fn=atan_sg):
    h = BETA * v + x
    s1 = spike_fn(h - (rho + 1.0))
    s2 = spike_fn(h - 1.0)
    rho = GAMMA * rho + s1
    v1 = h * (1.0 - s1)
    v2 = h - s2
    yy = torch.sigmoid(y)
    v = v1 * yy + v2 * (1.0 - yy)
    return s1, s2, v, rho


def eager_loop(x_seq, y_seq, v0, rho0):
    """Multistep reference: plain unrolled Python time loop."""
    T = x_seq.shape[0]
    v, rho = v0, rho0
    s1_l, s2_l = [], []
    for t in range(T):
        s1, s2, v, rho = core_step(x_seq[t], y_seq[t], v, rho)
        s1_l.append(s1)
        s2_l.append(s2)
    return torch.stack(s1_l), torch.stack(s2_l), v, rho


def eager_loop_save(x_seq, y_seq, v0, rho0):
    """Same loop, additionally returning the per-step history FlexSN keeps.

    ``h``, the pre-update ``rho`` and ``sigmoid(y)`` are exactly the residuals
    FlexSN's forward stores and its backward reloads. Returning them as outputs
    is the "plain autograd" way to ask AOTAutograd to keep them; note that it
    does *not* actually stop inductor's min-cut partitioner from rematerializing
    the step (see ../TRITON_CODEGEN.md).
    """
    T = x_seq.shape[0]
    v, rho = v0, rho0
    s1_l, s2_l, h_l, rho_prev_l, yy_l = [], [], [], [], []
    for t in range(T):
        h = BETA * v + x_seq[t]
        s1 = atan_sg(h - (rho + 1.0))
        s2 = atan_sg(h - 1.0)
        rho_prev_l.append(rho)
        rho = GAMMA * rho + s1
        yy = torch.sigmoid(y_seq[t])
        v = (h * (1.0 - s1)) * yy + (h - s2) * (1.0 - yy)
        s1_l.append(s1)
        s2_l.append(s2)
        h_l.append(h)
        yy_l.append(yy)
    return (
        torch.stack(s1_l),
        torch.stack(s2_l),
        v,
        rho,
        torch.stack(h_l),
        torch.stack(rho_prev_l),
        torch.stack(yy_l),
    )
