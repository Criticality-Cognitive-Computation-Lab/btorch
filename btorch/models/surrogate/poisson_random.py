import torch

from .base import SurrogateFunctionBase


class _PoissonRandomSpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        tau: float,
        rho: float,
        leak: float,
        k: float,
        damping: float,
        spiking: bool,
    ):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.tau = tau
            ctx.rho = rho
            ctx.leak = leak
            ctx.k = k
            ctx.damping = damping

        fr = torch.exp(rho * x) / tau
        if spiking:
            return (fr > torch.rand_like(fr)).to(x)

        # primitive (leaky ReLU-style) when not spiking
        mask = (x >= 0.0).to(x)
        return (leak * (1.0 - mask) + k * mask) * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x >= 0.0).to(x)
        grad_x = mask * ctx.k + (1.0 - mask) * ctx.leak
        grad_x = grad_x * ctx.damping
        return grad_output * grad_x, None, None, None, None, None, None


class PoissonRandomSpike(SurrogateFunctionBase):
    """Stochastic Poisson spike surrogate with piecewise constant gradient."""

    def __init__(
        self,
        spiking: bool = True,
        tau: float = 1.0,
        rho: float = 1.0,
        leak: float = 0.0,
        k: float = 1.0,
        damping_factor: float = 1.0,
    ):
        super().__init__(alpha=1.0, damping_factor=damping_factor, spiking=spiking)
        self.tau = tau
        self.rho = rho
        self.leak = leak
        self.k = k

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x >= 0.0).to(x)
        return (self.leak * (1.0 - mask) + self.k * mask) * x

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x >= 0.0).to(x)
        return mask * self.k + (1.0 - mask) * self.leak

    def forward(self, x: torch.Tensor):
        return _PoissonRandomSpikeFn.apply(
            x, self.tau, self.rho, self.leak, self.k, self.damping_factor, self.spiking
        )
