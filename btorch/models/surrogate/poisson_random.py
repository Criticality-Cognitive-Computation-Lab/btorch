import torch

from .base import SurrogateFunctionBase


def _poisson_grad(
    x: torch.Tensor, k: float, leak: float, damping: float
) -> torch.Tensor:
    mask = (x >= 0.0).to(x)
    return (mask * k + (1.0 - mask) * leak) * damping


class _PoissonRandomSpikeFn(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(
        x: torch.Tensor,
        tau: float,
        rho: float,
        leak: float,
        k: float,
        damping: float,
        spiking: bool,
    ):
        fr = torch.exp(rho * x) / tau
        if spiking:
            return (fr > torch.rand_like(fr)).to(x)

        # primitive (leaky ReLU-style) when not spiking
        mask = (x >= 0.0).to(x)
        return (leak * (1.0 - mask) + k * mask) * x

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, _tau, _rho, leak, k, damping, _spiking = inputs
        ctx.save_for_backward(x)
        ctx.save_for_forward(x)
        ctx.leak = leak
        ctx.k = k
        ctx.damping = damping

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = _poisson_grad(x, ctx.k, ctx.leak, ctx.damping)
        return grad_output * grad_x, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx, x_tangent, *_non_tensor_tangents):
        (x,) = ctx.saved_tensors
        return x_tangent * _poisson_grad(x, ctx.k, ctx.leak, ctx.damping)


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

    def derivative(
        self,
        x: torch.Tensor,
        grad_output: torch.Tensor,
        damping_factor: float = 1.0,
    ) -> torch.Tensor:
        return grad_output * _poisson_grad(x, self.k, self.leak, damping_factor)

    def forward(self, x: torch.Tensor):
        return _PoissonRandomSpikeFn.apply(
            x, self.tau, self.rho, self.leak, self.k, self.damping_factor, self.spiking
        )


def poisson_random_spike(
    x: torch.Tensor,
    spiking: bool = True,
    tau: float = 1.0,
    rho: float = 1.0,
    leak: float = 0.0,
    k: float = 1.0,
    damping_factor: float = 1.0,
) -> torch.Tensor:
    return PoissonRandomSpike(
        spiking=spiking,
        tau=tau,
        rho=rho,
        leak=leak,
        k=k,
        damping_factor=damping_factor,
    )(x)
