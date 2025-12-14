import torch


_heaviside = torch.jit.script(lambda x: (x >= 0).to(x))
_damp = torch.jit.script(lambda g, d: g * d)


class _SurrogateAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, module: "SurrogateFunctionBase"):
        ctx.module = module
        ctx.save_for_backward(x)
        if module.spiking:
            return _heaviside(x)
        return module.primitive(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        module: SurrogateFunctionBase = ctx.module
        grad = module.derivative(x)
        if module.damping_factor != 1.0:
            grad = _damp(grad, module.damping_factor)
        return grad_output * grad, None


class SurrogateFunctionBase(torch.nn.Module):
    """Minimal surrogate gradient base with optional damping.

    Parameters
    ----------
    alpha : float
        Shape/steepness parameter for the surrogate derivative.
    damping_factor : float
        Scales the surrogate gradient (1.0 keeps it unchanged).
    spiking : bool
        If True, forward returns a Heaviside spike; otherwise returns the
        primitive function.
    """

    def __init__(
        self, alpha: float = 1.0, damping_factor: float = 1.0, spiking: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.damping_factor = damping_factor
        self.spiking = spiking

    def primitive(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        return _SurrogateAutograd.apply(x, self)
