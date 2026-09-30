import torch

from btorch import jit


@jit.script
def _heaviside(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0).to(x)


class _SurrogateAutograd(torch.autograd.Function):
    """Surrogate-gradient autograd function (functorch-compatible).

    Supports ``torch.func.vjp/jvp/vmap`` via ``setup_context``, ``jvp`` and
    ``generate_vmap_rule``. The surrogate derivative is linear in its
    ``grad_output`` argument, so the same call serves backward and jvp.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor, module: "SurrogateFunctionBase"):
        if module.spiking:
            return _heaviside(x)
        return module.primitive(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, module = inputs
        ctx.module = module
        ctx.save_for_backward(x)
        ctx.save_for_forward(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        module: SurrogateFunctionBase = ctx.module
        grad_input = module.derivative(x, grad_output, module.damping_factor)
        return grad_input, None

    @staticmethod
    def jvp(ctx, x_tangent: torch.Tensor, module_tangent):
        (x,) = ctx.saved_tensors
        module: SurrogateFunctionBase = ctx.module
        return module.derivative(x, x_tangent, module.damping_factor)


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

    def derivative(
        self,
        x: torch.Tensor,
        grad_output: torch.Tensor,
        damping_factor: float = 1.0,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        return _SurrogateAutograd.apply(x, self)
