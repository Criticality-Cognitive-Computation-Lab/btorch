import torch
from spikingjelly.activation_based import surrogate


# TODO: not compat with torch.script
def add_damping(damping_factor=1.0):
    def decorator(cls: type[surrogate.SurrogateFunctionBase]):
        class DampedSurrogate(cls):
            def __init__(self, *args, damping_factor=damping_factor, **kwargs):
                super().__init__(*args, **kwargs)
                self.damping_factor = damping_factor

            @staticmethod
            def spiking_function(x, alpha, damping_factor=damping_factor):
                ret = cls.spiking_function(x, alpha)
                # Mix tracked (grad) and detached parts → gradient is damped
                return ret * damping_factor + ret.detach() * (1 - damping_factor)

            @staticmethod
            def primitive_function(x, alpha, damping_factor=damping_factor):
                ret = cls.primitive_function(x, alpha)
                return ret * damping_factor

            def forward(self, x: torch.Tensor):
                if self.spiking:
                    return self.spiking_function(x, self.alpha, self.damping_factor)
                else:
                    return self.primitive_function(x, self.alpha, self.damping_factor)

        DampedSurrogate.__name__ = cls.__name__
        # DampedSurrogate.__qualname__ = cls.__qualname__

        return DampedSurrogate

    return decorator


ATan = add_damping()(surrogate.ATan)
Erf = add_damping()(surrogate.Erf)
