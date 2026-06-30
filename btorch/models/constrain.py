import torch
from torch import nn


class HasConstraint:
    def constrain(self, *args, **kwargs):
        raise NotImplementedError()


def constrain_net(mod: nn.Module):
    with torch.no_grad():
        for child in mod.modules():
            if isinstance(child, HasConstraint) or getattr(
                child, "_btorch_constraint", False
            ):
                child.constrain()
