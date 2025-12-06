import torch
from torch import nn


class HasConstraint:
    def constrain(self, *args, **kwargs):
        raise NotImplementedError()


def constrain_net(mod: nn.Module):
    with torch.no_grad():
        for mod in mod.modules():
            if isinstance(mod, HasConstraint):
                mod.constrain()
