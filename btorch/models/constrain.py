from typing import Any

import torch
from torch import nn


class HasConstraint:
    """Mixin for modules that project their parameters onto a constraint
    set."""

    def constrain(self, *args: Any, **kwargs: Any) -> None:
        """Project parameters in place (called under ``torch.no_grad``)."""
        raise NotImplementedError()


def constrain_net(net: nn.Module) -> None:
    """Call ``constrain()`` on every :class:`HasConstraint` module in ``net``.

    Args:
        net: Network whose submodules (including ``net`` itself) are visited.
    """
    with torch.no_grad():
        for mod in net.modules():
            if isinstance(mod, HasConstraint):
                mod.constrain()
