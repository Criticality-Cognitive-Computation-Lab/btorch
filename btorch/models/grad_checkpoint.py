from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    ActivationWrapper,
)
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint

from . import environ
from .functional import named_hidden_states, set_hidden_states


class CheckpointWrapper(ActivationWrapper):
    """An ``nn.Module`` that wraps another ``nn.Module`` with checkpointing.

    Supports variable states in MemoryModule.

    Note that this module is not meant to be used directly but instead,
    it is to be used through the ``checkpoint_wrapper`` function.
    """

    def __init__(
        self,
        mod: torch.nn.Module,
        checkpoint_fn: Callable[..., Any] | None = None,
        **checkpoint_fn_kwargs: Any,
    ) -> None:
        super().__init__(mod)
        if checkpoint_fn is None:
            # use torch.utils.checkpoint
            self.checkpoint_fn = partial(
                torch_utils_checkpoint,
                use_reentrant=False,
                **checkpoint_fn_kwargs,
            )
        else:
            # Construct user-specified checkpoint function.
            self.checkpoint_fn = partial(
                checkpoint_fn,
                **checkpoint_fn_kwargs,
            )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        mem_states = named_hidden_states(self._checkpoint_wrapped_module)
        env = environ.all()

        def _pure_memory_module(env, states, *args, **kwargs):
            set_hidden_states(self._checkpoint_wrapped_module, states)
            with environ.context(**env):
                return self._checkpoint_wrapped_module(*args, **kwargs)

        return self.checkpoint_fn(  # type: ignore[misc]
            _pure_memory_module, env, mem_states, *args, **kwargs
        )


def checkpoint_wrapper(
    module: torch.nn.Module,
    checkpoint_fn: Callable[..., Any] | None = None,
    **checkpoint_fn_kwargs: Any,
) -> torch.nn.Module:
    """Wrap ``module`` so its forward is activation-checkpointed.

    Hidden states of MemoryModules and the active :mod:`environ` settings are
    passed through the checkpoint explicitly so the recompute is faithful.

    Args:
        module: Module to wrap.
        checkpoint_fn: Checkpoint function; defaults to
            :func:`torch.utils.checkpoint.checkpoint` with
            ``use_reentrant=False``.
        **checkpoint_fn_kwargs: Passed to the checkpoint function.

    Returns:
        A :class:`CheckpointWrapper` around ``module``.
    """
    return CheckpointWrapper(
        module,
        checkpoint_fn,
        **checkpoint_fn_kwargs,
    )
