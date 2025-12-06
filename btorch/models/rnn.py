from collections.abc import Callable, Sequence
from functools import partial
from typing import overload

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from . import base, environ, synapse
from .functional import filter_hidden_states, named_hidden_states, set_hidden_states


# TODO: handle multiple output
class RecurrentNNAbstract(base.MemoryModule):
    def __init__(
        self,
        update_state_names: Sequence[str] | None = None,
        step_mode="m",
        unroll: int | bool = 8,  # Changed: now accepts False
        grad_checkpoint: bool = False,
        save_grad_history: bool = False,
        grad_state_names: Sequence[str] | None = None,
    ):
        super().__init__()
        self.step_mode = step_mode
        self.update_state_names = update_state_names
        self.unroll = unroll
        self.grad_checkpoint = grad_checkpoint
        self.save_grad_history = save_grad_history
        self.grad_state_names = grad_state_names
        self._grad_history = {}

    def _detect_loop_args(self, *args):
        """Heuristic: use first arg's shape to detect loop args"""

        if len(args) == 1:
            return args[0].shape[0], (0,)
        shapes = [
            a.shape[0] if torch.is_tensor(a) and a.ndim > 0 else None for a in args
        ]
        T = shapes[0]
        assert T is not None
        loop_args = tuple(i for i, s in enumerate(shapes) if s == T)
        return T, loop_args

    def _slice_args(self, args, loop_args, t):
        def normalize_index(t):
            if t == Ellipsis or t == "...":
                return ...
            return t

        t = normalize_index(t)
        out = []
        for i, a in enumerate(args):
            if i in loop_args:
                out.append(a[t])
            else:
                out.append(a)
        return out

    def single_step_forward(
        *args, **kwargs
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

    def _init_grad_hist(self, state_names: Sequence[str], T: int):
        self._grad_history = {name: [None] * T for name in state_names}

    def _should_save_grad(self, state_name: str) -> bool:
        """Check if gradient should be saved for this state."""
        if not self.save_grad_history:
            return False
        if self.grad_state_names is None:
            return True
        return state_name in self.grad_state_names

    def _register_grad_hook(self, tensor: Tensor, state_name: str, timestep: int):
        """Register a hook to capture gradients during backward pass."""

        def grad_hook(grad):
            if grad is not None:
                self._grad_history[state_name][timestep] = grad.detach().clone()
            return grad

        if tensor.requires_grad:
            tensor.register_hook(grad_hook)

    def _multi_step_forward_unrolled(self, *args, loop_args=(0,), **kwargs):
        """Unrolled inner loop for checkpoint.

        IMPORTANT: This function now returns lists (not stacked tensors):
            - z_seq: list[Tensor] (length = block length)
            - states_seq: dict[str, list[Tensor]] where each list length = block length

        Stacking into a single tensor is done once by the caller (multi_step_forward).
        """
        T = args[loop_args[0]].shape[0]
        z_seq = []
        states_seq = {}

        for t in range(T):
            sliced = self._slice_args(args, loop_args, t)
            z, states = self.single_step_forward(*sliced, **kwargs)
            z_seq.append(z)
            for k, v in states.items():
                states_seq.setdefault(k, []).append(v)

        # Note: don't stack here. Return lists so caller stacks once at the end.
        return z_seq, states_seq

    # --- helper: checkpointed execution of a block ---
    def _checkpointed_block_fn(self, *block_args, loop_args=(0,), **kwargs):
        memories = named_hidden_states(self)
        env = environ.all()

        def _pure(env, memories, *inner_args):
            set_hidden_states(self, memories)
            with environ.context(**env):
                return self._multi_step_forward_unrolled(
                    *inner_args, loop_args=loop_args, **kwargs
                )

        return checkpoint(_pure, env, memories, *block_args, use_reentrant=False)

    @partial(torch.compiler.disable, recursive=False)
    def multi_step_forward(self, *args, loop_args=None, **kwargs):
        """Unified implementation for unroll=False and unroll=int."""

        # Reset gradient history
        if self.save_grad_history:
            self._grad_history = {}

        # Detect loop args and time length T
        if loop_args is None:
            T, loop_args = self._detect_loop_args(*args)
        else:
            T = args[loop_args[0]].shape[0]

        if self.grad_state_names:
            self._init_grad_hist(self.grad_state_names, T)

        # Unified block configuration
        if self.unroll is False:
            block_size = T  # no unrolling → one large block
            use_checkpoint = False  # disable checkpointing
        else:
            block_size = int(self.unroll)
            use_checkpoint = bool(self.grad_checkpoint)

        # Compute number of blocks
        num_blocks = (T + block_size - 1) // block_size  # ceil(T / block_size)

        # Accumulators (always lists; stack once at the end)
        all_z_list = []
        all_states_lists = {}

        # ---- unified per-block loop ----
        for block_id in range(num_blocks):
            start = block_id * block_size
            end = min(start + block_size, T)

            # Slice loop args for the block
            block_indices = slice(start, end)
            block_args = self._slice_args(args, loop_args, block_indices)

            # Optionally checkpoint block function
            if use_checkpoint:
                z_list_block, states_block = self._checkpointed_block_fn(
                    *block_args, loop_args=loop_args, **kwargs
                )
            else:
                z_list_block, states_block = self._multi_step_forward_unrolled(
                    *block_args, loop_args=loop_args, **kwargs
                )

            # Accumulate z
            all_z_list.extend(z_list_block)

            # Accumulate state
            for k, lst in states_block.items():
                all_states_lists.setdefault(k, []).extend(lst)

        # ---- register gradient hooks on final per-timestep tensors ----
        if self.save_grad_history:
            for state_name, tensors in all_states_lists.items():
                if not self._should_save_grad(state_name):
                    continue
                for t, tensor in enumerate(tensors):
                    self._register_grad_hook(tensor, state_name, t)

        # ---- stack once and return ----
        return (
            torch.stack(all_z_list, dim=0),
            {k: torch.stack(v, dim=0) for k, v in all_states_lists.items()},
        )

    def get_grad_history(
        self, state_name: str | None = None
    ) -> dict[str, list[Tensor]] | list[Tensor]:
        """Retrieve saved gradient history.

        Parameters
        ----------
        state_name : str | None
            If provided, returns gradients for that specific state.
            If None, returns all gradient histories as a dict.

        Returns
        -------
        dict[str, list[Tensor]] | list[Tensor]
            Gradient history for requested state(s).
        """
        if state_name is not None:
            return self._grad_history.get(state_name, [])
        return self._grad_history

    def clear_grad_history(self):
        """Clear all saved gradient history."""
        self._grad_history = {}


# ----------------------------------------------------------------------
# make_rnn factory + decorator
# ----------------------------------------------------------------------


@overload
def make_rnn(
    obj: type[base.MemoryModule], allow_buffer=False, **rnn_kwargs
) -> type[RecurrentNNAbstract]: ...
@overload
def make_rnn(
    obj: base.MemoryModule, allow_buffer=False, **rnn_kwargs
) -> RecurrentNNAbstract: ...
@overload
def make_rnn(
    obj: None = None, allow_buffer=False, **rnn_kwargs
) -> Callable[[type[base.MemoryModule]], type[RecurrentNNAbstract]]: ...
def make_rnn(
    obj=None,
    allow_buffer=False,
    **rnn_kwargs,
):
    """RNN wrapper."""

    def _build_rnn_class(
        neuron_cls: type[base.MemoryModule] | base.MemoryModule,
    ) -> type[RecurrentNNAbstract]:
        class RNNWrapped(RecurrentNNAbstract):
            def __init__(self, *args, **kwargs):
                super().__init__(**rnn_kwargs)
                if isinstance(neuron_cls, type):
                    self.rnn_cell = neuron_cls(*args, **kwargs)
                else:
                    self.rnn_cell = neuron_cls

            def single_step_forward(self, *args, **kwargs):
                out = self.rnn_cell(*args, **kwargs)
                states = filter_hidden_states(
                    self.rnn_cell, self.update_state_names, allow_buffer=allow_buffer
                )
                return out, states

        RNNWrapped.__name__ = (
            f"RNN_{getattr(neuron_cls, '__name__', type(neuron_cls).__name__)}"
        )
        return RNNWrapped

    if isinstance(obj, type):
        return _build_rnn_class(obj)
    elif isinstance(obj, base.MemoryModule):
        Wrapped = _build_rnn_class(obj)
        return Wrapped()
    elif obj is None:

        def decorator(cls: type[base.MemoryModule]) -> type[RecurrentNNAbstract]:
            return _build_rnn_class(cls)

        return decorator


class RecurrentNN(RecurrentNNAbstract):
    def __init__(
        self,
        neuron: nn.Module,
        synapse: synapse.Synapse,
        syn_inp_module: nn.Module | None = None,
        neuron_inp_module: nn.Module | None = None,
        *,
        update_state_names: Sequence[str] | None = None,
        unroll: int = 8,
        grad_checkpoint: bool = False,
        allow_buffer=False,
        **kwargs,
    ):
        super().__init__(
            update_state_names=update_state_names,
            unroll=unroll,
            grad_checkpoint=grad_checkpoint,
            **kwargs,
        )
        self.neuron = neuron
        self.synapse = synapse

        # single step modules
        self.neuron_inp_module = neuron_inp_module
        self.syn_inp_module = syn_inp_module
        self.allow_buffer = allow_buffer

    def single_step_forward(self, x: Tensor, x_syn: Tensor | None = None):
        if self.neuron_inp_module is not None:
            x = self.neuron_inp_module(x)
        if self.syn_inp_module is not None:
            x_syn = self.syn_inp_module(x_syn)
        z = self.neuron(self.synapse.psc + x)
        _ = self.synapse(z if x_syn is None else z + x_syn)
        states = filter_hidden_states(
            self, self.update_state_names, allow_buffer=self.allow_buffer
        )

        return z, states
