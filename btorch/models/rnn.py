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
        unroll: int | bool = 8,
        chunk_size: int | None = None,
        cpu_offload: bool = False,
        grad_checkpoint: bool = False,
        save_grad_history: bool = False,
        grad_state_names: Sequence[str] | None = None,
    ):
        super().__init__()
        self.step_mode = step_mode
        self.update_state_names = update_state_names
        self.unroll = unroll
        self.chunk_size = chunk_size
        self.cpu_offload = cpu_offload
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

    def _slice_args(self, args, loop_args, t: int):
        out = []
        for i, a in enumerate(args):
            if i in loop_args:
                out.append(a[t])
            else:
                out.append(a)
        return out

    @partial(torch.compiler.disable, recursive=True)
    def _slice_args_range(self, args, loop_args, t):
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

    def _process_small_chunk(
        self, *args, loop_args=(0,), unroll_steps: int = 1, **kwargs
    ):
        """Inner loop for processing a small chunk.

        Returns:
            - z_seq: list[Tensor]
            - states_seq: dict[str, list[Tensor]]
        """
        # Determine actual number of steps for this small chunk (might be remainder)
        # However, args are already sliced to the correct size by caller.
        T = args[loop_args[0]].shape[0]
        z_seq = []
        states_seq = {}

        loop_positions = tuple(loop_args)
        static_args = list(args)
        for t in range(T):
            for i in loop_positions:
                static_args[i] = args[i][t]
            z, states = self.single_step_forward(*static_args, **kwargs)
            z_seq.append(z)
            for k, v in states.items():
                states_seq.setdefault(k, []).append(v)

        return z_seq, states_seq

    @partial(torch.compiler.disable, recursive=False)
    def _process_large_chunk_impl(
        self, *chunk_args, loop_args=(0,), unroll_size=1, **kwargs
    ):
        """Process a large chunk by splitting it into small unroll blocks.

        This function is NOT checkpointed itself, but is the body of the
        checkpoint.
        """
        T_chunk = chunk_args[loop_args[0]].shape[0]

        # Calculate number of small blocks
        num_blocks = (T_chunk + unroll_size - 1) // unroll_size

        chunk_z = []
        chunk_states = {}

        for i in range(num_blocks):
            start = i * unroll_size
            end = min(start + unroll_size, T_chunk)

            # Slice args for small chunk
            sub_indices = slice(start, end)
            sub_args = self._slice_args_range(chunk_args, loop_args, sub_indices)

            # Process small chunk
            z_sub, states_sub = self._process_small_chunk(
                *sub_args, loop_args=loop_args, unroll_steps=end - start, **kwargs
            )

            chunk_z.extend(z_sub)
            for k, v in states_sub.items():
                chunk_states.setdefault(k, []).extend(v)

        return chunk_z, chunk_states

    def _checkpointed_large_chunk(
        self,
        *chunk_args,
        loop_args=(0,),
        unroll_size=1,
        **kwargs,
    ):
        memories = named_hidden_states(self)
        env = environ.all()

        def _pure(env, memories, *inner_args):
            set_hidden_states(self, memories)
            with environ.context(**env):
                return self._process_large_chunk_impl(
                    *inner_args,
                    loop_args=loop_args,
                    unroll_size=unroll_size,
                    **kwargs,
                )

        return checkpoint(_pure, env, memories, *chunk_args, use_reentrant=False)

    @partial(torch.compiler.disable, recursive=False)
    def multi_step_forward(self, *args, loop_args=None, **kwargs):
        """Unified implementation for chunked unrolling and CPU offloading."""
        # Reset gradient history
        if self.save_grad_history:
            self._grad_history = {}

        # Detect loop args and time length T
        if loop_args is None:
            T, loop_args = self._detect_loop_args(*args)
        else:
            T = args[loop_args[0]].shape[0]

        self._current_T = T

        if self.grad_state_names:
            self._init_grad_hist(self.grad_state_names, T)

        # ------------------------------------------------------------------
        # Determine Chunk Sizes
        # ------------------------------------------------------------------
        # Unroll size (small chunk)
        if self.unroll is False:
            unroll_size = T  # No inner unrolling
        else:
            unroll_size = int(self.unroll)

        # Large chunk size
        # Follow legacy behavior: if chunk_size is None, check if we need
        # block-checkpoints.
        if self.chunk_size is None:
            # If default (None), we treat unroll_size as the chunk unit if
            # checkpointing is ON, to match legacy behavior where unroll was the
            # only block size.
            # If checkpointing is OFF, large_chunk_size = T is fine (except for
            # offloading which needs chunks).
            if self.grad_checkpoint:
                large_chunk_size = unroll_size
            else:
                large_chunk_size = T
        else:
            large_chunk_size = self.chunk_size
            if self.unroll is not False:
                if large_chunk_size % unroll_size != 0:
                    raise ValueError(
                        f"chunk_size ({large_chunk_size}) must be a multiple of "
                        f"unroll ({unroll_size})"
                    )

        # Number of large chunks
        num_large_chunks = (T + large_chunk_size - 1) // large_chunk_size

        use_checkpoint = bool(self.grad_checkpoint)

        # Accumulators
        all_z_list = []
        all_states_lists = {}

        # ------------------------------------------------------------------
        # Outer Loop: Large Chunks (Checkpointing & CPU Offloading)
        # ------------------------------------------------------------------
        for chunk_id in range(num_large_chunks):
            start = chunk_id * large_chunk_size
            end = min(start + large_chunk_size, T)

            # Slice args for large chunk
            chunk_indices = slice(start, end)
            chunk_args = self._slice_args_range(args, loop_args, chunk_indices)

            # Process Large Chunk
            if use_checkpoint:
                z_chunk, states_chunk = self._checkpointed_large_chunk(
                    *chunk_args,
                    loop_args=loop_args,
                    unroll_size=unroll_size,
                    **kwargs,
                )
            else:
                z_chunk, states_chunk = self._process_large_chunk_impl(
                    *chunk_args,
                    loop_args=loop_args,
                    unroll_size=unroll_size,
                    **kwargs,
                )

            # Offload to CPU if requested
            if self.cpu_offload:
                z_chunk = [z.cpu() for z in z_chunk]
                states_chunk = {
                    k: [v.cpu() for v in lst] for k, lst in states_chunk.items()
                }

            # Accumulate
            all_z_list.extend(z_chunk)
            for k, lst in states_chunk.items():
                all_states_lists.setdefault(k, []).extend(lst)

        # ------------------------------------------------------------------
        # Post-process: Register gradient hooks and stack
        # ------------------------------------------------------------------
        # Register hooks BEFORE stacking, on the original tensors in the lists
        # This ensures hooks are on tensors that participate in the backward pass
        if self.save_grad_history:
            for state_name, tensors in all_states_lists.items():
                if not self._should_save_grad(state_name):
                    continue
                if state_name not in self._grad_history:
                    self._grad_history[state_name] = [None] * T
                for t, tensor in enumerate(tensors):
                    self._register_grad_hook(tensor, state_name, t)

        # Stack after registering hooks
        stacked_outputs = torch.stack(all_z_list, dim=0)
        stacked_states = {k: torch.stack(v, dim=0) for k, v in all_states_lists.items()}

        return (stacked_outputs, stacked_states)

    def get_grad_history(self) -> dict[str, list]:
        """Retrieve saved gradient history."""
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
) -> (
    type[RecurrentNNAbstract]
    | RecurrentNNAbstract
    | Callable[[type[base.MemoryModule]], type[RecurrentNNAbstract]]
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

    raise TypeError(
        "`make_rnn` expects a MemoryModule class, a MemoryModule instance, "
        "or `None` when used as a decorator."
    )


class RecurrentNN(RecurrentNNAbstract):
    def __init__(
        self,
        neuron: nn.Module,
        synapse: synapse.Synapse,
        syn_inp_module: nn.Module | None = None,
        neuron_inp_module: nn.Module | None = None,
        *,
        update_state_names: Sequence[str] | None = None,
        unroll: int | bool = 8,
        chunk_size: int | None = None,
        cpu_offload: bool = False,
        grad_checkpoint: bool = False,
        allow_buffer=False,
        **kwargs,
    ):
        super().__init__(
            update_state_names=update_state_names,
            unroll=unroll,
            chunk_size=chunk_size,
            cpu_offload=cpu_offload,
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
