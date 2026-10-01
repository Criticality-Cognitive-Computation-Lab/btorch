from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, overload

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from . import base, environ, synapse
from .cudagraph import CudaGraphRunner
from .functional import filter_hidden_states, named_hidden_states, set_hidden_states


def _split_loop_args(args: Sequence, loop_args: Sequence[int], size: int):
    """Yield positional-arg tuples, splitting only ``loop_args`` along time.

    ``torch.split`` returns zero-copy views; every other arg passes through
    unchanged at its position.
    """
    split = {i: torch.split(args[i], size, dim=0) for i in loop_args}
    for block_id in range(len(split[loop_args[0]])):
        yield tuple(
            split[i][block_id] if i in loop_args else args[i] for i in range(len(args))
        )


def _cat_chunks(chunks: list[Tensor]) -> Tensor:
    """Join per-chunk results along time, without a copy in the single-chunk
    case."""
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


def _append_chunk(z_list: list, states_lists: dict[str, list], z, states: dict) -> None:
    """Accumulate one chunk's outputs and states (lists or stacked tensors) in
    place."""
    z_list.append(z) if isinstance(z, Tensor) else z_list.extend(z)
    for k, v in states.items():
        lst = states_lists.setdefault(k, [])
        lst.append(v) if isinstance(v, Tensor) else lst.extend(v)


def _chunk_to_cpu(z, states: dict):
    """Move one chunk's outputs and states to CPU (tensors or lists of
    them)."""

    def to_cpu(x):
        return x.cpu() if isinstance(x, Tensor) else [t.cpu() for t in x]

    return to_cpu(z), {k: to_cpu(v) for k, v in states.items()}


# TODO: handle multiple output
class RecurrentNNAbstract(base.MemoryModule):
    """Base class for the time-unrolled recurrent loop.

    CUDA-graph acceleration comes in two flavours, one per use case:

    * Inference: ``cudagraph=True`` captures the loop's device work and replays
      it (see ``cudagraph.py``). It is the fastest option and composes with
      ``torch.compile`` and ``cpu_offload``, but is inference-only.
    * Training: wrap the module in ``torch.compile(model, mode="reduce-overhead")``
      instead. Its CUDAGraph Trees capture forward and backward separately, so it
      is correct through ``backward()`` and composes with ``grad_checkpoint`` and
      ``cpu_offload`` -- call ``torch.compiler.cudagraph_mark_step_begin()`` once
      per iteration. ``cudagraph=True`` refuses grad-recording calls and points
      here.
    """

    def __init__(
        self,
        update_state_names: Sequence[str] | None = None,
        step_mode: str = "m",
        unroll: int | bool = 8,
        chunk_size: int | None = None,
        cpu_offload: bool = False,
        grad_checkpoint: bool = False,
        save_grad_history: bool = False,
        grad_state_names: Sequence[str] | None = None,
        cudagraph: bool = False,
        cudagraph_warmup: int = 3,
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
        self.cudagraph = cudagraph
        self._cudagraph_runner = CudaGraphRunner(warmup=cudagraph_warmup)

    def _detect_loop_args(self, *args: Any) -> tuple[int, tuple[int, ...]]:
        """Infer ``(T, loop_args)`` from the first positional argument.

        Every positional tensor whose leading dim equals that of the first
        argument is treated as a time-major input to be sliced per step.

        Raises:
            ValueError: If the first argument is not a tensor with a time dim.
        """

        if len(args) == 1:
            return args[0].shape[0], (0,)
        shapes = [
            a.shape[0] if torch.is_tensor(a) and a.ndim > 0 else None for a in args
        ]
        T = shapes[0]
        if T is None:
            raise ValueError(
                "cannot infer the time dimension: the first positional argument "
                "must be a tensor with at least one dimension (time first), or "
                "pass `loop_args` explicitly."
            )
        loop_args = tuple(i for i, s in enumerate(shapes) if s == T)
        return T, loop_args

    @abstractmethod
    def single_step_forward(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Advance one timestep; return ``(output, states)``.

        Narrows :meth:`base.MemoryModule.single_step_forward` (which returns
        ``Any``) to the ``(output, states)`` pair the time loop accumulates.
        """
        raise NotImplementedError

    def _init_grad_hist(self, state_names: Sequence[str], T: int) -> None:
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

    def _run_unroll_block(
        self, *args: Any, loop_args: Sequence[int] = (0,), **kwargs: Any
    ) -> tuple[list[Tensor], dict[str, list[Tensor]]]:
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
    def _run_chunk_steps(self, *chunk_args, loop_args=(0,), unroll_size=1, **kwargs):
        """Process a large chunk by splitting it into small unroll blocks.

        This function is NOT checkpointed itself, but is the body of the
        checkpoint.
        """
        chunk_z = []
        chunk_states = {}

        for sub_args in _split_loop_args(chunk_args, loop_args, unroll_size):
            # Process small chunk
            z_sub, states_sub = self._run_unroll_block(
                *sub_args,
                loop_args=loop_args,
                **kwargs,
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
        **kwargs: Any,
    ) -> None:
        memories = named_hidden_states(self)
        env = environ.all()

        def _pure(env, memories, *inner_args):
            set_hidden_states(self, memories)
            with environ.context(**env):
                return self._run_chunk_steps(
                    *inner_args,
                    loop_args=loop_args,
                    unroll_size=unroll_size,
                    **kwargs,
                )

        return checkpoint(_pure, env, memories, *chunk_args, use_reentrant=False)

    @partial(torch.compiler.disable, recursive=False)
    def multi_step_forward(
        self, *args: Any, loop_args: Sequence[int] | None = None, **kwargs: Any
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the multi-step loop, optionally as replayed CUDA graphs."""
        if self.cudagraph:
            return self._cudagraph_multi_step(*args, loop_args=loop_args, **kwargs)
        return self._multi_step_forward_impl(*args, loop_args=loop_args, **kwargs)

    def _chunk_plan(
        self, *args: Any, loop_args: Sequence[int] | None = None
    ) -> tuple[int, Sequence[int], int, int]:
        """Resolve T, the loop args, and the two block sizes the time loop
        uses."""
        # Detect loop args and time length T
        if loop_args is None:
            T, loop_args = self._detect_loop_args(*args)
        else:
            T = args[loop_args[0]].shape[0]

        # Unroll size (small chunk)
        unroll_size = T if self.unroll is False else int(self.unroll)

        # Large chunk size. If chunk_size is None: with grad_checkpoint on, each
        # unroll block is its own chunk (so it is the checkpoint unit); otherwise
        # the whole sequence is a single chunk.
        if self.chunk_size is None:
            large_chunk_size = unroll_size if self.grad_checkpoint else T
        else:
            large_chunk_size = self.chunk_size
            if self.unroll is not False and large_chunk_size % unroll_size != 0:
                raise ValueError(
                    f"chunk_size ({large_chunk_size}) must be a multiple of "
                    f"unroll ({unroll_size})"
                )

        return T, loop_args, unroll_size, large_chunk_size

    def _iter_large_chunks(
        self, args: Sequence, loop_args: Sequence[int], chunk_size: int
    ):
        """Yield each large chunk's positional args.

        Only the loop args are split along time (``torch.split`` returns zero-copy
        views); every other arg passes through unchanged at its position.
        """
        yield from _split_loop_args(args, loop_args, chunk_size)

    # NOTE: `disable(recursive=False)` here is load-bearing, not cosmetic. It keeps
    # the O(T) python loop eager so only the unroll block is traced+compiled, which
    # is what makes compile time constant in T. Without it dynamo inlines all T
    # steps into one graph (T=1000 -> ~335s to compile).
    @partial(torch.compiler.disable, recursive=False)
    def _multi_step_forward_impl(
        self, *args: Any, loop_args: Sequence[int] | None = None, **kwargs: Any
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Unified implementation for chunked unrolling and CPU offloading."""
        # Reset gradient history
        if self.save_grad_history:
            self._grad_history = {}

        T, loop_args, unroll_size, large_chunk_size = self._chunk_plan(
            *args, loop_args=loop_args
        )

        if self.grad_state_names:
            self._init_grad_hist(self.grad_state_names, T)

        use_checkpoint = bool(self.grad_checkpoint)

        # Accumulators
        all_z_list = []
        all_states_lists = {}

        # ------------------------------------------------------------------
        # Outer Loop: Large Chunks (Checkpointing & CPU Offloading)
        # ------------------------------------------------------------------
        for chunk_args in self._iter_large_chunks(args, loop_args, large_chunk_size):
            # Process Large Chunk
            process = (
                self._checkpointed_large_chunk
                if use_checkpoint
                else self._run_chunk_steps
            )
            z_chunk, states_chunk = process(
                *chunk_args,
                loop_args=loop_args,
                unroll_size=unroll_size,
                **kwargs,
            )

            if self.cpu_offload:
                z_chunk, states_chunk = _chunk_to_cpu(z_chunk, states_chunk)
            _append_chunk(all_z_list, all_states_lists, z_chunk, states_chunk)

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

    @partial(torch.compiler.disable, recursive=False)
    def _stacked_chunk_forward(
        self, *chunk_args, loop_args=(0,), unroll_size=1, **kwargs
    ):
        """One large chunk's device work, stacked. This is the captured unit.

        Stacking inside the capture matters: the graph then returns one
        ``(chunk, ...)`` tensor per output instead of a list of per-step tensors,
        so copying results out of the replay pool costs one copy per chunk rather
        than one per timestep -- which would put O(T) eager work back into the
        loop and undo what capturing bought.
        """
        z_chunk, states_chunk = self._run_chunk_steps(
            *chunk_args, loop_args=loop_args, unroll_size=unroll_size, **kwargs
        )
        return (
            torch.stack(z_chunk, dim=0),
            {k: torch.stack(v, dim=0) for k, v in states_chunk.items()},
        )

    @partial(torch.compiler.disable, recursive=False)
    def _cudagraph_multi_step(
        self, *args: Any, loop_args: Sequence[int] | None = None, **kwargs: Any
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Chunked time loop whose per-chunk device work is a replayed CUDA
        graph.

        The capture boundary sits at the large chunk rather than the whole loop.
        With the default ``chunk_size=None`` the two coincide (one chunk spans T),
        so this is still a single replay per call. When the sequence *is* chunked,
        each chunk replays and every host-side step -- the ``cpu_offload`` moves,
        the accumulation, the final concat -- stays out here in eager python,
        where a replay cannot swallow it.

        Chunks of equal length share one capture; a short final remainder keys to
        its own.
        """
        T, loop_args, unroll_size, large_chunk_size = self._chunk_plan(
            *args, loop_args=loop_args
        )
        z_chunks = []
        state_chunks = {}
        for chunk_args in self._iter_large_chunks(args, loop_args, large_chunk_size):

            def fn(*inner_args):
                return self._stacked_chunk_forward(
                    *inner_args, loop_args=loop_args, unroll_size=unroll_size, **kwargs
                )

            # Offloading copies out of the pool itself, so skip the runner's clone.
            z, states = self._cudagraph_runner(
                self,
                fn,
                chunk_args,
                clone_outputs=not self.cpu_offload,
                incompatible=self._cudagraph_incompatibilities(),
            )

            if self.cpu_offload:
                z, states = _chunk_to_cpu(z, states)
            _append_chunk(z_chunks, state_chunks, z, states)

        return (
            _cat_chunks(z_chunks),
            {k: _cat_chunks(v) for k, v in state_chunks.items()},
        )

    def _cudagraph_incompatibilities(self) -> dict[str, str]:
        """Enabled options the CUDA-graph path cannot honour, with the reason.

        This is the single home of the RNN-side compatibility rule; the
        runner only reports what it is handed.
        """
        rules = {
            # Not merely unsupported: the capture path never routes through
            # _checkpointed_large_chunk, so this would be silently ignored -- no
            # recompute, no memory saved -- on top of being backward-only.
            "grad_checkpoint": "the capture path would silently ignore it",
            "save_grad_history": "its hooks only fire in the backward pass",
        }
        return {k: why for k, why in rules.items() if getattr(self, k, False)}

    def get_grad_history(self) -> dict[str, list]:
        """Retrieve saved gradient history."""
        return self._grad_history

    def clear_grad_history(self) -> None:
        """Clear all saved gradient history."""
        self._grad_history = {}


# ----------------------------------------------------------------------
# make_rnn factory + decorator
# ----------------------------------------------------------------------


@overload
def make_rnn(
    obj: type[base.MemoryModule], allow_buffer: bool = False, **rnn_kwargs
) -> type[RecurrentNNAbstract]: ...
@overload
def make_rnn(
    obj: base.MemoryModule, allow_buffer: bool = False, **rnn_kwargs
) -> RecurrentNNAbstract: ...
@overload
def make_rnn(
    obj: None = None, allow_buffer: bool = False, **rnn_kwargs
) -> Callable[[type[base.MemoryModule]], type[RecurrentNNAbstract]]: ...
def make_rnn(
    obj: type[base.MemoryModule] | base.MemoryModule | None = None,
    allow_buffer: bool = False,
    **rnn_kwargs: Any,
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
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(**rnn_kwargs)
                if isinstance(neuron_cls, type):
                    self.rnn_cell = neuron_cls(*args, **kwargs)
                else:
                    self.rnn_cell = neuron_cls

            def single_step_forward(
                self, *args: Any, **kwargs: Any
            ) -> tuple[Any, dict[str, Tensor]]:
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
        allow_buffer: bool = False,
        **kwargs: Any,
    ) -> None:
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

    def single_step_forward(
        self, x: Tensor, x_syn: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
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


class ApicalRecurrentNN(RecurrentNN):
    """Recurrent layer that supports an optional apical / top-down input.

    This subclass is useful when the neuron population contains models with
    multiple input ports (e.g.
    :class:`~btorch.models.neurons.TwoCompartmentGLIF`).  The extra
    ``x_apical`` tensor is forwarded to the neuron module unchanged.

    Optionally, a second ``synapse_apical`` can be supplied so that a subset
    of recurrent connections (e.g. SST→L5E or long-range E→L5E) drive the
    apical compartment while the remaining connections drive the somatic
    compartment.

    .. note::
        When calling :meth:`multi_step_forward` with a time-varying
        ``x_apical``, pass it as a **positional** argument
        (``brain(x, None, x_apical)``) so that the outer time-loop
        slices it correctly.  Keyword arguments are not unrolled by
        :class:`RecurrentNNAbstract`.

    Args:
        neuron: Neuron module (typically a
            :class:`~btorch.models.neurons.mixed.MixedNeuronPopulation`).
        synapse: Synapse model that provides recurrent currents to the soma.
        synapse_apical: Optional second synapse that provides recurrent
            currents to the apical compartment.
        syn_inp_module: Optional module applied to ``x_syn``.
        neuron_inp_module: Optional module applied to ``x``.
        update_state_names: Dotted state names to expose in the returned
            state dictionary.
        unroll: Inner unroll block size.
        chunk_size: Outer chunk size for gradient checkpointing / offloading.
        cpu_offload: Move chunk outputs to CPU during forward.
        grad_checkpoint: Use ``torch.utils.checkpoint`` on large chunks.
        allow_buffer: Allow collecting hidden states from non-MemoryModule
            buffers.
        **kwargs: Passed to :class:`RecurrentNNAbstract`.
    """

    def __init__(
        self,
        neuron: nn.Module,
        synapse: synapse.Synapse,
        synapse_apical: synapse.Synapse | None = None,
        syn_inp_module: nn.Module | None = None,
        neuron_inp_module: nn.Module | None = None,
        *,
        update_state_names: Sequence[str] | None = None,
        unroll: int | bool = 8,
        chunk_size: int | None = None,
        cpu_offload: bool = False,
        grad_checkpoint: bool = False,
        allow_buffer: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            neuron=neuron,
            synapse=synapse,
            syn_inp_module=syn_inp_module,
            neuron_inp_module=neuron_inp_module,
            update_state_names=update_state_names,
            unroll=unroll,
            chunk_size=chunk_size,
            cpu_offload=cpu_offload,
            grad_checkpoint=grad_checkpoint,
            allow_buffer=allow_buffer,
            **kwargs,
        )
        self.synapse_apical = synapse_apical

    def single_step_forward(
        self,
        x: Tensor,
        x_syn: Tensor | None = None,
        x_apical: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Advance one timestep with optional apical drive.

        Args:
            x: External input current of shape ``(*batch, n_neuron)``.
            x_syn: Optional direct synaptic input.
            x_apical: Optional apical / top-down input of the same shape as
                ``x``.  If ``synapse_apical`` is present, the apical synaptic
                current is *added* to this tensor before being passed to the
                neuron.

        Returns:
            ``(spikes, states)`` where ``spikes`` has shape
            ``(*batch, n_neuron)``.
        """
        if self.neuron_inp_module is not None:
            x = self.neuron_inp_module(x)
        if self.syn_inp_module is not None:
            x_syn = self.syn_inp_module(x_syn)

        # Somatic input
        total_input = self.synapse.psc + x

        # Apical input = recurrent apical current + external teacher signal
        apical_input = x_apical
        if self.synapse_apical is not None:
            if apical_input is None:
                apical_input = self.synapse_apical.psc
            else:
                apical_input = self.synapse_apical.psc + apical_input

        if apical_input is None:
            z = self.neuron(total_input)
        else:
            z = self.neuron(total_input, apical_input)

        _ = self.synapse(z if x_syn is None else z + x_syn)
        if self.synapse_apical is not None:
            _ = self.synapse_apical(z if x_syn is None else z + x_syn)

        states = filter_hidden_states(
            self, self.update_state_names, allow_buffer=self.allow_buffer
        )
        return z, states


class SomaApicalRecurrentNN(ApicalRecurrentNN):
    """Recurrent layer with dedicated somatic and apical synapses, both
    required.

    A specialisation of :class:`ApicalRecurrentNN` for architectures where
    recurrent connections are explicitly split into separate somatic and apical
    pathways:

    - **Somatic synapse** (``synapse_soma``): drives the somatic compartment.
    - **Apical synapse** (``synapse_apical``): drives the apical compartment
      (e.g. SST→L5E or long-range E→L5E connections).

    Both synapses receive the same spike output each step.  The external input
    ``x`` is added to the somatic current, and ``x_apical`` (if provided) is
    added to the apical current.

    Args:
        neuron: Neuron module (e.g.
            :class:`~btorch.models.neurons.mixed.MixedNeuronPopulation` or a
            single :class:`~btorch.models.neurons.TwoCompartmentGLIF`).
        synapse_soma: Synapse model for the somatic compartment.
        synapse_apical: Synapse model for the apical compartment.
        syn_inp_module: Optional module applied to ``x_syn``.
        neuron_inp_module: Optional module applied to ``x``.
        update_state_names: Dotted state names to expose.
        **kwargs: Passed to :class:`ApicalRecurrentNN`.
    """

    def __init__(
        self,
        neuron: nn.Module,
        synapse_soma: synapse.Synapse,
        synapse_apical: synapse.Synapse,
        syn_inp_module: nn.Module | None = None,
        neuron_inp_module: nn.Module | None = None,
        *,
        update_state_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            neuron=neuron,
            synapse=synapse_soma,
            synapse_apical=synapse_apical,
            syn_inp_module=syn_inp_module,
            neuron_inp_module=neuron_inp_module,
            update_state_names=update_state_names,
            **kwargs,
        )
        self.synapse_soma = synapse_soma
