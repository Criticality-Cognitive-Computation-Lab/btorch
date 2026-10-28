from abc import abstractmethod
from collections.abc import Callable, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial
from typing import Any, Literal, overload

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from . import base, environ, synapse
from .base import StepMode
from .cudagraph import CudaGraphRunner
from .functional import filter_hidden_states, named_hidden_states, set_hidden_states
from .gradients import BPTT


_sparse_scope_states: ContextVar[dict[int, tuple[int, tuple[int, torch.dtype]]]] = (
    ContextVar("btorch_sparse_run_scopes", default={})
)


def _restore_autograd_multithreading() -> None:
    torch.autograd.set_multithreading_enabled(True)


def _serialize_offload_backward(grad: Tensor) -> Tensor:
    """Serialize backward for a multi-chunk CPU-offloaded trajectory.

    The CPU and CUDA branches of this graph can be scheduled on
    different autograd device threads. Each compiled chunk then tries to
    enter the same AOTAutograd context, which is not re-entrant across
    threads. Disabling multithreading for this backward keeps the
    compiled cell graphs intact and makes those contexts run
    sequentially. The previous global setting is restored once the
    engine finishes.
    """
    if torch.autograd.is_multithreading_enabled():
        torch.autograd.set_multithreading_enabled(False)
        torch.autograd.Variable._execution_engine.queue_callback(
            _restore_autograd_multithreading
        )
    return grad


def _split_loop_args(
    args: Sequence,
    loop_args: Sequence[int],
    size: int,
    mode: Literal["split", "chunk"] = "split",
):
    """Yield time views using the selected Torch partition operation.

    Only loop inputs are split; every other argument passes through
    unchanged.
    """
    splitter = partial(torch.split, split_size_or_sections=size, dim=0)
    if mode == "chunk":
        steps = args[loop_args[0]].shape[0]
        splitter = partial(
            torch.chunk, chunks=max(1, (steps + size - 1) // size), dim=0
        )
    split = {i: splitter(args[i]) for i in loop_args}
    for block_id in range(len(split[loop_args[0]])):
        yield tuple(
            split[i][block_id] if i in loop_args else args[i] for i in range(len(args))
        )


def _append_chunk(z_list: list, states_lists: dict[str, list], z, states: dict) -> None:
    """Accumulate one chunk's outputs and states (lists or stacked tensors) in
    place."""
    z_list.append(z) if isinstance(z, Tensor) else z_list.extend(z)
    for k, v in states.items():
        lst = states_lists.setdefault(k, [])
        lst.append(v) if isinstance(v, Tensor) else lst.extend(v)


@partial(torch.compiler.disable, recursive=True)
def _chunk_to_cpu(z, states: dict):
    """Move one chunk's outputs and states to CPU (tensors or lists of them).

    Keep the host copies outside AOTAutograd. If they are captured, the
    backward graph contains both CPU and CUDA compiled subgraphs; multithreaded
    autograd can then enter the same compiled-autograd context concurrently and
    fail with "cannot enter context". This is a host-transfer helper, not a
    cell-computation boundary: the recursive=False unroll/chunk boundaries
    still compile the lower-level cell functions.
    """

    def to_cpu(x):
        if isinstance(x, Tensor):
            return x.cpu()
        # One transfer per chunk/record, preserving the existing list-of-steps
        # contract and its autograd edges without a copy launch per timestep.
        return list(torch.stack(x, dim=0).cpu().unbind(0))

    return to_cpu(z), {k: to_cpu(v) for k, v in states.items()}


def _tensor_storage_key(tensor: Tensor) -> tuple[str, int | None, int]:
    """Return a stable key for storage sharing, including views."""
    device = tensor.device
    return device.type, device.index, tensor.untyped_storage().data_ptr()


def _iter_tensors(value: Any):
    """Yield tensors nested in recurrent argument containers."""
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


@contextmanager
def _offload_saved_tensors(
    enabled: bool,
    persistent: Sequence[Tensor] = (),
    providers: Sequence[Callable[[], Sequence[Tensor]]] = (),
):
    """Move autograd intermediates to CPU while an offloaded run is built.

    ``_chunk_to_cpu`` moves the public chunk outputs, but it does not affect
    tensors saved by individual autograd nodes inside the chunk.  Keeping those
    tensors on the accelerator defeats the memory purpose of ``cpu_offload``
    when checkpointing is disabled.  Caller-owned inputs and module parameters
    remain resident; only newly allocated intermediates are packed with their
    original device and restored when backward consumes them. This avoids
    copying a large sparse topology or trainable weight for every chunk.

    The context is active only while autograd records a training forward.  In
    particular, inference and ``torch.no_grad`` retain the existing fast path.
    The hook is equivalent to ``save_on_cpu`` for private tensors while keeping
    persistent storage resident.
    """
    if not enabled or not torch.is_grad_enabled():
        yield
        return
    resident = {_tensor_storage_key(tensor) for tensor in persistent}

    def pack(tensor: Tensor):
        key = _tensor_storage_key(tensor)
        if key not in resident:
            # Sparse execution bindings are prepared after this outer context
            # opens and may be rebuilt for each chunk. Refresh those small
            # ownership protocols lazily so large topology/weight storage is
            # never copied into every chunk's private autograd saves.
            for provider in providers:
                resident.update(
                    _tensor_storage_key(value)
                    for value in provider()
                    if isinstance(value, Tensor)
                )
        try:
            version = tensor._version
        except RuntimeError:
            version = None
        if tensor.device.type == "cpu" or key in resident:
            # Hooks bypass SavedVariable's normal version check. Keep a detached
            # resident view and restore that check explicitly on unpack.
            return "resident", tensor.device, tensor.detach(), version
        return "cpu", tensor.device, tensor.detach().cpu(), None

    def unpack(value):
        kind, device, tensor, version = value
        if version is not None and tensor._version != version:
            raise RuntimeError("a saved recurrent tensor was modified in-place")
        return tensor if kind == "resident" else tensor.to(device)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def _sparse_run_arguments(loop_tensor: Tensor) -> tuple[int, torch.dtype]:
    """Infer the sample batch size and dtype for one recurrent trajectory."""
    if not torch.is_tensor(loop_tensor) or loop_tensor.ndim == 0:
        raise ValueError(
            "the first loop argument must be a tensor with a time dimension"
        )
    step = loop_tensor[0]
    batch_size = 1 if step.ndim == 0 else max(1, step.numel() // step.shape[-1])
    return batch_size, loop_tensor.dtype


@contextmanager
def _sparse_checkpoint_recompute_context(
    sparse_states: list[tuple[nn.Module, Any]],
    batch_size: int,
    dtype: torch.dtype,
):
    """Restore sparse runtime bindings while checkpointing recomputes a
    chunk."""
    scope_states = dict(_sparse_scope_states.get())
    previous = []
    for submodule, state in sparse_states:
        restore = submodule._install_checkpoint_binding
        previous.append((submodule, restore(state)))
        scope_states[id(submodule)] = (1, (batch_size, dtype))
    token = _sparse_scope_states.set(scope_states)
    try:
        yield
    finally:
        _sparse_scope_states.reset(token)
        for submodule, state in reversed(previous):
            submodule._restore_checkpoint_run_state(state)


def _sparse_checkpoint_contexts(
    sparse_states: list[tuple[nn.Module, Any]],
    batch_size: int,
    dtype: torch.dtype,
):
    """Create forward and recompute contexts accepted by TorchDynamo."""
    return nullcontext(), _sparse_checkpoint_recompute_context(
        sparse_states, batch_size, dtype
    )


@contextmanager
def _sparse_run_scope(
    module: nn.Module,
    loop_tensor: Tensor,
    *,
    stable_addresses: bool = False,
):
    """Notify sparse-runtime participants once for one nested recurrent run."""
    batch_size, dtype = _sparse_run_arguments(loop_tensor)

    def leave(submodule: nn.Module, end: Callable[[], None]) -> None:
        states = dict(_sparse_scope_states.get())
        depth, signature = states[id(submodule)]
        depth -= 1
        if depth:
            states[id(submodule)] = (depth, signature)
            _sparse_scope_states.set(states)
            return
        states.pop(id(submodule))
        _sparse_scope_states.set(states)
        end()

    with ExitStack() as stack:
        for submodule in module.modules():
            begin = getattr(submodule, "_begin_sparse_run", None)
            end = getattr(submodule, "_end_sparse_run", None)
            if begin is None and end is None:
                continue
            if not callable(begin) or not callable(end):
                raise TypeError(
                    f"{type(submodule).__name__} must define both "
                    "_begin_sparse_run() and _end_sparse_run()."
                )
            signature = (batch_size, dtype)
            states = dict(_sparse_scope_states.get())
            current = states.get(id(submodule))
            depth = 0 if current is None else current[0]
            if depth:
                if current[1] != signature:
                    raise RuntimeError(
                        "nested sparse runs must use the same batch size and dtype"
                    )
            else:
                try:
                    begin(
                        batch_size,
                        dtype,
                        stable_addresses=stable_addresses,
                    )
                except Exception:
                    end()
                    raise
            states[id(submodule)] = (depth + 1, signature)
            _sparse_scope_states.set(states)
            stack.callback(leave, submodule, end)
        yield


class RecurrentNNAbstract(base.MemoryModule):
    """Base class for the time-unrolled recurrent loop.

    ``single_step_forward`` returns one output tensor plus a dict of states per
    step; models with several output ports should pack them into one tensor.

    CUDA-graph acceleration comes in two flavours, one per use case:

    * Inference: ``cudagraph=True`` captures the loop's device work and replays
      it (see ``cudagraph.py``). It is the fastest option and composes with
      ``torch.compile`` and ``cpu_offload``, but is inference-only. Warm-up,
      capture and replay use pieces of at most ``chunk_size`` steps (default
      200). ``chunk_mode="split"`` uses torch.split; ``"chunk"`` uses
      torch.chunk with ceil(T / chunk_size) sections. A shorter final piece
      runs eagerly rather than creating a graph used only once per sequence.
      ``None`` explicitly opts out of outer chunking.
    * Training: wrap the module in ``torch.compile(model, mode="reduce-overhead")``
      instead. Its CUDAGraph Trees capture forward and backward separately, so it
      is correct through ``backward()`` and composes with ``grad_checkpoint`` and
      ``cpu_offload`` -- call ``torch.compiler.cudagraph_mark_step_begin()`` once
      per iteration. ``cudagraph=True`` refuses grad-recording calls and points
      here.

    A captured graph is tied to the structure it was recorded for. Submodules
    whose structure can change in place (a sparse connection that is rewired
    or loads another pattern) expose ``capture_version``; when it changes, the
    next ``cudagraph=True`` call captures again instead of replaying. A
    submodule whose ``capture_incompatibility()`` returns a reason makes
    ``cudagraph=True`` raise with it. See ``cudagraph.py`` for the protocol.
    """

    def __init__(
        self,
        update_state_names: Sequence[str] | None = None,
        step_mode: StepMode = "m",
        unroll: int | bool = 8,
        chunk_size: int | None = 200,
        chunk_mode: Literal["split", "chunk"] = "split",
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
        self.chunk_mode = chunk_mode
        self.cpu_offload = cpu_offload
        self.grad_checkpoint = grad_checkpoint
        self.save_grad_history = save_grad_history
        self.grad_state_names = grad_state_names
        self._grad_history = {}
        self.cudagraph = cudagraph
        self._cudagraph_runner = CudaGraphRunner(warmup=cudagraph_warmup)

    @property
    def capture_version(self) -> tuple:
        """Track scheduling and recording choices bound into a capture."""
        return (
            self.unroll,
            None if self.update_state_names is None else tuple(self.update_state_names),
        )

    def clear_cudagraph(self) -> None:
        """Release cached CUDA graphs without resetting model memories.

        Existing returned outputs remain independent of the replay pool.
        Use this before comparing execution policies to release old
        captures.
        """
        self._cudagraph_runner.reset()

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
        self,
        *args: Any,
        loop_args: Sequence[int] = (0,),
        timestep_offset: int = 0,
        **kwargs: Any,
    ) -> tuple[list[Tensor], dict[str, list[Tensor]]]:
        """Inner loop for processing a small chunk.

        Gradient-history hooks are attached here, before the state is cloned.
        The clone is only a record: nothing consumes it during backward, so a
        hook on the clone would never observe a cotangent. The live state does
        participate in the next step (and, at the end, in the output), which is
        where its adjoint is produced. Checkpointed runs register the same hook
        during recomputation, when the state tensors require grad.

        Returns:
            - z_seq: list[Tensor]
            - states_seq: dict[str, list[Tensor]]
        """
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
                if self._should_save_grad(k):
                    history = self._grad_history.setdefault(k, [])
                    needed = timestep_offset + t + 1
                    if len(history) < needed:
                        history.extend([None] * (needed - len(history)))
                    self._register_grad_hook(v, k, timestep_offset + t)
                # Records are snapshots of this logical timestep. Memories
                # such as a circular delay cursor can be updated in place on
                # the next step; retaining a reference would rewrite history.
                states_seq.setdefault(k, []).append(v.clone())

        return z_seq, states_seq

    @partial(torch.compiler.disable, recursive=False)
    def _run_chunk_steps(
        self,
        *chunk_args,
        loop_args=(0,),
        unroll_size=1,
        timestep_offset=0,
        **kwargs,
    ):
        """Process a large chunk by splitting it into small unroll blocks.

        This function is NOT checkpointed itself, but is the body of the
        checkpoint.
        """
        with _sparse_run_scope(self, chunk_args[loop_args[0]]):
            chunk_z = []
            chunk_states = {}

            block_offset = timestep_offset
            for sub_args in _split_loop_args(chunk_args, loop_args, unroll_size):
                z_sub, states_sub = self._run_unroll_block(
                    *sub_args,
                    loop_args=loop_args,
                    timestep_offset=block_offset,
                    **kwargs,
                )

                block_offset += sub_args[loop_args[0]].shape[0]

                chunk_z.extend(z_sub)
                for k, v in states_sub.items():
                    chunk_states.setdefault(k, []).extend(v)

            return chunk_z, chunk_states

    def _checkpointed_large_chunk(
        self,
        *chunk_args,
        loop_args=(0,),
        unroll_size=1,
        timestep_offset=0,
        **kwargs: Any,
    ) -> None:
        memories = named_hidden_states(self)
        env = environ.all()
        batch_size, dtype = _sparse_run_arguments(chunk_args[loop_args[0]])
        sparse_states = []
        for submodule in self.modules():
            snapshot = getattr(submodule, "_capture_checkpoint_binding", None)
            if callable(snapshot):
                sparse_states.append((submodule, snapshot()))

        def _pure(env, memories, *inner_args):
            set_hidden_states(self, memories)
            with environ.context(**env):
                history_kwargs = (
                    {"timestep_offset": timestep_offset}
                    if self.save_grad_history
                    else {}
                )
                return self._run_chunk_steps(
                    *inner_args,
                    loop_args=loop_args,
                    unroll_size=unroll_size,
                    **history_kwargs,
                    **kwargs,
                )

        return checkpoint(
            _pure,
            env,
            memories,
            *chunk_args,
            use_reentrant=False,
            context_fn=partial(
                _sparse_checkpoint_contexts,
                sparse_states,
                batch_size,
                dtype,
            ),
        )

    @partial(torch.compiler.disable, recursive=False)
    def multi_step_forward(
        self, *args: Any, loop_args: Sequence[int] | None = None, **kwargs: Any
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the multi-step loop, optionally as replayed CUDA graphs."""
        if loop_args is None:
            _, scoped_loop_args = self._detect_loop_args(*args)
        else:
            scoped_loop_args = loop_args
        if args[scoped_loop_args[0]].shape[0] == 0:
            raise ValueError("RecurrentNN requires at least one timestep")
        with _sparse_run_scope(
            self,
            args[scoped_loop_args[0]],
            stable_addresses=self.cudagraph,
        ):
            if self.cudagraph:
                return self._cudagraph_multi_step(*args, loop_args=loop_args, **kwargs)
            return self._multi_step_forward_impl(*args, loop_args=loop_args, **kwargs)

    def _chunk_plan(
        self, *args: Any, loop_args: Sequence[int] | None = None
    ) -> tuple[int, Sequence[int], int, int]:
        """Resolve T, the loop args, and the two block sizes the time loop
        uses."""
        if loop_args is None:
            T, loop_args = self._detect_loop_args(*args)
        else:
            T = args[loop_args[0]].shape[0]
        if any(args[i].shape[0] != T for i in loop_args):
            raise ValueError("Loop inputs must share the same time dimension")

        unroll_size = T if self.unroll is False else int(self.unroll)

        # Inner unroll blocks already support short tails; outer chunk lengths
        # need not be multiples of unroll, including the final short chunk.
        large_chunk_size = self.chunk_size
        if large_chunk_size is None:
            large_chunk_size = unroll_size if self.grad_checkpoint else T
        if large_chunk_size < 1:
            raise ValueError("chunk_size must be positive or None")
        if self.chunk_mode not in ("split", "chunk"):
            raise ValueError("chunk_mode must be 'split' or 'chunk'")

        return T, loop_args, unroll_size, large_chunk_size

    def _iter_large_chunks(
        self, args: Sequence, loop_args: Sequence[int], chunk_size: int
    ):
        """Yield time views, each at most chunk_size steps long."""
        yield from _split_loop_args(args, loop_args, chunk_size, self.chunk_mode)

    # NOTE: `disable(recursive=False)` here is load-bearing, not cosmetic. It keeps
    # the O(T) python loop eager so only the unroll block is traced+compiled, which
    # is what makes compile time constant in T. Without it dynamo inlines all T
    # steps into one graph (T=1000 -> ~335s to compile).
    @partial(torch.compiler.disable, recursive=False)
    def _multi_step_forward_impl(
        self, *args: Any, loop_args: Sequence[int] | None = None, **kwargs: Any
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Unified implementation for chunked unrolling and CPU offloading."""
        if self.save_grad_history:
            self._grad_history = {}

        T, loop_args, unroll_size, large_chunk_size = self._chunk_plan(
            *args, loop_args=loop_args
        )

        if self.grad_state_names:
            self._init_grad_hist(self.grad_state_names, T)

        use_checkpoint = bool(self.grad_checkpoint)

        all_z_list = []
        all_states_lists = {}
        timestep_offset = 0

        # ------------------------------------------------------------------
        # Outer Loop: Large Chunks (Checkpointing & CPU Offloading)
        # ------------------------------------------------------------------
        if self.cpu_offload and torch.is_grad_enabled():
            persistent_providers = []
            persistent_tensors = [
                tensor for value in (*args, kwargs) for tensor in _iter_tensors(value)
            ]
            persistent_tensors.extend(self.parameters())
            persistent_tensors.extend(self.buffers())
            for submodule in self.modules():
                provider = getattr(submodule, "_recurrent_persistent_tensors", None)
                if callable(provider):
                    persistent_providers.append(provider)
        else:
            persistent_tensors = ()
            persistent_providers = ()
        with _offload_saved_tensors(
            self.cpu_offload, persistent_tensors, persistent_providers
        ):
            for chunk_args in self._iter_large_chunks(
                args, loop_args, large_chunk_size
            ):
                process = (
                    self._checkpointed_large_chunk
                    if use_checkpoint
                    else self._run_chunk_steps
                )
                history_kwargs = (
                    {"timestep_offset": timestep_offset}
                    if self.save_grad_history
                    else {}
                )
                z_chunk, states_chunk = process(
                    *chunk_args,
                    loop_args=loop_args,
                    unroll_size=unroll_size,
                    **history_kwargs,
                    **kwargs,
                )

                timestep_offset += chunk_args[loop_args[0]].shape[0]

                if self.cpu_offload:
                    z_chunk, states_chunk = _chunk_to_cpu(z_chunk, states_chunk)
                _append_chunk(all_z_list, all_states_lists, z_chunk, states_chunk)

        # ------------------------------------------------------------------
        # Post-process: stack
        # ------------------------------------------------------------------
        # Native chunks already own time-major trajectories. Unbinding them
        # into per-step Tensor objects only to stack again adds O(T * records)
        # wrappers and garbage-collection work to host submission.
        if isinstance(z_chunk, Tensor):
            stacked_outputs = (
                all_z_list[0] if len(all_z_list) == 1 else torch.cat(all_z_list, dim=0)
            )
            stacked_states = {
                name: values[0] if len(values) == 1 else torch.cat(values, dim=0)
                for name, values in all_states_lists.items()
            }
        else:
            stacked_outputs = torch.stack(all_z_list, dim=0)
            stacked_states = {
                k: torch.stack(v, dim=0) for k, v in all_states_lists.items()
            }

        # Only compiled multi-chunk CPU offload needs serialization: its CPU
        # and CUDA compiled branches can enter the same AOTAutograd context on
        # different device threads. Eager offload keeps multithreaded autograd.
        if self.cpu_offload and large_chunk_size < T and torch._dynamo.is_compiling():
            for tensor in (stacked_outputs, *stacked_states.values()):
                if tensor.requires_grad:
                    tensor.register_hook(_serialize_offload_backward)

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
        The default ``chunk_size=200`` bounds warm-up and capture. With
        explicit ``chunk_size=None`` one chunk spans T. When chunked, each
        replay copies directly into one preallocated full-sequence output and
        state destination; CPU offload uses the same ownership model with CPU
        destinations. No graph-pool tensor escapes to the caller.

        Equal-length pieces share a capture. A shorter final piece runs eagerly
        through the same recording path instead of creating a one-use graph.
        """
        T, loop_args, unroll_size, large_chunk_size = self._chunk_plan(
            *args, loop_args=loop_args
        )
        # A single chunk can use the runner's owned clone directly.  For a
        # multi-chunk call, allocate the complete trajectory after the first
        # replay has revealed the output shapes, then copy each replay into its
        # final time slice.  This keeps graph-pool tensors from escaping and
        # avoids both per-chunk clones and a final concatenation.
        n_chunks = (T + large_chunk_size - 1) // large_chunk_size
        full_z = None
        full_states = None
        time_offset = 0
        replay_length = None
        from .kernels.operators import poisson_uniforms
        from .kernels.options import normalize_backend
        from .surrogate import PoissonRandomSpike, Truncated

        function = getattr(getattr(self, "neuron", None), "surrogate_function", None)
        while isinstance(function, Truncated):
            function = function.surrogate
        poisson_spiking = (
            isinstance(function, PoissonRandomSpike)
            and function.spiking
            and normalize_backend(self.backend).neuron != "torch"
        )
        for chunk_args in self._iter_large_chunks(args, loop_args, large_chunk_size):
            length = chunk_args[loop_args[0]].shape[0]
            if replay_length is None:
                replay_length = length
            uniform = (
                poisson_uniforms(chunk_args[loop_args[0]]) if poisson_spiking else None
            )
            capture_args = (*chunk_args, uniform) if poisson_spiking else chunk_args

            def fn(*inner_args, **step_kwargs):
                if poisson_spiking:
                    inner_args, inner_uniform = inner_args[:-1], inner_args[-1]
                else:
                    inner_uniform = None
                if poisson_spiking:
                    step_kwargs["uniform_seq"] = inner_uniform
                return self._stacked_chunk_forward(
                    *inner_args,
                    loop_args=loop_args,
                    unroll_size=unroll_size,
                    **step_kwargs,
                )

            # Multi-chunk calls copy out of the graph pool into one stable
            # destination.  A single chunk retains the runner's existing
            # clone/CPU-offload ownership behavior.
            if length < replay_length:
                z, states = fn(*capture_args, **kwargs)
            else:
                z, states = self._cudagraph_runner(
                    self,
                    fn,
                    capture_args,
                    kwargs,
                    clone_outputs=n_chunks == 1 and not self.cpu_offload,
                    incompatible=self._cudagraph_incompatibilities(),
                )

            if n_chunks == 1:
                if self.cpu_offload:
                    z, states = _chunk_to_cpu(z, states)
                return z, states

            chunk_length = z.shape[0]
            destination_device = torch.device("cpu") if self.cpu_offload else z.device
            if full_z is None:
                full_z = torch.empty(
                    (T, *z.shape[1:]),
                    dtype=z.dtype,
                    device=destination_device,
                )
                full_states = {
                    name: torch.empty(
                        (T, *value.shape[1:]),
                        dtype=value.dtype,
                        device=destination_device,
                    )
                    for name, value in states.items()
                }

            full_z[time_offset : time_offset + chunk_length].copy_(z)
            for name, value in states.items():
                full_states[name][time_offset : time_offset + chunk_length].copy_(value)
            time_offset += chunk_length

        assert full_z is not None and full_states is not None
        return full_z, full_states

    def _cudagraph_incompatibilities(self) -> dict[str, str]:
        """Enabled options the CUDA-graph path cannot honour, with the reason.

        This is the single home of the RNN-side compatibility rule; the
        runner only reports what it is handed. Submodules report their own
        limits through ``capture_incompatibility()``, which the runner
        collects itself (see ``cudagraph.py``).
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
        """Retrieve saved gradient history.

        Returns:
            A shallow copy: ``{state name: [grad per timestep]}``. Clearing or
            reassigning entries of the result does not affect the module; the
            gradient tensors in the lists are shared with the module (not
            cloned), and the lists are copied so appends do not leak back.
        """
        return {name: list(grads) for name, grads in self._grad_history.items()}

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
                out = base._single_step_call(self.rnn_cell, *args, **kwargs)
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
    """Run a stateful neuron population coupled through a synapse.

    Neuron and synapse modules retain their ordinary memory interface.
    Execution and recording options follow :class:`RecurrentNNAbstract`.

    Args:
        neuron: Population implementing the neuron dynamics.
        synapse: Recurrent connection and postsynaptic dynamics.
        syn_inp_module: Optional transformation of external synaptic input.
        neuron_inp_module: Optional transformation of external neuron input.
        chunk_size: Maximum timesteps per outer chunk; None disables outer
            chunking unless checkpointing selects unroll-sized pieces.
        chunk_mode: "split" uses fixed-length torch.split pieces; "chunk" uses
            torch.chunk with ceil(T / chunk_size) sections.
        cpu_offload: Store returned histories and private saved tensors on CPU;
            model parameters, buffers and caller inputs remain on their devices.
        grad_checkpoint: Recompute chunks during backward instead of retaining
            their private forward activations.
        backend: Kernel selection, independent of gradient mathematics.
        gradient: Recurrent gradient algorithm. ``BPTT(sparse=True)`` exploits
            compact surrogate support where dependency analysis permits it;
            it never silently truncates an unbounded surrogate.
        **kwargs: Recording, chunking, checkpointing and memory options.
    """

    def __init__(
        self,
        neuron: nn.Module,
        synapse: synapse.Synapse,
        syn_inp_module: nn.Module | None = None,
        neuron_inp_module: nn.Module | None = None,
        *,
        update_state_names: Sequence[str] | None = None,
        unroll: int | bool = 8,
        chunk_size: int | None = 200,
        chunk_mode: Literal["split", "chunk"] = "split",
        cpu_offload: bool = False,
        grad_checkpoint: bool = False,
        allow_buffer: bool = False,
        backend: base.Backend = "torch",
        gradient: BPTT = BPTT(),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            update_state_names=update_state_names,
            unroll=unroll,
            chunk_size=chunk_size,
            chunk_mode=chunk_mode,
            cpu_offload=cpu_offload,
            grad_checkpoint=grad_checkpoint,
            **kwargs,
        )
        self.neuron = neuron
        self.synapse = synapse

        self.neuron_inp_module = neuron_inp_module
        self.syn_inp_module = syn_inp_module
        self.allow_buffer = allow_buffer
        self.backend = backend
        self.gradient = gradient
        self.register_load_state_dict_post_hook(self._refresh_merge_validation)

    @property
    def gradient(self) -> BPTT:
        """Return the recurrent gradient method, independent of kernel
        selection."""
        return self._gradient

    @gradient.setter
    def gradient(self, value: BPTT) -> None:
        if not isinstance(value, BPTT):
            raise TypeError("gradient must be a supported recurrent gradient method")
        self._gradient = value

    def _refresh_merge_validation(self, module, incompatible_keys):
        from .kernels.options import normalize_backend

        if normalize_backend(self.backend).merge_synapse:
            from .kernels.recurrent import validate_merge_module

            validate_merge_module(self)

    def _apply(self, fn):
        result = super()._apply(fn)
        if hasattr(self, "_backend"):
            self._refresh_merge_validation(self, None)
        return result

    @property
    def backend(self) -> base.Backend:
        return self._backend

    @backend.setter
    def backend(self, value: base.Backend) -> None:
        from .kernels.options import normalize_backend

        options = normalize_backend(value)
        previous = getattr(self, "_backend", "torch")
        if normalize_backend(
            previous
        ).merge_synapse != options.merge_synapse and getattr(
            self.synapse, "_kernel_sequence_started", False
        ):
            raise RuntimeError("Reset synapse states before switching merge_synapse")
        if options.merge_synapse:
            from .kernels.recurrent import validate_merge_module

            validate_merge_module(self)
        self._backend = value

    @property
    def capture_version(self) -> tuple:
        """Track execution choices and tensors wired into captured kernels."""
        from .kernels.options import resolve_model_options
        from .kernels.registry import cell_backend_revision

        neuron_model = type(self.neuron).__dict__.get("kernel_model")
        options = resolve_model_options(
            self.backend, *(() if neuron_model is None else (neuron_model,))
        )
        execution = (
            super().capture_version,
            self.gradient,
            self.allow_buffer,
            options.neuron,
            options.connection,
            options.merge_synapse,
            options.enable_fp_fusion,
            options.precise_division,
            options.precise_exp,
            cell_backend_revision(options.neuron),
            options.connection_method,
        )

        parameters = tuple(
            (name, self._capture_tensor_token(value))
            for name, value in self.named_parameters()
        )
        buffers = []
        for module_name, module in self.named_modules():
            memories = getattr(module, "_memories_rv", {})
            for name, value in module.named_buffers(recurse=False):
                if name in memories:
                    continue
                qualified_name = f"{module_name}.{name}" if module_name else name
                buffers.append((qualified_name, self._capture_tensor_token(value)))

        dt = environ.all().get("dt")
        if torch.is_tensor(dt):
            dt_token = ("tensor", self._capture_tensor_token(dt))
            if dt.device.type == "cpu":
                # A host scalar becomes a kernel argument at capture time rather
                # than a device-memory read. Its address can stay fixed while the
                # captured value becomes stale, so an in-place write must recapture.
                try:
                    revision = ("version", dt._version)
                except RuntimeError:
                    # Inference tensors have no version counter. CPU values are
                    # cheap to inspect and scalar dt is the supported capture use.
                    revision = ("value", dt.item())
                dt_token = (*dt_token, revision)
        else:
            dt_token = ("value", dt)
        return (*execution, tuple(parameters), tuple(buffers), dt_token)

    @staticmethod
    def _capture_tensor_token(value: Tensor) -> tuple:
        """Describe a non-state tensor address wired into a CUDA graph."""
        stride = tuple(value.stride()) if value.layout == torch.strided else ()
        return (
            id(value),
            value.data_ptr(),
            value.device,
            value.dtype,
            tuple(value.shape),
            stride,
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if getattr(self.synapse, "_kernel_merge_active", False):
            destination[prefix + "_kernel_merge_mode"] = torch.tensor(1)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        from .kernels.options import normalize_backend

        key = prefix + "_kernel_merge_mode"
        if key in state_dict:
            if not normalize_backend(self.backend).merge_synapse:
                error_msgs.append(
                    "Merged recurrent checkpoint requires merge_synapse=True"
                )
            else:
                cell = self.synapse.base_psc
                for name in cell._memories_rv:
                    state_key = prefix + "synapse.base_psc." + name
                    if state_key in state_dict:
                        cell._buffers[name] = torch.empty_like(state_dict[state_key])
                self.synapse._kernel_merge_active = True
                # Restoring a merged trajectory also restores its mode lock:
                # independent receptor states cannot be recovered without reset.
                self.synapse._kernel_sequence_started = True
            state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _reject_native_conductance_recurrence(self) -> None:
        """Reject COBA until recurrence supplies the observed voltage."""
        from .kernels.options import normalize_backend

        recurrent_synapses = (self.synapse, getattr(self, "synapse_apical", None))
        for recurrent_synapse in recurrent_synapses:
            if type(recurrent_synapse) is not synapse.ConductancePSC:
                continue
            network_native = normalize_backend(self.backend).neuron != "torch"
            synapse_native = (
                normalize_backend(recurrent_synapse.backend).neuron != "torch"
            )
            if network_native or synapse_native:
                raise NotImplementedError(
                    "Native RecurrentNN does not yet support ConductancePSC: "
                    "voltage-aware recurrent scheduling is required"
                )

    def multi_step_forward(self, *args, loop_args=None, **kwargs):
        self._reject_native_conductance_recurrence()
        if self.backend == "torch":
            from .kernels.adapters import connection_scope
            from .kernels.options import normalize_backend

            with connection_scope(normalize_backend(self.backend)):
                return super().multi_step_forward(*args, loop_args=loop_args, **kwargs)
        if self.cudagraph:
            from .kernels.options import normalize_backend

            if normalize_backend(self.backend).merge_synapse and not getattr(
                self.synapse, "_kernel_merge_active", False
            ):
                from .kernels.recurrent import prepare_merged_state

                prepare_merged_state(self)
        if self.cudagraph or self.save_grad_history or self.grad_state_names:
            from .kernels.adapters import connection_scope
            from .kernels.options import normalize_backend

            with connection_scope(normalize_backend(self.backend)):
                return super().multi_step_forward(*args, loop_args=loop_args, **kwargs)
        from .kernels.recurrent import recurrent_sequence

        return recurrent_sequence(self, *args, loop_args=loop_args, **kwargs)

    @partial(torch.compiler.disable, recursive=False)
    def _stacked_chunk_forward(
        self, *chunk_args, loop_args=(0,), unroll_size=1, **kwargs
    ):
        """Capture native trajectories without unbinding and restacking
        records."""
        if self.backend == "torch":
            return super()._stacked_chunk_forward(
                *chunk_args, loop_args=loop_args, unroll_size=unroll_size, **kwargs
            )
        self._reject_native_conductance_recurrence()
        from .kernels.recurrent import recurrent_sequence

        return recurrent_sequence(
            self, *chunk_args, loop_args=loop_args, _chunk=True, **kwargs
        )

    def _run_chunk_steps(self, *chunk_args, loop_args=(0,), unroll_size=1, **kwargs):
        self._reject_native_conductance_recurrence()
        if self.backend == "torch":
            return super()._run_chunk_steps(
                *chunk_args, loop_args=loop_args, unroll_size=unroll_size, **kwargs
            )
        from .kernels.recurrent import recurrent_sequence

        return recurrent_sequence(
            self, *chunk_args, loop_args=loop_args, _chunk=True, **kwargs
        )

    def single_step_forward(
        self, x: Tensor, x_syn: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self._reject_native_conductance_recurrence()
        if self.neuron_inp_module is not None:
            x = self.neuron_inp_module(x)
        if self.syn_inp_module is not None:
            x_syn = self.syn_inp_module(x_syn)
        z = base._single_step_call(self.neuron, self.synapse.psc + x)
        from .kernels.adapters import connection_scope
        from .kernels.options import normalize_backend

        with connection_scope(normalize_backend(self.backend)):
            _ = base._single_step_call(self.synapse, z if x_syn is None else z + x_syn)
        states = filter_hidden_states(
            self, self.update_state_names, allow_buffer=self.allow_buffer
        )

        return z, states


class ConductanceRecurrentNN(RecurrentNN):
    r"""Run a recurrent conductance network with previous-voltage coupling.

    The recurrent current at step :math:`t` uses the conductance and post-reset
    membrane voltage left by step :math:`t-1`:

    .. math::
        I_t &= g_{t-1}\,(E - v_{t-1}) \\
        z_t &= \operatorname{neuron}(x_t + I_t) \\
        g_t &= \operatorname{ConductancePSC}(z_t + x_{\mathrm{syn},t})

    The neuron advances, fires, and resets before its emitted spikes update the
    conductance. A recurrent spike emitted at step :math:`t` can therefore
    affect membrane current beginning at step :math:`t+1`. This initial public
    schedule is deliberately Torch-only and requires the exact
    :class:`~btorch.models.synapse.ConductancePSC` class.

    Constructor arguments are inherited from :class:`RecurrentNN`.

    Raises:
        TypeError: If ``synapse`` is not an exact ``ConductancePSC`` or the
            neuron has no membrane state ``v``.
        NotImplementedError: If any backend-bearing module in the network is
            not plain ``"torch"``.

    Examples:
        Run a voltage-aware recurrent sequence while recording its physical
        states:

        >>> network = ConductanceRecurrentNN(
        ...     neuron,
        ...     conductance_psc,
        ...     backend="torch",
        ...     update_state_names=("neuron.v", "synapse.psc"),
        ... )
        >>> spikes, states = network(input_sequence)
    """

    def _reject_native_conductance_recurrence(self) -> None:
        """Validate the explicit Torch COBA schedule before state advances."""
        if type(self.synapse) is not synapse.ConductancePSC:
            raise TypeError(
                "ConductanceRecurrentNN requires an exact ConductancePSC synapse"
            )
        if not hasattr(self.neuron, "v"):
            raise TypeError(
                "ConductanceRecurrentNN requires a neuron with membrane state 'v'"
            )
        unsupported = []
        for name, module in self.named_modules():
            if not hasattr(module, "backend"):
                continue
            if module.backend != "torch":
                unsupported.append(name or "network")
        if unsupported:
            raise NotImplementedError(
                "ConductanceRecurrentNN currently requires plain backend='torch' "
                "for every backend-bearing module; unsupported: "
                + ", ".join(unsupported)
            )

    def single_step_forward(
        self, x: Tensor, x_syn: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Advance one previous-post-reset-voltage conductance step."""
        self._reject_native_conductance_recurrence()
        if self.neuron_inp_module is not None:
            x = self.neuron_inp_module(x)
        if self.syn_inp_module is not None:
            x_syn = self.syn_inp_module(x_syn)

        voltage_observed = self.neuron.v
        current = self.synapse.current_charge(voltage_observed)
        z = base._single_step_call(self.neuron, x + current)
        event = z if x_syn is None else z + x_syn
        _ = base._single_step_call(self.synapse, event, voltage_observed)
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
        chunk_size: Maximum outer chunk length (default 200).
            Applies to checkpointing, offloading and CUDA Graph warm-up/replay.
            None uses the full sequence, or an unroll block with checkpointing.
        chunk_mode: "split" uses torch.split with chunk_size timesteps per piece;
            "chunk" uses torch.chunk with ceil(T / chunk_size) sections.
        cpu_offload: Move output traces and private backward residuals to CPU.
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
        chunk_size: int | None = 200,
        chunk_mode: Literal["split", "chunk"] = "split",
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
            chunk_mode=chunk_mode,
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
        self._reject_native_conductance_recurrence()
        if self.neuron_inp_module is not None:
            x = self.neuron_inp_module(x)
        if self.syn_inp_module is not None:
            x_syn = self.syn_inp_module(x_syn)

        total_input = self.synapse.psc + x

        # Apical input = recurrent apical current + external teacher signal
        apical_input = x_apical
        if self.synapse_apical is not None:
            if apical_input is None:
                apical_input = self.synapse_apical.psc
            else:
                apical_input = self.synapse_apical.psc + apical_input

        if apical_input is None:
            z = base._single_step_call(self.neuron, total_input)
        else:
            z = base._single_step_call(self.neuron, total_input, apical_input)

        _ = base._single_step_call(self.synapse, z if x_syn is None else z + x_syn)
        if self.synapse_apical is not None:
            _ = base._single_step_call(
                self.synapse_apical, z if x_syn is None else z + x_syn
            )

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
