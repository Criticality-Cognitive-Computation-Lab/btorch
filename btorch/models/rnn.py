from collections.abc import Callable, Mapping
from functools import partial
from typing import overload

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..monitor import (
    EagerFrame,
    Expr,
    Recorder,
    RecordSpec,
    Resolver,
    TargetRef,
)
from . import base, environ, synapse
from .cudagraph import CudaGraphRunner
from .functional import filter_hidden_states, named_hidden_states, set_hidden_states


def _cat_chunks(chunks: list[Tensor]) -> Tensor:
    """Join per-chunk results along time, without a copy in the single-chunk
    case."""
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


def _split_state_specs(
    spec: RecordSpec | None,
) -> tuple[list[str] | None, list]:
    """Split ``update_state_names`` into legacy names + recorder specs.

    Plain dotted strings (and ``None`` = all memories) flow through the classic
    ``stacked_states`` path unchanged; :class:`~btorch.monitor.Expr` /
    ``grad(...)`` / ``{name: expr}`` specs route to the recording engine and are
    read via :meth:`get_records`.

    Returns ``(legacy_names, record_specs)`` where ``legacy_names`` is
    ``None`` (all memories), else a list of dotted strings; and ``record_specs``
    is a list of engine specs.
    """
    if spec is None:
        return None, []
    if isinstance(spec, (str, Expr)):
        spec = [spec]
    if isinstance(spec, Mapping):
        return [], [spec]
    legacy_names, record_specs = [], []
    for entry in spec:
        (legacy_names if isinstance(entry, str) else record_specs).append(entry)
    return legacy_names, record_specs


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

    ``update_state_names`` is the single recording spec: plain dotted strings
    populate ``stacked_states``; ``Expr`` / ``grad(...)`` / ``{name: expr}`` specs
    (see :mod:`btorch.monitor`) are read via :meth:`get_records`.  Value/reduction
    monitors compose with ``cudagraph=True`` (their fold is captured inside the
    chunk graph), single-chunk only; ``grad(...)`` monitors are refused there.
    """

    def __init__(
        self,
        update_state_names: RecordSpec | None = None,
        step_mode="m",
        unroll: int | bool = 8,
        chunk_size: int | None = None,
        cpu_offload: bool = False,
        grad_checkpoint: bool = False,
        cudagraph: bool = False,
        cudagraph_warmup: int = 3,
    ):
        super().__init__()
        # `update_state_names` is the single recording spec. Dotted strings (and
        # None = all memories) use the classic stacked_states path; Expr /
        # grad(...) / {name: expr} specs use the recording engine (get_records()).
        legacy_names, record_specs = _split_state_specs(update_state_names)
        self.step_mode = step_mode
        self.update_state_names = legacy_names
        self.unroll = unroll
        self.chunk_size = chunk_size
        self.cpu_offload = cpu_offload
        self.grad_checkpoint = grad_checkpoint
        self.cudagraph = cudagraph
        self._cudagraph_runner = CudaGraphRunner(warmup=cudagraph_warmup)

        # --- lazy recording engine (see btorch.monitor) ------------------
        self._record_specs = record_specs or None
        self._recorder = None  # built lazily at first multi_step_forward
        self._records: dict[str, Tensor | list] = {}
        self._records_active = False  # set once the recorder is built (skip if empty)
        self._record_has_grad = False
        self._record_grad_keys: list[str] = []
        self._record_grad_refs: list[TargetRef] = []
        self._state_resolver = None  # set when the recorder is built

    def _state_module(self) -> nn.Module:
        """The module the recorder resolves dotted state names against.

        A method rather than a stored attribute so we never register a module as
        its own submodule (which breaks named_modules / pickling).  Subclasses
        that hold the stateful cell elsewhere override this (see ``make_rnn``).
        """
        return self

    def _state_allow_buffer(self) -> bool:
        """Whether the recorder may resolve non-memory state on
        :meth:`_state_module`.

        Single source of truth for the recorder's resolution policy: it always
        mirrors the ``allow_buffer`` flag of the legacy ``stacked_states``
        collection (see :func:`btorch.models.functional.filter_hidden_states`),
        so the two recording paths cannot disagree.  Subclasses that take an
        ``allow_buffer`` argument override this alongside :meth:`_state_module`.
        """
        return False

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

    def single_step_forward(
        *args, **kwargs
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

    def _process_small_chunk(
        self, *args, loop_args=(0,), unroll_steps: int = 1, record_carry=None, **kwargs
    ):
        """Inner loop for processing a small chunk.

        Returns ``(z_seq, states_seq, record_state)`` where ``record_state`` is
        ``(carry, record_buffers, grad_snapshots)``:

        - ``carry``: the streaming reduction/window carry, threaded and returned
          (functional -- grad_checkpoint recompute discards its copy; the whole
          fold is tensor ops so cudagraph can capture it).
        - ``record_buffers``: per-step tensors of each materialise/raw stack-point
          node (``{node_id: [tensor, ...]}``).
        - ``grad_snapshots``: per-step source tensors for grad monitors.

        The monitor source read happens here -- inside the traced small chunk,
        right after ``single_step_forward`` -- so the just-rebound buffers hold
        this step's tensors (byte-identical, grad-connected, cudagraph-safe).
        """
        T = args[loop_args[0]].shape[0]
        z_seq = []
        states_seq = {}
        carry = record_carry
        record_buffers: dict[int, list] = {}
        grad_snapshots: list = []

        fold_records = self._records_active
        capture_grad = self._record_has_grad
        grad_refs = self._record_grad_refs
        loop_positions = tuple(loop_args)
        static_args = list(args)
        for t in range(T):
            for i in loop_positions:
                static_args[i] = args[i][t]
            z, states = self.single_step_forward(*static_args, **kwargs)
            z_seq.append(z)
            for k, v in states.items():
                states_seq.setdefault(k, []).append(v)
            if fold_records:
                snapshot = self._state_resolver.snapshot()
                carry, stack = self._recorder.step_kernel(carry, EagerFrame(snapshot))
                for node_id, value in stack.items():
                    record_buffers.setdefault(node_id, []).append(value)
                if capture_grad:
                    grad_snapshots.append([snapshot[ref] for ref in grad_refs])

        return z_seq, states_seq, (carry, record_buffers, grad_snapshots)

    @partial(torch.compiler.disable, recursive=False)
    def _process_large_chunk_impl(
        self, *chunk_args, loop_args=(0,), unroll_size=1, record_carry=None, **kwargs
    ):
        """Process a large chunk by splitting it into small unroll blocks.

        This function is NOT checkpointed itself, but is the body of the
        checkpoint.  It threads ``record_carry`` across blocks and returns the
        accumulated ``(carry, record_buffers, grad_snapshots)``.
        """
        # Split loop args into unroll-sized chunks using torch.split
        # torch.split returns views of the original tensor (zero-copy)
        split_tensors = {
            i: torch.split(chunk_args[i], unroll_size, dim=0) for i in loop_args
        }

        chunk_z = []
        chunk_states = {}
        carry = record_carry
        record_buffers: dict[int, list] = {}
        grad_snapshots: list = []

        # Iterate over the split chunks (all loop args have same number of chunks)
        num_blocks = len(split_tensors[loop_args[0]])
        for block_id in range(num_blocks):
            # Build sub_args preserving original arg positions
            # Split tensors get their chunk, scalars pass through unchanged
            sub_args = tuple(
                split_tensors[i][block_id] if i in loop_args else chunk_args[i]
                for i in range(len(chunk_args))
            )

            # Process small chunk
            z_sub, states_sub, (carry, buffers_sub, grads_sub) = (
                self._process_small_chunk(
                    *sub_args,
                    loop_args=loop_args,
                    unroll_steps=sub_args[loop_args[0]].shape[0],
                    record_carry=carry,
                    **kwargs,
                )
            )

            chunk_z.extend(z_sub)
            for k, v in states_sub.items():
                chunk_states.setdefault(k, []).extend(v)
            for node_id, values in buffers_sub.items():
                record_buffers.setdefault(node_id, []).extend(values)
            grad_snapshots.extend(grads_sub)

        return chunk_z, chunk_states, (carry, record_buffers, grad_snapshots)

    def _checkpointed_large_chunk(
        self,
        *chunk_args,
        loop_args=(0,),
        unroll_size=1,
        record_carry=None,
        **kwargs,
    ):
        memories = named_hidden_states(self)
        env = environ.all()

        def _pure(env, memories, record_carry, *inner_args):
            set_hidden_states(self, memories)
            with environ.context(**env):
                return self._process_large_chunk_impl(
                    *inner_args,
                    loop_args=loop_args,
                    unroll_size=unroll_size,
                    record_carry=record_carry,
                    **kwargs,
                )

        return checkpoint(
            _pure, env, memories, record_carry, *chunk_args, use_reentrant=False
        )

    @partial(torch.compiler.disable, recursive=False)
    def multi_step_forward(self, *args, loop_args=None, **kwargs):
        """Run the multi-step loop, optionally as replayed CUDA graphs."""
        if self.cudagraph:
            return self._cudagraph_multi_step(*args, loop_args=loop_args, **kwargs)
        return self._multi_step_forward_impl(*args, loop_args=loop_args, **kwargs)

    def _chunk_plan(self, *args, loop_args=None):
        """Resolve T, the loop args, and the two block sizes the time loop
        uses."""
        # Detect loop args and time length T
        if loop_args is None:
            T, loop_args = self._detect_loop_args(*args)
        else:
            T = args[loop_args[0]].shape[0]

        # Unroll size (small chunk)
        unroll_size = T if self.unroll is False else int(self.unroll)

        # Large chunk size. Follow legacy behavior: if chunk_size is None, treat
        # unroll_size as the chunk unit when checkpointing is ON (matching the
        # legacy behavior where unroll was the only block size), else the whole
        # sequence is one chunk.
        if self.chunk_size is None:
            large_chunk_size = unroll_size if self.grad_checkpoint else T
        else:
            large_chunk_size = self.chunk_size
            if self.unroll is not False and large_chunk_size % unroll_size != 0:
                raise ValueError(
                    f"chunk_size ({large_chunk_size}) must be a multiple of "
                    f"unroll ({unroll_size})"
                )

        num_large_chunks = (T + large_chunk_size - 1) // large_chunk_size
        return T, loop_args, unroll_size, large_chunk_size, num_large_chunks

    def _split_loop_args(self, args, loop_args, chunk_size):
        """Split only the loop args along time; torch.split returns zero-copy
        views."""
        return {i: torch.split(args[i], chunk_size, dim=0) for i in loop_args}

    # NOTE: `disable(recursive=False)` here is load-bearing, not cosmetic. It keeps
    # the O(T) python loop eager so only the unroll block is traced+compiled, which
    # is what makes compile time constant in T. Without it dynamo inlines all T
    # steps into one graph (T=1000 -> ~335s to compile).
    @partial(torch.compiler.disable, recursive=False)
    def _multi_step_forward_impl(self, *args, loop_args=None, **kwargs):
        """Unified implementation for chunked unrolling and CPU offloading."""
        self._ensure_recorder()

        T, loop_args, unroll_size, large_chunk_size, num_large_chunks = (
            self._chunk_plan(*args, loop_args=loop_args)
        )

        self._current_T = T

        use_checkpoint = bool(self.grad_checkpoint)

        # Accumulators
        all_z_list = []
        all_states_lists = {}
        # Streaming record carry, threaded across chunks (see _process_small_chunk).
        record_carry = self._init_record_carry() if self._records_active else None
        record_buffers: dict[int, list] = {}
        grad_snapshots: list = []

        # ------------------------------------------------------------------
        # Outer Loop: Large Chunks (Checkpointing & CPU Offloading)
        # ------------------------------------------------------------------
        split_tensors = self._split_loop_args(args, loop_args, large_chunk_size)

        for chunk_id in range(num_large_chunks):
            # Build chunk_args preserving original arg positions
            # Split tensors get their chunk, scalars pass through unchanged
            chunk_args = tuple(
                split_tensors[i][chunk_id] if i in loop_args else args[i]
                for i in range(len(args))
            )

            # Process Large Chunk
            chunk_fn = (
                self._checkpointed_large_chunk
                if use_checkpoint
                else self._process_large_chunk_impl
            )
            z_chunk, states_chunk, (record_carry, buffers_chunk, grads_chunk) = (
                chunk_fn(
                    *chunk_args,
                    loop_args=loop_args,
                    unroll_size=unroll_size,
                    record_carry=record_carry,
                    **kwargs,
                )
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
            for node_id, values in buffers_chunk.items():
                record_buffers.setdefault(node_id, []).extend(values)
            grad_snapshots.extend(grads_chunk)

        # Stack the legacy update_state_names channel.
        stacked_outputs = torch.stack(all_z_list, dim=0)
        stacked_states = {k: torch.stack(v, dim=0) for k, v in all_states_lists.items()}

        # Recording engine: finalise the streamed carry + materialise buffers.
        if self._records_active:
            self._finalize_records(record_carry, record_buffers, grad_snapshots, T)

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

        Records are folded and finalised *inside* the capture too (the fold is
        pure tensor ops on a tensor carry), so the record outputs are graph
        outputs that replay like the states.  This is only used for a single chunk
        spanning T (``_cudagraph_multi_step`` enforces that when records are on),
        so ``finalize`` sees the whole sequence.
        """
        carry = self._init_record_carry() if self._records_active else None
        z_chunk, states_chunk, (carry, record_buffers, _grads) = (
            self._process_large_chunk_impl(
                *chunk_args,
                loop_args=loop_args,
                unroll_size=unroll_size,
                record_carry=carry,
                **kwargs,
            )
        )
        stacked_z = torch.stack(z_chunk, dim=0)
        stacked_states = {k: torch.stack(v, dim=0) for k, v in states_chunk.items()}
        records = (
            self._recorder.finalize(carry, record_buffers)
            if self._records_active
            else {}
        )
        return stacked_z, stacked_states, records

    @partial(torch.compiler.disable, recursive=False)
    def _cudagraph_multi_step(self, *args, loop_args=None, **kwargs):
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

        value/reduction monitors are folded inside the captured chunk
        (see ``_stacked_chunk_forward``) and stored in ``self._records``; this
        requires a single chunk, and ``grad(...)`` monitors are refused (below).
        """
        T, loop_args, unroll_size, large_chunk_size, num_large_chunks = (
            self._chunk_plan(*args, loop_args=loop_args)
        )
        self._current_T = T

        self._ensure_recorder()
        if self._records_active:
            if self._record_has_grad:
                raise RuntimeError(
                    "cudagraph=True is incompatible with grad(...) monitors: their "
                    "hooks only fire in the backward pass, which replay never runs."
                )
            if num_large_chunks > 1:
                raise RuntimeError(
                    "cudagraph=True with recording requires a single chunk (the "
                    "default chunk_size=None spans T); chunked capture cannot "
                    "finalise streaming reductions across separate replays."
                )

        split_tensors = self._split_loop_args(args, loop_args, large_chunk_size)

        z_chunks = []
        state_chunks = {}
        for chunk_id in range(num_large_chunks):
            chunk_args = tuple(
                split_tensors[i][chunk_id] if i in loop_args else args[i]
                for i in range(len(args))
            )

            def fn(*inner_args):
                return self._stacked_chunk_forward(
                    *inner_args, loop_args=loop_args, unroll_size=unroll_size, **kwargs
                )

            # Offloading copies out of the pool itself, so skip the runner's clone.
            z, states, records = self._cudagraph_runner(
                self, fn, chunk_args, clone_outputs=not self.cpu_offload
            )

            if self.cpu_offload:
                z = z.cpu()
                states = {k: v.cpu() for k, v in states.items()}

            z_chunks.append(z)
            for k, v in states.items():
                state_chunks.setdefault(k, []).append(v)

        # Single chunk when records are on, so these records span all of T.
        if self._records_active:
            if self.cpu_offload:
                records = {
                    k: (v.cpu() if torch.is_tensor(v) else v)
                    for k, v in records.items()
                }
            self._records = records

        return (
            _cat_chunks(z_chunks),
            {k: _cat_chunks(v) for k, v in state_chunks.items()},
        )

    # ------------------------------------------------------------------
    # Recording engine (btorch.monitor): drives update_state_names' Expr /
    # grad(...) specs; plain dotted-string names use the stacked_states path.
    # ------------------------------------------------------------------
    def _ensure_recorder(self):
        if self._record_specs is None or self._recorder is not None:
            return
        resolver = Resolver(self._state_module())
        self._recorder = Recorder(self._record_specs, resolver=resolver)
        if not self._state_allow_buffer():
            # mirror filter_hidden_states' legacy policy so both recording
            # paths accept exactly the same targets
            self._reject_non_memory_targets(resolver)
        self._state_resolver = resolver
        self._records_active = not self._recorder.is_empty
        self._record_has_grad = self._recorder.has_grad_specs
        # refs / keys of grad monitors, in a stable order for per-step capture
        grad_specs = self._recorder.grad_specs
        self._record_grad_keys = [key for key, _ref, _name in grad_specs]
        self._record_grad_refs = [ref for _key, ref, _name in grad_specs]

    @staticmethod
    def _reject_non_memory_targets(resolver: Resolver) -> None:
        """Reject recorder targets outside a MemoryModule's memories.

        With ``allow_buffer=False`` the legacy ``stacked_states`` path collects
        only ``MemoryModule`` memories; silently accepting plain buffers here
        would let the two paths disagree on what gets recorded.
        """
        for module, attr in resolver.resolved():
            memories = getattr(module, "_memories_rv", None)
            if isinstance(memories, dict) and attr in memories:
                continue
            raise KeyError(
                f"monitor target {attr!r} on {type(module).__name__} is not a "
                "registered memory of a MemoryModule. Pass allow_buffer=True "
                "to record plain buffers."
            )

    def _init_record_carry(self) -> dict[str, Tensor | tuple[Tensor, ...]]:
        """Allocate the streaming carry (shapes via FakeTensor, no real alloc).

        Uses the current state buffers as shape/dtype/device examples, so it must
        run after ``init_state``/``reset`` (state exists by the first forward).
        """
        example = EagerFrame(self._state_resolver.snapshot())
        return self._recorder.init_carry(example)

    def _finalize_records(
        self, carry: dict, record_buffers: dict, grad_snapshots: list, T: int
    ) -> None:
        """Turn the streamed carry + materialise buffers into the records dict.

        ``grad_snapshots[t]`` holds the per-step source tensors (one per grad
        monitor, in ``_record_grad_refs`` order) for backward-time hooks.
        """
        self._records = self._recorder.finalize(carry, record_buffers)
        for i, key in enumerate(self._record_grad_keys):
            history: list = [None] * T
            self._records[key] = history
            for t in range(T):
                self._register_record_grad_hook(grad_snapshots[t][i], key, t)
        if self.cpu_offload:
            self._records = {
                k: (v.cpu() if torch.is_tensor(v) else v)
                for k, v in self._records.items()
            }

    def _register_record_grad_hook(self, tensor: Tensor, key: str, t: int):
        def grad_hook(grad):
            if grad is not None:
                self._records[key][t] = grad.detach().clone()
            return grad

        if torch.is_tensor(tensor) and tensor.requires_grad:
            tensor.register_hook(grad_hook)

    def get_records(self) -> dict[str, Tensor | list]:
        """Recorded monitor outputs (Expr / grad(...) specs of
        update_state_names).

        Reduction / derived / raw records are tensors; ``grad(...)`` monitors are
        ``list[Tensor | None]`` of length T, populated after ``backward()``.
        """
        return self._records

    def clear_records(self) -> None:
        """Drop the recorded outputs from the last forward."""
        self._records = {}


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
                # single source of truth for both recording paths (legacy
                # stacked_states collection + the btorch.monitor recorder)
                self.allow_buffer = allow_buffer

            def _state_module(self):
                # matches filter_hidden_states(self.rnn_cell, ...) in the step
                return self.rnn_cell

            def _state_allow_buffer(self) -> bool:
                return self.allow_buffer

            def single_step_forward(self, *args, **kwargs):
                out = self.rnn_cell(*args, **kwargs)
                states = filter_hidden_states(
                    self.rnn_cell,
                    self.update_state_names,
                    allow_buffer=self.allow_buffer,
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
        update_state_names: RecordSpec | None = None,
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
        # single source of truth for both recording paths (legacy
        # stacked_states collection + the btorch.monitor recorder)
        self.allow_buffer = allow_buffer

    def _state_allow_buffer(self) -> bool:
        return self.allow_buffer

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
        update_state_names: Recording spec (see RecurrentNNAbstract): dotted
            strings -> stacked_states; Expr / grad(...) / {name: expr} ->
            get_records().
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
        update_state_names: RecordSpec | None = None,
        unroll: int | bool = 8,
        chunk_size: int | None = None,
        cpu_offload: bool = False,
        grad_checkpoint: bool = False,
        allow_buffer=False,
        **kwargs,
    ):
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
        update_state_names: Recording spec; see RecurrentNNAbstract.
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
        update_state_names: RecordSpec | None = None,
        **kwargs,
    ):
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
