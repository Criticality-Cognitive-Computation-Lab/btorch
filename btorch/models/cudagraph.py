"""CUDA graph capture/replay for stateful multi-step loops.

A T-step RNN over a small network is launch-bound: the kernels are tiny and the
CPU cannot queue them fast enough to keep the GPU busy. Capturing the whole loop
collapses T*k launches into a single replay.

This composes with ``torch.compile`` rather than competing with it: Inductor
still fuses the unroll block, and the graph is captured around the resulting
compiled kernels, so the two wins stack. Measured on the launch-bound regime
(T=500, batch=1, hidden=32) against eager:

    torch.compile(mode="reduce-overhead")   4.1x   (one replay per unroll block)
    capture around eager                    3.2x   (one replay, unfused kernels)
    capture around torch.compile()          5.0x   (one replay, fused kernels)

Correctness rests on three things that in-place-ness makes subtle:

* btorch cells *rebind* their state (``self.h = tanh(...)``) rather than mutating
  it, so the address the graph reads at step 0 is not the one the module holds
  after ``reset_net_state``. The initial state is copied into the captured buffer
  on every replay, and the module is re-pointed at the captured final state
  afterwards.
* Replay writes into a private memory pool, so outputs are cloned out before
  being returned -- otherwise the next replay would silently rewrite tensors the
  caller is still holding.
* A graph is only valid for the shapes/dtypes it saw at capture, so entries are
  keyed by input signature and re-captured on a miss.
* A graph is also only valid for the *structure* it saw at capture. Replay
  re-runs recorded kernels, not python, so anything a submodule decides on the
  host (which branch its forward takes, which derived device buffers its
  backend keeps) is frozen at capture time. Submodules whose structure can
  change in place opt into the capture protocol below; the runner reads their
  versions on every call and captures again instead of replaying a graph
  recorded for a different structure.

Capture protocol (duck-typed; the runner imports no model code). Any submodule
may define either or both of:

* ``capture_version`` -- an attribute or property holding a hashable value
  (typically an ``int``) that changes whenever a previously captured graph of
  that submodule is no longer valid: rewired sparse topology, a re-planned
  execution path, reallocated backend buffers. In-place *value* edits
  (weights) must not change it: the graph reads those tensors by address.
* ``capture_incompatibility()`` -- returns ``None``, or a human-readable reason
  why the submodule cannot be captured in its current configuration (e.g. its
  forward synchronises with the host). Capture is refused with that reason.

What must stay *outside* the captured region: replay re-runs the captured device
kernels and nothing else, so any host-side work (a ``.cpu()`` move, a python-level
accumulation) would run once at capture and never again, silently returning
capture-time values on every replay. Callers therefore capture only device work
and keep host steps in eager python -- which is why ``rnn.py`` captures per
*chunk* and offloads between chunks, rather than capturing the whole loop.

Inference only -- a property of *this runner*, not of CUDA graphs. Capture under
grad mode does record an autograd graph (``out.grad_fn`` is set), but two separate
things then break backward:

* ``__call__`` writes the caller's data into the static buffers with ``copy_``.
  Those are in-place writes to tensors the capture-time autograd graph saved for
  backward (a matmul saves its input), so the version counter fires: "variable
  needed for gradient computation has been modified by an inplace operation".
* Even without that, the activations saved for backward live in the graph's
  private pool, which the next replay overwrites -- silently.

Note it is *these* buffers, the runner's own, that are the problem. The cells'
``self.h = tanh(...)`` is a rebind, not an in-place mutation: it leaves the old
tensor untouched at version 0, and is not what breaks backward.

Training with CUDA graphs works via ``torch.compile(model, mode="reduce-overhead")``,
whose cudagraph trees capture forward and backward as separate graphs (verified
against eager grads, including with ``grad_checkpoint``). ``make_graphed_callables``
is the equivalent manual API, but it warms up by running a backward, which this
loop's state mutation defeats. The runner refuses grad-recording calls up front
rather than let them fail later and elsewhere.
"""

import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from .functional import named_hidden_states, set_hidden_states


__all__ = [
    "CudaGraphRunner",
    "capture_incompatibilities",
    "capture_versions",
    "clone_graph_outputs",
]


def _capture_participants(
    module: torch.nn.Module,
) -> tuple[list[torch.nn.Module], list[tuple[str, torch.nn.Module]]]:
    """Submodules of ``module`` (itself included) that opt into the capture
    protocol: those defining ``capture_version``, and ``(dotted name, submodule)``
    for those defining ``capture_incompatibility()``."""
    versioned, checked = [], []
    for name, sub in module.named_modules():
        if hasattr(sub, "capture_version"):
            versioned.append(sub)
        if callable(getattr(sub, "capture_incompatibility", None)):
            checked.append((name or type(sub).__name__, sub))
    return versioned, checked


def capture_versions(module: torch.nn.Module) -> tuple:
    """Snapshot the ``capture_version`` of every participating submodule.

    A CUDA graph captured over ``module`` is valid only while this tuple is
    unchanged. :class:`CudaGraphRunner` checks it on every call; code that
    drives ``torch.cuda.CUDAGraph`` by hand can do the same.

    Args:
        module: Root of the module tree the captured region runs.

    Returns:
        One entry per submodule that defines ``capture_version``, in
        ``module.modules()`` order. Empty when none participates.

    Examples:
        >>> captured = capture_versions(model)            # doctest: +SKIP
        >>> with torch.cuda.graph(graph):
        ...     step()
        >>> if capture_versions(model) != captured:       # e.g. after rewiring
        ...     ...  # warm up and capture again instead of graph.replay()
    """
    return tuple(sub.capture_version for sub in _capture_participants(module)[0])


def capture_incompatibilities(module: torch.nn.Module) -> dict[str, str]:
    """Reasons why submodules of ``module`` cannot be captured in a CUDA graph.

    Args:
        module: Root of the module tree the captured region would run.

    Returns:
        ``{dotted submodule name: reason}`` for every submodule whose
        ``capture_incompatibility()`` returns a reason. Empty when the tree
        can be captured (or no submodule participates).
    """
    return _incompatibilities(_capture_participants(module)[1])


def _incompatibilities(checked: Sequence[tuple[str, torch.nn.Module]]) -> dict:
    reasons = {}
    for name, sub in checked:
        why = sub.capture_incompatibility()
        if why:
            reasons[name] = str(why)
    return reasons


def clone_graph_outputs(obj: Any) -> Any:
    """Copy replay results out of the graph's pool so they survive the next
    one."""
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: clone_graph_outputs(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(clone_graph_outputs(v) for v in obj)
    return obj


class _Entry:
    """One captured graph, specialised to a single input signature.

    Holds the static buffers the graph is hard-wired to read from and
    write to. Replay does not re-read the module's attributes, so every
    tensor the loop depends on must be copied *into* these buffers
    first, and everything the caller keeps must be copied *out* of them
    afterwards.
    """

    __slots__ = (
        "graph",
        "static_args",
        "static_state_in",
        "state_out",
        "outputs",
        "versions",
    )

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_args: tuple,
        static_state_in: dict[str, torch.Tensor],
        state_out: dict[str, Any],
        outputs: Any,
        versions: tuple = (),
    ) -> None:
        self.graph = graph
        self.static_args = static_args
        self.static_state_in = static_state_in
        self.state_out = state_out
        self.outputs = outputs
        # Capture versions (see the module docstring) this graph is valid for.
        self.versions = versions


class CudaGraphRunner:
    """Capture a stateful forward into a CUDA graph and replay it.

    Args:
        warmup: Iterations to run before capture. These flush pending
            ``torch.compile`` work and let the caching allocator settle; both
            would otherwise be recorded into -- and corrupt -- the capture.
        versions: Optional callable returning a hashable tuple that changes
            whenever captured graphs become invalid for a reason the module
            tree does not report itself (state ``fn`` closes over that lives
            outside ``module``). It is called on every call and combined with
            the ``capture_version`` of the module's submodules.

    Call as ``runner(module, fn, args, kwargs)``, where ``fn(*args, **kwargs)``
    runs the loop and ``module`` owns the hidden state that ``fn`` advances.

    Submodules of ``module`` take part in the capture protocol described in the
    module docstring: a changed ``capture_version`` makes the next call capture
    again (warm-up included) instead of replaying, and a non-``None``
    ``capture_incompatibility()`` is refused. The participating submodules are
    looked up once per module object (on the first call, since the runner is
    usually created before the module tree is complete) and their versions are
    read on every call. Call :meth:`reset` after adding or replacing submodules.

    Attributes:
        entries: Captured graphs by input signature.
        n_captures: Number of captures performed so far (never reset); a
            re-capture after a version change counts.
    """

    def __init__(
        self, warmup: int = 3, versions: Callable[[], tuple] | None = None
    ) -> None:
        self.warmup = warmup
        self.versions = versions
        self.entries: dict[tuple, _Entry] = {}
        self.n_captures = 0
        # (weakref to the module, versioned submodules, incompatibility checkers)
        self._participants: tuple | None = None

    def _participants_of(self, module: torch.nn.Module) -> tuple[list, list]:
        cached = self._participants
        if cached is None or cached[0]() is not module:
            cached = (weakref.ref(module), *_capture_participants(module))
            self._participants = cached
        return cached[1], cached[2]

    def _versions(self, module: torch.nn.Module) -> tuple:
        """Current capture versions: a handful of python reads per call."""
        versioned = self._participants_of(module)[0]
        current = tuple(sub.capture_version for sub in versioned)
        if self.versions is not None:
            current = (*current, self.versions())
        return current

    @staticmethod
    def _key(args: Sequence, kwargs: Mapping[str, Any]) -> tuple:
        def describe(v):
            if torch.is_tensor(v):
                return ("t", tuple(v.shape), v.dtype, v.device)
            return ("v", v)

        return (
            tuple(describe(a) for a in args),
            tuple(sorted((k, describe(v)) for k, v in kwargs.items())),
        )

    @staticmethod
    def _check_supported(
        module: torch.nn.Module,
        args: Sequence,
        incompatible: Mapping[str, str] | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("cudagraph=True requires CUDA to be available.")
        devices = {a.device for a in args if torch.is_tensor(a)}
        if any(d.type != "cuda" for d in devices):
            raise RuntimeError(
                f"cudagraph=True requires all inputs on a CUDA device, got {devices}."
            )
        # Catch training *before* the forward. Inspecting the args alone is not
        # enough: the usual training shape is plain data plus parameters that
        # require grad, which still records an autograd graph whose saved tensors
        # are the very buffers __call__ then rewrites with copy_ -- surfacing as an
        # inplace-version error from backward(), long after the call that caused it.
        if torch.is_grad_enabled() and (
            any(a.requires_grad for a in args if torch.is_tensor(a))
            or any(p.requires_grad for p in module.parameters())
        ):
            raise RuntimeError(
                "cudagraph=True is inference-only: each call rewrites the runner's "
                "static buffers in place, which invalidates any autograd graph "
                "recorded over them, and backward() then fails. Run under "
                "torch.no_grad() / torch.inference_mode(). To use CUDA graphs while "
                'training, use torch.compile(model, mode="reduce-overhead") instead '
                "-- its cudagraph trees do capture forward and backward."
            )
        for attr, why in (incompatible or {}).items():
            raise RuntimeError(
                f"cudagraph=True is incompatible with {attr}=True ({why})."
            )

    def _check_submodules(self, module: torch.nn.Module) -> None:
        """Refuse a module tree that reports it cannot be captured."""
        checked = self._participants_of(module)[1]
        for name, why in _incompatibilities(checked).items():
            raise RuntimeError(
                f"cudagraph=True is incompatible with submodule '{name}' ({why})."
            )

    def _capture(
        self,
        module: torch.nn.Module,
        fn: Callable[..., Any],
        args: Sequence,
        kwargs: Mapping[str, Any],
    ) -> _Entry:
        # Snapshot the entry state *by value*. Warmup and capture both advance the
        # module, and capture only records kernels without running them, so what
        # the module holds afterwards is not a state at all -- it is uninitialised
        # pool memory. Nothing below may read the module's live state expecting to
        # find the caller's initial one.
        snapshot = {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in named_hidden_states(module).items()
        }

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Runtime participants may allocate derived buffers lazily on the
            # first call. Even warmup=0 needs one uncaptured settle pass so no
            # task layout, workspace or compiled kernel is first created while
            # the graph is being recorded.
            for _ in range(max(1, self.warmup)):
                set_hidden_states(module, snapshot)
                fn(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        # Freeze inputs and initial state at addresses the graph can hard-wire.
        static_args = tuple(
            a.detach().clone() if torch.is_tensor(a) else a for a in args
        )
        static_state_in = {
            k: v.detach().clone() for k, v in snapshot.items() if torch.is_tensor(v)
        }
        set_hidden_states(module, {**snapshot, **static_state_in})

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = fn(*static_args, **kwargs)

        # After capture the module points at the graph's final-step tensors, and
        # replay writes to those same addresses -- keep them to re-point later.
        state_out = named_hidden_states(module)
        # Hand the module back its real pre-capture state, so the replay that
        # follows starts from that rather than from the pool garbage capture left
        # bound -- and so __call__ needs no special case for the capturing call.
        set_hidden_states(module, snapshot)
        self.n_captures += 1
        # Stamp with the versions read *after* capture: a submodule may settle
        # its own derived state lazily during warm-up (and bump its version
        # doing so), and the graph was recorded against the settled state.
        return _Entry(
            graph,
            static_args,
            static_state_in,
            state_out,
            outputs,
            self._versions(module),
        )

    def __call__(
        self,
        module: torch.nn.Module,
        fn: Callable[..., Any],
        args: Sequence,
        kwargs: Mapping[str, Any] | None = None,
        clone_outputs: bool = True,
        incompatible: Mapping[str, str] | None = None,
    ) -> Any:
        """Replay ``fn(*args, **kwargs)``, capturing it first if needed.

        Set ``clone_outputs=False`` only if the caller copies the results out of
        the pool itself before the next replay (``.cpu()`` counts); otherwise they
        are live only until then.

        ``incompatible`` maps option names the caller has enabled but capture
        cannot honour to the reason; any entry is refused. The runner knows
        nothing about the module's own options.
        """
        kwargs = kwargs or {}
        self._check_supported(module, args, incompatible)
        self._check_submodules(module)

        key = self._key(args, kwargs)
        entry = self.entries.get(key)
        if entry is not None and entry.versions != self._versions(module):
            # Recorded for a different structure (rewired topology, another
            # execution plan, ...): replaying it would silently compute with
            # the old one. Capture again, exactly like a first call. The stale
            # entry stays in ``entries`` until the new capture has snapshotted
            # the module state, which may still live in the stale graph's pool.
            entry = None
        if entry is None:
            entry = self._capture(module, fn, args, kwargs)
            self.entries[key] = entry
            # Graphs of other signatures that are stale as well can never be
            # replayed again; drop them now to release their memory pools.
            self.entries = {
                k: e for k, e in self.entries.items() if e.versions == entry.versions
            }

        for static, incoming in zip(entry.static_args, args):
            if torch.is_tensor(static):
                static.copy_(incoming)

        # The caller may have reset or advanced the state since capture, and the
        # cells rebind rather than mutate, so feed the state in by value.
        current = named_hidden_states(module)
        for name, buf in entry.static_state_in.items():
            buf.copy_(current[name])

        entry.graph.replay()

        set_hidden_states(module, entry.state_out)
        return clone_graph_outputs(entry.outputs) if clone_outputs else entry.outputs

    def reset(self) -> None:
        """Drop captured graphs and their memory pools, and forget which
        submodules take part in the capture protocol."""
        self.entries.clear()
        self._participants = None
