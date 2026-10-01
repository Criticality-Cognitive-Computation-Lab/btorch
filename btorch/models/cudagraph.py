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

from typing import Any

import torch

from .functional import named_hidden_states, set_hidden_states


__all__ = ["CudaGraphRunner", "clone_graph_outputs"]


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

    __slots__ = ("graph", "static_args", "static_state_in", "state_out", "outputs")

    def __init__(self, graph, static_args, static_state_in, state_out, outputs):
        self.graph = graph
        self.static_args = static_args
        self.static_state_in = static_state_in
        self.state_out = state_out
        self.outputs = outputs


class CudaGraphRunner:
    """Capture a stateful forward into a CUDA graph and replay it.

    Args:
        warmup: Iterations to run before capture. These flush pending
            ``torch.compile`` work and let the caching allocator settle; both
            would otherwise be recorded into -- and corrupt -- the capture.

    Call as ``runner(module, fn, args, kwargs)``, where ``fn(*args, **kwargs)``
    runs the loop and ``module`` owns the hidden state that ``fn`` advances.
    """

    def __init__(self, warmup: int = 3):
        self.warmup = warmup
        self.entries: dict[tuple, _Entry] = {}

    @staticmethod
    def _key(args, kwargs):
        def describe(v):
            if torch.is_tensor(v):
                return ("t", tuple(v.shape), v.dtype, v.device)
            return ("v", v)

        return (
            tuple(describe(a) for a in args),
            tuple(sorted((k, describe(v)) for k, v in kwargs.items())),
        )

    @staticmethod
    def _check_supported(module, args):
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
        # Not merely unsupported: the capture path never routes through
        # _checkpointed_large_chunk, so this would be silently ignored -- no
        # recompute, no memory saved -- on top of being backward-only.
        if getattr(module, "grad_checkpoint", False):
            raise RuntimeError(
                "cudagraph=True is incompatible with grad_checkpoint=True (the "
                "capture path would silently ignore it)."
            )
        # NOTE: value/reduction recording monitors DO compose with cudagraph --
        # their fold is captured inside the chunk graph (see
        # RecurrentNNAbstract._stacked_chunk_forward). grad(...) monitors and
        # multi-chunk recording are refused in _cudagraph_multi_step instead, where
        # the recorder and chunk count are known.

    def _capture(self, module, fn, args, kwargs):
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
            for _ in range(self.warmup):
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
        return _Entry(graph, static_args, static_state_in, state_out, outputs)

    def __call__(self, module, fn, args, kwargs=None, clone_outputs=True):
        """Replay ``fn(*args, **kwargs)``, capturing it first if needed.

        Set ``clone_outputs=False`` only if the caller copies the results out of
        the pool itself before the next replay (``.cpu()`` counts); otherwise they
        are live only until then.
        """
        kwargs = kwargs or {}
        self._check_supported(module, args)

        key = self._key(args, kwargs)
        entry = self.entries.get(key)
        if entry is None:
            entry = self._capture(module, fn, args, kwargs)
            self.entries[key] = entry

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

    def reset(self):
        """Drop captured graphs and their memory pools."""
        self.entries.clear()
