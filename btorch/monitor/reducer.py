"""Custom recording extension points.

Two levels, from safest to most powerful:

* :class:`Reducer` -- a **custom streaming reduction**.  Subclass it with pure
  ``init`` / ``update`` / ``finalize`` tensor functions and plug it in with
  ``col("neuron.v").fold(MyReducer())``.  The engine threads the (tensor) carry,
  so a reducer composes with ``torch.compile`` / ``cudagraph`` / ``cpu_offload`` /
  ``grad_checkpoint`` **automatically** -- there is nothing mode-specific to get
  right, as long as the three methods are pure tensor ops (no in-place mutation of
  the carry, no ``.item()``, no Python-scalar carry).  (At the RNN level,
  ``cudagraph=True`` recording additionally requires a single chunk.)  This is
  the recommended way to add custom recording.

* :class:`RecordEngine` -- the full driver contract (``init_carry`` / ``step_kernel``
  / ``finalize`` / ...).  A wholesale replacement for the DSL engine, driven by the
  RNN's recording loop.  It gives total control but you now **own** the
  mode-correctness invariants below; prefer :class:`Reducer` unless you truly need
  to replace the engine.

Use :func:`validate_reducer` to check a reducer against the invariants before
trusting it in a compiled / captured loop.
"""

from __future__ import annotations

import copy
from typing import Any, Protocol

import torch
from torch import Tensor

from .frame import StepFrame, TargetRef


class Reducer:
    """A custom streaming reduction over time.

    The carry may be any pytree of tensors.  ``valid`` in :meth:`update` is a scalar
    per-step validity -- ``1.0``, or ``0.0`` during a windowed input's warmup (e.g.
    after ``.diff()``).  Use it to keep counts/denominators correct: multiply it into
    additive accumulators (``total += value * valid``; ``count += valid``) or mask on
    it (``torch.where(valid > 0, value, ...)``).
    """

    def init(self, example: Tensor) -> Any:
        """Return the initial carry, given an example step value (for
        shape/dtype/ device).

        Must depend only on ``example``'s shape/dtype/device, not values.
        """
        raise NotImplementedError

    def update(self, carry: Any, value: Tensor, valid) -> Any:
        """Fold ``value`` (this step's ``[B, N]``) into ``carry``; return the
        new carry.

        ``valid`` is a scalar 0.0/1.0 validity (tensor or ``float``).
        """
        raise NotImplementedError

    def finalize(self, carry: Any) -> Tensor:
        """Reduce the final carry to the recorded tensor."""
        raise NotImplementedError


class RecordEngine(Protocol):
    """The full driver contract, duck-typed by the RNN loop (and
    ``Recorder.run``).

    A conforming object composes with all execution modes ONLY IF:

    1. ``step_kernel`` is **pure**: it must not mutate ``carry`` in place, and the
       returned carry must be a pytree of **tensors** (no Python scalars) -- so
       grad_checkpoint recompute is discardable and cudagraph can capture the fold.
    2. No ``.item()`` / numpy / data-dependent Python branching in ``step_kernel``.
    3. ``finalize`` is a pure function of the carry + buffers.

    :class:`~btorch.monitor.Recorder` is the reference implementation.  The RNN loop
    uses ``init_carry`` / ``step_kernel`` / ``finalize`` / ``is_empty`` /
    ``has_grad_specs`` / ``grad_specs``; ``stack_nids`` / ``new_buffers`` are the
    extra hooks the ``Recorder.run`` convenience driver uses.
    """

    def init_carry(self, example_frame: StepFrame) -> Any:
        """Allocate and return the initial carry, shaped from an example
        frame."""
        raise NotImplementedError

    def step_kernel(self, carry: Any, frame: StepFrame) -> tuple[Any, dict]:
        """Fold one step's ``frame`` into ``carry``; return ``(carry,
        stack)``."""
        raise NotImplementedError

    def finalize(self, carry: Any, buffers: dict) -> dict[str, Tensor]:
        """Reduce the final carry (+ stacked raw buffers) to records."""
        raise NotImplementedError

    def new_buffers(self) -> dict:
        """Fresh per-run containers for stacked raw (per-step) values."""
        raise NotImplementedError

    @property
    def stack_nids(self):
        """Ids of per-step values to materialise (stacked over time)."""
        raise NotImplementedError

    @property
    def grad_specs(self) -> list[tuple[str, TargetRef, str]]:
        """``(name, ref, mode)`` triples recorded via backward hooks."""
        raise NotImplementedError

    @property
    def has_grad_specs(self) -> bool:
        """Whether any gradient monitors are registered."""
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        """Whether no recording specs are registered at all."""
        raise NotImplementedError


def validate_reducer(reducer: Reducer, example: Tensor, *, steps: int = 4) -> None:
    """Check a :class:`Reducer` against the mode-correctness invariants.

    Raises a clear error on a violation: a non-tensor carry, in-place mutation of
    the carry / the ``value`` / the reducer instance, a carry structure that changes
    between ``init`` and ``update``, or code that is not ``fullgraph``-traceable
    (``.item()``, numpy, data-dependent Python branching).

    Coverage notes: ``finalize`` must return a ``Tensor``.  Attributes of an
    unknown (non-tensor/scalar/container) type are SKIPPED by the self-mutation
    check -- value-comparing arbitrary objects is unsound -- so keep mutable
    state in tensors or standard containers.

    Note: this resets the process-wide Dynamo compile cache
    (``torch._dynamo.reset()``) to measure traceability in isolation.
    """
    from torch.utils._pytree import tree_flatten

    carry0 = reducer.init(torch.zeros_like(example))
    leaves, init_spec = tree_flatten(carry0)
    if not leaves or any(not isinstance(x, Tensor) for x in leaves):
        raise TypeError(
            "Reducer.init must return a (non-empty) pytree of tensors; got a carry "
            f"with a non-tensor leaf: {carry0!r}"
        )

    # Snapshot everything update might illegally mutate: the carry, the input
    # value, and the reducer's own state (value-comparable attributes; see
    # _attr_snapshot).
    probe = torch.ones_like(example)  # non-zero, so in-place folds are observable
    carry_before = [x.clone() for x in leaves]
    probe_before = probe.clone()
    self_before = _attr_snapshot(reducer)

    new_carry = reducer.update(carry0, probe, 1.0)

    carry_after, _ = tree_flatten(carry0)
    if any(not torch.equal(a, b) for a, b in zip(carry_after, carry_before)):
        raise RuntimeError(
            "Reducer.update mutated its carry in place; it must be functional "
            "(return a new carry) so grad_checkpoint / cudagraph stay correct. "
            "Put all per-step state in the returned carry, not in the argument."
        )
    if not torch.equal(probe, probe_before):
        raise RuntimeError(
            "Reducer.update mutated its `value` argument in place; that corrupts the "
            "network's live state buffer. Treat `value` as read-only."
        )
    self_after = _attr_snapshot(reducer)
    for k, before in self_before.items():
        if _attr_changed(before, self_after[k]):
            raise RuntimeError(
                f"Reducer mutated its own state (self.{k}) during update; keep the "
                "instance immutable and thread all per-step state through the carry."
            )
    _, new_spec = tree_flatten(new_carry)
    if new_spec != init_spec:
        raise RuntimeError(
            f"Reducer.update returned a carry with structure {new_spec} but init "
            f"produced {init_spec}; the carry pytree must be constant across steps."
        )

    def run(x):
        carry = reducer.init(x)
        for _ in range(steps):
            carry = reducer.update(carry, x, x.new_ones(()))  # tensor validity
        carry = reducer.update(carry, x, x.new_zeros(()))  # zero validity (warmup)
        out = reducer.finalize(carry)
        if not isinstance(out, Tensor):
            raise TypeError(f"Reducer.finalize must return a Tensor, got {type(out)}")
        return out

    eager = run(example)
    torch._dynamo.reset()
    compiled = torch.compile(run, fullgraph=True)(example)
    torch.testing.assert_close(
        eager,
        compiled,
        msg="Reducer is not fullgraph-traceable (or differs under "
        "compile): avoid .item(), numpy, and data-dependent Python branching.",
    )


def _attr_snapshot(reducer) -> dict:
    """Snapshot a reducer's attributes for a before/after mutation check.

    Only attributes with a sound *value* comparison are tracked: tensors are
    cloned, immutable scalars copied by value, standard containers deep-copied.
    Arbitrary objects are skipped -- their equality semantics are unknowable
    (``repr`` can embed volatile state such as ids or counters), so comparing
    them risks flagging semantically-unchanged state as mutated.
    """
    snap: dict = {}
    for k, v in vars(reducer).items():
        if isinstance(v, Tensor):
            snap[k] = v.detach().clone()
        elif isinstance(v, int | float | bool | bytes | str | complex | type(None)):
            snap[k] = v
        elif isinstance(v, list | tuple | set | frozenset | dict):
            snap[k] = copy.deepcopy(v)
    return snap


def _attr_changed(before, after) -> bool:
    if isinstance(before, Tensor):
        return not (isinstance(after, Tensor) and torch.equal(before, after))
    return before != after
