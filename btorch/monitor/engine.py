"""The :class:`Recorder`: compile record specs once, drive them via a frame.

Consumer contract (RNN-agnostic): construct with specs + a :class:`Resolver`
rooted on the consumer's state module, then thread a functional carry:

    carry = recorder.init_carry(example_frame)
    for frame in frames:            # one per timestep
        carry, chunk_stack = recorder.step_kernel(carry, frame)
        for nid, t in chunk_stack.items():
            buffers[nid].append(t)  # cross-chunk append lives in the driver
    records = recorder.finalize(carry, buffers)

``carry`` is an immutable tensor pytree (grad-checkpoint-safe; a discarded
recompute carry changes nothing).  Raw columns also appear in ``raw_columns`` so
the consumer can alias them into its native channel (e.g. ``stacked_states``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from . import expr as E
from .expr import Expr
from .frame import Resolver, StepFrame
from .ir import build
from .lower import CompiledProgram, lower
from .reducer import _check_reducer_purity


#: What ``update_state_names=`` accepts: a single spec (a dotted ``str`` or an
#: :class:`~btorch.monitor.expr.Expr`), a sequence of them (entries may also be a
#: ``{name: expr}`` mapping renaming a processed monitor), or a top-level
#: ``{name: expr}`` mapping.  A top-level mapping routes every value (including
#: plain strings) through the recorder -- it renames AND switches channel:
#: the legacy ``stacked_states`` output is disabled entirely for that network.
RecordSpec = str | Expr | Mapping | Sequence


def _normalize_specs(specs) -> list:
    """Flatten specs into a list of ``str | Expr``.

    Accepts a sequence (entries may be ``str``, ``Expr``, or a ``{name: expr}``
    mapping renaming a processed monitor) or a single top-level mapping.
    """
    out: list = []
    if isinstance(specs, Mapping):
        items = specs.items()
    else:
        items = None
    if items is not None:
        for name, e in items:
            out.append(_alias(e, name))
        return out
    if isinstance(specs, (str, E.Expr)):
        return [specs]
    if not isinstance(specs, Sequence):
        raise TypeError(f"record specs must be a sequence or mapping, got {specs!r}")
    for spec in specs:
        if isinstance(spec, Mapping):
            for name, e in spec.items():
                out.append(_alias(e, name))
        else:
            out.append(spec)
    return out


def _alias(e, name: str):
    if isinstance(e, str):
        e = E.col(e)
    if not isinstance(e, E.Expr):
        raise TypeError(f"record value must be str or Expr, got {type(e)}")
    return e.alias(name)


class Recorder:
    """Compile record specs once against a :class:`Resolver`, then drive them.

    Args:
        specs: Record specs (see :data:`RecordSpec` for the accepted grammar);
            plain strings are treated as ``col(name)``.
        resolver: Resolves dotted names / bare tensors to refs under the
            consumer's state module.

    Drive with ``init_carry`` / ``step_kernel`` (one frame per timestep) /
    ``finalize`` -- threading the carry and appending stack outputs to buffers
    across chunk boundaries is the DRIVER's job, as in
    :meth:`RecurrentNNAbstract._multi_step_forward_impl` -- or one-shot via
    :meth:`run(frames)`, which returns ``{name: Tensor}``.

    Every ``fold(reducer)`` is purity-checked here at build time (cheap, no
    tracing); a reducer subclass can opt out with ``auto_validate = False``.
    """

    def __init__(self, specs, *, resolver: Resolver):
        self.resolver = resolver
        self.graph = build(_normalize_specs(specs), resolver)
        self.program: CompiledProgram = lower(self.graph)
        self._check_reducers()

    def _check_reducers(self):
        """Run the cheap invariant checks on each fold(reducer), once.

        Catches impure reducers at construction -- where the mistake is made --
        instead of silently corrupting results under checkpoint/cudagraph later.
        Shape-agnostic by design: a scalar example suffices.
        """
        example = torch.zeros(())
        for node in self.graph.nodes:
            if node.op == "creduce" and getattr(
                node.params["reducer"], "auto_validate", True
            ):
                _check_reducer_purity(node.params["reducer"], example)

    # -- introspection ----------------------------------------------------
    @property
    def has_grad_specs(self) -> bool:
        return bool(self.graph.grad_specs)

    @property
    def grad_specs(self):
        """List of ``(key, ref, dotted_name)`` for grad monitors (eager
        hooks)."""
        return self.graph.grad_specs

    @property
    def raw_columns(self) -> set[str]:
        """Names whose root is a bare column -> aliased into stacked_states."""
        names: set[str] = set()
        for nid in self.graph.roots:
            node = self.graph.node(nid)
            if node.op == "source":
                names.update(node.names)
        return names

    @property
    def materialized_columns(self) -> set[str]:
        names: set[str] = set()
        for nid in self.program.stack_nids:
            names.update(self.graph.node(nid).names)
        return names

    @property
    def stack_nids(self):
        return self.program.stack_nids

    @property
    def is_empty(self) -> bool:
        return not self.graph.roots and not self.graph.grad_specs

    # -- driving ----------------------------------------------------------
    def init_carry(self, example_frame: StepFrame) -> dict:
        return self.program.init_carry(example_frame)

    def step_kernel(self, carry, frame: StepFrame):
        return self.program.step_kernel(carry, frame)

    def new_buffers(self) -> dict:
        return {nid: [] for nid in self.program.stack_nids}

    def finalize(self, carry, buffers) -> dict:
        """Reduce the final carry (+ stacked raw buffers) to records.

        Raw source columns keep their autograd connection; every derived
        or reduced record is detached (reductions stream through an O(1)
        carry that is not part of the forward graph).
        """
        return self.program.finalize(carry, buffers)

    def run(self, frames: Sequence[StepFrame]) -> dict:
        """Convenience whole-sequence driver (tests / simple consumers)."""
        if not frames:
            raise ValueError("run() needs at least one frame")
        carry = self.init_carry(frames[0])
        buffers = self.new_buffers()
        for frame in frames:
            carry, stack = self.step_kernel(carry, frame)
            for nid, t in stack.items():
                buffers[nid].append(t)
        return self.finalize(carry, buffers)
