"""Accessor indirection for the recording engine.

Targets (dotted state names such as ``"neuron.v"``) resolve **once**, in eager,
to opaque :class:`TargetRef` handles; per-step values are pulled from a
:class:`StepFrame`.  :class:`EagerFrame` reads live module buffers by attribute;
a future ``ScanFrame`` would read a carry slot instead -- the monitor kernels
never learn which, so the backend is swappable.

Resolution is consumer-agnostic: the consumer supplies a *root* module and we
resolve dotted names (or bare live tensors, via identity lookup) against it.  The
root is a consumer input on purpose -- e.g. an RNN wrapper resolves against its
inner cell, not itself.  This module has **no** dependency on ``btorch.models``.
"""

from __future__ import annotations

from typing import NewType, Protocol

import torch.nn as nn
from torch import Tensor


# An opaque handle naming a logical state slot.  Concretely an index into a
# resolver's accessor list; monitors treat it as opaque.
TargetRef = NewType("TargetRef", int)


class StepFrame(Protocol):
    """Per-step view: maps a resolved ref to its current tensor value."""

    def __getitem__(self, ref: TargetRef) -> Tensor: ...


def _walk(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    """Resolve a dotted name to ``(owning_module, leaf_attr)``.

    Strips a leading ``self.`` / ``self`` (matching the convention used by
    ``btorch.models.functional``).  Raises a clear error listing available names
    if any path component is missing.
    """
    name = dotted.removeprefix("self.").removeprefix("self")
    parts = name.split(".")
    m: nn.Module = root
    for i, p in enumerate(parts[:-1]):
        if not hasattr(m, p):
            raise KeyError(
                f"cannot resolve target {dotted!r}: {p!r} not found on "
                f"{type(m).__name__} (at {'.'.join(parts[:i]) or '<root>'})"
            )
        m = getattr(m, p)
    return m, parts[-1]


def dotted_name_of_buffer(root: nn.Module, tensor: Tensor) -> str:
    """Reverse-lookup a live tensor to its dotted buffer name by **identity**.

    Used to normalise a bare ``mod.v`` reference into a dotted string, once, at
    registration time (buffers are reassigned every step, so identity is only
    valid before stepping begins).
    """
    for name, buf in root.named_buffers():
        if buf is tensor:
            return name
    raise ValueError(
        "bare-tensor target is not a registered buffer of the root module "
        "(is it a fresh tensor, or was it captured before init_state/reset?). "
        "Pass a dotted string like 'neuron.v' instead."
    )


class Resolver:
    """Resolves dotted names / bare tensors to :class:`TargetRef` under a root.

    One resolver per :class:`~btorch.monitor.engine.Recorder`.  ``resolve`` is
    idempotent per name, so shared targets across monitors collapse to one ref
    (and one per-step read).
    """

    def __init__(self, root: nn.Module, *, allow_buffer: bool = False):
        self.root = root
        self.allow_buffer = allow_buffer
        self._accessors: list[tuple[nn.Module, str]] = []
        self._by_name: dict[str, TargetRef] = {}

    def resolve(self, target: str | Tensor) -> tuple[TargetRef, str]:
        """Return ``(ref, dotted_name)`` for a dotted string or bare tensor."""
        name = self._to_name(target)
        if name in self._by_name:
            return self._by_name[name], name
        module, attr = _walk(self.root, name)
        if not hasattr(module, attr):
            avail = sorted(n for n, _ in self.root.named_buffers())
            raise KeyError(
                f"target {name!r} not found (buffer missing on "
                f"{type(module).__name__}). Available: {avail}. "
                "If state is created in init_state/reset, resolve after the "
                "first forward."
            )
        ref = TargetRef(len(self._accessors))
        self._accessors.append((module, attr))
        self._by_name[name] = ref
        return ref, name

    def _to_name(self, target: str | Tensor) -> str:
        if isinstance(target, str):
            return target.removeprefix("self.").removeprefix("self")
        if isinstance(target, Tensor):
            return dotted_name_of_buffer(self.root, target)
        raise TypeError(
            f"target must be a dotted str or a live buffer Tensor, got "
            f"{type(target).__name__}"
        )

    @property
    def n_refs(self) -> int:
        return len(self._accessors)

    def snapshot(self) -> list[Tensor]:
        """The current buffer tensors in ref order (no copy)."""
        return [getattr(m, a) for m, a in self._accessors]

    def frame(self) -> "EagerFrame":
        """Snapshot the current live buffers into a fresh frame.

        Snapshotting (not lazy reads) is essential: the consumer builds one frame
        right after ``single_step_forward`` so it captures *this* step's
        (freshly rebound) tensors, even if ``step_kernel`` runs slightly later.
        No copy -- it holds the tensor objects themselves.
        """
        return EagerFrame([getattr(m, a) for m, a in self._accessors])


class EagerFrame:
    """A per-step snapshot of resolved buffer tensors, indexed by
    :class:`TargetRef`."""

    __slots__ = ("_values",)

    def __init__(self, values: list[Tensor]):
        self._values = values

    def __getitem__(self, ref: TargetRef) -> Tensor:
        return self._values[ref]
