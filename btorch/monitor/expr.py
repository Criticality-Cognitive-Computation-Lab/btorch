"""Polars-style lazy expression DSL for recording temporal state.

The recorded run is a lazy frame: **columns are named state buffers** (dotted
names like ``"neuron.v"``), **rows are timesteps**.  Aggregations reduce over
*time*.  An :class:`Expr` is pure, immutable data describing a computation; it is
compiled once into a dataflow IR (:mod:`btorch.monitor.ir`) and lowered to a
single per-step kernel.

Native ops carry their own time-semantics (no ``s``/``m`` marker).  The only
single-step vs multi-step choice is at the custom-function boundary:

* :func:`map_step` -- ``fn(a_t, b_t, ...)`` per timestep on ``[B, N]``; streamable.
* :func:`map_seq`  -- ``fn(A, B, ...)`` once on the stacked ``[T, B, N]``; materialises.

Aggregations reduce over time; batch/feature dims are preserved and only reachable
via a custom fn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from torch import Tensor


_GRAD_MSG = (
    "grad(...) records raw per-step gradients and cannot be composed with "
    "expressions; only .alias() may wrap it. Record grad(...) and reduce in post."
)


class Expr:
    """Base expression node.

    Subclasses are immutable dataclasses.
    """

    is_grad: bool = False

    # -- internal guard ---------------------------------------------------
    def _no_compose(self) -> None:
        """Reject composition of a grad leaf (only .alias() may wrap grad)."""
        if self.is_grad:
            raise TypeError(_GRAD_MSG)

    # -- arithmetic (elementwise, streamable) -----------------------------
    def __add__(self, o) -> Expr:
        self._no_compose()
        return Elementwise("add", (self, _lit(o)))

    def __radd__(self, o) -> Expr:
        self._no_compose()
        return Elementwise("add", (_lit(o), self))

    def __sub__(self, o) -> Expr:
        self._no_compose()
        return Elementwise("sub", (self, _lit(o)))

    def __rsub__(self, o) -> Expr:
        self._no_compose()
        return Elementwise("sub", (_lit(o), self))

    def __mul__(self, o) -> Expr:
        self._no_compose()
        return Elementwise("mul", (self, _lit(o)))

    def __rmul__(self, o) -> Expr:
        self._no_compose()
        return Elementwise("mul", (_lit(o), self))

    def __truediv__(self, o) -> Expr:
        self._no_compose()
        return Elementwise("div", (self, _lit(o)))

    def __rtruediv__(self, o) -> Expr:
        self._no_compose()
        return Elementwise("div", (_lit(o), self))

    def __neg__(self) -> Expr:
        self._no_compose()
        return Elementwise("neg", (self,))

    def __abs__(self) -> Expr:
        self._no_compose()
        return Elementwise("abs", (self,))

    def abs(self) -> Expr:
        return self.__abs__()

    def relu(self) -> Expr:
        self._no_compose()
        return Elementwise("relu", (self,))

    def clamp(self, min=None, max=None) -> Expr:
        self._no_compose()
        return Elementwise("clamp", (self,), {"min": min, "max": max})

    def pow(self, exponent: float) -> Expr:
        self._no_compose()
        return Elementwise("pow", (self,), {"exponent": exponent})

    # -- reductions over time (streamable via carry) ----------------------
    def mean(self) -> Expr:
        self._no_compose()
        return Reduce("mean", self)

    def sum(self) -> Expr:
        self._no_compose()
        return Reduce("sum", self)

    def last(self) -> Expr:
        self._no_compose()
        return Reduce("last", self)

    def first(self) -> Expr:
        self._no_compose()
        return Reduce("first", self)

    def min(self) -> Expr:
        self._no_compose()
        return Reduce("min", self)

    def max(self) -> Expr:
        self._no_compose()
        return Reduce("max", self)

    def count(self) -> Expr:
        self._no_compose()
        return Reduce("count", self)

    def std(self) -> Expr:
        self._no_compose()
        return Reduce("std", self)

    def var(self) -> Expr:
        self._no_compose()
        return Reduce("var", self)

    # -- windowed (streamable via ring carry) -----------------------------
    def diff(self, n: int = 1) -> Expr:
        self._no_compose()
        return Window("diff", n, self)

    def shift(self, n: int = 1) -> Expr:
        self._no_compose()
        return Window("shift", n, self)

    # -- custom-function boundary (the s/m marker) ------------------------
    def map_step(self, fn: Callable[..., Tensor]) -> Expr:
        self._no_compose()
        return MapStep(fn, (self,))

    def map_seq(self, fn: Callable[..., Tensor]) -> Expr:
        self._no_compose()
        return MapSeq(fn, (self,))

    def pipe(self, fn: Callable[..., Tensor]) -> Expr:
        """Polars-style alias of :meth:`map_seq`: run ``fn`` on the stacked
        column."""
        return self.map_seq(fn)

    def fold(self, reducer) -> Expr:
        """Reduce this column over time with a custom streaming
        :class:`~btorch.monitor.reducer.Reducer` (O(1) carry, all-mode-
        safe)."""
        self._no_compose()
        return CustomReduce(reducer, self)

    # -- naming -----------------------------------------------------------
    def alias(self, name: str) -> Expr:
        return Alias(name, self)


def _lit(o) -> Expr:
    return o if isinstance(o, Expr) else Lit(o)


# ---------------------------------------------------------------------------
# Node types (immutable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Col(Expr):
    """A state target (column).

    ``target`` is a dotted name or a live tensor.
    """

    target: str | Tensor


@dataclass(frozen=True, eq=False)
class Lit(Expr):
    """A constant scalar/tensor broadcast into the per-step computation."""

    value: object


@dataclass(frozen=True, eq=False)
class Grad(Expr):
    """Per-step gradient of a target (eager-only, via ``register_hook``)."""

    target: str | Tensor
    is_grad: bool = True


@dataclass(frozen=True, eq=False)
class Elementwise(Expr):
    op: str
    args: tuple[Expr, ...]
    params: dict = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class Window(Expr):
    kind: str  # "diff" | "shift"
    n: int
    child: Expr


@dataclass(frozen=True, eq=False)
class Reduce(Expr):
    kind: str  # mean|sum|last|first|min|max|count|std|var
    child: Expr


@dataclass(frozen=True, eq=False)
class MapStep(Expr):
    fn: Callable[..., Tensor]
    args: tuple[Expr, ...]


@dataclass(frozen=True, eq=False)
class MapSeq(Expr):
    fn: Callable[..., Tensor]
    args: tuple[Expr, ...]


@dataclass(frozen=True, eq=False)
class CustomReduce(Expr):
    reducer: object  # a btorch.monitor.reducer.Reducer
    child: Expr


@dataclass(frozen=True, eq=False)
class Alias(Expr):
    name: str
    child: Expr


# ---------------------------------------------------------------------------
# Constructors (public)
# ---------------------------------------------------------------------------
def col(target: str | Tensor) -> Expr:
    """Select a state target as a column."""
    return Col(target)


def lit(value) -> Expr:
    """A constant."""
    return Lit(value)


def grad(target: str | Tensor) -> Expr:
    """Record per-step gradients of a target (eager-only, backward-time)."""
    return Grad(target)


def map_step(fn: Callable[..., Tensor], *cols: Expr) -> Expr:
    """``fn(a_t, b_t, ...)`` per step on ``[B, N]`` slices; streamable."""
    if not cols:
        raise ValueError("map_step requires at least one column expression")
    return MapStep(fn, tuple(cols))


def map_seq(fn: Callable[..., Tensor], *cols: Expr) -> Expr:
    """``fn(A, B, ...)`` once on stacked ``[T, B, N]`` columns;
    materialises."""
    if not cols:
        raise ValueError("map_seq requires at least one column expression")
    return MapSeq(fn, tuple(cols))
