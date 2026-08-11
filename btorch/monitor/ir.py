"""Typed dataflow IR for the recording engine.

All monitor expressions lower into one shared, hash-consed DAG (so a subexpression
used by several monitors is a single node -- ``Source`` dedup always pays).  Each
node is assigned a **temporal level** (three-class inference):

* ``static`` -- a constant (``lit``); time-invariant.
* ``step``   -- a per-timestep ``[B, N]`` value (sources, elementwise/window/map_step
  of step values).
* ``agg``    -- a time-reduced value (``Reduce`` output, or arithmetic among aggs).

The illegal case is a ``step`` node consuming an ``agg`` operand -- a *broadcast
back* (e.g. ``col("v") - col("v").mean()``) which needs the whole-sequence agg
before it can run per-step.  v1 rejects it loudly; two-pass materialisation is a
documented future extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import expr as E
from .frame import Resolver, TargetRef


STATIC, STEP, AGG = "static", "step", "agg"


@dataclass
class Node:
    nid: int
    op: str  # source|lit|grad|elementwise|window|reduce|mapstep|mapseq
    inputs: tuple[int, ...]
    params: dict
    level: str
    # roles (filled by the builder / lowering)
    is_stack_point: bool = False  # per-step output appended to a [T,...] buffer
    names: list[str] = field(default_factory=list)  # output column name(s) if a root


class Graph:
    """A hash-consed DAG of :class:`Node` plus the resolved roots."""

    def __init__(self):
        self.nodes: list[Node] = []
        self._intern: dict[tuple, int] = {}
        self.roots: list[int] = []  # nids that produce an output column
        self.grad_specs: list[tuple[str, TargetRef, str]] = []  # (key, ref, dotted)

    # -- interning --------------------------------------------------------
    def _add(self, op: str, inputs: tuple[int, ...], params: dict, level: str) -> int:
        key = (op, inputs, _param_key(params))
        if key in self._intern:
            return self._intern[key]
        nid = len(self.nodes)
        self.nodes.append(Node(nid, op, inputs, params, level))
        self._intern[key] = nid
        return nid

    def node(self, nid: int) -> Node:
        return self.nodes[nid]


def _param_key(params: dict):
    items = []
    for k, v in sorted(params.items()):
        if callable(v):
            items.append((k, id(v)))
        elif hasattr(v, "__hash__") and not isinstance(v, (list, dict)):
            try:
                hash(v)
                items.append((k, v))
            except TypeError:
                items.append((k, id(v)))
        else:
            items.append((k, id(v)))
    return tuple(items)


def build(specs, resolver: Resolver) -> Graph:
    """Parse a sequence of ``str | Expr`` specs into a shared
    :class:`Graph`."""
    g = Graph()
    seen_names: set[str] = set()

    def claim(name: str):
        if name in seen_names:
            raise ValueError(f"duplicate output/record key {name!r}")
        seen_names.add(name)

    for spec in specs:
        expr = E.col(spec) if isinstance(spec, str) else spec
        if not isinstance(expr, E.Expr):
            raise TypeError(f"record spec must be str or Expr, got {type(expr)}")

        name, inner = _strip_alias(expr)

        if isinstance(inner, E.Grad):
            _, dotted = resolver.resolve(inner.target)
            key = name or dotted
            claim(key)
            g.grad_specs.append((key, *resolver.resolve(inner.target)))
            continue

        nid = _emit(g, inner, resolver)
        node = g.node(nid)
        key = name or _default_key(g, inner, node)
        claim(key)
        node.names.append(key)
        if nid not in g.roots:
            g.roots.append(nid)
        _mark_stack_points(g, nid)

    # Global pass: every map_seq's inputs must be buffered, even when the map_seq
    # is nested under agg arithmetic (not a root) -- else finalize() KeyErrors.
    for node in g.nodes:
        if node.op == "mapseq":
            for i in node.inputs:
                g.node(i).is_stack_point = True

    return g


def _strip_alias(expr: E.Expr) -> tuple[str | None, E.Expr]:
    if isinstance(expr, E.Alias):
        if isinstance(expr.child, E.Grad) or not isinstance(expr.child, E.Alias):
            return expr.name, expr.child
        # nested alias: outermost wins
        _, inner = _strip_alias(expr.child)
        return expr.name, inner
    return None, expr


def _default_key(g: Graph, inner: E.Expr, node: Node) -> str:
    if node.op == "source":
        return node.params["name"]
    raise ValueError(
        "processed monitors (functions/reductions/multi-target) require an "
        "explicit name via .alias('name') or a {name: expr} mapping"
    )


def _emit(g: Graph, e: E.Expr, resolver: Resolver) -> int:
    """Recursively emit a node for ``e``, returning its nid.

    Enforces levels.
    """
    if isinstance(e, E.Alias):
        return _emit(g, e.child, resolver)

    if isinstance(e, E.Col):
        ref, name = resolver.resolve(e.target)
        return g._add("source", (), {"ref": int(ref), "name": name}, STEP)

    if isinstance(e, E.Lit):
        return g._add("lit", (), {"value": e.value}, STATIC)

    if isinstance(e, E.Grad):
        raise TypeError(E._GRAD_MSG)

    if isinstance(e, E.Elementwise):
        ins = tuple(_emit(g, a, resolver) for a in e.args)
        level = _combine_level(g, ins, ctx=f"op {e.op!r}")
        return g._add("elementwise", ins, {"op": e.op, **e.params}, level)

    if isinstance(e, E.MapStep):
        ins = tuple(_emit(g, a, resolver) for a in e.args)
        level = _combine_level(g, ins, ctx="map_step")
        if level == AGG:
            raise NotImplementedError(
                "map_step over already-reduced (agg) values is not supported; "
                "use map_seq for whole-sequence transforms."
            )
        return g._add("mapstep", ins, {"fn": e.fn}, STEP if level != STATIC else STATIC)

    if isinstance(e, E.Window):
        cid = _emit(g, e.child, resolver)
        if g.node(cid).level != STEP:
            raise NotImplementedError(
                f"{e.kind}() requires a per-step (step-level) input, got "
                f"{g.node(cid).level}"
            )
        return g._add("window", (cid,), {"kind": e.kind, "n": e.n}, STEP)

    if isinstance(e, E.Reduce):
        cid = _emit(g, e.child, resolver)
        clevel = g.node(cid).level
        if clevel == AGG:
            raise NotImplementedError(
                "nested reduction (reducing an already-reduced value) is not "
                "supported in v1."
            )
        return g._add("reduce", (cid,), {"kind": e.kind}, AGG)

    if isinstance(e, E.CustomReduce):
        cid = _emit(g, e.child, resolver)
        if g.node(cid).level != STEP:
            raise NotImplementedError(
                "fold() requires a per-step (step-level) input, got "
                f"{g.node(cid).level}"
            )
        return g._add("creduce", (cid,), {"reducer": e.reducer}, AGG)

    if isinstance(e, E.MapSeq):
        ins = tuple(_emit(g, a, resolver) for a in e.args)
        for i in ins:
            if g.node(i).level != STEP:
                raise NotImplementedError(
                    "map_seq inputs must be per-step columns/expressions (got a "
                    f"{g.node(i).level}-level input); it stacks each input to "
                    "[T, ...] and runs the fn once."
                )
        return g._add("mapseq", ins, {"fn": e.fn}, AGG)

    raise TypeError(f"unhandled expression node {type(e).__name__}")


def _combine_level(g: Graph, ins: tuple[int, ...], ctx: str) -> str:
    levels = {g.node(i).level for i in ins}
    if levels <= {STATIC}:
        return STATIC
    if AGG in levels and STEP in levels:
        raise NotImplementedError(
            f"broadcast-back not supported in v1 ({ctx}): an expression mixes a "
            "per-step value with a whole-sequence reduction (e.g. "
            "col('v') - col('v').mean()). Materialise and post-process instead."
        )
    if levels <= {STATIC, AGG}:
        return AGG
    return STEP


def _mark_stack_points(g: Graph, root: int) -> None:
    """A node is a stack-point if its per-step output must be buffered to
    [T,...].

    That is: a step-level root (raw / per-step-derived trace with no reduction),
    or a direct step-level input of a MapSeq.
    """
    node = g.node(root)
    if node.op == "mapseq":
        for i in node.inputs:
            if g.node(i).level == STEP:
                g.node(i).is_stack_point = True
    elif node.level == STEP:
        node.is_stack_point = True
    # agg roots (reductions / agg-arithmetic) need no buffer.
