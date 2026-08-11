"""Lower a :class:`~btorch.monitor.ir.Graph` to a per-step kernel + finalize.

The lowered program threads an **immutable tensor carry** (reduction accumulators,
window ring buffers, a global step counter) and, per step, appends *stack-point*
outputs to buffers the driver keeps outside any checkpoint (grad-checkpoint safe).
Each
step value is threaded with a scalar **validity** so that windowed-op warmup is
excluded from downstream reduction counts.  ``std``/``var`` use
Welford (numerically stable).  Node ids are already a valid topological order (children
are interned before parents).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .frame import StepFrame
from .ir import AGG, STEP, Graph, Node


def _apply_ew(op: str, args: list, params: dict):
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "div":
        return args[0] / args[1]
    if op == "neg":
        return -args[0]
    if op == "abs":
        return torch.abs(args[0])
    if op == "relu":
        return torch.relu(args[0])
    if op == "clamp":
        return torch.clamp(args[0], min=params.get("min"), max=params.get("max"))
    if op == "pow":
        return torch.pow(args[0], params["exponent"])
    if op == "getitem":
        return args[0][params["index"]]
    raise ValueError(f"unknown elementwise op {op!r}")


@dataclass
class _CarrySlot:
    """Shape/dtype/device/fill descriptor for one carry entry (see
    init_carry)."""

    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: object
    fill: float = 0.0


@dataclass
class _CustomReduce:
    """Runtime info for a ``fold(reducer)`` node (see init_carry / step /
    finalize)."""

    nid: int
    reducer: object  # btorch.monitor.reducer.Reducer
    child: int
    shape: tuple[int, ...]  # child value shape (input to reducer.init)
    dtype: torch.dtype
    device: object
    treespec: object  # pytree structure of the carry, for (un)flatten
    n_leaves: int


@dataclass
class CompiledProgram:
    graph: Graph
    step_nids: tuple[int, ...]  # nodes evaluated/folded per step (topo order)
    stack_nids: tuple[int, ...]  # nodes whose per-step output is buffered
    reduce_nids: tuple[int, ...]
    creduce_nids: tuple[int, ...] = ()  # custom fold(reducer) nodes
    # cached carry layout (built once via meta tensors); see init_carry
    _carry_slots: list[_CarrySlot] | None = None
    _creduce: dict | None = None  # nid -> _CustomReduce

    # ------------------------------------------------------------------ carry
    def init_carry(self, example_frame: StepFrame) -> dict:
        """A fresh zero carry.

        The carry layout (shapes/dtypes/devices) is inferred **once**
        via meta tensors (no allocation) and cached; subsequent calls
        just allocate the tensors, so the hot path never re-runs the
        inference.
        """
        from torch.utils._pytree import tree_flatten

        if self._carry_slots is None:
            self._carry_slots, self._creduce = self._infer_carry_slots(example_frame)
        carry = {
            s.key: (
                torch.zeros(s.shape, dtype=s.dtype, device=s.device)
                if s.fill == 0.0
                else torch.full(s.shape, s.fill, dtype=s.dtype, device=s.device)
            )
            for s in self._carry_slots
        }
        # Custom reducers: run reducer.init on a real zero example to get the
        # initial carry (values matter, not just shapes) -- pure ops, so this is
        # torch.compile / cudagraph-capturable.
        for cr in self._creduce.values():
            example = torch.zeros(cr.shape, dtype=cr.dtype, device=cr.device)
            leaves, _ = tree_flatten(cr.reducer.init(example))
            for i, leaf in enumerate(leaves):
                carry[f"{cr.nid}:c{i}"] = leaf
        return carry

    def _infer_carry_slots(self, example_frame: StepFrame):
        """Infer the carry layout from example source shapes, without
        allocating.

        Uses **meta** tensors (shape/dtype only, no storage) rather than
        FakeTensorMode -- meta avoids allocation just as well but does not trip
        inductor's CPU codegen when this runs inside a compiled RNN frame.
        """
        from torch.utils._pytree import tree_flatten

        g = self.graph
        creduce: dict[int, _CustomReduce] = {}
        mval: dict[int, object] = {}  # meta tensors (or lit values)
        ref_dev = ref_dt = None
        for nid in self.step_nids:
            node = g.node(nid)
            if node.op == "source":
                t = example_frame[node.params["ref"]]
                ref_dev, ref_dt = t.device, t.dtype  # the real device/dtype
                mval[nid] = t.detach().to("meta")
            elif node.op == "lit":
                mval[nid] = node.params["value"]
            elif node.op in ("elementwise", "mapstep"):
                args = [mval[i] for i in node.inputs]
                if node.op == "elementwise":
                    mval[nid] = _apply_ew(node.params["op"], args, node.params)
                else:
                    mval[nid] = node.params["fn"](*args)
            elif node.op == "window":
                mval[nid] = mval[node.inputs[0]]  # same shape as child
            elif node.op == "reduce":
                mval[nid] = mval[node.inputs[0]]
            elif node.op == "creduce":
                child = node.inputs[0]
                fv = mval[child]
                reducer = node.params["reducer"]
                leaves, treespec = tree_flatten(reducer.init(fv))
                creduce[nid] = _CustomReduce(
                    nid,
                    reducer,
                    child,
                    tuple(fv.shape),
                    fv.dtype,
                    ref_dev,
                    treespec,
                    len(leaves),
                )

        slots: list[_CarrySlot] = []

        def val_slot(key, nid, fill=0.0):
            fv = mval[nid]
            slots.append(_CarrySlot(key, tuple(fv.shape), fv.dtype, ref_dev, fill))

        def scalar_slot(key, dtype=None):
            slots.append(_CarrySlot(key, (), dtype or ref_dt, ref_dev, 0.0))

        slots.append(_CarrySlot("__step__", (), torch.long, ref_dev, 0.0))
        for nid in self.reduce_nids:
            kind = g.node(nid).params["kind"]
            cid = g.node(nid).inputs[0]
            if kind in ("mean", "sum"):
                val_slot(f"{nid}:sum", cid)
                if kind == "mean":
                    scalar_slot(f"{nid}:cnt")
            elif kind == "count":
                scalar_slot(f"{nid}:cnt")
            elif kind == "last":
                val_slot(f"{nid}:val", cid)
            elif kind == "min":
                val_slot(f"{nid}:val", cid, fill=float("inf"))
            elif kind == "max":
                val_slot(f"{nid}:val", cid, fill=float("-inf"))
            elif kind == "first":
                val_slot(f"{nid}:val", cid)
                scalar_slot(f"{nid}:has")
            elif kind in ("std", "var"):
                scalar_slot(f"{nid}:cnt")
                val_slot(f"{nid}:mean", cid)
                val_slot(f"{nid}:m2", cid)
        for nid in self.step_nids:
            node = g.node(nid)
            if node.op == "window":
                n = node.params["n"]
                fv = mval[node.inputs[0]]
                # value ring [n, *value] and a validity ring [n] so a window-of-
                # window knows its predecessor was itself valid (window warmup).
                slots.append(
                    _CarrySlot(f"{nid}:ring", (n, *fv.shape), fv.dtype, ref_dev)
                )
                slots.append(_CarrySlot(f"{nid}:vring", (n,), ref_dt, ref_dev))
        return slots, creduce

    # ------------------------------------------------------------------- step
    def step_kernel(self, carry: dict, frame: StepFrame) -> tuple[dict, dict]:
        g = self.graph
        carry = dict(carry)
        step = carry["__step__"]
        val: dict[int, object] = {}
        vld: dict[int, object] = {}  # validity: float 1.0 or a scalar tensor
        stack: dict[int, Tensor] = {}

        for nid in self.step_nids:
            node = g.node(nid)
            op = node.op
            if op == "source":
                val[nid] = frame[node.params["ref"]]
                vld[nid] = 1.0
            elif op == "lit":
                val[nid] = node.params["value"]
                vld[nid] = 1.0
            elif op in ("elementwise", "mapstep"):
                args = [val[i] for i in node.inputs]
                if op == "elementwise":
                    val[nid] = _apply_ew(node.params["op"], args, node.params)
                else:
                    val[nid] = node.params["fn"](*args)
                vld[nid] = _vprod([vld[i] for i in node.inputs])
            elif op == "window":
                cid = node.inputs[0]
                n = node.params["n"]
                x = val[cid]
                # ring holds the last n values in order [x_{t-n}, ..., x_{t-1}].
                # Static-index roll (no data-dependent index) -> torch.compile-safe.
                ring = carry[f"{nid}:ring"]
                vring = carry[f"{nid}:vring"]
                prev = ring[0]  # x_{t-n} (a zero during warmup, masked below)
                prev_valid = vring[0]  # child validity at t-n
                cur_valid = vld[cid]  # child validity now
                valid_step = (step >= n).to(x.dtype)
                if node.params["kind"] == "diff":
                    # valid iff both endpoints valid and the ring has filled
                    out_valid = _vmul(valid_step * prev_valid, cur_valid)
                    out = (x - prev) * out_valid
                else:  # shift: outputs x_{t-n}; validity is that of the predecessor
                    out_valid = valid_step * prev_valid
                    out = prev * out_valid
                wt = (
                    cur_valid
                    if isinstance(cur_valid, Tensor)
                    else torch.ones_like(valid_step)
                )
                carry[f"{nid}:ring"] = torch.cat([ring[1:], x.unsqueeze(0)], dim=0)
                carry[f"{nid}:vring"] = torch.cat([vring[1:], wt.unsqueeze(0)], dim=0)
                val[nid] = out
                vld[nid] = out_valid
            elif op == "reduce":
                self._fold(carry, node, val[node.inputs[0]], vld[node.inputs[0]])
            elif op == "creduce":
                self._fold_custom(carry, nid, val[node.inputs[0]], vld[node.inputs[0]])
            if node.is_stack_point:
                stack[nid] = val[nid]

        carry["__step__"] = step + 1
        return carry, stack

    def _fold(self, carry: dict, node: Node, x, w) -> None:
        nid = node.nid
        kind = node.params["kind"]
        if kind in ("mean", "sum"):
            carry[f"{nid}:sum"] = carry[f"{nid}:sum"] + x * w
            if kind == "mean":
                carry[f"{nid}:cnt"] = carry[f"{nid}:cnt"] + w
        elif kind == "count":
            carry[f"{nid}:cnt"] = carry[f"{nid}:cnt"] + w
        elif kind == "last":
            old = carry[f"{nid}:val"]
            carry[f"{nid}:val"] = old + (x - old) * _wpos(w)
        elif kind == "first":
            take = _wpos(w) * (1 - carry[f"{nid}:has"])
            old = carry[f"{nid}:val"]
            carry[f"{nid}:val"] = old + (x - old) * take
            carry[f"{nid}:has"] = carry[f"{nid}:has"] + take
        elif kind in ("min", "max"):
            old = carry[f"{nid}:val"]
            if isinstance(w, Tensor):
                big = float("inf") if kind == "min" else float("-inf")
                cand = torch.where(w > 0, x, torch.full_like(x, big))
            else:
                cand = x
            carry[f"{nid}:val"] = (
                torch.minimum(old, cand) if kind == "min" else torch.maximum(old, cand)
            )
        elif kind in ("std", "var"):
            cnt = carry[f"{nid}:cnt"] + w
            mean = carry[f"{nid}:mean"]
            delta = x - mean
            mean2 = mean + _wmul(w, delta) / torch.clamp(cnt, min=1)
            carry[f"{nid}:m2"] = carry[f"{nid}:m2"] + _wmul(w, delta) * (x - mean2)
            carry[f"{nid}:mean"] = mean2
            carry[f"{nid}:cnt"] = cnt

    def _fold_custom(self, carry: dict, nid: int, x, w) -> None:
        from torch.utils._pytree import tree_flatten, tree_unflatten

        cr = self._creduce[nid]
        leaves = [carry[f"{nid}:c{i}"] for i in range(cr.n_leaves)]
        state = cr.reducer.update(tree_unflatten(leaves, cr.treespec), x, w)
        new_leaves, treespec = tree_flatten(state)
        # The carry structure must be constant across steps (it was fixed by init).
        # This check is data-independent (static), so torch.compile folds it away.
        if treespec != cr.treespec:
            raise ValueError(
                f"Reducer.update returned a carry with structure {treespec} but "
                f"init produced {cr.treespec}; the carry pytree must be constant "
                "across steps."
            )
        for i, leaf in enumerate(new_leaves):
            carry[f"{nid}:c{i}"] = leaf

    # --------------------------------------------------------------- finalize
    def finalize(self, carry: dict, buffers: dict[int, list]) -> dict[str, Tensor]:
        g = self.graph
        out: dict[str, Tensor] = {}
        cache: dict[int, object] = {}

        def stacked(nid):
            return torch.stack(buffers[nid], dim=0)

        def ev(nid):
            if nid in cache:
                return cache[nid]
            node = g.node(nid)
            if node.op == "reduce":
                r = self._reduce_finalize(node, carry)
            elif node.op == "creduce":
                from torch.utils._pytree import tree_unflatten

                cr = self._creduce[nid]
                leaves = [carry[f"{nid}:c{i}"] for i in range(cr.n_leaves)]
                r = cr.reducer.finalize(tree_unflatten(leaves, cr.treespec))
                if not isinstance(r, Tensor):
                    raise TypeError(
                        f"Reducer.finalize must return a Tensor, got {type(r)}"
                    )
                r = r.detach()
            elif node.op == "lit":
                r = node.params["value"]
            elif node.op == "elementwise" and node.level == AGG:
                r = _apply_ew(
                    node.params["op"], [ev(i) for i in node.inputs], node.params
                )
            elif node.op == "mapseq":
                seqs = [stacked(i) for i in node.inputs]
                r = node.params["fn"](*seqs).detach()
            elif node.level == STEP:  # materialised step-level root (raw / derived)
                r = stacked(nid)
                if node.op != "source":  # raw source stays grad-connected (parity)
                    r = r.detach()
            else:
                raise RuntimeError(f"cannot finalize node op={node.op}")
            cache[nid] = r
            return r

        for nid in g.roots:
            v = ev(nid)
            for name in g.node(nid).names:
                out[name] = v
        return out

    def _reduce_finalize(self, node: Node, carry: dict) -> Tensor:
        nid = node.nid
        kind = node.params["kind"]
        if kind == "sum":
            r = carry[f"{nid}:sum"]
        elif kind == "mean":
            r = carry[f"{nid}:sum"] / torch.clamp(carry[f"{nid}:cnt"], min=1)
        elif kind == "count":
            r = carry[f"{nid}:cnt"]
        elif kind in ("last", "first", "min", "max"):
            r = carry[f"{nid}:val"]
        elif kind == "var":
            r = carry[f"{nid}:m2"] / torch.clamp(carry[f"{nid}:cnt"], min=1)
        elif kind == "std":
            var = carry[f"{nid}:m2"] / torch.clamp(carry[f"{nid}:cnt"], min=1)
            r = torch.sqrt(torch.clamp(var, min=0))
        else:
            raise ValueError(f"unknown reduce {kind!r}")
        return r.detach()


# -- validity helpers (float 1.0 or scalar tensor) --------------------------
def _vprod(ws: list):
    acc = 1.0
    for w in ws:
        acc = _vmul(acc, w)
    return acc


def _vmul(a, b):
    if isinstance(a, float) and a == 1.0:
        return b
    if isinstance(b, float) and b == 1.0:
        return a
    return a * b


def _wpos(w):
    # w is already 0/1 (valid_step) or float 1.0
    return w


def _wmul(w, x):
    if isinstance(w, float) and w == 1.0:
        return x
    return w * x


def lower(graph: Graph) -> CompiledProgram:
    """Partition the graph into the per-step region and roles."""
    step_nids = []
    stack_nids = []
    reduce_nids = []
    creduce_nids = []
    for nid, node in enumerate(graph.nodes):
        if node.op == "mapseq":
            continue
        if node.op == "reduce":
            step_nids.append(nid)
            reduce_nids.append(nid)
            continue
        if node.op == "creduce":
            step_nids.append(nid)
            creduce_nids.append(nid)
            continue
        if node.op == "elementwise" and node.level == AGG:
            continue  # agg-arithmetic runs in finalize
        # source / lit / step-elementwise / mapstep / window
        step_nids.append(nid)
        if node.is_stack_point:
            stack_nids.append(nid)
    return CompiledProgram(
        graph=graph,
        step_nids=tuple(step_nids),
        stack_nids=tuple(stack_nids),
        reduce_nids=tuple(reduce_nids),
        creduce_nids=tuple(creduce_nids),
    )
