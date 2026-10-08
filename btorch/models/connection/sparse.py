"""Explicit sparse connection."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor, nn

from ...sparse import COO, CSR, Hints, Sparse, as_sparse
from ...sparse.coo import _ravel
from ...sparse.runtime import RepresentationCache, kernels_aten, ops, planner, registry
from .base import Connection
from .synapse import Synapse
from .weight import Weight


Orientation = Literal["src_dst", "dst_src"]


class SparseConnection(Connection):
    """Connection with explicitly stored edges.

    ``current = conn(spikes)`` maps ``spikes [..., in_features]`` to
    ``current [..., out_features]``; any number of leading (time, batch)
    dimensions is supported.

    **Orientation.** The constructor takes the standard linear-algebra
    operator ``A`` with ``A.shape == (n_post, n_pre)``, i.e. ``y = A @ x``.
    Connectome matrices are usually stored the other way round (rows are
    sources, columns are targets, ``y = x @ W``): pass those through
    :meth:`from_adjacency`, which is the only place a transpose happens.

    **Model state.** The ``state_dict`` holds the canonical edge list
    (``indices`` = ``[post, pre]`` per edge slot, plus ``receptor`` / ``delay``
    when used) and the weight module. Compressed layouts used for execution
    are derived, non-persistent, and rebuilt whenever the edges change
    (loading a checkpoint, rewiring), so they can never go stale.

    **Receptors and delays** are edge attributes. With ``n_receptor`` receptor
    channels the output is ``[..., n_post * n_receptor]`` (index
    ``post * n_receptor + receptor``); with ``n_delay`` delay bins the input is
    ``[..., n_pre * n_delay]`` (index ``pre * n_delay + delay``, the layout
    produced by :class:`~btorch.models.history.SpikeHistory`). This expanded
    layout is one lowering of the semantic edge list, not the model itself.

    For a batch of different patterns (:func:`btorch.sparse.stack`) the
    stored ``indices`` are those of the block-diagonal operator
    (``member * n + id``); :meth:`edge_table` and :meth:`to_sparse` report
    network coordinates.

    **Network batch.** A matrix with a shared-pattern batch (or a weight
    module returning ``[G, n_edge]``) defines ``G`` networks; the input is
    then ``[G, ..., in_features]`` and a leading dimension of size 1
    broadcasts. Networks are never combined with samples implicitly.

    Args:
        matrix: Operator ``(n_post, n_pre)`` as a btorch
            :class:`~btorch.sparse.Sparse`, a PyTorch sparse tensor or a SciPy
            sparse array. Duplicate entries are summed.
        synapse: Weight / receptor / delay description. By default the matrix
            values become trainable per-edge weights.
        bias: Optional ``[out_features]`` bias (becomes a parameter).
        hints: Performance expectations for execution planning.
        device: Target device.
        dtype: Weight dtype. Defaults to the matrix dtype for btorch and
            PyTorch input, and to the PyTorch default dtype for SciPy input
            and for integer matrices.

    Examples:
        >>> conn = SparseConnection(A)                         # doctest: +SKIP
        >>> conn = SparseConnection.from_adjacency(W, Synapse(dale=True))
        >>> compiled = torch.compile(conn, fullgraph=True)
        >>> current = compiled(spikes)
    """

    indices: Tensor
    receptor: Tensor | None
    delay: Tensor | None
    bias: nn.Parameter | None

    def __init__(
        self,
        matrix: Any,
        synapse: Synapse | None = None,
        *,
        bias: Tensor | None = None,
        hints: Hints | None = None,
        device=None,
        dtype: torch.dtype | None = None,
        _adjacency: Sparse | None = None,
    ):
        super().__init__()
        synapse = synapse or Synapse()
        if dtype is None and not isinstance(matrix, (Sparse, Tensor)):
            # SciPy data is float64 by NumPy convention, not by choice.
            dtype = torch.get_default_dtype()
        A = as_sparse(matrix, device=device)
        if A.sparse_dim() != 2 or A.dense_dim():
            raise ValueError(
                "A connection matrix needs exactly two sparse dimensions and "
                f"scalar entries, got sparse_shape={A.sparse_shape}, "
                f"dense_shape={A.dense_shape}."
            )
        coo = A.tocoo()
        adjacency = coo if _adjacency is None else _adjacency.tocoo()
        values = coo.values()
        if dtype is not None:
            values = values.to(dtype)
        elif not values.is_floating_point():
            values = values.to(torch.get_default_dtype())
        n_post, n_pre = A.sparse_shape
        self.n_pre, self.n_post = int(n_pre), int(n_post)

        # A batch of different patterns is executed as one block-diagonal
        # operator over the concatenated populations (reference fallback).
        lead, post, pre = coo._edge_index()
        self._member_shape: tuple[int, ...] = ()
        if lead is not None:
            if synapse.receptor is not None or synapse.delay is not None:
                raise NotImplementedError(
                    "Receptor/delay routing for a batch of different sparsity "
                    "patterns is not implemented."
                )
            self._member_shape = tuple(coo.batch_shape[coo._value_batch_dim :])
            member = _ravel(lead, self._member_shape)
            post = member * n_post + post
            pre = member * n_pre + pre
        n_member = math.prod(self._member_shape)

        # Semantic routing -> lowered (expanded) operator coordinates.
        receptor, n_receptor, delay, n_delay = synapse.routing(coo.nnz, post.device)
        self.n_receptor, self.n_delay = n_receptor, n_delay
        self._lowered_shape = (
            n_member * self.n_post * n_receptor,
            n_member * self.n_pre * n_delay,
        )
        lowered = COO(
            torch.stack([post * n_receptor + receptor, pre * n_delay + delay]),
            values,
            (*values.shape[:-1], *self._lowered_shape),
            properties=coo.properties
            if n_receptor == n_delay == 1 and lead is None
            else None,
            check=False,
        )
        # One map moves weights and every piece of edge metadata together.
        canonical, edge_map = lowered.coalesce(return_map=True)
        row, col = canonical.row, canonical.col
        self.register_buffer(
            "indices", torch.stack([row // n_receptor, col // n_delay])
        )
        self.register_buffer(
            "receptor", row % n_receptor if synapse.receptor is not None else None
        )
        self.register_buffer(
            "delay", col % n_delay if synapse.delay is not None else None
        )

        self.weight: Weight = synapse.make_weight()
        self.weight.bind(values, edge_map, adjacency)
        self.weight.to(device=row.device)
        self.plasticity = synapse.plasticity
        self.bias = None if bias is None else nn.Parameter(torch.as_tensor(bias))

        # Version counters: which derived state a change invalidates.
        self.topology_version = 0
        self.routing_version = 0
        self.value_version = 0
        self._always_permute = False
        self.cache = RepresentationCache()
        self._rebuild()

        self._value_batch_dim = self.weight().ndim - 1
        self.hints = hints or A.hints
        self._plan = planner.plan(
            device=row.device.type,
            value_batched=self._value_batch_dim > 0,
            hints=self.hints,
        )
        self.register_load_state_dict_post_hook(_after_load)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    # ------------------------------------------------------------ builders
    @classmethod
    def from_adjacency(
        cls,
        matrix: Any,
        synapse: Synapse | None = None,
        *,
        orientation: Orientation = "src_dst",
        **kwargs,
    ) -> "SparseConnection":
        """Build from an adjacency / connectivity matrix.

        Args:
            matrix: Sparse matrix (btorch, PyTorch or SciPy).
            synapse: Synapse description; per-edge arrays are aligned with
                the stored entries of ``matrix``, and matrices inside it (e.g.
                a group-id matrix) use the same orientation as ``matrix``.
            orientation: ``"src_dst"`` (default): rows are sources, columns
                are targets, the connection computes ``x @ matrix``.
                ``"dst_src"``: the matrix already is the operator
                ``(n_post, n_pre)``.
            **kwargs: Passed to the constructor.
        """
        A = _as_sparse_default_dtype(matrix, kwargs)
        if orientation == "src_dst":
            return cls(A.transpose(), synapse, _adjacency=A, **kwargs)
        if orientation == "dst_src":
            return cls(A, synapse, **kwargs)
        raise ValueError(
            f"orientation must be 'src_dst' or 'dst_src', got {orientation!r}."
        )

    @classmethod
    def from_edges(
        cls,
        pre: Tensor,
        post: Tensor,
        n_pre: int,
        n_post: int,
        synapse: Synapse | None = None,
        *,
        values: Tensor | None = None,
        **kwargs,
    ) -> "SparseConnection":
        """Build from an edge list.

        Args:
            pre: ``[n_edge]`` source neuron of each edge.
            post: ``[n_edge]`` target neuron of each edge.
            n_pre: Number of source neurons.
            n_post: Number of target neurons.
            synapse: Synapse description aligned with the edge list.
            values: Optional ``[*batch, n_edge]`` initial weights (default 1).
            **kwargs: Passed to the constructor.
        """
        pre = torch.as_tensor(pre)
        post = torch.as_tensor(post, device=pre.device)
        if values is None:
            values = torch.ones(pre.shape[0], device=pre.device)
        coo = COO(torch.stack([post, pre]), values, (*values.shape[:-1], n_post, n_pre))
        return cls(coo, synapse, **kwargs)

    @classmethod
    def from_hetersynapse(
        cls,
        matrix: Any,
        synapse: Synapse | None = None,
        *,
        n_receptor: int = 1,
        n_delay: int = 1,
        **kwargs,
    ) -> "SparseConnection":
        """Build from a physically expanded hetersynapse matrix.

        Adapter for the output of
        :func:`btorch.connectome.connection.make_hetersynapse_conn` (and
        ``expand_conn_for_delays``): a ``src_dst`` matrix whose columns are
        ``post * n_receptor + receptor`` and whose rows are
        ``pre * n_delay + delay``. The expansion is decoded into per-edge
        receptor and delay attributes, so the connection stores the semantic
        edge list instead of the expanded matrix.

        Args:
            matrix: Expanded matrix of shape
                ``(n_pre * n_delay, n_post * n_receptor)``.
            synapse: Optional weight description (its ``receptor`` / ``delay``
                must be unset; they are decoded from ``matrix``). Matrices
                inside it use the same expanded layout.
            n_receptor: Number of receptor channels in the column expansion.
            n_delay: Number of delay bins in the row expansion.
            **kwargs: Passed to the constructor.
        """
        synapse = synapse or Synapse()
        if synapse.receptor is not None or synapse.delay is not None:
            raise ValueError("receptor/delay are decoded from the expanded matrix.")
        A = _as_sparse_default_dtype(matrix, kwargs)
        coo = A.tocoo()
        n_rows, n_cols = coo.sparse_shape
        if n_rows % n_delay or n_cols % n_receptor:
            raise ValueError(
                f"Matrix shape {coo.sparse_shape} is not divisible by "
                f"(n_delay={n_delay}, n_receptor={n_receptor})."
            )
        decoded = Synapse(
            weight=synapse.weight,
            delay=coo.row % n_delay if n_delay > 1 else None,
            receptor=coo.col % n_receptor if n_receptor > 1 else None,
            n_delay=n_delay if n_delay > 1 else None,
            n_receptor=n_receptor if n_receptor > 1 else None,
            dale=synapse.dale,
            plasticity=synapse.plasticity,
        )
        operator = COO(
            torch.stack([coo.col // n_receptor, coo.row // n_delay]),
            coo.values(),
            (*coo.values().shape[:-1], n_cols // n_receptor, n_rows // n_delay),
            check=False,
        )
        return cls(operator, decoded, _adjacency=A, **kwargs)

    # ---------------------------------------------------------- properties
    @property
    def in_features(self) -> int:
        return self.n_pre * self.n_delay

    @property
    def out_features(self) -> int:
        return self.n_post * self.n_receptor

    @property
    def nnz(self) -> int:
        """Number of edge slots (all networks together for a ragged batch)."""
        return int(self.indices.shape[1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Network-batch shape (empty for a single network)."""
        return tuple(self.weight().shape[:-1]) + self._member_shape

    # ---------------------------------------------------------- execution
    def forward(self, x: Tensor | Sparse) -> Tensor:
        """Propagate activity through the connection.

        Args:
            x: ``[*network_batch, ..., in_features]`` dense activity. For
                surrogate-gradient training this is the ordinary spike tensor;
                sparse execution is derived from it internally. Externally
                supplied events may instead be passed as a sparse
                :class:`~btorch.sparse.Sparse` array of shape
                ``[B, in_features]`` (see :meth:`propagate_events`).

        Returns:
            ``[*network_batch, ..., out_features]``.
        """
        if isinstance(x, Sparse):
            return self.propagate_events(x)
        n_in = self.in_features
        if x.shape[-1] != n_in:
            raise ValueError(
                f"Expected input with last dimension {n_in}, got {tuple(x.shape)}."
            )
        n_batch = self._value_batch_dim + len(self._member_shape)
        if n_batch and (
            x.ndim - 1 < n_batch
            or any(
                b != 1 and g != 1 and b != g
                for g, b in zip(self.batch_shape, x.shape[:n_batch])
            )
        ):
            raise ValueError(
                f"This connection holds a batch of networks {self.batch_shape}; "
                f"the input needs shape [*network_batch, ..., {n_in}], got "
                f"{tuple(x.shape)}. Networks are not combined with samples "
                "implicitly: use x.unsqueeze(0) to share samples across networks."
            )
        if self._member_shape:
            x = self._fold(x)

        cache = self.cache
        weight = self.weight()
        if self._always_permute or not cache.identity_perm:
            weight = weight.index_select(-1, cache.perm)
        if self._plan.algorithm == "adaptive-push":
            out = ops.spike_propagate(
                cache.crow,
                cache.col,
                weight,
                x,
                cache.t_crow,
                cache.t_col,
                cache.t_perm,
                self._plan.max_density,
            )
        else:
            out = ops.csr_propagate(
                cache.crow,
                cache.col,
                weight,
                x,
                cache.t_crow,
                cache.t_col,
                cache.t_perm,
            )
        if self._member_shape:
            out = self._unfold(out)
        if self.bias is not None:
            out = out + self.bias
        return out

    def propagate_events(self, events: Sparse) -> Tensor:
        """Propagate genuinely sparse input events.

        For input that is sparse by nature (event-camera data, precomputed
        spike trains) rather than produced by a differentiable neuron model.
        Only the out-edges of the listed events are visited and the input is
        never densified. Gradients flow to the weights and to the event
        values, not to absent events; for surrogate-gradient training pass
        the dense spike tensor to :meth:`forward` instead.

        Args:
            events: Sparse array of shape ``[B, in_features]`` (or
                ``[in_features]``) whose stored entries are the events.

        Returns:
            Dense ``[B, out_features]`` (or ``[out_features]``).
        """
        if self._value_batch_dim or self._member_shape:
            raise NotImplementedError("Sparse event input for a network batch.")
        coo = events.tocoo()
        if events.batch_dim() or events.dense_dim() or events.sparse_dim() > 2:
            raise ValueError("events must be a plain [B, N] or [N] sparse array.")
        if coo.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected events over {self.in_features} inputs, got {coo.shape}."
            )
        idx = coo.indices()
        vector = idx.shape[0] == 1
        n_sample = 1 if vector else coo.shape[0]
        sample = torch.zeros_like(idx[0]) if vector else idx[0]
        order = torch.argsort(sample, stable=True)
        counts = torch.bincount(sample, minlength=n_sample)
        ptr = torch.zeros(n_sample + 1, dtype=torch.long, device=idx.device)
        ptr[1:] = torch.cumsum(counts, 0)
        cache = self.cache
        weight = self.weight().index_select(-1, cache.perm)
        out = kernels_aten.spike_push_values(
            cache.t_crow,
            cache.t_col,
            cache.t_perm,
            weight,
            coo.values()[order],
            idx[-1][order],
            ptr,
            self.out_features,
        )
        if self.bias is not None:
            out = out + self.bias
        return out[0] if vector else out

    def _fold(self, x: Tensor) -> Tensor:
        """``[*members, *sample, n] -> [*sample, n_member * n]``."""
        n_m = len(self._member_shape)
        x = x.expand(*self._member_shape, *x.shape[n_m:])
        order = [*range(n_m, x.ndim - 1), *range(n_m), x.ndim - 1]
        x = x.permute(*order)
        return x.reshape(*x.shape[: x.ndim - n_m - 1], -1)

    def _unfold(self, y: Tensor) -> Tensor:
        """``[*sample, n_member * n] -> [*members, *sample, n]``."""
        n_m = len(self._member_shape)
        y = y.reshape(*y.shape[:-1], *self._member_shape, -1)
        n_s = y.ndim - n_m - 1
        return y.permute(*range(n_s, n_s + n_m), *range(n_s), y.ndim - 1)

    # ------------------------------------------------------------ topology
    def _lowered_edges(self) -> tuple[Tensor, Tensor]:
        row, col = self.indices[0], self.indices[1]
        if self.n_receptor > 1:
            row = row * self.n_receptor
            if self.receptor is not None:
                row = row + self.receptor
        if self.n_delay > 1:
            col = col * self.n_delay
            if self.delay is not None:
                col = col + self.delay
        return row, col

    def _rebuild(self) -> None:
        """Rebuild the derived execution layouts from the canonical edges."""
        row, col = self._lowered_edges()
        self.cache.build(row, col, self._lowered_shape, self.topology_version)

    @torch.no_grad()
    def set_edges_(
        self,
        slots: Tensor,
        post: Tensor,
        pre: Tensor,
        receptor: Tensor | None = None,
        delay: Tensor | None = None,
    ) -> None:
        """Rewire edge slots in place (structural plasticity).

        Tensor shapes do not change: slot ``k`` simply becomes a different
        edge. Derived execution layouts are rebuilt into their existing
        tensors and ``topology_version`` is incremented. Call this outside
        the forward pass (for example from an optimizer hook).

        Args:
            slots: ``[K]`` edge slots to rewire.
            post: ``[K]`` new target neuron of each slot.
            pre: ``[K]`` new source neuron of each slot.
            receptor: Optional ``[K]`` new receptor ids.
            delay: Optional ``[K]`` new delays.
        """
        if self._member_shape:
            raise NotImplementedError("Rewiring a batch of different patterns.")
        if slots.numel() == 0:
            return
        if (
            int(post.min()) < 0
            or int(post.max()) >= self.n_post
            or int(pre.min()) < 0
            or int(pre.max()) >= self.n_pre
        ):
            raise ValueError("Rewired edges must lie within the populations.")
        if not self._always_permute and not torch.compiler.is_compiling():
            # From now on slot order differs from execution order.
            self._always_permute = True
        self.indices[0, slots] = post.to(self.indices)
        self.indices[1, slots] = pre.to(self.indices)
        if receptor is not None:
            self.receptor[slots] = receptor.to(self.receptor)
            self.routing_version += 1
        if delay is not None:
            self.delay[slots] = delay.to(self.delay)
            self.routing_version += 1
        self.topology_version += 1
        self._rebuild()

    def constrain(self) -> None:
        """Project the weights onto their constraints (e.g. Dale's law).

        :func:`btorch.models.constrain.constrain_net` reaches the weight
        module on its own; this is a convenience for a single connection.
        """
        self.weight.constrain()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Refuse edges that do not fit this module *before* anything is
        # copied: population sizes and routing sizes are not in the
        # checkpoint, so out-of-range ids would silently address other
        # neurons or channels.
        error_msgs = args[4]
        limits = {
            "receptor": (self.n_receptor, None),
            "delay": (self.n_delay, None),
            "indices": (
                self._lowered_shape[0] // self.n_receptor,
                self._lowered_shape[1] // self.n_delay,
            ),
        }
        for name, (first, second) in limits.items():
            value = state_dict.get(prefix + name)
            if value is None or value.numel() == 0 or value.is_floating_point():
                continue
            bad = int(value.min()) < 0
            if second is None:
                bad = bad or int(value.max()) >= first
            elif value.ndim == 2 and value.shape[0] == 2:
                bad = (
                    bad or int(value[0].max()) >= first or int(value[1].max()) >= second
                )
            if bad:
                error_msgs.append(
                    f"'{prefix}{name}' in the checkpoint addresses neurons or "
                    f"channels outside this connection (n_pre={self.n_pre}, "
                    f"n_post={self.n_post}, n_receptor={self.n_receptor}, "
                    f"n_delay={self.n_delay})."
                )
                state_dict = {k: v for k, v in state_dict.items() if k != prefix + name}
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def enable_rewiring(self) -> None:
        """Declare that edges will be rewired.

        Makes the forward pass independent of whether the slots are currently
        in execution order, so later rewiring never changes the traced graph.
        Call before ``torch.compile``.
        """
        self._always_permute = True

    # -------------------------------------------------------- introspection
    def to_sparse(self, orientation: Orientation = "dst_src") -> Sparse:
        """Current effective operator as a :class:`~btorch.sparse.Sparse`.

        Args:
            orientation: ``"dst_src"`` returns the operator
                ``(out_features, in_features)`` as CSR; ``"src_dst"`` its
                transpose (the connectome convention).

        Returns:
            Sparse array whose values are the effective weights (gradients
            flow to the weight parameters).
        """
        weight = self.weight().index_select(-1, self.cache.perm)
        if self._member_shape:
            # Undo the block-diagonal lowering: report network coordinates.
            member, post, pre = self._member_coordinates(
                self.cache.crow_rows(), self.cache.col
            )
            out = COO(
                torch.cat([member, post[None], pre[None]]),
                weight,
                (*weight.shape[:-1], *self._member_shape, self.n_post, self.n_pre),
                batch_dim=weight.ndim - 1 + len(self._member_shape),
                check=False,
            )
            return out.transpose() if orientation == "src_dst" else out
        csr = CSR(
            self.cache.crow,
            self.cache.col,
            weight,
            (*weight.shape[:-1], *self._lowered_shape),
            check=False,
        )
        return csr.transpose() if orientation == "src_dst" else csr

    def _member_coordinates(self, row: Tensor, col: Tensor):
        """Split folded ids into ``(network coordinates, post, pre)``."""
        member = torch.div(row, self.n_post, rounding_mode="floor")
        coords = []
        for size in reversed(self._member_shape):
            coords.append(member % size)
            member = torch.div(member, size, rounding_mode="floor")
        return torch.stack(coords[::-1]), row % self.n_post, col % self.n_pre

    def edge_table(self) -> dict[str, Tensor]:
        """Semantic edge list: one entry per edge slot.

        Returns:
            Dict with ``pre``, ``post``, ``weight`` and, when present,
            ``receptor``, ``delay`` and ``group``.
        """
        table = {
            "pre": self.indices[1],
            "post": self.indices[0],
            "weight": self.weight(),
        }
        if self._member_shape:
            member, post, pre = self._member_coordinates(*self.indices)
            table.update(pre=pre, post=post, network=member)
        if self.receptor is not None:
            table["receptor"] = self.receptor
        if self.delay is not None:
            table["delay"] = self.delay
        group = getattr(self.weight, "group", None)
        if group is not None:
            table["group"] = group
        return table

    def explain(self, x: Tensor | None = None) -> str:
        """Describe how this connection is executed (debugging aid)."""
        device = self.indices.device.type
        plan = self._plan
        lines = []
        if x is not None:
            density = float((x != 0).float().mean()) if x.numel() else 0.0
            lines += [
                "Input:",
                f"    shape = {list(x.shape)}",
                "    representation = dense",
                f"    measured density = {density:.2%}",
            ]
        lines += [
            "Connection:",
            f"    logical shape = [{self.n_post}, {self.n_pre}]",
            f"    network batch = {list(self.batch_shape)}",
            f"    nnz = {self.nnz}",
            "    canonical state = edge slots [post, pre]"
            + (", receptor" if self.receptor is not None else "")
            + (", delay" if self.delay is not None else ""),
            f"    topology_version = {self.topology_version}",
            "Selected runtime:",
            f"    cached representation = {plan.representation}",
            f"    algorithm = {plan.algorithm} ({plan.reason})",
            f"    backend = {plan.backend}",
            "    available backends = "
            + str(
                registry.available(
                    "spike_push" if plan.algorithm == "adaptive-push" else "csr_matvec",
                    device,
                )
            ),
            f"Weight:\n    {type(self.weight).__name__}({self.weight.extra_repr()})",
            "Receptor / delay:",
            f"    n_receptor = {self.n_receptor}, n_delay = {self.n_delay} "
            "(expanded reference lowering)",
        ]
        return "\n".join(lines)

    def set_hints(self, hints: Hints) -> None:
        """Change the performance hints and re-plan execution."""
        self.hints = hints
        self._plan = planner.plan(
            device=self.indices.device.type,
            value_batched=self._value_batch_dim > 0,
            hints=hints,
        )

    def _apply(self, fn, recurse: bool = True):
        out = super()._apply(fn, recurse)
        # The device may have changed: backends are chosen per device.
        if "_plan" in self.__dict__:
            self.set_hints(self.hints)
        return out

    def extra_repr(self) -> str:
        parts = [f"n_pre={self.n_pre}", f"n_post={self.n_post}", f"nnz={self.nnz}"]
        if self.n_receptor > 1:
            parts.append(f"n_receptor={self.n_receptor}")
        if self.n_delay > 1:
            parts.append(f"n_delay={self.n_delay}")
        if self.batch_shape:
            parts.append(f"batch_shape={self.batch_shape}")
        return ", ".join(parts)


def _as_sparse_default_dtype(matrix: Any, kwargs: dict) -> Sparse:
    """``as_sparse`` with the SciPy default-dtype rule of the constructor."""
    if kwargs.get("dtype") is None and not isinstance(matrix, (Sparse, Tensor)):
        kwargs["dtype"] = torch.get_default_dtype()
    return as_sparse(matrix)


def _after_load(module: SparseConnection, incompatible_keys) -> None:
    """Checkpoint loaded: the canonical edges may differ, rebuild layouts."""
    module.topology_version += 1
    module.routing_version += 1
    module.value_version += 1
    module._rebuild()
