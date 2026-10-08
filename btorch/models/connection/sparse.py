"""Explicit sparse connection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from ...sparse import COO, CSR, Hints, Sparse, as_sparse
from ...sparse.coo import _ravel
from ...sparse.runtime import (
    RepresentationCache,
    kernels_aten,
    kernels_triton_push,
    kernels_triton_tasks,
    ops,
    planner,
    registry,
)
from .base import Connection
from .synapse import Synapse
from .weight import Weight


Orientation = Literal["pre_post", "post_pre"]


@dataclass(frozen=True)
class SparseCheckpointBinding:
    """Exact sparse execution binding captured for checkpoint recomputation."""

    route_binding: Any
    route_token: int
    max_density: float | None
    execution_weight: Tensor | None
    task_values: Tensor | None
    task_state: kernels_triton_tasks.PreparedTaskState | None


@dataclass(frozen=True)
class SparseCheckpointRestoreState:
    """Sparse execution state temporarily displaced by recomputation."""

    binding: SparseCheckpointBinding
    run_active: bool


def _check_orientation(orientation: str) -> None:
    if orientation not in ("pre_post", "post_pre"):
        raise ValueError(
            "orientation must be 'pre_post' (rows are sources, y = x @ W) or "
            f"'post_pre' (the operator, y = A @ x), got {orientation!r}."
        )


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
        bias: ``True`` for a zero-initialised bias, or an initial
            ``[out_features]`` tensor (copied; becomes a parameter).
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
        bias: Tensor | bool | None = None,
        hints: Hints | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        _adjacency: Sparse | None = None,
        _scipy_origin: bool = False,
    ):
        super().__init__()
        synapse = synapse or Synapse()
        # SciPy data is float64 by NumPy convention, not by choice: such
        # matrices (and integer ones) get the default dtype unless one is
        # requested. Explicitly supplied weight tensors keep their own dtype.
        value_dtype = dtype
        if dtype is None and (
            _scipy_origin or not isinstance(matrix, (Sparse, Tensor))
        ):
            value_dtype = torch.get_default_dtype()
        if synapse.plasticity is not None:
            raise NotImplementedError(
                "Synapse.plasticity is reserved for online plasticity rules "
                "and not implemented yet."
            )
        A = as_sparse(matrix, device=device)
        if getattr(A, "_reordered", False) and synapse.has_edge_arrays():
            raise ValueError(
                "The matrix is an uncoalesced PyTorch COO tensor that requires "
                "grad, so its entries had to be merged and reordered; per-edge "
                "Synapse arrays cannot be aligned with it. Coalesce the tensor "
                "first, or pass a btorch/SciPy matrix."
            )
        if A.sparse_dim() != 2 or A.dense_dim():
            raise ValueError(
                "A connection matrix needs exactly two sparse dimensions and "
                f"scalar entries, got sparse_shape={A.sparse_shape}, "
                f"dense_shape={A.dense_shape}."
            )
        coo = A.tocoo()
        adjacency = coo if _adjacency is None else _adjacency.tocoo()
        values = coo.values()
        if value_dtype is not None:
            values = values.to(value_dtype)
        elif not values.is_floating_point():
            values = values.to(torch.get_default_dtype())
        n_post, n_pre = A.sparse_shape
        self.n_pre, self.n_post = int(n_pre), int(n_post)

        # A batch of different patterns is executed as one block-diagonal
        # operator over the concatenated populations (reference fallback).
        lead, post, pre = coo._edge_index()
        self._member_shape: tuple[int, ...] = ()
        if lead is not None and coo._value_batch_dim:
            raise NotImplementedError(
                "A batch of different patterns that also has a shared-pattern "
                "value batch is not supported."
            )
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
        self.orientation: Orientation = "post_pre"
        self.bias = None
        self.register_buffer(
            "layout",
            torch.tensor(
                [self.n_post, self.n_pre, n_receptor, n_delay, *self._member_shape],
                device=row.device,
            ),
        )

        # Version counters: which derived state a change invalidates.
        self.topology_version = 0
        self.routing_version = 0
        self._layout_version = 0
        self._always_permute = False
        self._run_weight: Tensor | None = None
        self._run_task_values: Tensor | None = None
        self._run_active = False
        self._task_epoch = 0
        self._prepared_task_values: Tensor | None = None
        self._prepared_task_signature: tuple | None = None
        self._prepared_task_epoch = 0
        self._prepared_weight: Tensor | None = None
        self._prepared_weight_signature: tuple | None = None
        self._route_token = 0
        self.cache = RepresentationCache()
        self._rebuild()

        effective = self.weight()
        if bias is True:
            bias = torch.zeros(self.out_features)
        if bias is not None and bias is not False:
            # Own copy on the connection's device, in the weight dtype.
            bias = torch.as_tensor(bias, dtype=effective.dtype, device=row.device)
            if bias.shape != (self.out_features,):
                raise ValueError(
                    f"bias must have shape ({self.out_features},) "
                    f"(out_features), got {tuple(bias.shape)}."
                )
            self.bias = nn.Parameter(bias.clone())
        self._value_batch_dim = effective.ndim - 1
        if self._value_batch_dim and self._member_shape:
            raise NotImplementedError(
                "Batched weights on a batch of different patterns are not supported."
            )
        self.hints = hints or A.hints
        self._replan()
        self._prepare()
        self.register_load_state_dict_post_hook(_after_load)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def _replan(self) -> None:
        self._clear_prepared_task_values()
        plan = planner.plan(
            device=self.indices.device.type,
            value_batched=self._value_batch_dim > 0 or bool(self._member_shape),
            hints=self.hints,
            dtype=self.weight().dtype,
        )
        old_plan = getattr(self, "_plan", None)
        if (
            old_plan is not None
            and old_plan.route.fingerprint != plan.route.fingerprint
        ):
            kernels_triton_push.invalidate_task_state(
                self.cache.t_crow,
                self.cache.t_col,
                self.cache.t_perm,
            )
        self._plan = plan
        self._route = registry.bind_route(self._plan.route)
        self._route_token = ops.bind_route(self._route)
        self._fallback_route = None
        self._fallback_route_token = 0
        if self._plan.algorithm == "push":
            fallback = planner.plan(
                device=self.indices.device.type,
                value_batched=self._value_batch_dim > 0 or bool(self._member_shape),
                hints=Hints(),
                dtype=self.weight().dtype,
            )
            if fallback.algorithm == "pull":
                self._fallback_route = registry.bind_route(fallback.route)
                self._fallback_route_token = ops.bind_route(self._fallback_route)
        # ``None`` selects destination-driven propagation in ``forward``.
        self._max_density = (
            None if self._plan.algorithm == "pull" else self._plan.max_density
        )

    def _ensure_plan_current(self) -> None:
        """Refresh planning once when a planning dependency changed."""
        override = registry.override_snapshot()
        plan = self._plan
        if (
            plan.route.registry_generation != registry.registry_generation
            or plan.route.override != override.name
            or plan.deterministic != torch.are_deterministic_algorithms_enabled()
        ):
            self._replan()
            self._prepare()

    def __getstate__(self) -> dict:
        """Serialize canonical module state without runtime route caches."""
        state = super().__getstate__().copy()
        for name in (
            "_plan",
            "_route",
            "_route_token",
            "_fallback_route",
            "_fallback_route_token",
            "_run_weight",
            "_run_task_values",
            "_prepared_task_values",
            "_prepared_task_signature",
            "_prepared_weight",
            "_prepared_weight_signature",
        ):
            state.pop(name, None)
        state["_run_active"] = False
        state["_prepared_task_epoch"] = 0
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore canonical state and rebuild process-local runtime caches."""
        super().__setstate__(state)
        object.__setattr__(self, "_run_weight", None)
        object.__setattr__(self, "_run_task_values", None)
        self._run_active = False
        self._prepared_task_values = None
        self._prepared_task_signature = None
        self._prepared_task_epoch = 0
        self._prepared_weight = None
        self._prepared_weight_signature = None
        self._route_token = 0
        self._fallback_route = None
        self._fallback_route_token = 0
        self._replan()
        self._prepare()

    # ------------------------------------------------------------ builders
    @classmethod
    def from_adjacency(
        cls,
        matrix: Any,
        synapse: Synapse | None = None,
        *,
        orientation: Orientation = "pre_post",
        **kwargs,
    ) -> "SparseConnection":
        """Build from an adjacency / connectivity matrix.

        Args:
            matrix: Sparse matrix (btorch, PyTorch or SciPy).
            synapse: Synapse description; per-edge arrays are aligned with
                the stored entries of ``matrix``, and matrices inside it (e.g.
                a group-id matrix) use the same orientation as ``matrix``.
            orientation: ``"pre_post"`` (default): rows are sources, columns
                are targets, the connection computes ``x @ matrix``.
                ``"post_pre"``: the matrix already is the operator
                ``(n_post, n_pre)``. :meth:`to_sparse` returns the matrix in
                the orientation given here.
            **kwargs: Passed to the constructor.
        """
        _check_orientation(orientation)
        A = _as_sparse_default_dtype(matrix, kwargs)
        if orientation == "pre_post":
            conn = cls(A.transpose(), synapse, _adjacency=A, **kwargs)
        else:
            conn = cls(A, synapse, **kwargs)
        conn.orientation = orientation
        return conn

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
        if pre.ndim != 1 or pre.shape != post.shape:
            raise ValueError(
                "pre and post must be 1-D with one entry per edge, got shapes "
                f"{tuple(pre.shape)} and {tuple(post.shape)}."
            )
        for name, ids, n in (("pre", pre, n_pre), ("post", post, n_post)):
            if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= n):
                raise ValueError(
                    f"{name} ids must be in [0, {n}), got range "
                    f"[{int(ids.min())}, {int(ids.max())}]."
                )
        if values is None:
            values = torch.ones(pre.shape[0], device=pre.device)
        coo = COO(
            torch.stack([post, pre]),
            values,
            (*values.shape[:-1], n_post, n_pre),
            check=False,
        )
        conn = cls(coo, synapse, **kwargs)
        conn.orientation = "pre_post"
        return conn

    @classmethod
    def from_hetersynapse(
        cls,
        matrix: Any,
        synapse: Synapse | None = None,
        *,
        n_receptor: int | None = None,
        n_delay: int = 1,
        receptor_type_index: Any = None,
        **kwargs,
    ) -> "SparseConnection":
        """Build from a physically expanded hetersynapse matrix.

        Adapter for the output of
        :func:`btorch.connectome.connection.make_hetersynapse_conn` (and
        ``expand_conn_for_delays``): a ``pre_post`` matrix whose columns are
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
            n_receptor: Number of receptor channels in the column expansion
                (default 1, or the length of ``receptor_type_index``).
            n_delay: Number of delay bins in the row expansion.
            receptor_type_index: The receptor index table returned by
                ``make_hetersynapse_conn``; only its length is used, to set
                ``n_receptor``. Keep the table itself to map channel ids to
                receptor names.
            **kwargs: Passed to the constructor.
        """
        if receptor_type_index is not None:
            if n_receptor is not None and n_receptor != len(receptor_type_index):
                raise ValueError(
                    f"n_receptor={n_receptor} disagrees with the "
                    f"{len(receptor_type_index)} rows of receptor_type_index."
                )
            n_receptor = len(receptor_type_index)
        n_receptor = 1 if n_receptor is None else n_receptor
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
        conn = cls(operator, decoded, _adjacency=A, **kwargs)
        conn.orientation = "pre_post"
        return conn

    # ---------------------------------------------------------- properties
    @property
    def in_features(self) -> int:
        return self.n_pre * self.n_delay

    @property
    def out_features(self) -> int:
        return self.n_post * self.n_receptor

    @property
    def pre(self) -> Tensor:
        """``[n_edge]`` source neuron of every edge slot."""
        return self.indices[1]

    @property
    def post(self) -> Tensor:
        """``[n_edge]`` target neuron of every edge slot."""
        return self.indices[0]

    def find_edges(self, pre: Tensor | int, post: Tensor | int) -> Tensor:
        """Edge slots connecting the given neurons.

        Args:
            pre: Source neuron id(s), a scalar or 1-D tensor.
            post: Target neuron id(s), a scalar or 1-D tensor.

        Returns:
            1-D tensor of the slots whose source is in ``pre`` and whose
            target is in ``post`` (several per pair with receptors, delays or
            after rewiring; empty if unconnected).
        """
        device = self.indices.device
        pre = torch.as_tensor(pre, device=device).reshape(-1)
        post = torch.as_tensor(post, device=device).reshape(-1)
        hit = torch.isin(self.indices[1], pre) & torch.isin(self.indices[0], post)
        return hit.nonzero().squeeze(1)

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
        if isinstance(x, Sparse) or (
            isinstance(x, Tensor) and x.layout != torch.strided
        ):
            return self.propagate_events(x)
        if not self._run_active and not torch.compiler.is_compiling():
            self._ensure_plan_current()
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

        temporary_run = (
            not self._run_active
            and not torch.compiler.is_compiling()
            and self._plan.algorithm == "push"
        )
        if temporary_run:
            batch_size = x.numel() // x.shape[-1]
            self._begin_sparse_run(batch_size, x.dtype)
        try:
            cache = self.cache
            weight = self._run_weight
            if weight is None:
                weight = self.weight()
                if self._always_permute or not cache.identity_perm:
                    weight = weight.index_select(-1, cache.perm)
            propagate = (
                ops.propagate_bound
                if torch.compiler.is_compiling()
                else ops.propagate_binding
            )
            compiled = torch.compiler.is_compiling()
            use_fallback = (
                self._plan.algorithm == "push"
                and self._run_task_values is None
                and self._fallback_route is not None
            )
            if compiled:
                route = (
                    self._fallback_route_token if use_fallback else self._route_token
                )
            else:
                route = self._fallback_route if use_fallback else self._route
            out = propagate(
                route,
                cache.crow,
                cache.col,
                weight,
                x,
                cache.t_crow,
                cache.t_col,
                cache.t_perm,
                None if use_fallback else self._max_density,
                self._run_task_values,
            )
        finally:
            if temporary_run:
                self._end_sparse_run()
        if self._member_shape:
            out = self._unfold(out)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _begin_sparse_run(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        stable_addresses: bool = False,
    ) -> None:
        """Prepare execution values once for a recurrent trajectory."""
        if self._run_active:
            raise RuntimeError(
                "SparseConnection does not support overlapping recurrent "
                "trajectories on the same module instance."
            )
        object.__setattr__(self, "_run_weight", None)
        object.__setattr__(self, "_run_task_values", None)
        self._run_active = True
        try:
            self._ensure_plan_current()
            weight = self.weight()
            cache = self.cache
            if self._always_permute or not cache.identity_perm:
                ordered = weight.index_select(-1, cache.perm)
                if stable_addresses:
                    signature = (
                        self.topology_version,
                        self.routing_version,
                        str(ordered.device),
                        ordered.dtype,
                        tuple(ordered.shape),
                    )
                    if self._prepared_weight_signature != signature:
                        self._prepared_weight = ordered.detach().clone()
                        self._prepared_weight_signature = signature
                    else:
                        self._prepared_weight.copy_(ordered.detach())
                    weight = self._prepared_weight
                else:
                    weight = ordered
            object.__setattr__(self, "_run_weight", weight)
            if self._max_density is None or self._value_batch_dim or self._member_shape:
                return
            if weight.dtype != dtype:
                raise ValueError(
                    "SparseConnection input dtype must match its weight dtype; "
                    f"got input {dtype} and weight {weight.dtype}."
                )
            signature = self._task_value_signature(batch_size, dtype)
            if (
                self._prepared_task_values is not None
                and self._prepared_task_signature == signature
            ):
                object.__setattr__(self, "_run_task_values", self._prepared_task_values)
                return
            prepare_values = self._route.implementation.prepare_values
            if prepare_values is None:
                return
            task_values, epoch = prepare_values(
                cache.t_crow,
                cache.t_col,
                cache.t_perm,
                weight,
                batch_size,
                route_key=self._plan.route.fingerprint,
            )
            object.__setattr__(self, "_run_task_values", task_values)
            self._prepared_task_values = task_values
            self._prepared_task_signature = (
                signature if task_values is not None else None
            )
            self._prepared_task_epoch = epoch
            if epoch and epoch != self._task_epoch:
                self._task_epoch = epoch
                self._layout_version += 1
        except BaseException:
            self._end_sparse_run()
            raise

    def _end_sparse_run(self) -> None:
        """Release tensors whose lifetime is one recurrent trajectory."""
        object.__setattr__(self, "_run_weight", None)
        object.__setattr__(self, "_run_task_values", None)
        self._run_active = False

    def _capture_checkpoint_binding(self) -> SparseCheckpointBinding:
        """Capture the exact run binding used by checkpoint forward."""
        task_state = None
        if self._run_task_values is not None:
            task_binding = kernels_triton_tasks.state_for_values(
                registry.kernels,
                self._run_task_values,
                self.cache.t_crow,
                self.cache.t_col,
                self.cache.t_perm,
            )
            task_state = None if task_binding is None else task_binding.task_state
        return SparseCheckpointBinding(
            route_binding=self._route,
            route_token=self._route_token,
            max_density=self._max_density,
            execution_weight=self._run_weight,
            task_values=self._run_task_values,
            task_state=task_state,
        )

    def _install_checkpoint_binding(
        self, checkpoint_binding: SparseCheckpointBinding
    ) -> SparseCheckpointRestoreState:
        """Install a forward run binding for checkpoint recomputation."""
        previous_state = SparseCheckpointRestoreState(
            binding=self._capture_checkpoint_binding(),
            run_active=self._run_active,
        )
        if (
            checkpoint_binding.task_state is not None
            and checkpoint_binding.task_values is not None
        ):
            kernels_triton_tasks.restore_binding(
                registry.kernels,
                checkpoint_binding.task_state,
                checkpoint_binding.task_values,
                self.cache.t_crow,
                self.cache.t_col,
                self.cache.t_perm,
            )
        self._route = checkpoint_binding.route_binding
        self._route_token = checkpoint_binding.route_token
        self._max_density = checkpoint_binding.max_density
        object.__setattr__(self, "_run_weight", checkpoint_binding.execution_weight)
        object.__setattr__(
            self,
            "_run_task_values",
            checkpoint_binding.task_values,
        )
        self._run_active = True
        return previous_state

    def _restore_checkpoint_run_state(
        self, runtime_state: SparseCheckpointRestoreState
    ) -> None:
        """Restore the runtime binding present before recomputation."""
        binding = runtime_state.binding
        self._route = binding.route_binding
        self._route_token = binding.route_token
        self._max_density = binding.max_density
        object.__setattr__(self, "_run_weight", binding.execution_weight)
        object.__setattr__(self, "_run_task_values", binding.task_values)
        self._run_active = runtime_state.run_active

    def _task_value_signature(self, batch_size: int, dtype: torch.dtype) -> tuple:
        """Identity of task-packed values reusable by checkpoint
        recomputation."""
        tensors = (*self.weight.parameters(), *self.weight.buffers())
        return (
            self.topology_version,
            self.routing_version,
            registry.kernels.generation,
            batch_size,
            dtype,
            tuple((str(t.device), t.dtype, t.data_ptr(), t._version) for t in tensors),
        )

    def _clear_prepared_task_values(self) -> None:
        self._prepared_task_values = None
        self._prepared_task_signature = None
        self._prepared_task_epoch = 0
        self._prepared_weight = None
        self._prepared_weight_signature = None

    @torch.compiler.disable
    def propagate_events(self, events: Sparse | Tensor) -> Tensor:
        """Propagate genuinely sparse input events.

        For input that is sparse by nature (event-camera data, precomputed
        spike trains) rather than produced by a differentiable neuron model.
        Only the out-edges of the listed events are visited and the input is
        never densified. Gradients flow to the weights and to the event
        values, not to absent events; for surrogate-gradient training pass
        the dense spike tensor to :meth:`forward` instead. This path uses the
        reference kernel on every device and is excluded from
        ``torch.compile`` tracing (a compiled module falls back to eager for
        this call).

        Args:
            events: Sparse array of shape ``[B, in_features]`` (or
                ``[in_features]``) whose stored entries are the events.

        Returns:
            Dense ``[B, out_features]`` (or ``[out_features]``).
        """
        if self._value_batch_dim or self._member_shape:
            raise NotImplementedError("Sparse event input for a network batch.")
        events = as_sparse(events)
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
        self._clear_prepared_task_values()
        if "_plan" in self.__dict__:
            kernels_triton_push.invalidate_task_state(
                self.cache.t_crow,
                self.cache.t_col,
                self.cache.t_perm,
            )
        row, col = self._lowered_edges()
        cache = self.cache
        before = tuple(getattr(cache, n).data_ptr() for n in cache._NAMES)
        cache.build(row, col, self._lowered_shape, self.topology_version)
        if not cache.identity_perm:
            # Slot order differs from execution order from now on; never
            # switch back, so a traced forward keeps one shape.
            self._always_permute = True
        if before != tuple(getattr(cache, n).data_ptr() for n in cache._NAMES):
            self._layout_version += 1
        if "_plan" in self.__dict__:
            self._prepare()

    def _prepare(self) -> None:
        """Prepare only the selected route, outside execution and capture."""
        prepare = self._route.implementation.prepare
        if prepare is None:
            return
        c = self.cache
        epoch = prepare(c.crow, c.col, c.t_crow, c.t_col, c.t_perm)
        if epoch is not None and epoch != getattr(self, "_backend_epoch", epoch):
            self._layout_version += 1
        self._backend_epoch = epoch

    @property
    def value_version(self) -> int:
        """Counter that changes whenever a weight tensor is written in place
        (optimizer steps, ``constrain``, loading a checkpoint)."""
        return sum(
            t._version
            for t in (*self.weight.parameters(), *self.weight.buffers())
            if t.is_floating_point()
        )

    @property
    def capture_version(self) -> tuple:
        """Changes whenever a CUDA graph captured around this connection is no
        longer valid (edges, routing, plan, device or backend layouts changed).

        Weight updates do not change it: values are read at replay time.
        """
        return (
            self.topology_version,
            self.routing_version,
            self._layout_version,
            self._max_density,
            self._always_permute,
            self._task_epoch,
            registry.kernels.generation,
            str(self.indices.device),
            self._plan.route.fingerprint,
            registry.registry_generation,
            registry.override_snapshot(),
            torch.are_deterministic_algorithms_enabled(),
            tuple(
                (str(t.device), t.dtype, t.data_ptr())
                for t in (*self.weight.parameters(), *self.weight.buffers())
            ),
            None
            if self.bias is None
            else (str(self.bias.device), self.bias.dtype, self.bias.data_ptr()),
        )

    def capture_incompatibility(self) -> str | None:
        """Why this connection cannot be captured in a CUDA graph, if so."""
        if self._plan.algorithm == "adaptive-push":
            return (
                "a density-hinted connection packs spikes on the host on this "
                "backend, which cannot be captured in a CUDA graph; remove the "
                "expected_density hint or use the Triton backend"
            )
        return None

    @torch.no_grad()
    def set_edges_(
        self,
        slots: Tensor,
        *,
        pre: Tensor,
        post: Tensor,
        receptor: Tensor | None = None,
        delay: Tensor | None = None,
    ) -> None:
        """Rewire edge slots in place (structural plasticity).

        Tensor shapes do not change: slot ``k`` simply becomes a different
        edge. Derived execution layouts are rebuilt into their existing
        tensors and ``topology_version`` is incremented. Call this outside
        the forward pass (for example from an optimizer hook). The number of
        edge slots is fixed; two slots may describe the same pair, in which
        case their weights add. A CUDA graph captured before the call must
        be captured again (see :attr:`capture_version`).

        Args:
            slots: ``[K]`` edge slots to rewire.
            pre: ``[K]`` new source neuron of each slot.
            post: ``[K]`` new target neuron of each slot.
            receptor: Optional ``[K]`` new receptor ids.
            delay: Optional ``[K]`` new delays.
        """
        slots, pre, post = (torch.as_tensor(t) for t in (slots, pre, post))
        receptor = None if receptor is None else torch.as_tensor(receptor)
        delay = None if delay is None else torch.as_tensor(delay)
        if self._member_shape:
            raise NotImplementedError("Rewiring a batch of different patterns.")
        if slots.numel() == 0:
            return
        for name, ids in (
            ("post", post),
            ("pre", pre),
            ("receptor", receptor),
            ("delay", delay),
        ):
            if ids is not None and ids.shape != slots.shape:
                raise ValueError(
                    f"{name} has shape {tuple(ids.shape)}, expected one entry "
                    f"per rewired slot {tuple(slots.shape)}."
                )
        if (
            int(post.min()) < 0
            or int(post.max()) >= self.n_post
            or int(pre.min()) < 0
            or int(pre.max()) >= self.n_pre
        ):
            raise ValueError("Rewired edges must lie within the populations.")
        for name, ids, count in (
            ("receptor", receptor, self.n_receptor),
            ("delay", delay, self.n_delay),
        ):
            if ids is None:
                continue
            if getattr(self, name) is None:
                raise ValueError(f"This connection has no per-edge {name}.")
            if int(ids.min()) < 0 or int(ids.max()) >= count:
                raise ValueError(f"{name} ids must be in [0, {count}).")
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
        saved = state_dict.get(prefix + "layout")
        if saved is None and prefix + "indices" in state_dict:
            # A checkpoint of the removed SparseConn layers: same key name,
            # different weight keys, and nothing to validate the sizes with.
            error_msgs.append(
                f"'{prefix}indices' comes from a checkpoint without "
                f"'{prefix}layout' (written by the removed SparseConn / "
                "SparseConstrainedConn layers, or incomplete). It cannot be "
                "loaded; rebuild the connection from its matrix with "
                "SparseConnection.from_adjacency and copy the weights."
            )
            for key in [k for k in state_dict if k.startswith(prefix)]:
                state_dict.pop(key)
            return
        if saved is not None and saved.tolist() != self.layout.tolist():
            names = ("n_post", "n_pre", "n_receptor", "n_delay")
            error_msgs.append(
                f"'{prefix}layout': the checkpoint describes a connection with "
                f"{dict(zip(names, saved.tolist()))}, this module has "
                f"{dict(zip(names, self.layout.tolist()))}."
            )
            # Copy nothing from a checkpoint of a different connection.
            for key in [k for k in state_dict if k.startswith(prefix)]:
                state_dict.pop(key)
            return
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
            if value is None or value.numel() == 0:
                continue
            if value.is_floating_point():
                error_msgs.append(f"'{prefix}{name}' must hold integer ids.")
                state_dict = {k: v for k, v in state_dict.items() if k != prefix + name}
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
    def to_sparse(self, orientation: Orientation | None = None) -> Sparse:
        """Current effective matrix as a :class:`~btorch.sparse.Sparse`.

        Args:
            orientation: ``"post_pre"`` returns the operator
                ``(out_features, in_features)`` as CSR; ``"pre_post"`` its
                transpose (the connectome convention). Defaults to the
                orientation the connection was built from, so
                ``from_adjacency(W).to_sparse()`` has the layout of ``W``.

        Returns:
            Sparse array whose values are the effective weights (gradients
            flow to the weight parameters).
        """
        orientation = self.orientation if orientation is None else orientation
        _check_orientation(orientation)
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
            return out.transpose() if orientation == "pre_post" else out
        csr = CSR(
            self.cache.crow,
            self.cache.col,
            weight,
            (*weight.shape[:-1], *self._lowered_shape),
            check=False,
        )
        return csr.transpose() if orientation == "pre_post" else csr

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
            Dict with ``pre``, ``post``, ``weight`` (a detached snapshot)
            and, when present, ``receptor``, ``delay`` and ``group``.
        """
        table = {
            "pre": self.indices[1].clone(),
            "post": self.indices[0].clone(),
            "weight": self.weight().detach().clone(),
        }
        if self._member_shape:
            member, post, pre = self._member_coordinates(*self.indices)
            table.update(pre=pre, post=post, network=member)
        if self.receptor is not None:
            table["receptor"] = self.receptor.clone()
        if self.delay is not None:
            table["delay"] = self.delay.clone()
        group = getattr(self.weight, "group", None)
        if group is not None:
            table["group"] = group.clone()
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
            "    backend = "
            + registry.name(
                "csr_matvec"
                if plan.algorithm == "pull"
                else "spike_push_dense"
                if registry.has("spike_push_dense", device)
                else "spike_push",
                device,
            ),
            "    available backends = "
            + str(
                registry.available(
                    "csr_matvec" if plan.algorithm == "pull" else "spike_push",
                    device,
                )
            ),
            f"Weight:\n    {type(self.weight).__name__}({self.weight.extra_repr()})",
            "Receptor / delay:",
            f"    n_receptor = {self.n_receptor}, n_delay = {self.n_delay} "
            "(expanded reference lowering)",
        ]
        return "\n".join(lines)

    @property
    def hints(self) -> Hints:
        """Performance expectations used by the execution planner."""
        return self._hints

    @hints.setter
    def hints(self, hints: Hints) -> None:
        if not isinstance(hints, Hints):
            raise TypeError(
                "hints must be a Hints instance, got " f"{type(hints).__name__}."
            )
        if getattr(self, "_hints", None) == hints:
            return
        self._hints = hints
        if "_plan" in self.__dict__:
            self._replan()
            self._prepare()

    def set_hints(self, hints: Hints) -> None:
        """Change the performance hints and re-plan execution."""
        self.hints = hints

    def update_hints(self, **changes) -> None:
        """Change several performance expectations with one re-plan.

        Args:
            **changes: Fields of :class:`~btorch.sparse.Hints` to replace.
        """
        self.hints = self.hints.replace(**changes)

    def _apply(self, fn, recurse: bool = True):
        before = None
        old_task_sources = None
        if "_plan" in self.__dict__:
            old_task_sources = (
                self.cache.t_crow,
                self.cache.t_col,
                self.cache.t_perm,
            )
            before = (
                str(self.indices.device),
                tuple(
                    (str(t.device), t.dtype, t.data_ptr())
                    for t in (*self.weight.parameters(), *self.weight.buffers())
                ),
                None
                if self.bias is None
                else (str(self.bias.device), self.bias.dtype, self.bias.data_ptr()),
                tuple(getattr(self.cache, n).data_ptr() for n in self.cache._NAMES),
            )
        out = super()._apply(fn, recurse)
        # The device may have changed: backends are chosen per device and
        # keep derived layouts per buffer.
        if before is not None:
            after = (
                str(self.indices.device),
                tuple(
                    (str(t.device), t.dtype, t.data_ptr())
                    for t in (*self.weight.parameters(), *self.weight.buffers())
                ),
                None
                if self.bias is None
                else (str(self.bias.device), self.bias.dtype, self.bias.data_ptr()),
                tuple(getattr(self.cache, n).data_ptr() for n in self.cache._NAMES),
            )
        if before is not None and after != before:
            kernels_triton_push.invalidate_task_state(*old_task_sources)
            self._clear_prepared_task_values()
            self._layout_version += 1
            self._replan()
            self._prepare()
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
    """``as_sparse`` that remembers SciPy origin for the default-dtype rule."""
    if not isinstance(matrix, (Sparse, Tensor)):
        kwargs["_scipy_origin"] = True
    return as_sparse(matrix)


def _after_load(module: SparseConnection, incompatible_keys) -> None:
    """Checkpoint loaded: the canonical edges may differ, rebuild layouts."""
    module.topology_version += 1
    module.routing_version += 1
    module._rebuild()
