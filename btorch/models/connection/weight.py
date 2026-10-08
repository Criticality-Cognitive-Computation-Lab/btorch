"""Synaptic weight parameterisations.

A weight module owns whatever is trainable about the strength of a set
of edges and returns one effective value per edge. It is independent of
how the edges are stored or executed, so every parameterisation works
with every connection realisation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ...sparse import EdgeMap, Sparse, as_sparse
from ..constrain import HasConstraint


class Weight(nn.Module, HasConstraint):
    """Base class: ``forward()`` returns ``[*batch, n_edge]`` effective weights.

    Subclasses are created with data aligned to the edges *as the user
    supplied them*. The owning connection calls :meth:`bind` once with the map
    from those edges to its canonical edge order; after that the module is
    aligned with the connection's edge slots.
    """

    def forward(self) -> Tensor:
        raise NotImplementedError

    def bind(self, base: Tensor, edge_map: EdgeMap, adjacency: Sparse) -> None:
        """Align with a connection's canonical edges.

        Args:
            base: ``[*batch, n_input_edge]`` values stored in the matrix the
                connection was built from (used when the weight was not given
                explicit values).
            edge_map: Map from input edges to canonical edge slots.
            adjacency: The input matrix in standard orientation
                (``(n_post, n_pre)``, input edge order), for weights that
                locate edges by coordinate.
        """
        raise NotImplementedError

    def constrain(self, *args: Any, **kwargs: Any) -> None:
        """Project the parameters onto their constraint set (no-op by
        default)."""

    def reset_slots(
        self, slots: Tensor, value: float | Tensor = 0.0, sign: Tensor | None = None
    ) -> None:
        """Re-initialise the trainable state of rewired edge slots.

        Args:
            slots: ``[K]`` edge slots.
            value: New weight (scalar or ``[K]``).
            sign: Optional ``[K]`` new Dale reference signs.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support structural rewiring."
        )


def _dale_project(value: Tensor, sign: Tensor) -> Tensor:
    return (value * sign).relu() * sign


class EdgeWeight(Weight):
    """One weight per edge.

    Args:
        value: ``[*batch, n_edge]`` initial weights aligned with the input
            edges, or ``None`` to take them from the connection matrix.
        trainable: Store the weights as a parameter (else a buffer).
        dale: Enforce Dale's law: every weight keeps the sign it had at
            construction. :meth:`constrain` clamps weights that crossed zero
            to zero. The reference sign is part of the ``state_dict``.

    Attributes:
        value: The stored weights (parameter or buffer).
        sign: Reference sign per edge (only with ``dale``).
    """

    value: Tensor
    sign: Tensor

    def __init__(
        self,
        value: Tensor | None = None,
        *,
        trainable: bool = True,
        dale: bool = False,
    ):
        super().__init__()
        self.trainable = trainable
        self.dale = dale
        self._given = value is not None
        if value is not None:
            self._set(torch.as_tensor(value).detach().clone())

    def _set(self, value: Tensor) -> None:
        if not value.is_floating_point():
            value = value.to(torch.get_default_dtype())
        if self.trainable:
            self.value = nn.Parameter(value)
        else:
            self.register_buffer("value", value)
        if self.dale:
            self.register_buffer("sign", torch.sign(value))

    def bind(self, base: Tensor, edge_map: EdgeMap, adjacency: Sparse) -> None:
        value = self.value.detach() if self._given else base
        value = value.to(device=edge_map.target.device)
        for name in ("value", "sign"):
            self._parameters.pop(name, None)
            self._buffers.pop(name, None)
        self._set(edge_map.sum(value, dim=-1).detach().clone())
        self._given = True

    def forward(self) -> Tensor:
        return self.value

    @torch.no_grad()
    def constrain(self, *args: Any, **kwargs: Any) -> None:
        if self.dale:
            self.value.copy_(_dale_project(self.value, self.sign))

    @torch.no_grad()
    def reset_slots(
        self, slots: Tensor, value: float | Tensor = 0.0, sign: Tensor | None = None
    ) -> None:
        self.value[..., slots] = value
        if self.dale and sign is not None:
            self.sign[..., slots] = sign.to(self.sign.dtype)

    def extra_repr(self) -> str:
        return (
            f"n_edge={self.value.shape[-1]}, trainable={self.trainable}, "
            f"dale={self.dale}"
        )


class ConstantWeight(Weight):
    """The same fixed weight on every edge.

    Parallel edges between the same pair (multapses) that a connection merges
    into one slot count with their multiplicity.

    Args:
        value: Scalar weight.

    Attributes:
        value: The scalar weight (buffer).
        multiplicity: ``[n_edge]`` number of input edges merged into each
            slot (persistent buffer).
    """

    value: Tensor
    multiplicity: Tensor

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.register_buffer("value", torch.as_tensor(float(value)))
        self.register_buffer("multiplicity", torch.zeros(0))

    def bind(self, base: Tensor, edge_map: EdgeMap, adjacency: Sparse) -> None:
        ones = torch.ones(edge_map.n_old, device=edge_map.target.device)
        self.multiplicity = edge_map.sum(ones, dim=-1)

    def forward(self) -> Tensor:
        return self.value * self.multiplicity

    @torch.no_grad()
    def reset_slots(
        self, slots: Tensor, value: float | Tensor = 0.0, sign: Tensor | None = None
    ) -> None:
        self.multiplicity[slots] = 1.0

    def extra_repr(self) -> str:
        return f"value={float(self.value):g}, n_edge={self.multiplicity.shape[0]}"


class ConstrainedWeight(Weight):
    """Grouped (tied) weights: one learnable scale per group of edges.

    .. math::
        w_e = \\mathrm{base}_e \\cdot \\mathrm{scale}_{\\mathrm{group}_e}

    ``base`` is fixed; only the per-group ``scale`` is trained, so all edges
    of a group keep their relative strengths.

    Args:
        group: Group of every edge. Either a ``[n_edge]`` integer tensor of
            0-based group ids aligned with the input edges, or a sparse matrix
            (anything :func:`btorch.sparse.as_sparse` accepts) with the same
            orientation as the connection matrix whose entries are *1-based*
            group ids; edges are then matched by coordinate.
        base: ``[n_edge]`` fixed per-edge weight, or ``None`` to take it from
            the connection matrix.
        scale: Initial ``[*batch, n_group]`` scales (default: ones).
        n_group: Number of groups (default: largest id + 1).
        dale: Keep every scale non-negative, so no edge can change sign.

    Attributes:
        base: ``[n_edge]`` fixed weights (persistent buffer).
        group: ``[n_edge]`` 0-based group id per edge (persistent buffer).
        scale: ``[*batch, n_group]`` learnable scales.
    """

    base: Tensor
    group: Tensor
    scale: nn.Parameter

    def __init__(
        self,
        group: Tensor | Any,
        base: Tensor | None = None,
        scale: Tensor | None = None,
        *,
        n_group: int | None = None,
        dale: bool = False,
    ):
        super().__init__()
        self.dale = dale
        self._group_matrix = None
        self._base_given = base is not None
        if isinstance(group, Tensor) and group.layout == torch.strided:
            group = group.to(torch.long)
        else:
            # Coordinate-matched in ``bind``; ids in a matrix are 1-based
            # because 0 means "no entry".
            self._group_matrix = as_sparse(group)
            group = torch.zeros(0, dtype=torch.long)
        self.register_buffer("group", group)
        self.register_buffer(
            "base",
            torch.zeros(0) if base is None else torch.as_tensor(base).detach().clone(),
        )
        if n_group is None and scale is not None:
            n_group = int(torch.as_tensor(scale).shape[-1])
        self._n_group = n_group
        self._init_scale(scale)

    def _init_scale(self, scale: Tensor | None) -> None:
        n_group = self._n_group
        if n_group is None:
            n_group = int(self.group.max()) + 1 if self.group.numel() else 0
        if scale is None:
            scale = torch.ones(n_group, dtype=self.base.dtype)
        self.scale = nn.Parameter(torch.as_tensor(scale).detach().clone())
        if self.group.numel() and int(self.group.max()) >= self.scale.shape[-1]:
            raise ValueError(
                f"group ids go up to {int(self.group.max())} but there are only "
                f"{self.scale.shape[-1]} scales."
            )

    def bind(self, base: Tensor, edge_map: EdgeMap, adjacency: Sparse) -> None:
        device = edge_map.target.device
        if self._group_matrix is not None:
            group = _match_by_coordinate(adjacency, self._group_matrix).to(device)
            self._group_matrix = None
        else:
            group = self.group.to(device)
        if group.shape[0] != edge_map.n_old:
            raise ValueError(
                f"group has {group.shape[0]} entries but the connection has "
                f"{edge_map.n_old} input edges."
            )
        if (base if not self._base_given else self.base).ndim != 1:
            raise ValueError(
                "ConstrainedWeight needs one base weight per edge; batch the "
                "scales ([*batch, n_group]) instead of the base."
            )
        base = (self.base if self._base_given else base).to(device)
        scale = self.scale.detach() if self._n_group is not None else None
        # Merged duplicates must agree on their group, otherwise no single
        # scale describes the merged edge.
        self.group = edge_map.take(group, check=True)
        self.base = edge_map.sum(base, dim=-1).detach().clone()
        self._base_given = True
        self._init_scale(None if scale is None else scale.to(self.base.dtype))

    def forward(self) -> Tensor:
        return self.base * self.scale.index_select(-1, self.group)

    @torch.no_grad()
    def constrain(self, *args: Any, **kwargs: Any) -> None:
        if self.dale:
            self.scale.clamp_(min=0)

    # ------------------------------------------------------- introspection
    @property
    def n_group(self) -> int:
        return int(self.scale.shape[-1])

    def group_sizes(self) -> Tensor:
        """``[n_group]`` number of edges in every group."""
        return torch.bincount(self.group, minlength=self.n_group)

    def group_info(self, include_weights: bool = False):
        """Summarise the groups as a DataFrame.

        Args:
            include_weights: Add mean / standard deviation of the base
                weights of each group.

        Returns:
            ``pandas.DataFrame`` with ``group_id``, ``num_connections``,
            ``scale`` (and the base-weight statistics if requested).
        """
        import pandas as pd

        if self.scale.ndim != 1:
            raise ValueError("group_info is defined for unbatched scales.")
        sizes = self.group_sizes()
        info = {
            "group_id": range(self.n_group),
            "num_connections": sizes.tolist(),
            "scale": self.scale.detach().tolist(),
        }
        if include_weights:
            base = self.base.double()
            count = sizes.clamp(min=1).double()
            zeros = torch.zeros(self.n_group, dtype=base.dtype, device=base.device)
            mean = zeros.index_add(0, self.group, base) / count
            sq = zeros.index_add(0, self.group, base**2) / count
            info["mean_base_weight"] = mean.tolist()
            info["std_base_weight"] = (sq - mean**2).clamp(min=0).sqrt().tolist()
        return pd.DataFrame(info)

    @torch.no_grad()
    def set_scale(self, group_id: int, value: float | Tensor) -> None:
        """Set the scale of one group."""
        self.scale[..., group_id] = value

    def weights_by_group(self) -> dict[int, Tensor]:
        """Effective weights of the edges of every group."""
        weight = self.forward()
        return {g: weight[..., self.group == g] for g in range(self.n_group)}

    def extra_repr(self) -> str:
        return f"n_edge={self.base.shape[0]}, n_group={self.n_group}, dale={self.dale}"


def _match_by_coordinate(adjacency: Sparse, labels: Sparse) -> Tensor:
    """0-based label of every entry of ``adjacency`` from a 1-based label
    matrix with the same shape."""
    if tuple(labels.sparse_shape) != tuple(adjacency.sparse_shape):
        raise ValueError(
            f"The group matrix has shape {labels.sparse_shape}, expected "
            f"{adjacency.sparse_shape} (same orientation as the connection)."
        )
    lab = labels.tocoo()
    n_col = adjacency.sparse_shape[1]
    lab_key = lab.row * n_col + lab.col
    keep = lab.data != 0
    lab_key, lab_val = lab_key[keep], lab.data[keep].to(torch.long)
    lab_key, order = torch.sort(lab_key)
    lab_val = lab_val[order]
    _, row, col = adjacency._edge_index()
    key = (row * n_col + col).to(lab_key.device)
    pos = torch.searchsorted(lab_key, key).clamp(max=max(lab_key.shape[0] - 1, 0))
    if lab_key.shape[0] == 0 or not bool((lab_key[pos] == key).all()):
        raise ValueError("Constraint missing for some connections.")
    return lab_val[pos] - 1
