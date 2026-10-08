"""Structural properties, performance hints and edge maps of sparse objects.

Properties are facts about a stored representation that correctness may
rely on (for example that indices are sorted). Hints are expectations
about how the object will be used; they never change results.
"""

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class Properties:
    """Structural facts about a stored sparse representation.

    A ``True`` value is a guarantee; ``False`` means "not known", never
    "known to be violated".

    Args:
        sorted: Entries are in lexicographic (row-major) index order.
        unique: No index tuple is stored twice.
        symmetric: Stored values satisfy ``A == A.T``.
        structurally_symmetric: The sparsity pattern is symmetric.
        triangular: ``"lower"``, ``"upper"`` or ``None``.
        fixed_pattern: The pattern is never mutated (no rewiring).
        shared_pattern: A batch of matrices shares one pattern.
    """

    sorted: bool = False
    unique: bool = False
    symmetric: bool = False
    structurally_symmetric: bool = False
    triangular: str | None = None
    fixed_pattern: bool = False
    shared_pattern: bool = False

    @property
    def canonical(self) -> bool:
        """Whether entries are sorted and free of duplicates."""
        return self.sorted and self.unique

    def replace(self, **changes) -> "Properties":
        """Return a copy with some fields changed."""
        return replace(self, **changes)


@dataclass(frozen=True)
class Hints:
    """Performance expectations; advisory only.

    Only ``expected_density`` is consulted by the planner today; the other
    two are accepted and reserved.

    Args:
        expected_calls: How many products the object will take part in
            (reserved).
        expected_density: Expected fraction of non-zero input entries, in
            ``[0, 1]``.
        expected_batch: Expected number of right-hand sides per call
            (reserved).

    Raises:
        ValueError: If a value is out of range.
    """

    expected_calls: int | None = None
    expected_density: float | None = None
    expected_batch: int | None = None

    def __post_init__(self) -> None:
        density = self.expected_density
        if density is not None and not (
            isinstance(density, (int, float)) and 0.0 <= density <= 1.0
        ):
            raise ValueError(f"expected_density must be in [0, 1], got {density!r}.")
        for name in ("expected_calls", "expected_batch"):
            value = getattr(self, name)
            if value is not None and not (isinstance(value, int) and value > 0):
                raise ValueError(f"{name} must be a positive int, got {value!r}.")

    def replace(self, **changes) -> "Hints":
        """Return a copy with selected expectations changed."""
        return replace(self, **changes)


@dataclass(frozen=True)
class EdgeMap:
    """Correspondence between the entries of two representations.

    A representation change (sorting, duplicate reduction) maps every old
    entry ``e`` to a new entry ``target[e]``. Every edge-aligned array (values,
    group ids, receptors, delays, plasticity state) must be moved with the same
    map, so the map is computed once and applied to each of them.

    Args:
        target: ``[n_old]`` new position of every old entry.
        n_new: Number of entries after the change.
    """

    target: Tensor
    n_new: int

    @property
    def n_old(self) -> int:
        return int(self.target.shape[0])

    @property
    def is_permutation(self) -> bool:
        """Whether no entries were merged."""
        return self.n_old == self.n_new

    def source(self) -> Tensor:
        """``[n_new]`` index of one old entry for every new entry.

        For merged duplicates this is the last old entry of the group.
        """
        old = torch.arange(self.n_old, device=self.target.device)
        if self.is_permutation:
            src = torch.empty_like(old)
            src[self.target] = old
            return src
        # Deterministic on every device (an indexed assignment with repeated
        # targets is not).
        src = self.target.new_zeros(self.n_new)
        return src.scatter_reduce(0, self.target, old, "amax", include_self=False)

    def sum(self, values: Tensor, dim: int = -1) -> Tensor:
        """Move additive data (weights); merged entries are summed.

        Differentiable: the gradient of a merged entry flows to every old
        entry that was merged into it.

        Args:
            values: Tensor whose axis ``dim`` has length ``n_old``.
            dim: Edge axis.
        """
        if self.is_permutation:
            return values.index_select(dim, self.source())
        shape = list(values.shape)
        shape[dim] = self.n_new
        return values.new_zeros(shape).index_add(dim, self.target, values)

    def take(self, meta: Tensor, dim: int = 0, check: bool = True) -> Tensor:
        """Move categorical metadata (group, receptor, delay ids).

        Args:
            meta: Tensor whose axis ``dim`` has length ``n_old``.
            dim: Edge axis.
            check: Raise if merged entries carry different metadata, because
                no single value can then represent the merged entry.

        Raises:
            ValueError: If ``check`` and merged duplicates disagree.
        """
        out = meta.index_select(dim, self.source())
        if check and not self.is_permutation:
            if not torch.equal(out.index_select(dim, self.target), meta):
                raise ValueError(
                    "Duplicate entries carry different metadata and cannot be "
                    "merged; split them into separate connections or make the "
                    "metadata equal first."
                )
        return out

    def compose(self, then: "EdgeMap") -> "EdgeMap":
        """Map equivalent to applying ``self`` and then ``then``."""
        return EdgeMap(then.target[self.target], then.n_new)

    @staticmethod
    def identity(n: int, device=None) -> "EdgeMap":
        return EdgeMap(torch.arange(n, device=device), n)
