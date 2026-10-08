"""NEST-style connection rules: *which* pre/post pairs are connected.

A :class:`ConnectionRule` is model semantics. It is not a sparse format and it
does not decide how a connection is executed: the same rule can be realised
once as an explicit sparse matrix, regenerated at runtime, or (for
:class:`OneToOne` / :class:`AllToAll`) replaced by a structured operator that
never stores a matrix (:meth:`ConnectionRule.as_operator`).

Rules produce edge lists in population-local indices:

>>> import torch
>>> from btorch.models.connection.rule import FixedIndegree
>>> g = torch.Generator().manual_seed(0)
>>> pre, post = FixedIndegree(3).edges(n_pre=10, n_post=4, generator=g)
>>> torch.bincount(post, minlength=4)
tensor([3, 3, 3, 3])

The standard operator of a projection has shape ``(n_post, n_pre)`` with
``A[post, pre]`` (so ``current = A @ spikes``); building it from
``(pre, post)`` is the job of the connection layer, not of the rule.

Terms (as in NEST): an *autapse* is a connection of a neuron onto itself,
which is only defined when a population projects onto itself; *multapses* are
several connections between the same ordered pair.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any, Literal

import torch
from torch import Tensor

from btorch.sparse import as_sparse
from btorch.sparse.operator import (
    ConstantOperator,
    DiagonalOperator,
    LinearOperator,
)


# Upper bound on the number of elements of a temporary tensor in the chunked
# samplers (2**24 float64 values are 128 MB).
_CHUNK_ELEMENTS = 1 << 24


def _check_sizes(n_pre: int, n_post: int) -> tuple[int, int]:
    for name, n in (("n_pre", n_pre), ("n_post", n_post)):
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"{name} must be a non-negative int, got {n!r}.")
    return n_pre, n_post


def _check_autapses(rule: "ConnectionRule", n_pre: int, n_post: int) -> None:
    """An autapse pairs index ``i`` of both populations: they must be one."""
    if not rule.allow_autapses and n_pre != n_post:
        raise ValueError(
            f"{type(rule).__name__}(allow_autapses=False) is only defined when "
            "a population projects onto itself (pre index i and post index i "
            f"are the same neuron), got n_pre={n_pre} and n_post={n_post}. Use "
            "allow_autapses=True between different populations."
        )


def _sample_device(generator: torch.Generator | None, device) -> torch.device:
    """Device on which random numbers are drawn.

    ``torch`` requires the generator and the sampled tensor to live on one
    device, so sampling happens on the generator device and the result is
    moved afterwards. A seeded CPU generator therefore gives the same edges
    on every target device.
    """
    if generator is not None:
        return generator.device
    return torch.device(device) if device is not None else torch.device("cpu")


def _empty(device) -> tuple[Tensor, Tensor]:
    e = torch.empty(0, dtype=torch.long, device=device)
    return e, e.clone()


def _sample_distinct(
    n_rows: int,
    n_choices: int,
    k: int,
    exclude_self: bool,
    generator: torch.Generator | None,
    device: torch.device,
) -> Tensor:
    """Draw ``k`` distinct integers in ``[0, n_choices)`` for each row.

    Every row gets a uniformly random ``k``-subset (in uniformly random
    order), independently of the other rows. With ``exclude_self`` the value
    equal to the row index is never drawn. Both methods below are exact.

    - Sparse regime (``k**2 <= n``): draw ``k`` values with replacement and
      redraw the *whole* row while it contains a repeat. Conditional on having
      no repeat, an i.i.d. tuple is a uniform ordered ``k``-subset. A row is
      rejected with probability below ``k**2 / (2 n) <= 1/2``, so the number
      of rounds is logarithmic in ``n_rows`` and memory is ``O(n_rows * k)``.
    - Dense regime: give every candidate an i.i.d. uniform score and keep the
      ``k`` smallest (a uniform random permutation prefix). Rows are processed
      in chunks so that the score matrix stays below ``_CHUNK_ELEMENTS``.

    Returns:
        ``[n_rows, k]`` int64 tensor on ``device``.
    """
    n = n_choices - int(exclude_self)
    out = torch.empty(n_rows, k, dtype=torch.long, device=device)
    if n_rows == 0 or k == 0:
        return out
    if k * k <= n:
        todo = torch.arange(n_rows, device=device)
        while todo.numel() > 0:
            draw = torch.randint(
                n, (todo.numel(), k), generator=generator, device=device
            )
            ordered = draw.sort(dim=1).values
            ok = (ordered[:, 1:] != ordered[:, :-1]).all(dim=1)
            out[todo[ok]] = draw[ok]
            todo = todo[~ok]
        if exclude_self:
            # Values in [0, n_choices - 1) skip over the row's own index.
            out += out >= torch.arange(n_rows, device=device)[:, None]
        return out
    chunk = max(1, _CHUNK_ELEMENTS // max(n_choices, 1))
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        scores = torch.rand(stop - start, n_choices, generator=generator, device=device)
        if exclude_self:
            rows = torch.arange(start, stop, device=device)
            scores[rows - start, rows] = math.inf
        out[start:stop] = scores.topk(k, dim=1, largest=False).indices
    return out


class ConnectionRule:
    """Rule deciding which pre/post pairs of a projection are connected.

    Subclasses implement :meth:`edges` and :meth:`expected_nnz`. A rule holds
    only its parameters: population sizes are passed at call time, so one rule
    object can be reused for several projections.

    Randomness comes exclusively from the ``generator`` argument of
    :meth:`edges`. With a seeded ``torch.Generator`` the result is
    reproducible; random numbers are drawn on the generator's device and the
    edges are then moved to ``device``. ``generator=None`` uses the global
    PyTorch RNG.
    """

    allow_autapses: bool = True

    def edges(
        self,
        n_pre: int,
        n_post: int,
        *,
        generator: torch.Generator | None = None,
        device=None,
    ) -> tuple[Tensor, Tensor]:
        """Generate the connected pairs.

        Args:
            n_pre: Size of the presynaptic (source) population.
            n_post: Size of the postsynaptic (target) population.
            generator: Source of randomness (ignored by deterministic rules).
            device: Device of the returned tensors. ``None`` means CPU (or
                the generator's device) for generated rules and the device
                of the stored data for :class:`DistanceDependent`,
                :class:`FromEdges` and :class:`FromSparse`.

        Returns:
            ``(pre, post)``: two int64 tensors of equal length ``E``; edge
            ``e`` connects source ``pre[e]`` to target ``post[e]``. The edge
            order is rule-specific.
        """
        raise NotImplementedError

    def values(self, n_pre: int, n_post: int, *, device=None) -> Tensor | None:
        """Values carried by the rule, aligned with :meth:`edges`.

        Most rules only describe structure and return ``None``; weights then
        come from the synapse specification. Rules built from existing data
        (:class:`FromEdges`, :class:`FromSparse`) return ``[E, ...]`` values.
        """
        return None

    def expected_nnz(self, n_pre: int, n_post: int) -> float:
        """Expected number of edges, without generating them."""
        raise NotImplementedError

    def as_operator(
        self,
        n_pre: int,
        n_post: int,
        *,
        dtype: torch.dtype | None = None,
        device=None,
    ) -> LinearOperator | None:
        """Structured realisation of the rule with unit weights, if any.

        Returns:
            A :class:`~btorch.sparse.operator.LinearOperator` of shape
            ``(n_post, n_pre)`` equal to the 0/1 matrix of :meth:`edges`
            without storing it, or ``None`` if the rule has no such closed
            form (the default) and must be realised from its edges.
        """
        return None

    def __repr__(self) -> str:
        args = ", ".join(
            f"{k}={v!r}"
            for k, v in vars(self).items()
            if not k.startswith("_") and not isinstance(v, Tensor) and not callable(v)
        )
        return f"{type(self).__name__}({args})"


class OneToOne(ConnectionRule):
    """Connect source ``i`` to target ``i``; needs equal population sizes.

    The structured realisation is the identity
    (:class:`~btorch.sparse.operator.DiagonalOperator`).
    """

    @staticmethod
    def _check(n_pre: int, n_post: int) -> int:
        _check_sizes(n_pre, n_post)
        if n_pre != n_post:
            raise ValueError(
                "OneToOne needs populations of the same size, got "
                f"n_pre={n_pre} and n_post={n_post}."
            )
        return n_pre

    def edges(self, n_pre, n_post, *, generator=None, device=None):
        index = torch.arange(self._check(n_pre, n_post), device=device)
        return index, index.clone()

    def expected_nnz(self, n_pre, n_post):
        return float(self._check(n_pre, n_post))

    def as_operator(self, n_pre, n_post, *, dtype=None, device=None):
        n = self._check(n_pre, n_post)
        return DiagonalOperator(torch.ones(n, dtype=dtype, device=device))


class AllToAll(ConnectionRule):
    """Connect every source to every target.

    Edges are ordered by target, then source.

    The structured realisation is a
    :class:`~btorch.sparse.operator.ConstantOperator` (minus the identity
    without autapses), so a projection can apply an all-to-all connection in
    ``O(n_pre + n_post)`` without a dense matrix.

    Args:
        allow_autapses: Keep the pairs ``pre == post``. ``False`` requires
            ``n_pre == n_post``.
    """

    def __init__(self, allow_autapses: bool = True):
        self.allow_autapses = bool(allow_autapses)

    def edges(self, n_pre, n_post, *, generator=None, device=None):
        _check_sizes(n_pre, n_post)
        _check_autapses(self, n_pre, n_post)
        post = torch.arange(n_post, device=device).repeat_interleave(n_pre)
        pre = torch.arange(n_pre, device=device).repeat(n_post)
        if not self.allow_autapses:
            keep = pre != post
            pre, post = pre[keep], post[keep]
        return pre, post

    def expected_nnz(self, n_pre, n_post):
        _check_sizes(n_pre, n_post)
        _check_autapses(self, n_pre, n_post)
        return float(n_pre * n_post - (0 if self.allow_autapses else n_pre))

    def as_operator(self, n_pre, n_post, *, dtype=None, device=None):
        _check_sizes(n_pre, n_post)
        _check_autapses(self, n_pre, n_post)
        ones = ConstantOperator((n_post, n_pre), 1.0, dtype=dtype, device=device)
        if self.allow_autapses:
            return ones
        return ones - DiagonalOperator(torch.ones(n_pre, dtype=dtype, device=device))


class _FixedDegree(ConnectionRule):
    """Shared implementation of :class:`FixedIndegree` /
    :class:`FixedOutdegree`."""

    def __init__(
        self, k: int, allow_autapses: bool = True, allow_multapses: bool = True
    ):
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError(f"The degree k must be a non-negative int, got {k!r}.")
        self.k = k
        self.allow_autapses = bool(allow_autapses)
        self.allow_multapses = bool(allow_multapses)

    def _draw(
        self,
        n_rows: int,
        n_choices: int,
        what: str,
        generator: torch.Generator | None,
        device,
    ) -> tuple[Tensor, Tensor]:
        """Give each of ``n_rows`` neurons ``k`` partners out of ``n_choices``.

        Returns:
            ``(rows, partners)``, both ``[n_rows * k]``, grouped by row.
        """
        k = self.k
        exclude = not self.allow_autapses
        available = n_choices - int(exclude)
        if n_rows > 0 and k > 0:
            if available <= 0:
                raise ValueError(
                    f"{type(self).__name__}(k={k}): there is no eligible {what} "
                    f"neuron to connect to (population size {n_choices}, "
                    f"allow_autapses={self.allow_autapses})."
                )
            if not self.allow_multapses and k > available:
                raise ValueError(
                    f"{type(self).__name__}(k={k}, allow_multapses=False) needs "
                    f"k distinct {what} neurons per neuron but only {available} "
                    f"are eligible (population size {n_choices}, "
                    f"allow_autapses={self.allow_autapses}). Reduce k or allow "
                    "multapses."
                )
        sample_device = _sample_device(generator, device)
        rows = torch.arange(n_rows, device=sample_device)
        if n_rows == 0 or k == 0:
            partners = torch.empty(n_rows, k, dtype=torch.long, device=sample_device)
        elif self.allow_multapses:
            # k independent uniform draws per row (sampling with replacement).
            partners = torch.randint(
                available, (n_rows, k), generator=generator, device=sample_device
            )
            if exclude:
                # Uniform on the n_choices - 1 values other than the row index.
                partners += partners >= rows[:, None]
        else:
            partners = _sample_distinct(
                n_rows, n_choices, k, exclude, generator, sample_device
            )
        rows = rows.repeat_interleave(k)
        return rows.to(device=device), partners.reshape(-1).to(device=device)


class FixedIndegree(_FixedDegree):
    """Every target receives exactly ``k`` connections from random sources.

    Sampling is exact: each target independently draws its sources uniformly
    from the eligible ones, with replacement if ``allow_multapses`` (``k``
    i.i.d. draws) and without replacement otherwise (a uniform ``k``-subset;
    whole-row rejection when ``k**2 <= n_pre``, random-score top-``k`` in
    chunks otherwise). Memory is ``O(n_post * k)`` in the first case and
    bounded chunks of ``n_pre`` scores in the second. Edges are grouped by
    target.

    Args:
        k: In-degree of every target neuron.
        allow_autapses: Allow ``pre == post``. ``False`` requires
            ``n_pre == n_post``.
        allow_multapses: Allow several connections from the same source to a
            target. ``False`` requires ``k <= n_pre`` (``n_pre - 1`` without
            autapses).
    """

    def edges(self, n_pre, n_post, *, generator=None, device=None):
        _check_sizes(n_pre, n_post)
        _check_autapses(self, n_pre, n_post)
        post, pre = self._draw(n_post, n_pre, "presynaptic", generator, device)
        return pre, post

    def expected_nnz(self, n_pre, n_post):
        _check_sizes(n_pre, n_post)
        return float(self.k * n_post)


class FixedOutdegree(_FixedDegree):
    """Every source makes exactly ``k`` connections onto random targets.

    The mirror image of :class:`FixedIndegree` (same sampling methods and
    exactness, with the roles of the populations swapped). Edges are grouped
    by source.

    Args:
        k: Out-degree of every source neuron.
        allow_autapses: Allow ``pre == post``. ``False`` requires
            ``n_pre == n_post``.
        allow_multapses: Allow several connections onto the same target from
            a source. ``False`` requires ``k <= n_post`` (``n_post - 1``
            without autapses).
    """

    def edges(self, n_pre, n_post, *, generator=None, device=None):
        _check_sizes(n_pre, n_post)
        _check_autapses(self, n_pre, n_post)
        return self._draw(n_pre, n_post, "postsynaptic", generator, device)

    def expected_nnz(self, n_pre, n_post):
        _check_sizes(n_pre, n_post)
        return float(self.k * n_pre)


class PairwiseBernoulli(ConnectionRule):
    """Connect every pair independently with probability ``p``.

    No ``n_post x n_pre`` mask is built. The pairs are laid out in the
    flattened order ``post * n_pre + pre`` and the sampler jumps from one
    connected pair to the next: the number of unconnected pairs skipped before
    each connection is geometric,

    .. math::
        G = \\left\\lfloor \\frac{\\ln U}{\\ln(1 - p)} \\right\\rfloor,
        \\qquad U \\sim \\mathcal{U}(0, 1], \\qquad
        P(G \\ge g) = (1 - p)^g .

    This reproduces the i.i.d. Bernoulli field exactly (so the edge count is
    ``Binomial(n_pre * n_post, p)``) up to the resolution of the float64
    uniform draws; time and memory are proportional to the number of edges.
    Positions are strictly increasing, so there are no multapses and the
    edges are ordered by target, then source. Without autapses the pairs
    ``pre == post`` are dropped afterwards, which leaves the other pairs
    independent with probability ``p``.

    Args:
        p: Connection probability in ``[0, 1]``.
        allow_autapses: Allow ``pre == post``. ``False`` requires
            ``n_pre == n_post``.
    """

    def __init__(self, p: float, allow_autapses: bool = True):
        p = float(p)
        if not 0.0 <= p <= 1.0:  # also rejects NaN
            raise ValueError(f"The probability p must be in [0, 1], got {p}.")
        self.p = p
        self.allow_autapses = bool(allow_autapses)

    def edges(self, n_pre, n_post, *, generator=None, device=None):
        _check_sizes(n_pre, n_post)
        _check_autapses(self, n_pre, n_post)
        total = n_pre * n_post
        p = self.p
        if total == 0 or p == 0.0:
            return _empty(device)
        if p == 1.0:
            return AllToAll(self.allow_autapses).edges(n_pre, n_post, device=device)

        sample_device = _sample_device(generator, device)
        log_q = math.log1p(-p)
        # Expected count plus six standard deviations: one batch almost always
        # reaches the end; the loop below is still correct if it does not.
        mean = total * p
        batch = int(mean + 6.0 * math.sqrt(mean * (1.0 - p)) + 16)
        batch = min(batch, _CHUNK_ELEMENTS)
        chunks = []
        last = -1  # flattened position of the last connected pair
        while True:
            u = 1.0 - torch.rand(
                batch, dtype=torch.float64, generator=generator, device=sample_device
            )
            # Gaps beyond the end are clamped so the cumulative sum cannot
            # overflow; clamping does not change which positions are < total.
            gaps = torch.floor(torch.log(u) / log_q).clamp_(max=total).long()
            position = last + torch.cumsum(gaps + 1, dim=0)
            if int(position[-1]) >= total:
                chunks.append(position[position < total])
                break
            chunks.append(position)
            last = int(position[-1])
        flat = torch.cat(chunks)
        post = torch.div(flat, n_pre, rounding_mode="floor")
        pre = flat - post * n_pre
        if not self.allow_autapses:
            keep = pre != post
            pre, post = pre[keep], post[keep]
        return pre.to(device=device), post.to(device=device)

    def expected_nnz(self, n_pre, n_post):
        _check_sizes(n_pre, n_post)
        _check_autapses(self, n_pre, n_post)
        pairs = n_pre * n_post - (0 if self.allow_autapses else n_pre)
        return self.p * pairs

    def as_operator(self, n_pre, n_post, *, dtype=None, device=None):
        # p == 1 is all-to-all; every other p is genuinely random.
        if self.p == 1.0:
            return AllToAll(self.allow_autapses).as_operator(
                n_pre, n_post, dtype=dtype, device=device
            )
        return None


class DistanceDependent(ConnectionRule):
    """Connect pairs with a probability that depends on their distance.

    Every pair is an independent Bernoulli trial with probability
    ``probability(||post_pos - pre_pos||)`` (Euclidean distance), which is
    exact. Target neurons are processed in chunks so that the temporary
    ``[chunk, n_pre]`` distance block stays bounded.

    With ``max_distance`` only pairs within that distance are candidates
    (the probability is treated as zero beyond it); they are found with a
    ``scipy.spatial.cKDTree`` so the cost scales with the number of candidate
    pairs instead of ``n_pre * n_post``.

    Edges are ordered by target, then source.

    Args:
        pre_pos: ``[n_pre, D]`` positions of the source neurons.
        post_pos: ``[n_post, D]`` positions of the target neurons.
        probability: Elementwise function mapping a 1-D tensor of distances
            to connection probabilities in ``[0, 1]``.
        max_distance: Optional cutoff radius (inclusive).
        allow_autapses: Allow ``pre == post``. ``False`` requires
            ``n_pre == n_post`` (the positions are then expected to describe
            the same population).

    Examples:
        >>> pos = torch.rand(100, 2)
        >>> rule = DistanceDependent(
        ...     pos, pos, lambda d: torch.exp(-d / 0.1), max_distance=0.3,
        ...     allow_autapses=False,
        ... )
        >>> pre, post = rule.edges(100, 100, generator=torch.Generator())
    """

    def __init__(
        self,
        pre_pos: Tensor,
        post_pos: Tensor,
        probability: Callable[[Tensor], Tensor],
        max_distance: float | None = None,
        allow_autapses: bool = True,
    ):
        pre_pos = torch.as_tensor(pre_pos)
        post_pos = torch.as_tensor(post_pos)
        if pre_pos.ndim != 2 or post_pos.ndim != 2:
            raise ValueError(
                "Positions must be [n, D] tensors, got shapes "
                f"{tuple(pre_pos.shape)} and {tuple(post_pos.shape)}."
            )
        if pre_pos.shape[1] != post_pos.shape[1]:
            raise ValueError(
                "pre_pos and post_pos need the same number of spatial "
                f"dimensions, got {pre_pos.shape[1]} and {post_pos.shape[1]}."
            )
        if not pre_pos.is_floating_point():
            pre_pos = pre_pos.to(torch.get_default_dtype())
        if not callable(probability):
            raise TypeError("probability must be a callable distance -> probability.")
        if max_distance is not None and not max_distance >= 0:
            raise ValueError(f"max_distance must be >= 0, got {max_distance}.")
        self._pre_pos = pre_pos.detach()
        self._post_pos = post_pos.detach().to(pre_pos)
        self._probability = probability
        self.max_distance = None if max_distance is None else float(max_distance)
        self.allow_autapses = bool(allow_autapses)

    def _check(self, n_pre: int, n_post: int) -> None:
        _check_sizes(n_pre, n_post)
        if n_pre != self._pre_pos.shape[0] or n_post != self._post_pos.shape[0]:
            raise ValueError(
                f"The rule holds positions of {self._pre_pos.shape[0]} "
                f"presynaptic and {self._post_pos.shape[0]} postsynaptic "
                f"neurons but was applied to n_pre={n_pre}, n_post={n_post}."
            )
        _check_autapses(self, n_pre, n_post)

    def _candidates(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        """Yield ``(pre, post, probability)`` of the candidate pairs of one
        chunk of targets, ordered by target then source."""
        pre_pos, post_pos = self._pre_pos, self._post_pos
        n_pre, n_post = pre_pos.shape[0], post_pos.shape[0]
        device = pre_pos.device
        chunk = max(1, _CHUNK_ELEMENTS // max(n_pre, 1))
        tree = None
        if self.max_distance is not None:
            from scipy.spatial import cKDTree

            pre_np = pre_pos.cpu().numpy()
            post_np = post_pos.cpu().numpy()
            tree = cKDTree(pre_np)
        for start in range(0, n_post, chunk):
            stop = min(start + chunk, n_post)
            if tree is None:
                dist = torch.cdist(post_pos[start:stop], pre_pos).reshape(-1)
                post = torch.arange(start, stop, device=device).repeat_interleave(n_pre)
                pre = torch.arange(n_pre, device=device).repeat(stop - start)
            else:
                # All pairs within the radius as records (i, j, distance),
                # including pairs at distance zero.
                found = cKDTree(post_np[start:stop]).sparse_distance_matrix(
                    tree, self.max_distance, output_type="ndarray"
                )
                post = torch.as_tensor(found["i"], dtype=torch.long) + start
                pre = torch.as_tensor(found["j"], dtype=torch.long)
                dist = torch.as_tensor(found["v"]).to(pre_pos.dtype)
                # The tree traversal order is an implementation detail: sort
                # so the random draws below pair up deterministically.
                order = torch.argsort(post * n_pre + pre)
                post, pre, dist = (
                    t[order].to(device=device) for t in (post, pre, dist)
                )
            if not self.allow_autapses:
                keep = pre != post
                pre, post, dist = pre[keep], post[keep], dist[keep]
            prob = self._probability(dist)
            if prob.shape != dist.shape:
                raise ValueError(
                    "probability must be elementwise: it returned shape "
                    f"{tuple(prob.shape)} for distances of shape "
                    f"{tuple(dist.shape)}."
                )
            if prob.numel() and not bool(((prob >= 0) & (prob <= 1)).all()):
                raise ValueError(
                    "probability returned values outside [0, 1] (range "
                    f"[{float(prob.min())}, {float(prob.max())}])."
                )
            yield pre, post, prob

    def edges(self, n_pre, n_post, *, generator=None, device=None):
        self._check(n_pre, n_post)
        pos_device = self._pre_pos.device
        sample_device = _sample_device(generator, pos_device)
        if device is None:
            device = pos_device
        pres, posts = [], []
        for pre, post, prob in self._candidates():
            u = torch.rand(
                prob.shape, dtype=prob.dtype, generator=generator, device=sample_device
            )
            keep = u.to(pos_device) < prob
            pres.append(pre[keep])
            posts.append(post[keep])
        if not pres:
            return _empty(device)
        return torch.cat(pres).to(device=device), torch.cat(posts).to(device=device)

    def expected_nnz(self, n_pre, n_post):
        """Sum of the pair probabilities.

        Unlike the other rules this is not ``O(1)``: it evaluates the
        probability of every candidate pair (in chunks).
        """
        self._check(n_pre, n_post)
        return float(sum(float(prob.sum()) for _, _, prob in self._candidates()))


class FromEdges(ConnectionRule):
    """Use an explicit edge list.

    The edges are returned as given (same order, multapses kept).

    Args:
        pre: ``[E]`` source index of each edge.
        post: ``[E]`` target index of each edge.
        values: Optional ``[E, ...]`` values (for example weights) aligned
            with the edges.
    """

    def __init__(self, pre: Tensor, post: Tensor, values: Tensor | None = None):
        pre = torch.as_tensor(pre)
        post = torch.as_tensor(post, device=pre.device)
        for name, index in (("pre", pre), ("post", post)):
            if index.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {tuple(index.shape)}.")
            if index.is_floating_point() or index.dtype == torch.bool:
                raise TypeError(f"{name} must be an integer tensor, got {index.dtype}.")
        if pre.shape != post.shape:
            raise ValueError(
                f"pre and post need the same length, got {pre.shape[0]} and "
                f"{post.shape[0]}."
            )
        if values is not None:
            values = torch.as_tensor(values)
            if values.ndim < 1 or values.shape[0] != pre.shape[0]:
                raise ValueError(
                    f"values must have shape [{pre.shape[0]}, ...] (one entry "
                    f"per edge), got {tuple(values.shape)}."
                )
        self._pre = pre.long()
        self._post = post.long()
        self._values = values
        self._max_pre = int(pre.max()) if pre.numel() else -1
        self._max_post = int(post.max()) if post.numel() else -1
        if pre.numel() and (int(pre.min()) < 0 or int(post.min()) < 0):
            raise ValueError("Edge indices must be non-negative.")

    def _check(self, n_pre: int, n_post: int) -> None:
        _check_sizes(n_pre, n_post)
        if self._max_pre >= n_pre or self._max_post >= n_post:
            raise ValueError(
                f"The edge list refers to presynaptic index {self._max_pre} and "
                f"postsynaptic index {self._max_post}, which do not fit "
                f"populations of size n_pre={n_pre}, n_post={n_post}."
            )

    def edges(self, n_pre, n_post, *, generator=None, device=None):
        self._check(n_pre, n_post)
        return self._pre.to(device=device), self._post.to(device=device)

    def values(self, n_pre, n_post, *, device=None):
        self._check(n_pre, n_post)
        return None if self._values is None else self._values.to(device=device)

    def expected_nnz(self, n_pre, n_post):
        self._check(n_pre, n_post)
        return float(self._pre.numel())


class FromSparse(ConnectionRule):
    """Use the stored entries of a sparse matrix as edges and values.

    The matrix is interpreted through :func:`btorch.sparse.as_sparse` (the
    single conversion path), so a btorch ``Sparse``, a PyTorch sparse tensor
    and a SciPy sparse array/matrix all work, and the conversion itself never
    transposes. Which matrix axis is the source population is stated only by
    ``orientation``:

    - ``"post_pre"``: the standard operator, shape
      ``(n_post, n_pre)``, ``A[post, pre]``, used as ``y = A @ x``.
    - ``"pre_post"`` (default, as in ``SparseConnection.from_adjacency``):
      rows are sources, shape ``(n_pre, n_post)``,
      ``A[pre, post]``, the connectome convention ``y = x @ A``.

    Every stored entry is one edge (explicit zeros and duplicate coordinates
    included), in the COO order of the matrix; :meth:`values` returns the
    stored values in the same order and keeps their autograd history.

    Args:
        A: Sparse matrix without batch dimensions.
        orientation: ``"post_pre"`` or ``"pre_post"``.

    Raises:
        TypeError: For dense input (never sparsified implicitly).
    """

    def __init__(
        self, A: Any, orientation: Literal["pre_post", "post_pre"] = "pre_post"
    ):
        if orientation not in ("post_pre", "pre_post"):
            raise ValueError(
                "orientation must be 'post_pre' (matrix is (n_post, n_pre), "
                "y = A @ x) or 'pre_post' (matrix is (n_pre, n_post), "
                f"y = x @ A), got {orientation!r}."
            )
        # SciPy data is float64 by NumPy convention; connections built from
        # it use the default dtype unless one is requested.
        self.from_scipy = not isinstance(A, Tensor) and not hasattr(A, "sparse_shape")
        matrix = as_sparse(A)
        if matrix.batch_dim() != 0 or matrix.sparse_dim() != 2:
            raise ValueError(
                "FromSparse needs a single sparse matrix, got shape "
                f"{matrix.shape} with {matrix.batch_dim()} batch and "
                f"{matrix.sparse_dim()} sparse dimensions."
            )
        self.orientation = orientation
        self._matrix = matrix
        # Convert once so that edges() and values() share one entry order.
        self._coo = matrix.tocoo()

    @property
    def matrix(self):
        """The matrix as given (a :class:`~btorch.sparse.Sparse`), in its own
        orientation."""
        return self._matrix

    def _check(self, n_pre: int, n_post: int) -> None:
        _check_sizes(n_pre, n_post)
        expected = (
            (n_post, n_pre) if self.orientation == "post_pre" else (n_pre, n_post)
        )
        got = tuple(self._matrix.sparse_shape)
        if got != expected:
            layout = (
                "(n_post, n_pre)"
                if self.orientation == "post_pre"
                else "(n_pre, n_post)"
            )
            raise ValueError(
                f"FromSparse(orientation={self.orientation!r}) expects a matrix "
                f"of shape {layout} = {expected} for n_pre={n_pre}, "
                f"n_post={n_post}, got {got}. The matrix is never transposed "
                "implicitly: pass the other orientation if its rows are the "
                f"{'sources' if self.orientation == 'post_pre' else 'targets'}."
            )

    def edges(self, n_pre, n_post, *, generator=None, device=None):
        self._check(n_pre, n_post)
        row, col = self._coo.row, self._coo.col
        if device is None:
            device = row.device
        pre, post = (col, row) if self.orientation == "post_pre" else (row, col)
        return pre.to(device=device), post.to(device=device)

    def values(self, n_pre, n_post, *, device=None):
        self._check(n_pre, n_post)
        return self._coo.values().to(device=device)

    def expected_nnz(self, n_pre, n_post):
        self._check(n_pre, n_post)
        return float(self._coo.nnz)


__all__ = [
    "AllToAll",
    "ConnectionRule",
    "DistanceDependent",
    "FixedIndegree",
    "FixedOutdegree",
    "FromEdges",
    "FromSparse",
    "OneToOne",
    "PairwiseBernoulli",
]
