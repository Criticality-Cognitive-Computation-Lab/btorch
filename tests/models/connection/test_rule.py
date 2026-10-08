"""NEST-style connection rules.

A ``ConnectionRule`` answers one question: *which* presynaptic (source)
neurons connect to which postsynaptic (target) neurons. It returns an edge
list ``(pre, post)`` in population-local indices and nothing else: no sparse
format, no execution strategy.

The tests check, for every rule,

- the structural guarantee it makes (exact degrees, no autapses, no
  multapses, statistics of the edge count),
- reproducibility: all randomness comes from an explicit
  ``torch.Generator``,
- clear validation errors,

and, for the rules that import existing data, that the matrix orientation is
stated once and never silently flipped.

Populations have different sizes wherever the rule allows it, so swapping
``pre`` and ``post`` produces out-of-range indices or wrong counts instead of
passing by accident.
"""

import math

import numpy as np
import pytest
import scipy.sparse
import torch

from btorch import sparse
from btorch.models.connection.rule import (
    AllToAll,
    ConnectionRule,
    DistanceDependent,
    FixedIndegree,
    FixedOutdegree,
    FromEdges,
    FromSparse,
    OneToOne,
    PairwiseBernoulli,
)
from btorch.sparse.operator import LinearOperator


N_PRE, N_POST = 12, 7


def _gen(seed=0):
    """A seeded generator: the only source of randomness a rule may use."""
    return torch.Generator().manual_seed(seed)


def _adjacency(pre, post, n_pre, n_post):
    """Dense count matrix ``[n_post, n_pre]`` of an edge list.

    Entry ``[j, i]`` is the number of edges from source ``i`` to target ``j``
    (the standard operator orientation, ``current = A @ spikes``).
    """
    A = torch.zeros(n_post, n_pre, dtype=torch.long)
    A.index_put_((post, pre), torch.ones_like(pre), accumulate=True)
    return A


def _check_edge_list(pre, post, n_pre, n_post):
    """The basic contract of ``edges``: two aligned int64 index vectors."""
    assert pre.dtype == torch.long and post.dtype == torch.long
    assert pre.ndim == 1 and pre.shape == post.shape
    if pre.numel():
        assert 0 <= int(pre.min()) and int(pre.max()) < n_pre
        assert 0 <= int(post.min()) and int(post.max()) < n_post


# --------------------------------------------------------------- determinism
def test_one_to_one():
    """Source ``i`` connects to target ``i``; sizes must match."""
    rule = OneToOne()
    pre, post = rule.edges(5, 5)
    _check_edge_list(pre, post, 5, 5)
    assert torch.equal(pre, torch.arange(5)) and torch.equal(post, pre)
    assert rule.expected_nnz(5, 5) == 5
    assert rule.values(5, 5) is None  # a structural rule carries no weights
    with pytest.raises(ValueError, match="same size"):
        rule.edges(N_PRE, N_POST)


@pytest.mark.parametrize("allow_autapses", [True, False])
def test_all_to_all(allow_autapses):
    """Every pair once; without autapses the diagonal is missing."""
    n = 6
    rule = AllToAll(allow_autapses=allow_autapses)
    pre, post = rule.edges(n, n)
    _check_edge_list(pre, post, n, n)
    expected = torch.ones(n, n, dtype=torch.long)
    if not allow_autapses:
        expected -= torch.eye(n, dtype=torch.long)
    assert torch.equal(_adjacency(pre, post, n, n), expected)
    assert rule.expected_nnz(n, n) == pre.numel()


def test_all_to_all_between_different_populations():
    pre, post = AllToAll().edges(N_PRE, N_POST)
    assert torch.equal(
        _adjacency(pre, post, N_PRE, N_POST),
        torch.ones(N_POST, N_PRE, dtype=torch.long),
    )
    assert AllToAll().expected_nnz(N_PRE, N_POST) == N_PRE * N_POST


# ------------------------------------------------------------- fixed degrees
@pytest.mark.parametrize("allow_multapses", [True, False])
@pytest.mark.parametrize("k", [0, 1, 5, N_PRE])
def test_fixed_indegree_gives_every_target_exactly_k_inputs(k, allow_multapses):
    """In-degree is exactly ``k`` for every target; out-degree is free.

    ``k = N_PRE`` without multapses is the extreme case: every target must
    receive from every source exactly once.
    """
    rule = FixedIndegree(k, allow_multapses=allow_multapses)
    pre, post = rule.edges(N_PRE, N_POST, generator=_gen())
    _check_edge_list(pre, post, N_PRE, N_POST)
    A = _adjacency(pre, post, N_PRE, N_POST)
    # Row sums of the (post, pre) matrix are in-degrees.
    assert torch.equal(A.sum(1), torch.full((N_POST,), k))
    assert rule.expected_nnz(N_PRE, N_POST) == k * N_POST == pre.numel()
    if not allow_multapses:
        assert int(A.max()) <= 1
        if k == N_PRE:
            assert torch.equal(A, torch.ones_like(A))


@pytest.mark.parametrize("allow_multapses", [True, False])
@pytest.mark.parametrize("k", [0, 1, 4, N_POST])
def test_fixed_outdegree_gives_every_source_exactly_k_outputs(k, allow_multapses):
    """The mirror image: column sums (out-degrees) are exactly ``k``."""
    rule = FixedOutdegree(k, allow_multapses=allow_multapses)
    pre, post = rule.edges(N_PRE, N_POST, generator=_gen())
    _check_edge_list(pre, post, N_PRE, N_POST)
    A = _adjacency(pre, post, N_PRE, N_POST)
    assert torch.equal(A.sum(0), torch.full((N_PRE,), k))
    assert rule.expected_nnz(N_PRE, N_POST) == k * N_PRE == pre.numel()
    if not allow_multapses:
        assert int(A.max()) <= 1


@pytest.mark.parametrize("rule_cls", [FixedIndegree, FixedOutdegree])
@pytest.mark.parametrize("allow_multapses", [True, False])
@pytest.mark.parametrize("n, k", [(40, 3), (9, 8), (9, 5)])
def test_fixed_degree_without_autapses(rule_cls, allow_multapses, n, k):
    """``allow_autapses=False`` never connects a neuron to itself.

    The three sizes exercise both samplers of the no-multapse case: ``(40,
    3)`` has ``k**2 <= n`` (whole-row rejection), ``(9, 5)`` uses the dense
    random-score top-k, and ``(9, 8)`` leaves no freedom at all: every neuron
    must connect to all 8 others.
    """
    rule = rule_cls(k, allow_autapses=False, allow_multapses=allow_multapses)
    pre, post = rule.edges(n, n, generator=_gen(1))
    A = _adjacency(pre, post, n, n)
    assert int(A.diagonal().sum()) == 0
    degree = A.sum(1) if rule_cls is FixedIndegree else A.sum(0)
    assert torch.equal(degree, torch.full((n,), k))
    if not allow_multapses:
        assert int(A.max()) <= 1
        if k == n - 1:
            assert torch.equal(A, 1 - torch.eye(n, dtype=torch.long))


def test_multapses_do_occur_when_allowed():
    """With replacement and ``k`` close to ``n_pre`` repeats are certain in
    practice, which shows that ``allow_multapses=False`` is doing work.

    For one target, P(no repeat) = 6!/6^6 = 1.5 %; over 50 targets the
    probability that no target has a multapse is below 1e-90.
    """
    pre, post = FixedIndegree(6).edges(6, 50, generator=_gen())
    assert int(_adjacency(pre, post, 6, 50).max()) > 1


def test_fixed_indegree_sources_are_uniform():
    """Sources are drawn uniformly, with and without multapses.

    Each of the ``n_post * k = 20000`` edges picks one of 20 sources, so every
    source is expected ``1000`` times. Without replacement the counts are
    slightly *less* variable than a Binomial(20000, 1/20) (std 30.8), so a
    +-6 sigma window of 185 around the mean holds for both variants and all
    20 sources with overwhelming probability.
    """
    n_pre, n_post, k = 20, 4000, 5
    for allow_multapses in (True, False):
        rule = FixedIndegree(k, allow_multapses=allow_multapses)
        pre, _ = rule.edges(n_pre, n_post, generator=_gen(2))
        counts = torch.bincount(pre, minlength=n_pre).double()
        std = math.sqrt(n_post * k * (1 / n_pre) * (1 - 1 / n_pre))
        assert (counts - n_post * k / n_pre).abs().max() < 6 * std


def test_fixed_degree_dense_sampler_is_chunked(monkeypatch):
    """The dense top-k sampler gives the same guarantees when it has to work in
    several chunks (forced here by shrinking the chunk budget)."""
    from btorch.models.connection import rule as rule_module

    monkeypatch.setattr(rule_module, "_CHUNK_ELEMENTS", 3 * N_PRE)  # 3 rows
    pre, post = FixedIndegree(8, allow_multapses=False).edges(
        N_PRE, N_POST, generator=_gen()
    )
    A = _adjacency(pre, post, N_PRE, N_POST)
    assert int(A.max()) == 1
    assert torch.equal(A.sum(1), torch.full((N_POST,), 8))


def test_fixed_degree_validation():
    """Impossible requests are refused with an explanation."""
    # More distinct sources than exist.
    with pytest.raises(ValueError, match="allow_multapses=False"):
        FixedIndegree(N_PRE + 1, allow_multapses=False).edges(N_PRE, N_POST)
    # ... which is fine with multapses.
    pre, _ = FixedIndegree(N_PRE + 1).edges(N_PRE, N_POST, generator=_gen())
    assert pre.numel() == (N_PRE + 1) * N_POST
    # Without autapses one source fewer is eligible.
    with pytest.raises(ValueError, match="only 4"):
        FixedIndegree(5, allow_autapses=False, allow_multapses=False).edges(5, 5)
    with pytest.raises(ValueError, match="only 6"):
        FixedOutdegree(7, allow_autapses=False, allow_multapses=False).edges(7, 7)
    # Nothing to connect to.
    with pytest.raises(ValueError, match="no eligible"):
        FixedIndegree(1).edges(0, 3)
    with pytest.raises(ValueError, match="non-negative int"):
        FixedIndegree(-1)
    with pytest.raises(ValueError, match="non-negative int"):
        FixedIndegree(2.5)
    with pytest.raises(ValueError, match="n_pre"):
        FixedIndegree(1).edges(-1, 3)


@pytest.mark.parametrize(
    "rule",
    [
        AllToAll(allow_autapses=False),
        FixedIndegree(2, allow_autapses=False),
        FixedOutdegree(2, allow_autapses=False),
        PairwiseBernoulli(0.5, allow_autapses=False),
    ],
)
def test_autapse_exclusion_needs_one_population(rule):
    """ "No autapses" only means something when pre and post are the same
    population: index ``i`` must denote the same neuron on both sides.

    Indices are population-local, so with different sizes the request is a
    modelling error and is reported instead of being ignored.
    """
    with pytest.raises(ValueError, match="projects onto itself"):
        rule.edges(N_PRE, N_POST)


# ---------------------------------------------------------- pairwise Bernoulli
def test_pairwise_bernoulli_edge_count_statistics():
    """The number of edges is Binomial(n_pre * n_post, p).

    With ``n = 400 * 300 = 120000`` pairs and ``p = 0.05`` the count has mean
    ``6000`` and standard deviation ``sqrt(n p (1 - p)) = 75.5``. The seed is
    fixed, so the test is deterministic; the 5-sigma window (false alarm
    probability 6e-7 for a correct sampler) is tight enough to catch a
    sampler that is off by 7 % in ``p`` (5.5 sigma away from the mean).
    """
    n_pre, n_post, p = 400, 300, 0.05
    rule = PairwiseBernoulli(p)
    pre, post = rule.edges(n_pre, n_post, generator=_gen(3))
    _check_edge_list(pre, post, n_pre, n_post)

    n = n_pre * n_post
    assert rule.expected_nnz(n_pre, n_post) == pytest.approx(n * p)
    assert abs(pre.numel() - n * p) < 5 * math.sqrt(n * p * (1 - p))

    # Each pair is tried once: no multapses.
    flat = post * n_pre + pre
    assert flat.unique().numel() == flat.numel()

    # Every target's in-degree is Binomial(n_pre, p): mean 20, std 4.36. The
    # largest deviation over 300 targets stays below 6 sigma, and the same
    # holds for the out-degrees (Binomial(300, 0.05): mean 15, std 3.77), so
    # the edges are not concentrated on some rows or columns.
    indeg = torch.bincount(post, minlength=n_post).double()
    outdeg = torch.bincount(pre, minlength=n_pre).double()
    assert (indeg - n_pre * p).abs().max() < 6 * math.sqrt(n_pre * p * (1 - p))
    assert (outdeg - n_post * p).abs().max() < 6 * math.sqrt(n_post * p * (1 - p))


def test_pairwise_bernoulli_every_pair_has_probability_p():
    """Averaged over many draws each individual pair is present with
    probability ``p`` (the geometric-skip sampler has no position bias).

    ``R = 4000`` draws of a 5 x 6 block with ``p = 0.3``: each pair's
    frequency has std ``sqrt(p (1 - p) / R) = 0.0072``; all 30 pairs must be
    within 5 sigma (0.036) of 0.3.
    """
    n_pre, n_post, p, repeats = 5, 6, 0.3, 4000
    rule = PairwiseBernoulli(p)
    g = _gen(4)
    total = torch.zeros(n_post, n_pre, dtype=torch.long)
    for _ in range(repeats):
        total += _adjacency(*rule.edges(n_pre, n_post, generator=g), n_pre, n_post)
    frequency = total.double() / repeats
    assert (frequency - p).abs().max() < 5 * math.sqrt(p * (1 - p) / repeats)


def test_pairwise_bernoulli_scales_with_edges_not_pairs():
    """10^5 x 10^5 neurons are 10^10 pairs: a dense mask would need 10 GB.

    The sampler only ever touches the ~10^5 connected pairs. The count
    is Binomial(1e10, 1e-5): mean 1e5, std 316.
    """
    n, p = 100_000, 1e-5
    pre, post = PairwiseBernoulli(p).edges(n, n, generator=_gen(5))
    _check_edge_list(pre, post, n, n)
    assert abs(pre.numel() - n * n * p) < 5 * math.sqrt(n * n * p)


def test_pairwise_bernoulli_is_exact_across_internal_batches(monkeypatch):
    """Positions continue correctly when the sampler needs several batches of
    random numbers (forced here with a tiny batch size)."""
    from btorch.models.connection import rule as rule_module

    monkeypatch.setattr(rule_module, "_CHUNK_ELEMENTS", 64)
    n_pre, n_post, p = 200, 150, 0.2
    pre, post = PairwiseBernoulli(p).edges(n_pre, n_post, generator=_gen(6))
    n = n_pre * n_post
    assert abs(pre.numel() - n * p) < 5 * math.sqrt(n * p * (1 - p))
    flat = post * n_pre + pre
    # Strictly increasing flattened positions: sorted by (post, pre), unique.
    assert bool((flat[1:] > flat[:-1]).all())
    assert int(flat[-1]) > 0.99 * n  # the last batch reaches the end


def test_pairwise_bernoulli_limits_and_autapses():
    """``p = 0`` is empty, ``p = 1`` is all-to-all, autapses can be removed."""
    assert PairwiseBernoulli(0.0).edges(N_PRE, N_POST)[0].numel() == 0
    pre, post = PairwiseBernoulli(1.0).edges(N_PRE, N_POST)
    assert int(_adjacency(pre, post, N_PRE, N_POST).min()) == 1

    n = 60
    rule = PairwiseBernoulli(0.5, allow_autapses=False)
    pre, post = rule.edges(n, n, generator=_gen())
    assert not bool((pre == post).any())
    assert rule.expected_nnz(n, n) == pytest.approx(0.5 * (n * n - n))
    # Binomial(3540, 0.5): std 29.7.
    assert abs(pre.numel() - 0.5 * (n * n - n)) < 5 * 29.75

    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            PairwiseBernoulli(bad)


# ----------------------------------------------------------- reproducibility
RANDOM_RULES = {
    "indegree": lambda: FixedIndegree(3),
    "indegree_distinct": lambda: FixedIndegree(3, allow_multapses=False),
    "outdegree": lambda: FixedOutdegree(3),
    "bernoulli": lambda: PairwiseBernoulli(0.3),
    "distance": lambda: DistanceDependent(
        torch.rand(N_PRE, 2, generator=_gen(7)),
        torch.rand(N_POST, 2, generator=_gen(8)),
        lambda d: torch.exp(-d),
    ),
}


@pytest.mark.parametrize("name", list(RANDOM_RULES))
def test_seeded_generator_reproduces_edges(name):
    """Same seed, same edges; a different seed gives different edges.

    The rule object itself holds no random state, so it can be reused.
    """
    rule = RANDOM_RULES[name]()
    first = rule.edges(N_PRE, N_POST, generator=_gen(11))
    again = rule.edges(N_PRE, N_POST, generator=_gen(11))
    other = rule.edges(N_PRE, N_POST, generator=_gen(12))
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
    assert first[0].shape != other[0].shape or not (
        torch.equal(first[0], other[0]) and torch.equal(first[1], other[1])
    )
    # One generator advances: two consecutive draws are different networks.
    g = _gen(11)
    a = rule.edges(N_PRE, N_POST, generator=g)
    b = rule.edges(N_PRE, N_POST, generator=g)
    assert a[0].shape != b[0].shape or not (
        torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    )


# ------------------------------------------------------------------ distance
def _grid(n):
    """``n`` points on a line with unit spacing: distances are integers."""
    return torch.arange(n, dtype=torch.float64)[:, None]


@pytest.mark.parametrize("max_distance", [None, 2.0])
def test_distance_dependent_step_kernel_is_deterministic(max_distance):
    """With a probability of exactly 0 or 1 the result is known.

    Sources and targets sit on a line; ``probability = 1`` up to distance 2
    connects each target to the sources at most two steps away. The dense
    path (``max_distance=None``) and the KD-tree path must agree, and both
    must equal the band matrix.
    """
    n_pre, n_post = 9, 6
    rule = DistanceDependent(
        _grid(n_pre),
        _grid(n_post),
        lambda d: (d <= 2).to(d.dtype),
        max_distance=max_distance,
    )
    pre, post = rule.edges(n_pre, n_post, generator=_gen())
    _check_edge_list(pre, post, n_pre, n_post)
    i, j = torch.arange(n_pre), torch.arange(n_post)
    band = ((j[:, None] - i[None, :]).abs() <= 2).long()
    assert torch.equal(_adjacency(pre, post, n_pre, n_post), band)
    assert rule.expected_nnz(n_pre, n_post) == int(band.sum())


def test_distance_dependent_cutoff_and_autapses():
    """``max_distance`` removes far pairs even where the kernel is positive;
    ``allow_autapses=False`` removes the zero-distance self pairs."""
    n = 8
    pos = _grid(n)
    rule = DistanceDependent(
        pos, pos, lambda d: torch.ones_like(d), max_distance=1.0, allow_autapses=False
    )
    pre, post = rule.edges(n, n, generator=_gen())
    # Only nearest neighbours remain: |i - j| == 1.
    assert torch.equal((pre - post).abs(), torch.ones_like(pre))
    assert pre.numel() == 2 * (n - 1) == rule.expected_nnz(n, n)
    # With autapses the zero-distance pairs are candidates as well.
    with_self = DistanceDependent(pos, pos, lambda d: torch.ones_like(d), 1.0)
    assert with_self.edges(n, n)[0].numel() == 2 * (n - 1) + n


@pytest.mark.parametrize("max_distance", [None, 0.25])
def test_distance_dependent_statistics(max_distance, monkeypatch):
    """The edge count matches the sum of the pair probabilities.

    The count is a sum of independent Bernoulli variables (Poisson binomial)
    with mean ``sum(p_ij)`` and variance ``sum(p_ij (1 - p_ij)) <= mean``, so
    ``|count - mean| < 5 sqrt(mean)`` is a conservative 5-sigma window. The
    reference mean is computed here with a dense distance matrix; the rule is
    forced to work in several chunks of targets to cover the chunked path.
    """
    from btorch.models.connection import rule as rule_module

    n_pre, n_post = 300, 200
    monkeypatch.setattr(rule_module, "_CHUNK_ELEMENTS", 37 * n_pre)  # 37 targets
    pre_pos = torch.rand(n_pre, 2, generator=_gen(1), dtype=torch.float64)
    post_pos = torch.rand(n_post, 2, generator=_gen(2), dtype=torch.float64)

    def kernel(d):
        return 0.8 * torch.exp(-d / 0.1)

    dist = torch.cdist(post_pos, pre_pos)
    prob = kernel(dist)
    if max_distance is not None:
        prob = prob * (dist <= max_distance)
    mean = float(prob.sum())

    rule = DistanceDependent(pre_pos, post_pos, kernel, max_distance=max_distance)
    assert rule.expected_nnz(n_pre, n_post) == pytest.approx(mean)
    pre, post = rule.edges(n_pre, n_post, generator=_gen(3))
    _check_edge_list(pre, post, n_pre, n_post)
    assert abs(pre.numel() - mean) < 5 * math.sqrt(mean)
    # Pairs are tried once, and never beyond the cutoff.
    flat = post * n_pre + pre
    assert flat.unique().numel() == flat.numel()
    assert bool((flat[1:] > flat[:-1]).all())  # ordered by target, then source
    if max_distance is not None:
        assert float(dist[post, pre].max()) <= max_distance
    # Connected pairs are closer than average: the kernel decays.
    assert float(dist[post, pre].mean()) < 0.5 * float(dist.mean())


def test_distance_dependent_validation():
    pos = torch.rand(5, 2)
    with pytest.raises(ValueError, match=r"\[n, D\]"):
        DistanceDependent(torch.rand(5), pos, torch.exp)
    with pytest.raises(ValueError, match="spatial"):
        DistanceDependent(torch.rand(5, 3), pos, torch.exp)
    with pytest.raises(ValueError, match="max_distance"):
        DistanceDependent(pos, pos, torch.exp, max_distance=-1.0)
    # The rule holds positions, so the population sizes are fixed by them.
    rule = DistanceDependent(pos, pos, lambda d: torch.exp(-d))
    with pytest.raises(ValueError, match="positions of 5"):
        rule.edges(4, 5)
    # A "probability" above one is a modelling error, not clipped silently.
    bad = DistanceDependent(pos, pos, lambda d: 2.0 * torch.ones_like(d))
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        bad.edges(5, 5)


# ---------------------------------------------------------------- from data
def test_from_edges_returns_the_list_and_its_values():
    """An explicit edge list is passed through unchanged, values aligned."""
    pre = torch.tensor([3, 0, 3, 11])
    post = torch.tensor([6, 1, 6, 0])  # (3 -> 6) twice: a multapse is kept
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4])
    rule = FromEdges(pre, post, weight)
    got_pre, got_post = rule.edges(N_PRE, N_POST)
    assert torch.equal(got_pre, pre) and torch.equal(got_post, post)
    assert torch.equal(rule.values(N_PRE, N_POST), weight)
    assert rule.expected_nnz(N_PRE, N_POST) == 4
    assert FromEdges(pre, post).values(N_PRE, N_POST) is None

    # The list must fit the populations it is applied to. Swapping the
    # population sizes is caught because index 11 does not fit n_pre = 7.
    with pytest.raises(ValueError, match="do not fit"):
        rule.edges(N_POST, N_PRE)
    with pytest.raises(ValueError, match="same length"):
        FromEdges(pre, post[:2])
    with pytest.raises(ValueError, match="one entry per edge"):
        FromEdges(pre, post, weight[:2])
    with pytest.raises(TypeError, match="integer"):
        FromEdges(pre.float(), post)
    with pytest.raises(ValueError, match="non-negative"):
        FromEdges(torch.tensor([-1]), torch.tensor([0]))


def _asymmetric_matrix():
    """A non-square, asymmetric operator ``W [n_post, n_pre]`` as an edge list
    ``(post, pre, weight)`` with distinct weights.

    Target 0 receives from sources 1 and 4, target 2 from source 0, target 1
    from nobody: no symmetry can hide a transpose.
    """
    post = np.array([0, 0, 2, 3, 3, 3])
    pre = np.array([1, 4, 0, 2, 3, 4])
    weight = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return post, pre, weight, (4, 5)  # n_post = 4, n_pre = 5


def _as_library(kind, rows, cols, values, shape):
    """The same matrix as a SciPy, PyTorch or btorch sparse object."""
    if kind.startswith("scipy"):
        mat = scipy.sparse.coo_array((values, (rows, cols)), shape=shape)
        return mat.asformat(kind.split("_")[1])
    indices = torch.tensor(np.stack([rows, cols]))
    coo = torch.sparse_coo_tensor(indices, torch.tensor(values), shape).coalesce()
    if kind == "torch_coo":
        return coo
    if kind == "torch_csr":
        return coo.to_sparse_csr()
    A = sparse.from_edges(
        torch.tensor(rows), torch.tensor(cols), torch.tensor(values), shape
    )
    return getattr(A, f"to{kind.split('_')[1]}")()


LIBRARIES = [
    "scipy_coo",
    "scipy_csr",
    "scipy_csc",
    "torch_coo",
    "torch_csr",
    "btorch_coo",
    "btorch_csr",
    "btorch_csc",
]


def _weights(rule, n_pre, n_post):
    """Dense ``[n_post, n_pre]`` weight matrix described by a rule."""
    pre, post = rule.edges(n_pre, n_post)
    W = torch.zeros(n_post, n_pre, dtype=torch.float64)
    W.index_put_((post, pre), rule.values(n_pre, n_post).double(), accumulate=True)
    return W


@pytest.mark.parametrize("kind", LIBRARIES)
def test_from_sparse_post_pre_is_the_standard_operator(kind):
    """``orientation="post_pre"``: the matrix is ``(n_post, n_pre)``.

    Rows are targets, columns are sources (``y = A @ x``). The same matrix
    given as a SciPy, PyTorch or btorch object (any storage format) yields
    the same edges and values, because all of them go through
    ``btorch.sparse.as_sparse``.
    """
    post, pre, weight, (n_post, n_pre) = _asymmetric_matrix()
    A = _as_library(kind, post, pre, weight, (n_post, n_pre))
    rule = FromSparse(A)  # post_pre is the default
    got_pre, got_post = rule.edges(n_pre, n_post)
    _check_edge_list(got_pre, got_post, n_pre, n_post)
    assert rule.expected_nnz(n_pre, n_post) == len(weight)

    reference = torch.zeros(n_post, n_pre, dtype=torch.float64)
    reference[post, pre] = torch.tensor(weight)
    torch.testing.assert_close(_weights(rule, n_pre, n_post), reference)
    # Spot check one entry by hand: source 4 -> target 0 has weight 2.
    hit = (got_pre == 4) & (got_post == 0)
    assert rule.values(n_pre, n_post)[hit].tolist() == [2.0]


@pytest.mark.parametrize("kind", LIBRARIES)
def test_from_sparse_pre_post_is_the_connectome_convention(kind):
    """``orientation="pre_post"``: rows are sources (``y = x @ A``).

    The connectome matrix ``C [n_pre, n_post]`` is the transpose of the
    operator above. Declaring the orientation gives the *same* edges and the
    same ``[n_post, n_pre]`` weights: this is the one and only place where a
    transpose is expressed.
    """
    post, pre, weight, (n_post, n_pre) = _asymmetric_matrix()
    # Rows = pre, columns = post.
    C = _as_library(kind, pre, post, weight, (n_pre, n_post))
    rule = FromSparse(C, orientation="pre_post")

    reference = torch.zeros(n_post, n_pre, dtype=torch.float64)
    reference[post, pre] = torch.tensor(weight)
    torch.testing.assert_close(_weights(rule, n_pre, n_post), reference)

    # The connectome convention in one line: x @ C == W @ x.
    x = torch.arange(1.0, n_pre + 1, dtype=torch.float64)
    C_dense = sparse.as_sparse(C).to_dense().double()
    torch.testing.assert_close(x @ C_dense, _weights(rule, n_pre, n_post) @ x)


@pytest.mark.parametrize("kind", LIBRARIES)
def test_from_sparse_never_transposes_twice_or_silently(kind):
    """The anti-double-transpose test.

    A classic bug is a conversion helper that "helpfully" transposes
    connectome matrices and a caller that transposes again (or not at all).
    Here the conversion never transposes, so:

    1. the matrix the rule holds is exactly the input (same shape, same
       dense content), whatever the orientation;
    2. reading an ``(n_post, n_pre)`` operator and its ``(n_pre, n_post)``
       transpose with the matching orientations gives identical edge sets;
    3. using the wrong orientation on a non-square matrix is a shape error
       that names the fix. It is never "repaired" by a hidden transpose.
    """
    post, pre, weight, (n_post, n_pre) = _asymmetric_matrix()
    W = _as_library(kind, post, pre, weight, (n_post, n_pre))  # operator
    C = _as_library(kind, pre, post, weight, (n_pre, n_post))  # connectome

    # 1. No transpose inside the rule.
    for obj, orientation, shape in (
        (W, "post_pre", (n_post, n_pre)),
        (C, "pre_post", (n_pre, n_post)),
    ):
        held = FromSparse(obj, orientation=orientation).matrix
        assert held.shape == shape
        torch.testing.assert_close(held.to_dense(), sparse.as_sparse(obj).to_dense())

    # 2. Both descriptions are the same set of (pre, post, weight) triples.
    def triples(rule):
        p, q = rule.edges(n_pre, n_post)
        v = rule.values(n_pre, n_post)
        return sorted(zip(p.tolist(), q.tolist(), v.tolist()))

    expected = sorted(zip(pre.tolist(), post.tolist(), weight.tolist()))
    assert triples(FromSparse(W, orientation="post_pre")) == expected
    assert triples(FromSparse(C, orientation="pre_post")) == expected

    # 3. The wrong orientation fails loudly instead of flipping the matrix.
    with pytest.raises(ValueError, match="never transposed"):
        FromSparse(W, orientation="pre_post").edges(n_pre, n_post)
    with pytest.raises(ValueError, match="never transposed"):
        FromSparse(C, orientation="post_pre").edges(n_pre, n_post)


def test_from_sparse_square_matrix_orientation_changes_the_result():
    """On a square matrix no shape check can help: the orientation argument
    alone decides the direction, and the two readings are transposes."""
    A = sparse.from_edges(
        torch.tensor([0]), torch.tensor([2]), torch.tensor([1.0]), (3, 3)
    )
    # As an operator, A[0, 2] is "source 2 -> target 0".
    pre, post = FromSparse(A, orientation="post_pre").edges(3, 3)
    assert (pre.tolist(), post.tolist()) == ([2], [0])
    # As a connectome matrix, A[0, 2] is "source 0 -> target 2".
    pre, post = FromSparse(A, orientation="pre_post").edges(3, 3)
    assert (pre.tolist(), post.tolist()) == ([0], [2])


def test_from_sparse_keeps_values_differentiable_and_validates():
    """Values from a btorch/PyTorch matrix keep their autograd history, so
    weights initialised from a matrix can be trained."""
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    A = sparse.from_edges(torch.tensor([0, 1]), torch.tensor([2, 0]), values, (2, 3))
    rule = FromSparse(A)
    (grad,) = torch.autograd.grad(rule.values(3, 2).sum(), values)
    assert grad.tolist() == [1.0, 1.0]

    with pytest.raises(ValueError, match="orientation"):
        FromSparse(A, orientation="transposed")
    # Dense input is never sparsified implicitly (as_sparse refuses it).
    with pytest.raises(TypeError):
        FromSparse(torch.ones(2, 3))
    with pytest.raises(ValueError, match="single sparse matrix"):
        FromSparse(sparse.stack([A, A]))


# --------------------------------------------------------------- as_operator
def _edge_matrix(rule, n_pre, n_post):
    pre, post = rule.edges(n_pre, n_post)
    return _adjacency(pre, post, n_pre, n_post).double()


@pytest.mark.parametrize(
    "rule, n_pre, n_post",
    [
        (OneToOne(), 6, 6),
        (AllToAll(), N_PRE, N_POST),
        (AllToAll(allow_autapses=False), 6, 6),
        (PairwiseBernoulli(1.0), N_PRE, N_POST),
    ],
)
def test_as_operator_equals_the_materialised_edges(rule, n_pre, n_post):
    """The structured realisation is the 0/1 matrix of the edges.

    ``as_operator`` lets a projection apply a one-to-one or all-to-all
    connection without ever building the edge list. Its shape is the
    standard ``(n_post, n_pre)``; applying it, transposing it and
    materialising it must all agree with the matrix built from ``edges``.
    """
    op = rule.as_operator(n_pre, n_post, dtype=torch.float64)
    assert isinstance(op, LinearOperator)
    assert op.shape == (n_post, n_pre)
    reference = _edge_matrix(rule, n_pre, n_post)

    x = torch.randn(3, n_pre, dtype=torch.float64, generator=_gen())
    y = torch.randn(3, n_post, dtype=torch.float64, generator=_gen(1))
    torch.testing.assert_close(op.matvec(x), x @ reference.T)
    torch.testing.assert_close(op.rmatvec(y), y @ reference)
    torch.testing.assert_close(op.to_dense(dtype=torch.float64), reference)
    torch.testing.assert_close(op.materialize().to_dense(), reference)


def test_all_to_all_operator_avoids_the_dense_matrix():
    """The point of ``as_operator``: 10^5 x 10^5 all-to-all has 10^10 edges,
    but the operator applies with two vectors' worth of memory."""
    n = 100_000
    op = AllToAll(allow_autapses=False).as_operator(n, n)
    y = op.matvec(torch.ones(n))
    # Every target receives from the n - 1 other neurons.
    assert y.shape == (n,) and float(y[0]) == n - 1 and float(y[-1]) == n - 1


def test_rules_without_closed_form_return_none():
    """Random and data-driven rules have no structured realisation; the default
    tells the caller to realise them from their edges."""
    pos = torch.rand(4, 2)
    for rule in (
        FixedIndegree(2),
        FixedOutdegree(2),
        PairwiseBernoulli(0.5),
        DistanceDependent(pos, pos, lambda d: torch.exp(-d)),
        FromEdges(torch.tensor([0]), torch.tensor([1])),
    ):
        assert isinstance(rule, ConnectionRule)
        assert rule.as_operator(4, 4) is None
    # Invalid sizes are still reported by the rules that do have one.
    with pytest.raises(ValueError, match="same size"):
        OneToOne().as_operator(N_PRE, N_POST)


def test_rule_is_not_a_sparse_array():
    """Design invariant: a rule is modelling semantics, not a matrix."""
    rule = FixedIndegree(2)
    assert not isinstance(rule, sparse.Sparse)
    assert not isinstance(rule, LinearOperator)
    assert "FixedIndegree(k=2" in repr(rule)
