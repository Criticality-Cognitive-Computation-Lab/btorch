"""``Projection``: a connection rule plus a synapse between two populations.

A projection is the NEST-style way to build a connection (spec sections
11-13, 42.7)::

    proj = Projection(pre=E, post=I, rule=FixedIndegree(100),
                      synapse=Synapse(weight=0.1, delay=2))
    current = proj(spikes)

It states *which* neurons are connected (the rule) and *what* the edges do
(the synapse). It does not state how the edges are stored or executed: the
projection builds a ``Connection`` (``proj.connection``) and delegates to it.

Every numerical test compares the projection with a dense matrix that is
written down independently: the rule is asked for its edge list with a
generator of the same seed, and the edges are scattered into a dense
``[n_post, n_pre]`` matrix (duplicates add up). ``current = x @ A.T`` is then
the reference. Populations have different sizes wherever the rule allows it,
so a transposed result has the wrong shape and cannot pass by accident.

Terms: an *edge* is one ``(pre, post)`` pair of the rule's edge list; a
*multapse* is an edge that occurs more than once; an *edge slot* is one stored
entry of the realised ``SparseConnection`` (parallel edges are merged into one
slot).
"""

import inspect

import pytest
import scipy.sparse
import torch
from torch import nn

from btorch import sparse
from btorch.models.connection import (
    AllToAll,
    ConnectionRule,
    ConstantWeight,
    ConstrainedWeight,
    DistanceDependent,
    EdgeWeight,
    FixedIndegree,
    FixedOutdegree,
    FromEdges,
    FromSparse,
    OneToOne,
    PairwiseBernoulli,
    Projection,
    SparseConnection,
    StructuredConnection,
    Synapse,
)
from btorch.models.constrain import constrain_net
from btorch.models.neurons.lif import LIF
from btorch.sparse import Sparse
from btorch.sparse.operator import is_linear_operator
from btorch.sparse.runtime import RepresentationCache
from tests.sparse.helpers import DEVICES


# Rectangular populations (most rules) and a square one (rules that pair
# index i of both populations: OneToOne, allow_autapses=False).
N_PRE, N_POST, N_SQ = 7, 5, 6


def _gen(seed=0):
    """A fresh seeded generator.

    ``rule.edges`` consumes the generator, so the reference and the projection
    each get their own generator with the same seed.
    """
    return torch.Generator().manual_seed(seed)


# Fixed inputs of the data-carrying rules.
_POS_PRE = torch.rand(N_PRE, 2, generator=_gen(10))
_POS_POST = torch.rand(N_POST, 2, generator=_gen(11))
# An explicit edge list: unsorted, with the pair (pre 3 -> post 1) twice.
_EDGE_PRE = torch.tensor([3, 0, 6, 3, 2, 5, 1])
_EDGE_POST = torch.tensor([1, 4, 0, 1, 2, 2, 3])
_EDGE_VAL = torch.tensor([0.5, -1.0, 2.0, 0.25, 1.5, -3.0, 0.75])
# The same edges as a sparse operator (n_post, n_pre) and as an adjacency
# matrix (n_pre, n_post); the two are transposes of each other.
_OPERATOR = sparse.from_edges(_EDGE_POST, _EDGE_PRE, _EDGE_VAL, (N_POST, N_PRE))
_ADJACENCY = sparse.from_edges(_EDGE_PRE, _EDGE_POST, _EDGE_VAL, (N_PRE, N_POST))

# name -> (rule, n_pre, n_post). Every rule class appears at least once; the
# small populations make the multapse-allowing rules actually produce
# parallel edges.
RULES = {
    "one_to_one": (OneToOne(), N_SQ, N_SQ),
    "all_to_all": (AllToAll(), N_PRE, N_POST),
    "all_to_all_no_autapses": (AllToAll(allow_autapses=False), N_SQ, N_SQ),
    "fixed_indegree": (FixedIndegree(4), N_PRE, N_POST),
    "fixed_indegree_distinct": (
        FixedIndegree(3, allow_multapses=False),
        N_PRE,
        N_POST,
    ),
    "fixed_outdegree": (FixedOutdegree(4), N_PRE, N_POST),
    "fixed_outdegree_distinct_no_autapses": (
        FixedOutdegree(2, allow_autapses=False, allow_multapses=False),
        N_SQ,
        N_SQ,
    ),
    "pairwise_bernoulli": (PairwiseBernoulli(0.4), N_PRE, N_POST),
    "distance_dependent": (
        DistanceDependent(_POS_PRE, _POS_POST, lambda d: torch.exp(-d / 0.5)),
        N_PRE,
        N_POST,
    ),
    "from_edges": (FromEdges(_EDGE_PRE, _EDGE_POST, _EDGE_VAL), N_PRE, N_POST),
    "from_sparse_post_pre": (
        FromSparse(_OPERATOR, orientation="post_pre"),
        N_PRE,
        N_POST,
    ),
    "from_sparse_pre_post": (
        # "pre_post" is the default; spelled out because the key names it.
        FromSparse(_ADJACENCY, orientation="pre_post"),
        N_PRE,
        N_POST,
    ),
}
RULE_NAMES = list(RULES)


def _edges(name, seed=0):
    """Edge list ``(pre, post, values)`` of a rule of ``RULES``.

    ``values`` is what the rule itself carries (``FromEdges`` / ``FromSparse``)
    or ``None`` for purely structural rules.
    """
    rule, n_pre, n_post = RULES[name]
    pre, post = rule.edges(n_pre, n_post, generator=_gen(seed))
    return pre, post, rule.values(n_pre, n_post)


def _dense(
    pre,
    post,
    weight,
    n_pre,
    n_post,
    receptor=None,
    n_receptor=1,
    delay=None,
    n_delay=1,
):
    """Dense reference operator of an edge list, independent of btorch.

    Returns ``[n_post * n_receptor, n_pre * n_delay]`` with
    ``A[post * n_receptor + receptor, pre * n_delay + delay] += weight`` for
    every edge; ``current = x @ A.T``. Parallel edges add up.
    """
    zero = torch.zeros_like(pre)
    receptor = zero if receptor is None else receptor
    delay = zero if delay is None else delay
    A = torch.zeros(n_post * n_receptor, n_pre * n_delay, dtype=weight.dtype)
    A.index_put_(
        (post * n_receptor + receptor, pre * n_delay + delay), weight, accumulate=True
    )
    return A


def _input(proj, batch=(3,), seed=1):
    """Random ``[*batch, in_features]`` activity."""
    return torch.rand(*batch, proj.in_features, generator=_gen(seed))


# ------------------------------------------------------- rule -> same currents
@pytest.mark.parametrize("weight_kind", ["default", "scalar", "per_edge", "parameter"])
@pytest.mark.parametrize("name", RULE_NAMES)
def test_every_rule_matches_dense_reference(name, weight_kind):
    """A projection computes ``x @ A.T`` for the dense matrix of its rule.

    Four ways to specify the weight:

    - ``default``: no synapse. Rules that carry values use them, all other
      rules start from unit weights (so ``A`` counts the edges).
    - ``scalar``: one fixed number; the values of the rule are ignored.
    - ``per_edge`` / ``parameter``: a ``[n_edge]`` tensor or ``nn.Parameter``
      aligned with the rule's edge list.
    """
    rule, n_pre, n_post = RULES[name]
    pre, post, values = _edges(name)
    n_edge = pre.shape[0]
    per_edge = torch.randn(n_edge, generator=_gen(2))
    if weight_kind == "default":
        synapse, weight = None, values if values is not None else torch.ones(n_edge)
    elif weight_kind == "scalar":
        synapse, weight = Synapse(weight=0.5), torch.full((n_edge,), 0.5)
    elif weight_kind == "per_edge":
        synapse, weight = Synapse(weight=per_edge), per_edge
    else:
        synapse, weight = Synapse(weight=nn.Parameter(per_edge.clone())), per_edge
    A = _dense(pre, post, weight, n_pre, n_post)

    proj = Projection(n_pre, n_post, rule, synapse, generator=_gen())

    assert (proj.n_pre, proj.n_post) == (n_pre, n_post)
    assert (proj.in_features, proj.out_features) == (n_pre, n_post)
    # Any number of leading (time, batch) dimensions.
    x = _input(proj, batch=(2, 3))
    out = proj(x)
    assert out.shape == (2, 3, n_post)
    torch.testing.assert_close(out, x @ A.T)


@pytest.mark.parametrize("name", RULE_NAMES)
def test_projection_uses_the_generator_and_nothing_else(name):
    """The same seed gives the same projection; the projection's edges are
    exactly the rule's edges for that seed (as a multiset: the connection
    stores them in its own canonical order)."""
    rule, n_pre, n_post = RULES[name]
    pre, post, _ = _edges(name, seed=3)
    count = _dense(pre, post, torch.ones(pre.shape[0]), n_pre, n_post)

    def build():
        # A unit weight on explicit edges: the operator counts the edges.
        return Projection(
            n_pre,
            n_post,
            rule,
            Synapse(weight=1.0),
            generator=_gen(3),
            realization="sparse",
        ).connection

    first, second = build(), build()
    assert torch.equal(first.indices, second.indices)
    torch.testing.assert_close(first.to_sparse("post_pre").to_dense(), count)


# ------------------------------------------------------------------ populations
class _Sized(nn.Module):
    """Stand-in population exposing only ``n_neuron``."""

    def __init__(self, n_neuron):
        super().__init__()
        self.n_neuron = n_neuron


@pytest.mark.parametrize(
    "pre, post, expected",
    [
        (N_PRE, N_POST, (N_PRE, N_POST)),
        # Neuron modules carry their flat size in ``size`` (and the shape in
        # ``n_neuron``).
        (LIF(n_neuron=N_PRE), LIF(n_neuron=N_POST), (N_PRE, N_POST)),
        (LIF(n_neuron=N_PRE), N_POST, (N_PRE, N_POST)),
        # A multi-dimensional population counts all of its neurons.
        (LIF(n_neuron=(2, 3)), _Sized((4,)), (6, 4)),
        (_Sized(N_PRE), _Sized((5, 1)), (N_PRE, 5)),
    ],
    ids=["ints", "modules", "module_and_int", "nd_module", "n_neuron_only"],
)
def test_population_given_as_int_or_module(pre, post, expected):
    """``pre`` / ``post`` are neuron counts or modules that know their size."""
    proj = Projection(pre, post, AllToAll())
    assert (proj.n_pre, proj.n_post) == expected
    assert proj(torch.ones(expected[0])).shape == (expected[1],)


def test_module_and_int_populations_build_the_same_connection():
    """A module is only read for its size: the edges are the same."""
    a = Projection(N_PRE, N_POST, FixedIndegree(3), generator=_gen())
    b = Projection(
        LIF(n_neuron=N_PRE), LIF(n_neuron=N_POST), FixedIndegree(3), generator=_gen()
    )
    assert torch.equal(a.connection.indices, b.connection.indices)
    # The population module is not adopted as a submodule of the projection.
    assert [n for n, _ in b.named_children()] == ["connection"]


@pytest.mark.parametrize(
    "bad", [object(), torch.zeros(4), "E", 2.5, None, True], ids=repr
)
def test_population_without_a_size_is_rejected(bad):
    """An object from which no neuron count can be read raises ``TypeError``
    naming the argument (``True`` is not a population of one neuron)."""
    with pytest.raises(TypeError, match="pre"):
        Projection(bad, N_POST, AllToAll())
    with pytest.raises(TypeError, match="post"):
        Projection(N_PRE, bad, AllToAll())


def test_population_sizes_are_checked_by_the_rule():
    """Sizes that do not fit the rule raise the rule's own error."""
    with pytest.raises(ValueError, match="same size"):
        Projection(N_PRE, N_POST, OneToOne())
    with pytest.raises(ValueError, match="non-negative"):
        Projection(-1, N_POST, AllToAll())


# ------------------------------------------------------------------ realisation
# Rules with a closed form (``rule.as_operator`` is not None).
STRUCTURED_RULES = {
    "one_to_one": (OneToOne(), N_SQ, N_SQ),
    "all_to_all": (AllToAll(), N_PRE, N_POST),
    "all_to_all_no_autapses": (AllToAll(allow_autapses=False), N_SQ, N_SQ),
    # p == 1 is all-to-all.
    "bernoulli_p1": (PairwiseBernoulli(1.0), N_PRE, N_POST),
}


@pytest.mark.parametrize("weight", [0.5, 2], ids=["float", "int"])
@pytest.mark.parametrize("name", list(STRUCTURED_RULES))
def test_auto_picks_structured_for_fixed_scalar_weight(name, weight):
    """``OneToOne`` / ``AllToAll`` with one fixed weight and no receptor or
    delay never store a matrix: the projection applies a closed-form operator.

    The structured and the explicit realisation give the same currents. The
    few tensors a closed-form operator consists of (the diagonal of
    ``OneToOne``, nothing at all for a plain ``AllToAll``) are registered on
    the connection as ``operator_<i>_<attr>`` buffers so that they follow
    ``.to()`` and are saved; none of them is trainable for a fixed weight.
    """
    rule, n_pre, n_post = STRUCTURED_RULES[name]
    synapse = Synapse(weight=weight)
    auto = Projection(n_pre, n_post, rule, synapse)
    explicit = Projection(n_pre, n_post, rule, synapse, realization="sparse")

    assert isinstance(auto.connection, StructuredConnection)
    assert isinstance(explicit.connection, SparseConnection)
    # No edge list and nothing trainable. The only state is the operator's
    # own tensors, registered as buffers: at most O(n) numbers, never the
    # n_post * n_pre entries of a matrix.
    assert not hasattr(auto.connection, "indices")
    assert list(auto.parameters()) == []
    state = auto.state_dict()
    assert all(k.startswith("connection.operator_") for k in state)
    assert set(state) == {n for n, _ in auto.named_buffers()}
    assert sum(v.numel() for v in state.values()) <= max(n_pre, n_post)
    assert (auto.in_features, auto.out_features) == (n_pre, n_post)

    x = _input(auto, batch=(2, 3))
    pre, post = rule.edges(n_pre, n_post)
    A = _dense(pre, post, torch.full((pre.shape[0],), float(weight)), n_pre, n_post)
    torch.testing.assert_close(auto(x), x @ A.T)
    torch.testing.assert_close(auto(x), explicit(x))


def _per_edge(rule, n_pre, n_post, fill):
    return torch.full((int(rule.expected_nnz(n_pre, n_post)),), fill)


@pytest.mark.parametrize(
    "make_synapse, kwargs",
    [
        # Trainable weights need one parameter per edge.
        (lambda n: None, {}),
        (lambda n: Synapse(), {}),
        (lambda n: Synapse(weight=nn.Parameter(torch.full((n,), 0.5))), {}),
        # Fixed, but one value per edge.
        (lambda n: Synapse(weight=torch.full((n,), 0.5)), {}),
        # Receptor / delay are edge attributes: explicit edges are needed.
        (lambda n: Synapse(weight=0.5, receptor=0), {}),
        (lambda n: Synapse(weight=0.5, delay=0), {}),
        (lambda n: Synapse(weight=0.5, receptor=1, delay=2), {}),
        # The user asks for explicit edges.
        (lambda n: Synapse(weight=0.5), {"realization": "sparse"}),
    ],
    ids=[
        "no_synapse",
        "default_synapse",
        "parameter",
        "per_edge_tensor",
        "receptor",
        "delay",
        "receptor_and_delay",
        "realization_sparse",
    ],
)
@pytest.mark.parametrize("name", list(STRUCTURED_RULES))
def test_auto_picks_sparse_otherwise(name, make_synapse, kwargs):
    """Anything a closed-form operator cannot express is realised with explicit
    edges, also for a rule that has a closed form."""
    rule, n_pre, n_post = STRUCTURED_RULES[name]
    n_edge = int(rule.expected_nnz(n_pre, n_post))
    proj = Projection(n_pre, n_post, rule, make_synapse(n_edge), **kwargs)
    assert isinstance(proj.connection, SparseConnection)
    assert proj.connection.nnz == n_edge


@pytest.mark.parametrize(
    "name", [n for n in RULE_NAMES if n not in STRUCTURED_RULES], ids=str
)
def test_rules_without_closed_form_are_always_sparse(name):
    """A random or data-defined rule has no structured operator, so a fixed
    scalar weight is realised as a ``ConstantWeight`` on explicit edges."""
    rule, n_pre, n_post = RULES[name]
    assert rule.as_operator(n_pre, n_post) is None
    proj = Projection(n_pre, n_post, rule, Synapse(weight=0.5), generator=_gen())
    assert isinstance(proj.connection, SparseConnection)
    assert isinstance(proj.connection.weight, ConstantWeight)
    assert list(proj.parameters()) == []


def test_structured_and_sparse_realisations_agree_with_other_dtypes():
    """Both realisations follow the dtype of the input the same way."""
    synapse = Synapse(weight=0.5)
    auto = Projection(N_SQ, N_SQ, AllToAll(allow_autapses=False), synapse)
    explicit = Projection(
        N_SQ,
        N_SQ,
        AllToAll(allow_autapses=False),
        synapse,
        realization="sparse",
        dtype=torch.float64,
    )
    x = _input(auto).double()
    assert auto(x).dtype == explicit(x).dtype == torch.float64
    torch.testing.assert_close(auto(x), explicit(x))


def test_unknown_realization_is_rejected():
    with pytest.raises(ValueError, match="realization"):
        Projection(N_PRE, N_POST, AllToAll(), realization="dense")


@pytest.mark.parametrize(
    "routing", [{"n_receptor": 2}, {"n_delay": 3}], ids=["n_receptor", "n_delay"]
)
def test_channel_counts_without_ids_give_the_same_layout(routing):
    """``n_receptor`` / ``n_delay`` define the output / input layout even when
    every edge uses channel 0; both realisations must agree on it.

    Regression test: ``realization="auto"`` used to look only at the per-edge
    ``receptor`` / ``delay`` ids and built a structured operator of shape
    ``(n_post, n_pre)``, i.e. with other ``in_features`` / ``out_features``
    than the explicit realisation. A closed-form operator has no channel
    axis, so such a synapse is now realised with explicit edges.
    """
    synapse = Synapse(weight=0.5, **routing)
    auto = Projection(N_SQ, N_SQ, OneToOne(), synapse)
    explicit = Projection(N_SQ, N_SQ, OneToOne(), synapse, realization="sparse")
    assert isinstance(auto.connection, SparseConnection)
    n_receptor, n_delay = routing.get("n_receptor", 1), routing.get("n_delay", 1)
    assert (auto.in_features, auto.out_features) == (N_SQ * n_delay, N_SQ * n_receptor)
    assert (auto.in_features, auto.out_features) == (
        explicit.in_features,
        explicit.out_features,
    )
    # Every edge uses channel 0: the dense reference is the scaled identity
    # placed at the channel-0 positions of the expanded layout.
    idx = torch.arange(N_SQ)
    A = _dense(
        idx, idx, torch.full((N_SQ,), 0.5), N_SQ, N_SQ, None, n_receptor, None, n_delay
    )
    x = _input(auto)
    torch.testing.assert_close(auto(x), x @ A.T)
    torch.testing.assert_close(auto(x), explicit(x))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["one_to_one", "all_to_all", "fixed_indegree"])
def test_device_argument(name, device):
    """``device=`` builds the connection on that device; a seeded CPU generator
    gives the same edges everywhere."""
    rule, n_pre, n_post = RULES[name]
    pre, post, _ = _edges(name)
    A = _dense(pre, post, torch.full((pre.shape[0],), 0.5), n_pre, n_post)
    proj = Projection(
        n_pre, n_post, rule, Synapse(weight=0.5), generator=_gen(), device=device
    )
    x = _input(proj)
    out = proj(x.to(device))
    assert out.device.type == device
    torch.testing.assert_close(out.cpu(), x @ A.T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "name", ["one_to_one", "all_to_all_no_autapses", "all_to_all", "fixed_indegree"]
)
def test_module_to_moves_the_projection(name):
    """A projection is an ``nn.Module``: ``proj.to(device)`` moves it.

    Regression test for the structured rules (``one_to_one``,
    ``all_to_all_no_autapses``): the tensor inside their operator used to be
    a plain attribute that ``Module.to()`` left on the CPU. It is a
    registered buffer now.
    """
    rule, n_pre, n_post = RULES[name]
    proj = Projection(n_pre, n_post, rule, Synapse(weight=0.5), generator=_gen())
    x = _input(proj)
    expected = proj(x)
    out = proj.to("cuda")(x.cuda())
    assert out.device.type == "cuda"
    assert all(b.device.type == "cuda" for b in proj.buffers())
    torch.testing.assert_close(out.cpu(), expected)


# ------------------------------------------------------- FromSparse / FromEdges
# The dense operator [n_post, n_pre] of the explicit edge list above (the
# duplicate pair adds up).
_W = _dense(_EDGE_PRE, _EDGE_POST, _EDGE_VAL, N_PRE, N_POST)


@pytest.mark.parametrize(
    "make_rule",
    [
        lambda: FromSparse(_OPERATOR, orientation="post_pre"),
        lambda: FromSparse(_OPERATOR.tocsr(), orientation="post_pre"),
        lambda: FromSparse(_ADJACENCY),  # the default is "pre_post"
        lambda: FromSparse(_ADJACENCY.to_torch(), orientation="pre_post"),
        lambda: FromEdges(_EDGE_PRE, _EDGE_POST, _EDGE_VAL),
    ],
    ids=["post_pre_coo", "post_pre_csr", "pre_post_coo", "pre_post_torch", "edges"],
)
def test_rules_with_values_become_trainable_weights(make_rule):
    """Without a synapse the values of ``FromSparse`` / ``FromEdges`` are the
    initial per-edge weights, and they are trainable.

    The orientation of a matrix is stated once, on the rule: ``"post_pre"`` is
    the operator ``(n_post, n_pre)``, ``"pre_post"`` the connectome adjacency
    ``(n_pre, n_post)``. Both describe the same connection here.
    """
    proj = Projection(N_PRE, N_POST, make_rule())
    weight = proj.connection.weight
    assert isinstance(weight, EdgeWeight) and isinstance(weight.value, nn.Parameter)
    assert weight.value.dtype == torch.float32
    # The duplicated pair was merged into one slot: 7 edges, 6 slots.
    assert proj.connection.nnz == _EDGE_PRE.shape[0] - 1
    x = _input(proj)
    torch.testing.assert_close(proj(x), x @ _W.T)
    torch.testing.assert_close(proj.connection.to_sparse("post_pre").to_dense(), _W)


def test_from_sparse_orientation_is_never_guessed():
    """A matrix in the other orientation has the wrong shape and is refused; it
    is not transposed to make it fit."""
    # The default orientation is "pre_post" (rows are sources), the same as
    # ``SparseConnection.from_adjacency``: an operator passed without saying
    # so is refused, not flipped.
    with pytest.raises(ValueError, match="never transposed"):
        Projection(N_PRE, N_POST, FromSparse(_OPERATOR))
    with pytest.raises(ValueError, match="never transposed"):
        Projection(N_PRE, N_POST, FromSparse(_OPERATOR, orientation="pre_post"))
    with pytest.raises(ValueError, match="never transposed"):
        Projection(N_PRE, N_POST, FromSparse(_ADJACENCY, orientation="post_pre"))


def test_synapse_weight_overrides_rule_values():
    """An explicit weight in the synapse replaces the values of the rule."""
    rule = FromEdges(_EDGE_PRE, _EDGE_POST, _EDGE_VAL)
    count = _dense(_EDGE_PRE, _EDGE_POST, torch.ones(7), N_PRE, N_POST)
    x = torch.rand(3, N_PRE, generator=_gen(1))

    constant = Projection(N_PRE, N_POST, rule, Synapse(weight=0.5))
    torch.testing.assert_close(constant(x), x @ (0.5 * count).T)

    other = torch.arange(1.0, 8.0)
    fixed = Projection(N_PRE, N_POST, rule, Synapse(weight=other))
    assert list(fixed.parameters()) == []
    A = _dense(_EDGE_PRE, _EDGE_POST, other, N_PRE, N_POST)
    torch.testing.assert_close(fixed(x), x @ A.T)


def test_from_edges_without_values_starts_from_unit_weights():
    """A bare edge list gives trainable weights of 1 per edge (2 for the merged
    parallel pair)."""
    proj = Projection(N_PRE, N_POST, FromEdges(_EDGE_PRE, _EDGE_POST))
    assert isinstance(proj.connection.weight.value, nn.Parameter)
    count = _dense(_EDGE_PRE, _EDGE_POST, torch.ones(7), N_PRE, N_POST)
    torch.testing.assert_close(proj.connection.to_sparse("post_pre").to_dense(), count)
    assert count.max() == 2


def test_from_sparse_scipy_uses_default_dtype_like_sparse_connection():
    """SciPy data is float64 by NumPy convention; ``SparseConnection(S)``
    stores it in the PyTorch default dtype and a projection does too.

    Regression test: the projection used to hand the float64 SciPy values on
    as a btorch array, for which the default-dtype rule did not apply. A
    btorch / PyTorch matrix is a deliberate choice of dtype and keeps it.
    """
    S = scipy.sparse.random(N_POST, N_PRE, density=0.4, format="csr", random_state=0)
    assert S.dtype == "float64"
    direct = SparseConnection(S)
    # ``S`` is the operator ``(n_post, n_pre)``, as for the constructor.
    proj = Projection(N_PRE, N_POST, FromSparse(S, orientation="post_pre"))
    assert direct.weight.value.dtype == torch.get_default_dtype()
    assert proj.connection.weight.value.dtype == direct.weight.value.dtype
    torch.testing.assert_close(
        proj.connection.to_sparse("post_pre").to_dense(),
        torch.as_tensor(S.toarray(), dtype=torch.get_default_dtype()),
    )
    # dtype= still overrides the rule, and a btorch matrix keeps its dtype.
    explicit = Projection(
        N_PRE, N_POST, FromSparse(S, orientation="post_pre"), dtype=torch.float64
    )
    assert explicit.connection.weight.value.dtype == torch.float64
    kept = Projection(
        N_PRE, N_POST, FromSparse(sparse.as_sparse(S), orientation="post_pre")
    )
    assert kept.connection.weight.value.dtype == torch.float64


# ------------------------------------------------------- per-edge synapse arrays
@pytest.mark.parametrize(
    "name", ["fixed_outdegree_distinct_no_autapses", "pairwise_bernoulli", "from_edges"]
)
def test_per_edge_arrays_align_with_rule_edges(name):
    """Entry ``e`` of every per-edge array belongs to edge ``e`` of
    ``rule.edges(...)``, whatever order the connection stores the edges in.

    The arrays below are functions of the edge index only, so a projection
    that paired them with the edges in any other order (for example the
    connection's sorted order; ``FixedOutdegree`` lists edges by source,
    ``FromEdges`` here is unsorted) gives another matrix.
    """
    rule, n_pre, n_post = RULES[name]
    pre, post, _ = _edges(name)
    n_edge = pre.shape[0]
    n_receptor, n_delay = 2, 3
    weight = torch.arange(1.0, n_edge + 1)
    receptor = torch.arange(n_edge) % n_receptor
    delay = (torch.arange(n_edge) // 2) % n_delay
    synapse = Synapse(
        weight=nn.Parameter(weight.clone()),
        receptor=receptor,
        delay=delay,
        n_receptor=n_receptor,
        n_delay=n_delay,
    )
    A = _dense(pre, post, weight, n_pre, n_post, receptor, n_receptor, delay, n_delay)

    proj = Projection(n_pre, n_post, rule, synapse, generator=_gen())

    x = _input(proj)
    torch.testing.assert_close(proj(x), x @ A.T)
    # The semantic edge list holds the same (pre, post, receptor, delay,
    # weight) tuples as the input. The duplicated pair of ``from_edges`` has
    # receptors 0 and 1, so it is *not* merged: all edges are kept.
    table = proj.connection.edge_table()
    assert proj.connection.nnz == n_edge
    got = sorted(
        zip(
            *(table[k].tolist() for k in ("pre", "post", "receptor", "delay", "weight"))
        )
    )
    want = sorted(
        zip(*(t.tolist() for t in (pre, post, receptor, delay, weight))),
    )
    assert got == want


# A rule that is certain to produce multapses: each of the 2 targets draws 6
# sources out of 3 with replacement.
_MULTI = FixedIndegree(6)
_MULTI_PRE, _MULTI_POST = 3, 2


def _multi_edges():
    pre, post = _MULTI.edges(_MULTI_PRE, _MULTI_POST, generator=_gen())
    count = _dense(pre, post, torch.ones(12), _MULTI_PRE, _MULTI_POST)
    assert count.max() > 1, "the rule must produce parallel edges for this test"
    return pre, post, count


def _multi_projection(synapse):
    return Projection(_MULTI_PRE, _MULTI_POST, _MULTI, synapse, generator=_gen())


def test_multapses_are_merged_and_weights_add():
    """Parallel edges between the same pair become one edge slot whose weight
    is the sum of the weights of the merged edges."""
    pre, post, count = _multi_edges()
    n_slot = int((count > 0).sum())
    assert n_slot < pre.shape[0]

    # Per-edge weights: the slot holds the sum.
    weight = torch.arange(1.0, 13.0)
    proj = _multi_projection(Synapse(weight=nn.Parameter(weight.clone())))
    assert proj.connection.nnz == n_slot
    A = _dense(pre, post, weight, _MULTI_PRE, _MULTI_POST)
    torch.testing.assert_close(proj.connection.to_sparse("post_pre").to_dense(), A)

    # No synapse: unit weight per edge, so the slot holds the multiplicity.
    proj = _multi_projection(None)
    torch.testing.assert_close(proj.connection.to_sparse("post_pre").to_dense(), count)
    x = _input(proj)
    torch.testing.assert_close(proj(x), x @ count.T)


def test_constant_weight_counts_multiplicity():
    """A fixed scalar weight applies to every edge of the rule, so a pair
    connected ``m`` times carries ``m * weight``.

    The multiplicity is model
    state (``weight.multiplicity``).
    """
    pre, post, count = _multi_edges()
    proj = _multi_projection(Synapse(weight=0.5))
    conn = proj.connection
    assert isinstance(conn.weight, ConstantWeight)
    torch.testing.assert_close(conn.to_sparse("post_pre").to_dense(), 0.5 * count)
    # One multiplicity per slot; together they account for every edge.
    assert conn.weight.multiplicity.shape == (conn.nnz,)
    assert conn.weight.multiplicity.sum() == pre.shape[0]
    assert "connection.weight.multiplicity" in proj.state_dict()
    x = _input(proj)
    torch.testing.assert_close(proj(x), x @ (0.5 * count).T)


@pytest.mark.parametrize("attribute", ["receptor", "delay"])
def test_parallel_edges_with_different_routing_stay_separate(attribute):
    """Receptor and delay are part of an edge's identity: two edges between the
    same pair that use different receptors (or delays) are different edges and
    are not merged.

    Only edges that agree on (pre, post, receptor, delay) are merged, so
    no conflict can arise.
    """
    pre, post, count = _multi_edges()
    ids = torch.arange(12) % 2
    weight = torch.arange(1.0, 13.0)
    proj = _multi_projection(Synapse(weight=weight, **{attribute: ids}))
    conn = proj.connection

    # Slots = distinct (pre, post, id) triples: more than the distinct pairs.
    triples = {(a, b, c) for a, b, c in zip(pre.tolist(), post.tolist(), ids.tolist())}
    assert conn.nnz == len(triples) > int((count > 0).sum())
    table = conn.edge_table()
    assert {
        (a, b, c)
        for a, b, c in zip(
            table["pre"].tolist(), table["post"].tolist(), table[attribute].tolist()
        )
    } == triples

    kwargs = {attribute: ids, f"n_{attribute}": 2}
    A = _dense(pre, post, weight, _MULTI_PRE, _MULTI_POST, **kwargs)
    x = _input(proj)
    torch.testing.assert_close(proj(x), x @ A.T)


def test_merged_edges_with_conflicting_groups_are_rejected():
    """A merged edge has one weight, hence one group.

    Parallel edges that belong to different groups cannot be merged and
    raise instead of silently keeping one of the groups.
    """
    pre, _, _ = _multi_edges()
    conflicting = torch.arange(12) % 2
    with pytest.raises(ValueError, match="different metadata"):
        _multi_projection(Synapse(weight=ConstrainedWeight(group=conflicting)))

    # Groups that are a function of the pair (here: of the source) agree on
    # every merged edge and are accepted.
    proj = _multi_projection(Synapse(weight=ConstrainedWeight(group=pre)))
    table = proj.connection.edge_table()
    assert torch.equal(table["group"], table["pre"])


@pytest.mark.parametrize(
    "make_synapse",
    [
        lambda: Synapse(weight=torch.ones(5)),
        lambda: Synapse(delay=torch.ones(5, dtype=torch.long)),
        lambda: Synapse(receptor=torch.ones(5, dtype=torch.long)),
        lambda: Synapse(
            weight=ConstrainedWeight(group=torch.zeros(5, dtype=torch.long))
        ),
    ],
    ids=["weight", "delay", "receptor", "group"],
)
def test_per_edge_array_of_wrong_length_is_rejected(make_synapse):
    """A per-edge array must have one entry per edge of the rule (12 here).

    Every such array is rejected with a ``ValueError`` that names the two
    lengths (5 given, 12 edges); nothing is silently broadcast or truncated.
    """
    with pytest.raises(ValueError, match=r"(?s)5.*12|12.*5"):
        _multi_projection(make_synapse())


def test_float_delay_is_rejected():
    """Delays are time steps and receptors are channel ids: integers."""
    with pytest.raises(TypeError, match="integers"):
        Projection(N_PRE, N_POST, AllToAll(), Synapse(delay=1.5))


# -------------------------------------------------- constrained weights and Dale
def test_constrained_weight_with_dale_through_a_projection():
    """``Synapse(weight=ConstrainedWeight(group=...), dale=True)``: the group
    ids align with the rule's edges, one scale per group is trained, and the
    ``dale`` flag of the synapse switches the constraint of the connection's
    weight module on (scales stay non-negative).

    A weight module is used as is: it *becomes* ``conn.weight`` (the object
    the user holds is the one that is trained and constrained), which is why
    it serves exactly one connection. Describing a second projection with
    the same module is an error instead of two networks silently sharing, or
    silently not sharing, their scales.
    """
    rule, n_pre, n_post = RULES["fixed_indegree_distinct"]
    pre, post, _ = _edges("fixed_indegree_distinct")
    group = pre % 3  # group of an edge = a function of its source
    weight = ConstrainedWeight(group=group)
    assert weight.dale is False

    proj = Projection(
        n_pre, n_post, rule, Synapse(weight=weight, dale=True), generator=_gen()
    )
    conn = proj.connection

    # Identity: the user's module is the connection's weight (and what
    # ``Projection.weight`` delegates to), with the Dale flag switched on.
    assert conn.weight is weight
    assert proj.weight is weight
    assert weight.dale is True
    # It is bound to this connection now; a second one needs its own module.
    with pytest.raises(RuntimeError, match="already bound"):
        Projection(
            n_pre, n_post, rule, Synapse(weight=weight, dale=True), generator=_gen()
        )
    assert [n for n, _ in proj.named_parameters()] == ["connection.weight.scale"]
    table = conn.edge_table()
    assert torch.equal(table["group"], table["pre"] % 3)

    # Effective weight = base (1 per edge) * scale[group].
    scale = torch.tensor([2.0, -1.0, 0.5])
    with torch.no_grad():
        weight.scale.copy_(scale)
    x = _input(proj)
    A = _dense(pre, post, scale[group], n_pre, n_post)
    torch.testing.assert_close(proj(x), x @ A.T)

    # The projection (run by constrain_net after an optimizer step) clamps
    # the negative scale to zero; it is found through the projection module.
    constrain_net(proj)
    torch.testing.assert_close(weight.scale.detach(), scale.clamp(min=0))
    A = _dense(pre, post, scale.clamp(min=0)[group], n_pre, n_post)
    torch.testing.assert_close(proj(x), x @ A.T)
    # Binding aligned the module with the connection's edge slots: its group
    # ids are now in slot order (still "source % 3" for every edge), not in
    # the rule's edge order they were given in.
    assert torch.equal(weight.group, conn.pre % 3)
    assert weight.group.shape == (conn.nnz,)


def test_dale_on_per_edge_weights_through_a_projection():
    """``Synapse(dale=True)`` without a weight keeps the sign of the rule's
    values: a weight that crossed zero is clamped by ``constrain_net``."""
    proj = Projection(
        N_PRE, N_POST, FromEdges(_EDGE_PRE, _EDGE_POST, _EDGE_VAL), Synapse(dale=True)
    )
    weight = proj.connection.weight
    sign = weight.sign.clone()
    assert set(sign.tolist()) == {-1.0, 1.0}
    with torch.no_grad():
        weight.value.mul_(-1.0)  # every weight now has the wrong sign
    constrain_net(proj)
    assert torch.equal(weight.value.detach(), torch.zeros_like(sign))
    assert torch.equal(weight.sign, sign)


def test_dale_with_a_weight_module_that_cannot_enforce_it():
    """Asking for Dale's law on a weight that has no sign to keep is an error,
    not a silent no-op."""
    with pytest.raises(ValueError, match="Dale"):
        Projection(
            N_PRE,
            N_POST,
            FixedIndegree(2),
            Synapse(weight=ConstantWeight(0.5), dale=True),
        )


# -------------------------------------------------------------------- gradients
@pytest.mark.parametrize(
    "name", ["fixed_indegree", "pairwise_bernoulli", "all_to_all", "from_edges"]
)
def test_gradients_reach_trainable_weights(name):
    """The gradient of every edge slot equals the gradient of the matching
    entry of a dense trainable matrix (the sum over merged parallel edges is
    one parameter)."""
    rule, n_pre, n_post = RULES[name]
    proj = Projection(n_pre, n_post, rule, generator=_gen())
    conn = proj.connection
    x = _input(proj, batch=(4,)).requires_grad_()
    target = torch.randn(4, n_post, generator=_gen(5))

    loss = ((proj(x) - target) ** 2).sum()
    grad_w, grad_x = torch.autograd.grad(loss, [conn.weight.value, x])

    # Dense twin with the same weights.
    A = conn.to_sparse("post_pre").to_dense().detach().requires_grad_()
    loss_ref = ((x @ A.T - target) ** 2).sum()
    grad_A, grad_x_ref = torch.autograd.grad(loss_ref, [A, x])

    post, pre = conn.indices
    torch.testing.assert_close(grad_w, grad_A[post, pre])
    torch.testing.assert_close(grad_x, grad_x_ref)
    assert grad_w.abs().sum() > 0


def test_optimizer_step_trains_a_projection():
    """An ordinary training step through ``proj.parameters()`` lowers the loss;
    the edges do not move."""
    proj = Projection(N_PRE, N_POST, FixedIndegree(3), generator=_gen())
    optimizer = torch.optim.SGD(proj.parameters(), lr=0.05)
    x = _input(proj, batch=(8,))
    target = torch.randn(8, N_POST, generator=_gen(5))
    indices = proj.connection.indices.clone()

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = ((proj(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert torch.equal(proj.connection.indices, indices)


@pytest.mark.parametrize("name", list(STRUCTURED_RULES))
def test_gradient_flows_through_a_structured_projection(name):
    """A structured projection has no parameters; the gradient with respect to
    its input is that of the dense matrix."""
    rule, n_pre, n_post = STRUCTURED_RULES[name]
    proj = Projection(n_pre, n_post, rule, Synapse(weight=0.5))
    pre, post = rule.edges(n_pre, n_post)
    A = _dense(pre, post, torch.full((pre.shape[0],), 0.5), n_pre, n_post)
    x = _input(proj).requires_grad_()
    coeff = torch.randn(3, n_post, generator=_gen(5))
    (grad,) = torch.autograd.grad((proj(x) * coeff).sum(), x)
    torch.testing.assert_close(grad, coeff @ A)


# ---------------------------------------------------------------- torch.compile
@pytest.fixture
def fresh_dynamo():
    """Each compile test starts without cached graphs."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _routed_projection():
    """Sparse projection with trainable weights, receptors and delays."""
    rule, n_pre, n_post = RULES["fixed_indegree_distinct"]
    n_edge = int(rule.expected_nnz(n_pre, n_post))
    synapse = Synapse(receptor=torch.arange(n_edge) % 2, delay=torch.arange(n_edge) % 3)
    return Projection(n_pre, n_post, rule, synapse, generator=_gen())


@pytest.mark.parametrize(
    "make",
    [
        lambda: Projection(N_PRE, N_POST, FixedIndegree(4), generator=_gen()),
        _routed_projection,
        lambda: Projection(N_PRE, N_POST, PairwiseBernoulli(0.4), Synapse(weight=0.5)),
    ],
    ids=["trainable", "routed", "constant"],
)
def test_compile_sparse_projection(make, fresh_dynamo):
    """``torch.compile(proj, fullgraph=True)`` works for a sparse projection:

    one graph, same output and same weight gradient as eager.
    """
    proj = make()
    compiled = torch.compile(proj, fullgraph=True)
    x = _input(proj, batch=(4,))
    torch.testing.assert_close(compiled(x), proj(x))
    # A different batch size and leading dimensions (recompiles are allowed).
    x2 = _input(proj, batch=(2, 3), seed=7)
    torch.testing.assert_close(compiled(x2), proj(x2))

    params = list(proj.parameters())
    if params:
        grad_c = torch.autograd.grad(compiled(x).square().sum(), params)
        grad_e = torch.autograd.grad(proj(x).square().sum(), params)
        for c, e in zip(grad_c, grad_e):
            torch.testing.assert_close(c, e)


@pytest.mark.parametrize("name", list(STRUCTURED_RULES))
def test_compile_structured_projection(name, fresh_dynamo):
    """``torch.compile(proj, fullgraph=True)`` works for a structured
    projection, including the gradient with respect to the input."""
    rule, n_pre, n_post = STRUCTURED_RULES[name]
    proj = Projection(n_pre, n_post, rule, Synapse(weight=0.5))
    assert isinstance(proj.connection, StructuredConnection)
    compiled = torch.compile(proj, fullgraph=True)
    x = _input(proj, batch=(4,)).requires_grad_()
    torch.testing.assert_close(compiled(x), proj(x))
    (grad_c,) = torch.autograd.grad(compiled(x).square().sum(), x)
    (grad_e,) = torch.autograd.grad(proj(x).square().sum(), x)
    torch.testing.assert_close(grad_c, grad_e)


# -------------------------------------------------------------------- state_dict
def _routing_arrays(n_edge):
    return {
        "receptor": torch.arange(n_edge) % 2,
        "delay": torch.arange(n_edge) % 3,
        "n_receptor": 2,
        "n_delay": 3,
    }


# kind -> (synapse factory taking n_edge and a seed, expected state keys).
# ``layout`` = [n_post, n_pre, n_receptor, n_delay] describes the connection
# the edge ids refer to; it is saved so that a checkpoint cannot be loaded
# into a connection between other populations. Buffers of the connection come
# first, then the state of its weight module.
STATE_KINDS = {
    "default": (lambda n, s: None, ["indices", "layout", "weight.value"]),
    "per_edge": (
        lambda n, s: Synapse(weight=nn.Parameter(torch.randn(n, generator=_gen(s)))),
        ["indices", "layout", "weight.value"],
    ),
    "constant": (
        lambda n, s: Synapse(weight=0.5 + s),
        ["indices", "layout", "weight.value", "weight.multiplicity"],
    ),
    "dale": (
        lambda n, s: Synapse(
            weight=nn.Parameter(torch.randn(n, generator=_gen(s))), dale=True
        ),
        ["indices", "layout", "weight.value", "weight.sign"],
    ),
    "routed": (
        lambda n, s: Synapse(
            weight=nn.Parameter(torch.randn(n, generator=_gen(s))),
            **_routing_arrays(n),
        ),
        ["indices", "receptor", "delay", "layout", "weight.value"],
    ),
    "constrained": (
        lambda n, s: Synapse(
            weight=ConstrainedWeight(
                group=(torch.arange(n) + s) % 3,
                base=torch.randn(n, generator=_gen(s)),
                scale=torch.tensor([1.0, 2.0, 3.0]) + s,
            )
        ),
        ["indices", "layout", "weight.scale", "weight.group", "weight.base"],
    ),
}


@pytest.mark.parametrize("kind", list(STATE_KINDS))
@pytest.mark.parametrize(
    "name", ["fixed_indegree_distinct", "fixed_outdegree_distinct_no_autapses"]
)
def test_checkpoint_restores_the_saved_edges(name, kind):
    """A stochastic rule draws new edges every time a projection is built, so
    the module that receives a checkpoint starts with *other* edges. Loading
    must replace them: afterwards the module is the saved network.

    The two rules used here always produce the same number of distinct edges,
    which is what makes the checkpoint loadable (see the next test).
    """
    rule, n_pre, n_post = RULES[name]
    make_synapse, keys = STATE_KINDS[kind]
    n_edge = int(rule.expected_nnz(n_pre, n_post))

    saved = Projection(n_pre, n_post, rule, make_synapse(n_edge, 0), generator=_gen(0))
    fresh = Projection(n_pre, n_post, rule, make_synapse(n_edge, 1), generator=_gen(1))
    x = _input(saved, batch=(4,))
    expected = saved(x).detach()
    # The precondition of the test: the fresh module is another network.
    assert not torch.equal(saved.connection.indices, fresh.connection.indices)
    assert not torch.allclose(fresh(x), expected)

    state = saved.state_dict()
    assert list(state) == [f"connection.{k}" for k in keys]
    fresh.load_state_dict(state)

    assert torch.equal(fresh.connection.indices, saved.connection.indices)
    # Derived execution layouts were rebuilt: the output is the saved one.
    torch.testing.assert_close(fresh(x), expected)
    torch.testing.assert_close(
        fresh.connection.to_sparse("post_pre").to_dense(),
        saved.connection.to_sparse("post_pre").to_dense(),
    )
    # The rule object is not part of the state and was not consulted again.
    assert not any("rule" in k for k in state)


@pytest.mark.parametrize("name", ["pairwise_bernoulli", "fixed_indegree"])
def test_checkpoint_needs_the_same_number_of_edge_slots(name):
    """The number of edge slots is fixed at construction.

    Rules whose edge
    count is random (``PairwiseBernoulli``) or that produce multapses
    (merged into fewer slots) give a different count for a different seed,
    and such a checkpoint is refused with a size mismatch instead of being
    loaded partially. Rebuilding with the same seed gives a module that
    accepts it.
    """
    rule, n_pre, n_post = RULES[name]
    saved = Projection(n_pre, n_post, rule, generator=_gen(0))
    with torch.no_grad():
        saved.connection.weight.value.mul_(3.0)  # "trained" weights
    other = Projection(n_pre, n_post, rule, generator=_gen(1))
    assert saved.connection.nnz != other.connection.nnz, "pick seeds that differ"

    before = other.connection.indices.clone()
    with pytest.raises(RuntimeError, match="size mismatch"):
        other.load_state_dict(saved.state_dict())
    assert torch.equal(other.connection.indices, before)

    same_seed = Projection(n_pre, n_post, rule, generator=_gen(0))
    same_seed.load_state_dict(saved.state_dict())
    x = _input(saved)
    torch.testing.assert_close(same_seed(x), saved(x))


def test_structured_projection_state_is_its_operator_tensors():
    """The state of a structured projection is the handful of tensors its
    closed-form operator consists of (here the unit diagonal of ``OneToOne``),
    saved as ``connection.operator_<i>_<attr>``.

    The scalar weight is a constructor argument and not part of the
    state. There is no edge list, so the checkpoint of an explicit
    realisation of the same rule does not fit.
    """
    synapse = Synapse(weight=0.5)
    auto = Projection(N_SQ, N_SQ, OneToOne(), synapse)
    explicit = Projection(N_SQ, N_SQ, OneToOne(), synapse, realization="sparse")
    state = auto.state_dict()
    assert list(state) == ["connection.operator_0_diag"]
    torch.testing.assert_close(state["connection.operator_0_diag"], torch.ones(N_SQ))
    assert list(auto.parameters()) == []

    # The saved tensor is the one the operator applies: a checkpoint with
    # another diagonal changes the output of the module that loads it.
    x = _input(auto)
    torch.testing.assert_close(auto(x), 0.5 * x)
    diag = torch.arange(1.0, N_SQ + 1)
    other = Projection(N_SQ, N_SQ, OneToOne(), synapse)
    other.load_state_dict({"connection.operator_0_diag": diag})
    torch.testing.assert_close(other(x), 0.5 * diag * x)
    # ... and a plain round trip leaves the module as it was.
    other.load_state_dict(state)
    torch.testing.assert_close(other(x), auto(x))

    with pytest.raises(RuntimeError, match="Unexpected key"):
        auto.load_state_dict(explicit.state_dict())


def test_projection_inside_a_model_checkpoint():
    """The usual situation: projections are submodules of a model and the
    whole model is saved and restored."""

    def build(seed):
        g = _gen(seed)
        model = nn.Module()
        model.exc = Projection(
            N_PRE, N_POST, FixedIndegree(3, allow_multapses=False), generator=g
        )
        model.inh = Projection(
            N_POST, N_POST, AllToAll(allow_autapses=False), Synapse(weight=-0.2)
        )
        return model

    a, b = build(0), build(1)
    b.load_state_dict(a.state_dict())
    x = torch.rand(3, N_PRE, generator=_gen(1))
    torch.testing.assert_close(b.exc(x), a.exc(x))
    y = torch.rand(3, N_POST, generator=_gen(2))
    torch.testing.assert_close(b.inh(y), a.inh(y))
    # The sparse projection contributes its edges, layout and weights; the
    # structured one only the tensors of its closed-form operator.
    keys = list(a.state_dict())
    assert [k for k in keys if k.startswith("exc.")] == [
        "exc.connection.indices",
        "exc.connection.layout",
        "exc.connection.weight.value",
    ]
    inh = [k for k in keys if not k.startswith("exc.")]
    assert inh and all(k.startswith("inh.connection.operator_") for k in inh)


# ------------------------------------------------------- in_features/out_features
@pytest.mark.parametrize(
    "routing, n_delay, n_receptor",
    [
        ({}, 1, 1),
        # An int applies to every edge; the count defaults to id + 1.
        ({"delay": 2}, 3, 1),
        ({"receptor": 1}, 1, 2),
        ({"delay": 2, "receptor": 1}, 3, 2),
        # Explicit counts may be larger than the ids in use.
        ({"delay": 0, "n_delay": 4}, 4, 1),
        ({"receptor": 0, "n_receptor": 3}, 1, 3),
        ({"delay": 1, "n_delay": 4, "receptor": 2, "n_receptor": 5}, 4, 5),
    ],
    ids=["none", "delay", "receptor", "both", "n_delay", "n_receptor", "both_counts"],
)
@pytest.mark.parametrize("name", ["all_to_all", "fixed_indegree"])
def test_features_with_delays_and_receptors(name, routing, n_delay, n_receptor):
    """The projection reads ``[..., n_pre * n_delay]`` (index.

    ``pre * n_delay + delay``) and writes ``[..., n_post * n_receptor]`` (index
    ``post * n_receptor + receptor``); ``n_pre`` / ``n_post`` stay neuron
    counts.
    """
    rule, n_pre, n_post = RULES[name]
    proj = Projection(
        n_pre, n_post, rule, Synapse(weight=0.5, **routing), generator=_gen()
    )
    assert (proj.n_pre, proj.n_post) == (n_pre, n_post)
    assert proj.in_features == n_pre * n_delay
    assert proj.out_features == n_post * n_receptor

    x = _input(proj, batch=(2,))
    out = proj(x)
    assert out.shape == (2, n_post * n_receptor)
    # Only the selected delay bin is read and only the selected receptor
    # channel is written.
    pre, post, _ = _edges(name)
    count = _dense(pre, post, torch.full((pre.shape[0],), 0.5), n_pre, n_post)
    delay, receptor = routing.get("delay", 0), routing.get("receptor", 0)
    x_bin = x.reshape(2, n_pre, n_delay)[..., delay]
    out = out.reshape(2, n_post, n_receptor)
    torch.testing.assert_close(out[..., receptor], x_bin @ count.T)
    other = [r for r in range(n_receptor) if r != receptor]
    assert torch.count_nonzero(out[..., other]) == 0

    with pytest.raises(ValueError, match="last dimension"):
        proj(torch.rand(2, proj.in_features + 1))


def test_per_edge_ids_out_of_range_are_rejected():
    """An id that does not fit the declared number of channels is an error."""
    with pytest.raises(ValueError, match="out of range"):
        Projection(N_PRE, N_POST, AllToAll(), Synapse(delay=3, n_delay=3))
    with pytest.raises(ValueError, match="non-negative"):
        Projection(N_PRE, N_POST, AllToAll(), Synapse(receptor=-1))


# ------------------------------------------------------ architectural invariants
# Spec sections 12, 52 and 58 as executable checks.

# Names that would mean a rule knows how it is stored or executed.
_EXECUTION_NAMES = {
    "format",
    "nnz",
    "indptr",
    "indices",
    "crow_indices",
    "col_indices",
    "matvec",
    "rmatvec",
    "tocsr",
    "tocoo",
    "backend",
    "algorithm",
    "realization",
    "hints",
    "cache",
    "forward",
}


@pytest.mark.parametrize("name", RULE_NAMES)
def test_rule_is_not_a_sparse_array_and_holds_no_execution_format(name):
    """``ConnectionRule != Sparse`` and ``ConnectionRule != execution
    representation`` (invariants 3 and 4).

    A rule is a plain parameter object: not a sparse array, not a linear
    operator, not a module, and without any attribute describing a storage
    format, an algorithm or a backend. Data-carrying rules keep the data they
    were given (``FromSparse.matrix`` is the user's matrix, in the user's
    orientation); that is input, not an execution layout.
    """
    rule, n_pre, n_post = RULES[name]
    assert isinstance(rule, ConnectionRule)
    assert not isinstance(rule, Sparse)
    assert not isinstance(rule, nn.Module)
    assert not is_linear_operator(rule)
    assert _EXECUTION_NAMES.isdisjoint(dir(rule))

    # What the rule returns is an edge list in population-local indices.
    pre, post = rule.edges(n_pre, n_post, generator=_gen())
    assert pre.dtype == post.dtype == torch.long and pre.shape == post.shape
    assert pre.ndim == 1

    # Building projections of either realisation leaves the rule untouched:
    # no cached plan, layout or connection is written back to it.
    before = dict(vars(rule))
    for realization in ("auto", "sparse"):
        Projection(
            n_pre,
            n_post,
            rule,
            Synapse(weight=0.5),
            generator=_gen(),
            realization=realization,
        )
    after = vars(rule)
    assert after.keys() == before.keys()
    assert all(after[k] is before[k] for k in before)
    assert not any(isinstance(v, RepresentationCache) for v in after.values())


def test_rule_class_is_unrelated_to_sparse():
    """No rule class derives from, or is registered as, a sparse array."""
    assert not issubclass(ConnectionRule, Sparse)
    assert not issubclass(Sparse, ConnectionRule)
    for cls in (
        OneToOne,
        AllToAll,
        FixedIndegree,
        FixedOutdegree,
        PairwiseBernoulli,
        DistanceDependent,
        FromEdges,
        FromSparse,
    ):
        assert issubclass(cls, ConnectionRule) and not issubclass(cls, Sparse)


@pytest.mark.parametrize(
    "make",
    [
        lambda: Projection(N_SQ, N_SQ, OneToOne(), Synapse(weight=0.5)),
        lambda: Projection(N_PRE, N_POST, FixedIndegree(3), generator=_gen()),
    ],
    ids=["structured", "sparse"],
)
def test_projection_is_not_a_sparse_array(make):
    """``Projection != Sparse`` (invariant 1): a projection is a module that
    owns a connection; it has no format, no ``nnz`` and no ``@``."""
    proj = make()
    assert isinstance(proj, nn.Module)
    assert not isinstance(proj, Sparse)
    assert not issubclass(Projection, Sparse)
    assert not is_linear_operator(proj)
    for attr in ("format", "nnz", "indptr", "indices", "tocsr", "matvec"):
        assert not hasattr(proj, attr), attr
    with pytest.raises(TypeError):
        proj @ torch.ones(proj.in_features)
    # The model-level API is the call.
    assert proj(torch.ones(proj.in_features)).shape == (proj.out_features,)


def test_one_rule_object_serves_several_realisations():
    """The same rule object is realised as a structured operator, as explicit
    edges, and for other population sizes: the rule decides none of this."""
    rule = AllToAll()
    structured = Projection(N_PRE, N_POST, rule, Synapse(weight=1.0))
    explicit = Projection(N_PRE, N_POST, rule)
    smaller = Projection(3, 2, rule)
    assert isinstance(structured.connection, StructuredConnection)
    assert isinstance(explicit.connection, SparseConnection)
    assert smaller.connection.nnz == 6
    assert structured.rule is explicit.rule is smaller.rule is rule
    # The sparse connection's execution layout lives in its cache, not in the
    # rule and not in the projection.
    assert isinstance(explicit.connection.cache, RepresentationCache)


def test_projection_signature_has_no_execution_choices():
    """Traversal, format and backend are not arguments of ``Projection`` or of
    any rule (section 52: "do not put traversal choice into Projection").

    The only realisation switch is structured-versus-explicit, and ``hints``
    are expectations, not instructions.
    """
    forbidden = {"format", "layout", "backend", "algorithm", "traversal", "kernel"}
    assert forbidden.isdisjoint(inspect.signature(Projection.__init__).parameters)
    for rule, _, _ in RULES.values():
        params = inspect.signature(type(rule).__init__).parameters
        assert forbidden.isdisjoint(params)
        assert forbidden.isdisjoint(inspect.signature(rule.edges).parameters)
