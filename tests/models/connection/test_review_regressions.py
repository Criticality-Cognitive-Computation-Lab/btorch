"""Regression tests for the second review of the connection layer.

Each section pins down one deliberate change of behaviour. The sections are
independent; the heading of each states the rule that is tested.

Conventions used throughout:

- a *connection matrix* / *operator* ``A`` has shape ``(n_post, n_pre)`` and
  the connection computes ``y = x @ A.T`` along the last axis;
- an *edge slot* is one stored entry of a ``SparseConnection``; slots are
  sorted by target and then by source, which generally differs from the order
  in which the user supplied the edges;
- every numerical expectation comes from a dense matrix built by scattering
  the user's edge list (``_dense``), independently of ``btorch.sparse``.

Populations have different sizes, so a transposed result has the wrong shape.
A test marked ``xfail(strict=True)`` documents a confirmed library bug; its
``reason`` names the offending line.
"""

import copy
import io

import numpy as np
import pytest
import scipy.sparse
import torch
from torch import nn

from btorch import sparse
from btorch.analysis.dynamic_tools import complexity
from btorch.models.connection import (
    AllToAll,
    ConstantWeight,
    ConstrainedWeight,
    EdgeWeight,
    FromSparse,
    ImplicitConnection,
    OneToOne,
    PairwiseBernoulli,
    Projection,
    SparseConnection,
    StructuredConnection,
    Synapse,
)
from btorch.sparse.operator import (
    ConstantOperator,
    DiagonalOperator,
    ImplicitOperator,
    LowRankOperator,
)
from btorch.sparse.runtime import registry, use_backend
from tests.sparse.helpers import DEVICES


N_PRE, N_POST = 7, 5
# Two networks between the same populations. Both edge lists are unsorted, so
# the order of the edge slots differs from the order given here, and they
# have different numbers of edges.
PRE_A = torch.tensor([6, 0, 3, 1, 5, 2])
POST_A = torch.tensor([4, 4, 0, 2, 1, 0])
W_A = torch.tensor([0.5, -1.0, 2.0, -0.25, 1.5, -3.0])
PRE_B = torch.tensor([2, 6, 0, 4, 1, 3, 5, 0])
POST_B = torch.tensor([3, 1, 0, 4, 1, 2, 0, 2])
W_B = torch.tensor([-2.0, 0.75, 1.25, -0.5, 3.0, -1.5, 0.25, 4.0])
# A third pattern with as many edges as A (for checkpoints and per-edge arrays).
PRE_C = torch.tensor([1, 4, 2, 6, 0, 3])
POST_C = torch.tensor([0, 3, 2, 1, 4, 3])


def _dense(pre, post, weight, n_pre=N_PRE, n_post=N_POST):
    """Dense reference ``A [n_post, n_pre]`` of an edge list (parallel edges
    add up)."""
    weight = torch.as_tensor(weight)
    A = torch.zeros(n_post, n_pre, dtype=weight.dtype)
    return A.index_put((torch.as_tensor(post), torch.as_tensor(pre)), weight, True)


def _matrix(conn):
    """The operator ``[n_post, n_pre]`` a connection currently applies, as a
    dense tensor.

    The orientation is named: without an argument ``to_sparse`` returns the
    layout the connection was built from (``pre_post`` for an edge list).
    """
    return conn.to_sparse("post_pre").to_dense().detach()


def _x(n=N_PRE, batch=(3,), seed=1, dtype=None):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(*batch, n, generator=gen).to(dtype or torch.get_default_dtype())


def _state_copy(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def _assert_state_equal(module, snapshot):
    state = module.state_dict()
    assert list(state) == list(snapshot)
    for key, value in snapshot.items():
        assert state[key].dtype == value.dtype, key
        assert torch.equal(state[key], value), key


# =============================================================================
# 1. A weight module belongs to one connection; a Synapse without one is a
#    reusable description
# =============================================================================
# ``Synapse.make_weight()`` returns a user-supplied weight *module* as is: it
# becomes ``conn.weight`` (same object), the connection aligns ("binds") it
# with its own edge slots, and binding it to a second connection raises.
# Every other kind of ``weight`` (``None``, a number, a tensor, a callable, a
# distribution) creates a fresh module per connection, so such a ``Synapse``
# can describe any number of connections.
def test_edge_weight_module_is_adopted_and_a_plain_synapse_is_reusable():
    """``Synapse(weight=EdgeWeight(dale=True))``: the module is the
    connection's weight, and a second connection cannot have it too.

    The reusable spelling of the same thing is ``Synapse(dale=True)``: used
    for two different networks, each connection gets its own weights and
    Dale signs from its own matrix.
    """
    user_weight = EdgeWeight(dale=True)
    module_synapse = Synapse(weight=user_weight)
    adopted = SparseConnection.from_edges(
        PRE_A, POST_A, N_PRE, N_POST, module_synapse, values=W_A
    )
    # Identity, not a copy: what the user holds is what is trained.
    assert adopted.weight is user_weight
    assert user_weight.value is adopted.weight.value
    torch.testing.assert_close(_matrix(adopted), _dense(PRE_A, POST_A, W_A))
    # One module, one connection. The refused construction changes nothing.
    before = _state_copy(adopted)
    with pytest.raises(RuntimeError, match="already bound"):
        SparseConnection.from_edges(
            PRE_B, POST_B, N_PRE, N_POST, module_synapse, values=W_B
        )
    _assert_state_equal(adopted, before)
    torch.testing.assert_close(_matrix(adopted), _dense(PRE_A, POST_A, W_A))

    # The description without a module serves both networks.
    synapse = Synapse(dale=True)
    conn_a = SparseConnection.from_edges(
        PRE_A, POST_A, N_PRE, N_POST, synapse, values=W_A
    )
    conn_b = SparseConnection.from_edges(
        PRE_B, POST_B, N_PRE, N_POST, synapse, values=W_B
    )

    for conn, pre, post, w in (
        (conn_a, PRE_A, POST_A, W_A),
        (conn_b, PRE_B, POST_B, W_B),
    ):
        A = _dense(pre, post, w)
        torch.testing.assert_close(_matrix(conn), A)
        torch.testing.assert_close(conn(_x()), _x() @ A.T)
        # The Dale reference sign of every slot is the sign of its own edge.
        slot_post, slot_pre = conn.indices
        assert torch.equal(conn.weight.sign, torch.sign(A[slot_post, slot_pre]))
        assert conn.weight.dale is True

    # Three different modules: the adopted one and one per reuse.
    assert conn_a.weight is not user_weight and conn_b.weight is not user_weight
    assert conn_a.weight is not conn_b.weight
    assert conn_a.weight.value is not conn_b.weight.value

    # The connections train independently.
    with torch.no_grad():
        conn_a.weight.value.mul_(-2.0)  # every weight crosses zero
    conn_a.constrain()  # Dale's law clamps them to zero
    torch.testing.assert_close(_matrix(conn_a), torch.zeros(N_POST, N_PRE))
    torch.testing.assert_close(_matrix(conn_b), _dense(PRE_B, POST_B, W_B))


def test_edge_weight_with_values_is_not_reordered_by_the_connections():
    """Explicit per-edge values are aligned with the edges *as supplied*.

    Two connections with different edge lists each place value ``k`` on their
    own edge ``k``; the user's tensor keeps its order. A tensor in a
    ``Synapse`` is copied per connection, so the synapse is reusable; an
    ``EdgeWeight`` module holding the same values is bound in place (it then
    holds them in slot order) and serves one connection only.
    """
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    synapse = Synapse(weight=values)
    conn_a = SparseConnection.from_edges(PRE_A, POST_A, N_PRE, N_POST, synapse)
    conn_c = SparseConnection.from_edges(PRE_C, POST_C, N_PRE, N_POST, synapse)

    torch.testing.assert_close(_matrix(conn_a), _dense(PRE_A, POST_A, values))
    torch.testing.assert_close(_matrix(conn_c), _dense(PRE_C, POST_C, values))
    # Slot order is not input order for these edge lists: the connection
    # holds a permutation of ``values``, the user's tensor is untouched.
    assert not torch.equal(conn_a.weight.value.detach(), values)
    assert torch.equal(values, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    # The module spelling: same matrix, but the module itself was aligned.
    user_weight = EdgeWeight(values)
    module_synapse = Synapse(weight=user_weight)
    conn_m = SparseConnection.from_edges(PRE_A, POST_A, N_PRE, N_POST, module_synapse)
    assert conn_m.weight is user_weight
    torch.testing.assert_close(_matrix(conn_m), _dense(PRE_A, POST_A, values))
    assert torch.equal(user_weight.value.detach(), conn_a.weight.value.detach())
    assert not torch.equal(user_weight.value.detach(), values)
    with pytest.raises(RuntimeError, match="already bound"):
        SparseConnection.from_edges(PRE_C, POST_C, N_PRE, N_POST, module_synapse)
    # The refused bind did not re-align the first connection's weights.
    torch.testing.assert_close(_matrix(conn_m), _dense(PRE_A, POST_A, values))


def _group_matrix():
    """1-based group id of *every* pair ``(post, pre)``: ``1 + (post + pre) %
    3``.

    A group matrix is matched with a connection matrix by coordinate, so
    one matrix covering all pairs fits every pattern between the
    populations.
    """
    post, pre = torch.meshgrid(torch.arange(N_POST), torch.arange(N_PRE), indexing="ij")
    return scipy.sparse.coo_array(((post + pre) % 3 + 1).numpy())


def test_one_constrained_weight_module_per_connection():
    """One ``ConstrainedWeight`` per connection (group ids given as a matrix,
    three group scales), for two networks with different patterns.

    Each connection computes ``w[e] = base[e] * scale[group[e]]`` with the
    base weights of its own matrix and the groups found at its own
    coordinates. The module handed in is the connection's weight; the same
    module for a second network is refused.
    """
    scale = torch.tensor([2.0, -1.0, 0.5])
    networks = ((PRE_A, POST_A, W_A), (PRE_B, POST_B, W_B))
    # One module per connection, built from the same description.
    modules = [ConstrainedWeight(group=_group_matrix(), scale=scale) for _ in networks]
    conns = [
        SparseConnection.from_edges(
            pre, post, N_PRE, N_POST, Synapse(weight=module), values=w
        )
        for module, (pre, post, w) in zip(modules, networks)
    ]
    for conn, module in zip(conns, modules):
        assert conn.weight is module
    # Reusing the first module for the second network raises and leaves the
    # connection that owns it exactly as it was.
    before = _state_copy(conns[0])
    with pytest.raises(RuntimeError, match="already bound"):
        SparseConnection.from_edges(
            PRE_B, POST_B, N_PRE, N_POST, Synapse(weight=modules[0]), values=W_B
        )
    _assert_state_equal(conns[0], before)
    for conn, (pre, post, w) in zip(conns, networks):
        A = _dense(pre, post, w * scale[(post + pre) % 3])
        torch.testing.assert_close(_matrix(conn), A)
        torch.testing.assert_close(conn(_x()), _x() @ A.T)
        table = conn.edge_table()
        assert torch.equal(table["group"], (table["post"] + table["pre"]) % 3)
        assert [n for n, _ in conn.named_parameters()] == ["weight.scale"]

    # Binding filled the user's modules in: one group id and one base weight
    # per edge slot of *their* connection.
    for conn, module in zip(conns, modules):
        assert module.group.shape == module.base.shape == (conn.nnz,)

    # Training one connection leaves the other alone.
    conn_a, conn_b = conns
    assert conn_a.weight.scale is not conn_b.weight.scale
    with torch.no_grad():
        conn_a.weight.scale.fill_(1.0)
    torch.testing.assert_close(_matrix(conn_a), _dense(PRE_A, POST_A, W_A))
    A_b = _dense(PRE_B, POST_B, W_B * scale[(POST_B + PRE_B) % 3])
    torch.testing.assert_close(_matrix(conn_b), A_b)
    torch.testing.assert_close(modules[1].scale.detach(), scale)


def test_constrained_weight_with_per_edge_groups_is_aligned_with_input_order():
    """Per-edge group ids are aligned with the supplied edges, like per-edge
    values: group ``k`` belongs to edge ``k`` of the edge list, whatever slot
    the connection stores that edge in. The caller's tensors are not
    modified; the module is (it is the connection's weight)."""
    group = torch.tensor([0, 1, 2, 2, 1, 0])
    scale = torch.tensor([2.0, -1.0, 0.5])
    for pre, post in ((PRE_A, POST_A), (PRE_C, POST_C)):
        user_weight = ConstrainedWeight(group=group, scale=scale)
        assert user_weight.dale is False
        conn = SparseConnection.from_edges(
            pre, post, N_PRE, N_POST, Synapse(weight=user_weight, dale=True), values=W_A
        )
        torch.testing.assert_close(_matrix(conn), _dense(pre, post, W_A * scale[group]))
        # ``Synapse.dale`` switched the flag of the user's module on: it is
        # the module that ``constrain`` will act on.
        assert conn.weight is user_weight and user_weight.dale is True
        # The module's group ids are in slot order now; read back per edge
        # they are the ids that were supplied.
        slot_of_edge = [int(conn.find_edges(a, b)) for a, b in zip(pre, post)]
        assert user_weight.group[slot_of_edge].tolist() == group.tolist()
        assert torch.equal(user_weight.scale.detach(), scale)
    # The tensors the modules were described with were never written to.
    assert group.tolist() == [0, 1, 2, 2, 1, 0]
    assert scale.tolist() == [2.0, -1.0, 0.5]


def _bind_arguments(pre, post, values):
    """What a connection passes to ``Weight.bind``: the matrix values, the map
    from input edges to edge slots, and the input matrix itself."""
    adjacency = sparse.from_edges(post, pre, values, (N_POST, N_PRE))
    _, edge_map = adjacency.coalesce(return_map=True)
    return values, edge_map, adjacency


@pytest.mark.parametrize(
    "make_weight",
    [
        lambda: EdgeWeight(),
        lambda: EdgeWeight(W_A.clone(), dale=True),
        lambda: ConstrainedWeight(group=torch.tensor([0, 1, 2, 2, 1, 0])),
        pytest.param(
            lambda: ConstantWeight(0.5),
        ),
    ],
    ids=["edge", "edge_given_dale", "constrained", "constant"],
)
def test_binding_a_weight_module_twice_is_an_error(make_weight):
    """A weight module belongs to one connection.

    Binding it a second time (which would re-align already aligned data)
    raises instead of corrupting the first connection.
    """
    weight = make_weight()
    weight.bind(*_bind_arguments(PRE_A, POST_A, W_A))
    first = weight().detach().clone()
    with pytest.raises(RuntimeError, match="already bound"):
        weight.bind(*_bind_arguments(PRE_C, POST_C, W_A))
    assert torch.equal(weight().detach(), first)


# =============================================================================
# 2. Operator connections own the tensors of their operator
# =============================================================================
# ``OperatorConnection`` (``StructuredConnection``, ``ImplicitConnection`` and
# structured projections) registers every tensor found inside its operator:
# ``nn.Parameter`` s as parameters, everything else as buffers, under the names
# ``operator_<i>_<attr>``. They follow ``.to()``, are optimised and are saved.
N_SQ = 6


def _composite(diag, value=0.5):
    """``value * ones + diag(diag)`` as a lazy sum of two closed forms."""
    return ConstantOperator((N_SQ, N_SQ), torch.tensor(value)) + DiagonalOperator(diag)


def _composite_dense(diag, value=0.5):
    return torch.full((N_SQ, N_SQ), value, dtype=diag.dtype) + torch.diag(diag)


def test_operator_tensors_are_parameters_and_buffers():
    """Parameters inside the operator are parameters of the connection (an
    optimiser reaches them); other tensors are buffers; all are in the
    ``state_dict``.

    Parts of a composite operator are found as well.
    """
    diag = nn.Parameter(torch.arange(1.0, N_SQ + 1))
    conn = StructuredConnection(_composite(diag))

    assert [n for n, _ in conn.named_parameters()] == ["operator_1_diag"]
    assert [n for n, _ in conn.named_buffers()] == ["operator_0_value"]
    assert list(conn.state_dict()) == ["operator_1_diag", "operator_0_value"]
    assert next(conn.parameters()) is diag  # the very tensor the operator applies

    # One SGD step on ``sum(y^2)``, checked against the dense expression.
    x = _x(N_SQ)
    ref = diag.detach().clone().requires_grad_()
    (x @ _composite_dense(ref).T).square().sum().backward()
    optimiser = torch.optim.SGD(conn.parameters(), lr=0.01)
    conn(x).square().sum().backward()
    torch.testing.assert_close(diag.grad, ref.grad)
    optimiser.step()
    stepped = (ref - 0.01 * ref.grad).detach()
    torch.testing.assert_close(conn(x), x @ _composite_dense(stepped).T)

    low_rank = StructuredConnection(
        LowRankOperator(
            nn.Parameter(torch.ones(N_POST, 2)), nn.Parameter(torch.ones(N_PRE, 2))
        )
    )
    assert [n for n, _ in low_rank.named_parameters()] == [
        "operator_0_U",
        "operator_1_V",
    ]


def test_operator_connection_state_dict_round_trip():
    """A saved operator connection restores into a freshly built one: the
    loaded tensors are the ones the operator applies afterwards."""
    saved = StructuredConnection(
        _composite(nn.Parameter(torch.arange(1.0, N_SQ + 1)), 0.5)
    )
    fresh = StructuredConnection(_composite(nn.Parameter(torch.zeros(N_SQ)), -2.0))
    x = _x(N_SQ)
    expected = x @ _composite_dense(torch.arange(1.0, N_SQ + 1)).T
    assert not torch.allclose(fresh(x), expected)

    buffer = io.BytesIO()
    torch.save(saved.state_dict(), buffer)
    buffer.seek(0)
    fresh.load_state_dict(torch.load(buffer))
    torch.testing.assert_close(fresh(x), expected)
    # Still trainable after loading: the gradient reaches the parameter.
    fresh(x).sum().backward()
    assert next(fresh.parameters()).grad is not None

    # A deep copy is an independent module with its own operator tensors.
    clone = copy.deepcopy(saved)
    with torch.no_grad():
        next(clone.parameters()).zero_()
    torch.testing.assert_close(saved(x), expected)
    torch.testing.assert_close(clone(x), x @ torch.full((N_SQ, N_SQ), 0.5).T)


def test_operator_connection_follows_dtype_moves():
    """``conn.double()`` converts the operator's tensors; the operator applies
    the converted ones."""
    diag = torch.arange(1.0, N_SQ + 1)
    conn = StructuredConnection(_composite(nn.Parameter(diag.clone()))).double()
    x = _x(N_SQ, dtype=torch.float64)
    out = conn(x)
    assert out.dtype == torch.float64
    assert all(t.dtype == torch.float64 for t in conn.state_dict().values())
    torch.testing.assert_close(out, x @ _composite_dense(diag.double()).T)


# name -> (rule, n_pre, n_post) of the rules that have a closed form.
_STRUCTURED = {
    "one_to_one": (OneToOne(), N_SQ, N_SQ),
    "all_to_all": (AllToAll(), N_PRE, N_POST),
    "all_to_all_no_autapses": (AllToAll(allow_autapses=False), N_SQ, N_SQ),
    "bernoulli_p1": (PairwiseBernoulli(1.0), N_PRE, N_POST),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("name", list(_STRUCTURED))
def test_structured_projection_moves_to_cuda_and_back(name):
    """``proj.to("cuda")`` moves a structured projection completely, and its
    checkpoint taken on the GPU loads into a CPU module."""
    rule, n_pre, n_post = _STRUCTURED[name]
    proj = Projection(n_pre, n_post, rule, Synapse(weight=0.5))
    assert isinstance(proj.connection, StructuredConnection)
    x = _x(n_pre)
    pre, post = rule.edges(n_pre, n_post)
    expected = x @ _dense(pre, post, torch.full((pre.shape[0],), 0.5), n_pre, n_post).T

    proj.to("cuda")
    assert all(t.device.type == "cuda" for t in proj.state_dict().values())
    out = proj(x.cuda())
    assert out.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), expected)

    cpu = Projection(n_pre, n_post, rule, Synapse(weight=0.5))
    cpu.load_state_dict(proj.state_dict())
    torch.testing.assert_close(cpu(x), expected)
    torch.testing.assert_close(proj.to("cpu")(x), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_composite_operator_connection_moves_to_cuda():
    """Every part of a composite operator moves, parameters included."""
    diag = torch.arange(1.0, N_SQ + 1)
    conn = StructuredConnection(_composite(nn.Parameter(diag.clone()))).to("cuda")
    x = _x(N_SQ)
    out = conn(x.cuda())
    torch.testing.assert_close(out.cpu(), x @ _composite_dense(diag).T)
    out.sum().backward()
    assert next(conn.parameters()).grad.device.type == "cuda"


@pytest.mark.parametrize("part", ["sparse_alone", "sum_with_sparse"])
def test_operator_connection_with_a_sparse_part_follows_dtype_moves(part):
    """The operator may be a ``Sparse`` array or a composite containing one
    (the documented argument types of ``OperatorConnection``); such a
    connection is an ordinary module that can be converted."""
    A = sparse.from_edges(POST_A, PRE_A, W_A, (N_POST, N_PRE))
    dense = _dense(PRE_A, POST_A, W_A)
    operator = A
    if part == "sum_with_sparse":
        operator = ConstantOperator((N_POST, N_PRE), torch.tensor(0.5)) + A
        dense = dense + 0.5
    conn = StructuredConnection(operator)
    torch.testing.assert_close(conn(_x()), _x() @ dense.T)  # fine before the move
    conn = conn.double()
    x = _x(dtype=torch.float64)
    torch.testing.assert_close(conn(x), x @ dense.double().T)


def test_implicit_connection_without_tensors_has_no_state():
    """A procedural operator that holds no tensors contributes no state; the
    connection still is a module with an ``explain()``."""
    operator = ImplicitOperator(
        (N_POST, N_PRE),
        matvec=lambda x: x[..., :N_POST],
        rmatvec=lambda y: nn.functional.pad(y, (0, N_PRE - N_POST)),
    )
    conn = ImplicitConnection(operator)
    assert len(conn.state_dict()) == 0 and list(conn.parameters()) == []
    x = _x()
    torch.testing.assert_close(conn(x), x @ torch.eye(N_POST, N_PRE).T)
    text = conn.explain()
    assert "ImplicitConnection" in text and "ImplicitOperator" in text
    assert f"[{N_POST}, {N_PRE}]" in text


def test_projection_explain_for_both_realisations():
    """``Projection.explain()`` describes the realised connection: a structured
    one names its operator, a sparse one its runtime plan."""
    structured = Projection(N_SQ, N_SQ, OneToOne(), Synapse(weight=0.5))
    text = structured.explain()
    assert text == structured.connection.explain()
    assert "StructuredConnection" in text and "without a stored matrix" in text

    explicit = Projection(
        N_SQ, N_SQ, OneToOne(), Synapse(weight=0.5), realization="sparse"
    )
    text = explicit.explain(_x(N_SQ))
    assert f"nnz = {N_SQ}" in text and "Selected runtime:" in text
    assert "measured density" in text  # the input section is passed through


# =============================================================================
# 3. `layout` buffer and checkpoint validation of SparseConnection
# =============================================================================
# ``layout = [n_post, n_pre, n_receptor, n_delay, *member_shape]`` is saved
# with the edges. The edge ids of a checkpoint only mean something for that
# layout, so ``load_state_dict`` refuses another one, and it refuses index
# tensors that are not valid ids. Errors are reported the PyTorch way.
_LOAD_ERROR = r"Error\(s\) in loading state_dict"
_RECEPTOR = torch.tensor([0, 1, 0, 1, 1, 0])
_DELAY = torch.tensor([2, 0, 1, 0, 2, 1])


def _routed(
    n_pre=N_PRE, n_post=N_POST, n_receptor=2, n_delay=3, pre=PRE_A, post=POST_A
):
    """A connection with per-edge receptor and delay ids."""
    synapse = Synapse(
        receptor=_RECEPTOR, delay=_DELAY, n_receptor=n_receptor, n_delay=n_delay
    )
    return SparseConnection.from_edges(pre, post, n_pre, n_post, synapse, values=W_A)


def test_layout_buffer_describes_the_connection():
    """The buffer holds population and channel counts, plus the batch shape for
    a batch of different patterns; it is persistent."""
    plain = SparseConnection.from_edges(PRE_A, POST_A, N_PRE, N_POST)
    assert plain.layout.tolist() == [N_POST, N_PRE, 1, 1]
    assert "layout" in plain.state_dict()
    assert not plain.layout.is_floating_point()

    assert _routed().layout.tolist() == [N_POST, N_PRE, 2, 3]

    # Two networks with different patterns (and different edge counts).
    ragged = SparseConnection(
        sparse.stack(
            [
                sparse.from_edges(POST_A, PRE_A, W_A, (N_POST, N_PRE)),
                sparse.from_edges(POST_B, PRE_B, W_B, (N_POST, N_PRE)),
            ]
        )
    )
    assert ragged.layout.tolist() == [N_POST, N_PRE, 1, 1, 2]


# what differs -> (kwargs of the saved connection, kwargs of the loading one).
# The two always have tensors of identical shapes, and the saved ids are in
# range for the loading module, so nothing but ``layout`` can tell them apart.
_LAYOUT_MISMATCH = {
    "n_post": ({"n_post": N_POST}, {"n_post": N_POST + 1}),
    "n_pre": ({"n_pre": N_PRE}, {"n_pre": N_PRE + 2}),
    "n_receptor": ({"n_receptor": 2}, {"n_receptor": 3}),
    "n_delay": ({"n_delay": 3}, {"n_delay": 4}),
}


@pytest.mark.parametrize("what", list(_LAYOUT_MISMATCH))
def test_load_rejects_a_checkpoint_with_another_layout(what):
    """Edge ``(post 4, receptor 1)`` is output 9 with two receptor channels
    but output 13 with three: the same ids describe another network. Such a
    checkpoint is refused and the message names both layouts."""
    saved_kwargs, target_kwargs = _LAYOUT_MISMATCH[what]
    saved, target = (
        _routed(**saved_kwargs),
        _routed(pre=PRE_C, post=POST_C, **target_kwargs),
    )
    assert saved.layout.tolist() != target.layout.tolist()
    with pytest.raises(RuntimeError, match=_LOAD_ERROR) as error:
        target.load_state_dict(saved.state_dict())
    message = str(error.value)
    assert "layout" in message
    assert f"'{what}': {saved_kwargs[what]}" in message
    assert f"'{what}': {target_kwargs[what]}" in message


def test_load_rejects_another_layout_inside_a_parent_module():
    """The check also runs when the connection is a submodule, and the message
    carries the full key."""

    def model(n_post):
        net = nn.Module()
        net.recurrent = SparseConnection.from_edges(PRE_A, POST_A, N_PRE, n_post)
        return net

    with pytest.raises(RuntimeError, match=r"recurrent\.layout"):
        model(N_POST + 1).load_state_dict(model(N_POST).state_dict())


@pytest.mark.parametrize("what", list(_LAYOUT_MISMATCH))
def test_rejected_layout_leaves_the_module_unchanged_and_usable(what):
    """A refused checkpoint must not be applied: afterwards the module is the
    network it was, and a checkpoint of the right layout still loads."""
    saved_kwargs, target_kwargs = _LAYOUT_MISMATCH[what]
    saved, target = (
        _routed(**saved_kwargs),
        _routed(pre=PRE_C, post=POST_C, **target_kwargs),
    )
    before = _state_copy(target)
    x = _x(target.in_features)
    expected = target(x).detach()

    with pytest.raises(RuntimeError, match=_LOAD_ERROR):
        target.load_state_dict(saved.state_dict())

    assert target.layout.tolist() == before["layout"].tolist()
    _assert_state_equal(target, before)
    torch.testing.assert_close(target(x), expected)
    # A checkpoint of a module with the same layout is accepted.
    twin = _routed(pre=PRE_A, post=POST_A, **target_kwargs)
    target.load_state_dict(twin.state_dict())
    torch.testing.assert_close(target(x), twin(x))


def _bad_indices(state, kind):
    """A checkpoint with one corrupted index tensor."""
    state = dict(state)
    if kind == "float_indices":
        state["indices"] = state["indices"].to(torch.float32)
    elif kind == "float_receptor":
        state["receptor"] = state["receptor"].to(torch.float64)
    elif kind == "post_too_large":
        state["indices"] = state["indices"].clone()
        state["indices"][0, 0] = N_POST
    elif kind == "pre_too_large":
        state["indices"] = state["indices"].clone()
        state["indices"][1, 2] = N_PRE
    elif kind == "negative_index":
        state["indices"] = state["indices"].clone()
        state["indices"][1, 0] = -1
    elif kind == "delay_too_large":
        state["delay"] = state["delay"].clone()
        state["delay"][3] = 3
    return state


@pytest.mark.parametrize(
    "kind, refused, message",
    [
        ("float_indices", "indices", "'indices' must hold integer ids"),
        ("float_receptor", "receptor", "'receptor' must hold integer ids"),
        ("post_too_large", "indices", "'indices' in the checkpoint addresses"),
        ("pre_too_large", "indices", "'indices' in the checkpoint addresses"),
        ("negative_index", "indices", "'indices' in the checkpoint addresses"),
        ("delay_too_large", "delay", "'delay' in the checkpoint addresses"),
    ],
)
def test_load_rejects_invalid_index_tensors_and_stays_usable(kind, refused, message):
    """Floating-point index tensors and ids outside the populations or
    channels are refused: the invalid tensor ``refused`` is not copied (and a
    floating-point one is not cast). The module still runs with valid integer
    ids and accepts a valid checkpoint afterwards.

    As usual for ``load_state_dict``, the load is not atomic: the *other*
    tensors of the refused checkpoint may have been copied, so the test
    checks the forward pass against the edge list the module reports after
    the failed load rather than against its state before it.
    """
    saved, target = _routed(), _routed(pre=PRE_C, post=POST_C)
    before = getattr(target, refused).clone()
    bad = _bad_indices(saved.state_dict(), kind)

    with pytest.raises(RuntimeError, match=_LOAD_ERROR) as error:
        target.load_state_dict(bad)
    assert message in str(error.value)

    assert torch.equal(getattr(target, refused), before)
    assert not getattr(target, refused).is_floating_point()
    assert target.layout.tolist() == [N_POST, N_PRE, 2, 3]

    # Usable: the forward pass is that of the edge list the module now
    # reports, written as a dense matrix over the expanded layout
    # (row = post * n_receptor + receptor, column = pre * n_delay + delay).
    table = target.edge_table()
    A = _dense(
        table["pre"] * 3 + table["delay"],
        table["post"] * 2 + table["receptor"],
        table["weight"].detach(),
        N_PRE * 3,
        N_POST * 2,
    )
    x = _x(target.in_features)
    torch.testing.assert_close(target(x), x @ A.T)

    # ... and the uncorrupted checkpoint loads and reproduces the saved net.
    target.load_state_dict(saved.state_dict())
    torch.testing.assert_close(target(x), saved(x))
    assert torch.equal(target.indices, saved.indices)


# =============================================================================
# 4. Projection(realization="auto") and the structured realisation
# =============================================================================
@pytest.mark.parametrize(
    "routing, structured",
    [
        ({}, True),
        # Channel counts of 1 are the same as no channels.
        ({"n_receptor": 1, "n_delay": 1}, True),
        # More channels change in/out_features: explicit edges are needed.
        ({"n_receptor": 2}, False),
        ({"n_delay": 3}, False),
    ],
    ids=["plain", "one_channel", "n_receptor", "n_delay"],
)
@pytest.mark.parametrize("name", list(_STRUCTURED))
def test_auto_realisation_is_structured_only_without_channels(
    name, routing, structured
):
    """A closed-form operator maps ``n_pre`` inputs to ``n_post`` outputs and
    stores no weights.

    It is chosen only when the synapse asks for nothing else; in every
    case the output is that of the dense matrix.
    """
    rule, n_pre, n_post = _STRUCTURED[name]
    proj = Projection(n_pre, n_post, rule, Synapse(weight=0.5, **routing))
    kind = StructuredConnection if structured else SparseConnection
    assert isinstance(proj.connection, kind)

    n_receptor, n_delay = routing.get("n_receptor", 1), routing.get("n_delay", 1)
    assert (proj.in_features, proj.out_features) == (
        n_pre * n_delay,
        n_post * n_receptor,
    )
    # All edges use channel 0 of the expanded layout.
    pre, post = rule.edges(n_pre, n_post)
    A = _dense(
        pre * n_delay,
        post * n_receptor,
        torch.full((pre.shape[0],), 0.5),
        n_pre * n_delay,
        n_post * n_receptor,
    )
    x = _x(proj.in_features)
    torch.testing.assert_close(proj(x), x @ A.T)


@pytest.mark.parametrize("name", list(_STRUCTURED))
def test_dale_on_a_fixed_scalar_weight_is_rejected(name):
    """Dale's law is a statement about stored per-edge weights whose sign could
    change.

    One fixed scalar cannot change sign, so ``dale=True`` would
    do nothing: it is refused with a message that says so, instead of quietly
    switching the projection to the sparse realisation. Per-edge weights with
    the same value do support it (and need explicit edges).
    """
    rule, n_pre, n_post = _STRUCTURED[name]
    with pytest.raises(ValueError, match="dale=True has no effect"):
        Projection(n_pre, n_post, rule, Synapse(weight=0.5, dale=True))
    with pytest.raises(ValueError, match="dale=True has no effect"):
        pre, post = rule.edges(n_pre, n_post)
        SparseConnection.from_edges(
            pre, post, n_pre, n_post, Synapse(weight=0.5, dale=True)
        )

    pre, post = rule.edges(n_pre, n_post)
    per_edge = torch.full((pre.shape[0],), 0.5)
    proj = Projection(n_pre, n_post, rule, Synapse(weight=per_edge, dale=True))
    assert isinstance(proj.connection, SparseConnection)
    assert proj.weight.dale is True
    A = _dense(pre, post, per_edge, n_pre, n_post)
    x = _x(n_pre)
    torch.testing.assert_close(proj(x), x @ A.T)


@pytest.mark.parametrize("orientation", ["post_pre", "pre_post"])
def test_from_sparse_with_scipy_matrix_gets_the_default_dtype(orientation):
    """A SciPy matrix is float64 because NumPy is; through ``FromSparse`` it
    becomes default-dtype weights, exactly like ``SparseConnection(scipy)``.

    A btorch matrix is a deliberate choice and keeps its dtype.
    """
    S = scipy.sparse.random(N_POST, N_PRE, density=0.4, format="coo", random_state=0)
    assert S.dtype == np.float64
    given = S if orientation == "post_pre" else S.T.tocoo()
    proj = Projection(N_PRE, N_POST, FromSparse(given, orientation=orientation))
    assert proj.connection.weight.value.dtype == torch.get_default_dtype()
    assert SparseConnection(S).weight.value.dtype == torch.get_default_dtype()
    reference = torch.as_tensor(S.toarray(), dtype=torch.get_default_dtype())
    torch.testing.assert_close(_matrix(proj.connection), reference)

    kept = Projection(
        N_PRE, N_POST, FromSparse(sparse.as_sparse(given), orientation=orientation)
    )
    assert kept.connection.weight.value.dtype == torch.float64


# =============================================================================
# 5. dtype rule: the default-dtype cast applies to matrix values only
# =============================================================================
# SciPy (float64 by convention) and integer matrices get the default dtype.
# A weight tensor the user supplies explicitly is a deliberate choice: float64
# stays float64 unless ``dtype=`` says otherwise.
_S = scipy.sparse.coo_array(
    (np.arange(1.0, 7.0), (POST_A.numpy(), PRE_A.numpy())), shape=(N_POST, N_PRE)
)
# Not representable in float32: a silent round trip through float32 is visible.
_W64 = 1.0 + torch.arange(1, 7, dtype=torch.float64) * 1e-9

_BUILDERS = {
    "constructor_scipy": lambda synapse, **kw: SparseConnection(_S, synapse, **kw),
    "from_adjacency_scipy": lambda synapse, **kw: SparseConnection.from_adjacency(
        _S.T.tocoo(), synapse, **kw
    ),
    "projection_from_sparse_scipy": lambda synapse, **kw: Projection(
        # ``_S`` is the operator (n_post, n_pre); the default is "pre_post".
        N_PRE,
        N_POST,
        FromSparse(_S, orientation="post_pre"),
        synapse,
        **kw,
    ).connection,
    "integer_matrix": lambda synapse, **kw: SparseConnection(
        sparse.from_edges(POST_A, PRE_A, torch.arange(1, 7), (N_POST, N_PRE)),
        synapse,
        **kw,
    ),
}
_EXPLICIT_WEIGHTS = {
    "tensor": lambda: _W64.clone(),
    "parameter": lambda: nn.Parameter(_W64.clone()),
    "numpy": lambda: _W64.numpy().copy(),
    "edge_weight_module": lambda: EdgeWeight(_W64.clone()),
    "constrained_base": lambda: ConstrainedWeight(
        group=torch.zeros(6, dtype=torch.long), base=_W64.clone()
    ),
}


@pytest.mark.parametrize("weight", list(_EXPLICIT_WEIGHTS))
@pytest.mark.parametrize("builder", list(_BUILDERS))
def test_explicit_float64_weights_are_not_downcast(builder, weight):
    """Explicit float64 weights on a SciPy or integer matrix stay float64, bit
    for bit; the matrix only supplies the pattern here."""
    conn = _BUILDERS[builder](Synapse(weight=_EXPLICIT_WEIGHTS[weight]()))
    assert conn.weight().dtype == torch.float64
    A = _dense(PRE_A, POST_A, _W64)
    assert torch.equal(_matrix(conn), A)  # exact: no float32 detour
    x = _x(dtype=torch.float64)
    out = conn(x)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, x @ A.T, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("builder", list(_BUILDERS))
def test_matrix_values_and_explicit_dtype_follow_the_documented_rule(builder):
    """Without explicit weights the matrix values are cast to the default
    dtype, and ``dtype=`` overrides everything, explicit weights included."""
    default = torch.get_default_dtype()
    assert default != torch.float64, "the test assumes the float32 default"
    assert _BUILDERS[builder](None).weight().dtype == default
    assert _BUILDERS[builder](None, dtype=torch.float64).weight().dtype == torch.float64
    forced = _BUILDERS[builder](Synapse(weight=_W64.clone()), dtype=torch.float32)
    assert forced.weight().dtype == torch.float32
    torch.testing.assert_close(_matrix(forced), _dense(PRE_A, POST_A, _W64.float()))


# =============================================================================
# 6. bias: copied, on the connection's device, in the weight dtype
# =============================================================================
@pytest.mark.parametrize(
    "make_bias",
    [
        lambda: torch.arange(1.0, N_POST + 1),
        lambda: nn.Parameter(torch.arange(1.0, N_POST + 1)),
        lambda: np.arange(1.0, N_POST + 1, dtype=np.float32),
    ],
    ids=["tensor", "parameter", "numpy"],
)
def test_bias_is_a_private_copy(make_bias):
    """The connection's bias parameter shares no memory with what the caller
    passed: training the connection does not change the caller's tensor and
    vice versa."""
    source = make_bias()
    original = torch.arange(1.0, N_POST + 1)
    conn = SparseConnection.from_edges(
        PRE_A, POST_A, N_PRE, N_POST, values=W_A, bias=source
    )
    assert isinstance(conn.bias, nn.Parameter) and conn.bias is not source
    x = _x()
    expected = x @ _dense(PRE_A, POST_A, W_A).T + original
    torch.testing.assert_close(conn(x), expected)

    with torch.no_grad():
        conn.bias.add_(10.0)  # "training" the connection
    torch.testing.assert_close(torch.as_tensor(source).detach(), original)

    with torch.no_grad():
        conn.bias.sub_(10.0)
        if isinstance(source, np.ndarray):
            source += 100.0
        else:
            source.add_(100.0)  # the caller reuses its tensor
    torch.testing.assert_close(conn.bias.detach(), original)
    torch.testing.assert_close(conn(x), expected)


@pytest.mark.parametrize(
    "bias, kwargs, expected",
    [
        # A float64 bias does not make a float32 connection compute in float64.
        (torch.ones(N_POST, dtype=torch.float64), {}, torch.float32),
        (torch.ones(N_POST, dtype=torch.long), {}, torch.float32),
        (torch.ones(N_POST), {"dtype": torch.float64}, torch.float64),
        ([1.0] * N_POST, {}, torch.float32),
    ],
    ids=["float64_bias", "integer_bias", "float64_connection", "list"],
)
def test_bias_is_cast_to_the_weight_dtype(bias, kwargs, expected):
    """Bias, weights and (for an input of that dtype) the output share one
    dtype."""
    conn = SparseConnection.from_edges(
        PRE_A, POST_A, N_PRE, N_POST, values=W_A, bias=bias, **kwargs
    )
    assert conn.bias.dtype == conn.weight().dtype == expected
    x = _x(dtype=expected)
    out = conn(x)
    assert out.dtype == expected
    torch.testing.assert_close(out, x @ _dense(PRE_A, POST_A, W_A).to(expected).T + 1.0)


def test_bias_follows_explicit_float64_weights():
    """With explicit float64 weights (which are kept, see section 5) the bias
    is float64 as well."""
    conn = SparseConnection(
        _S, Synapse(weight=_W64.clone()), bias=torch.ones(N_POST, dtype=torch.float64)
    )
    assert conn.weight().dtype == torch.float64
    assert conn.bias.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("where", ["device_argument", "matrix_on_cuda"])
def test_bias_is_created_on_the_connection_device(where):
    """A CPU bias handed to a CUDA connection lands on the GPU (and the
    caller's tensor stays where it was)."""
    bias = torch.arange(1.0, N_POST + 1)
    if where == "device_argument":
        conn = SparseConnection.from_edges(
            PRE_A, POST_A, N_PRE, N_POST, values=W_A, bias=bias, device="cuda"
        )
    else:
        conn = SparseConnection.from_edges(
            PRE_A.cuda(), POST_A.cuda(), N_PRE, N_POST, values=W_A.cuda(), bias=bias
        )
    assert conn.bias.device.type == "cuda" and bias.device.type == "cpu"
    x = _x()
    expected = x @ _dense(PRE_A, POST_A, W_A).T + bias
    torch.testing.assert_close(conn(x.cuda()).cpu(), expected)


# =============================================================================
# 7. set_edges_ validates before it mutates
# =============================================================================
def _rewire_snapshot(conn):
    return (
        _state_copy(conn),
        (conn.topology_version, conn.routing_version, conn.value_version),
    )


def _assert_not_rewired(conn, snapshot, x, expected):
    state, versions = snapshot
    _assert_state_equal(conn, state)
    assert (conn.topology_version, conn.routing_version, conn.value_version) == versions
    torch.testing.assert_close(conn(x), expected)


_SLOT = torch.tensor([1])
_ONE = torch.tensor([1])


@pytest.mark.parametrize(
    "routed, kwargs, message",
    [
        (True, {"receptor": torch.tensor([2])}, r"receptor ids must be in \[0, 2\)"),
        (True, {"receptor": torch.tensor([-1])}, r"receptor ids must be in \[0, 2\)"),
        (True, {"delay": torch.tensor([3])}, r"delay ids must be in \[0, 3\)"),
        (True, {"delay": torch.tensor([-1])}, r"delay ids must be in \[0, 3\)"),
        # A valid receptor does not excuse an invalid delay.
        (
            True,
            {"receptor": torch.tensor([1]), "delay": torch.tensor([7])},
            r"delay ids must be in \[0, 3\)",
        ),
        (False, {"receptor": torch.tensor([0])}, "no per-edge receptor"),
        (False, {"delay": torch.tensor([0])}, "no per-edge delay"),
    ],
    ids=[
        "receptor_too_large",
        "receptor_negative",
        "delay_too_large",
        "delay_negative",
        "valid_receptor_bad_delay",
        "receptor_without_receptors",
        "delay_without_delays",
    ],
)
def test_set_edges_rejects_bad_routing_without_changing_anything(
    routed, kwargs, message
):
    """Receptor / delay ids outside the channels, or for a connection that
    has no such per-edge attribute, raise a ``ValueError`` *before* any edge
    is touched: state, version counters and output are as before."""
    conn = (
        _routed()
        if routed
        else SparseConnection.from_edges(PRE_A, POST_A, N_PRE, N_POST, values=W_A)
    )
    x = _x(conn.in_features)
    expected = conn(x).detach()
    snapshot = _rewire_snapshot(conn)
    with pytest.raises(ValueError, match=message):
        conn.set_edges_(_SLOT, pre=_ONE, post=_ONE, **kwargs)
    _assert_not_rewired(conn, snapshot, x, expected)

    # The connection is still rewirable with valid arguments.
    valid = {k: torch.tensor([1]) for k in kwargs} if routed else {}
    conn.set_edges_(_SLOT, pre=torch.tensor([6]), post=torch.tensor([3]), **valid)
    assert conn.indices[:, 1].tolist() == [3, 6]  # stored as [post, pre]
    assert (int(conn.pre[1]), int(conn.post[1])) == (6, 3)


def test_set_edges_with_an_empty_slot_list_is_a_no_op():
    """Nothing to rewire (a rewiring step that found no candidate): no error,
    no version bump, no change."""
    empty = torch.empty(0, dtype=torch.long)
    for conn in (_routed(), SparseConnection.from_edges(PRE_A, POST_A, N_PRE, N_POST)):
        x = _x(conn.in_features)
        expected = conn(x).detach()
        snapshot = _rewire_snapshot(conn)
        conn.set_edges_(empty, pre=empty, post=empty)
        if conn.receptor is not None:
            conn.set_edges_(empty, pre=empty, post=empty, receptor=empty, delay=empty)
        _assert_not_rewired(conn, snapshot, x, expected)


def test_set_edges_with_valid_routing_matches_dense_reference():
    """The positive case next to the rejections: a rewired slot moves its
    weight to the new ``(post, receptor) <- (pre, delay)`` position."""
    conn = _routed()
    slots = torch.tensor([0, 4])
    post, pre = torch.tensor([2, 3]), torch.tensor([5, 0])
    receptor, delay = torch.tensor([1, 0]), torch.tensor([2, 1])
    conn.set_edges_(slots, pre=pre, post=post, receptor=receptor, delay=delay)

    table = conn.edge_table()
    assert torch.equal(table["post"][slots], post)
    assert torch.equal(table["pre"][slots], pre)
    assert torch.equal(table["receptor"][slots], receptor)
    assert torch.equal(table["delay"][slots], delay)
    A = _dense(
        table["pre"] * 3 + table["delay"],
        table["post"] * 2 + table["receptor"],
        table["weight"].detach(),
        N_PRE * 3,
        N_POST * 2,
    )
    x = _x(conn.in_features)
    torch.testing.assert_close(conn(x), x @ A.T)


@pytest.mark.parametrize("wrong", ["pre", "receptor", "delay"])
def test_set_edges_with_mismatching_lengths_changes_nothing(wrong):
    """An argument with another length than ``slots`` is an error, and like
    every other rejected call it must leave the connection untouched."""
    conn = _routed()
    x = _x(conn.in_features)
    expected = conn(x).detach()
    snapshot = _rewire_snapshot(conn)
    arguments = {
        "post": torch.tensor([3]),
        "pre": torch.tensor([6]),
        "receptor": torch.tensor([1]),
        "delay": torch.tensor([1]),
    }
    arguments[wrong] = torch.tensor([1, 1, 1])  # three values for one slot
    with pytest.raises((ValueError, RuntimeError)):
        conn.set_edges_(_SLOT, **arguments)
    _assert_not_rewired(conn, snapshot, x, expected)


# =============================================================================
# 8. Unsupported batch combinations are refused at construction
# =============================================================================
def _ragged():
    """Two networks with different patterns: 6 + 8 edges."""
    return sparse.stack(
        [
            sparse.from_edges(POST_A, PRE_A, W_A, (N_POST, N_PRE)),
            sparse.from_edges(POST_B, PRE_B, W_B, (N_POST, N_PRE)),
        ]
    )


def test_batch_of_different_patterns_alone_is_supported():
    """The supported base case the rejections below are variations of:
    network ``g`` of the batch is applied to ``x[g]``."""
    conn = SparseConnection(_ragged())
    assert conn.batch_shape == (2,)
    x = _x(batch=(2, 3))
    expected = torch.stack(
        [x[0] @ _dense(PRE_A, POST_A, W_A).T, x[1] @ _dense(PRE_B, POST_B, W_B).T]
    )
    torch.testing.assert_close(conn(x), expected)


def test_different_patterns_with_a_shared_pattern_value_batch_are_rejected():
    """A matrix that is both a batch of different patterns (the network index
    is stored per entry) and a shared-pattern batch of values (``[G, nnz]``
    values) would need two kinds of network batch at once."""
    # Entries (network, post, pre) of two networks; values for 3 value-batch
    # members on top: shape (3, 2, n_post, n_pre).
    indices = torch.stack(
        [
            torch.tensor([0, 0, 1, 1, 1]),
            torch.tensor([0, 1, 0, 1, 2]),
            torch.tensor([1, 2, 0, 3, 4]),
        ]
    )
    matrix = sparse.COO(indices, torch.ones(3, 5), (3, 2, N_POST, N_PRE), batch_dim=2)
    with pytest.raises(NotImplementedError, match="shared-pattern value batch"):
        SparseConnection(matrix)
    # The same entries without the value batch are fine.
    plain = sparse.COO(indices, torch.ones(5), (2, N_POST, N_PRE), batch_dim=1)
    assert SparseConnection(plain).batch_shape == (2,)


@pytest.mark.parametrize(
    "make_weight",
    [
        lambda n: nn.Parameter(torch.ones(3, n)),
        lambda n: torch.ones(3, n),
        lambda n: ConstrainedWeight(
            group=torch.zeros(n, dtype=torch.long), scale=torch.ones(3, 1)
        ),
    ],
    ids=["parameter", "tensor", "constrained_scales"],
)
def test_different_patterns_with_batched_weights_are_rejected(make_weight):
    """Weights ``[G, n_edge]`` (or group scales ``[G, n_group]``) define ``G``
    networks on one pattern; on top of a batch of different patterns that is
    refused when the connection is built, not at the first forward."""
    matrix = _ragged()
    with pytest.raises(NotImplementedError, match="Batched weights"):
        SparseConnection(matrix, Synapse(weight=make_weight(matrix.nnz)))


# =============================================================================
# 9. Per-edge arrays: clear errors, and NumPy arrays / lists as weights
# =============================================================================
@pytest.mark.parametrize(
    "make_synapse, message",
    [
        (lambda: Synapse(weight=torch.ones(4)), "weight has 4 entries.*6 input edges"),
        (
            lambda: Synapse(weight=nn.Parameter(torch.ones(9))),
            "weight has 9 entries.*6 input edges",
        ),
        (lambda: Synapse(weight=np.ones(4)), "weight has 4 entries.*6 input edges"),
        (
            lambda: Synapse(weight=EdgeWeight(torch.ones(4))),
            "weight has 4 entries.*6 input edges",
        ),
        (
            lambda: Synapse(delay=torch.ones(4, dtype=torch.long)),
            r"delay must be a scalar or have one entry per edge \(6\).*\(4,\)",
        ),
        (
            lambda: Synapse(receptor=torch.ones(4, dtype=torch.long)),
            r"receptor must be a scalar or have one entry per edge \(6\).*\(4,\)",
        ),
        # One entry per edge, but not one-dimensional.
        (
            lambda: Synapse(delay=torch.ones(1, 6, dtype=torch.long)),
            r"delay must be a scalar or have one entry per edge \(6\).*\(1, 6\)",
        ),
        # A single-element array is not a scalar: it would broadcast silently.
        (
            lambda: Synapse(receptor=torch.ones(1, dtype=torch.long)),
            r"receptor must be a scalar or have one entry per edge \(6\).*\(1,\)",
        ),
    ],
    ids=[
        "weight_tensor",
        "weight_parameter",
        "weight_numpy",
        "weight_module",
        "delay",
        "receptor",
        "delay_2d",
        "receptor_length_1",
    ],
)
def test_per_edge_array_of_wrong_length_names_both_lengths(make_synapse, message):
    """A per-edge array that does not have one entry per edge is a
    ``ValueError`` whose message states what was given and what is needed (it
    used to be a ``RuntimeError`` from inside an indexing operation)."""
    with pytest.raises(ValueError, match=message):
        SparseConnection.from_edges(PRE_A, POST_A, N_PRE, N_POST, make_synapse())


def test_scalar_routing_ids_still_apply_to_every_edge():
    """The length check does not affect the documented scalar form."""
    conn = SparseConnection.from_edges(
        PRE_A,
        POST_A,
        N_PRE,
        N_POST,
        Synapse(delay=2, receptor=torch.tensor(1)),
        values=W_A,
    )
    assert conn.delay.tolist() == [2] * 6 and conn.receptor.tolist() == [1] * 6
    assert (conn.n_delay, conn.n_receptor) == (3, 2)


@pytest.mark.parametrize(
    "weight, dtype",
    [
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), torch.float64),
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32), torch.float32),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], torch.float32),
        # Integer weights become default-dtype floats, like an integer matrix.
        (np.arange(1, 7), torch.float32),
    ],
    ids=["numpy_float64", "numpy_float32", "list", "numpy_int"],
)
def test_numpy_array_or_list_as_weight(weight, dtype):
    """``Synapse(weight=<array or list>)`` means fixed per-edge weights aligned
    with the supplied edges, exactly like a tensor."""
    conn = SparseConnection.from_edges(
        PRE_A, POST_A, N_PRE, N_POST, Synapse(weight=weight)
    )
    assert isinstance(conn.weight, EdgeWeight)
    assert list(conn.parameters()) == []  # fixed, as for a plain tensor
    assert conn.weight().dtype == dtype
    expected = _dense(PRE_A, POST_A, torch.arange(1.0, 7.0).to(dtype))
    torch.testing.assert_close(_matrix(conn), expected)
    from_tensor = SparseConnection.from_edges(
        PRE_A, POST_A, N_PRE, N_POST, Synapse(weight=torch.as_tensor(weight))
    )
    torch.testing.assert_close(_matrix(conn), _matrix(from_tensor))


def test_numpy_scalar_weight_is_a_constant_weight():
    """A zero-dimensional array is a single weight for every edge."""
    conn = SparseConnection.from_edges(
        PRE_A, POST_A, N_PRE, N_POST, Synapse(weight=np.float64(0.25))
    )
    assert isinstance(conn.weight, ConstantWeight)
    torch.testing.assert_close(
        _matrix(conn), _dense(PRE_A, POST_A, torch.full((6,), 0.25))
    )


# =============================================================================
# 10. explain() reports the backend that would run now
# =============================================================================
def _backend_line(conn):
    (line,) = [
        ln.strip() for ln in conn.explain().splitlines() if "    backend = " in ln
    ]
    return line.removeprefix("backend = ")


@pytest.mark.parametrize("device", DEVICES)
def test_explain_respects_use_backend(device):
    """The backend is resolved when ``explain()`` is called, so it follows
    ``runtime.use_backend`` (the old report was the choice made when the
    connection was built).

    On CUDA the default (Triton, when installed) is
    replaced by ``aten`` inside ``use_backend("aten")``.
    """
    conn = SparseConnection.from_edges(
        PRE_A, POST_A, N_PRE, N_POST, values=W_A, device=device
    )
    available = registry.available("csr_matvec", device)
    assert "aten" in available
    default = available[0]  # best first
    x = _x().to(device)
    expected = _x() @ _dense(PRE_A, POST_A, W_A).T

    assert _backend_line(conn) == default
    for name in available:
        with use_backend(name):
            assert _backend_line(conn) == name
            # ... and that is the backend the forward pass resolves.
            assert registry.name("csr_matvec", device) == name
            torch.testing.assert_close(conn(x).cpu(), expected)
    assert _backend_line(conn) == default


def test_explain_does_not_remember_the_backend_at_construction():
    """A connection built while a backend is forced reports the default one
    once the override is gone."""
    forced = registry.available("csr_matvec", "cpu")[-1]
    default = registry.available("csr_matvec", "cpu")[0]
    if forced == default:
        pytest.skip("only one csr_matvec backend is available on the CPU")
    with use_backend(forced):
        conn = SparseConnection.from_edges(PRE_A, POST_A, N_PRE, N_POST, values=W_A)
        assert _backend_line(conn) == forced
    assert _backend_line(conn) == default


# =============================================================================
# 13. Gain sweep on a model whose recurrent weights are a fixed buffer
# =============================================================================
@pytest.mark.parametrize("device", DEVICES)
def test_gain_sweep_scales_fixed_sparse_weights(device, monkeypatch):
    """``compute_gain_stability_sensitivity`` multiplies the recurrent weights
    by every gain ``g`` before it simulates.

    The stand-in model has the attribute contract of the function
    (``model.brain.synapse.linear`` is the recurrent layer, ``model.brain.
    neuron`` exists, the forward returns ``(_, {"neuron": {"spike": ...}})``).
    Its recurrent layer is a ``SparseConnection`` with fixed weights, i.e. a
    *buffer*. ``Module.to(device)`` replaces buffers by new tensors, so a
    weight tensor looked up before the move is not the one the model uses
    afterwards: the sweep then scaled a stale tensor and simulated every gain
    with the original weights. (On the CPU the move is a no-op, so only the
    CUDA case could fail before the fix.)

    The Lyapunov estimate is replaced by a recorder of what the simulation
    was run with.
    """
    weight = EdgeWeight(W_A.clone(), trainable=False)
    conn = SparseConnection.from_edges(
        PRE_A, POST_A, N_PRE, N_POST, Synapse(weight=weight)
    )
    assert list(conn.parameters()) == [] and "value" in dict(
        conn.weight.named_buffers()
    )
    original = conn.weight().detach().clone()  # slot order, on the CPU
    A = _dense(PRE_A, POST_A, W_A)

    model = nn.Module()
    model.brain = nn.Module()
    model.brain.synapse = nn.Module()
    model.brain.synapse.linear = conn
    model.brain.neuron = nn.Module()
    n_step = 20

    def forward(x):
        # "Spikes" that expose the weights in use: the recurrent input caused
        # by constant unit activity, repeated over time -> [T, 1, n_post].
        current = conn(torch.ones(N_PRE, device=x.device))
        return None, {"neuron": {"spike": current.expand(n_step, 1, N_POST)}}

    model.forward = forward

    import btorch.models.functional as functional
    import btorch.models.init as minit

    monkeypatch.setattr(functional, "reset_net", lambda *a, **k: None)
    monkeypatch.setattr(minit, "uniform_v_", lambda *a, **k: None)
    monkeypatch.setattr(
        complexity, "compute_continuous_spiking_rate", lambda s, dt: s.cpu().numpy()
    )
    seen_weights, seen_rates = [], []

    def record(mean_rate):
        # Called once per gain with the population rate [T] of the simulation.
        seen_weights.append(conn.weight().detach().cpu().clone())
        seen_rates.append(float(mean_rate[0]))
        return float(len(seen_weights))

    monkeypatch.setattr(complexity, "compute_max_lyapunov_exponent", record)

    gains = np.array([0.5, 2.0, 3.0])
    slope, _, g_out, lam = complexity.compute_gain_stability_sensitivity(
        model, [{"input": torch.zeros(1, n_step, 1)}], g_values=gains, device=device
    )

    assert conn.weight.value.device.type == device
    assert len(seen_weights) == len(gains)
    base_rate = float((A @ torch.ones(N_PRE)).mean())
    for g, w, rate in zip(gains, seen_weights, seen_rates):
        # The weights of the model, and therefore its activity, scale with g.
        torch.testing.assert_close(w, original * g)
        assert rate == pytest.approx(g * base_rate, rel=1e-5)
    # The recorder returned 1, 2, 3 for gains 0.5, 2, 3.
    assert lam.tolist() == [1.0, 2.0, 3.0] and np.array_equal(g_out, gains)
    assert slope > 0
    # After the sweep the model has its original weights back.
    torch.testing.assert_close(conn.weight().detach().cpu(), original)
    torch.testing.assert_close(_matrix(conn).cpu(), A)
