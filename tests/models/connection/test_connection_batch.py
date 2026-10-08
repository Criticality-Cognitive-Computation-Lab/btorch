"""Batched connections: sample batch versus network batch (spec sections
21-23, 45, 46).

Two different things are called "batch":

- the *sample* batch: leading dimensions of the input (trials, time), all
  propagated through the same network: ``[B, N] -> [B, M]``;
- the *network* batch: ``G`` networks living in one module. The input then
  carries the network dimension first: ``[G, B, N] -> [G, B, M]``.

Networks and samples are never combined implicitly. To run the same samples
through every network, write the broadcast explicitly: ``x.unsqueeze(0)``.

A network batch can be defined in three ways, all tested here against dense
``einsum`` references, eager and under ``torch.compile(fullgraph=True)``:

1. shared pattern, values ``[G, E]`` (indices stored once);
2. an unbatched matrix with ``ConstrainedWeight(scale=[G, n_group])``;
3. different patterns with different edge counts (``btorch.sparse.stack``).
"""

import pytest
import torch

from btorch import sparse
from btorch.models.connection import ConstrainedWeight, SparseConnection, Synapse


M, N, G, B, T = 4, 6, 3, 5, 2  # n_post, n_pre, networks, samples, time steps
DTYPE = torch.float64
_gen = torch.Generator().manual_seed(0)

# Shared pattern: 8 edges, unsorted; post-neuron 2 receives nothing.
POST = torch.tensor([3, 0, 1, 0, 3, 1, 0, 3])
PRE = torch.tensor([5, 1, 0, 4, 0, 2, 2, 3])
E = POST.shape[0]
VALUES = torch.randn(G, E, generator=_gen, dtype=DTYPE)  # one row per network
GROUP = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
SCALES = torch.randn(G, 3, generator=_gen, dtype=DTYPE)  # [G, n_group]

# Different patterns: 3, 7 and 0 edges (an empty network is a valid member).
RAGGED_POST = [torch.tensor([0, 2, 3]), torch.tensor([1, 0, 3, 3, 2, 0, 1]), POST[:0]]
RAGGED_PRE = [torch.tensor([5, 0, 1]), torch.tensor([0, 3, 2, 5, 4, 4, 1]), PRE[:0]]
RAGGED_VALUES = [torch.randn(len(p), generator=_gen, dtype=DTYPE) for p in RAGGED_POST]


def dense_stack(posts, pres, values):
    """Dense ``[G, M, N]`` from one edge list per network (plain indexing)."""
    dense = torch.zeros(len(values), M, N, dtype=DTYPE)
    for g, (post, pre, value) in enumerate(zip(posts, pres, values)):
        dense[g, post, pre] = value
    return dense


DENSE = {
    "shared": dense_stack([POST] * G, [PRE] * G, VALUES),
    "constrained": dense_stack([POST] * G, [PRE] * G, VALUES[0] * SCALES[:, GROUP]),
    "ragged": dense_stack(RAGGED_POST, RAGGED_PRE, RAGGED_VALUES),
}


def build(kind):
    if kind == "shared":
        return SparseConnection.from_edges(PRE, POST, N, M, values=VALUES)
    if kind == "constrained":
        # The matrix itself is unbatched; only the group scales are batched.
        weight = ConstrainedWeight(GROUP, scale=SCALES)
        return SparseConnection.from_edges(
            PRE, POST, N, M, Synapse(weight), values=VALUES[0]
        )
    members = [
        sparse.from_edges(post, pre, value, (M, N))
        for post, pre, value in zip(RAGGED_POST, RAGGED_PRE, RAGGED_VALUES)
    ]
    return SparseConnection(sparse.stack(members))


def reference(dense, x):
    """``y[g] = A[g] @ x[g]``; ``x`` may broadcast over the networks."""
    return torch.einsum("gmn,g...n->g...m", dense, x.expand(G, *x.shape[1:]))


def randn(*shape):
    return torch.randn(*shape, generator=_gen, dtype=DTYPE)


@pytest.fixture(params=["eager", "compiled"])
def run(request):
    """Call a connection eagerly or through ``torch.compile(fullgraph=True)``
    (one compiled module per connection, reused across calls)."""
    torch._dynamo.reset()
    compiled = {}

    def call(conn, x):
        if request.param == "eager":
            return conn(x)
        if id(conn) not in compiled:
            compiled[id(conn)] = torch.compile(conn, fullgraph=True)
        return compiled[id(conn)](x)

    yield call
    torch._dynamo.reset()


# ------------------------------------------------------------- sample batch
@pytest.mark.parametrize("shape", [(N,), (B, N), (T, B, N), (0, N)], ids=str)
def test_sample_batch_only(shape, run):
    """One network: every leading dimension is a sample dimension."""
    conn = SparseConnection.from_edges(PRE, POST, N, M, values=VALUES[0])
    assert conn.batch_shape == ()
    x = randn(*shape)
    out = run(conn, x)
    assert out.shape == (*shape[:-1], M)
    torch.testing.assert_close(out, x @ DENSE["shared"][0].T)


# ------------------------------------------------------------ network batch
@pytest.mark.parametrize("kind", ["shared", "constrained", "ragged"])
def test_network_batch_matches_dense_einsum(kind, run):
    """``[G, ..., N] -> [G, ..., M]`` with network ``g`` applied to ``x[g]``
    for all three definitions of a network batch."""
    conn = build(kind)
    assert conn.batch_shape == (G,)
    assert (conn.n_pre, conn.n_post) == (N, M)
    assert (conn.in_features, conn.out_features) == (N, M)
    for shape in [(G, B, N), (G, T, B, N), (G, N), (G, 1, N)]:
        x = randn(*shape)
        out = run(conn, x)
        assert out.shape == (*shape[:-1], M)
        torch.testing.assert_close(out, reference(DENSE[kind], x))


@pytest.mark.parametrize("kind", ["shared", "constrained", "ragged"])
def test_same_samples_through_all_networks_by_explicit_broadcast(kind, run):
    """``x.unsqueeze(0)`` shares ``B`` samples across the ``G`` networks: a
    network dimension of size 1 broadcasts, like any PyTorch dimension."""
    conn = build(kind)
    x = randn(B, N)
    out = run(conn, x.unsqueeze(0))
    assert out.shape == (G, B, M)
    expected = torch.stack([x @ DENSE[kind][g].T for g in range(G)])
    torch.testing.assert_close(out, expected)
    # Time-major data: [T, B, N] -> [1, T, B, N] -> [G, T, B, M].
    xt = randn(T, B, N)
    torch.testing.assert_close(run(conn, xt[None]), reference(DENSE[kind], xt[None]))


@pytest.mark.parametrize("kind", ["shared", "constrained", "ragged"])
def test_no_implicit_cartesian_product(kind):
    """An input without the network dimension is rejected.

    It is *not* silently expanded to "every sample through every
    network".
    """
    conn = build(kind)
    with pytest.raises(ValueError, match="batch of networks"):
        conn(randn(N))
    # [B, N] with B != G cannot be read as [G, N] either. (The error for this
    # case is PyTorch's broadcasting error, not a btorch message.)
    with pytest.raises((ValueError, RuntimeError)):
        conn(randn(B, N))
    # The compiled module refuses as well (Dynamo wraps the error).
    torch._dynamo.reset()
    compiled = torch.compile(conn, fullgraph=True)
    for shape in [(N,), (B, N)]:
        with pytest.raises(RuntimeError):  # Dynamo's ``Unsupported``
            compiled(randn(*shape))
    # The explicit spelling works.
    assert compiled(randn(B, N).unsqueeze(0)).shape == (G, B, M)
    torch._dynamo.reset()


def test_shared_pattern_constructors_are_equivalent():
    """A shared-pattern batch can come from an edge list with ``[G, E]``
    values, from ``sparse.csr(..., values[G, E], shape=(G, M, N))`` or from
    ``sparse.stack`` of same-pattern members: the indices are stored once."""
    # CSR of the pattern by hand: sort the edges by (post, pre).
    order = torch.argsort(POST * N + PRE)
    crow = torch.zeros(M + 1, dtype=torch.long)
    crow[1:] = torch.cumsum(torch.bincount(POST, minlength=M), 0)
    csr = sparse.csr(crow, PRE[order], VALUES[:, order], (G, M, N))
    stacked = sparse.stack(
        [sparse.from_edges(POST, PRE, VALUES[g], (M, N)) for g in range(G)]
    )
    x = randn(G, B, N)
    expected = reference(DENSE["shared"], x)
    for matrix in (csr, stacked):
        assert matrix.batch_shape == (G,)
        conn = SparseConnection(matrix)
        assert conn.batch_shape == (G,)
        assert conn.indices.shape == (2, E)  # one pattern, not G copies
        assert conn.weight().shape == (G, E)
        torch.testing.assert_close(conn(x), expected)
    # An adjacency (src_dst) batch is transposed per network.
    adjacency = sparse.stack(
        [sparse.from_edges(PRE, POST, VALUES[g], (N, M)) for g in range(G)]
    )
    conn = SparseConnection.from_adjacency(adjacency)
    torch.testing.assert_close(conn(x), expected)


def test_different_patterns_keep_their_own_edge_counts():
    """A batch of different patterns stores ``sum(nnz)`` edges: no padding to
    the largest member, and an empty member contributes nothing."""
    conn = build("ragged")
    assert conn.nnz == sum(len(p) for p in RAGGED_POST) == 10
    assert conn.weight().shape == (10,)
    out = conn(randn(G, B, N))
    assert torch.equal(out[2], torch.zeros(B, M, dtype=DTYPE))  # empty network
    assert "network batch = [3]" in conn.explain()


def test_different_patterns_introspection_uses_network_coordinates():
    """``to_sparse()`` / ``edge_table()`` describe the model, not the way a
    ragged batch happens to be executed."""
    conn = build("ragged")
    table = conn.edge_table()
    assert int(table["post"].max()) < M and int(table["pre"].max()) < N
    assert conn.to_sparse().shape == (G, M, N)
    torch.testing.assert_close(conn.to_sparse().to_dense(), DENSE["ragged"])


def test_batched_weights_with_per_edge_group_ids():
    """The effective weights of the constrained batch are ``base[e] * scale[g,
    group[e]]``, one row per network."""
    conn = build("constrained")
    assert conn.weight().shape == (G, E)
    assert [n for n, _ in conn.named_parameters()] == ["weight.scale"]
    assert conn.weight.scale.shape == (G, 3)
    dense = conn.to_sparse().to_dense()
    torch.testing.assert_close(dense, DENSE["constrained"])


# ---------------------------------------------------------------- gradients
@pytest.mark.parametrize("broadcast", [False, True], ids=["x[G,B,N]", "x[1,B,N]"])
@pytest.mark.parametrize("kind", ["shared", "constrained", "ragged"])
def test_batched_gradients(kind, broadcast, run):
    """Gradients of the batched cases against the dense twin.

    - ``d/dx``: ``upstream[g] @ A[g]``; for a broadcast input summed over the
      networks that shared it.
    - ``d/dparam``: finite differences (``gradcheck``), plus the dense
      formula where the parameter layout is part of the public contract.
    """
    conn = build(kind)
    (param,) = conn.parameters()
    x = randn(1 if broadcast else G, B, N).requires_grad_()
    upstream = randn(G, B, M)
    out = run(conn, x)
    g_param, g_x = torch.autograd.grad((out * upstream).sum(), (param, x))

    dense = DENSE[kind].clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    (reference(dense, x_ref) * upstream).sum().backward()
    torch.testing.assert_close(g_x, x_ref.grad)
    assert g_x.shape == x.shape

    if kind == "shared":
        # values[g, k] is the weight of edge post[k] <- pre[k] in network g.
        post, pre = conn.indices
        torch.testing.assert_close(g_param, dense.grad[:, post, pre])
    elif kind == "constrained":
        # d/dscale[g, j] = sum_{e in group j} base[e] * dL/dA[g, post e, pre e]
        per_edge = VALUES[0] * dense.grad[:, POST, PRE]
        expected = torch.zeros(G, 3, dtype=DTYPE).index_add(1, GROUP, per_edge)
        torch.testing.assert_close(g_param, expected)
    else:
        # Packed storage: empty-network edges do not exist, every stored
        # edge gets a gradient; its value is checked numerically below.
        assert g_param.shape == (10,)


@pytest.mark.parametrize("kind", ["shared", "constrained", "ragged"])
def test_batched_gradcheck(kind):
    """Finite-difference check of the trainable tensor and the input."""
    conn = build(kind)
    ((name, param),) = conn.named_parameters()
    x = randn(G, 2, N).requires_grad_()
    assert torch.autograd.gradcheck(conn, (x,))
    assert torch.autograd.gradcheck(
        lambda p: torch.func.functional_call(conn, {name: p}, (x.detach(),)), (param,)
    )
    # Broadcast input: one sample set shared by all networks.
    shared = randn(1, 2, N).requires_grad_()
    assert torch.autograd.gradcheck(conn, (shared,))


def test_training_networks_independently():
    """An SGD step on the loss of network 0 only moves network 0."""
    conn = build("shared")
    x = randn(G, B, N)
    before = conn(x).detach()
    optimizer = torch.optim.SGD(conn.parameters(), lr=0.1)
    conn(x)[0].square().sum().backward()
    optimizer.step()
    after = conn(x).detach()
    assert not torch.allclose(after[0], before[0])
    torch.testing.assert_close(after[1:], before[1:])
