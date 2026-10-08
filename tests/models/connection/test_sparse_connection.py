"""``SparseConnection`` construction and semantics, against dense references:

- every accepted matrix type builds the *same* module (one conversion path);
- the orientation contract (the place where a transpose may happen);
- what a :class:`Synapse` weight description turns into and what is trained;
- Dale's law, bias, dtype / device rules, and the debugging helpers.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch import nn

from btorch import sparse
from btorch.models.connection import (
    ConstantWeight,
    ConstrainedWeight,
    EdgeWeight,
    SparseConnection,
    Synapse,
)
from btorch.models.constrain import constrain_net
from btorch.sparse import Hints
from tests.sparse.helpers import DEVICES


# The operator A, shape (n_post=4, n_pre=6): non-square and asymmetric, so a
# transposed result has the wrong shape or the wrong numbers. Row 2 is empty
# (a neuron without inputs), column 3 too (a neuron without outputs).
A_DENSE = torch.tensor(
    [
        [0.0, 1.5, 0.0, 0.0, -2.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 2.5, 0.0, 0.0, 0.25],
    ]
)
N_POST, N_PRE = A_DENSE.shape
NNZ = int((A_DENSE != 0).sum())
# Entries of A in a shuffled order, to build inputs that are not canonical.
_post, _pre = A_DENSE.nonzero(as_tuple=True)
_order = torch.tensor([4, 0, 6, 2, 5, 1, 3])
POST, PRE = _post[_order], _pre[_order]
VALUE = A_DENSE[POST, PRE]


def x_batch(dtype=torch.float32, device="cpu"):
    gen = torch.Generator().manual_seed(0)
    return torch.randn(3, N_PRE, generator=gen).to(device=device, dtype=dtype)


def dense_of(pre, post, weight, shape=(N_POST, N_PRE)):
    """Dense operator ``[n_post, n_pre]`` from an edge list (duplicates
    summed), built without any btorch code."""
    dense = torch.zeros(shape, dtype=weight.dtype)
    return dense.index_put((post, pre), weight, accumulate=True)


# ------------------------------------------------------------- construction
SOURCES = {
    "btorch_coo": lambda: sparse.from_edges(POST, PRE, VALUE, A_DENSE.shape),
    "btorch_csr": lambda: sparse.from_dense(A_DENSE).tocsr(),
    "btorch_csc": lambda: sparse.from_dense(A_DENSE).tocsc(),
    "torch_coo": lambda: A_DENSE.to_sparse_coo(),
    "torch_csr": lambda: A_DENSE.to_sparse_csr(),
    "scipy_coo_array": lambda: sp.coo_array(A_DENSE.numpy()),
    "scipy_csr_array": lambda: sp.csr_array(A_DENSE.numpy()),
    "scipy_csc_matrix": lambda: sp.csc_matrix(A_DENSE.numpy()),
    "scipy_coo_matrix": lambda: sp.coo_matrix(A_DENSE.numpy()),
}


@pytest.mark.parametrize("source", SOURCES)
def test_every_matrix_type_builds_the_same_module(source):
    """Btorch, PyTorch and SciPy input go through one converter: the canonical
    state (edge slots sorted by ``(post, pre)``, one weight per slot) and the
    forward pass are identical for all of them."""
    conn = SparseConnection(SOURCES[source]())
    reference = SparseConnection(sparse.from_dense(A_DENSE))
    assert (conn.n_post, conn.n_pre, conn.nnz) == (N_POST, N_PRE, NNZ)
    assert (conn.in_features, conn.out_features) == (N_PRE, N_POST)
    assert torch.equal(conn.indices, reference.indices)
    assert torch.equal(conn.indices, torch.stack(A_DENSE.nonzero(as_tuple=True)))
    assert torch.equal(conn.weight(), A_DENSE[A_DENSE != 0])
    assert conn.batch_shape == ()
    x = x_batch()
    torch.testing.assert_close(conn(x), x @ A_DENSE.T)
    torch.testing.assert_close(conn(x[0]), A_DENSE @ x[0])  # a single vector


def test_invalid_inputs_raise():
    conn = SparseConnection(A_DENSE.to_sparse_coo())
    with pytest.raises(ValueError, match="last dimension 6"):
        conn(torch.zeros(3, N_POST))  # a post-sized vector is not an input
    with pytest.raises(ValueError, match="two sparse dimensions"):
        SparseConnection(torch.zeros(2, 3, 4).to_sparse_coo())
    with pytest.raises(ValueError, match="orientation"):
        SparseConnection.from_adjacency(A_DENSE.to_sparse_coo(), orientation="pre_post")


# -------------------------------------------------------------- orientation
@pytest.mark.parametrize("fmt", ["btorch", "torch", "scipy"])
def test_orientation_contract(fmt):
    """The anti-double-transpose test (spec section 6).

    - ``SparseConnection(A)``: ``A`` is the operator, ``y = A @ x``.
    - ``from_adjacency(W)``: ``W[src, dst]`` is a connectome matrix,
      ``y = x @ W`` (the only place where a transpose happens).
    - ``from_adjacency(A, orientation="dst_src")`` is the constructor.
    """
    convert = {
        "btorch": sparse.from_dense,
        "torch": torch.Tensor.to_sparse_coo,
        "scipy": lambda d: sp.coo_array(d.numpy()),
    }[fmt]
    x = x_batch()
    W_DENSE = A_DENSE.T.contiguous()  # [n_pre, n_post]: rows are sources
    expected = torch.stack([A_DENSE @ row for row in x])  # explicit A @ x

    operator = SparseConnection(convert(A_DENSE))
    adjacency = SparseConnection.from_adjacency(convert(W_DENSE))
    explicit = SparseConnection.from_adjacency(convert(A_DENSE), orientation="dst_src")
    for conn in (operator, adjacency, explicit):
        assert (conn.n_pre, conn.n_post) == (N_PRE, N_POST)
        torch.testing.assert_close(conn(x), expected)
        torch.testing.assert_close(conn(x), x @ W_DENSE)
        assert torch.equal(conn.indices, operator.indices)
        assert torch.equal(conn.weight(), operator.weight())

    # Reading the matrix back names the orientation explicitly as well.
    assert torch.equal(operator.to_sparse().to_dense(), A_DENSE)
    assert torch.equal(operator.to_sparse("dst_src").to_dense(), A_DENSE)
    assert torch.equal(adjacency.to_sparse("src_dst").to_dense(), W_DENSE)
    assert operator.to_sparse("src_dst").shape == (N_PRE, N_POST)

    # Treating the operator as an adjacency matrix is a different network
    # (here even a different shape): nothing transposes silently.
    swapped = SparseConnection.from_adjacency(convert(A_DENSE))
    assert (swapped.n_pre, swapped.n_post) == (N_POST, N_PRE)
    with pytest.raises(ValueError):
        swapped(x)


# ------------------------------------------------------ duplicates and edges
def test_duplicate_entries_are_summed():
    """Duplicate coordinates mean "add", as in SciPy and PyTorch; the
    connection stores one edge slot per distinct coordinate."""
    post = torch.cat([POST, POST[:3]])
    pre = torch.cat([PRE, PRE[:3]])
    value = torch.cat([VALUE, torch.tensor([10.0, 20.0, 30.0])])
    expected = dense_of(pre, post, value)
    x = x_batch()
    for matrix in (
        sparse.from_edges(post, pre, value, A_DENSE.shape),
        torch.sparse_coo_tensor(torch.stack([post, pre]), value, A_DENSE.shape),
        sp.coo_array((value.numpy(), (post.numpy(), pre.numpy())), A_DENSE.shape),
    ):
        conn = SparseConnection(matrix)
        assert conn.nnz == NNZ
        torch.testing.assert_close(conn(x), x @ expected.T)
    # The same rule for a constant weight: every *input* edge carries the
    # constant, so a pair connected twice receives it twice.
    conn = SparseConnection.from_edges(pre, post, N_PRE, N_POST, Synapse(weight=0.5))
    expected = dense_of(pre, post, torch.full_like(value, 0.5))
    torch.testing.assert_close(conn(x), x @ expected.T)


def test_from_edges():
    """``from_edges(pre, post, n_pre, n_post)`` takes the semantic edge list;
    weights default to 1 and stay aligned with the edges as given."""
    x = x_batch()
    ones = SparseConnection.from_edges(PRE, POST, N_PRE, N_POST)
    torch.testing.assert_close(ones(x), x @ (A_DENSE != 0).float().T)
    conn = SparseConnection.from_edges(PRE, POST, N_PRE, N_POST, values=VALUE)
    torch.testing.assert_close(conn(x), x @ A_DENSE.T)
    table = conn.edge_table()  # one row per edge slot, canonical order
    rebuilt = dense_of(table["pre"], table["post"], table["weight"].detach())
    assert torch.equal(rebuilt, A_DENSE)
    empty = SparseConnection.from_edges(PRE[:0], POST[:0], N_PRE, N_POST)
    assert empty.nnz == 0 and torch.equal(empty(x), torch.zeros(3, N_POST))


# ------------------------------------------------------------- weight kinds
GROUP = torch.tensor([0, 1, 2, 0, 1, 2, 0])  # 0-based group of each input edge
SCALE = torch.tensor([2.0, -1.0, 0.5])
NEW_W = torch.arange(1.0, NNZ + 1)  # replacement weights, input-edge order


def group_matrix(transpose=False):
    """Group ids as a matrix: 1-based, because 0 means "no entry"."""
    rows, cols = (PRE, POST) if transpose else (POST, PRE)
    shape = (N_PRE, N_POST) if transpose else (N_POST, N_PRE)
    return sp.coo_array(((GROUP + 1).numpy(), (rows.numpy(), cols.numpy())), shape)


# name -> (Synapse.weight, expected per-input-edge weight, trainable
# parameters, persistent state besides the edge list).
WEIGHT_KINDS = {
    "matrix_values": (lambda: None, VALUE, ["weight.value"], ["weight.value"]),
    "constant": (lambda: 0.5, torch.full((NNZ,), 0.5), [], ["weight.value"]),
    "fixed_tensor": (lambda: NEW_W, NEW_W, [], ["weight.value"]),
    "parameter": (
        lambda: nn.Parameter(NEW_W),
        NEW_W,
        ["weight.value"],
        ["weight.value"],
    ),
    "edge_weight": (
        lambda: EdgeWeight(NEW_W, dale=True),
        NEW_W,
        ["weight.value"],
        ["weight.value", "weight.sign"],
    ),
    "constrained_ids": (
        lambda: ConstrainedWeight(GROUP, scale=SCALE),
        VALUE * SCALE[GROUP],
        ["weight.scale"],
        ["weight.scale", "weight.group", "weight.base"],
    ),
    "constrained_id_matrix": (
        lambda: ConstrainedWeight(group_matrix(), base=NEW_W, scale=SCALE),
        NEW_W * SCALE[GROUP],
        ["weight.scale"],
        ["weight.scale", "weight.group", "weight.base"],
    ),
}


@pytest.mark.parametrize("kind", WEIGHT_KINDS)
def test_synapse_weight_kinds(kind):
    """``Synapse(weight=...)`` selects the parameterisation.

    Per-edge arrays are aligned with the edges *as supplied* (here a
    shuffled COO), the connection re-aligns them with its canonical
    slots. Only the documented tensors are trainable; the edge list
    never is.
    """
    make, per_edge, trainable, state = WEIGHT_KINDS[kind]
    matrix = sparse.from_edges(POST, PRE, VALUE, A_DENSE.shape)
    conn = SparseConnection(matrix, Synapse(weight=make()))
    assert [name for name, _ in conn.named_parameters()] == trainable
    assert {"indices", *state} <= set(conn.state_dict())
    assert not any("cache" in key for key in conn.state_dict())
    x = x_batch()
    expected = dense_of(PRE, POST, per_edge)
    out = conn(x)
    torch.testing.assert_close(out, x @ expected.T)
    assert out.requires_grad == bool(trainable)
    assert not conn.indices.requires_grad and not conn.indices.is_floating_point()


def test_weight_module_types_and_group_introspection():
    assert isinstance(SparseConnection(SOURCES["torch_coo"]()).weight, EdgeWeight)
    conn = SparseConnection(SOURCES["torch_coo"](), Synapse(weight=2.0))
    assert isinstance(conn.weight, ConstantWeight)
    # An adjacency (src_dst) matrix takes its id matrix in the same layout.
    W = sp.coo_array((VALUE.numpy(), (PRE.numpy(), POST.numpy())), (N_PRE, N_POST))
    weight = ConstrainedWeight(group_matrix(transpose=True), scale=SCALE)
    conn = SparseConnection.from_adjacency(W, Synapse(weight=weight))
    x = x_batch()
    expected = dense_of(PRE, POST, VALUE * SCALE[GROUP])
    torch.testing.assert_close(conn(x), x @ expected.T)
    assert conn.weight.n_group == 3
    assert conn.weight.group_sizes().tolist() == [3, 2, 2]
    assert torch.equal(conn.edge_table()["group"], conn.weight.group)
    info = conn.weight.group_info(include_weights=True)
    assert info["num_connections"].tolist() == [3, 2, 2]
    # A group matrix that misses an edge is an error, not a silent group 0.
    partial = sp.coo_array(([1], ([0], [1])), (N_POST, N_PRE))
    with pytest.raises(ValueError, match="Constraint missing"):
        SparseConnection(
            SOURCES["torch_coo"](), Synapse(weight=ConstrainedWeight(partial))
        )


# ----------------------------------------------------------------- Dale's law
@pytest.mark.parametrize("how", ["synapse_flag", "edge_weight", "parameter"])
def test_dale_edge_weights_shrink_to_zero_but_never_flip(how):
    """Dale's law for per-edge weights: every weight keeps the sign it had at
    construction.

    An update may shrink it to zero, never past zero.
    """
    synapse = {
        "synapse_flag": lambda: Synapse(dale=True),
        "edge_weight": lambda: Synapse(weight=EdgeWeight(VALUE, dale=True)),
        "parameter": lambda: Synapse(weight=nn.Parameter(VALUE.clone()), dale=True),
    }[how]()
    conn = SparseConnection(sparse.from_edges(POST, PRE, VALUE, A_DENSE.shape), synapse)
    initial = conn.weight().detach().clone()
    sign = torch.sign(initial)
    assert torch.equal(conn.weight.sign, sign)

    # One SGD step that drags every weight by 1.0 towards (and for the small
    # ones across) zero, then the projection, as a training loop would do.
    optimizer = torch.optim.SGD(conn.parameters(), lr=1.0)
    (conn.weight() * sign).sum().backward()
    optimizer.step()
    assert (torch.sign(conn.weight()) == -sign).any()  # some did cross zero
    constrain_net(conn)  # visits the weight module (a ``HasConstraint``)

    weight = conn.weight().detach()
    crossed = initial.abs() <= 1.0
    assert torch.equal(weight[crossed], torch.zeros(int(crossed.sum())))
    torch.testing.assert_close(weight[~crossed], (initial - sign)[~crossed])
    assert ((weight * sign) >= 0).all()
    # Without the flag the same step is left alone by ``constrain``.
    free = SparseConnection(SOURCES["torch_coo"]())
    with torch.no_grad():
        free.weight.value.neg_()
    constrain_net(free)
    assert torch.equal(free.weight(), -A_DENSE[A_DENSE != 0])


def test_dale_constrained_weights_keep_the_base_sign():
    """Dale's law for grouped weights: scales stay non-negative, so every edge
    keeps the sign of its fixed base weight (zero at worst)."""
    weight = ConstrainedWeight(GROUP, dale=True)
    conn = SparseConnection(
        sparse.from_edges(POST, PRE, VALUE, A_DENSE.shape), Synapse(weight)
    )
    base_sign = torch.sign(conn.weight.base)
    with torch.no_grad():
        conn.weight.scale.copy_(torch.tensor([-0.7, 0.3, 2.0]))
    assert ((conn.weight() * base_sign) < 0).any()  # group 0 flipped ...
    constrain_net(conn)
    assert torch.equal(conn.weight.scale.detach(), torch.tensor([0.0, 0.3, 2.0]))
    assert ((conn.weight() * base_sign) >= 0).all()  # ... and is back at zero
    x = x_batch()
    expected = dense_of(PRE, POST, VALUE * torch.tensor([0.0, 0.3, 2.0])[GROUP])
    torch.testing.assert_close(conn(x), x @ expected.T)


# ------------------------------------------------------- bias, dtype, device
def test_bias_is_a_parameter_added_to_the_output():
    bias = torch.arange(float(N_POST))
    conn = SparseConnection(SOURCES["torch_coo"](), bias=bias)
    assert [n for n, _ in conn.named_parameters()] == ["bias", "weight.value"]
    x = x_batch()
    torch.testing.assert_close(conn(x), x @ A_DENSE.T + bias)
    conn(x).sum().backward()
    assert torch.equal(conn.bias.grad, torch.full((N_POST,), 3.0))


@pytest.mark.parametrize(
    "make, kwargs, expected",
    [
        # SciPy is float64 by NumPy convention, not by choice -> default dtype.
        (lambda: sp.coo_array(A_DENSE.double().numpy()), {}, torch.float32),
        (lambda: sp.csr_matrix(A_DENSE.double().numpy()), {}, torch.float32),
        (lambda: sp.coo_array(A_DENSE.numpy().astype(np.int64)), {}, torch.float32),
        # A torch / btorch dtype is a deliberate choice and is kept.
        (lambda: A_DENSE.double().to_sparse_coo(), {}, torch.float64),
        (lambda: sparse.from_dense(A_DENSE.double()), {}, torch.float64),
        (lambda: A_DENSE.long().to_sparse_coo(), {}, torch.float32),
        # An explicit dtype always wins.
        (
            lambda: sp.coo_array(A_DENSE.numpy()),
            {"dtype": torch.float64},
            torch.float64,
        ),
        (
            lambda: A_DENSE.double().to_sparse_coo(),
            {"dtype": torch.float32},
            torch.float32,
        ),
    ],
)
@pytest.mark.parametrize("builder", ["constructor", "from_adjacency"])
def test_dtype_rules(make, kwargs, expected, builder):
    build = (
        SparseConnection
        if builder == "constructor"
        else SparseConnection.from_adjacency
    )
    conn = build(make(), **kwargs)
    assert conn.weight().dtype == expected
    assert conn.indices.dtype == torch.long
    x = torch.ones(conn.in_features, dtype=expected)
    assert conn(x).dtype == expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["matrix_values", "constant", "constrained_ids"])
def test_to_device_and_dtype(kind, device):
    """``.to()`` moves canonical state and derived layouts together; the moved
    module computes the same function in the new dtype / device."""
    make, per_edge, _, _ = WEIGHT_KINDS[kind]
    matrix = sparse.from_edges(POST, PRE, VALUE, A_DENSE.shape)
    conn = SparseConnection(matrix, Synapse(weight=make()), bias=torch.ones(N_POST))
    conn = conn.to(device).to(torch.float64)
    assert {t.device.type for t in [*conn.parameters(), *conn.buffers()]} == {device}
    assert conn.weight().dtype == conn.bias.dtype == torch.float64
    assert conn.indices.dtype == torch.long  # integer state is not cast
    x = x_batch(torch.float64, device)
    expected = dense_of(PRE, POST, per_edge).double().to(device)
    out = conn(x)
    assert out.dtype == torch.float64 and out.device.type == device
    torch.testing.assert_close(out, x @ expected.T + 1.0)
    # device= at construction is the same as moving afterwards.
    direct = SparseConnection(A_DENSE.to_sparse_coo(), device=device)
    torch.testing.assert_close(direct(x.float()), x.float() @ A_DENSE.T.to(device))


# ------------------------------------------------------------ explain / hints
def test_explain_reports_algorithm_and_backend():
    """``explain()`` is the debugging window into the hidden planner."""
    conn = SparseConnection(SOURCES["scipy_coo_array"]())
    text = conn.explain()
    assert isinstance(text, str)
    assert "algorithm" in text and "backend" in text and f"nnz = {NNZ}" in text
    with_input = conn.explain(torch.zeros(3, N_PRE))
    assert "measured density = 0.00%" in with_input


@pytest.mark.parametrize("density", [0.0, 0.01, 1.0])
def test_hints_change_the_plan_not_the_result(density):
    """Hints are advisory: an expected input density switches the planned
    algorithm (visible in ``explain``), results and gradients stay equal."""
    plain = SparseConnection(SOURCES["torch_coo"]())
    hinted = SparseConnection(
        SOURCES["torch_coo"](), hints=Hints(expected_density=0.01)
    )
    assert hinted.hints.expected_density == 0.01

    def algorithm(conn):
        (line,) = [ln for ln in conn.explain().splitlines() if "algorithm" in ln]
        return line

    assert algorithm(plain) != algorithm(hinted)
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(8, 20, N_PRE, generator=gen)
    x = (x * (torch.rand(8, 20, N_PRE, generator=gen) < density)).requires_grad_()
    outs = [plain(x), hinted(x)]
    torch.testing.assert_close(outs[0], outs[1])
    torch.testing.assert_close(outs[1], x @ A_DENSE.T)
    grads = [torch.autograd.grad(o.sum(), x)[0] for o in outs]
    torch.testing.assert_close(grads[0], grads[1])
    torch.testing.assert_close(grads[1], A_DENSE.sum(0).expand_as(x))
    # Hints can be changed later; the plan follows.
    plain.set_hints(Hints(expected_density=0.01))
    assert algorithm(plain) == algorithm(hinted)
    torch.testing.assert_close(plain(x), outs[0])
