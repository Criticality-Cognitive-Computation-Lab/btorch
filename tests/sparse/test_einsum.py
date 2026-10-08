"""Experimental ``sparse.einsum``: one N-D sparse operand times dense tensors.

Every test compares ``einsum(subscripts, A, *operands)`` with
``torch.einsum(subscripts, dense_A, *operands)``, where ``dense_A`` is built
with a plain ``index_put_`` scatter-add that does not use ``btorch.sparse``.

A sparse array has the logical shape ``[*batch, *sparse, *dense]``, so the
term of ``A`` in the subscripts has one letter per logical axis, e.g.
``"bijkd"`` for one batch, three sparse and one dense dimension. Each letter
has its own size throughout this file: a result with permuted axes then has
the wrong shape and cannot pass by accident.
"""

import pytest
import torch

from btorch import sparse
from btorch.sparse.einsum import einsum

from .helpers import DEVICES, build, random_edges


# Size of every subscript letter used below.
SIZE = dict(a=2, b=3, c=2, d=6, e=2, i=4, j=5, k=3, l=2, m=5, n=8)  # noqa: E741


def _randn(*shape, seed=0, dtype=torch.float64):
    return torch.randn(*shape, dtype=dtype, generator=torch.manual_seed(seed))


def random_coo(sparse_shape, batch=(), dense=(), nnz=12, seed=0, dtype=torch.float64):
    """Random COO array ``[*batch, *sparse, *dense]`` and its dense tensor.

    The entries are in random order and the first three coordinates are stored
    twice, so every test also covers unsorted input and duplicates (which are
    summed). ``batch`` is a *shared-pattern* batch: one set of indices, values
    of shape ``[*batch, nnz, *dense]``.
    """
    gen = torch.manual_seed(seed)
    idx = torch.stack(
        [torch.randint(0, s, (nnz,), generator=gen) for s in sparse_shape]
    )
    idx = torch.cat([idx, idx[:, : min(3, nnz)]], dim=1)
    values = torch.randn(*batch, idx.shape[1], *dense, generator=gen, dtype=dtype)
    A = sparse.coo(idx, values, (*batch, *sparse_shape, *dense), dense_dim=len(dense))
    # Independent reference: scatter-add entry by entry with the entry axis
    # in front, then move the batch dimensions back to the front.
    nb, ns = len(batch), len(sparse_shape)
    ref = torch.zeros(*sparse_shape, *batch, *dense, dtype=dtype)
    ref.index_put_(tuple(idx), values.movedim(nb, 0), accumulate=True)
    ref = ref.movedim(tuple(range(ns, ns + nb)), tuple(range(nb)))
    return A, ref


def _check(subscripts, A, dense, operands):
    """Assert that the sparse and the dense einsum agree in shape and value."""
    expected = torch.einsum(subscripts, dense, *operands)
    result = einsum(subscripts, A, *operands)
    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected)


# (subscripts, number of batch dims of A, number of dense dims of A).
# The remaining letters of A's term are sparse dimensions.
CASES = [
    # --- the example of the design: batch + 3 sparse dims + a dense payload.
    ("bijkd,bk->bijd", 1, 1),
    # --- 1-D sparse vectors.
    ("i,i->", 0, 0),  # dot product, scalar output
    ("i,ib->b", 0, 0),
    ("i->i", 0, 0),  # no dense operand: plain densification
    # --- 2-D: matrix products, transposes and reductions.
    ("mn,bn->bm", 0, 0),  # matvec over a stack of vectors
    ("ij,jk->ki", 0, 0),  # matrix product with a permuted output
    ("ij->ji", 0, 0),
    ("ij->", 0, 0),  # sum of all entries
    ("ij->j", 0, 0),  # `i` appears only in A and is summed out
    ("ij,i,j->", 0, 0),  # bilinear form: two dense operands
    ("ij,jk,kl->il", 0, 0),  # `k` appears only in dense operands
    ("ij,j,jk->ki", 0, 0),  # `j` is shared with several dense operands
    ("ij,ab->bia", 0, 0),  # outer product with an unrelated operand
    ("ij,jk", 0, 0),  # implicit output: letters appearing once, sorted -> "ik"
    ("ji", 0, 0),  # implicit output "ij": a transpose
    # --- 3-D and 4-D sparse arrays.
    ("ijk,k->ij", 0, 0),
    ("ijk,jk->i", 0, 0),
    ("ijk,j->k", 0, 0),
    ("ijk,ka,ib->baj", 0, 0),
    ("ijkl,jl->ki", 0, 0),
    ("ijkl->lj", 0, 0),
    ("ijkl,i,j,k,l->", 0, 0),
    # --- shared-pattern value batches (the batch letter is an axis of values).
    ("bij,bj->bi", 1, 0),
    ("bij,cj->cbi", 1, 0),  # every batch member times every sample
    ("abij,j->ba", 2, 0),
    ("bij->", 1, 0),
    # --- dense payload dimensions.
    ("ijd,jd->id", 0, 1),
    ("ijde,e->dij", 0, 2),
    ("aijd,d,ab->bji", 1, 1),
    # --- diagonals: a repeated sparse letter keeps entries with equal coords.
    ("ii->i", 0, 0),  # diagonal of a matrix
    ("ii", 0, 0),  # trace (implicit scalar output)
    ("iij,j->i", 0, 0),
    ("iji,ib->jb", 0, 0),
    ("biid,b->d", 1, 1),
    ("ijj,jj->i", 0, 0),  # the dense operand is gathered on its own diagonal
    ("ijee->ji", 0, 2),  # a repeated *dense* letter: diagonal of the payload
]


@pytest.mark.parametrize("subscripts,n_batch,n_dense", CASES)
def test_einsum_matches_dense(subscripts, n_batch, n_dense):
    """Every supported expression equals ``torch.einsum`` on the dense
    array."""
    terms = subscripts.split("->")[0].split(",")
    shape = [SIZE[c] for c in terms[0]]
    n_sparse = len(shape) - n_batch - n_dense
    A, dense = random_coo(
        shape[n_batch : n_batch + n_sparse],
        batch=shape[:n_batch],
        dense=shape[n_batch + n_sparse :],
    )
    assert not A.properties.canonical  # duplicates, unsorted
    operands = [
        _randn(*[SIZE[c] for c in term], seed=s) for s, term in enumerate(terms[1:], 1)
    ]
    _check(subscripts, A, dense, operands)


@pytest.mark.parametrize(
    "subscripts,x_shape",
    [
        ("bijkd,bk->bijd", (3, 3)),  # the design example again
        ("bijkd,bk->djb", (3, 3)),  # permuted output, `i` and `k` summed out
        ("bijkd,ck->cb", (2, 3)),  # the batch pairs with every sample `c`
        ("bijkd->b", None),  # per-member sum
    ],
)
def test_different_pattern_batch(subscripts, x_shape):
    """A batch built with ``sparse.stack`` has one pattern per member.

    Here the batch index is stored as a coordinate row (like a sparse
    dimension), not as an axis of the values; the subscripts are the
    same as for a shared-pattern batch and so is the result.
    """
    members = [
        random_coo((4, 5, 3), dense=(6,), nnz=4 + 3 * g, seed=g) for g in range(3)
    ]
    A = sparse.stack([m[0] for m in members])
    dense = torch.stack([m[1] for m in members])
    assert A.shape == (3, 4, 5, 3, 6) and A.indices().shape[0] == 4
    _check(subscripts, A, dense, [] if x_shape is None else [_randn(*x_shape)])


@pytest.mark.parametrize("fmt", ["coo", "csr", "csc"])
@pytest.mark.parametrize("variant", ["plain", "duplicates", "unsorted"])
def test_matrix_formats_and_matvec_consistency(fmt, variant):
    """CSR / CSC inputs work, and ``"mn,bn->bm"`` is exactly ``matvec``."""
    rows, cols, values, dense = random_edges(variant)
    A, dense = build(fmt, rows, cols, values), torch.as_tensor(dense)
    x = _randn(3, A.shape[1])
    _check("mn,bn->bm", A, dense, [x])
    torch.testing.assert_close(einsum("mn,bn->bm", A, x), A.matvec(x))
    torch.testing.assert_close(einsum("mn,bm->bn", A, x[:, :5]), A.rmatvec(x[:, :5]))
    # A shared-pattern batch of matrices keeps its format through `stack`.
    G = sparse.stack([A, A.with_values(2 * A.values())])
    xg = _randn(2, 3, A.shape[1], seed=1)
    torch.testing.assert_close(einsum("gmn,gbn->gbm", G, xg), G.matvec(xg))


def test_empty_array():
    """An array without stored entries is all zeros."""
    A = sparse.coo(torch.zeros((3, 0), dtype=torch.long), torch.zeros(0), (4, 5, 3))
    assert A.nnz == 0
    dense = torch.zeros(4, 5, 3)
    _check("ijk,k->ji", A, dense, [torch.randn(3)])
    _check("ijk->", A, dense, [])
    _check(
        "iji->j",
        sparse.coo(A.indices(), A.values(), (4, 5, 4)),
        dense[..., :1].expand(4, 5, 4),
        [],
    )


@pytest.mark.parametrize("device", DEVICES)
def test_devices_and_dtypes(device):
    """Works on every device; the result dtype is the promoted dtype."""
    A, dense = random_coo((4, 5, 3), batch=(3,), dense=(6,), dtype=torch.float32)
    A, dense = A.to(device), dense.to(device)
    x = _randn(3, 3, dtype=torch.float32).to(device)
    y = einsum("bijkd,bk->bijd", A, x)
    assert y.dtype == torch.float32 and y.device.type == device
    torch.testing.assert_close(
        y, torch.einsum("bijkd,bk->bijd", dense, x), atol=1e-5, rtol=1e-5
    )
    # Integer / bool operands (e.g. spikes) are promoted to the float dtype;
    # ``torch.einsum`` itself would raise on mixed dtypes.
    spikes = x > 0
    counts = torch.arange(6, device=device)
    y = einsum("bijkd,bk,d->ij", A, spikes, counts)
    expected = torch.einsum("bijkd,bk,d->ij", dense, spikes.float(), counts.float())
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, expected, atol=1e-4, rtol=1e-4)
    # A float64 operand promotes a float32 array.
    assert einsum("bijkd,bk->", A, x.double()).dtype == torch.float64


@pytest.mark.parametrize(
    "subscripts,shapes",
    [
        ("bijd,bj,d->ib", [(3, 5), (6,)]),  # scatter path
        ("bijd,bj,d->", [(3, 5), (6,)]),  # no sparse letter in the output
        ("biid,bi->d", [(3, 4)]),  # diagonal filter
    ],
)
def test_gradcheck(subscripts, shapes):
    """Gradients w.r.t. the stored values and every dense operand are exact.

    The array has duplicates, so this also checks that each duplicate
    entry receives its own gradient.
    """
    sparse_shape = (4, 4) if "ii" in subscripts else (4, 5)
    A, _ = random_coo(sparse_shape, batch=(3,), dense=(6,), nnz=6)
    values = A.values().clone().requires_grad_(True)
    operands = [_randn(*s, seed=k).requires_grad_(True) for k, s in enumerate(shapes)]

    def fn(v, *ops):
        return einsum(subscripts, A.with_values(v), *ops)

    assert torch.autograd.gradcheck(fn, (values, *operands))


def test_errors():
    """Unsupported or inconsistent input raises instead of guessing."""
    A, _ = random_coo((4, 5))
    x = torch.randn(5)
    # An ellipsis is not supported: name the dimensions.
    with pytest.raises(NotImplementedError, match="ellipsis"):
        einsum("mn,...n->...m", A, x)
    # Only one sparse operand; sparse x sparse needs a different algorithm.
    with pytest.raises(NotImplementedError, match="one sparse operand"):
        einsum("ij,kj->ik", A, A)
    # The sparse array must be the first operand.
    with pytest.raises(TypeError, match="first operand"):
        einsum("j,ij->i", x, A)
    # One letter per logical axis of A (batch, sparse and dense).
    with pytest.raises(ValueError, match="2 dimensions"):
        einsum("ijk,k->ij", A, x)
    with pytest.raises(ValueError, match="operand 1"):
        einsum("ij,jk->ik", A, x)
    # The number of terms must match the number of operands.
    with pytest.raises(ValueError, match="2 operand"):
        einsum("ij->i", A, x)
    # A letter must have one size everywhere; size-1 axes do not broadcast.
    with pytest.raises(ValueError, match="size 4"):
        einsum("ij,j->i", A, torch.randn(4))
    with pytest.raises(ValueError, match="do not broadcast"):
        einsum("ij,j->i", A, torch.randn(1))
    with pytest.raises(ValueError, match="size 5"):
        einsum("ii->i", A)  # a diagonal needs equal sizes
    # Output letters must be unique and come from the inputs.
    with pytest.raises(ValueError, match="more than once"):
        einsum("ij->ii", A)
    with pytest.raises(ValueError, match="do not appear"):
        einsum("ij->k", A)
    with pytest.raises(ValueError, match="ASCII letters"):
        einsum("i1->i", A)
    # A letter repeated across a sparse and a dense axis of A is not handled.
    B, _ = random_coo((4, 5), dense=(4,))
    with pytest.raises(NotImplementedError, match="diagonal"):
        einsum("iji->j", B)
