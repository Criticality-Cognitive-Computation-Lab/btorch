"""Batches of sparse matrices ("network batch") against dense ``einsum``.

Two kinds of batch exist and look the same from the outside
(``A.shape == (G, M, N)``, ``A.batch_shape == (G,)``):

- *Shared pattern*: ``G`` weight sets on one topology. The indices are stored
  once and the values are ``[G, E]``.
- *Different patterns*: ``sparse.stack([A0, A1, ...])`` of members that may
  have different numbers of entries (no padding).

The batch dimensions of ``A`` align with the *leading* dimensions of the
input, ``[G, B, N] -> [G, B, M]``, and broadcast like any torch operation. A
plain sample batch ``[B, N]`` is never silently combined with all ``G``
networks.
"""

import numpy as np
import pytest
import torch

from btorch import sparse

from .helpers import SHAPE, build, dense_from_edges, random_edges


M, N = SHAPE
G, B = 3, 4


def _randn(*shape, seed=0):
    return torch.randn(*shape, dtype=torch.float64, generator=torch.manual_seed(seed))


def _shared(fmt, variant="plain"):
    """Shared-pattern batch ``(G, M, N)`` and its dense ``[G, M, N]`` tensor.

    Built with the public constructors: one index set, values ``[G, E]``.
    """
    rows, cols, _, _ = random_edges(variant)
    values = _randn(G, len(rows), seed=3)
    dense = np.stack([dense_from_edges(rows, cols, v.numpy(), SHAPE) for v in values])
    if fmt == "coo":
        indices = torch.as_tensor(np.stack([rows, cols]))
        A = sparse.coo(indices, values, shape=(G, M, N))
    else:
        # `variant` is "plain" here, i.e. the entries are already row-major.
        crow = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=M))])
        A = sparse.csr(crow, cols, values, shape=(G, M, N))
    return A, torch.as_tensor(dense)


def _stacked():
    """Three members with 13, 9 and 0 entries, each in a different format."""
    members, dense = [], []
    for fmt, variant, seed in [("coo", "duplicates", 1), ("csr", "empty_rows", 2)]:
        rows, cols, values, d = random_edges(variant, seed=seed)
        members.append(build(fmt, rows, cols, values))
        dense.append(d)
    no_index = torch.empty(0, dtype=torch.long)
    pointer = torch.zeros(N + 1, dtype=torch.long)
    values = torch.empty(0, dtype=torch.float64)
    members.append(sparse.csc(pointer, no_index, values, SHAPE))
    dense.append(np.zeros(SHAPE))
    return members, torch.as_tensor(np.stack(dense))


def _stack_members(members, dense):
    return sparse.stack(members), dense


CASES = {
    "shared_csr": lambda: _shared("csr"),
    "shared_coo": lambda: _shared("coo"),
    "shared_coo_duplicates": lambda: _shared("coo", "duplicates"),
    "stacked": lambda: _stack_members(*_stacked()),
}
case = pytest.mark.parametrize("name", list(CASES))


@pytest.mark.parametrize("fmt", ["csr", "coo"])
def test_shared_pattern_stores_the_pattern_once(fmt):
    A, dense = _shared(fmt)
    n_edges = int((dense[0] != 0).sum())
    assert A.shape == (G, M, N) and A.format == fmt
    assert (A.batch_shape, A.sparse_shape, A.batch_dim()) == ((G,), SHAPE, 1)
    # nnz counts the entries of one member; the indices carry no batch axis.
    assert A.nnz == n_edges and A.values().shape == (G, n_edges)
    assert A.properties.shared_pattern
    index = A.col_indices() if fmt == "csr" else A.indices()
    assert index.shape[-1] == n_edges and index.numel() <= 2 * n_edges
    torch.testing.assert_close(A.to_dense(), dense)


@case
def test_network_and_sample_batch(name):
    """``[G, B, N] -> [G, B, M]``: network ``g`` only sees ``x[g]``."""
    A, dense = CASES[name]()
    x, x_left = _randn(G, B, N), _randn(G, B, M)
    y = A.matvec(x)
    assert y.shape == (G, B, M)
    torch.testing.assert_close(y, torch.einsum("gmn,gbn->gbm", dense, x))
    torch.testing.assert_close(
        A.rmatvec(x_left), torch.einsum("gmn,gbm->gbn", dense, x_left)
    )
    # Network batch only, and more than one sample dimension ([G, T, B, N]).
    torch.testing.assert_close(
        A.matvec(x[:, 0]), torch.einsum("gmn,gn->gm", dense, x[:, 0])
    )
    x = _randn(G, 2, B, N)
    torch.testing.assert_close(A.matvec(x), torch.einsum("gmn,gtbn->gtbm", dense, x))


@case
def test_size_one_batch_dim_broadcasts(name):
    """The same ``B`` samples on all networks: ``x.unsqueeze(0)``."""
    A, dense = CASES[name]()
    x = _randn(B, N)
    y = A.matvec(x.unsqueeze(0))
    assert y.shape == (G, B, M)
    torch.testing.assert_close(y, torch.einsum("gmn,bn->gbm", dense, x))
    x_left = _randn(1, B, M)
    torch.testing.assert_close(
        A.rmatvec(x_left), torch.einsum("gmn,bm->gbn", dense, x_left[0])
    )


@case
def test_missing_batch_dims_are_an_error(name):
    """No hidden Cartesian product of networks and samples."""
    A, _ = CASES[name]()
    # A bare vector has no network dimension at all.
    with pytest.raises(ValueError, match="never combined implicitly"):
        A.matvec(torch.ones(N, dtype=torch.float64))
    # [B, N] with B != G: the leading dimension is not a network dimension.
    assert B not in (1, G)
    with pytest.raises(ValueError, match="do not broadcast"):
        A.matvec(torch.ones(B, N, dtype=torch.float64))
    with pytest.raises(ValueError, match="do not broadcast"):
        A.rmatvec(torch.ones(B, M, dtype=torch.float64))


@case
def test_matmul_of_a_batch_follows_torch_matmul(name):
    """``@`` aligns the batch with the dims in front of the matrix dims."""
    A, dense = CASES[name]()
    for X in (_randn(G, N, 2), _randn(N, 2), _randn(5, G, N, 2), _randn(N)):
        expected = torch.matmul(dense, X)
        assert (A @ X).shape == expected.shape
        torch.testing.assert_close(A @ X, expected)
    for X in (_randn(G, 2, M), _randn(2, M), _randn(5, 1, 2, M), _randn(M)):
        expected = torch.matmul(X, dense)
        assert (X @ A).shape == expected.shape
        torch.testing.assert_close(X @ A, expected)


def test_stack_of_different_patterns_does_not_pad():
    """Members keep their own entry counts; the result is a batched COO."""
    members, dense = _stacked()
    counts = [m.nnz for m in members]
    assert len(set(counts)) == 3 and counts[-1] == 0  # deliberately ragged
    A = sparse.stack(members)
    assert A.format == "coo" and A.shape == (3, M, N) and A.batch_shape == (3,)
    assert A.sparse_shape == SHAPE and A.batch_dim() == 1
    assert A.nnz == sum(counts) and A.values().shape == (sum(counts),)
    assert not A.properties.shared_pattern
    torch.testing.assert_close(A.to_dense(), dense)
    # Coalescing merges duplicates within a member, never across members.
    canon = A.coalesce()
    assert canon.nnz == sum(counts) - 4
    torch.testing.assert_close(canon.to_dense(), dense)
    torch.testing.assert_close(A.T.to_dense(), dense.mT)


def test_stack_of_identical_patterns_shares_the_pattern():
    """Same indices in every member: indices once, values ``[G, E]``."""
    rows, cols, _, _ = random_edges("plain")
    values = _randn(G, len(rows), seed=4)
    for fmt in ("coo", "csr", "csc"):
        members = [build(fmt, rows, cols, v) for v in values]
        A = sparse.stack(members)
        assert type(A) is type(members[0]) and A.properties.shared_pattern
        assert A.nnz == len(rows) and A.values().shape == (G, len(rows))
        assert A.shape == (G, M, N)
        dense = torch.stack([m.to_dense() for m in members])
        x = _randn(G, B, N)
        torch.testing.assert_close(A.matvec(x), torch.einsum("gmn,gbn->gbm", dense, x))


def test_stack_rejects_mismatched_members():
    rows, cols, values, _ = random_edges()
    a = build("coo", rows, cols, values)
    with pytest.raises(ValueError, match="same shape"):
        sparse.stack([a, a.T])
    with pytest.raises(ValueError):
        sparse.stack([])
    with pytest.raises(TypeError):
        sparse.stack([a, a.to_dense()])


@case
def test_export_of_batched_arrays(name):
    """``to_torch`` gives a torch sparse tensor of the full ``(G, M, N)``."""
    A, dense = CASES[name]()
    t = A.to_torch()
    # The default layout is the stored format; torch has no shared-pattern
    # batch, so the pattern is repeated per member on export.
    assert t.layout == getattr(torch, f"sparse_{A.format}")
    assert tuple(t.shape) == (G, M, N)
    torch.testing.assert_close(t.to_dense(), dense)
    torch.testing.assert_close(A.to_torch(layout="coo").to_dense(), dense)
    # SciPy arrays are 2-D: a batch cannot be exported in one piece.
    with pytest.raises(ValueError, match="one batch member"):
        A.to_scipy()


def test_batched_torch_csr_round_trip():
    """A batched torch CSR comes back as a batched btorch array."""
    A, dense = _shared("csr")
    back = sparse.from_torch(A.to_torch())
    # Same pattern in every member: recognised and stored once again.
    assert back.format == "csr" and back.batch_shape == (G,)
    assert back.nnz == A.nnz and back.values().shape == A.values().shape
    x = _randn(G, B, N)
    torch.testing.assert_close(back.matvec(x), torch.einsum("gmn,gbn->gbm", dense, x))

    # torch's batched CSR needs equal nnz per member but allows different
    # patterns; those become a batched COO with per-member coordinates.
    dense = torch.zeros(2, 3, 4, dtype=torch.float64)
    dense[0, 0, 1], dense[0, 2, 3], dense[1, 1, 0], dense[1, 1, 2] = 1, 2, 3, 4
    back = sparse.from_torch(dense.to_sparse_csr())
    assert back.format == "coo" and back.batch_shape == (2,) and back.nnz == 4
    x = _randn(2, B, 4)
    torch.testing.assert_close(back.matvec(x), torch.einsum("gmn,gbn->gbm", dense, x))
