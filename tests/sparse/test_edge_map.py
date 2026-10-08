"""``EdgeMap``: moving edge-aligned data through a representation change.

Sorting entries or merging duplicates (``coalesce``, ``tocsr``, ``tocsc``)
changes which stored position an edge lives at. The map returned with
``return_map=True`` says where every *old* entry went, so weights
(``sum``) and categorical metadata such as receptor or delay ids (``take``)
can be moved with one consistent permutation instead of each guessing it.
"""

import numpy as np
import pytest
import torch

from btorch import sparse

from .helpers import SHAPE, random_edges


def _coo(variant):
    rows, cols, values, dense = random_edges(variant)
    rows, cols, values = map(torch.as_tensor, (rows, cols, values))
    return sparse.from_edges(rows, cols, values, SHAPE), rows, cols, values, dense


def _new_coordinates(out):
    """``(row, col)`` of every stored entry of a COO, CSR or CSC array."""
    if out.format == "coo":
        return out.row, out.col
    return out.row_indices(), out.col_indices()


@pytest.mark.parametrize("variant", ["plain", "unsorted", "duplicates"])
@pytest.mark.parametrize("method", ["coalesce", "tocsr", "tocsc"])
def test_map_sends_every_old_entry_to_its_new_position(method, variant):
    """``target[e]`` is the new position of old entry ``e``."""
    A, rows, cols, values, dense = _coo(variant)
    out, edge_map = getattr(A, method)(return_map=True)

    n_merged = 4 if variant == "duplicates" else 0
    assert edge_map.n_old == A.nnz and edge_map.n_new == out.nnz == A.nnz - n_merged
    assert edge_map.is_permutation == (variant != "duplicates")

    # Structure: the new entry an old edge maps to has the same coordinates.
    new_rows, new_cols = _new_coordinates(out)
    assert torch.equal(new_rows[edge_map.target], rows)
    assert torch.equal(new_cols[edge_map.target], cols)

    # Values: `sum` reproduces exactly what the conversion stored, and the
    # converted array is still the same matrix.
    torch.testing.assert_close(edge_map.sum(values), out.data)
    np.testing.assert_allclose(out.to_dense().numpy(), dense)
    # The result is the same with and without the map.
    plain = getattr(A, method)()
    assert torch.equal(_new_coordinates(plain)[1], new_cols)
    torch.testing.assert_close(plain.data, out.data)


def test_map_of_a_canonical_array_is_the_identity():
    A, *_ = _coo("plain")
    canon, first = A.coalesce(return_map=True)
    # Coalescing again has nothing left to do.
    again, second = canon.coalesce(return_map=True)
    assert again is canon
    assert torch.equal(second.target, torch.arange(canon.nnz))
    assert first.is_permutation and second.is_permutation


def test_take_moves_metadata_and_sum_moves_weights_consistently():
    """Weights and ids of an edge end up at the same new position."""
    # Edge list (src, dst) with one duplicate pair: (1, 2) appears twice.
    rows = torch.tensor([1, 0, 1, 2])
    cols = torch.tensor([2, 3, 2, 0])
    weight = torch.tensor([1.0, 2.0, 4.0, 8.0])
    receptor = torch.tensor([7, 5, 7, 9])  # equal for the duplicate pair
    A = sparse.from_edges(rows, cols, weight, (3, 4))

    csr, edge_map = A.tocsr(return_map=True)
    # Canonical row-major order: (0, 3), (1, 2), (2, 0).
    assert csr.row_indices().tolist() == [0, 1, 2]
    assert csr.col_indices().tolist() == [3, 2, 0]
    assert edge_map.target.tolist() == [1, 0, 1, 2]
    assert edge_map.sum(weight).tolist() == [2.0, 5.0, 8.0]
    assert edge_map.take(receptor).tolist() == [5, 7, 9]

    # Both helpers act along a chosen axis, e.g. per-network weights [G, E]
    # or per-edge feature rows [E, F].
    batched = torch.stack([weight, 10 * weight])
    assert edge_map.sum(batched, dim=-1).tolist() == [[2, 5, 8], [20, 50, 80]]
    features = torch.stack([receptor, receptor + 1], dim=1)
    assert edge_map.take(features, dim=0).tolist() == [[5, 6], [7, 8], [9, 10]]


def test_take_refuses_to_merge_conflicting_metadata():
    """Duplicates with different ids cannot be represented by one entry."""
    rows, cols = torch.tensor([1, 0, 1]), torch.tensor([2, 3, 2])
    A = sparse.from_edges(rows, cols, torch.ones(3), (3, 4))
    _, edge_map = A.coalesce(return_map=True)
    delay = torch.tensor([1, 5, 2])  # the two (1, 2) edges disagree
    with pytest.raises(ValueError, match="different metadata"):
        edge_map.take(delay)
    # Opting out of the check picks one of the merged values.
    picked = edge_map.take(delay, check=False)
    assert picked[0] == 5 and picked[1] in (1, 2)
    # Without merging there is never a conflict.
    B = sparse.from_edges(rows[:2], cols[:2], torch.ones(2), (3, 4))
    _, permutation = B.coalesce(return_map=True)
    assert permutation.take(delay[:2]).tolist() == [5, 1]


def test_maps_compose_across_two_representation_changes():
    """COO -> transposed COO -> CSR is described by the composed map."""
    A, rows, cols, values, dense = _coo("duplicates")
    canon, first = A.coalesce(return_map=True)
    # Transposing makes the canonical entries column-major, so converting to
    # CSR has to permute them again.
    csr_t, second = canon.T.tocsr(return_map=True)
    both = first.compose(second)
    assert both.n_old == A.nnz and both.n_new == csr_t.nnz
    torch.testing.assert_close(both.sum(values), csr_t.data)
    # csr_t is the transpose: its rows are the original columns.
    assert torch.equal(csr_t.row_indices()[both.target], cols)
    assert torch.equal(csr_t.col_indices()[both.target], rows)
    np.testing.assert_allclose(csr_t.to_dense().numpy(), dense.T)
