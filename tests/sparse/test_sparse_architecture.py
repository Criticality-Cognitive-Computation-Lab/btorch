"""Tests for the btorch.sparse refactor.

Each test demonstrates one aspect of the new API and can serve as a
usage example for downstream code.
"""

import pytest
import scipy.sparse
import torch
import torch.utils._pytree as pytree

from btorch.models.linear import DenseConn, DenseLinear, SparseConn, SparseLinear
from btorch.sparse import (
    COO,
    CSC,
    CSR,
    ELL,
    BackendUnavailableError,
    BinaryEvents,
    Bounded,
    GroupedMagnitude,
    NonNegative,
    SparseLinear as SparseSparseLinear,
    SparseParam,
    SparseProperties,
    SparseTensor,
    SpikeListEvents,
    Symmetric,
    constrain,
)


# ── Construction and basic accessors ─────────────────────────────────────────


def test_csr_from_edges_basic():
    # from_edges canonicalizes (sorts, deduplicates) edges and builds CSR
    W = CSR.from_edges(
        row=torch.tensor([1, 0, 0]),
        col=torch.tensor([0, 1, 0]),
        data=torch.tensor([3.0, 2.0, 1.0]),
        shape=(2, 2),
    )
    # After dedup: (0→0)=1.0, (0→1)=2.0, (1→0)=3.0
    assert W.nnz() == 3
    assert W.shape == (2, 2)
    dense = W.to_dense()
    torch.testing.assert_close(
        dense,
        torch.tensor([[1.0, 2.0], [3.0, 0.0]]),
    )


def test_csr_from_edges_deduplicates_by_summing():
    # Duplicate edges accumulate their values
    W = CSR.from_edges(
        row=torch.tensor([0, 0]),
        col=torch.tensor([1, 1]),
        data=torch.tensor([1.5, 0.5]),
        shape=(1, 3),
    )
    assert W.nnz() == 1
    torch.testing.assert_close(W.data, torch.tensor([2.0]))


def test_coo_construction_and_to_dense():
    W = COO.from_edges(
        row=torch.tensor([0, 1]),
        col=torch.tensor([1, 0]),
        data=torch.tensor([5.0, 7.0]),
        shape=(2, 2),
    )
    dense = W.to_dense()
    torch.testing.assert_close(
        dense,
        torch.tensor([[0.0, 5.0], [7.0, 0.0]]),
    )


def test_csc_from_edges():
    # CSC stores by column order; to_dense matches CSR reference
    src = torch.tensor([0, 1, 0])
    dst = torch.tensor([0, 0, 1])
    vals = torch.tensor([1.0, 2.0, 3.0])
    shape = (2, 2)
    W_csr = CSR.from_edges(src, dst, vals, shape=shape)
    W_csc = CSC.from_edges(src, dst, vals, shape=shape)
    torch.testing.assert_close(W_csr.to_dense(), W_csc.to_dense())


def test_ell_from_indices():
    # ELL stores (n_pre, width) arrays; to_dense matches
    indices = torch.tensor([[1, 2], [0, 2]])
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    shape = (2, 3)
    W = ELL.from_indices(indices, values, shape=shape)
    dense = W.to_dense()
    expected = torch.zeros(2, 3)
    expected[0, 1] = 1.0
    expected[0, 2] = 2.0
    expected[1, 0] = 3.0
    expected[1, 2] = 4.0
    torch.testing.assert_close(dense, expected)
    assert W.properties.max_fan_out == W.width


def test_ell_rejects_mismatched_fixed_fanout():
    with pytest.raises(ValueError, match="physical width"):
        ELL.from_indices(
            torch.tensor([[0, 1], [1, 2]]),
            torch.ones(2, 2),
            shape=(2, 3),
            properties=SparseProperties(max_fan_out=1),
        )


def test_ell_marks_fixed_fanout_when_other_properties_are_provided():
    W = ELL.from_indices(
        torch.tensor([[0, 1], [1, 2]]),
        torch.ones(2, 2),
        shape=(2, 3),
        properties=SparseProperties(sorted_indices=True),
    )
    assert W.properties.max_fan_out == 2


def test_ell_random_reproducible_no_self_loop():
    # ELL.random generates identical matrices for the same seed
    W1 = ELL.random(shape=(6, 6), fan_out=2, seed=42, allow_self=False)
    W2 = ELL.random(shape=(6, 6), fan_out=2, seed=42, allow_self=False)
    torch.testing.assert_close(W1.indices, W2.indices)
    torch.testing.assert_close(W1.data, W2.data)
    # No row i connects to column i
    row_idx = torch.arange(6).unsqueeze(1)
    assert not torch.any(W1.indices == row_idx)


def test_csr_from_scipy():
    sp = scipy.sparse.coo_array(
        ([1.0, 2.0, 3.0], ([0, 0, 1], [0, 1, 0])),
        shape=(2, 2),
    )
    W = CSR.from_scipy(sp)
    torch.testing.assert_close(W.to_dense(), torch.tensor([[1.0, 2.0], [3.0, 0.0]]))


# ── Format conversions ────────────────────────────────────────────────────────


def test_csr_to_coo_to_csr_roundtrip():
    W = CSR.from_edges(
        row=torch.tensor([0, 0, 1]),
        col=torch.tensor([0, 2, 1]),
        data=torch.tensor([1.0, 2.0, 3.0]),
        shape=(2, 3),
    )
    W2 = W.to_coo().to_csr()
    torch.testing.assert_close(W.to_dense(), W2.to_dense())


def test_csr_to_csc_to_csr_roundtrip():
    W = CSR.from_edges(
        row=torch.tensor([0, 1, 1]),
        col=torch.tensor([1, 0, 2]),
        data=torch.tensor([1.0, 2.0, 3.0]),
        shape=(2, 3),
    )
    W2 = W.to_csc().to_csr()
    torch.testing.assert_close(W.to_dense(), W2.to_dense())


def test_fixed_degree_properties_survive_format_conversion():
    W = ELL.random(shape=(4, 6), fan_out=2, seed=3)
    assert W.to_coo().properties.max_fan_out == 2
    assert W.to_csr().properties.max_fan_out == 2
    assert W.to_csc().properties.max_fan_out == 2


def test_conversion_preserves_edge_attribute_alignment():
    W = CSR.from_edges(
        row=torch.tensor([0, 0, 1]),
        col=torch.tensor([1, 0, 0]),
        data=torch.tensor([1.0, 2.0, 3.0]),
        shape=(2, 2),
        labels={"edge_id": torch.tensor([10, 20, 30])},
    )
    converted = W.to_csc()
    edges = converted.logical_edges()
    observed = {
        (int(source), int(destination)): int(edge_id)
        for source, destination, edge_id in zip(
            edges.row,
            edges.col,
            converted.attributes.labels["edge_id"],
            strict=True,
        )
    }
    assert observed == {(0, 1): 10, (0, 0): 20, (1, 0): 30}


def test_csr_to_ell_uniform_fanout():
    # to_ell works when all rows have the same number of nonzeros
    W = CSR.from_edges(
        row=torch.tensor([0, 0, 1, 1]),
        col=torch.tensor([0, 1, 0, 2]),
        data=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        shape=(2, 3),
    )
    W_ell = W.to_ell()
    assert isinstance(W_ell, ELL)
    assert W_ell.width == 2
    torch.testing.assert_close(W.to_dense(), W_ell.to_dense())


# ── mm() correctness with gradients ──────────────────────────────────────────


@pytest.mark.parametrize("Format", [COO, CSR, CSC])
def test_mm_matches_dense_matmul(Format):
    # x @ W should equal x @ W.to_dense()
    src = torch.tensor([0, 0, 1])
    dst = torch.tensor([0, 1, 1])
    vals = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    shape = (2, 2)
    W = Format.from_edges(src, dst, vals.detach().clone(), shape=shape)
    # Rebuild with grad-tracked values via _new_unsafe
    x = torch.randn(5, 2, requires_grad=True)
    out = W.mm(x)
    expected = x @ W.to_dense()
    torch.testing.assert_close(out, expected)


def test_ell_mm_matches_dense():
    W = ELL.random(shape=(4, 6), fan_out=3, seed=7)
    x = torch.randn(10, 4)
    torch.testing.assert_close(W.mm(x), x @ W.to_dense())


def test_mm_gradient_flows_through_values():
    # Gradients must reach the values tensor
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 0])
    vals = torch.tensor([2.0, 3.0], requires_grad=True)
    W = CSR.from_edges(src, dst, vals.detach().clone(), shape=(2, 2))
    # Rebuild inner with the grad tensor through a SparseParam
    param = SparseParam(W)
    x = torch.ones(1, 2)
    out = param.mm(x)
    out.sum().backward()
    # Values parameter has a gradient
    for p in param.parameters():
        assert p.grad is not None


# ── SparseTensor auto-selection ───────────────────────────────────────────────


def test_sparsetensor_defaults_to_csr():
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 0])
    W = SparseTensor.from_edges(src, dst, torch.ones(2), shape=(2, 2))
    assert W.selected_format is CSR
    torch.testing.assert_close(W.mm(torch.eye(2)), W.to_dense())


def test_sparsetensor_selects_ell_for_fixed_fanout():
    props = SparseProperties(max_fan_out=2)
    W = SparseTensor.from_edges(
        row=torch.tensor([0, 0, 1, 1]),
        col=torch.tensor([0, 1, 0, 2]),
        data=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        shape=(2, 3),
        properties=props,
    )
    assert W.selected_format is ELL
    assert W.properties.max_fan_out == 2


def test_sparsetensor_rejects_invalid_fixed_fanout():
    with pytest.raises(ValueError, match="max_fan_out"):
        SparseTensor.from_edges(
            row=torch.tensor([0, 1, 1, 1]),
            col=torch.tensor([0, 0, 1, 2]),
            data=torch.ones(4),
            shape=(2, 3),
            properties=SparseProperties(max_fan_out=2),
        )


def test_sparsetensor_selects_csc_for_valid_fixed_fanin():
    W = SparseTensor.from_edges(
        row=torch.tensor([0, 1, 0, 1]),
        col=torch.tensor([0, 0, 1, 1]),
        data=torch.ones(4),
        shape=(2, 2),
        properties=SparseProperties(max_fan_in=2),
    )
    assert W.selected_format is CSC
    assert W.properties.max_fan_in == 2


def test_direct_csr_rejects_invalid_fixed_fanout():
    with pytest.raises(ValueError, match="max_fan_out"):
        CSR(
            indptr=torch.tensor([0, 1, 3]),
            indices=torch.tensor([0, 0, 1]),
            data=torch.ones(3),
            shape=(2, 2),
            properties=SparseProperties(max_fan_out=1),
        )


def test_direct_coo_rejects_invalid_fixed_fanout():
    with pytest.raises(ValueError, match="max_fan_out"):
        COO(
            row=torch.tensor([0, 1, 1]),
            col=torch.tensor([0, 0, 1]),
            data=torch.ones(3),
            shape=(2, 2),
            properties=SparseProperties(max_fan_out=1),
        )


def test_direct_csc_rejects_invalid_fixed_fanin():
    with pytest.raises(ValueError, match="max_fan_in"):
        CSC(
            indptr=torch.tensor([0, 1, 3]),
            indices=torch.tensor([0, 0, 1]),
            data=torch.ones(3),
            shape=(2, 2),
            properties=SparseProperties(max_fan_in=1),
        )


def test_direct_format_rejects_false_unique_edges_property():
    with pytest.raises(ValueError, match="unique_edges"):
        COO(
            row=torch.tensor([0, 0]),
            col=torch.tensor([1, 1]),
            data=torch.ones(2),
            shape=(1, 2),
        )


def test_direct_format_rejects_false_sorted_indices_property():
    with pytest.raises(ValueError, match="sorted_indices"):
        CSR(
            indptr=torch.tensor([0, 2]),
            indices=torch.tensor([1, 0]),
            data=torch.ones(2),
            shape=(1, 2),
            properties=SparseProperties(sorted_indices=True),
        )


# ── SparseParam – nn.Module bridge ───────────────────────────────────────────


def test_sparse_param_registers_parameter_and_buffers():
    W = CSR.from_edges(
        row=torch.tensor([0, 1]),
        col=torch.tensor([1, 0]),
        data=torch.tensor([1.0, 2.0]),
        shape=(2, 2),
    )
    sp = SparseParam(W)
    param_names = [n for n, _ in sp.named_parameters()]
    buf_names = [n for n, _ in sp.named_buffers()]
    assert "data" in param_names
    assert "indptr" in buf_names
    assert "indices" in buf_names


def test_sparse_param_state_dict_contains_values():
    W = CSR.from_edges(
        row=torch.tensor([0, 1]),
        col=torch.tensor([1, 0]),
        data=torch.tensor([2.0, 3.0]),
        shape=(2, 2),
    )
    sp = SparseParam(W)
    sd = sp.state_dict()
    # Only persistent parameters appear in state_dict (buffers registered
    # persistent=False are excluded).
    assert "data" in sd


def test_sparse_linear_forward():
    W = CSR.from_edges(
        row=torch.tensor([0, 1]),
        col=torch.tensor([1, 0]),
        data=torch.tensor([1.0, 2.0]),
        shape=(2, 2),
    )
    layer = SparseSparseLinear(W, bias=True)
    x = torch.ones(3, 2)
    out = layer(x)
    assert out.shape == (3, 2)


def test_models_sparse_linear_is_alias():
    # btorch.models.SparseLinear is the same class as btorch.sparse.param.SparseLinear
    from btorch.models.linear import SparseLinear as ModelSparseLinear
    from btorch.sparse.param import SparseLinear as SparseSparseLinear2

    W = CSR.from_edges(
        row=torch.tensor([0]),
        col=torch.tensor([0]),
        data=torch.tensor([1.0]),
        shape=(1, 1),
    )
    layer = ModelSparseLinear(W)
    assert isinstance(layer, SparseSparseLinear2)


# ── Constraints (projection) ──────────────────────────────────────────────────


def test_nonnegative_projection():
    W = CSR.from_edges(
        row=torch.tensor([0, 0, 1]),
        col=torch.tensor([0, 1, 0]),
        data=torch.tensor([-1.0, 2.0, -3.0]),
        shape=(2, 2),
        constraint=NonNegative(),
    )
    W_constrained = constrain(W)
    assert torch.all(W_constrained.effective_values() >= 0)


def test_symmetric_projection():
    # After projection, W[i,j] == W[j,i]
    W = CSR.from_edges(
        row=torch.tensor([0, 1]),
        col=torch.tensor([1, 0]),
        data=torch.tensor([2.0, 4.0]),
        shape=(2, 2),
        constraint=Symmetric(),
    )
    W_c = constrain(W)
    vals = W_c.effective_values()
    torch.testing.assert_close(vals[0], vals[1])
    torch.testing.assert_close(vals[0], torch.tensor(3.0))


def test_bounded_projection():
    W = CSR.from_edges(
        row=torch.tensor([0, 1]),
        col=torch.tensor([0, 0]),
        data=torch.tensor([-2.0, 5.0]),
        shape=(2, 1),
        constraint=Bounded(min_val=0.0, max_val=3.0),
    )
    W_c = constrain(W)
    vals = W_c.effective_values()
    torch.testing.assert_close(vals, torch.tensor([0.0, 3.0]))


def test_constrain_caches_prepared_projection():
    W = CSR.from_edges(
        row=torch.tensor([0]),
        col=torch.tensor([0]),
        data=torch.tensor([1.0]),
        shape=(1, 1),
        constraint=NonNegative(),
    )
    _ = constrain(W)
    assert "projection" in W._cache
    # Second call does not recompute (cache hit)
    cached = W._cache["projection"]
    _ = constrain(W)
    assert W._cache["projection"] is cached


# ── GroupedMagnitude parameterization ────────────────────────────────────────


def test_grouped_magnitude_effective_values():
    # values[i] = initial_weight[i] * magnitude[group_index[i]]
    initial_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    group_index = torch.tensor([0, 0, 1, 1])
    magnitude = torch.tensor([2.0, 3.0])
    param = GroupedMagnitude(
        initial_weight=initial_weight,
        group_index=group_index,
        magnitude=magnitude,
    )
    eff = param.effective_values()
    torch.testing.assert_close(eff, torch.tensor([2.0, 2.0, 3.0, 3.0]))


def test_grouped_magnitude_on_sparse_tensor():
    W = CSR.from_edges(
        row=torch.tensor([0, 0, 1, 1]),
        col=torch.tensor([0, 1, 0, 1]),
        data=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        shape=(2, 2),
    )
    param = GroupedMagnitude(
        initial_weight=W.data.clone(),
        group_index=torch.tensor([0, 0, 1, 1]),
        magnitude=torch.tensor([5.0, 7.0]),
    )
    W = W.with_parameterization(param)
    eff = W.effective_values()
    torch.testing.assert_close(eff, torch.tensor([5.0, 5.0, 7.0, 7.0]))


# ── PyTree registration ───────────────────────────────────────────────────────


@pytest.mark.parametrize("Format", [COO, CSR, CSC])
def test_pytree_roundtrip(Format):
    W = Format.from_edges(
        row=torch.tensor([0, 1]),
        col=torch.tensor([1, 0]),
        data=torch.tensor([1.0, 2.0]),
        shape=(2, 2),
    )
    leaves, treespec = pytree.tree_flatten(W)
    W2 = pytree.tree_unflatten(leaves, treespec)
    torch.testing.assert_close(W.to_dense(), W2.to_dense())


def test_ell_pytree_roundtrip():
    W = ELL.random(shape=(4, 6), fan_out=2, seed=1)
    leaves, treespec = pytree.tree_flatten(W)
    W2 = pytree.tree_unflatten(leaves, treespec)
    torch.testing.assert_close(W.to_dense(), W2.to_dense())


def test_sparsetensor_pytree_roundtrip():
    W = SparseTensor.from_edges(
        row=torch.tensor([0, 1]),
        col=torch.tensor([1, 0]),
        data=torch.tensor([3.0, 4.0]),
        shape=(2, 2),
    )
    leaves, treespec = pytree.tree_flatten(W)
    W2 = pytree.tree_unflatten(leaves, treespec)
    torch.testing.assert_close(W.to_dense(), W2.to_dense())


# ── Event sparse mm ───────────────────────────────────────────────────────────


def test_event_mm_matches_numeric_mm():
    # event_mm applied to binary events must equal mm applied to dense 0/1 tensor
    W = CSR.from_edges(
        row=torch.tensor([0, 0, 2]),
        col=torch.tensor([0, 1, 1]),
        data=torch.tensor([1.0, 2.0, 3.0]),
        shape=(3, 2),
    )
    numeric = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    expected = W.mm(numeric)
    events = BinaryEvents(numeric, threshold=0.5)
    out = W.event_mm(events)
    torch.testing.assert_close(out, expected)


def test_spike_list_events_mm():
    W = CSR.from_edges(
        row=torch.tensor([0, 1]),
        col=torch.tensor([0, 0]),
        data=torch.tensor([2.0, 3.0]),
        shape=(2, 1),
    )
    # Batch of 1, spike at index 0
    events = SpikeListEvents(
        count=torch.tensor([1], dtype=torch.int32),
        indices=torch.tensor([[0]]),
        size=2,
    )
    out = W.event_mm(events)
    torch.testing.assert_close(out, torch.tensor([[2.0]]))


# ── DenseLinear orientation ───────────────────────────────────────────────────


def test_dense_linear_source_to_destination_orientation():
    # DenseConn uses (in_features, out_features) weight; x @ W gives output.
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    layer = DenseConn(2, 2, bias=False, weight=weight)
    x = torch.tensor([[2.0, 1.0]])
    torch.testing.assert_close(layer(x), x @ weight)


def test_linear_aliases_remain_available():
    assert DenseLinear is DenseConn
    assert SparseLinear is SparseConn


# ── padded_csr_layout ────────────────────────────────────────────────────────


def test_padded_csr_layout_cached():
    W = CSR.from_edges(
        row=torch.tensor([0, 0, 1]),
        col=torch.tensor([0, 2, 1]),
        data=torch.tensor([1.0, 2.0, 3.0]),
        shape=(2, 3),
    )
    layout = W.padded_csr_layout()
    layout2 = W.padded_csr_layout()
    assert layout is layout2  # same object (cached)
    assert layout.row_stride == 2  # max row length = 2 (row 0 has 2 edges)
    assert layout.indices.shape == (2, 2)


def test_ell_padded_csr_layout():
    W = ELL.random(shape=(4, 8), fan_out=3, seed=99)
    layout = W.padded_csr_layout()
    assert layout.row_stride == 3
    assert layout.indices is W.indices


# ── Backend availability ──────────────────────────────────────────────────────


def test_available_backends_includes_native():
    from btorch.sparse import available_backends

    backends = available_backends()
    assert "native" in backends


def test_torch_sparse_raises_backend_unavailable_when_missing():
    from btorch.sparse.backends import torch_sparse

    if torch_sparse.is_available():
        pytest.skip("torch_sparse is installed in this environment.")
    W = CSR.from_edges(
        row=torch.tensor([0]),
        col=torch.tensor([0]),
        data=torch.tensor([1.0]),
        shape=(1, 1),
    )
    with pytest.raises(BackendUnavailableError):
        torch_sparse.sparse_mm(W, torch.ones(1, 1))
