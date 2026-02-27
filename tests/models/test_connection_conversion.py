import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import torch

from btorch.connectome.connection import (
    make_hetersynapse_conn,
    make_hetersynapse_constrained_conn,
)
from btorch.models.connection_conversion import (
    convert_connection_layer,
    convert_connection_layer_from_checkpoint,
)
from btorch.models.linear import DenseConn, SparseConn, SparseConstrainedConn


def _layer_weight_dense(layer: torch.nn.Module) -> np.ndarray:
    with torch.no_grad():
        x = torch.eye(layer.in_features, dtype=torch.float32)
        y = layer(x)
    return y.detach().cpu().numpy()


def _sum_receptors_per_neuron(y_heter: torch.Tensor, n_receptor: int) -> torch.Tensor:
    assert y_heter.shape[-1] % n_receptor == 0
    n_post = y_heter.shape[-1] // n_receptor
    return y_heter.reshape(*y_heter.shape[:-1], n_post, n_receptor).sum(dim=-1)


def _assert_base_heter_forward_match(
    base_layer: torch.nn.Module,
    heter_layer: torch.nn.Module,
    x: torch.Tensor,
    n_receptor: int,
) -> None:
    with torch.no_grad():
        y_base = base_layer(x)
        y_heter = heter_layer(x)
    y_heter_collapsed = _sum_receptors_per_neuron(y_heter, n_receptor)
    torch.testing.assert_close(y_base, y_heter_collapsed, atol=1e-6, rtol=0.0)


def test_convert_sparse_conn_neuron_mode_no_split_roundtrip():
    neurons = pd.DataFrame(
        {
            "root_id": [100, 101, 102, 103],
            "simple_id": [0, 1, 2, 3],
            "EI": ["E", "I", "E", "I"],
            "cell_type": ["a", "a", "b", "b"],
        }
    )
    conn_base = scipy.sparse.coo_array(
        ([1.0, 2.0, 3.0, 4.0], ([0, 1, 2, 3], [1, 2, 3, 0])),
        shape=(4, 4),
    )
    conn_ref, receptor_idx = make_hetersynapse_conn(
        neurons=neurons,
        connections=conn_base,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
    )
    layer = SparseConn(conn=conn_base, enforce_dale=False)

    layer_heter = convert_connection_layer(
        layer,
        target_layout="heter",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx,
        neurons=neurons,
        receptor_type_col="EI",
    )
    dense_heter = _layer_weight_dense(layer_heter)
    np.testing.assert_allclose(dense_heter, conn_ref.toarray(), atol=1e-6)

    n_receptor = len(receptor_idx)
    x = torch.tensor([0.5, -1.0, 2.0, 1.0], dtype=torch.float32)
    x_batch = torch.stack([x, x + 0.3], dim=0)
    _assert_base_heter_forward_match(layer, layer_heter, x, n_receptor)
    _assert_base_heter_forward_match(layer, layer_heter, x_batch, n_receptor)

    base_coo = conn_base.tocoo()
    for pre, post, value in zip(base_coo.row, base_coo.col, base_coo.data):
        receptor_slice = dense_heter[pre, post * n_receptor : (post + 1) * n_receptor]
        assert np.count_nonzero(np.abs(receptor_slice) > 1e-8) == 1
        assert np.isclose(receptor_slice.sum(), value)

    layer_base = convert_connection_layer(
        layer_heter,
        target_layout="base",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx,
    )
    np.testing.assert_allclose(
        _layer_weight_dense(layer_base), conn_base.toarray(), atol=1e-6
    )


def test_convert_sparse_conn_connection_mode_with_assignment_roundtrip():
    neurons = pd.DataFrame(
        {
            "root_id": [10, 11, 12, 13],
            "simple_id": [0, 1, 2, 3],
            "cell_type": ["a", "a", "b", "b"],
        }
    )
    conn_df = pd.DataFrame(
        {
            "pre_simple_id": [0, 1, 2, 3],
            "post_simple_id": [1, 2, 3, 0],
            "syn_count": [2.0, 1.0, 4.0, 3.0],
            "EI": ["E", "I", "E", "I"],
        }
    )
    conn_df["pre_root_id"] = conn_df["pre_simple_id"].map(
        dict(zip(neurons["simple_id"], neurons["root_id"]))
    )
    conn_df["post_root_id"] = conn_df["post_simple_id"].map(
        dict(zip(neurons["simple_id"], neurons["root_id"]))
    )

    conn_ref, receptor_idx = make_hetersynapse_conn(
        neurons=neurons,
        connections=conn_df,
        receptor_type_col="EI",
        receptor_type_mode="connection",
    )
    conn_base = scipy.sparse.coo_array(
        (
            conn_df["syn_count"].values,
            (conn_df["pre_simple_id"].values, conn_df["post_simple_id"].values),
        ),
        shape=(4, 4),
    )
    receptor_lookup = receptor_idx.set_index("receptor_type")["receptor_index"]
    assignment = conn_df[["pre_simple_id", "post_simple_id", "EI"]].copy()
    assignment["receptor_index"] = assignment["EI"].map(receptor_lookup)
    assignment = assignment[["pre_simple_id", "post_simple_id", "receptor_index"]]

    layer = SparseConn(conn=conn_base, enforce_dale=False)
    layer_heter = convert_connection_layer(
        layer,
        target_layout="heter",
        receptor_type_mode="connection",
        receptor_type_index=receptor_idx,
        edge_receptor_assignment=assignment,
    )
    np.testing.assert_allclose(
        _layer_weight_dense(layer_heter), conn_ref.toarray(), atol=1e-6
    )
    n_receptor = len(receptor_idx)
    x = torch.tensor([-0.2, 0.7, 1.2, -0.4], dtype=torch.float32)
    x_batch = torch.stack([x, x - 0.5], dim=0)
    _assert_base_heter_forward_match(layer, layer_heter, x, n_receptor)
    _assert_base_heter_forward_match(layer, layer_heter, x_batch, n_receptor)

    layer_base = convert_connection_layer(
        layer_heter,
        target_layout="base",
        receptor_type_mode="connection",
        receptor_type_index=receptor_idx,
    )
    np.testing.assert_allclose(
        _layer_weight_dense(layer_base), conn_base.toarray(), atol=1e-6
    )

    with pytest.raises(ValueError, match="base->heter conversion requires"):
        convert_connection_layer(
            layer,
            target_layout="heter",
            receptor_type_mode="connection",
            receptor_type_index=receptor_idx,
        )


def test_convert_sparse_constrained_group_policy_and_roundtrip():
    neurons = pd.DataFrame(
        {
            "root_id": [100, 101, 102, 103],
            "simple_id": [0, 1, 2, 3],
            "EI": ["E", "I", "E", "I"],
            "cell_type": ["a", "a", "b", "b"],
        }
    )
    conn_base = scipy.sparse.coo_array(
        ([1.0, 2.0, -1.0, -2.0], ([0, 1, 2, 3], [1, 0, 3, 2])),
        shape=(4, 4),
    )
    constraint = scipy.sparse.coo_array(
        ([1, 1, 2, 2], ([0, 1, 2, 3], [1, 0, 3, 2])),
        shape=(4, 4),
    )
    _, receptor_idx = make_hetersynapse_conn(
        neurons=neurons,
        connections=conn_base,
        receptor_type_col="EI",
        receptor_type_mode="neuron",
    )
    layer = SparseConstrainedConn(
        conn=conn_base,
        constraint=constraint,
        enforce_dale=False,
    )
    layer.magnitude.data = torch.tensor([1.5, 0.5], dtype=layer.magnitude.dtype)
    source_dense = _layer_weight_dense(layer)

    heter_ind = convert_connection_layer(
        layer,
        target_layout="heter",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx,
        neurons=neurons,
        group_policy="independent",
    )
    heter_shared = convert_connection_layer(
        layer,
        target_layout="heter",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx,
        neurons=neurons,
        group_policy="shared",
    )
    assert isinstance(heter_ind, SparseConstrainedConn)
    assert isinstance(heter_shared, SparseConstrainedConn)
    assert heter_ind.magnitude.numel() > heter_shared.magnitude.numel()
    np.testing.assert_allclose(
        _layer_weight_dense(heter_ind), _layer_weight_dense(heter_shared), atol=1e-6
    )
    n_receptor = len(receptor_idx)
    x = torch.tensor([1.0, -1.0, 0.5, 2.0], dtype=torch.float32)
    x_batch = torch.stack([x, x * 0.5], dim=0)
    _assert_base_heter_forward_match(layer, heter_ind, x, n_receptor)
    _assert_base_heter_forward_match(layer, heter_ind, x_batch, n_receptor)
    _assert_base_heter_forward_match(layer, heter_shared, x, n_receptor)
    _assert_base_heter_forward_match(layer, heter_shared, x_batch, n_receptor)

    base_roundtrip = convert_connection_layer(
        heter_ind,
        target_layout="base",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx,
    )
    np.testing.assert_allclose(
        _layer_weight_dense(base_roundtrip),
        source_dense,
        atol=1e-6,
    )


def test_convert_dense_conn_neuron_mode_no_split_roundtrip():
    neurons = pd.DataFrame(
        {
            "root_id": [100, 101, 102, 103],
            "simple_id": [0, 1, 2, 3],
            "EI": ["E", "I", "E", "I"],
        }
    )
    base_weight = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
            [4.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    conn_ref, receptor_idx = make_hetersynapse_conn(
        neurons=neurons,
        connections=scipy.sparse.coo_array(base_weight),
        receptor_type_col="EI",
        receptor_type_mode="neuron",
    )
    layer = DenseConn(
        in_features=4,
        out_features=4,
        weight=torch.tensor(base_weight, dtype=torch.float32),
        enforce_dale=False,
    )

    layer_heter = convert_connection_layer(
        layer,
        target_layout="heter",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx,
        neurons=neurons,
        receptor_type_col="EI",
    )
    assert isinstance(layer_heter, DenseConn)
    dense_heter = _layer_weight_dense(layer_heter)
    np.testing.assert_allclose(dense_heter, conn_ref.toarray(), atol=1e-6)

    n_receptor = len(receptor_idx)
    x = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float32)
    x_batch = torch.stack([x, x + 0.6], dim=0)
    _assert_base_heter_forward_match(layer, layer_heter, x, n_receptor)
    _assert_base_heter_forward_match(layer, layer_heter, x_batch, n_receptor)

    row, col = np.nonzero(base_weight)
    for pre, post in zip(row, col):
        receptor_slice = dense_heter[pre, post * n_receptor : (post + 1) * n_receptor]
        assert np.count_nonzero(np.abs(receptor_slice) > 1e-8) == 1
        assert np.isclose(receptor_slice.sum(), base_weight[pre, post])

    layer_base = convert_connection_layer(
        layer_heter,
        target_layout="base",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx,
    )
    assert isinstance(layer_base, DenseConn)
    np.testing.assert_allclose(_layer_weight_dense(layer_base), base_weight, atol=1e-6)


def test_convert_dense_conn_connection_mode_with_assignment_roundtrip():
    conn_df = pd.DataFrame(
        {
            "pre_simple_id": [0, 1, 2, 3],
            "post_simple_id": [1, 2, 3, 0],
            "syn_count": [2.0, 1.0, 4.0, 3.0],
            "EI": ["E", "I", "E", "I"],
        }
    )
    receptor_idx = pd.DataFrame(
        [(0, "E"), (1, "I")], columns=["receptor_index", "receptor_type"]
    )
    receptor_lookup = receptor_idx.set_index("receptor_type")["receptor_index"]
    assignment = conn_df[["pre_simple_id", "post_simple_id", "EI"]].copy()
    assignment["receptor_index"] = assignment["EI"].map(receptor_lookup)
    assignment = assignment[["pre_simple_id", "post_simple_id", "receptor_index"]]

    base_weight = np.zeros((4, 4), dtype=np.float32)
    for _, row in conn_df.iterrows():
        base_weight[int(row["pre_simple_id"]), int(row["post_simple_id"])] = row[
            "syn_count"
        ]

    layer = DenseConn(
        in_features=4,
        out_features=4,
        weight=torch.tensor(base_weight, dtype=torch.float32),
        bias=torch.tensor([0.1, 0.2, -0.1, 0.3], dtype=torch.float32),
        enforce_dale=False,
    )
    layer_heter = convert_connection_layer(
        layer,
        target_layout="heter",
        receptor_type_mode="connection",
        receptor_type_index=receptor_idx,
        edge_receptor_assignment=assignment,
    )
    assert isinstance(layer_heter, DenseConn)
    n_receptor = len(receptor_idx)
    x = torch.tensor([1.0, 0.5, -1.0, 0.2], dtype=torch.float32)
    x_batch = torch.stack([x, x - 0.8], dim=0)
    _assert_base_heter_forward_match(layer, layer_heter, x, n_receptor)
    _assert_base_heter_forward_match(layer, layer_heter, x_batch, n_receptor)

    layer_base = convert_connection_layer(
        layer_heter,
        target_layout="base",
        receptor_type_mode="connection",
        receptor_type_index=receptor_idx,
    )
    assert isinstance(layer_base, DenseConn)
    expected_with_bias = base_weight + layer.bias.detach().cpu().numpy()[None, :]
    np.testing.assert_allclose(
        _layer_weight_dense(layer_base), expected_with_bias, atol=1e-6
    )
    np.testing.assert_allclose(
        layer_base.bias.detach().cpu().numpy(),
        layer.bias.detach().cpu().numpy(),
        atol=1e-6,
    )


def test_convert_connection_layer_from_checkpoint_matches_instance_path():
    neurons = pd.DataFrame(
        {
            "root_id": [100, 101, 102, 103],
            "simple_id": [0, 1, 2, 3],
            "EI": ["E", "I", "E", "I"],
            "cell_type": ["a", "a", "b", "b"],
        }
    )
    conn_df = pd.DataFrame(
        {
            "pre_simple_id": [0, 1, 2, 3],
            "post_simple_id": [1, 2, 3, 0],
            "syn_count": [3.0, 2.0, 1.0, 4.0],
            "pre_root_id": [100, 101, 102, 103],
            "post_root_id": [101, 102, 103, 100],
            "EI": ["E", "I", "E", "I"],
        }
    )
    conn_base = scipy.sparse.coo_array(
        (
            conn_df["syn_count"].values,
            (conn_df["pre_simple_id"].values, conn_df["post_simple_id"].values),
        ),
        shape=(4, 4),
    )
    receptor_idx_conn = pd.DataFrame(
        [(0, "E"), (1, "I")], columns=["receptor_index", "receptor_type"]
    )
    assignment = conn_df[["pre_simple_id", "post_simple_id", "EI"]].copy()
    assignment["receptor_index"] = assignment["EI"].map(
        {"E": 0, "I": 1},
    )
    assignment = assignment[["pre_simple_id", "post_simple_id", "receptor_index"]]

    src_sparse = SparseConn(conn=conn_base, enforce_dale=False)
    src_sparse.magnitude.data *= 1.1
    inst_sparse = convert_connection_layer(
        src_sparse,
        target_layout="heter",
        receptor_type_mode="connection",
        receptor_type_index=receptor_idx_conn,
        edge_receptor_assignment=assignment,
    )
    ckpt_sparse = convert_connection_layer_from_checkpoint(
        state_dict=src_sparse.state_dict(),
        source_class=SparseConn,
        conn=conn_base,
        enforce_dale=False,
        target_layout="heter",
        receptor_type_mode="connection",
        receptor_type_index=receptor_idx_conn,
        edge_receptor_assignment=assignment,
    )
    np.testing.assert_allclose(
        _layer_weight_dense(inst_sparse), _layer_weight_dense(ckpt_sparse), atol=1e-6
    )

    conn_heter, constraint_heter, receptor_idx_neuron = (
        make_hetersynapse_constrained_conn(
            neurons=neurons,
            connections=conn_df,
            cell_type_col="cell_type",
            receptor_type_col="EI",
            receptor_type_mode="neuron",
            constraint_mode="full",
        )
    )
    conn_heter_coo = conn_heter.tocoo()
    topology_only_conn = scipy.sparse.coo_array(
        (
            np.ones_like(conn_heter_coo.data),
            (conn_heter_coo.row, conn_heter_coo.col),
        ),
        shape=conn_heter_coo.shape,
    )

    src_constrained_false = SparseConstrainedConn(
        conn=conn_heter,
        constraint=constraint_heter,
        enforce_dale=False,
    )
    src_constrained_false.magnitude.data *= 0.8
    inst_constrained_false = convert_connection_layer(
        src_constrained_false,
        target_layout="base",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx_neuron,
    )
    ckpt_constrained_false = convert_connection_layer_from_checkpoint(
        state_dict=src_constrained_false.state_dict(),
        source_class=SparseConstrainedConn,
        conn=conn_heter,
        constraint=constraint_heter,
        enforce_dale=False,
        target_layout="base",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx_neuron,
    )
    np.testing.assert_allclose(
        _layer_weight_dense(inst_constrained_false),
        _layer_weight_dense(ckpt_constrained_false),
        atol=1e-6,
    )

    src_constrained_true = SparseConstrainedConn(
        conn=conn_heter,
        constraint=constraint_heter,
        enforce_dale=False,
        persist_initial_weight=True,
    )
    src_constrained_true.magnitude.data *= 1.1
    inst_constrained_true = convert_connection_layer(
        src_constrained_true,
        target_layout="base",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx_neuron,
    )
    ckpt_constrained_true = convert_connection_layer_from_checkpoint(
        state_dict=src_constrained_true.state_dict(),
        source_class=SparseConstrainedConn,
        conn=conn_heter,
        constraint=constraint_heter,
        enforce_dale=False,
        target_layout="base",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx_neuron,
    )
    np.testing.assert_allclose(
        _layer_weight_dense(inst_constrained_true),
        _layer_weight_dense(ckpt_constrained_true),
        atol=1e-6,
    )
    ckpt_constrained_true_no_conn = convert_connection_layer_from_checkpoint(
        state_dict=src_constrained_true.state_dict(),
        source_class=SparseConstrainedConn,
        conn=None,
        constraint=constraint_heter,
        enforce_dale=False,
        target_layout="base",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx_neuron,
    )
    np.testing.assert_allclose(
        _layer_weight_dense(inst_constrained_true),
        _layer_weight_dense(ckpt_constrained_true_no_conn),
        atol=1e-6,
    )
    ckpt_constrained_true_conn_override = convert_connection_layer_from_checkpoint(
        state_dict=src_constrained_true.state_dict(),
        source_class=SparseConstrainedConn,
        conn=topology_only_conn,
        constraint=constraint_heter,
        enforce_dale=False,
        target_layout="base",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx_neuron,
    )
    source_override = SparseConstrainedConn(
        conn=topology_only_conn,
        constraint=constraint_heter,
        enforce_dale=False,
        persist_initial_weight=True,
    )
    source_override_state = dict(src_constrained_true.state_dict())
    source_override_state.pop("indices", None)
    source_override_state.pop("initial_weight", None)
    source_override.load_state_dict(source_override_state, strict=False)
    expected_override = convert_connection_layer(
        source_override,
        target_layout="base",
        receptor_type_mode="neuron",
        receptor_type_index=receptor_idx_neuron,
    )
    np.testing.assert_allclose(
        _layer_weight_dense(ckpt_constrained_true_conn_override),
        _layer_weight_dense(expected_override),
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="requires either conn"):
        convert_connection_layer_from_checkpoint(
            state_dict=src_constrained_false.state_dict(),
            source_class=SparseConstrainedConn,
            conn=None,
            constraint=constraint_heter,
            enforce_dale=False,
            target_layout="base",
            receptor_type_mode="neuron",
            receptor_type_index=receptor_idx_neuron,
        )

    dense_weight = conn_base.toarray().astype(np.float32)
    src_dense = DenseConn(
        in_features=4,
        out_features=4,
        weight=torch.tensor(dense_weight, dtype=torch.float32),
        bias=torch.tensor([0.0, 0.2, -0.1, 0.3], dtype=torch.float32),
        enforce_dale=False,
    )
    inst_dense = convert_connection_layer(
        src_dense,
        target_layout="heter",
        receptor_type_mode="connection",
        receptor_type_index=receptor_idx_conn,
        edge_receptor_assignment=assignment,
    )
    ckpt_dense = convert_connection_layer_from_checkpoint(
        state_dict=src_dense.state_dict(),
        source_class=DenseConn,
        in_features=4,
        out_features=4,
        enforce_dale=False,
        target_layout="heter",
        receptor_type_mode="connection",
        receptor_type_index=receptor_idx_conn,
        edge_receptor_assignment=assignment,
    )
    assert isinstance(inst_dense, DenseConn)
    assert isinstance(ckpt_dense, DenseConn)
    np.testing.assert_allclose(
        _layer_weight_dense(inst_dense), _layer_weight_dense(ckpt_dense), atol=1e-6
    )
    np.testing.assert_allclose(
        inst_dense.bias.detach().cpu().numpy(),
        ckpt_dense.bias.detach().cpu().numpy(),
        atol=1e-6,
    )


def test_convert_connection_layer_split_and_validation_errors():
    receptor_idx = pd.DataFrame(
        [(0, "E"), (1, "I")], columns=["receptor_index", "receptor_type"]
    )
    conn_base = scipy.sparse.coo_array(([4.0], ([0], [1])), shape=(2, 2))
    layer = SparseConn(conn=conn_base, enforce_dale=False)

    edge_receptor_weight = pd.DataFrame(
        {
            "pre_simple_id": [0, 0],
            "post_simple_id": [1, 1],
            "receptor_index": [0, 1],
            "weight_coeff": [0.25, 0.75],
        }
    )
    heter_split = convert_connection_layer(
        layer,
        target_layout="heter",
        receptor_type_mode="connection",
        receptor_type_index=receptor_idx,
        allow_weight_split=True,
        edge_receptor_weight=edge_receptor_weight,
    )
    dense = _layer_weight_dense(heter_split)
    assert np.isclose(dense[0, 1 * 2 + 0], 1.0)
    assert np.isclose(dense[0, 1 * 2 + 1], 3.0)

    with pytest.raises(ValueError, match="allow_weight_split=False"):
        convert_connection_layer(
            layer,
            target_layout="heter",
            receptor_type_mode="connection",
            receptor_type_index=receptor_idx,
            edge_receptor_weight=edge_receptor_weight,
        )

    bad_assignment = pd.DataFrame(
        {
            "pre_simple_id": [0],
            "post_simple_id": [1],
            "receptor_index": [2],
        }
    )
    with pytest.raises(ValueError, match="out of range"):
        convert_connection_layer(
            layer,
            target_layout="heter",
            receptor_type_mode="connection",
            receptor_type_index=receptor_idx,
            edge_receptor_assignment=bad_assignment,
        )
