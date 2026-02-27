"""Connection-layer conversion utilities.

This module converts connection layers between:

1. Base layout: ``(N_pre, N_post)``
2. Heter layout: ``(N_pre, N_post * n_receptor)``

Supported layer classes:

- ``DenseConn``
- ``SparseConn``
- ``SparseConstrainedConn``

Dense conversion is implemented directly on dense tensors and does not route
through sparse conversion.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import scipy.sparse
import torch

from .linear import DenseConn, SparseBackend, SparseConn, SparseConstrainedConn


ReceptorTypeMode = Literal["neuron", "connection"]
TargetLayout = Literal["base", "heter"]
GroupPolicy = Literal["independent", "shared"]
ConnectionLayer = DenseConn | SparseConn | SparseConstrainedConn
SourceLayerClass = type[DenseConn] | type[SparseConn] | type[SparseConstrainedConn]

__all__ = [
    "ConnectionLayer",
    "GroupPolicy",
    "ReceptorTypeMode",
    "TargetLayout",
    "convert_connection_layer",
    "convert_connection_layer_from_checkpoint",
]


def _as_coo(conn: scipy.sparse.sparray) -> scipy.sparse.coo_array:
    if not isinstance(conn, scipy.sparse.coo_array):
        conn = conn.tocoo()
    conn = conn.copy()
    conn.sum_duplicates()
    return conn


def _require_columns(df: pd.DataFrame, required_cols: set[str], name: str) -> None:
    missing = required_cols - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_str}")


def _normalize_receptor_type_mode(mode: str) -> ReceptorTypeMode:
    mode = mode.lower()
    if mode not in get_args(ReceptorTypeMode):
        raise ValueError(
            "receptor_type_mode must be one of "
            f"{get_args(ReceptorTypeMode)}, got {mode!r}"
        )
    return mode  # type: ignore[return-value]


def _edge_df_from_sparse_conn(conn: scipy.sparse.coo_array) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pre_simple_id": conn.row.astype(np.int64),
            "post_simple_id": conn.col.astype(np.int64),
            "weight": conn.data,
        }
    )


def _edge_df_from_dense_weight(weight: np.ndarray) -> pd.DataFrame:
    row, col = np.nonzero(weight)
    if row.size == 0:
        return pd.DataFrame(
            columns=["pre_simple_id", "post_simple_id", "weight"],
        )
    return pd.DataFrame(
        {
            "pre_simple_id": row.astype(np.int64),
            "post_simple_id": col.astype(np.int64),
            "weight": weight[row, col],
        }
    )


def _resolve_assignment_from_neurons(
    edge_df: pd.DataFrame,
    receptor_type_index: pd.DataFrame,
    neurons: pd.DataFrame,
    receptor_type_col: str,
) -> pd.DataFrame:
    _require_columns(
        neurons,
        required_cols={"simple_id", receptor_type_col},
        name="neurons",
    )
    receptor_type_index = receptor_type_index.copy()
    _require_columns(
        receptor_type_index,
        required_cols={"receptor_index"},
        name="receptor_type_index",
    )

    if edge_df.empty:
        return pd.DataFrame(
            columns=["pre_simple_id", "post_simple_id", "receptor_index"],
        )

    neuron_lookup = neurons.set_index("simple_id")[receptor_type_col]
    edge_df = edge_df.copy()
    edge_df["pre_receptor_type"] = edge_df["pre_simple_id"].map(neuron_lookup)
    edge_df["post_receptor_type"] = edge_df["post_simple_id"].map(neuron_lookup)
    if (
        edge_df["pre_receptor_type"].isna().any()
        or edge_df["post_receptor_type"].isna().any()
    ):
        raise ValueError(
            "neurons/receptor_type_col does not provide receptor types for all edges."
        )

    if {
        "pre_receptor_type",
        "post_receptor_type",
    }.issubset(receptor_type_index.columns):
        lookup = receptor_type_index[
            ["receptor_index", "pre_receptor_type", "post_receptor_type"]
        ]
        merged = edge_df.merge(
            lookup,
            on=["pre_receptor_type", "post_receptor_type"],
            how="left",
            validate="many_to_one",
        )
    elif "receptor_type" in receptor_type_index.columns:
        lookup = receptor_type_index[["receptor_index", "receptor_type"]]
        merged = edge_df.merge(
            lookup,
            left_on="pre_receptor_type",
            right_on="receptor_type",
            how="left",
            validate="many_to_one",
        )
    else:
        raise ValueError(
            "receptor_type_index must contain either "
            "('pre_receptor_type', 'post_receptor_type') or 'receptor_type'."
        )

    if merged["receptor_index"].isna().any():
        missing = merged[merged["receptor_index"].isna()][
            ["pre_receptor_type", "post_receptor_type"]
        ].drop_duplicates()
        raise ValueError(
            "Failed to resolve receptor_index from neurons/receptor_type_index for "
            f"some edges. Missing receptor types: {missing.to_dict('records')}"
        )

    return merged[["pre_simple_id", "post_simple_id", "receptor_index"]]


def _validate_assignment_df(
    assignment_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    n_receptor: int,
    allow_split: bool,
) -> pd.DataFrame:
    required_cols = {"pre_simple_id", "post_simple_id", "receptor_index"}
    _require_columns(
        assignment_df,
        required_cols=required_cols,
        name="assignment dataframe",
    )
    assignment_df = assignment_df.copy()
    assignment_df["receptor_index"] = assignment_df["receptor_index"].astype(np.int64)
    if (assignment_df["receptor_index"] < 0).any() or (
        assignment_df["receptor_index"] >= n_receptor
    ).any():
        raise ValueError(
            "receptor_index out of range for receptor_type_index. "
            f"Valid range is [0, {n_receptor - 1}]."
        )

    if edge_df.empty:
        return assignment_df

    merge_validate = "one_to_one" if not allow_split else "many_to_one"
    merged = assignment_df.merge(
        edge_df[["pre_simple_id", "post_simple_id"]],
        on=["pre_simple_id", "post_simple_id"],
        how="left",
        indicator=True,
        validate=merge_validate,
    )
    bad = merged[merged["_merge"] != "both"]
    if not bad.empty:
        raise ValueError(
            "assignment dataframe contains edges that do not exist in source matrix."
        )
    return assignment_df


def _expand_edges_with_one_hot(
    edge_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    n_receptor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignment_df = _validate_assignment_df(
        assignment_df=assignment_df,
        edge_df=edge_df,
        n_receptor=n_receptor,
        allow_split=False,
    )

    if edge_df.empty:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    merged = edge_df.merge(
        assignment_df,
        on=["pre_simple_id", "post_simple_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["receptor_index"].isna().any():
        raise ValueError(
            "Each nonzero base edge must have exactly one receptor assignment "
            "in no-split mode."
        )

    rows = merged["pre_simple_id"].to_numpy(dtype=np.int64)
    cols = merged["post_simple_id"].to_numpy(dtype=np.int64) * n_receptor + merged[
        "receptor_index"
    ].to_numpy(dtype=np.int64)
    vals = merged["weight"].to_numpy()
    return rows, cols, vals


def _expand_edges_with_split(
    edge_df: pd.DataFrame,
    edge_receptor_weight: pd.DataFrame,
    n_receptor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required_cols = {
        "pre_simple_id",
        "post_simple_id",
        "receptor_index",
        "weight_coeff",
    }
    _require_columns(
        edge_receptor_weight,
        required_cols=required_cols,
        name="edge_receptor_weight",
    )

    weights_df = edge_receptor_weight.copy()
    weights_df["receptor_index"] = weights_df["receptor_index"].astype(np.int64)
    if (weights_df["receptor_index"] < 0).any() or (
        weights_df["receptor_index"] >= n_receptor
    ).any():
        raise ValueError(
            "receptor_index out of range for receptor_type_index in "
            "edge_receptor_weight."
        )

    weights_df = _validate_assignment_df(
        assignment_df=weights_df[["pre_simple_id", "post_simple_id", "receptor_index"]],
        edge_df=edge_df,
        n_receptor=n_receptor,
        allow_split=True,
    ).merge(
        weights_df[list(required_cols)],
        on=["pre_simple_id", "post_simple_id", "receptor_index"],
        how="left",
        validate="one_to_one",
    )

    if edge_df.empty:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    merged = edge_df.merge(
        weights_df,
        on=["pre_simple_id", "post_simple_id"],
        how="left",
        validate="one_to_many",
    )
    if merged["receptor_index"].isna().any():
        raise ValueError(
            "edge_receptor_weight must define at least one receptor row "
            "for every nonzero base edge."
        )

    rows = merged["pre_simple_id"].to_numpy(dtype=np.int64)
    cols = merged["post_simple_id"].to_numpy(dtype=np.int64) * n_receptor + merged[
        "receptor_index"
    ].to_numpy(dtype=np.int64)
    vals = merged["weight"].to_numpy() * merged["weight_coeff"].to_numpy()
    return rows, cols, vals


def _collapse_heter_sparse_to_base(
    conn_heter: scipy.sparse.sparray,
    n_receptor: int,
) -> scipy.sparse.coo_array:
    conn_heter = _as_coo(conn_heter)
    if conn_heter.shape[1] % n_receptor != 0:
        raise ValueError(
            "heter connection output dimension must be divisible by receptor count. "
            f"Got shape={conn_heter.shape}, n_receptor={n_receptor}."
        )
    n_post = conn_heter.shape[1] // n_receptor
    post = conn_heter.col // n_receptor
    conn_base = scipy.sparse.coo_array(
        (conn_heter.data, (conn_heter.row, post)),
        shape=(conn_heter.shape[0], n_post),
    )
    conn_base.sum_duplicates()
    return conn_base


def _collapse_heter_dense_to_base(
    weight_heter: np.ndarray,
    n_receptor: int,
) -> np.ndarray:
    if weight_heter.shape[1] % n_receptor != 0:
        raise ValueError(
            "heter connection output dimension must be divisible by receptor count. "
            f"Got shape={weight_heter.shape}, n_receptor={n_receptor}."
        )
    n_post = weight_heter.shape[1] // n_receptor
    return weight_heter.reshape(weight_heter.shape[0], n_post, n_receptor).sum(axis=2)


def _factorized_groups(group_keys: Any) -> np.ndarray:
    if not isinstance(group_keys, (np.ndarray, pd.Series, pd.Index)):
        group_keys = pd.Index(group_keys)
    labels, _ = pd.factorize(group_keys)
    return labels + 1


def _build_constraint_from_groups(
    row: np.ndarray,
    col: np.ndarray,
    group_keys: Any,
    shape: tuple[int, int],
) -> scipy.sparse.coo_array:
    group_ids = _factorized_groups(group_keys)
    return scipy.sparse.coo_array((group_ids, (row, col)), shape=shape)


def _layer_conn_coo(
    layer: SparseConn | SparseConstrainedConn,
) -> scipy.sparse.coo_array:
    effective_weight = layer._get_effective_weight().detach().cpu().numpy()
    indices = layer.indices.detach().cpu().numpy()
    row = indices[1]
    col = indices[0]
    conn = scipy.sparse.coo_array(
        (effective_weight, (row, col)),
        shape=(layer.in_features, layer.out_features),
    )
    conn.sum_duplicates()
    return conn


def _layer_constraint_coo(layer: SparseConstrainedConn) -> scipy.sparse.coo_array:
    indices = layer.indices.detach().cpu().numpy()
    group = (layer._constraint_scatter_indices.detach().cpu().numpy() + 1).astype(int)
    row = indices[1]
    col = indices[0]
    constraint = scipy.sparse.coo_array(
        (group, (row, col)),
        shape=(layer.in_features, layer.out_features),
    )
    constraint.sum_duplicates()
    return constraint


def _expand_constraint_from_source(
    conn_base: scipy.sparse.coo_array,
    conn_heter: scipy.sparse.coo_array,
    source_constraint: scipy.sparse.coo_array | None,
    group_policy: GroupPolicy,
) -> scipy.sparse.coo_array:
    if group_policy not in get_args(GroupPolicy):
        raise ValueError(
            f"group_policy must be one of {get_args(GroupPolicy)}, got {group_policy!r}"
        )

    if source_constraint is None:
        group_keys = np.arange(conn_heter.nnz, dtype=np.int64)
    else:
        base_edge_df = pd.DataFrame(
            {
                "pre_simple_id": conn_base.row.astype(np.int64),
                "post_simple_id": conn_base.col.astype(np.int64),
                "source_group": source_constraint.data.astype(np.int64),
            }
        )
        n_receptor = conn_heter.shape[1] // conn_base.shape[1]
        expanded_df = pd.DataFrame(
            {
                "pre_simple_id": conn_heter.row.astype(np.int64),
                "post_simple_id": (conn_heter.col // n_receptor).astype(np.int64),
                "receptor_index": (conn_heter.col % n_receptor).astype(np.int64),
            }
        )
        merged = expanded_df.merge(
            base_edge_df,
            on=["pre_simple_id", "post_simple_id"],
            how="left",
            validate="many_to_one",
        )
        if merged["source_group"].isna().any():
            raise ValueError(
                "Failed to map expanded heter edges back to source constraint groups."
            )
        if group_policy == "shared":
            group_keys = merged["source_group"].to_numpy(dtype=np.int64)
        else:
            group_keys = list(
                zip(
                    merged["source_group"].to_numpy(dtype=np.int64),
                    merged["receptor_index"].to_numpy(dtype=np.int64),
                )
            )

    return _build_constraint_from_groups(
        row=conn_heter.row.astype(np.int64),
        col=conn_heter.col.astype(np.int64),
        group_keys=group_keys,
        shape=conn_heter.shape,
    )


def _convert_bias_base_to_heter(
    bias: torch.Tensor | None,
    n_receptor: int,
) -> torch.Tensor | None:
    if bias is None:
        return None
    bias = bias.detach().clone()
    return bias.repeat_interleave(n_receptor) / float(n_receptor)


def _convert_bias_heter_to_base(
    bias: torch.Tensor | None,
    n_receptor: int,
) -> torch.Tensor | None:
    if bias is None:
        return None
    if bias.numel() % n_receptor != 0:
        raise ValueError(
            "heter bias length must be divisible by receptor count for collapse. "
            f"Got len={bias.numel()}, n_receptor={n_receptor}."
        )
    n_post = bias.numel() // n_receptor
    return bias.detach().clone().reshape(n_post, n_receptor).sum(dim=1)


def _dense_weight_matrix(layer: DenseConn) -> np.ndarray:
    return layer.weight.detach().cpu().numpy().T


def _dense_enforce_dale(layer: DenseConn) -> bool:
    return layer.initial_sign is not None


def _validate_target_class(
    layer: ConnectionLayer,
    target_class: type[DenseConn]
    | type[SparseConn]
    | type[SparseConstrainedConn]
    | None,
) -> type[DenseConn] | type[SparseConn] | type[SparseConstrainedConn]:
    if isinstance(layer, DenseConn):
        default = DenseConn
        allowed = {DenseConn}
    else:
        default = (
            SparseConstrainedConn
            if isinstance(layer, SparseConstrainedConn)
            else SparseConn
        )
        allowed = {SparseConn, SparseConstrainedConn}

    if target_class is None:
        return default
    if target_class not in allowed:
        raise ValueError(
            "target_class is incompatible with source layer type. "
            f"Got target_class={target_class!r}, source={type(layer).__name__}."
        )
    return target_class


def _build_dense_conn(
    weight: np.ndarray,
    bias: torch.Tensor | None,
    enforce_dale: bool,
    *,
    device,
    dtype,
) -> DenseConn:
    weight_dtype = dtype or (bias.dtype if bias is not None else torch.float32)
    weight_tensor = torch.as_tensor(weight, dtype=weight_dtype, device=device)
    bias_tensor = None
    if bias is not None:
        bias_tensor = bias.detach().clone().to(device=device, dtype=weight_dtype)
    return DenseConn(
        in_features=int(weight.shape[0]),
        out_features=int(weight.shape[1]),
        weight=weight_tensor,
        bias=bias_tensor,
        enforce_dale=enforce_dale,
        device=device,
        dtype=weight_dtype,
    )


def _convert_dense_layer(
    layer: DenseConn,
    *,
    target_layout: TargetLayout,
    receptor_type_mode: ReceptorTypeMode | None,
    receptor_type_index: pd.DataFrame,
    neurons: pd.DataFrame | None,
    receptor_type_col: str,
    edge_receptor_assignment: pd.DataFrame | None,
    allow_weight_split: bool,
    edge_receptor_weight: pd.DataFrame | None,
    enforce_dale: bool,
    device,
    dtype,
) -> DenseConn:
    src_weight = _dense_weight_matrix(layer)
    n_receptor = len(receptor_type_index)

    if target_layout == "base":
        weight_target = _collapse_heter_dense_to_base(src_weight, n_receptor)
        bias_target = _convert_bias_heter_to_base(layer.bias, n_receptor)
        return _build_dense_conn(
            weight_target,
            bias_target,
            enforce_dale,
            device=device,
            dtype=dtype,
        )

    # target_layout == "heter"
    edge_df = _edge_df_from_dense_weight(src_weight)
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray

    if allow_weight_split:
        rows, cols, vals = _expand_edges_with_split(
            edge_df=edge_df,
            edge_receptor_weight=edge_receptor_weight,  # type: ignore[arg-type]
            n_receptor=n_receptor,
        )
    else:
        assignment_df = edge_receptor_assignment
        if (
            assignment_df is None
            and receptor_type_mode == "neuron"
            and neurons is not None
        ):
            assignment_df = _resolve_assignment_from_neurons(
                edge_df=edge_df,
                receptor_type_index=receptor_type_index,
                neurons=neurons,
                receptor_type_col=receptor_type_col,
            )
        if assignment_df is None and not edge_df.empty:
            raise ValueError(
                "base->heter conversion requires one-receptor-per-edge mapping. "
                "Provide edge_receptor_assignment, or provide "
                "neurons/receptor_type_col for neuron mode."
            )
        if assignment_df is None:
            rows = np.array([], dtype=np.int64)
            cols = np.array([], dtype=np.int64)
            vals = np.array([], dtype=src_weight.dtype)
        else:
            rows, cols, vals = _expand_edges_with_one_hot(
                edge_df=edge_df,
                assignment_df=assignment_df,
                n_receptor=n_receptor,
            )

    weight_target = np.zeros(
        (src_weight.shape[0], src_weight.shape[1] * n_receptor),
        dtype=src_weight.dtype,
    )
    if rows.size > 0:
        np.add.at(weight_target, (rows, cols), vals)

    bias_target = _convert_bias_base_to_heter(layer.bias, n_receptor)
    return _build_dense_conn(
        weight_target,
        bias_target,
        enforce_dale,
        device=device,
        dtype=dtype,
    )


def _convert_sparse_layer(
    layer: SparseConn | SparseConstrainedConn,
    *,
    target_layout: TargetLayout,
    receptor_type_mode: ReceptorTypeMode | None,
    receptor_type_index: pd.DataFrame,
    neurons: pd.DataFrame | None,
    receptor_type_col: str,
    edge_receptor_assignment: pd.DataFrame | None,
    allow_weight_split: bool,
    edge_receptor_weight: pd.DataFrame | None,
    group_policy: GroupPolicy,
    target_class: type[SparseConn] | type[SparseConstrainedConn],
    enforce_dale: bool,
    sparse_backend: SparseBackend,
    device,
    dtype,
) -> SparseConn | SparseConstrainedConn:
    n_receptor = len(receptor_type_index)
    src_conn = _layer_conn_coo(layer)
    persist_initial_weight = (
        layer.persist_initial_weight
        if isinstance(layer, SparseConstrainedConn)
        else True
    )
    src_constraint = (
        _layer_constraint_coo(layer)
        if isinstance(layer, SparseConstrainedConn)
        else None
    )

    if target_layout == "base":
        conn_target = _collapse_heter_sparse_to_base(src_conn, n_receptor=n_receptor)
        bias_target = _convert_bias_heter_to_base(layer.bias, n_receptor=n_receptor)
        if target_class is SparseConn:
            return SparseConn(
                conn=conn_target,
                bias=bias_target,
                enforce_dale=enforce_dale,
                sparse_backend=sparse_backend,
                device=device,
                dtype=dtype,
            )
        group_keys = np.arange(conn_target.nnz, dtype=np.int64)
        constraint_target = _build_constraint_from_groups(
            row=conn_target.row.astype(np.int64),
            col=conn_target.col.astype(np.int64),
            group_keys=group_keys,
            shape=conn_target.shape,
        )
        return SparseConstrainedConn(
            conn=conn_target,
            constraint=constraint_target,
            enforce_dale=enforce_dale,
            bias=bias_target,
            sparse_backend=sparse_backend,
            device=device,
            dtype=dtype,
            persist_initial_weight=persist_initial_weight,
        )

    edge_df = _edge_df_from_sparse_conn(src_conn)
    if allow_weight_split:
        rows, cols, vals = _expand_edges_with_split(
            edge_df=edge_df,
            edge_receptor_weight=edge_receptor_weight,  # type: ignore[arg-type]
            n_receptor=n_receptor,
        )
    else:
        assignment_df = edge_receptor_assignment
        if (
            assignment_df is None
            and receptor_type_mode == "neuron"
            and neurons is not None
        ):
            assignment_df = _resolve_assignment_from_neurons(
                edge_df=edge_df,
                receptor_type_index=receptor_type_index,
                neurons=neurons,
                receptor_type_col=receptor_type_col,
            )
        if assignment_df is None and not edge_df.empty:
            raise ValueError(
                "base->heter conversion requires one-receptor-per-edge mapping. "
                "Provide edge_receptor_assignment, or provide "
                "neurons/receptor_type_col for neuron mode."
            )
        if assignment_df is None:
            rows = np.array([], dtype=np.int64)
            cols = np.array([], dtype=np.int64)
            vals = np.array([], dtype=src_conn.data.dtype)
        else:
            rows, cols, vals = _expand_edges_with_one_hot(
                edge_df=edge_df,
                assignment_df=assignment_df,
                n_receptor=n_receptor,
            )

    conn_target = scipy.sparse.coo_array(
        (vals, (rows, cols)),
        shape=(src_conn.shape[0], src_conn.shape[1] * n_receptor),
    )
    conn_target.sum_duplicates()

    bias_target = _convert_bias_base_to_heter(layer.bias, n_receptor=n_receptor)
    if target_class is SparseConn:
        return SparseConn(
            conn=conn_target,
            bias=bias_target,
            enforce_dale=enforce_dale,
            sparse_backend=sparse_backend,
            device=device,
            dtype=dtype,
        )

    constraint_target = _expand_constraint_from_source(
        conn_base=src_conn,
        conn_heter=conn_target,
        source_constraint=src_constraint,
        group_policy=group_policy,
    )
    out = SparseConstrainedConn(
        conn=conn_target,
        constraint=constraint_target,
        enforce_dale=enforce_dale,
        bias=bias_target,
        sparse_backend=sparse_backend,
        device=device,
        dtype=dtype,
        persist_initial_weight=persist_initial_weight,
    )
    out.constraint_info = {"receptor_type_index": receptor_type_index}
    return out


def convert_connection_layer(
    layer: ConnectionLayer,
    *,
    target_layout: TargetLayout,
    receptor_type_mode: ReceptorTypeMode | None = None,
    receptor_type_index: pd.DataFrame,
    neurons: pd.DataFrame | None = None,
    receptor_type_col: str = "EI",
    edge_receptor_assignment: pd.DataFrame | None = None,
    allow_weight_split: bool = False,
    edge_receptor_weight: pd.DataFrame | None = None,
    group_policy: GroupPolicy = "independent",
    target_class: type[DenseConn]
    | type[SparseConn]
    | type[SparseConstrainedConn]
    | None = None,
    enforce_dale: bool | None = None,
    sparse_backend: SparseBackend | None = None,
    device=None,
    dtype=None,
) -> ConnectionLayer:
    """Convert a connection layer between base and heter layouts.

    Args:
        layer: Source connection layer.
        target_layout: Desired output layout, ``"base"`` or ``"heter"``.
        receptor_type_mode: Optional receptor semantics, ``"neuron"`` or
            ``"connection"``. Required only when inferring receptor assignment
            from ``neurons`` in ``base -> heter`` conversion.
        receptor_type_index: Receptor index table (must include
            ``receptor_index``).
        neurons: Optional neuron metadata used for assignment inference in
            neuron mode.
        receptor_type_col: Receptor-type column in ``neurons``.
        edge_receptor_assignment: One-receptor-per-edge mapping dataframe with
            ``pre_simple_id``, ``post_simple_id``, ``receptor_index``.
        allow_weight_split: Enable explicit split mode.
        edge_receptor_weight: Split dataframe with
            ``pre_simple_id``, ``post_simple_id``, ``receptor_index``,
            ``weight_coeff``.
        group_policy: Constraint expansion policy for constrained sparse output.
        target_class: Optional output class override. Cross-family conversion
            (dense<->sparse) is intentionally not supported.
        enforce_dale: Optional Dale-mode override for output layer.
        sparse_backend: Sparse backend override for sparse output.
        device: Optional output device.
        dtype: Optional output dtype.

    Returns:
        Converted connection layer instance.

    Raises:
        ValueError: If required metadata is missing or inconsistent.
    """
    target_layout = target_layout.lower()  # type: ignore[assignment]
    if target_layout not in get_args(TargetLayout):
        raise ValueError(
            "target_layout must be one of "
            f"{get_args(TargetLayout)}, got {target_layout!r}"
        )
    resolved_receptor_type_mode: ReceptorTypeMode | None = None
    if receptor_type_mode is not None:
        resolved_receptor_type_mode = _normalize_receptor_type_mode(receptor_type_mode)
    if target_layout == "heter":
        infer_from_neurons = (
            not allow_weight_split
            and edge_receptor_assignment is None
            and neurons is not None
        )
        if infer_from_neurons:
            if resolved_receptor_type_mode is None:
                raise ValueError(
                    "receptor_type_mode is required when inferring receptor "
                    "assignment from neurons."
                )
            if resolved_receptor_type_mode != "neuron":
                raise ValueError(
                    "receptor_type_mode must be 'neuron' when inferring receptor "
                    "assignment from neurons."
                )

    _require_columns(
        receptor_type_index,
        required_cols={"receptor_index"},
        name="receptor_type_index",
    )
    n_receptor = int(len(receptor_type_index))
    if n_receptor <= 0:
        raise ValueError("receptor_type_index must contain at least one receptor.")

    if edge_receptor_weight is not None and not allow_weight_split:
        raise ValueError(
            "edge_receptor_weight was provided but allow_weight_split=False. "
            "Set allow_weight_split=True to enable split mode."
        )
    if allow_weight_split and edge_receptor_weight is None:
        raise ValueError(
            "allow_weight_split=True requires edge_receptor_weight to be provided."
        )

    resolved_target_class = _validate_target_class(layer, target_class)

    if isinstance(layer, DenseConn):
        resolved_enforce_dale = (
            _dense_enforce_dale(layer) if enforce_dale is None else enforce_dale
        )
        return _convert_dense_layer(
            layer,
            target_layout=target_layout,
            receptor_type_mode=resolved_receptor_type_mode,
            receptor_type_index=receptor_type_index,
            neurons=neurons,
            receptor_type_col=receptor_type_col,
            edge_receptor_assignment=edge_receptor_assignment,
            allow_weight_split=allow_weight_split,
            edge_receptor_weight=edge_receptor_weight,
            enforce_dale=resolved_enforce_dale,
            device=device,
            dtype=dtype,
        )

    resolved_enforce_dale = layer.enforce_dale if enforce_dale is None else enforce_dale
    resolved_sparse_backend = (
        layer.sparse_backend if sparse_backend is None else sparse_backend
    )

    return _convert_sparse_layer(
        layer,
        target_layout=target_layout,
        receptor_type_mode=resolved_receptor_type_mode,
        receptor_type_index=receptor_type_index,
        neurons=neurons,
        receptor_type_col=receptor_type_col,
        edge_receptor_assignment=edge_receptor_assignment,
        allow_weight_split=allow_weight_split,
        edge_receptor_weight=edge_receptor_weight,
        group_policy=group_policy,
        target_class=resolved_target_class,  # type: ignore[arg-type]
        enforce_dale=resolved_enforce_dale,
        sparse_backend=resolved_sparse_backend,
        device=device,
        dtype=dtype,
    )


def convert_connection_layer_from_checkpoint(
    *,
    state_dict: dict[str, Any],
    source_class: SourceLayerClass,
    conn: scipy.sparse.sparray | None = None,
    constraint: scipy.sparse.sparray | None = None,
    in_features: int | None = None,
    out_features: int | None = None,
    mask: float | torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    enforce_dale: bool | None = None,
    sparse_backend: SparseBackend | None = None,
    target_layout: TargetLayout,
    receptor_type_mode: ReceptorTypeMode | None = None,
    receptor_type_index: pd.DataFrame,
    neurons: pd.DataFrame | None = None,
    receptor_type_col: str = "EI",
    edge_receptor_assignment: pd.DataFrame | None = None,
    allow_weight_split: bool = False,
    edge_receptor_weight: pd.DataFrame | None = None,
    group_policy: GroupPolicy = "independent",
    target_class: type[DenseConn]
    | type[SparseConn]
    | type[SparseConstrainedConn]
    | None = None,
    target_enforce_dale: bool | None = None,
    target_sparse_backend: SparseBackend | None = None,
    device=None,
    dtype=None,
) -> ConnectionLayer:
    """Convert a connection layer from checkpoint state plus topology metadata.

    For sparse sources, topology must be provided by the caller (`conn` and,
    for constrained sources, `constraint`).
    For constrained sparse sources, ``conn`` is authoritative when provided.
    If ``conn`` is omitted, both ``state_dict['initial_weight']`` and
    ``state_dict['indices']`` must be available.

    For dense sources, topology is inferred from ``state_dict['weight']`` unless
    ``in_features``/``out_features`` are explicitly supplied.
    """
    if source_class is DenseConn:
        if "weight" not in state_dict:
            raise ValueError(
                "DenseConn checkpoint must include 'weight' in state_dict."
            )
        weight_sd = state_dict["weight"]
        if not isinstance(weight_sd, torch.Tensor):
            raise ValueError("state_dict['weight'] must be a torch.Tensor.")
        out_from_sd, in_from_sd = weight_sd.shape
        in_features = in_from_sd if in_features is None else in_features
        out_features = out_from_sd if out_features is None else out_features
        if in_features != in_from_sd or out_features != out_from_sd:
            raise ValueError(
                "Provided in_features/out_features do not match checkpoint weight."
            )

        if enforce_dale is None:
            enforce_dale = "initial_sign" in state_dict

        if bias is None and "bias" in state_dict:
            bias = torch.zeros(
                out_features,
                dtype=state_dict["bias"].dtype,
                device=state_dict["bias"].device,
            )

        if mask is None and "mask" in state_dict:
            state_mask = state_dict["mask"]
            if not isinstance(state_mask, torch.Tensor):
                raise ValueError(
                    "state_dict['mask'] must be a torch.Tensor when present."
                )
            mask = torch.ones_like(state_mask)

        source_layer: ConnectionLayer = DenseConn(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            mask=mask,
            enforce_dale=enforce_dale,
            device=device,
            dtype=dtype,
        )
        source_layer.load_state_dict(state_dict)
    elif source_class is SparseConn:
        if conn is None:
            raise ValueError("conn must be provided when source_class='SparseConn'.")
        conn = _as_coo(conn)
        if enforce_dale is None:
            enforce_dale = "initial_sign" in state_dict
        if bias is None and "bias" in state_dict:
            bias = torch.zeros(
                conn.shape[1],
                dtype=state_dict["bias"].dtype,
                device=state_dict["bias"].device,
            )
        source_layer = SparseConn(
            conn=conn,
            bias=bias,
            enforce_dale=enforce_dale,
            sparse_backend=sparse_backend,
            device=device,
            dtype=dtype,
        )
        source_layer.load_state_dict(state_dict)
    elif source_class is SparseConstrainedConn:
        if constraint is None:
            raise ValueError(
                "constraint must be provided when source_class='SparseConstrainedConn'."
            )
        if conn is not None:
            # conn is authoritative for constrained initial weights when present.
            conn = _as_coo(conn)
        else:
            init_weight = state_dict.get("initial_weight")
            indices_sd = state_dict.get("indices")
            if init_weight is None or indices_sd is None:
                raise ValueError(
                    "SparseConstrainedConn conversion requires either conn, or both "
                    "state_dict['initial_weight'] and state_dict['indices']."
                )
            if not isinstance(init_weight, torch.Tensor):
                raise ValueError("state_dict['initial_weight'] must be a torch.Tensor.")
            if not isinstance(indices_sd, torch.Tensor):
                raise ValueError("state_dict['indices'] must be a torch.Tensor.")
            if indices_sd.ndim != 2 or indices_sd.shape[0] != 2:
                raise ValueError(
                    "state_dict['indices'] must have shape (2, nnz) for "
                    "SparseConstrainedConn."
                )
            if indices_sd.shape[1] != init_weight.numel():
                raise ValueError(
                    "state_dict indices/value mismatch for SparseConstrainedConn. "
                    f"Got indices nnz={indices_sd.shape[1]} vs "
                    f"initial_weight nnz={init_weight.numel()}."
                )

            pre = indices_sd[1].detach().cpu().numpy().astype(np.int64)
            post = indices_sd[0].detach().cpu().numpy().astype(np.int64)
            values = init_weight.detach().cpu().reshape(-1).numpy()
            shape = constraint.shape
            if pre.size > 0:
                if pre.max(initial=0) >= shape[0] or post.max(initial=0) >= shape[1]:
                    raise ValueError(
                        "state_dict indices are out of bounds for constraint shape "
                        f"{shape}."
                    )
            conn = scipy.sparse.coo_array((values, (pre, post)), shape=shape)
            conn.sum_duplicates()
        if enforce_dale is None:
            enforce_dale = True
        if bias is None and "bias" in state_dict:
            bias = torch.zeros(
                conn.shape[1],
                dtype=state_dict["bias"].dtype,
                device=state_dict["bias"].device,
            )
        source_layer = SparseConstrainedConn(
            conn=conn,
            constraint=constraint,
            enforce_dale=enforce_dale,
            bias=bias,
            sparse_backend=sparse_backend,
            device=device,
            dtype=dtype,
            persist_initial_weight=True,
        )
        state_for_load = dict(state_dict)
        # Topology and initial weights are reconstructed from conn/constraint.
        # Keep constructor indices to avoid stale sparse backend tensors.
        state_for_load.pop("initial_weight", None)
        state_for_load.pop("indices", None)
        source_layer.load_state_dict(state_for_load, strict=False)
    else:
        raise ValueError(
            "source_class must be one of "
            "(DenseConn, SparseConn, SparseConstrainedConn), got "
            f"{source_class!r}"
        )

    return convert_connection_layer(
        layer=source_layer,
        target_layout=target_layout,
        receptor_type_mode=receptor_type_mode,
        receptor_type_index=receptor_type_index,
        neurons=neurons,
        receptor_type_col=receptor_type_col,
        edge_receptor_assignment=edge_receptor_assignment,
        allow_weight_split=allow_weight_split,
        edge_receptor_weight=edge_receptor_weight,
        group_policy=group_policy,
        target_class=target_class,
        enforce_dale=target_enforce_dale,
        sparse_backend=target_sparse_backend,
        device=device,
        dtype=dtype,
    )
