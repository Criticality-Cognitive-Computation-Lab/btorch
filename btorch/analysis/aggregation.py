from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse
import torch

from ..types import TensorLike


def agg_by_neuron(
    y,
    neurons: pd.DataFrame,
    agg: Literal["mean", "sum", "std"] = "mean",
    neuron_type_column: str = "cell_type",
    **kwargs,
) -> dict:
    """Aggregate data by neuron type."""
    agg_func = getattr(np, agg) if isinstance(y, np.ndarray) else getattr(torch, agg)
    ret = {}
    for neuron_type, group in neurons.groupby(
        neuron_type_column, dropna=True, **kwargs
    ):
        ret[neuron_type] = agg_func(y[..., group.simple_id.to_numpy()], -1)
    return ret


def agg_by_neuropil(
    y,
    neurons: pd.DataFrame | None = None,
    connections: pd.DataFrame | None = None,
    mode: Literal["top_innervated", "all_innervated"] = "all_innervated",
    agg: Literal["mean", "sum", "std"] = "mean",
    use_polars: bool = False,
):
    """Aggregate activations by neuropil under a validated aggregation mode."""
    agg_func = getattr(np, agg) if isinstance(y, np.ndarray) else getattr(torch, agg)
    if use_polars:
        try:
            import polars as pl
        except ImportError:
            use_polars = False

    if mode == "top_innervated":
        assert neurons is not None, "neurons must be provided for top_innervated mode"
        tmp = neurons[["group", "simple_id"]].copy()
        tmp = tmp[tmp["simple_id"] < y.shape[-1]]
        pre_ret: dict = {}
        post_ret: dict = {}

        if use_polars:
            tmp["pre"] = tmp["group"].str.split(".").str[0]
            tmp["post"] = tmp["group"].str.split(".").str[-1]
            ptbl = pl.from_pandas(tmp)

            pre_groups = ptbl.group_by("pre", maintain_order=True).agg(
                pl.col("simple_id")
            )
            for row in pre_groups.iter_rows(named=True):
                sid = np.asarray(row["simple_id"], dtype=int)
                pre_ret[row["pre"]] = agg_func(y[..., sid], -1)

            post_groups = ptbl.group_by("post", maintain_order=True).agg(
                pl.col("simple_id")
            )
            for row in post_groups.iter_rows(named=True):
                sid = np.asarray(row["simple_id"], dtype=int)
                post_ret[row["post"]] = agg_func(y[..., sid], -1)
        else:
            tmp["pre"] = tmp["group"].apply(lambda x: x.split(".")[0])
            tmp["post"] = tmp["group"].apply(lambda x: x.split(".")[-1])
            for pre, group in tmp.groupby("pre", dropna=True):
                pre_ret[pre] = agg_func(y[..., group.simple_id], -1)
            for post, group in tmp.groupby("post", dropna=True):
                post_ret[post] = agg_func(y[..., group.simple_id], -1)
        return pre_ret, post_ret
    if mode == "all_innervated":
        assert (
            connections is not None
        ), "connections must be provided for all_innervated mode"
        tmp = connections[["pre_simple_id", "post_simple_id", "neuropil"]]
        tmp = tmp[
            (tmp["pre_simple_id"] < y.shape[-1]) & (tmp["post_simple_id"] < y.shape[-1])
        ]
        pre_ret: dict = {}
        post_ret: dict = {}

        if use_polars:
            ptbl = pl.from_pandas(tmp)
            groups = ptbl.group_by("neuropil", maintain_order=True).agg(
                pl.col("pre_simple_id"), pl.col("post_simple_id")
            )
            for row in groups.iter_rows(named=True):
                neuropil = row["neuropil"]
                pre_ids = np.asarray(row["pre_simple_id"], dtype=int)
                post_ids = np.asarray(row["post_simple_id"], dtype=int)
                pre_ret[neuropil] = agg_func(y[..., pre_ids], -1)
                post_ret[neuropil] = agg_func(y[..., post_ids], -1)
        else:
            for neuropil, group in tmp.groupby("neuropil", dropna=True):
                pre_ret[neuropil] = agg_func(y[..., group.pre_simple_id.to_numpy()], -1)
                post_ret[neuropil] = agg_func(
                    y[..., group.post_simple_id.to_numpy()], -1
                )
        return pre_ret, post_ret

    raise ValueError(
        "Invalid `mode` for `agg_by_neuropil`. "
        "Expected one of: {'top_innervated', 'all_innervated'}."
    )


def agg_conn(
    y,
    conn: pd.DataFrame,
    conn_weight: scipy.sparse.sparray | None = None,
    neurons: pd.DataFrame | None = None,
    mode: Literal["neuropil", "neuron"] = "neuron",
    neuron_type_column: str = "cell_type",
    agg: Literal["mean", "sum", "std"] = "mean",
):
    """Aggregate connectivity weights by neuropil or neuron-type pairs."""
    if conn_weight is not None:
        conn_weight = conn_weight.tocoo()
        conn = conn.merge(
            pd.DataFrame(
                {
                    "pre_simple_id": conn_weight.row,
                    "post_simple_id": conn_weight.col,
                    "weight": conn_weight.data,
                }
            ),
            how="left",
            on=["pre_simple_id", "post_simple_id"],
        )
    if mode == "neuropil":
        return conn.groupby("neuropil")["weight"].agg(agg)
    if mode == "neuron":
        assert neurons is not None, "neurons must be provided for neuron mode"
        conn = conn.merge(
            neurons[["simple_id", neuron_type_column]].rename(
                columns={
                    "simple_id": "pre_simple_id",
                    neuron_type_column: f"pre_{neuron_type_column}",
                }
            ),
            how="left",
            on="pre_simple_id",
        )
        conn = conn.merge(
            neurons[["simple_id", neuron_type_column]].rename(
                columns={
                    "simple_id": "post_simple_id",
                    neuron_type_column: f"post_{neuron_type_column}",
                }
            ),
            how="left",
            on="post_simple_id",
        )
        return conn.groupby(
            [f"pre_{neuron_type_column}", f"post_{neuron_type_column}"]
        )["weight"].agg(agg)

    raise ValueError(
        "Invalid `mode` for `agg_conn`. Expected one of: {'neuropil', 'neuron'}."
    )


def build_group_frame(
    values: TensorLike,
    neurons_df: pd.DataFrame,
    group_by: str,
    *,
    simple_id_col: str = "simple_id",
    value_name: str = "value",
    dropna: bool = True,
) -> pd.DataFrame:
    """Convert neuron-aligned values to a tidy frame for grouped analyses.

    Args:
        values: Array/tensor shaped `[N]` or `[..., N]` where the last axis is
            neuron. All leading dimensions are flattened into independent
            samples (e.g., trials, conditions, or time points).
        neurons_df: DataFrame containing at least `simple_id_col` and `group_by`.
        group_by: Column in `neurons_df` used as grouping key.
        simple_id_col: Column mapping rows in `neurons_df` to neuron index in
            `values`.
        value_name: Name for the output value column.
        dropna: Drop missing values in group/value columns when `True`.
    """
    y = _to_numpy(values)
    if y.ndim < 1:
        raise ValueError("`values` must have at least one dimension.")

    if simple_id_col not in neurons_df.columns:
        raise ValueError(f"Missing `{simple_id_col}` in `neurons_df`.")
    if group_by not in neurons_df.columns:
        raise ValueError(f"Missing `{group_by}` in `neurons_df`.")

    metadata = neurons_df.loc[:, [simple_id_col, group_by]].copy()
    if dropna:
        metadata = metadata.dropna(subset=[group_by])
    if metadata.empty:
        raise ValueError("No neuron metadata available after filtering.")

    if metadata[simple_id_col].duplicated().any():
        raise ValueError(f"`{simple_id_col}` must be unique in `neurons_df`.")

    try:
        simple_ids = pd.to_numeric(metadata[simple_id_col], errors="raise").to_numpy(
            dtype=np.int64
        )
    except Exception as exc:
        raise ValueError(f"`{simple_id_col}` must be numeric.") from exc

    n_neurons = y.shape[-1]
    out_of_range = (simple_ids < 0) | (simple_ids >= n_neurons)
    if np.any(out_of_range):
        bad_ids = simple_ids[out_of_range]
        raise ValueError(
            f"Found `{simple_id_col}` outside [0, {n_neurons - 1}]: {bad_ids.tolist()}"
        )

    selected = y[..., simple_ids]
    n_samples = int(np.prod(selected.shape[:-1], dtype=np.int64))
    n_samples = max(1, n_samples)

    flattened = selected.reshape(n_samples, len(simple_ids))
    group_labels = metadata[group_by].to_numpy()

    frame = pd.DataFrame(
        {
            group_by: np.repeat(group_labels, n_samples),
            value_name: flattened.T.reshape(-1),
        }
    )

    if dropna:
        frame = frame.dropna(subset=[value_name])
    if frame.empty:
        raise ValueError("No values available after filtering.")

    return frame


def group_values(
    values: TensorLike,
    neurons_df: pd.DataFrame,
    group_by: str,
    *,
    simple_id_col: str = "simple_id",
    value_name: str = "value",
    group_order: Sequence | None = None,
    dropna: bool = True,
) -> dict[object, np.ndarray]:
    """Return grouped value arrays, keyed by group label in plotting order."""
    frame = build_group_frame(
        values,
        neurons_df,
        group_by,
        simple_id_col=simple_id_col,
        value_name=value_name,
        dropna=dropna,
    )
    order = _resolve_group_order(frame, group_by, group_order)
    return {
        group: frame.loc[frame[group_by] == group, value_name].to_numpy(dtype=float)
        for group in order
    }


def group_summary(
    values: TensorLike,
    neurons_df: pd.DataFrame,
    group_by: str,
    *,
    simple_id_col: str = "simple_id",
    value_name: str = "value",
    group_order: Sequence | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Compute per-group summary statistics from neuron-aligned values."""
    grouped = group_values(
        values,
        neurons_df,
        group_by,
        simple_id_col=simple_id_col,
        value_name=value_name,
        group_order=group_order,
        dropna=dropna,
    )

    rows = []
    for group, vals in grouped.items():
        rows.append(
            {
                group_by: group,
                "n": int(vals.size),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "q25": float(np.quantile(vals, 0.25)),
                "median": float(np.median(vals)),
                "q75": float(np.quantile(vals, 0.75)),
                "max": float(np.max(vals)),
            }
        )

    return pd.DataFrame(rows)


def group_ecdf(
    values: TensorLike,
    neurons_df: pd.DataFrame,
    group_by: str,
    *,
    simple_id_col: str = "simple_id",
    value_name: str = "value",
    group_order: Sequence | None = None,
    dropna: bool = True,
) -> dict[object, pd.DataFrame]:
    """Compute grouped ECDF points ready for plotting or analysis."""
    grouped = group_values(
        values,
        neurons_df,
        group_by,
        simple_id_col=simple_id_col,
        value_name=value_name,
        group_order=group_order,
        dropna=dropna,
    )

    ret: dict[object, pd.DataFrame] = {}
    for group, vals in grouped.items():
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1, dtype=float) / len(x)
        ret[group] = pd.DataFrame({value_name: x, "ecdf": y})
    return ret


def _resolve_group_order(
    frame: pd.DataFrame,
    group_by: str,
    group_order: Sequence | None,
) -> list[object]:
    if group_order is None:
        return list(pd.unique(frame[group_by]))

    requested = list(group_order)
    available = set(frame[group_by].tolist())
    missing = [group for group in requested if group not in available]
    if missing:
        raise ValueError(f"`group_order` contains unknown groups: {missing}")
    return requested


def _to_numpy(values: TensorLike) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        return values
    raise TypeError("`values` must be a numpy array or torch tensor.")
