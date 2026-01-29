from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse
import torch


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
    elif mode == "all_innervated":
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


def agg_conn(
    y,
    conn: pd.DataFrame,
    conn_weight: scipy.sparse.sparray | None = None,
    neurons: pd.DataFrame | None = None,
    mode: Literal["neuropil", "neuron"] = "neuron",
    neuron_type_column: str = "cell_type",
    agg: Literal["mean", "sum", "std"] = "mean",
):
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
    elif mode == "neuron":
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
