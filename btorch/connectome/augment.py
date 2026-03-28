"""Data augmentation and sampling utilities for connectome data.

This module provides functions for manipulating neuron and connection
DataFrames, including:

- Slicing neurons and their connections by ID subsets
- Sampling strategies for representative neuron subsets
- Downsampling connections while preserving network structure
- Dropping neurons/connections and computing degrees
- E/I connection matrix construction

Expected DataFrame schemas:
    neurons DataFrame:
        - root_id: Unique neuron identifier (64-bit int)
        - simple_id: Linear 0..N-1 index
        - flow, super_class, class, nt_type: Classification columns
        - area_nm: Surface area for capacitance computation

    connections DataFrame:
        - pre_root_id, post_root_id: Root IDs
        - pre_simple_id, post_simple_id: Simple IDs
        - syn_count: Synapse count

Side effects:
    Functions may modify DataFrame indices, add new columns (e.g.,
    original_simple_id for backup), or remap IDs to local ranges.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse

from . import simple_id_to_root_id


def slice_neurons_connections(
    simple_id: int | list[int] | np.ndarray | None = None,
    root_id: int | list[int] | np.ndarray | None = None,
    neurons: pd.DataFrame | None = None,
    connections: pd.DataFrame | None = None,
    connection_mode: Literal["both", "pre", "post", "any"] = "both",
    keep_original_ids: bool = True,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, np.ndarray]:
    """Slices neuron metadata and connections while preserving input sequence
    order.

    Args:
        simple_id: Ordered linear IDs for slicing.
        root_id: Ordered non-linear IDs (requires `neurons` for resolution).
        neurons: Metadata containing 'simple_id' and 'root_id'.
        connections: Edge list containing 'pre_simple_id' and 'post_simple_id'.
        connection_mode: Filtering logic for edges relative to selected neurons.
        keep_original_ids: Back up global IDs before remapping to local 0..N-1 range.

    Returns:
        tuple: (ordered_neurons, filtered_connections, original_indices)
    """
    if (simple_id is None) == (root_id is None):
        raise ValueError("Provide exactly one of 'simple_id' or 'root_id'.")

    # 1. Resolve ordered target indices
    if root_id is not None:
        if neurons is None:
            raise ValueError("`neurons` DataFrame required to resolve `root_id`.")
        targets = np.atleast_1d(root_id)
        # Reorder neurons to match root_id input sequence
        sliced_neurons = neurons.set_index("root_id").loc[targets].reset_index()
        indices = sliced_neurons["simple_id"].values
    else:
        indices = np.atleast_1d(simple_id)
        sliced_neurons = (
            neurons.set_index("simple_id").loc[indices].reset_index()
            if neurons is not None
            else None
        )

    # 2. Map original IDs (Only on the sliced subset to save memory)
    if keep_original_ids:
        if (
            sliced_neurons is not None
            and "original_simple_id" not in sliced_neurons.columns
        ):
            sliced_neurons["original_simple_id"] = indices

    # 3. Create mapping for 0..N-1 reassignment
    # pd.Series is more memory efficient than dict for large ID sets
    mapper = pd.Series(np.arange(len(indices)), index=indices)

    if sliced_neurons is not None:
        sliced_neurons["simple_id"] = mapper.values

    # 4. Filter and remap connections
    sliced_conn = None
    if connections is not None:
        # Masking on global dataframe avoids premature copies
        pre_in = connections["pre_simple_id"].isin(indices)
        post_in = connections["post_simple_id"].isin(indices)

        mask = {
            "both": pre_in & post_in,
            "pre": pre_in,
            "post": post_in,
            "any": pre_in | post_in,
        }[connection_mode]

        sliced_conn = connections[mask].copy()

        if keep_original_ids:
            if "original_pre_simple_id" not in sliced_conn.columns:
                sliced_conn["original_pre_simple_id"] = sliced_conn["pre_simple_id"]
                sliced_conn["original_post_simple_id"] = sliced_conn["post_simple_id"]

        # Update IDs in-place using masks to prevent NaN-induced float conversion
        for col in ["pre_simple_id", "post_simple_id"]:
            m = sliced_conn[col].isin(indices)
            sliced_conn.loc[m, col] = sliced_conn.loc[m, col].map(mapper)

    return sliced_neurons, sliced_conn, indices


def sample_by_column_expand_none(
    df: pd.DataFrame,
    group_cols: list,
    k: int = 5,
    j=10,
    random_state: int = 0,
    product_sample: bool = False,
) -> pd.DataFrame:
    """Sample rows from DataFrame based on complete groupings and null columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    group_cols : list
        List of column names to group on.

    k : int
        Number of samples to draw per fully-specified group
        (non-null in all group_cols).

    j : int or dict
        If int: number of samples from all partially-specified rows (non-product mode),
                or per partial group (product mode).
        If dict: mapping of column name to number of samples if that column is null.

    random_state : int
        Random seed.

    product_sample : bool
        If True, group partially-specified rows by their non-null group_cols and sample
        `j` per group (or per-column null if j is a dict). If False, sample globally.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of full group and partial samples.
    """
    # --- Fully-specified groups ---
    mask_full = df[group_cols].notnull().all(axis=1)
    df_full = df[mask_full]
    df_partial = df[~mask_full]

    grouped = df_full.groupby(group_cols, dropna=False, sort=False)
    sampled_full = []
    for _, group in grouped:
        sampled = group.sample(min(len(group), k), random_state=random_state)
        sampled_full.append(sampled)
    sampled_full = (
        pd.concat(sampled_full, ignore_index=True)
        if sampled_full
        else pd.DataFrame(columns=df.columns)
    )

    # --- Partial sampling ---
    sampled_partial = []

    if isinstance(j, dict):
        for col, n in j.items():
            null_rows = df_partial[df_partial[col].isnull()]
            if len(null_rows) == 0:
                continue
            if product_sample:
                # Group by other non-null columns (excluding the null one)
                other_cols = [c for c in group_cols if c != col]
                grouped = null_rows[
                    null_rows[other_cols].notnull().all(axis=1)
                ].groupby(other_cols, dropna=False, sort=False)
                for _, group in grouped:
                    sampled = group.sample(
                        min(len(group), n), random_state=random_state
                    )
                    sampled_partial.append(sampled)
            else:
                sampled = null_rows.sample(
                    min(len(null_rows), n), random_state=random_state
                )
                sampled_partial.append(sampled)
    else:
        if product_sample:
            partial_group_keys = df_partial[group_cols].drop_duplicates()
            for _, row in partial_group_keys.iterrows():
                mask = pd.Series(True, index=df_partial.index)
                for col in group_cols:
                    if pd.notnull(row[col]):
                        mask &= df_partial[col] == row[col]
                subset = df_partial[mask]
                if len(subset) > 0:
                    sampled = subset.sample(
                        min(len(subset), j), random_state=random_state
                    )
                    sampled_partial.append(sampled)
        else:
            sampled_partial = [
                df_partial.sample(min(len(df_partial), j), random_state=random_state)
            ]

    sampled_partial = (
        pd.concat(sampled_partial, ignore_index=True)
        if sampled_partial
        else pd.DataFrame(columns=df.columns)
    )

    return pd.concat([sampled_full, sampled_partial], ignore_index=True)


def sample_neuron_representative(
    neurons,
    k: int = 1,
    j: int | dict = 2,
    random_state: int | None = None,
    product_sample: bool = True,
):
    return sample_by_column_expand_none(
        neurons,
        ["flow", "super_class", "class", "nt_type"],
        k=k,
        j=j,
        random_state=random_state,
        product_sample=product_sample,
    )


def mask_neurons_in_conn_mat(
    conn_mat: scipy.sparse.sparray,
    ids: int | Sequence[int],
    id_type: Literal["root_id", "simple_id"],
    neurons: pd.DataFrame | None = None,
) -> scipy.sparse.sparray:
    if isinstance(ids, int):
        ids = [ids]
    if id_type == "root_id":
        assert neurons is not None
        to_root_id = simple_id_to_root_id(neurons, reverse=True)
        ids = [to_root_id[i] for i in ids]
    conn_mat_lil = conn_mat.tolil()

    conn_mat_lil[ids, :] = 0
    conn_mat_lil[:, ids] = 0

    if isinstance(conn_mat, scipy.sparse.csr_array):
        return conn_mat_lil.tocsr()
    elif isinstance(conn_mat, scipy.sparse.coo_array):
        return conn_mat_lil.tocsr().tocoo()
    else:
        raise ValueError()


def drop_neurons(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    root_id: Sequence[int] | pd.DataFrame | pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    neurons = neurons[~neurons.root_id.isin(root_id)]
    connections = connections[
        ~(
            connections["pre_root_id"].isin(root_id)
            | connections["post_root_id"].isin(root_id)
        )
    ]
    return neurons, connections


def drop_no_conn(neurons: pd.DataFrame, connections: pd.DataFrame) -> pd.DataFrame:
    combined_series = pd.concat(
        [connections["pre_root_id"], connections["post_root_id"]]
    )
    unique_series = combined_series.drop_duplicates()
    return neurons[neurons.root_id.isin(unique_series)]


def find_neuron_inout_degree(neurons: pd.DataFrame, connections: pd.DataFrame):
    degrees = neurons["root_id"]
    degrees = pd.merge(
        degrees,
        connections.groupby("pre_root_id")["syn_count"].sum(),
        how="left",
        left_on="root_id",
        right_on="pre_root_id",
    ).rename(columns={"syn_count": "out_degree"})
    degrees["out_degree"] = degrees["out_degree"].fillna(0)
    degrees = degrees.merge(
        connections.groupby("post_root_id")["syn_count"].sum(),
        how="left",
        left_on="root_id",
        right_on="post_root_id",
    ).rename(columns={"syn_count": "in_degree"})
    degrees["in_degree"] = degrees["in_degree"].fillna(0)
    return degrees


def downsample_neurons(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    samples: int,
    random_state=42,
    method: Literal[
        "random_sample_connections", "degree"
    ] = "random_sample_connections",
    **params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method == "random_sample_connections":
        connections = connections.sample(samples, random_state=random_state)
        remaining_ids = pd.concat(
            [connections["pre_root_id"], connections["post_root_id"]]
        ).drop_duplicates()
        neurons = neurons[neurons["root_id"].isin(remaining_ids)]
    elif method == "degree":
        degrees = find_neuron_inout_degree(neurons, connections)
        degrees["degree"] = degrees["in_degree"] + degrees["out_degree"]
        lowest_first = params.get("smallest_first", True)
        degrees = degrees.sort_values(
            by=params.get("by", "degree"), ascending=not lowest_first
        )
        degrees = degrees.iloc[:samples]
        neurons = neurons[neurons["root_id"].isin(degrees.root_id)]
        connections = connections[
            connections["pre_root_id"].isin(degrees.root_id)
            | connections["post_root_id"].isin(degrees.root_id)
        ]
    else:
        raise ValueError("Unsupported downsample method")

    return neurons, connections


def make_ei_conn_mat(
    conn_mats: dict[str, scipy.sparse.sparray], rebalance_ratio: float = 0.5
):
    # return 2 * (
    #     rebalance_ratio * conn_mats["E"] - (1 - rebalance_ratio) * conn_mats["I"]
    # )
    return conn_mats["E"] - rebalance_ratio * conn_mats["I"]


def empirical_membrane_capacitance(neurons: pd.DataFrame) -> pd.Series:
    area_cm = neurons.area_nm / 10**12
    area_cm = area_cm.fillna(area_cm.mean(skipna=True))
    return area_cm.to_numpy() * 800
