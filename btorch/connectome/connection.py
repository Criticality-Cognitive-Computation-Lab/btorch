import itertools
import warnings
from collections import OrderedDict
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.spatial

from ..utils.pandas_utils import groupby_to_dict
from . import simple_id_to_root_id


def make_sparse_mat(
    connections: pd.DataFrame, shape, field="syn_count"
) -> scipy.sparse.sparray:
    # Important: Connections can have duplicated pre and post neuron pairs
    #            if they innervate between different neuropils.
    tmp_connections = connections[[field, "pre_simple_id", "post_simple_id"]]
    tmp_connections = tmp_connections.groupby(
        ["pre_simple_id", "post_simple_id"], as_index=False
    ).agg({field: "sum"})

    # Note: this .T has nothing to do with transposing the full weight matrix.
    #       it only makes (N, 2) to (2, N) where the 2 rows correspond to pre and post
    pre, post = (
        tmp_connections[["pre_simple_id", "post_simple_id"]].to_numpy(dtype=int).T
    )

    ret = scipy.sparse.coo_array(
        (
            tmp_connections[[field]].to_numpy().flatten(),
            (pre, post),
        ),
        shape=shape,
    )

    return ret


def neuron_subset_to_conn_mat(
    subset: Sequence[int] | pd.Series | pd.DataFrame,
    id_type: Literal["root_id", "simple_id"],
    size: int,
    neurons: Optional[pd.DataFrame] = None,
    remove_nan: bool = False,
    return_mode: Literal["sparray", "scatter"] = "scatter",
) -> scipy.sparse.sparray | np.ndarray:
    if isinstance(subset, (Sequence, pd.Series)):
        df = pd.DataFrame(subset, columns=[id_type])
        # convert to root_id
        if id_type == "root_id":
            df["simple_id"] = df.root_id.map(
                simple_id_to_root_id(neurons, reverse=True).get
            )
    elif isinstance(subset, pd.DataFrame):
        assert hasattr(subset, "root_id")
        if not hasattr(subset, "simple_id"):
            df = subset.copy()
            df["simple_id"] = df.root_id.map(
                simple_id_to_root_id(neurons, reverse=True).get
            )
        else:
            df = subset
    else:
        raise ValueError(f"Not a valid neuron subset, type {type(subset)}")

    unmapped = df[df["simple_id"].isna()]
    if not unmapped.empty and remove_nan:
        warnings.warn(
            f"Found unknown root_id, removing {unmapped['root_id'].to_list()}.\n"
            "Either the current Flywire version doesn't contain these neurons,\n"
            "or they don't have neurotransmitter prediction",
        )
        df = df.dropna(subset="simple_id")
    else:
        assert unmapped.empty, (
            f"Found unknown root_id, {unmapped['root_id'].to_list()}.\n"
            "Either the current Flywire version doesn't contain these neurons,\n"
            "or they don't have neurotransmitter prediction"
        )

    simple_id_subset = df["simple_id"].to_numpy().flatten()
    if return_mode == "scatter":
        return simple_id_subset

    input_size = simple_id_subset.size
    ret = scipy.sparse.coo_array(
        (np.ones_like(simple_id_subset), (np.arange(input_size), simple_id_subset)),
        shape=(input_size, size),
    )
    return ret


def make_constraint_by_neuron_type(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    nan_in_same_group: bool = True,
) -> scipy.sparse.sparray:
    """Create a sparse constraint matrix where each non-zero entry represents a
    unique (pre_cell_type, post_cell_type) pair. This can be used to group
    connections based on cell types for constrained learning.

    Args:
        neurons (pd.DataFrame): Must contain 'root_id', 'cell_type'.
        connections (pd.DataFrame): Must contain 'pre_root_id', 'post_root_id',
            'pre_simple_id', and 'post_simple_id'.
        nan_in_same_group (bool): If True, missing cell types are grouped together.
            If False, missing values are assigned unique dummy types.
        format (Literal["coo", "csr"]): Return format for the sparse matrix.

    Returns:
        scipy.sparse.sparray: Sparse array with shape (num_neurons, num_neurons),
            where each non-zero value is a group ID for that connection.
    """

    tmp_neurons = neurons.copy()

    if not nan_in_same_group:
        none_mask = tmp_neurons["cell_type"].isna()
        # Assign unique dummy types for each missing cell type
        tmp_neurons.loc[none_mask, "cell_type"] = [
            f"__none_{i}__" for i in range(none_mask.sum())
        ]

    # Build mapping from root_id to resolved cell_type
    root_id_to_cell_type = dict(
        tmp_neurons[["root_id", "cell_type"]].itertuples(index=False, name=None)
    )

    # Drop duplicate synapses to avoid repeated entries in constraint matrix
    tmp_conns = connections.drop_duplicates(
        subset=["pre_simple_id", "post_simple_id"]
    ).copy()

    # Map root IDs to cell types
    tmp_conns["pre_cell_type"] = tmp_conns["pre_root_id"].map(root_id_to_cell_type)
    tmp_conns["post_cell_type"] = tmp_conns["post_root_id"].map(root_id_to_cell_type)

    # Create group ID for each (pre, post) cell type pair, starting from 1
    tmp_conns["cell_type_pair_id"] = (
        pd.factorize(tmp_conns[["pre_cell_type", "post_cell_type"]].agg(tuple, axis=1))[
            0
        ]
        + 1
    )

    # Construct sparse array with group IDs as values
    constraint_group = scipy.sparse.coo_array(
        (
            tmp_conns["cell_type_pair_id"].to_numpy(dtype=int).flatten(),
            tmp_conns[["pre_simple_id", "post_simple_id"]].to_numpy(dtype=int).T,
        ),
        shape=(neurons.shape[0], neurons.shape[0]),
    )

    return constraint_group


# prompt:
# 1. forgot :-b
# 2. support the include_self flag for both num and radius modes.
# 3. avoid using for loop, try numpy
def make_spatial_localised_conn(
    neurons: pd.DataFrame,
    mode: Literal["num", "radius"] = "num",
    num: int = 5,
    radius: float = 5.0,
    include_self: bool = True,
) -> scipy.sparse.sparray:
    """Constructs a spatially localized connection matrix."""
    positions = neurons[["x", "y", "z"]].values
    n_neurons = len(positions)
    tree = scipy.spatial.cKDTree(positions)

    if mode == "num":
        num = num + 1 if include_self else num  # ensure enough neighbors to drop self
        _, indices = tree.query(positions, k=num)

        if num == 1:
            # tree.query returns shape (n,) instead of (n, num) when num=1
            indices = indices[:, np.newaxis]

        if not include_self:
            mask = indices != np.arange(n_neurons)[:, None]
            indices = indices[mask].reshape(n_neurons, num - 1)
        else:
            indices = indices[:, :num]

        row_indices = np.repeat(np.arange(n_neurons), indices.shape[1])
        col_indices = indices.flatten()

    elif mode == "radius":
        _, indices = tree.query_radius(
            positions, r=int(radius), return_distance=False, sort_results=False
        )
        # Flatten indices
        row_indices = np.repeat(np.arange(n_neurons), [len(ids) for ids in indices])
        col_indices = np.concatenate(indices)

        if not include_self:
            mask = row_indices != col_indices
            row_indices = row_indices[mask]
            col_indices = col_indices[mask]

    else:
        raise ValueError("mode must be 'num' or 'radius'")

    data = np.ones_like(row_indices, dtype=np.float32)
    conn_matrix = scipy.sparse.coo_array(
        (data, (row_indices, col_indices)), shape=(n_neurons, n_neurons)
    )

    return conn_matrix


def make_hetersynapse_conn(
    neurons: pd.DataFrame,
    connections: scipy.sparse.sparray | pd.DataFrame,
    receptor_type_col="EI",
    receptor_type_mode: Literal["neuron", "connection"] = "neuron",
    return_dict: bool = False,
    dropna: Literal["error", "filter", "unknown"] = "error",
    ignore_post_type: bool = False,
) -> tuple[scipy.sparse.sparray, pd.DataFrame] | tuple[OrderedDict, pd.DataFrame]:
    """Transforms a connectivity matrix to represent heterosynaptic connections
    based on receptor types.

    This function can handle two modes:
    1. **'neuron' mode**: Receptor types are properties of the pre- and post-synaptic
       neurons. The output matrix will have `num_neurons` rows and `num_neurons
       * num_receptor_pairs` columns, where each block of `n_receptor_type` columns
       corresponds to all receptors of a specific neuron. (e.g., 'E' to 'I').
    2. **'connection' mode**: Receptor types are properties of the connections
       themselves (possible cotransmission). The output matrix will have
       `num_neurons` rows and `num_neurons * num_receptor_types` columns, where
       each block of `n_receptor_type` columns corresponds to all receptors of a
       single neuron.

    Args:
        neurons: DataFrame with neuron information. It must contain the
        `receptor_type_col` and a column that can be used as a unique identifier
        (e.g., 'simple_id').
        connections: A `scipy.sparse.sparray` or `pandas.DataFrame` representing
        the network connections. If a DataFrame, it should have 'syn_count',
        'pre_simple_id' and 'post_simple_id' columns. If a sparray, it should be
        in (pre_neuron, post_neuron)
        receptor_type_col: The column name in the `neurons` or `connections`
        DataFrame that specifies the receptor type.
        receptor_type_mode: Specifies whether receptor types are associated with
        'neuron' or 'connection'.
        return_dict: If True, returns OrderedDict mapping receptor type pairs to
        sparse matrices instead of a single stacked matrix.
        dropna: How to handle NaN receptor types. Options:
            - 'error' (default): Raise ValueError if NaN found
            - 'filter': Remove connections involving NaN neurons
              (preserves neuron count)
            - 'unknown': Treat NaN as a separate receptor type category
        ignore_post_type: If True, clusters connections only by pre-synaptic
            receptor type, ignoring post-synaptic receptor type. Resulting matrix
            will have fewer columns. Only valid for 'neuron' mode.

    Returns:
        A tuple containing:
            - The transformed sparse array (`scipy.sparse.sparray`) or OrderedDict
              of sparse arrays (if return_dict=True) in (pre_neuron,
              post_neuron * receptor_type).
            - A DataFrame mapping the new rows to receptor types. For 'neuron'
              mode, this will include 'pre_receptor_type' and
              'post_receptor_type'. For 'connection' mode, it will just have
              'receptor_type'.

    Raises:
        ValueError: If `connections` is not a DataFrame or a sparse array, or if
                    the `receptor_type_mode` is 'connection' but the input is a
                    sparse array, or if NaN values found and dropna='error'.

    Note:
        The neuron count is always preserved (based on simple_id indexing).
        When dropna='filter', only connections are removed, not neurons.
        When dropna='unknown', NaN becomes a valid receptor type.
    """

    if isinstance(connections, pd.DataFrame):
        if receptor_type_mode == "neuron":
            connections = make_sparse_mat(connections, (len(neurons), len(neurons)))
            return make_hetersynapse_conn(
                neurons,
                connections,
                receptor_type_col,
                receptor_type_mode,
                return_dict=return_dict,
                dropna=dropna,
                ignore_post_type=ignore_post_type,
            )

        # create sparse mat for each connection's receptor_type
        # major difference from neuron receptor_type is that we don't need
        # itertools.product since receptor_type is a property of connection itself
        shape = (len(neurons), len(neurons))
        connections_groups = groupby_to_dict(
            connections, by=receptor_type_col, sort=True
        )
        conn_receptor_type_groups = OrderedDict(
            {k: make_sparse_mat(v, shape) for k, v in connections_groups.items()}
        )
        receptor_type_index = list(enumerate(conn_receptor_type_groups.keys()))
        receptor_type_index = pd.DataFrame(
            receptor_type_index, columns=["receptor_index", "receptor_type"]
        )
        if return_dict:
            return conn_receptor_type_groups, receptor_type_index
        receptor_type_index_groups = list(enumerate(conn_receptor_type_groups.values()))

        n_receptor_type = len(conn_receptor_type_groups)
        new_shape = (shape[0], shape[1] * n_receptor_type)

        new_row = []
        new_col = []
        new_val = []
        for i, conn_mat in receptor_type_index_groups:
            new_row.append(conn_mat.row)
            new_col.append(conn_mat.col * n_receptor_type + i)
            new_val.append(conn_mat.data)

        return (
            scipy.sparse.coo_array(
                (
                    np.concatenate(new_val),
                    (np.concatenate(new_row), np.concatenate(new_col)),
                ),
                shape=new_shape,
            ),
            receptor_type_index,
        )

    elif isinstance(connections, scipy.sparse.sparray):
        assert receptor_type_mode == "neuron"

        if ignore_post_type and receptor_type_mode != "neuron":
            raise ValueError("ignore_post_type only valid for 'neuron' mode")

        # Validate no NaN in connection data
        connections = connections.tocoo()
        if np.isnan(connections.data).any():
            raise ValueError("NaN values detected in connection matrix data")

        conn_mat_df = pd.DataFrame(
            {"pre": connections.row, "post": connections.col, "data": connections.data}
        )
        shape = connections.shape

        # Check for NaN receptor types in neurons
        nan_mask = neurons[receptor_type_col].isna()
        if nan_mask.any():
            if dropna == "error":
                nan_neurons = neurons[nan_mask]["simple_id"].tolist()
                raise ValueError(
                    f"NaN receptor types found for neurons: {nan_neurons}. "
                    "Set dropna='filter' to remove connections involving them, "
                    "or dropna='unknown' to treat NaN as a separate receptor type."
                )
            elif dropna == "filter":
                # Filter out connections involving NaN neurons
                nan_neuron_ids = set(neurons[nan_mask]["simple_id"])
                n_before = len(conn_mat_df)
                conn_mat_df = conn_mat_df[
                    ~conn_mat_df["pre"].isin(nan_neuron_ids)
                    & ~conn_mat_df["post"].isin(nan_neuron_ids)
                ]
                n_filtered = n_before - len(conn_mat_df)
                if n_filtered > 0:
                    warnings.warn(
                        f"Filtered out {n_filtered} connections involving "
                        f"{len(nan_neuron_ids)} neurons with NaN receptor types"
                    )
            elif dropna == "unknown":
                # Replace NaN with 'unknown' string to make it a valid category
                neurons = neurons.copy()
                neurons.loc[nan_mask, receptor_type_col] = "unknown"
                warnings.warn(
                    f"Treating {nan_mask.sum()} neurons with NaN receptor types "
                    "as 'unknown' receptor type"
                )
            else:
                raise ValueError(
                    f"dropna must be 'error', 'filter', or 'unknown', got {dropna}"
                )

        # Group neurons by receptor type
        receptor_type_groups = OrderedDict(
            groupby_to_dict(
                neurons,
                "simple_id",
                by=receptor_type_col,
                sort=True,
                dropna=(dropna == "filter"),  # Only drop NaN groups if filtering
            )
        )

        if ignore_post_type:
            # Only iterate over pre-synaptic receptor types
            # We treat all post-synaptic neurons as targets regardless of their type
            receptor_type_index_groups = list(enumerate(receptor_type_groups.items()))
            receptor_type_index = [
                (i, pre_k) for i, (pre_k, _) in receptor_type_index_groups
            ]
        else:
            receptor_type_groups_product = list(
                itertools.product(receptor_type_groups.items(), repeat=2)
            )
            receptor_type_index_groups = list(enumerate(receptor_type_groups_product))
            receptor_type_index = [
                (i, pre_k, post_k)
                for i, ((pre_k, _), (post_k, _)) in receptor_type_index_groups
            ]

        n_receptor_type = len(receptor_type_index_groups)
        new_shape = (shape[0], shape[1] * n_receptor_type)

        if return_dict:
            # Return dict mapping keys -> sparse matrix
            result_dict = OrderedDict()
            if ignore_post_type:
                for i, (pre_type, pre_group) in receptor_type_index_groups:
                    conn = conn_mat_df[conn_mat_df.pre.isin(pre_group)]
                    if len(conn) > 0:
                        result_dict[pre_type] = scipy.sparse.coo_array(
                            (conn.data.values, (conn.pre.values, conn.post.values)),
                            shape=shape,
                        )
                return result_dict, pd.DataFrame(
                    receptor_type_index,
                    columns=["receptor_index", "receptor_type"],
                )
            else:
                for i, (pre, post) in receptor_type_index_groups:
                    _, pre_group = pre
                    _, post_group = post
                    pre_k, _ = pre
                    post_k, _ = post

                    conn = conn_mat_df[
                        conn_mat_df.pre.isin(pre_group)
                        & conn_mat_df.post.isin(post_group)
                    ]

                    if len(conn) > 0:
                        result_dict[(pre_k, post_k)] = scipy.sparse.coo_array(
                            (conn.data.values, (conn.pre.values, conn.post.values)),
                            shape=shape,
                        )

                return result_dict, pd.DataFrame(
                    receptor_type_index,
                    columns=[
                        "receptor_index",
                        "pre_receptor_type",
                        "post_receptor_type",
                    ],
                )

        # Default: return stacked matrix
        new_row = []
        new_col = []
        new_val = []

        if ignore_post_type:
            for i, (pre_type, pre_group) in receptor_type_index_groups:
                conn = conn_mat_df[conn_mat_df.pre.isin(pre_group)]
                new_row.append(conn.pre.values)
                # Stack columns based on pre-synaptic type index
                # Each block corresponds to one pre-synaptic channel
                new_col.append(conn.post.values * n_receptor_type + i)
                new_val.append(conn.data.values)

            return scipy.sparse.coo_array(
                (
                    np.concatenate(new_val),
                    (np.concatenate(new_row), np.concatenate(new_col)),
                ),
                shape=new_shape,
            ), pd.DataFrame(
                receptor_type_index,
                columns=["receptor_index", "receptor_type"],
            )
        else:
            for i, (pre, post) in receptor_type_index_groups:
                _, pre_group = pre
                _, post_group = post

                conn = conn_mat_df[
                    conn_mat_df.pre.isin(pre_group) & conn_mat_df.post.isin(post_group)
                ]
                new_row.append(conn.pre.values)
                new_col.append(conn.post.values * n_receptor_type + i)
                new_val.append(conn.data.values)

            return scipy.sparse.coo_array(
                (
                    np.concatenate(new_val),
                    (np.concatenate(new_row), np.concatenate(new_col)),
                ),
                shape=new_shape,
            ), pd.DataFrame(
                receptor_type_index,
                columns=["receptor_index", "pre_receptor_type", "post_receptor_type"],
            )
    else:
        raise ValueError("connections must be a DataFrame or a scipy.sparse.sparray")


def make_hetersynapse_constraint(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    cell_type_col: str = "cell_type",
    receptor_type_col: str = "EI",
    receptor_type_mode: Literal["neuron", "connection"] = "neuron",
    constraint_mode: Literal["full", "cell_only", "cell_and_receptor"] = "full",
    nan_in_same_group: bool = True,
    ignore_post_type: bool = False,
) -> scipy.sparse.sparray:
    """Create constraint matrix for hetersynaptic connections grouped by cell
    and receptor types.

    Args:
        neurons (pd.DataFrame): Must contain 'root_id', cell_type_col,
            and receptor_type_col.
        connections (pd.DataFrame): Must contain 'pre_root_id', 'post_root_id',
            'pre_simple_id', and 'post_simple_id'.
        cell_type_col: Column name for cell type information.
        receptor_type_col: Column name for receptor type information.
        receptor_type_mode: Whether receptor types are per 'neuron' or 'connection'.
        constraint_mode:
            - "full": Separate constraint for each (pre_cell_type, post_cell_type,
                      pre_receptor, post_receptor) combination
            - "cell_only": Same constraint for all receptor types with matching
                           (pre_cell_type, post_cell_type)
            - "cell_and_receptor": Constraint by (pre_cell_type, post_cell_type)
                                   and whether it's E-E, E-I, I-E, or I-I
        nan_in_same_group (bool): If True, missing cell types are grouped together.
        ignore_post_type: If True, constraints ignore post-receptor type.

    Returns:
        scipy.sparse.sparray: Sparse array matching heterosynapse connection shape,
            where each non-zero value is a group ID for that connection.
    """
    # First get the basic cell type constraint
    cell_constraint = make_constraint_by_neuron_type(
        neurons, connections, nan_in_same_group=nan_in_same_group
    )

    if constraint_mode == "cell_only":
        # Just use cell type constraints, replicate across all receptor types
        # Get receptor type info to determine how many receptor pairs
        conn_temp, receptor_idx = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col,
            receptor_type_mode,
            return_dict=False,
            ignore_post_type=ignore_post_type,
        )
        n_receptor_pairs = len(receptor_idx)

        # Replicate constraint for each receptor pair
        cell_constraint = cell_constraint.tocoo()
        new_row = []
        new_col = []
        new_val = []

        for i in range(n_receptor_pairs):
            new_row.append(cell_constraint.row)
            new_col.append(cell_constraint.col * n_receptor_pairs + i)
            new_val.append(cell_constraint.data)

        return scipy.sparse.coo_array(
            (
                np.concatenate(new_val),
                (np.concatenate(new_row), np.concatenate(new_col)),
            ),
            shape=(
                cell_constraint.shape[0],
                cell_constraint.shape[1] * n_receptor_pairs,
            ),
        )

    # For "full" or "cell_and_receptor", need to expand with receptor type info
    conn_mat, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col,
        receptor_type_mode,
        return_dict=False,
        ignore_post_type=ignore_post_type,
    )

    # Build mapping from connection to cell type pair and receptor pair
    tmp_neurons = neurons.copy()

    if not nan_in_same_group:
        none_mask = tmp_neurons[cell_type_col].isna()
        tmp_neurons.loc[none_mask, cell_type_col] = [
            f"__none_{i}__" for i in range(none_mask.sum())
        ]

    root_id_to_cell_type = dict(
        tmp_neurons[["root_id", cell_type_col]].itertuples(index=False, name=None)
    )

    tmp_conns = connections.drop_duplicates(
        subset=["pre_simple_id", "post_simple_id"]
    ).copy()

    tmp_conns["pre_cell_type"] = tmp_conns["pre_root_id"].map(root_id_to_cell_type)
    tmp_conns["post_cell_type"] = tmp_conns["post_root_id"].map(root_id_to_cell_type)

    # Get heterosynapse connection matrix to know the expanded structure
    conn_mat = conn_mat.tocoo()

    # Create DataFrame of connections in hetersynapse space
    hetero_conn_df = pd.DataFrame(
        {
            "pre": conn_mat.row,
            "post_hetero": conn_mat.col,
        }
    )

    # Map back to original post neuron and receptor index
    n_receptor_pairs = len(receptor_idx)
    hetero_conn_df["post"] = hetero_conn_df["post_hetero"] // n_receptor_pairs
    hetero_conn_df["receptor_idx"] = hetero_conn_df["post_hetero"] % n_receptor_pairs

    # Merge with receptor type info
    hetero_conn_df = hetero_conn_df.merge(
        receptor_idx, left_on="receptor_idx", right_on="receptor_index", how="left"
    )

    # Merge with connection info to get cell types
    original_conn_df = tmp_conns[
        ["pre_simple_id", "post_simple_id", "pre_cell_type", "post_cell_type"]
    ]
    hetero_conn_df = hetero_conn_df.merge(
        original_conn_df,
        left_on=["pre", "post"],
        right_on=["pre_simple_id", "post_simple_id"],
        how="left",
    )

    if constraint_mode == "full":
        # Each (pre_cell_type, post_cell_type, pre_receptor, post_receptor)
        # gets unique ID
        if ignore_post_type:
            constraint_key = hetero_conn_df[
                ["pre_cell_type", "post_cell_type", "receptor_type"]
            ].agg(tuple, axis=1)
        elif receptor_type_mode == "neuron":
            constraint_key = hetero_conn_df[
                [
                    "pre_cell_type",
                    "post_cell_type",
                    "pre_receptor_type",
                    "post_receptor_type",
                ]
            ].agg(tuple, axis=1)
        else:
            constraint_key = hetero_conn_df[
                ["pre_cell_type", "post_cell_type", "receptor_type"]
            ].agg(tuple, axis=1)
    else:  # "cell_and_receptor"
        # Group by cell type and receptor category (E-E, E-I, I-E, I-I)
        if ignore_post_type:
            constraint_key = hetero_conn_df[
                ["pre_cell_type", "post_cell_type", "receptor_type"]
            ].agg(tuple, axis=1)
        elif receptor_type_mode == "neuron":
            hetero_conn_df["receptor_category"] = (
                hetero_conn_df["pre_receptor_type"].astype(str)
                + "-"
                + hetero_conn_df["post_receptor_type"].astype(str)
            )
            constraint_key = hetero_conn_df[
                ["pre_cell_type", "post_cell_type", "receptor_category"]
            ].agg(tuple, axis=1)
        else:
            constraint_key = hetero_conn_df[
                ["pre_cell_type", "post_cell_type", "receptor_type"]
            ].agg(tuple, axis=1)

    hetero_conn_df["constraint_group_id"] = pd.factorize(constraint_key)[0] + 1

    return scipy.sparse.coo_array(
        (
            hetero_conn_df["constraint_group_id"].values,
            (hetero_conn_df["pre"].values, hetero_conn_df["post_hetero"].values),
        ),
        shape=conn_mat.shape,
    )


def make_hetersynapse_constrained_conn(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    cell_type_col: str = "cell_type",
    receptor_type_col: str = "EI",
    receptor_type_mode: Literal["neuron", "connection"] = "neuron",
    constraint_mode: Literal["full", "cell_only", "cell_and_receptor"] = "full",
    nan_in_same_group: bool = True,
    dropna: Literal["error", "filter", "unknown"] = "error",
    ignore_post_type: bool = False,
) -> tuple[scipy.sparse.sparray, scipy.sparse.sparray, pd.DataFrame]:
    """Create both heterosynaptic connection and constraint matrices.

    This is a convenience function that combines make_hetersynapse_conn and
    make_hetersynapse_constraint to produce outputs ready for SparseConstrainedConn.

    Args:
        neurons (pd.DataFrame): Must contain 'root_id', cell_type_col,
            and receptor_type_col.
        connections (pd.DataFrame): Must contain 'pre_root_id', 'post_root_id',
            'pre_simple_id', and 'post_simple_id'.
        cell_type_col: Column name for cell type information.
        receptor_type_col: Column name for receptor type information.
        receptor_type_mode: Whether receptor types are per 'neuron' or
            'connection'.
        constraint_mode: Controls constraint granularity (see
            make_hetersynapse_constraint).
        nan_in_same_group (bool): If True, missing cell types are grouped together.
        dropna: How to handle NaN receptor types ('error', 'filter', or 'unknown').
        ignore_post_type: If True, ignore post-synaptic receptor distinction.

    Returns:
        Tuple of (conn_matrix, constraint_matrix, receptor_type_index).
        Can be used directly with SparseConstrainedConn.from_hetersynapse(
            conn, constraint, receptor_idx
        ).
    """
    conn_mat, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col=receptor_type_col,
        receptor_type_mode=receptor_type_mode,
        return_dict=False,
        dropna=dropna,
        ignore_post_type=ignore_post_type,
    )

    constraint_mat = make_hetersynapse_constraint(
        neurons,
        connections,
        cell_type_col=cell_type_col,
        receptor_type_col=receptor_type_col,
        receptor_type_mode=receptor_type_mode,
        constraint_mode=constraint_mode,
        nan_in_same_group=nan_in_same_group,
        ignore_post_type=ignore_post_type,
    )

    return conn_mat, constraint_mat, receptor_idx


def stack_hetersynapse(
    conn_dict: dict,
    receptor_type_index: pd.DataFrame,
    ignore_post_type: bool = False,
) -> scipy.sparse.sparray | tuple[scipy.sparse.sparray, pd.DataFrame]:
    """Convert dict of receptor-specific matrices back to stacked matrix
    format.

    This is the inverse of make_hetersynapse_conn with return_dict=True.
    Useful after modifying individual receptor type matrices.

    Args:
        conn_dict: OrderedDict mapping receptor type pairs to sparse matrices.
            Keys are (pre_receptor, post_receptor) tuples for neuron mode,
            or receptor_type strings for connection mode.
        receptor_type_index: DataFrame with receptor type mappings, must
            include 'receptor_index' column and either
            ('pre_receptor_type', 'post_receptor_type') for neuron mode
            or 'receptor_type' for connection mode.

    Returns:
        Stacked sparse matrix in (pre_neuron, post_neuron * n_receptor_types) format.
        If ignore_post_type is True, returns (stacked_conn, new_receptor_type_index)
        as the receptor index changes.

    Example:
        >>> # Get dict, modify matrices, then stack back
        >>> conn_dict, receptor_idx = make_hetersynapse_conn(
        ...     neurons, connections, return_dict=True
        ... )
        >>> # Modify E->I connections
        >>> conn_dict[('E', 'I')].data *= 2.0
        >>> # Convert back to stacked format
        >>> conn_stacked = stack_hetersynapse(conn_dict, receptor_idx)
    """
    if len(conn_dict) == 0:
        raise ValueError("conn_dict is empty")

    n_neuron = next(iter(conn_dict.values())).shape[0]

    # Check if ignore_post_type is requested
    if ignore_post_type:
        # We need to collapse columns.
        # Iterate over conn_dict, aggregating by pre-synaptic type
        collapsed_conn_dict = OrderedDict()

        # Check if we are in neuron mode (keys are tuples)
        is_tuple_key = isinstance(next(iter(conn_dict.keys())), tuple)

        if is_tuple_key:
            # Group by pre-synaptic type
            for (pre, post), mat in conn_dict.items():
                if pre not in collapsed_conn_dict:
                    collapsed_conn_dict[pre] = mat.copy()
                else:
                    # In standard make_hetersynapse_conn with ignore_post_type=True,
                    # we create one block of columns per Pre-Receptor-Type.
                    # Inside that block, for a given pre-neuron, the row connects to
                    # ALL post-neurons (regardless of post-type) that possess that
                    # pre-receptor connectivity.
                    #
                    # If we have E->E and E->I connections in conn_dict:
                    # 'E' block should contain BOTH.
                    # Since conn_dict split them into separate matrices (same size
                    # N x N), E->E matrix has entries where post is E. E->I has
                    # entries where post is I. They are mutually exclusive.
                    # So we should sum them up.
                    collapsed_conn_dict[pre] += mat

            # Build new receptor index
            new_receptor_index = []
            for i, pre in enumerate(collapsed_conn_dict.keys()):
                new_receptor_index.append((i, pre))

            new_receptor_index_df = pd.DataFrame(
                new_receptor_index, columns=["receptor_index", "receptor_type"]
            )

            # Stack using new types
            n_receptor_types = len(collapsed_conn_dict)
            new_shape = (n_neuron, n_neuron * n_receptor_types)

            new_row = []
            new_col = []
            new_val = []

            for i, (pre, mat) in enumerate(collapsed_conn_dict.items()):
                mat = mat.tocoo()
                new_row.append(mat.row)
                new_col.append(mat.col * n_receptor_types + i)
                new_val.append(mat.data)

            stacked = scipy.sparse.coo_array(
                (
                    np.concatenate(new_val),
                    (np.concatenate(new_row), np.concatenate(new_col)),
                ),
                shape=new_shape,
            )
            return stacked, new_receptor_index_df

        else:
            # Connection mode - keys are already just receptor types (strings)
            # If ignore_post_type is requested here... connection mode essentially
            # already ignores post neuron type distinction (it groups by connection
            # property). So we can just behave like normal stacking, but return
            # Tuple to match signature? Assuming standard behavior: ignore_post_type
            # implies clustering purely by the key property. In connection mode,
            # the key IS the property.

            # Let's just fall back to normal stacking but return index.
            # Actually, if the input index has 'receptor_type', it's already compatible.
            pass

    # Standard Stacking Logic
    # Get the shape from the first matrix
    first_mat = next(iter(conn_dict.values()))
    base_shape = first_mat.shape
    n_receptor_pairs = len(receptor_type_index)

    # Detect mode from receptor_type_index columns
    has_pre_post = (
        "pre_receptor_type" in receptor_type_index.columns
        and "post_receptor_type" in receptor_type_index.columns
    )

    new_shape = (base_shape[0], base_shape[1] * n_receptor_pairs)
    new_row = []
    new_col = []
    new_val = []

    # Build mapping from receptor pair to index
    if has_pre_post:
        # Neuron mode
        receptor_to_idx = {}
        for _, row in receptor_type_index.iterrows():
            key = (row["pre_receptor_type"], row["post_receptor_type"])
            receptor_to_idx[key] = row["receptor_index"]

        for (pre_type, post_type), mat in conn_dict.items():
            if (pre_type, post_type) not in receptor_to_idx:
                continue  # Skip if not in index (or raise error?)
            idx = receptor_to_idx[(pre_type, post_type)]
            mat = mat.tocoo()

            new_row.append(mat.row)
            new_col.append(mat.col * n_receptor_pairs + idx)
            new_val.append(mat.data)
    else:
        # Connection mode
        receptor_to_idx = dict(
            receptor_type_index[["receptor_type", "receptor_index"]].itertuples(
                index=False, name=None
            )
        )

        for receptor_type, mat in conn_dict.items():
            idx = receptor_to_idx[receptor_type]
            mat = mat.tocoo()

            new_row.append(mat.row)
            new_col.append(mat.col * n_receptor_pairs + idx)
            new_val.append(mat.data)

    return scipy.sparse.coo_array(
        (
            np.concatenate(new_val),
            (np.concatenate(new_row), np.concatenate(new_col)),
        ),
        shape=new_shape,
    )
