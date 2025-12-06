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


def make_sparse_mat(connections: pd.DataFrame, shape) -> scipy.sparse.sparray:
    # Important: Connections can have duplicated pre and post neuron pairs
    #            if they innervate between different neuropils.
    tmp_connections = connections[["syn_count", "pre_simple_id", "post_simple_id"]]
    tmp_connections = tmp_connections.groupby(
        ["pre_simple_id", "post_simple_id"], as_index=False
    ).agg({"syn_count": "sum"})

    # Note: this .T has nothing to do with transposing the full weight matrix.
    #       it only makes (N, 2) to (2, N) where the 2 rows correspond to pre and post
    pre, post = (
        tmp_connections[["pre_simple_id", "post_simple_id"]].to_numpy(dtype=int).T
    )

    ret = scipy.sparse.coo_array(
        (
            tmp_connections[["syn_count"]].to_numpy().flatten(),
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
) -> tuple[scipy.sparse.sparray, pd.DataFrame]:
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

    Returns:
        A tuple containing:
            - The transformed sparse array (`scipy.sparse.sparray`) in
              (pre_neuron, post_neuron * receptor_type).
            - A DataFrame mapping the new rows to receptor types. For 'neuron'
              mode, this will include 'pre_receptor_type' and
              'post_receptor_type'. For 'connection' mode, it will just have
              'receptor_type'.

    Raises:
        ValueError: If `connections` is not a DataFrame or a sparse array, or if
                    the `receptor_type_mode` is 'connection' but the input is a
                    sparse array.
    """

    # TODO: handle na values
    # TODO: make this compatible with make_constraint_by_neuron_type

    if isinstance(connections, pd.DataFrame):
        if receptor_type_mode == "neuron":
            connections = make_sparse_mat(connections, (len(neurons), len(neurons)))
            return make_hetersynapse_conn(
                neurons,
                connections,
                receptor_type_col,
                receptor_type_mode,
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

        return scipy.sparse.coo_array(
            (
                np.concatenate(new_val),
                (np.concatenate(new_row), np.concatenate(new_col)),
            ),
            shape=new_shape,
        ), pd.DataFrame(
            receptor_type_index, columns=["receptor_index", "receptor_type"]
        )

    elif isinstance(connections, scipy.sparse.sparray):
        assert receptor_type_mode == "neuron"
        connections = connections.tocoo()
        conn_mat_df = pd.DataFrame(
            {"pre": connections.row, "post": connections.col, "data": connections.data}
        )
        shape = connections.shape

        receptor_type_groups = OrderedDict(
            groupby_to_dict(
                neurons, "simple_id", by=receptor_type_col, sort=True, dropna=False
            )
        )
        receptor_type_groups = list(
            itertools.product(receptor_type_groups.items(), repeat=2)
        )
        receptor_type_index_groups = list(enumerate(receptor_type_groups))
        receptor_type_index = [
            (i, pre_k, post_k)
            for i, ((pre_k, _), (post_k, _)) in receptor_type_index_groups
        ]

        n_receptor_type = len(receptor_type_groups)
        new_shape = (shape[0], shape[1] * n_receptor_type)
        new_row = []
        new_col = []
        new_val = []
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
