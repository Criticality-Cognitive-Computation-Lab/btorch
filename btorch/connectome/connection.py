import itertools
import warnings
from collections import OrderedDict
from typing import Literal, Sequence, overload

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.spatial

from ..utils.pandas_utils import groupby_to_dict
from . import simple_id_to_root_id
from .augment import IdType


ReceptorTypeMode = Literal["neuron", "connection"]
ConstraintMode = Literal["full", "cell_only", "cell_and_receptor"]
DropNaMode = Literal["error", "filter", "unknown"]


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


@overload
def neuron_subset_to_conn_mat(
    subset: Sequence[int] | pd.Series | pd.DataFrame,
    id_type: IdType,
    size: int,
    neurons: pd.DataFrame | None = None,
    remove_nan: bool = False,
    return_mode: Literal["scatter"] = "scatter",
) -> np.ndarray: ...


@overload
def neuron_subset_to_conn_mat(
    subset: Sequence[int] | pd.Series | pd.DataFrame,
    id_type: IdType,
    size: int,
    neurons: pd.DataFrame | None = None,
    remove_nan: bool = False,
    *,
    return_mode: Literal["sparray"],
) -> scipy.sparse.sparray: ...


@overload
def neuron_subset_to_conn_mat(
    subset: Sequence[int] | pd.Series | pd.DataFrame,
    id_type: IdType,
    size: int,
    neurons: pd.DataFrame | None = None,
    remove_nan: bool = False,
    *,
    return_mode: Literal["sparray", "scatter"],
) -> scipy.sparse.sparray | np.ndarray: ...


def neuron_subset_to_conn_mat(
    subset: Sequence[int] | pd.Series | pd.DataFrame,
    id_type: IdType,
    size: int,
    neurons: pd.DataFrame | None = None,
    remove_nan: bool = False,
    return_mode: Literal["sparray", "scatter"] = "scatter",
) -> scipy.sparse.sparray | np.ndarray:
    """Convert a neuron subset to simple ids or a selection matrix.

    Args:
        subset: Neuron ids (sequence or Series), or a DataFrame with a
            ``root_id`` column (and optionally ``simple_id``).
        id_type: Kind of ids in ``subset`` when it is a sequence or Series.
        size: Total number of neurons (columns of the selection matrix).
        neurons: Neuron table with ``root_id`` and ``simple_id`` columns.
            Required whenever ``root_id`` values must be mapped to ``simple_id``.
        remove_nan: Drop (with a warning) ids that cannot be mapped instead of
            raising.
        return_mode: ``"scatter"`` returns the simple ids as an array;
            ``"sparray"`` returns a ``(len(subset), size)`` selection matrix.

    Returns:
        Array of simple ids, or a sparse selection matrix.

    Raises:
        ValueError: If ``subset`` has an invalid type or lacks ``root_id``, if
            ``neurons`` is needed but missing, or if ids are unmappable and
            ``remove_nan`` is False.
    """
    needs_mapping = (
        isinstance(subset, pd.DataFrame)
        and "simple_id" not in subset.columns
        or not isinstance(subset, pd.DataFrame)
        and id_type == "root_id"
    )
    if needs_mapping and neurons is None:
        raise ValueError(
            "`neurons` is required to map root_id to simple_id for this subset."
        )
    if isinstance(subset, (Sequence, pd.Series)):
        df = pd.DataFrame(subset, columns=[id_type])
        if id_type == "root_id":
            df["simple_id"] = df.root_id.map(
                simple_id_to_root_id(neurons, reverse=True).get
            )
    elif isinstance(subset, pd.DataFrame):
        if "root_id" not in subset.columns:
            raise ValueError("Neuron subset DataFrame must have a 'root_id' column.")
        if "simple_id" not in subset.columns:
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
        if not unmapped.empty:
            raise ValueError(
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

    Returns:
        scipy.sparse.sparray: Sparse array with shape (num_neurons, num_neurons),
            where each non-zero value is a group ID for that connection.
    """

    tmp_neurons = neurons.copy()

    if not nan_in_same_group:
        none_mask = tmp_neurons["cell_type"].isna()
        tmp_neurons.loc[none_mask, "cell_type"] = [
            f"__none_{i}__" for i in range(none_mask.sum())
        ]

    root_id_to_cell_type = dict(
        tmp_neurons[["root_id", "cell_type"]].itertuples(index=False, name=None)
    )

    # Drop duplicate synapses to avoid repeated entries in constraint matrix
    tmp_conns = connections.drop_duplicates(
        subset=["pre_simple_id", "post_simple_id"]
    ).copy()

    tmp_conns["pre_cell_type"] = tmp_conns["pre_root_id"].map(root_id_to_cell_type)
    tmp_conns["post_cell_type"] = tmp_conns["post_root_id"].map(root_id_to_cell_type)

    # Group ids start at 1 so that 0 stays "no connection" in the sparse matrix.
    tmp_conns["cell_type_pair_id"] = (
        pd.factorize(tmp_conns[["pre_cell_type", "post_cell_type"]].agg(tuple, axis=1))[
            0
        ]
        + 1
    )

    constraint_group = scipy.sparse.coo_array(
        (
            tmp_conns["cell_type_pair_id"].to_numpy(dtype=int).flatten(),
            tmp_conns[["pre_simple_id", "post_simple_id"]].to_numpy(dtype=int).T,
        ),
        shape=(neurons.shape[0], neurons.shape[0]),
    )

    return constraint_group


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


# ---------------------------------------------------------------------------
# make_hetersynapse_conn helpers
#
# Internally every input is normalised to an "edge table": a DataFrame with
# columns ``pre``, ``post``, ``data`` (and ``delay`` when delays are expanded).
# Receptor builders split the edge table into ``(key, edges)`` groups, and a
# single assembler turns the groups into a stacked matrix or a dict.  Delay
# expansion is applied per edge in the assembler, so the delay of an edge can
# never get out of step with the entry it belongs to.
# ---------------------------------------------------------------------------


def _delay_rows(pre: np.ndarray, delays, n_delay_bins: int) -> np.ndarray:
    """Row index ``pre * n_delay_bins + clip(delay)`` of each delayed edge."""
    delays = np.asarray(delays)
    if np.any(delays < 0):
        raise ValueError("delays must be non-negative")
    delay_bins = np.clip(delays, 0, n_delay_bins - 1).astype(int)
    return pre * n_delay_bins + delay_bins


def _aggregate_edges(connections: pd.DataFrame, delay_col: str | None) -> pd.DataFrame:
    """Sum duplicated (pre, post) pairs into an edge table.

    Duplicates occur when two neurons are connected in several
    neuropils. Delays of merged duplicates are averaged and truncated to
    integer steps.
    """
    agg = {"syn_count": "sum"}
    if delay_col is not None:
        agg[delay_col] = "mean"
    tmp = connections.groupby(["pre_simple_id", "post_simple_id"], as_index=False).agg(
        agg
    )
    edges = pd.DataFrame(
        {
            "pre": tmp["pre_simple_id"].to_numpy(dtype=int),
            "post": tmp["post_simple_id"].to_numpy(dtype=int),
            "data": tmp["syn_count"].to_numpy(),
        }
    )
    if delay_col is not None:
        edges["delay"] = tmp[delay_col].to_numpy().astype(int)
    return edges


def _validate_hetersynapse_args(
    connections, receptor_type_col, receptor_type_mode, return_dict, delay_col
) -> None:
    """Reject argument combinations that no branch can serve."""
    is_df = isinstance(connections, pd.DataFrame)
    if delay_col is not None and not is_df:
        raise ValueError("delay_col can only be used with DataFrame connections")
    if receptor_type_col is None:
        if not is_df:
            raise ValueError(
                "receptor_type_col=None only supported with DataFrame connections"
            )
        if delay_col is None:
            raise ValueError(
                "Must provide either receptor_type_col or delay_col (or both)"
            )
        if return_dict:
            raise ValueError(
                "return_dict=True requires receptor_type_col: without receptor "
                "types there is nothing to key the dict by"
            )
    if is_df:
        if delay_col is not None and delay_col not in connections.columns:
            raise KeyError(
                f"delay_col '{delay_col}' not found in connections DataFrame"
            )
    elif isinstance(connections, scipy.sparse.sparray):
        if receptor_type_mode != "neuron":
            raise ValueError(
                "Sparse-array input only supports receptor_type_mode='neuron', "
                f"got {receptor_type_mode!r}"
            )
    else:
        raise ValueError("connections must be a DataFrame or a scipy.sparse.sparray")


def _resolve_nan_receptors(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    receptor_type_col: str,
    dropna: DropNaMode,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the ``dropna`` policy to neurons without a receptor type.

    Returns the (possibly relabelled) neurons and the (possibly
    filtered) edges.
    """
    nan_mask = neurons[receptor_type_col].isna()
    if not nan_mask.any():
        return neurons, edges
    if dropna == "error":
        nan_neurons = neurons[nan_mask]["simple_id"].tolist()
        raise ValueError(
            f"NaN receptor types found for neurons: {nan_neurons}. "
            "Set dropna='filter' to remove connections involving them, "
            "or dropna='unknown' to treat NaN as a separate receptor type."
        )
    if dropna == "filter":
        nan_neuron_ids = set(neurons[nan_mask]["simple_id"])
        n_before = len(edges)
        edges = edges[
            ~edges["pre"].isin(nan_neuron_ids) & ~edges["post"].isin(nan_neuron_ids)
        ]
        n_filtered = n_before - len(edges)
        if n_filtered > 0:
            warnings.warn(
                f"Filtered out {n_filtered} connections involving "
                f"{len(nan_neuron_ids)} neurons with NaN receptor types"
            )
        return neurons, edges
    if dropna == "unknown":
        neurons = neurons.copy()
        neurons.loc[nan_mask, receptor_type_col] = "unknown"
        warnings.warn(
            f"Treating {nan_mask.sum()} neurons with NaN receptor types "
            "as 'unknown' receptor type"
        )
        return neurons, edges
    raise ValueError(f"dropna must be 'error', 'filter', or 'unknown', got {dropna}")


def _neuron_receptor_groups(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    receptor_type_col: str,
    dropna: DropNaMode,
    ignore_post_type: bool,
) -> tuple[list[tuple], pd.DataFrame]:
    """Split edges by the receptor types of their pre/post neurons.

    Returns ``(groups, index_df)``; ``groups[i]`` is ``(key, edges_i)`` where
    ``key`` is the pre type (``ignore_post_type``) or ``(pre, post)`` type pair.
    """
    neurons, edges = _resolve_nan_receptors(neurons, edges, receptor_type_col, dropna)
    type_members = OrderedDict(
        groupby_to_dict(
            neurons,
            "simple_id",
            by=receptor_type_col,
            sort=True,
            dropna=(dropna == "filter"),  # only drop NaN groups if filtering
        )
    )

    groups = []
    if ignore_post_type:
        # One channel per pre-synaptic type; all post neurons are targets.
        for pre_k, pre_ids in type_members.items():
            groups.append((pre_k, edges[edges["pre"].isin(pre_ids)]))
        index_df = pd.DataFrame(
            [(i, key) for i, (key, _) in enumerate(groups)],
            columns=["receptor_index", "receptor_type"],
        )
    else:
        for (pre_k, pre_ids), (post_k, post_ids) in itertools.product(
            type_members.items(), repeat=2
        ):
            mask = edges["pre"].isin(pre_ids) & edges["post"].isin(post_ids)
            groups.append(((pre_k, post_k), edges[mask]))
        index_df = pd.DataFrame(
            [(i, *key) for i, (key, _) in enumerate(groups)],
            columns=["receptor_index", "pre_receptor_type", "post_receptor_type"],
        )
    return groups, index_df


def _connection_receptor_groups(
    connections: pd.DataFrame, receptor_type_col: str, delay_col: str | None
) -> tuple[list[tuple], pd.DataFrame]:
    """Split connections by their own receptor type (possible
    cotransmission)."""
    by_type = groupby_to_dict(connections, by=receptor_type_col, sort=True)
    groups = [(k, _aggregate_edges(v, delay_col)) for k, v in by_type.items()]
    index_df = pd.DataFrame(
        [(i, key) for i, (key, _) in enumerate(groups)],
        columns=["receptor_index", "receptor_type"],
    )
    return groups, index_df


def _assemble_hetersynapse(
    groups: list[tuple],
    shape: tuple[int, int],
    return_dict: bool,
    n_delay_bins: int,
) -> scipy.sparse.sparray | OrderedDict:
    """Turn ``(key, edges)`` groups into a stacked matrix or a dict of
    matrices.

    Stacked: edge ``pre -> post`` of group ``i`` lands at column
    ``post * n_groups + i``.  Dict: one ``shape`` matrix per non-empty group.
    With ``n_delay_bins > 1`` rows become ``pre * n_delay_bins + delay_bin``.
    """
    n_pre, n_post = shape
    expand = n_delay_bins > 1
    n_rows = n_pre * n_delay_bins if expand else n_pre
    n_groups = len(groups)

    def rows_of(edges: pd.DataFrame) -> np.ndarray:
        if expand:
            return _delay_rows(edges["pre"].to_numpy(), edges["delay"], n_delay_bins)
        return edges["pre"].to_numpy()

    if return_dict:
        return OrderedDict(
            (
                key,
                scipy.sparse.coo_array(
                    (
                        edges["data"].to_numpy(),
                        (rows_of(edges), edges["post"].to_numpy()),
                    ),
                    shape=(n_rows, n_post),
                ),
            )
            for key, edges in groups
            if len(edges) > 0
        )

    rows, cols, vals = [], [], []
    for i, (_, edges) in enumerate(groups):
        rows.append(rows_of(edges))
        cols.append(edges["post"].to_numpy() * n_groups + i)
        vals.append(edges["data"].to_numpy())
    if not groups:
        rows, cols, vals = (
            [np.empty(0, dtype=int)],
            [np.empty(0, dtype=int)],
            [np.empty(0)],
        )
    return scipy.sparse.coo_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_rows, n_post * n_groups),
    )


@overload
def make_hetersynapse_conn(
    neurons: pd.DataFrame,
    connections: scipy.sparse.sparray | pd.DataFrame,
    receptor_type_col: str | None = "EI",
    receptor_type_mode: ReceptorTypeMode = "neuron",
    return_dict: Literal[False] = False,
    dropna: DropNaMode = "error",
    ignore_post_type: bool = False,
    delay_col: str | None = None,
    n_delay_bins: int = 5,
) -> tuple[scipy.sparse.sparray, pd.DataFrame]: ...


@overload
def make_hetersynapse_conn(
    neurons: pd.DataFrame,
    connections: scipy.sparse.sparray | pd.DataFrame,
    receptor_type_col: str | None = "EI",
    receptor_type_mode: ReceptorTypeMode = "neuron",
    return_dict: Literal[True] = ...,
    dropna: DropNaMode = "error",
    ignore_post_type: bool = False,
    delay_col: str | None = None,
    n_delay_bins: int = 5,
) -> tuple[OrderedDict, pd.DataFrame]: ...


@overload
def make_hetersynapse_conn(
    neurons: pd.DataFrame,
    connections: scipy.sparse.sparray | pd.DataFrame,
    receptor_type_col: str | None = "EI",
    receptor_type_mode: ReceptorTypeMode = "neuron",
    return_dict: bool = ...,
    dropna: DropNaMode = "error",
    ignore_post_type: bool = False,
    delay_col: str | None = None,
    n_delay_bins: int = 5,
) -> tuple[scipy.sparse.sparray, pd.DataFrame] | tuple[OrderedDict, pd.DataFrame]: ...


def make_hetersynapse_conn(
    neurons: pd.DataFrame,
    connections: scipy.sparse.sparray | pd.DataFrame,
    receptor_type_col: str | None = "EI",
    receptor_type_mode: ReceptorTypeMode = "neuron",
    return_dict: bool = False,
    dropna: DropNaMode = "error",
    ignore_post_type: bool = False,
    delay_col: str | None = None,
    n_delay_bins: int = 5,
) -> tuple[scipy.sparse.sparray, pd.DataFrame] | tuple[OrderedDict, pd.DataFrame]:
    """Transforms a connectivity matrix to represent hetersynaptic connections
    based on receptor types, with optional delay support.

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

    When delay_col is provided (DataFrame only), the output matrix rows are
    expanded to include delay bins: shape becomes
    (n_neurons * n_delay_bins, n_neurons * n_receptors).

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
        return_dict: If True, returns OrderedDict mapping receptor types (pre
        type with ``ignore_post_type``, ``(pre, post)`` pairs otherwise, the
        connection's receptor type in 'connection' mode) to sparse matrices
        instead of a single stacked matrix. Receptor groups without any
        connection are omitted from the dict. Not available with
        ``receptor_type_col=None``.
        dropna: How to handle NaN receptor types. Options:
            - 'error' (default): Raise ValueError if NaN found
            - 'filter': Remove connections involving NaN neurons
              (preserves neuron count)
            - 'unknown': Treat NaN as a separate receptor type category
        ignore_post_type: If True, clusters connections only by pre-synaptic
            receptor type, ignoring post-synaptic receptor type. Resulting matrix
            will have fewer columns. Only valid for 'neuron' mode.
        delay_col: Optional column name in connections DataFrame containing
            delay values (in dt steps). If provided, expands matrix rows for
            delays. Only works with DataFrame connections. Delays of
            duplicated (pre, post) pairs are averaged (truncated to int);
            values above ``n_delay_bins - 1`` are clipped. With
            ``return_dict=True`` every matrix has ``n_neurons * n_delay_bins``
            rows.
        n_delay_bins: Number of delay bins when delay_col is provided
            (default: 5; a value of 1 means no delay
            expansion).

    Returns:
        A tuple containing:
            - The transformed sparse array (`scipy.sparse.sparray`) or OrderedDict
              of sparse arrays (if return_dict=True). With delays, shape is
              (n_neurons * n_delay_bins, n_neurons * n_receptors).
            - A DataFrame mapping the new columns to receptor types. For 'neuron'
              mode, this will include 'pre_receptor_type' and
              'post_receptor_type'. For 'connection' mode, it will just have
              'receptor_type'.

    Raises:
        ValueError: If `connections` is not a DataFrame or a sparse array, or if
                    the `receptor_type_mode` is 'connection' but the input is a
                    sparse array, or if NaN values found and dropna='error', or
                    if delay_col provided with sparse array connections, or if
                    ``receptor_type_col`` is None together with ``return_dict``,
                    a sparse array, or no ``delay_col``.

    Note:
        ``receptor_type_col=None`` with ``delay_col`` builds a delays-only matrix
        of shape ``(n_neurons * n_delay_bins, n_neurons)`` and returns it with an
        index frame of ``simple_id`` (plus ``delay_index`` when expanded).
        ``dropna`` and ``ignore_post_type`` only apply to 'neuron' mode.
        The neuron count is always preserved (based on simple_id indexing).
        When dropna='filter', only connections are removed, not neurons.
        When dropna='unknown', NaN becomes a valid receptor type.
    """
    _validate_hetersynapse_args(
        connections, receptor_type_col, receptor_type_mode, return_dict, delay_col
    )
    # n_delay_bins == 1 means "no delay expansion"; the delay column is then unused.
    n_bins = n_delay_bins if delay_col is not None and n_delay_bins > 1 else 1
    edge_delay_col = delay_col if n_bins > 1 else None
    n = len(neurons)

    if receptor_type_col is None:
        # Delays only: a single channel, no receptor expansion.
        edges = _aggregate_edges(connections, edge_delay_col)
        result = _assemble_hetersynapse([(None, edges)], (n, n), False, n_bins)
        index_df = pd.DataFrame({"simple_id": range(n)})
        if n_delay_bins > 1:
            index_df["delay_index"] = 0
        return result, index_df

    if receptor_type_mode == "connection":
        # Receptor type is a property of the connection itself (no pre x post
        # product), so DataFrame input is required and dropna/ignore_post_type
        # do not apply.
        groups, index_df = _connection_receptor_groups(
            connections, receptor_type_col, edge_delay_col
        )
        shape = (n, n)
    else:
        if isinstance(connections, pd.DataFrame):
            edges = _aggregate_edges(connections, edge_delay_col)
            shape = (n, n)
        else:
            coo = connections.tocoo()
            if np.isnan(coo.data).any():
                raise ValueError("NaN values detected in connection matrix data")
            edges = pd.DataFrame({"pre": coo.row, "post": coo.col, "data": coo.data})
            shape = coo.shape
        groups, index_df = _neuron_receptor_groups(
            neurons, edges, receptor_type_col, dropna, ignore_post_type
        )

    return _assemble_hetersynapse(groups, shape, return_dict, n_bins), index_df


def make_hetersynapse_constraint(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    cell_type_col: str = "cell_type",
    receptor_type_col: str = "EI",
    receptor_type_mode: ReceptorTypeMode = "neuron",
    constraint_mode: ConstraintMode = "full",
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
        scipy.sparse.sparray: Sparse array matching hetersynapse connection shape,
            where each non-zero value is a group ID for that connection.
    """
    cell_constraint = make_constraint_by_neuron_type(
        neurons, connections, nan_in_same_group=nan_in_same_group
    )

    if constraint_mode == "cell_only":
        conn_temp, receptor_idx = make_hetersynapse_conn(
            neurons,
            connections,
            receptor_type_col,
            receptor_type_mode,
            return_dict=False,
            ignore_post_type=ignore_post_type,
        )
        n_receptor_pairs = len(receptor_idx)

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

    conn_mat, receptor_idx = make_hetersynapse_conn(
        neurons,
        connections,
        receptor_type_col,
        receptor_type_mode,
        return_dict=False,
        ignore_post_type=ignore_post_type,
    )

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

    conn_mat = conn_mat.tocoo()

    hetersynapse_conn_df = pd.DataFrame(
        {
            "pre": conn_mat.row,
            "post_hetersynapse": conn_mat.col,
        }
    )

    # Map back to original post neuron and receptor index
    n_receptor_pairs = len(receptor_idx)
    hetersynapse_conn_df["post"] = (
        hetersynapse_conn_df["post_hetersynapse"] // n_receptor_pairs
    )
    hetersynapse_conn_df["receptor_idx"] = (
        hetersynapse_conn_df["post_hetersynapse"] % n_receptor_pairs
    )

    hetersynapse_conn_df = hetersynapse_conn_df.merge(
        receptor_idx, left_on="receptor_idx", right_on="receptor_index", how="left"
    )

    original_conn_df = tmp_conns[
        ["pre_simple_id", "post_simple_id", "pre_cell_type", "post_cell_type"]
    ]
    hetersynapse_conn_df = hetersynapse_conn_df.merge(
        original_conn_df,
        left_on=["pre", "post"],
        right_on=["pre_simple_id", "post_simple_id"],
        how="left",
    )

    if constraint_mode == "full":
        # Each (pre_cell_type, post_cell_type, pre_receptor, post_receptor)
        # gets unique ID
        if ignore_post_type:
            constraint_key = hetersynapse_conn_df[
                ["pre_cell_type", "post_cell_type", "receptor_type"]
            ].agg(tuple, axis=1)
        elif receptor_type_mode == "neuron":
            constraint_key = hetersynapse_conn_df[
                [
                    "pre_cell_type",
                    "post_cell_type",
                    "pre_receptor_type",
                    "post_receptor_type",
                ]
            ].agg(tuple, axis=1)
        else:
            constraint_key = hetersynapse_conn_df[
                ["pre_cell_type", "post_cell_type", "receptor_type"]
            ].agg(tuple, axis=1)
    else:  # "cell_and_receptor"
        # Group by cell type and receptor category (E-E, E-I, I-E, I-I)
        if ignore_post_type:
            constraint_key = hetersynapse_conn_df[
                ["pre_cell_type", "post_cell_type", "receptor_type"]
            ].agg(tuple, axis=1)
        elif receptor_type_mode == "neuron":
            hetersynapse_conn_df["receptor_category"] = (
                hetersynapse_conn_df["pre_receptor_type"].astype(str)
                + "-"
                + hetersynapse_conn_df["post_receptor_type"].astype(str)
            )
            constraint_key = hetersynapse_conn_df[
                ["pre_cell_type", "post_cell_type", "receptor_category"]
            ].agg(tuple, axis=1)
        else:
            constraint_key = hetersynapse_conn_df[
                ["pre_cell_type", "post_cell_type", "receptor_type"]
            ].agg(tuple, axis=1)

    hetersynapse_conn_df["constraint_group_id"] = pd.factorize(constraint_key)[0] + 1

    return scipy.sparse.coo_array(
        (
            hetersynapse_conn_df["constraint_group_id"].values,
            (
                hetersynapse_conn_df["pre"].values,
                hetersynapse_conn_df["post_hetersynapse"].values,
            ),
        ),
        shape=conn_mat.shape,
    )


def make_hetersynapse_constrained_conn(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    cell_type_col: str = "cell_type",
    receptor_type_col: str = "EI",
    receptor_type_mode: ReceptorTypeMode = "neuron",
    constraint_mode: ConstraintMode = "full",
    nan_in_same_group: bool = True,
    dropna: DropNaMode = "error",
    ignore_post_type: bool = False,
) -> tuple[scipy.sparse.sparray, scipy.sparse.sparray, pd.DataFrame]:
    """Create both hetersynaptic connection and constraint matrices.

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
    ignore_receptor_type: Literal["pre", "post", "all"] | None = None,
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
        ignore_receptor_type: Optional collapse strategy to reduce receptor
            distinctions. None keeps full indexing. "post" collapses only
            post-receptor types in neuron mode (treated as "all" in
            connection mode). "pre" is accepted for connection mode and
            behaves like "all". "all" sums all receptor channels into one.

    Returns:
        Stacked sparse matrix with shape (N, N * n_receptor_types). When
        collapsing, also returns a new receptor_type_index describing the
        reduced mapping.
    """

    if not conn_dict:
        raise ValueError("conn_dict is empty")

    first_mat = next(iter(conn_dict.values()))
    n_neuron = first_mat.shape[0]

    neuron_mode = {
        "pre_receptor_type",
        "post_receptor_type",
    }.issubset(receptor_type_index.columns)
    connection_mode = "receptor_type" in receptor_type_index.columns and not neuron_mode

    def _stack(items, idx_lookup, n_receptor_pairs):
        rows, cols, vals = [], [], []
        for key, mat in items:
            if key not in idx_lookup:
                raise KeyError(f"Receptor key {key} missing from receptor_type_index")
            idx = idx_lookup[key]
            mat = mat.tocoo()
            rows.append(mat.row)
            cols.append(mat.col * n_receptor_pairs + idx)
            vals.append(mat.data)

        return scipy.sparse.coo_array(
            (
                np.concatenate(vals),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=(n_neuron, n_neuron * n_receptor_pairs),
        )

    def _make_index_and_lookup(labels: list[str]) -> tuple[pd.DataFrame, dict]:
        idx_df = pd.DataFrame(
            [(i, lab) for i, lab in enumerate(labels)],
            columns=["receptor_index", "receptor_type"],
        )
        lookup = dict(
            idx_df[["receptor_type", "receptor_index"]].itertuples(
                index=False, name=None
            )
        )
        return idx_df, lookup

    def _collapse_all(conn_dict: dict) -> tuple[scipy.sparse.sparray, pd.DataFrame]:
        summed = None
        for mat in conn_dict.values():
            summed = mat.copy() if summed is None else summed + mat
        idx_df = pd.DataFrame([(0, "all")], columns=["receptor_index", "receptor_type"])
        return summed.tocoo(), idx_df

    if ignore_receptor_type in {"pre", "post", "all"}:
        # Connection mode: 'pre'/'post' behave like 'all'
        if connection_mode:
            return _collapse_all(conn_dict)

        if ignore_receptor_type == "all":
            return _collapse_all(conn_dict)

        collapsed = OrderedDict()
        if ignore_receptor_type == "post":
            for (pre, _post), mat in conn_dict.items():
                if pre not in collapsed:
                    collapsed[pre] = mat.copy()
                else:
                    collapsed[pre] += mat
            labels = list(collapsed.keys())
            idx_df, lookup = _make_index_and_lookup(labels)
            return _stack(collapsed.items(), lookup, len(labels)), idx_df

        if ignore_receptor_type == "pre":
            for (_pre, post), mat in conn_dict.items():
                if post not in collapsed:
                    collapsed[post] = mat.copy()
                else:
                    collapsed[post] += mat
            labels = list(collapsed.keys())
            idx_df, lookup = _make_index_and_lookup(labels)
            return _stack(collapsed.items(), lookup, len(labels)), idx_df

    if neuron_mode:
        idx_lookup = {
            (row.pre_receptor_type, row.post_receptor_type): row.receptor_index
            for row in receptor_type_index.itertuples(index=False)
        }
        return _stack(conn_dict.items(), idx_lookup, len(receptor_type_index))

    if connection_mode:
        idx_lookup = {
            row.receptor_type: row.receptor_index
            for row in receptor_type_index.itertuples(index=False)
        }
        return _stack(conn_dict.items(), idx_lookup, len(receptor_type_index))

    raise ValueError(
        "receptor_type_index must contain either "
        "(pre_receptor_type, post_receptor_type) "
        "or receptor_type columns"
    )


def expand_conn_for_delays(
    conn: scipy.sparse.sparray,
    delays: np.ndarray,
    n_delay_bins: int = 5,
) -> scipy.sparse.sparray:
    """Expand connection matrix for delay dimension.

    Expands the ROW dimension (pre-synaptic neurons) to accommodate
    spike history. Each pre-neuron gets n_delay_bins "virtual" neurons
    representing its spike at different delays.

    The output matrix shape is (n_neurons * n_delay_bins, n_neurons),
    where row i * n_delay_bins + d corresponds to neuron i at delay d.

    Args:
        conn: Base connection matrix of shape (n_neurons, n_neurons).
        delays: Per-connection delay values in dt steps, array of length
            conn.nnz (number of non-zero connections).
        n_delay_bins: Number of delay bins (0 to n_delay_bins-1).

    Returns:
        Expanded sparse matrix of shape (n_neurons * n_delay_bins, n_neurons).

    Raises:
        ValueError: If delays contains negative values.

    Example:
        >>> import scipy.sparse
        >>> import numpy as np
        >>> conn = scipy.sparse.coo_array(
        ...     ([1.0, 2.0], ([0, 1], [1, 2])),
        ...     shape=(3, 3)
        ... )
        >>> delays = np.array([1, 3])  # conn[0,1] has delay 1, conn[1,2] has delay 3
        >>> conn_d = expand_conn_for_delays(conn, delays, n_delay_bins=5)
        >>> conn_d.shape
        (15, 3)  # 3 neurons * 5 delays, 3 neurons
        >>> # conn[0,1]=1.0 with delay 1 -> conn_d[0*5+1, 1] = 1.0
        >>> # conn[1,2]=2.0 with delay 3 -> conn_d[1*5+3, 2] = 2.0
    """
    conn_coo = conn.tocoo()
    n_pre, n_post = conn.shape
    # Expand rows: each pre neuron gets n_delay_bins slots; post is unchanged.
    new_row = _delay_rows(conn_coo.row, delays, n_delay_bins)
    new_shape = (n_pre * n_delay_bins, n_post)
    return scipy.sparse.coo_array(
        (conn_coo.data, (new_row, conn_coo.col)), shape=new_shape
    )
