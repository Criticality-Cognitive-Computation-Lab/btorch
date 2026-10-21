import numpy as np
import scipy.sparse


def _fill_random_block(
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    count: int,
    *targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """Fill preallocated COO row/column arrays with a balanced block sample.

    Row counts are balanced around ``count // n_rows``; the remainder rows are
    selected uniformly without replacement. Each row then samples unique
    columns only, so the helper never materializes the block or a global
    permutation. This keeps the recipe valid when a block's Cartesian product
    is too large for NumPy's hypergeometric argument range.
    """
    if count == 0:
        return
    n_rows = int(row_ids.size)
    n_cols = int(col_ids.size)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("A non-empty sample needs non-empty row and column sets.")

    base, remainder = divmod(count, n_rows)
    extra_rows = np.zeros(n_rows, dtype=bool)
    if remainder:
        extra_rows[np.random.choice(n_rows, size=remainder, replace=False)] = True

    cursor = 0
    for row_pos, row in enumerate(row_ids):
        take = base + int(extra_rows[row_pos])
        if take:
            stop = cursor + take
            if take == n_cols:
                selected = col_ids
            else:
                selected = col_ids[np.random.choice(n_cols, size=take, replace=False)]
            for rows, cols in targets:
                rows[cursor:stop] = row
                cols[cursor:stop] = selected
            cursor = stop

    if cursor != count:
        raise RuntimeError("Internal block sampler count mismatch.")


def _block_count(n_rows: int, n_cols: int, density: float) -> int:
    """Return the historical floor of a block's requested edge count."""
    return int(n_rows * n_cols * density)


def build_dense_mat(
    n_e_neurons: int,
    n_i_neurons: int,
    split: bool = False,
    i_e_ratio=100,
    e_to_e_mean=4.0e-3,
    e_to_e_std=1.9e-3,
    e_i_mean=5e-2,
    i_i_mean=25e-4,
):
    n_neurons = n_e_neurons + n_i_neurons
    e_idx = np.arange(n_e_neurons)
    i_idx = np.arange(n_e_neurons, n_neurons)

    # Initialize random weights (uniform 0–0.05)
    full_mat = np.random.rand(n_neurons, n_neurons) * 0.05

    # E to E lognormal
    m = np.log(e_to_e_mean**2 / np.sqrt(e_to_e_std**2 + e_to_e_mean**2))
    s = np.sqrt(np.log(1 + (e_to_e_std**2 / e_to_e_mean**2)))
    e_to_e_weights = np.exp(
        np.random.normal(loc=m, scale=s, size=(n_e_neurons, n_e_neurons))
    )
    full_mat[np.ix_(e_idx, e_idx)] = e_to_e_weights

    # I to E Gaussian
    mean_e_weights = e_to_e_weights.mean(axis=0)  # mean over presynaptic E
    mean_i_weights = mean_e_weights * i_e_ratio
    for e_neuron in e_idx:
        i_to_e_weights = np.abs(
            np.random.normal(
                loc=mean_i_weights[e_neuron],
                scale=mean_i_weights[e_neuron] * 0.25,
                size=n_i_neurons,
            )
        )
        full_mat[i_idx, e_neuron] = -i_to_e_weights  # current-based inhibition

    # E to I: homogeneous
    full_mat[np.ix_(e_idx, i_idx)] = e_i_mean

    # I to I: homogeneous
    full_mat[np.ix_(i_idx, i_idx)] = -i_i_mean

    if not split:
        return full_mat, e_idx, i_idx

    # Split by presynaptic neuron class
    e_matrix = np.zeros_like(full_mat)
    e_matrix[e_idx, :] = full_mat[e_idx, :]

    i_matrix = np.zeros_like(full_mat)
    i_matrix[i_idx, :] = -full_mat[i_idx, :]

    return e_matrix, i_matrix, e_idx, i_idx


def build_sparse_mat(
    n_e_neurons: int,
    n_i_neurons: int,
    split: bool = False,
    density: float = 1.0,
    i_e_ratio=100,
    e_to_e_mean=4.0e-3,
    e_to_e_std=1.9e-3,
    e_i_mean=5e-2,
    i_i_mean=25e-4,
):
    """Build a sparse neural connection matrix without dense intermediates.

    E-E, E-I, and I-I blocks receive exactly the floor of ``block_size *
    density`` edges. The historical I-E block gives every E target exactly
    ``floor(n_i_neurons * density)`` incoming I edges. Within each sampled
    block, row counts differ by at most one and the remainder rows are
    randomized; columns are sampled without replacement per row. This
    preserves the old block counts and uniqueness while avoiding all ``N*N``
    arrays.

    The sparse recipe intentionally differs from :func:`build_dense_mat`: E-E
    lognormal weights are drawn only for selected edges, and I-E weight scales
    use the empirical mean of selected incoming E-E edges per target. Targets
    with no selected E-E input fall back to ``e_to_e_mean``. This changes the
    random-number sequence by design and avoids a hidden dense draw.

    Args:
        n_e_neurons (int): The number of excitatory neurons.
        n_i_neurons (int): The number of inhibitory neurons.
        split (bool, optional): If True, returns separate excitatory
                                and inhibitory matrices. Defaults to False.
        density (float, optional): The fraction of connections to keep. A value of 1.0
                                    means a dense matrix (all connections).
                                    Defaults to 1.0.
    Returns:
        tuple: If split is False, returns a single sparse matrix and the neuron indices.
               If split is True, returns two sparse matrices and the neuron indices.
    """
    if not 0.0 <= density <= 1.0:
        raise ValueError("Density must be between 0.0 and 1.0")

    n_neurons = n_e_neurons + n_i_neurons
    e_idx = np.arange(n_e_neurons)
    i_idx = np.arange(n_e_neurons, n_neurons)

    n_ee = _block_count(n_e_neurons, n_e_neurons, density)
    # Preserve the historical I-to-E topology: every excitatory target gets
    # exactly floor(n_i * density) selected inhibitory presynaptic neurons.
    n_ie = n_e_neurons * int(n_i_neurons * density)
    n_ei = _block_count(n_e_neurons, n_i_neurons, density)
    n_ii = _block_count(n_i_neurons, n_i_neurons, density)
    n_e_edges = n_ee + n_ei
    n_i_edges = n_ie + n_ii
    n_edges = n_e_edges + n_i_edges

    # Store E rows first and I rows second. Split outputs are views into this
    # one backing allocation; a full signed matrix is materialized only when
    # the caller requests the unsplit return layout.
    all_rows = np.empty(n_edges, dtype=int)
    all_cols = np.empty(n_edges, dtype=int)
    all_vals = np.empty(n_edges, dtype=float)
    e_rows = all_rows[:n_e_edges]
    e_cols = all_cols[:n_e_edges]
    e_vals = all_vals[:n_e_edges]
    i_rows = all_rows[n_e_edges:]
    i_cols = all_cols[n_e_edges:]
    i_vals = all_vals[n_e_edges:]

    # E to E lognormal. Only selected edges receive a random weight.
    m = np.log(e_to_e_mean**2 / np.sqrt(e_to_e_std**2 + e_to_e_mean**2))
    s = np.sqrt(np.log(1 + (e_to_e_std**2 / e_to_e_mean**2)))
    ee_slice = slice(0, n_ee)
    _fill_random_block(e_idx, e_idx, n_ee, (all_rows[ee_slice], all_cols[ee_slice]))
    all_vals[ee_slice] = np.exp(np.random.normal(m, s, size=n_ee))

    incoming_count = np.bincount(all_cols[ee_slice], minlength=n_e_neurons)
    incoming_sum = np.bincount(
        all_cols[ee_slice], weights=all_vals[ee_slice], minlength=n_e_neurons
    )
    mean_e_weights = np.full(n_e_neurons, e_to_e_mean, dtype=float)
    nonempty = incoming_count != 0
    mean_e_weights[nonempty] = incoming_sum[nonempty] / incoming_count[nonempty]

    # I to E Gaussian. Rows are inhibitory presynaptic neurons and columns are
    # excitatory targets; scales follow selected incoming E-E means.
    ie_start = n_e_edges
    ie_slice = slice(ie_start, ie_start + n_ie)
    # The old implementation iterated over each E target and sampled I
    # presynaptic columns. Swap the target arrays so this helper retains that
    # exact per-target count and random topology convention.
    _fill_random_block(e_idx, i_idx, n_ie, (all_cols[ie_slice], all_rows[ie_slice]))
    ie_scale = mean_e_weights[all_cols[ie_slice]] * i_e_ratio
    ie_positive = np.abs(np.random.normal(ie_scale, ie_scale * 0.25, size=n_ie))
    if split:
        i_vals[:n_ie] = ie_positive
    else:
        np.negative(ie_positive, out=all_vals[ie_slice])

    # E to I: homogeneous positive weights.
    ei_start = n_ee
    ei_slice = slice(ei_start, ei_start + n_ei)
    _fill_random_block(e_idx, i_idx, n_ei, (all_rows[ei_slice], all_cols[ei_slice]))
    all_vals[ei_slice] = e_i_mean

    # I to I: homogeneous inhibitory weights in the full matrix.
    ii_start = n_e_edges + n_ie
    ii_slice = slice(ii_start, ii_start + n_ii)
    _fill_random_block(i_idx, i_idx, n_ii, (all_rows[ii_slice], all_cols[ii_slice]))
    i_positive = np.full(n_ii, i_i_mean, dtype=float)
    if split:
        i_vals[n_ie:] = i_positive
    else:
        np.negative(i_positive, out=all_vals[ii_slice])

    if not split:
        full_matrix = scipy.sparse.coo_array(
            (all_vals, (all_rows, all_cols)), shape=(n_neurons, n_neurons)
        )
        return full_matrix, e_idx, i_idx

    e_matrix = scipy.sparse.coo_array(
        (e_vals, (e_rows, e_cols)), shape=(n_neurons, n_neurons)
    )
    i_matrix = scipy.sparse.coo_array(
        (i_vals, (i_rows, i_cols)), shape=(n_neurons, n_neurons)
    )

    return e_matrix, i_matrix, e_idx, i_idx
