from collections.abc import Sequence

import numpy as np
import scipy
import torch

from . import base
from .shape import expand_leading_dims
from .types import TensorLike


def _tensor_with_default(
    v: float | TensorLike | None, cond: bool, default, default_fallback, device
) -> torch.Tensor:
    if v is None:
        if cond:
            v = default
        else:
            # fallback generic range
            v = default_fallback
    return torch.as_tensor(v, device=device)


# store the random init information for a network somewhere
@torch.no_grad()
def uniform_state_(
    neuron: base.BaseNode,
    name: str | Sequence[str],
    *,
    low: float | TensorLike | None = None,
    high: float | TensorLike | None = None,
    set_reset_value: bool = False,
    rand_batch: bool = False,
    rng: torch.Generator | int | None = None,
) -> base.BaseNode:
    """Uniformly initialize **any state variable(s)** of a neuron.

    Args:
        neuron: BaseNode instance.
        name: variable name or list of names (e.g., "v", ["v", "w"]).
        low, high:
            Range for initialization. If None:
                - if name == "v": uses (v_reset, v_threshold)
                - else: uses (0, 1)
        set_reset_value:
            If True, also sets reset value for this variable.
        rand_batch:
            Distinct value per batch entry (True)
            or shared across batch (False).
        rng:
            Seed or generator for reproducibility.

    Returns:
        neuron (modified in place)
    """

    # Convert names to list
    if isinstance(name, str):
        names = [name]
    else:
        names = list(name)

    # Setup RNG
    if isinstance(rng, int):
        generator = torch.Generator(device=next(neuron.parameters()).device)
        generator.manual_seed(rng)
    else:
        generator = rng

    for var in names:
        x: torch.Tensor = getattr(neuron, var)
        batch_size = neuron._batch_dim_detect(var)

        # ----- default range -----
        lo = _tensor_with_default(low, var == "v", neuron.v_reset, 0.0, x.device)
        hi = _tensor_with_default(high, var == "v", neuron.v_threshold, 1.0, x.device)

        # ----- random sampling -----
        if rand_batch:
            # unique value for each batch entry
            val = torch.rand(
                x.shape, dtype=x.dtype, device=x.device, generator=generator
            )
            val = val * (hi - lo) + lo

            if set_reset_value:
                neuron.set_reset_value(var, val, has_batch=batch_size is not None)

        else:
            # shared across batch
            shared_shape = x.shape[0 if batch_size is None else len(batch_size) :]
            shared = torch.rand(
                shared_shape, dtype=x.dtype, device=x.device, generator=generator
            )
            shared = shared * (hi - lo) + lo

            if batch_size is not None:
                val = expand_leading_dims(shared, batch_size, view=False)
            else:
                val = shared

            if set_reset_value:
                neuron.set_reset_value(var, shared)

        setattr(neuron, var, val)

    return neuron


def uniform_v_(
    neuron: base.BaseNode,
    *,
    low: float | TensorLike | None = None,
    high: float | TensorLike | None = None,
    set_reset_value: bool = False,
    rand_batch: bool = False,
    rng: torch.Generator | int | None = None,
):
    return uniform_state_(
        neuron,
        "v",
        low=low,
        high=high,
        set_reset_value=set_reset_value,
        rand_batch=rand_batch,
        rng=rng,
    )


# TODO: for demo purpose, not supposed to be here


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
    """Builds a sparse matrix representing neural connections with a specified
    density.

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

    all_rows, all_cols, all_vals = [], [], []
    e_rows_list, e_cols_list, e_vals_list = [], [], []
    i_rows_list, i_cols_list, i_vals_list = [], [], []

    # E to E lognormal

    m = np.log(e_to_e_mean**2 / np.sqrt(e_to_e_std**2 + e_to_e_mean**2))
    s = np.sqrt(np.log(1 + (e_to_e_std**2 / e_to_e_mean**2)))
    e_to_e_weights = np.exp(
        np.random.normal(loc=m, scale=s, size=(n_e_neurons, n_e_neurons))
    )

    e_rows, e_cols = np.meshgrid(e_idx, e_idx, indexing="ij")
    flat_rows = e_rows.flatten()
    flat_cols = e_cols.flatten()
    flat_vals = e_to_e_weights.flatten()

    # Apply density for E to E
    num_e_to_e_connections = int(len(flat_rows) * density)
    if density < 1.0:
        keep_indices = np.random.choice(
            len(flat_rows), num_e_to_e_connections, replace=False
        )
        flat_rows = flat_rows[keep_indices]
        flat_cols = flat_cols[keep_indices]
        flat_vals = flat_vals[keep_indices]

    all_rows.append(flat_rows)
    all_cols.append(flat_cols)
    all_vals.append(flat_vals)
    e_rows_list.append(flat_rows)
    e_cols_list.append(flat_cols)
    e_vals_list.append(flat_vals)

    # I to E Gaussian
    mean_e_weights = e_to_e_weights.mean(axis=0)
    mean_i_weights = mean_e_weights * i_e_ratio

    for e_neuron in e_idx:
        i_to_e_weights = np.abs(
            np.random.normal(
                loc=mean_i_weights[e_neuron],
                scale=mean_i_weights[e_neuron] * 0.25,
                size=n_i_neurons,
            )
        )
        pre = i_idx
        post = np.full(n_i_neurons, e_neuron)
        vals = i_to_e_weights

        # Apply density for I to E
        num_i_to_e_connections = int(len(pre) * density)
        if density < 1.0:
            keep_indices = np.random.choice(
                len(pre), num_i_to_e_connections, replace=False
            )
            pre = pre[keep_indices]
            post = post[keep_indices]
            vals = vals[keep_indices]

        all_rows.append(pre)
        all_cols.append(post)
        all_vals.append(-vals)
        i_rows_list.append(pre)
        i_cols_list.append(post)
        i_vals_list.append(vals)

    # E to I: homogeneous
    e_rows, i_cols = np.meshgrid(e_idx, i_idx, indexing="ij")
    flat_rows = e_rows.flatten()
    flat_cols = i_cols.flatten()
    flat_vals = np.full(flat_rows.size, e_i_mean)

    # Apply density for E to I
    num_e_to_i_connections = int(len(flat_rows) * density)
    if density < 1.0:
        keep_indices = np.random.choice(
            len(flat_rows), num_e_to_i_connections, replace=False
        )
        flat_rows = flat_rows[keep_indices]
        flat_cols = flat_cols[keep_indices]
        flat_vals = flat_vals[keep_indices]

    all_rows.append(flat_rows)
    all_cols.append(flat_cols)
    all_vals.append(flat_vals)
    e_rows_list.append(flat_rows)
    e_cols_list.append(flat_cols)
    e_vals_list.append(flat_vals)

    # I to I: homogeneous
    i_rows, i_cols = np.meshgrid(i_idx, i_idx, indexing="ij")
    flat_rows = i_rows.flatten()
    flat_cols = i_cols.flatten()
    flat_vals = np.full(flat_rows.size, i_i_mean)

    # Apply density for I to I
    num_i_to_i_connections = int(len(flat_rows) * density)
    if density < 1.0:
        keep_indices = np.random.choice(
            len(flat_rows), num_i_to_i_connections, replace=False
        )
        flat_rows = flat_rows[keep_indices]
        flat_cols = flat_cols[keep_indices]
        flat_vals = flat_vals[keep_indices]

    all_rows.append(flat_rows)
    all_cols.append(flat_cols)
    all_vals.append(-flat_vals)
    i_rows_list.append(flat_rows)
    i_cols_list.append(flat_cols)
    i_vals_list.append(flat_vals)

    # Construct matrices
    all_rows = np.concatenate(all_rows)
    all_cols = np.concatenate(all_cols)
    all_vals = np.concatenate(all_vals)

    full_matrix = scipy.sparse.coo_array(
        (all_vals, (all_rows, all_cols)), shape=(n_neurons, n_neurons)
    )

    if not split:
        return full_matrix, e_idx, i_idx

    e_matrix = scipy.sparse.coo_array(
        (
            np.concatenate(e_vals_list),
            (np.concatenate(e_rows_list), np.concatenate(e_cols_list)),
        ),
        shape=(n_neurons, n_neurons),
    )

    i_matrix = scipy.sparse.coo_array(
        (
            np.concatenate(i_vals_list),
            (np.concatenate(i_rows_list), np.concatenate(i_cols_list)),
        ),
        shape=(n_neurons, n_neurons),
    )

    return e_matrix, i_matrix, e_idx, i_idx
