import numpy as np
import pandas as pd
import scipy.sparse

from btorch.analysis.aggregation import (
    agg_by_neuron,
    agg_conn,
    build_group_frame,
    group_ecdf,
    group_summary,
)


def _make_realistic_grouped_data(
    *,
    n_trials: int = 30,
    n_neurons_per_group: int = 20,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Generate heterogeneous grouped responses for analysis tests."""
    rng = np.random.default_rng(11)
    group_order = ["visual", "olfactory", "motor"]

    neurons = []
    values_by_group = []
    for group in group_order:
        n = n_neurons_per_group
        simple_ids = np.arange(len(neurons), len(neurons) + n)

        if group == "visual":
            group_values = rng.normal(0.2, 0.12, size=(n_trials, n))
        elif group == "olfactory":
            base = rng.normal(0.05, 0.09, size=(n_trials, n))
            bursts = rng.exponential(0.30, size=(n_trials, n))
            mask = rng.random((n_trials, n)) < 0.10
            group_values = base + bursts * mask
        else:
            shift = rng.choice([-0.15, 0.15], size=n)
            group_values = rng.normal(0.1 + shift, 0.14, size=(n_trials, n))

        values_by_group.append(group_values)
        neurons.extend(
            {"simple_id": int(simple_id), "neuropil": group} for simple_id in simple_ids
        )

    values = np.concatenate(values_by_group, axis=1)
    neurons_df = pd.DataFrame(neurons)
    return values, neurons_df, group_order


def test_agg_by_neuron_groups_by_cell_type():
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    neurons = pd.DataFrame(
        {
            "simple_id": [0, 1, 2],
            "cell_type": ["A", "B", "A"],
        }
    )

    result = agg_by_neuron(y, neurons, agg="mean")
    np.testing.assert_allclose(result["A"], np.array([2.0, 5.0]))
    np.testing.assert_allclose(result["B"], np.array([2.0, 5.0]))


def test_agg_conn_with_sparse_weights_by_neuropil():
    conn = pd.DataFrame(
        {
            "pre_simple_id": [0, 1],
            "post_simple_id": [1, 0],
            "neuropil": ["alpha", "alpha"],
        }
    )
    weights = scipy.sparse.coo_array(
        (
            np.array([0.5, 1.5]),
            (np.array([0, 1]), np.array([1, 0])),
        ),
        shape=(2, 2),
    )

    aggregated = agg_conn(
        y=np.array([]), conn=conn, conn_weight=weights, mode="neuropil", agg="sum"
    )

    np.testing.assert_allclose(aggregated.loc["alpha"], 2.0)


def test_build_group_frame_and_summary_preserve_statistics():
    """Grouped long-format data should preserve exact per-group means."""
    values, neurons_df, group_order = _make_realistic_grouped_data()

    frame = build_group_frame(
        values,
        neurons_df,
        group_by="neuropil",
        value_name="response",
    )
    summary = group_summary(
        values,
        neurons_df,
        group_by="neuropil",
        value_name="response",
        group_order=group_order,
    )

    assert len(frame) == values.shape[0] * values.shape[1]
    assert summary["neuropil"].tolist() == group_order

    for group in group_order:
        group_ids = neurons_df.loc[
            neurons_df["neuropil"] == group, "simple_id"
        ].to_numpy(dtype=np.int64)
        expected_mean = values[..., group_ids].reshape(-1).mean()
        actual_mean = summary.loc[summary["neuropil"] == group, "mean"].item()
        assert np.isclose(actual_mean, expected_mean)


def test_group_ecdf_is_sorted_and_bounded():
    """ECDF outputs should be monotonic and finish at 1.0 for every group."""
    values, neurons_df, group_order = _make_realistic_grouped_data()
    ecdf_by_group = group_ecdf(
        values,
        neurons_df,
        group_by="neuropil",
        value_name="response",
        group_order=group_order,
    )

    assert list(ecdf_by_group.keys()) == group_order
    for group in group_order:
        ecdf = ecdf_by_group[group]
        x = ecdf["response"].to_numpy()
        y = ecdf["ecdf"].to_numpy()
        assert np.all(np.diff(x) >= 0)
        assert np.all(np.diff(y) > 0)
        assert np.isclose(y[0], 1.0 / len(y))
        assert np.isclose(y[-1], 1.0)
