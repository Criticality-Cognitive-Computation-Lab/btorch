import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btorch.utils.file import save_fig
from btorch.visualisation.aggregation import (
    plot_group_box,
    plot_group_ecdf,
    plot_group_violin,
    plot_neuropil_timeseries_overview,
    plot_neuropil_timeseries_panels,
)


def _make_realistic_grouped_data(
    *,
    n_trials: int = 40,
    n_neurons_per_group: int = 30,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Create a realistic grouped dataset with distinct distribution shapes.

    The three groups are intentionally different so users can see why they might
    pick violin vs box vs ECDF in practice:
    - `visual`: approximately Gaussian responses.
    - `olfactory`: right-skewed responses with bursty outliers.
    - `motor`: broader, slightly bimodal responses.
    """
    rng = np.random.default_rng(7)
    group_order = ["visual", "olfactory", "motor"]

    neurons = []
    values_by_group = []
    for group in group_order:
        n = n_neurons_per_group
        simple_ids = np.arange(len(neurons), len(neurons) + n)

        if group == "visual":
            neuron_baseline = rng.normal(loc=0.20, scale=0.06, size=n)
            trial_noise = rng.normal(loc=0.0, scale=0.10, size=(n_trials, n))
            group_values = neuron_baseline + trial_noise
        elif group == "olfactory":
            neuron_baseline = rng.normal(loc=0.05, scale=0.04, size=n)
            trial_noise = rng.normal(loc=0.0, scale=0.08, size=(n_trials, n))
            burst = rng.exponential(scale=0.35, size=(n_trials, n))
            burst_mask = rng.random((n_trials, n)) < 0.12
            group_values = neuron_baseline + trial_noise + burst * burst_mask
        else:  # motor
            subtype_shift = rng.choice([-0.16, 0.16], size=n)
            neuron_baseline = rng.normal(loc=0.10, scale=0.07, size=n) + subtype_shift
            trial_noise = rng.normal(loc=0.0, scale=0.12, size=(n_trials, n))
            group_values = neuron_baseline + trial_noise

        values_by_group.append(group_values)
        neurons.extend(
            {
                "simple_id": int(simple_id),
                "neuropil": group,
            }
            for simple_id in simple_ids
        )

    values = np.concatenate(values_by_group, axis=1)
    neurons_df = pd.DataFrame(neurons)
    return values, neurons_df, group_order


def test_group_distribution_style_comparison_plot():
    """Render one practical comparison figure across violin/box/ECDF styles."""
    values, neurons_df, group_order = _make_realistic_grouped_data()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    common = {
        "neurons_df": neurons_df,
        "group_by": "neuropil",
        "group_order": group_order,
        "value_name": "response",
    }

    plot_group_violin(
        values,
        ax=axes[0],
        title="Violin: Distribution Shape",
        **common,
    )
    plot_group_box(
        values,
        ax=axes[1],
        title="Box: Quantiles and Spread",
        showfliers=False,
        **common,
    )
    plot_group_ecdf(
        values,
        ax=axes[2],
        title="ECDF: Cumulative Comparison",
        linewidth=2.0,
        **common,
    )

    # Ensure group order is respected in categorical axes.
    expected_labels = group_order
    violin_labels = [tick.get_text() for tick in axes[0].get_xticklabels()]
    box_labels = [tick.get_text() for tick in axes[1].get_xticklabels()]
    assert violin_labels == expected_labels
    assert box_labels == expected_labels

    # ECDF should contain one line per group.
    assert len(axes[2].lines) == len(group_order)

    fig.tight_layout()
    save_fig(fig, name="group_distribution_style_comparison")
    plt.close(fig)


def _make_neuropil_trace_dict(n_time: int = 120) -> dict[str, np.ndarray]:
    t = np.linspace(0, 4 * np.pi, n_time)
    return {
        "AL": np.sin(t) * 0.6,
        "MB": np.cos(t) * 0.4 + 0.2,
        "LH": np.linspace(-0.2, 0.5, n_time),
    }


def test_plot_neuropil_timeseries_overview_wave():
    traces = _make_neuropil_trace_dict()

    fig, ax = plot_neuropil_timeseries_overview(
        traces,
        dt=0.01,
        kind="wave",
        top_n=3,
    )

    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Neuropil Activity (z-scored)"
    save_fig(fig, name="neuropil_overview_wave")
    plt.close(fig)


def test_plot_neuropil_timeseries_panels_grid():
    traces = _make_neuropil_trace_dict(n_time=100)

    fig, _ = plot_neuropil_timeseries_panels(
        traces,
        dt=0.02,
        regions=["AL", "MB", "LH"],
        cols=2,
    )

    # Three selected regions should create three active axes.
    assert len(fig.axes) == 3
    save_fig(fig, name="neuropil_panels")
    plt.close(fig)
