"""Tests for connectome augmentation sampling helpers.

These tests validate that `sample_neuron_representative` remains a stable,
intentional public API over the lower-level generic sampler.
"""

import pandas as pd
import pytest

from btorch.connectome.augment import (
    sample_by_column_expand_none,
    sample_neuron_representative,
)


def _build_neurons_fixture() -> pd.DataFrame:
    """Build a compact metadata table with full and partial taxonomy rows."""
    return pd.DataFrame(
        [
            # Fully specified groups
            {
                "root_id": 1,
                "flow": "sensorimotor",
                "super_class": "A",
                "class": "alpha",
                "nt_type": "GABA",
            },
            {
                "root_id": 2,
                "flow": "sensorimotor",
                "super_class": "A",
                "class": "alpha",
                "nt_type": "GABA",
            },
            {
                "root_id": 3,
                "flow": "sensorimotor",
                "super_class": "B",
                "class": "beta",
                "nt_type": "ACh",
            },
            # Partially specified rows (with nulls)
            {
                "root_id": 4,
                "flow": "sensorimotor",
                "super_class": "A",
                "class": "alpha",
                "nt_type": None,
            },
            {
                "root_id": 5,
                "flow": "visual",
                "super_class": "C",
                "class": None,
                "nt_type": "Glu",
            },
            {
                "root_id": 6,
                "flow": "visual",
                "super_class": None,
                "class": "gamma",
                "nt_type": "Glu",
            },
        ]
    )


def test_sample_neuron_representative_matches_generic_sampler():
    """Wrapper should preserve exact sampling semantics for canonical columns.

    This protects downstream code that relies on current representative
    sampling behavior while still allowing internal implementation
    cleanup.
    """
    neurons = _build_neurons_fixture()

    expected = sample_by_column_expand_none(
        neurons,
        ["flow", "super_class", "class", "nt_type"],
        k=1,
        j=2,
        random_state=7,
        product_sample=True,
    ).sort_values("root_id", ignore_index=True)
    actual = sample_neuron_representative(
        neurons,
        k=1,
        j=2,
        random_state=7,
        product_sample=True,
    ).sort_values("root_id", ignore_index=True)

    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_sample_neuron_representative_validates_required_columns():
    """The wrapper should fail fast when taxonomy metadata is incomplete."""
    bad = _build_neurons_fixture().drop(columns=["nt_type"])

    with pytest.raises(
        ValueError,
        match="sample_neuron_representative requires columns",
    ):
        sample_neuron_representative(bad, k=1, j=1)
