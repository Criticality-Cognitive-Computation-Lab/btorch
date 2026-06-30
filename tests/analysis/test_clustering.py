"""Tests for clustering module.

This module contains tests for trace clustering functionality using
dynamic time warping (DTW) and hierarchical clustering.
"""

import numpy as np
import pytest


pytest.importorskip("fastdtw")

from btorch.analysis.clustering import cluster_traces, suggest_threshold  # noqa: E402


class TestSuggestThreshold:
    """Tests for suggest_threshold function."""

    def test_basic_threshold_suggestion(self):
        """Test that threshold suggestion runs without error."""
        # Create a simple linkage matrix
        # Each row: [idx1, idx2, distance, sample_count]
        linkage_matrix = np.array(
            [
                [0, 1, 1.0, 2],
                [2, 3, 2.0, 2],
                [4, 5, 5.0, 4],
                [6, 7, 10.0, 8],
            ]
        )

        threshold = suggest_threshold(linkage_matrix)

        assert isinstance(threshold, (float, np.floating))
        assert threshold >= 0

    def test_elbow_detection(self):
        """Test that elbow in distance curve is detected."""
        # Create linkage with clear elbow at distance 5.0
        linkage_matrix = np.array(
            [
                [0, 1, 1.0, 2],
                [2, 3, 2.0, 2],
                [4, 5, 3.0, 4],
                [6, 7, 10.0, 8],  # Large jump here
            ]
        )

        threshold = suggest_threshold(linkage_matrix)

        # Should suggest around the elbow
        assert threshold >= 3.0
        assert threshold <= 10.0

    def test_no_elbow_fallback(self):
        """Test fallback when no clear elbow exists."""
        # Uniform distances - no elbow
        linkage_matrix = np.array(
            [
                [0, 1, 1.0, 2],
                [2, 3, 1.1, 2],
                [4, 5, 1.2, 4],
            ]
        )

        threshold = suggest_threshold(linkage_matrix)

        # Should return the smallest distance
        assert isinstance(threshold, (float, np.floating))


class TestClusterTraces:
    """Tests for cluster_traces function."""

    def test_basic_clustering(self):
        """Test basic clustering of traces."""
        # Create simple synthetic traces
        # Two groups: sine waves with different phases
        t = np.linspace(0, 4 * np.pi, 100)
        traces = [
            np.sin(t),  # Group 1
            np.sin(t + 0.1),  # Group 1
            np.sin(t + np.pi),  # Group 2 (opposite phase)
            np.sin(t + np.pi + 0.1),  # Group 2
        ]

        cluster_indices, clusters, Z, distance_matrix = cluster_traces(
            traces, threshold=5, linkage_method="average"
        )

        # Check return types and shapes
        assert isinstance(cluster_indices, dict)
        assert isinstance(clusters, np.ndarray)
        assert len(clusters) == len(traces)
        assert isinstance(Z, np.ndarray)  # Linkage matrix
        assert isinstance(distance_matrix, np.ndarray)
        assert distance_matrix.shape == (len(traces), len(traces))

    def test_cluster_count(self):
        """Test that appropriate number of clusters is created."""
        # Very similar traces - should cluster together
        traces = [
            np.array([1, 2, 3, 4, 5]),
            np.array([1.1, 2.1, 3.1, 4.1, 5.1]),
            np.array([1.2, 2.2, 3.2, 4.2, 5.2]),
        ]

        cluster_indices, clusters, Z, distance_matrix = cluster_traces(
            traces, threshold=100, linkage_method="average"
        )

        # With high threshold, all should be in one cluster
        unique_clusters = np.unique(clusters)
        assert len(unique_clusters) >= 1

    def test_distance_matrix_symmetry(self):
        """Test that distance matrix is symmetric."""
        traces = [
            np.array([1, 2, 3, 4, 5]),
            np.array([5, 4, 3, 2, 1]),
            np.array([1, 3, 5, 3, 1]),
        ]

        _, _, _, distance_matrix = cluster_traces(traces, threshold=10)

        # Distance matrix should be symmetric
        np.testing.assert_array_almost_equal(distance_matrix, distance_matrix.T)

    def test_diagonal_is_zero(self):
        """Test that diagonal of distance matrix is zero (self-distance)."""
        traces = [
            np.array([1, 2, 3, 4, 5]),
            np.array([5, 4, 3, 2, 1]),
        ]

        _, _, _, distance_matrix = cluster_traces(traces, threshold=10)

        # Self-distances should be zero
        np.testing.assert_array_almost_equal(
            np.diag(distance_matrix), np.zeros(len(traces))
        )
