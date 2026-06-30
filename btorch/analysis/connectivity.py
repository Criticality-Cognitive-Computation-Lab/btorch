"""Connectivity analysis utilities for neural networks.

This module provides tools for analyzing connectivity properties of
neural circuits, including:

- Inhibitory/excitatory (IE) ratio computation
- Hop distance analysis using BFS
- Network reachability statistics
- Shortest path reconstruction

Expected DataFrame schemas:
    neurons DataFrame:
        - simple_id: Linear neuron index (0..N-1)
        - root_id: Unique neuron identifier
        - EI: Excitatory/inhibitory type ("E" or "I")

    connections/edges DataFrame:
        - pre_simple_id: Presynaptic neuron simple_id
        - post_simple_id: Postsynaptic neuron simple_id
        - syn_count: Number of synapses (optional)

The `HopDistanceModel` class provides efficient hop distance computation
using either sparse matrix or edge list representations.
"""

from collections import deque
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import sparray


def compute_ie_ratio(
    excitatory_mat: sparray,
    inhibitory_mat: sparray,
    excitatory_neuron_only: bool = True,
    neurons: Optional[pd.DataFrame] = None,
    warn_strict: bool = True,
) -> tuple[float, np.ndarray]:
    """Compute inhibitory/excitatory ratio per neuron and whole-brain mean."""
    if excitatory_neuron_only:
        assert neurons is not None

    sum_inhibitory = inhibitory_mat.sum(axis=0).astype(float)
    sum_excitatory = excitatory_mat.sum(axis=0).astype(float)

    # neurons that have zero inputs from both e and i connections are likely input,
    # so don't warn on them
    input_indices = (sum_excitatory == 0) & (sum_inhibitory == 0)

    sum_excitatory[sum_excitatory == 0] = np.nan
    if excitatory_neuron_only:
        sum_excitatory[neurons[neurons.EI == "E"].simple_id.to_numpy()] = np.nan

    ie_ratios = sum_inhibitory / sum_excitatory

    if warn_strict:
        nan_indices = (np.isnan(ie_ratios) & ~input_indices).nonzero()[0]
        if nan_indices.size > 0:
            print(f"Warning: IE ratio contains NaN values at indices {nan_indices}")

        inf_indices = np.isinf(ie_ratios).nonzero()[0]
        if inf_indices.size > 0:
            print(f"Warning: IE ratio contains Inf values at indices {inf_indices}")

    ie_ratios = np.where(np.isinf(ie_ratios), np.nan, ie_ratios)
    ie_ratio_whole_brain: float = np.nanmean(ie_ratios[ie_ratios != 0])

    return ie_ratio_whole_brain, ie_ratios


class HopDistanceModel:
    """Fast computation of hop distances in networks using BFS."""

    def __init__(
        self,
        edges: Optional[pd.DataFrame] = None,
        adjacency: Optional[sparse.sparray] = None,
        node_mapping: Optional[Dict] = None,
        source: str = "source",
        target: str = "target",
    ):
        if edges is None and adjacency is None:
            raise ValueError("Must provide either edges DataFrame or adjacency matrix")

        if adjacency is not None:
            self.adjacency = adjacency.tocsr()
            self.use_sparse = True
            if node_mapping is None:
                self.node_mapping = {i: i for i in range(adjacency.shape[0])}
                self.reverse_mapping = {i: i for i in range(adjacency.shape[0])}
            else:
                self.node_mapping = node_mapping
                self.reverse_mapping = {v: k for k, v in node_mapping.items()}
            self.n_nodes = adjacency.shape[0]
        else:
            self.edges = edges.copy()
            self.source = source
            self.target = target
            self.use_sparse = False

            assert source in edges.columns, f'edges must contain "{source}" column'
            assert target in edges.columns, f'edges must contain "{target}" column'

            all_nodes = np.unique(
                np.concatenate([edges[source].values, edges[target].values])
            )
            self.node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
            self.reverse_mapping = {
                idx: node for node, idx in self.node_mapping.items()
            }
            self.n_nodes = len(all_nodes)

            self._build_adjacency_dict()

    def _build_adjacency_dict(self):
        """Build adjacency dictionary for fast neighbor lookup."""
        self.adj_dict = {}
        for _, row in self.edges.iterrows():
            src_idx = self.node_mapping[row[self.source]]
            tgt_idx = self.node_mapping[row[self.target]]

            if src_idx not in self.adj_dict:
                self.adj_dict[src_idx] = []
            self.adj_dict[src_idx].append(tgt_idx)

        for node in self.adj_dict:
            self.adj_dict[node] = np.array(self.adj_dict[node], dtype=np.int32)

    def compute_distances(
        self, seeds: List[Union[int, str]], max_hops: Optional[int] = None
    ) -> pd.DataFrame:
        """Compute hop distances from seeds to all reachable nodes using
        BFS."""
        seed_indices = [self.node_mapping.get(seed, None) for seed in seeds]
        missing = [s for s in seeds if s not in self.node_mapping]
        if len(missing) != 0:
            print(f"Seeds not found in network: {missing}")
            seed_indices = [s for s in seed_indices if s is not None]

        distances = np.full(self.n_nodes, -1, dtype=np.int32)
        predecessors = np.full(self.n_nodes, -1, dtype=np.int32)

        queue = deque()
        for seed_idx in seed_indices:
            distances[seed_idx] = 0
            predecessors[seed_idx] = seed_idx
            queue.append(seed_idx)

        while queue:
            current_idx = queue.popleft()
            current_dist = distances[current_idx]

            if max_hops is not None and current_dist >= max_hops:
                continue

            if self.use_sparse:
                neighbors = self.adjacency[[current_idx]].indices.flatten()
            else:
                neighbors = self.adj_dict.get(current_idx, np.array([], dtype=np.int32))

            for neighbor_idx in neighbors:
                if distances[neighbor_idx] == -1:
                    distances[neighbor_idx] = current_dist + 1
                    predecessors[neighbor_idx] = current_idx
                    queue.append(neighbor_idx)

        reachable_mask = distances >= 0
        result_nodes = [
            self.reverse_mapping[idx] for idx in np.where(reachable_mask)[0]
        ]
        result_distances = distances[reachable_mask]
        result_predecessors = [
            self.reverse_mapping[predecessors[idx]]
            for idx in np.where(reachable_mask)[0]
        ]

        return pd.DataFrame(
            {
                "node": result_nodes,
                "distance": result_distances,
                "predecessor": result_predecessors,
            }
        )

    def hop_statistics(
        self, seeds: List[Union[int, str]], max_hops: Optional[int] = None
    ) -> pd.DataFrame:
        """Compute statistics about network reachability by hop distance."""
        distances_df = self.compute_distances(seeds, max_hops)

        if distances_df.empty:
            return pd.DataFrame(
                columns=[
                    "hops",
                    "nodes_count",
                    "nodes_percentage",
                    "cumulative_count",
                    "cumulative_percentage",
                ]
            )

        hop_counts = distances_df["distance"].value_counts().sort_index()

        stats = []
        cumulative = 0
        for hops in range(hop_counts.index.min(), hop_counts.index.max() + 1):
            count = hop_counts.get(hops, 0)
            cumulative += count

            stats.append(
                {
                    "hops": hops,
                    "nodes_count": count,
                    "nodes_percentage": 100.0 * count / self.n_nodes,
                    "cumulative_count": cumulative,
                    "cumulative_percentage": 100.0 * cumulative / self.n_nodes,
                }
            )

        return pd.DataFrame(stats)

    def reconstruct_path(
        self,
        source_node: Union[int, str],
        target_node: Union[int, str],
        distances_df: Optional[pd.DataFrame] = None,
    ) -> List[Union[int, str]]:
        """Reconstruct shortest path from source to target using predecessor
        info."""
        if distances_df is None:
            distances_df = self.compute_distances([source_node])

        target_row = distances_df[distances_df["node"] == target_node]
        if target_row.empty:
            return []

        path = []
        current = target_node
        predecessors_dict = dict(zip(distances_df["node"], distances_df["predecessor"]))

        while current != source_node:
            path.append(current)
            if current not in predecessors_dict:
                return []
            current = predecessors_dict[current]

        path.append(source_node)
        return path[::-1]
