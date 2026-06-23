from __future__ import annotations

from dataclasses import dataclass

import torch


class Parameterization:
    def effective_values(self) -> torch.Tensor:
        raise NotImplementedError

    def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]:
        raise NotImplementedError

    def _fixed_tensors(self) -> list[tuple[str, torch.Tensor]]:
        raise NotImplementedError

    def _rebuild(self, parameters: dict, buffers: dict) -> "Parameterization":
        raise NotImplementedError


@dataclass
class GroupedMagnitude(Parameterization):
    """Values[i] = initial_weight[i] * magnitude[group_index[i]]

    initial_weight : (nnz,)      fixed buffer from connectome, never trained
    group_index    : (nnz,)      int64, structural — maps each edge to its group
    magnitude      : (n_groups,) the only trainable parameter
    dale           : bool         if True, enforce magnitude >= 0 by clamping
    """

    initial_weight: torch.Tensor
    group_index: torch.Tensor
    magnitude: torch.Tensor
    dale: bool = False

    def effective_values(self) -> torch.Tensor:
        return self.initial_weight * self.magnitude[self.group_index]

    def _trainable_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [("magnitude", self.magnitude)]

    def _fixed_tensors(self) -> list[tuple[str, torch.Tensor]]:
        return [
            ("initial_weight", self.initial_weight),
            ("group_index", self.group_index),
        ]

    def _rebuild(self, parameters: dict, buffers: dict) -> "GroupedMagnitude":
        return GroupedMagnitude(
            initial_weight=buffers["initial_weight"],
            group_index=buffers["group_index"],
            magnitude=parameters["magnitude"],
            dale=self.dale,
        )
