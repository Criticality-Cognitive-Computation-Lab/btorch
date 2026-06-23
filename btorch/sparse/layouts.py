from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PaddedCSRLayout:
    """Presynaptic-row padded structural layout with packed edge values."""

    row_length: torch.Tensor
    row_offset: torch.Tensor
    indices: torch.Tensor
    row_stride: int
