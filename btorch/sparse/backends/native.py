from __future__ import annotations

import torch


def sparse_mm(
    matrix,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute ``x @ W`` using native PyTorch sparse ops."""
    if x.shape[-1] != matrix.shape[0]:
        raise ValueError(
            f"Expected input size {matrix.shape[0]}, received {x.shape[-1]}."
        )
    return matrix.mm(x)
