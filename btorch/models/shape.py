from typing import Literal

import torch


def expand_dims(
    tensor: torch.Tensor,
    target_shape: int | tuple[int, ...],
    match_full_shape: bool,
    position: Literal["leading", "trailing"],
    broadcast_only=False,
    view=True,
) -> torch.Tensor:
    if isinstance(target_shape, int):
        target_shape = (target_shape,)

    if match_full_shape:
        expanded_shape = target_shape
        if position == "leading":
            num_new_dims = len(target_shape) - len(tensor.shape)
            tensor = tensor[(None,) * num_new_dims + (...,)]
        else:  # trailing
            num_new_dims = len(target_shape) - len(tensor.shape)
            tensor = tensor[(...,) + (None,) * num_new_dims]
    else:
        if position == "leading":
            expanded_shape = target_shape + tensor.shape
            tensor = tensor[(None,) * len(target_shape) + (...,)]
        else:  # trailing
            expanded_shape = tensor.shape + target_shape
            tensor = tensor[(...,) + (None,) * len(target_shape)]
    if broadcast_only:
        return tensor
    if view:
        return tensor.expand(expanded_shape)
    else:
        return tensor.expand(expanded_shape).clone()


def expand_leading_dims(
    tensor: torch.Tensor,
    target_leading_shape: int | tuple[int, ...],
    match_full_shape: bool = False,
    broadcast_only=False,
    view=True,
) -> torch.Tensor:
    return expand_dims(
        tensor,
        target_leading_shape,
        match_full_shape,
        position="leading",
        view=view,
        broadcast_only=broadcast_only,
    )


def expand_trailing_dims(
    tensor: torch.Tensor,
    target_trailing_shape: int | tuple[int, ...],
    match_full_shape: bool = False,
    broadcast_only: bool = False,
    view=True,
) -> torch.Tensor:
    return expand_dims(
        tensor,
        target_trailing_shape,
        match_full_shape,
        position="trailing",
        view=view,
        broadcast_only=broadcast_only,
    )
