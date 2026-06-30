from collections.abc import Sequence

import torch

from ..types import TensorLike
from . import base
from .shape import expand_leading_dims


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
    """Uniformly initialize any state variable(s) of a neuron.

    Args:
        neuron: BaseNode instance.
        name: Variable name or list of names (e.g., ``"v"``, ``["v", "w"]``).
        low: Lower bound for initialization. If None:
            uses ``(v_reset, v_threshold)`` when ``name == "v"``,
            otherwise ``(0, 1)``.
        high: Upper bound for initialization. If None:
            uses ``(v_reset, v_threshold)`` when ``name == "v"``,
            otherwise ``(0, 1)``.
        set_reset_value: If True, also stores the initialized value
            as the memory reset value.
        rand_batch: If True, draws a distinct value per batch entry.
            If False, shares the same value across the batch.
        rng: Integer seed or ``torch.Generator`` for reproducibility.

    Returns:
        The modified ``neuron`` (in-place).

    Example:
        >>> from btorch.models.init import uniform_state_
        >>> uniform_state_(neuron, "v", low=-60.0, high=-45.0)
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


@torch.no_grad()
def uniform_v_(
    neuron: base.BaseNode,
    *,
    low: float | TensorLike | None = None,
    high: float | TensorLike | None = None,
    set_reset_value: bool = False,
    rand_batch: bool = False,
    rng: torch.Generator | int | None = None,
):
    """Uniformly initialize the membrane voltage ``v`` of a neuron.

    This is a convenience wrapper around :func:`uniform_state_` for the
    common case of initializing ``v``.

    Args:
        neuron: BaseNode instance.
        low: Lower bound. Defaults to ``v_reset``.
        high: Upper bound. Defaults to ``v_threshold``.
        set_reset_value: If True, stores the value as the reset value.
        rand_batch: Distinct value per batch entry.
        rng: Seed or generator for reproducibility.

    Returns:
        The modified ``neuron`` (in-place).

    Example:
        >>> from btorch.models.init import uniform_v_
        >>> uniform_v_(neuron, set_reset_value=True)
    """
    return uniform_state_(
        neuron,
        "v",
        low=low,
        high=high,
        set_reset_value=set_reset_value,
        rand_batch=rand_batch,
        rng=rng,
    )
