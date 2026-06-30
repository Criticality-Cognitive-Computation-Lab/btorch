from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Normal:
    mean: float = 0.0
    std: float = 1.0

    def __call__(
        self,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        return torch.normal(
            mean=self.mean,
            std=self.std,
            size=shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )


def initialize_values(
    value,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if callable(value):
        result = value(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    elif isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=dtype)
    else:
        result = torch.full(shape, float(value), device=device, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"Expected initialized values with shape {shape}.")
    return result
