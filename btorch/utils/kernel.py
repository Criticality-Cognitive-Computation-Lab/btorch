"""Own prepared GPU pointer arguments and their tensor storage lifetimes."""

from dataclasses import dataclass, field

from torch import Tensor


def _argument(value):
    return value.data_ptr() if isinstance(value, Tensor) else value


@dataclass
class PreparedLaunch:
    """Own typed kernel storage and its prepared pointer arguments.

    Owners stay alive as long as their addresses can be submitted.
    Updating a slot replaces both its owner and its address; no bare
    pointer is cached across unrelated invocations or module storage
    replacement. Static storage must not be rebound while this
    invocation is active; a new invocation prepares fresh bindings.
    Changing slots refresh their pointers on every call and preserve
    their prepared layout contract.
    """

    launch: object
    owners: list
    changing: tuple[int, ...]
    constants: tuple
    arguments: list = field(init=False)
    layouts: dict = field(init=False)

    def __post_init__(self):
        tensors = tuple(value for value in self.owners if isinstance(value, Tensor))
        if (
            not tensors
            or tensors[0].device.type != "cuda"
            or any(value.device != tensors[0].device for value in tensors)
        ):
            raise ValueError("Prepared GPU arguments must share one CUDA device")
        self.arguments = [_argument(value) for value in self.owners]
        self.layouts = {
            index: tensor_layout(self.owners[index])
            for index in self.changing
            if isinstance(self.owners[index], Tensor)
        }

    def __call__(self, values):
        for index in self.changing:
            value = values[index]
            if index in self.layouts and (
                not isinstance(value, Tensor)
                or tensor_layout(value) != self.layouts[index]
            ):
                raise ValueError(
                    "Prepared input layout/dtype changed; prepare a new execution"
                )
            self.owners[index] = value
            self.arguments[index] = _argument(value)
        self.submit()

    def submit(self):
        """Submit the owned argument vector after prepared slot updates."""
        self.launch(*self.arguments, *self.constants)


def storage_span(value, span=None):
    """Expose reachable storage without changing logical parameter offsets."""
    if span is None:
        span = (
            0
            if not value.numel()
            else 1
            + sum(
                (size - 1) * stride for size, stride in zip(value.shape, value.stride())
            )
        )
    return value.as_strided((span,), (1,), value.storage_offset())


def tensor_layout(value):
    """Return the layout contract of an invocation's replaceable tensor."""
    return value.shape, value.stride(), value.dtype, value.device
