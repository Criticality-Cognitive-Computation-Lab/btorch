"""Independent choices for cell dynamics and connection propagation."""

from dataclasses import dataclass
from typing import Literal


CellBackend = str
ConnectionBackend = Literal["auto", "triton", "aten"]


@dataclass(frozen=True)
class KernelOptions:
    """Select cell and connection implementations independently.

    Args:
        cell: Implementation of neuron and PSC dynamics.
        connection: Sparse runtime implementation; ``auto`` preserves routing.
        merge_synapse: Combine compatible linear receptor states explicitly.
            Independent trainable parameters cannot be merged by value.
    """

    cell: CellBackend = "torch"
    connection: ConnectionBackend = "auto"
    merge_synapse: bool = False

    def __post_init__(self) -> None:
        from .registry import cell_backend_names

        if self.cell not in cell_backend_names():
            raise ValueError(f"Unknown cell backend: {self.cell!r}")
        if self.connection not in ("auto", "triton", "aten"):
            raise ValueError(f"Unknown connection backend: {self.connection!r}")
        if not isinstance(self.merge_synapse, bool):
            raise TypeError("merge_synapse must be a bool")


def normalize_backend(backend: str | KernelOptions) -> KernelOptions:
    """Resolve a backend string to independent kernel options.

    Args:
        backend: A cell backend name or an explicit options object.

    Returns:
        Immutable options; existing options retain their identity.
    """
    if isinstance(backend, KernelOptions):
        return backend
    if not isinstance(backend, str):
        raise TypeError("backend must be a string or KernelOptions")
    return KernelOptions(cell=backend)
