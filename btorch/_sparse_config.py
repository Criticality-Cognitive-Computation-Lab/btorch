"""Configuration objects for sparse connection backends."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any


@dataclass
class SparseBackendConfig:
    """Base configuration for sparse backends without algorithm options."""


@dataclass
class TritonSparseConfig(SparseBackendConfig):
    """Configure the event-driven Triton sparse backend.

    The three optimization switches are deliberately independent. Disabling all
    of them selects the source-binned, direct-atomic baseline.
    """

    reorder: bool = True
    block: bool = True
    hash: bool = True
    block_sources: int = 32
    edge_block: int = 256
    long_fanout_threshold: int = 256
    hash_capacity: int = 512
    hash_max_probe: int = 8
    hash_min_edges: int = 64
    num_warps: int = 4

    def validate(self) -> None:
        """Validate algorithm parameters."""

        if self.block_sources <= 0:
            raise ValueError("block_sources must be positive.")
        if self.edge_block <= 0 or self.edge_block & (self.edge_block - 1):
            raise ValueError("edge_block must be a positive power of two.")
        if self.long_fanout_threshold <= 0:
            raise ValueError("long_fanout_threshold must be positive.")
        if self.hash_capacity <= 0 or self.hash_capacity & (self.hash_capacity - 1):
            raise ValueError("hash_capacity must be a positive power of two.")
        if self.hash and self.hash_capacity < self.edge_block:
            raise ValueError("hash_capacity must be at least edge_block.")
        if self.hash_max_probe <= 0:
            raise ValueError("hash_max_probe must be positive.")
        if self.hash_min_edges < 0:
            raise ValueError("hash_min_edges must be non-negative.")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be 1, 2, 4, or 8.")


class SparseConfigRegistry:
    """Store the selected sparse backend and per-backend global templates."""

    def __init__(self, selected: str | None = None):
        self._selected = selected.lower() if selected else None
        self._configs: dict[str, SparseBackendConfig] = {
            "native": SparseBackendConfig(),
            "torch_sparse": SparseBackendConfig(),
            "triton": TritonSparseConfig(),
        }

    @property
    def selected(self) -> str | None:
        """Return the explicitly selected global backend, if any."""

        return self._selected

    def backend(self, name: str | None = None) -> SparseBackendConfig:
        """Select a backend and return its mutable global config template.

        Calling without ``name`` returns the currently selected backend config.
        """

        if name is not None:
            normalized = name.lower()
            if normalized not in self._configs:
                choices = ", ".join(sorted(self._configs))
                raise ValueError(
                    f"Unknown sparse backend '{name}'; expected one of {choices}."
                )
            self._selected = normalized
        if self._selected is None:
            raise RuntimeError(
                "No global sparse backend is selected; call "
                "config.sparse.backend('triton') first."
            )
        return self._configs[self._selected]

    def resolve(
        self,
        name: str,
        override: SparseBackendConfig | dict[str, Any] | None = None,
    ) -> SparseBackendConfig:
        """Return an isolated, validated config for one module."""

        resolved = deepcopy(self._configs[name])
        if override is not None:
            values = asdict(override) if is_dataclass(override) else dict(override)
            valid_fields = {field.name for field in fields(resolved)}
            unknown = set(values) - valid_fields
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"Unknown {name} sparse config fields: {names}.")
            for key, value in values.items():
                setattr(resolved, key, value)
        validate = getattr(resolved, "validate", None)
        if validate is not None:
            validate()
        return resolved


__all__ = [
    "SparseBackendConfig",
    "SparseConfigRegistry",
    "TritonSparseConfig",
]
