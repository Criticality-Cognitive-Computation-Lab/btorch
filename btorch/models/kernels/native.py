"""Bind native Torch operators and compare complete user model executions."""

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten


@dataclass(frozen=True)
class KernelBinding:
    """Select inference or training without replacing Torch autograd.

    Args:
        inference: Callable returning output, final state and requested records.
            It must not allocate backward-only tape.
        training: Callable with native Torch autograd support. Its backward may
            use an entirely different algorithm. All participating parameters
            and initial states must be explicit tensor inputs.
    """

    inference: Callable
    training: Callable | None = None

    def __call__(self, *inputs):
        leaves, _ = tree_flatten(inputs)
        needs_backward = torch.is_grad_enabled() and any(
            isinstance(value, Tensor) and value.requires_grad for value in leaves
        )
        if needs_backward:
            if self.training is None:
                raise RuntimeError("This kernel binding has no training implementation")
            return self.training(*inputs)
        return self.inference(*inputs)


def load_kernel_package(module: str | Path) -> Mapping[str, KernelBinding]:
    """Import a trusted Python kernel package and return its declared bindings.

    A package exposes ``KERNEL_BINDINGS``, a nonempty mapping of names to
    :class:`KernelBinding`. Native operator registration remains the package's
    responsibility. A path may identify a Python file or package directory.
    Import executes Python code; use only sources you trust.

    Args:
        module: Importable module name or durable source path.

    Returns:
        The package's validated binding mapping.
    """
    source = Path(module)
    if source.exists():
        source = source.resolve()
        if source.is_dir():
            source = source / "__init__.py"
        name = "_btorch_oot_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, source)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot import kernel package {source}")
            package = importlib.util.module_from_spec(spec)
            sys.modules[name] = package
            try:
                spec.loader.exec_module(package)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        package = sys.modules[name]
    else:
        if isinstance(module, Path):
            raise FileNotFoundError(module)
        package = importlib.import_module(module)
    bindings = getattr(package, "KERNEL_BINDINGS", None)
    if (
        not isinstance(bindings, Mapping)
        or not bindings
        or any(
            not isinstance(name, str) or not isinstance(binding, KernelBinding)
            for name, binding in bindings.items()
        )
    ):
        raise ValueError(
            "Kernel package must declare a nonempty KERNEL_BINDINGS mapping"
        )
    return bindings
