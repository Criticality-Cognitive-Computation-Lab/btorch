"""Independent choices for neuron dynamics and connection execution."""

from dataclasses import dataclass, replace
from typing import Literal

from ...sparse.runtime.method import PropagationMethod, validate_method
from .registry import _NATIVE_GPU_BACKENDS


NeuronBackend = str
ConnectionBackend = Literal["auto", "triton", "cusparse", "torch"]
_PRECISE_DIVISION_DEFAULT_MODELS = frozenset(("alif", "elif"))


@dataclass(frozen=True)
class KernelOptions:
    """Select neuron and connection implementations independently.

    Args:
        neuron: Implementation of neuron and PSC dynamics.
        connection: Sparse implementation: ``cusparse`` selects native CUDA CSR
            products, ``triton`` selects generated kernels, ``torch`` selects
            the general PyTorch reference, and ``auto`` preserves routing.
        connection_method: Explicit sparse execution method. ``auto`` inherits
            connection settings and hints; ``push`` requests spike-list SpMSpV,
            and ``pull`` requests destination-driven execution. Unsupported
            explicit requests raise instead of falling back.
        merge_synapse: Combine compatible linear receptor states explicitly.
            Independent trainable parameters cannot be merged by value.
        enable_fp_fusion: Permit the GPU compiler to contract compatible
            floating-point operations, such as multiply-add pairs. Disable it
            when stepwise FP32 rounding parity with the Torch reference matters
            more than the potential throughput benefit. This option does not
            select approximate transcendental functions or a general fast-math
            mode.
            Explicit fused operations that reproduce a Torch surrogate remain
            fused; this flag controls compiler-selected contraction.
        precise_division: Request accurately rounded floating-point division in
            built-in GPU cell kernels. The default preserves each backend's
            historical division implementation. Plain backend strings select accurate
            division for ALIF/ELIF; explicit options retain this boolean.
            This option does not select a
            general precise-math mode or control transcendental functions,
            square roots, denormals or flush-to-zero behavior.
        precise_exp: Request accurate floating-point ``exp`` evaluation in
            built-in GPU cell kernels. The default preserves each backend's
            historical exponential implementation, and shorthand backends do
            not enable this option automatically. This option does not control
            ``expm1``, division, compiler contraction, or other operations.
    """

    neuron: NeuronBackend = "torch"
    connection: ConnectionBackend = "auto"
    connection_method: PropagationMethod = "auto"
    merge_synapse: bool = False
    enable_fp_fusion: bool = True
    precise_division: bool = False
    precise_exp: bool = False

    def __post_init__(self) -> None:
        from .registry import cell_backend_names

        if self.neuron not in cell_backend_names():
            raise ValueError(f"Unknown neuron backend: {self.neuron!r}")
        if self.connection not in ("auto", "triton", "cusparse", "torch"):
            raise ValueError(f"Unknown connection backend: {self.connection!r}")
        validate_method(self.connection_method)
        if not isinstance(self.merge_synapse, bool):
            raise TypeError("merge_synapse must be a bool")
        if not isinstance(self.enable_fp_fusion, bool):
            raise TypeError("enable_fp_fusion must be a bool")
        if not isinstance(self.precise_division, bool):
            raise TypeError("precise_division must be a bool")
        if not isinstance(self.precise_exp, bool):
            raise TypeError("precise_exp must be a bool")


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
    return KernelOptions(neuron=backend)


def resolve_model_options(backend: str | KernelOptions, *models: str) -> KernelOptions:
    """Apply model defaults to shorthand backends while preserving options."""
    options = normalize_backend(backend)
    if (
        isinstance(backend, str)
        and options.neuron in _NATIVE_GPU_BACKENDS
        and any(model in _PRECISE_DIVISION_DEFAULT_MODELS for model in models)
    ):
        return replace(options, precise_division=True)
    return options
