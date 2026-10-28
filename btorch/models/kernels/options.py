"""Independent choices for neuron dynamics and connection execution."""

from dataclasses import dataclass, replace
from typing import Literal

from ...sparse.runtime.method import PropagationMethod, validate_method
from .registry import _NATIVE_GPU_BACKENDS


NeuronBackend = str
ConnectionBackend = Literal["auto", "triton", "tilelang", "cusparse", "torch"]
RecurrentStrategy = Literal["standard", "helion"]


@dataclass(frozen=True)
class KernelOptions:
    """Select neuron and connection implementations independently.

    Args:
        neuron: Implementation of neuron and PSC dynamics.
        connection: Sparse implementation: ``cusparse`` selects native CUDA CSR
            products, ``triton`` and ``tilelang`` select their sparse CUDA kernels,
            ``torch`` selects
            the general PyTorch reference, and ``auto`` preserves routing.
        connection_method: Explicit sparse execution method. ``auto`` inherits
            connection settings and hints; ``push`` requests spike-list SpMSpV,
            and ``pull`` requests destination-driven execution. Unsupported
            explicit requests raise instead of falling back. CUDA push is not
            bitwise reproducible, including under CUDA Graph replay: rounding
            differences can change later recurrent spikes and gradients. Use
            ``pull`` when reproducible sparse accumulation is required. In a
            1000-step FP32 Izhikevich/ExponentialPSC experiment (60k neurons,
            33,072,348 edges, batch 1, dt=1 ms, about 5 Hz, RTX 5090), repeat
            differences reached 0.676 mV at step 1000 and about 95 mV
            transiently at differing resets; one repeat had 12 differing
            spike bins out of 60 million. These are observations, not bounds.
        recurrent: Recurrent schedule. ``helion`` selects the experimental
            cooperative fused inference path; unsupported configurations raise.
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
        fast_math: Use approximate GPU exponentials and division when True.
            False selects libdevice exponentials (including cancellation-safe
            ``expm1``) and accurately rounded division. FP64 transcendentals
            always use libdevice. This does not change storage dtype or select
            compiler FMA contraction; control that with ``enable_fp_fusion``.
            Torch reference modules retain PyTorch's native arithmetic.
    """

    neuron: NeuronBackend = "torch"
    connection: ConnectionBackend = "auto"
    recurrent: RecurrentStrategy = "standard"
    connection_method: PropagationMethod = "auto"
    merge_synapse: bool = False
    enable_fp_fusion: bool = True
    fast_math: bool = True

    def __post_init__(self) -> None:
        from .registry import cell_backend_names

        if self.neuron not in cell_backend_names():
            raise ValueError(f"Unknown neuron backend: {self.neuron!r}")
        if self.connection not in ("auto", "triton", "tilelang", "cusparse", "torch"):
            raise ValueError(f"Unknown connection backend: {self.connection!r}")
        if self.recurrent not in ("standard", "helion"):
            raise ValueError(f"Unknown recurrent backend: {self.recurrent!r}")
        validate_method(self.connection_method)
        if not isinstance(self.merge_synapse, bool):
            raise TypeError("merge_synapse must be a bool")
        if not isinstance(self.enable_fp_fusion, bool):
            raise TypeError("enable_fp_fusion must be a bool")
        if not isinstance(self.fast_math, bool):
            raise TypeError("fast_math must be a bool")


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
    from .contracts import NeuronSpec
    from .templates import get_template, has_generated_model

    options = normalize_backend(backend)
    if (
        isinstance(backend, str)
        and options.neuron in _NATIVE_GPU_BACKENDS
        and any(
            has_generated_model(model)
            and not get_template(NeuronSpec(model)).default_fast_math
            for model in models
        )
    ):
        return replace(options, fast_math=False)
    return options
