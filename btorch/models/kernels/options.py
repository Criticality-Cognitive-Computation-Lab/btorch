"""Independent choices for neuron dynamics and connection execution."""

from dataclasses import dataclass, replace
from typing import Literal

from ...sparse.runtime.method import PropagationMethod, validate_method
from .registry import _NATIVE_GPU_BACKENDS


NeuronBackend = str
ConnectionBackend = Literal["auto", "triton", "tilelang", "cusparse", "torch"]
RecurrentStrategy = Literal["standard", "helion"]
SpikeSelection = Literal["auto", "scan", "spike_list"]


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
            connection settings and hints; ``push`` requests source-driven
            execution (spike-list SpMSpV in standard recurrence), and ``pull``
            requests destination-driven execution. HELION activity discovery
            is controlled by ``spike_selection``. Unsupported
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
            Its single kernel uses the neuron renderer for propagation as well;
            connection must be ``auto`` or match neuron.
        spike_selection: Activity discovery for recurrent propagation.
            ``auto`` retains the schedule's default: lists for standard
            push execution, scanning for HELION. ``scan`` discovers
            active rows from dense spikes, while ``spike_list`` consumes
            the existing neuron-produced block-local queue. For zero-delay
            event-only STDP, either explicit choice combines propagation
            and learning in one kernel; ``standard`` retains a separate
            PSC/neuron kernel and ``helion`` fuses the timestep.
            Static list execution also requires zero delay; static scan
            requires HELION. Explicit choices require native neuron templates
            and push propagation. Unsupported combinations raise before stepping.
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
    spike_selection: SpikeSelection = "auto"
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
        if self.spike_selection not in ("auto", "scan", "spike_list"):
            raise ValueError(f"Unknown spike selection: {self.spike_selection!r}")
        if self.spike_selection != "auto" and self.neuron not in ("triton", "tilelang"):
            raise NotImplementedError(
                "Explicit spike discovery requires Triton or TileLang "
                "neuron/PSC templates"
            )
        if self.recurrent == "helion" and self.connection not in ("auto", self.neuron):
            raise ValueError(
                "HELION uses one backend for its fused neuron/propagation/PSC kernel; "
                "connection must be 'auto' or match neuron. Use recurrent='standard' "
                "for independent neuron and sparse backends."
            )
        validate_method(self.connection_method)
        if self.spike_selection != "auto" and self.connection_method == "pull":
            raise ValueError(
                "Explicit spike discovery requires source-driven propagation; use "
                "connection_method='push' or spike_selection='auto'"
            )
        if not isinstance(self.merge_synapse, bool):
            raise TypeError("merge_synapse must be a bool")
        if not isinstance(self.enable_fp_fusion, bool):
            raise TypeError("enable_fp_fusion must be a bool")
        if not isinstance(self.fast_math, bool):
            raise TypeError("fast_math must be a bool")


def _event_connection_backend(options: KernelOptions | None) -> str | None:
    """Resolve the shared propagation/learning renderer for explicit
    discovery."""
    if (
        options is None
        or options.recurrent != "standard"
        or options.spike_selection == "auto"
    ):
        return None
    if options.neuron not in ("triton", "tilelang"):
        raise NotImplementedError(
            "Two-kernel event recurrence requires Triton or TileLang "
            "neuron/PSC templates"
        )
    backend = options.neuron if options.connection == "auto" else options.connection
    if backend not in ("triton", "tilelang"):
        raise NotImplementedError(
            "Sparse propagation/plasticity fusion requires a Triton "
            "or TileLang connection backend"
        )
    return backend


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
