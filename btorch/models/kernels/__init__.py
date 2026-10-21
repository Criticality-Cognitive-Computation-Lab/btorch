"""Public cell backend extension and pure tensor sequence interfaces."""

from .contracts import CellSpec, HistoryPlan, PoissonOptions
from .options import KernelOptions
from .registry import (
    CellBackendEntry,
    cell_backend_names,
    register_cell_backend,
    resolve_cell_backend,
)
from .templates import (
    BackendFragments,
    ModelTemplate,
    get_template,
    register_model_template,
)


__all__ = [
    "BackendFragments",
    "CellBackendEntry",
    "CellSpec",
    "HistoryPlan",
    "KernelOptions",
    "ModelTemplate",
    "PoissonOptions",
    "cell_backend_names",
    "get_template",
    "register_cell_backend",
    "register_model_template",
    "run_multi_step",
    "resolve_cell_backend",
    "generate_cell_source",
    "KernelBinding",
    "ModelParityCase",
    "ModelParityOptions",
    "check_model_parity",
    "load_kernel_package",
    "ModelSnapshot",
    "ResponseTrace",
    "snapshot_model",
    "restore_model",
    "simulate_response",
    "validate_response",
    "plot_response",
    "describe_response",
]


def __getattr__(name):
    # Keep stateful-model imports lazy: models import KernelOptions while
    # their classes are still being initialized.
    native_names = {
        "KernelBinding",
        "ModelParityCase",
        "ModelParityOptions",
        "check_model_parity",
        "load_kernel_package",
    }
    response_names = {
        "ModelSnapshot",
        "ResponseTrace",
        "snapshot_model",
        "restore_model",
        "simulate_response",
        "validate_response",
        "plot_response",
        "describe_response",
    }
    if name in native_names | response_names:
        from importlib import import_module

        module = import_module(
            ".native" if name in native_names else ".response", package=__name__
        )
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate_cell_source(*args, **kwargs):
    """Render inspectable DSL source using the shared backend templates.

    Args:
        *args: Backend, CellSpec, flattened batch/neuron lane count, sequence
            length and history names, in that order.
        **kwargs: Optional adaptation component count ``modes``, ``backward``
            selection and ``parameter_mask`` for gradient buffers.

    Returns:
        Source text and ordered buffer names with their element counts. Source
        generation does not import or compile optional GPU dependencies.
    """
    from .backends.engine import generate_source

    return generate_source(*args, **kwargs)


def run_multi_step(*args, **kwargs):
    """Execute the public pure tensor sequence operator lazily.

    Args:
        *args: Input sequence, initial states, floating parameters, CellSpec and
            optional HistoryPlan, in that order.
        **kwargs: Registered ``backend`` name.

    Returns:
        Output sequence, final states and requested histories as a flat tuple.
    """
    from .operators import run_multi_step as execute

    return execute(*args, **kwargs)
