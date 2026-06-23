from .base import SparseTensorBase
from .constraints import (
    BlockUnitNorm,
    Bounded,
    BoundedProjection,
    GroupedWeights,
    GroupProjection,
    NonNegative,
    NonNegativeProjection,
    Symmetric,
    SymmetricProjection,
    constrain,
    prepare_projection,
    project,
)
from .errors import BackendUnavailableError, SparseError, UnsupportedCapabilityError
from .events import BinaryEvents, EventRepresentation, SpikeListEvents
from .initializers import Normal
from .layouts import PaddedCSRLayout
from .matrices import COO, CSC, CSR, ELL, SparseTensor
from .operations import (
    EventSchedule,
    SparseBackend,
    available_backends,
    compact_events,
    event_sparse_mm,
    sparse_mm,
)
from .param import SparseConn, SparseLinear, SparseParam
from .parameterization import GroupedMagnitude, Parameterization
from .procedural import ProceduralSparseMatrix
from .properties import EdgeAttributes, LogicalEdges, SparseProperties


__all__ = [
    # Base
    "SparseTensorBase",
    # Formats
    "COO",
    "CSR",
    "CSC",
    "ELL",
    "SparseTensor",
    # nn.Module wrappers
    "SparseParam",
    "SparseConn",
    "SparseLinear",
    # Constraints
    "NonNegative",
    "GroupedWeights",
    "Symmetric",
    "Bounded",
    "BlockUnitNorm",
    "GroupProjection",
    "NonNegativeProjection",
    "SymmetricProjection",
    "BoundedProjection",
    "constrain",
    "prepare_projection",
    "project",
    # Parameterization
    "Parameterization",
    "GroupedMagnitude",
    # Properties & metadata
    "SparseProperties",
    "EdgeAttributes",
    "LogicalEdges",
    # Layouts
    "PaddedCSRLayout",
    # Initializers
    "Normal",
    # Events
    "BinaryEvents",
    "EventRepresentation",
    "SpikeListEvents",
    # Operations
    "EventSchedule",
    "SparseBackend",
    "available_backends",
    "compact_events",
    "event_sparse_mm",
    "sparse_mm",
    # Procedural
    "ProceduralSparseMatrix",
    # Errors
    "BackendUnavailableError",
    "SparseError",
    "UnsupportedCapabilityError",
]
