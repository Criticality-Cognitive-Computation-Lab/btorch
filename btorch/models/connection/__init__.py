"""Connectivity: projections, rules, synapses and connection realisations.

The modelling layer sits above the numerical :mod:`btorch.sparse` API:

- :class:`SparseConnection` and the other :class:`Connection` realisations are
  the modules a model calls (``current = conn(spikes)``).
- :class:`Projection` combines a :class:`ConnectionRule` (which pairs are
  connected) with a :class:`Synapse` and builds the connection.
- :class:`HardDeepR` rewires a sparse connection during training.
- :class:`Synapse` describes weight, receptor, delay and plasticity of a set
  of edges; :class:`EdgeWeight`, :class:`ConstantWeight` and
  :class:`ConstrainedWeight` are the weight parameterisations.
"""

from .base import (
    Connection,
    HybridConnection,
    ImplicitConnection,
    OperatorConnection,
    StructuredConnection,
)
from .projection import Projection
from .rewire import HardDeepR, HardDeepROptions
from .rule import (
    AllToAll,
    ConnectionRule,
    DistanceDependent,
    FixedIndegree,
    FixedOutdegree,
    FromEdges,
    FromSparse,
    OneToOne,
    PairwiseBernoulli,
)
from .sparse import SparseConnection
from .synapse import Synapse
from .weight import ConstantWeight, ConstrainedWeight, EdgeWeight, Weight


SparseLinear = SparseConnection


__all__ = [
    "AllToAll",
    "Connection",
    "ConnectionRule",
    "ConstantWeight",
    "ConstrainedWeight",
    "DistanceDependent",
    "EdgeWeight",
    "FixedIndegree",
    "FixedOutdegree",
    "FromEdges",
    "FromSparse",
    "HardDeepR",
    "HardDeepROptions",
    "HybridConnection",
    "ImplicitConnection",
    "OneToOne",
    "OperatorConnection",
    "PairwiseBernoulli",
    "Projection",
    "SparseConnection",
    "SparseLinear",
    "StructuredConnection",
    "Synapse",
    "Weight",
]
