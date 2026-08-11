"""Polars-style lazy recording/monitor engine.

Standalone temporal recording: describe *what* to record over a stepped
computation with expressions (``col("neuron.v").mean().alias("v_mean")``); the
engine infers streaming (O(1) carry) vs materialise (stack ``[T, ...]``) and
lowers to one per-step kernel.  See :class:`Recorder`.
"""

from .engine import Recorder, RecordSpec
from .expr import Expr, col, grad, lit, map_seq, map_step
from .frame import EagerFrame, Resolver, StepFrame, TargetRef
from .reducer import RecordEngine, Reducer, validate_reducer


__all__ = [
    "Recorder",
    "RecordSpec",
    "RecordEngine",
    "Reducer",
    "validate_reducer",
    "Resolver",
    "StepFrame",
    "EagerFrame",
    "TargetRef",
    "Expr",
    "col",
    "lit",
    "grad",
    "map_step",
    "map_seq",
]
