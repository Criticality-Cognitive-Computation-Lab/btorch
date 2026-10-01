"""Model-driven fitting and tuning utilities.

Unlike :mod:`btorch.analysis` (post-hoc statistics of recorded activity), the
tools here simulate :mod:`btorch.models` networks and optimise their parameters.

- :mod:`.tuning`: constant-current f-I / V-I sweeps of a neuron class.
- :mod:`.two_compartment`: Allen-data fitting of the two-compartment GLIF neuron.
"""

from . import tuning, two_compartment


__all__ = ["tuning", "two_compartment"]
