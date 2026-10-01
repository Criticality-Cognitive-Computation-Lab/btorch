"""Dynamical-systems metrics for spiking networks.

Modules: ``criticality``, ``complexity``, ``lyapunov_dynamics``,
``attractor_dynamics``, ``micro_scale``, ``ei_balance`` and ``fano``
(rate-compensated Fano factors).

Failure convention: an undefined or failed estimate is returned as NaN (scalar
``float("nan")``, NaN-filled arrays of the success shape, or ``(np.nan, None)``
for fit pairs) and is accompanied by a warning when the failure is unexpected;
invalid arguments raise ``ValueError``. See ``README.md`` in this directory.
"""
