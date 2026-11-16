"""Stateful short-term synaptic transmission rules."""

from __future__ import annotations

import torch
from torch import Tensor

from .plasticity import SynapticRule


class TransmissionRule(SynapticRule):
    """Own edge-local release state without changing canonical weights.

    A transmission rule consumes the delayed, canonical edge activity for one
    eager timestep and returns one release efficacy per edge. The connection
    multiplies its learned/fixed weights by that efficacy for propagation.
    Transmission state is independent from :class:`PlasticityRule` state.
    """

    def step(self, arrivals: Tensor, dt: float | Tensor) -> Tensor:
        """Advance release state and return per-edge release efficacy.

        Args:
            arrivals: Canonical delayed binary edge events.
            dt: Positive timestep as a number or one-element tensor.
        """
        raise NotImplementedError


class TsodyksMarkram(TransmissionRule):
    r"""Apply binned Tsodyks--Markram short-term depression/facilitation.

    The rule uses resource :math:`x` and utilization :math:`u` per canonical
    edge. At each timestep it first recovers resources and decays facilitation,
    then applies the event update, releases
    :math:`u x` for a binary arrival, and consumes that resource:

    .. math::
        x_r &= 1 - (1-x)e^{-dt/\tau_{rec}}\\
        u_r &= u e^{-dt/\tau_{fac}}\\
        u_e &= u_r + U(1-u_r)\quad\text{on an arrival}\\
        q &= u_e x_r\\
        x' &= x_r - q\quad\text{on an arrival}

    ``tau_fac=0`` disables facilitation and resets utilization toward zero
    before each event. Arrivals must be binary; fractional or multiple events
    require a separately declared release law.

    Args:
        U: Utilization increment, scalar or one value per canonical edge in
            ``[0, 1]``.
        tau_rec: Resource recovery time constant, positive scalar or edge
            array.
        tau_fac: Facilitation decay time constant, nonnegative scalar or edge
            array. Zero disables facilitation.

    Notes:
        The memories initialize to ``x=1`` and ``u=0``. The rule is eager-only
        for now: minibatches, externally supplied sparse events, BPTT and
        optimized recurrent lowering are rejected by the owning connection.
    """

    state_domains = {"x": "edge", "u": "edge"}

    def __init__(
        self,
        U: float | Tensor = 0.2,
        tau_rec: float | Tensor = 800.0,
        tau_fac: float | Tensor = 0.0,
    ):
        super().__init__()
        self._register_coefficient("U", U, nonnegative=True)
        self._register_coefficient("tau_rec", tau_rec, positive=True)
        self._register_coefficient("tau_fac", tau_fac, nonnegative=True)
        self.register_memory("x", 1.0, (), persistent=True)
        self.register_memory("u", 0.0, (), persistent=True)

    def _validate_coefficients(self) -> None:
        """Require utilization increments no larger than one."""
        super()._validate_coefficients()
        if not ((self.U >= 0) & (self.U <= 1)).all():
            raise ValueError("U must be in [0, 1]")

    @staticmethod
    def _validate_binary(arrivals: Tensor) -> None:
        """Reject fractional or multiple events before mutating memories."""
        valid = (arrivals == 0) | (arrivals == 1)
        if arrivals.device.type == "cpu":
            if not bool(valid.all()):
                raise ValueError("TsodyksMarkram requires binary edge arrivals")
        else:
            torch._assert_async(
                valid.all(), "TsodyksMarkram requires binary edge arrivals"
            )

    def step(self, arrivals: Tensor, dt: float | Tensor) -> Tensor:
        """Advance memories and return the release efficacy per edge."""
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Short-term transmission requires no_grad or inference_mode; "
                "BPTT is unsupported"
            )
        self._validate_state()
        if arrivals.ndim != 1 or arrivals.numel() != self.x.numel():
            raise ValueError(
                "TsodyksMarkram arrivals must be one canonical edge vector with "
                f"shape {(self.x.numel(),)}"
            )
        if arrivals.device != self.x.device:
            raise ValueError("Transmission arrivals and memories must share a device")
        if arrivals.requires_grad:
            raise RuntimeError("Transmission arrivals must not require gradients")
        self._validate_binary(arrivals)
        dt_value = self._time_step(dt, self.x)

        rec_decay = torch.exp(-dt_value / self.tau_rec)
        x_recovered = 1 - (1 - self.x.reshape(-1)) * rec_decay
        fac_enabled = self.tau_fac > 0
        fac_tau = torch.where(fac_enabled, self.tau_fac, torch.ones_like(self.tau_fac))
        fac_decay = torch.where(
            fac_enabled,
            torch.exp(-dt_value / fac_tau),
            torch.zeros_like(self.tau_fac),
        )
        u_recovered = self.u.reshape(-1) * fac_decay
        event = arrivals != 0
        u_event = u_recovered + self.U * (1 - u_recovered)
        efficacy = torch.where(
            event, u_event * x_recovered, torch.zeros_like(x_recovered)
        )
        released = arrivals * efficacy
        self.x.copy_((x_recovered - released).reshape_as(self.x))
        self.u.copy_(torch.where(event, u_event, u_recovered).reshape_as(self.u))
        return efficacy
