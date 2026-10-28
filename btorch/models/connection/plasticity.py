"""Arrival-based online learning rules over canonical connection edges."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ...sparse import EdgeMap
from ..base import MemoryModule


class PairSTDP(MemoryModule):
    r"""Apply additive pair STDP using strictly earlier arrival traces.

    A presynaptic event is a spike arriving at an edge, including its integer
    delay. At each step, decay both traces, compute both weight increments
    from those old traces, clip their sum once, then increment the traces:

    .. math::
        x^-_e = e^{-dt/\tau_{+,e}} x_e,\quad
        y^-_e = e^{-dt/\tau_{-,e}} y_e

        \Delta w_e = -A_{-,e}a_e y^-_e + A_{+,e}z_{j(e)}x^-_e

    Coincident events do not pair with each other; both may still pair with
    earlier events. Propagation uses the weight before this step's update.
    Traces are edge-local, allowing heterogeneous time constants even among
    edges sharing a source, delay, or target. Prepared implementations may
    share traces only when their event streams and time constants agree.

    The connection owns this module after binding. Reset clears traces, not
    learned weights. Traces and hyperparameters are included in state_dict.
    Online mutation is independent of BPTT: execute under torch.no_grad() or
    torch.inference_mode(). Shared-weight minibatch learning and simultaneous
    optimizer updates are not defined by this rule.

    Args:
        a_plus: Nonnegative LTP amplitude, scalar or input-edge array.
        a_minus: Nonnegative LTD amplitude, scalar or input-edge array.
        tau_plus: Positive pre-trace time constant in the same units as dt.
        tau_minus: Positive post-trace time constant.
        w_min: Lower weight bound, scalar or input-edge array.
        w_max: Upper weight bound, scalar or input-edge array.
    """

    def __init__(
        self,
        a_plus: float | Tensor = 0.01,
        a_minus: float | Tensor = 0.012,
        tau_plus: float | Tensor = 20.0,
        tau_minus: float | Tensor = 20.0,
        w_min: float | Tensor = 0.0,
        w_max: float | Tensor = 1.0,
    ):
        super().__init__()
        self._bound = False
        for name, value in (
            ("a_plus", a_plus),
            ("a_minus", a_minus),
            ("tau_plus", tau_plus),
            ("tau_minus", tau_minus),
            ("w_min", w_min),
            ("w_max", w_max),
        ):
            value = torch.as_tensor(value)
            if value.requires_grad:
                raise ValueError("STDP hyperparameters are fixed, not BPTT parameters")
            value = value.detach().clone()
            if value.ndim > 1 or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite and scalar or edge-aligned")
            if not value.is_floating_point():
                value = value.to(torch.get_default_dtype())
            if name.startswith("tau_") and not (value > 0).all():
                raise ValueError(f"{name} must be positive")
            if name.startswith("a_") and not (value >= 0).all():
                raise ValueError(f"{name} must be nonnegative")
            self.register_buffer(name, value)

    def bind(self, edge_map: EdgeMap, weight: Tensor) -> None:
        """Align hyperparameters and allocate persistent trace memories
        once."""
        if self._bound:
            raise ValueError("Create one PairSTDP module per connection")
        if not edge_map.is_permutation:
            raise ValueError(
                "STDP needs distinct canonical edges; coalesce duplicates first"
            )
        source = edge_map.source()
        for name in ("a_plus", "a_minus", "tau_plus", "tau_minus", "w_min", "w_max"):
            value = getattr(self, name).to(device=weight.device, dtype=weight.dtype)
            if value.ndim:
                if value.shape != (edge_map.n_old,):
                    raise ValueError(f"{name} must have one value per input edge")
                value = value.index_select(0, source)
            setattr(self, name, value)
        if not (self.w_min <= self.w_max).all():
            raise ValueError("w_min must not exceed w_max")
        for name in ("pre_trace", "post_trace"):
            self.register_memory(name, 0.0, edge_map.n_new, persistent=True)
        self.init_state(device=weight.device, dtype=weight.dtype)
        self._bound = True

    def single_step_forward(
        self, weight: Tensor, arrival: Tensor, post: Tensor, dt: float | Tensor
    ) -> None:
        """Update canonical weights and trace memories in place.

        Args:
            weight: Mutable canonical per-edge weights.
            arrival: Arriving activity per edge from the delay-expanded input.
            post: Postsynaptic activity per edge.
            dt: Positive simulation timestep in trace-time-constant units.
        """
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Online STDP requires no_grad or inference_mode; BPTT is unsupported"
            )
        if isinstance(dt, (int, float)) and (not math.isfinite(dt) or dt <= 0):
            raise ValueError("STDP dt must be finite and positive")
        if isinstance(dt, Tensor) and (dt.numel() != 1 or dt.requires_grad):
            raise ValueError("STDP dt must be a non-differentiable scalar")
        if isinstance(dt, Tensor):
            valid_dt = torch.isfinite(dt) & (dt > 0)
            if dt.device.type == "cpu":
                if not bool(valid_dt):
                    raise ValueError("STDP dt must be finite and positive")
            else:
                torch._assert_async(
                    valid_dt.all(), "STDP dt must be finite and positive"
                )
        dt = torch.as_tensor(dt, device=weight.device, dtype=weight.dtype)
        x = self.pre_trace.reshape(-1) * torch.exp(-dt / self.tau_plus)
        y = self.post_trace.reshape(-1) * torch.exp(-dt / self.tau_minus)
        delta = self.a_plus * post * x - self.a_minus * arrival * y
        weight.copy_(
            torch.maximum(self.w_min, torch.minimum(self.w_max, weight + delta))
        )
        self.pre_trace.copy_((x + arrival).reshape_as(self.pre_trace))
        self.post_trace.copy_((y + post).reshape_as(self.post_trace))
