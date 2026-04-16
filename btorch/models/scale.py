from collections.abc import Sequence
from numbers import Number
from typing import Any, Literal

import numpy as np
import torch

from ..types import TensorLike
from .base import MemoryModule


# DECRECATED: this was designed to scale/unscale the network weight
#   automatically. Scaling makes param values compatible with gradient learning
#   rate. However, through practice, it seems easier to pass scaled param
#   directly to the model.
#   TODO: revisit for a better design.

# TODO: needs rework. use scale_state_ for now.
#   - how does it fit in states dict and memory?
#   - how to handle weight and input scaling?
#   - how to scale buffer and parameters in the same way?
#   - must be explicit and optional. once enabled, must be automatic later
#   - consider torch parametrise or brevitas QuantTensor (not perferable,
#     because scaling couples with model)?


@torch.no_grad()
def scale_state_(
    states: dict[str, Any],
    *,
    scale: TensorLike | None = None,
    zeropoint: TensorLike | None = None,
    unscale: bool = False,
    store: bool = False,
):
    """Scale or unscale a state dictionary in-place.

    Args:
        states: Dictionary of state tensors to scale.
        scale: Scaling factor. If None, inferred from
            ``states["v_threshold"] - states["v_reset"]``.
        zeropoint: Zero point for scaling. If None, inferred from
            ``states["v_reset"]``.
        unscale: If True, apply unscaling instead of scaling.
        store: If True, store ``scale``, ``zeropoint``, and ``scaled``
            flag into ``states``.

    Returns:
        Tuple of ``(scale, zeropoint)`` used.

    Raises:
        ValueError: If required keys for inference are missing.
    """
    if scale is None:
        if "scale" in states:
            scale = states["scale"]
        elif ("v_threshold" in states) and ("v_reset" in states):
            scale = states["v_threshold"] - states["v_reset"]
    assert scale is not None, "Must specify either scale or v_threshold and v_reset"

    if zeropoint is None:
        if "zeropoint" in states:
            zeropoint = states["zeropoint"]
        elif "v_reset" in states:
            zeropoint = states["v_reset"]
        else:
            raise ValueError("Must specify either zeropoint or v_reset")

    if not unscale:
        if "scaled" in states and states["scaled"]:
            return zeropoint, scale
    else:
        if "scaled" in states and not states["scaled"]:
            return zeropoint, scale

    def _scale(v, zero=True):
        if zero:
            return (v - zeropoint) / scale
        else:
            return v / scale

    def _unscale(v, zero=True):
        if zero:
            return v * scale + zeropoint
        else:
            return v * scale

    fn = _unscale if unscale else _scale

    if "v" in states:
        states["v"] = fn(states["v"])

    if "Iasc" in states:
        states["Iasc"] = fn(states["Iasc"], zero=False)

    if "v_threshold" in states:
        states["v_threshold"] = fn(states["v_threshold"])

    if "v_reset" in states:
        states["v_reset"] = fn(states["v_reset"])

    if "psc" in states:
        states["psc"] = fn(states["psc"], zero=False)

    if "asc_amps" in states:
        # very annoying to write code for both torch and numpy
        scale = scale if isinstance(scale, Number) else scale[..., None]
        asc_amps = states["asc_amps"]
        assert not isinstance(asc_amps, Number)
        asc_amps = np.array(asc_amps) if isinstance(asc_amps, Sequence) else asc_amps
        if unscale:
            states["asc_amps"] = asc_amps * scale
        else:
            states["asc_amps"] = asc_amps / scale

    if store:
        states["zeropoint"] = zeropoint
        states["scale"] = scale
        if unscale:
            states["scaled"] = False
        else:
            states["scaled"] = True

    return scale, zeropoint


_ENFORCE_MODE = Literal["ignore", "assert", "repeated"]


class SupportScaleState(MemoryModule):
    """Mixin providing automatic state scaling for neuron parameters.

    Still experimental; see module notes for limitations.
    You are suggested to use :func:`scale_state_` directly for now.
    """

    scaled: torch.Tensor
    neuron_scale: torch.Tensor
    neuron_zeropoint: torch.Tensor

    def _init_scale_state(self):
        self.register_buffer("scaled", torch.tensor(False), persistent=True)

    def _scale_state(self, states: dict | None = None):
        self.scaled |= True

    def _unscale_state(self, states: dict | None = None):
        self.scaled &= False

    def init_scale_state(self):
        self._init_scale_state()
        self.register_buffer(
            "neuron_scale", self.v_threshold - self.v_rest, persistent=True
        )
        self.register_buffer("neuron_zeropoint", self.v_rest, persistent=True)

    def scale_func(self, v, zero=True, enforce: _ENFORCE_MODE = "assert"):
        if not hasattr(self, "scaled"):
            self.init_scale_state()
        if self.scaled:
            if enforce == "assert":
                assert "{self} already scaled"
            elif enforce == "ignore":
                return
        if zero:
            return (v - self.neuron_zeropoint) / self.neuron_scale
        else:
            return v / self.neuron_scale

    def unscale_func(self, v, zero=True, enforce: _ENFORCE_MODE = "ignore"):
        if not self.scaled:
            if enforce == "assert":
                assert "{self} already unscaled"
            elif enforce == "ignore":
                return

        if zero:
            return v * self.neuron_scale + self.neuron_zeropoint
        else:
            return v * self.neuron_scale

    # TODO: complicated, how does this play with memories and memories_rv?
    # TODO: should I scale memories already inited by init_state?
    @torch.no_grad()
    def scale_state(
        self,
        states: dict | None = None,
        enforce: _ENFORCE_MODE = "assert",
        force_memories_rv: bool = True,
    ):
        if not hasattr(self, "scaled"):
            self.init_scale_state()
        if self.scaled:
            if enforce == "assert":
                assert "{self} already scaled"
            elif enforce == "ignore":
                return

        if states is not None:
            scale_state_(
                states,
                scale=self.neuron_scale,
                zeropoint=self.neuron_zeropoint,
            )
            return

        states = {
            "v_threshold": self.v_threshold,
            "v_reset": self.v_rest,
            "asc_amps": self.asc_amps,
        }
        if force_memories_rv:
            states.update(
                {
                    "v": self._memories_rv["v"].value,
                    "Iasc": self._memories_rv["Iasc"].value,
                }
            )
        scale_state_(states, scale=self.neuron_scale, zeropoint=self.neuron_zeropoint)
        self.v_threshold = states["v_threshold"]
        self.v_reset = states["v_reset"]
        self.v_rest = states["v_reset"]
        self.asc_amps = states["asc_amps"]
        if force_memories_rv:
            self.set_reset_value("v", states["v"])
            self.set_reset_value("Iasc", states["Iasc"])

        self._unscale_state(states)

    @torch.no_grad()
    def unscale_state(
        self,
        states: dict | None = None,
        enforce: _ENFORCE_MODE = "ignore",
        force_memories_rv=True,
    ):
        if not self.scaled:
            if enforce == "assert":
                assert "{self} already unscaled"
            elif enforce == "ignore":
                return

        if states is not None:
            scale_state_(
                states,
                scale=self.neuron_scale,
                zeropoint=self.neuron_zeropoint,
                unscale=True,
            )
            return

        states = {
            "v_threshold": self.v_threshold,
            "v_reset": self.v_rest,
            "asc_amps": self.asc_amps,
        }
        if force_memories_rv:
            states.update(
                {
                    "v": self._memories_rv["v"].value,
                    "Iasc": self._memories_rv["Iasc"].value,
                }
            )
        scale_state_(
            states,
            scale=self.neuron_scale,
            zeropoint=self.neuron_zeropoint,
            unscale=True,
        )
        self.v_threshold = states["v_threshold"]
        self.v_reset = states["v_reset"]
        self.v_rest = states["v_reset"]
        self.asc_amps = states["asc_amps"]
        if force_memories_rv:
            self.set_reset_value("v", states["v"])
            self.set_reset_value("Iasc", states["Iasc"])
        self._unscale_state(states)
