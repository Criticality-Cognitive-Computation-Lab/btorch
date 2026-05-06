"""Heterogeneous neuron population that mixes multiple neuron models.

This module allows a single recurrent layer to contain different neuron
types (e.g. ``GLIF3`` and ``TwoCompartmentGLIF``) by slicing the input
current and dispatching to sub-populations.
"""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .two_compartment import TwoCompartmentGLIF


class MixedNeuronPopulation(nn.Module):
    """Mix multiple neuron populations into a single logical layer.

    The input current tensor is sliced along the last (neuron) dimension and
    dispatched to each sub-population.  Spikes from all groups are
    concatenated back together so the layer behaves like a single neuron
    module from the point of view of :class:`~btorch.models.rnn.RecurrentNN`.

    Args:
        groups: Either a sequence of ``(count, neuron_module)`` tuples, or a
            mapping of ``name -> (count, neuron_module)``.  When a sequence
            is given, groups are auto-named ``group_0``, ``group_1``, etc.
        step_mode: ``"s"`` for single-step or ``"m"`` for multi-step
            dispatch.  Default: ``"s"``.

    Raises:
        ValueError: If ``groups`` is empty or contains a non-positive count.

    Examples:
        >>> from btorch.models.neurons import GLIF3, TwoCompartmentGLIF
        >>> glif = GLIF3(n_neuron=60)
        >>> tc = TwoCompartmentGLIF(n_neuron=40)
        >>> mixed = MixedNeuronPopulation([(60, glif), (40, tc)])
    """

    def __init__(
        self,
        groups: Sequence[tuple[int, nn.Module]] | Mapping[str, tuple[int, nn.Module]],
        step_mode: str = "s",
    ):
        super().__init__()
        if not groups:
            raise ValueError("groups must contain at least one population.")

        # Normalise to a list of (name, count, neuron).
        if isinstance(groups, Mapping):
            items = [(name, count, neuron) for name, (count, neuron) in groups.items()]
        else:
            items = [
                (f"group_{i}", count, neuron)
                for i, (count, neuron) in enumerate(groups)
            ]

        counts: list[int] = []
        total = 0
        for name, count, neuron in items:
            if count <= 0:
                raise ValueError(f"Group {name!r} count must be positive, got {count}.")
            self.add_module(name, neuron)
            counts.append(count)
            if hasattr(neuron, "size"):
                total += int(neuron.size)
            else:
                total += count

        self.counts = counts
        self._cumsum = [0] + torch.cumsum(torch.tensor(counts), dim=0).tolist()
        self.n_neuron = (total,)
        self.size = total
        self.step_mode = step_mode

    def _slice(self, x: Tensor, idx: int) -> Tensor:
        """Slice ``x`` along the last dimension for group *idx*."""
        start, end = self._cumsum[idx], self._cumsum[idx + 1]
        return x[..., start:end]

    def single_step_forward(
        self,
        x: Tensor,
        i_apical: Tensor | None = None,
    ) -> Tensor:
        """Advance all sub-populations by one timestep.

        Args:
            x: Input current of shape ``(*batch, n_neuron)``.
            i_apical: Optional apical current of the same shape.  Only slices
                corresponding to :class:`TwoCompartmentGLIF` groups are used.

        Returns:
            Spike tensor of shape ``(*batch, n_neuron)``.
        """
        spikes: list[Tensor] = []
        for idx, (_, neuron) in enumerate(self.named_children()):
            soma = self._slice(x, idx)
            out: Tensor | tuple[Tensor, ...]

            if isinstance(neuron, TwoCompartmentGLIF):
                apical = self._slice(i_apical, idx) if i_apical is not None else None
                out = neuron(soma, apical)
            else:
                out = neuron(soma)

            # Normalise return value to spikes only.
            spikes.append(out[0] if isinstance(out, tuple) else out)

        return torch.cat(spikes, dim=-1)

    def multi_step_forward(
        self,
        x: Tensor,
        i_apical: Tensor | None = None,
    ) -> Tensor:
        """Advance all sub-populations over a full time sequence.

        Args:
            x: Input sequence of shape ``(T, *batch, n_neuron)``.
            i_apical: Optional apical sequence of the same shape.

        Returns:
            Spike sequence of shape ``(T, *batch, n_neuron)``.
        """
        spike_seq: list[Tensor] = []
        T = x.shape[0]
        for t in range(T):
            step_apical = None if i_apical is None else i_apical[t]
            spike_seq.append(self.single_step_forward(x[t], step_apical))
        return torch.stack(spike_seq, dim=0)

    def forward(
        self,
        x: Tensor,
        i_apical: Tensor | None = None,
    ) -> Tensor:
        """Dispatch to single-step or multi-step forward."""
        if self.step_mode == "m":
            return self.multi_step_forward(x, i_apical)
        return self.single_step_forward(x, i_apical)

    def extra_repr(self) -> str:
        parts = [f"n_neuron={self.n_neuron}"]
        for i, (name, neuron) in enumerate(self.named_children()):
            parts.append(f"{name}={neuron.__class__.__name__}(n={self.counts[i]})")
        return ", ".join(parts)
