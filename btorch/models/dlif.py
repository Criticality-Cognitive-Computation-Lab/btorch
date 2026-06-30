"""Dendritic LIF-style recurrent cells.

This module implements composite recurrent cells that keep responsibilities
separate:

- optional receptor-wise synaptic dynamics via a ``BasePSC`` module,
- dendritic receptor mixing via bilinear + linear terms,
- somatic spiking via :class:`btorch.models.neurons.lif.LIF`.

The cells are designed for recurrent usage through ``make_rnn(cell)``.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .base import MemoryModule, normalize_n_neuron
from .bilinear import SymmetricBilinear
from .neurons.lif import LIF
from .synapse import BasePSC, DualExponentialPSC


class DendriticLIF(MemoryModule):
    """Composite dendritic-soma LIF recurrent cell.

    Per step, this cell computes:

    1. optional receptor-wise synaptic filtering,
    2. bilinear + linear receptor mixing,
    3. somatic LIF spike update.

    Args:
        n_neuron: Number of neurons (int or tuple).
        n_receptor: Number of receptor channels per neuron.
        synapse_module: Optional pre-built synapse module. When ``None``,
            input current is used directly (DLIF-like mode) unless
            ``synapse_cls`` is provided.
        synapse_cls: Optional ``BasePSC`` subclass used to construct a synapse
            module internally.
        synapse_kwargs: Keyword arguments passed to ``synapse_cls``.
        bilinear_mask: Optional mask for :class:`SymmetricBilinear`.
        soma: Optional pre-built soma module.
        soma_kwargs: Keyword arguments used to construct default ``LIF`` soma.
        step_mode: Step mode label. Default: ``"s"``.
        backend: Backend label. Default: ``"torch"``.

    Shape:
        Input per step:
            ``(*batch, *n_neuron, n_receptor)``.
            If ``n_receptor == 1``, ``(*batch, *n_neuron)`` is also accepted.
        Output per step:
            ``(*batch, *n_neuron)`` spike tensor.
    """

    synapse_module: BasePSC | None

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        n_receptor: int,
        *,
        synapse_module: BasePSC | None = None,
        synapse_cls: type[BasePSC] | None = None,
        synapse_kwargs: dict | None = None,
        bilinear_mask: float | Tensor | None = None,
        soma: LIF | None = None,
        soma_kwargs: dict | None = None,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        super().__init__()

        if n_receptor < 1:
            raise ValueError(f"n_receptor must be >= 1, got {n_receptor}.")
        if synapse_module is not None and synapse_cls is not None:
            raise ValueError("Provide either synapse_module or synapse_cls, not both.")
        if soma is not None and soma_kwargs is not None:
            raise ValueError("Provide either soma or soma_kwargs, not both.")

        self.n_neuron, self.size = normalize_n_neuron(n_neuron)
        self.n_receptor = int(n_receptor)
        self.step_mode = step_mode
        self.backend = backend

        if synapse_module is None and synapse_cls is not None:
            synapse_module = self._build_synapse(synapse_cls, synapse_kwargs)
        self.synapse_module = synapse_module

        self.bilinear = SymmetricBilinear(
            in_features=self.n_receptor,
            out_features=1,
            bias=True,
            mask=bilinear_mask,
        )

        if soma is None:
            soma_kwargs = {} if soma_kwargs is None else dict(soma_kwargs)
            soma = LIF(n_neuron=self.n_neuron, **soma_kwargs)
        self.soma = soma

    def _build_synapse(
        self,
        synapse_cls: type[BasePSC],
        synapse_kwargs: dict | None,
    ) -> BasePSC:
        synapse_kwargs = {} if synapse_kwargs is None else dict(synapse_kwargs)
        total_channels = self.size * self.n_receptor

        linear = nn.Linear(total_channels, total_channels, bias=False)
        nn.init.eye_(linear.weight)

        return synapse_cls(
            n_neuron=total_channels,
            linear=linear,
            **synapse_kwargs,
        )

    def _prepare_step_input(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, tuple[int, ...]]:
        neuron_rank = len(self.n_neuron)
        compact_shape = self.n_neuron
        full_shape = (*self.n_neuron, self.n_receptor)

        if self.n_receptor == 1 and x.shape[-neuron_rank:] == compact_shape:
            x = x.unsqueeze(-1)

        if x.shape[-(neuron_rank + 1) :] != full_shape:
            raise RuntimeError(
                "DendriticLIFCell input shape mismatch. "
                f"Expected trailing shape {full_shape}, got {tuple(x.shape)}."
            )

        leading = x.shape[: -(neuron_rank + 1)]
        x_flat = x.reshape(*leading, self.size * self.n_receptor)
        return x, x_flat, leading

    def single_step_forward(self, x: Tensor) -> Tensor:
        receptor_input, receptor_input_flat, leading = self._prepare_step_input(x)

        if self.synapse_module is None:
            receptor_current = receptor_input
        else:
            receptor_current_flat = self.synapse_module.single_step_forward(
                receptor_input_flat
            )
            receptor_current = receptor_current_flat.reshape(
                *leading, *self.n_neuron, self.n_receptor
            )

        soma_current = (
            self.bilinear(receptor_current) + receptor_current.sum(dim=-1, keepdim=True)
        ).squeeze(-1)

        return self.soma.single_step_forward(soma_current)

    def multi_step_forward(self, x_seq: Tensor) -> Tensor:
        out_seq = []
        for x in x_seq:
            out_seq.append(self.single_step_forward(x))
        return torch.stack(out_seq, dim=0)


class DLIF(DendriticLIF):
    """Dendritic LIF cell without synaptic dynamics (identity receptor
    current)."""

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        n_receptor: int = 1,
        *,
        bilinear_mask: float | Tensor | None = None,
        soma: LIF | None = None,
        soma_kwargs: dict | None = None,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        super().__init__(
            n_neuron=n_neuron,
            n_receptor=n_receptor,
            synapse_module=None,
            bilinear_mask=bilinear_mask,
            soma=soma,
            soma_kwargs=soma_kwargs,
            step_mode=step_mode,
            backend=backend,
        )


class DBNN(DendriticLIF):
    """Dendritic LIF cell with synaptic dynamics (default: dual-exponential
    PSC)."""

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        n_receptor: int,
        *,
        synapse_cls: type[BasePSC] = DualExponentialPSC,
        synapse_kwargs: dict | None = None,
        bilinear_mask: float | Tensor | None = None,
        soma: LIF | None = None,
        soma_kwargs: dict | None = None,
        step_mode: str = "s",
        backend: str = "torch",
    ):
        synapse_kwargs = {} if synapse_kwargs is None else dict(synapse_kwargs)
        if issubclass(synapse_cls, DualExponentialPSC):
            synapse_kwargs.setdefault("tau_decay", 20.0)
            synapse_kwargs.setdefault("tau_rise", 5.0)

        super().__init__(
            n_neuron=n_neuron,
            n_receptor=n_receptor,
            synapse_cls=synapse_cls,
            synapse_kwargs=synapse_kwargs,
            bilinear_mask=bilinear_mask,
            soma=soma,
            soma_kwargs=soma_kwargs,
            step_mode=step_mode,
            backend=backend,
        )


__all__ = ["DendriticLIF", "DLIF", "DBNN"]
