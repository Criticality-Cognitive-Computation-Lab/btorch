"""Convolution layers.

Contains Conv1dSpatial for sparse spatial connections. For hexagonal
convolution, use btorch.models.hex.Conv2dHex.
"""

import pandas as pd
import torch
from jaxtyping import Float
from torch import nn

from ..connectome.connection import make_spatial_localised_conn


class Conv1dSpatial(nn.Conv1d):
    """1D Convolution with spatial locality - each neuron connects to nearest neighbors.

    A special case of sparse constrained connection where every post-synaptic neuron
    connects to `n_neighbor` closest neighbor pre-synaptic neurons.

    Connection weights from each pre-syn neuron are shared,
    constituting the kernel of Conv1d. This is equivalent to graph
    attention module.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        neurons: DataFrame with 'x', 'y', 'z' columns for spatial positions.
        n_neighbor: Number of neighbors to connect to.
        include_self: Whether to include self-connections.
        bias: Whether to use bias.
        device: Device for computation.
        dtype: Data type for parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neurons: pd.DataFrame,
        n_neighbor: int,
        include_self: bool = True,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.include_self = include_self
        self.n_neighbor = n_neighbor
        self.n_neuron = len(neurons)

        kernel_size = (
            n_neighbor + 1 if include_self else n_neighbor
        )  # +1 for self connection

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        # Rows are post neurons, columns are pre neurons.
        conn = make_spatial_localised_conn(
            neurons, mode="num", num=n_neighbor, include_self=include_self
        ).tocsr()

        self.register_buffer(
            "indices",
            torch.tensor(conn.indices, device=device, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self, x: Float[torch.Tensor, "... {self.in_channels} n_neuron"]
    ) -> Float[torch.Tensor, "... {self.out_channels} n_neuron"]:
        """
        Forward pass:
        1. Gather neighbors for each neuron: (..., in_channels, n_neurons, kernel_size)
        2. Apply matrix multiplication with shared weights
        """
        *leading_dims, in_channels, n_neurons = x.shape

        kernel_size = self.kernel_size[0]  # (n,) -> n

        neighbor_indices = self.indices.view(n_neurons, kernel_size)

        expanded_shape = x.shape + (
            kernel_size,
        )  # (..., in_channels, n_neurons, kernel_size)
        neighbor_indices_expanded = neighbor_indices.view(
            *([1] * len(leading_dims)), 1, n_neurons, kernel_size
        ).expand(*leading_dims, in_channels, -1, -1)

        x_expanded = x.unsqueeze(-1).expand(*expanded_shape)

        x_neighbors = torch.gather(
            x_expanded,
            dim=-2,  # gather along neuron dimension
            index=neighbor_indices_expanded,
        )  # Shape: (..., in_channels, n_neurons, kernel_size)

        # Contract (in_channels, kernel_size) against weight (out_channels,
        # in_channels, kernel_size) to get (..., out_channels, n_neurons).
        x_reshaped = x_neighbors.permute(*range(len(leading_dims)), -2, -3, -1)
        # Shape: (..., n_neurons, in_channels, kernel_size)

        *batch_dims, n_neurons_dim, in_ch_dim, kernel_dim = x_reshaped.shape
        x_flat = x_reshaped.reshape(*batch_dims, n_neurons_dim, in_ch_dim * kernel_dim)

        weight_flat = self.weight.view(
            self.out_channels, self.in_channels * kernel_size
        )

        output = torch.matmul(x_flat.contiguous(), weight_flat.T)
        # Shape: (..., n_neurons, out_channels)

        output = output.transpose(-2, -1)

        if self.bias is not None:
            output = output + self.bias.view(
                *([1] * len(leading_dims)), self.out_channels, 1
            )

        return output
