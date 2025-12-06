import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from torch import nn

from ..utils.hex_utils import get_hex_coords


class Conv2dHex(nn.Conv2d):
    """Convolution with regularly, hexagonally shaped filters (in cartesian map
    storage).

    Reference to map storage:
    https://www.redblobgames.com/grids/hexagons/#map-storage

    Info:
        kernel_size must be odd!

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding added to input.
        **kwargs: Additional keyword arguments for Conv2d.

    Attributes:
        mask: Mask for hexagonal convolution.
        _filter_to_hex: Whether to apply hexagonal filter.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

        if not kernel_size % 2:
            raise ValueError(f"{kernel_size} is even. Must be odd.")
        if kernel_size > 1:
            u, v = get_hex_coords(kernel_size // 2)
            u -= u.min()
            v -= v.min()
            mask = np.zeros(tuple(self.weight.shape))
            mask[..., u, v] = 1
            self.register_buffer("mask", torch.tensor(mask), persistent=False)
            self.weight.data.mul_(self.mask)
            self._filter_to_hex = True
        else:
            self._filter_to_hex = False

    def filter_to_hex(self):
        """Apply hexagonal filter to weights."""
        return self.weight.data.mul_(self.mask)

    def forward(
        self, x: Float[torch.Tensor, "... {self.in_channels} H W"]
    ) -> Float[torch.Tensor, "... {self.out_channels} Hout Wout"]:
        """Forward pass of the Conv2dHexSpace layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after hexagonal convolution.
        """
        if self._filter_to_hex:
            self.filter_to_hex()
        return super().forward(x)


class Conv1dSpatial(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neurons: pd.DataFrame,
        n_neighbor: int,
        include_self: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        """A special case of SparseConstraintConn, where every post-syn neuron
        connects to `n_neighbor` closest neighbor pre-syn neurons.

        Connection weights from each pre-syn neuron are shared,
        constituting the kernel of Conv1d. This is equivalent to graph
        attention module.
        """
        from ..connectome.connection import make_spatial_localised_conn

        self.include_self = include_self
        self.n_neighbor = n_neighbor
        self.n_neuron = len(neurons)

        # Calculate kernel size before calling super().__init__
        kernel_size = (
            n_neighbor + 1 if include_self else n_neighbor
        )  # +1 for self connection

        # Call parent constructor to initialize weight and bias parameters
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        # Create connection matrix
        # interprete row as post-neuron, and col as pre-neuron
        # coo doesn't support slicing??, have to use csr
        conn = make_spatial_localised_conn(
            neurons, mode="num", num=n_neighbor, include_self=include_self
        ).tocsr()

        self.register_buffer(
            "indices",
            torch.tensor(conn.indices, device=device, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self, x: Float[torch.Tensor, "... {self.in_channels} {self.n_neuron}"]
    ) -> Float[torch.Tensor, "... {self.out_channels} {self.n_neuron}"]:
        """
        Forward pass:
        1. Gather neighbors for each neuron: (..., in_channels, n_neurons, kernel_size)
        2. Apply matrix multiplication with shared weights
        """
        # Handle arbitrary leading dimensions (..., in_channels, n_neurons)
        *leading_dims, in_channels, n_neurons = x.shape

        kernel_size = self.kernel_size[0]  # (n,) -> n

        # Step 1: Gather neighbor values for each neuron
        # Reshape indices to (n_neurons, kernel_size)
        neighbor_indices = self.indices.view(n_neurons, kernel_size)

        # Gather neighbor values
        # x shape: (..., in_channels, n_neurons)
        # neighbor_indices shape: (n_neurons, kernel_size)
        # We want: (..., in_channels, n_neurons, kernel_size)

        # Expand neighbor indices for all leading dimensions and channels
        expanded_shape = x.shape + (
            kernel_size,
        )  # (..., in_channels, n_neurons, kernel_size)
        neighbor_indices_expanded = neighbor_indices.view(
            *([1] * len(leading_dims)), 1, n_neurons, kernel_size
        ).expand(*leading_dims, in_channels, -1, -1)

        # Expand x for gathering
        x_expanded = x.unsqueeze(-1).expand(*expanded_shape)

        # Gather along the neuron dimension (index -2)
        x_neighbors = torch.gather(
            x_expanded,
            dim=-2,  # gather along neuron dimension
            index=neighbor_indices_expanded,
        )  # Shape: (..., in_channels, n_neurons, kernel_size)

        # Step 2: Apply matrix multiplication
        # self.weight shape: (out_channels, in_channels, kernel_size)
        # x_neighbors shape: (..., in_channels, n_neurons, kernel_size)
        # We want: (..., out_channels, n_neurons)

        # Rearrange for matrix multiplication
        # Move kernel_size to the end and combine with in_channels
        x_reshaped = x_neighbors.permute(*range(len(leading_dims)), -2, -3, -1)
        # Shape: (..., n_neurons, in_channels, kernel_size)

        # Flatten last two dimensions for matrix multiplication
        *batch_dims, n_neurons_dim, in_ch_dim, kernel_dim = x_reshaped.shape
        x_flat = x_reshaped.reshape(*batch_dims, n_neurons_dim, in_ch_dim * kernel_dim)

        # Flatten weight for matrix multiplication
        weight_flat = self.weight.view(
            self.out_channels, self.in_channels * kernel_size
        )

        # Matrix multiplication: (..., n_neurons, in_channels * kernel_size) @
        # (out_channels, in_channels * kernel_size).T
        output = torch.matmul(x_flat.contiguous(), weight_flat.T)
        # Shape: (..., n_neurons, out_channels)

        # Transpose to get (..., out_channels, n_neurons)
        output = output.transpose(-2, -1)

        # Add bias if present
        if self.bias is not None:
            # bias shape: (out_channels,)
            output = output + self.bias.view(
                *([1] * len(leading_dims)), self.out_channels, 1
            )

        return output
