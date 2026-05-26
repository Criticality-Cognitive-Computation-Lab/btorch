from typing import Literal, get_args

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
from jaxtyping import Float

from btorch.config import SPARSE_BACKEND

from .constrain import HasConstraint


try:
    from torch_sparse import SparseTensor
except ImportError:
    SparseTensor = None

SparseBackend = Literal["native", "torch_sparse"]
DEFAULT_SPARSE_BACKEND = SPARSE_BACKEND


def _resolve_sparse_backend(backend: str | None) -> SparseBackend:
    backend = (backend or DEFAULT_SPARSE_BACKEND).lower()
    if backend not in get_args(SparseBackend):
        raise ValueError(
            f"sparse_backend must be 'native' or 'torch_sparse', got '{backend}'."
        )
    if backend == "torch_sparse" and SparseTensor is None:
        raise ImportError("torch_sparse is required for sparse_backend='torch_sparse'.")
    return backend  # type: ignore[return-value]


def available_sparse_backends() -> list[SparseBackend]:
    """Return the sparse backends that can be used in this environment."""
    backends = list(get_args(SparseBackend))
    if SparseTensor is None and "torch_sparse" in backends:
        backends.remove("torch_sparse")
    return backends


# TODO: cleanup and abstract out the logic of native and torch_sparse backends


class DenseConn(nn.Linear):
    # Matrix product using y = x @ A.
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight=None,
        bias=None,
        device=None,
        dtype=None,
    ) -> None:
        """
        :param bias0: The initial bias, if bias0=None or not assigned a value,
            use the random initial value assigned by bias.
        :type bias0: torch.Tensor
        """
        # if weight0 is given, in_features and out_features are ignored
        super().__init__(
            in_features, out_features, bias=bias is not None, device=device, dtype=dtype
        )
        if weight is not None:
            self.weight.data = weight.T
        if bias is not None:
            self.bias.data = bias


@torch.compiler.disable
def spmat(A, x):
    # pytorch_sparse bug. doesn't support torch.compile atm.
    # https://github.com/rusty1s/pytorch_sparse/issues/400
    return A @ x


class BaseSparseConn(nn.Module):
    """Abstract base class for sparse linear layers using a fixed sparse
    connection matrix.

    Attributes:
        in_features (int): Number of source neurons (input features).
        out_features (int): Number of destination neurons (output features).
        shape (Tuple[int, int]): Shape of the internal sparse matrix
            (num_dst, num_src) used for sparse @ dense.
        indices (Tensor): Stacked row/col indices for the transposed matrix.
        native_sparse_tensor (Tensor | None): Cached native sparse tensor.
        sparse_tensor (SparseTensor): Sparse tensor for torch_sparse mode.
        bias (Parameter or None): Optional bias term
    """

    in_features: int
    out_features: int
    shape: tuple[int, int]
    indices: torch.Tensor
    sparse_tensor: SparseTensor | torch.Tensor
    bias: torch.nn.Parameter | None

    def __init__(
        self,
        conn: scipy.sparse.sparray,
        bias=None,
        sparse_backend: SparseBackend | None = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            conn (scipy.sparse.sparray): Sparse connection matrix (num_src, num_dst).
            bias (Tensor, optional): Optional bias vector of shape (num_dst,).
            sparse_backend: "native" or "torch_sparse".
        """
        super().__init__()
        self.sparse_backend = _resolve_sparse_backend(sparse_backend)
        if not isinstance(conn, scipy.sparse.coo_array):
            conn = conn.tocoo()
        # transpose A to compute x @ A via A^T @ x^T.
        conn = conn.T
        # also sort it
        conn.sum_duplicates()
        self.in_features, self.out_features = conn.shape
        indices = torch.stack(
            [
                torch.tensor(conn.row, dtype=torch.long, device=device),
                torch.tensor(conn.col, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        self.register_buffer("indices", indices)
        value = torch.tensor(conn.data, dtype=dtype, device=device)
        shape = (self.out_features, self.in_features)

        self.shape = shape
        # TODO: should update at each time of mod.load_state
        #       maybe a source of checkpoint loading bug!!
        self.sparse_tensor = None
        if dtype is None:
            value = value.to(torch.float32)
        if self.sparse_backend == "native":
            native_sparse = torch.sparse_coo_tensor(
                indices=indices,
                values=value,
                size=self.shape,
                device=device,
                dtype=dtype,
                is_coalesced=True,
            )
            self.sparse_tensor = native_sparse
        elif self.sparse_backend == "torch_sparse":
            self.sparse_tensor = SparseTensor(
                row=self.indices[0],
                col=self.indices[1],
                value=None,
                sparse_sizes=self.shape,
                is_sorted=True,
                trust_data=True,
            ).to(device=device, dtype=dtype)  # type: ignore
        self.bias = nn.Parameter(bias) if bias is not None else None
        self._init_weights(value)

    def _apply(self, fn, recurse=True):
        self.sparse_tensor = fn(self.sparse_tensor)
        return super()._apply(fn, recurse=recurse)

    def _init_weights(self, value: torch.Tensor):
        """Abstract method to initialize layer-specific weights.

        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_effective_weight(self):
        """Abstract method to compute effective weights for the sparse matrix.

        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def forward(
        self, x: Float[torch.Tensor, "... {self.in_features}"]
    ) -> Float[torch.Tensor, "... {self.out_features}"]:
        """Applies the sparse linear transformation.

        Args:
            x (Tensor): Input tensor of shape (..., num_src)

        Returns:
            Tensor: Output tensor of shape (..., num_dst) computed as x @ conn.
        """
        no_batch = x.ndim == 1
        if no_batch:
            x = x[None, :]

        effective_value = self._get_effective_weight()
        if effective_value.device != x.device or effective_value.dtype != x.dtype:
            effective_value = effective_value.to(device=x.device, dtype=x.dtype)
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if self.sparse_backend == "native":
            sp = self.sparse_tensor
            sp = torch.sparse_coo_tensor(
                indices=sp.indices(),
                values=effective_value,
                size=self.shape,
                is_coalesced=True,
            )
            # (A^T @ x^T)^T == x @ A
            out = torch.sparse.mm(sp, x_2d.T).T
        else:
            sparse_tensor = self.sparse_tensor.set_value(effective_value, layout="coo")
            out = spmat(sparse_tensor, x_2d.T)
            out = out.T
        out = out.reshape(*leading_shape, self.out_features)
        if no_batch:
            out = out[0, :]

        if self.bias is not None:
            out += self.bias

        return out


class SparseConn(BaseSparseConn, HasConstraint):
    """Sparse linear transformation using a fixed sparse matrix.

    Optionally enforces Dale's law by maintaining a fixed sign per connection
    and applying ReLU to learned magnitudes.

    Attributes:
        enforce_dale (bool): If True, enforces Dale's law via fixed sign and ReLU.
        initial_sign (Tensor): Fixed signs if Dale's law is enforced.
        magnitude (Parameter): Learnable weights or magnitudes.
    """

    enforce_dale: bool
    initial_sign: torch.Tensor
    magnitude: torch.nn.Parameter

    def __init__(
        self,
        conn: scipy.sparse.sparray,
        bias=None,
        enforce_dale: bool = True,
        sparse_backend: SparseBackend | None = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            conn (scipy.sparse.sparray): Sparse connection matrix (num_src, num_dst).
            bias (Tensor, optional): Optional bias vector of shape (num_dst,).
            enforce_dale (bool): If True, enforces Dale's law via fixed sign and ReLU.
            sparse_backend: "native" or "torch_sparse".
        """
        self.enforce_dale = enforce_dale
        super().__init__(
            conn,
            bias=bias,
            sparse_backend=sparse_backend,
            device=device,
            dtype=dtype,
        )

    def _init_weights(self, value: torch.Tensor):
        if self.enforce_dale:
            self.register_buffer("initial_sign", torch.sign(value), persistent=False)
        self.magnitude = nn.Parameter(value)

    def _get_effective_weight(self):
        return self.magnitude

    def constrain(self, *args, **kwargs):
        if self.enforce_dale:
            self.magnitude.data = (self.magnitude * self.initial_sign).relu() * (
                self.initial_sign
            )


class SparseConstrainedConn(BaseSparseConn, HasConstraint):
    """Sparse linear layer with connection constraints and optional Dale's law
    enforcement.

    This layer uses a sparse connection matrix and a constraint matrix to
    parameterize groups of weights. Each group shares a learnable magnitude,
    allowing structured learning across connections.

    Attributes:
        enforce_dale (bool): Whether to enforce Dale's law (weights never change sign).
        initial_weight (Tensor): Initial signed weights.
        magnitude (Parameter): Learnable magnitudes per constraint group.
        _constraint_scatter_indices (Tensor): Mapping from connection to group index.
    """

    enforce_dale: bool
    initial_weight: torch.Tensor
    magnitude: torch.nn.Parameter
    _constraint_scatter_indices: torch.Tensor

    def __init__(
        self,
        conn: scipy.sparse.sparray,
        constraint: scipy.sparse.sparray,
        enforce_dale: bool = True,
        bias: torch.Tensor | None = None,
        sparse_backend: SparseBackend | None = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            conn (scipy.sparse.sparray): Sparse matrix with initial weights
            (num_src, num_dst).
            constraint (scipy.sparse.sparray): Constraint matrix, entries are
            group IDs (starting from 1).
            enforce_dale (bool): If True, applies ReLU to enforce Dale's law.
            bias (Tensor, optional): Optional bias of shape (num_dst,).
            sparse_backend: "native" or "torch_sparse".
        """
        self.enforce_dale = enforce_dale
        constraint = constraint.T
        constraint.eliminate_zeros()
        if not isinstance(constraint, scipy.sparse.coo_array):
            constraint = constraint.tocoo()
        self._constraint_matrix = constraint
        super().__init__(
            conn,
            bias=bias,
            sparse_backend=sparse_backend,
            device=device,
            dtype=dtype,
        )

    def _init_weights(self, value: torch.Tensor):
        initial_weight = value
        self.register_buffer("initial_weight", initial_weight, persistent=False)
        num_groups = int(self._constraint_matrix.data.max())
        self.magnitude = nn.Parameter(torch.empty(num_groups))
        self.register_buffer(
            "_constraint_scatter_indices",
            torch.tensor(
                self._precompute_scatter_indices(self._constraint_matrix),
                dtype=torch.long,
                device=self.magnitude.device,
            ),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes magnitude parameters to 1."""
        nn.init.ones_(self.magnitude)

    def _precompute_scatter_indices(
        self, constraint: scipy.sparse.coo_array
    ) -> np.ndarray:
        """Matches the (row, col) pairs in the connection matrix with their
        constraint group ID.

        Returns:
            ndarray: Index mapping from connection to group ID (zero-indexed).
        """
        indices = self.indices.cpu().numpy()
        coo_df = pd.DataFrame({"row": indices[0], "col": indices[1]})
        constraint_df = pd.DataFrame(
            {
                "row": constraint.row,
                "col": constraint.col,
                "group_id": constraint.data.astype(int),
            }
        )
        merged = coo_df.merge(constraint_df, how="left", on=["row", "col"])
        assert (
            merged["group_id"].notnull().all()
        ), "Constraint missing for some connections."
        # Convert group ID from 1-based to 0-based indexing
        return merged["group_id"].values - 1

    def _get_effective_weight(self):
        magnitude = self.magnitude[self._constraint_scatter_indices]
        return self.initial_weight * magnitude

    def constrain(self, *args, **kwargs):
        if self.enforce_dale:
            self.magnitude.data = self.magnitude.relu()


class SparseEventConn(nn.Module):
    """Forward-only sparse event propagation using manual span buffers.

    This module is intended for event-driven synaptic current accumulation
    where the input has shape ``(..., n_pre)`` and the output has shape
    ``(..., n_post)``. It supports two traversal modes over the same padded
    CSR-like storage:

    - ``pre_span``: one program works on one presynaptic row at a time.
    - ``post_span``: GeNN-style slot-parallel traversal over the presynaptic
      rows, which changes the kernel schedule without changing the storage.

    The storage uses:

    - ``row_length`` stores the valid number of edges per row
    - ``ind`` stores column indices in a padded ``[rows, row_stride]`` table
    - ``weight`` stores the edge weights in the same padded layout
    """

    in_features: int
    out_features: int
    event_mode: Literal["pre_span", "post_span"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        event_mode: Literal["pre_span", "post_span"] = "pre_span",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.event_mode = event_mode
        self.device = device
        self.dtype = dtype

        self.register_buffer("pre_row_length", None)
        self.register_buffer("pre_ind", None)
        self.register_buffer("pre_weight", None)
        self.register_buffer("post_row_length", None)
        self.register_buffer("post_ind", None)
        self.register_buffer("post_weight", None)

        self.pre_row_stride = 0
        self.post_row_stride = 0

    def set_pre_span_data(
        self,
        row_length: torch.Tensor,
        ind: torch.Tensor,
        weight: torch.Tensor,
        row_stride: int | None = None,
    ) -> None:
        """Register the presynaptic span layout buffers."""
        if row_length.ndim != 1:
            raise ValueError("pre row_length must be 1D.")
        if row_length.shape[0] != self.in_features:
            raise ValueError("pre row_length must have length in_features.")
        if ind.ndim != 2 or weight.ndim != 2:
            raise ValueError("pre ind and weight must be 2D.")
        if ind.shape != weight.shape:
            raise ValueError("pre ind and weight must have identical shapes.")
        if ind.shape[0] != self.in_features:
            raise ValueError("pre ind must have one row per input feature.")

        stride = ind.shape[1] if row_stride is None else row_stride
        if stride != ind.shape[1]:
            raise ValueError("pre row_stride must match ind.shape[1].")

        target_device = self.device if self.device is not None else row_length.device
        target_dtype = self.dtype if self.dtype is not None else weight.dtype
        self.pre_row_length = row_length.to(device=target_device, dtype=torch.int64)
        self.pre_ind = ind.to(device=target_device, dtype=torch.int64)
        self.pre_weight = weight.to(device=target_device, dtype=target_dtype)
        self.pre_row_stride = stride

    def set_post_span_data(
        self,
        row_length: torch.Tensor,
        ind: torch.Tensor,
        weight: torch.Tensor,
        row_stride: int | None = None,
    ) -> None:
        """Register the GeNN-style post-span layout buffers.

        The storage remains presynaptic-row based:

        - ``row_length`` has length ``in_features``
        - ``ind`` and ``weight`` have shape ``[in_features, row_stride]``

        The difference from ``pre_span`` is the kernel traversal strategy rather
        than the underlying sparse storage.
        """
        if row_length.ndim != 1:
            raise ValueError("post row_length must be 1D.")
        if row_length.shape[0] != self.in_features:
            raise ValueError("post row_length must have length in_features.")
        if ind.ndim != 2 or weight.ndim != 2:
            raise ValueError("post ind and weight must be 2D.")
        if ind.shape != weight.shape:
            raise ValueError("post ind and weight must have identical shapes.")
        if ind.shape[0] != self.in_features:
            raise ValueError("post ind must have one row per input feature.")

        stride = ind.shape[1] if row_stride is None else row_stride
        if stride != ind.shape[1]:
            raise ValueError("post row_stride must match ind.shape[1].")

        target_device = self.device if self.device is not None else row_length.device
        target_dtype = self.dtype if self.dtype is not None else weight.dtype
        self.post_row_length = row_length.to(device=target_device, dtype=torch.int64)
        self.post_ind = ind.to(device=target_device, dtype=torch.int64)
        self.post_weight = weight.to(device=target_device, dtype=target_dtype)
        self.post_row_stride = stride

    def forward_events(
        self,
        spike: torch.Tensor,
        *,
        mode: Literal["pre_span", "post_span"] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sparse event propagation for spike-like inputs."""
        from btorch.backend.triton import post_span_spmm, pre_span_spmm

        mode = mode or self.event_mode
        no_batch = spike.ndim == 1
        if no_batch:
            spike = spike.unsqueeze(0)
        if spike.ndim != 2:
            raise ValueError("spike must have shape (n_pre,) or (batch_size, n_pre).")
        if spike.shape[-1] != self.in_features:
            raise ValueError("The last spike dimension must equal in_features.")

        if mode == "pre_span":
            if (
                self.pre_row_length is None
                or self.pre_ind is None
                or self.pre_weight is None
            ):
                raise RuntimeError("pre-span buffers are not initialized.")
            out_tensor = pre_span_spmm(
                spike,
                self.pre_row_length,
                self.pre_ind,
                self.pre_weight,
                size_m=self.out_features,
                is_bool_float=False,
                out=out,
            )
        elif mode == "post_span":
            if (
                self.post_row_length is None
                or self.post_ind is None
                or self.post_weight is None
            ):
                raise RuntimeError("post-span buffers are not initialized.")
            out_tensor = post_span_spmm(
                spike,
                self.post_row_length,
                self.post_ind,
                self.post_weight,
                size_m=self.out_features,
                is_bool_float=False,
                out=out,
            )
        else:
            raise ValueError(f"Unsupported event mode '{mode}'.")

        if no_batch:
            return out_tensor[0]
        return out_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_events(x)
