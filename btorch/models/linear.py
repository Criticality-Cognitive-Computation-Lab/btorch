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
        self.in_features, self.out_features = conn.shape
        self.shape = (self.in_features, self.out_features)
        conn = conn.T
        # also sort it
        conn.sum_duplicates()
        indices = torch.stack(
            [
                torch.tensor(conn.row, dtype=torch.long, device=device),
                torch.tensor(conn.col, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        self.register_buffer("indices", indices)
        value = torch.tensor(conn.data, dtype=dtype, device=device)

        # TODO: should update at each time of mod.load_state
        #       maybe a source of checkpoint loading bug!!
        self.sparse_tensor = None
        if dtype is None:
            value = value.to(torch.float32)
        if self.sparse_backend == "native":
            native_sparse = torch.sparse_coo_tensor(
                indices=indices,
                values=value,
                size=conn.shape,
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
                sparse_sizes=conn.shape,
                is_sorted=True,
                trust_data=False,
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
                size=sp.shape,
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
