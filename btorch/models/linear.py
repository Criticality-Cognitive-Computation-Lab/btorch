import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
from jaxtyping import Float

from .constrain import HasConstraint


USE_NATIVE_SPARSE = True

if not USE_NATIVE_SPARSE:
    from torch_sparse import SparseTensor


class DenseConn(nn.Linear):
    # Matrix-matrix product for dense matrix
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


def sort_indices(coo: scipy.sparse.coo_array):
    coo.sum_duplicates()


class BaseSparseConn(nn.Module):
    """Abstract base class for sparse linear layers using a fixed sparse
    connection matrix.

    Attributes:
        shape (Tuple[int, int]): Shape of the sparse matrix (num_src, num_dst)
        row (Tensor): Row indices of nonzero connections
        col (Tensor): Column indices of nonzero connections
        sparse_tensor (SparseTensor): Torch sparse tensor for efficient computation
        bias (Parameter or None): Optional bias term
    """

    def __init__(
        self,
        conn: scipy.sparse.sparray,
        bias=None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            conn (scipy.sparse.sparray): Sparse connection matrix (num_src, num_dst).
            bias (Tensor, optional): Optional bias vector of shape (num_dst,).
        """
        super().__init__()
        conn = conn.T
        if not isinstance(conn, scipy.sparse.coo_array):
            conn = conn.tocoo()
        sort_indices(conn)
        row = torch.tensor(conn.row, dtype=torch.long, device=device)
        col = torch.tensor(conn.col, dtype=torch.long, device=device)
        value = torch.tensor(conn.data, dtype=dtype, device=device)
        shape = conn.shape

        self.shape = shape
        self.register_buffer("row", row, persistent=False)
        self.register_buffer("col", col, persistent=False)
        # TODO: should update at each time of mod.load_state
        #       maybe a source of checkpoint loading bug!!
        if USE_NATIVE_SPARSE:
            indices = torch.stack([self.row, self.col], dim=0)
            self.sparse_tensor = torch.sparse_coo_tensor(
                indices=indices,
                values=torch.ones_like(value),
                size=self.shape,
                device=device,
                dtype=dtype,
            ).coalesce()
        else:
            self.sparse_tensor = SparseTensor(
                row=self.row,
                col=self.col,
                value=None,
                sparse_sizes=self.shape,
                is_sorted=True,
                trust_data=True,
            ).to_device(device=device)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self._init_weights(value)

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
        self, x: Float[torch.Tensor, "... {self.shape[1]}"]
    ) -> Float[torch.Tensor, "... {self.shape[0]}"]:
        """Applies the sparse linear transformation.

        Args:
            x (Tensor): Input tensor of shape (..., num_src)

        Returns:
            Tensor: Output tensor of shape (..., num_dst)
        """
        no_batch = x.ndim == 1
        if no_batch:
            x = x[None, :]

        effective_value = self._get_effective_weight()
        if USE_NATIVE_SPARSE:
            sp = torch.sparse_coo_tensor(
                indices=self.sparse_tensor.indices(),
                values=effective_value,
                size=self.shape,
                device=x.device,
                dtype=x.dtype,
            )
            out = torch.sparse.mm(sp, x.T).T
        else:
            self.sparse_tensor = (
                self.sparse_tensor.device_as(device=x.device)
                .type_as(type=x.dtype)
                .set_value(effective_value, layout="coo")
            )
            out = spmat(self.sparse_tensor, x.T)
            out = out.T
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

    def __init__(
        self,
        conn: scipy.sparse.sparray,
        bias=None,
        enforce_dale: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Args:
            conn (scipy.sparse.sparray): Sparse connection matrix (num_src, num_dst).
            bias (Tensor, optional): Optional bias vector of shape (num_dst,).
            enforce_dale (bool): If True, enforces Dale's law via fixed sign and ReLU.
        """
        self.enforce_dale = enforce_dale
        super().__init__(conn, bias=bias, device=device, dtype=dtype)

    def _init_weights(self, value: torch.Tensor):
        if self.enforce_dale:
            self.register_buffer("initial_sign", torch.sign(value), persistent=False)
        self.magnitude = nn.Parameter(value)

    def _get_effective_weight(self):
        return self.magnitude

    def constrain(self, *args, **kwargs):
        if self.enforce_dale:
            self.magnitude = (self.magnitude * self.initial_sign).relu() * (
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

    def __init__(
        self,
        conn: scipy.sparse.sparray,
        constraint: scipy.sparse.sparray,
        enforce_dale: bool = True,
        bias: torch.Tensor = None,
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
        """
        self.enforce_dale = enforce_dale
        constraint = constraint.T
        constraint.eliminate_zeros()
        if not isinstance(constraint, scipy.sparse.coo_array):
            constraint = constraint.tocoo()
        self._constraint_matrix = constraint
        super().__init__(conn, bias=bias, device=device, dtype=dtype)

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
        # Use self.row and self.col from registered buffers
        coo_df = pd.DataFrame(
            {"row": self.row.cpu().numpy(), "col": self.col.cpu().numpy()}
        )
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
