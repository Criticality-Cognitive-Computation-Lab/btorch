from typing import Literal, get_args

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .base import ParamBufferMixin
from .constrain import HasConstraint


try:
    from torch_sparse import spmm
except ImportError:
    spmm = None

SparseBackend = Literal["native", "torch_sparse"]


def _resolve_sparse_backend(backend: str | None) -> SparseBackend:
    if backend is None:
        return "torch_sparse" if spmm is not None else "native"  # type: ignore[return-value]

    backend = backend.lower()
    if backend not in get_args(SparseBackend):
        raise ValueError(
            f"sparse_backend must be 'native' or 'torch_sparse', got '{backend}'."
        )
    if backend == "torch_sparse" and spmm is None:
        import warnings

        warnings.warn(
            "torch_sparse not installed; falling back to native sparse backend.",
            stacklevel=3,
        )
        backend = "native"
    return backend  # type: ignore[return-value]


def available_sparse_backends() -> list[SparseBackend]:
    """Return the sparse backends that can be used in this environment."""
    backends = list(get_args(SparseBackend))
    if spmm is None and "torch_sparse" in backends:
        backends.remove("torch_sparse")
    return backends


class LearnableScale(ParamBufferMixin, nn.Module):
    """Learnable scalar affine transform via softplus-parameterized scale/bias.

    This is the idiomatic way to apply learnable scaling and offset to
    noise generators or other modules: wrap them with ``LearnableScale``.

    trainable controls which parts are trainable:
      - True: both ``scale`` and ``bias`` (if present) are trainable
      - False: neither is trainable (stored as buffers)
      - "scale": only ``scale`` is trainable
      - "bias": only ``bias`` is trainable (requires ``bias`` to be provided)

    Args:
        scale: Initial scale value (softplus-ed internally, so always > 0).
        bias: Initial bias value (softplus-ed internally, so always > 0).
            If None, no bias term is applied.
        trainable: Which components are trainable.
    """

    def __init__(
        self,
        scale: float = 1.0,
        bias: float | None = None,
        trainable: bool | Literal["bias", "scale"] = True,
    ):
        ParamBufferMixin.__init__(self)
        nn.Module.__init__(self)

        if not (
            trainable is True or trainable is False or trainable in ("bias", "scale")
        ):
            raise ValueError("trainable must be a bool or one of 'bias' or 'scale'")
        if trainable == "bias" and bias is None:
            raise ValueError("trainable='bias' requires a non-None bias value")

        self.has_bias = bias is not None

        # Build trainable_param set for def_param
        if trainable is True:
            tp = {"scale", "bias"} if self.has_bias else {"scale"}
        elif trainable is False:
            tp = set()
        elif trainable == "scale":
            tp = {"scale"}
        else:
            tp = {"bias"}

        def _inv_softplus(y: float) -> torch.Tensor:
            # softplus(x) = log(1 + exp(x)); inverse is log(exp(y) - 1).
            # Clamp for y == 0 to avoid -inf.
            val = torch.exp(torch.tensor(float(y))) - 1.0
            return torch.log(val.clamp_min(1e-8))

        scale_init = _inv_softplus(scale)
        self.def_param(
            "scale",
            scale_init,
            sizes=(),
            trainable_param=tp,
            trainable_shape="scalar",
        )

        if self.has_bias:
            bias_init = _inv_softplus(bias)
            self.def_param(
                "bias",
                bias_init,
                sizes=(),
                trainable_param=tp,
                trainable_shape="scalar",
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.scale_value * x
        if self.has_bias:
            out = out + self.bias_value
        return out

    @property
    def scale_value(self) -> Tensor:
        return nn.functional.softplus(self.scale)

    @property
    def bias_value(self) -> Tensor:
        if not self.has_bias:
            raise AttributeError("bias_value is unavailable when bias=None")
        return nn.functional.softplus(self.bias)


# TODO: cleanup and abstract out the logic of native and torch_sparse backends


class DenseConn(nn.Linear, HasConstraint):
    # Matrix product using y = x @ A.
    mask: Tensor | None
    initial_sign: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        mask: float | Tensor | None = None,
        enforce_dale: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        """Dense connection layer with optional weight masking and Dale's Law
        enforcement.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            weight: Initial weight matrix (in_features, out_features).
                Note: internally it is stored as (out_features, in_features)
                and transposed.
            bias: Initial bias vector (out_features,).
            mask: If float, a random binary mask with this density is generated.
                If Tensor, it must have shape (out_features, in_features).
            enforce_dale: If True, enforces weights to maintain their initial sign
                via ReLU in the `constrain()` method.
            device: Torch device.
            dtype: Torch dtype.
        """
        # if weight is given, in_features and out_features are ignored
        super().__init__(
            in_features, out_features, bias=bias is not None, device=device, dtype=dtype
        )
        if weight is not None:
            self.weight.data = weight.T
        if bias is not None:
            self.bias.data = bias

        if mask is not None:
            if isinstance(mask, (int, float)):
                mask = (
                    torch.rand(self.weight.shape, device=device, dtype=dtype) < mask
                ).to(dtype=self.weight.dtype)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        if enforce_dale:
            self.register_buffer("initial_sign", torch.sign(self.weight.data))
        else:
            self.initial_sign = None

    def constrain(self, *args, **kwargs):
        """Apply the weight mask and Dale's Law constraints to the weight
        matrix."""
        if self.mask is not None:
            self.weight.data *= self.mask
        if self.initial_sign is not None:
            self.weight.data = (
                self.weight.data * self.initial_sign
            ).relu() * self.initial_sign


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
    sparse_tensor: torch.Tensor | None
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
        else:
            self.sparse_tensor = None
        self.bias = nn.Parameter(bias) if bias is not None else None
        self._init_weights(value)

    def _apply(self, fn, recurse=True):
        if self.sparse_tensor is not None:
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

    def get_sparse_matrix(self) -> torch.Tensor:
        """Return the sparse connection matrix as a
        ``torch.sparse_coo_tensor``.

        Values are taken from :meth:`_get_effective_weight`, so gradients
        flow back to the underlying learnable parameters.

        Returns:
            Sparse COO tensor of shape ``(in_features, out_features)``.
        """
        effective_value = self._get_effective_weight()
        # self.indices stores (dst, src) into the transposed matrix;
        # swap back to (src, dst) for the original (in_features, out_features)
        # orientation.
        indices = torch.stack([self.indices[1], self.indices[0]], dim=0)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=effective_value,
            size=(self.in_features, self.out_features),
            is_coalesced=True,
        )

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
            out = spmm(self.indices, effective_value, *self.shape[::-1], x_2d.T)
            out = out.T
        out = out.reshape(*leading_shape, self.out_features)
        if no_batch:
            out = out[0, :]

        if self.bias is not None:
            out = out + self.bias

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
    persist_initial_weight: bool
    initial_weight: torch.Tensor
    magnitude: torch.nn.Parameter
    _constraint_scatter_indices: torch.Tensor
    constraint_info: dict | None

    def __init__(
        self,
        conn: scipy.sparse.sparray,
        constraint: scipy.sparse.sparray,
        enforce_dale: bool = True,
        bias: torch.Tensor | None = None,
        sparse_backend: SparseBackend | None = None,
        device=None,
        dtype=None,
        persist_initial_weight: bool = False,
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
            persist_initial_weight: If True, saves ``initial_weight`` in
                ``state_dict`` for checkpoint reproducibility.
        """
        self.enforce_dale = enforce_dale
        self.persist_initial_weight = persist_initial_weight
        self.constraint_info = None  # Will be populated by from_hetersynapse
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
        self.register_buffer(
            "initial_weight",
            initial_weight,
            persistent=self.persist_initial_weight,
        )
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

    @classmethod
    def from_hetersynapse(
        cls,
        conn: scipy.sparse.sparray,
        constraint: scipy.sparse.sparray,
        receptor_type_index: pd.DataFrame,
        enforce_dale: bool = True,
        bias: torch.Tensor | None = None,
        sparse_backend: SparseBackend | None = None,
        device=None,
        dtype=None,
        persist_initial_weight: bool = False,
    ) -> "SparseConstrainedConn":
        """Create from make_hetersynapse_constrained_conn() output.

        Args:
            conn: Connection sparse matrix from make_hetersynapse_constrained_conn
            constraint: Constraint sparse matrix from make_hetersynapse_constrained_conn
            receptor_type_index: DataFrame mapping receptor indices to receptor types
            enforce_dale: If True, applies ReLU to enforce Dale's law
            bias: Optional bias of shape (num_dst,)
            sparse_backend: "native" or "torch_sparse"
            device: Device to place tensors on
            dtype: Data type for tensors
            persist_initial_weight: If True, includes ``initial_weight`` in
                ``state_dict``.

        Returns:
            SparseConstrainedConn instance with constraint_info populated
        """
        instance = cls(
            conn=conn,
            constraint=constraint,
            enforce_dale=enforce_dale,
            bias=bias,
            sparse_backend=sparse_backend,
            device=device,
            dtype=dtype,
            persist_initial_weight=persist_initial_weight,
        )
        instance.constraint_info = {
            "receptor_type_index": receptor_type_index,
        }
        return instance

    def get_group_info(self, include_weights: bool = False) -> pd.DataFrame:
        """Get information about each constraint group.

        Args:
            include_weights: If True, include mean weight statistics

        Returns:
            DataFrame with group_id, num_connections, current_magnitude,
            and optionally receptor type info.
        """
        n_groups = len(self.magnitude)
        group_info = []

        for group_id in range(n_groups):
            mask = self._constraint_scatter_indices == group_id
            num_connections = mask.sum().item()
            current_mag = self.magnitude[group_id].item()

            info = {
                "group_id": group_id,
                "num_connections": num_connections,
                "current_magnitude": current_mag,
            }

            if include_weights:
                weights = self.initial_weight[mask]
                info["mean_initial_weight"] = weights.mean().item()
                # Use unbiased=False to avoid warning for single-element groups
                info["std_initial_weight"] = weights.std(unbiased=False).item()

            group_info.append(info)

        df = pd.DataFrame(group_info)

        # Add receptor type info if available
        if self.constraint_info is not None:
            receptor_idx = self.constraint_info["receptor_type_index"]
            # Note: Constraint groups may not directly map to receptor indices
            # This is a simple mapping that works when constraint_mode="full"
            if len(receptor_idx) == n_groups:
                df = pd.concat([df, receptor_idx.reset_index(drop=True)], axis=1)

        return df

    def set_group_magnitude(
        self,
        group_id: int | None = None,
        receptor_pair: tuple[str, str] | None = None,
        value: float | torch.Tensor = 1.0,
    ):
        """Set magnitude for a specific constraint group.

        Args:
            group_id: Zero-indexed group ID (mutually exclusive with receptor_pair)
            receptor_pair: (pre_receptor, post_receptor) if created from hetersynapse
                          (mutually exclusive with group_id)
            value: Magnitude value to set
        """
        if (group_id is None) == (receptor_pair is None):
            raise ValueError(
                "Exactly one of group_id or receptor_pair must be specified"
            )

        if receptor_pair is not None:
            if self.constraint_info is None:
                raise ValueError(
                    "receptor_pair can only be used if created via from_hetersynapse"
                )
            receptor_idx = self.constraint_info["receptor_type_index"]

            # Check if neuron mode (has pre/post receptor types)
            if "pre_receptor_type" in receptor_idx.columns:
                pre_type, post_type = receptor_pair
                idx = receptor_idx.set_index(
                    ["pre_receptor_type", "post_receptor_type"]
                )
                group_id = idx.loc[(pre_type, post_type), "receptor_index"]
            else:
                raise ValueError(
                    "receptor_pair requires neuron mode "
                    "(pre_receptor_type, post_receptor_type)"
                )

        if isinstance(value, (int, float)):
            value = torch.tensor(
                value, dtype=self.magnitude.dtype, device=self.magnitude.device
            )

        self.magnitude.data[group_id] = value

    def get_weights_by_group(self) -> dict[int, torch.Tensor]:
        """Get actual weight values grouped by constraint group.

        Returns:
            Dict mapping group_id (0-indexed) to tensor of weights in that group.
        """
        n_groups = len(self.magnitude)
        result = {}

        for group_id in range(n_groups):
            mask = self._constraint_scatter_indices == group_id
            weights = self.initial_weight[mask] * self.magnitude[group_id]
            result[group_id] = weights

        return result
