"""Format-agnostic sparse array."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .operator import LinearOperator
from .properties import Hints, Properties


if TYPE_CHECKING:
    from .coo import COO
    from .csr import CSC, CSR


class Sparse:
    """Sparse array with logical shape ``[*batch, *sparse, *dense]``.

    ``Sparse`` is the format-agnostic interface. It describes *what* the array
    is (shape, dtype, device, number of stored entries) and offers the
    operations every format supports. Format-specific data such as ``indptr``
    is only available on the concrete classes (:class:`~btorch.sparse.COO`,
    :class:`~btorch.sparse.CSR`, :class:`~btorch.sparse.CSC`), obtained with
    ``tocsr()`` / ``as_csr()`` and friends.

    The three shape groups are:

    - ``batch_shape``: independent sparse arrays (for example an ensemble of
      networks).
    - ``sparse_shape``: the dimensions that are stored sparsely.
    - ``dense_shape``: a dense payload attached to every stored entry.

    Stored values have shape ``[*value_batch, nnz, *dense_shape]``. A batch
    whose members share one sparsity pattern stores the pattern once and only
    the values per member.

    ``Sparse`` is a plain Python object holding tensors; it is not a
    ``torch.Tensor`` subclass. Matrix products follow standard linear algebra:
    for ``A.shape == (M, N)``, ``A @ x`` maps a length-``N`` vector to a
    length-``M`` vector. Conversions never transpose.
    """

    format: str = "sparse"
    # Let ``ndarray @ A`` reach ``__rmatmul__`` instead of NumPy treating the
    # sparse array as an object scalar.
    __array_ufunc__ = None

    _values: Tensor
    _shape: tuple[int, ...]
    _batch_dim: int
    _dense_dim: int
    properties: Properties
    hints: Hints

    # ------------------------------------------------------------------ shape
    @property
    def shape(self) -> tuple[int, ...]:
        """Logical shape ``(*batch, *sparse, *dense)``."""
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def batch_dim(self) -> int:
        """Number of batch dimensions."""
        return self._batch_dim

    def dense_dim(self) -> int:
        """Number of dense (per-entry payload) dimensions."""
        return self._dense_dim

    def sparse_dim(self) -> int:
        """Number of sparsely stored dimensions."""
        return self.ndim - self._batch_dim - self._dense_dim

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._shape[: self._batch_dim]

    @property
    def sparse_shape(self) -> tuple[int, ...]:
        return self._shape[self._batch_dim : self.ndim - self._dense_dim]

    @property
    def dense_shape(self) -> tuple[int, ...]:
        return self._shape[self.ndim - self._dense_dim :]

    @property
    def dtype(self) -> torch.dtype:
        return self._values.dtype

    @property
    def device(self) -> torch.device:
        return self._values.device

    @property
    def nnz(self) -> int:
        """Number of stored entries (of one batch member for a shared pattern,
        of all members together otherwise)."""
        raise NotImplementedError

    @property
    def requires_grad(self) -> bool:
        return self._values.requires_grad

    # ------------------------------------------------------- format changes
    def tocoo(self) -> "COO":
        """Convert to COO (may allocate)."""
        raise NotImplementedError

    def tocsr(self) -> "CSR":
        """Convert to CSR (may sort and merge duplicates)."""
        raise NotImplementedError

    def tocsc(self) -> "CSC":
        """Convert to CSC (may sort and merge duplicates)."""
        raise NotImplementedError

    def tobsr(self, blocksize: tuple[int, int]):
        """Convert to BSR; not implemented as a native format.

        Use ``to_torch(layout="bsr", blocksize=...)`` or
        ``to_scipy(format="bsr")`` to export block storage.
        """
        raise NotImplementedError(
            "BSR is not a native btorch format yet; export with "
            "to_torch(layout='bsr', blocksize=...) or to_scipy(format='bsr')."
        )

    def _as(self, fmt: str):
        if self.format != fmt:
            hint = (
                "BSR is not a native format; export with to_torch(layout='bsr')."
                if fmt == "bsr"
                else f"Call to{fmt}() to convert explicitly."
            )
            raise TypeError(
                f"This sparse array is stored as {self.format.upper()}, not "
                f"{fmt.upper()}; as_{fmt}() never converts. {hint}"
            )
        return self

    def as_coo(self) -> "COO":
        """Return ``self`` if it is stored as COO, else raise ``TypeError``."""
        return self._as("coo")

    def as_csr(self) -> "CSR":
        """Return ``self`` if it is stored as CSR, else raise ``TypeError``."""
        return self._as("csr")

    def as_csc(self) -> "CSC":
        """Return ``self`` if it is stored as CSC, else raise ``TypeError``."""
        return self._as("csc")

    def as_bsr(self, blocksize: tuple[int, int] | None = None):
        """BSR is not a native format; always raises ``TypeError``."""
        return self._as("bsr")

    # -------------------------------------------------------------- interop
    @classmethod
    def from_torch(cls, tensor: Tensor) -> "Sparse":
        """Wrap a ``torch`` sparse tensor (see
        :func:`btorch.sparse.from_torch`)."""
        from .conversion import from_torch

        return from_torch(tensor)

    @classmethod
    def from_scipy(cls, array: Any, *, device=None, dtype=None) -> "Sparse":
        """Convert a SciPy sparse array or matrix (see
        :func:`btorch.sparse.from_scipy`)."""
        from .conversion import from_scipy

        return from_scipy(array, device=device, dtype=dtype)

    def to_torch(self, layout: str | torch.layout | None = None, **kwargs) -> Tensor:
        """Export as a ``torch`` sparse tensor.

        Args:
            layout: ``"coo"``, ``"csr"``, ``"csc"``, ``"bsr"`` or ``"bsc"``
                (or the ``torch.sparse_*`` layout). Defaults to the stored
                format.
            **kwargs: ``blocksize`` for the block layouts.
        """
        from .conversion import to_torch

        return to_torch(self, layout, **kwargs)

    def to_scipy(self, format: str | None = None, **kwargs):
        """Export as a ``scipy.sparse`` array (CPU, detached).

        Args:
            format: Any SciPy format name; defaults to the stored format.
            **kwargs: ``blocksize`` for ``"bsr"``.
        """
        from .conversion import to_scipy

        return to_scipy(self, format, **kwargs)

    def to_dense(self) -> Tensor:
        """Materialise as a dense tensor of shape :attr:`shape`."""
        return self.tocoo().to_dense()

    # ------------------------------------------------------- tensor handling
    def with_values(self, values: Tensor) -> "Sparse":
        """Return an array with the same pattern and new values.

        ``values`` is ``[*batch, nnz, *dense]``; a different leading batch
        shape creates a shared-pattern batch.
        """
        raise NotImplementedError

    def _map_tensors(self, index_fn, value_fn) -> "Sparse":
        """Apply ``index_fn`` to index tensors and ``value_fn`` to values."""
        raise NotImplementedError

    def to(self, *args, **kwargs) -> "Sparse":
        """Move to a device and/or cast the values (indices stay integer)."""
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)

        return self._map_tensors(
            lambda index: index.to(device=device),
            lambda values: values.to(device=device, dtype=dtype),
        )

    def cpu(self) -> "Sparse":
        return self.to("cpu")

    def cuda(self, device=None) -> "Sparse":
        return self.to("cuda" if device is None else device)

    def float(self) -> "Sparse":
        return self.to(torch.float32)

    def double(self) -> "Sparse":
        return self.to(torch.float64)

    def detach(self) -> "Sparse":
        return self.with_values(self._values.detach())

    # ------------------------------------------------------------- products
    def matvec(self, x: Tensor) -> Tensor:
        """Apply the operator along the last axis of ``x``.

        See :func:`btorch.sparse.matvec`.
        """
        from .ops import matvec

        return matvec(self, x)

    def rmatvec(self, x: Tensor) -> Tensor:
        """Apply the transposed operator along the last axis of ``x``."""
        from .ops import rmatvec

        return rmatvec(self, x)

    def __matmul__(self, other):
        from .ops import matmul

        return matmul(self, other)

    def __rmatmul__(self, other):
        from .ops import matmul

        return matmul(other, self)

    @property
    def T(self) -> "Sparse":
        """Transpose of the two sparse dimensions."""
        return self.transpose()

    def transpose(self) -> "Sparse":
        raise NotImplementedError

    def __repr__(self) -> str:
        parts = [f"shape={self.shape}", f"nnz={self.nnz}", f"dtype={self.dtype}"]
        if self.batch_dim():
            parts.append(f"batch_shape={self.batch_shape}")
        if self.dense_dim():
            parts.append(f"dense_shape={self.dense_shape}")
        if self.device.type != "cpu":
            parts.append(f"device={self.device}")
        return f"{type(self).__name__}({', '.join(parts)})"


# A sparse array is a linear operator, without inheriting the operator
# algebra (``A + B`` on sparse arrays is not defined yet).
LinearOperator.register(Sparse)


def _check_index(name: str, index: Tensor) -> Tensor:
    if index.numel() == 0:
        # An empty Python list becomes a float tensor; it is still a valid
        # (empty) index.
        return index.to(torch.long)
    if index.is_floating_point() or index.dtype == torch.bool:
        raise TypeError(f"{name} must be an integer tensor, got {index.dtype}.")
    return index.to(torch.long)


def _as_tensor(x, *, dtype=None, device=None) -> Tensor:
    if isinstance(x, Tensor):
        return (
            x if dtype is None and device is None else x.to(device=device, dtype=dtype)
        )
    return torch.as_tensor(x, dtype=dtype, device=device)
