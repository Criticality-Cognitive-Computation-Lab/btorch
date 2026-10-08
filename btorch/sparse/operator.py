"""Linear operators that are not explicit sparse arrays.

A :class:`LinearOperator` is anything that maps ``x [..., N]`` to
``y [..., M]`` linearly. A stored sparse array (:class:`~btorch.sparse.Sparse`)
is one way to do that; this module holds the other ones:

- :class:`StructuredOperator`: closed-form products with no stored matrix
  (:class:`ConstantOperator`, :class:`DiagonalOperator`,
  :class:`LowRankOperator`).
- :class:`ImplicitOperator`: user callables; the matrix is never built unless
  the user provides an edge enumeration and asks for it.
- :class:`CompositeOperator`: lazy sums, products, scalings and transposes of
  other operators (:class:`SumOperator`, :class:`ProductOperator`,
  :class:`ScaledOperator`, :class:`TransposedOperator`).

Procedural operators are deliberately *not* ``Sparse``: they have no ``nnz``,
no format and no stored indices. What an operator can do is queried with
:meth:`LinearOperator.supports`; asking for an unsupported capability raises
``NotImplementedError`` naming it.

Products follow the conventions of :func:`btorch.sparse.matmul`: for
``A.shape == (M, N)``, ``A @ x`` maps a length-``N`` vector to a length-``M``
vector, and ``A.matvec(x)`` applies ``A`` along the last axis of ``x``.

A ``Sparse`` is accepted wherever an operator is expected (as a part of a sum
or a product, on either side of ``@`` and ``+``).

Examples:
    >>> import torch
    >>> from btorch.sparse.operator import ConstantOperator, DiagonalOperator
    >>> A = ConstantOperator((2, 3), 0.5) @ DiagonalOperator(torch.ones(3))
    >>> A @ torch.tensor([1.0, 2.0, 3.0])
    tensor([3., 3.])
"""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor


# NOTE: this module must not import the rest of ``btorch.sparse`` at module
# level, so that ``base.py`` can import ``LinearOperator`` from here and call
# ``LinearOperator.register(Sparse)``. The sparse constructors needed for
# materialisation are imported lazily inside the functions that use them.

CAPABILITIES: frozenset[str] = frozenset(
    {"matvec", "matmat", "rmatvec", "enumerate_edges", "materialize_sparse"}
)
"""Optional capabilities of a linear operator.

- ``"matvec"``: ``A.matvec(x)`` and ``A @ x``.
- ``"matmat"``: ``A @ X`` with a matrix right-hand side.
- ``"rmatvec"``: the transpose apply, ``A.rmatvec(x)``, ``x @ A`` and ``A.T``.
- ``"enumerate_edges"``: ``A.enumerate_edges()`` lists ``(rows, cols,
  values)``.
- ``"materialize_sparse"``: ``A.materialize()`` / ``tocoo()`` / ``tocsr()``.
"""

_APPLY = frozenset({"matvec", "matmat", "rmatvec"})
_ALL = CAPABILITIES

Edges = tuple[Tensor, Tensor, Tensor]


class LinearOperator(metaclass=ABCMeta):
    """Base class of everything that acts as a matrix of shape ``(M, N)``.

    Subclasses implement the private hooks ``_matvec`` / ``_rmatvec`` (and
    optionally ``_matmat`` / ``_enumerate_edges``) and list what they provide
    in ``_capabilities``. The public methods validate shapes and capabilities
    and are shared by all operators.

    Operators are plain Python objects holding tensors, like
    :class:`~btorch.sparse.Sparse`; they are not ``nn.Module``. A learnable
    tensor used by an operator must be registered on the owning module, which
    passes it to the operator (see the examples of the concrete classes).

    Operator algebra is lazy and never builds a matrix:

    - ``A + B``, ``A - B``: :class:`SumOperator`.
    - ``A @ B`` with another operator or a ``Sparse``: :class:`ProductOperator`.
    - ``alpha * A``, ``A * alpha``, ``A / alpha``, ``-A``:
      :class:`ScaledOperator`.
    - ``A.T``: :class:`TransposedOperator`.

    The class is an ABC only so that explicit sparse arrays can be registered
    as virtual subclasses (``LinearOperator.register(Sparse)``); it has no
    abstract methods.
    """

    _capabilities: frozenset[str] = frozenset()
    _shape: tuple[int, int]
    dtype: torch.dtype | None = None
    device: torch.device | None = None

    # ----------------------------------------------------------------- shape
    @property
    def shape(self) -> tuple[int, int]:
        """``(M, N)``: the operator maps length-``N`` to length-``M``."""
        return self._shape

    @property
    def ndim(self) -> int:
        return 2

    # ---------------------------------------------------------- capabilities
    def supports(self, name: str) -> bool:
        """Whether the optional capability ``name`` is available.

        Args:
            name: One of :data:`CAPABILITIES`.

        Raises:
            ValueError: If ``name`` is not a known capability.
        """
        if name not in CAPABILITIES:
            raise ValueError(
                f"Unknown capability {name!r}; expected one of {sorted(CAPABILITIES)}."
            )
        return name in self._capabilities

    def _require(self, name: str) -> None:
        if name not in self._capabilities:
            raise NotImplementedError(
                f"{type(self).__name__} of shape {self.shape} does not support "
                f"the capability {name!r} (supported: "
                f"{sorted(self._capabilities)})."
            )

    # --------------------------------------------------------------- products
    def matvec(self, x: Tensor) -> Tensor:
        """Apply the operator along the last axis: ``[..., N] -> [..., M]``.

        .. math::
            y_{\\ldots m} = \\sum_n A_{m n}\\, x_{\\ldots n}

        All leading dimensions of ``x`` are independent samples.
        """
        self._require("matvec")
        if x.shape[-1] != self._shape[1]:
            raise ValueError(
                f"matvec: the last dimension of the input is {x.shape[-1]}, "
                f"expected {self._shape[1]} for an operator of shape {self.shape}."
            )
        return self._matvec(x)

    def rmatvec(self, x: Tensor) -> Tensor:
        """Apply the transpose along the last axis: ``[..., M] -> [..., N]``.

        Equals ``self.T.matvec(x)`` and, for 1-D ``x``, ``x @ self``.
        """
        self._require("rmatvec")
        if x.shape[-1] != self._shape[0]:
            raise ValueError(
                f"rmatvec: the last dimension of the input is {x.shape[-1]}, "
                f"expected {self._shape[0]} for an operator of shape {self.shape}."
            )
        return self._rmatvec(x)

    def matmat(self, x: Tensor) -> Tensor:
        """``A @ X`` for ``X [..., N, K]``, returning ``[..., M, K]``."""
        self._require("matmat")
        if x.ndim < 2 or x.shape[-2] != self._shape[1]:
            raise ValueError(
                f"matmat: the input needs shape [..., {self._shape[1]}, K] for "
                f"an operator of shape {self.shape}, got {tuple(x.shape)}."
            )
        return self._matmat(x)

    def _matvec(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def _rmatvec(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def _matmat(self, x: Tensor) -> Tensor:
        # Columns of X are vectors: move them to the last axis and back.
        return self.matvec(x.mT).mT

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            if other.ndim == 0:
                raise ValueError("A @ x needs x with at least one dimension.")
            if other.ndim == 1:
                return self.matvec(other)
            return self.matmat(other)
        if is_linear_operator(other):
            return ProductOperator([self, other])
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, Tensor):
            if other.ndim == 0:
                raise ValueError("x @ A needs x with at least one dimension.")
            # 1-D: x [M] -> [N]; matrix: X [..., K, M] -> [..., K, N]. In both
            # cases the transpose acts along the last axis.
            return self.rmatvec(other)
        if is_linear_operator(other):
            return ProductOperator([other, self])
        return NotImplemented

    # ---------------------------------------------------------- lazy algebra
    def __add__(self, other):
        if is_linear_operator(other):
            return SumOperator([self, other])
        return NotImplemented

    def __radd__(self, other):
        if is_linear_operator(other):
            return SumOperator([other, self])
        return NotImplemented

    def __sub__(self, other):
        if is_linear_operator(other):
            return SumOperator([self, ScaledOperator(other, -1.0)])
        return NotImplemented

    def __rsub__(self, other):
        if is_linear_operator(other):
            return SumOperator([other, ScaledOperator(self, -1.0)])
        return NotImplemented

    def __mul__(self, alpha):
        if _is_scalar(alpha):
            return ScaledOperator(self, alpha)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, alpha):
        if _is_scalar(alpha):
            return ScaledOperator(self, 1.0 / alpha)
        return NotImplemented

    def __neg__(self):
        return ScaledOperator(self, -1.0)

    @property
    def T(self) -> "LinearOperator":
        """Lazy transpose; needs the ``"rmatvec"`` capability."""
        return TransposedOperator(self)

    def transpose(self) -> "LinearOperator":
        return self.T

    # -------------------------------------------------------- materialisation
    def enumerate_edges(self) -> Edges:
        """List the entries as ``(rows [E], cols [E], values [E])``.

        Entries may repeat; repeated coordinates are summed, as in
        :class:`~btorch.sparse.COO`. ``values`` keeps its autograd history.
        """
        self._require("enumerate_edges")
        return self._enumerate_edges()

    def _enumerate_edges(self) -> Edges:
        raise NotImplementedError

    def materialize(self, format: str | None = None):
        """Build an explicit :class:`~btorch.sparse.Sparse` with these entries.

        Args:
            format: Optional ``"coo"``, ``"csr"`` or ``"csc"``. Without it the
                result is format-agnostic.

        Raises:
            NotImplementedError: If the operator cannot list its entries
                (capability ``"materialize_sparse"``).
        """
        self._require("materialize_sparse")
        from . import from_edges

        rows, cols, values = self._enumerate_edges()
        out = from_edges(rows, cols, values, self.shape)
        if format is not None and out.format != format:
            out = getattr(out, f"to{format}")()
        return out

    def tocoo(self):
        """Materialise as COO (capability ``"materialize_sparse"``)."""
        return self.materialize("coo")

    def tocsr(self):
        """Materialise as CSR (capability ``"materialize_sparse"``)."""
        return self.materialize("csr")

    def tocsc(self):
        """Materialise as CSC (capability ``"materialize_sparse"``)."""
        return self.materialize("csc")

    def to_dense(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Dense ``[M, N]`` matrix; an explicit request, never done implicitly.

        The default applies the operator to the identity (``N`` right-hand
        sides at once), so it only needs ``"matvec"`` and stays
        differentiable.

        Args:
            dtype: Dtype of the identity; defaults to the operator dtype if
                it has one.
            device: Device of the identity; defaults to the operator device.
        """
        eye = torch.eye(
            self._shape[1],
            dtype=dtype if dtype is not None else self.dtype,
            device=device if device is not None else self.device,
        )
        # Row n of matvec(eye) is A @ e_n, i.e. column n of A.
        return self.matvec(eye).mT

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"


def is_linear_operator(obj: Any) -> bool:
    """Whether ``obj`` can be used as a linear operator here.

    True for a :class:`LinearOperator` and for anything that looks like one
    (a 2-D ``shape`` with ``matvec`` and ``rmatvec``), which is how a
    :class:`~btorch.sparse.Sparse` matrix is accepted without inheriting from
    :class:`LinearOperator`. Dense tensors are data, not operators.
    """
    if isinstance(obj, LinearOperator):
        return True
    if isinstance(obj, Tensor):
        return False
    return (
        callable(getattr(obj, "matvec", None))
        and callable(getattr(obj, "rmatvec", None))
        and hasattr(obj, "shape")
    )


def _is_scalar(alpha: Any) -> bool:
    if isinstance(alpha, Tensor):
        return alpha.ndim == 0
    return isinstance(alpha, int | float | complex) and not isinstance(alpha, bool)


def _as_part(obj: Any) -> Any:
    """Validate an operand of a composite (operator or plain 2-D
    ``Sparse``)."""
    if not is_linear_operator(obj):
        raise TypeError(
            f"Expected a LinearOperator or a sparse matrix, got {type(obj).__name__}."
        )
    if len(obj.shape) != 2:
        raise ValueError(
            "Only plain matrices can be combined into an operator, got shape "
            f"{tuple(obj.shape)} (batch or dense entry dimensions are not "
            "supported in composites)."
        )
    return obj


def _capabilities_of(part: Any) -> frozenset[str]:
    # Operators defined here carry ``_capabilities``. Anything else is an
    # explicit sparse array (duck-typed or registered as a virtual subclass),
    # which supports every capability.
    return getattr(part, "_capabilities", _ALL)


def _supports(part: Any, name: str) -> bool:
    return name in _capabilities_of(part)


def _edges_of(part: Any) -> Edges:
    if hasattr(part, "_capabilities"):
        return part.enumerate_edges()
    coo = part.tocoo()
    return coo.row, coo.col, coo.values()


def _first(parts: Sequence[Any], name: str):
    for part in parts:
        value = getattr(part, name, None)
        if value is not None:
            return value
    return None


def _composite_dtype(parts: Sequence[Any]) -> torch.dtype | None:
    dtype = _first(parts, "dtype")
    if dtype is None:
        return None
    for part in parts:
        part_dtype = getattr(part, "dtype", None)
        if part_dtype is not None:
            dtype = torch.promote_types(dtype, part_dtype)
    return dtype


def _composite_device(parts: Sequence[Any]) -> torch.device | None:
    devices = {
        torch.device(part_device)
        for part in parts
        if (part_device := getattr(part, "device", None)) is not None
    }
    if len(devices) > 1:
        raise ValueError(
            "Composite operators require every operand on one device, got "
            f"{sorted(map(str, devices))}."
        )
    return next(iter(devices), None)


# ---------------------------------------------------------------- structured
class StructuredOperator(LinearOperator):
    """Operator with a closed-form product and no stored matrix.

    Structured operators support every apply capability and can list their
    entries on request (:meth:`enumerate_edges` / :meth:`materialize`), which
    is the only time memory proportional to the number of entries is used.
    """

    _capabilities = _ALL


class ConstantOperator(StructuredOperator):
    """All-to-all operator with one shared weight: ``A[m, n] = value``.

    .. math::
        y_{\\ldots m} = \\text{value} \\cdot \\sum_n x_{\\ldots n}

    The product costs ``O(M + N)`` instead of ``O(M N)``.

    Args:
        shape: ``(M, N)``.
        value: Scalar weight: a Python number or a 0-dim tensor (for example
            an ``nn.Parameter``; gradients flow to it).
        dtype: Dtype used when ``value`` is a Python number.
        device: Device used when ``value`` is a Python number.

    Examples:
        >>> A = ConstantOperator((2, 3), 2.0)
        >>> A @ torch.tensor([1.0, 2.0, 3.0])
        tensor([12., 12.])
    """

    def __init__(
        self,
        shape: tuple[int, int],
        value: Tensor | float = 1.0,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        self._shape = _check_shape(shape)
        if isinstance(value, Tensor):
            if value.ndim != 0:
                raise ValueError(
                    "ConstantOperator needs a scalar value, got shape "
                    f"{tuple(value.shape)}."
                )
            self.dtype, self.device = value.dtype, value.device
        else:
            self.dtype = dtype
            self.device = torch.device(device) if device is not None else None
        self.value = value

    def _matvec(self, x: Tensor) -> Tensor:
        total = x.sum(-1, keepdim=True) * self.value
        return total.expand(*x.shape[:-1], self._shape[0])

    def _rmatvec(self, x: Tensor) -> Tensor:
        total = x.sum(-1, keepdim=True) * self.value
        return total.expand(*x.shape[:-1], self._shape[1])

    def _value_tensor(self) -> Tensor:
        if isinstance(self.value, Tensor):
            return self.value
        return torch.tensor(self.value, dtype=self.dtype, device=self.device)

    def _enumerate_edges(self) -> Edges:
        m, n = self._shape
        value = self._value_tensor()
        rows = torch.arange(m, device=value.device).repeat_interleave(n)
        cols = torch.arange(n, device=value.device).repeat(m)
        return rows, cols, value.expand(m * n)

    def to_dense(self, *, dtype=None, device=None) -> Tensor:
        return self._value_tensor().to(dtype=dtype, device=device).expand(self._shape)


class DiagonalOperator(StructuredOperator):
    """One-to-one operator: ``A = diag(d)``, ``y = d * x``.

    Args:
        diag: ``[N]`` diagonal (may be an ``nn.Parameter``).

    Examples:
        >>> DiagonalOperator(torch.tensor([1.0, 2.0])) @ torch.tensor([3.0, 4.0])
        tensor([3., 8.])
    """

    def __init__(self, diag: Tensor):
        if not isinstance(diag, Tensor) or diag.ndim != 1:
            raise ValueError("DiagonalOperator needs a 1-D tensor.")
        self.diag = diag
        self._shape = (diag.shape[0], diag.shape[0])
        self.dtype, self.device = diag.dtype, diag.device

    def _matvec(self, x: Tensor) -> Tensor:
        return x * self.diag

    _rmatvec = _matvec

    def _enumerate_edges(self) -> Edges:
        index = torch.arange(self._shape[0], device=self.diag.device)
        return index, index, self.diag

    def to_dense(self, *, dtype=None, device=None) -> Tensor:
        return torch.diag(self.diag).to(dtype=dtype, device=device)


class LowRankOperator(StructuredOperator):
    """Low-rank operator ``A = U V^T`` applied as two thin products.

    .. math::
        y = U\\,(V^\\top x)

    The product costs ``O((M + N) r)``. ``U V^T`` is dense, so listing its
    entries is not offered (``"enumerate_edges"`` and ``"materialize_sparse"``
    are unsupported); use :meth:`to_dense` for the explicit matrix.

    Args:
        U: ``[M, r]`` left factor.
        V: ``[N, r]`` right factor. Both may be ``nn.Parameter``.
    """

    _capabilities = _APPLY

    def __init__(self, U: Tensor, V: Tensor):
        if U.ndim != 2 or V.ndim != 2 or U.shape[1] != V.shape[1]:
            raise ValueError(
                "LowRankOperator needs U [M, r] and V [N, r] with the same "
                f"rank, got {tuple(U.shape)} and {tuple(V.shape)}."
            )
        self.U, self.V = U, V
        self._shape = (U.shape[0], V.shape[0])
        self.dtype, self.device = U.dtype, U.device

    @property
    def rank(self) -> int:
        return int(self.U.shape[1])

    def _matvec(self, x: Tensor) -> Tensor:
        return (x @ self.V) @ self.U.mT

    def _rmatvec(self, x: Tensor) -> Tensor:
        return (x @ self.U) @ self.V.mT

    def to_dense(self, *, dtype=None, device=None) -> Tensor:
        return (self.U @ self.V.mT).to(dtype=dtype, device=device)


def _check_shape(shape: Sequence[int]) -> tuple[int, int]:
    shape = tuple(int(s) for s in shape)
    if len(shape) != 2 or min(shape) < 0:
        raise ValueError(f"An operator shape is (M, N) with M, N >= 0, got {shape}.")
    return shape


# ------------------------------------------------------------------ implicit
class ImplicitOperator(LinearOperator):
    """Operator defined by user callables; nothing is stored or materialised.

    Only ``matvec`` is required. Every other capability exists exactly when
    its callable was given, so an apply-only operator refuses ``.T``,
    ``x @ A`` and ``tocsr()`` with a clear error, while an enumerable
    procedural operator can be materialised on request.

    The callables are stored as given and called directly: the operator is
    differentiable when they are, and works under ``torch.compile`` when they
    do.

    Args:
        shape: ``(M, N)``.
        matvec: ``x [..., N] -> y [..., M]``, applied along the last axis.
        rmatvec: Optional transpose apply ``x [..., M] -> y [..., N]``.
        enumerate_edges: Optional callable without arguments returning
            ``(rows [E], cols [E], values [E])``. Enables
            :meth:`~LinearOperator.materialize`, ``tocoo()`` and ``tocsr()``.
        dtype: Optional dtype, used by :meth:`~LinearOperator.to_dense`.
        device: Optional device, used by :meth:`~LinearOperator.to_dense`.
        matmat: Optional ``X [..., N, K] -> [..., M, K]``. Defaults to
            ``matvec`` applied to the columns.

    Examples:
        A circular shift, never stored as a matrix:

        >>> shift = ImplicitOperator(
        ...     (4, 4),
        ...     matvec=lambda x: x.roll(1, -1),
        ...     rmatvec=lambda x: x.roll(-1, -1),
        ... )
        >>> shift @ torch.tensor([1.0, 2.0, 3.0, 4.0])
        tensor([4., 1., 2., 3.])
    """

    def __init__(
        self,
        shape: tuple[int, int],
        matvec: Callable[[Tensor], Tensor],
        rmatvec: Callable[[Tensor], Tensor] | None = None,
        enumerate_edges: Callable[[], Edges] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        *,
        matmat: Callable[[Tensor], Tensor] | None = None,
    ):
        if not callable(matvec):
            raise TypeError("ImplicitOperator needs a callable matvec.")
        self._shape = _check_shape(shape)
        self._matvec_fn = matvec
        self._rmatvec_fn = rmatvec
        self._matmat_fn = matmat
        self._edges_fn = enumerate_edges
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else None
        capabilities = {"matvec", "matmat"}
        if rmatvec is not None:
            capabilities.add("rmatvec")
        if enumerate_edges is not None:
            capabilities |= {"enumerate_edges", "materialize_sparse"}
        self._capabilities = frozenset(capabilities)

    def _matvec(self, x: Tensor) -> Tensor:
        return self._matvec_fn(x)

    def _rmatvec(self, x: Tensor) -> Tensor:
        return self._rmatvec_fn(x)

    def _matmat(self, x: Tensor) -> Tensor:
        if self._matmat_fn is not None:
            return self._matmat_fn(x)
        return self._matvec_fn(x.mT).mT

    def _enumerate_edges(self) -> Edges:
        rows, cols, values = self._edges_fn()
        return rows, cols, values


# ----------------------------------------------------------------- composite
class CompositeOperator(LinearOperator):
    """Operator built lazily from other operators and sparse matrices.

    The concrete composites are :class:`SumOperator`,
    :class:`ProductOperator`, :class:`ScaledOperator` and
    :class:`TransposedOperator`; they are normally created with ``+``, ``@``,
    ``*`` and ``.T``. A composite supports a capability only if all of its
    parts do (a ``Sparse`` part supports everything).

    Attributes:
        parts: The operands, in order.
    """

    parts: tuple[Any, ...]

    def _set_parts(self, parts: Sequence[Any]) -> None:
        self.parts = tuple(_as_part(p) for p in parts)
        if not self.parts:
            raise ValueError(f"{type(self).__name__} needs at least one operand.")
        self.dtype = _composite_dtype(self.parts)
        self.device = _composite_device(self.parts)

    def _common(self, allowed: frozenset[str] = _ALL) -> frozenset[str]:
        out = allowed
        for part in self.parts:
            out = out & _capabilities_of(part)
        return out

    def __repr__(self) -> str:
        inner = ", ".join(repr(p) for p in self.parts)
        return f"{type(self).__name__}(shape={self.shape}, parts=[{inner}])"


class SumOperator(CompositeOperator):
    """Lazy sum ``A_1 + A_2 + ...`` of operators with the same shape.

    Listing the entries concatenates the entries of the parts (coordinates
    present in several parts are stored several times and summed by every
    sparse operation), so no sparse addition is run.

    Args:
        parts: Operators and/or sparse matrices of one shape.
    """

    def __init__(self, parts: Sequence[Any]):
        flat: list[Any] = []
        for part in parts:
            # Flatten nested sums so a long chain stays one level deep.
            flat.extend(part.parts if isinstance(part, SumOperator) else [part])
        self._set_parts(flat)
        shape = tuple(self.parts[0].shape)
        for part in self.parts:
            if tuple(part.shape) != shape:
                raise ValueError(
                    "Cannot add operators of different shapes: "
                    f"{shape} and {tuple(part.shape)}."
                )
        self._shape = shape
        self._capabilities = self._common()

    def _matvec(self, x: Tensor) -> Tensor:
        out = self.parts[0].matvec(x)
        for part in self.parts[1:]:
            out = out + part.matvec(x)
        return out

    def _rmatvec(self, x: Tensor) -> Tensor:
        out = self.parts[0].rmatvec(x)
        for part in self.parts[1:]:
            out = out + part.rmatvec(x)
        return out

    def _enumerate_edges(self) -> Edges:
        edges = [_edges_of(part) for part in self.parts]
        dtype = edges[0][2].dtype
        for _, _, values in edges[1:]:
            dtype = torch.promote_types(dtype, values.dtype)
        return (
            torch.cat([e[0] for e in edges]),
            torch.cat([e[1] for e in edges]),
            torch.cat([e[2].to(dtype) for e in edges]),
        )


class ProductOperator(CompositeOperator):
    """Lazy composition ``A_1 @ A_2 @ ...`` (the rightmost is applied first).

    The product is never formed: ``matvec`` applies the parts right to left
    and ``rmatvec`` left to right. Its entries cannot be listed (that would
    be a sparse-sparse product), so ``"enumerate_edges"`` and
    ``"materialize_sparse"`` are unsupported.

    Args:
        parts: Operators and/or sparse matrices with chained shapes
            ``(M, K_1), (K_1, K_2), ..., (K_j, N)``.
    """

    def __init__(self, parts: Sequence[Any]):
        flat: list[Any] = []
        for part in parts:
            flat.extend(part.parts if isinstance(part, ProductOperator) else [part])
        self._set_parts(flat)
        dtypes = {
            part_dtype
            for part in self.parts
            if (part_dtype := getattr(part, "dtype", None)) is not None
        }
        if len(dtypes) > 1:
            raise ValueError(
                "Product operators require every operand to use one dtype, got "
                f"{sorted(map(str, dtypes))}."
            )
        for left, right in zip(self.parts[:-1], self.parts[1:]):
            if left.shape[1] != right.shape[0]:
                raise ValueError(
                    "Cannot compose operators of shapes "
                    f"{tuple(left.shape)} and {tuple(right.shape)}: the inner "
                    "dimensions differ."
                )
        self._shape = (self.parts[0].shape[0], self.parts[-1].shape[1])
        self._capabilities = self._common(_APPLY)
        self._reversed = self.parts[::-1]

    def _matvec(self, x: Tensor) -> Tensor:
        for part in self._reversed:
            x = part.matvec(x)
        return x

    def _rmatvec(self, x: Tensor) -> Tensor:
        for part in self.parts:
            x = part.rmatvec(x)
        return x


class ScaledOperator(CompositeOperator):
    """Lazy scalar multiple ``alpha * A``.

    Args:
        operator: Operator or sparse matrix.
        alpha: Python number or 0-dim tensor (may be an ``nn.Parameter``).
    """

    def __init__(self, operator: Any, alpha: Tensor | float):
        if not _is_scalar(alpha):
            raise TypeError("ScaledOperator needs a scalar factor.")
        self._set_parts([operator])
        self.alpha = alpha
        if isinstance(alpha, Tensor):
            if self.device is not None and alpha.device != self.device:
                raise ValueError(
                    "Scaled operators require the factor and operand on one "
                    f"device, got {alpha.device} and {self.device}."
                )
            if self.dtype is not None:
                self.dtype = torch.promote_types(self.dtype, alpha.dtype)
            else:
                self.dtype = alpha.dtype
            self.device = alpha.device if self.device is None else self.device
        elif self.dtype is not None:
            alpha_dtype = torch.tensor(alpha).dtype
            self.dtype = torch.promote_types(self.dtype, alpha_dtype)
        self._shape = tuple(operator.shape)
        self._capabilities = self._common()

    def _matvec(self, x: Tensor) -> Tensor:
        return self.parts[0].matvec(x) * self.alpha

    def _rmatvec(self, x: Tensor) -> Tensor:
        return self.parts[0].rmatvec(x) * self.alpha

    def _enumerate_edges(self) -> Edges:
        rows, cols, values = _edges_of(self.parts[0])
        return rows, cols, values * self.alpha


class TransposedOperator(CompositeOperator):
    """Lazy transpose ``A^T``; swaps ``matvec`` and ``rmatvec``.

    Created by :attr:`LinearOperator.T`. Building it fails immediately (not on
    first use) if the operator has no ``"rmatvec"`` capability.

    Args:
        operator: Operator or sparse matrix.
    """

    def __init__(self, operator: Any):
        self._set_parts([operator])
        if not _supports(operator, "rmatvec"):
            operator._require("rmatvec")  # raises, naming the capability
        inner = _capabilities_of(operator)
        capabilities = {"matvec", "matmat"} | (
            inner & {"enumerate_edges", "materialize_sparse"}
        )
        if "matvec" in inner:
            capabilities.add("rmatvec")
        self._capabilities = frozenset(capabilities)
        self._shape = (operator.shape[1], operator.shape[0])

    def _matvec(self, x: Tensor) -> Tensor:
        return self.parts[0].rmatvec(x)

    def _rmatvec(self, x: Tensor) -> Tensor:
        return self.parts[0].matvec(x)

    def _enumerate_edges(self) -> Edges:
        rows, cols, values = _edges_of(self.parts[0])
        return cols, rows, values

    @property
    def T(self):
        return self.parts[0]


__all__ = [
    "CAPABILITIES",
    "CompositeOperator",
    "ConstantOperator",
    "DiagonalOperator",
    "ImplicitOperator",
    "LinearOperator",
    "LowRankOperator",
    "ProductOperator",
    "ScaledOperator",
    "StructuredOperator",
    "SumOperator",
    "TransposedOperator",
    "is_linear_operator",
]
