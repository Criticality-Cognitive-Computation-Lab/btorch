"""Linear operators that are not explicit sparse arrays.

``btorch.sparse.operator`` holds the procedural side of the numerical
hierarchy: structured operators with a closed-form product, implicit
operators defined by callables, and lazy composites. None of them stores a
matrix, so every test here compares against a *dense reference matrix* that
is written down independently, and checks the same contract for each
operator:

- ``matvec`` / ``rmatvec`` act along the last axis and accept any leading
  sample dimensions;
- ``@`` follows ``torch.matmul`` on both sides;
- gradients reach the parameters an operator was built from;
- optional capabilities are reported by ``supports`` and refused with a
  ``NotImplementedError`` that names the capability.

All matrices are non-square (where the operator allows it) so that a
transposed result has the wrong shape and cannot pass by accident.
"""

import pytest
import torch
from torch import nn

from btorch import sparse
from btorch.sparse.operator import (
    CAPABILITIES,
    CompositeOperator,
    ConstantOperator,
    DiagonalOperator,
    ImplicitOperator,
    LinearOperator,
    LowRankOperator,
    ProductOperator,
    ScaledOperator,
    StructuredOperator,
    SumOperator,
    TransposedOperator,
    is_linear_operator,
)


M, N = 4, 6  # operator shape (M, N): maps length-N to length-M
DTYPE = torch.float64  # double precision so gradients can be compared tightly


def _rand(*shape, seed=0):
    return torch.randn(*shape, dtype=DTYPE, generator=torch.manual_seed(seed))


def _sparse_matrix(shape=(M, N), seed=3):
    """A small explicit ``Sparse`` and its dense matrix.

    Used to check that a ``Sparse`` is accepted wherever an operator is.
    """
    dense = _rand(*shape, seed=seed)
    dense = dense * (dense.abs() > 0.7)  # drop about half of the entries
    rows, cols = dense.nonzero(as_tuple=True)
    return sparse.from_edges(rows, cols, dense[rows, cols], shape), dense


# ------------------------------------------------------------------ fixtures
# Each case builds ``(operator, dense reference, parameters)``. The dense
# reference is computed from the parameters with ordinary dense algebra, so
# autograd through the reference gives the expected parameter gradients.
def _constant():
    value = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    return ConstantOperator((M, N), value), value.expand(M, N), [value]


def _diagonal():
    diag = _rand(N, seed=1).requires_grad_()
    return DiagonalOperator(diag), torch.diag(diag), [diag]


def _low_rank():
    U = _rand(M, 2, seed=1).requires_grad_()
    V = _rand(N, 2, seed=2).requires_grad_()
    return LowRankOperator(U, V), U @ V.T, [U, V]


def _implicit():
    # An implicit operator that happens to wrap a dense weight: the callables
    # are the only thing the operator knows about.
    W = _rand(M, N, seed=4).requires_grad_()
    op = ImplicitOperator(
        (M, N), matvec=lambda x: x @ W.T, rmatvec=lambda x: x @ W, dtype=DTYPE
    )
    return op, W, [W]


def _sum():
    # structured + explicit sparse: the lazy sum never adds the matrices.
    value = torch.tensor(-0.3, dtype=DTYPE, requires_grad=True)
    A, dense = _sparse_matrix()
    return ConstantOperator((M, N), value) + A, value.expand(M, N) + dense, [value]


def _product():
    # (M, N) = (M, 3) @ (3, N): a low-rank map followed by a row mixing.
    U = _rand(M, 3, seed=5).requires_grad_()
    diag = _rand(3, seed=6).requires_grad_()
    A, dense = _sparse_matrix((3, N), seed=7)
    op = ImplicitOperator((M, 3), lambda x: x @ U.T, lambda x: x @ U)
    return op @ DiagonalOperator(diag) @ A, U @ torch.diag(diag) @ dense, [U, diag]


def _scaled():
    alpha = torch.tensor(1.5, dtype=DTYPE, requires_grad=True)
    U = _rand(M, 2, seed=8).requires_grad_()
    V = _rand(N, 2, seed=9).requires_grad_()
    return alpha * LowRankOperator(U, V), alpha * (U @ V.T), [alpha, U, V]


def _transposed():
    # The transpose of an (N, M) operator is an (M, N) operator.
    U = _rand(N, 2, seed=10).requires_grad_()
    V = _rand(M, 2, seed=11).requires_grad_()
    return LowRankOperator(U, V).T, (U @ V.T).T, [U, V]


CASES = {
    "constant": _constant,
    "diagonal": _diagonal,
    "low_rank": _low_rank,
    "implicit": _implicit,
    "sum": _sum,
    "product": _product,
    "scaled": _scaled,
    "transposed": _transposed,
}


@pytest.fixture(params=list(CASES))
def case(request):
    return CASES[request.param]()


# ------------------------------------------------------------------ products
def test_shape_and_type(case):
    """Every operator is a ``LinearOperator`` with a 2-D ``(M, N)`` shape."""
    op, dense, _ = case
    assert isinstance(op, LinearOperator)
    assert op.shape == tuple(dense.shape)
    assert op.ndim == 2
    assert is_linear_operator(op)


@pytest.mark.parametrize("lead", [(), (3,), (2, 3)])
def test_matvec_and_rmatvec_match_dense(case, lead):
    """``matvec`` maps ``[..., N] -> [..., M]`` and ``rmatvec`` the reverse.

    The leading dimensions are independent samples (for example batch and
    time), exactly as for ``btorch.sparse.matvec``.
    """
    op, dense, _ = case
    m, n = op.shape
    x = _rand(*lead, n, seed=20)
    y = _rand(*lead, m, seed=21)
    torch.testing.assert_close(op.matvec(x), x @ dense.T)
    torch.testing.assert_close(op.rmatvec(y), y @ dense)


def test_matmul_both_sides_follows_torch_matmul(case):
    """``A @ x``, ``A @ X``, ``x @ A`` and ``X @ A`` equal the dense ones.

    A 1-D operand is a vector; a 2-D-or-more operand is a (stack of)
    matrices with ``torch.matmul`` semantics, i.e. ``A @ X`` acts on the
    second-to-last axis of ``X [..., N, K]``.
    """
    op, dense, _ = case
    m, n = op.shape
    x, y = _rand(n, seed=22), _rand(m, seed=23)
    X, Y = _rand(2, n, 5, seed=24), _rand(2, 5, m, seed=25)
    torch.testing.assert_close(op @ x, dense @ x)
    torch.testing.assert_close(op @ X, dense @ X)
    torch.testing.assert_close(y @ op, y @ dense)
    torch.testing.assert_close(Y @ op, Y @ dense)
    # matmat is the named form of ``A @ X``.
    torch.testing.assert_close(op.matmat(X), dense @ X)


def test_transpose_is_lazy_and_consistent(case):
    """``A.T`` swaps the shape and the two apply directions; ``A.T.T is A`` up
    to laziness (it applies exactly like ``A``)."""
    op, dense, _ = case
    m, n = op.shape
    y = _rand(3, m, seed=26)
    assert op.T.shape == (n, m)
    torch.testing.assert_close(op.T.matvec(y), y @ dense)
    torch.testing.assert_close(op.T.T.matvec(_rand(n)), dense @ _rand(n))


def test_to_dense_matches_reference(case):
    """``to_dense()`` is the explicit way to look at the whole matrix."""
    op, dense, _ = case
    torch.testing.assert_close(op.to_dense(dtype=DTYPE), dense)


def test_gradients_reach_parameters(case):
    """Gradients w.r.t. the parameters and the input equal the dense ones.

    The operators hold references to the tensors they were built from, so a
    ``nn.Parameter`` passed in is trained like any other.
    """
    op, dense, params = case
    x = _rand(3, op.shape[1], seed=27).requires_grad_()
    got = torch.autograd.grad(op.matvec(x).square().sum(), [x, *params])
    # ``dense`` is itself a function of the parameters (e.g. ``U @ V.T``) and
    # is differentiated twice below, hence ``retain_graph``.
    want = torch.autograd.grad(
        (x @ dense.T).square().sum(), [x, *params], retain_graph=True
    )
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w)

    # The transpose apply is differentiable too.
    y = _rand(3, op.shape[0], seed=28)
    got = torch.autograd.grad(op.rmatvec(y).square().sum(), params)
    want = torch.autograd.grad((y @ dense).square().sum(), params)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w)


def test_wrong_input_size_is_rejected(case):
    """A vector of the wrong length is a ``ValueError``, not a broadcast."""
    op, _, _ = case
    m, n = op.shape
    with pytest.raises(ValueError, match="matvec"):
        op.matvec(_rand(n + 1))
    with pytest.raises(ValueError, match="rmatvec"):
        op.rmatvec(_rand(m + 1))


# ---------------------------------------------------------------- structured
def test_constant_operator_never_stores_a_matrix():
    """An all-to-all operator of 10^6 x 10^6 applies in ``O(M + N)``.

    The explicit matrix would need 10^12 entries; the closed form is
    ``y_m = value * sum(x)`` for every ``m``.
    """
    n = 1_000_000
    A = ConstantOperator((n, n), 2.0)
    x = torch.ones(n)
    y = A @ x
    assert y.shape == (n,)
    assert float(y[0]) == 2.0 * n and float(y[-1]) == 2.0 * n


def test_constant_operator_python_number_and_parameter():
    """``value`` may be a Python number or a 0-dim (trainable) tensor."""
    x = _rand(N)
    assert isinstance(ConstantOperator((M, N), 2.0), StructuredOperator)
    torch.testing.assert_close(
        ConstantOperator((M, N), 2.0) @ x, (2.0 * x.sum()).expand(M)
    )
    weight = nn.Parameter(torch.tensor(0.5, dtype=DTYPE))
    (ConstantOperator((M, N), weight) @ x).sum().backward()
    # d/dvalue sum_m value * sum(x) = M * sum(x)
    torch.testing.assert_close(weight.grad, M * x.sum())
    with pytest.raises(ValueError, match="scalar"):
        ConstantOperator((M, N), torch.ones(M))


def test_structured_constructors_validate():
    with pytest.raises(ValueError, match="1-D"):
        DiagonalOperator(torch.ones(2, 2))
    with pytest.raises(ValueError, match="same rank"):
        LowRankOperator(torch.ones(M, 2), torch.ones(N, 3))
    with pytest.raises(ValueError, match="shape"):
        ConstantOperator((M, N, 2), 1.0)
    assert LowRankOperator(torch.ones(M, 2), torch.ones(N, 2)).rank == 2


# -------------------------------------------------------------- capabilities
def test_capabilities_of_structured_operators():
    """Structured operators with few entries can list them; a low-rank operator
    is dense, so it only applies."""
    assert all(DiagonalOperator(torch.ones(3)).supports(c) for c in CAPABILITIES)
    assert all(ConstantOperator((M, N), 1.0).supports(c) for c in CAPABILITIES)
    low_rank = LowRankOperator(torch.ones(M, 1), torch.ones(N, 1))
    assert low_rank.supports("matvec") and low_rank.supports("rmatvec")
    assert not low_rank.supports("enumerate_edges")
    assert not low_rank.supports("materialize_sparse")
    with pytest.raises(NotImplementedError, match="materialize_sparse"):
        low_rank.tocsr()
    # A misspelled capability is an error rather than a silent ``False``.
    with pytest.raises(ValueError, match="Unknown capability"):
        low_rank.supports("transpose_apply")


def test_structured_materialize_matches_dense():
    """``materialize()`` turns a structured operator into a real ``Sparse``.

    This is the only place a structured operator allocates per-entry
    memory, and it happens only on request. Gradients still reach the
    parameter.
    """
    value = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    A = ConstantOperator((M, N), value).materialize()
    assert isinstance(A, sparse.Sparse) and A.nnz == M * N
    torch.testing.assert_close(A.to_dense(), value.expand(M, N))
    A.to_dense().sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor(float(M * N), dtype=DTYPE))

    diag = _rand(N)
    D = DiagonalOperator(diag)
    assert D.tocsr().format == "csr" and D.tocoo().format == "coo"
    assert D.tocsc().format == "csc" and D.tocsr().nnz == N
    torch.testing.assert_close(D.tocsr().to_dense(), torch.diag(diag))
    rows, cols, values = D.enumerate_edges()
    assert torch.equal(rows, torch.arange(N)) and torch.equal(cols, rows)
    torch.testing.assert_close(values, diag)


# ------------------------------------------------------------------ implicit
def test_apply_only_implicit_operator_refuses_everything_else():
    """An operator given only ``matvec`` can apply and nothing more.

    "Procedural != sparse": a matrix-free operator has no entries to convert,
    so ``tocsr()`` (and the transpose, which was not provided) must fail with
    an error that names the missing capability instead of densifying.
    """
    # A cumulative sum is a lower-triangular all-ones matrix, never built.
    op = ImplicitOperator((N, N), matvec=lambda x: x.cumsum(-1))
    x = _rand(2, N)
    torch.testing.assert_close(op.matvec(x), x.cumsum(-1))
    torch.testing.assert_close(op @ x[0], x[0].cumsum(-1))
    assert op.supports("matvec") and op.supports("matmat")
    assert not op.supports("rmatvec")
    assert not op.supports("enumerate_edges")
    assert not op.supports("materialize_sparse")

    for call in (op.tocsr, op.tocoo, op.materialize):
        with pytest.raises(NotImplementedError, match="materialize_sparse"):
            call()
    with pytest.raises(NotImplementedError, match="enumerate_edges"):
        op.enumerate_edges()
    with pytest.raises(NotImplementedError, match="rmatvec"):
        op.rmatvec(x)
    with pytest.raises(NotImplementedError, match="rmatvec"):
        x @ op
    # The lazy transpose is refused when it is built, not on its first use.
    with pytest.raises(NotImplementedError, match="rmatvec"):
        op.T

    # ``to_dense`` only needs matvec (it applies the operator to the
    # identity) and remains available as an explicit request.
    torch.testing.assert_close(
        op.to_dense(dtype=DTYPE), torch.ones(N, N, dtype=DTYPE).tril()
    )


def test_enumerable_implicit_operator_materialises_correctly():
    """A procedural operator that can list its edges converts on request.

    The operator is a ring shift: ``y[m] = w[m] * x[(m - 1) % N]``. The
    product is implemented with ``roll`` (no indices at all); the edge
    enumeration is a second, independent description of the same matrix.
    """
    w = _rand(N, seed=30).requires_grad_()
    index = torch.arange(N)

    op = ImplicitOperator(
        (N, N),
        matvec=lambda x: w * x.roll(1, -1),
        rmatvec=lambda y: (w * y).roll(-1, -1),
        enumerate_edges=lambda: (index, (index - 1) % N, w),
        dtype=DTYPE,
    )
    dense = torch.zeros(N, N, dtype=DTYPE)
    dense[index, (index - 1) % N] = w.detach()

    assert all(op.supports(c) for c in CAPABILITIES)
    # The enumerated matrix, the matvec and the rmatvec all agree.
    csr = op.tocsr()
    assert csr.format == "csr" and csr.shape == (N, N) and csr.nnz == N
    torch.testing.assert_close(csr.to_dense(), dense)
    torch.testing.assert_close(op.materialize().to_dense(), dense)
    torch.testing.assert_close(op.to_dense(), dense)
    y = _rand(N)
    torch.testing.assert_close(y @ op, y @ dense)
    # The materialised matrix keeps the autograd link to ``w``.
    (grad,) = torch.autograd.grad(csr.matvec(torch.ones(N, dtype=DTYPE)).sum(), w)
    torch.testing.assert_close(grad, torch.ones(N, dtype=DTYPE))


def test_implicit_operator_custom_matmat():
    """An optional ``matmat`` callable replaces the column-wise default."""
    W = _rand(M, N)
    calls = []

    def matmat(X):
        calls.append(X.shape)
        return W @ X

    op = ImplicitOperator((M, N), lambda x: x @ W.T, matmat=matmat)
    X = _rand(N, 3)
    torch.testing.assert_close(op @ X, W @ X)
    assert calls == [X.shape]


# ----------------------------------------------------------------- composite
def test_operator_algebra_builds_lazy_composites():
    """``+``, ``-``, ``@``, scalar ``*`` / ``/`` and unary ``-`` are lazy."""
    C = ConstantOperator((M, N), 2.0, dtype=DTYPE)
    L = LowRankOperator(_rand(M, 2), _rand(N, 2, seed=1))
    D = DiagonalOperator(_rand(N, seed=2))
    c, low, d = C.to_dense(), L.to_dense(), D.to_dense()

    assert isinstance(C + L, SumOperator)
    assert isinstance(C @ D, ProductOperator)
    assert isinstance(2 * C, ScaledOperator)
    assert isinstance(C.T, TransposedOperator)
    for op in (C + L, C @ D, 2 * C, C.T):
        assert isinstance(op, CompositeOperator)

    expression = (C - 0.5 * L) @ D / 4 + (-L)
    reference = (c - 0.5 * low) @ d / 4 - low
    torch.testing.assert_close(expression.to_dense(dtype=DTYPE), reference)
    x = _rand(3, N)
    torch.testing.assert_close(expression.matvec(x), x @ reference.T)

    # Chains are flattened, so a long sum or product stays one level deep.
    assert len((C + L + C + L).parts) == 4
    assert len((C @ D @ D @ D).parts) == 4


def test_composite_shape_checks():
    """Sums need equal shapes; products need matching inner dimensions."""
    C = ConstantOperator((M, N), 1.0)
    with pytest.raises(ValueError, match="different shapes"):
        C + ConstantOperator((N, M), 1.0)
    with pytest.raises(ValueError, match="inner"):
        C @ ConstantOperator((M, N), 1.0)
    # (M, N) @ (N, M) is fine and has shape (M, M).
    assert (C @ C.T).shape == (M, M)
    # Only scalars scale an operator; a vector is not silently broadcast.
    with pytest.raises(TypeError):
        C * torch.ones(N)
    with pytest.raises(TypeError):
        C + 1.0


def test_composite_reports_promoted_dtype_and_rejects_mixed_devices():
    """Composite metadata matches tensor promotion and device constraints."""
    left = DiagonalOperator(torch.ones(N, dtype=torch.float32))
    right = DiagonalOperator(torch.ones(N, dtype=torch.float64))
    summed = left + right

    assert summed.dtype == torch.float64
    assert summed.device == torch.device("cpu")
    assert summed.matvec(torch.ones(N)).dtype == torch.float64

    meta = DiagonalOperator(torch.ones(N, device="meta"))
    with pytest.raises(ValueError, match="one device"):
        left + meta

    with pytest.raises(ValueError, match="one dtype"):
        left @ right

    complex_scaled = (1 + 2j) * left
    assert complex_scaled.dtype == torch.complex64
    assert complex_scaled.matvec(torch.ones(N)).dtype == torch.complex64


def test_scalar_matmul_reports_invalid_tensor_rank():
    """A zero-dimensional tensor gives the operator's shape error."""
    operator = ConstantOperator((M, N), 1.0)
    scalar = torch.tensor(1.0)

    with pytest.raises(ValueError, match="at least one dimension"):
        operator @ scalar
    with pytest.raises(ValueError, match="at least one dimension"):
        scalar @ operator


def test_sparse_is_accepted_wherever_an_operator_is():
    """An explicit ``Sparse`` combines with operators on either side.

    ``Sparse`` does not inherit from ``LinearOperator``; it is recognised by
    its ``shape`` / ``matvec`` / ``rmatvec``. The reflected methods of the
    operator handle ``sparse @ op`` and ``sparse + op``.
    """
    A, dense = _sparse_matrix()
    D = DiagonalOperator(_rand(N, seed=2))
    C = ConstantOperator((M, N), 0.5, dtype=DTYPE)
    d, c = D.to_dense(), C.to_dense()
    assert is_linear_operator(A)
    assert not is_linear_operator(dense)  # dense tensors are data

    cases = {
        "sparse @ op": (A @ D, dense @ d),
        "op @ sparse": (C.T @ A, c.T @ dense),
        "sparse + op": (A + C, dense + c),
        "op + sparse": (C + A, c + dense),
        "sparse - op": (A - C, dense - c),
        "explicit": (SumOperator([A, A]), 2 * dense),
        "scaled": (ScaledOperator(A, 3.0), 3 * dense),
        "transposed": (TransposedOperator(A), dense.T),
    }
    for name, (op, reference) in cases.items():
        assert isinstance(op, CompositeOperator), name
        torch.testing.assert_close(op.to_dense(dtype=DTYPE), reference)
        y = _rand(2, op.shape[0], seed=5)
        torch.testing.assert_close(op.rmatvec(y), y @ reference)

    # Batched sparse arrays are not plain matrices and are refused.
    with pytest.raises(ValueError, match="plain matrices"):
        C + sparse.stack([A, A])


def test_composite_capabilities_follow_the_parts():
    """A composite can do what all of its parts can do.

    - A sum of enumerable parts materialises by concatenating their entries
      (overlapping coordinates are stored twice and summed, as in COO).
    - A sum with an apply-only part cannot materialise.
    - A product never materialises (that would be a sparse-sparse product).
    """
    A, dense = _sparse_matrix()
    C = ConstantOperator((M, N), 0.5, dtype=DTYPE)
    L = LowRankOperator(_rand(M, 2), _rand(N, 2, seed=1))
    apply_only = ImplicitOperator((M, N), lambda x: x @ dense.T)

    total = A + 2 * C - A.T.T
    assert total.supports("materialize_sparse") and total.supports("enumerate_edges")
    S = total.materialize()
    assert isinstance(S, sparse.Sparse)
    # Entries are concatenated, not merged: nnz(A) + M*N + nnz(A).
    assert S.nnz == 2 * A.nnz + M * N
    torch.testing.assert_close(S.to_dense(), 2 * C.to_dense())
    torch.testing.assert_close(total.tocsr().to_dense(), 2 * C.to_dense())

    # The transpose of an enumerable operator is enumerable.
    torch.testing.assert_close((A + C).T.tocsr().to_dense(), (dense + 0.5).T)

    assert not (A + L).supports("materialize_sparse")
    with pytest.raises(NotImplementedError, match="materialize_sparse"):
        (A + L).tocsr()

    product = C.T @ A
    assert product.supports("matvec") and product.supports("rmatvec")
    assert not product.supports("materialize_sparse")
    with pytest.raises(NotImplementedError, match="materialize_sparse"):
        product.tocsr()

    # One part without a transpose apply removes it from the whole sum.
    assert (A + apply_only).supports("matvec")
    assert not (A + apply_only).supports("rmatvec")
    with pytest.raises(NotImplementedError, match="rmatvec"):
        (A + apply_only).rmatvec(_rand(M))


# ------------------------------------------------------------- torch.compile
@pytest.fixture
def fresh_dynamo():
    """Compile from a clean slate so tests do not share guard caches."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class _OperatorLayer(nn.Module):
    """A layer whose weight matrix is ``gain * 1 1^T + ring shift``.

    This is the intended usage pattern: operators are plain objects (like
    ``Sparse``), so the *module* owns the parameters and hands them to the
    operators. The structured part is an all-to-all coupling with one
    trainable gain; the implicit part is a matrix-free ring shift scaled by a
    trainable per-neuron weight.

    Inside compiled code call ``matvec`` (or ``op.__matmul__``): TorchDynamo
    does not dispatch the binary ``@`` to user-defined objects.
    """

    def __init__(self, n: int):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.1))
        self.ring = nn.Parameter(torch.linspace(0.5, 1.5, n))
        self.structured = ConstantOperator((n, n), self.gain)
        self.implicit = ImplicitOperator(
            (n, n),
            matvec=self._shift,
            rmatvec=self._shift_transposed,
        )
        self.total = self.structured + self.implicit

    def _shift(self, x):
        return self.ring * x.roll(1, -1)

    def _shift_transposed(self, y):
        return (self.ring * y).roll(-1, -1)

    def forward(self, x):
        return self.total.matvec(x) + self.total.rmatvec(x)


def test_compile_module_with_structured_and_implicit_operator(fresh_dynamo):
    """``torch.compile(model, fullgraph=True)`` traces through the operators.

    The compiled forward and its gradients must equal eager mode, and the
    whole forward must be one graph (``fullgraph=True`` raises on a break).
    """
    n = 7
    model = _OperatorLayer(n)
    x = torch.randn(3, n, generator=torch.manual_seed(0))

    # Independent dense reference of the same matrix.
    index = torch.arange(n)
    shift = torch.zeros(n, n)
    shift[index, (index - 1) % n] = model.ring.detach()
    W = model.gain.detach() + shift
    expected = x @ W.T + x @ W

    eager = model(x)
    torch.testing.assert_close(eager, expected)
    eager_grads = torch.autograd.grad(eager.square().sum(), list(model.parameters()))

    compiled = torch.compile(model, fullgraph=True)
    for _ in range(2):  # the second call reuses the compiled graph
        out = compiled(x)
        torch.testing.assert_close(out, expected)
        grads = torch.autograd.grad(out.square().sum(), list(model.parameters()))
        for g, e in zip(grads, eager_grads):
            torch.testing.assert_close(g, e)

    # The compiled graph reads the live parameters: an optimiser step is seen.
    with torch.no_grad():
        model.gain.add_(1.0)
    torch.testing.assert_close(compiled(x), expected + 2.0 * x.sum(-1, keepdim=True))
