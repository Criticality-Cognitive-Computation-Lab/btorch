"""Autograd and ``torch.compile`` behaviour of sparse products.

Gradients are checked numerically (``gradcheck`` in float64) with respect to
the stored values and the dense input. Index tensors are integers and never
receive gradients.
"""

import pytest
import torch
from torch.autograd import gradcheck

from btorch import sparse

from .helpers import FORMATS, SHAPE, build, random_edges


M, N = SHAPE
G, B = 2, 3


def _randn(*shape, seed=0):
    return torch.randn(*shape, dtype=torch.float64, generator=torch.manual_seed(seed))


def _leaf(*shape, seed=0):
    return _randn(*shape, seed=seed).requires_grad_()


def _pattern(fmt, variant="plain"):
    """Float64 array in ``fmt``; only its pattern is used by the tests."""
    rows, cols, values, _ = random_edges(variant)
    return build(fmt, rows, cols, values)


# ------------------------------------------------------------- gradcheck
@pytest.mark.parametrize("variant", ["plain", "duplicates"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_gradcheck_values_and_input(fmt, variant):
    """Unbatched: d/d(values) and d/dx of every product spelling."""
    A = _pattern(fmt, variant)
    values = _leaf(A.nnz)

    # `with_values` swaps in new values on the same pattern, which is how a
    # trainable weight vector is attached to a fixed topology.
    assert gradcheck(lambda v, x: A.with_values(v) @ x, (values, _leaf(N)))
    assert gradcheck(lambda v, x: A.with_values(v) @ x, (values, _leaf(N, 2)))
    assert gradcheck(lambda v, x: x @ A.with_values(v), (values, _leaf(2, M)))
    assert gradcheck(lambda v, x: A.with_values(v).matvec(x), (values, _leaf(B, N)))
    assert gradcheck(lambda v, x: A.with_values(v).rmatvec(x), (values, _leaf(B, M)))


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_gradcheck_shared_pattern_batch(fmt):
    """``values [G, E]`` on one pattern, input ``[G, B, N]``."""
    A = _pattern(fmt)
    values, x = _leaf(G, A.nnz), _leaf(G, B, N)

    def product(v, x):
        batched = A.with_values(v)
        assert batched.shape == (G, M, N)
        return batched.matvec(x)

    assert gradcheck(product, (values, x))
    # A broadcast sample batch [1, B, N] sums the gradient over networks.
    assert gradcheck(product, (values, _leaf(1, B, N)))
    assert gradcheck(lambda v, x: A.with_values(v) @ x, (values, _leaf(N, 2)))


def test_gradcheck_stacked_different_patterns():
    """Stacked members with different nnz: one flat value vector."""
    members = [_pattern("coo", "duplicates"), _pattern("csr", "empty_rows")]
    A = sparse.stack(members)
    assert members[0].nnz != members[1].nnz and A.batch_shape == (2,)
    values = _leaf(A.nnz)
    assert gradcheck(lambda v, x: A.with_values(v).matvec(x), (values, _leaf(2, B, N)))
    assert gradcheck(lambda v, x: A.with_values(v).rmatvec(x), (values, _leaf(1, B, M)))
    assert gradcheck(lambda v, x: A.with_values(v) @ x, (values, _leaf(N, 2)))


@pytest.mark.parametrize("fmt", FORMATS)
def test_gradcheck_dense_entry_dimension(fmt):
    """Vector-valued entries ``values [E, D]`` with ``matvec``."""
    rows, cols, _, _ = random_edges("duplicates")
    coo = sparse.from_edges(rows, cols, _randn(len(rows), 2), (M, N, 2), dense_dim=1)
    A = getattr(coo, f"to{fmt}")()
    values = _leaf(A.nnz, 2)
    assert gradcheck(lambda v, x: A.with_values(v).matvec(x), (values, _leaf(B, N)))


@pytest.mark.parametrize("method", ["coalesce", "tocsr", "tocsc"])
def test_gradcheck_through_canonicalisation(method):
    """Merging duplicates is differentiable w.r.t.

    the original values.
    """
    A = _pattern("coo", "duplicates")
    values, x = _leaf(A.nnz), _leaf(N)
    assert gradcheck(lambda v, x: getattr(A.with_values(v), method)() @ x, (values, x))
    assert gradcheck(lambda v: getattr(A.with_values(v), method)().to_dense(), values)


# ------------------------------------------------------ explicit gradients
@pytest.mark.parametrize("canonicalise", [None, "coalesce", "tocsr", "tocsc"])
def test_duplicates_each_receive_the_gradient(canonicalise):
    """Both copies of a duplicated entry get the gradient of the merged one.

    ``y = A @ x`` with ``A[1, 2] = v0 + v2`` gives ``dy_1/dv0 = dy_1/dv2 =
    x[2]``, whether the duplicates are merged before the product or not.
    """
    rows, cols = torch.tensor([1, 0, 1]), torch.tensor([2, 0, 2])
    values = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    A = sparse.from_edges(rows, cols, values, (2, 3))
    if canonicalise is not None:
        A = getattr(A, canonicalise)()
        assert A.nnz == 2 and A.data.sum() == 7.0
    x = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
    y = A @ x
    assert y.tolist() == [20.0, 150.0]
    # Weight the outputs so that a row mix-up would change the result.
    (y * torch.tensor([1.0, 100.0])).sum().backward()
    assert values.grad.tolist() == [3000.0, 10.0, 3000.0]
    assert x.grad.tolist() == [2.0, 0.0, 500.0]


def test_gradients_match_the_dense_reference():
    """Gradients of a shared-pattern batch equal those of dense einsum."""
    A = _pattern("csr")
    coo = A.tocoo()
    values, x = _leaf(G, A.nnz), _leaf(G, B, N)
    weight = _randn(G, B, M, seed=5)

    (A.with_values(values).matvec(x) * weight).sum().backward()

    dense = torch.zeros(G, M, N, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        dense[:, coo.row, coo.col] = values
    x_ref = x.detach().clone().requires_grad_()
    (torch.einsum("gmn,gbn->gbm", dense, x_ref) * weight).sum().backward()
    torch.testing.assert_close(values.grad, dense.grad[:, coo.row, coo.col])
    torch.testing.assert_close(x.grad, x_ref.grad)


# ------------------------------------------------------------ torch.compile
@pytest.fixture
def fresh_dynamo():
    """Compile from a clean slate so tests do not share guard caches."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _compile_case(fmt):
    """Array with trainable values, an input and an eager reference."""
    A = _pattern(fmt, "duplicates").float()
    A = A.with_values(A.values().detach().requires_grad_())
    x = torch.randn(N, generator=torch.manual_seed(0), requires_grad=True)
    return A, x


def _check_compiled(fn, A, x):
    """Forward and backward of ``compile(fn, fullgraph=True)`` match eager."""
    values = A.values()
    expected = fn(x)
    grad_x, grad_v = torch.autograd.grad(expected.square().sum(), (x, values))

    compiled = torch.compile(fn, fullgraph=True)
    for _ in range(2):  # the second call reuses the compiled graph
        out = compiled(x)
        torch.testing.assert_close(out, expected)
        got_x, got_v = torch.autograd.grad(out.square().sum(), (x, values))
        torch.testing.assert_close(got_x, grad_x)
        torch.testing.assert_close(got_v, grad_v)


SPELLINGS = {
    "matvec": lambda A: lambda x: A.matvec(x),
    "sparse.matmul": lambda A: lambda x: sparse.matmul(A, x),
    "__matmul__": lambda A: lambda x: A.__matmul__(x),
}


@pytest.mark.parametrize("spelling", list(SPELLINGS))
@pytest.mark.parametrize("fmt", ["coo", "csr", "csc"])
def test_compile_closure_over_sparse(fmt, spelling, fresh_dynamo):
    """A ``Sparse`` captured by a compiled closure traces as one graph.

    The array is a plain Python object, so Dynamo reads its tensors as
    constants of the closure; no graph break is allowed (``fullgraph``).
    """
    A, x = _compile_case(fmt)
    _check_compiled(SPELLINGS[spelling](A), A, x)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_compile_before_any_eager_call(fmt, fresh_dynamo):
    """The very first use of a CSR/CSC array may be the compiled one.

    CSR/CSC expand their pointer array into per-entry coordinates lazily
    on first use; that expansion then happens while tracing.
    """
    A, x = _compile_case(fmt)
    compiled = torch.compile(lambda x: A.matvec(x), fullgraph=True)
    out = compiled(x)  # no eager product has run on `A` before this line
    grad = torch.autograd.grad(out.square().sum(), (x, A.values()))
    expected = A.matvec(x)
    torch.testing.assert_close(out, expected)
    reference = torch.autograd.grad(expected.square().sum(), (x, A.values()))
    torch.testing.assert_close(grad[0], reference[0])
    torch.testing.assert_close(grad[1], reference[1])
    torch.testing.assert_close(compiled(x), expected)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_compile_before_any_eager_call_does_not_recompile(fmt, fresh_dynamo):
    """A cold CSR/CSC should need one compilation, not two."""
    A, x = _compile_case(fmt)
    compiled = torch.compile(lambda x: A.matvec(x), fullgraph=True)
    with torch._dynamo.config.patch(error_on_recompile=True):
        compiled(x)
        compiled(x)


@pytest.mark.xfail(
    strict=True,
    raises=torch._dynamo.exc.Unsupported,
    reason="TorchDynamo (torch 2.11) does not dispatch the binary `@` "
    "operator to a user-defined object's __matmul__/__rmatmul__: it lowers "
    "`A @ x` to operator.matmul(UserDefinedObjectVariable, TensorVariable) "
    "and fails with 'Failed to convert args/kwargs to proxy'. A.matvec(x), "
    "sparse.matmul(A, x) and A.__matmul__(x) compile fine.",
)
@pytest.mark.parametrize("side", ["A @ x", "x @ A"])
@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_compile_closure_with_matmul_operator(fmt, side, fresh_dynamo):
    """``torch.compile(lambda x: A @ x, fullgraph=True)``, the PLAN.md section
    37 experiment ("desirable but secondary")."""
    A, x = _compile_case(fmt)
    if side == "A @ x":
        _check_compiled(lambda x: A @ x, A, x)
    else:
        x = torch.randn(M, generator=torch.manual_seed(0), requires_grad=True)
        _check_compiled(lambda x: x @ A, A, x)
