"""Regression tests for defects found when reviewing the Triton kernels.

Each test reproduces one confirmed finding through the public entry point
(:func:`btorch.sparse.runtime.ops.propagate`) and compares against the ATen
reference backend, which defines the semantics.
"""

import pytest
import torch

from btorch.sparse import runtime
from btorch.sparse.runtime import RepresentationCache, ops


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "triton" not in runtime.registry.available("csr_matvec", "cuda"),
    reason="needs CUDA and the Triton backend",
)


def _buffers(row, col, shape):
    """Derived layouts of an edge list, on the GPU."""
    cache = RepresentationCache()
    cache.build(torch.tensor(row), torch.tensor(col), shape, 0)
    return cache.cuda()


def _propagate(cache, values, x):
    return ops.propagate(
        cache.crow, cache.col, values, x, cache.t_crow, cache.t_col, cache.t_perm
    )


def test_backward_with_no_destinations_and_several_samples():
    """A connection onto an empty population (``M == 0``) must still run a
    backward pass for a batch of samples.

    The fused backward transposed the output gradient to a sample-minor
    layout before checking for zero-size shapes, and that reshape is
    ambiguous for zero elements.
    """
    n_in = 6
    cache = RepresentationCache()
    cache.build(
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), (0, n_in), 0
    )
    cache = cache.cuda()
    values = torch.empty(0, device="cuda", requires_grad=True)
    x = torch.rand(4, n_in, device="cuda", requires_grad=True)

    out = _propagate(cache, values, x)
    assert out.shape == (4, 0)
    out.sum().backward()
    # Nothing is connected, so nothing flows back.
    assert x.grad.shape == x.shape and not x.grad.any()
    assert values.grad.shape == (0,)


def test_replacing_index_data_is_noticed():
    """``tensor.data = other`` swaps the contents of an index buffer without
    changing the tensor object or its version counter.

    The kernels cache derived copies per tensor object; the cache must
    not keep serving the layout of the old contents.
    """
    row, col = [0, 0, 1, 2], [0, 1, 2, 3]
    cache = _buffers(row, col, (3, 4))
    other = _buffers([0, 1, 1, 2], [3, 0, 1, 2], (3, 4))
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    x = torch.rand(5, 4, device="cuda")

    _propagate(cache, values, x)  # fills the kernel's cache for these objects
    for name in ("crow", "col", "perm", "t_crow", "t_col", "t_perm"):
        getattr(cache, name).data = getattr(other, name).data.clone()

    out = _propagate(cache, values, x)
    with runtime.use_backend("aten"):
        expected = _propagate(cache, values, x)
    torch.testing.assert_close(out, expected)


def test_eager_route_is_never_traced():
    """If a compiled frame is skipped and runs in the interpreter, Dynamo must
    not begin compiling the kernel wrappers of the eager route: they contain
    data-dependent cache checks that cannot be traced."""
    assert torch._dynamo.eval_frame.innermost_fn(ops._propagate_eager) is not None
    cache = _buffers([0, 1, 2], [1, 2, 0], (3, 3))
    values = torch.tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)
    x = torch.rand(2, 3, device="cuda")

    def direct(x):
        # Calls the eager helper explicitly from inside a compiled function.
        return ops._propagate_eager(
            cache.crow,
            cache.col,
            values,
            x,
            cache.t_crow,
            cache.t_col,
            cache.t_perm,
            None,
        )

    compiled = torch.compile(direct)
    torch.testing.assert_close(compiled(x), direct(x))
