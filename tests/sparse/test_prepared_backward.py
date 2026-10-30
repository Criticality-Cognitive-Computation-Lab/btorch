"""Check invocation-owned sparse reverse preparation against dense
mathematics."""

import gc
import weakref

import pytest
import torch

from btorch.sparse.runtime import RepresentationCache, kernels_cusparse


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _pattern():
    # A duplicate slot and an empty output row rule out a dense reconstruction
    # that loses stored-edge identity or misinterprets source-major permutation.
    cache = RepresentationCache()
    cache.build(
        torch.tensor([0, 0, 0, 2, 3, 3]), torch.tensor([1, 1, 4, 0, 2, 4]), (4, 5), 0
    )
    return cache.cuda()


def test_cusparse_prepared_reverse_rebinds_inputs_and_keeps_edge_vjps():
    """Mixed/broadcast weights, sliced storage and masks preserve full VJPs."""
    cache = _pattern()
    values = torch.linspace(0.1, 0.7, 12, device="cuda").reshape(2, 6).requires_grad_()
    x = (
        torch.linspace(-0.4, 0.8, 60, device="cuda", dtype=torch.float64)
        .reshape(1, 3, 20)[..., 1::4]
        .requires_grad_()
    )
    grad = torch.linspace(-0.5, 0.8, 48, device="cuda", dtype=torch.float64).reshape(
        2, 3, 8
    )[..., ::2]
    row = torch.repeat_interleave(torch.arange(4, device="cuda"), cache.crow.diff())
    output = x.new_zeros(2, 3, 4).index_add(
        -1, row, x.index_select(-1, cache.col) * values[:, None, :]
    )
    reference = torch.autograd.grad(output, (values, x), grad)
    with torch.no_grad():
        prepared = kernels_cusparse.prepare_backward(
            cache.crow,
            cache.col,
            cache.t_crow,
            cache.t_col,
            cache.t_perm,
            values,
            x,
            grad,
        )
        # Current values stay fixed within an invocation; each source and
        # cotangent replacement must read its own storage, not the warm inputs.
        for scale in (1.0, -0.3):
            source = torch.empty_strided(
                x.shape, x.stride(), device=x.device, dtype=x.dtype
            )
            source.copy_(x * scale)
            cotangent = torch.empty_strided(
                grad.shape, grad.stride(), device=grad.device, dtype=grad.dtype
            )
            cotangent.copy_(grad * scale)
            actual_values, actual_x = prepared(source, cotangent)
            torch.testing.assert_close(
                actual_values.to(values.dtype), reference[0] * scale**2
            )
            torch.testing.assert_close(
                actual_x.sum_to_size(x.shape), reference[1] * scale
            )
        mask = torch.tensor([True, False, True, False, True], device="cuda").expand_as(
            actual_x
        )
        _, masked = prepared(x, grad, mask)
        torch.testing.assert_close(masked.sum_to_size(x.shape), reference[1] * mask[:1])
    # Only invocation-local preparation may retain input/weight owners.
    refs = weakref.ref(values), weakref.ref(x), weakref.ref(grad)
    del prepared, output, reference, values, x, grad, source, cotangent
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_cusparse_prepared_reverse_accumulates_and_replays_graph():
    """Graph replay reuses destinations and adds each weight VJP exactly
    once."""
    cache = _pattern()
    values = torch.linspace(0.1, 0.6, 6, device="cuda")
    x = torch.linspace(-0.4, 0.8, 10, device="cuda").reshape(2, 5)
    gradient = torch.linspace(-0.5, 0.8, 8, device="cuda").reshape(2, 4)
    accumulated = torch.zeros_like(values)
    row = torch.repeat_interleave(torch.arange(4, device="cuda"), cache.crow.diff())
    prepared = kernels_cusparse.prepare_backward(
        cache.crow,
        cache.col,
        cache.t_crow,
        cache.t_col,
        cache.t_perm,
        values,
        x,
        gradient,
        grad_values=accumulated,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared(x, gradient)
    torch.cuda.current_stream().wait_stream(stream)
    accumulated.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # A complete training graph prepares reverse inside capture, just as
        # the recurrent operator does. Its transposed-value refresh must replay
        # after optimizer updates; a snapshot prepared outside would be stale.
        captured = kernels_cusparse.prepare_backward(
            cache.crow,
            cache.col,
            cache.t_crow,
            cache.t_col,
            cache.t_perm,
            values,
            x,
            gradient,
            grad_values=accumulated,
        )
        weight_gradient, input_gradient = captured(x, gradient)
    assert weight_gradient is None
    for scale in (0.5, 1.5):
        accumulated.zero_()
        values.add_(0.07)
        x.fill_(scale)
        graph.replay()
        expected_weight = (gradient.index_select(-1, row) * scale).sum(0)
        expected_input = torch.zeros_like(x).index_add(
            -1, cache.col, gradient.index_select(-1, row) * values
        )
        torch.testing.assert_close(accumulated, expected_weight)
        torch.testing.assert_close(input_gradient, expected_input)


@pytest.mark.parametrize("shape,samples", [((0, 5), 3), ((4, 5), 0), ((4, 0), 2)])
def test_cusparse_prepared_empty_products(shape, samples):
    """Empty populations, sample batches and source populations keep exact
    zeros."""
    cache = RepresentationCache()
    cache.build(
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), shape, 0
    )
    cache = cache.cuda()
    values = torch.empty(0, device="cuda")
    x = torch.empty(samples, shape[1], device="cuda")
    gradient = torch.empty(samples, shape[0], device="cuda")
    prepared = kernels_cusparse.prepare_backward(
        cache.crow,
        cache.col,
        cache.t_crow,
        cache.t_col,
        cache.t_perm,
        values,
        x,
        gradient,
    )
    grad_values, grad_x = prepared(x, gradient)
    assert grad_values.shape == values.shape and grad_x.shape == x.shape
    torch.testing.assert_close(grad_x, torch.zeros_like(x))


def test_cusparse_prepared_amp_preserves_promoted_outputs_and_vjps():
    """Autocast must not demote prepared CSR products or their explicit
    VJPs."""
    cache = _pattern()
    values = torch.linspace(0.1, 0.6, 6, device="cuda", requires_grad=True)
    x = (
        torch.linspace(-0.4, 0.8, 10, device="cuda", dtype=torch.float16)
        .reshape(2, 5)
        .requires_grad_()
    )
    gradient = torch.linspace(-0.5, 0.8, 8, device="cuda").reshape(2, 4)
    row = torch.repeat_interleave(torch.arange(4, device="cuda"), cache.crow.diff())
    expected = torch.zeros(2, 4, device="cuda").index_add(
        -1, row, x.float().index_select(-1, cache.col) * values
    )
    trusted = torch.autograd.grad(expected, (values, x), gradient)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        forward = kernels_cusparse.prepare_matvec(cache.crow, cache.col, values, x)
        actual = forward(x)
        reverse = kernels_cusparse.prepare_backward(
            cache.crow,
            cache.col,
            cache.t_crow,
            cache.t_col,
            cache.t_perm,
            values,
            x,
            gradient,
        )
        grad_values, grad_x = reverse(x, gradient)
    assert actual.dtype == grad_values.dtype == grad_x.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(grad_values, trusted[0])
    torch.testing.assert_close(grad_x.to(x.dtype), trusted[1])


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_sparse_reverse_recipes_release_tapes_between_calls(backend):
    """Cached executables must rebind operands and own no private VJP
    storage."""
    from btorch.sparse.runtime import kernels_tilelang, kernels_triton

    kernels = {"triton": kernels_triton, "tilelang": kernels_tilelang}[backend]
    if not kernels.is_available():
        pytest.skip(f"{backend} is unavailable")
    cache = _pattern()
    row = torch.repeat_interleave(torch.arange(4, device="cuda"), cache.crow.diff())

    def run(offset):
        values = (
            torch.linspace(0.1, 0.6, 6, device="cuda", dtype=torch.float64) + offset
        )
        x = torch.linspace(-0.4, 0.8, 15, device="cuda", dtype=torch.float64).reshape(
            3, 5
        )
        gradient = torch.linspace(
            -0.5, 0.8, 12, device="cuda", dtype=torch.float64
        ).reshape(3, 4)
        result = kernels.csr_backward(
            cache.crow,
            cache.col,
            cache.t_crow,
            cache.t_col,
            cache.t_perm,
            values,
            x,
            gradient,
        )
        expected_weight = (
            gradient.index_select(-1, row) * x.index_select(-1, cache.col)
        ).sum(0)
        expected_input = torch.zeros_like(x).index_add(
            -1, cache.col, gradient.index_select(-1, row) * values
        )
        torch.testing.assert_close(result[0], expected_weight, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(result[1], expected_input, atol=1e-11, rtol=1e-11)
        return tuple(weakref.ref(value) for value in (values, x, gradient, *result))

    # Same static layout, genuinely different operands. Keeping compiled stage
    # metadata alive must not keep either invocation's inputs/outputs alive.
    references = [run(0.0), run(0.2)]
    gc.collect()
    assert all(reference() is None for group in references for reference in group)
