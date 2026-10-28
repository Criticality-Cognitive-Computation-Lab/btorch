"""Qualify TileLang's sparse stages against independent Torch mathematics.

Cases target different mechanisms rather than a backend/dtype/shape
product: split hubs, permuted edges, block-local spike prefixes, mixed
precision, noncontiguous operands and prepared graph replay/gradient
accumulation.
"""

import weakref

import pytest
import torch

from btorch.sparse import Hints
from btorch.sparse.runtime import (
    KernelCache,
    csr_execution,
    kernels_tilelang as tile,
    kernels_torch,
    kernels_triton,
    kernels_triton_spikelist,
    use_backend,
)
from btorch.sparse.runtime.activity import BLOCK_SIZE
from btorch.sparse.runtime.cache import RepresentationCache
from btorch.sparse.runtime.ops import csr_propagate
from btorch.sparse.runtime.planner import Planner


pytestmark = pytest.mark.skipif(not tile.is_available(), reason="needs TileLang/CUDA")


def _case(n_out=17, n_in=259, hub=True):
    # Include duplicate edges, an empty destination row, and a >512-edge row.
    # Source order is deliberately different from destination CSR order.
    gen = torch.Generator(device="cuda").manual_seed(31)
    row = torch.randint(1, n_out, (700 if hub else 71,), device="cuda", generator=gen)
    if hub:
        row[:600] = 1
    col = torch.randint(n_in, row.shape, device="cuda", generator=gen)
    cache = RepresentationCache()
    cache.build(row, col, (n_out, n_in), 0)
    return cache


def _buffers(c):
    return c.crow, c.col, c.t_crow, c.t_col, c.t_perm


@pytest.mark.parametrize("dtype,samples", [(torch.float64, 1), (torch.float32, 5)])
def test_segmented_pull_and_full_vjp(dtype, samples):
    c = _case()
    gen = torch.Generator(device="cuda").manual_seed(4)
    # Slices exercise real storage offsets and strides, not just a transposed
    # contiguous copy. Every gradient is compared to independent scatter math.
    values = torch.randn(
        c.col.numel() * 2 + 1, device="cuda", dtype=dtype, generator=gen
    )[1::2].requires_grad_()
    x = torch.randn(samples, 259 * 2, device="cuda", dtype=dtype, generator=gen)[
        :, ::2
    ].requires_grad_()
    with use_backend("tilelang"):
        output = csr_propagate(c.crow, c.col, values, x, c.t_crow, c.t_col, c.t_perm)
    gradient = torch.randn_like(output)
    actual = torch.autograd.grad(output, (values, x), gradient)
    row = torch.repeat_interleave(
        torch.arange(17, device="cuda"), c.crow[1:] - c.crow[:-1]
    )
    # Torch sparse-CSR backward coalesces duplicate slots on this version;
    # explicit dense scatter retains one differentiable weight per edge.
    expected = x.new_zeros((samples, 17)).index_add(
        1, row, x.index_select(1, c.col) * values
    )
    reference = torch.autograd.grad(expected, (values, x), gradient)
    atol, rtol = (1e-11, 1e-11) if dtype == torch.float64 else (2e-5, 2e-5)
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    if dtype == torch.float64:
        source = x.contiguous()
        bound = tile._prepare_matvec(KernelCache(), c.crow, c.col, values, source)
        bound(source)
        with pytest.raises(ValueError, match="layout/dtype changed"):
            bound(source.float())
        # A prepared weight VJP fixes both operand layouts. Replacements with
        # identical layouts remain valid; changing either stride must require
        # preparation, otherwise the old single-sample addressing reads the
        # wrong values. Exercise the shared contract through both renderers.
        grad = gradient.contiguous()
        strided_grad = torch.empty(
            *grad.shape[:-1], grad.shape[-1] * 2, device=grad.device, dtype=grad.dtype
        )[..., ::2]
        strided_grad.copy_(grad)
        for execution in (tile._Execution(), kernels_triton._Execution(KernelCache())):
            bound_grad = csr_execution._prepare_edge_grad(
                execution, KernelCache(), c.crow, c.col, grad, source, 0
            )
            torch.testing.assert_close(
                bound_grad(grad.clone(), source.clone()),
                reference[0],
                atol=atol,
                rtol=rtol,
            )
            for g, operand in ((grad, x), (strided_grad, source)):
                with pytest.raises(ValueError, match="layout/dtype changed"):
                    bound_grad(g, operand)


def test_local_spike_lists_and_persistent_fanout_match_triton():
    c = _case(n_out=23, n_in=259)
    values = torch.randn(c.col.numel() * 2 + 1, device="cuda", dtype=torch.float64)[
        1::2
    ]
    x = torch.zeros(3, 519, device="cuda", dtype=torch.float64)[:, 1::2]
    # Silent producer blocks, zero-degree active sources, a batch boundary in
    # the middle of a producer block, then all-active and all-silent reuse.
    x[:, (0, 128, 258)] = torch.tensor([1.0, -2.0, 0.5], device="cuda", dtype=x.dtype)
    args = (c.t_crow, c.t_col, c.t_perm, values, x, 23)
    a = tile.prepare_spike_list(*args, cache=KernelCache())
    b = kernels_triton_spikelist.prepare_spike_list(*args, cache=KernelCache())
    for mode in ("sparse", "dense", "silent"):
        if mode == "dense":
            x.fill_(1)
        elif mode == "silent":
            x.zero_()
        actual = a(x)
        expected = kernels_torch.csr_matvec(c.crow, c.col, values, x)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(actual, b(x), atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(a.spikes.counts, b.spikes.counts)
        for group, count in enumerate(a.spikes.counts.cpu().tolist()):
            sl = slice(group * BLOCK_SIZE, group * BLOCK_SIZE + count)
            torch.testing.assert_close(a.spikes.indices[sl], b.spikes.indices[sl])
            torch.testing.assert_close(a.spikes.cumulative[sl], b.spikes.cumulative[sl])
    # A replaceable input may rebind storage while preserving its layout.
    # Both renderers must refresh the address, even when the Tensor object is
    # unchanged; cached flat TileLang views must not keep reading old spikes.
    replacement = torch.empty_strided(
        x.shape, x.stride(), device=x.device, dtype=x.dtype
    )
    replacement.fill_(1)
    x.set_(replacement)
    for execute in (a, b):
        torch.testing.assert_close(
            execute(x),
            kernels_torch.csr_matvec(c.crow, c.col, values, x),
            atol=1e-11,
            rtol=1e-11,
        )
    # Dropping an invocation must release its queue/output immediately; a
    # launch closure retaining its own execution creates a per-chunk cycle
    # that keeps GPU workspaces alive until Python's cyclic collector runs.
    released = weakref.ref(a.output)
    del actual, a, execute
    assert released() is None
    # Check the shared input guard after the lifetime check: an exception's
    # traceback deliberately retains its frame until Python collects it.
    with pytest.raises(ValueError, match="layout/dtype changed"):
        b(x.contiguous())


def test_prepared_mixed_precision_graph_replay_uses_no_triton(monkeypatch):
    c = _case(hub=False)
    values = torch.randn(c.col.numel(), device="cuda", dtype=torch.float16)
    x = torch.randn(2, 259 * 2, device="cuda", dtype=torch.float32)[:, ::2]
    with use_backend("tilelang"):
        for method in ("pull", "push"):
            plan = Planner().plan(
                device="cuda",
                value_batched=False,
                hints=Hints(firing_rate_hz=1),
                dtype=values.dtype,
                method=method,
                shape=(17, 259),
            )
            assert plan.backend == "tilelang"

    # Prepared kernels must remain TileLang even though Triton is installed.
    # Fail the actual launcher, not merely inspect a route's name.
    def forbidden(*args, **kwargs):
        raise AssertionError("TileLang route attempted a Triton launch")

    monkeypatch.setattr(kernels_triton, "_launch", forbidden)
    pull = tile._prepare_matvec(KernelCache(), c.crow, c.col, values, x)
    push = tile.prepare_spike_list(
        c.t_crow, c.t_col, c.t_perm, values, x, 17, cache=KernelCache()
    )
    for execute in (pull, push):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            execute(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = execute(x)
        # Replay must read changed operands through the same addresses.
        x.neg_()
        graph.replay()
        torch.testing.assert_close(
            output,
            kernels_torch.csr_matvec(c.crow, c.col, values, x),
            atol=2e-5,
            rtol=2e-5,
        )


def test_task_queues_hash_collisions_keep_every_edge(monkeypatch):
    from btorch.sparse.runtime import csr_execution, kernels_triton_tasks, task_layout

    # One long source exercises direct tasks; short sources share colliding
    # destinations and exercise hash aggregation. Deliberately tiny tables
    # force the identical bounded-probe fallback used by Triton.
    source = torch.cat(
        (
            torch.zeros(300, device="cuda", dtype=torch.long),
            torch.arange(1, 33, device="cuda").repeat_interleave(7),
        )
    )
    destination = torch.arange(source.numel(), device="cuda") % 19
    c = RepresentationCache()
    c.build(destination, source, (23, 40), 0)
    monkeypatch.setattr(task_layout, "_HASH_CAPACITY", 2)
    monkeypatch.setattr(task_layout, "_HASH_MAX_PROBE", 1)
    cache = KernelCache()
    state = task_layout.prepare(cache, c.t_crow, c.t_col, c.t_perm, 2, torch.float64)
    assert state.hash_aggregation_mask.any()
    assert (~state.hash_aggregation_mask).any()
    values = torch.randn(source.numel(), device="cuda", dtype=torch.float64)
    packed = task_layout.bind_values(cache, state, values)
    # Task kernels preserve source/batch strides directly. This offset slice
    # has holes between sources and must expose its reachable storage span.
    x = torch.ones(2, 81, device="cuda", dtype=torch.float64)[:, 1::2]
    forward = csr_execution.prepare_tasks(
        tile._Execution(), state, packed, x, 23, cache
    )
    for silent in (False, True):
        if silent:
            x.zero_()
        actual = forward(x)
        expected = kernels_torch.csr_matvec(c.crow, c.col, values, x)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(
            actual,
            kernels_triton_tasks.run(state, packed, x, 23, cache),
            atol=1e-11,
            rtol=1e-11,
        )


def test_listed_backward_preserves_weight_union_and_masked_input():
    from btorch.sparse.runtime.activity import GradientList

    c = _case(n_out=23, n_in=259)
    values = torch.randn(c.col.numel(), device="cuda", dtype=torch.float64)
    x = torch.zeros(1, 259, device="cuda", dtype=torch.float64)
    x[:, ::3] = 1
    gradient = torch.randn(1, 23, device="cuda", dtype=torch.float64)
    tape = torch.zeros(GradientList.size(x.numel()), device="cuda", dtype=torch.int32)
    queue = GradientList(tape, x.numel())
    queue.mask[1::4] = 1
    for union in (False, True):
        indices, counts = queue.work(union)
        selected = (
            queue.mask.bool() | x.flatten().bool() if union else queue.mask.bool()
        )
        for group in range(counts.numel()):
            start = group * BLOCK_SIZE
            rows = selected[start : start + BLOCK_SIZE].nonzero().flatten() + start
            counts[group] = rows.numel()
            indices[start : start + rows.numel()] = rows.to(indices.dtype)
    destination = torch.zeros_like(values)
    execute = tile._prepare_backward(
        KernelCache(),
        *_buffers(c),
        values,
        x,
        gradient,
        True,
        True,
        input_mask=queue,
        grad_values=destination,
    )
    for _ in range(2):
        returned, gx = execute(x, gradient, queue)
        assert returned is None
        expected_values = kernels_torch.edge_grad(c.crow, c.col, gradient, x, 0)
        expected_x = kernels_torch.csr_matvec(
            c.t_crow, c.t_col, values[c.t_perm], gradient
        )
        torch.testing.assert_close(
            gx,
            torch.where(queue.mask.reshape_as(x).bool(), expected_x, 0),
            atol=1e-11,
            rtol=1e-11,
        )
    torch.testing.assert_close(destination, expected_values * 2, atol=1e-11, rtol=1e-11)


def test_compiled_autocast_mixed_low_precision_preserves_full_vjp():
    c = _case(hub=False)
    values = torch.randn(
        c.col.numel(), device="cuda", dtype=torch.float16, requires_grad=True
    )
    x = torch.randn(3, 259, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    row = torch.repeat_interleave(
        torch.arange(17, device="cuda"), c.crow[1:] - c.crow[:-1]
    )

    def product(weight, source):
        return csr_propagate(c.crow, c.col, weight, source, c.t_crow, c.t_col, c.t_perm)

    compiled = torch.compile(product, fullgraph=True)
    with use_backend("tilelang"), torch.autocast("cuda", dtype=torch.float16):
        actual = compiled(values, x)
        gradient = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, (values, x), gradient)
    # Sparse products retain Torch promotion, independent of ambient autocast.
    # FP16 + BF16 promotes to FP32; adjoints return each original operand dtype.
    expected = torch.zeros(3, 17, device="cuda").index_add(
        1, row, x.float().index_select(1, c.col) * values.float()
    )
    oracle = torch.autograd.grad(expected, (values, x), gradient)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for a, b in zip(grads, oracle):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_dense_hubs_and_host_packed_fanout_keep_triton_schedule(monkeypatch):
    from btorch.sparse.runtime import registry, source_layout

    # One 2,200-edge source crosses the dense/hub ownership boundary. The
    # other source has a tiny fanout. Two samples expose packed sample lookup.
    source = torch.cat(
        (
            torch.zeros(2200, device="cuda", dtype=torch.long),
            torch.tensor([1, 1, 8], device="cuda"),
        )
    )
    destination = torch.arange(source.numel(), device="cuda") % 23
    c = RepresentationCache()
    c.build(destination, source, (23, 9), 0)
    values = torch.randn(source.numel(), device="cuda", dtype=torch.float64)
    x = torch.zeros(2, 9, device="cuda", dtype=torch.float64)
    x[0, 0] = 1
    x[1, 8] = -1
    row = torch.repeat_interleave(
        torch.arange(23, device="cuda"), c.crow[1:] - c.crow[:-1]
    )
    expected = x.new_zeros((2, 23)).index_add(1, row, x.index_select(1, c.col) * values)
    active, ptr = kernels_torch.pack_spikes(x)
    with use_backend("tilelang"):
        dense = registry.resolve("spike_push_dense", "cuda")
        packed = registry.resolve("spike_push", "cuda")
    # Force the packed-vs-dense policy's two real device stages independently;
    # default planning still uses the exact shared heuristic.
    for use_dense in (True, False):
        monkeypatch.setattr(source_layout, "_prefer_dense", lambda *args: use_dense)
        result = packed(c.t_crow, c.t_col, c.t_perm, values, x, active, ptr, 23)
        torch.testing.assert_close(result, expected, atol=1e-11, rtol=1e-11)
    result = dense(c.t_crow, c.t_col, c.t_perm, values, x, 23)
    torch.testing.assert_close(result, expected, atol=1e-11, rtol=1e-11)
