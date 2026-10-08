"""Acceptance tests of the sparse execution runtime.

The runtime is the layer below ``btorch.models.connection``: registered custom
operators (the ``torch.compile`` boundary), the cache of derived layouts and
the kernel-backend registry. Model code never calls these directly, but they
define the numerical contract every connection relies on, so they are tested
against dense references built here with NumPy (never with the runtime).
"""

import numpy as np
import pytest
import torch
from torch import nn

from btorch.sparse import runtime
from btorch.sparse.runtime import (
    BackendRegistry,
    RepresentationCache,
    kernels_aten,
    ops,
)
from tests.sparse.helpers import DEVICES


M, N = 5, 7  # non-square: a transposed product cannot pass by accident


def make_layout(seed=0, variant="plain", dtype=torch.float64, device="cpu", batch=()):
    """Random operator ``A [M, N]`` as raw buffers plus its dense tensor.

    Buffers, as every operator takes them: ``crow [M + 1]``, ``col [E]``
    (destination-major CSR), ``values [*batch, E]`` in that order, and the
    same edges sorted by source: ``t_crow [N + 1]``, ``t_col [E]`` and
    ``t_perm [E]``, the CSR position of every source-major entry.

    Derived with NumPy sorts, independently of :class:`RepresentationCache`.
    ``variant``: ``"plain"``, ``"empty_rows"`` (rows 1, 4 and column 0 hold
    no entry) or ``"nnz0"`` (no entry at all).
    """
    rng = np.random.default_rng(seed)
    flat = np.sort(rng.choice(M * N, size=M * N // 3, replace=False))
    row, col = flat // N, flat % N  # row-major sorted == CSR order
    if variant == "empty_rows":
        keep = (row != 1) & (row != 4) & (col != 0)
        row, col = row[keep], col[keep]
    elif variant == "nnz0":
        row, col = row[:0], col[:0]
    n_edge = len(row)
    values = rng.normal(size=(*batch, n_edge))
    dense = np.zeros((*batch, M, N))
    dense[..., row, col] = values
    crow = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=M))])
    t_perm = np.lexsort((row, col))  # sort by source, then destination
    t_crow = np.concatenate([[0], np.cumsum(np.bincount(col, minlength=N))])

    index = [crow, col, t_crow, row[t_perm], t_perm]
    crow, col, t_crow, t_col, t_perm = (
        torch.as_tensor(np.asarray(a), dtype=torch.long, device=device) for a in index
    )
    values, dense = (
        torch.as_tensor(a, dtype=dtype, device=device) for a in (values, dense)
    )
    return (crow, col, values, t_crow, t_col, t_perm), dense


def propagate(buffers, x, values=None):
    crow, col, v, t_crow, t_col, t_perm = buffers
    v = v if values is None else values
    return ops.csr_propagate(crow, col, v, x, t_crow, t_col, t_perm)


def spike_propagate(buffers, x, max_density, values=None):
    crow, col, v, t_crow, t_col, t_perm = buffers
    v = v if values is None else values
    return ops.spike_propagate(crow, col, v, x, t_crow, t_col, t_perm, max_density)


def dense_apply(dense, x):
    """Reference ``y = A @ x`` along the last axis with a value batch."""
    n_vb = dense.ndim - 2
    lhs = "".join("abc"[:n_vb])
    return torch.einsum(f"{lhs}mn,{lhs}...n->{lhs}...m", dense, x)


# One entry per situation the operators must handle:
# (variant, value batch, shape of x, dtype).
SAMPLES = {
    "vector": ("plain", (), (N,), torch.float32),
    "batch": ("plain", (), (3, N), torch.float64),
    "time_batch": ("plain", (), (2, 3, N), torch.float32),
    "empty_rows": ("empty_rows", (), (3, N), torch.float64),
    "nnz0": ("nnz0", (), (3, N), torch.float32),
    "value_batch": ("plain", (2,), (2, 3, N), torch.float64),
    # A size-1 network dimension of x broadcasts against the value batch.
    "value_batch_broadcast": ("plain", (2,), (1, 3, N), torch.float32),
}


def make_sample(name, device="cpu", requires_grad=False, dtype=None):
    variant, batch, x_shape, sample_dtype = SAMPLES[name]
    dtype = dtype or sample_dtype
    buffers, dense = make_layout(1, variant, dtype, device, batch)
    x = torch.randn(x_shape, dtype=dtype, device=device)
    if requires_grad:
        buffers[2].requires_grad_()
        x.requires_grad_()
    return buffers, dense, x


# -------------------------------------------------------- forward / backward
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", SAMPLES)
def test_csr_propagate_matches_dense(name, device):
    """``csr_propagate`` is ``A @ x`` for every supported input shape, and.

    its gradients are the dense ones: ``dL/dx = upstream @ A`` and
    ``dL/dvalues[k] = sum_samples upstream[row[k]] * x[col[k]]``.
    """
    buffers, dense, x = make_sample(name, device, requires_grad=True)
    dense.requires_grad_()  # the dense twin gets the reference gradients
    tol = {"atol": 1e-5, "rtol": 1e-5}
    out = propagate(buffers, x)
    expected = dense_apply(dense, x)
    assert out.shape == expected.shape and out.dtype == x.dtype
    torch.testing.assert_close(out, expected, **tol)
    # The raw (non-differentiable) kernel entry point computes the same.
    raw = ops.csr_matvec(buffers[0], buffers[1], buffers[2].detach(), x.detach())
    torch.testing.assert_close(raw, expected, **tol)

    upstream = torch.randn_like(expected)
    g_values, g_x = torch.autograd.grad((out * upstream).sum(), (buffers[2], x))
    ref_dense, ref_x = torch.autograd.grad((expected * upstream).sum(), (dense, x))
    row = torch.repeat_interleave(torch.arange(M, device=device), buffers[0].diff())
    torch.testing.assert_close(g_x, ref_x, **tol)
    torch.testing.assert_close(g_values, ref_dense[..., row, buffers[1]], **tol)
    # The raw ``csr_edge_grad`` kernel is that sampled outer product
    # ``upstream^T x`` at the stored entries, summed over the samples.
    n_vb = buffers[2].ndim - 1
    raw = ops.csr_edge_grad(buffers[0], buffers[1], upstream, x.detach(), n_vb)
    torch.testing.assert_close(raw, ref_dense[..., row, buffers[1]], **tol)


def test_row_index_memo_is_not_keyed_by_memory_address():
    """Regression test.

    Layouts are freed and rebuilt all the time (a second
    model, a moved module). A pointer tensor that happens to live at the
    address of an earlier one must be read, not recognised: memoising row
    indices by ``data_ptr`` once gave silently wrong weight gradients. The
    address reuse is forced here by wrapping one NumPy buffer twice.
    """
    storage = np.zeros(M + 1, dtype=np.int64)
    x = torch.randn(3, N, dtype=torch.float64)
    grad = torch.randn(3, M, dtype=torch.float64)
    full = grad.T @ x  # dense dL/dA; the value gradient is its stored entries
    with runtime.use_backend("aten"):
        for seed in (0, 1):  # two patterns with the same number of entries
            (crow, col, *_), _ = make_layout(seed)
            row = torch.repeat_interleave(torch.arange(M), crow.diff())
            storage[:] = crow.numpy()
            out = ops.csr_edge_grad(torch.from_numpy(storage), col, grad, x, 0)
            torch.testing.assert_close(out, full[row, col])


# ------------------------------------------------------- opcheck / gradcheck
@pytest.mark.parametrize("name", SAMPLES)
def test_opcheck(name):
    """Schema, fake kernel, autograd registration and AOT dispatch of every
    registered operator (``torch.library.opcheck``, spec section 36)."""
    buffers, dense, x = make_sample(name, requires_grad=True)
    crow, col, values, *transposed = buffers
    torch.library.opcheck(ops.csr_propagate, (crow, col, values, x, *transposed))
    if values.ndim == 1:
        # 1.0: always source-driven; 0.0: always the destination-driven
        # fallback. Both must satisfy the operator contract.
        for max_density in (0.0, 1.0):
            args = (crow, col, values, x, *transposed, max_density)
            torch.library.opcheck(ops.spike_propagate, args)
    # The raw kernels used by the backward are registered operators too.
    values, x = values.detach(), x.detach()
    torch.library.opcheck(ops.csr_matvec, (crow, col, values, x))
    grad = torch.randn(dense_apply(dense, x).shape, dtype=x.dtype)
    torch.library.opcheck(ops.csr_edge_grad, (crow, col, grad, x, values.ndim - 1))


@pytest.mark.parametrize("name", SAMPLES)
def test_gradcheck_values_and_input(name):
    """Analytic gradients w.r.t.

    ``values`` and ``x`` match finite
    differences, including the broadcast of a size-1 network dimension of
    ``x`` (its gradient is summed over the networks).
    """
    buffers, _, x = make_sample(name, requires_grad=True, dtype=torch.float64)
    values = buffers[2]
    assert torch.autograd.gradcheck(lambda v, x: propagate(buffers, x, v), (values, x))
    if values.ndim == 1:
        for max_density in (0.0, 1.0):
            assert torch.autograd.gradcheck(
                lambda v, x: spike_propagate(buffers, x, max_density, v), (values, x)
            )


NON_CONTIGUOUS = {
    "expanded": lambda v: v[:1].expand(v.shape[0]),  # stride 0: one constant
    "strided": lambda v: torch.stack([v, -v], dim=1)[:, 0],  # stride 2
}


@pytest.mark.parametrize("layout", NON_CONTIGUOUS)
@pytest.mark.parametrize("device", DEVICES)
def test_non_contiguous_values(layout, device):
    """``values`` may be any view (a weight module can return an expanded
    scalar or a slice); the result must not depend on its memory layout."""
    buffers, _, x = make_sample("batch", device)
    values = NON_CONTIGUOUS[layout](buffers[2])
    assert not values.is_contiguous()
    with runtime.use_backend("aten"):
        out = propagate(buffers, x, values)
        expected = propagate(buffers, x, values.contiguous())
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------- spike propagation
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_density", [0.0, 0.05, 1.0])
@pytest.mark.parametrize("density", [0.0, 0.02, 0.05, 0.3, 1.0])
def test_spike_propagate_equals_csr_propagate(density, max_density, device):
    """The activity-adaptive operator is mathematically ``csr_propagate``.

    Input densities on both sides of ``max_density`` exercise the
    source-driven path (packed spikes) and the destination-driven fallback.
    The amplitudes are not 0/1 on purpose: packed indices are an execution
    aid, the dense tensor supplies the values.
    """
    buffers, dense, _ = make_sample("batch", device)
    gen = torch.Generator().manual_seed(3)
    x = torch.rand(4, 6, N, generator=gen, dtype=torch.float64)
    x = (x * (torch.rand(4, 6, N, generator=gen) < density)).to(device)
    out = spike_propagate(buffers, x, max_density)
    torch.testing.assert_close(out, propagate(buffers, x))
    torch.testing.assert_close(out, dense_apply(dense, x))
    # A value batch is not handled by the push kernel: same result anyway.
    buffers, dense, _ = make_sample("value_batch", device)
    out = spike_propagate(buffers, x[:2], max_density)
    torch.testing.assert_close(out, dense_apply(dense, x[:2]))


@pytest.mark.parametrize("max_density", [0.0, 1.0])
def test_spike_propagate_gradient_reaches_silent_inputs(max_density):
    """Surrogate-gradient requirement (spec section 24).

    A silent neuron (``x == 0``) contributes nothing to the forward pass, but
    its gradient is generally non-zero: ``dL/dx = upstream @ A`` for *every*
    entry. A backward that only visited the packed active indices would
    return zero there and break surrogate-gradient training.
    """
    buffers, dense, _ = make_sample("batch", requires_grad=True)
    x = torch.zeros(3, N, dtype=torch.float64)
    x[0, 2] = 1.0  # one spike in 21 entries: the push path is taken
    x.requires_grad_()
    upstream = torch.randn(3, M, dtype=torch.float64)
    (spike_propagate(buffers, x, max_density) * upstream).sum().backward()
    expected = upstream @ dense  # dense reference, [3, N]
    torch.testing.assert_close(x.grad, expected)
    silent = x.detach() == 0
    assert (x.grad[silent] != 0).sum() > silent.sum() // 2
    # Edge weights, in contrast, only learn from the one active source.
    assert (buffers[2].grad != 0).sum() == (buffers[1] == 2).sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_pack_spikes_contract(device):
    """``pack_spikes(x [B, N]) -> (active_idx, ptr)``: sample ``b`` owns
    ``active_idx[ptr[b]:ptr[b + 1]]``, the ascending non-zero positions."""
    x = torch.tensor([[0, 1, 0, 2.5], [0, 0, 0, 0], [1, 1, 1, 1]], device=device)
    active_idx, ptr = runtime.pack_spikes(x)
    assert ptr.tolist() == [0, 2, 2, 6]  # an all-silent sample owns nothing
    assert active_idx.tolist() == [1, 3, 0, 1, 2, 3]
    assert active_idx.dtype == ptr.dtype == torch.long
    assert active_idx.device == ptr.device == x.device
    empty_idx, empty_ptr = runtime.pack_spikes(torch.zeros(2, 4, device=device))
    assert empty_idx.numel() == 0 and empty_ptr.tolist() == [0, 0, 0]


# --------------------------------------------------------- RepresentationCache
def cache_dense(cache, values, transposed):
    """Dense matrix rebuilt from one derived layout.

    ``values`` are in slot
    order; ``perm`` maps them to CSR order, ``t_perm`` on to source-major
    order. Duplicate edges stay separate entries and are summed here.
    """
    n_out, n_in = cache.shape
    v_csr = values[cache.perm]
    dense = torch.zeros(n_out, n_in, dtype=values.dtype)
    if transposed:
        src = torch.repeat_interleave(torch.arange(n_in), cache.t_crow.diff())
        dense.index_put_((cache.t_col, src), v_csr[cache.t_perm], accumulate=True)
    else:
        dst = torch.repeat_interleave(torch.arange(n_out), cache.crow.diff())
        dense.index_put_((dst, cache.col), v_csr, accumulate=True)
    return dense


# Slot lists (row = destination, col = source), deliberately unsorted and with
# the edge (3, 1) stored twice; rows 1 and 4 are empty.
ROW_A = torch.tensor([3, 0, 2, 3, 0, 3])
COL_A = torch.tensor([1, 6, 2, 0, 1, 1])
ROW_B = torch.tensor([4, 4, 1, 0, 2, 1])  # a different pattern of equal nnz
COL_B = torch.tensor([0, 5, 5, 3, 3, 2])


@pytest.mark.parametrize("transposed", [False, True], ids=["csr", "source_csr"])
@pytest.mark.parametrize("pattern", ["unsorted_duplicates", "sorted"])
def test_cache_layouts_reconstruct_the_matrix(pattern, transposed):
    """Both derived layouts describe exactly the canonical edge list."""
    row, col = ROW_A, COL_A
    if pattern == "sorted":
        order = torch.argsort(row * N + col, stable=True)
        row, col = row[order], col[order]
    values = torch.arange(1.0, 7.0, dtype=torch.float64)
    expected = torch.zeros(M, N, dtype=torch.float64)
    expected.index_put_((row, col), values, accumulate=True)

    cache = RepresentationCache()
    cache.build(row, col, (M, N), version=7)
    assert cache.topology_version == 7 and cache.shape == (M, N)
    # Slots already in execution order need no permutation of the values.
    assert cache.identity_perm == (pattern == "sorted")
    assert cache.crow.shape == (M + 1,) and cache.t_crow.shape == (N + 1,)
    assert int(cache.crow[-1]) == int(cache.t_crow[-1]) == 6
    torch.testing.assert_close(cache_dense(cache, values, transposed), expected)


def test_cache_rebuild_is_in_place_for_equal_nnz():
    """Rewiring / checkpoint loading with an unchanged number of edges writes
    into the existing tensors, so compiled graphs that captured them stay
    valid."""
    cache = RepresentationCache()
    cache.build(ROW_A, COL_A, (M, N), version=0)
    before = {name: getattr(cache, name) for name in cache._NAMES}
    cache.build(ROW_B, COL_B, (M, N), version=1)
    for name, tensor in before.items():
        assert getattr(cache, name) is tensor, name
    values = torch.arange(1.0, 7.0)
    expected = torch.zeros(M, N).index_put_((ROW_B, COL_B), values, accumulate=True)
    for transposed in (False, True):  # ... and the content is the new pattern
        torch.testing.assert_close(cache_dense(cache, values, transposed), expected)
    assert cache.topology_version == 1


def test_cache_is_not_model_state_and_follows_to():
    """Derived layouts never enter a ``state_dict`` but move with the owner."""
    owner = nn.Module()
    owner.cache = RepresentationCache()
    owner.cache.build(ROW_A, COL_A, (M, N), version=0)
    assert list(owner.state_dict()) == []
    assert {n for n, _ in owner.named_buffers()} == {
        f"cache.{n}" for n in owner.cache._NAMES
    }
    owner.to(torch.float64)  # a dtype move leaves the integer layouts alone
    assert all(b.dtype == torch.long for b in owner.buffers())
    if torch.cuda.is_available():
        owner.to("cuda")
        assert all(b.device.type == "cuda" for b in owner.buffers())
        owner.cache.build(ROW_B.cuda(), COL_B.cuda(), (M, N), version=1)
        assert owner.cache.crow.device.type == "cuda"


# ------------------------------------------------------------ backend registry
def test_registry_priority_availability_and_override():
    """A kernel resolves to its highest-priority *available* backend; an
    override forces a backend where it exists and is ignored elsewhere."""
    reg = BackendRegistry()  # a private registry: the global one is untouched
    reg.register("k", "slow", lambda: "slow", device="cpu")
    reg.register("k", "fast", lambda: "fast", device="cpu", priority=5)
    reg.register("k", "off", lambda: "off", priority=9, available=lambda: False)
    reg.register("other", "slow", lambda: "other-slow", device="cpu")
    assert reg.available("k", "cpu") == ["fast", "slow"]
    assert reg.name("k", "cpu") == "fast" and reg.resolve("k", "cpu")() == "fast"

    reg.set_backend("slow")
    assert reg.resolve("k", "cpu")() == "slow"
    reg.set_backend("fast")  # "other" has no "fast" backend: default is kept
    assert reg.name("k", "cpu") == "fast" and reg.name("other", "cpu") == "slow"
    reg.set_backend(None)
    assert reg.name("k", "cpu") == "fast"

    # Unknown kernels, and known kernels on a device without a backend.
    for kernel, device in [("no_such_kernel", "cpu"), ("k", "cuda")]:
        with pytest.raises(RuntimeError, match="No backend for kernel"):
            reg.resolve(kernel, device)

    # The global registry: ``use_backend`` forces a backend inside the block
    # and restores the previous choice afterwards, also when the block raises.
    registry = runtime.registry
    assert registry.name("csr_matvec", "cpu") == "aten"  # the reference backend
    with pytest.raises(KeyError), runtime.use_backend("torch_sparse"):
        assert registry._override == "torch_sparse"
        raise KeyError("boom")
    assert registry._override is None
    assert registry.name("csr_matvec", "cpu") == "aten"


@pytest.mark.skipif(
    not kernels_aten.torch_sparse_available(), reason="torch_sparse not installed"
)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", SAMPLES)
def test_torch_sparse_backend_equals_aten(name, device):
    """The optional legacy backend is selectable and numerically identical,
    forward and backward (the backward goes through the same kernel)."""
    buffers, dense, x = make_sample(name, device, requires_grad=True)
    registry = runtime.registry
    defaults = [registry.name(k, device) for k in ("csr_matvec", "edge_grad")]
    with runtime.use_backend("aten"):
        reference = propagate(buffers, x)
        g_ref = torch.autograd.grad(reference.sum(), (buffers[2], x))
    with runtime.use_backend("torch_sparse"):
        assert registry.name("csr_matvec", device) == "torch_sparse"
        # Kernels the backend does not implement keep their default.
        assert registry.name("edge_grad", device) == defaults[1]
        out = propagate(buffers, x)
        g_out = torch.autograd.grad(out.sum(), (buffers[2], x))
    assert registry.name("csr_matvec", device) == defaults[0]
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out, dense_apply(dense, x), atol=1e-5, rtol=1e-5)
    for a, b in zip(g_out, g_ref):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------- CUDA / AMP
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float16])
def test_half_precision_input_under_autocast(value_dtype):
    """Mixed precision: activity produced under ``torch.autocast`` is float16
    (no native sparse product). Both algorithms give the dense result; the
    output dtype is ``promote_types(values, x)`` whatever the autocast region
    says (as the fake kernels used for tracing declare), and gradients come
    back in the dtypes of the leaves."""
    buffers, dense, x = make_sample("batch", "cuda", dtype=torch.float32)
    values = buffers[2].to(value_dtype).requires_grad_()
    x = x.half().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = [
            propagate(buffers, x, values),
            spike_propagate(buffers, x, 1.0, values),
        ]
    tol = {"atol": 2e-2, "rtol": 2e-2}
    expected = dense_apply(dense, x.detach().float())
    for out in outputs:
        assert out.dtype == torch.promote_types(value_dtype, torch.float16)
        torch.testing.assert_close(out.float(), expected, **tol)
        g_values, g_x = torch.autograd.grad(out.float().sum(), (values, x))
        assert g_values.dtype == value_dtype and g_x.dtype == torch.float16
        ones = torch.ones(3, M, device="cuda")
        torch.testing.assert_close(g_x.float(), ones @ dense, **tol)
