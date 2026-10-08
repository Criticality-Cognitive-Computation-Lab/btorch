"""Triton push kernels against independent references.

The kernels under test (``btorch.sparse.runtime.kernels_triton_push``) compute

    out[b, dst(e)] += values[slot(e)] * x[b, src(e)]    for every edge e
                                                        with x[b, src(e)] != 0

with float32 ``atomic_add``. Two references are used:

- the ATen kernel ``kernels_aten.spike_push`` (the contract to reproduce), and
- a float64 dense product ``x @ A.T`` on the CPU, built with a NumPy
  scatter-add that shares no code with ``btorch.sparse``.

Atomic accumulation adds the contributions to one destination in a
non-deterministic order, so results are compared with a *derived* bound
instead of bit-wise (see ``rounding_bound``).
"""

import gc
import importlib.util

import numpy as np
import pytest
import torch

from btorch.sparse.runtime import kernels_aten, kernels_triton_push as ktp
from btorch.sparse.runtime.backend import BackendRegistry, KernelCache
from btorch.sparse.runtime.cache import RepresentationCache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="needs CUDA and triton",
)

DEVICE = "cuda"

# Non-square on purpose: N sources, M destinations. Mixing them up (or using
# the destination-major CSR by mistake) produces a wrong shape or wrong values.
N_SRC, N_DST = 1500, 1100
# Out-degree of the hub (source 7). It must exceed every chunk size of the
# kernels so that its out-edges are split across several programs / tasks.
HUB, HUB_DEGREE = 7, 6000
# Sources that are deliberately left without any out-edge.
SILENT_SOURCES = (0, 3, N_SRC - 1)

# The three ways a result can be produced:
# - "packed": thread-per-gathered-edge kernel on the packed indices,
# - "auto-dense": ``spike_push`` switching to the dense kernel by density,
# - "dense": ``spike_push_dense`` on the dense input, without packing.
MODES = ("packed", "auto-dense", "dense")


class Dense:
    """Float64 dense references of one graph.

    ``matrix [M, N]`` is the operator (duplicate edges summed); ``magnitude``
    sums the absolute values of the same edges and feeds ``rounding_bound``.
    """

    def __init__(self, matrix, magnitude):
        self.matrix = matrix
        self.magnitude = magnitude

    def product(self, x):
        """Exact (float64) ``[B, M]`` result for the float32 input ``x``."""
        return x.double().cpu().numpy() @ self.matrix.T

    def error(self, out, x):
        """``|out - exact|`` as a float64 array."""
        return np.abs(out.double().cpu().numpy() - self.product(x))


def make_graph(seed=0, hub_degree=HUB_DEGREE):
    """Random graph with a hub, silent sources and duplicate edges.

    Returns the :class:`RepresentationCache` on the GPU (the same derived
    buffers a real connection hands to the kernel), float32 ``values`` in
    destination-major CSR order, and a :class:`Dense` pair of float64
    ``[M, N]`` matrices (the operator and its entry-wise magnitude).
    """
    rng = np.random.default_rng(seed)
    n_edge = 30_000
    src = rng.integers(0, N_SRC, n_edge)
    dst = rng.integers(0, N_DST, n_edge)
    # Keep the hub out of the random part so its out-degree is exactly
    # ``hub_degree`` (the threshold test below depends on it).
    src[src == HUB] = HUB + 1
    # The hub fans out to ``hub_degree`` targets; with only 1100 destinations
    # many (hub, dst) pairs repeat, which also exercises duplicate edges
    # between the same two neurons (they must be summed, not overwritten).
    src = np.concatenate([src, np.full(hub_degree, HUB)])
    dst = np.concatenate([dst, rng.integers(0, N_DST, hub_degree)])
    # Explicit duplicates of ordinary edges as well.
    src = np.concatenate([src, src[:50]])
    dst = np.concatenate([dst, dst[:50]])
    keep = ~np.isin(src, SILENT_SOURCES)
    src, dst = src[keep], dst[keep]
    weight = rng.normal(size=src.shape[0]).astype(np.float32)

    dense = np.zeros((N_DST, N_SRC), dtype=np.float64)
    np.add.at(dense, (dst, src), weight.astype(np.float64))
    # Entry-wise sum of |weight|. It differs from ``|dense|`` wherever
    # duplicate edges of opposite sign cancel; the rounding error of the
    # kernel scales with the terms it actually adds, i.e. with this matrix.
    magnitude = np.zeros_like(dense)
    np.add.at(magnitude, (dst, src), np.abs(weight.astype(np.float64)))

    cache = RepresentationCache().to(DEVICE)
    cache.build(
        torch.as_tensor(dst, device=DEVICE),
        torch.as_tensor(src, device=DEVICE),
        (N_DST, N_SRC),
        version=0,
    )
    # ``values`` follow the destination-major CSR order: slot ``perm[k]`` of
    # the edge list sits at CSR position ``k``.
    values = torch.as_tensor(weight, device=DEVICE)[cache.perm]
    return cache, values, Dense(dense, magnitude)


def make_input(kind, n_batch, seed=0):
    """Dense ``[B, N]`` float32 input of the requested activity pattern."""
    gen = torch.Generator().manual_seed(seed)
    amplitude = torch.randn(n_batch, N_SRC, generator=gen)  # non-binary, signed
    if kind == "silent":
        x = torch.zeros(n_batch, N_SRC)
    elif kind == "all":
        x = amplitude.abs() + 0.1  # every entry non-zero
    elif kind == "binary":
        x = (torch.rand(n_batch, N_SRC, generator=gen) < 0.05).float()
        x[:, HUB] = 1.0
    else:  # "sparse": ~1% active with graded amplitudes, hub + silent sources
        x = amplitude * (torch.rand(n_batch, N_SRC, generator=gen) < 0.01)
        x[:, HUB] = 1.5
        # Active sources *without* out-edges must contribute nothing.
        x[:, list(SILENT_SOURCES)] = 2.0
        # Leave one sample completely silent when there is more than one, so
        # an empty ``ptr`` range sits in the middle of the batch.
        if n_batch > 2:
            x[2] = 0.0
    return x.to(DEVICE)


def force_kernel(monkeypatch, mode):
    """Make ``spike_push`` take the packed or the dense kernel.

    It normally chooses by input size, density and mean out-degree
    (``_prefer_dense``); that decision is replaced by a constant so each
    kernel is exercised on every input.
    """
    packed = mode == "packed"
    monkeypatch.setattr(ktp, "_prefer_dense", lambda *_: not packed)


def run(mode, cache, values, x, kernel_cache, monkeypatch):
    """Produce ``[B, M]`` through one of the kernel paths in ``MODES``."""
    args = (cache.t_crow, cache.t_col, cache.t_perm, values, x)
    if mode == "dense":
        return ktp.spike_push_dense(*args, N_DST, cache=kernel_cache)
    force_kernel(monkeypatch, mode)
    active_idx, ptr = kernels_aten.pack_spikes(x)
    return ktp.spike_push(*args, active_idx, ptr, N_DST, cache=kernel_cache)


def rounding_bound(dense, x):
    """Elementwise bound on the float32 error of an arbitrary-order sum.

    Summing ``K`` float32 terms in any order differs from the exact sum by at
    most ``(K - 1) * eps * sum(|terms|)`` (Higham, *Accuracy and Stability of
    Numerical Algorithms*, 2nd ed., eq. 4.4), and each product adds one more
    rounding. With ``K`` bounded by the largest in-degree (duplicates
    included) this gives ``(K + 1) * eps * (|A| @ |x|)``, where ``|A|`` sums
    the magnitudes of duplicate edges (``dense.magnitude``). Observed errors are
    two to three orders of magnitude below this bound; a wrong edge, a lost
    update or a missed sample exceeds it by many orders.
    """
    eps = float(np.finfo(np.float32).eps)
    magnitude = np.abs(x.double().cpu().numpy()) @ dense.magnitude.T
    return (MAX_IN_DEGREE + 1) * eps * magnitude + 1e-30


# In-degree bound used by ``rounding_bound``: edges per destination, counting
# duplicates. 36k edges over 1100 destinations average 33; 256 is generous.
MAX_IN_DEGREE = 256


@pytest.fixture(scope="module")
def graph():
    return make_graph()


@pytest.fixture()
def kernel_cache():
    # A private cache per test: nothing leaks into the global registry.
    return KernelCache()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n_batch", [1, 5])
@pytest.mark.parametrize("kind", ["sparse", "binary", "silent", "all"])
def test_matches_references(graph, kernel_cache, monkeypatch, mode, n_batch, kind):
    """Every path equals the float64 dense product and the ATen kernel."""
    cache, values, dense = graph
    x = make_input(kind, n_batch)
    out = run(mode, cache, values, x, kernel_cache, monkeypatch)

    assert out.shape == (n_batch, N_DST)
    assert out.dtype == torch.float32 and out.device.type == "cuda"

    # Independent reference: float64 dense product on the CPU.
    bound = rounding_bound(dense, x)
    error = dense.error(out, x)
    assert (error <= bound).all(), f"max error {error.max():.3e}"

    # The ATen kernel is itself a float32 sum (in another order), so the two
    # may differ by twice the bound.
    active_idx, ptr = kernels_aten.pack_spikes(x)
    reference = kernels_aten.spike_push(
        cache.t_crow, cache.t_col, cache.t_perm, values, x, active_idx, ptr, N_DST
    )
    difference = (out - reference).abs().double().cpu().numpy()
    assert (difference <= 2 * bound).all()

    if kind == "silent":
        # No atomics at all: the output is exactly zero.
        assert not out.any()


@pytest.mark.parametrize("mode", MODES)
def test_hub_is_fully_delivered(graph, kernel_cache, monkeypatch, mode):
    """A lone spike of the hub reaches each of its 6000 out-edges once.

    With only the hub active the result is one column of the matrix. The hub
    is longer than a block of the packed kernel and than ``_HUB_DEGREE`` /
    ``_HUB_CHUNK`` of the dense kernel, so a chunk that is dropped, repeated
    or mis-aligned changes the sum of the delivered weights.
    """
    cache, values, dense = graph
    assert HUB_DEGREE > max(ktp._BLOCK, ktp._TILE_BLOCK, ktp._HUB_DEGREE)
    x = torch.zeros(1, N_SRC, device=DEVICE)
    x[0, HUB] = 1.0
    out = run(mode, cache, values, x, kernel_cache, monkeypatch)
    # The exact result is column HUB of the matrix.
    np.testing.assert_array_equal(dense.product(x)[0], dense.matrix[:, HUB])
    error = dense.error(out, x)
    assert (error <= rounding_bound(dense, x)).all(), f"max error {error.max():.3e}"


@pytest.mark.parametrize("hub_degree", [ktp._HUB_DEGREE, ktp._HUB_DEGREE + 1])
def test_hub_threshold_boundary(kernel_cache, hub_degree):
    """Dense kernel: a source is served by its tile *or* by hub tasks.

    Out-degree exactly ``_HUB_DEGREE`` stays in the tile, one more moves to
    the hub task list. Either way every edge is delivered exactly once; an
    off-by-one in the two complementary tests would deliver the source twice
    or not at all.
    """
    cache, values, dense = make_graph(seed=1, hub_degree=hub_degree)
    degree = int(cache.t_crow[HUB + 1] - cache.t_crow[HUB])
    assert degree == hub_degree
    layout = ktp._layout(kernel_cache, cache.t_crow, cache.t_col, cache.t_perm)
    assert (layout.n_hub > 0) == (degree > ktp._HUB_DEGREE)
    x = make_input("sparse", 2)
    out = ktp.spike_push_dense(
        cache.t_crow, cache.t_col, cache.t_perm, values, x, N_DST, cache=kernel_cache
    )
    assert (dense.error(out, x) <= rounding_bound(dense, x)).all()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_source_major_values(graph, kernel_cache, monkeypatch, mode, dtype):
    """``source_major_values=True`` reads ``values`` without ``t_perm``.

    The caller then supplies ``values[t_perm]`` (one value per source-major
    entry). The result must be identical to the default call that is handed
    the destination-major values; float64 exercises the ATen fallback of the
    same option.
    """
    cache, values, dense = graph
    x = make_input("sparse", 3).to(dtype)
    values_t = values[cache.t_perm].to(dtype)
    args = (cache.t_crow, cache.t_col, cache.t_perm, values_t, x)
    if mode == "dense":
        out = ktp.spike_push_dense(
            *args, N_DST, source_major_values=True, cache=kernel_cache
        )
    else:
        force_kernel(monkeypatch, mode)
        active_idx, ptr = kernels_aten.pack_spikes(x)
        out = ktp.spike_push(
            *args, active_idx, ptr, N_DST, source_major_values=True, cache=kernel_cache
        )
    assert out.dtype == dtype
    assert (dense.error(out, x) <= rounding_bound(dense, x)).all()


def test_fallback_for_unsupported_inputs(graph, kernel_cache):
    """Non-float32 or CPU inputs are forwarded to the ATen kernel unchanged.

    The Triton kernels are float32/CUDA only. Float64 goes through the very
    same ATen code as the reference, so (up to the atomics of ``index_add_``)
    the results agree to float64 precision, and nothing is cached.
    """
    cache, values, dense = graph
    x = make_input("sparse", 2).double()
    active_idx, ptr = kernels_aten.pack_spikes(x)
    args = (cache.t_crow, cache.t_col, cache.t_perm, values.double(), x)
    out = ktp.spike_push(*args, active_idx, ptr, N_DST, cache=kernel_cache)
    assert out.dtype == torch.float64
    exact = dense.product(x)
    np.testing.assert_allclose(out.cpu().numpy(), exact, rtol=1e-10, atol=1e-10)
    out = ktp.spike_push_dense(*args, N_DST, cache=kernel_cache)
    np.testing.assert_allclose(out.cpu().numpy(), exact, rtol=1e-10, atol=1e-10)
    assert ("triton_push", "layouts") not in kernel_cache._store

    # CPU tensors: same forwarding.
    cpu = [t.cpu() for t in (cache.t_crow, cache.t_col, cache.t_perm, values)]
    x_cpu = make_input("sparse", 2).cpu()
    active_idx, ptr = kernels_aten.pack_spikes(x_cpu)
    out = ktp.spike_push(*cpu, x_cpu, active_idx, ptr, N_DST, cache=kernel_cache)
    assert out.device.type == "cpu"
    assert (dense.error(out, x_cpu) <= rounding_bound(dense, x_cpu)).all()


def test_non_contiguous_input(graph, kernel_cache, monkeypatch):
    """A strided view of ``x`` gives the same result as its contiguous copy.

    The kernels address ``x`` as a flat ``[B * N]`` array, so a view must be
    materialised first; reading the underlying storage instead would use the
    wrong samples.
    """
    cache, values, dense = graph
    wide = make_input("sparse", 6)
    view = wide[::2]  # samples 0, 2, 4: not contiguous
    assert not view.is_contiguous()
    for mode in MODES:
        out = run(mode, cache, values, view, kernel_cache, monkeypatch)
        assert (dense.error(out, view) <= rounding_bound(dense, view)).all()


def test_layout_follows_in_place_rewiring(kernel_cache, monkeypatch):
    """Derived int32 layouts are refreshed when the cache is rewired in place.

    ``RepresentationCache.build`` writes a new topology with the same number
    of edges into the *existing* buffers: same tensor objects, same addresses,
    new contents. A layout keyed by identity or address alone would keep
    pushing along the old edges. The kernel must notice (through the tensors'
    version counters) and refresh its copies, re-using its own buffers too.
    """
    cache, values, dense = make_graph(seed=2)
    x = make_input("sparse", 3)
    before = ktp._layout(kernel_cache, cache.t_crow, cache.t_col, cache.t_perm)
    col_buffer = before.col

    # Rewire: reverse the destination of every edge (M - 1 - dst). The edge
    # count is unchanged, so ``build`` copies into the existing buffers.
    t_col_object, t_col_address = cache.t_col, cache.t_col.data_ptr()
    row = torch.repeat_interleave(
        torch.arange(N_DST, device=DEVICE), cache.crow[1:] - cache.crow[:-1]
    )
    weight = values.clone()  # per CSR entry, re-ordered below
    new_row, new_col = N_DST - 1 - row, cache.col.clone()
    cache.build(new_row, new_col, (N_DST, N_SRC), version=1)
    assert cache.t_col is t_col_object and cache.t_col.data_ptr() == t_col_address
    # Edge list slot k is now at CSR position given by ``perm``.
    values = weight[cache.perm]
    # The same reversal on the dense references.
    dense = Dense(dense.matrix[::-1].copy(), dense.magnitude[::-1].copy())

    for mode in MODES:
        out = run(mode, cache, values, x, kernel_cache, monkeypatch)
        assert (dense.error(out, x) <= rounding_bound(dense, x)).all(), mode

    after = ktp._layout(kernel_cache, cache.t_crow, cache.t_col, cache.t_perm)
    # Same holder and same int32 buffer, refreshed in place: tensors captured
    # by a compiled graph or a CUDA graph stay valid across rewiring.
    assert after is before and after.col is col_buffer
    assert torch.equal(after.col, cache.t_col.to(torch.int32))


def test_layout_cache_is_bounded(kernel_cache, monkeypatch):
    """Layouts of dead connections are dropped; unchanged ones are re-used."""
    cache, values, _ = make_graph(seed=3)
    first = ktp._layout(kernel_cache, cache.t_crow, cache.t_col, cache.t_perm)
    again = ktp._layout(kernel_cache, cache.t_crow, cache.t_col, cache.t_perm)
    assert again is first  # hit: nothing rebuilt
    holders = kernel_cache.get(("triton_push", "layouts"), dict)
    assert len(holders) == 1
    # Free the connection: its holder (and with it the int32 copies on the
    # device) goes away at once, not only when another layout is built.
    del cache, first, again
    gc.collect()
    assert len(holders) == 0
    # Then build another one: the new buffers may land on the old addresses
    # and must get their own holder.
    other, _, _ = make_graph(seed=4)
    ktp._layout(kernel_cache, other.t_crow, other.t_col, other.t_perm)
    assert len(holders) == 1

    # Graphs that stay alive are bounded by ``_MAX_LAYOUTS`` (oldest first);
    # an evicted layout is rebuilt on its next use.
    monkeypatch.setattr(ktp, "_MAX_LAYOUTS", 3)
    alive = [make_graph(seed=10 + i)[0] for i in range(5)]
    for c in alive:
        ktp._layout(kernel_cache, c.t_crow, c.t_col, c.t_perm)
        assert len(holders) <= 3
    c = alive[0]
    assert (id(c.t_crow), id(c.t_col), id(c.t_perm)) not in holders
    rebuilt = ktp._layout(kernel_cache, c.t_crow, c.t_col, c.t_perm)
    assert torch.equal(rebuilt.col, c.t_col.to(torch.int32))


@pytest.mark.parametrize("mode", MODES)
def test_views_sharing_an_address_are_not_confused(kernel_cache, monkeypatch, mode):
    """Views with equal address, shape and version get their own layouts.

    ``base[:E]`` and ``base[::2]`` of a ``[2 E]`` buffer start at the same
    address, have the same shape and share one version counter; only their
    strides differ. Each view is a different graph and must be pushed along
    its own edges: the layout is tied to the tensor *object*, and the strided
    view is copied element by element into the kernel's contiguous buffers.
    """
    cache, values, dense = make_graph(seed=5)
    x = make_input("sparse", 3)
    n_edge = cache.t_col.shape[0]
    gen = torch.Generator(device=DEVICE).manual_seed(6)
    # ``head`` is the real destination list, ``strided`` a different one.
    base = torch.randint(0, N_DST, (2 * n_edge,), device=DEVICE, generator=gen)
    base[:n_edge] = cache.t_col
    head, strided = base[:n_edge], base[::2]
    assert head.data_ptr() == strided.data_ptr() and head._version == strided._version
    assert not torch.equal(head, strided)

    def push(t_col):
        args = (cache.t_crow, t_col, cache.t_perm, values, x)
        if mode == "dense":
            return ktp.spike_push_dense(*args, N_DST, cache=kernel_cache)
        force_kernel(monkeypatch, mode)
        active_idx, ptr = kernels_aten.pack_spikes(x)
        return ktp.spike_push(*args, active_idx, ptr, N_DST, cache=kernel_cache)

    def expected(t_col):
        # Plain indexing: edge e of source s (t_crow) reaches t_col[e].
        src = torch.repeat_interleave(
            torch.arange(N_SRC, device=DEVICE), cache.t_crow[1:] - cache.t_crow[:-1]
        )
        contrib = x.double()[:, src] * values.double()[cache.t_perm]
        out = torch.zeros(x.shape[0], N_DST, dtype=torch.float64, device=DEVICE)
        return out.index_add(1, t_col, contrib)

    for t_col in (head, strided, head):
        out, exact = push(t_col), expected(t_col)
        scale = float(exact.abs().max())
        assert float((out - exact).abs().max()) <= 1e-4 * scale


def test_mismatched_sizes_and_index_types_fall_back(graph, kernel_cache, monkeypatch):
    """Operands the kernels would read out of bounds go to the reference.

    The kernels index ``values``, the pointers and the packed lists without
    bounds checks and read the packed lists as contiguous int64. Anything
    else is routed to the ATen kernel. The reference is replaced by a marker
    and the launcher by a tripwire, so only the routing is observed.
    """
    cache, values, dense = graph
    x = make_input("sparse", 3)
    active_idx, ptr = kernels_aten.pack_spikes(x)

    def tripwire(*args, **kwargs):
        raise AssertionError("a Triton kernel was launched")

    monkeypatch.setattr(ktp.kernels_triton, "_launch", tripwire)
    monkeypatch.setattr(ktp, "_reference", lambda *a: "reference")
    buffers = (cache.t_crow, cache.t_col, cache.t_perm)

    def packed(*args):
        return ktp.spike_push(*args, N_DST, cache=kernel_cache)

    # One value missing; an input that is one neuron too wide.
    assert packed(*buffers, values[:-1], x, active_idx, ptr) == "reference"
    wide = torch.cat([x, x[:, :1]], dim=1)
    assert packed(*buffers, values, wide, active_idx, ptr) == "reference"
    assert (
        ktp.spike_push_dense(*buffers, values, wide, N_DST, cache=kernel_cache)
        == "reference"
    )
    # A permutation of the wrong length.
    short_perm = (cache.t_crow, cache.t_col, cache.t_perm[:-1])
    assert packed(*short_perm, values, x, active_idx, ptr) == "reference"
    # int32 packed indices; sample offsets of the wrong length.
    assert packed(*buffers, values, x, active_idx.int(), ptr) == "reference"
    assert packed(*buffers, values, x, active_idx, ptr[:-1]) == "reference"


def test_non_contiguous_packed_indices(graph, kernel_cache, monkeypatch):
    """A strided view of the packed index list is read element by element."""
    cache, values, dense = graph
    x = make_input("sparse", 3)
    active_idx, ptr = kernels_aten.pack_spikes(x)
    doubled = torch.stack([active_idx, torch.zeros_like(active_idx)], dim=1)
    view = doubled.reshape(-1)[::2]
    assert not view.is_contiguous() and torch.equal(view, active_idx)
    force_kernel(monkeypatch, "packed")
    out = ktp.spike_push(
        cache.t_crow,
        cache.t_col,
        cache.t_perm,
        values,
        x,
        view,
        ptr,
        N_DST,
        cache=kernel_cache,
    )
    assert (dense.error(out, x) <= rounding_bound(dense, x)).all()


@pytest.mark.parametrize("mode", MODES)
def test_direct_launch_matches_public_launch(graph, kernel_cache, monkeypatch, mode):
    """The cached-handle launch runs the same kernels as Triton's own path.

    With a known Triton version the kernels are launched through a
    cached compiled handle (no per-call argument specialisation). Both
    launch paths must give correct results, also on the second (steady-
    state) call.
    """
    cache, values, dense = graph
    x = make_input("sparse", 3)
    for direct in (True, False):
        monkeypatch.setattr(ktp.kernels_triton, "_DIRECT_LAUNCH", direct)
        for _ in range(2):
            out = run(mode, cache, values, x, kernel_cache, monkeypatch)
            assert (dense.error(out, x) <= rounding_bound(dense, x)).all()


def test_register_on_private_registry(graph):
    """``register`` adds a preferred CUDA backend and leaves the CPU alone.

    A fresh registry is used so the global one (and therefore every
    other test) keeps resolving what it resolved before.
    """
    cache, values, dense = graph
    registry = BackendRegistry()
    registry.register("spike_push", "aten", kernels_aten.spike_push)
    ktp.register(registry)
    assert registry.available("spike_push", "cuda") == ["triton", "aten"]
    # The pack-free kernel is offered under its own name.
    assert registry.available("spike_push_dense", "cuda") == ["triton"]
    assert registry.available("spike_push", "cpu") == ["aten"]
    assert registry.name("spike_push", "cuda") == "triton"

    x = make_input("sparse", 4)
    active_idx, ptr = kernels_aten.pack_spikes(x)
    push = registry.resolve("spike_push", "cuda")
    out = push(
        cache.t_crow, cache.t_col, cache.t_perm, values, x, active_idx, ptr, N_DST
    )
    assert (dense.error(out, x) <= rounding_bound(dense, x)).all()
    dense_out = registry.resolve("spike_push_dense", "cuda")(
        cache.t_crow, cache.t_col, cache.t_perm, values, x, N_DST
    )
    assert (dense.error(dense_out, x) <= rounding_bound(dense, x)).all()
    # The derived layout went into *this* registry's kernel cache.
    assert len(registry.kernels.get(("triton_push", "layouts"), dict)) == 1
