"""Triton pull kernels (``csr_matvec`` / ``edge_grad``) against the ATen
reference.

The ATen kernels in ``btorch.sparse.runtime.kernels_aten`` define the
semantics of every backend. These tests feed both backends the same raw CSR
buffers and compare the results, so they exercise the Triton kernels directly
(the backend does not have to be registered for them to run).

The whole module is skipped unless a CUDA device and Triton are available.
Sizes are kept small enough for a 2 GB card.
"""

import functools
import gc

import numpy as np
import pytest
import torch

from btorch.sparse.runtime import kernels_aten, kernels_triton
from btorch.sparse.runtime.backend import BackendRegistry, KernelCache


pytestmark = pytest.mark.skipif(
    not kernels_triton.is_available(), reason="needs CUDA and triton"
)

DEVICE = "cuda"

# The two backends sum the products of a row in different orders, so float32
# results agree only up to rounding. The tolerance is relative to the largest
# reference magnitude because a hub row adds thousands of terms.
RTOL = 2e-5


def patterns_of(cache: KernelCache) -> dict:
    """The cached patterns (derived layouts per ``(crow, col)`` pair)."""
    return cache.get(kernels_triton._PATTERNS_KEY, dict)


@functools.cache
def make_csr(variant: str, seed: int = 0):
    """Random destination-major CSR ``(crow [M + 1], col [E], M, N)`` on GPU.

    Each graph is built once and kept alive for the whole session; tests
    share the buffers read-only, and the tests that rewire a graph in place
    work on clones. Keeping them alive also guarantees that two different
    graphs never occupy the same address, so nothing that memoises on a
    buffer address can confuse them.

    Every variant is non-square (``M != N``) so a transposed result cannot
    pass by accident.

    Args:
        variant: Row-length profile of the graph.

            - ``"plain"``: short random rows.
            - ``"empty_rows"``: as above, but the first, the last and every
              third row have no entries (also the rows around block edges).
            - ``"hub"``: one row with 5000 entries, far longer than any tile
              of the kernel (32 entries per step), next to short rows and an
              empty row. Columns repeat within the hub row because it is
              longer than ``N`` — duplicates must simply be summed.
            - ``"multi_hub"``: several long rows, including the first and the
              last row and lengths at the edges of the kernel's row splitting
              (rows longer than 512 entries are reduced in 512-entry segments
              whose partial sums are combined by a second launch).
            - ``"duplicates"``: every row lists the same column several times.
            - ``"single_row"``: ``M = 1``, fewer rows than one row block.
        seed: Seed of the NumPy generator.
    """
    rng = np.random.default_rng(seed)
    if variant == "single_row":
        m, n = 1, 37
        counts = np.array([23])
    else:
        m, n = 203, 157
        counts = rng.integers(1, 40, size=m)
        if variant == "empty_rows":
            counts[::3] = 0
            counts[[0, -1]] = 0
        elif variant == "hub":
            counts[77] = 5000
            counts[78] = 0
        elif variant == "multi_hub":
            counts[0] = 1024  # first row, an exact multiple of the segment
            counts[1] = 513  # one entry more than a segment
            counts[2] = 512  # exactly one segment: not split
            counts[100] = 3000
            counts[101] = 0
            counts[-1] = 2000  # last row
    cols = rng.integers(0, n, size=int(counts.sum()))
    if variant == "duplicates":
        # Only three distinct columns: almost every entry is a duplicate.
        cols = rng.integers(0, 3, size=int(counts.sum()))
    crow = np.concatenate([[0], np.cumsum(counts)])
    return (
        torch.as_tensor(crow, dtype=torch.long, device=DEVICE),
        torch.as_tensor(cols, dtype=torch.long, device=DEVICE),
        m,
        n,
    )


def assert_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Same shape, dtype and device; values equal up to float32 rounding."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    scale = float(expected.abs().max()) if expected.numel() else 0.0
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=RTOL * scale + 1e-6)


VARIANTS = ("plain", "empty_rows", "hub", "multi_hub", "duplicates", "single_row")

# Leading shapes of ``x`` in front of the neuron axis: 1-D, 2-D and 3-D
# inputs. A single sample uses the kernel's single-sample tile; 3 and 5
# samples a partly filled sample tile (16 samples per program); 2 * 20 = 40
# samples several sample tiles with a ragged last one.
SAMPLE_SHAPES = ((), (1,), (3,), (5,), (2, 20))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("sample", SAMPLE_SHAPES)
def test_csr_matvec_unbatched_values(variant, sample):
    # One value vector ``[E]`` shared by all samples: the plain SpMV / SpMM.
    crow, col, m, n = make_csr(variant)
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    values = torch.randn(col.shape[0], device=DEVICE, generator=gen)
    x = torch.randn(*sample, n, device=DEVICE, generator=gen)

    out = kernels_triton.csr_matvec(crow, col, values, x)

    assert out.shape == (*sample, m)
    assert out.is_contiguous()
    assert_matches(out, kernels_aten.csr_matvec(crow, col, values, x))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("sample", ((), (5,), (2, 3)))
def test_csr_matvec_batched_values(variant, sample):
    # ``values [G, E]``: G operators sharing one pattern, each applied to its
    # own slice ``x[g]``. Output is ``[G, *sample, M]``.
    crow, col, m, n = make_csr(variant)
    gen = torch.Generator(device=DEVICE).manual_seed(2)
    values = torch.randn(3, col.shape[0], device=DEVICE, generator=gen)
    x = torch.randn(3, *sample, n, device=DEVICE, generator=gen)

    out = kernels_triton.csr_matvec(crow, col, values, x)

    assert out.shape == (3, *sample, m)
    assert_matches(out, kernels_aten.csr_matvec(crow, col, values, x))


def test_csr_matvec_broadcast_batch():
    # Either side may have size-1 value-batch dimensions that broadcast
    # against the other. The kernel passes the shared side once (stride 0).
    crow, col, m, n = make_csr("hub")
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    values = torch.randn(3, col.shape[0], device=DEVICE, generator=gen)
    x = torch.randn(3, 4, n, device=DEVICE, generator=gen)

    # x shared by the three members.
    out = kernels_triton.csr_matvec(crow, col, values, x[:1])
    assert_matches(out, kernels_aten.csr_matvec(crow, col, values, x[:1]))
    # values shared by the three members.
    out = kernels_triton.csr_matvec(crow, col, values[:1], x)
    assert_matches(out, kernels_aten.csr_matvec(crow, col, values[:1], x))

    # Two value-batch dimensions; a partial broadcast ([2, 1] against [1, 3])
    # is delegated to the reference kernel and must still be correct.
    values2 = torch.randn(2, 1, col.shape[0], device=DEVICE, generator=gen)
    x2 = torch.randn(1, 3, 4, n, device=DEVICE, generator=gen)
    out = kernels_triton.csr_matvec(crow, col, values2, x2)
    assert out.shape == (2, 3, 4, m)
    assert_matches(out, kernels_aten.csr_matvec(crow, col, values2, x2))


def test_csr_matvec_non_contiguous_and_mixed_dtype():
    # Strided inputs are made contiguous internally; a bool spike tensor
    # promotes to float32 exactly like the reference does.
    crow, col, m, n = make_csr("plain")
    gen = torch.Generator(device=DEVICE).manual_seed(4)
    values = torch.randn(col.shape[0], device=DEVICE, generator=gen)
    wide = torch.randn(6, 2 * n, device=DEVICE, generator=gen)

    x = wide[:, ::2]
    assert not x.is_contiguous()
    assert_matches(
        kernels_triton.csr_matvec(crow, col, values, x),
        kernels_aten.csr_matvec(crow, col, values, x),
    )

    spikes = x > 0.5
    assert_matches(
        kernels_triton.csr_matvec(crow, col, values, spikes),
        kernels_aten.csr_matvec(crow, col, values, spikes),
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("sample", SAMPLE_SHAPES)
def test_edge_grad_unbatched(variant, sample):
    # ``out[k] = sum_s grad[s, row[k]] * x[s, col[k]]`` with ``n_vb = 0``:
    # every leading dimension of grad / x is a sample dimension and is
    # summed away.
    crow, col, m, n = make_csr(variant)
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    grad = torch.randn(*sample, m, device=DEVICE, generator=gen)
    x = torch.randn(*sample, n, device=DEVICE, generator=gen)

    out = kernels_triton.edge_grad(crow, col, grad, x, 0)

    assert out.shape == (col.shape[0],)
    assert_matches(out, kernels_aten.edge_grad(crow, col, grad, x, 0))


@pytest.mark.parametrize("variant", ("plain", "empty_rows", "hub", "multi_hub"))
@pytest.mark.parametrize("sample", ((), (4,), (2, 3)))
def test_edge_grad_batched(variant, sample):
    # ``n_vb = 1``: the leading dimension indexes value-batch members and is
    # kept, giving one gradient vector per member, ``[G, E]``.
    crow, col, m, n = make_csr(variant)
    gen = torch.Generator(device=DEVICE).manual_seed(6)
    grad = torch.randn(3, *sample, m, device=DEVICE, generator=gen)
    x = torch.randn(3, *sample, n, device=DEVICE, generator=gen)

    out = kernels_triton.edge_grad(crow, col, grad, x, 1)
    assert out.shape == (3, col.shape[0])
    assert_matches(out, kernels_aten.edge_grad(crow, col, grad, x, 1))

    # The forward input may be shared by all members (size-1 batch dim).
    out = kernels_triton.edge_grad(crow, col, grad, x[:1], 1)
    assert_matches(out, kernels_aten.edge_grad(crow, col, grad, x[:1], 1))


def test_edge_grad_matches_autograd():
    # Independent check that does not involve the ATen kernel: the gradient
    # of sum(y * g) w.r.t. the values of a dense-indexed product.
    crow, col, m, n = make_csr("hub")
    row = torch.repeat_interleave(torch.arange(m, device=DEVICE), crow[1:] - crow[:-1])
    gen = torch.Generator(device=DEVICE).manual_seed(7)
    values = torch.randn(col.shape[0], device=DEVICE, generator=gen)
    values.requires_grad_()
    x = torch.randn(4, n, device=DEVICE, generator=gen)
    g = torch.randn(4, m, device=DEVICE, generator=gen)

    y = torch.zeros(4, m, device=DEVICE).index_add(1, row, x[:, col] * values)
    (expected,) = torch.autograd.grad((y * g).sum(), values)

    assert_matches(kernels_triton.edge_grad(crow, col, g, x, 0), expected)


@pytest.mark.parametrize(
    "m, n, n_edge, sample",
    [
        (0, 7, 0, (3,)),  # no rows
        (5, 7, 0, (3,)),  # rows but no entries
        (5, 7, 9, (0,)),  # no samples
        (5, 0, 0, ()),  # no inputs
    ],
)
def test_zero_size_inputs(m, n, n_edge, sample):
    # Degenerate shapes must give correctly shaped (all-zero) results rather
    # than launching an empty grid.
    gen = torch.Generator(device=DEVICE).manual_seed(8)
    counts = torch.zeros(m, dtype=torch.long, device=DEVICE)
    if n_edge:
        counts[0] = n_edge
    crow = torch.zeros(m + 1, dtype=torch.long, device=DEVICE)
    crow[1:] = torch.cumsum(counts, 0)
    col = torch.randint(0, max(n, 1), (n_edge,), device=DEVICE, generator=gen)
    values = torch.randn(n_edge, device=DEVICE, generator=gen)
    x = torch.randn(*sample, n, device=DEVICE, generator=gen)
    grad = torch.randn(*sample, m, device=DEVICE, generator=gen)

    # The expected results are written out explicitly (zeros of the contract
    # shape) instead of taken from the ATen kernels: with nothing to sum the
    # answer is known, and the reference ``edge_grad`` itself cannot reshape
    # a gradient with ``M == 0``.
    out = kernels_triton.csr_matvec(crow, col, values, x)
    assert_matches(out, torch.zeros(*sample, m, device=DEVICE))
    if n_edge == 0 or 0 in sample:
        assert not out.any()

    out = kernels_triton.edge_grad(crow, col, grad, x, 0)
    assert_matches(out, torch.zeros(n_edge, device=DEVICE))

    # Zero value-batch members: a leading dimension of size 0 on every
    # batched argument.
    x0 = x.expand(0, *x.shape)
    out = kernels_triton.csr_matvec(crow, col, values.expand(0, -1), x0)
    assert out.shape == (0, *sample, m)
    out = kernels_triton.edge_grad(crow, col, grad.expand(0, *grad.shape), x0, 1)
    assert out.shape == (0, n_edge)


@pytest.mark.parametrize("variant", ("hub", "multi_hub"))
@pytest.mark.parametrize("sample", ((), (7,), (40,)))
def test_deterministic(variant, sample):
    # One writer per output and a fixed reduction order: repeated runs must
    # be bitwise identical, also for the 5000-entry hub row whose sum is the
    # most sensitive to ordering.
    crow, col, m, n = make_csr(variant)
    gen = torch.Generator(device=DEVICE).manual_seed(9)
    values = torch.randn(col.shape[0], device=DEVICE, generator=gen)
    x = torch.randn(*sample, n, device=DEVICE, generator=gen)
    grad = torch.randn(*sample, m, device=DEVICE, generator=gen)

    first = kernels_triton.csr_matvec(crow, col, values, x)
    first_grad = kernels_triton.edge_grad(crow, col, grad, x, 0)
    for _ in range(5):
        assert torch.equal(kernels_triton.csr_matvec(crow, col, values, x), first)
        assert torch.equal(kernels_triton.edge_grad(crow, col, grad, x, 0), first_grad)


def test_fallback_dtypes_and_cpu():
    # Anything the Triton kernels do not implement is delegated to the ATen
    # kernel: float64 keeps its dtype, CPU tensors stay on the CPU.
    crow, col, m, n = make_csr("plain")
    gen = torch.Generator(device=DEVICE).manual_seed(10)
    values = torch.randn(col.shape[0], device=DEVICE, generator=gen)
    x = torch.randn(3, n, device=DEVICE, generator=gen)
    grad = torch.randn(3, m, device=DEVICE, generator=gen)

    out = kernels_triton.csr_matvec(crow, col, values.double(), x)
    assert out.dtype == torch.float64
    assert_matches(out, kernels_aten.csr_matvec(crow, col, values.double(), x))

    out = kernels_triton.edge_grad(crow, col, grad.double(), x, 0)
    assert out.dtype == torch.float64
    assert_matches(out, kernels_aten.edge_grad(crow, col, grad.double(), x, 0))

    cpu = [t.cpu() for t in (crow, col, values, x, grad)]
    out = kernels_triton.csr_matvec(*cpu[:4])
    assert out.device.type == "cpu"
    assert_matches(out, kernels_aten.csr_matvec(*cpu[:4]))
    out = kernels_triton.edge_grad(cpu[0], cpu[1], cpu[4], cpu[3], 0)
    assert out.device.type == "cpu"


def test_register_and_layout_cache_invalidation():
    # ``register`` adds the backend above the ATen one (priority 0) and binds
    # the derived layouts (int32 index copies, expanded row index) to the
    # registry's kernel cache.
    reg = BackendRegistry()
    reg.register("csr_matvec", "aten", kernels_aten.csr_matvec)
    reg.register("edge_grad", "aten", kernels_aten.edge_grad)
    kernels_triton.register(reg)
    assert reg.name("csr_matvec", "cuda") == "triton"
    assert reg.name("edge_grad", "cuda") == "triton"
    # Not registered for the CPU.
    assert reg.name("csr_matvec", "cpu") == "aten"
    matvec = reg.resolve("csr_matvec", "cuda")
    edge_grad = reg.resolve("edge_grad", "cuda")

    crow, col, m, n = make_csr("plain", seed=0)
    crow, col = crow.clone(), col.clone()  # rewired in place below
    gen = torch.Generator(device=DEVICE).manual_seed(11)
    values = torch.randn(col.shape[0], device=DEVICE, generator=gen)
    x = torch.randn(3, n, device=DEVICE, generator=gen)
    grad = torch.randn(3, m, device=DEVICE, generator=gen)

    def check():
        assert_matches(
            matvec(crow, col, values, x),
            kernels_aten.csr_matvec(crow, col, values, x),
        )
        assert_matches(
            edge_grad(crow, col, grad, x, 0),
            kernels_aten.edge_grad(crow, col, grad, x, 0),
        )

    def n_layouts():
        return len(patterns_of(reg.kernels))

    check()
    # One pattern for the ``(crow, col)`` pair holds everything both kernels
    # derive from it (row pointers, expanded row index, int32 ``col``). A
    # second call reuses it: the very same derived tensors are read again.
    assert n_layouts() == 1
    (pattern,) = patterns_of(reg.kernels).values()
    derived = (pattern.rows, pattern.row32, pattern.col32)
    assert all(d is not None for d in derived)
    check()
    assert n_layouts() == 1
    (again,) = patterns_of(reg.kernels).values()
    assert again is pattern
    assert (again.rows, again.row32, again.col32) == derived

    # Rewiring rewrites the index buffers IN PLACE (same tensor objects, same
    # addresses, new contents) — exactly what ``RepresentationCache.build``
    # does when the number of edges is unchanged. A stale layout would
    # silently compute with the old graph, so the results must follow the new
    # contents, and the cache must not grow.
    col.copy_(torch.randint(0, n, col.shape, device=DEVICE, generator=gen))
    check()
    # Move entries between rows: same E, different row pointers.
    counts = (crow[1:] - crow[:-1]).flip(0)
    crow[1:] = torch.cumsum(counts, 0)
    check()
    assert n_layouts() == 1

    # A different number of edges replaces the buffers by new tensors.
    crow, col, m, n = make_csr("hub", seed=1)
    values = torch.randn(col.shape[0], device=DEVICE, generator=gen)
    check()


def test_layout_cache_releases_dead_buffers():
    # The cache holds only a weak reference to the buffer a layout was built
    # from. When the buffer dies its layout is dropped, so a new tensor that
    # happens to be allocated at the same address is never served the stale
    # layout and no GPU memory is pinned by dead graphs.
    cache = KernelCache()
    crow, col, m, n = make_csr("plain")
    crow, col = crow.clone(), col.clone()  # private buffers that can die
    values = torch.randn(col.shape[0], device=DEVICE)
    x = torch.randn(n, device=DEVICE)
    kernels_triton._csr_matvec(cache, crow, col, values, x)
    assert len(patterns_of(cache)) == 1

    # The entry itself goes away, not only its tensors: the cache does not
    # accumulate keys of dead graphs. One dead buffer of the pair is enough.
    del col
    assert len(patterns_of(cache)) == 0
    del crow


def test_layout_cache_is_bounded_and_evicts_dead_patterns(monkeypatch):
    # Regression: 200 transient patterns used to leave ~200 dead entries
    # behind. Entries are deleted when their buffers die; buffers that stay
    # alive are bounded by ``_MAX_PATTERNS`` (oldest dropped first), and an
    # evicted pattern is simply rebuilt on its next use.
    cache = KernelCache()
    crow, col, m, n = make_csr("plain")
    values = torch.randn(col.shape[0], device=DEVICE)
    x = torch.randn(n, device=DEVICE)
    expected = kernels_aten.csr_matvec(crow, col, values, x)

    for _ in range(200):
        transient = col.clone()
        kernels_triton._csr_matvec(cache, crow, transient, values, x)
        del transient
    gc.collect()
    assert len(patterns_of(cache)) == 0

    monkeypatch.setattr(kernels_triton, "_MAX_PATTERNS", 4)
    alive = [col.clone() for _ in range(10)]
    for c in alive:
        assert_matches(kernels_triton._csr_matvec(cache, crow, c, values, x), expected)
        assert len(patterns_of(cache)) <= 4
    # The first buffers were evicted while alive; they still work.
    assert (id(crow), id(alive[0])) not in patterns_of(cache)
    assert_matches(
        kernels_triton._csr_matvec(cache, crow, alive[0], values, x), expected
    )
    assert len(patterns_of(cache)) <= 4


def test_views_sharing_an_address_are_not_confused():
    # Regression: two views of one buffer can have the same address, shape
    # and version counter but different strides (``base[:k]`` against
    # ``base[::2]``). A layout cache keyed on those properties served the
    # first view's layout to the second, i.e. computed with the wrong graph.
    #
    # The expected values are written out with plain indexing: the CUDA CSR
    # product behind the ATen kernel reads strided index buffers as if they
    # were contiguous, so it cannot serve as the reference here.
    cache = KernelCache()
    m, k, n = 40, 200, 50
    gen = torch.Generator(device=DEVICE).manual_seed(20)
    x = torch.randn(3, n, device=DEVICE, generator=gen)
    grad = torch.randn(3, m, device=DEVICE, generator=gen)

    def check(crow, col):
        values = torch.randn(col.shape[0], device=DEVICE, generator=gen)
        n_rows = crow.shape[0] - 1
        row = torch.repeat_interleave(
            torch.arange(n_rows, device=DEVICE), crow[1:] - crow[:-1]
        )
        expected = torch.zeros(3, n_rows, device=DEVICE).index_add(
            1, row, x[:, col] * values
        )
        assert_matches(
            kernels_triton._csr_matvec(cache, crow, col, values, x), expected
        )
        expected = (grad[:, row] * x[:, col]).sum(0)
        assert_matches(
            kernels_triton._edge_grad(cache, crow, col, grad, x, 0), expected
        )

    # Column indices: the first half of a buffer against every second entry.
    base = torch.randint(0, n, (2 * k,), device=DEVICE, generator=gen)
    crow = torch.arange(0, k + 1, k // m, device=DEVICE)
    head, strided = base[:k], base[::2]
    assert head.data_ptr() == strided.data_ptr() and head.shape == strided.shape
    assert head._version == strided._version
    assert not torch.equal(head, strided)
    for col in (head, strided, head):
        check(crow, col)

    # Row pointers: the first pointers of a longer array (rows of 5 entries)
    # against every second pointer (rows of 10 entries).
    wide = torch.arange(0, 2 * k + 1, k // m, device=DEVICE)
    first, merged = wide[: m + 1], wide[::2]
    assert merged.data_ptr() == first.data_ptr() and merged.shape == first.shape
    for ptr, cols in ((first, base[:k]), (merged, base), (first, base[:k])):
        check(ptr, cols)


def test_too_few_dimensions_raise_like_the_reference():
    # Regression: ``values [G, E]`` needs ``x [G, ..., N]``. With ``x [G]``
    # (and N == G) the shapes still "broadcast", and the kernel used to
    # return ``[G, M]`` read from beyond the input buffer. The rank is now
    # checked first and the reference kernel raises its error.
    crow, col, m, n = make_csr("plain")
    values = torch.randn(3, col.shape[0], device=DEVICE)
    x = torch.randn(3, device=DEVICE)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        kernels_aten.csr_matvec(crow, col, values, x)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        kernels_triton.csr_matvec(crow, col, values, x)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        kernels_triton.csr_matvec_gather(
            crow, col, torch.arange(col.shape[0], device=DEVICE), values, x
        )
    # ``edge_grad``: operands without the value-batch dimensions never reach
    # the kernel either (whatever the reference does with them).
    grad = torch.randn(3, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        kernels_triton.edge_grad(crow, col, grad, x, 1)


def test_mismatched_sizes_never_reach_the_kernel(monkeypatch):
    # The kernels index their operands without bounds checks, so operands
    # whose sizes do not match the pattern are left to the reference kernel
    # instead of being read out of bounds. The reference is replaced by a
    # marker and the launcher by a tripwire to observe the routing only.
    crow, col, m, n = make_csr("plain")
    x = torch.randn(n, device=DEVICE)

    def tripwire(*args, **kwargs):
        raise AssertionError("the Triton kernel was launched")

    monkeypatch.setattr(kernels_triton, "_launch", tripwire)
    monkeypatch.setattr(kernels_aten, "csr_matvec", lambda *a: "reference")
    monkeypatch.setattr(kernels_aten, "edge_grad", lambda *a: "reference")

    short = torch.randn(col.shape[0] - 1, device=DEVICE)  # one value missing
    assert kernels_triton.csr_matvec(crow, col, short, x) == "reference"
    grad = torch.randn(m - 1, device=DEVICE)  # one output missing
    assert kernels_triton.edge_grad(crow, col, grad, x, 0) == "reference"
    # A permutation of the wrong length.
    perm = torch.arange(col.shape[0] - 1, device=DEVICE)
    values = torch.randn(col.shape[0], device=DEVICE)
    out = kernels_triton.csr_matvec_gather(crow, col, perm, values, x)
    assert out == "reference"


def test_every_argument_must_be_on_the_kernel_device():
    # All four tensors are checked, not only the last one: a CPU operand
    # mixed with CUDA buffers is the reference kernel's error, never a launch
    # on a host pointer.
    crow, col, m, n = make_csr("plain")
    values = torch.randn(col.shape[0], device=DEVICE)
    x = torch.randn(2, n, device=DEVICE)
    grad = torch.randn(2, m, device=DEVICE)
    for args in (
        (crow.cpu(), col, values, x),
        (crow, col.cpu(), values, x),
        (crow, col, values.cpu(), x),
        (crow, col, values, x.cpu()),
    ):
        with pytest.raises((RuntimeError, ValueError, TypeError)):
            kernels_triton.csr_matvec(*args)
    for args in ((crow, col, grad.cpu(), x), (crow, col, grad, x.cpu())):
        with pytest.raises((RuntimeError, ValueError, TypeError)):
            kernels_triton.edge_grad(*args, 0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_buffers_on_another_gpu_fall_back():
    # Triton launches on the current device; tensors of another GPU are
    # handled by the reference kernel on their own device.
    crow, col, m, n = make_csr("plain")
    other = [t.to("cuda:1") for t in (crow, col)]
    values = torch.randn(col.shape[0], device="cuda:1")
    x = torch.randn(2, n, device="cuda:1")
    out = kernels_triton.csr_matvec(*other, values, x)
    assert out.device == x.device
    assert_matches(out, kernels_aten.csr_matvec(*other, values, x))


def test_layout_cache_survives_address_reuse():
    # Graphs that are created and dropped in a loop tend to land on the same
    # GPU address with the same shape and a fresh version counter. A cache
    # keyed on (address, version, shape) alone would then serve the layout of
    # the previous graph. Each iteration builds a *new* graph (bypassing the
    # session memo of ``make_csr``) and checks the result against a
    # reference written out here with plain indexing.
    cache = KernelCache()
    gen = torch.Generator(device=DEVICE).manual_seed(12)
    addresses = set()
    for seed in range(6):
        crow, col, m, n = make_csr.__wrapped__("empty_rows", seed=seed)
        addresses.add((crow.data_ptr(), col.data_ptr()))
        row = torch.repeat_interleave(
            torch.arange(m, device=DEVICE), crow[1:] - crow[:-1]
        )
        values = torch.randn(col.shape[0], device=DEVICE, generator=gen)
        x = torch.randn(3, n, device=DEVICE, generator=gen)
        grad = torch.randn(3, m, device=DEVICE, generator=gen)

        expected = torch.zeros(3, m, device=DEVICE).index_add(
            1, row, x[:, col] * values
        )
        assert_matches(
            kernels_triton._csr_matvec(cache, crow, col, values, x), expected
        )
        expected = (grad[:, row] * x[:, col]).sum(0)
        assert_matches(
            kernels_triton._edge_grad(cache, crow, col, grad, x, 0), expected
        )
        del crow, col, row
    # Not an assertion of the test, but the scenario it is about: on CUDA the
    # freed buffers are normally reused, giving fewer addresses than graphs.
    assert len(addresses) >= 1


@pytest.mark.parametrize("variant", ("plain", "multi_hub"))
def test_direct_launch_matches_public_launch(variant, monkeypatch):
    # With a known Triton version the kernels are launched through a cached
    # compiled handle, skipping Triton's per-call argument specialisation.
    # Both launch paths must run the same binary: results are bitwise equal.
    crow, col, m, n = make_csr(variant)
    gen = torch.Generator(device=DEVICE).manual_seed(13)
    values = torch.randn(col.shape[0], device=DEVICE, generator=gen)
    x = torch.randn(5, n, device=DEVICE, generator=gen)
    grad = torch.randn(5, m, device=DEVICE, generator=gen)

    def run():
        # Called twice: the first call may compile, the second one takes the
        # steady-state path.
        for _ in range(2):
            out = (
                kernels_triton.csr_matvec(crow, col, values, x),
                kernels_triton.edge_grad(crow, col, grad, x, 0),
            )
        return out

    direct = run()
    monkeypatch.setattr(kernels_triton, "_DIRECT_LAUNCH", False)
    public = run()
    for a, b in zip(direct, public):
        assert torch.equal(a, b)


def test_inference_mode_buffers():
    # Tensors created under ``torch.inference_mode`` have no version counter,
    # so their derived layouts cannot be validated and are rebuilt on every
    # call instead of cached. The results must be unaffected.
    crow, col, m, n = make_csr("multi_hub")
    cache = KernelCache()
    with torch.inference_mode():
        crow, col = crow.clone(), col.clone()
        values = torch.randn(col.shape[0], device=DEVICE)
        x = torch.randn(3, n, device=DEVICE)
        grad = torch.randn(3, m, device=DEVICE)
        for _ in range(2):
            out = kernels_triton._csr_matvec(cache, crow, col, values, x)
            out_grad = kernels_triton._edge_grad(cache, crow, col, grad, x, 0)
        assert_matches(out, kernels_aten.csr_matvec(crow, col, values, x))
        row = torch.repeat_interleave(
            torch.arange(m, device=DEVICE), crow[1:] - crow[:-1]
        )
        assert_matches(out_grad, (grad[:, row] * x[:, col]).sum(0))
    # No pattern was cached for the version-less buffers.
    assert len(patterns_of(cache)) == 0


@pytest.mark.parametrize("variant", ("plain", "empty_rows", "multi_hub"))
@pytest.mark.parametrize("sample", ((), (5,), (2, 20)))
@pytest.mark.parametrize("batched", (False, True))
def test_csr_matvec_gather(variant, sample, batched):
    # ``csr_matvec_gather(crow, col, perm, values, x)`` is ``csr_matvec`` on
    # ``values[..., perm]``: the kernel reads the values through ``perm``
    # instead of from a gathered copy (the transposed product of the
    # backward, where ``perm`` maps source-major entries to the
    # destination-major values). Same reduction order, so the results are
    # bitwise equal to the two-step computation.
    crow, col, m, n = make_csr(variant)
    n_edge = col.shape[0]
    gen = torch.Generator(device=DEVICE).manual_seed(14)
    perm = torch.randperm(n_edge, device=DEVICE, generator=gen)
    lead = (3,) if batched else ()
    values = torch.randn(*lead, n_edge, device=DEVICE, generator=gen)
    x = torch.randn(*lead, *sample, n, device=DEVICE, generator=gen)

    out = kernels_triton.csr_matvec_gather(crow, col, perm, values, x)

    gathered = values.index_select(-1, perm)
    assert torch.equal(out, kernels_triton.csr_matvec(crow, col, gathered, x))
    assert_matches(out, kernels_aten.csr_matvec(crow, col, gathered, x))


def test_csr_matvec_gather_follows_the_permutation_buffer():
    # The int32 copy of ``perm`` is cached with the pattern and validated by
    # identity and version, like the pattern itself: an in-place rewrite and
    # a different permutation tensor must both be picked up. A strided view
    # of a permutation is copied correctly, and unsupported inputs
    # (float64) fall back to the reference.
    cache = KernelCache()
    crow, col, m, n = make_csr("plain")
    n_edge = col.shape[0]
    gen = torch.Generator(device=DEVICE).manual_seed(15)
    values = torch.randn(n_edge, device=DEVICE, generator=gen)
    x = torch.randn(4, n, device=DEVICE, generator=gen)

    def check(perm, values=values):
        out = kernels_triton._csr_matvec(cache, crow, col, values, x, perm)
        expected = kernels_aten.csr_matvec(crow, col, values[perm], x)
        assert_matches(out, expected)

    perm = torch.randperm(n_edge, device=DEVICE, generator=gen)
    check(perm)
    perm.copy_(torch.randperm(n_edge, device=DEVICE, generator=gen))  # in place
    check(perm)
    check(torch.randperm(n_edge, device=DEVICE, generator=gen))  # new tensor
    double = torch.randperm(2 * n_edge, device=DEVICE, generator=gen)
    check(double[::2] // 2)
    check(perm, values.double())


@pytest.mark.parametrize("variant", ("plain", "multi_hub"))
@pytest.mark.parametrize("sample", ((), (5,), (2, 20)))
@pytest.mark.parametrize("batched", (False, True))
def test_csr_backward_equals_the_separate_kernels(variant, sample, batched):
    # ``csr_backward`` returns both gradients of ``y = A x``: the per-entry
    # gradient (``edge_grad``) and the transposed product over the
    # source-major CSR of the same entries, read through ``t_perm``. It only
    # shares work between the two kernels (one transposed copy of ``grad``),
    # so each result is bitwise the separate kernel's.
    crow, col, m, n = make_csr(variant)
    n_edge = col.shape[0]
    gen = torch.Generator(device=DEVICE).manual_seed(17)
    # Source-major CSR of the same entries: sort the entries by column.
    row = torch.repeat_interleave(torch.arange(m, device=DEVICE), crow[1:] - crow[:-1])
    t_perm = torch.argsort(col * m + row, stable=True)
    t_col = row[t_perm]
    t_crow = torch.zeros(n + 1, dtype=torch.long, device=DEVICE)
    t_crow[1:] = torch.cumsum(torch.bincount(col, minlength=n), 0)

    lead = (3,) if batched else ()
    values = torch.randn(*lead, n_edge, device=DEVICE, generator=gen)
    x = torch.randn(*lead, *sample, n, device=DEVICE, generator=gen)
    grad = torch.randn(*lead, *sample, m, device=DEVICE, generator=gen)
    buffers = (crow, col, t_crow, t_col, t_perm)

    grad_values, grad_x = kernels_triton.csr_backward(*buffers, values, x, grad)

    n_vb = len(lead)
    assert torch.equal(grad_values, kernels_triton.edge_grad(crow, col, grad, x, n_vb))
    gathered = values.index_select(-1, t_perm)
    assert torch.equal(grad_x, kernels_triton.csr_matvec(t_crow, t_col, gathered, grad))
    # Against autograd through a dense-indexed product (no kernel involved).
    v, xr = values.clone().requires_grad_(), x.clone().requires_grad_()
    y = torch.zeros(*lead, *sample, m, device=DEVICE).index_add(
        -1, row, xr[..., col] * v.reshape(*lead, *(1,) * len(sample), n_edge)
    )
    expected_values, expected_x = torch.autograd.grad((y * grad).sum(), (v, xr))
    assert_matches(grad_values, expected_values)
    assert_matches(grad_x, expected_x)

    # A gradient that is not requested is not computed.
    only_x = kernels_triton.csr_backward(*buffers, values, x, grad, need_values=False)
    assert only_x[0] is None and torch.equal(only_x[1], grad_x)
    only_values = kernels_triton.csr_backward(*buffers, values, x, grad, need_x=False)
    assert only_values[1] is None and torch.equal(only_values[0], grad_values)


@pytest.mark.parametrize("batched", (False, True))
def test_fused_backward_operator(batched):
    # ``btorch::csr_propagate_backward`` is the registered operator compiled
    # graphs call for the backward of a propagation: both gradients behind
    # one operator boundary. It must satisfy the operator contract
    # (``opcheck``: schema, fake kernel, AOT dispatch) and return exactly
    # what autograd through ``csr_propagate`` returns. It is not
    # differentiable itself, so ``opcheck`` gets inputs that do not require
    # gradients, like the other raw operators.
    from btorch.sparse.runtime import RepresentationCache, ops

    m, n, n_edge = 23, 17, 120
    gen = torch.Generator().manual_seed(18)
    row = torch.randint(0, m, (n_edge,), generator=gen)
    col = torch.randint(0, n, (n_edge,), generator=gen)
    cache = RepresentationCache()
    cache.build(row, col, (m, n), version=0)
    cache = cache.to(DEVICE)
    n_edge = cache.col.shape[0]
    forward = (cache.crow, cache.col)
    transposed = (cache.t_crow, cache.t_col, cache.t_perm)

    lead = (2,) if batched else ()
    gen = torch.Generator(device=DEVICE).manual_seed(19)
    values = torch.randn(*lead, n_edge, device=DEVICE, generator=gen)
    x = torch.randn(*lead, 5, n, device=DEVICE, generator=gen)
    grad = torch.randn(*lead, 5, m, device=DEVICE, generator=gen)

    for need_values, need_x in ((True, True), (True, False), (False, True)):
        args = (*forward, values, x, *transposed, grad, need_values, need_x)
        torch.library.opcheck(ops.csr_propagate_backward, args)
        grad_values, grad_x = ops.csr_propagate_backward(*args)
        # A gradient that is not needed is an empty placeholder.
        assert grad_values.shape == (values.shape if need_values else (0,))
        assert grad_x.shape == (x.shape if need_x else (0,))

    v, xr = values.clone().requires_grad_(), x.clone().requires_grad_()
    y = ops.csr_propagate(*forward, v, xr, *transposed)
    expected_values, expected_x = torch.autograd.grad((y * grad).sum(), (v, xr))
    args = (*forward, values, x, *transposed, grad, True, True)
    grad_values, grad_x = ops.csr_propagate_backward(*args)
    assert torch.equal(grad_values, expected_values)
    assert torch.equal(grad_x, expected_x)
