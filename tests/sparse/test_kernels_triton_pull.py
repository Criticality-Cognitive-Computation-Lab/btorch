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
        # Layout slots are keyed ("triton", kind, address, shape, ...); the
        # cache may also hold compiled-kernel handles, which are not counted.
        return sum(len(key) > 2 for key in reg.kernels._store)

    check()
    # Three layouts: row pointers and expanded row index (both derived from
    # ``crow``) and the int32 copy of ``col``. A second call reuses them.
    assert n_layouts() == 3
    check()
    assert n_layouts() == 3

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
    assert n_layouts() == 3

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
    # Layout slots only (the cache also holds compiled-kernel handles).
    slots = [v for k, v in cache._store.items() if len(k) > 2]
    assert len(slots) == 2  # row pointers of ``crow``, int32 ``col``
    assert all(slot.payload is not None for slot in slots)

    del crow, col
    assert all(slot.payload is None for slot in slots)


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
    # No layout slot was created for the version-less buffers.
    assert not [key for key in cache._store if len(key) > 2]
