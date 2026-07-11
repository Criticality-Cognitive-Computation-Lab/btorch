"""Activity-match tests for the cudagraph-vs-persistent comparison providers.

Every provider in ``benchmark_rsnn_cudagraph_compare`` must reproduce the dense
reference's *spiking activity*, not merely run fast. These tests assert, for
each provider, that (a) the reference produces substantial activity (so a match
is non-trivial, not "both silent") and (b) the provider's spikes match the dense
reference to within a tight tolerance. The two event-driven hand-CUDA providers
(``persistent_prespan_cuda`` cooperative, ``cudagraph_prespan_cuda`` graph) may
differ from the dense reference by a handful of near-threshold spikes at large T
due to atomicAdd fan-out summation order (a deterministic float effect, see the
report in results/persistent_kernel/); the fractional tolerance below covers
that while still catching any real activity mismatch.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_compare_module():
    path = REPO_ROOT / "benchmark" / "benchmark_rsnn_cudagraph_compare.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_rsnn_cudagraph_compare", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cmp = _load_compare_module()

# Min fraction of (T*B*N) entries that must spike in the reference for the match
# to be meaningful, and the max fraction of spike entries allowed to differ.
MIN_ACTIVITY = 1e-3
MAX_SPIKE_MISMATCH_FRAC = 2e-4

# Extension-build / capability failures that should skip rather than fail.
_SKIPPABLE = (
    "cuda_home",
    "nvcc",
    "ninja",
    "cooperative",
    "no kernel image",
    "unsupported",
    "cuda_runtime",
    "libcudart",
    "triton",
)


def _case(t_steps: int = 16):
    return cmp.BenchCase(
        n_neuron=1024,
        batch_size=2,
        t_steps=t_steps,
        fanout=8,
        event_rate=0.05,
        input_amplitude=30.0,
    )


def _reference(case, device):
    x_seq = cmp.make_input_sequence(case, device)
    matrix = cmp.make_recurrent_csr(case, device)
    ref = cmp.dense_rsnn_forward(x_seq, cmp.csr_to_dense(matrix), case)
    return x_seq, matrix, ref


def _run_provider(name, x_seq, matrix, case, ref):
    """Dispatch a provider by name to its spike output (or skip on build
    fail)."""
    max_events = cmp._max_events_for_case(case, ref)
    try:
        if name == "eager_native_sparse":
            return cmp.run_eager_native_sparse(x_seq, matrix, case).spikes
        if name == "cudagraph_native_sparse":
            return cmp.CUDAGraphNativeSparseProvider()(x_seq, matrix, case).spikes
        if name == "eager_prespan":
            return cmp.run_eager_prespan(
                x_seq, matrix, case, max_events=max_events
            ).spikes
        if name == "cudagraph_prespan":
            return cmp.CUDAGraphPreSpanProvider()(
                x_seq, matrix, case, max_events=max_events
            ).spikes
        if name == "cudagraph_native_sparse_chunked":
            return (
                cmp.ChunkedCUDAGraphNativeSparseProvider()
                .run_full(x_seq, matrix, case, chunk_size=8)
                .spikes
            )
        if name == "cudagraph_prespan_chunked":
            return (
                cmp.ChunkedCUDAGraphPreSpanProvider()
                .run_full(x_seq, matrix, case, chunk_size=8, max_events=max_events)
                .spikes
            )
        if name == "persistent_prespan_cuda":
            return cmp.run_persistent(
                x_seq, matrix, case, backend="cuda_persistent"
            ).spikes
        if name == "cudagraph_prespan_cuda":
            return cmp.CUDAGraphPreSpanCudaProvider()(x_seq, matrix, case).spikes
        raise ValueError(name)
    except (RuntimeError, OSError) as exc:
        if any(tok in str(exc).lower() for tok in _SKIPPABLE):
            pytest.skip(f"{name} unavailable: {exc}")
        raise


PROVIDERS = [
    "eager_native_sparse",
    "cudagraph_native_sparse",
    "cudagraph_native_sparse_chunked",
    "eager_prespan",
    "cudagraph_prespan",
    "cudagraph_prespan_chunked",
    "persistent_prespan_cuda",
    "cudagraph_prespan_cuda",
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("provider", PROVIDERS)
def test_provider_activity_matches_dense_reference(provider):
    """Each provider reproduces the dense reference's spiking activity."""
    device = torch.device("cuda")
    case = _case()
    x_seq, matrix, ref = _reference(case, device)

    total = int(ref.spikes.numel())
    ref_spikes = int(ref.spikes.sum().item())
    # Non-trivial activity: guards against a "both silent" degenerate pass.
    assert (
        ref_spikes > MIN_ACTIVITY * total
    ), f"reference activity too low ({ref_spikes}/{total}) to be a real test"

    spikes = _run_provider(provider, x_seq, matrix, case, ref)
    assert spikes.shape == ref.spikes.shape
    mismatch = int((spikes != ref.spikes).sum().item())
    frac = mismatch / total
    assert frac < MAX_SPIKE_MISMATCH_FRAC, (
        f"{provider}: {mismatch}/{total} spike entries differ from dense "
        f"reference (frac={frac:.2e} >= {MAX_SPIKE_MISMATCH_FRAC:.0e})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_same_code_pair_matches_each_other():
    """persistent_prespan_cuda and cudagraph_prespan_cuda are the same phase
    code (cooperative vs graph dispatch); their spike outputs must agree."""
    device = torch.device("cuda")
    case = _case()
    x_seq, matrix, ref = _reference(case, device)

    coop = _run_provider("persistent_prespan_cuda", x_seq, matrix, case, ref)
    graph = _run_provider("cudagraph_prespan_cuda", x_seq, matrix, case, ref)
    mismatch = int((coop != graph).sum().item())
    frac = mismatch / coop.numel()
    assert (
        frac < MAX_SPIKE_MISMATCH_FRAC
    ), f"same-code pair disagree: {mismatch}/{coop.numel()} (frac={frac:.2e})"
