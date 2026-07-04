import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "benchmark" / "benchmark_persistent_snn.py"
    spec = importlib.util.spec_from_file_location("benchmark_persistent_snn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


def _small_case():
    return bench.BenchCase(
        n_neuron=8,
        batch_size=2,
        t_steps=5,
        fanout=3,
        event_rate=0.25,
        input_amplitude=30.0,
    )


def test_cpu_benchmark_skips_unavailable_event_and_compile_providers():
    """CPU smoke test: unavailable providers should produce skipped CSV rows.

    This keeps the benchmark runnable on Windows or CPU-only machines while
    preserving the CUDA/Triton event providers for real performance runs.
    """

    case = _small_case()
    rows = bench.bench_case(
        case,
        device=torch.device("cpu"),
        providers=("event_pre_span", "event_post_span", "torch_compile_dense", "persistent"),
        persistent_backend="torch_stub",
        warmup=1,
        repeat=1,
        skip_correctness=False,
    )

    by_provider = {row["provider"]: row for row in rows}
    assert by_provider["event_pre_span"]["correctness_status"] == "skipped:requires_cuda"
    assert by_provider["event_post_span"]["correctness_status"] == "skipped:requires_cuda"
    assert by_provider["persistent"]["correctness_status"] == "stub_only"
    assert by_provider["persistent"]["latency_ms"] >= 0.0


def test_dense_reference_uses_same_csr_weights():
    """Dense reference should use the same CSR graph as event providers."""

    case = _small_case()
    device = torch.device("cpu")
    x_seq = bench.make_input_sequence(case, device)
    matrix = bench.make_recurrent_csr(case, device)
    dense = bench.csr_to_dense(matrix)

    ref = bench.dense_rsnn_forward(x_seq, dense, case)
    direct = bench.dense_rsnn_forward(x_seq, matrix.mm(torch.eye(case.n_neuron)), case)

    torch.testing.assert_close(ref.spikes, direct.spikes)
    torch.testing.assert_close(ref.v, direct.v)
    torch.testing.assert_close(ref.psc, direct.psc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_event_span_providers_match_dense_reference_when_available():
    """CUDA/Triton event span providers should match dense RSNN dynamics."""

    if not bench._triton_available():
        pytest.skip("requires Triton")

    case = _small_case()
    device = torch.device("cuda")
    x_seq = bench.make_input_sequence(case, device)
    matrix = bench.make_recurrent_csr(case, device)
    dense = bench.csr_to_dense(matrix)
    ref = bench.dense_rsnn_forward(x_seq, dense, case)

    pre = bench.event_rsnn_forward(x_seq, matrix, case, schedule="pre_span")
    post = bench.event_rsnn_forward(x_seq, matrix, case, schedule="post_span")

    torch.testing.assert_close(pre.spikes, ref.spikes, atol=0, rtol=0)
    torch.testing.assert_close(pre.v, ref.v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pre.psc, ref.psc, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(post.spikes, ref.spikes, atol=0, rtol=0)
    torch.testing.assert_close(post.v, ref.v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(post.psc, ref.psc, atol=1e-5, rtol=1e-5)
