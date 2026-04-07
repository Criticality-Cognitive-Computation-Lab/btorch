"""Benchmark spiking analysis functions with large inputs.

Tests performance of CV, Fano factor, Kurtosis, Local Variation, ECI, and
lag correlation with realistic large-scale inputs: T=1000, B=8, N=100000
"""

import time

import torch

from btorch.analysis.dynamic_tools.ei_balance import (
    compute_eci,
    compute_ei_balance,
    compute_lag_correlation,
)
from btorch.analysis.spiking import (
    fano,
    isi_cv,
    kurtosis,
    local_variation,
)


def _run_benchmark(
    *,
    label: str,
    input_shape: tuple[int, int, int],
    output_shape: torch.Size | tuple[int, ...] | None,
    value_unit: str,
    warmup_fn,
    run_fn,
    extra_result: str | None = None,
) -> float:
    """Run a benchmark with consistent warmup/timing/reporting."""
    T, B, N = input_shape
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {label}: T={T}, B={B}, N={N}")
    print(f"{'=' * 60}")

    warmup_fn()

    start = time.time()
    run_fn()
    elapsed = time.time() - start

    if output_shape is not None:
        print(f"  Shape: {input_shape} -> {label} shape: {tuple(output_shape)}")
    else:
        print(f"  Shape: {input_shape}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {(T * B * N) / elapsed / 1e6:.2f}M {value_unit}/sec")
    if extra_result is not None:
        print(f"  Results: {extra_result}")

    return elapsed


def _benchmark_header(name: str, T: int, B: int, N: int, device, dtype) -> None:
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {name}: T={T}, B={B}, N={N}, device={device}, dtype={dtype}")
    print(f"{'=' * 60}")


def generate_spike_data(T, B, N, rate=0.05, device="cpu", dtype=torch.float32):
    """Generate synthetic spike data.

    Args:
        T: Time steps
        B: Batch size (trials)
        N: Number of neurons
        rate: Firing probability per time step
        device: torch device
        dtype: torch dtype

    Returns:
        spike_data: Tensor of shape [T, B, N]
    """
    torch.manual_seed(42)
    spikes = (torch.rand(T, B, N, device=device, dtype=torch.float32) < rate).to(dtype)
    return spikes


def generate_current_data(T, B, N, device="cpu", dtype=torch.float32):
    """Generate synthetic current data for E/I balance analysis.

    Args:
        T: Time steps
        B: Batch size (trials)
        N: Number of neurons
        device: torch device
        dtype: torch dtype

    Returns:
        I_e, I_i: Excitatory and inhibitory currents [T, B, N]
    """
    torch.manual_seed(42)
    I_e = torch.randn(T, B, N, device=device, dtype=dtype) * 0.5 + 1.0
    I_i = -torch.randn(T, B, N, device=device, dtype=dtype) * 0.5 - 0.5
    return I_e, I_i


def benchmark_cv(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark ISI CV computation."""
    _benchmark_header("CV", T, B, N, device, dtype)

    spikes = generate_spike_data(T, B, N, device=device, dtype=dtype)
    cv_values = None

    def warmup():
        isi_cv(spikes, dt_ms=1.0, batch_axis=(1,))

    def run():
        nonlocal cv_values
        cv_values, _ = isi_cv(spikes, dt_ms=1.0, batch_axis=(1,))

    return _run_benchmark(
        label="CV",
        input_shape=(T, B, N),
        output_shape=cv_values.shape if cv_values is not None else spikes.shape[1:],
        value_unit="spikes",
        warmup_fn=warmup,
        run_fn=run,
    )


def benchmark_fano(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark Fano factor computation."""
    _benchmark_header("Fano", T, B, N, device, dtype)

    spikes = generate_spike_data(T, B, N, device=device, dtype=dtype)
    fano_values = None

    def warmup():
        fano(spikes, window=100, batch_axis=(1,))

    def run():
        nonlocal fano_values
        fano_values, _ = fano(spikes, window=100, batch_axis=(1,))

    return _run_benchmark(
        label="Fano",
        input_shape=(T, B, N),
        output_shape=fano_values.shape if fano_values is not None else spikes.shape[1:],
        value_unit="spikes",
        warmup_fn=warmup,
        run_fn=run,
    )


def benchmark_kurtosis(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark Kurtosis computation."""
    _benchmark_header("Kurtosis", T, B, N, device, dtype)

    spikes = generate_spike_data(T, B, N, device=device, dtype=dtype)
    kurt_values = None

    def warmup():
        kurtosis(spikes, window=100, batch_axis=(1,))

    def run():
        nonlocal kurt_values
        kurt_values, _ = kurtosis(spikes, window=100, batch_axis=(1,))

    return _run_benchmark(
        label="Kurtosis",
        input_shape=(T, B, N),
        output_shape=kurt_values.shape if kurt_values is not None else spikes.shape[1:],
        value_unit="spikes",
        warmup_fn=warmup,
        run_fn=run,
    )


def benchmark_lv(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark Local Variation computation."""
    _benchmark_header("LV", T, B, N, device, dtype)

    spikes = generate_spike_data(T, B, N, device=device, dtype=dtype)
    lv_values = None

    def warmup():
        local_variation(spikes, dt_ms=1.0, batch_axis=(1,))

    def run():
        nonlocal lv_values
        lv_values, _ = local_variation(spikes, dt_ms=1.0, batch_axis=(1,))

    return _run_benchmark(
        label="LV",
        input_shape=(T, B, N),
        output_shape=lv_values.shape if lv_values is not None else spikes.shape[1:],
        value_unit="spikes",
        warmup_fn=warmup,
        run_fn=run,
    )


def benchmark_eci(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark ECI computation."""
    _benchmark_header("ECI", T, B, N, device, dtype)

    I_e, I_i = generate_current_data(T, B, N, device=device, dtype=dtype)
    eci = None

    def warmup():
        compute_eci(I_e, I_i, batch_axis=(1,))

    def run():
        nonlocal eci
        eci, _ = compute_eci(I_e, I_i, batch_axis=(1,))

    return _run_benchmark(
        label="ECI",
        input_shape=(T, B, N),
        output_shape=eci.shape if eci is not None else I_e.shape[1:],
        value_unit="values",
        warmup_fn=warmup,
        run_fn=run,
    )


def benchmark_lag_correlation(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark lag correlation computation."""
    _benchmark_header("Lag Correlation", T, B, N, device, dtype)

    I_e, I_i = generate_current_data(T, B, N, device=device, dtype=dtype)
    peak_corr = None

    def warmup():
        compute_lag_correlation(I_e, -I_i, dt=1.0, max_lag_ms=30.0, batch_axis=(1,))

    def run():
        nonlocal peak_corr
        peak_corr, _, _ = compute_lag_correlation(
            I_e, -I_i, dt=1.0, max_lag_ms=30.0, batch_axis=(1,)
        )

    return _run_benchmark(
        label="Lag Correlation",
        input_shape=(T, B, N),
        output_shape=peak_corr.shape if peak_corr is not None else I_e.shape[1:],
        value_unit="values",
        warmup_fn=warmup,
        run_fn=run,
    )


def benchmark_ei_full(T=1000, B=8, N=100000, device="cpu", dtype=torch.float32):
    """Benchmark full E/I balance computation."""
    _benchmark_header("E/I Balance Full", T, B, N, device, dtype)

    I_e, I_i = generate_current_data(T, B, N, device=device, dtype=dtype)
    eci = None
    peak_corr = None

    def warmup():
        compute_ei_balance(I_e, I_i, batch_axis=(1,))

    def run():
        nonlocal eci, peak_corr
        eci, peak_corr, _, _ = compute_ei_balance(I_e, I_i, batch_axis=(1,))

    return _run_benchmark(
        label="E/I Balance Full",
        input_shape=(T, B, N),
        output_shape=None,
        value_unit="values",
        warmup_fn=warmup,
        run_fn=run,
        extra_result=(
            f"eci_mean={eci.mean():.3f}, peak_corr_mean={peak_corr.mean():.3f}"
            if eci is not None and peak_corr is not None
            else None
        ),
    )


def compare_dtypes(T=1000, B=8, N=100000, device="cpu"):
    """Compare float16 vs float32 performance."""
    print(f"\n{'='*60}")
    print(f"Comparing dtypes: T={T}, B={B}, N={N}, device={device}")
    print(f"{'='*60}")

    for dtype in [torch.float16, torch.float32]:
        spikes = generate_spike_data(T, B, N, device=device, dtype=dtype)
        mem_mb = spikes.numel() * spikes.element_size() / 1024 / 1024

        start = time.time()
        _ = fano(spikes, window=100, batch_axis=(1,))
        elapsed = time.time() - start

        print(f"  {dtype}: {mem_mb:.1f}MB, {elapsed:.3f}s")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("Spiking Analysis Benchmarks")
    print("=" * 60)

    T, B, N = 1000, 8, 100000

    # CPU benchmarks
    print("\n" + "=" * 60)
    print("CPU BENCHMARKS")
    print("=" * 60)

    benchmark_cv(T, B, N, device="cpu")
    benchmark_fano(T, B, N, device="cpu")
    benchmark_kurtosis(T, B, N, device="cpu")
    benchmark_lv(T, B, N, device="cpu")

    # E/I balance benchmarks
    print("\n" + "=" * 60)
    print("E/I BALANCE BENCHMARKS (CPU)")
    print("=" * 60)

    benchmark_eci(T, B, N, device="cpu")
    benchmark_lag_correlation(T, B, N, device="cpu")
    benchmark_ei_full(T, B, N, device="cpu")

    # GPU benchmarks if available
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU BENCHMARKS")
        print("=" * 60)

        benchmark_cv(T, B, N, device="cuda")
        benchmark_fano(T, B, N, device="cuda")
        benchmark_kurtosis(T, B, N, device="cuda")
        benchmark_lv(T, B, N, device="cuda")

        # E/I balance GPU benchmarks
        print("\n" + "=" * 60)
        print("E/I BALANCE BENCHMARKS (GPU)")
        print("=" * 60)

        benchmark_eci(T, B, N, device="cuda")
        benchmark_lag_correlation(T, B, N, device="cuda")
        benchmark_ei_full(T, B, N, device="cuda")

        # Compare dtypes on GPU
        compare_dtypes(T, B, N, device="cuda")
    else:
        print("\nCUDA not available, skipping GPU benchmarks")

    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
