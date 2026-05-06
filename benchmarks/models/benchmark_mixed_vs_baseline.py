"""Speed benchmark: MixedNeuronPopulation vs raw single-type RNN.

Three configurations are compared with identical total neuron counts:
  1. Baseline: 100 %% GLIF3 inside plain RecurrentNN
  2. Adapter-only overhead: 100 %% GLIF3 inside MixedNeuronPopulation
     + ApicalRecurrentNN (no apical input)
  3. Real mixed: 50 %% GLIF3 + 50 %% TwoCompartmentGLIF inside
     ApicalRecurrentNN (with apical input)
"""

from __future__ import annotations

import time
from typing import Any

import torch

from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.linear import DenseConn
from btorch.models.neurons import GLIF3, TwoCompartmentGLIF
from btorch.models.neurons.mixed import MixedNeuronPopulation
from btorch.models.rnn import ApicalRecurrentNN, RecurrentNN
from btorch.models.synapse import AlphaPSC


DTYPE = torch.float32
WARMUP_REPS = 2
BENCH_REPS = 5


def _build_baseline_rnn(n_neuron: int, unroll: int | bool) -> RecurrentNN:
    neuron = GLIF3(n_neuron=n_neuron, step_mode="s")
    conn = DenseConn(n_neuron, n_neuron, bias=None)
    psc = AlphaPSC(n_neuron=n_neuron, tau_syn=5.0, linear=conn, step_mode="s")
    return RecurrentNN(neuron=neuron, synapse=psc, step_mode="m", unroll=unroll)


def _build_adapter_only_rnn(n_neuron: int, unroll: int | bool) -> ApicalRecurrentNN:
    mixed = MixedNeuronPopulation(
        [(n_neuron, GLIF3(n_neuron=n_neuron, step_mode="s"))],
        step_mode="s",
    )
    conn = DenseConn(n_neuron, n_neuron, bias=None)
    psc = AlphaPSC(n_neuron=n_neuron, tau_syn=5.0, linear=conn, step_mode="s")
    return ApicalRecurrentNN(neuron=mixed, synapse=psc, step_mode="m", unroll=unroll)


def _build_mixed_rnn(n_neuron: int, unroll: int | bool) -> ApicalRecurrentNN:
    n_glif = n_neuron // 2
    n_tc = n_neuron - n_glif
    mixed = MixedNeuronPopulation(
        [
            (n_glif, GLIF3(n_neuron=n_glif, step_mode="s")),
            (n_tc, TwoCompartmentGLIF(n_neuron=n_tc, step_mode="s")),
        ],
        step_mode="s",
    )
    conn = DenseConn(n_neuron, n_neuron, bias=None)
    psc = AlphaPSC(n_neuron=n_neuron, tau_syn=5.0, linear=conn, step_mode="s")
    return ApicalRecurrentNN(neuron=mixed, synapse=psc, step_mode="m", unroll=unroll)


def _time_forward(
    net: torch.nn.Module,
    x: torch.Tensor,
    x_apical: torch.Tensor | None,
    reps: int,
) -> float:
    times: list[float] = []
    for _ in range(reps):
        reset_net_state(net, batch_size=x.shape[1])
        t0 = time.perf_counter()
        with torch.no_grad(), environ.context(dt=1.0):
            if x_apical is not None:
                net(x, None, x_apical)
            else:
                net(x)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(torch.tensor(times).median().item())


def _time_train(
    net: torch.nn.Module,
    x: torch.Tensor,
    x_apical: torch.Tensor | None,
    reps: int,
) -> float:
    times: list[float] = []
    for _ in range(reps):
        reset_net_state(net, batch_size=x.shape[1])
        if x.grad is not None:
            x.grad.zero_()
        t0 = time.perf_counter()
        with environ.context(dt=1.0):
            if x_apical is not None:
                spikes, _ = net(x, None, x_apical)
            else:
                spikes, _ = net(x)
        loss = spikes.sum()
        loss.backward()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(torch.tensor(times).median().item())


def _run_triplet(
    label: str,
    T: int,
    batch_size: int,
    n_neuron: int,
    unroll: int | bool,
) -> dict[str, Any]:
    x = torch.randn(T, batch_size, n_neuron, dtype=DTYPE)
    x_apical = torch.randn(T, batch_size, n_neuron, dtype=DTYPE)

    baseline = _build_baseline_rnn(n_neuron, unroll)
    init_net_state(baseline, batch_size=batch_size, dtype=DTYPE)
    _time_forward(baseline, x, None, WARMUP_REPS)
    fwd_base = _time_forward(baseline, x, None, BENCH_REPS)
    train_base = _time_train(baseline, x, None, BENCH_REPS)

    adapter = _build_adapter_only_rnn(n_neuron, unroll)
    init_net_state(adapter, batch_size=batch_size, dtype=DTYPE)
    _time_forward(adapter, x, None, WARMUP_REPS)
    fwd_adapt = _time_forward(adapter, x, None, BENCH_REPS)
    train_adapt = _time_train(adapter, x, None, BENCH_REPS)

    mixed = _build_mixed_rnn(n_neuron, unroll)
    init_net_state(mixed, batch_size=batch_size, dtype=DTYPE)
    _time_forward(mixed, x, x_apical, WARMUP_REPS)
    fwd_mixed = _time_forward(mixed, x, x_apical, BENCH_REPS)
    train_mixed = _time_train(mixed, x, x_apical, BENCH_REPS)

    return {
        "label": label,
        "T": T,
        "batch": batch_size,
        "n": n_neuron,
        "unroll": str(unroll),
        "fwd_base_ms": fwd_base * 1000,
        "fwd_adapt_ms": fwd_adapt * 1000,
        "fwd_mixed_ms": fwd_mixed * 1000,
        "train_base_ms": train_base * 1000,
        "train_adapt_ms": train_adapt * 1000,
        "train_mixed_ms": train_mixed * 1000,
        "adapt_vs_base_fwd": fwd_adapt / fwd_base,
        "mixed_vs_base_fwd": fwd_mixed / fwd_base,
        "adapt_vs_base_train": train_adapt / train_base,
        "mixed_vs_base_train": train_mixed / train_base,
    }


def main() -> None:
    print("=" * 90)
    print("MixedNeuronPopulation speed benchmark")
    print("=" * 90)
    print(f"Warm-up reps: {WARMUP_REPS}, benchmark reps: {BENCH_REPS}")
    print("Device: CPU (CUDA not available)")
    print()

    configs = [
        ("small ", 100, 1, 64, 8),
        ("small ", 100, 16, 64, 8),
        ("medium", 200, 1, 256, 8),
        ("medium", 200, 16, 256, 8),
        ("large ", 500, 1, 1024, 8),
        ("large ", 500, 16, 1024, 8),
    ]

    results: list[dict[str, Any]] = []
    for label, T, batch_size, n, unroll in configs:
        result = _run_triplet(label, T, batch_size, n, unroll)
        results.append(result)

    header = (
        f"{'Config':>7} | {'T':>4} | {'B':>3} | {'N':>5} | {'U':>5} |"
        f" {'BaseFwd':>8} | {'AdaptFwd':>8} | {'MixFwd':>8} |"
        f" {'A/B':>5} | {'M/B':>5} |"
        f" {'BaseTrn':>8} | {'AdaptTrn':>8} | {'MixTrn':>8} |"
        f" {'A/B':>5} | {'M/B':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:>7} | {r['T']:>4} | {r['batch']:>3} |"
            f" {r['n']:>5} | {r['unroll']:>5} |"
            f" {r['fwd_base_ms']:>7.1f}ms | {r['fwd_adapt_ms']:>7.1f}ms |"
            f" {r['fwd_mixed_ms']:>7.1f}ms |"
            f" {r['adapt_vs_base_fwd']:>4.2f}x |"
            f" {r['mixed_vs_base_fwd']:>4.2f}x |"
            f" {r['train_base_ms']:>7.1f}ms | {r['train_adapt_ms']:>7.1f}ms |"
            f" {r['train_mixed_ms']:>7.1f}ms |"
            f" {r['adapt_vs_base_train']:>4.2f}x |"
            f" {r['mixed_vs_base_train']:>4.2f}x"
        )

    print()
    print("Legend:")
    print("  Base  = raw GLIF3-only inside plain RecurrentNN")
    print("  Adapt = same GLIF3 neurons wrapped in MixedNeuronPopulation")
    print("  Mix   = 50/50 GLIF3 + TwoCompartmentGLIF + apical input")
    print("  A/B   = Adapt slowdown relative to Base (pure wrapper overhead)")
    print("  M/B   = Mix slowdown relative to Base (wrapper + TC cost)")


if __name__ == "__main__":
    main()
