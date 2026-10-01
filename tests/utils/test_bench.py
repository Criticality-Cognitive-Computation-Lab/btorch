"""Tests for :func:`btorch.utils.bench.do_bench` (CPU only, mocked clock)."""

import pytest

from btorch.utils import bench


class _FakeClock:
    """Deterministic ``perf_counter`` that advances 1 s on every call."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 1.0
        return self.now


def test_duration_mode_tiny_rep_still_takes_one_sample(monkeypatch):
    """With ``rep`` far below one call's duration the sample list used to be
    empty (mean of an empty tensor is NaN).

    At least one sample is now taken.
    """
    monkeypatch.setattr(bench.time, "perf_counter", _FakeClock())
    calls = []
    # warmup=0.0 and rep=1e-6 are float -> duration mode.
    out = bench.do_bench(
        lambda: calls.append(1), warmup=0.0, rep=1e-6, timing_method="cpu"
    )
    assert len(calls) == 1
    assert out == out  # not NaN
    assert out > 0.0


def test_duration_mode_runs_until_budget_is_spent(monkeypatch):
    monkeypatch.setattr(bench.time, "perf_counter", _FakeClock())
    calls = []
    bench.do_bench(lambda: calls.append(1), warmup=0.0, rep=5000.0)
    assert len(calls) > 1


def test_iteration_mode_counts_reps():
    calls = []
    bench.do_bench(lambda: calls.append(1), warmup=2, rep=3, timing_method="cpu")
    assert len(calls) == 5


def test_invalid_rep_raises():
    with pytest.raises(ValueError):
        bench.do_bench(lambda: None, warmup=0, rep=0)


def test_return_mode_all_returns_dict_of_statistics():
    """``return_mode="all"`` returns every statistic, as the docstring
    promises.

    Regression test: the CPU path used to call ``torch.all`` on the timings.
    """
    stats = bench.do_bench(lambda: None, warmup=0, rep=5, return_mode="all")
    assert set(stats) == {"min", "max", "mean", "median"}
    # min <= median <= max and min <= mean <= max for any sample
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["min"] <= stats["mean"] <= stats["max"]


@pytest.fixture
def fake_clock(monkeypatch):
    """Patch ``perf_counter`` so every call advances the clock by 1 s."""
    clock = _FakeClock()
    monkeypatch.setattr(bench.time, "perf_counter", clock)
    return clock


def test_duration_warmup_runs_until_budget_is_spent(fake_clock):
    """Warmup in ms: the loop polls the clock once per check.

    With a 1 s tick and ``warmup=2500`` ms the checks read 1000, 2000 and 3000 ms
    after the start, so ``fn`` runs twice during warmup (before measurement).
    """
    calls = []
    # float budgets -> duration mode; rep=1 ms is shorter than a single call.
    bench.do_bench(lambda: calls.append(1), warmup=2500.0, rep=1.0)
    # 2 warmup calls + exactly 1 measured call (rep budget < one call).
    assert len(calls) == 3


def test_duration_measurement_counts_follow_clock(fake_clock):
    """Each timed call costs 1000 ms of fake time; ``rep=3500`` ms allows
    two."""
    calls = []
    stats = bench.do_bench(
        lambda: calls.append(1), warmup=0.0, rep=3500.0, return_mode="all"
    )
    assert len(calls) == 2
    # PerfTimer reads the clock twice per sample -> every sample is 1000 ms.
    assert stats == {"min": 1000.0, "max": 1000.0, "mean": 1000.0, "median": 1000.0}


def test_mixed_int_float_budget_is_duration_mode(fake_clock):
    """Duration mode needs *both* budgets to be ints to count iterations."""
    calls = []
    bench.do_bench(lambda: calls.append(1), warmup=0, rep=1e-6)
    assert len(calls) == 1  # float rep -> duration mode -> single sample


@pytest.mark.parametrize(
    "warmup,rep",
    [(-1, 1), (0, 0), (-1.0, 1.0), (0.0, 0.0), (0.0, -2.0)],
)
def test_invalid_budgets_raise(warmup, rep):
    with pytest.raises(ValueError):
        bench.do_bench(lambda: None, warmup=warmup, rep=rep)


def test_non_callable_and_unknown_timing_method_raise():
    with pytest.raises(TypeError):
        bench.do_bench(42)
    with pytest.raises(ValueError, match="timing_method"):
        bench.do_bench(lambda: None, timing_method="bogus")


def test_total_timing_method_is_alias_for_cpu():
    calls = []
    bench.do_bench(lambda: calls.append(1), warmup=0, rep=2, timing_method="total")
    assert len(calls) == 2


def test_gpu_request_without_cuda_warns_and_uses_cpu(monkeypatch):
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: False)
    calls = []
    with pytest.warns(UserWarning, match="Falling back to cpu"):
        bench.do_bench(lambda: calls.append(1), warmup=0, rep=2, timing_method="gpu")
    assert len(calls) == 2


def test_grad_to_none_is_reset_before_each_measured_call():
    """Gradients are cleared before every timed repetition, not in warmup."""
    import torch

    p = torch.nn.Parameter(torch.ones(1))
    seen = []

    def fn():
        seen.append(p.grad)
        p.grad = torch.ones(1)  # pretend backward() ran

    bench.do_bench(fn, warmup=0, rep=3, grad_to_none=[p])
    assert seen == [None, None, None]


def test_quantiles_return_list_or_scalar():
    two = bench.do_bench(lambda: None, warmup=0, rep=5, quantiles=[0.2, 0.8])
    assert isinstance(two, list) and len(two) == 2 and two[0] <= two[1]
    one = bench.do_bench(lambda: None, warmup=0, rep=5, quantiles=[0.5])
    assert isinstance(one, float)


@pytest.mark.parametrize("mode", ["min", "max", "mean", "median"])
def test_summarize_cpu_single_statistic(mode):
    times = [1.0, 2.0, 6.0]
    expected = {"min": 1.0, "max": 6.0, "mean": 3.0, "median": 2.0}[mode]
    assert bench._summarize_cpu(times, None, mode) == pytest.approx(expected)


def test_resolve_budget_returns_mode_and_normalised_values():
    assert bench._resolve_budget(2, 3) == (True, 2, 3)
    assert bench._resolve_budget(2, 3.0) == (False, 2.0, 3.0)
