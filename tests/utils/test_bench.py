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
