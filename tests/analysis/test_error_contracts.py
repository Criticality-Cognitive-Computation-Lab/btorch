"""Tests for the error-handling contracts of the analysis helpers.

Each test documents one contract: failures are reported as NaN (with a
warning) or explicit exceptions, never as fake numeric values or strings.
"""

import numpy as np
import pytest
import torch
import yaml

from btorch.analysis.branching import branching_ratio
from btorch.analysis.dynamic_tools import complexity, spiking as dyn_spiking
from btorch.analysis.statistics import describe_array
from btorch.utils.yaml_utils import load_yaml, save_yaml


def _spikes(T=2000, N=4, p=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((T, N)) < p).astype(np.float32)


def test_compare_fano_methods_failure_is_nan_not_string(monkeypatch):
    """A failing compensation method yields NaN and a warning, not 'Error:

    ..'.
    """

    def boom(*args, **kwargs):
        raise ValueError("not enough data")

    monkeypatch.setattr(dyn_spiking, "fano_mean_matching", boom)
    with pytest.warns(UserWarning, match="mean_matching"):
        results = dyn_spiking.compare_fano_methods(_spikes(), dt_ms=1.0)

    assert np.isnan(results["mean_matching"])
    # No value in the result dict may be a string.
    assert not any(isinstance(v, str) for v in results.values())


def test_compare_fano_methods_programming_errors_propagate(monkeypatch):
    """TypeError (a bug) must not be swallowed into NaN."""

    def bug(*args, **kwargs):
        raise TypeError("bad argument")

    monkeypatch.setattr(dyn_spiking, "fano_mean_matching", bug)
    with pytest.raises(TypeError):
        dyn_spiking.compare_fano_methods(_spikes(), dt_ms=1.0)


def test_fano_operational_time_rejects_overlap():
    """``overlap`` used to be silently ignored; now it raises."""
    with pytest.raises(ValueError, match="overlap"):
        dyn_spiking.fano_operational_time(_spikes(), overlap=0.5)
    # overlap=0 / None (no overlap) stays accepted.
    dyn_spiking.fano_operational_time(_spikes(), overlap=0)


def test_branching_ratio_no_maxslopes():
    """The legacy ``maxslopes`` synonym of ``k_max`` was removed."""
    counts = np.random.default_rng(0).poisson(5, size=500).astype(float)
    with pytest.raises(TypeError):
        branching_ratio(counts, maxslopes=10)


def test_describe_array_returns_stats(capsys):
    """describe_array returns a dict and can be silenced with verbose=False."""
    x = np.arange(101, dtype=float)
    stats = describe_array(x, verbose=False)
    assert capsys.readouterr().out == ""
    assert stats["mean"] == 50.0 and stats["median"] == 50.0
    assert stats["q25"] == 25.0 and stats["q75"] == 75.0
    assert stats["min"] == 0.0 and stats["max"] == 100.0
    # Default remains verbose for backward compatibility.
    describe_array(x)
    assert "Mean:" in capsys.readouterr().out


def test_save_yaml_fallback_and_explicit_failure(tmp_path):
    """Objects with a YAML-safe __dict__ are saved; others raise TypeError."""

    class Args:
        def __init__(self):
            self.lr = 0.1

    save_yaml(Args(), str(tmp_path), "ok.yaml")
    assert load_yaml(str(tmp_path / "ok.yaml")) == {"lr": 0.1}

    class Bad:
        def __init__(self):
            self.fn = lambda: None  # not YAML-safe

    with pytest.raises(TypeError):
        save_yaml(Bad(), str(tmp_path), "bad.yaml")
    assert not (tmp_path / "bad.yaml").exists()
    with pytest.raises(TypeError):
        save_yaml(
            object.__new__(type("Slots", (), {"__slots__": ()})),
            str(tmp_path / "s.yaml"),
        )
    assert yaml is not None


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.magnitude = torch.nn.Parameter(torch.ones(2))


def test_gain_stability_missing_layer_raises():
    """A model without brain.synapse.linear raises instead of returning 0.0."""
    with pytest.raises(AttributeError, match="brain.synapse.linear"):
        complexity.calculate_gain_stability_sensitivity(
            torch.nn.Linear(1, 1), [{"input": torch.zeros(1, 4, 1)}], device="cpu"
        )


def test_gain_stability_restores_weights_and_returns_nan(monkeypatch):
    """Weights are restored on failure; failed LE gives NaN and a NaN slope."""
    model = torch.nn.Module()
    model.brain = torch.nn.Module()
    model.brain.synapse = torch.nn.Module()
    model.brain.synapse.linear = _Linear()
    model.brain.neuron = torch.nn.Module()

    import btorch.models.functional as functional
    import btorch.models.init as minit

    monkeypatch.setattr(functional, "reset_net", lambda *a, **k: None)
    monkeypatch.setattr(minit, "uniform_v_", lambda *a, **k: None)
    monkeypatch.setattr(
        complexity, "get_continuous_spiking_rate", lambda s, dt: s.numpy()
    )
    # Fake forward pass returning random spikes.
    model.forward = lambda x: (
        None,
        {"neuron": {"spike": torch.rand(50, 1, 3)}},
    )
    loader = [{"input": torch.zeros(1, 50, 1)}]

    # 1) LE estimation fails for every gain -> NaN everywhere, NaN slope.
    def le_fail(*a, **k):
        raise ValueError("series too short")

    monkeypatch.setattr(complexity, "compute_max_lyapunov_exponent", le_fail)
    slope, intercept, g, lam = complexity.calculate_gain_stability_sensitivity(
        model, loader, g_values=np.array([1.0, 2.0, 3.0]), device="cpu"
    )
    assert np.isnan(slope) and np.isnan(intercept)
    assert np.isnan(lam).all() and len(lam) == 3

    # 2) A hard failure mid-sweep propagates, and weights are restored.
    def hard_fail(*a, **k):
        raise KeyError("programming error")

    monkeypatch.setattr(complexity, "compute_max_lyapunov_exponent", hard_fail)
    with pytest.raises(KeyError):
        complexity.calculate_gain_stability_sensitivity(
            model, loader, g_values=np.array([5.0]), device="cpu"
        )
    assert torch.equal(model.brain.synapse.linear.magnitude.data, torch.ones(2))


def test_calculate_pcist_svd_failure_is_nan_with_warning(monkeypatch):
    """A non-converging SVD yields NaN plus a warning, never a fake 0.0."""

    def boom(*args, **kwargs):
        raise RuntimeError("svd did not converge")

    monkeypatch.setattr(torch.linalg, "svd", boom)
    resp = torch.randn(50, 4)
    base = torch.randn(50, 4)
    with pytest.warns(UserWarning, match="SVD failed"):
        score = complexity.calculate_pcist(resp, base)
    assert np.isnan(score)


def test_fano_mean_matching_too_few_windows_warns_and_returns_nan():
    """Too few windows -> NaN result, UserWarning, info dict without
    'error'."""
    spikes = _spikes(T=12, N=3)
    with pytest.warns(UserWarning, match="windows"):
        result, info = dyn_spiking.fano_mean_matching(spikes, window=10)
    assert np.all(np.isnan(result))
    assert "error" not in info


@pytest.mark.parametrize("window, overlap", [(0, 0), (101, 0), (10, 10), (10, 11)])
def test_spiking_window_args_raise_value_error(window, overlap):
    """Invalid window/overlap raise ValueError (not AssertionError)."""
    from btorch.analysis.spiking import fano

    with pytest.raises(ValueError, match="window|overlap"):
        fano(_spikes(T=100, N=2), window=window, overlap=overlap)
