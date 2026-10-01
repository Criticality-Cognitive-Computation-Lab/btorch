"""Characterization tests for ``compute_stat`` / ``compute_stats_batch``.

These pin the numeric behaviour of the statistics helpers (NumPy and torch
backends, NaN/Inf policies, dimension handling and the ``use_stats`` decorator
plumbing) so that the implementation can be refactored safely.  Every expected
value is computed independently with plain NumPy.
"""

import warnings

import numpy as np
import pytest
import torch

from btorch.analysis.statistics import (
    compute_stat,
    compute_stats_batch,
    use_percentiles,
    use_stats,
)


ALL_STATS = ["mean", "median", "max", "min", "std", "var", "argmax", "argmin", "cv"]


def _reference(x: np.ndarray, stat: str, axis, *, torch_semantics: bool = False):
    """Independent NumPy reference for each supported statistic.

    Torch's ``std``/``var``/``cv`` are unbiased (``ddof=1``) whereas NumPy's are
    population statistics (``ddof=0``); the library keeps each backend's native
    convention, so the reference mirrors that with ``torch_semantics``.
    """
    ddof = 1 if torch_semantics else 0
    if stat == "cv":
        return x.std(axis=axis, ddof=ddof) / (x.mean(axis=axis) + 1e-10)
    if stat in ("std", "var"):
        return getattr(x, stat)(axis=axis, ddof=ddof)
    if stat == "median":
        return np.median(x, axis=axis)
    return getattr(x, stat)(axis=axis)


@pytest.fixture
def data() -> np.ndarray:
    rng = np.random.default_rng(0)
    # Odd lengths so torch (lower median) and NumPy medians agree.
    return rng.normal(loc=2.0, scale=1.0, size=(13, 5))


@pytest.mark.parametrize("stat", ALL_STATS)
def test_compute_stat_numpy_flatten_and_axis(data, stat):
    """With ``dim=None`` the array is flattened; an int ``dim`` is an axis."""
    np.testing.assert_allclose(
        compute_stat(data, stat), _reference(data.reshape(-1), stat, 0)
    )
    np.testing.assert_allclose(
        compute_stat(data, stat, dim=0), _reference(data, stat, 0)
    )


@pytest.mark.parametrize("stat", ALL_STATS)
def test_compute_stat_torch_flatten_returns_python_float(data, stat):
    """Torch inputs reduce to Python scalars when the result is a single
    value."""
    out = compute_stat(torch.from_numpy(data), stat)
    assert isinstance(out, (int, float))
    expected = _reference(data.reshape(-1), stat, 0, torch_semantics=True)
    assert out == pytest.approx(float(expected))


@pytest.mark.parametrize("stat", ALL_STATS)
def test_batch_matches_single_numpy_and_torch(data, stat):
    """The batch API equals the single-stat API, for both backends."""
    batch = compute_stats_batch(data, ALL_STATS, dim=0)
    np.testing.assert_allclose(batch[stat], _reference(data, stat, 0))
    batch_t = compute_stats_batch(torch.from_numpy(data), ALL_STATS, dim=0)
    np.testing.assert_allclose(
        np.asarray(batch_t[stat]),
        _reference(data, stat, 0, torch_semantics=True),
        rtol=1e-6,
    )
    one = compute_stats_batch(torch.from_numpy(data), [stat])
    expected = _reference(data.reshape(-1), stat, 0, torch_semantics=True)
    assert one[stat] == pytest.approx(float(expected))


def test_torch_axis_reduction_keeps_tensor():
    """Multi-element torch results stay tensors (only scalars become
    floats)."""
    x = torch.arange(12.0).reshape(4, 3)
    out = compute_stat(x, "mean", dim=0)
    assert isinstance(out, torch.Tensor) and out.shape == (3,)
    scalar = compute_stat(x, "mean")
    assert isinstance(scalar, float)


def test_nan_inf_policies(data):
    """Skip drops NaN/Inf, warn warns and then skips, assert raises."""
    x = data.reshape(-1).copy()
    x[3] = np.nan
    x[7] = np.inf
    finite = x[np.isfinite(x)]
    # NaN is skipped by default, Inf propagates by default.
    assert np.isinf(compute_stat(x, "max"))
    assert compute_stat(x, "mean", inf_policy="skip") == pytest.approx(finite.mean())
    with pytest.raises(ValueError, match="NaN values found"):
        compute_stat(x, "mean", nan_policy="assert")
    with pytest.raises(ValueError, match="Inf values found"):
        compute_stats_batch(x, ["mean"], inf_policy="assert")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compute_stat(x, "mean", nan_policy="warn", inf_policy="skip")
    assert any("NaN values found" in str(w.message) for w in caught)
    assert out == pytest.approx(finite.mean())


def test_unknown_stat_raises(data):
    with pytest.raises(ValueError, match="Unknown stat"):
        compute_stat(data, "nope")
    with pytest.raises(ValueError, match="Unknown stat"):
        compute_stats_batch(data, ["mean", "nope"])


def test_use_stats_single_and_multi_stat_info_agree():
    """stat_info with one name and with several names give identical values."""

    @use_stats(value_key="v")
    def f():
        return np.arange(10.0), {}

    _, one = f(stat_info="std")
    _, many = f(stat_info=["std", "cv", "max"])
    assert one["v_std"] == many["v_std"]
    assert many["v_cv"] == pytest.approx(
        np.arange(10.0).std() / (np.arange(10.0).mean() + 1e-10)
    )


def test_use_stats_positional_stat_dict_and_dim_dict():
    """Per-position stat / dim dictionaries, with info keys per position."""

    @use_stats(value_key={0: "a", 1: "b"}, dim={0: 0, 1: None})
    def f():
        return np.arange(6.0).reshape(3, 2), np.arange(6.0).reshape(3, 2), {}

    a, b, info = f(stat={0: "mean", 1: "max"})
    np.testing.assert_allclose(a, [2.0, 3.0])  # dim=0 -> per-column mean
    assert b == 5.0  # dim=None -> flattened max
    assert "a_mean" in info and "b_max" in info
    with pytest.raises(IndexError):
        f(stat={2: "mean"})


def test_use_stats_policies_forwarded_only_when_accepted():
    """``nan_policy`` / ``inf_policy`` reach the wrapped function iff
    declared."""
    seen = {}

    @use_stats(default_nan_policy="warn")
    def g(*, nan_policy="x"):
        seen["nan"] = nan_policy
        return np.ones(3), {}

    g()
    assert seen["nan"] == "warn"
    g(nan_policy="assert")
    assert seen["nan"] == "assert"


def test_use_percentiles_roundtrip():
    @use_percentiles(value_key="v")
    def f():
        return np.arange(101.0), {}

    _, info = f(percentiles=(10, 90))
    assert info["v_levels"] == (10.0, 90.0)
    assert info["v_percentiles"] == pytest.approx([10.0, 90.0])


def test_decorated_signatures_are_honest():
    """Decorated functions no longer swallow unknown keywords (no dummy
    ``**kwargs``) and the decorators expose their added parameters."""
    from btorch.analysis import isi_cv
    from btorch.analysis.dynamic_tools.ei_balance import compute_eci

    I_e = np.ones((20, 3))
    # compute_eci returns ``(eci, info)`` once decorated.
    eci, info = compute_eci(I_e, -I_e)
    assert eci.shape == (3,) and isinstance(info, dict)
    # A misspelt keyword used to be silently ignored; now it is an error.
    with pytest.raises(TypeError):
        compute_eci(I_e, -I_e, batch_axs=(1,))
    with pytest.raises(TypeError):
        isi_cv(np.zeros((20, 3)), bogus=1)
    # The wrapped function is still reachable for introspection.
    assert compute_eci.__wrapped__.__name__ == "compute_eci"
