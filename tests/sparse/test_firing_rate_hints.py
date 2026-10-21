"""Low-rate routing preserves sparse products and their dense-input
gradients."""

import numpy as np
import pytest
import torch

from btorch.models.connection import SparseConnection
from btorch.sparse import Hints
from btorch.sparse.runtime import use_backend, use_method
from btorch.sparse.runtime.planner import Planner
from tests.sparse.helpers import random_edges


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf"), True, "4"])
def test_invalid_firing_rates(rate):
    # Invalid units/values must fail before topology preparation or execution.
    with pytest.raises(ValueError, match="firing_rate_hz"):
        Hints(firing_rate_hz=rate)


@pytest.mark.parametrize(
    ("rate", "density", "algorithm"),
    [
        (0, None, "push"),
        (4.999, None, "push"),
        (4, 1.0, "push"),
        (5, None, "pull"),
        (6, 0.001, "push"),
        (6, 1.0, "pull"),
    ],
)
def test_rate_boundary_and_density_precedence(monkeypatch, rate, density, algorithm):
    # Exercise the real route registry without launching CUDA. Availability is
    # the only simulated fact; kernel binding and dtype filtering stay real.
    # Registration stores the original availability function, which consults
    # torch.cuda.is_available dynamically.
    pytest.importorskip("triton")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    plan = Planner().plan(
        device="cuda",
        value_batched=False,
        hints=Hints(firing_rate_hz=rate, expected_density=density),
        dtype=torch.float32,
    )
    assert plan.algorithm == algorithm
    if rate < 5:
        assert plan.backend == "triton"
        assert plan.route.key.representation == "spike_list_csr"
        assert "firing_rate_hz" in plan.reason


@pytest.mark.parametrize("constraint", ["deterministic", "batched", "float64", "cpu"])
def test_low_rate_respects_execution_constraints(monkeypatch, constraint):
    pytest.importorskip("triton")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch,
        "are_deterministic_algorithms_enabled",
        lambda: constraint == "deterministic",
    )
    plan = Planner().plan(
        device="cpu" if constraint == "cpu" else "cuda",
        value_batched=constraint == "batched",
        hints=Hints(firing_rate_hz=1),
        dtype=torch.float64 if constraint == "float64" else torch.float32,
    )
    # Low activity cannot override a missing supported route, reproducibility,
    # or shared-pattern value batching. CPU keeps its existing density policy.
    assert plan.algorithm == "pull"
    with pytest.raises(RuntimeError, match="method='push'"):
        Planner().plan(
            device="cpu" if constraint == "cpu" else "cuda",
            value_batched=constraint == "batched",
            hints=Hints(),
            dtype=torch.float64 if constraint == "float64" else torch.float32,
            method="push",
        )


def test_explicit_method_overrides_hints_and_rejects_unsupported_routes(monkeypatch):
    # Explicit requests constrain the real route registry. Unlike an advisory
    # hint, they must not silently fall back to another execution method.
    pytest.importorskip("triton")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = dict(device="cuda", value_batched=False, dtype=torch.float32)
    planner = Planner()
    push = planner.plan(**args, method="push", hints=Hints(expected_density=1.0))
    assert push.route.key.representation == "spike_list_csr"
    pull = planner.plan(**args, method="pull", hints=Hints(firing_rate_hz=1))
    assert pull.algorithm == "pull"
    with use_backend("torch"), pytest.raises(RuntimeError, match="method='push'"):
        planner.plan(**args, method="push", hints=Hints())
    with pytest.raises(RuntimeError, match="method='push'"):
        planner.plan(**{**args, "dtype": torch.float64}, method="push", hints=Hints())
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    with pytest.raises(RuntimeError, match="atomic"):
        planner.plan(**args, method="push", hints=Hints())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_explicit_connection_and_scoped_methods_preserve_product_and_restore(
    monkeypatch,
):
    pytest.importorskip("triton")
    from btorch.sparse.runtime import kernels_triton_spikelist

    launches = []
    original = kernels_triton_spikelist.spike_push_list

    def observed(*args, **kwargs):
        launches.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernels_triton_spikelist, "spike_push_list", observed)
    # Exercise execution and restoration, not just the selected plan. A dense
    # input intentionally exceeds every low-activity heuristic.
    connection = SparseConnection.from_edges(
        torch.tensor([0, 1, 1], device="cuda"),
        torch.tensor([1, 0, 1], device="cuda"),
        2,
        2,
        values=torch.tensor([0.5, -0.25, 0.75], device="cuda"),
        method="push",
    )
    x = torch.tensor([[1.0, -2.0], [0.0, 0.0]], device="cuda", requires_grad=True)
    expected = x @ torch.tensor([[0.0, 0.5], [-0.25, 0.75]], device="cuda")
    for method in ("push", "pull", "push"):
        scope = use_method("pull" if method == "pull" else "auto")
        with scope:
            actual = connection(x)
            torch.testing.assert_close(actual, expected)
            assert connection._plan.algorithm == method
            grad = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
            torch.testing.assert_close(
                grad, torch.autograd.grad(expected.sum(), x, retain_graph=True)[0]
            )
    # Fullgraph calls bind the explicit connection method without a rate hint.
    actual = torch.compile(connection, backend="eager", fullgraph=True)(x)
    torch.testing.assert_close(actual, expected)
    # The same compiled callable must honor later scoped method changes,
    # including restoration; it cannot retain the first traced route.
    compiled = torch.compile(connection, backend="eager", fullgraph=True)
    for method in ("pull", "push", "auto"):
        launches.clear()
        with use_method(method):
            torch.testing.assert_close(compiled(x), expected)
        assert bool(launches) == (method != "pull")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "variant", ["plain", "empty_rows", "duplicates", "unsorted", "empty", "hub"]
)
@pytest.mark.parametrize("activity", ["silent", "sparse", "dense"])
def test_device_spike_list_preserves_product_and_vjp(variant, activity):
    pytest.importorskip("triton")
    shape = (4097, 521) if variant == "hub" else (5, 8)
    if variant in ("empty", "hub"):
        post = np.arange(shape[0]) if variant == "hub" else np.empty(0, dtype=np.int64)
        pre = np.zeros_like(post)
        values = np.linspace(-0.5, 0.7, len(post), dtype=np.float32)
        if variant == "hub":
            # Dyadic weights/cotangents make the long fanout reduction exact.
            # This case isolates topology and gradient coverage, rather than
            # comparing two unrelated FP32 summation orders.
            values = (np.arange(len(post), dtype=np.float32) % 16 - 8) / 16
        dense = np.zeros(shape, dtype=np.float32)
        dense[post, pre] = values
    else:
        post, pre, values, dense = random_edges(variant, shape=shape, dtype=np.float32)
    conn = SparseConnection.from_edges(
        torch.as_tensor(pre, device="cuda"),
        torch.as_tensor(post, device="cuda"),
        shape[1],
        shape[0],
        values=torch.as_tensor(values, device="cuda"),
        hints=Hints(firing_rate_hz=4),
    )
    # Use a strided source view: compaction and fanout must consume the same
    # contiguous snapshot, while autograd still returns the dense-input VJP.
    storage = torch.empty((3, shape[1], 2), device="cuda")
    x = storage[..., 0]
    x.copy_(torch.linspace(-1.0, 1.0, x.numel(), device="cuda").reshape_as(x))
    if activity == "silent":
        x.zero_()
    elif activity == "sparse":
        x[:, 1:] = 0
    # Dense activity deliberately exceeds the hint. Batch 1 is always silent;
    # the hub case covers compaction tails and fanout longer than one block.
    x[1].zero_()
    x.requires_grad_()
    expected = x @ torch.as_tensor(dense, device="cuda").T
    actual = conn(x)
    assert conn._plan.algorithm == "push"
    assert conn._plan.route.key.representation == "spike_list_csr"
    torch.testing.assert_close(actual, expected)
    cotangent = (
        (torch.arange(actual.numel(), device="cuda") % 19 - 9) / 16
    ).reshape_as(actual)
    actual_grad, weight_grad = torch.autograd.grad(
        actual, (x, conn.weight.value), cotangent
    )
    expected_grad = torch.autograd.grad(expected, x, cotangent)[0]
    # Silent samples retain the full dense-input derivative; each edge-slot
    # weight derivative uses its actual endpoints, including duplicate slots.
    torch.testing.assert_close(actual_grad, expected_grad)
    expected_weight_grad = (x[:, conn.pre] * cotangent[:, conn.post]).sum(0)
    torch.testing.assert_close(weight_grad, expected_weight_grad)
