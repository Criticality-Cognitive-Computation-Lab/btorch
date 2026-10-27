"""Tests for the private sparse runtime scope around one RNN chunk."""

from dataclasses import replace

import pytest
import torch

from btorch.models import functional
from btorch.models.base import MemoryModule
from btorch.models.connection import SparseConnection
from btorch.models.rnn import make_rnn
from btorch.sparse import Hints
from btorch.sparse.runtime import (
    kernels_triton_tasks as task_kernels,
    registry,
    use_backend,
)


@pytest.fixture
def sparse_reference_backend():
    # These cases exercise task preparation, not the small-network dense route.
    with use_backend("torch"):
        yield


class _ScopedCell(MemoryModule):
    """Small stateful cell that records sparse-run scope notifications."""

    def __init__(self, n_neuron: int):
        super().__init__()
        self.n_neuron = n_neuron
        self.register_memory("h", torch.zeros(1), n_neuron)
        self.init_state()
        self.scope_calls = []

    def _begin_sparse_run(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        stable_addresses: bool = False,
    ) -> None:
        self.scope_calls.append(("begin", batch_size, dtype))

    def _end_sparse_run(self) -> None:
        self.scope_calls.append(("end",))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.h = self.h + x
        return self.h


@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_sparse_run_scope_covers_chunks_and_checkpoint_recompute(grad_checkpoint):
    """A trajectory enters once; checkpoint recomputes enter once per chunk."""
    cell = make_rnn(
        _ScopedCell,
        chunk_size=2,
        unroll=2,
        grad_checkpoint=grad_checkpoint,
    )(n_neuron=3)
    functional.init_net_state(cell, batch_size=4)
    x = torch.ones(4, 4, 3, dtype=torch.float64, requires_grad=True)

    output, _ = cell(x)
    scoped = cell.rnn_cell
    assert scoped.scope_calls == [
        ("begin", 4, torch.float64),
        ("end",),
    ]
    if grad_checkpoint:
        output.sum().backward()

    begins = [call for call in scoped.scope_calls if call[0] == "begin"]
    ends = [call for call in scoped.scope_calls if call[0] == "end"]
    assert len(begins) == 1
    assert len(ends) == 1
    assert all(call == ("begin", 4, torch.float64) for call in begins)


def test_sparse_run_scope_uses_batch_one_for_unbatched_time_input():
    """A ``[T, N]`` loop tensor reports one sample and its dtype."""
    cell = make_rnn(_ScopedCell, unroll=2)(n_neuron=3)
    functional.init_net_state(cell)
    x = torch.ones(4, 3, dtype=torch.float64)

    cell._run_chunk_steps(x, loop_args=(0,), unroll_size=2)

    assert cell.rnn_cell.scope_calls == [
        ("begin", 1, torch.float64),
        ("end",),
    ]


def test_sparse_run_scope_flattens_all_sample_dimensions():
    """Every sample dimension before neurons contributes to queue capacity."""
    cell = make_rnn(_ScopedCell, unroll=2)(n_neuron=3)
    functional.init_net_state(cell, batch_size=(2, 4))
    x = torch.ones(4, 2, 4, 3)

    cell._run_chunk_steps(x, loop_args=(0,), unroll_size=2)

    assert cell.rnn_cell.scope_calls == [
        ("begin", 8, torch.float32),
        ("end",),
    ]


class _SparseCell(MemoryModule):
    """Stateless cell exposing a real sparse connection to the RNN scope."""

    def __init__(self):
        super().__init__()
        self.connection = SparseConnection.from_edges(
            torch.tensor([0, 1, 2, 0]),
            torch.tensor([1, 2, 0, 2]),
            3,
            3,
            hints=Hints(expected_density=0.0001),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.connection(x)


def test_sparse_connection_materializes_once_for_a_multichunk_trajectory(
    monkeypatch, sparse_reference_backend
):
    """Weight ordering and task preparation are outside the timestep loop."""
    cell = make_rnn(_SparseCell, chunk_size=2, unroll=1)()
    connection = cell.rnn_cell.connection
    calls = {"weight": 0, "prepare": 0}
    original_weight = connection.weight.forward

    def weight():
        calls["weight"] += 1
        return original_weight()

    def prepare(*args, **kwargs):
        calls["prepare"] += 1
        return None, 0

    monkeypatch.setattr(connection.weight, "forward", weight)
    connection._route = replace(
        connection._route,
        implementation=replace(
            connection._route.implementation,
            prepare_values=prepare,
        ),
    )

    x = torch.ones(6, 4, 3)
    output, _ = cell(x)

    assert output.shape == (6, 4, 3)
    assert calls == {"weight": 1, "prepare": 1}
    assert connection._run_weight is None
    assert connection._run_task_values is None


def test_pull_connection_materializes_once_for_a_multichunk_trajectory(monkeypatch):
    """Destination-driven execution also materializes weights once."""
    cell = make_rnn(_SparseCell, chunk_size=2, unroll=1)()
    connection = cell.rnn_cell.connection
    connection.hints = Hints()
    calls = 0
    original_weight = connection.weight.forward

    def weight():
        nonlocal calls
        calls += 1
        return original_weight()

    monkeypatch.setattr(connection.weight, "forward", weight)
    output, _ = cell(torch.ones(6, 4, 3))

    assert output.shape == (6, 4, 3)
    assert calls == 1
    assert connection._run_weight is None


def test_bound_route_does_not_resolve_registry_inside_trajectory(monkeypatch):
    """Planning binds every callback before the recurrent timestep loop."""
    cell = make_rnn(_SparseCell, chunk_size=2, unroll=1)()

    def fail_resolve(*args, **kwargs):
        raise AssertionError("registry.resolve ran inside the trajectory")

    monkeypatch.setattr(registry, "resolve", fail_resolve)
    output, _ = cell(torch.ones(8, 4, 3))
    assert output.shape == (8, 4, 3)


def test_checkpoint_recompute_reuses_prepared_task_values(
    monkeypatch, sparse_reference_backend
):
    """Backward recomputation reuses the trajectory's packed task values."""
    cell = make_rnn(
        _SparseCell,
        chunk_size=2,
        unroll=1,
        grad_checkpoint=True,
    )()
    calls = 0

    def prepare(t_crow, t_col, t_perm, weight, batch_size, **kwargs):
        nonlocal calls
        calls += 1
        return weight.detach().clone(), 1

    connection = cell.rnn_cell.connection
    connection._route = replace(
        connection._route,
        implementation=replace(
            connection._route.implementation,
            prepare_values=prepare,
        ),
    )
    x = torch.ones(6, 4, 3, requires_grad=True)
    output, _ = cell(x)
    monkeypatch.setattr(
        connection,
        "_ensure_plan_current",
        lambda: (_ for _ in ()).throw(
            AssertionError("checkpoint recompute reconsidered the route")
        ),
    )
    output.sum().backward()
    assert calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_task_route_prepares_once_and_only_runs_per_timestep(monkeypatch):
    """A real PR70 trajectory pays conversion once and execution ``T``
    times."""
    if (
        not registry.has("spike_push_dense", "cuda")
        or registry.name("spike_push_dense", "cuda") != "triton"
    ):
        pytest.skip("the Triton task route is unavailable")

    cell = make_rnn(_SparseCell, chunk_size=2, unroll=1)().cuda()
    connection = cell.rnn_cell.connection
    calls = {"prepare": 0, "bind": 0, "pack": 0, "run": 0}

    def counted(name, function):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        task_kernels,
        "prepare",
        counted("prepare", task_kernels.prepare),
    )
    monkeypatch.setattr(
        task_kernels,
        "bind_values",
        counted("bind", task_kernels.bind_values),
    )
    monkeypatch.setattr(
        task_kernels,
        "pack_values",
        counted("pack", task_kernels.pack_values),
    )
    monkeypatch.setattr(
        task_kernels,
        "run",
        counted("run", task_kernels.run),
    )

    pointers = tuple(
        getattr(connection.cache, name).data_ptr() for name in connection.cache._NAMES
    )
    time_steps = 8
    output, _ = cell(torch.ones(time_steps, 4, 3, device="cuda"))

    assert output.shape == (time_steps, 4, 3)
    assert calls == {
        "prepare": 1,
        "bind": 1,
        "pack": 1,
        "run": time_steps,
    }
    assert pointers == tuple(
        getattr(connection.cache, name).data_ptr() for name in connection.cache._NAMES
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_task_route_supports_a_standalone_connection_call():
    """A task-planned connection creates one temporary single-step scope."""
    connection = _SparseCell().connection.cuda()
    x = torch.ones(4, 3, device="cuda")

    output = connection(x)

    assert output.shape == (4, 3)
    assert connection._run_weight is None
    assert connection._run_task_values is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "backend,weight_dtype,input_dtype",
    [
        ("triton", torch.float16, torch.float16),
        ("triton", torch.float32, torch.float64),
        ("triton", torch.float64, torch.float16),
        ("triton", torch.bfloat16, torch.float16),
        ("cusparse", torch.float16, torch.float16),
        ("cusparse", torch.bfloat16, torch.bfloat16),
    ],
)
def test_cuda_sparse_routes_preserve_typed_activity_and_gradients(
    backend, weight_dtype, input_dtype, monkeypatch
):
    """Sparse routes use actual operand dtypes, including strided views.

    These selected cases cover half storage, double activity with single
    weights, double weights with half activity, and Torch's FP16/BF16
    promotion. A fractional FP64 perturbation detects accidental FP32
    calculation; the independent dense product checks both input and
    trainable edge gradients.
    """
    from btorch.sparse.runtime import kernels_torch, use_backend

    def no_reference(*args, **kwargs):
        pytest.fail("Supported floating operands must execute native sparse kernels")

    if backend == "triton":
        monkeypatch.setattr(kernels_torch, "csr_matvec", no_reference)
        monkeypatch.setattr(kernels_torch, "edge_grad", no_reference)
    connection = _SparseCell().connection.cuda().to(dtype=weight_dtype)
    storage = torch.empty((4, 7), device="cuda", dtype=input_dtype)
    x = storage[:, 1::2]
    x.copy_(torch.arange(12, device="cuda").reshape(4, 3) / 8 + 2**-30)
    if input_dtype == torch.float64:
        x.add_(2**-30)
    x.requires_grad_()
    # The cuSPARSE cases exercise Torch's existing low-precision CUDA CSR
    # support, while Triton cases cover direct/hash task propagation.
    with use_backend(backend):
        output = connection(x)

    dtype = torch.promote_types(weight_dtype, input_dtype)
    dense = torch.zeros((3, 3), device="cuda", dtype=dtype).index_put(
        (connection.post, connection.pre),
        connection.weight().to(dtype),
        accumulate=True,
    )
    expected = x.to(dtype) @ dense.T
    tolerance = {
        torch.float16: 2e-3,
        torch.bfloat16: 1e-2,
        torch.float32: 3e-5,
        torch.float64: 1e-12,
    }[dtype]
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)
    cotangent = torch.linspace(
        -0.7, 0.8, output.numel(), device="cuda", dtype=dtype
    ).reshape_as(output)
    roots = (x, *connection.parameters())
    actual_gradients = torch.autograd.grad(output, roots, cotangent)
    expected_gradients = torch.autograd.grad(expected, roots, cotangent)
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_standalone_task_plan_uses_prebound_pull_fallback():
    """Compilation never tries to prepare task state inside the graph."""
    connection = _SparseCell().connection.cuda()
    x = torch.ones(4, 3, device="cuda")
    expected = connection(x)

    actual = torch.compile(connection, fullgraph=True)(x)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_task_route_reprepares_after_kernel_cache_clear(monkeypatch):
    """Clearing process-local artefacts invalidates packed task bindings."""
    cell = make_rnn(_SparseCell, unroll=1)().cuda()
    calls = 0
    prepare_values = cell.rnn_cell.connection._route.implementation.prepare_values
    assert prepare_values is not None

    def prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return prepare_values(*args, **kwargs)

    connection = cell.rnn_cell.connection
    connection._route = replace(
        connection._route,
        implementation=replace(
            connection._route.implementation,
            prepare_values=prepare,
        ),
    )
    x = torch.ones(4, 2, 3, device="cuda")

    first, _ = cell(x)
    registry.kernels.clear()
    second, _ = cell(x)

    torch.testing.assert_close(second, first)
    assert calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_checkpoint_recompute_restores_evicted_task_binding(monkeypatch):
    """Checkpoint recomputation restores its exact task binding after
    eviction."""
    if (
        not registry.has("spike_push_dense", "cuda")
        or registry.name("spike_push_dense", "cuda") != "triton"
    ):
        pytest.skip("the Triton task route is unavailable")

    cell = make_rnn(
        _SparseCell,
        chunk_size=2,
        unroll=1,
        grad_checkpoint=True,
    )().cuda()
    calls = {"prepare": 0, "bind": 0, "pack": 0}

    def counted(name, function):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        task_kernels,
        "prepare",
        counted("prepare", task_kernels.prepare),
    )
    monkeypatch.setattr(
        task_kernels,
        "bind_values",
        counted("bind", task_kernels.bind_values),
    )
    monkeypatch.setattr(
        task_kernels,
        "pack_values",
        counted("pack", task_kernels.pack_values),
    )

    x = torch.ones(8, 4, 3, device="cuda", requires_grad=True)
    output, _ = cell(x)
    registry.kernels.clear()
    output.sum().backward()
    assert calls == {"prepare": 1, "bind": 1, "pack": 1}
