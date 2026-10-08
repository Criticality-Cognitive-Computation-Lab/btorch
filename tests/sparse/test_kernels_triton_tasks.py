"""Tests for the PR70 task-queue Triton sparse-push runtime.

The task backend is deliberately tested below the public connection API. This
keeps the tests focused on the implementation's important contracts:

* source-major tasks preserve the canonical edge/value mapping;
* direct and hash-aggregated tasks produce the same result as independent
  dense and ATen references;
* queue/workspace storage is reused for a stable shape and grows for a larger
  batch; and
* an in-place topology rewrite refreshes task metadata and its epoch.

The module is skipped until the task backend is present. It is intentionally
not a CPU reference test: PR70's queue and hash kernels are CUDA/Triton code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pytest
import torch

from btorch.sparse.runtime import kernels_aten, kernels_triton_push
from btorch.sparse.runtime.backend import KernelCache
from btorch.sparse.runtime.cache import RepresentationCache


task_kernels = pytest.importorskip(
    "btorch.sparse.runtime.kernels_triton_tasks",
    reason="the PR70 task-queue backend is not installed yet",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not getattr(task_kernels, "_available", lambda: False)(),
    reason="needs CUDA and Triton",
)

DEVICE = torch.device("cuda")
N_SRC, N_DST = 8, 20
HUB_SOURCE = 3


def _graph() -> dict[str, Any]:
    """Build a non-square graph independently of the task backend.

    The hub has repeated destinations, which exercises duplicate
    accumulation and gives the task builder enough edges to split its
    work. Sources 6 and 7 have no outgoing edges, so active silent
    sources and all-silent samples are represented in the same fixture.
    """
    ordinary_source = np.asarray([0, 1, 2, 4, 5, 0, 1, 2], dtype=np.int64)
    ordinary_destination = np.asarray([1, 2, 3, 4, 5, 1, 2, 3], dtype=np.int64)
    # The hub has 640 edges, so the fixture exercises both long-fanout task
    # splitting and the bounded hash path. Repeated destinations are
    # intentional: every edge remains a distinct slot and must contribute.
    hub_destination = np.asarray([0, 4, 8, 12, 16] * 128, dtype=np.int64)
    source = np.concatenate(
        [ordinary_source, np.full(hub_destination.shape, HUB_SOURCE, dtype=np.int64)]
    )
    destination = np.concatenate([ordinary_destination, hub_destination])
    slot_values = torch.arange(source.size, dtype=torch.float32) * 0.03125 - 0.5

    dense = torch.zeros(N_DST, N_SRC, dtype=torch.float64)
    dense.index_put_(
        (
            torch.as_tensor(destination),
            torch.as_tensor(source),
        ),
        slot_values.double(),
        accumulate=True,
    )

    representation = RepresentationCache()
    representation.build(
        torch.as_tensor(destination),
        torch.as_tensor(source),
        (N_DST, N_SRC),
        version=0,
    )
    representation = representation.to(DEVICE)
    slot_values = slot_values.to(DEVICE)
    csr_values = slot_values.index_select(0, representation.perm)

    spikes = torch.zeros(4, N_SRC, device=DEVICE)
    spikes[0, HUB_SOURCE] = 1.25
    spikes[0, 0] = -0.75
    spikes[0, 7] = 2.0  # active source with no outgoing edges
    spikes[2, HUB_SOURCE] = -0.5
    spikes[3, 6] = 3.0  # another source with no outgoing edges

    return {
        "representation": representation,
        "slot_values": slot_values,
        "csr_values": csr_values,
        "dense": dense.to(DEVICE),
        "spikes": spikes,
        "source": source,
        "destination": destination,
    }


def _prepare(graph: dict[str, Any], kernel_cache: KernelCache, batch_size: int):
    """Prepare one task state using the interface under test."""
    representation = graph["representation"]
    state = task_kernels.prepare(
        kernel_cache,
        representation.t_crow,
        representation.t_col,
        representation.t_perm,
        batch_size,
        torch.float32,
    )
    assert state is not None
    return state


def _run_task(graph: dict[str, Any], state, kernel_cache: KernelCache, spikes=None):
    """Pack destination-CSR values and run the task backend."""
    spikes = graph["spikes"] if spikes is None else spikes
    packed = task_kernels.pack_values(state, graph["csr_values"])
    return task_kernels.run(state, packed, spikes, N_DST, kernel_cache)


def _aten_reference(graph: dict[str, Any], spikes: torch.Tensor) -> torch.Tensor:
    """Reference source-driven result using the existing ATen
    implementation."""
    representation = graph["representation"]
    active, ptr = kernels_aten.pack_spikes(spikes)
    return kernels_aten.spike_push(
        representation.t_crow,
        representation.t_col,
        representation.t_perm,
        graph["csr_values"],
        spikes,
        active,
        ptr,
        N_DST,
    )


def _assert_matches_references(
    graph: dict[str, Any],
    actual: torch.Tensor,
    *,
    silent_rows: tuple[int, ...] = (),
) -> None:
    """Compare float-atomic output with independent dense and ATen results."""
    spikes = graph["spikes"]
    dense = spikes.double() @ graph["dense"].T
    aten = _aten_reference(graph, spikes)

    assert actual.shape == spikes.shape[:-1] + (N_DST,)
    # The task kernels use atomic additions, so the comparison allows the
    # small order-dependent error from the 640-edge fanout.
    torch.testing.assert_close(actual, dense.float(), rtol=2e-5, atol=2e-3)
    torch.testing.assert_close(actual, aten, rtol=2e-5, atol=2e-3)
    for row in silent_rows:
        assert torch.equal(actual[row], torch.zeros_like(actual[row]))


def _permutation_fixture() -> dict[str, Any]:
    """Build a small graph whose source/destination pairs are unique."""
    source = torch.tensor([0, 1, 2, 4, 5, 0, 1, 2], dtype=torch.long)
    destination = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    representation = RepresentationCache()
    representation.build(destination, source, (N_DST, N_SRC), version=0)
    representation = representation.to(DEVICE)
    slot_values = torch.arange(source.numel(), device=DEVICE, dtype=torch.float32)
    return {
        "representation": representation,
        "csr_values": slot_values.index_select(0, representation.perm),
    }


def _tensor_data_ptrs(value: Any) -> set[int]:
    """Collect tensor addresses from a task state or workspace dataclass."""
    if torch.is_tensor(value):
        return {value.data_ptr()}
    if isinstance(value, Mapping):
        pointers: set[int] = set()
        for child in value.values():
            pointers.update(_tensor_data_ptrs(child))
        return pointers
    if is_dataclass(value) and not isinstance(value, type):
        pointers = set()
        for field in fields(value):
            pointers.update(_tensor_data_ptrs(getattr(value, field.name)))
        return pointers
    if isinstance(value, (tuple, list)):
        pointers = set()
        for child in value:
            pointers.update(_tensor_data_ptrs(child))
        return pointers
    return set()


def _workspace(value: Any) -> Any:
    """Return the explicitly named workspace when the state exposes one."""
    if isinstance(value, Mapping):
        return value.get("workspace", value)
    return getattr(value, "workspace", value)


def _capacity(value: Any) -> int | None:
    """Read the PR70 queue capacity without prescribing a state container."""
    candidate = _workspace(value)
    for name in ("queue_capacity", "capacity", "workspace_capacity"):
        capacity = getattr(candidate, name, None)
        if capacity is not None:
            return int(capacity)
        if isinstance(candidate, Mapping) and name in candidate:
            return int(candidate[name])
    return None


def _epoch(value: Any) -> int | None:
    """Read the task-layout epoch exposed by the prepared state."""
    for name in ("epoch", "layout_epoch", "topology_epoch"):
        epoch = getattr(value, name, None)
        if epoch is not None:
            return int(epoch)
        if isinstance(value, Mapping) and name in value:
            return int(value[name])
    return None


@pytest.fixture()
def graph():
    return _graph()


@pytest.fixture()
def kernel_cache():
    return KernelCache()


def test_task_queue_and_hash_aggregation_match_dense_and_aten(graph, kernel_cache):
    """Direct and hash-capable task execution preserves all edge
    contributions."""
    state = _prepare(graph, kernel_cache, batch_size=graph["spikes"].shape[0])
    actual = _run_task(graph, state, kernel_cache)
    _assert_matches_references(graph, actual, silent_rows=(1,))


def test_pack_values_composes_source_and_task_permutations(graph, kernel_cache):
    """Packed values remain aligned after CSR-to-source-to-task reordering."""
    permutation_graph = _permutation_fixture()
    representation = permutation_graph["representation"]
    state = _prepare(permutation_graph, kernel_cache, batch_size=1)
    packed = task_kernels.pack_values(state, permutation_graph["csr_values"])

    csr_rows = torch.repeat_interleave(
        torch.arange(N_DST, device=DEVICE),
        representation.crow[1:] - representation.crow[:-1],
    )
    pair_to_csr = {
        (int(source), int(destination)): position
        for position, (source, destination) in enumerate(
            zip(representation.col.tolist(), csr_rows.tolist())
        )
    }
    expected_destination_perm = torch.tensor(
        [
            pair_to_csr[(int(source), int(destination))]
            for source, destination in zip(
                state.task_pre_indices.tolist(), state.task_post_indices.tolist()
            )
        ],
        device=DEVICE,
        dtype=torch.long,
    )
    torch.testing.assert_close(
        state.task_destination_perm.to(torch.long), expected_destination_perm
    )
    torch.testing.assert_close(
        packed,
        permutation_graph["csr_values"].index_select(0, expected_destination_perm),
    )
    assert sorted(state.task_destination_perm.tolist()) == list(range(state.edge_count))


def test_bound_task_values_execute_without_conversion_in_the_timestep(
    graph, kernel_cache, monkeypatch
):
    """The recurrent step consumes prepared state without rebuilding
    layouts."""
    representation = graph["representation"]
    state = _prepare(graph, kernel_cache, batch_size=graph["spikes"].shape[0])
    packed = task_kernels.bind_values(kernel_cache, state, graph["csr_values"])
    task_binding = task_kernels.state_for_values(
        kernel_cache,
        packed,
        representation.t_crow,
        representation.t_col,
        representation.t_perm,
    )
    assert task_binding is not None
    assert task_binding.task_state is state

    def fail_prepare(*args, **kwargs):
        raise AssertionError("task prepare ran inside the timestep")

    def fail_layout(*args, **kwargs):
        raise AssertionError("layout conversion ran inside the timestep")

    monkeypatch.setattr(task_kernels, "prepare", fail_prepare)
    monkeypatch.setattr(kernels_triton_push, "_layout", fail_layout)
    actual = kernels_triton_push.spike_push_dense(
        representation.t_crow,
        representation.t_col,
        representation.t_perm,
        graph["csr_values"],
        graph["spikes"],
        N_DST,
        task_values=packed,
        cache=kernel_cache,
    )
    _assert_matches_references(graph, actual, silent_rows=(1,))


def test_bound_task_values_reject_a_different_topology(graph, kernel_cache):
    """Packed values cannot select task metadata from another connection."""
    task_state = _prepare(
        graph,
        kernel_cache,
        batch_size=graph["spikes"].shape[0],
    )
    packed_values = task_kernels.bind_values(
        kernel_cache,
        task_state,
        graph["csr_values"],
    )

    other_representation = RepresentationCache()
    other_representation.build(
        torch.as_tensor(N_DST - 1 - graph["destination"]),
        torch.as_tensor(graph["source"]),
        (N_DST, N_SRC),
        version=0,
    )
    other_representation = other_representation.to(DEVICE)

    assert (
        task_kernels.state_for_values(
            kernel_cache,
            packed_values,
            other_representation.t_crow,
            other_representation.t_col,
            other_representation.t_perm,
        )
        is None
    )
    with pytest.raises(RuntimeError, match="no prepared task state"):
        kernels_triton_push.spike_push_dense(
            other_representation.t_crow,
            other_representation.t_col,
            other_representation.t_perm,
            graph["csr_values"],
            graph["spikes"],
            N_DST,
            task_values=packed_values,
            require_task=True,
            cache=kernel_cache,
        )


def test_independent_task_values_run_on_two_cuda_streams(graph, kernel_cache):
    """Concurrent task executions own disjoint values and queue workspaces."""
    task_state = _prepare(
        graph,
        kernel_cache,
        batch_size=graph["spikes"].shape[0],
    )
    first_values = task_kernels.create_task_values(
        kernel_cache,
        task_state,
        graph["csr_values"],
    )
    second_values = task_kernels.create_task_values(
        kernel_cache,
        task_state,
        graph["csr_values"] * 2,
    )
    representation = graph["representation"]
    first_binding = task_kernels.state_for_values(
        kernel_cache,
        first_values,
        representation.t_crow,
        representation.t_col,
        representation.t_perm,
    )
    second_binding = task_kernels.state_for_values(
        kernel_cache,
        second_values,
        representation.t_crow,
        representation.t_col,
        representation.t_perm,
    )
    assert first_binding is not None
    assert second_binding is not None
    assert first_values.data_ptr() != second_values.data_ptr()
    assert _tensor_data_ptrs(first_binding.workspace) != _tensor_data_ptrs(
        second_binding.workspace
    )

    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    with torch.cuda.stream(first_stream):
        first_output = task_kernels.run(
            task_state,
            first_values,
            graph["spikes"],
            N_DST,
            kernel_cache,
            workspace=first_binding.workspace,
        )
    with torch.cuda.stream(second_stream):
        second_output = task_kernels.run(
            task_state,
            second_values,
            graph["spikes"],
            N_DST,
            kernel_cache,
            workspace=second_binding.workspace,
        )
    torch.cuda.synchronize()

    expected = _aten_reference(graph, graph["spikes"])
    torch.testing.assert_close(first_output, expected, rtol=2e-5, atol=2e-3)
    torch.testing.assert_close(second_output, expected * 2, rtol=2e-5, atol=4e-3)


def test_workspace_reuses_storage_and_grows_for_a_larger_batch(graph, kernel_cache):
    """Repeated preparation is allocation-free until the batch needs growth."""
    first = _prepare(graph, kernel_cache, batch_size=1)
    second = _prepare(graph, kernel_cache, batch_size=1)
    assert first is second
    assert first.layout_version == second.layout_version
    assert _tensor_data_ptrs(_workspace(first)) == _tensor_data_ptrs(_workspace(second))

    first_capacity = _capacity(first)
    if first_capacity is None:
        pytest.skip("task state does not expose queue capacity")

    larger_batch = max(2, first_capacity + 1)
    grown = _prepare(graph, kernel_cache, batch_size=larger_batch)
    grown_capacity = _capacity(grown)
    assert grown_capacity is not None and grown_capacity >= larger_batch
    assert grown is not first
    assert grown.layout_version != first.layout_version
    assert _prepare(graph, kernel_cache, batch_size=1) is grown
    grown_spikes = graph["spikes"][:1].expand(larger_batch, -1).clone()
    grown_spikes[-1].zero_()
    actual = _run_task(
        graph,
        grown,
        kernel_cache,
        spikes=grown_spikes,
    )
    expected_graph = dict(graph)
    expected_graph["spikes"] = grown_spikes
    _assert_matches_references(expected_graph, actual, silent_rows=(larger_batch - 1,))


def test_inplace_topology_refresh_updates_task_layout_and_epoch(graph, kernel_cache):
    """Rewriting same-size CSR buffers cannot leave the old task layout
    cached."""
    representation = graph["representation"]
    buffer_ids = tuple(
        id(getattr(representation, name)) for name in RepresentationCache._NAMES
    )
    state_before = _prepare(graph, kernel_cache, batch_size=graph["spikes"].shape[0])
    epoch_before = _epoch(state_before)
    if epoch_before is None:
        pytest.skip("task state does not expose a layout epoch")
    assert (
        task_kernels.layout_epoch(
            kernel_cache,
            representation.t_crow,
            representation.t_col,
            representation.t_perm,
            graph["spikes"].shape[0],
            torch.float32,
        )
        == epoch_before
    )

    new_source = torch.as_tensor(graph["source"], device=DEVICE)
    new_destination = torch.as_tensor((N_DST - 1 - graph["destination"]), device=DEVICE)
    representation.build(new_destination, new_source, (N_DST, N_SRC), version=1)
    assert buffer_ids == tuple(
        id(getattr(representation, name)) for name in RepresentationCache._NAMES
    )

    updated = dict(graph)
    updated["csr_values"] = graph["slot_values"].index_select(0, representation.perm)
    updated["dense"] = torch.zeros(N_DST, N_SRC, dtype=torch.float64, device=DEVICE)
    updated["dense"].index_put_(
        (new_destination, new_source),
        graph["slot_values"].double(),
        accumulate=True,
    )

    state_after = _prepare(updated, kernel_cache, batch_size=updated["spikes"].shape[0])
    epoch_after = _epoch(state_after)
    assert epoch_after != epoch_before
    assert (
        task_kernels.layout_epoch(
            kernel_cache,
            representation.t_crow,
            representation.t_col,
            representation.t_perm,
            updated["spikes"].shape[0],
            torch.float32,
        )
        == epoch_after
    )
    actual = _run_task(updated, state_after, kernel_cache)
    _assert_matches_references(updated, actual, silent_rows=(1,))


def test_bounded_hash_probe_fallback_preserves_colliding_edges(
    monkeypatch, kernel_cache
):
    """When available, bounded probing must spill rather than drop an edge."""
    monkeypatch.setattr(task_kernels, "_HASH_CAPACITY", 2)
    monkeypatch.setattr(task_kernels, "_HASH_MAX_PROBE", 1)
    graph = _graph()
    state = _prepare(graph, kernel_cache, batch_size=graph["spikes"].shape[0])
    assert bool(state.hash_aggregation_mask.any())
    hub_destinations = set(graph["destination"][8:].tolist())
    assert len(hub_destinations) > 2
    assert all((destination * 2654435761) & 1 == 0 for destination in hub_destinations)
    actual = _run_task(graph, state, kernel_cache)
    _assert_matches_references(graph, actual, silent_rows=(1,))
