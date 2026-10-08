"""A captured CUDA graph must not be replayed for a different network
structure.

``RecurrentNN(cudagraph=True)`` records the device kernels of the time loop once
and replays them. Replay runs no python, so everything a submodule decides on
the host is frozen at capture time: which branch its forward takes, and which
derived device buffers its execution backend reads. A sparse connection whose
edges are rewired in place (``load_state_dict`` of another same-size pattern,
``set_edges_`` / ``HardDeepR``) changes both, and the old graph then silently
computes with the old wiring.

The capture layer (``btorch/models/cudagraph.py``) therefore implements a small
duck-typed protocol, tested here:

* a submodule exposing ``capture_version`` takes part in graph validity: when the
  value differs from the one a graph was captured with, the runner captures again
  (warm-up included) instead of replaying;
* a submodule whose ``capture_incompatibility()`` returns a reason makes
  ``cudagraph=True`` refuse to run, with that reason in the message.

Every structural test compares against a *dense twin*: an eager network with the
same neurons whose recurrent weights are the dense matrix of the connection's
current edges. The twin shares no code with the sparse execution path or with
the capture machinery, so agreeing with it means the right wiring was used.
"""

import pytest
import torch
from torch import nn

from btorch.models import environ
from btorch.models.connection import SparseConnection
from btorch.models.cudagraph import (
    CudaGraphRunner,
    capture_incompatibilities,
    capture_versions,
)
from btorch.models.functional import init_net_state, reset_net_state
from btorch.models.linear import Linear
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC
from btorch.sparse import Hints
from btorch.sparse.runtime import registry, use_backend


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cudagraph capture requires CUDA"
)

# Tiny on purpose: the point is which edges are used, not throughput, and the
# suite has to fit next to other processes on a small GPU.
N, B, T, N_EDGE = 48, 4, 10, 400
DEVICE, DTYPE = "cuda", torch.float32


# --- helpers --------------------------------------------------------------------


_NOOP_MOVE_REASON = (
    "SparseConnection._apply bumps capture_version on a no-op .to(), so every "
    "reset_net_state forces a re-capture and capture counts cannot be checked"
)


def _noop_move_changes_version() -> bool:
    """Whether ``conn.to()`` *without* a new device or dtype changes
    ``capture_version``.

    It must not: ``reset_net_state`` -- called before every simulation -- starts
    with exactly that no-op ``net.to(device=None, dtype=None)``. A connection
    that counts it as "moved" invalidates its graph on every call, so the
    network captures each time and never replays (correct, but slower than not
    using ``cudagraph`` at all).
    """
    pre, post, weight = _edges(seed=0)
    conn = SparseConnection.from_edges(pre, post, N, N, values=weight)
    before = conn.capture_version
    conn.to(device=None, dtype=None)
    return conn.capture_version != before


def _assert_replayed(runner: CudaGraphRunner, n_captures: int) -> None:
    """The calls since the last check were replays: the count is still
    ``n_captures``.

    Only checkable while a no-op move leaves the connection's version alone
    (see :func:`_noop_move_changes_version`). Until then the test is reported
    as an expected failure at this point; call this last, after the
    correctness assertions.
    """
    if _noop_move_changes_version():
        # Everything before this call (the comparisons against the dense twin)
        # has run and passed; only the capture count cannot hold yet. Report
        # that instead of silently passing a check that was never made.
        pytest.xfail(_NOOP_MOVE_REASON)
    assert runner.n_captures == n_captures


@pytest.fixture(params=["aten", None], ids=["aten", "default-backend"])
def backend(request):
    """Run the whole test under one sparse kernel backend.

    The two backends go stale for *different* reasons, so both are covered:

    * ``aten`` reads the connection's CSR buffers directly (those are rewritten
      in place and would survive a replay), but after rewiring the forward takes
      a new python branch -- a weight ``index_select`` through the slot
      permutation -- that the old graph never recorded.
    * the default backend (Triton when installed) additionally keeps derived
      device buffers of its own that only host-side code refreshes.

    ``use_backend`` is process-global state, hence a fixture around the test body
    (connection construction included: the execution plan is made at build time).
    """
    with use_backend(request.param):
        yield request.param


def _edges(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``N_EDGE`` distinct random ``(pre, post)`` pairs and their weights.

    Every seed yields the same number of edges, so two patterns fit the same
    connection (``load_state_dict`` checks sizes, not which pairs are wired).
    """
    gen = torch.Generator().manual_seed(seed)
    flat = torch.randperm(N * N, generator=gen)[:N_EDGE]
    weight = torch.randn(N_EDGE, generator=gen) * 0.6
    return flat // N, flat % N, weight


def _connection(seed: int, **kwargs) -> SparseConnection:
    pre, post, weight = _edges(seed)
    conn = SparseConnection.from_edges(pre, post, N, N, values=weight, **kwargs)
    return conn.to(DEVICE)


def _network(linear: nn.Module, **rnn_kwargs) -> RecurrentNN:
    """LIF neurons with an exponential synapse over the recurrent
    ``linear``."""
    net = RecurrentNN(
        neuron=LIF(n_neuron=N, v_threshold=1.0, v_reset=0.0, tau=5.0),
        synapse=ExponentialPSC(N, tau_syn=5.0, linear=linear),
        step_mode="m",
        update_state_names=("neuron.v", "synapse.psc"),
        **rnn_kwargs,
    ).to(DEVICE)
    init_net_state(net, batch_size=B, device=DEVICE, dtype=DTYPE)
    return net


def _drive() -> torch.Tensor:
    """External current ``[T, B, N]`` strong enough that about half of the
    neurons spike each step, so the recurrent wiring visibly shapes the raster
    (two different patterns differ in hundreds of spikes within ``T``
    steps)."""
    gen = torch.Generator().manual_seed(123)
    return (torch.randn(T, B, N, generator=gen) + 0.8).to(DEVICE, DTYPE)


def _run(net: RecurrentNN, drive: torch.Tensor):
    """One simulation from the reset state; returns ``(spikes, states)``."""
    with torch.no_grad(), environ.context(dt=1.0):
        reset_net_state(net, batch_size=B)
        return net(drive)


def _dense_twin(conn: SparseConnection, drive: torch.Tensor):
    """Eager reference run for the connection's *current* edges and weights.

    ``to_sparse("pre_post")`` is the connectome layout (rows are sources), which
    is transposed for ``Linear``, whose weight follows ``[post, pre]``.
    """
    weight = conn.to_sparse("pre_post").to_dense().detach()
    return _run(_network(Linear(N, N, weight=weight.T, bias=False)), drive)


def _assert_matches(result, reference) -> None:
    """Graphed sparse run equals the eager dense run.

    Spikes are compared exactly: they are binary, and a network that used the
    wrong wiring is off by hundreds of them. The continuous states only differ
    by the summation order of a dense matmul versus a CSR product.
    """
    spikes, states = result
    ref_spikes, ref_states = reference
    assert spikes.sum() > 0, "a silent network would make the comparison trivial"
    n_wrong = int((spikes != ref_spikes).sum())
    assert n_wrong == 0, f"{n_wrong} of {spikes.numel()} spikes differ"
    for name, value in states.items():
        torch.testing.assert_close(value, ref_states[name], atol=1e-4, rtol=1e-4)


def _shuffled_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    """State dict of another pattern of the same size, slots in random order.

    A checkpoint written after structural plasticity has its edge slots in
    arbitrary order. Shuffling ``indices`` together with the per-slot weights
    describes the very same matrix, but the loading connection must now gather
    its weights through a permutation -- the python-side branch change a stale
    graph misses on every backend.
    """
    state = {k: v.clone() for k, v in _connection(seed).state_dict().items()}
    order = torch.randperm(N_EDGE, generator=torch.Generator().manual_seed(seed))
    order = order.to(DEVICE)
    state["indices"] = state["indices"][:, order]
    state["weight.value"] = state["weight.value"][order]
    return state


# --- a real sparse connection ---------------------------------------------------


def test_capture_then_replay_matches_dense(backend):
    """Baseline: one capture, then replays, all equal to the dense twin."""
    conn = _connection(seed=0)
    net = _network(conn, cudagraph=True)
    runner = net._cudagraph_runner
    drive = _drive()
    reference = _dense_twin(conn, drive)

    _assert_matches(_run(net, drive), reference)  # warm-up + capture + replay
    assert runner.n_captures == 1

    # Nothing changed: these are pure replays of the same graph.
    _assert_matches(_run(net, drive), reference)
    _assert_matches(_run(net, drive), reference)
    _assert_replayed(runner, 1)


def test_replay_does_not_recapture(backend, request):
    """The usual loop -- ``reset_net_state`` then a forward, repeated --
    captures once.

    ``reset_net_state`` issues a no-op ``net.to()``, which must not count
    as a structural change of the connection.
    """
    if _noop_move_changes_version():
        # Connection-side defect, outside the capture layer: the runner does
        # what the version tells it to. Strict, so the marker cannot outlive
        # the fix -- and it is not even applied once the probe turns False.
        request.applymarker(pytest.mark.xfail(strict=True, reason=_NOOP_MOVE_REASON))
    conn = _connection(seed=0)
    net = _network(conn, cudagraph=True)
    drive = _drive()

    for _ in range(4):
        version = conn.capture_version
        _run(net, drive)  # reset_net_state + forward
        assert conn.capture_version == version, "a no-op .to() changed the version"
    assert net._cudagraph_runner.n_captures == 1


def test_load_state_dict_of_another_pattern_recaptures(backend):
    """Loading a different same-size pattern invalidates the captured graph."""
    conn = _connection(seed=0)
    net = _network(conn, cudagraph=True)
    runner = net._cudagraph_runner
    drive = _drive()

    _assert_matches(_run(net, drive), _dense_twin(conn, drive))
    assert runner.n_captures == 1
    old_reference = _dense_twin(conn, drive)
    version = conn.capture_version

    # Same shapes everywhere, different wiring: nothing in the input signature
    # the graphs are keyed by tells the two networks apart.
    conn.load_state_dict(_shuffled_checkpoint(seed=1))
    assert conn.capture_version != version
    new_reference = _dense_twin(conn, drive)
    # The two patterns really behave differently, so replaying the old graph
    # could not pass the check below by accident.
    assert (old_reference[0] != new_reference[0]).sum() > 100

    _assert_matches(_run(net, drive), new_reference)
    assert runner.n_captures == 2, "the stale graph must be captured again"
    # The stale graph (and its memory pool) is gone, not kept next to the new one.
    assert len(runner.entries) == 1

    # ... and the new graph is replayed from then on.
    _assert_matches(_run(net, drive), new_reference)
    _assert_replayed(runner, 2)


def test_set_edges_recaptures(backend):
    """Structural rewiring in place (what ``HardDeepR`` does) invalidates the
    captured graph, once per rewiring step."""
    conn = _connection(seed=0)
    net = _network(conn, cudagraph=True)
    runner = net._cudagraph_runner
    drive = _drive()

    _assert_matches(_run(net, drive), _dense_twin(conn, drive))
    old_reference = _dense_twin(conn, drive)
    version = conn.capture_version

    # Move a third of the edge slots to currently unconnected pairs; choosing
    # free pairs keeps every edge unique. Shapes and the weight parameter are
    # untouched: slot k simply becomes a different edge.
    occupied = torch.zeros(N, N, dtype=torch.bool)
    occupied[conn.indices[1].cpu(), conn.indices[0].cpu()] = True  # [pre, post]
    free = (~occupied).flatten().nonzero().squeeze(1)
    gen = torch.Generator().manual_seed(7)
    slots = torch.randperm(N_EDGE, generator=gen)[: N_EDGE // 3]
    target = free[torch.randperm(free.numel(), generator=gen)[: slots.numel()]]
    conn.set_edges_(
        slots.to(DEVICE), pre=(target // N).to(DEVICE), post=(target % N).to(DEVICE)
    )
    assert conn.capture_version != version
    new_reference = _dense_twin(conn, drive)
    assert (old_reference[0] != new_reference[0]).sum() > 100

    _assert_matches(_run(net, drive), new_reference)
    assert runner.n_captures == 2

    # A replay of the re-captured graph, without any further capture.
    _assert_matches(_run(net, drive), new_reference)
    _assert_replayed(runner, 2)


def test_inplace_weight_edit_does_not_recapture(backend):
    """Weights are *values*: the graph reads them by address at replay time, so
    an in-place edit (an optimizer step, ``constrain``) shows up in the output
    without paying for a new capture."""
    conn = _connection(seed=0)
    net = _network(conn, cudagraph=True)
    runner = net._cudagraph_runner
    drive = _drive()

    _assert_matches(_run(net, drive), _dense_twin(conn, drive))
    old_reference = _dense_twin(conn, drive)
    version = conn.capture_version

    with torch.no_grad():
        conn.weight.value.mul_(-1.5)  # flip and scale every edge in place
    assert conn.capture_version == version
    new_reference = _dense_twin(conn, drive)
    assert (old_reference[0] != new_reference[0]).sum() > 100

    _assert_matches(_run(net, drive), new_reference)
    _assert_replayed(runner, 1)  # a value edit stays a plain replay


def test_capture_version_tracks_backend_override_context():
    """Entering or leaving a backend override invalidates a captured route."""
    with use_backend(None):
        conn = _connection(seed=0)
        automatic = conn.capture_version
        with use_backend("aten"):
            assert conn.capture_version != automatic
        assert conn.capture_version != automatic


def test_capture_version_tracks_deterministic_mode():
    """Deterministic mode replans every CUDA atomic route as pull."""
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        conn = _connection(seed=0, hints=Hints(expected_density=0.01))
        nondeterministic = conn.capture_version
        torch.use_deterministic_algorithms(True)
        assert conn.capture_version != nondeterministic
        with use_backend("aten"):
            conn(torch.zeros(48, device="cuda"))
            assert "algorithm = pull" in conn.explain()
    finally:
        torch.use_deterministic_algorithms(previous)


def test_parameter_replacement_changes_capture_version():
    """A new Parameter allocation invalidates graphs that captured its
    address."""
    conn = _connection(seed=0)
    version = conn.capture_version
    conn.weight.value = nn.Parameter(conn.weight.value.detach().clone())
    assert conn.capture_version != version


def test_task_route_refreshes_packed_weights_without_recapture():
    """Task-order weights keep one address and refresh before every replay."""
    with use_backend(None):
        if (
            not registry.has("spike_push_dense", "cuda")
            or registry.name("spike_push_dense", "cuda") != "triton"
        ):
            pytest.skip("the Triton task route is unavailable")
        conn = _connection(seed=0, hints=Hints(expected_density=0.01))
        net = _network(conn, cudagraph=True)
        runner = net._cudagraph_runner
        drive = _drive()

        _assert_matches(_run(net, drive), _dense_twin(conn, drive))
        assert runner.n_captures == 1
        version = conn.capture_version

        with torch.no_grad():
            conn.weight.value.mul_(-1.5)
        assert conn.capture_version == version

        _assert_matches(_run(net, drive), _dense_twin(conn, drive))
        _assert_replayed(runner, 1)


def test_host_packed_connection_is_refused():
    """A density-hinted connection on a backend without a device-compacting
    push kernel packs spikes with ``nonzero`` on the host.

    That would be recorded once and never run again, so capture is
    refused up front with the connection's own explanation instead of
    replaying capture-time spikes.
    """
    with use_backend("aten"):  # the reference backend has no device compaction
        conn = _connection(seed=0, hints=Hints(expected_density=0.01))
        reason = conn.capture_incompatibility()
        if reason is None:
            pytest.skip("this build captures hinted connections on every backend")
        net = _network(conn, cudagraph=True)

        assert capture_incompatibilities(net) == {"synapse.linear": reason}
        with pytest.raises(RuntimeError, match="incompatible with submodule") as err:
            _run(net, _drive())
        assert reason in str(err.value)
        assert net._cudagraph_runner.n_captures == 0


# --- the protocol itself, with stand-in modules ---------------------------------


class _HostGain(nn.Module):
    """Stand-in for any module with host-side structure.

    ``gain`` is a python float: capture bakes its value into the recorded
    multiply as a constant, exactly like a rewired connection's branch choice.
    The module follows the protocol and bumps ``capture_version`` whenever the
    float changes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gain = 1.0
        self.capture_version = 0

    def set_gain(self, gain: float) -> None:
        self.gain = gain
        self.capture_version += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain


class _NotCapturable(nn.Module):
    """Stand-in that reports it cannot be captured; ``reason=None`` lifts
    it."""

    def __init__(self, reason: str | None) -> None:
        super().__init__()
        self.reason = reason

    def capture_incompatibility(self) -> str | None:
        return self.reason

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _dense_network(inp_module: nn.Module | None = None, **rnn_kwargs) -> RecurrentNN:
    """Network without any sparse connection; ``inp_module`` filters the
    drive."""
    torch.manual_seed(0)
    weight = torch.randn(N, N) * 0.1
    return _network(
        Linear(N, N, weight=weight.T, bias=False),
        neuron_inp_module=inp_module,
        **rnn_kwargs,
    )


def test_changed_capture_version_of_any_submodule_recaptures():
    """The protocol is duck-typed: the capture layer knows nothing about sparse
    connections, only about the ``capture_version`` attribute."""
    eager = _dense_network(_HostGain())
    graphed = _dense_network(_HostGain(), cudagraph=True)
    runner = graphed._cudagraph_runner
    drive = _drive()

    assert capture_versions(graphed) == (0,)
    assert torch.equal(_run(graphed, drive)[0], _run(eager, drive)[0])
    assert torch.equal(_run(graphed, drive)[0], _run(eager, drive)[0])
    assert runner.n_captures == 1

    # Change the host-side structure on both twins. Without the version bump the
    # graphed twin would keep multiplying by the captured 1.0.
    eager.neuron_inp_module.set_gain(-0.5)
    graphed.neuron_inp_module.set_gain(-0.5)
    assert capture_versions(graphed) == (1,)
    old = _run(_dense_network(_HostGain()), drive)[0]
    new = _run(eager, drive)[0]
    assert (old != new).sum() > 100

    assert torch.equal(_run(graphed, drive)[0], new)
    assert runner.n_captures == 2
    assert torch.equal(_run(graphed, drive)[0], new)
    assert runner.n_captures == 2


def test_recapture_covers_every_input_signature():
    """Each captured graph remembers the versions it was recorded for.

    With ``chunk_size`` the loop replays one graph per chunk and a second graph
    for the short remainder. After a version change both are stale, and each
    must be captured again -- not only the first one the loop reaches.
    """
    eager = _dense_network(_HostGain(), unroll=2, chunk_size=4)
    graphed = _dense_network(_HostGain(), unroll=2, chunk_size=4, cudagraph=True)
    runner = graphed._cudagraph_runner
    drive = _drive()  # T=10 -> chunks of 4, 4 and a remainder of 2

    assert torch.equal(_run(graphed, drive)[0], _run(eager, drive)[0])
    assert (runner.n_captures, len(runner.entries)) == (2, 2)

    eager.neuron_inp_module.set_gain(2.0)
    graphed.neuron_inp_module.set_gain(2.0)
    assert torch.equal(_run(graphed, drive)[0], _run(eager, drive)[0])
    assert (runner.n_captures, len(runner.entries)) == (4, 2)


def test_reported_incompatibility_refuses_capture():
    """``capture_incompatibility()`` is consulted on every call, before any
    capture, and its reason is passed on verbatim."""
    reason = "the frobnicator synchronises with the host"
    blocker = _NotCapturable(reason)
    graphed = _dense_network(blocker, cudagraph=True)
    drive = _drive()

    # The message names the offending submodule by its dotted path.
    with pytest.raises(RuntimeError, match="neuron_inp_module") as err:
        _run(graphed, drive)
    assert reason in str(err.value)
    assert graphed._cudagraph_runner.n_captures == 0

    # Once the module reports no problem, the same network captures fine ...
    blocker.reason = None
    assert capture_incompatibilities(graphed) == {}
    _run(graphed, drive)
    assert graphed._cudagraph_runner.n_captures == 1

    # ... and a problem that appears later is still caught (no stale verdict).
    blocker.reason = reason
    with pytest.raises(RuntimeError, match="frobnicator"):
        _run(graphed, drive)


def test_tree_without_participants_behaves_as_before():
    """No submodule opts in -> no versions, no extra capture, and in-place
    parameter updates (here a full ``load_state_dict``) are still picked up by
    plain replays, as they always were."""
    eager = _dense_network()
    graphed = _dense_network(cudagraph=True)
    runner = graphed._cudagraph_runner
    drive = _drive()

    assert capture_versions(graphed) == ()
    assert capture_incompatibilities(graphed) == {}
    for _ in range(3):
        assert torch.equal(_run(graphed, drive)[0], _run(eager, drive)[0])
    assert runner.n_captures == 1

    state = {k: v * -2.0 for k, v in eager.synapse.linear.state_dict().items()}
    eager.synapse.linear.load_state_dict(state)
    graphed.synapse.linear.load_state_dict(state)
    assert torch.equal(_run(graphed, drive)[0], _run(eager, drive)[0])
    assert runner.n_captures == 1


def test_runner_versions_callable_for_state_outside_the_module():
    """Direct use of the runner: ``versions=`` covers structure the module tree
    cannot report, such as host values the captured function closes over."""
    net = _dense_network()
    drive = _drive()
    host = {"gain": 1.0, "version": 0}
    runner = CudaGraphRunner(versions=lambda: (host["version"],))

    def fn(x):
        # ``host["gain"]`` is read by python: frozen into the graph at capture.
        return net._stacked_chunk_forward(x * host["gain"], unroll_size=T)[0]

    def call():
        with torch.no_grad(), environ.context(dt=1.0):
            reset_net_state(net, batch_size=B)
            return runner(net, fn, (drive,))

    def reference():
        return _run(net, drive * host["gain"])[0]

    assert torch.equal(call(), reference())
    assert torch.equal(call(), reference())
    assert runner.n_captures == 1

    host.update(gain=-1.0, version=1)
    assert torch.equal(call(), reference())
    assert runner.n_captures == 2
