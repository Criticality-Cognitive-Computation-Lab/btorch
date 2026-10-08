"""Fixed-slot hard rewiring (:class:`HardDeepR`).

The connection owns ``K`` edge *slots*. Rewiring changes which ``(post, pre)``
pair a slot stands for; it never changes ``K``, a tensor shape or the weight
parameter object. The tests below force dormancy deterministically (by writing
a weight of the wrong sign into chosen slots) instead of waiting for training
to produce it; only the end-to-end test relies on training.
"""

import copy

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from btorch import sparse
from btorch.models.connection import (
    ConstantWeight,
    ConstrainedWeight,
    SparseConnection,
    Synapse,
)
from btorch.models.connection.rewire import HardDeepR, HardDeepROptions
from btorch.sparse import as_sparse


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
N_PRE, N_POST, K = 12, 9, 30


def _adjacency(n_pre=N_PRE, n_post=N_POST, k=K, seed=0, dale=True):
    """Random ``src_dst`` adjacency (rows = sources) with exactly ``k`` edges.

    With ``dale`` every source neuron has one sign (even rows excitatory, odd
    rows inhibitory), the situation Dale's law describes.
    """
    g = torch.Generator().manual_seed(seed)
    flat = torch.randperm(n_pre * n_post, generator=g)[:k]
    pre, post = flat // n_post, flat % n_post
    magnitude = torch.rand(k, generator=g) + 0.5
    if dale:
        sign = torch.where(pre % 2 == 0, 1.0, -1.0)
    else:
        sign = torch.where(torch.rand(k, generator=g) < 0.5, 1.0, -1.0)
    return torch.sparse_coo_tensor(
        torch.stack([pre, post]), sign * magnitude, (n_pre, n_post)
    ).coalesce()


def _make(dale=True, device="cpu", seed=0, **kwargs):
    """Connection plus controller; returns ``(conn, rewire)``."""
    conn = SparseConnection.from_adjacency(
        _adjacency(seed=seed, dale=dale), Synapse(dale=dale)
    ).to(device)
    options = kwargs.pop("options", None) or HardDeepROptions(**kwargs)
    return conn, HardDeepR(conn, options, generator=torch.Generator().manual_seed(1))


def _kill(conn, rewire, slots):
    """Make ``slots`` dormant: move their weight across zero (theta < 0)."""
    slots = torch.as_tensor(slots, device=conn.indices.device)
    with torch.no_grad():
        conn.weight.value[slots] = -0.1 * rewire._signs()[slots]
    return slots


def _dense(conn):
    """Dense operator ``[n_post, n_pre]`` rebuilt from the semantic edge list.

    ``index_put_(accumulate=True)`` would sum duplicate edges, so a forward
    pass that agrees with this matrix also shows that the execution layouts
    describe exactly the edges of the table.
    """
    table = conn.edge_table()
    dense = torch.zeros(
        conn.n_post, conn.n_pre, dtype=table["weight"].dtype, device=conn.indices.device
    )
    return dense.index_put_(
        (table["post"], table["pre"]), table["weight"].detach(), accumulate=True
    )


def _keys(conn):
    """One integer per edge: equal keys <=> the same ``(post, pre)`` pair."""
    return (conn.indices[0] * conn.n_pre + conn.indices[1]).cpu()


# ------------------------------------------------------------ edge budget
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dale", [True, False])
def test_slot_count_fixed_and_no_duplicates(dale, device):
    """K slots before, K distinct edges after; rewired slots really move."""
    conn, rewire = _make(dale=dale, device=device)
    param = conn.weight.value
    old_indices = conn.indices.clone()
    old_keys = _keys(conn)
    slots = _kill(conn, rewire, [0, 3, 7, 11, 20])

    assert rewire.dormant_slots().tolist() == slots.tolist()
    assert rewire.step() == 5

    # The budget: same number of slots, same tensors, same parameter object.
    assert conn.nnz == K and conn.indices.shape == (2, K)
    assert conn.weight.value is param and param.shape == (K,)
    # No two slots describe the same connection.
    new_keys = _keys(conn)
    assert new_keys.unique().numel() == K
    # Every rewired slot moved to a position that was unconnected before
    # (positions vacated in this update are not reused within it) ...
    assert not torch.isin(new_keys[slots.cpu()], old_keys).any()
    # ... and every other slot is untouched.
    others = torch.ones(K, dtype=torch.bool)
    others[slots.cpu()] = False
    assert torch.equal(conn.indices[:, others], old_indices[:, others])
    # New coordinates are valid neuron ids.
    assert int(conn.indices[0].max()) < N_POST and int(conn.indices[1].max()) < N_PRE
    # Nothing is dormant any more: new connections start at theta = init > 0.
    assert rewire.dormant_slots().numel() == 0
    theta = (param * rewire._signs())[slots]
    assert torch.allclose(theta, torch.full_like(theta, rewire.options.init))


def test_no_autapses_and_candidate_restriction():
    """``allow_autapses=False`` and ``candidate`` restrict the new positions.

    A recurrent (square) connection is rewired completely many times; no
    new edge may be a self-connection, and all of them must come from
    the allowed sources (the first half of the population).
    """
    n = 10
    W = _adjacency(n, n, 20, dale=True)
    conn = SparseConnection.from_adjacency(W, Synapse(dale=True))
    allowed_pre = torch.arange(n) < n // 2
    rewire = HardDeepR(
        conn,
        HardDeepROptions(
            allow_autapses=False, candidate=lambda post, pre: allowed_pre[pre]
        ),
        generator=torch.Generator().manual_seed(0),
    )
    for _ in range(20):
        _kill(conn, rewire, torch.arange(conn.nnz))
        assert rewire.step() == conn.nnz
        post, pre = conn.indices
        assert not (post == pre).any()
        assert allowed_pre[pre].all()
        assert _keys(conn).unique().numel() == conn.nnz


def test_sampling_is_uniform_over_free_positions():
    """New positions are spread uniformly over the unconnected pairs.

    One slot of a small connection is rewired many times from the same
    starting topology; every free position must be hit, about equally
    often (a loose 5-sigma band around the binomial expectation).
    """
    conn, rewire = _make(dale=True)
    start = copy.deepcopy(conn.state_dict())
    free = N_PRE * N_POST - K
    counts = torch.zeros(N_POST * N_PRE)
    n_draw = 20 * free
    for _ in range(n_draw):
        conn.load_state_dict(start)
        _kill(conn, rewire, [4])
        rewire.step()
        counts[_keys(conn)[4]] += 1
    occupied = torch.zeros(N_POST * N_PRE, dtype=torch.bool)
    conn.load_state_dict(start)
    occupied[_keys(conn)] = True
    assert counts[occupied].sum() == 0
    p = 1 / free
    sigma = (n_draw * p * (1 - p)) ** 0.5
    assert (counts[~occupied] - n_draw * p).abs().max() < 5 * sigma


def test_reproducible_with_generator():
    """The same generator seed gives the same rewiring."""
    results = []
    for _ in range(2):
        conn, rewire = _make(dale=False)
        _kill(conn, rewire, [1, 2, 3, 4, 5])
        rewire.step()
        results.append((conn.indices.clone(), conn.weight.value.detach().clone()))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


# ------------------------------------------------------------------ signs
def test_dale_sign_of_new_edges_follows_source():
    """With Dale's law a new edge takes the sign of its source neuron.

    ``sign="pre"`` (the Dale default) derives the per-source sign from the
    existing edges: in the fixture even sources are excitatory and odd ones
    inhibitory. The weight's own ``sign`` buffer is updated, so the Dale
    projection keeps protecting the new edges.
    """
    conn, rewire = _make(dale=True, init=0.01)
    for _ in range(10):
        _kill(conn, rewire, torch.arange(0, K, 2))
        rewire.step()
    pre = conn.indices[1]
    has_edges = _adjacency().indices()[0].unique()
    known = torch.isin(pre, has_edges)  # sources without edges: random sign
    expected = torch.where(pre % 2 == 0, 1.0, -1.0)
    assert torch.equal(conn.weight.sign[known], expected[known])
    assert (conn.weight.value * conn.weight.sign >= 0).all()
    # Sources without initial edges still have ONE sign for all their edges.
    for source in pre[~known].unique():
        assert conn.weight.sign[pre == source].unique().numel() == 1

    # An explicit per-source sign tensor overrides the derived one.
    conn, _ = _make(dale=True)
    rewire = HardDeepR(conn, HardDeepROptions(sign=-torch.ones(N_PRE)))
    slots = _kill(conn, rewire, [0, 1, 2])
    rewire.step()
    assert (conn.weight.sign[slots] == -1).all()
    assert (conn.weight.value[slots] < 0).all()


def test_dormancy_without_dale_tracks_activation_sign():
    """Without Dale's law a weight is dormant once it crosses zero.

    The reference is the sign the weight had when its slot was
    (re)activated; a weight that merely changes magnitude stays active.
    """
    conn, rewire = _make(dale=False)
    assert rewire.dormant_slots().numel() == 0
    with torch.no_grad():
        conn.weight.value[2] *= 3.0  # same sign: still active
        conn.weight.value[5] *= -1.0  # crossed zero: dormant
        conn.weight.value[6] = 0.0  # reached zero: dormant
    assert rewire.dormant_slots().tolist() == [5, 6]
    assert rewire.step() == 2
    assert rewire.dormant_slots().numel() == 0
    # The tracked sign of a rewired slot is the sign of its new weight.
    assert torch.equal(torch.sign(conn.weight.value.detach()), rewire._signs())


# -------------------------------------------------------------- execution
@pytest.mark.parametrize("device", DEVICES)
def test_forward_matches_dense_after_rewiring(device):
    """The forward pass uses the *new* edges, with unchanged shapes."""
    conn, rewire = _make(dale=True, device=device, init=0.3)
    x = torch.randn(4, 5, N_PRE, device=device)
    shapes = {k: v.shape for k, v in conn.state_dict().items()}
    before = conn(x)
    dense_before = _dense(conn)
    assert torch.allclose(before, x @ dense_before.T, atol=1e-5)

    for event in range(3):
        _kill(conn, rewire, torch.arange(event, K, 3))
        assert rewire.step() > 0
        out = conn(x)
        dense = _dense(conn)
        # The operator really changed, and the forward follows it.
        assert not torch.equal(dense != 0, dense_before != 0)
        assert out.shape == before.shape
        assert torch.allclose(out, x @ dense.T, atol=1e-5)
        dense_before = dense
    # Persistent state: same keys and shapes as before any rewiring.
    assert {k: v.shape for k, v in conn.state_dict().items()} == shapes


def test_gradients_flow_to_rewired_slots():
    """A rewired slot trains like any other: its gradient is the dense one."""
    conn, rewire = _make(dale=True, init=0.2)
    slots = _kill(conn, rewire, [0, 5, 9])
    rewire.step()
    x = torch.randn(6, N_PRE)
    target = torch.randn(6, N_POST)
    ((conn(x) - target) ** 2).sum().backward()
    dense = _dense(conn).requires_grad_()
    ((x @ dense.T - target) ** 2).sum().backward()
    post, pre = conn.indices
    assert torch.allclose(conn.weight.value.grad, dense.grad[post, pre], atol=1e-4)
    assert conn.weight.value.grad[slots].abs().sum() > 0


def test_topology_version_and_cache_consistency():
    """``topology_version`` moves only when edges move; the cache follows.

    The derived layouts (CSR, transposed CSR, permutations) are rebuilt
    *into the same tensors*. They must equal the layouts of a connection
    built from scratch from the new edge list, i.e. nothing of the old
    pattern survives.
    """
    conn, rewire = _make(dale=False, init=0.5)
    cache_tensors = {n: getattr(conn.cache, n) for n in conn.cache._NAMES}

    # Nothing dormant: no update, no version bump.
    version = conn.topology_version
    assert rewire.step() == 0
    assert conn.topology_version == version

    for _ in range(3):
        _kill(conn, rewire, [2, 3, 17, 29])
        assert rewire.step() == 4
        assert conn.topology_version == version + 1
        assert conn.cache.topology_version == conn.topology_version
        version = conn.topology_version

        table = conn.edge_table()
        fresh = SparseConnection.from_edges(
            table["pre"], table["post"], N_PRE, N_POST, values=table["weight"].detach()
        )
        for name in ("crow", "col", "t_crow", "t_col", "t_perm"):
            assert torch.equal(getattr(conn.cache, name), getattr(fresh.cache, name))
        # ``perm`` maps execution order to slots; slot order differs between
        # the two connections, the weights in execution order do not.
        assert torch.equal(
            conn.weight()[conn.cache.perm], fresh.weight()[fresh.cache.perm]
        )
        x = torch.randn(3, N_PRE)
        assert torch.allclose(conn(x), fresh(x), atol=1e-6)
    # In-place rebuild: compiled / CUDA graphs that captured the buffers
    # keep seeing the current topology.
    for name, tensor in cache_tensors.items():
        assert getattr(conn.cache, name) is tensor


def test_checkpoint_after_rewiring_roundtrip(tmp_path):
    """A checkpoint taken after rewiring restores the rewired network.

    The canonical edge list is part of the ``state_dict``; the execution
    layouts are derived from it on load. The fresh connection is built from
    the *original* matrix (same K, different edges).
    """
    conn, rewire = _make(dale=True, init=0.4)
    for event in range(4):
        _kill(conn, rewire, torch.arange(event, K, 4))
        rewire.step()
    x = torch.randn(5, N_PRE)
    expected = conn(x)
    torch.save(conn.state_dict(), tmp_path / "conn.pt")

    fresh = SparseConnection.from_adjacency(_adjacency(), Synapse(dale=True))
    assert not torch.allclose(fresh(x), expected)
    fresh.load_state_dict(torch.load(tmp_path / "conn.pt"))  # strict
    assert torch.equal(fresh.indices, conn.indices)
    assert torch.equal(fresh.weight.sign, conn.weight.sign)
    assert torch.allclose(fresh(x), expected, atol=1e-6)


# -------------------------------------------------------- optimizer state
def _train_steps(conn, optimizer, n=3):
    for _ in range(n):
        optimizer.zero_grad()
        conn(torch.randn(8, N_PRE)).pow(2).sum().backward()
        optimizer.step()


OPTIMIZERS = {
    "adam": (
        lambda p: torch.optim.Adam(p, lr=1e-3, amsgrad=True),
        ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
    ),
    "sgd": (
        lambda p: torch.optim.SGD(p, lr=1e-3, momentum=0.9),
        ("momentum_buffer",),
    ),
}


@pytest.mark.parametrize("policy", ["reset", "keep"])
@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_optimizer_state_policy(name, policy):
    """Per-slot optimizer state of rewired slots is reset or kept, explicitly.

    A few ordinary steps populate the state. Then the controller is attached
    and one more ``optimizer.step()`` is taken with zero gradients and
    ``lr``-independent dormant weights, so the state the step leaves behind is
    known, and the hook rewires exactly the chosen slots.
    """
    make, state_names = OPTIMIZERS[name]
    conn, rewire = _make(dale=True, optimizer_state=policy)
    param = conn.weight.value
    optimizer = make(conn.parameters())
    _train_steps(conn, optimizer)
    assert rewire.dormant_slots().numel() == 0  # tiny lr: nothing crossed zero

    handle = rewire.attach(optimizer)
    slots = _kill(conn, rewire, [1, 8, 15, 22])
    others = torch.ones(K, dtype=torch.bool)
    others[slots] = False

    # One real step; afterwards compare with the state right before the hook,
    # obtained from an identical optimizer without the hook.
    reference = make([param])
    reference.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    optimizer.zero_grad()
    conn(torch.randn(8, N_PRE)).pow(2).sum().backward()
    value_before = param.detach().clone()
    reference.step()  # plain step: the state the hook starts from
    expected = {k: reference.state[param][k].clone() for k in state_names}
    with torch.no_grad():
        param.copy_(value_before)
    optimizer.step()  # same step + structural update

    assert rewire.n_rewired == 4
    state = optimizer.state[param]
    for key in state_names:
        # Slots that were not rewired never change, under either policy.
        assert torch.equal(state[key][others], expected[key][others])
        assert expected[key][slots].abs().min() > 0
        if policy == "reset":
            assert (state[key][slots] == 0).all()
        else:
            assert torch.equal(state[key][slots], expected[key][slots])
    if name == "adam":
        # Scalar state (the step counter) is never touched.
        assert float(state["step"]) == 4
    # The optimizer still holds the very same parameter object.
    assert optimizer.param_groups[0]["params"][0] is conn.weight.value is param

    # detach() removes the hook: further steps do not rewire.
    rewire.detach()
    assert handle.id not in optimizer._optimizer_step_post_hooks
    _kill(conn, rewire, [0])
    optimizer.step()
    assert rewire.n_rewired == 4


def test_every_runs_update_on_nth_step():
    """``every=3``: the structural update runs on optimizer steps 3, 6, ..."""
    conn, rewire = _make(dale=True, every=3)
    optimizer = torch.optim.SGD(conn.parameters(), lr=0.0)
    rewire.attach(optimizer)
    _kill(conn, rewire, [0, 1])
    rewired = []
    for _ in range(6):
        optimizer.zero_grad()
        conn(torch.randn(2, N_PRE)).sum().backward()
        optimizer.step()
        rewired.append(rewire.n_rewired)
    assert rewired == [0, 0, 2, 2, 2, 2]


def test_l1_and_noise_act_on_theta():
    """``l1`` shrinks theta by ``lr * l1`` per step; ``noise`` adds a random
    walk with standard deviation ``sqrt(2 * lr * T)``.

    ``every`` is large so that no structural update interferes, and the
    learning rate only scales the two terms (the gradient is zero).
    """
    lr = 0.1
    conn, rewire = _make(dale=True, l1=0.5, every=1000)
    optimizer = torch.optim.SGD(conn.parameters(), lr=lr)
    rewire.attach(optimizer)
    theta = (conn.weight.value * conn.weight.sign).detach().clone()
    optimizer.step()
    theta_new = (conn.weight.value * conn.weight.sign).detach()
    assert torch.allclose(theta_new, theta - lr * 0.5, atol=1e-6)

    big = SparseConnection.from_adjacency(
        _adjacency(200, 200, 20000, dale=False), Synapse()
    )
    rewire = HardDeepR(big, HardDeepROptions(noise=2.0, every=1000))
    optimizer = torch.optim.SGD(big.parameters(), lr=lr)
    rewire.attach(optimizer)
    before = big.weight.value.detach().clone()
    optimizer.step()
    delta = big.weight.value.detach() - before
    assert abs(float(delta.std()) - (2 * lr * 2.0) ** 0.5) < 0.02
    assert abs(float(delta.mean())) < 0.02


# ---------------------------------------------------------- torch.compile
@pytest.fixture
def fresh_dynamo():
    """Compile from a clean slate so tests do not share guard caches."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("device", DEVICES)
def test_compile_survives_rewiring_without_recompiling(device, fresh_dynamo):
    """One compiled graph serves every topology.

    ``CompileCounterWithBackend`` wraps the real backend and counts how often
    Dynamo hands it a new frame: a guard failure caused by rewiring (a changed
    Python attribute, a replaced buffer, a changed shape) would show up as a
    second frame. Rewiring only writes new values into existing buffers, so
    the count must stay at one. ``error_on_recompile`` is a second, independent
    check: it turns any recompilation into an exception.
    """
    conn, rewire = _make(dale=True, device=device, init=0.3)
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(conn, backend=counter, fullgraph=True)
    x = torch.randn(4, N_PRE, device=device)

    assert torch.allclose(compiled(x), x @ _dense(conn).T, atol=1e-5)
    assert counter.frame_count == 1

    with torch._dynamo.config.patch(error_on_recompile=True):
        for event in range(6):
            _kill(conn, rewire, torch.arange(event, K, 5))
            assert rewire.step() > 0
            out = compiled(x)
            assert torch.allclose(out, x @ _dense(conn).T, atol=1e-5)
            assert torch.allclose(out, conn(x), atol=1e-6)
    assert counter.frame_count == 1

    # Gradients through the compiled module reach the (unchanged) parameter.
    compiled(x).sum().backward()
    assert conn.weight.value.grad is not None


# ------------------------------------------------------------- end to end
def test_training_recovers_support_with_rewiring():
    """Toy regression whose true support differs from the initial one.

    The teacher is a sparse non-negative matrix; the student starts with
    the same number of edges at *different* positions. Without rewiring
    the student cannot represent the teacher at all (its best fit is
    "all weights zero"). With Hard Deep R, misplaced edges are driven to
    zero, go dormant and are re-drawn until they land on useful
    positions, so the loss keeps decreasing below that floor while the
    edge count never changes.
    """
    torch.manual_seed(0)
    n, k = 8, 12
    g = torch.Generator().manual_seed(3)
    flat = torch.randperm(n * n, generator=g)
    true_idx, init_idx = flat[:k], flat[k : 2 * k]  # disjoint supports
    W_true = torch.zeros(n * n)
    W_true[true_idx] = 1.0
    W_true = W_true.view(n, n)  # [post, pre]

    def student(rewiring: bool):
        init = torch.sparse_coo_tensor(
            torch.stack([init_idx // n, init_idx % n]), torch.full((k,), 0.5), (n, n)
        )
        conn = SparseConnection(init, Synapse(dale=True))
        optimizer = torch.optim.Adam(conn.parameters(), lr=0.05)
        rewire = HardDeepR(
            conn,
            HardDeepROptions(init=0.0, sign=torch.ones(n)),
            generator=torch.Generator().manual_seed(0),
        )
        if rewiring:
            rewire.attach(optimizer)
        data = torch.Generator().manual_seed(1)
        losses = []
        for _ in range(1500):
            x = torch.randn(32, n, generator=data)
            loss = (conn(x) - x @ W_true.T).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not rewiring:
                conn.weight.constrain()  # plain Dale projection
            losses.append(float(loss.detach()))
        return conn, rewire, losses

    conn, rewire, losses = student(rewiring=True)
    _, _, fixed_losses = student(rewiring=False)

    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    fixed_last = sum(fixed_losses[-20:]) / 20
    assert rewire.n_rewired > 0
    assert conn.nnz == k and _keys(conn).unique().numel() == k
    # The loss decreased, and well below what the fixed topology can reach.
    assert last < 0.5 * first
    assert last < 0.5 * fixed_last
    # Most of the teacher's connections were found.
    found = torch.isin(conn.indices[0] * n + conn.indices[1], true_idx).sum()
    assert found >= k // 2
    # Dale's law holds throughout without a separate projection.
    assert (conn.weight.value >= 0).all()


# ------------------------------------------------------------ error cases
def test_unsupported_connections_raise():
    """Rewiring needs one trainable, unbatched weight per edge slot."""
    W = _adjacency()
    # Tied weights: a slot has no parameter of its own.
    group = torch.zeros(K, dtype=torch.long)
    conn = SparseConnection.from_adjacency(W, Synapse(weight=ConstrainedWeight(group)))
    with pytest.raises(TypeError, match="ConstrainedWeight"):
        HardDeepR(conn)
    conn = SparseConnection.from_adjacency(W, Synapse(weight=ConstantWeight(1.0)))
    with pytest.raises(TypeError, match="ConstantWeight"):
        HardDeepR(conn)
    # Fixed per-edge weights can never become dormant.
    conn = SparseConnection.from_adjacency(W, Synapse(weight=W.values()))
    with pytest.raises(TypeError, match="trainable"):
        HardDeepR(conn)
    # A batch of networks sharing the pattern: [G, K] weights.
    conn = SparseConnection.from_edges(
        W.indices()[0], W.indices()[1], N_PRE, N_POST, values=torch.ones(2, K)
    )
    with pytest.raises(NotImplementedError, match="batched"):
        HardDeepR(conn)
    # A batch of networks with different patterns (ragged network batch).
    members = [
        as_sparse(torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]).to_sparse()),
        as_sparse(torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]).to_sparse()),
    ]
    ragged = sparse.stack(members)
    conn = SparseConnection(ragged)
    with pytest.raises(NotImplementedError, match="different"):
        HardDeepR(conn)
    # A full connection has no unconnected position.
    conn = SparseConnection(torch.ones(3, 3).to_sparse())
    with pytest.raises(ValueError, match="dense"):
        HardDeepR(conn)


def test_option_and_attach_errors():
    """Invalid options, foreign optimizers and exhausted candidates."""
    with pytest.raises(ValueError, match="optimizer_state"):
        HardDeepROptions(optimizer_state="transfer")
    with pytest.raises(ValueError, match="every"):
        HardDeepROptions(every=0)
    with pytest.raises(ValueError, match="sign"):
        HardDeepROptions(sign="post")
    conn, _ = _make()
    with pytest.raises(ValueError, match="source"):
        HardDeepR(conn, HardDeepROptions(sign=torch.ones(N_PRE + 1)))

    conn, rewire = _make()
    other = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="does not optimise"):
        rewire.attach(torch.optim.SGD(other.parameters(), lr=0.1))
    optimizer = torch.optim.SGD(conn.parameters(), lr=0.1)
    rewire.attach(optimizer)
    with pytest.raises(RuntimeError, match="already attached"):
        rewire.attach(optimizer)

    # No admissible position: the update fails loudly and changes nothing.
    conn, rewire = _make(
        candidate=lambda post, pre: torch.zeros_like(pre, dtype=torch.bool),
        max_tries=3,
    )
    _kill(conn, rewire, [0])
    indices, version = conn.indices.clone(), conn.topology_version
    with pytest.raises(RuntimeError, match="could not find"):
        rewire.step()
    assert torch.equal(conn.indices, indices) and conn.topology_version == version


def test_receptor_and_delay_are_kept_per_slot():
    """With receptors / delays a rewired slot keeps both attributes.

    Only ``(post, pre)`` is re-drawn; uniqueness is judged on the full
    ``(post, pre, receptor, delay)`` tuple, and the forward pass on the
    expanded layout matches a dense operator rebuilt from the edge table.
    """
    W = _adjacency()
    g = torch.Generator().manual_seed(5)
    synapse = Synapse(
        receptor=torch.randint(2, (K,), generator=g),
        delay=torch.randint(3, (K,), generator=g),
        dale=True,
    )
    conn = SparseConnection.from_adjacency(W, synapse)
    rewire = HardDeepR(conn, HardDeepROptions(init=0.2), generator=g)
    receptor, delay = conn.receptor.clone(), conn.delay.clone()
    x = torch.randn(3, conn.in_features)
    for _ in range(5):
        _kill(conn, rewire, torch.arange(0, K, 2))
        assert rewire.step() == K // 2
        assert torch.equal(conn.receptor, receptor)
        assert torch.equal(conn.delay, delay)
        t = conn.edge_table()
        row = t["post"] * conn.n_receptor + t["receptor"]
        col = t["pre"] * conn.n_delay + t["delay"]
        assert (row * conn.in_features + col).unique().numel() == K
        dense = torch.zeros(conn.out_features, conn.in_features)
        dense.index_put_((row, col), t["weight"].detach(), accumulate=True)
        assert torch.allclose(conn(x), x @ dense.T, atol=1e-5)
