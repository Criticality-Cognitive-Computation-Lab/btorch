"""Fixed-slot hard rewiring (:class:`HardDeepR`).

The connection owns ``K`` edge *slots*. Rewiring changes which ``(post, pre)``
pair a slot stands for; it never changes ``K``, a tensor shape or the weight
parameter object. The tests below force dormancy deterministically (by writing
a weight of the wrong sign into chosen slots) instead of waiting for training
to produce it; only the end-to-end test relies on training.
"""

import copy
import warnings

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
    """Random ``pre_post`` adjacency (rows = sources) with exactly ``k`` edges.

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
            # ``candidate`` is called as ``candidate(pre, post)``.
            allow_autapses=False,
            candidate=lambda pre, post: allowed_pre[pre],
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
        x = torch.randn(8, N_PRE, device=conn.indices.device)
        conn(x).pow(2).sum().backward()
        optimizer.step()


# name -> (factory, first-moment-like state, second-moment-like state).
# ``foreach`` and ``fused`` are different implementations of the same update;
# they must keep one state tensor per parameter with the same names, which is
# what the controller indexes by slot.
OPTIMIZERS = {
    "adam": (
        lambda p: torch.optim.Adam(p, lr=1e-3, amsgrad=True),
        ("exp_avg",),
        ("exp_avg_sq", "max_exp_avg_sq"),
    ),
    "adam_foreach": (
        lambda p: torch.optim.Adam(p, lr=1e-3, amsgrad=True, foreach=True),
        ("exp_avg",),
        ("exp_avg_sq", "max_exp_avg_sq"),
    ),
    "adam_fused": (
        lambda p: torch.optim.Adam(p, lr=1e-3, amsgrad=True, fused=True),
        ("exp_avg",),
        ("exp_avg_sq", "max_exp_avg_sq"),
    ),
    "sgd": (
        lambda p: torch.optim.SGD(p, lr=1e-3, momentum=0.9),
        ("momentum_buffer",),
        (),
    ),
    "rmsprop": (
        lambda p: torch.optim.RMSprop(p, lr=1e-3, momentum=0.5, centered=True),
        ("momentum_buffer", "grad_avg"),
        ("square_avg",),
    ),
    "adagrad": (lambda p: torch.optim.Adagrad(p, lr=1e-3), (), ("sum",)),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("policy", ["neutral", "reset", "keep"])
@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_optimizer_state_policy(name, policy, device):
    """Per-slot optimizer state of rewired slots follows an explicit policy.

    A few ordinary steps populate the state. Then the controller is attached
    and one more ``optimizer.step()`` is taken, and the hook rewires exactly
    the chosen slots. The state right before the hook is obtained from an
    identical optimizer without the hook.

    - ``"keep"``: nothing changes.
    - ``"reset"``: every per-slot entry of a rewired slot is zero.
    - ``"neutral"``: first moments are zero, second moments are the mean over
      the slots that were not rewired.
    """
    make, first_names, second_names = OPTIMIZERS[name]
    state_names = first_names + second_names
    conn, rewire = _make(dale=True, device=device, optimizer_state=policy)
    param = conn.weight.value
    optimizer = make(conn.parameters())
    _train_steps(conn, optimizer)
    assert rewire.dormant_slots().numel() == 0  # tiny lr: nothing crossed zero

    handle = rewire.attach(optimizer)
    slots = _kill(conn, rewire, [1, 8, 15, 22])
    others = torch.ones(K, dtype=torch.bool, device=device)
    others[slots] = False

    # One real step; afterwards compare with the state right before the hook,
    # obtained from an identical optimizer without the hook.
    reference = make([param])
    reference.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    optimizer.zero_grad()
    conn(torch.randn(8, N_PRE, device=device)).pow(2).sum().backward()
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
        if policy == "keep":
            assert torch.equal(state[key][slots], expected[key][slots])
        elif policy == "reset" or key in first_names:
            assert (state[key][slots] == 0).all()
        else:
            # Population statistic of the established connections only.
            mean = expected[key][others].mean()
            assert torch.allclose(state[key][slots], mean.expand(4))
    if "step" in state:
        # Scalar state (the global step counter) is never touched.
        assert float(state["step"]) == 4
    # The optimizer still holds the very same parameter object.
    assert optimizer.param_groups[0]["params"][0] is conn.weight.value is param

    # detach() removes the hook: further steps do not rewire.
    rewire.detach()
    assert handle.id not in optimizer._optimizer_step_post_hooks
    _kill(conn, rewire, [0])
    optimizer.step()
    assert rewire.n_rewired == 4


def _first_step_ratio(make, policy, n_warmup=1000):
    """Size of a rewired slot's first step relative to an established slot.

    Every slot receives the same constant gradient, written directly into
    ``.grad``, so after the warm-up all slots have identical optimizer state
    and an established slot moves by a known amount per step. Slot 0 is then
    rewired by the hook (``init=0`` puts its new weight at exactly zero) and
    the next step is measured for slot 0 and for the untouched slot 1.
    """
    conn, rewire = _make(dale=True, optimizer_state=policy, init=0.0, every=n_warmup)
    param = conn.weight.value
    optimizer = make([param])
    rewire.attach(optimizer)
    grad = torch.full_like(param, 0.3)
    for _ in range(n_warmup - 1):
        param.grad = grad.clone()
        optimizer.step()
    assert rewire.n_rewired == 0  # lr is tiny: no weight came close to zero
    _kill(conn, rewire, [0])
    param.grad = grad.clone()
    optimizer.step()  # step ``n_warmup``: the structural update runs
    assert rewire.n_rewired == 1 and float(param.detach()[0]) == 0.0
    before = param.detach().clone()
    rewire.detach()  # measure the plain optimizer step
    param.grad = grad.clone()
    optimizer.step()
    moved = (param.detach() - before).abs()
    # ``moved[1]`` is a float32 difference of weights of order one, so the
    # ratio is only accurate to about 1e-3 (the tolerances below reflect
    # that, not an inaccuracy of the controller).
    return float(moved[0] / moved[1])


def test_first_step_size_of_rewired_slot():
    """``optimizer_state`` decides how large a new connection's first step is.

    ``step`` (Adam's bias-correction counter) is global, so zeroing the
    second moment of one slot late in training makes its denominator tiny:
    with ``"reset"`` the new connection takes a first step several times
    larger than an established one. ``"neutral"`` gives the slot the
    population's second moment instead, so its step never exceeds the
    population's: RMSprop (no momentum) steps exactly like an established
    slot, Adam ramps up from ``(1 - beta1)`` of it as its momentum builds.
    """

    def adam(p):
        return torch.optim.Adam(p, lr=1e-4)

    def rmsprop(p):
        return torch.optim.RMSprop(p, lr=1e-4)

    # Neutral: the step of a new connection matches the population scale.
    assert _first_step_ratio(rmsprop, "neutral") == pytest.approx(1.0, rel=1e-2)
    assert _first_step_ratio(adam, "neutral") == pytest.approx(0.1, rel=1e-2)
    # Reset: inflated. Adam at t = 1000 with the default betas:
    # (1 - b1) / sqrt(1 - b2) * sqrt(1 - b2**t) = 2.5; RMSprop with
    # alpha = 0.99: 1 / sqrt(1 - alpha) = 10.
    assert _first_step_ratio(adam, "reset") == pytest.approx(2.5, rel=2e-2)
    assert _first_step_ratio(rmsprop, "reset") == pytest.approx(10.0, rel=1e-2)
    # Keep: the slot continues with the removed connection's moments.
    assert _first_step_ratio(adam, "keep") == pytest.approx(1.0, rel=1e-2)
    # The default is the policy that does not inflate.
    assert HardDeepROptions().optimizer_state == "neutral"


def test_neutral_policy_when_every_slot_is_rewired():
    """Without any established slot the mean is taken over all slots."""
    conn, rewire = _make(dale=True)
    param = conn.weight.value
    optimizer = torch.optim.Adam([param], lr=1e-3)
    _train_steps(conn, optimizer)
    expected = optimizer.state[param]["exp_avg_sq"].mean()
    _kill(conn, rewire, torch.arange(K))
    assert rewire.step(optimizer) == K
    state = optimizer.state[param]
    assert torch.allclose(state["exp_avg_sq"], expected.expand(K))
    assert (state["exp_avg"] == 0).all()


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
    (conn(torch.randn(2, N_PRE)) * 0).sum().backward()  # a zero gradient
    optimizer.step()
    theta_new = (conn.weight.value * conn.weight.sign).detach()
    assert torch.allclose(theta_new, theta - lr * 0.5, atol=1e-6)

    # A manual loop gets the same term from ``regularize``; ``step`` alone is
    # purely structural and leaves the weights of active slots alone.
    rewire.detach()
    rewire.step(optimizer)
    assert torch.equal((conn.weight.value * conn.weight.sign).detach(), theta_new)
    rewire.regularize(optimizer)
    theta_manual = (conn.weight.value * conn.weight.sign).detach()
    assert torch.allclose(theta_manual, theta - 2 * lr * 0.5, atol=1e-6)
    with pytest.raises(ValueError, match="needs an optimizer"):
        rewire.regularize()  # detached: no optimizer to take ``lr`` from

    big = SparseConnection.from_adjacency(
        _adjacency(200, 200, 20000, dale=False), Synapse()
    )
    rewire = HardDeepR(big, HardDeepROptions(noise=2.0, every=1000))
    optimizer = torch.optim.SGD(big.parameters(), lr=lr)
    rewire.attach(optimizer)
    before = big.weight.value.detach().clone()
    (big(torch.randn(2, 200)) * 0).sum().backward()
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
    """Invalid options, foreign optimizers and repeated ``attach``."""
    with pytest.raises(ValueError, match="optimizer_state"):
        HardDeepROptions(optimizer_state="transfer")
    with pytest.raises(ValueError, match="every"):
        HardDeepROptions(every=0)
    with pytest.raises(ValueError, match="max_tries"):
        HardDeepROptions(max_tries=-1)
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
    handle = rewire.attach(optimizer)
    assert rewire.attached
    with pytest.raises(RuntimeError, match="already attached"):
        rewire.attach(optimizer)
    assert len(optimizer._optimizer_step_post_hooks) == 1  # no second hook

    # A second controller for the same weights would apply l1 / noise twice
    # per step and race for the dormant slots: refused on the same optimizer.
    twin = HardDeepR(conn)
    with pytest.raises(RuntimeError, match="Another HardDeepR"):
        twin.attach(optimizer)

    # detach() is idempotent, and the controller can be attached again.
    rewire.detach()
    rewire.detach()
    assert not rewire.attached and len(optimizer._optimizer_step_post_hooks) == 0
    handle = rewire.attach(optimizer)
    # Removing the hook through the returned handle is noticed as well.
    handle.remove()
    assert not rewire.attached
    rewire.attach(optimizer)
    assert rewire.attached and len(optimizer._optimizer_step_post_hooks) == 1


# ----------------------------------------------------------- unplaced slots
def test_more_dormant_slots_than_free_positions_does_not_raise():
    """A nearly dense layer: place what fits, keep the rest dormant, retry.

    A 3x3 connection with 8 edges has one unconnected position. Two slots go
    dormant in the same optimizer step. The hook must not raise (that would
    kill a training run after the weights were already updated): one slot
    takes the free position, the other keeps its position with a weight of
    exactly zero. Positions vacated in an update become free at the next one,
    so the waiting slot then moves to where the first one used to be.
    """
    dense = torch.ones(3, 3)
    dense[2, 2] = 0.0
    conn = SparseConnection(dense.to_sparse(), Synapse(dale=True))
    rewire = HardDeepR(conn, generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.SGD(conn.parameters(), lr=0.1)
    rewire.attach(optimizer)
    old = conn.indices.clone()
    with torch.no_grad():
        conn.weight.value[:2] = -1.0  # far below zero: dormant after the step

    def train_step():
        optimizer.zero_grad()
        # Gradient -2 on every slot: the (positive) weights only grow, so
        # nothing but the two killed slots is ever dormant.
        (-conn(torch.ones(2, 3))).sum().backward()
        optimizer.step()

    with pytest.warns(RuntimeWarning, match="could not place 1 of 2"):
        train_step()
    assert rewire.n_rewired == 1 and rewire.n_unplaced == 1
    # Slot 0 took the only free position; slot 1 waits at its old position.
    assert conn.indices[:, 0].tolist() == [2, 2]
    assert torch.equal(conn.indices[:, 1:], old[:, 1:])
    assert float(conn.weight.value.detach()[1]) == 0.0
    assert rewire.dormant_slots().tolist() == [1]
    assert _keys(conn).unique().numel() == 8
    # The waiting slot contributes nothing to the forward pass.
    x = torch.randn(4, 3)
    assert torch.allclose(conn(x), x @ _dense(conn).T, atol=1e-6)
    assert _dense(conn)[old[0, 1], old[1, 1]] == 0

    # Next update: the position slot 0 left is free now, slot 1 takes it.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning: everything was placed
        train_step()
    assert rewire.n_rewired == 2 and rewire.n_unplaced == 0
    assert conn.indices[:, 1].tolist() == old[:, 0].tolist()
    assert rewire.dormant_slots().numel() == 0
    assert _keys(conn).unique().numel() == 8


def test_unplaced_slots_survive_a_checkpoint():
    """The waiting slots and the "already warned" flag are checkpointed.

    Same nearly dense layer as above, saved while one slot waits. In the
    restored run the slot's weight is zero in the connection's
    checkpoint, but only the controller knows that it is *waiting*:
    without that the next gradient step would revive the connection in
    place. The restored controller holds it at zero, places it at the
    next update (the position vacated before the save is free now) and
    does not warn a second time.
    """

    def build():
        dense = torch.ones(3, 3)
        dense[2, 2] = 0.0
        conn = SparseConnection(dense.to_sparse(), Synapse(dale=True))
        # ``every=2``: updates on even steps, so a step without one follows.
        rewire = HardDeepR(conn, HardDeepROptions(every=2))
        optimizer = torch.optim.SGD(conn.parameters(), lr=0.1)
        rewire.attach(optimizer)
        return conn, rewire, optimizer

    def train_step(conn, optimizer):
        optimizer.zero_grad()
        (-conn(torch.ones(2, 3))).sum().backward()  # weights grow
        optimizer.step()

    conn, rewire, optimizer = build()
    with torch.no_grad():
        conn.weight.value[:2] = -1.0
    with pytest.warns(RuntimeWarning, match="could not place"):
        train_step(conn, optimizer)
        train_step(conn, optimizer)  # step 2: the structural update
    assert rewire.n_unplaced == 1 and rewire.dormant_slots().tolist() == [1]
    saved = copy.deepcopy((conn.state_dict(), rewire.state_dict()))
    assert saved[1]["unplaced"].tolist() == [False, True] + [False] * 6

    conn, rewire, optimizer = build()
    conn.load_state_dict(saved[0])
    rewire.load_state_dict(saved[1])
    assert rewire.n_unplaced == 1 and rewire.dormant_slots().tolist() == [1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the warning was already shown
        train_step(conn, optimizer)  # step 3: no update, slot 1 held at zero
        assert float(conn.weight.value.detach()[1]) == 0.0
        assert rewire.n_unplaced == 1
        train_step(conn, optimizer)  # step 4: slot 1 is placed
    assert rewire.n_unplaced == 0 and rewire.dormant_slots().numel() == 0
    assert _keys(conn).unique().numel() == 8


@pytest.mark.parametrize("enumerate_free", [True, False])
def test_unplaced_slots_stay_dormant_until_a_position_opens(
    enumerate_free, monkeypatch
):
    """Unplaced slots are held at zero and retried; one warning in total.

    The ``candidate`` restriction first forbids every position, later allows
    them again. While a slot waits, the optimizer keeps producing a gradient
    for it and ``l1`` / ``noise`` are active; none of them may revive the
    connection in place, and its weight must be exactly zero after every
    step. ``enumerate_free=False`` shrinks the enumeration limit to zero,
    which is how a layer too large to enumerate behaves: rejection sampling
    gives up after ``max_tries`` rounds, equally without raising.
    """
    if not enumerate_free:
        monkeypatch.setattr("btorch.models.connection.rewire._ENUMERATE_MAX", 0)
    allow = [False]
    conn, rewire = _make(
        dale=True,
        candidate=lambda pre, post: torch.full_like(pre, allow[0], dtype=torch.bool),
        max_tries=3,
        l1=0.1,
        noise=0.1,
    )
    optimizer = torch.optim.SGD(conn.parameters(), lr=1e-3)
    rewire.attach(optimizer)
    slots = _kill(conn, rewire, [0, 7])
    indices, version = conn.indices.clone(), conn.topology_version

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            optimizer.zero_grad()
            conn(torch.randn(4, N_PRE)).sum().backward()
            assert conn.weight.value.grad[slots].abs().min() > 0
            optimizer.step()
            assert rewire.n_unplaced == 2 and rewire.n_rewired == 0
            assert (conn.weight.value[slots] == 0).all()
            assert rewire.dormant_slots().tolist() == slots.tolist()
    # One warning per controller, not one per step.
    assert len([w for w in caught if w.category is RuntimeWarning]) == 1
    # Nothing was placed, so the topology and its version are untouched.
    assert torch.equal(conn.indices, indices) and conn.topology_version == version

    # A manual ``step`` behaves the same way and reports the count.
    assert rewire.step() == 0 and rewire.n_unplaced == 2

    allow[0] = True
    optimizer.zero_grad()
    conn(torch.randn(4, N_PRE)).sum().backward()
    optimizer.step()
    assert rewire.n_unplaced == 0 and rewire.n_rewired == 2
    assert not torch.equal(conn.indices[:, slots], indices[:, slots])
    assert _keys(conn).unique().numel() == K


def test_exact_enumeration_is_uniform_and_fills_every_free_position():
    """With few free positions they are enumerated, not rejection-sampled.

    A 6x6 recurrent connection with 30 edges, no autapses and a ``candidate``
    that excludes one source leaves only a handful of admissible positions.
    Rewiring single slots must reach each of them equally often (same
    5-sigma band as the rejection-sampling test), and when more slots are
    dormant than positions exist exactly the admissible ones are filled.
    """
    n, k = 6, 30
    W = _adjacency(n, n, k, seed=2, dale=True)
    options = HardDeepROptions(
        allow_autapses=False, candidate=lambda pre, post: pre != 0
    )
    conn = SparseConnection.from_adjacency(W, Synapse(dale=True))
    rewire = HardDeepR(conn, options, generator=torch.Generator().manual_seed(0))
    start = copy.deepcopy(conn.state_dict())
    post, pre = torch.arange(n * n) // n, torch.arange(n * n) % n
    free = torch.ones(n * n, dtype=torch.bool)
    free[_keys(conn)] = False
    free &= (post != pre) & (pre != 0)
    n_free = int(free.sum())
    assert 2 <= n_free < 6

    counts = torch.zeros(n * n)
    n_draw = 300 * n_free
    for _ in range(n_draw):
        conn.load_state_dict(start)
        _kill(conn, rewire, [4])
        assert rewire.step() == 1
        counts[_keys(conn)[4]] += 1
    assert counts[~free].sum() == 0
    p = 1 / n_free
    sigma = (n_draw * p * (1 - p)) ** 0.5
    assert (counts[free] - n_draw * p).abs().max() < 5 * sigma

    # Ten dormant slots, n_free positions: exactly those get filled.
    conn.load_state_dict(start)
    slots = _kill(conn, rewire, torch.arange(10))
    with pytest.warns(RuntimeWarning, match="could not place"):
        assert rewire.step() == n_free
    assert rewire.n_unplaced == 10 - n_free
    placed = slots[conn.weight.value[slots] != 0]
    assert sorted(_keys(conn)[placed].tolist()) == free.nonzero()[:, 0].tolist()
    assert _keys(conn).unique().numel() == k


# ------------------------------------------------------------ checkpointing
def _resumable_run(n_steps, dale, tmp_path=None, split=None):
    """Train for ``n_steps``; optionally stop at ``split``, save and resume.

    Everything the run needs is rebuilt from scratch after the save, the way
    a restarted job would: connection from the *initial* matrix, a fresh
    optimizer, a fresh controller with a differently seeded generator. Only
    the three ``state_dict`` s carry information across.
    """
    # Inputs are generated up front so both runs see identical data.
    data = torch.Generator().manual_seed(11)
    xs = [torch.randn(8, N_PRE, generator=data) for _ in range(n_steps)]
    target = torch.randn(N_POST, N_PRE, generator=data)

    def build(seed):
        # ``k=20`` leaves sources without any edge, so with Dale's law the
        # per-source sign table uses its random fallback; it differs between
        # generators and must therefore come from the checkpoint.
        W = _adjacency(k=20, seed=4, dale=dale)
        conn = SparseConnection.from_adjacency(W, Synapse(dale=dale))
        optimizer = torch.optim.Adam(conn.parameters(), lr=0.05)
        rewire = HardDeepR(
            conn,
            HardDeepROptions(l1=1e-2, noise=1e-3, every=2),
            generator=torch.Generator().manual_seed(seed),
        )
        rewire.attach(optimizer)
        return conn, optimizer, rewire

    conn, optimizer, rewire = build(seed=5)
    for i, x in enumerate(xs):
        if i == split:
            torch.save(
                {
                    "conn": conn.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rewire": rewire.state_dict(),
                },
                tmp_path / "ckpt.pt",
            )
            conn, optimizer, rewire = build(seed=99)
            ckpt = torch.load(tmp_path / "ckpt.pt")
            conn.load_state_dict(ckpt["conn"])
            optimizer.load_state_dict(ckpt["optimizer"])
            rewire.load_state_dict(ckpt["rewire"])
        optimizer.zero_grad()
        (conn(x) - x @ target.T).pow(2).mean().backward()
        optimizer.step()
    return conn, rewire


@pytest.mark.parametrize("dale", [True, False])
def test_checkpoint_resume_reproduces_uninterrupted_run(dale, tmp_path):
    """``k`` steps, save, rebuild, load, ``k`` steps == ``2k`` steps, exactly.

    The controller's state that the connection and the optimizer do not
    hold: the generator (noise, new positions, random signs), the per-source
    sign table, the per-slot reference signs without Dale's law, the step
    counter that drives ``every`` and the unplaced-slot mask.
    """
    k = 40
    conn_a, rewire_a = _resumable_run(2 * k, dale)
    conn_b, rewire_b = _resumable_run(2 * k, dale, tmp_path, split=k)
    assert rewire_a.n_rewired > 10  # the run really rewired, before and after
    assert rewire_b.n_steps == rewire_a.n_steps == 2 * k
    assert rewire_b.n_rewired == rewire_a.n_rewired
    assert torch.equal(conn_b.indices, conn_a.indices)
    assert torch.equal(conn_b.weight.value, conn_a.weight.value)
    assert torch.equal(rewire_b._signs(), rewire_a._signs())
    # The generators are in the same state: the next draws agree.
    assert torch.equal(
        torch.rand(4, generator=rewire_b.generator),
        torch.rand(4, generator=rewire_a.generator),
    )


def test_state_dict_contents_and_mismatch():
    """The state is a plain dict of tensors / ints and is validated on load."""
    conn, rewire = _make(dale=False, sign="pre")
    state = rewire.state_dict()
    assert set(state) == {
        "n_steps",
        "n_rewired",
        "n_unplaced",
        "warned",
        "unplaced",
        "pre_sign",
        "slot_sign",
        "generator",
    }
    assert all(isinstance(v, int | torch.Tensor) for v in state.values())
    assert state["pre_sign"].shape == (N_PRE,) and state["slot_sign"].shape == (K,)
    # The saved tensors are copies: later rewiring does not change them.
    saved = state["slot_sign"].clone()
    _kill(conn, rewire, torch.arange(K))
    rewire.step()
    assert torch.equal(state["slot_sign"], saved)
    rewire.load_state_dict(state)
    assert torch.equal(rewire._slot_sign, saved) and rewire.n_rewired == 0

    # A controller with another configuration saves other entries
    # (Dale's law: no slot signs; no generator: no generator state).
    other = HardDeepR(SparseConnection.from_adjacency(_adjacency(), Synapse(dale=True)))
    assert set(other.state_dict()) == set(state) - {"slot_sign", "generator"}
    with pytest.raises(ValueError, match="does not match"):
        other.load_state_dict(state)
    # A checkpoint of a connection with another number of slots is refused.
    small = SparseConnection.from_adjacency(_adjacency(k=20, dale=False), Synapse())
    with pytest.raises(ValueError, match="slot_sign|unplaced"):
        HardDeepR(
            small, HardDeepROptions(sign="pre"), generator=torch.Generator()
        ).load_state_dict(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_state_dict_roundtrip_with_cuda_generator():
    """A CUDA generator's state is saved and restored as well."""
    conn = SparseConnection.from_adjacency(_adjacency(), Synapse(dale=True)).cuda()
    rewire = HardDeepR(conn, generator=torch.Generator("cuda").manual_seed(3))
    state = rewire.state_dict()
    _kill(conn, rewire, [0, 1, 2])
    rewire.step()
    first = conn.indices.clone()
    start = SparseConnection.from_adjacency(_adjacency(), Synapse(dale=True)).cuda()
    conn.load_state_dict(start.state_dict())
    rewire.load_state_dict(state)
    _kill(conn, rewire, [0, 1, 2])
    rewire.step()
    assert torch.equal(conn.indices, first)


# ------------------------------------------------------- hook corner cases
def test_step_without_gradient_and_second_optimizer():
    """Steps in which the weight has no gradient, and a second optimizer.

    ``.grad is None`` means the optimizer skips the parameter entirely (no
    update, no weight decay, no state). The controller treats ``l1`` and
    ``noise`` the same way, so a connection that is not part of the current
    loss does not drift, shrink or get pruned; the structural update itself
    still runs on schedule and must cope with the missing optimizer state.
    """
    conn, rewire = _make(dale=True, l1=0.5, noise=0.5)
    param = conn.weight.value
    optimizer = torch.optim.Adam(conn.parameters(), lr=0.1)
    rewire.attach(optimizer)
    before = param.detach().clone()
    assert param.grad is None
    optimizer.step()
    assert rewire.n_steps == 1 and torch.equal(param.detach(), before)
    assert param not in optimizer.state  # Adam created no state

    # Dormant slots are still rewired in such a step, without state to fix.
    slots = _kill(conn, rewire, [3, 4])
    optimizer.step()
    assert rewire.n_rewired == 2 and rewire.dormant_slots().numel() == 0
    assert param not in optimizer.state

    # The weight in two optimizers: only the attached one follows the policy.
    # The other keeps the moments of the removed connections unless it is
    # passed to a manual ``step`` (which has nothing left to rewire here).
    conn, rewire = _make(dale=True, optimizer_state="reset")
    param = conn.weight.value
    first = torch.optim.Adam([param], lr=1e-3)
    second = torch.optim.Adam([param], lr=1e-3)
    for optimizer in (first, second):
        _train_steps(conn, optimizer)
    rewire.attach(first)
    slots = _kill(conn, rewire, [2, 9])
    kept = second.state[param]["exp_avg_sq"].clone()
    _train_steps(conn, first, n=1)
    assert rewire.n_rewired == 2
    assert (first.state[param]["exp_avg_sq"][slots] == 0).all()
    assert torch.equal(second.state[param]["exp_avg_sq"], kept)


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
    # The loop above rewires many slots at once, which enumerates the free
    # positions per (receptor, delay) channel. Single slots take the
    # rejection-sampling path; it must respect the channels just the same.
    for slot in range(K):
        _kill(conn, rewire, [slot])
        assert rewire.step() == 1
        t = conn.edge_table()
        row = t["post"] * conn.n_receptor + t["receptor"]
        col = t["pre"] * conn.n_delay + t["delay"]
        assert (row * conn.in_features + col).unique().numel() == K
    assert torch.equal(conn.receptor, receptor) and torch.equal(conn.delay, delay)
