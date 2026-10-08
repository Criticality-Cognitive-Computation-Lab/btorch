"""Model state of ``SparseConnection``: ``state_dict``, checkpoints, copies.

Spec sections 34 and 38: the persistent state is the *canonical* description
of the network (edge list, routing ids, weight state). Everything used for
execution (compressed layouts, plans, backends) is derived and must be
rebuilt, never serialised and never left stale.

The central scenario: a module built for network B receives the checkpoint of
network A (another pattern with the same number of edges, other weights,
groups, signs, receptors, delays). Afterwards it must *be* network A; this is
checked against a dense matrix of A written down independently.
"""

import copy
import io
import pickle
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from btorch.models.connection import ConstrainedWeight, SparseConnection, Synapse
from btorch.models.constrain import constrain_net
from tests.sparse.helpers import DEVICES


N_PRE, N_POST, N_RECEPTOR, N_DELAY, G = 5, 4, 2, 3, 2
# Two networks with 8 distinct edges each (input-edge order, unsorted).
NET_A = SimpleNamespace(
    pre=torch.tensor([4, 0, 2, 1, 3, 0, 4, 2]),
    post=torch.tensor([0, 3, 1, 1, 2, 0, 3, 3]),
    w=torch.tensor([0.5, -1.0, 2.0, -0.25, 1.5, 3.0, -2.0, 0.75]),
    const=0.5,
    group=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
    scale=torch.tensor([2.0, 0.5, 1.5]),
    receptor=torch.tensor([0, 1, 1, 0, 0, 1, 0, 1]),
    delay=torch.tensor([2, 0, 1, 1, 0, 2, 0, 1]),
)
NET_B = SimpleNamespace(
    pre=torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]),
    post=torch.tensor([0, 0, 0, 1, 1, 2, 2, 3]),
    w=torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
    const=2.0,
    group=torch.tensor([2, 2, 1, 1, 0, 0, 2, 1]),
    scale=torch.tensor([1.0, 1.0, 1.0]),
    receptor=torch.tensor([1, 0, 0, 1, 1, 0, 1, 0]),
    delay=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
)
N_EDGE = 8
for _net in (NET_A, NET_B):
    _net.w_batch = torch.stack([_net.w, _net.w.flip(0) * 0.5])  # [G, E]

# kind -> persistent keys that must be present besides "indices".
KINDS = {
    "edge": ["weight.value"],
    "fixed": ["weight.value"],
    "constant": ["weight.value"],
    "dale": ["weight.value", "weight.sign"],
    "constrained": ["weight.scale", "weight.group", "weight.base"],
    "batched": ["weight.value"],
    "routed": ["receptor", "delay", "weight.value"],
    "routed_constrained": [
        "receptor",
        "delay",
        "weight.scale",
        "weight.group",
        "weight.base",
    ],
}


def build(kind, net):
    """Connection of ``kind`` for the network description ``net``."""
    routing = {}
    if kind.startswith("routed"):
        routing = {
            "receptor": net.receptor,
            "delay": net.delay,
            "n_receptor": N_RECEPTOR,
            "n_delay": N_DELAY,
        }
    weight, values = None, net.w
    if kind == "fixed":
        weight = net.w
    elif kind == "constant":
        weight = net.const
    elif kind.endswith("constrained"):
        weight = ConstrainedWeight(net.group, scale=net.scale, dale=True)
    elif kind == "batched":
        values = net.w_batch
    synapse = Synapse(weight=weight, dale=kind == "dale", **routing)
    return SparseConnection.from_edges(
        net.pre, net.post, N_PRE, N_POST, synapse, values=values
    )


def dense_of(kind, net):
    """Dense operator ``[*batch, out_features, in_features]`` of ``net``.

    Receptors expand the output (``post * n_receptor + receptor``), delays
    the input (``pre * n_delay + delay``): the layout a routed connection
    exposes. Written with plain indexing, independent of the library.
    """
    weight = net.w
    if kind == "constant":
        weight = torch.full((N_EDGE,), net.const)
    elif kind.endswith("constrained"):
        weight = net.w * net.scale[net.group]
    elif kind == "batched":
        weight = net.w_batch
    row, col, shape = net.post, net.pre, (N_POST, N_PRE)
    if kind.startswith("routed"):
        row = net.post * N_RECEPTOR + net.receptor
        col = net.pre * N_DELAY + net.delay
        shape = (N_POST * N_RECEPTOR, N_PRE * N_DELAY)
    dense = torch.zeros(*weight.shape[:-1], *shape)
    dense[..., row, col] = weight
    return dense


def check_is_network(conn, kind, net, device="cpu", dtype=torch.float32):
    """``conn`` computes exactly the dense operator of ``net``."""
    dense = dense_of(kind, net).to(device=device, dtype=dtype)
    gen = torch.Generator().manual_seed(0)
    shape = (G, 3, dense.shape[-1]) if kind == "batched" else (3, dense.shape[-1])
    x = torch.randn(shape, generator=gen).to(device=device, dtype=dtype)
    expected = torch.einsum("...mn,...bn->...bm", dense, x)
    torch.testing.assert_close(conn(x), expected, atol=1e-5, rtol=1e-5)


def roundtrip(state_dict):
    """Serialise to bytes and back with the safe (tensors-only) loader."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


# --------------------------------------------------------- state_dict content
@pytest.mark.parametrize("kind", KINDS)
def test_state_dict_holds_canonical_state_only(kind):
    """Exactly the canonical edge list, routing ids and weight state; no
    derived layout (``cache.*``), plan or backend object."""
    conn = build(kind, NET_A)
    state = conn.state_dict()
    assert {"indices", *KINDS[kind]} <= set(state)
    assert all(isinstance(v, torch.Tensor) for v in state.values())
    assert not any("cache" in key or "plan" in key for key in state)
    # The derived layouts exist as buffers, they are just not persistent.
    assert {n for n, _ in conn.named_buffers() if n.startswith("cache.")} == {
        f"cache.{n}" for n in ("crow", "col", "perm", "t_crow", "t_col", "t_perm")
    }
    # ``indices`` are un-expanded neuron ids [post, pre] in canonical order.
    assert state["indices"].shape == (2, N_EDGE)
    assert int(state["indices"][0].max()) < N_POST
    assert int(state["indices"][1].max()) < N_PRE
    edges = set(zip(state["indices"][1].tolist(), state["indices"][0].tolist()))
    assert edges == set(zip(NET_A.pre.tolist(), NET_A.post.tolist()))


# ------------------------------------------------------------- save and load
@pytest.mark.parametrize("kind", KINDS)
def test_checkpoint_of_another_pattern_determines_the_forward(kind):
    """Load network A's checkpoint into a module built for network B (same edge
    count, different edges / weights / groups / signs / routing)."""
    target = build(kind, NET_B)
    check_is_network(target, kind, NET_B)
    result = target.load_state_dict(roundtrip(build(kind, NET_A).state_dict()))
    assert not result.missing_keys and not result.unexpected_keys
    check_is_network(target, kind, NET_A)
    # ... and reading the model back gives A as well, not a stale B.
    lowered = target.to_sparse("post_pre").to_dense()
    torch.testing.assert_close(lowered, dense_of(kind, NET_A))


@pytest.mark.parametrize("order", ["load_then_move", "move_then_load"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["edge", "constant", "constrained", "routed"])
def test_checkpoint_and_device_dtype_moves(kind, device, order):
    """The same holds when the module is moved to another device and dtype
    before or after loading: the rebuilt layouts live where the module is."""
    checkpoint = roundtrip(build(kind, NET_A).state_dict())  # CPU, float32
    target = build(kind, NET_B)
    if order == "move_then_load":
        target.to(device=device, dtype=torch.float64)
    target.load_state_dict(checkpoint)
    if order == "load_then_move":
        target.to(device=device, dtype=torch.float64)
    assert {b.device.type for b in target.buffers()} == {device}
    assert target.weight().dtype == torch.float64
    check_is_network(target, kind, NET_A, device, torch.float64)


class Wrapper(nn.Module):
    """A model that owns a connection, as real networks do (``psc.linear``)."""

    def __init__(self, conn):
        super().__init__()
        self.readout = nn.Linear(conn.out_features, 2)
        self.conn = conn


@pytest.mark.parametrize("kind", ["edge", "routed_constrained"])
def test_checkpoint_loaded_through_a_parent_module(kind):
    """Loading the *parent's* checkpoint rebuilds the nested connection."""
    source, target = Wrapper(build(kind, NET_A)), Wrapper(build(kind, NET_B))
    version = target.conn.topology_version
    target.load_state_dict(roundtrip(source.state_dict()))
    assert target.conn.topology_version > version
    check_is_network(target.conn, kind, NET_A)


def test_dale_signs_come_from_the_checkpoint():
    """Dale's reference signs are model state: after loading A into B the
    projection protects A's signs, not the ones B was built with."""
    target = build("dale", NET_B)
    target.load_state_dict(roundtrip(build("dale", NET_A).state_dict()))
    post, pre = target.indices
    sign_a = torch.sign(dense_of("dale", NET_A))[post, pre]
    assert torch.equal(target.weight.sign, sign_a)
    with torch.no_grad():  # flip every second weight, as a bad update would
        target.weight.value[::2].neg_()
    constrain_net(target)
    expected = dense_of("dale", NET_A)[post, pre]
    expected[::2] = 0.0  # flipped weights are clamped to zero, others kept
    assert torch.equal(target.weight().detach(), expected)


def test_constraint_base_and_group_come_from_the_checkpoint():
    target = build("constrained", NET_B)
    target.load_state_dict(roundtrip(build("constrained", NET_A).state_dict()))
    post, pre = target.indices
    # Dense lookups of A's per-edge base weight and group id.
    base = dense_of("edge", NET_A)[post, pre]
    group = torch.zeros(N_POST, N_PRE, dtype=torch.long)
    group[NET_A.post, NET_A.pre] = NET_A.group
    assert torch.equal(target.weight.base, base)
    assert torch.equal(target.weight.group, group[post, pre])
    assert torch.equal(target.weight.scale.detach(), NET_A.scale)
    assert torch.equal(target.edge_table()["group"], group[post, pre])


def test_load_rejects_a_checkpoint_with_another_edge_count():
    """A different number of edges is a shape mismatch, reported by PyTorch as
    usual instead of being loaded half-way."""
    smaller = SparseConnection.from_edges(NET_A.pre[:5], NET_A.post[:5], N_PRE, N_POST)
    target = build("edge", NET_B)
    with pytest.raises(RuntimeError, match="size mismatch"):
        target.load_state_dict(smaller.state_dict())
    check_is_network(target, "edge", NET_B)  # the failed load changed nothing


def test_load_rejects_routing_ids_outside_the_module():
    """Ids that do not fit the module's receptor channels cannot be routed;
    loading them must fail loudly instead of aliasing onto another neuron."""
    pre, post = torch.tensor([0, 1, 2, 3]), torch.tensor([1, 0, 2, 2])

    def build_with(receptor, n_receptor):
        synapse = Synapse(receptor=torch.tensor(receptor), n_receptor=n_receptor)
        return SparseConnection.from_edges(pre, post, 4, 3, synapse)

    three_channels = build_with([2, 0, 1, 0], 3)
    two_channels = build_with([0, 1, 1, 0], 2)
    with pytest.raises((ValueError, RuntimeError)):
        two_channels.load_state_dict(three_channels.state_dict())


# ----------------------------------------------------------- version counters
def test_version_counters():
    """``topology_version`` counts structural changes only (spec section 34):

    loading a checkpoint and rewiring increment it, a weight update does
    not. The derived layouts always carry the version they were built
    for. ``value_version`` is the complementary counter: it is derived from
    the in-place version counters of the weight tensors, so it moves on every
    weight write (and cannot be assigned).
    """
    conn = build("routed", NET_B)
    assert conn.topology_version == conn.cache.topology_version == 0

    # 1. An optimizer step changes values, not topology.
    optimizer = torch.optim.SGD(conn.parameters(), lr=0.1)
    layouts = {n: b.clone() for n, b in conn.cache.named_buffers()}
    values_before = conn.value_version
    conn(torch.ones(N_PRE * N_DELAY)).sum().backward()
    assert conn.value_version == values_before  # forward/backward write nothing
    optimizer.step()
    assert conn.topology_version == conn.routing_version == 0
    assert conn.value_version > values_before
    assert all(torch.equal(b, layouts[n]) for n, b in conn.cache.named_buffers())
    with pytest.raises(AttributeError):
        conn.value_version = 0  # a read-only property, not a counter to bump

    # 2. Loading a checkpoint may change the edges: version + rebuild.
    conn.load_state_dict(build("routed", NET_A).state_dict())
    assert conn.topology_version == conn.cache.topology_version == 1
    assert conn.routing_version == 1
    assert "topology_version = 1" in conn.explain()
    check_is_network(conn, "routed", NET_A)

    # 3. Rewiring two slots in place: version + rebuild, shapes unchanged.
    #    Slots are canonical positions; move them to coordinates A lacks.
    slots = torch.tensor([0, 5])
    new_post, new_pre = torch.tensor([2, 2]), torch.tensor([0, 1])
    new_receptor, new_delay = torch.tensor([1, 0]), torch.tensor([2, 2])
    weight = conn.weight().detach().clone()
    post, pre = conn.indices.clone()
    receptor, delay = conn.receptor.clone(), conn.delay.clone()
    conn.set_edges_(
        slots, pre=new_pre, post=new_post, receptor=new_receptor, delay=new_delay
    )
    assert conn.topology_version == conn.cache.topology_version == 2
    assert conn.routing_version > 1 and conn.nnz == N_EDGE
    post[slots], pre[slots] = new_post, new_pre
    receptor[slots], delay[slots] = new_receptor, new_delay
    dense = torch.zeros(N_POST * N_RECEPTOR, N_PRE * N_DELAY)
    dense[post * N_RECEPTOR + receptor, pre * N_DELAY + delay] = weight
    x = torch.randn(3, N_PRE * N_DELAY)
    torch.testing.assert_close(conn(x), x @ dense.T, atol=1e-6, rtol=1e-6)


# ------------------------------------------------------------ copy and pickle
@pytest.mark.parametrize("how", ["deepcopy", "pickle", "torch_save_module"])
@pytest.mark.parametrize("kind", KINDS)
def test_copy_and_pickle_round_trip(kind, how):
    """A connection can be deep-copied and pickled as a whole module; the copy
    computes the same network and shares no state with the original."""
    original = build(kind, NET_A)
    if how == "deepcopy":
        clone = copy.deepcopy(original)
    elif how == "pickle":
        clone = pickle.loads(pickle.dumps(original))
    else:
        buffer = io.BytesIO()
        torch.save(original, buffer)
        buffer.seek(0)
        clone = torch.load(buffer, weights_only=False)
    assert type(clone) is SparseConnection
    assert sorted(clone.state_dict()) == sorted(original.state_dict())
    assert clone.explain() == original.explain()
    check_is_network(clone, kind, NET_A)

    # Independence: turning the clone into network B (weights *and* derived
    # layouts) leaves the original untouched, and the clone still trains.
    clone.load_state_dict(build(kind, NET_B).state_dict())
    check_is_network(clone, kind, NET_B)
    check_is_network(original, kind, NET_A)
    parameters = list(clone.parameters())
    if parameters:
        x = torch.ones(clone.in_features).expand(*clone.batch_shape, 2, -1)
        clone(x).sum().backward()
        assert all(p.grad is not None for p in parameters)
        assert all(p.grad is None for p in original.parameters())
