"""Receptors and delays as edge attributes (spec sections 17, 18 and 43).

The legacy helpers encode receptors and delays *physically*: a ``src_dst``
matrix with rows ``pre * n_delay + delay`` and columns
``post * n_receptor + receptor``. The connection layer stores the semantic
edge list ``(pre, post, weight, receptor, delay)`` instead and treats that
expansion as one possible lowering. Three ways to build the same network must
therefore produce identical currents:

(a) ``from_adjacency(expanded)``: the old pathway, the expansion taken at
    face value (a plain matrix between expanded populations);
(b) ``from_hetersynapse(expanded, n_receptor=, n_delay=)``: the adapter that
    decodes the expansion into edge attributes;
(c) ``from_edges(pre, post, ..., Synapse(receptor=, delay=))``: no expansion
    anywhere in user code.

Each is compared with a dense matrix scattered here from the semantic edges.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from torch import nn

from btorch.connectome.connection import (
    expand_conn_for_delays,
    make_hetersynapse_conn,
    make_hetersynapse_constrained_conn,
)
from btorch.models import environ
from btorch.models.connection import ConstrainedWeight, SparseConnection, Synapse
from btorch.models.functional import init_net_state
from btorch.models.history import SpikeHistory
from btorch.models.synapse import ExponentialPSC, HeterSynapsePSC


N, N_BINS = 6, 5
NEURONS = pd.DataFrame(
    {
        "root_id": np.arange(100, 100 + N),
        "simple_id": np.arange(N),
        "cell_type": ["a", "b", "a", "b", "c", "c"],
        "EI": ["E", "I", "E", "E", "I", "E"],  # receptor type in neuron mode
    }
)
# Ten connections between distinct (pre, post) pairs, every E/I combination
# present, delays within [0, N_BINS - 1].
_PRE = np.array([0, 0, 1, 2, 3, 4, 5, 5, 2, 1])
_POST = np.array([1, 2, 3, 0, 5, 2, 0, 4, 4, 4])
CONNS = pd.DataFrame(
    {
        "pre_simple_id": _PRE,
        "post_simple_id": _POST,
        "pre_root_id": _PRE + 100,
        "post_root_id": _POST + 100,
        "syn_count": np.arange(1, 11),
        "delay": [0, 1, 2, 3, 1, 0, 2, 4, 1, 3],
        # Transmitter of the connection: receptor type in connection mode.
        "nt": ["ach", "gaba", "ach", "glu", "gaba", "ach", "glu", "ach", "gaba", "glu"],
    }
)
# Cotransmission: the pairs 0->1 and 5->4 additionally release a second
# transmitter, with another strength and another delay.
CONNS_CO = pd.concat(
    [
        CONNS,
        CONNS.iloc[[0, 7]].assign(nt=["gaba", "glu"], syn_count=[20, 30], delay=[3, 0]),
    ],
    ignore_index=True,
)


def scenario(name):
    """One reference network: the legacy expanded matrix and, written down
    independently, its semantic edge list.

    Returns a dict with ``expanded`` (SciPy, ``src_dst``), ``n_receptor``,
    ``n_delay``, the per-edge arrays ``pre``, ``post``, ``weight``,
    ``receptor``, ``delay`` and the receptor index frame ``index``.
    """
    mode, _, delayed = name.partition("+")
    conns = CONNS_CO if mode == "connection" else CONNS
    delay_kw = {"delay_col": "delay", "n_delay_bins": N_BINS} if delayed else {}
    pre, post = conns["pre_simple_id"].to_numpy(), conns["post_simple_id"].to_numpy()
    if mode == "delay_only":
        base = sp.coo_array((conns["syn_count"], (pre, post)), shape=(N, N))
        expanded = expand_conn_for_delays(base, conns["delay"].to_numpy(), N_BINS)
        index, receptor = None, np.zeros(len(conns), dtype=int)
    elif mode == "neuron":
        expanded, index = make_hetersynapse_conn(
            NEURONS, conns, "EI", "neuron", **delay_kw
        )
        ei = NEURONS["EI"].to_numpy()
        lookup = dict(
            zip(
                zip(index["pre_receptor_type"], index["post_receptor_type"]),
                index.index,
            )
        )
        receptor = np.array([lookup[ei[i], ei[j]] for i, j in zip(pre, post)])
    else:
        expanded, index = make_hetersynapse_conn(
            NEURONS, conns, "nt", "connection", **delay_kw
        )
        lookup = dict(zip(index["receptor_type"], index.index))
        receptor = conns["nt"].map(lookup).to_numpy()
    has_delay = bool(delayed) or mode == "delay_only"
    return {
        "expanded": expanded,
        "index": index,
        "n_receptor": 1 if index is None else len(index),
        "n_delay": N_BINS if has_delay else 1,
        "pre": torch.as_tensor(pre),
        "post": torch.as_tensor(post),
        "weight": torch.as_tensor(conns["syn_count"].to_numpy(), dtype=torch.float32),
        "receptor": torch.as_tensor(receptor),
        "delay": torch.as_tensor(conns["delay"].to_numpy() if has_delay else 0 * pre),
    }


def dense_operator(s, weight=None):
    """Dense ``[N * n_receptor, N * n_delay]`` operator from the semantic.

    edges: output ``post * n_receptor + receptor``, input
    ``pre * n_delay + delay`` (plain scatter-add, no library code).
    """
    weight = s["weight"] if weight is None else weight
    dense = torch.zeros(N * s["n_receptor"], N * s["n_delay"], dtype=weight.dtype)
    row = s["post"] * s["n_receptor"] + s["receptor"]
    col = s["pre"] * s["n_delay"] + s["delay"]
    return dense.index_put((row, col), weight, accumulate=True)


def pathways(s, weight=None, group=None):
    """The three constructions (a), (b), (c) of the module docstring.

    Args:
        weight: Optional factory of a weight module for the matrix pathways.
        group: Per-edge 0-based group ids for the semantic pathway.
    """
    n_receptor, n_delay = s["n_receptor"], s["n_delay"]
    make = (lambda: Synapse(weight=weight())) if weight else Synapse
    semantic_weight = None
    if group is not None:
        semantic_weight = ConstrainedWeight(group, n_group=int(group.max()) + 1)
    synapse = Synapse(
        weight=semantic_weight,
        receptor=s["receptor"] if n_receptor > 1 else None,
        delay=s["delay"] if n_delay > 1 else None,
        n_receptor=n_receptor if n_receptor > 1 else None,
        n_delay=n_delay if n_delay > 1 else None,
    )
    return {
        "expanded": SparseConnection.from_adjacency(s["expanded"], make()),
        "adapter": SparseConnection.from_hetersynapse(
            s["expanded"], make(), n_receptor=n_receptor, n_delay=n_delay
        ),
        "semantic": SparseConnection.from_edges(
            s["pre"], s["post"], N, N, synapse, values=s["weight"]
        ),
    }


def edge_rows(table):
    """Edge table as a sorted list of ``(pre, post, receptor, delay, w)``."""
    n_edge = table["pre"].shape[0]
    zeros = torch.zeros(n_edge, dtype=torch.long)
    columns = [table["pre"], table["post"], table.get("receptor", zeros)]
    columns += [table.get("delay", zeros), table["weight"].detach()]
    return sorted(zip(*(c.tolist() for c in columns)))


SCENARIOS = ["neuron", "neuron+delay", "connection", "connection+delay", "delay_only"]


# ------------------------------------------------- old and new pathways agree
@pytest.mark.parametrize("name", SCENARIOS)
def test_three_pathways_give_identical_currents(name):
    s = scenario(name)
    n_in, n_out = N * s["n_delay"], N * s["n_receptor"]
    assert s["expanded"].shape == (n_in, n_out)  # the legacy layout itself
    dense = dense_operator(s)
    # The helper's matrix is this very expansion of the semantic edges.
    assert np.array_equal(s["expanded"].toarray(), dense.T.numpy())

    x = torch.randn(2, 3, n_in, generator=torch.Generator().manual_seed(0))
    expected = x @ dense.T
    conns = pathways(s)
    for label, conn in conns.items():
        assert (conn.in_features, conn.out_features) == (n_in, n_out), label
        torch.testing.assert_close(conn(x), expected, msg=label)
        # Every pathway lowers to the same expanded operator.
        assert torch.equal(conn.to_sparse().to_dense(), dense), label
    assert torch.equal(conns["adapter"](x), conns["semantic"](x))


@pytest.mark.parametrize("name", SCENARIOS)
def test_semantic_buffers_hold_unexpanded_neuron_ids(name):
    """The adapter and the semantic pathway store neuron ids plus per-edge
    receptor / delay ids; only the old pathway knows expanded populations."""
    s = scenario(name)
    conns = pathways(s)
    reference = edge_rows(
        {k: s[k] for k in ("pre", "post", "receptor", "delay", "weight")}
    )
    for label in ("adapter", "semantic"):
        conn = conns[label]
        assert (conn.n_pre, conn.n_post) == (N, N)
        assert (conn.n_receptor, conn.n_delay) == (s["n_receptor"], s["n_delay"])
        assert conn.indices.shape == (2, len(reference)) and int(conn.indices.max()) < N
        assert edge_rows(conn.edge_table()) == reference, label
        # Routing ids are persistent model state exactly when they are used.
        state = conn.state_dict()
        assert ("receptor" in state) == (s["n_receptor"] > 1)
        assert ("delay" in state) == (s["n_delay"] > 1)
        if s["n_receptor"] > 1:
            assert torch.equal(conn.receptor, conn.edge_table()["receptor"])
            assert int(conn.receptor.max()) < s["n_receptor"]
        if s["n_delay"] > 1:
            assert int(conn.delay.max()) < s["n_delay"]
    for key in ("indices", "receptor", "delay"):
        a, b = getattr(conns["adapter"], key), getattr(conns["semantic"], key)
        assert (a is None and b is None) or torch.equal(a, b)
    old = conns["expanded"]
    assert (old.n_pre, old.n_post) == (N * s["n_delay"], N * s["n_receptor"])
    assert old.receptor is None and old.delay is None


def test_cotransmission_and_multiple_delays_stay_separate_edges():
    """Edges are identified by ``(pre, post, receptor, delay)``: the same pair
    with two receptors, or with two delays, is two edges; only exact duplicates
    are summed."""
    pre = torch.tensor([0, 0, 0, 1, 1])
    post = torch.tensor([1, 1, 1, 0, 0])
    receptor = torch.tensor([0, 1, 1, 0, 0])
    delay = torch.tensor([0, 0, 0, 1, 2])
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    conn = SparseConnection.from_edges(
        pre, post, 2, 2, Synapse(receptor=receptor, delay=delay), values=weight
    )
    assert conn.nnz == 4  # entries 1 and 2 are the same edge: 2 + 3
    assert (conn.n_receptor, conn.n_delay) == (2, 3)  # default: largest id + 1
    assert edge_rows(conn.edge_table()) == [
        (0, 1, 0, 0, 1.0),  # 0 -> 1 on receptor 0 ...
        (0, 1, 1, 0, 5.0),  # ... and on receptor 1 (cotransmission)
        (1, 0, 0, 1, 4.0),  # 1 -> 0 with delay 1 ...
        (1, 0, 0, 2, 5.0),  # ... and with delay 2
    ]
    # Dense [n_post * 2 receptors, n_pre * 3 delays], duplicates summed.
    dense = torch.zeros(4, 6).index_put(
        (post * 2 + receptor, pre * 3 + delay), weight, accumulate=True
    )
    x = torch.randn(4, 6)
    torch.testing.assert_close(conn(x), x @ dense.T)
    # A scalar id applies to every edge; n_* reserves unused channels / bins.
    conn = SparseConnection.from_edges(
        pre[:1], post[:1], 2, 2, Synapse(receptor=1, delay=2, n_receptor=3, n_delay=4)
    )
    assert (conn.in_features, conn.out_features) == (8, 6)
    assert conn(torch.eye(8)).nonzero().tolist() == [[2, 4]]  # in 0*4+2 -> out 1*3+1


# ------------------------------------------------------- constrained weights
@pytest.mark.parametrize("mode", ["full", "cell_only", "cell_and_receptor"])
def test_constrained_weights_through_hetersynapse_constraint(mode):
    """``make_hetersynapse_constrained_conn`` returns the expanded matrix and a
    matrix of 1-based group ids in the same layout. Both matrix pathways locate
    the group of every edge by coordinate; the semantic pathway gets.

    plain per-edge ids. Currents and group-scale gradients agree with
    ``w[e] = base[e] * scale[group[e]]``.
    """
    s = scenario("neuron")
    conn_mat, constraint_mat, index = make_hetersynapse_constrained_conn(
        NEURONS, CONNS, constraint_mode=mode
    )
    assert np.array_equal(conn_mat.toarray(), s["expanded"].toarray())
    n_receptor = len(index)
    ids = constraint_mat.toarray()[s["pre"], s["post"] * n_receptor + s["receptor"]]
    group = torch.as_tensor(ids, dtype=torch.long) - 1  # per semantic edge
    n_group = int(group.max()) + 1
    assert n_group > 1 and int(group.min()) >= 0
    s["expanded"] = conn_mat
    conns = pathways(s, weight=lambda: ConstrainedWeight(constraint_mat), group=group)

    scale = torch.linspace(0.5, 2.0, n_group, requires_grad=True)
    x = torch.randn(3, N, generator=torch.Generator().manual_seed(1))
    upstream = torch.randn(
        3, N * n_receptor, generator=torch.Generator().manual_seed(2)
    )
    expected = x @ dense_operator(s, s["weight"] * scale[group]).T
    (expected_grad,) = torch.autograd.grad((expected * upstream).sum(), scale)
    for label, conn in conns.items():
        assert conn.weight.n_group == n_group, label
        assert [n for n, _ in conn.named_parameters()] == ["weight.scale"]
        with torch.no_grad():
            conn.weight.scale.copy_(scale)
        out = conn(x)
        torch.testing.assert_close(out, expected.detach(), msg=label)
        (grad,) = torch.autograd.grad((out * upstream).sum(), conn.weight.scale)
        torch.testing.assert_close(grad, expected_grad, msg=label)
    assert torch.equal(conns["adapter"].weight.group, conns["semantic"].weight.group)
    assert (
        conns["adapter"].weight.group_sizes().tolist() == torch.bincount(group).tolist()
    )


# ------------------------------------------------- simulation-level integration
def test_delays_with_spike_history():
    """``SpikeHistory.get_flattened`` yields ``[..., pre * n_delay + delay]``,

    the input layout of a delayed connection. The current at step ``t`` is
    ``sum_e w[e] * spike[t - delay[e], pre[e]]``, computed here directly from
    the edge list and the spike train.
    """
    s = scenario("delay_only")
    steps, batch = 9, 2
    gen = torch.Generator().manual_seed(0)
    spikes = (torch.rand(steps, batch, N, generator=gen) < 0.4).float()
    expected = torch.zeros(steps, batch, N)
    for pre, post, w, d in zip(
        *(s[k].tolist() for k in ("pre", "post", "weight", "delay"))
    ):
        expected[d:, :, post] += w * spikes[: steps - d, :, pre]

    for label, conn in pathways(s).items():
        history = SpikeHistory(N, max_delay_steps=N_BINS)
        history.init_state(batch_size=batch)
        out = []
        for t in range(steps):
            history.update(spikes[t])
            out.append(conn(history.get_flattened(N_BINS)))
        torch.testing.assert_close(torch.stack(out), expected, msg=label)


@pytest.mark.parametrize(
    "name", ["neuron", "connection", "neuron+delay", "connection+delay"]
)
def test_receptors_with_hetersynapse_psc(name):
    """A routed connection is a drop-in ``linear`` of ``HeterSynapsePSC``
    (which owns the delay history and one PSC state per receptor channel):

    the simulation equals the one driven by a dense ``nn.Linear``.
    """
    s = scenario(name)
    n_receptor, n_delay = s["n_receptor"], s["n_delay"]
    dense = nn.Linear(N * n_delay, N * n_receptor, bias=False)
    with torch.no_grad():
        dense.weight.copy_(dense_operator(s))
    gen = torch.Generator().manual_seed(0)
    spikes = (torch.rand(12, 2, N, generator=gen) < 0.3).float()

    def simulate(linear):
        with environ.context(dt=1.0):
            psc = HeterSynapsePSC(
                N,
                n_receptor,
                s["index"],
                linear,
                base_psc=ExponentialPSC,
                max_delay_steps=n_delay,
                tau_syn=3.0,
            )
            init_net_state(psc, batch_size=2, dtype=torch.float32)
            return torch.stack([psc(z) for z in spikes])

    reference = simulate(dense)
    assert reference.abs().sum() > 0
    for label, conn in pathways(s).items():
        torch.testing.assert_close(
            simulate(conn), reference, msg=label, atol=1e-5, rtol=1e-5
        )


def test_routed_connection_compiles_and_trains():
    """Routing is resolved at construction: the compiled forward is one
    graph, and weight gradients reach the semantic (un-expanded) slots."""
    s = scenario("connection+delay")
    conn = pathways(s)["semantic"]
    x = torch.randn(3, conn.in_features)
    torch._dynamo.reset()
    report = torch._dynamo.explain(conn)(x)
    assert (report.graph_count, report.graph_break_count) == (1, 0)
    compiled = torch.compile(conn, fullgraph=True)
    dense = dense_operator(s).requires_grad_()
    compiled(x).square().sum().backward()
    (x @ dense.T).square().sum().backward()
    row = conn.indices[0] * conn.n_receptor + conn.receptor
    col = conn.indices[1] * conn.n_delay + conn.delay
    torch.testing.assert_close(conn.weight.value.grad, dense.grad[row, col])
    torch._dynamo.reset()


# ----------------------------------------------------------------- validation
def test_validation_errors():
    pre, post = torch.tensor([0, 1]), torch.tensor([1, 0])

    def edges(**synapse):
        return SparseConnection.from_edges(pre, post, 3, 3, Synapse(**synapse))

    with pytest.raises(ValueError, match="non-negative"):
        edges(delay=torch.tensor([0, -1]))
    with pytest.raises(ValueError, match="non-negative"):
        edges(receptor=torch.tensor([-2, 0]))
    with pytest.raises(ValueError, match="out of range"):
        edges(receptor=torch.tensor([0, 3]), n_receptor=2)
    with pytest.raises(ValueError, match="out of range"):
        edges(delay=torch.tensor([5, 0]), n_delay=5)
    with pytest.raises(TypeError, match="integers"):
        edges(delay=torch.tensor([0.5, 1.0]))  # delays are time steps
    with pytest.raises((ValueError, RuntimeError)):
        edges(receptor=torch.tensor([0, 1, 0]))  # one id per edge

    expanded = scenario("connection+delay")["expanded"]  # (N * 5, N * 3)
    for bad in ({"n_receptor": 4, "n_delay": 5}, {"n_receptor": 3, "n_delay": 4}):
        with pytest.raises(ValueError, match="not divisible"):
            SparseConnection.from_hetersynapse(expanded, **bad)
    # The adapter decodes routing from the matrix; giving it twice is an error.
    with pytest.raises(ValueError, match="decoded"):
        SparseConnection.from_hetersynapse(
            expanded, Synapse(receptor=0), n_receptor=3, n_delay=5
        )
