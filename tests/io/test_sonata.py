"""Check a specific circuit mapping against btorch and independent SONATA
readers."""

import numpy as np
import pytest
import torch

from btorch.models import environ, functional
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.connection.weight import EdgeWeight
from btorch.models.neurons.izhikevich import Izhikevich
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC
from examples.sonata_izhikevich import IzhikevichSonataIO


@pytest.fixture
def adapter():
    pytest.importorskip("h5py")
    pytest.importorskip("libsonata")
    return IzhikevichSonataIO()


def _network(topology, heter):
    # Include an isolated neuron, an inhibitory edge and a self-loop. Empty
    # connections must remain empty rather than acquiring default connectivity.
    pre, post = ([0, 1, 2, 2], [1, 2, 0, 2]) if topology == "recurrent" else ([], [])
    connection = SparseConnection.from_edges(
        torch.tensor(pre, dtype=torch.long),
        torch.tensor(post, dtype=torch.long),
        4,
        4,
        values=torch.tensor([40.0, 25.0, -10.0, 5.0] if pre else []),
        synapse=Synapse(weight=EdgeWeight(trainable=True)),
    )
    neuron = Izhikevich(
        4,
        a=torch.tensor([0.03, 0.02, 0.04, 0.03]) if heter else 0.03,
        d=100.0,
        hard_reset=True,
        trainable_param={"a", "d"},
    )
    psc = AlphaPSC(
        4,
        tau_syn=torch.tensor([2.0, 3.0, 4.0, 5.0]) if heter else 3.0,
        linear=connection,
    )
    # PSC currently accepts tensors, not trainable_param. Register the desired
    # physical parameters explicitly, as a user-owned network definition does.
    for name in ("tau_syn", "g_max"):
        value = getattr(psc, name)
        delattr(psc, name)
        psc.register_parameter(name, torch.nn.Parameter(value))
    return RecurrentNN(
        neuron,
        psc,
        update_state_names=("neuron.v", "neuron.u", "synapse.psc", "synapse.h"),
    )


@pytest.mark.parametrize("topology", ["recurrent", "empty"])
@pytest.mark.parametrize("heter", [False, True])
def test_izhikevich_roundtrip_activity_and_gradients(
    tmp_path, adapter, topology, heter
):
    original = _network(topology, heter)
    config = adapter.export_network(original, tmp_path, dt=0.25)
    imported = adapter.import_network(config)
    stimulus = torch.zeros(120, 2, 4)
    stimulus[8:110, :, :2] = 1500.0
    results = []
    for network in (original, imported.network):
        x = stimulus.clone().requires_grad_()
        with environ.context(dt=imported.dt):
            functional.init_net_state(network, batch_size=2)
            spikes, states = network(x)
            # State losses exercise recovery and both AlphaPSC stages even
            # when identical binary spikes alone would hide parameter errors.
            loss = spikes.sum() + sum(
                value.square().mean() * 1e-5 for value in states.values()
            )
            parameters = dict(network.named_parameters())
            gradients = torch.autograd.grad(
                loss, (x, *parameters.values()), allow_unused=True
            )
        results.append((spikes, states, dict(zip(("input", *parameters), gradients))))
    assert results[0][0].sum() > 0
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name, values in results[0][1].items():
        torch.testing.assert_close(values, results[1][1][name], rtol=0, atol=0)
    assert results[0][2].keys() == results[1][2].keys()
    for name, grad in results[0][2].items():
        actual = results[1][2][name]
        if grad is None:
            assert actual is None
        else:
            assert torch.isfinite(grad).all()
            torch.testing.assert_close(grad, actual, rtol=0, atol=0)


def test_libsonata_reads_circuit_and_edge_indices(tmp_path, adapter):
    import libsonata

    network = _network("recurrent", True)
    config = adapter.export_network(network, tmp_path, dt=0.25)
    # Config-driven discovery also verifies portable paths, not just direct
    # reading of files whose locations are already known to this test.
    circuit = libsonata.CircuitConfig.from_file(str(config))
    assert circuit.node_population("izhikevich").size == 4
    assert circuit.edge_population("recurrent").size == 4
    nodes = libsonata.NodeStorage(str(tmp_path / "nodes.h5")).open_population(
        "izhikevich"
    )
    edges = libsonata.EdgeStorage(str(tmp_path / "edges.h5")).open_population(
        "recurrent"
    )
    selected = nodes.select_all()
    np.testing.assert_array_equal(
        nodes.get_dynamics_attribute("a", selected), network.neuron.a.detach()
    )
    np.testing.assert_array_equal(
        nodes.get_dynamics_attribute("tau_syn", selected),
        network.synapse.tau_syn.detach(),
    )
    table = network.synapse.linear.edge_table()
    np.testing.assert_array_equal(edges.source_nodes(edges.select_all()), table["pre"])
    np.testing.assert_array_equal(edges.target_nodes(edges.select_all()), table["post"])
    np.testing.assert_array_equal(
        edges.get_attribute("syn_weight", edges.select_all()), table["weight"]
    )
    # Both indexed traversal directions include self-loops and exclude isolated
    # nodes. Check endpoints, so an index which exists but is wrong cannot pass.
    for node in range(4):
        incoming = edges.afferent_edges([node])
        outgoing = edges.efferent_edges([node])
        np.testing.assert_array_equal(
            edges.source_nodes(incoming), table["pre"][table["post"] == node]
        )
        np.testing.assert_array_equal(
            edges.target_nodes(outgoing), table["post"][table["pre"] == node]
        )


def test_official_pysonata_reads_circuit_and_edge_indices(tmp_path, adapter):
    pytest.importorskip("sonata")
    from sonata.circuit import File

    network = _network("recurrent", True)
    adapter.export_network(network, tmp_path, dt=0.25)
    nodes = File(str(tmp_path / "nodes.h5"), str(tmp_path / "node_types.csv")).nodes[
        "izhikevich"
    ]
    edges = File(str(tmp_path / "edges.h5"), str(tmp_path / "edge_types.csv")).edges[
        "recurrent"
    ]
    assert [node.node_id for node in nodes] == list(range(4))
    assert {node["model_template"] for node in nodes} == {"btorch:Izhikevich"}
    table = network.synapse.linear.edge_table()
    rows = list(edges)
    assert [(edge.source_node_id, edge.target_node_id) for edge in rows] == list(
        zip(table["pre"].tolist(), table["post"].tolist())
    )
    np.testing.assert_array_equal(
        [edge["syn_weight"] for edge in rows], table["weight"]
    )
    # Official pySONATA's Node.dynamics_params is unimplemented; libsonata
    # above verifies those fields. Its indexed topology traversal is supported.
    for node in range(4):
        assert [edge.source_node_id for edge in edges.get_target(node)] == table["pre"][
            table["post"] == node
        ].tolist()
        assert [edge.target_node_id for edge in edges.get_source(node)] == table[
            "post"
        ][table["pre"] == node].tolist()


def test_import_rejects_delays_outside_network_mapping(tmp_path, adapter):
    import h5py

    config = adapter.export_network(_network("recurrent", False), tmp_path, dt=0.25)
    # A reader must not silently turn an explicitly delayed circuit into the
    # zero-extra-delay RecurrentNN implemented by this specific mapping.
    with h5py.File(tmp_path / "edges.h5", "r+") as file:
        file["edges/recurrent/0/delay"][0] = 0.5
    with pytest.raises(ValueError, match="explicit delays"):
        adapter.import_network(config)


def test_import_rejects_heterogeneous_rows_for_shared_parameter(tmp_path, adapter):
    import h5py

    config = adapter.export_network(_network("recurrent", False), tmp_path, dt=0.25)
    # Scalar metadata denotes a genuinely shared parameter, not a request to
    # discard heterogeneity by picking the first row after an external edit.
    with h5py.File(tmp_path / "nodes.h5", "r+") as file:
        file["nodes/izhikevich/0/dynamics_params/a"][1] = 0.07
    with pytest.raises(ValueError, match="Shared parameter a"):
        adapter.import_network(config)


def test_export_rejects_parameter_aliasing_across_fields(tmp_path, adapter):
    network = _network("recurrent", False)
    network.neuron.b = network.neuron.a
    # Independent SONATA columns cannot preserve shared optimizer identity.
    # Reject this circuit instead of importing two independently trained values.
    with pytest.raises(ValueError, match="parameter sharing across fields"):
        adapter.export_network(network, tmp_path, dt=0.25)


def test_import_rejects_changed_model_template(tmp_path, adapter):
    config = adapter.export_network(_network("recurrent", False), tmp_path, dt=0.25)
    types = tmp_path / "node_types.csv"
    types.write_text(types.read_text().replace("btorch:Izhikevich", "btorch:LIF"))
    # Circuit type tables are authoritative. A stale btorch metadata block must
    # not reinterpret an externally changed neuron model as Izhikevich.
    with pytest.raises(ValueError, match="Unsupported SONATA type table"):
        adapter.import_network(config)
