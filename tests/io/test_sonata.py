"""Round-trip both concrete examples and read their files independently."""

import json

import numpy as np
import pytest
import torch

from btorch.models import environ
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.connection.weight import EdgeWeight
from examples.rsnn import build_model as build_glif3
from examples.rsnn_common import ExampleOptions, initialize, stimulus
from examples.rsnn_izhikevich import build_model as build_izhikevich
from examples.rsnn_sonata import RSNNSonataIO


@pytest.fixture
def adapter():
    pytest.importorskip("h5py")
    pytest.importorskip("libsonata")
    return RSNNSonataIO()


@pytest.mark.parametrize("kind", ["glif3", "izhikevich"])
def test_example_circuit_roundtrip_activity_gradients_and_native_reader(
    tmp_path, adapter, kind
):
    import libsonata

    network = {"glif3": build_glif3, "izhikevich": build_izhikevich}[kind](
        ExampleOptions(neurons=6, density=0.5)
    )
    # Parallel source/target pairs in distinct receptors/delay bins must remain
    # distinct mechanisms. Include a self-loop and an isolated final neuron.
    original = network.synapse.linear
    connection = SparseConnection.from_edges(
        torch.tensor([0, 0, 1, 2]),
        torch.tensor([1, 1, 2, 2]),
        6,
        6,
        synapse=Synapse(
            weight=EdgeWeight(trainable=True),
            receptor=torch.tensor([0, 1, 0, 1]),
            n_receptor=2,
            delay=torch.tensor([0, 3, 1, 2]) if kind == "glif3" else None,
            n_delay=4 if kind == "glif3" else None,
        ),
        values=torch.tensor([0.4, -0.2, 0.3, -0.1]),
    )
    network.synapse.linear = network.synapse.base_psc.linear = connection
    del original
    circuit_config = adapter.export_network(network, tmp_path, dt=0.25)
    loaded = adapter.import_network(circuit_config)
    assert loaded.dt == 0.25
    inputs = stimulus(kind, 96, 6, "cpu")
    results = []
    for model in (network, loaded.network):
        initialize(model, "cpu")
        with environ.context(dt=loaded.dt), torch.inference_mode():
            raster, records = model(inputs)
        inference = (raster.clone(), {k: v.clone() for k, v in records.items()})
        # Reset memories created under inference_mode before building autograd.
        initialize(model, "cpu")
        value = inputs.clone().requires_grad_()
        with environ.context(dt=loaded.dt):
            raster, records = model(value)
            loss = raster.sum() + sum(
                v.square().mean() * 1e-5 for v in records.values()
            )
        parameters = dict(model.named_parameters())
        gradients = torch.autograd.grad(loss, (value, *parameters.values()))
        assert all(torch.isfinite(v).all() for v in gradients)
        assert parameters["synapse.linear.weight.value"].requires_grad
        assert (
            gradients[list(parameters).index("synapse.linear.weight.value") + 1]
            .abs()
            .max()
            > 0
        )
        results.append((inference, raster, records, parameters, gradients))
    assert results[0][1].sum() > 0
    for a, b in zip(results[0][:3], results[1][:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert results[0][3].keys() == results[1][3].keys()
    for a, b in zip(results[0][4], results[1][4]):
        # Equivalent sparse index/reduction layouts may differ by one FP32
        # rounding; activity/state comparisons above remain bitwise strict.
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)
    # Query the circuit config and real afferent/efferent indices through an
    # independent reader, rather than trusting our own serializer twice.
    circuit = libsonata.CircuitConfig.from_file(str(circuit_config))
    nodes, edges = circuit.node_population(kind), circuit.edge_population("recurrent")
    table = connection.edge_table()
    assert nodes.size == 6 and edges.size == 4
    selected = edges.select_all()
    np.testing.assert_array_equal(edges.source_nodes(selected), table["pre"])
    np.testing.assert_array_equal(edges.target_nodes(selected), table["post"])
    for column in ("receptor", "syn_weight"):
        np.testing.assert_array_equal(
            edges.get_attribute(column, selected),
            table["weight" if column == "syn_weight" else column],
        )
    delay = table.get("delay", torch.zeros(4)) * loaded.dt
    np.testing.assert_array_equal(edges.get_attribute("delay", selected), delay)
    for node in range(6):
        incoming, outgoing = edges.afferent_edges([node]), edges.efferent_edges([node])
        np.testing.assert_array_equal(
            edges.source_nodes(incoming), table["pre"][table["post"] == node]
        )
        np.testing.assert_array_equal(
            edges.target_nodes(outgoing), table["post"][table["pre"] == node]
        )
    meta = json.loads(circuit_config.read_text())["btorch"]
    for name, layout in meta["layouts"].items():
        prefix, field = name.split(".")
        module = network.neuron if prefix == "neuron" else network.synapse.base_psc
        value = getattr(module, field).detach()
        expected = (
            value.reshape(layout["target"])
            if value.numel() == np.prod(layout["target"])
            else value.expand(layout["target"])
        )
        actual = np.stack(
            [
                nodes.get_dynamics_attribute(col, nodes.select_all())
                for col in layout["columns"]
            ],
            axis=1,
        )
        np.testing.assert_array_equal(actual.reshape(layout["target"]), expected)


@pytest.mark.parametrize("kind", ["glif3", "izhikevich"])
def test_official_pysonata_reads_receptors_and_delay_indices(tmp_path, adapter, kind):
    pytest.importorskip("sonata")
    from sonata.circuit import File

    network = {"glif3": build_glif3, "izhikevich": build_izhikevich}[kind](
        ExampleOptions(neurons=4, density=0.5)
    )
    adapter.export_network(network, tmp_path, dt=0.25)
    nodes = File(str(tmp_path / "nodes.h5"), str(tmp_path / "node_types.csv")).nodes[
        kind
    ]
    edges = File(str(tmp_path / "edges.h5"), str(tmp_path / "edge_types.csv")).edges[
        "recurrent"
    ]
    table = network.synapse.linear.edge_table()
    assert [node.node_id for node in nodes] == list(range(4))
    rows = list(edges)
    np.testing.assert_array_equal([edge.source_node_id for edge in rows], table["pre"])
    np.testing.assert_array_equal([edge.target_node_id for edge in rows], table["post"])
    np.testing.assert_array_equal(
        [edge["receptor"] for edge in rows], table["receptor"]
    )
    for node in range(4):
        assert [edge.source_node_id for edge in edges.get_target(node)] == table["pre"][
            table["post"] == node
        ].tolist()
        assert [edge.target_node_id for edge in edges.get_source(node)] == table[
            "post"
        ][table["pre"] == node].tolist()
    # Official pySONATA leaves Node.dynamics_params unimplemented. The native
    # reader test above checks every dynamics component independently.


def test_empty_connectivity_and_invalid_delay_grid_are_not_silently_changed(
    tmp_path, adapter
):
    import h5py

    network = build_glif3(ExampleOptions(neurons=4, density=0.5))
    circuit_config = adapter.export_network(network, tmp_path, dt=0.25)
    with h5py.File(tmp_path / "edges.h5", "r+") as file:
        file["edges/recurrent/0/delay"][0] = 0.1
    with pytest.raises(ValueError, match="delays"):
        adapter.import_network(circuit_config)
    empty = SparseConnection.from_edges(
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        4,
        4,
        synapse=Synapse(
            weight=EdgeWeight(trainable=True),
            receptor=torch.empty(0, dtype=torch.long),
            n_receptor=2,
            delay=torch.empty(0, dtype=torch.long),
            n_delay=4,
        ),
        values=torch.empty(0),
    )
    network.synapse.linear = network.synapse.base_psc.linear = empty
    circuit_config = adapter.export_network(network, tmp_path / "empty", dt=0.25)
    restored = adapter.import_network(circuit_config).network
    x = stimulus("glif3", 16, 4, "cpu")
    outputs = []
    for model in (network, restored):
        initialize(model, "cpu")
        with environ.context(dt=0.25), torch.inference_mode():
            outputs.append(model(x))
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert restored.synapse.linear.nnz == 0


def test_external_edits_cannot_discard_shared_parameters_or_model_types(
    tmp_path, adapter
):
    import h5py

    model = build_izhikevich(ExampleOptions(neurons=4))
    circuit_config = adapter.export_network(model, tmp_path, dt=0.25)
    with h5py.File(tmp_path / "nodes.h5", "r+") as file:
        file["nodes/izhikevich/0/dynamics_params/neuron_b_0"][1] = 0.7
    with pytest.raises(ValueError, match="Shared parameter neuron.b"):
        adapter.import_network(circuit_config)
    circuit_config = adapter.export_network(model, tmp_path, dt=0.25)
    types = tmp_path / "node_types.csv"
    types.write_text(types.read_text().replace("btorch:Izhikevich", "btorch:LIF"))
    with pytest.raises(ValueError, match="type table"):
        adapter.import_network(circuit_config)
    model.neuron.b = model.neuron.a
    with pytest.raises(ValueError, match="parameter sharing"):
        adapter.export_network(model, tmp_path / "alias", dt=0.25)


def test_glif_dale_projection_and_zero_weight_signs_survive_roundtrip(
    tmp_path, adapter
):
    from btorch.models.constrain import constrain_net

    network = build_glif3(ExampleOptions(neurons=8, density=0.5))
    weight = network.synapse.linear.weight
    table = network.synapse.linear.edge_table()
    # Every outgoing edge follows its source's E/I identity, not the target.
    expected_sign = torch.where(table["pre"] % 5 == 0, -1.0, 1.0)
    torch.testing.assert_close(weight.sign, expected_sign)
    # A deliberately large optimizer step crosses zero for both populations.
    # Projection must clamp them without forgetting their reference signs.
    optimizer = torch.optim.SGD(network.parameters(), lr=1.0)
    (weight.value * expected_sign).sum().backward()
    optimizer.step()
    constrain_net(network)
    torch.testing.assert_close(weight.value, torch.zeros_like(weight.value))
    circuit_config = adapter.export_network(network, tmp_path, dt=1.0)
    restored = adapter.import_network(circuit_config).network
    torch.testing.assert_close(restored.synapse.linear.weight.sign, expected_sign)
    # Resume learning away from zero after import. Reconstructing signs from
    # current zero weights would silently freeze all these synapses forever.
    for model in (network, restored):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        optimizer.zero_grad(set_to_none=True)
        weights = model.synapse.linear.weight
        loss = -(weights.value * expected_sign).sum()
        loss.backward()
        optimizer.step()
        constrain_net(model)
        torch.testing.assert_close(weights.value, expected_sign * 0.2)
        initialize(model, "cpu")
    inputs = stimulus("glif3", 64, 8, "cpu")
    with environ.context(dt=1.0), torch.inference_mode():
        spikes, records = network(inputs)
        expected, expected_records = restored(inputs)
    torch.testing.assert_close(spikes, expected, rtol=0, atol=0)
    for name in records:
        torch.testing.assert_close(
            records[name], expected_records[name], rtol=0, atol=0
        )
