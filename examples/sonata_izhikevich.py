"""Round-trip an Izhikevich/AlphaPSC RSNN through a specific SONATA mapping.

Run ``python examples/sonata_izhikevich.py --directory ./sonata-izh``.
Requires h5py and libsonata. The adapter intentionally handles this network
only; agents supply traversal and mappings for their own/custom networks.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from btorch.io.sonata import (
    SonataImport,
    SonataPopulation,
    read_population,
    write_edge_indices,
    write_population,
)
from btorch.models import KernelOptions, environ, functional
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.connection.weight import EdgeWeight
from btorch.models.neurons.izhikevich import Izhikevich
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import Sigmoid
from btorch.models.synapse import AlphaPSC


PARAMETERS = ("v_threshold", "v_reset", "v_rest", "v_peak", "c_m", "k", "a", "b", "d")
NODE_POPULATION = "izhikevich"
EDGE_POPULATION = "recurrent"
UNITS = {
    "dt": "ms",
    "v_threshold": "mV",
    "v_reset": "mV",
    "v_rest": "mV",
    "v_peak": "mV",
    "c_m": "pF",
    "k": "nS/mV",
    "a": "1/ms",
    "b": "nS",
    "d": "pA",
    "tau_syn": "ms",
    "g_max": "1",
    "syn_weight": "pA",
}


class IzhikevichSonataIO:
    """Map one Izhikevich/AlphaPSC sparse recurrent population to SONATA.

    Supports scalar/per-neuron parameters and plain per-edge weights.
    Node IDs equal neuron positions, sources are pre and targets are
    post. Values use mV/ms/pA/pF/nS; syn_weight is current drive in pA
    and g_max is dimensionless. Explicit edge delay is zero; RecurrentNN
    still uses previous-step PSC. Runtime state, optimizers and
    execution tuning are not circuit data.
    """

    def export_network(
        self, network: RecurrentNN, directory: Path, *, dt: float
    ) -> Path:
        """Export this network definition without advancing or resetting it."""
        if type(network) is not RecurrentNN:
            raise ValueError("This mapping requires the standard RecurrentNN scheduler")
        neuron, psc = network.neuron, network.synapse
        if type(neuron) is not Izhikevich or type(psc) is not AlphaPSC:
            raise ValueError("This mapping requires Izhikevich and AlphaPSC")
        connection = psc.linear
        if (
            type(connection) is not SparseConnection
            or type(connection.weight) is not EdgeWeight
            or connection.weight.dale
            or connection.bias is not None
            or connection.batch_shape
            or connection.delay is not None
            or connection.receptor is not None
            or psc.n_delay != 1
            or connection.n_receptor != 1
            or len(neuron.n_neuron) != 1
            or network.syn_inp_module is not None
            or network.neuron_inp_module is not None
        ):
            raise ValueError(
                "This mapping requires plain single-population edges, "
                "no additional delay/receptors/input modules"
            )
        n = neuron.n_neuron[0]
        if (
            connection.n_pre != n
            or connection.n_post != n
            or tuple(psc.n_neuron) != (n,)
        ):
            raise ValueError("The recurrent population sizes must match")
        surrogate = neuron.surrogate_function
        if type(surrogate) is not Sigmoid or not surrogate.spiking:
            raise ValueError("This mapping supports spiking Sigmoid surrogate only")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        parameters = {name: getattr(neuron, name) for name in PARAMETERS}
        parameters.update(tau_syn=psc.tau_syn, g_max=psc.g_max)
        # Keep parameter ownership honest: this example records scalar/heter
        # layouts, but does not reconstruct ties between separate model fields.
        tensors = [*parameters.values(), connection.weight.value]
        addresses = [
            value.untyped_storage().data_ptr() for value in tensors if value.numel()
        ]
        if len(addresses) != len(set(addresses)):
            raise ValueError(
                "This mapping does not support parameter sharing across fields"
            )
        shapes, dynamics = {}, {}
        for name, value in parameters.items():
            value = value.detach().cpu()
            if value.dtype not in (torch.float32, torch.float64):
                raise ValueError("This example maps FP32/FP64 model parameters")
            if value.ndim and tuple(value.shape) != (n,):
                raise ValueError(f"{name} must be scalar or per-neuron")
            shapes[name] = list(value.shape)
            dynamics[name] = value.expand(n).numpy().copy()
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        write_population(
            directory / "nodes.h5",
            NODE_POPULATION,
            SonataPopulation(
                {
                    "node_id": np.arange(n, dtype=np.uint64),
                    "node_type_id": np.zeros(n, np.uint32),
                },
                dynamics,
            ),
            kind="nodes",
        )
        edges = connection.edge_table()
        count = len(edges["pre"])
        write_population(
            directory / "edges.h5",
            EDGE_POPULATION,
            SonataPopulation(
                {
                    "source_node_id": edges["pre"].cpu().numpy().astype(np.uint64),
                    "target_node_id": edges["post"].cpu().numpy().astype(np.uint64),
                    "edge_type_id": np.zeros(count, np.uint32),
                    "syn_weight": edges["weight"].cpu().numpy(),
                    "delay": np.zeros(count, dtype=np.float64),
                },
                source=NODE_POPULATION,
                target=NODE_POPULATION,
            ),
            kind="edges",
        )
        write_edge_indices(
            directory / "edges.h5", EDGE_POPULATION, source_count=n, target_count=n
        )
        (directory / "node_types.csv").write_text(
            "node_type_id model_type model_template\n0 point_neuron btorch:Izhikevich\n"
        )
        (directory / "edge_types.csv").write_text(
            "edge_type_id model_template nsyns\n0 btorch:AlphaPSC 1\n"
        )
        metadata = {
            "mapping": "izhikevich-alpha-v1",
            "dt": dt,
            "units": UNITS,
            "parameter_shapes": shapes,
            "trainable_parameters": [
                name for name, value in parameters.items() if value.requires_grad
            ],
            "trainable_weights": connection.weight.value.requires_grad,
            "hard_reset": neuron.hard_reset,
            "detach_reset": neuron.detach_reset,
            "pre_spike": neuron.pre_spike_v,
            "surrogate": {
                "alpha": surrogate.alpha,
                "damping_factor": surrogate.damping_factor,
            },
            "update_state_names": network.update_state_names,
            "timing": "neuron consumes previous PSC; emitted spikes update current PSC",
        }
        config = directory / "circuit_config.json"
        config.write_text(
            json.dumps(
                {
                    "manifest": {"$NETWORK_DIR": "."},
                    "networks": {
                        "nodes": [
                            {
                                "nodes_file": "$NETWORK_DIR/nodes.h5",
                                "node_types_file": "$NETWORK_DIR/node_types.csv",
                                "populations": {NODE_POPULATION: {"type": "point"}},
                            }
                        ],
                        "edges": [
                            {
                                "edges_file": "$NETWORK_DIR/edges.h5",
                                "edge_types_file": "$NETWORK_DIR/edge_types.csv",
                                "populations": {EDGE_POPULATION: {"type": "chemical"}},
                            }
                        ],
                    },
                    "btorch": metadata,
                },
                indent=2,
            )
            + "\n"
        )
        return config

    def import_network(
        self,
        config: Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        backend: str | KernelOptions = "torch",
    ) -> SonataImport[RecurrentNN]:
        """Restore this mapping; the caller initializes memories and
        environ."""
        config = Path(config)
        definition = json.loads(config.read_text())
        metadata = definition["btorch"]
        if metadata["mapping"] != "izhikevich-alpha-v1":
            raise ValueError("Unsupported network mapping")
        if metadata["units"] != UNITS:
            raise ValueError("This mapping requires its declared parameter units")
        dt = float(metadata["dt"])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        def resolve(path):
            base = definition["manifest"]["$NETWORK_DIR"]
            return config.parent / path.replace("$NETWORK_DIR", base)

        if (
            len(definition["networks"]["nodes"]) != 1
            or len(definition["networks"]["edges"]) != 1
        ):
            raise ValueError("This mapping supports one node and one edge population")
        expected_types = {
            "nodes": [
                {
                    "node_type_id": "0",
                    "model_type": "point_neuron",
                    "model_template": "btorch:Izhikevich",
                }
            ],
            "edges": [
                {"edge_type_id": "0", "model_template": "btorch:AlphaPSC", "nsyns": "1"}
            ],
        }
        for kind in ("nodes", "edges"):
            key = "node_types_file" if kind == "nodes" else "edge_types_file"
            path = resolve(definition["networks"][kind][0][key])
            with path.open() as file:
                rows = list(csv.DictReader(file, delimiter=" ", skipinitialspace=True))
            if rows != expected_types[kind]:
                raise ValueError("Unsupported SONATA type table for this mapping")
        nodes = read_population(
            resolve(definition["networks"]["nodes"][0]["nodes_file"]),
            NODE_POPULATION,
            kind="nodes",
        )
        edges = read_population(
            resolve(definition["networks"]["edges"][0]["edges_file"]),
            EDGE_POPULATION,
            kind="edges",
        )
        n = len(nodes.columns["node_id"])
        if (
            not np.array_equal(nodes.columns["node_id"], np.arange(n))
            or np.any(nodes.columns["node_type_id"] != 0)
            or np.any(edges.columns["edge_type_id"] != 0)
            or edges.source != NODE_POPULATION
            or edges.target != NODE_POPULATION
            or np.any(edges.columns["delay"] != 0)
        ):
            raise ValueError(
                "Unexpected node IDs, types, population endpoints or explicit delays"
            )
        parameters = {}
        for name, shape in metadata["parameter_shapes"].items():
            values = nodes.dynamics[name]
            if shape == []:
                if not len(values) or not np.all(values == values[0]):
                    raise ValueError(f"Shared parameter {name} has inconsistent rows")
                values = values[0]
            elif shape != [n]:
                raise ValueError(f"Unsupported parameter shape for {name}")
            parameters[name] = torch.as_tensor(values, device=device, dtype=dtype)
        tau_syn, g_max = parameters.pop("tau_syn"), parameters.pop("g_max")
        trainable = set(metadata["trainable_parameters"])
        neuron = Izhikevich(
            n,
            **parameters,
            trainable_param=trainable.intersection(PARAMETERS),
            hard_reset=metadata["hard_reset"],
            detach_reset=metadata["detach_reset"],
            pre_spike=metadata["pre_spike"],
            surrogate_function=Sigmoid(**metadata["surrogate"]),
            device=device,
            dtype=dtype,
        )
        connection = SparseConnection.from_edges(
            torch.as_tensor(
                edges.columns["source_node_id"].astype(np.int64), device=device
            ),
            torch.as_tensor(
                edges.columns["target_node_id"].astype(np.int64), device=device
            ),
            n,
            n,
            synapse=Synapse(weight=EdgeWeight(trainable=metadata["trainable_weights"])),
            values=torch.as_tensor(
                edges.columns["syn_weight"], device=device, dtype=dtype
            ),
        )
        psc = AlphaPSC(n, tau_syn=tau_syn, g_max=g_max, linear=connection)
        for name in ("tau_syn", "g_max"):
            if name in trainable:
                value = getattr(psc, name)
                delattr(psc, name)
                psc.register_parameter(name, torch.nn.Parameter(value))
        network = RecurrentNN(
            neuron,
            psc,
            backend=backend,
            update_state_names=metadata["update_state_names"],
        )
        return SonataImport(network, dt)


def make_example() -> RecurrentNN:
    """Build a small recurrent circuit with heterogeneous recovery
    timescales."""
    connection = SparseConnection.from_edges(
        torch.tensor([0, 1, 2, 2]),
        torch.tensor([1, 2, 0, 2]),
        4,
        4,
        values=torch.tensor([40.0, 25.0, -10.0, 5.0]),
    )
    return RecurrentNN(
        Izhikevich(4, a=torch.tensor([0.03, 0.02, 0.04, 0.03]), hard_reset=True),
        AlphaPSC(4, tau_syn=3.0, linear=connection),
        update_state_names=("neuron.v", "neuron.u", "synapse.psc", "synapse.h"),
    )


def run_example(directory: Path) -> Path:
    """Export, import and compare activity using RecurrentNN scheduling."""
    original = make_example()
    adapter = IzhikevichSonataIO()
    config = adapter.export_network(original, directory, dt=0.25)
    restored = adapter.import_network(config)
    inputs = torch.zeros(120, 1, 4)
    inputs[8:110, :, :2] = 1500.0
    outputs = []
    with environ.context(dt=restored.dt), torch.no_grad():
        for network in (original, restored.network):
            functional.init_net_state(network, batch_size=1)
            outputs.append(network(inputs))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    for name, values in outputs[0][1].items():
        torch.testing.assert_close(values, outputs[1][1][name], rtol=0, atol=0)
    print(f"SONATA round-trip activity matches: {config}")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("sonata-izh"))
    run_example(parser.parse_args().directory)
