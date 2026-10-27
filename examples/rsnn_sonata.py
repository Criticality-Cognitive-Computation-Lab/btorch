"""Map the two RSNN examples explicitly; share SONATA mechanics with btorch.

Scope: GLIF3/heterogeneous AlphaPSC with discrete delays, or
Izhikevich/heterogeneous DualExponentialPSC without extra delays. Node IDs
are tensor positions; edges are pre -> post, with signed current weights and
receptor IDs. SONATA delay is ms = delay_bin * dt, in addition to RecurrentNN's
previous-PSC schedule. This is circuit IO, not a runtime/optimizer checkpoint.
"""

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
from btorch.models import BPTT, KernelOptions
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.connection.weight import EdgeWeight
from btorch.models.neurons import GLIF3, Izhikevich
from btorch.models.rnn import RecurrentNN
from btorch.models.surrogate import Sigmoid, Truncated
from btorch.models.synapse import AlphaPSC, DualExponentialPSC, HeterSynapsePSC


NEURON_PARAMETERS = {
    "glif3": (
        "v_threshold",
        "v_reset",
        "v_rest",
        "c_m",
        "tau",
        "k",
        "asc_amps",
        "tau_ref",
    ),
    "izhikevich": (
        "v_threshold",
        "v_reset",
        "v_rest",
        "v_peak",
        "c_m",
        "k",
        "a",
        "b",
        "d",
    ),
}
PSC_PARAMETERS = {
    "glif3": ("tau_syn", "g_max"),
    "izhikevich": ("tau_rise", "tau_decay", "a"),
}
MODELS = {"glif3": (GLIF3, AlphaPSC), "izhikevich": (Izhikevich, DualExponentialPSC)}
UNITS = {
    "voltage": "mV",
    "time": "ms",
    "current": "pA",
    "capacitance": "pF",
    "conductance": "nS",
    "receptor": "dimensionless integer",
    "delay": "ms; exact integer multiples of dt",
    "syn_weight": "pA",
    "glif3.k": "1/ms",
    "izhikevich.k": "nS/mV",
    "izhikevich.a": "1/ms",
    "izhikevich.b": "nS",
    "izhikevich.d": "pA",
    "psc.a": "1/ms",
    "psc.g_max": "dimensionless",
}


class RSNNSonataIO:
    """Round-trip the two example circuits through a versioned SONATA mapping.

    Scalar/shared, per-neuron, adaptation-mode and receptor parameter
    layouts are preserved. Each tensor component is a scalar dynamics
    column per node; the extension records its original broadcast shape.
    Plain EdgeWeight, signed receptors, and exact delay bins are
    supported. Cross-field parameter sharing, input modules, batched
    connectivity and custom models are rejected. Execution tuning and
    runtime memories are deliberately caller-owned.
    """

    def export_network(
        self, network: RecurrentNN, directory: Path, *, dt: float
    ) -> Path:
        """Export a circuit without advancing or resetting its memories."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if (
            type(network) is not RecurrentNN
            or type(network.synapse) is not HeterSynapsePSC
        ):
            raise ValueError("This mapping requires RecurrentNN and HeterSynapsePSC")
        kind = next(
            (key for key, (cls, _) in MODELS.items() if type(network.neuron) is cls),
            None,
        )
        if kind is None or type(network.synapse.base_psc) is not MODELS[kind][1]:
            raise ValueError("Expected GLIF3/AlphaPSC or Izhikevich/DualExponentialPSC")
        neuron, synapse = network.neuron, network.synapse
        psc, connection = synapse.base_psc, synapse.linear
        if (
            type(connection) is not SparseConnection
            or type(connection.weight) is not EdgeWeight
            or connection.bias is not None
            or connection.batch_shape
            or network.syn_inp_module is not None
            or network.neuron_inp_module is not None
            or len(neuron.n_neuron) != 1
            or connection.receptor is None
            or synapse.receptor_type_index is not None
            or (kind == "izhikevich" and connection.delay is not None)
        ):
            raise ValueError(
                "Unsupported connection, receptor metadata or input module"
            )
        n, receptors = neuron.n_neuron[0], synapse.n_receptor
        if (
            connection.n_pre != n
            or connection.n_post != n
            or connection.n_receptor != receptors
        ):
            raise ValueError("Population and receptor dimensions must match")
        function = neuron.surrogate_function
        width = function.width if type(function) is Truncated else None
        function = function.surrogate if width is not None else function
        if type(function) is not Sigmoid or not function.spiking:
            raise ValueError("This example maps spiking Sigmoid, optionally Truncated")
        dynamics, layouts, addresses = {}, {}, []
        for prefix, module, names in (
            ("neuron", neuron, NEURON_PARAMETERS[kind]),
            ("psc", psc, PSC_PARAMETERS[kind]),
        ):
            for name in names:
                value = getattr(module, name)
                if value is None:
                    continue
                addresses.append(value.untyped_storage().data_ptr())
                target = (
                    (n, receptors)
                    if prefix == "psc"
                    else (
                        (n, neuron.n_Iasc)
                        if kind == "glif3" and name in ("k", "asc_amps")
                        else (n,)
                    )
                )
                # PSC uses flattened post*receptor lanes; preserve its actual
                # tensor shape rather than treating receptors as extra neurons.
                source = value.detach().cpu()
                if prefix == "psc" and source.shape == (n * receptors,):
                    expanded = source.reshape(target)
                else:
                    expanded = source.expand(target)
                flat = expanded.reshape(n, -1).numpy()
                columns = [f"{prefix}_{name}_{i}" for i in range(flat.shape[1])]
                dynamics.update(
                    {column: flat[:, i].copy() for i, column in enumerate(columns)}
                )
                layouts[f"{prefix}.{name}"] = dict(
                    shape=list(source.shape),
                    target=list(target),
                    columns=columns,
                    parameter=isinstance(value, torch.nn.Parameter),
                    requires_grad=value.requires_grad,
                )
        addresses.append(connection.weight.value.untyped_storage().data_ptr())
        if len(addresses) != len(set(addresses)):
            raise ValueError(
                "This mapping cannot preserve parameter sharing across fields"
            )
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        write_population(
            directory / "nodes.h5",
            kind,
            SonataPopulation(
                {
                    "node_id": np.arange(n, dtype=np.uint64),
                    "node_type_id": np.zeros(n, np.uint32),
                },
                dynamics,
            ),
            kind="nodes",
        )
        table = connection.edge_table()
        count = len(table["pre"])
        delay = table.get("delay", torch.zeros(count, dtype=torch.int64))
        write_population(
            directory / "edges.h5",
            "recurrent",
            SonataPopulation(
                dict(
                    source_node_id=table["pre"].cpu().numpy().astype(np.uint64),
                    target_node_id=table["post"].cpu().numpy().astype(np.uint64),
                    edge_type_id=table["receptor"].cpu().numpy().astype(np.uint32),
                    receptor=table["receptor"].cpu().numpy(),
                    syn_weight=table["weight"].cpu().numpy(),
                    delay=delay.cpu().numpy() * float(dt),
                    **(
                        {"dale_sign": connection.weight.sign.detach().cpu().numpy()}
                        if connection.weight.dale
                        else {}
                    ),
                ),
                source=kind,
                target=kind,
            ),
            kind="edges",
        )
        write_edge_indices(
            directory / "edges.h5", "recurrent", source_count=n, target_count=n
        )
        neuron_type, psc_type = (cls.__name__ for cls in MODELS[kind])
        (directory / "node_types.csv").write_text(
            "node_type_id model_type model_template\n"
            f"0 point_neuron btorch:{neuron_type}{psc_type}\n"
        )
        (directory / "edge_types.csv").write_text(
            "edge_type_id model_template nsyns\n"
            + "".join(f"{r} btorch:{psc_type} 1\n" for r in range(receptors))
        )
        metadata = dict(
            mapping="example-rsnn-v2",
            kind=kind,
            dt=dt,
            units=UNITS,
            layouts=layouts,
            receptors=receptors,
            delay_bins=connection.n_delay,
            delayed=connection.delay is not None,
            dale=connection.weight.dale,
            trainable_weights=isinstance(connection.weight.value, torch.nn.Parameter),
            weight_requires_grad=connection.weight.value.requires_grad,
            hard_reset=neuron.hard_reset,
            detach_reset=neuron.detach_reset,
            pre_spike=neuron.pre_spike_v,
            surrogate=dict(
                alpha=function.alpha, damping_factor=function.damping_factor
            ),
            surrogate_width=width,
            sparse_gradient=network.gradient.sparse,
            update_state_names=list(network.update_state_names),
        )
        circuit_config = directory / "circuit_config.json"
        circuit_config.write_text(
            json.dumps(
                dict(
                    manifest={"$NETWORK_DIR": "."},
                    networks={
                        "nodes": [
                            dict(
                                nodes_file="$NETWORK_DIR/nodes.h5",
                                node_types_file="$NETWORK_DIR/node_types.csv",
                                populations={kind: {"type": "point"}},
                            )
                        ],
                        "edges": [
                            dict(
                                edges_file="$NETWORK_DIR/edges.h5",
                                edge_types_file="$NETWORK_DIR/edge_types.csv",
                                populations={"recurrent": {"type": "chemical"}},
                            )
                        ],
                    },
                    btorch=metadata,
                ),
                indent=2,
            )
            + "\n"
        )
        return circuit_config

    def import_network(
        self,
        circuit_config: Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        backend: str | KernelOptions = "torch",
    ) -> SonataImport[RecurrentNN]:
        """Restore this mapping; leave environ and memory initialization
        untouched."""
        circuit_config = Path(circuit_config)
        definition = json.loads(circuit_config.read_text())
        meta = definition["btorch"]
        if (
            meta["mapping"] != "example-rsnn-v2"
            or meta["units"] != UNITS
            or meta["kind"] not in MODELS
        ):
            raise ValueError("Unsupported RSNN mapping or units")
        kind, dt = meta["kind"], float(meta["dt"])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        def resolve(path):
            return circuit_config.parent / path.replace(
                "$NETWORK_DIR", definition["manifest"]["$NETWORK_DIR"]
            )

        neuron_type, psc_type = (cls.__name__ for cls in MODELS[kind])
        expected = {
            "nodes": [
                {
                    "node_type_id": "0",
                    "model_type": "point_neuron",
                    "model_template": f"btorch:{neuron_type}{psc_type}",
                }
            ],
            "edges": [
                dict(
                    edge_type_id=str(r), model_template=f"btorch:{psc_type}", nsyns="1"
                )
                for r in range(meta["receptors"])
            ],
        }
        populations = {}
        for group in ("nodes", "edges"):
            entries = definition["networks"][group]
            if len(entries) != 1:
                raise ValueError("Expected one node and one edge population")
            types_key = "node_types_file" if group == "nodes" else "edge_types_file"
            with resolve(entries[0][types_key]).open() as file:
                if (
                    list(csv.DictReader(file, delimiter=" ", skipinitialspace=True))
                    != expected[group]
                ):
                    raise ValueError("Unsupported SONATA type table")
            populations[group] = read_population(
                resolve(entries[0][group + "_file"]),
                kind if group == "nodes" else "recurrent",
                kind=group,
            )
        nodes, edges = populations["nodes"], populations["edges"]
        n = len(nodes.columns["node_id"])
        receptor = edges.columns["receptor"]
        bins = edges.columns["delay"] / dt
        if (
            not np.array_equal(nodes.columns["node_id"], np.arange(n))
            or np.any(nodes.columns["node_type_id"] != 0)
            or edges.source != kind
            or edges.target != kind
            or np.any(receptor < 0)
            or np.any(receptor >= meta["receptors"])
            or not np.array_equal(receptor, edges.columns["edge_type_id"])
            or not np.all(np.isfinite(bins))
            or not np.allclose(bins, np.rint(bins), rtol=0, atol=1e-10)
            or np.any(bins < 0)
            or np.any(bins >= meta["delay_bins"])
            or (not meta["delayed"] and np.any(bins != 0))
        ):
            raise ValueError(
                "Invalid IDs, receptors or non-grid-aligned/out-of-range delays"
            )
        values = {}
        for name, layout in meta["layouts"].items():
            full = torch.as_tensor(
                np.stack([nodes.dynamics[col] for col in layout["columns"]], axis=1),
                device=device,
                dtype=dtype,
            ).reshape(layout["target"])
            shape, target = tuple(layout["shape"]), tuple(layout["target"])
            if shape == (math.prod(target),) and len(target) > 1:
                value = full.reshape(shape)
            else:
                padded = (1,) * (len(target) - len(shape)) + shape
                selection = tuple(
                    slice(None) if size != 1 else slice(0, 1) for size in padded
                )
                value = full[selection].reshape(shape)
                if not torch.equal(value.expand(target), full):
                    raise ValueError(f"Shared parameter {name} has inconsistent rows")
            values[name] = value.clone()
        neuron_args = {
            name.split(".")[1]: value
            for name, value in values.items()
            if name.startswith("neuron.")
        }
        trainable = {
            name.split(".")[1]
            for name, layout in meta["layouts"].items()
            if name.startswith("neuron.") and layout["parameter"]
        }
        if kind == "glif3" and "v_rest" in trainable:
            trainable.remove("v_rest")
            trainable.add("_v_rest")
        surrogate = Sigmoid(**meta["surrogate"])
        if meta["surrogate_width"] is not None:
            surrogate = Truncated(surrogate, meta["surrogate_width"])
        neuron = MODELS[kind][0](
            n,
            **neuron_args,
            trainable_param=trainable,
            hard_reset=meta["hard_reset"],
            detach_reset=meta["detach_reset"],
            surrogate_function=surrogate,
            device=device,
            dtype=dtype,
            **(
                {"pre_spike_v": meta["pre_spike"]}
                if kind == "glif3"
                else {"pre_spike": meta["pre_spike"]}
            ),
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
            synapse=Synapse(
                weight=EdgeWeight(
                    trainable=meta["trainable_weights"], dale=meta["dale"]
                ),
                receptor=torch.as_tensor(receptor.astype(np.int64), device=device),
                n_receptor=meta["receptors"],
                delay=torch.as_tensor(np.rint(bins).astype(np.int64), device=device)
                if meta["delayed"]
                else None,
                n_delay=meta["delay_bins"] if meta["delayed"] else None,
            ),
            values=torch.as_tensor(
                edges.columns["syn_weight"], device=device, dtype=dtype
            ),
        )
        if meta["dale"]:
            signs = edges.columns["dale_sign"]
            weights = edges.columns["syn_weight"]
            if (
                not np.isin(signs, [-1, 0, 1]).all()
                or np.any(weights * signs < 0)
                or np.any((signs == 0) & (weights != 0))
            ):
                raise ValueError("Dale reference signs conflict with edge weights")
            # from_edges coalesces into expanded (post, receptor, pre, delay)
            # order. Preserve signs even when a projected weight is now zero.
            keys = (
                (
                    edges.columns["target_node_id"].astype(np.int64) * meta["receptors"]
                    + receptor
                )
                * (n * meta["delay_bins"])
                + edges.columns["source_node_id"].astype(np.int64) * meta["delay_bins"]
                + np.rint(bins).astype(np.int64)
            )
            if np.any(keys[1:] < keys[:-1]):
                signs = signs[np.argsort(keys, kind="stable")]
            if connection.nnz != len(signs):
                raise ValueError(
                    "Dale reference signs require distinct edge mechanisms"
                )
            connection.weight.sign.copy_(
                torch.as_tensor(
                    signs, device=device, dtype=connection.weight.value.dtype
                )
            )
        connection.weight.value.requires_grad_(meta["weight_requires_grad"])
        psc_args = {
            name.split(".")[1]: value
            for name, value in values.items()
            if name.startswith("psc.")
        }
        if kind == "izhikevich":
            psc_args.pop("a")
            psc_args["A"] = 1.0  # restore the exact derived coefficient below
        synapse = HeterSynapsePSC(
            n, meta["receptors"], None, connection, base_psc=MODELS[kind][1], **psc_args
        )
        for name, value in values.items():
            prefix, field = name.split(".")
            module = neuron if prefix == "neuron" else synapse.base_psc
            if prefix == "psc":
                if meta["layouts"][name]["parameter"]:
                    delattr(module, field)
                    module.register_parameter(field, torch.nn.Parameter(value))
                else:
                    setattr(module, field, value)
            getattr(module, field).requires_grad_(
                meta["layouts"][name]["requires_grad"]
            )
        return SonataImport(
            RecurrentNN(
                neuron,
                synapse,
                backend=backend,
                gradient=BPTT(sparse=meta["sparse_gradient"]),
                update_state_names=meta["update_state_names"],
            ),
            dt,
        )
