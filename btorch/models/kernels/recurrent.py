"""Prepare stateful recurrent modules for a whole tensor sequence operator."""

import json

import torch
from torch import nn

from ..connection.sparse import SparseConnection
from ..synapse import HeterSynapsePSC
from .adapters import connection_scope, prepare_cell
from .operators import encode_spec
from .options import normalize_backend


@torch.compiler.assume_constant_result
def _encode_recurrent_spec(spec):
    return json.dumps(spec, sort_keys=True)


def _prepare_connection(connection):
    if isinstance(connection, nn.Identity):
        return {"connection": "identity"}, (), ()
    if isinstance(connection, nn.Linear):
        return (
            {"connection": "linear"},
            (
                connection.weight.float(),
                *(() if connection.bias is None else (connection.bias.float(),)),
            ),
            (),
        )
    if isinstance(connection, SparseConnection):
        if connection._member_shape:
            raise NotImplementedError(
                "Recurrent tensor operators do not yet fold sparse member batches"
            )
        cache = connection.cache
        values = connection._run_weight
        if values is None:
            values = connection.weight()
            if connection._always_permute or not cache.identity_perm:
                values = values.index_select(-1, cache.perm)
        route = (
            connection._fallback_route_token
            if connection._plan.algorithm == "push"
            and connection._run_task_values is None
            and connection._fallback_route is not None
            else connection._route_token
        )
        return (
            {"connection": "sparse", "route": route},
            (
                values.float(),
                *(() if connection.bias is None else (connection.bias.float(),)),
            ),
            (cache.crow, cache.col, cache.t_crow, cache.t_col, cache.t_perm),
        )
    raise NotImplementedError(
        f"No recurrent tensor connection adapter for {type(connection).__name__}"
    )


def _merge_parameter_signature(module):
    from .adapters import _model_layout

    _, _, names = _model_layout(module.synapse.base_psc)
    return tuple(
        (
            id(value),
            value._version,
            value.requires_grad,
            tuple(value.shape),
            tuple(value.stride()),
        )
        for value in (getattr(module.synapse.base_psc, name) for name in names)
    )


def validate_merge_module(module):
    """Validate receptor compatibility before compiling or capturing
    kernels."""
    from .adapters import _model_layout

    synapse = module.synapse
    if not isinstance(synapse, HeterSynapsePSC):
        raise ValueError("merge_synapse requires a homogeneous receptor PSC population")
    if module.update_state_names is not None and any(
        "base_psc" in name or "receptor" in name for name in module.update_state_names
    ):
        raise ValueError("Merged states cannot record independent receptor currents")
    _, _, names = _model_layout(synapse.base_psc)
    for name in names:
        parameter = getattr(synapse.base_psc, name)
        if parameter.numel() == 1:
            continue
        if parameter.requires_grad:
            raise ValueError(
                "Independent trainable receptor parameters cannot be merged"
            )
        values = parameter.reshape(synapse.size, synapse.n_receptor)
        if not torch.equal(values, values[:, :1].expand_as(values)):
            raise ValueError("Receptor dynamics parameters must match before merging")
    synapse._kernel_merge_parameter_signature = _merge_parameter_signature(module)


def _merge_receptors(module, states, parameters):
    synapse = module.synapse
    if not torch.compiler.is_compiling() and getattr(
        synapse, "_kernel_merge_parameter_signature", None
    ) != _merge_parameter_signature(module):
        validate_merge_module(module)
    reduced_parameters = []
    for parameter in parameters:
        if parameter.numel() == 1:
            reduced_parameters.append(parameter)
            continue
        values = parameter.reshape(synapse.size, synapse.n_receptor)
        reduced_parameters.append(values[:, 0])
    if states[0].shape[-1] == synapse.size * synapse.n_receptor:
        states = tuple(
            value.reshape(*value.shape[:-1], synapse.size, synapse.n_receptor).sum(-1)
            for value in states
        )
    synapse._kernel_merge_active = True
    return states, tuple(reduced_parameters)


def prepare_merged_state(module):
    """Set stable merged state shapes before CUDA graph snapshots are taken."""
    cell = module.synapse.base_psc
    _, names, states, parameters = prepare_cell(cell, cell.psc.unsqueeze(0))
    states, _ = _merge_receptors(module, states, parameters)
    for name, value in zip(names, states):
        setattr(cell, name, value)
    module.synapse.psc = cell.psc.reshape(*cell.psc.shape[:-1], *module.neuron.n_neuron)


def recurrent_sequence(
    module, x_seq, x_syn=None, *, loop_args=None, _chunk=False, **kwargs
):
    """Execute a complete recurrent trajectory with explicit tensor inputs."""
    from ..rnn import _sparse_run_scope

    if kwargs or loop_args not in (None, (0,), (0, 1)):
        raise ValueError("Cell recurrent sequences accept time-major x and x_syn only")
    if not _chunk and (
        module.chunk_size or module.grad_checkpoint or module.cpu_offload
    ):
        with connection_scope(normalize_backend(module.backend)):
            return module._multi_step_forward_impl(
                x_seq, *(() if x_syn is None else (x_syn,)), loop_args=loop_args
            )
    if module.neuron_inp_module is not None:
        x_seq = module.neuron_inp_module(x_seq)
    if module.syn_inp_module is not None:
        x_syn = module.syn_inp_module(x_syn)
    options = normalize_backend(module.backend)
    neuron, synapse = module.neuron, module.synapse
    cell = synapse.base_psc if isinstance(synapse, HeterSynapsePSC) else synapse
    if not x_seq.shape[0]:
        return x_seq.clone(), {}
    neuron_spec, neuron_names, neuron_states, neuron_parameters = prepare_cell(
        neuron, x_seq
    )
    synapse_spec, synapse_names, synapse_states, synapse_parameters = prepare_cell(
        cell, cell.psc.unsqueeze(0)
    )
    if options.merge_synapse:
        synapse_states, synapse_parameters = _merge_receptors(
            module, synapse_states, synapse_parameters
        )
    states = [*neuron_states, *synapse_states]
    n_delay = synapse.n_delay if synapse.delayed_spike is not None else 0
    if n_delay:
        if synapse.delayed_spike.use_circular_buffer:
            raise NotImplementedError(
                "Recurrent tensor sequences require differentiable delay history"
            )
        states.append(
            synapse.delayed_spike.history.movedim(
                -len(synapse.delayed_spike.n_neuron) - 1, 0
            ).float()
        )
    state_names = [
        *(f"neuron.{name}" for name in neuron_names),
        *(
            f"synapse.{'base_psc.' if cell is not synapse else ''}{name}"
            for name in synapse_names
        ),
        *(["synapse.delayed_spike.history"] if n_delay else []),
    ]
    charged_names = []
    charged_attributes = []
    if module.neuron.pre_spike_v:
        charged_names.append("v_charged")
        charged_attributes.append("neuron.v_pre_spike")
        if "u_pre_spike" in module.neuron._memories_rv:
            charged_names.append("u_charged")
            charged_attributes.append("neuron.u_pre_spike")
    all_record_names = [*state_names, *charged_attributes]
    requested = (
        list(module.update_state_names)
        if module.update_state_names is not None
        else all_record_names
    )
    # Sum receptor PSC records from the exact same per-receptor history used by
    # backward; retain the individual dynamics for get_psc and continuation.
    aliases = {"synapse.psc": len(neuron_names)} if cell is not synapse else {}
    if options.merge_synapse and module.update_state_names is None:
        requested = ["neuron." + name for name in neuron_names] + ["synapse.psc"]
    indices = [
        aliases[name] if name in aliases else all_record_names.index(name)
        for name in requested
    ]
    x_syn_seq = torch.zeros_like(x_seq) if x_syn is None else x_syn
    if torch.compiler.is_compiling():
        connection_spec, connection_parameters, topology = _prepare_connection(
            cell.linear
        )
        return _execute(
            module,
            x_seq,
            x_syn_seq,
            states,
            neuron_parameters,
            synapse_parameters,
            connection_parameters,
            topology,
            connection_spec,
            neuron_spec,
            synapse_spec,
            neuron_names,
            synapse_names,
            n_delay,
            requested,
            indices,
            charged_names,
            options,
            cell,
        )
    with connection_scope(options), _sparse_run_scope(module, x_seq):
        connection_spec, connection_parameters, topology = _prepare_connection(
            cell.linear
        )
        return _execute(
            module,
            x_seq,
            x_syn_seq,
            states,
            neuron_parameters,
            synapse_parameters,
            connection_parameters,
            topology,
            connection_spec,
            neuron_spec,
            synapse_spec,
            neuron_names,
            synapse_names,
            n_delay,
            requested,
            indices,
            charged_names,
            options,
            cell,
        )


def _execute(
    module,
    x_seq,
    x_syn_seq,
    states,
    neuron_parameters,
    synapse_parameters,
    connection_parameters,
    topology,
    connection_spec,
    neuron_spec,
    synapse_spec,
    neuron_names,
    synapse_names,
    n_delay,
    requested,
    indices,
    charged_names,
    options,
    cell,
):
    from .recurrent_operator import inference, training

    parameters = [*neuron_parameters, *synapse_parameters, *connection_parameters]
    needs_backward = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x_seq, x_syn_seq, *states, *parameters)
    )
    charged_indices = list(range(len(states), len(states) + len(charged_names)))
    history_indices = list(
        dict.fromkeys(
            (*range(len(states)), *indices, *charged_indices)
            if needs_backward
            else (*indices, *charged_indices)
        )
    )
    encoded = _encode_recurrent_spec(
        {
            **connection_spec,
            "neuron": encode_spec(neuron_spec),
            "synapse": encode_spec(synapse_spec),
            "neuron_states": len(neuron_names),
            "synapse_states": len(synapse_names),
            "neuron_parameters": len(neuron_parameters),
            "synapse_parameters": len(synapse_parameters),
            "delay": n_delay,
            "charged_names": charged_names,
            "receptors": module.synapse.n_receptor
            if cell is not module.synapse and not options.merge_synapse
            else 0,
            "merge_receptors": module.synapse.n_receptor
            if options.merge_synapse
            else 0,
            "neuron_shape": list(module.neuron.n_neuron),
        }
    )
    operator = training if needs_backward else inference
    result = operator(
        x_seq,
        x_syn_seq,
        states,
        parameters,
        list(topology),
        encoded,
        history_indices,
        options.cell,
        torch.as_tensor(neuron_spec.dt, dtype=torch.float32, device=x_seq.device),
    )
    ns = len(neuron_names)
    for name, value in zip(neuron_names, result[1 : 1 + ns]):
        setattr(module.neuron, name, value)
    for name, value in zip(synapse_names, result[1 + ns : 1 + ns + len(synapse_names)]):
        setattr(cell, name, value)
    if n_delay:
        delay = module.synapse.delayed_spike
        delay.history = result[len(states)].movedim(0, -len(delay.n_neuron) - 1)
    histories = dict(zip(history_indices, result[1 + len(states) :]))
    for offset, name in enumerate(charged_names):
        attribute = "v_pre_spike" if name == "v_charged" else "u_pre_spike"
        setattr(module.neuron, attribute, histories[len(states) + offset][-1])
    records = {}
    for name, index in zip(requested, indices):
        value = histories[index]
        if (
            cell is not module.synapse
            and not options.merge_synapse
            and name == "synapse.psc"
        ):
            value = value.reshape(
                *value.shape[:-1], *module.neuron.n_neuron, module.synapse.n_receptor
            ).sum(-1)
        records[name] = value
    module.synapse._kernel_sequence_started = True
    if options.merge_synapse:
        module.synapse.psc = cell.psc
    elif cell is not module.synapse:
        module.synapse.psc = cell.psc.reshape(
            *cell.psc.shape[:-1], *module.neuron.n_neuron, module.synapse.n_receptor
        ).sum(-1)
    return result[0], records
