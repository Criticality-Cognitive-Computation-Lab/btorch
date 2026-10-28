"""Prepare stateful recurrent modules for a whole tensor sequence operator."""

import json

import torch
from torch import nn

from ..connection.sparse import SparseConnection
from ..synapse import HeterSynapsePSC
from .adapters import connection_scope, prepare_cell
from .contracts import surrogate_support
from .operators import (
    _validate_compile_policy,
    _validate_parameter_shapes,
    _validate_poisson_fp32,
    encode_spec,
)
from .options import normalize_backend, resolve_model_options
from .registry import _NATIVE_GPU_BACKENDS, resolve_cell_backend
from .templates import get_template


_PAIRED_CHECKPOINT_INTERVAL = 16


def _reject_stochastic_input_composition(module):
    """Reject known input RNG whose step order cannot be preserved.

    Custom input callbacks used with a Poisson neuron must be
    deterministic.
    """
    from ..surrogate import PoissonRandomSpike

    stochastic = isinstance(module.neuron.surrogate_function, PoissonRandomSpike)
    if not stochastic:
        return
    for input_module in (module.neuron_inp_module, module.syn_inp_module):
        if input_module is None:
            continue
        if any(
            isinstance(
                child,
                (PoissonRandomSpike, nn.modules.dropout._DropoutNd, nn.RReLU),
            )
            for child in input_module.modules()
        ):
            raise NotImplementedError(
                "Native recurrent Poisson execution does not support stochastic "
                "input modules because their draws cannot be interleaved with "
                "per-step neuron execution; custom input callbacks must be "
                "deterministic"
            )


@torch.compiler.assume_constant_result
def _encode_recurrent_spec(spec):
    return json.dumps(spec, sort_keys=True)


@torch.compiler.assume_constant_result
def _compiled_sparse_route(connection, options, method):
    """Bind static RSNN execution choices before tracing tensor operations."""
    with connection_scope(options):
        return connection._compiled_method_route(method)


def _prepare_connection(connection, options=None):
    if isinstance(connection, nn.Identity):
        return {"connection": "identity"}, (), ()
    if isinstance(connection, nn.Linear):
        return (
            {"connection": "linear"},
            (
                connection.weight,
                *(() if connection.bias is None else (connection.bias,)),
            ),
            (),
        )
    if isinstance(connection, SparseConnection):
        if connection._member_shape:
            raise NotImplementedError(
                "Recurrent tensor operators do not yet fold sparse member batches"
            )
        cache = connection.cache
        from ...sparse.runtime.method import resolve_method

        method = resolve_method(
            options.connection_method
            if options is not None and options.connection_method != "auto"
            else connection.method
        )
        explicit_route = (
            _compiled_sparse_route(connection, options, method)
            if options is not None
            and (
                options.connection_method != "auto"
                or options.connection != "auto"
                or method != connection._plan.method
            )
            else None
        )
        values = connection._run_weight
        if values is None:
            values = connection.weight()
            if connection._always_permute or not cache.identity_perm:
                values = values.index_select(-1, cache.perm)
        use_fallback = (
            connection._plan.algorithm == "push"
            and connection._run_task_values is None
            and connection._plan.route.key.representation != "spike_list_csr"
            and connection._fallback_route is not None
        )
        route = (
            explicit_route[0]
            if explicit_route is not None
            else (
                connection._fallback_route_token
                if use_fallback
                else connection._route_token
            )
        )
        topology = (cache.crow, cache.col, cache.t_crow, cache.t_col, cache.t_perm)
        if connection._run_task_values is not None and explicit_route is None:
            topology += (connection._run_task_values,)
        return (
            {
                "connection": "sparse",
                "route": route,
                "max_density": explicit_route[1]
                if explicit_route is not None
                else (None if use_fallback else connection._max_density),
            },
            (
                values,
                *(() if connection.bias is None else (connection.bias,)),
            ),
            topology,
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
    module,
    x_seq,
    x_syn=None,
    *,
    loop_args=None,
    _chunk=False,
    uniform_seq=None,
    **kwargs,
):
    """Execute a complete recurrent trajectory with explicit tensor inputs."""
    from ..rnn import _sparse_run_scope

    if kwargs or loop_args not in (None, (0,), (0, 1)):
        raise ValueError("Cell recurrent sequences accept time-major x and x_syn only")
    chunked = module.chunk_size is not None and x_seq.shape[0] > module.chunk_size
    if not _chunk and (chunked or module.grad_checkpoint or module.cpu_offload):
        # Bind sparse execution before checkpoint snapshots are taken. Preparing
        # it inside the first chunk would save a different topology/task-value
        # input list than recomputation restores, and repeat preparation per
        # chunk instead of owning it for the whole trajectory.
        with (
            connection_scope(normalize_backend(module.backend)),
            _sparse_run_scope(module, x_seq),
        ):
            return module._multi_step_forward_impl(
                x_seq, *(() if x_syn is None else (x_syn,)), loop_args=loop_args
            )
    _reject_stochastic_input_composition(module)
    if module.neuron_inp_module is not None:
        x_seq = module.neuron_inp_module(x_seq)
    if module.syn_inp_module is not None:
        x_syn = module.syn_inp_module(x_syn)
    backend = module.backend
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
    options = resolve_model_options(backend, neuron_spec.model, synapse_spec.model)
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
            )
        )
    state_names = [
        *(f"neuron.{name}" for name in neuron_names),
        *(
            f"synapse.{'base_psc.' if cell is not synapse else ''}{name}"
            for name in synapse_names
        ),
        *(["synapse.delayed_spike.history"] if n_delay else []),
    ]
    from .templates import get_template

    charged_bindings = tuple(
        (name, attribute)
        for name, attribute in get_template(neuron_spec).record_attributes
        if attribute in module.neuron._memories_rv
    )
    charged_names = [name for name, _ in charged_bindings]
    charged_attributes = [f"neuron.{attribute}" for _, attribute in charged_bindings]
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
    if x_syn is not None and (
        x_syn.shape != x_seq.shape or x_syn.device != x_seq.device
    ):
        raise ValueError("x_syn must have the same time-first shape and device as x")
    x_syn_seq = x_seq.new_empty(0) if x_syn is None else x_syn
    connection_tensors = tuple(
        (f"connection parameter {name}", value)
        for name, value in cell.linear.named_parameters()
    )
    delay_tensors = (
        (("initial delay history", synapse.delayed_spike.history),) if n_delay else ()
    )
    _validate_poisson_fp32(
        neuron_spec,
        (
            ("x_seq", x_seq),
            ("x_syn_seq", x_syn_seq),
            *(
                (f"initial neuron {name}", value)
                for name, value in zip(neuron_names, neuron_states)
            ),
            *(
                (f"initial synapse {name}", value)
                for name, value in zip(synapse_names, synapse_states)
            ),
            *(
                (f"neuron parameter {name}", value)
                for name, value in zip(
                    get_template(neuron_spec).parameter_names, neuron_parameters
                )
            ),
            *(
                (f"synapse parameter {name}", value)
                for name, value in zip(
                    get_template(synapse_spec).parameter_names, synapse_parameters
                )
            ),
            *delay_tensors,
            *connection_tensors,
        ),
    )
    if torch.compiler.is_compiling():
        connection_spec, connection_parameters, topology = _prepare_connection(
            cell.linear, options
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
            uniform_seq,
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
            uniform_seq,
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
    uniform_seq=None,
):
    from .operators import poisson_uniforms
    from .recurrent_operator import inference, training

    _validate_compile_policy(
        options.neuron,
        options.enable_fp_fusion,
        options.precise_division,
        options.precise_exp,
    )
    if (x_syn_seq.shape != x_seq.shape and x_syn_seq.shape != (0,)) or (
        x_syn_seq.device != x_seq.device
    ):
        raise ValueError("x_syn must have the same time-first shape and device as x")
    # Resolve both implementations and all tensor broadcast layouts before a
    # stochastic neuron consumes the first global-generator draw. CUDA dt value
    # checks remain device assertions so capture does not introduce host sync.
    resolve_cell_backend(options.neuron, neuron_spec, x_seq.device)
    resolve_cell_backend(options.neuron, synapse_spec, x_seq.device)
    _validate_parameter_shapes(
        neuron_parameters,
        neuron_spec,
        states[: len(neuron_names)],
        x_seq.shape[1:],
        label="Neuron",
    )
    # Autocast applies to the eligible linear operation, not to neuron/PSC
    # memories or parameters. Cast once outside the opaque trajectory so
    # autograd retains the master-weight cast and reverse/checkpoint replay
    # consume exactly the forward dtype even outside the autocast context.
    if connection_spec["connection"] == "linear" and torch.is_autocast_enabled(
        x_seq.device.type
    ):
        dtype = torch.get_autocast_dtype(x_seq.device.type)
        connection_parameters = [
            value.to(dtype)
            if value.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else value
            for value in connection_parameters
        ]
    parameters = [*neuron_parameters, *synapse_parameters, *connection_parameters]
    needs_backward = torch.is_grad_enabled() and any(
        t.requires_grad for t in (x_seq, x_syn_seq, *states, *parameters)
    )
    charged_indices = list(range(len(states), len(states) + len(charged_names)))
    checkpoint_interval = (
        _PAIRED_CHECKPOINT_INTERVAL
        if needs_backward
        and options.neuron in _NATIVE_GPU_BACKENDS
        and x_seq.device.type == "cuda"
        and get_template(neuron_spec).recurrent_checkpoint
        and not (neuron_spec.poisson is not None and neuron_spec.poisson.spiking)
        and not neuron_spec.refractory
        and get_template(synapse_spec).recurrent_checkpoint
        and len(neuron_names) == 1
        and len(synapse_names) == 1
        and not n_delay
        and cell is module.synapse
        and not options.merge_synapse
        and all(index >= len(states) for index in indices)
        else 0
    )
    if needs_backward:
        residual_indices = () if checkpoint_interval else range(len(states))
        history_indices = list(dict.fromkeys((*residual_indices, *indices)))
        # The general reverse path retains physical trajectories; the selected
        # paired checkpoint plan reconstructs them. Charged values only need a
        # trajectory when the caller requested one publicly; otherwise the
        # final pre-spike memory is enough to seed its last-step cotangent.
        history_indices.extend(
            -offset - 1
            for offset, index in enumerate(charged_indices)
            if index not in history_indices
        )
    else:
        history_indices = list(dict.fromkeys(indices))
        # Negative entries request only the final charged value. This keeps
        # pre-spike module state current without retaining an unrequested
        # time-by-state trajectory during inference.
        history_indices.extend(
            -offset - 1
            for offset, index in enumerate(charged_indices)
            if index not in history_indices
        )
    gradient_activity = (
        needs_backward
        and module.gradient.sparse
        and connection_spec["connection"] == "sparse"
        and options.neuron in _NATIVE_GPU_BACKENDS
        and x_seq.device.type == "cuda"
        and get_template(neuron_spec).spike_gradient_input is not None
        and surrogate_support(neuron_spec) is not None
        and not n_delay
        and not x_syn_seq.numel()
    )
    if gradient_activity:
        activity_index = len(states) + len(charged_names)
        history_indices.append(activity_index)
    # A Python branch specializes the algorithm choice with a shape guard.
    # bool(SymBool) can remain symbolic in a JSON metadata dictionary under
    # Dynamo's automatic dynamic shapes and cannot cross the custom-op ABI.
    gradient_list = False
    if gradient_activity and options.neuron == "triton":
        if x_seq.numel() // max(1, x_seq.shape[0]) < 2**31:
            if (
                not connection_parameters[0].requires_grad
                or x_seq[0].numel() == topology[2].numel() - 1
            ):
                gradient_list = True
    recurrent_spec = {
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
        "merge_receptors": module.synapse.n_receptor if options.merge_synapse else 0,
        "neuron_shape": list(module.neuron.n_neuron),
        # Execution policy is deliberately outside both semantic NeuronSpec
        # encodings, but remains part of this captured recurrent call.
        "enable_fp_fusion": options.enable_fp_fusion,
        "precise_division": options.precise_division,
        "precise_exp": options.precise_exp,
        "gradient_activity": activity_index if gradient_activity else None,
        "gradient_list": gradient_list,
    }
    if (
        needs_backward
        and x_seq.dtype != torch.float32
        and (options.neuron in _NATIVE_GPU_BACKENDS or states[0].dtype == torch.float32)
    ):
        # The public raster follows input dtype, while reverse must consume
        # the producer's existing calculation-precision output without recasts.
        recurrent_spec["save_compute_spikes"] = True
    if checkpoint_interval:
        # This first residual slice is the native paired LIF + ExponentialPSC
        # schedule only. The encoded layout is part of operator specialization.
        recurrent_spec["paired_checkpoint_interval"] = checkpoint_interval
    encoded = _encode_recurrent_spec(recurrent_spec)
    operator = training if needs_backward else inference
    poisson = neuron_spec.poisson
    if poisson is not None and options.neuron not in _NATIVE_GPU_BACKENDS:
        raise NotImplementedError(
            "Native recurrent Poisson execution requires a built-in GPU backend"
        )
    if poisson is not None and poisson.spiking:
        if uniform_seq is None:
            uniform_seq = poisson_uniforms(x_seq)
        elif (
            uniform_seq.shape != x_seq.shape
            or uniform_seq.device != x_seq.device
            or uniform_seq.dtype != torch.float32
            or uniform_seq.requires_grad
        ):
            raise ValueError(
                "Poisson uniforms must be non-differentiable FP32 values with "
                "the time-first input shape and device"
            )
    else:
        uniform_seq = x_seq.new_empty(0, dtype=torch.float32)
    arguments = (
        x_seq,
        x_syn_seq,
        states,
        parameters,
        list(topology),
        encoded,
        history_indices,
        options.neuron,
        torch.as_tensor(neuron_spec.dt, dtype=torch.float32, device=x_seq.device),
    )
    result = operator(
        *arguments,
        *((uniform_seq,) if poisson is not None and poisson.spiking else ()),
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
    for offset, (name, index) in enumerate(zip(charged_names, charged_indices)):
        attribute = dict(get_template(neuron_spec).record_attributes)[name]
        value = histories[index][-1] if index in histories else histories[-offset - 1]
        setattr(module.neuron, attribute, value)
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
