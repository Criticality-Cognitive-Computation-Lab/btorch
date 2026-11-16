"""Prepare stateful recurrent modules for a whole tensor sequence operator."""

import json
from contextlib import nullcontext
from functools import lru_cache

import torch
from torch import nn

from ..connection.sparse import SparseConnection
from ..synapse import DelayedPSC, HeterSynapsePSC
from .adapters import connection_scope, prepare_cell
from .contracts import surrogate_support
from .operators import (
    _validate_compile_policy,
    _validate_parameter_shapes,
    _validate_poisson_fp32,
    encode_spec,
)
from .options import _event_connection_backend, normalize_backend, resolve_model_options
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
    return _encoded_recurrent_items(
        tuple(
            (name, tuple(value) if isinstance(value, list) else value)
            for name, value in sorted(spec.items())
        )
    )


@lru_cache(maxsize=256)
def _encoded_recurrent_items(items):
    return json.dumps(dict(items), sort_keys=True)


@torch.compiler.assume_constant_result
def _compiled_sparse_route(connection, options, method):
    """Bind static RSNN execution choices before tracing tensor operations."""
    with connection_scope(options):
        return connection._compiled_method_route(method)


@torch.compiler.assume_constant_result
def _delay_segments(connection):
    """Prepare topology-derived segment metadata outside compiled stepping."""
    return connection.cache.prepare_delay_segments(
        connection.n_pre, max(connection.n_delay, 1)
    )


@torch.compiler.assume_constant_result
def _edge_segments(connection):
    """Prepare the canonical inverse for heterogeneous delayed segment
    reads."""
    return connection.cache.prepare_edge_segments(
        connection.n_pre, max(connection.n_delay, 1)
    )


def _prepare_connection(connection, options=None, *, online=False):
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
        if (
            options is not None
            and options.spike_selection != "auto"
            and method == "auto"
        ):
            method = "push"
        if (
            options is not None
            and options.spike_selection != "auto"
            and method != "push"
        ):
            raise NotImplementedError(
                "Explicit spike discovery requires a sparse push connection; use "
                "spike_selection='auto' for the ordinary pull path"
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
        if (
            connection._run_task_values is not None
            and explicit_route is None
            and not online
        ):
            topology += (connection._run_task_values,)
        if online:
            # Online STDP owns canonical edge identity.  Keep the edge index
            # arrays beside the execution topology, but never pass packed task
            # values: they are snapshots and would become stale after step 0.
            topology += (
                connection.indices[1],
                connection.indices[0],
                connection.delay
                if connection.delay is not None
                else connection.indices[1].new_empty(0),
                cache.perm,
            )
        connection_spec = {
            "connection": "sparse",
            "online_permuted": bool(
                online and (connection._always_permute or not cache.identity_perm)
            ),
            "method": method,
            "route": route,
            "max_density": explicit_route[1]
            if explicit_route is not None
            else (None if use_fallback else connection._max_density),
        }
        event_backend = _event_connection_backend(options) if online else None
        event_connection = event_backend is not None
        if event_connection:
            connection_spec["event_connection"] = event_backend
        shared_traces = online and any(
            domain != "edge" for domain in connection.plasticity.state_domains.values()
        )
        segment_signals = online and any(
            domain == "segment" for _, domain in connection.plasticity.signal_domains
        )
        if options is not None and (
            options.recurrent == "helion"
            or event_connection
            or shared_traces
            or segment_signals
        ):
            connection_spec["max_degree"] = cache.max_source_degree
            if online:
                from .plasticity import get_plasticity_template

                rule = connection.plasticity
                template = get_plasticity_template(
                    rule._kernel_model,
                    tuple(rule.state_domains.items()),
                    rule.signal_domains,
                )
                if (
                    template.events is not None
                    or "segment" in rule.state_domains.values()
                    or segment_signals
                ):
                    if event_connection and template.events is None:
                        raise NotImplementedError(
                            "Two-kernel sparse learning requires exact "
                            "event-only silent evolution"
                        )
                    segments = _delay_segments(connection)
                    # Event traversal shares existing canonical edge maps;
                    # continuous traversal needs no segment representation.
                    topology += segments[:-1]
                    connection_spec["n_segments"] = segments[1].numel()
                    if (shared_traces or segment_signals) and connection.n_delay > 1:
                        topology += (_edge_segments(connection),)
                else:
                    if event_connection:
                        raise NotImplementedError(
                            "Two-kernel sparse learning requires a rule with "
                            "exact event-only silent evolution"
                        )
                if template.events is None:
                    # Continuous canonical-edge traversal needs receptor IDs
                    # even when compact traces also request segment metadata.
                    topology += (
                        connection.receptor
                        if connection.receptor is not None
                        else connection.indices[0].new_empty(0),
                    )
        return (
            connection_spec,
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
    *signal_inputs,
    loop_args=None,
    _chunk=False,
    uniform_seq=None,
    plasticity_signals=None,
    **kwargs,
):
    """Execute a complete recurrent trajectory with explicit tensor inputs."""
    from ..rnn import _sparse_run_scope

    signal_names = getattr(module, "_stdp_sequence_names", ())
    if (
        kwargs
        or len(signal_inputs) != len(signal_names)
        or (
            loop_args is not None
            and (
                not set(loop_args) <= {0, 1, *range(2, 2 + len(signal_inputs))}
                or not {0, *range(2, 2 + len(signal_inputs))} <= set(loop_args)
            )
        )
    ):
        raise ValueError(
            "Recurrent sequences accept time-major x, x_syn and declared "
            "plasticity signals"
        )
    if (
        not _chunk
        and module.cudagraph is True
        and torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    ):
        from .recurrent_graph import RecurrentGraphs

        steps = x_seq.shape[0]
        if module.chunk_size is not None and steps > module.chunk_size:
            sections = (steps + module.chunk_size - 1) // module.chunk_size
            steps = (
                module.chunk_size
                if module.chunk_mode == "split"
                else (steps + sections - 1) // sections
            )
        graphs = getattr(module, "_training_graphs", None)
        if graphs is None or graphs.steps != steps:
            module._training_graphs = RecurrentGraphs(
                steps, module._cudagraph_runner.warmup
            )
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
                x_seq,
                *(
                    (x_syn, *signal_inputs)
                    if signal_inputs or x_syn is not None
                    else ()
                ),
                loop_args=loop_args,
                **(
                    {"plasticity_signals": plasticity_signals}
                    if plasticity_signals is not None
                    else {}
                ),
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
    if (
        isinstance(cell.linear, SparseConnection)
        and cell.linear.transmission is not None
    ):
        raise NotImplementedError(
            "Transmission rules currently support eager connection "
            "steps only; optimized recurrent execution is not implemented"
        )
    options = normalize_backend(backend)
    if (
        isinstance(cell.linear, SparseConnection)
        and cell.linear.plasticity is not None
        and (
            options.recurrent == "helion"
            or options.spike_selection != "auto"
            or "segment" in cell.linear.plasticity.state_domains.values()
            or any(
                domain == "segment"
                for _, domain in cell.linear.plasticity.signal_domains
            )
        )
    ):
        # Build lazy topology buffers before Dynamo guards the module's buffer
        # shapes during precision discovery. Preparation is outside the graph.
        from .plasticity import get_plasticity_template

        if (
            get_plasticity_template(
                cell.linear.plasticity._kernel_model,
                tuple(cell.linear.plasticity.state_domains.items()),
                cell.linear.plasticity.signal_domains,
            ).traces
            is not None
        ):
            _delay_segments(cell.linear)
            if (
                "segment" in cell.linear.plasticity.state_domains.values()
                or any(
                    domain == "segment"
                    for _, domain in cell.linear.plasticity.signal_domains
                )
            ) and cell.linear.n_delay > 1:
                _edge_segments(cell.linear)
    neuron_spec, neuron_names, neuron_states, neuron_parameters = prepare_cell(
        neuron, x_seq
    )
    synapse_spec, synapse_names, synapse_states, synapse_parameters = prepare_cell(
        cell, cell.psc.unsqueeze(0)
    )
    options = resolve_model_options(backend, neuron_spec.model, synapse_spec.model)
    if options.recurrent == "helion" and isinstance(synapse, DelayedPSC):
        raise NotImplementedError(
            "HELION recurrent execution does not support the uniform DelayedPSC "
            "wrapper; use per-edge delay-expanded sparse connections"
        )
    if options.recurrent == "helion" and options.merge_synapse:
        raise NotImplementedError(
            "HELION recurrent execution does not support merge_synapse"
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
    online_connection = (
        cell.linear if isinstance(cell.linear, SparseConnection) else None
    )
    if online_connection is not None and online_connection.plasticity is not None:
        from .plasticity import get_plasticity_template

        rule = online_connection.plasticity
        plasticity_template = get_plasticity_template(
            rule._kernel_model, tuple(rule.state_domains.items()), rule.signal_domains
        )
        online_names = (
            *(
                "synapse.linear.plasticity." + name
                for name in plasticity_template.states
            ),
            "synapse.linear.weight.value",
        )
    else:
        online_names = ()
    available_record_names = [*all_record_names, *online_names]
    requested = (
        list(module.update_state_names)
        if module.update_state_names is not None
        else [*all_record_names, *online_names[:-1]]
    )
    # Sum receptor PSC records from the exact same per-receptor history used by
    # backward; retain the individual dynamics for get_psc and continuation.
    aliases = {"synapse.psc": len(neuron_names)} if cell is not synapse else {}
    if options.merge_synapse and module.update_state_names is None:
        requested = ["neuron." + name for name in neuron_names] + ["synapse.psc"]
    indices = [
        aliases[name] if name in aliases else available_record_names.index(name)
        for name in requested
    ]
    if x_syn is not None and (
        x_syn.shape != x_seq.shape or x_syn.device != x_seq.device
    ):
        raise ValueError("x_syn must have the same time-first shape and device as x")
    x_syn_seq = x_seq.new_empty(0) if x_syn is None else x_syn
    stdp_bindings = getattr(module, "_stdp_bindings", ())
    online_binding = stdp_bindings[0] if len(stdp_bindings) == 1 else None
    online = online_binding is not None
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
            cell.linear, options, online=online
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
            online_binding,
            {
                **(plasticity_signals or {}),
                **dict(zip(signal_names, signal_inputs, strict=True)),
            },
        )
    with connection_scope(options), _sparse_run_scope(module, x_seq):
        connection_spec, connection_parameters, topology = _prepare_connection(
            cell.linear, options, online=online
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
            online_binding,
            {
                **(plasticity_signals or {}),
                **dict(zip(signal_names, signal_inputs, strict=True)),
            },
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
    online_binding=None,
    plasticity_signals=None,
):
    from .operators import poisson_uniforms
    from .recurrent_operator import (
        _physical_dtypes,
        inference,
        online_inference,
        training,
    )

    _validate_compile_policy(
        options.neuron,
        options.enable_fp_fusion,
        (not options.fast_math),
        (not options.fast_math),
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
    online_weight = None
    online_traces = ()
    online_parameters = ()
    online_signals = []
    signal_bindings = []
    plasticity_model = None
    if online_binding is not None:
        from .plasticity import get_plasticity_template

        connection, _ = online_binding
        rule = connection.plasticity
        plasticity_model = rule._kernel_model
        plasticity_template = get_plasticity_template(
            plasticity_model, tuple(rule.state_domains.items()), rule.signal_domains
        )
        online_weight = connection.weight()
        online_traces = tuple(
            getattr(rule, name) for name in plasticity_template.states
        )
        online_parameters = tuple(
            getattr(rule, name) for name in plasticity_template.parameters
        )
        from ..connection.plasticity import ExternalSignal

        external = plasticity_signals or {}
        attributes = dict(get_template(neuron_spec).record_attributes)
        live_attributes = dict(module._stdp_signal_bindings)
        for name, _domain in plasticity_template.signals:
            source = rule.signals[name]
            if isinstance(source, ExternalSignal):
                signal_bindings.append(
                    (
                        name,
                        "sequence" if source.time_first else "constant",
                        len(online_signals),
                    )
                )
                online_signals.append(external[name])
            else:
                attribute = live_attributes[name]
                if source.stage == "state":
                    signal_bindings.append(
                        (name, "state", neuron_names.index(attribute))
                    )
                else:
                    record = next(
                        key for key in charged_names if attributes[key] == attribute
                    )
                    signal_bindings.append(
                        (name, "charged", charged_names.index(record))
                    )
        # The canonical weight is a mutable custom-op argument. The first
        # execution parameter remains the live ordered CSR view when needed.
        connection_parameters = connection_parameters[1:]
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
        if options.recurrent == "helion":
            raise NotImplementedError(
                "HELION recurrent execution currently supports inference only"
            )
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
        "precise_division": (not options.fast_math),
        "precise_exp": (not options.fast_math),
        "recurrent": options.recurrent,
        "spike_selection": options.spike_selection,
        "gradient_activity": activity_index if gradient_activity else None,
        "gradient_list": gradient_list,
    }
    if online_binding is not None:
        from .plasticity import plasticity_template_revision

        recurrent_spec["plasticity"] = plasticity_model
        recurrent_spec["plasticity_revision"] = plasticity_template_revision(
            plasticity_model
        )
        recurrent_spec["plasticity_domains"] = tuple(rule.state_domains.items())
        recurrent_spec["plasticity_signal_domains"] = rule.signal_domains
        recurrent_spec["plasticity_signals"] = signal_bindings
    if options.spike_selection == "spike_list" and (
        connection_spec["connection"] != "sparse"
        or connection_spec.get("method") != "push"
    ):
        raise NotImplementedError(
            "spike_selection='spike_list' requires a sparse push connection"
        )
    if (
        options.recurrent == "standard"
        and options.spike_selection == "scan"
        and (not connection_spec.get("event_connection"))
    ):
        raise NotImplementedError(
            "Standard scan discovery currently requires event-only online STDP; "
            "use recurrent='helion' for static scan execution"
        )
    if (
        options.recurrent == "standard"
        and options.spike_selection == "spike_list"
        and (n_delay or x_syn_seq.numel())
    ):
        raise NotImplementedError(
            "Explicit neuron-produced lists currently require zero delay and no "
            "x_syn input; use spike_selection='auto' for route-side packing"
        )
    if connection_spec.get("event_connection") and (
        n_delay or x_syn_seq.numel() or options.merge_synapse
    ):
        raise NotImplementedError(
            "Two-kernel sparse event learning currently requires zero delay, "
            "no x_syn input and unmerged receptor states"
        )
    if options.recurrent == "helion":
        if options.neuron not in ("triton", "tilelang"):
            raise NotImplementedError(
                "HELION recurrent execution requires the Triton or TileLang "
                "neuron backend"
            )
        if x_syn_seq.numel():
            raise NotImplementedError(
                "HELION recurrent execution currently does not support x_syn"
            )
        if connection_spec["connection"] != "sparse":
            raise NotImplementedError(
                "HELION recurrent execution requires a static sparse CSR connection"
            )
        if connection_spec.get("method") != "push":
            raise NotImplementedError(
                "HELION recurrent execution requires connection_method='push'"
            )
    neuron_dtype, psc_dtype = _physical_dtypes(
        x_seq,
        x_syn_seq,
        states,
        [*neuron_parameters, *synapse_parameters, online_weight, *connection_parameters]
        if online_binding is not None
        else parameters,
        recurrent_spec,
    )
    if needs_backward and x_seq.dtype != neuron_dtype:
        # The public raster follows input dtype, while reverse must consume
        # the producer's existing calculation-precision output without recasts.
        recurrent_spec["save_compute_spikes"] = True
    if checkpoint_interval:
        # This first residual slice is the native paired LIF + ExponentialPSC
        # schedule only. The encoded layout is part of operator specialization.
        recurrent_spec["paired_checkpoint_interval"] = checkpoint_interval
    encoded = _encode_recurrent_spec(recurrent_spec)
    if online_binding is not None and needs_backward:
        raise RuntimeError("Online STDP cannot be combined with autograd")
    operator = (
        online_inference
        if online_binding is not None
        else training
        if needs_backward
        else inference
    )
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
    graphs = getattr(module, "_training_graphs", None)
    capture = (
        online_binding is None
        and needs_backward
        and module.cudagraph is True
        and graphs is not None
        and x_seq.is_cuda
        and options.neuron in _NATIVE_GPU_BACKENDS
        and x_seq.shape[0] == graphs.steps
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
    )
    if capture:
        from ..cudagraph import capture_versions
        from .registry import cell_backend_revision
        from .templates import model_template_revision

        version = (
            capture_versions(module),
            cell_backend_revision(options.neuron),
            model_template_revision(neuron_spec.model),
            model_template_revision(synapse_spec.model),
        )
        if graphs.version != version:
            graphs.stages.clear()
            graphs.version = version
    arguments = (
        x_seq,
        x_syn_seq,
        states,
        parameters,
        list(topology),
        encoded,
        history_indices,
        options.neuron,
        torch.as_tensor(neuron_spec.dt, device=x_seq.device),
    )
    if online_binding is not None:
        arguments = (
            x_seq,
            x_syn_seq,
            states,
            parameters,
            list(topology),
            online_weight,
            list(online_traces),
            list(online_parameters),
            online_signals,
            encoded,
            history_indices,
            options.neuron,
            torch.as_tensor(
                neuron_spec.dt,
                device=x_seq.device,
            ),
        )
    uniform_args = (uniform_seq,) if poisson is not None and poisson.spiking else ()
    with graphs.activate() if capture else nullcontext():
        if online_binding is not None:
            history_sources = [
                *states,
                *(
                    getattr(
                        module.neuron,
                        dict(get_template(neuron_spec).record_attributes)[name],
                    )
                    for name in charged_names
                ),
                *online_traces,
                online_weight,
            ]
            state_output_dtypes = (
                *(neuron_dtype for _ in neuron_names),
                *(psc_dtype for _ in synapse_names),
                *(
                    value.dtype
                    for value in states[len(neuron_names) + len(synapse_names) :]
                ),
            )
            history_dtypes = (
                *state_output_dtypes,
                *(neuron_dtype for _ in charged_names),
                *(value.dtype for value in online_traces),
                online_weight.dtype,
            )
            result = [
                torch.empty_like(x_seq),
                *(
                    torch.empty_like(value, dtype=dtype)
                    for value, dtype in zip(states, state_output_dtypes)
                ),
                *(
                    history_sources[index].new_empty(
                        (x_seq.shape[0], *history_sources[index].shape),
                        dtype=history_dtypes[index],
                    )
                    if index >= 0
                    else torch.empty_like(
                        history_sources[len(states) - index - 1], dtype=neuron_dtype
                    )
                    for index in history_indices
                ),
            ]
            # Mutable operators with Tensor[] returns cannot be automatically
            # functionalized by Torch. Preallocated destinations keep the mutation
            # explicit and let compile preserve its normal functionalization path.
            operator(*arguments[:9], result, *arguments[9:], *uniform_args)
        else:
            result = operator(*arguments, *uniform_args)
    persistent_state = online_binding is not None or options.recurrent == "helion"
    ns = len(neuron_names)
    for name, value in zip(neuron_names, result[1 : 1 + ns]):
        if persistent_state and getattr(module.neuron, name).dtype == value.dtype:
            # Preserve physical memory across compiler CUDA Graph tree iterations;
            # a graph-pool output may be overwritten by the next invocation.
            getattr(module.neuron, name).copy_(value)
        else:
            setattr(module.neuron, name, value)
    for name, value in zip(synapse_names, result[1 + ns : 1 + ns + len(synapse_names)]):
        if persistent_state and getattr(cell, name).dtype == value.dtype:
            getattr(cell, name).copy_(value)
        else:
            setattr(cell, name, value)
    if n_delay:
        delay = module.synapse.delayed_spike
        history = result[len(states)].movedim(0, -len(delay.n_neuron) - 1)
        if persistent_state and delay.history.dtype == history.dtype:
            delay.history.copy_(history)
        else:
            delay.history = history
    histories = dict(zip(history_indices, result[1 + len(states) :]))
    for offset, (name, index) in enumerate(zip(charged_names, charged_indices)):
        attribute = dict(get_template(neuron_spec).record_attributes)[name]
        # Live memory must own one timestep, not a view retaining the complete
        # requested trajectory. Clone preserves its continuation cotangent.
        value = (
            histories[index][-1].clone()
            if index in histories
            else histories[-offset - 1]
        )
        if persistent_state and getattr(module.neuron, attribute).dtype == value.dtype:
            getattr(module.neuron, attribute).copy_(value)
        else:
            setattr(module.neuron, attribute, value)
    records = {}
    for name, index in zip(requested, indices):
        value = histories[index]
        if n_delay and name == "synapse.delayed_spike.history":
            # Operators carry delay first; public records retain the memory's
            # batch-first layout, just like eager MemoryModule recordings.
            value = value.movedim(1, -len(module.synapse.delayed_spike.n_neuron) - 1)
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
        psc = cell.psc.reshape(
            *cell.psc.shape[:-1], *module.neuron.n_neuron, module.synapse.n_receptor
        ).sum(-1)
        if persistent_state and module.synapse.psc.dtype == psc.dtype:
            module.synapse.psc.copy_(psc)
        else:
            module.synapse.psc = psc
    return result[0], records
