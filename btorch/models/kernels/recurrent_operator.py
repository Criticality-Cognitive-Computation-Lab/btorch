"""Register whole recurrent trajectories with explicit cell reverse passes."""

import json

import torch
from torch import Tensor
from torch.nn import functional as F

from .operators import decode_spec
from .registry import cell_backend_revision, resolve_cell_backend
from .templates import get_template, model_template_revision


def _connection(source, parameters, topology, spec):
    kind = spec["connection"]
    if kind == "identity":
        return source
    if kind == "linear":
        return F.linear(
            source, parameters[0], parameters[1] if len(parameters) > 1 else None
        )
    from ...sparse.runtime.ops import propagate_bound

    crow, col, t_crow, t_col, t_perm = topology
    output = propagate_bound(
        spec["route"], crow, col, parameters[0], source, t_crow, t_col, t_perm
    )
    return output + parameters[1] if len(parameters) > 1 else output


def _connection_backward(source, parameters, topology, spec, grad_drive, needs):
    if spec["connection"] == "identity":
        return grad_drive, ()
    if spec["connection"] == "linear":
        grad_source = grad_drive @ parameters[0]
        grad_weight = (
            grad_drive.flatten(0, -2).T @ source.flatten(0, -2) if needs[0] else None
        )
    else:
        from ...sparse.runtime.ops import route_propagate_backward

        crow, col, t_crow, t_col, t_perm = topology
        grad_weight, grad_source = route_propagate_backward(
            crow,
            col,
            parameters[0],
            source,
            t_crow,
            t_col,
            t_perm,
            grad_drive,
            needs[0],
            True,
            spec["route"],
        )
        if not needs[0]:
            grad_weight = None
    grad_parameters = [grad_weight]
    if len(parameters) > 1:
        grad_parameters.append(
            grad_drive.sum_to_size(parameters[1].shape) if needs[1] else None
        )
    return grad_source, grad_parameters


def _parts(states, parameters, spec):
    ns, np, sp = (
        spec["neuron_states"],
        spec["neuron_parameters"],
        spec["synapse_parameters"],
    )
    return (
        states[:ns],
        states[ns : ns + spec["synapse_states"]],
        parameters[:np],
        parameters[np : np + sp],
        parameters[np + sp :],
    )


def _source(spikes, ring, n_delay):
    if not n_delay:
        return spikes, ring
    next_ring = torch.cat((spikes.unsqueeze(0), ring[:-1]), dim=0)
    return next_ring.movedim(0, -1).flatten(-2), next_ring


def _current(psc, spec):
    if spec["receptors"]:
        return psc.reshape(
            *psc.shape[:-1], *spec["neuron_shape"], spec["receptors"]
        ).sum(-1)
    neuron_shape = tuple(spec["neuron_shape"])
    return (
        psc
        if psc.shape[-len(neuron_shape) :] == neuron_shape
        else psc.reshape(*psc.shape[:-1], *neuron_shape)
    )


def _cell_forward(x, states, parameters, cell_spec, backend, history_names=()):
    impl = resolve_cell_backend(backend, cell_spec, x.device)
    output, final, histories = impl.forward(
        x.unsqueeze(0), tuple(states), tuple(parameters), cell_spec, history_names
    )
    return output[0], list(final), tuple(value[0] for value in histories)


def _cell_backward(
    x,
    states,
    parameters,
    cell_spec,
    grad_output,
    grad_states,
    next_states,
    backend,
    parameter_needs,
    charged_records=None,
    charged_gradients=None,
):
    impl = resolve_cell_backend(backend, cell_spec, x.device)
    if impl.backward is None:
        raise NotImplementedError(
            f"Cell backend {backend!r} has no training reverse pass"
        )
    names = get_template(cell_spec).state_names
    histories = tuple(value.unsqueeze(0) for value in next_states)
    gradient_histories = tuple(torch.zeros_like(h) for h in histories)
    if charged_records:
        charged_names = tuple(charged_records)
        names = (*names, *charged_names)
        histories = (
            *histories,
            *(charged_records[name].unsqueeze(0) for name in charged_names),
        )
        gradient_histories = (
            *gradient_histories,
            *(charged_gradients[name].unsqueeze(0) for name in charged_names),
        )
    gx, gs, gp = impl.backward(
        x.unsqueeze(0),
        tuple(states),
        tuple(parameters),
        cell_spec,
        names,
        histories,
        grad_output.unsqueeze(0),
        tuple(grad_states),
        gradient_histories,
        needs_input_grad=(*((True,) * (1 + len(states))), *parameter_needs),
    )
    return gx[0], gs, gp


def _trajectory(
    x_seq,
    x_syn_seq,
    initial_states,
    parameters,
    topology,
    encoded,
    history_names,
    backend,
    dt,
):
    spec = json.loads(encoded)
    spec["dt"] = dt
    neuron_spec, synapse_spec = (
        decode_spec(spec["neuron"], spec["dt"]),
        decode_spec(spec["synapse"], spec["dt"]),
    )
    (
        neuron_states,
        synapse_states,
        neuron_parameters,
        synapse_parameters,
        connection_parameters,
    ) = _parts(initial_states, parameters, spec)
    ring = initial_states[-1] if spec["delay"] else x_seq.new_empty(0)
    histories = {}
    output_seq = torch.empty_like(x_seq)
    charged_names = tuple(spec.get("charged_names", ()))
    use_fused = (
        backend in ("triton", "tilelang", "cupy") and x_seq.device.type == "cuda"
    )
    if use_fused:
        from .backends.fused import fused_step_forward

        spikes, neuron_states, neuron_charged = _cell_forward(
            x_seq[0] + _current(synapse_states[0], spec),
            neuron_states,
            neuron_parameters,
            neuron_spec,
            backend,
            charged_names,
        )
    for time_index in range(x_seq.shape[0]):
        if not use_fused:
            spikes, neuron_states, neuron_charged = _cell_forward(
                x_seq[time_index] + _current(synapse_states[0], spec),
                neuron_states,
                neuron_parameters,
                neuron_spec,
                backend,
                charged_names,
            )
        source, ring = _source(
            (spikes.float() + x_syn_seq[time_index].float()).flatten(
                -len(spec["neuron_shape"])
            ),
            ring,
            spec["delay"],
        )
        drive = _connection(source, connection_parameters, topology, spec)
        drive_shape = synapse_states[0].shape
        if not spec.get("merge_receptors", 0):
            drive = drive.reshape(drive_shape)
        current_spikes, current_neuron_states = spikes, neuron_states
        current_neuron_charged = neuron_charged
        if use_fused and time_index + 1 < x_seq.shape[0]:
            spikes, neuron_states, synapse_states, next_charged, _ = fused_step_forward(
                x_seq[time_index + 1],
                drive,
                tuple(neuron_states),
                tuple(synapse_states),
                tuple(neuron_parameters),
                tuple(synapse_parameters),
                neuron_spec,
                synapse_spec,
                backend=backend,
                neuron_history_names=charged_names,
            )
            neuron_states, synapse_states = list(neuron_states), list(synapse_states)
            neuron_charged = tuple(value[0] for value in next_charged)
        else:
            if spec.get("merge_receptors", 0):
                drive = drive.reshape(
                    *drive.shape[:-1], -1, spec["merge_receptors"]
                ).sum(-1)
            drive = drive.reshape(drive_shape)
            _, synapse_states, _ = _cell_forward(
                drive, synapse_states, synapse_parameters, synapse_spec, backend
            )
        states = [
            *current_neuron_states,
            *synapse_states,
            *([ring] if spec["delay"] else []),
        ]
        output_seq[time_index].copy_(current_spikes)
        for name in history_names:
            history_values = (*states, *current_neuron_charged)
            value = history_values[int(name)]
            if name not in histories:
                histories[name] = value.new_empty((x_seq.shape[0], *value.shape))
            histories[name][time_index].copy_(value)
    # Custom operators must not alias their input or another output. Final
    # states are already independent tensor allocations for nonempty runs;
    # histories own one preallocated sequence allocation throughout forward.
    final = [*neuron_states, *synapse_states, *([ring] if spec["delay"] else [])]
    return [
        output_seq,
        *final,
        *(histories[name] for name in history_names),
    ]


@torch.library.custom_op("btorch_cells::recurrent_inference", mutates_args=())
def inference(
    x_seq: Tensor,
    x_syn_seq: Tensor,
    states: list[Tensor],
    parameters: list[Tensor],
    topology: list[Tensor],
    encoded: str,
    history_names: list[int],
    backend: str,
    dt: Tensor,
) -> list[Tensor]:
    return _trajectory(
        x_seq,
        x_syn_seq,
        states,
        parameters,
        topology,
        encoded,
        history_names,
        backend,
        dt,
    )


@torch.library.custom_op("btorch_cells::recurrent_training", mutates_args=())
def training(
    x_seq: Tensor,
    x_syn_seq: Tensor,
    states: list[Tensor],
    parameters: list[Tensor],
    topology: list[Tensor],
    encoded: str,
    history_names: list[int],
    backend: str,
    dt: Tensor,
) -> list[Tensor]:
    return _trajectory(
        x_seq,
        x_syn_seq,
        states,
        parameters,
        topology,
        encoded,
        history_names,
        backend,
        dt,
    )


def _fake(
    x_seq, x_syn_seq, states, parameters, topology, encoded, history_names, backend, dt
):
    spec = json.loads(encoded)
    charged_names = spec.get("charged_names", ())
    template = get_template(decode_spec(spec["neuron"]))
    state_names = template.state_names
    shape_sources = dict(template.history_sources)
    history_sources = [
        *states,
        *(states[state_names.index(shape_sources[name])] for name in charged_names),
    ]
    return [
        torch.empty_like(x_seq),
        *(torch.empty_like(s) for s in states),
        *(
            source.new_empty((x_seq.shape[0], *source.shape))
            for source in (history_sources[int(name)] for name in history_names)
        ),
    ]


inference.register_fake(_fake)
training.register_fake(_fake)


def _reverse_paired(
    x_seq,
    x_syn_seq,
    spikes_seq,
    initial_states,
    parameters,
    topology,
    spec,
    history_names,
    histories,
    grad_output,
    grad_states,
    grad_history,
    needs,
    backend,
):
    """Reverse the tail PSC, paired stages, and leading neuron in order."""
    from .backends.fused import fused_step_backward

    neuron_spec, psc_spec = (
        decode_spec(spec["neuron"], spec["dt"]),
        decode_spec(spec["synapse"], spec["dt"]),
    )
    _, _, neuron_parameters, psc_parameters, connection_parameters = _parts(
        initial_states, parameters, spec
    )
    ns, ss, n_initial = (
        spec["neuron_states"],
        spec["synapse_states"],
        len(initial_states),
    )
    np = len(neuron_parameters)
    offset = np + len(psc_parameters)
    neuron_needs = tuple(needs[2 + n_initial : 2 + n_initial + np])
    psc_needs = tuple(needs[2 + n_initial + np : 2 + n_initial + offset])
    connection_needs = needs[2 + n_initial + offset :]
    gradients = [
        torch.zeros_like(p) if need else None
        for p, need in zip(parameters, needs[2 + n_initial :])
    ]
    grad_x, grad_x_syn = torch.empty_like(x_seq), torch.empty_like(x_syn_seq)
    records = {int(name): value for name, value in zip(history_names, histories)}
    grad_records = {
        int(name): value for name, value in zip(history_names, grad_history)
    }

    def state(index, time_index):
        return initial_states[index] if time_index < 0 else records[index][time_index]

    def source(time_index):
        raw_spikes = spikes_seq[time_index].float() + x_syn_seq[time_index].float()
        raw_spikes = raw_spikes.flatten(-len(spec["neuron_shape"]))
        ring = (
            state(n_initial - 1, time_index - 1)
            if spec["delay"]
            else x_seq.new_empty(0)
        )
        return _source(raw_spikes, ring, spec["delay"])[0]

    def add_parameters(values, start):
        for i, value in enumerate(values):
            if value is not None:
                gradients[start + i] += value.sum_to_size(parameters[start + i].shape)

    last = x_seq.shape[0] - 1
    grad_neuron_states = list(grad_states[:ns])
    grad_psc_states = list(grad_states[ns : ns + ss])
    grad_ring = grad_states[-1] if spec["delay"] else None
    last_source = source(last)
    drive = _connection(last_source, connection_parameters, topology, spec)
    if spec.get("merge_receptors", 0):
        drive = drive.reshape(*drive.shape[:-1], -1, spec["merge_receptors"]).sum(-1)
    previous_psc = tuple(state(ns + i, last - 1) for i in range(ss))
    drive = drive.reshape(previous_psc[0].shape)
    grad_psc_states = [
        value + grad_records[ns + i][last] for i, value in enumerate(grad_psc_states)
    ]
    grad_drive, grad_psc_states, grad_psc_parameters = _cell_backward(
        drive,
        previous_psc,
        psc_parameters,
        psc_spec,
        torch.zeros_like(drive),
        grad_psc_states,
        tuple(state(ns + i, last) for i in range(ss)),
        backend,
        psc_needs,
    )
    add_parameters(grad_psc_parameters, np)
    if spec.get("merge_receptors", 0):
        grad_drive = (
            grad_drive.flatten(-len(spec["neuron_shape"]))
            .unsqueeze(-1)
            .expand(
                *grad_drive.flatten(-len(spec["neuron_shape"])).shape,
                spec["merge_receptors"],
            )
            .flatten(-2)
        )
    for time_index in range(last, -1, -1):
        step_source = source(time_index)
        grad_source, connection_gradients = _connection_backward(
            step_source,
            connection_parameters,
            topology,
            spec,
            grad_drive.reshape(*step_source.shape[:-1], -1),
            connection_needs,
        )
        add_parameters(connection_gradients, offset)
        if spec["delay"]:
            grad_ring_next = (
                grad_source.reshape(*step_source.shape[:-1], -1, spec["delay"]).movedim(
                    -1, 0
                )
                + grad_ring
                + grad_records[n_initial - 1][time_index]
            )
            grad_spikes = grad_ring_next[0].reshape(spikes_seq[time_index].shape)
            grad_ring = torch.cat(
                (grad_ring_next[1:], torch.zeros_like(grad_ring_next[:1])), dim=0
            )
        else:
            grad_spikes = grad_source.reshape(spikes_seq[time_index].shape)
        grad_x_syn[time_index] = grad_spikes
        grad_neuron_states = [
            value + grad_records[i][time_index]
            for i, value in enumerate(grad_neuron_states)
        ]
        neuron_records = {
            name: grad_records[n_initial + i][time_index]
            for i, name in enumerate(spec.get("charged_names", ()))
        }
        previous_neuron = tuple(state(i, time_index - 1) for i in range(ns))
        if time_index:
            previous_source = source(time_index - 1)
            previous_drive = _connection(
                previous_source, connection_parameters, topology, spec
            )
            previous_psc = tuple(state(ns + i, time_index - 2) for i in range(ss))
            grad_psc_states = [
                value + grad_records[ns + i][time_index - 1]
                for i, value in enumerate(grad_psc_states)
            ]
            (
                gx,
                grad_drive,
                grad_neuron_states,
                grad_psc_states,
                grad_neuron_parameters,
                grad_psc_parameters,
            ) = fused_step_backward(
                x_seq[time_index].float(),
                previous_drive,
                previous_neuron,
                previous_psc,
                tuple(neuron_parameters),
                tuple(psc_parameters),
                neuron_spec,
                psc_spec,
                grad_output[time_index] + grad_spikes,
                tuple(grad_neuron_states),
                tuple(grad_psc_states),
                grad_neuron_records=neuron_records,
                needs_neuron_parameters=neuron_needs,
                needs_psc_parameters=psc_needs,
                backend=backend,
            )
            add_parameters(grad_psc_parameters, np)
        else:
            current = x_seq[0].float() + _current(initial_states[ns], spec)
            gx, grad_neuron_states, grad_neuron_parameters = _cell_backward(
                current,
                previous_neuron,
                neuron_parameters,
                neuron_spec,
                grad_output[0] + grad_spikes,
                grad_neuron_states,
                tuple(state(i, 0) for i in range(ns)),
                backend,
                neuron_needs,
                {
                    name: records[n_initial + i][0]
                    for i, name in enumerate(spec.get("charged_names", ()))
                },
                neuron_records,
            )
            grad_current = gx
            if spec["receptors"]:
                flat = gx.flatten(-len(spec["neuron_shape"]))
                grad_current = (
                    flat.unsqueeze(-1)
                    .expand(*flat.shape, spec["receptors"])
                    .flatten(-2)
                )
            grad_psc_states = list(grad_psc_states)
            grad_psc_states[0] = grad_psc_states[0] + grad_current.reshape(
                grad_psc_states[0].shape
            )
        grad_x[time_index] = gx
        add_parameters(grad_neuron_parameters, 0)
    result = [
        grad_x,
        grad_x_syn,
        *grad_neuron_states,
        *grad_psc_states,
        *([grad_ring] if spec["delay"] else []),
        *gradients,
    ]
    return [value if need else None for value, need in zip(result, needs)]


def _reverse_impl(
    x_seq,
    x_syn_seq,
    spikes_seq,
    initial_states,
    parameters,
    topology,
    spec,
    history_names,
    histories,
    grad_output,
    grad_states,
    grad_history,
    needs,
    backend,
):
    if backend in ("triton", "tilelang", "cupy") and x_seq.device.type == "cuda":
        return _reverse_paired(
            x_seq,
            x_syn_seq,
            spikes_seq,
            initial_states,
            parameters,
            topology,
            spec,
            history_names,
            histories,
            grad_output,
            grad_states,
            grad_history,
            needs,
            backend,
        )
    neuron_spec, synapse_spec = (
        decode_spec(spec["neuron"], spec["dt"]),
        decode_spec(spec["synapse"], spec["dt"]),
    )
    _, _, neuron_parameters, synapse_parameters, connection_parameters = _parts(
        initial_states, parameters, spec
    )
    ns, ss = spec["neuron_states"], spec["synapse_states"]
    grad_parameters = [
        torch.zeros_like(p) if need else None
        for p, need in zip(parameters, needs[2 + len(initial_states) :])
    ]
    grad_x, grad_x_syn = torch.empty_like(x_seq), torch.empty_like(x_syn_seq)
    state_histories = {int(name): h for name, h in zip(history_names, histories)}
    history_gradients = {int(name): h for name, h in zip(history_names, grad_history)}
    for time_index in range(x_seq.shape[0] - 1, -1, -1):
        previous = [
            s if time_index == 0 else state_histories[i][time_index - 1]
            for i, s in enumerate(initial_states)
        ]
        following = [state_histories[i][time_index] for i in range(len(initial_states))]
        for i in range(len(grad_states)):
            grad_states[i] = grad_states[i] + history_gradients[i][time_index]
        current = x_seq[time_index] + _current(previous[ns], spec)
        spikes = spikes_seq[time_index]
        source, _ = _source(
            (spikes.float() + x_syn_seq[time_index].float()).flatten(
                -len(spec["neuron_shape"])
            ),
            previous[-1] if spec["delay"] else x_seq.new_empty(0),
            spec["delay"],
        )
        drive = _connection(source, connection_parameters, topology, spec)
        if spec.get("merge_receptors", 0):
            drive = drive.reshape(*drive.shape[:-1], -1, spec["merge_receptors"]).sum(
                -1
            )
        drive = drive.reshape(previous[ns].shape)
        grad_drive, grad_psc_previous, grad_psc_parameters = _cell_backward(
            drive,
            previous[ns : ns + ss],
            synapse_parameters,
            synapse_spec,
            torch.zeros_like(drive),
            grad_states[ns : ns + ss],
            following[ns : ns + ss],
            backend,
            needs[
                2 + len(initial_states) + spec["neuron_parameters"] : 2
                + len(initial_states)
                + spec["neuron_parameters"]
                + spec["synapse_parameters"]
            ],
        )
        # Bind backward to the exact sparse route selected before forward.
        # Its transpose and per-edge gradient kernels reuse the existing
        # topology, avoiding dense weights and time-by-batch-by-edge storage.
        offset = spec["neuron_parameters"] + spec["synapse_parameters"]
        if spec.get("merge_receptors", 0):
            grad_drive = (
                grad_drive.unsqueeze(-1)
                .expand(*grad_drive.shape, spec["merge_receptors"])
                .flatten(-2)
            )
        grad_drive = grad_drive.reshape(*source.shape[:-1], -1)
        connection_needs = needs[2 + len(initial_states) + offset :]
        grad_source, grad_connection = _connection_backward(
            source, connection_parameters, topology, spec, grad_drive, connection_needs
        )
        for i, value in enumerate(grad_connection):
            if value is not None:
                grad_parameters[offset + i] += value
        if spec["delay"]:
            grad_ring_next = (
                grad_source.reshape(*source.shape[:-1], -1, spec["delay"]).movedim(
                    -1, 0
                )
                + grad_states[-1]
            )
            grad_spikes = grad_ring_next[0]
            grad_ring = torch.cat(
                (grad_ring_next[1:], torch.zeros_like(grad_ring_next[:1])), dim=0
            )
        else:
            grad_spikes, grad_ring = grad_source, None
        grad_spikes = grad_spikes.reshape(spikes.shape)
        grad_x_syn[time_index] = grad_spikes
        gx, grad_neuron_previous, grad_neuron_parameters = _cell_backward(
            current,
            previous[:ns],
            neuron_parameters,
            neuron_spec,
            grad_output[time_index] + grad_spikes,
            grad_states[:ns],
            following[:ns],
            backend,
            needs[
                2 + len(initial_states) : 2
                + len(initial_states)
                + spec["neuron_parameters"]
            ],
            {
                name: state_histories[len(initial_states) + offset][time_index]
                for offset, name in enumerate(spec.get("charged_names", ()))
            },
            {
                name: history_gradients[len(initial_states) + offset][time_index]
                for offset, name in enumerate(spec.get("charged_names", ()))
            },
        )
        grad_x[time_index] = gx
        grad_psc_previous = list(grad_psc_previous)
        grad_current = (
            gx if spec["receptors"] else gx.reshape(grad_psc_previous[0].shape)
        )
        if spec["receptors"]:
            grad_current = (
                gx.flatten(-len(spec["neuron_shape"]))
                .unsqueeze(-1)
                .expand(
                    *gx.flatten(-len(spec["neuron_shape"])).shape, spec["receptors"]
                )
                .flatten(-2)
            )
        grad_psc_previous[0] = grad_psc_previous[0] + grad_current
        grad_states = [
            *grad_neuron_previous,
            *grad_psc_previous,
            *([grad_ring] if spec["delay"] else []),
        ]
        for i, value in enumerate((*grad_neuron_parameters, *grad_psc_parameters)):
            if value is not None:
                grad_parameters[i] += value.sum_to_size(parameters[i].shape)
    return [
        value if need else None
        for value, need in zip(
            (grad_x, grad_x_syn, *grad_states, *grad_parameters), needs
        )
    ]


@torch.library.custom_op(
    "btorch_cells::recurrent_reverse",
    mutates_args=(),
    schema=(
        "(Tensor x_seq, Tensor x_syn_seq, Tensor spikes_seq, Tensor[] states, "
        "Tensor[] parameters, Tensor[] topology, str encoded, int[] history_names, "
        "Tensor[] histories, Tensor grad_output, Tensor[] grad_states, "
        "Tensor[] grad_history, bool[] needs, str backend, int backend_revision, "
        "int[] model_revisions, Tensor dt) -> Tensor?[]"
    ),
)
def reverse(
    x_seq: Tensor,
    x_syn_seq: Tensor,
    spikes_seq: Tensor,
    states: list[Tensor],
    parameters: list[Tensor],
    topology: list[Tensor],
    encoded: str,
    history_names: list[int],
    histories: list[Tensor],
    grad_output: Tensor,
    grad_states: list[Tensor],
    grad_history: list[Tensor],
    needs: list[bool],
    backend: str,
    backend_revision: int,
    model_revisions: list[int],
    dt: Tensor,
) -> list[Tensor | None]:
    spec = json.loads(encoded)
    spec["dt"] = dt
    if cell_backend_revision(backend) != backend_revision or any(
        model_template_revision(decode_spec(spec[name]).model) != revision
        for name, revision in zip(("neuron", "synapse"), model_revisions)
    ):
        raise RuntimeError(
            "Cell backend or model registration changed between forward and backward; "
            "complete outstanding backward passes before replacing registrations"
        )
    return _reverse_impl(
        x_seq,
        x_syn_seq,
        spikes_seq,
        states,
        parameters,
        topology,
        spec,
        history_names,
        histories,
        grad_output,
        grad_states,
        grad_history,
        needs,
        backend,
    )


@reverse.register_fake
def _fake_reverse(
    x_seq,
    x_syn_seq,
    spikes_seq,
    states,
    parameters,
    topology,
    encoded,
    history_names,
    histories,
    grad_output,
    grad_states,
    grad_history,
    needs,
    backend,
    backend_revision,
    model_revisions,
    dt,
):
    return [
        torch.empty_like(t) if need else None
        for t, need in zip((x_seq, x_syn_seq, *states, *parameters), needs)
    ]


def _setup(ctx, inputs, output):
    (
        x_seq,
        x_syn_seq,
        states,
        parameters,
        topology,
        encoded,
        history_names,
        backend,
        dt,
    ) = inputs
    ctx.counts = len(states), len(parameters), len(topology)
    ctx.encoded, ctx.history_names, ctx.backend = encoded, history_names, backend
    spec = json.loads(encoded)
    ctx.backend_revision = cell_backend_revision(backend)
    ctx.model_revisions = [
        model_template_revision(decode_spec(spec[name]).model)
        for name in ("neuron", "synapse")
    ]
    ctx.save_for_backward(
        x_seq,
        x_syn_seq,
        *states,
        *parameters,
        *topology,
        *output[1 + len(states) :],
        output[0],
        dt,
    )
    ctx.set_materialize_grads(False)


def _backward(ctx, grad_results):
    if torch.is_grad_enabled():
        raise NotImplementedError(
            "Cell sequence kernels support first-order gradients only; "
            "create_graph=True is unsupported"
        )
    ns, np, nt = ctx.counts
    saved = ctx.saved_tensors
    x_seq, x_syn_seq = saved[:2]
    states = list(saved[2 : 2 + ns])
    parameters = list(saved[2 + ns : 2 + ns + np])
    topology = list(saved[2 + ns + np : 2 + ns + np + nt])
    histories = list(saved[2 + ns + np + nt : -2])
    spikes_seq = saved[-2]
    g_output = (
        grad_results[0] if grad_results[0] is not None else torch.zeros_like(x_seq)
    )
    g_states = [
        g if g is not None else torch.zeros_like(s)
        for g, s in zip(grad_results[1 : 1 + ns], states)
    ]
    g_history = [
        g if g is not None else torch.zeros_like(h)
        for g, h in zip(grad_results[1 + ns :], histories)
    ]
    nx, nxs, nstates, nparameters = ctx.needs_input_grad[:4]
    needs = [nx, nxs, *nstates, *nparameters]
    gradients = reverse(
        x_seq,
        x_syn_seq,
        spikes_seq,
        states,
        parameters,
        topology,
        ctx.encoded,
        ctx.history_names,
        histories,
        g_output,
        g_states,
        g_history,
        needs,
        ctx.backend,
        ctx.backend_revision,
        ctx.model_revisions,
        saved[-1],
    )
    values = [g if need else None for g, need in zip(gradients, needs)]
    return (
        values[0],
        values[1],
        values[2 : 2 + ns],
        values[2 + ns :],
        [None] * nt,
        None,
        None,
        None,
        None,
    )


training.register_autograd(_backward, setup_context=_setup)
