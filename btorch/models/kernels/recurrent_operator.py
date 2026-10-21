"""Register whole recurrent trajectories with explicit cell reverse passes."""

import json

import torch
from torch import Tensor
from torch.nn import functional as F

from ..ode import _exp_euler_coefficient
from .operators import decode_spec
from .registry import _NATIVE_GPU_BACKENDS, cell_backend_revision, resolve_cell_backend
from .templates import ResidualTemplate, get_template, model_template_revision


def _connection(source, parameters, topology, spec):
    kind = spec["connection"]
    if kind == "identity":
        return source
    if kind == "linear":
        return F.linear(
            source, parameters[0], parameters[1] if len(parameters) > 1 else None
        )
    from ...sparse.runtime.ops import propagate_bound

    crow, col, t_crow, t_col, t_perm = topology[:5]
    task_values = topology[5] if len(topology) > 5 else None
    output = propagate_bound(
        spec["route"],
        crow,
        col,
        parameters[0],
        source,
        t_crow,
        t_col,
        t_perm,
        spec.get("max_density"),
        task_values,
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

        crow, col, t_crow, t_col, t_perm = topology[:5]
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


def _glif_membrane_coefficients(spec, parameters, *, backward):
    if spec.model != "glif3":
        return None
    names = get_template(spec).parameter_names
    tau = parameters[names.index("tau")]
    phi, delta = _exp_euler_coefficient(-1.0 / tau, spec.dt)
    return (phi, delta) if backward else (phi,)


def _checkpoint_count(steps, interval):
    return (steps - 1) // interval if interval else 0


def _cell_forward(
    x,
    states,
    parameters,
    cell_spec,
    backend,
    history_names=(),
    *,
    enable_fp_fusion=True,
    precise_division=False,
    precise_exp=False,
    glif_membrane_coefficients=None,
    uniform=None,
):
    impl = resolve_cell_backend(backend, cell_spec, x.device)
    arguments = (
        x.unsqueeze(0),
        tuple(states),
        tuple(parameters),
        cell_spec,
        history_names,
    )
    if backend in _NATIVE_GPU_BACKENDS and x.device.type == "cuda":
        output, final, histories = impl.forward(
            *arguments,
            enable_fp_fusion=enable_fp_fusion,
            precise_division=precise_division,
            precise_exp=precise_exp,
            glif_membrane_coefficients=glif_membrane_coefficients,
            uniform_seq=None if uniform is None else uniform.unsqueeze(0),
        )
    else:
        output, final, histories = impl.forward(*arguments)
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
    charged_gradients=None,
    *,
    enable_fp_fusion=True,
    precise_division=False,
    precise_exp=False,
    glif_membrane_coefficients=None,
    authoritative_output=None,
):
    impl = resolve_cell_backend(backend, cell_spec, x.device)
    if impl.backward is None:
        raise NotImplementedError(
            f"Cell backend {backend!r} has no training reverse pass"
        )
    template = get_template(cell_spec)
    names = template.state_names
    histories = tuple(value.unsqueeze(0) for value in next_states)
    gradient_histories = tuple(torch.zeros_like(h) for h in histories)
    if charged_gradients:
        charged_names = tuple(charged_gradients)
        state_values = dict(zip(template.state_names, next_states))
        names = (*names, *charged_names)
        histories = (
            *histories,
            *(
                state_values[template.history_state_name(name)].unsqueeze(0)
                for name in charged_names
            ),
        )
        gradient_histories = (
            *gradient_histories,
            *(charged_gradients[name].unsqueeze(0) for name in charged_names),
        )
    arguments = (
        x.unsqueeze(0),
        tuple(states),
        tuple(parameters),
        cell_spec,
        names,
        histories,
        grad_output.unsqueeze(0),
        tuple(grad_states),
        gradient_histories,
    )
    needs = (*((True,) * (1 + len(states))), *parameter_needs)
    if backend in _NATIVE_GPU_BACKENDS and x.device.type == "cuda":
        gx, gs, gp = impl.backward(
            *arguments,
            needs_input_grad=needs,
            enable_fp_fusion=enable_fp_fusion,
            precise_division=precise_division,
            precise_exp=precise_exp,
            glif_membrane_coefficients=glif_membrane_coefficients,
            residual_template=(
                ResidualTemplate(template.state_names, save_output=True)
                if authoritative_output is not None
                else None
            ),
            residual_output=(
                None
                if authoritative_output is None
                else authoritative_output.unsqueeze(0)
            ),
        )
    else:
        gx, gs, gp = impl.backward(*arguments, needs_input_grad=needs)
    return gx[0], gs, gp


def _charged_gradient_plan(spec, gradients, n_states, last):
    """Return charged cotangents for a reverse timestep."""
    plans = {}
    for offset, name in enumerate(spec.get("charged_names", ())):
        full_index = n_states + offset
        if full_index in gradients:
            plans[name] = (gradients[full_index], None, None)
        else:
            final = gradients[-offset - 1]
            plans[name] = (None, final, torch.zeros_like(final))

    def at(time_index):
        return {
            name: full[time_index]
            if full is not None
            else final
            if time_index == last
            else zero
            for name, (full, final, zero) in plans.items()
        }

    return at


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
    uniform_seq=None,
):
    if uniform_seq is not None and not uniform_seq.numel():
        uniform_seq = None
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
    enable_fp_fusion = bool(spec.get("enable_fp_fusion", True))
    precise_division = bool(spec.get("precise_division", False))
    precise_exp = bool(spec.get("precise_exp", False))
    use_fused = backend in _NATIVE_GPU_BACKENDS and x_seq.device.type == "cuda"
    glif_membrane_coefficients = (
        _glif_membrane_coefficients(neuron_spec, neuron_parameters, backward=False)
        if use_fused
        else None
    )
    checkpoint_interval = int(spec.get("paired_checkpoint_interval", 0))
    if checkpoint_interval and not use_fused:
        raise RuntimeError("Paired recurrent checkpoints require a native CUDA backend")
    n_checkpoints = _checkpoint_count(x_seq.shape[0], checkpoint_interval)
    checkpoints = (
        [
            state.new_empty((n_checkpoints, *state.shape))
            for state in (*neuron_states, *synapse_states)
        ]
        if checkpoint_interval
        else []
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
            enable_fp_fusion=enable_fp_fusion,
            precise_division=precise_division,
            precise_exp=precise_exp,
            glif_membrane_coefficients=glif_membrane_coefficients,
            uniform=None if uniform_seq is None else uniform_seq[0],
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
                enable_fp_fusion=enable_fp_fusion,
                precise_division=precise_division,
                precise_exp=precise_exp,
                glif_membrane_coefficients=glif_membrane_coefficients,
                uniform=None if uniform_seq is None else uniform_seq[time_index],
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
                enable_fp_fusion=enable_fp_fusion,
                precise_division=precise_division,
                precise_exp=precise_exp,
                glif_membrane_coefficients=glif_membrane_coefficients,
                uniform=(None if uniform_seq is None else uniform_seq[time_index + 1]),
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
                drive,
                synapse_states,
                synapse_parameters,
                synapse_spec,
                backend,
                enable_fp_fusion=enable_fp_fusion,
                precise_division=precise_division,
                precise_exp=precise_exp,
            )
        states = [
            *current_neuron_states,
            *synapse_states,
            *([ring] if spec["delay"] else []),
        ]
        if (
            checkpoint_interval
            and (time_index + 1) % checkpoint_interval == 0
            and time_index + 1 < x_seq.shape[0]
        ):
            checkpoint_index = time_index // checkpoint_interval
            # The paired loop carries the next neuron state alongside the
            # current PSC state. This staggered boundary resumes the exact
            # fused schedule without replaying a threshold decision.
            for target, value in zip(checkpoints, (*neuron_states, *synapse_states)):
                target[checkpoint_index].copy_(value)
        output_seq[time_index].copy_(current_spikes)
        for name in history_names:
            if name < 0:
                continue
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
        *(
            histories[name] if name >= 0 else current_neuron_charged[-name - 1]
            for name in history_names
        ),
        *checkpoints,
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
    uniform_seq: Tensor | None = None,
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
        uniform_seq,
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
    uniform_seq: Tensor | None = None,
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
        uniform_seq,
    )


def _fake(
    x_seq,
    x_syn_seq,
    states,
    parameters,
    topology,
    encoded,
    history_names,
    backend,
    dt,
    uniform_seq=None,
):
    spec = json.loads(encoded)
    charged_names = spec.get("charged_names", ())
    template = get_template(decode_spec(spec["neuron"]))
    state_names = template.state_names
    history_sources = [
        *states,
        *(
            states[state_names.index(template.history_state_name(name))]
            for name in charged_names
        ),
    ]
    checkpoint_interval = int(spec.get("paired_checkpoint_interval", 0))
    n_checkpoints = _checkpoint_count(x_seq.shape[0], checkpoint_interval)
    return [
        torch.empty_like(x_seq),
        *(torch.empty_like(s) for s in states),
        *(
            history_sources[int(name)].new_empty(
                (x_seq.shape[0], *history_sources[int(name)].shape)
            )
            if name >= 0
            else torch.empty_like(history_sources[len(states) - name - 1])
            for name in history_names
        ),
        *(
            state.new_empty((n_checkpoints, *state.shape))
            for state in states
            if checkpoint_interval
        ),
    ]


inference.register_fake(_fake)
training.register_fake(_fake)


def _paired_checkpoint_state_reader(
    x_seq,
    x_syn_seq,
    spikes_seq,
    initial_states,
    parameters,
    topology,
    spec,
    checkpoints,
    backend,
    neuron_spec,
    psc_spec,
):
    """Replay one native LIF + ExponentialPSC block at a time.

    This is intentionally the first narrow recurrent residual layout: one
    neuron state, one PSC state, no refractory state, delay or receptor merge.
    Checkpoints hold the paired loop boundary ``(v[s], psc[s - 1])``.
    """
    from .backends.fused import fused_step_forward

    interval = int(spec["paired_checkpoint_interval"])
    ns, ss = spec["neuron_states"], spec["synapse_states"]
    if ns != 1 or ss != 1 or len(checkpoints) != ns + ss:
        raise RuntimeError("Invalid paired LIF + ExponentialPSC checkpoint layout")
    (
        _,
        _,
        neuron_parameters,
        psc_parameters,
        connection_parameters,
    ) = _parts(initial_states, parameters, spec)
    enable_fp_fusion = bool(spec.get("enable_fp_fusion", True))
    precise_division = bool(spec.get("precise_division", False))
    precise_exp = bool(spec.get("precise_exp", False))
    cached_block = -1
    cached_start = 0
    cached_states = ()
    cached_drives = ()

    def replay(block):
        nonlocal cached_block, cached_start, cached_states, cached_drives
        start = block * interval
        stop = min(start + interval, x_seq.shape[0])
        if block == 0:
            _, neuron_states, _ = _cell_forward(
                x_seq[0] + _current(initial_states[ns], spec),
                initial_states[:ns],
                neuron_parameters,
                neuron_spec,
                backend,
                enable_fp_fusion=enable_fp_fusion,
                precise_division=precise_division,
                precise_exp=precise_exp,
            )
            psc_states = list(initial_states[ns : ns + ss])
        else:
            checkpoint_index = block - 1
            neuron_states = [
                checkpoints[index][checkpoint_index] for index in range(ns)
            ]
            psc_states = [
                checkpoints[ns + index][checkpoint_index] for index in range(ss)
            ]
        values = [[] for _ in range(ns + ss)]
        drives = []
        for time_index in range(start, stop):
            source = (
                spikes_seq[time_index].float() + x_syn_seq[time_index].float()
            ).flatten(-len(spec["neuron_shape"]))
            drive = _connection(source, connection_parameters, topology, spec)
            drive = drive.reshape(psc_states[0].shape)
            drives.append(drive)
            current_neuron_states = tuple(neuron_states)
            if time_index + 1 < x_seq.shape[0]:
                (
                    _,
                    next_neuron_states,
                    next_psc_states,
                    _,
                    _,
                ) = fused_step_forward(
                    x_seq[time_index + 1],
                    drive,
                    current_neuron_states,
                    tuple(psc_states),
                    tuple(neuron_parameters),
                    tuple(psc_parameters),
                    neuron_spec,
                    psc_spec,
                    backend=backend,
                    enable_fp_fusion=enable_fp_fusion,
                    precise_division=precise_division,
                    precise_exp=precise_exp,
                )
                neuron_states = list(next_neuron_states)
                psc_states = list(next_psc_states)
            else:
                _, psc_states, _ = _cell_forward(
                    drive,
                    psc_states,
                    psc_parameters,
                    psc_spec,
                    backend,
                    enable_fp_fusion=enable_fp_fusion,
                    precise_division=precise_division,
                    precise_exp=precise_exp,
                )
            for target, value in zip(values, (*current_neuron_states, *psc_states)):
                target.append(value)
        cached_block = block
        cached_start = start
        cached_states = tuple(tuple(state_values) for state_values in values)
        cached_drives = tuple(drives)

    def state(index, time_index):
        if time_index < 0:
            return initial_states[index]
        block = time_index // interval
        if block != cached_block:
            replay(block)
        return cached_states[index][time_index - cached_start]

    def connection_drive(time_index):
        block = time_index // interval
        if block != cached_block:
            replay(block)
        return cached_drives[time_index - cached_start]

    return state, connection_drive


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
    checkpoints,
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
    psc_state_names = get_template(psc_spec).state_names
    enable_fp_fusion = bool(spec.get("enable_fp_fusion", True))
    precise_division = bool(spec.get("precise_division", False))
    precise_exp = bool(spec.get("precise_exp", False))
    _, _, neuron_parameters, psc_parameters, connection_parameters = _parts(
        initial_states, parameters, spec
    )
    glif_membrane_coefficients = _glif_membrane_coefficients(
        neuron_spec, neuron_parameters, backward=True
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
    last = x_seq.shape[0] - 1
    charged_gradients = _charged_gradient_plan(spec, grad_records, n_initial, last)

    if spec.get("paired_checkpoint_interval", 0):
        state, replayed_drive = _paired_checkpoint_state_reader(
            x_seq,
            x_syn_seq,
            spikes_seq,
            initial_states,
            parameters,
            topology,
            spec,
            checkpoints,
            backend,
            neuron_spec,
            psc_spec,
        )
    else:
        replayed_drive = None

        def state(index, time_index):
            return (
                initial_states[index] if time_index < 0 else records[index][time_index]
            )

    zero_record_gradients = [torch.zeros_like(state) for state in initial_states]

    def record_gradient(index, time_index):
        gradient = grad_records.get(index)
        return (
            zero_record_gradients[index] if gradient is None else gradient[time_index]
        )

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

    grad_neuron_states = list(grad_states[:ns])
    grad_psc_states = list(grad_states[ns : ns + ss])
    grad_ring = grad_states[-1] if spec["delay"] else None
    last_source = source(last)
    drive = _connection(last_source, connection_parameters, topology, spec)
    if spec.get("merge_receptors", 0):
        drive = drive.reshape(*drive.shape[:-1], -1, spec["merge_receptors"]).sum(-1)
    previous_psc = tuple(state(ns + i, last - 1) for i in range(ss))
    drive = drive.reshape(previous_psc[0].shape)
    if spec.get("paired_checkpoint_interval", 0):
        # The terminal PSC update is independent of the next neuron update.
        # Replay it directly from its predecessor: requesting state(last)
        # would evict the preceding block at an exact checkpoint boundary,
        # only for the first reverse step to replay that whole block again.
        _, terminal_psc, _ = _cell_forward(
            drive,
            previous_psc,
            psc_parameters,
            psc_spec,
            backend,
            enable_fp_fusion=enable_fp_fusion,
            precise_division=precise_division,
            precise_exp=precise_exp,
        )
    else:
        terminal_psc = tuple(state(ns + i, last) for i in range(ss))
    grad_psc_states = [
        value + record_gradient(ns + i, last) for i, value in enumerate(grad_psc_states)
    ]
    grad_drive, grad_psc_states, grad_psc_parameters = _cell_backward(
        drive,
        previous_psc,
        psc_parameters,
        psc_spec,
        torch.zeros_like(drive),
        grad_psc_states,
        terminal_psc,
        backend,
        psc_needs,
        enable_fp_fusion=enable_fp_fusion,
        precise_division=precise_division,
        precise_exp=precise_exp,
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
                + record_gradient(n_initial - 1, time_index)
            )
            grad_spikes = grad_ring_next[0].reshape(spikes_seq[time_index].shape)
            grad_ring = torch.cat(
                (grad_ring_next[1:], torch.zeros_like(grad_ring_next[:1])), dim=0
            )
        else:
            grad_spikes = grad_source.reshape(spikes_seq[time_index].shape)
        grad_x_syn[time_index] = grad_spikes
        grad_neuron_states = [
            value + record_gradient(i, time_index)
            for i, value in enumerate(grad_neuron_states)
        ]
        neuron_records = charged_gradients(time_index)
        previous_neuron = tuple(state(i, time_index - 1) for i in range(ns))
        if time_index:
            previous_source = source(time_index - 1)
            # Replay already propagated this saved spike source. Retain only
            # the current block's drives, sharing them with reverse instead
            # of launching the same connection operation a second time.
            previous_drive = (
                replayed_drive(time_index - 1)
                if replayed_drive is not None
                else _connection(previous_source, connection_parameters, topology, spec)
            )
            previous_psc = tuple(state(ns + i, time_index - 2) for i in range(ss))
            grad_psc_records = {
                name: gradient[time_index - 1]
                for i, name in enumerate(psc_state_names)
                if (gradient := grad_records.get(ns + i)) is not None
            }
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
                authoritative_spike=spikes_seq[time_index],
                grad_neuron_records=neuron_records,
                grad_psc_records=grad_psc_records,
                needs_neuron_parameters=neuron_needs,
                needs_psc_parameters=psc_needs,
                backend=backend,
                enable_fp_fusion=enable_fp_fusion,
                precise_division=precise_division,
                precise_exp=precise_exp,
                glif_membrane_coefficients=glif_membrane_coefficients,
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
                neuron_records,
                enable_fp_fusion=enable_fp_fusion,
                precise_division=precise_division,
                precise_exp=precise_exp,
                glif_membrane_coefficients=glif_membrane_coefficients,
                authoritative_output=spikes_seq[0],
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
    checkpoints,
    grad_output,
    grad_states,
    grad_history,
    needs,
    backend,
):
    if backend in _NATIVE_GPU_BACKENDS and x_seq.device.type == "cuda":
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
            checkpoints,
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
    enable_fp_fusion = bool(spec.get("enable_fp_fusion", True))
    precise_division = bool(spec.get("precise_division", False))
    precise_exp = bool(spec.get("precise_exp", False))
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
    last = x_seq.shape[0] - 1
    charged_gradients = _charged_gradient_plan(
        spec, history_gradients, len(initial_states), last
    )
    for time_index in range(last, -1, -1):
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
            enable_fp_fusion=enable_fp_fusion,
            precise_division=precise_division,
            precise_exp=precise_exp,
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
            charged_gradients(time_index),
            enable_fp_fusion=enable_fp_fusion,
            precise_division=precise_division,
            precise_exp=precise_exp,
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
        "Tensor[] histories, Tensor[] checkpoints, Tensor grad_output, "
        "Tensor[] grad_states, "
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
    checkpoints: list[Tensor],
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
        checkpoints,
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
    checkpoints,
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
        *_optional,
    ) = inputs
    ctx.counts = len(states), len(parameters), len(topology)
    ctx.encoded, ctx.history_names, ctx.backend = encoded, history_names, backend
    spec = json.loads(encoded)
    neuron_spec = decode_spec(spec["neuron"])
    ctx.has_uniform = neuron_spec.poisson is not None and neuron_spec.poisson.spiking
    ctx.output_counts = (
        len(history_names),
        len(states) if spec.get("paired_checkpoint_interval", 0) else 0,
    )
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
    residuals = list(saved[2 + ns + np + nt : -2])
    n_histories, n_checkpoints = ctx.output_counts
    histories = residuals[:n_histories]
    checkpoints = residuals[n_histories : n_histories + n_checkpoints]
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
        checkpoints,
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
    gradients = (
        values[0],
        values[1],
        values[2 : 2 + ns],
        values[2 + ns :],
        [None] * nt,
        None,
        [] if not ctx.history_names else None,
        None,
        None,
    )
    return (
        *gradients,
        *((None,) if ctx.has_uniform else ()),
    )


training.register_autograd(_backward, setup_context=_setup)
