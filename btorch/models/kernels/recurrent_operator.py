"""Register whole recurrent trajectories with explicit cell reverse passes."""

import json
from dataclasses import replace

import torch
from torch import Tensor
from torch.nn import functional as F

from .operators import decode_spec
from .registry import _NATIVE_GPU_BACKENDS, cell_backend_revision, resolve_cell_backend
from .templates import (
    ResidualTemplate,
    get_template,
    model_template_revision,
    prepare_parameters,
)


def _connection(source, parameters, topology, spec):
    kind = spec["connection"]
    if kind == "identity":
        return source
    if kind == "linear":
        return F.linear(
            source.to(parameters[0].dtype),
            parameters[0],
            parameters[1] if len(parameters) > 1 else None,
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


def _prepare_linear_operand(value, dtype):
    """Reuse conversion storage when a trajectory operand changes precision."""
    workspace = (
        None if value.dtype == dtype else value.new_empty(value.shape, dtype=dtype)
    )

    def operand(input):
        nonlocal workspace
        if input.dtype == dtype:
            return input
        # A standalone tail VJP can follow storage precision while paired
        # interior VJPs use calculation precision. Allocate at most once.
        if workspace is None:
            workspace = input.new_empty(input.shape, dtype=dtype)
        workspace.copy_(input)
        return workspace

    return operand


def _prepare_connection_forward(source, parameters, topology, spec):
    from ...sparse.runtime.backend import PreparedPropagation

    if spec["connection"] == "linear":
        weight = parameters[0]
        operand = _prepare_linear_operand(source, weight.dtype)
        output = source.new_empty(
            (*source.shape[:-1], weight.shape[0]), dtype=weight.dtype
        )
        flat_output = output.view(-1, weight.shape[0])

        def forward(input):
            input = operand(input).reshape(-1, weight.shape[1])
            if len(parameters) > 1:
                torch.addmm(parameters[1], input, weight.T, out=flat_output)
            else:
                torch.mm(input, weight.T, out=flat_output)
            return output

        return PreparedPropagation(forward)
    if spec["connection"] != "sparse":
        return None
    from ...sparse.runtime.ops import prepare_bound_forward

    crow, col, t_crow, t_col, t_perm = topology[:5]
    forward = prepare_bound_forward(
        spec["route"],
        (
            crow,
            col,
            parameters[0],
            source,
            t_crow,
            t_col,
            t_perm,
            spec.get("max_density"),
            topology[5] if len(topology) > 5 else None,
        ),
    )
    if forward is None or len(parameters) == 1:
        return forward

    def with_bias(callback):
        def biased(input):
            output = callback(input)
            output.add_(parameters[1])
            return output

        return biased

    return replace(
        forward,
        forward=with_bias(forward.forward),
        from_spikes=with_bias(forward.from_spikes)
        if forward.from_spikes is not None
        else None,
    )


def _prepare_connection_backward(
    source,
    parameters,
    topology,
    spec,
    grad_drive,
    needs,
    *,
    activity=None,
    grad_values=None,
):
    if spec["connection"] == "linear":
        weight = parameters[0]
        input_operand = _prepare_linear_operand(source, weight.dtype)
        grad_operand = _prepare_linear_operand(grad_drive, weight.dtype)
        grad_source_storage = source.new_empty(source.shape, dtype=weight.dtype)
        grad_source = (
            grad_source_storage
            if source.dtype == weight.dtype
            else source.new_empty(source.shape)
        )
        grad_weight = (
            (torch.empty_like(weight) if grad_values is None else grad_values)
            if needs[0]
            else None
        )

        def backward(input, gradient, activity=None):
            input = input_operand(input).reshape(-1, weight.shape[1])
            gradient = grad_operand(gradient).reshape(-1, weight.shape[0])
            torch.mm(
                gradient, weight, out=grad_source_storage.view(-1, weight.shape[1])
            )
            if grad_source is not grad_source_storage:
                grad_source.copy_(grad_source_storage)
            if grad_weight is not None:
                if grad_values is None:
                    torch.mm(gradient.T, input, out=grad_weight)
                else:
                    torch.addmm(grad_values, gradient.T, input, out=grad_values)
            return grad_weight if grad_values is None else None, grad_source

        return backward
    if spec["connection"] != "sparse":
        return None
    from ...sparse.runtime.ops import prepare_bound_backward

    crow, col, t_crow, t_col, t_perm = topology[:5]
    return prepare_bound_backward(
        spec["route"],
        (
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
        ),
        input_mask=activity,
        grad_values=grad_values,
    )


def _connection_backward(
    source,
    parameters,
    topology,
    spec,
    grad_drive,
    needs,
    prepared=None,
    *,
    activity=None,
):
    if spec["connection"] == "identity":
        return grad_drive, ()
    if spec["connection"] == "linear":
        if prepared is not None:
            grad_weight, grad_source = prepared(source, grad_drive, activity)
        else:
            weight = parameters[0]
            gradient = grad_drive.to(weight.dtype)
            grad_source = (gradient @ weight).to(source.dtype)
            grad_weight = (
                gradient.flatten(0, -2).T @ source.to(weight.dtype).flatten(0, -2)
                if needs[0]
                else None
            )
    else:
        from ...sparse.runtime.ops import route_propagate_backward

        crow, col, t_crow, t_col, t_perm = topology[:5]
        if prepared is not None:
            grad_weight, grad_source = prepared(source, grad_drive, activity)
        else:
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
            (
                grad_drive.to(parameters[0].dtype)
                if spec["connection"] == "linear"
                else grad_drive
            ).sum_to_size(parameters[1].shape)
            if needs[1]
            else None
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


def _spike_input(spikes, x_syn_seq, time_index):
    """Keep absent synaptic input absent instead of adding a zero sequence."""
    return spikes + x_syn_seq[time_index] if x_syn_seq.numel() else spikes


def _source(spikes, ring, n_delay):
    if not n_delay:
        return spikes, ring
    next_ring = torch.cat((spikes.to(ring.dtype).unsqueeze(0), ring[:-1]), dim=0)
    return next_ring.movedim(0, -1).flatten(-2), next_ring


def _prepare_source(spikes, x_syn_seq, ring, spec, ring_history):
    """Prepare invocation-owned synaptic input and delay destinations."""
    neuron_dims = len(spec["neuron_shape"])
    source_dtype = (
        torch.promote_types(spikes.dtype, x_syn_seq.dtype)
        if x_syn_seq.numel()
        else spikes.dtype
    )
    combined = (
        spikes.new_empty(spikes.shape, dtype=source_dtype)
        if x_syn_seq.numel()
        else None
    )
    delayed = bool(spec["delay"])
    if delayed:
        source_dtype = ring.dtype
    rings = (
        tuple(ring.new_empty(ring.shape, dtype=source_dtype) for _ in range(2))
        if delayed and ring_history is None
        else None
    )
    flat = spikes.flatten(-neuron_dims)
    source = (
        flat.new_empty(
            (*flat.shape[:-1], flat.shape[-1] * spec["delay"]), dtype=source_dtype
        )
        if delayed
        else None
    )

    def forward(spikes, ring, time_index):
        if combined is not None:
            torch.add(spikes, x_syn_seq[time_index], out=combined)
            spikes = combined
        flat = spikes.flatten(-neuron_dims)
        if not delayed:
            return flat, ring
        next_ring = (
            ring_history[time_index]
            if ring_history is not None
            else rings[time_index % 2]
        )
        torch.cat((flat.unsqueeze(0), ring[:-1]), dim=0, out=next_ring)
        source.view(*flat.shape, spec["delay"]).copy_(next_ring.movedim(0, -1))
        return source, next_ring

    return forward


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
    prepared_parameters=None,
    uniform=None,
    spike_list=None,
    gradient_activity=None,
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
            prepared_parameters=prepared_parameters,
            uniform_seq=None if uniform is None else uniform.unsqueeze(0),
            **({"spike_list": spike_list} if spike_list is not None else {}),
            **(
                {"gradient_activity": gradient_activity}
                if gradient_activity is not None
                else {}
            ),
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
    prepared_parameters=None,
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
            prepared_parameters=prepared_parameters,
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
    neuron_template = get_template(neuron_spec)
    synapse_template = get_template(synapse_spec)
    ring = initial_states[-1] if spec["delay"] else x_seq.new_empty(0)
    histories = {}
    output_seq = torch.empty_like(x_seq)
    charged_names = tuple(spec.get("charged_names", ()))
    enable_fp_fusion = bool(spec.get("enable_fp_fusion", True))
    precise_division = bool(spec.get("precise_division", False))
    precise_exp = bool(spec.get("precise_exp", False))
    use_fused = backend in _NATIVE_GPU_BACKENDS and x_seq.device.type == "cuda"
    output_dtype = x_seq.dtype
    if use_fused or spec.get("save_compute_spikes"):
        output_seq = torch.empty(x_seq.shape, device=x_seq.device, dtype=torch.float32)
    activity_index = spec.get("gradient_activity")
    gradient_activity = (
        torch.empty_like(output_seq) if activity_index is not None else None
    )
    if spec.get("gradient_list"):
        from ...sparse.runtime.activity import GradientList

        gradient_activity = torch.empty(
            (x_seq.shape[0], GradientList.size(output_seq[0].numel())),
            dtype=torch.int32,
            device=x_seq.device,
        )
    prepared_parameters = (
        prepare_parameters(neuron_spec, neuron_parameters, backward=False)
        if use_fused
        else None
    )
    synapse_prepared_parameters = (
        prepare_parameters(synapse_spec, synapse_parameters) if use_fused else {}
    )
    checkpoint_interval = int(spec.get("paired_checkpoint_interval", 0))
    if checkpoint_interval and not use_fused:
        raise RuntimeError("Paired recurrent checkpoints require a native CUDA backend")
    n_checkpoints = _checkpoint_count(x_seq.shape[0], checkpoint_interval)
    checkpoints = (
        [
            state.new_empty((n_checkpoints, *state.shape), dtype=torch.float32)
            for state in (*neuron_states, *synapse_states)
        ]
        if checkpoint_interval
        else []
    )
    prepared_step = None
    step_outputs = None
    prepared_connection = None
    spike_list = None
    if (
        use_fused
        and backend == "triton"
        and spec["connection"] == "sparse"
        and not spec["delay"]
        and not x_syn_seq.numel()
    ):
        prepared_connection = _prepare_connection_forward(
            output_seq[0].flatten(-len(spec["neuron_shape"])),
            connection_parameters,
            topology,
            spec,
        )
        if prepared_connection is not None:
            spike_list = prepared_connection.spikes
    if use_fused:
        from .backends.fused import prepare_fused_step

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
            prepared_parameters=prepared_parameters,
            uniform=None if uniform_seq is None else uniform_seq[0],
            spike_list=spike_list,
            gradient_activity=None
            if gradient_activity is None
            else gradient_activity[0],
        )
    if use_fused:
        # Templates determine state/record shapes. The trajectory owns these
        # destinations, so neither an earlier forward nor a saved input is
        # overwritten by this invocation's private working state.
        samples = (
            *neuron_states,
            *synapse_states,
            *([ring] if spec["delay"] else []),
            *neuron_charged,
        )
        histories = {}
        for name in history_names:
            if name < 0:
                continue
            if name == activity_index:
                histories[name] = gradient_activity
                continue
            sample = samples[name]
            dtype = (
                torch.float32
                if len(neuron_states) <= name < len(neuron_states) + len(synapse_states)
                else sample.dtype
            )
            if spec["delay"] and name == len(initial_states) - 1:
                dtype = ring.dtype
            histories[name] = sample.new_empty(
                (x_seq.shape[0], *sample.shape), dtype=dtype
            )
    prepared_source = (
        _prepare_source(
            spikes,
            x_syn_seq,
            ring,
            spec,
            histories.get(len(initial_states) - 1) if spec["delay"] else None,
        )
        if use_fused
        else None
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
                prepared_parameters=prepared_parameters,
                uniform=None if uniform_seq is None else uniform_seq[time_index],
            )
        if prepared_source is not None:
            source, ring = prepared_source(spikes, ring, time_index)
        else:
            source, ring = _source(
                _spike_input(spikes, x_syn_seq, time_index).flatten(
                    -len(spec["neuron_shape"])
                ),
                ring,
                spec["delay"],
            )
        if use_fused and time_index == 0 and prepared_connection is None:
            prepared_connection = _prepare_connection_forward(
                source, connection_parameters, topology, spec
            )
        drive = (
            prepared_connection.from_spikes(source)
            if spike_list is not None
            else prepared_connection(source)
            if prepared_connection is not None
            else _connection(source, connection_parameters, topology, spec)
        )
        drive_shape = synapse_states[0].shape
        if not spec.get("merge_receptors", 0):
            drive = drive.reshape(drive_shape)
        current_spikes, current_neuron_states = spikes, neuron_states
        current_neuron_charged = neuron_charged
        if use_fused and time_index + 1 < x_seq.shape[0]:
            if prepared_step is None:
                prepared_step = prepare_fused_step(
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
                    neuron_prepared_parameters=prepared_parameters,
                    psc_prepared_parameters=synapse_prepared_parameters,
                    spike_list=spike_list,
                    gradient_activity=(
                        None
                        if gradient_activity is None
                        else gradient_activity[time_index + 1]
                    ),
                    uniform=(
                        None if uniform_seq is None else uniform_seq[time_index + 1]
                    ),
                )
                # Two working sets preserve the previous step until its
                # records/checkpoint boundary has been consumed. Recorded
                # states instead bind directly to their sequence destinations.
                step_outputs = (
                    prepared_step.allocate_outputs(),
                    prepared_step.allocate_outputs(),
                )
            outputs = dict(step_outputs[time_index % 2])
            outputs["spike"] = output_seq[time_index + 1]
            bindings = {
                "neuron_input": x_seq[time_index + 1],
                "drive": drive,
            }
            if gradient_activity is not None:
                bindings["gradient_activity"] = gradient_activity[time_index + 1]
            for prefix, template, values, offset, record_time in (
                ("neuron_", neuron_template, neuron_states, 0, time_index + 1),
                (
                    "psc_",
                    synapse_template,
                    synapse_states,
                    len(neuron_states),
                    time_index,
                ),
            ):
                for index, (name, value) in enumerate(
                    zip(template.state_names, values)
                ):
                    bindings[prefix + name + "_init"] = value
                    record = histories.get(offset + index)
                    if record is not None:
                        outputs[prefix + name + "_final"] = record[record_time]
            for index, name in enumerate(charged_names):
                record = histories.get(len(initial_states) + index)
                if record is not None:
                    outputs["neuron_" + name + "_seq"] = record[
                        time_index + 1
                    ].unsqueeze(0)
            if uniform_seq is not None:
                bindings["neuron_uniform"] = uniform_seq[time_index + 1]
            prepared_step.launch_into({**outputs, **bindings})
            spikes = outputs["spike"]
            neuron_states = [
                outputs["neuron_" + name + "_final"]
                for name in neuron_template.state_names
            ]
            synapse_states = [
                outputs["psc_" + name + "_final"]
                for name in synapse_template.state_names
            ]
            neuron_charged = tuple(
                outputs["neuron_" + name + "_seq"][0] for name in charged_names
            )
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
                prepared_parameters=synapse_prepared_parameters,
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
        if not use_fused or time_index == 0:
            output_seq[time_index].copy_(current_spikes)
        for name in history_names:
            if name < 0:
                continue
            if name == activity_index:
                continue
            history_values = (*states, *current_neuron_charged)
            value = history_values[int(name)]
            if name not in histories:
                histories[name] = value.new_empty((x_seq.shape[0], *value.shape))
            # Native interior stages already wrote these records. Only
            # the leading neuron and trailing PSC need copies.
            ns, ss = len(neuron_states), len(synapse_states)
            direct = use_fused and (
                (name < ns and time_index > 0)
                or (ns <= name < ns + ss and time_index + 1 < x_seq.shape[0])
                or (name >= len(initial_states) and time_index > 0)
                or (spec["delay"] and name == len(initial_states) - 1)
            )
            if not direct:
                histories[name][time_index].copy_(value)
    # Custom operators must not alias their input or another output. Final
    # states are already independent tensor allocations for nonempty runs;
    # histories own one preallocated sequence allocation throughout forward.
    final = [*neuron_states, *synapse_states, *([ring] if spec["delay"] else [])]
    if use_fused:
        # Final outputs cannot alias a recorded trajectory under the functional
        # custom-op schema. Copy once at the trajectory boundary, never per step.
        final = [value.clone() for value in final]
    return [
        output_seq.to(output_dtype),
        *final,
        *(
            histories[name] if name >= 0 else current_neuron_charged[-name - 1]
            for name in history_names
        ),
        *checkpoints,
        *([output_seq] if spec.get("save_compute_spikes") else []),
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
    native = backend in _NATIVE_GPU_BACKENDS and x_seq.device.type == "cuda"
    if native:
        state_outputs = [
            state.new_empty(state.shape, dtype=torch.float32) for state in states
        ]
        if spec["delay"]:
            state_outputs[-1] = torch.empty_like(states[-1])
    else:
        state_outputs = [torch.empty_like(state) for state in states]
    history_sources = [
        *state_outputs,
        *(
            state_outputs[state_names.index(template.history_state_name(name))]
            for name in charged_names
        ),
    ]
    if spec.get("gradient_activity") is not None:
        if spec.get("gradient_list"):
            from ...sparse.runtime.activity import GradientList

            history_sources.append(
                torch.empty(
                    (GradientList.size(x_seq[0].numel()),),
                    dtype=torch.int32,
                    device=x_seq.device,
                )
            )
        else:
            history_sources.append(state_outputs[0])
    checkpoint_interval = int(spec.get("paired_checkpoint_interval", 0))
    n_checkpoints = _checkpoint_count(x_seq.shape[0], checkpoint_interval)
    return [
        x_seq.new_empty(x_seq.shape)
        if native or spec.get("save_compute_spikes")
        else torch.empty_like(x_seq),
        *state_outputs,
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
            for state in state_outputs
            if checkpoint_interval
        ),
        *(
            [torch.empty(x_seq.shape, device=x_seq.device, dtype=torch.float32)]
            if spec.get("save_compute_spikes")
            else []
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
    """Replay one template-qualified paired block at a time.

    The current representation excludes stochastic draws, refractory
    state, delay and receptor merge. Template capabilities select
    eligible cells. Checkpoints hold the staggered neuron/PSC boundary
    between paired stages.
    """
    from .backends.fused import prepare_fused_step

    interval = int(spec["paired_checkpoint_interval"])
    ns, ss = spec["neuron_states"], spec["synapse_states"]
    if len(checkpoints) != ns + ss:
        raise RuntimeError("Paired checkpoint state count disagrees with the templates")
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
    neuron_prepared = prepare_parameters(neuron_spec, neuron_parameters)
    psc_prepared = prepare_parameters(psc_spec, psc_parameters)
    neuron_template = get_template(neuron_spec)
    psc_template = get_template(psc_spec)
    source_spikes = spikes_seq
    source_workspace = _prepare_source(source_spikes[0], x_syn_seq, None, spec, None)
    prepared_connection = _prepare_connection_forward(
        source_workspace(source_spikes[0], None, 0)[0],
        connection_parameters,
        topology,
        spec,
    )
    prepared_step = None
    working_outputs = None
    block_states = None
    block_drives = None
    replay_count = 0
    cached_block = -1
    cached_start = 0
    cached_states = ()
    cached_drives = ()

    def replay(block):
        nonlocal cached_block, cached_start, cached_states, cached_drives
        nonlocal \
            prepared_step, \
            working_outputs, \
            block_states, \
            block_drives, \
            replay_count
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
                prepared_parameters=neuron_prepared,
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
        if block_states is None:
            # A reverse stage may read neuron state from one block and PSC
            # state from its predecessor. Two slots preserve both until the
            # consuming launch completes, without retaining the full replay.
            samples = (*neuron_states, *psc_states)
            block_states = tuple(
                tuple(value.new_empty((interval, *value.shape)) for value in samples)
                for _ in range(2)
            )
            block_drives = tuple(
                psc_states[0].new_empty((interval, *psc_states[0].shape))
                for _ in range(2)
            )
        slot = replay_count % 2
        replay_count += 1
        values = block_states[slot]
        drives = block_drives[slot]
        for target, value in zip(values[:ns], neuron_states):
            target[0].copy_(value)
        neuron_states = [value[0] for value in values[:ns]]
        for time_index in range(start, stop):
            local_time = time_index - start
            source = source_workspace(source_spikes[time_index], None, time_index)[0]
            drive = (
                prepared_connection(source)
                if prepared_connection is not None
                else _connection(source, connection_parameters, topology, spec)
            )
            # Reverse consumes these drives after replay. Preserve each once
            # in the bounded block, rather than redoing sparse propagation.
            drives[local_time].copy_(drive.reshape(psc_states[0].shape))
            drive = drives[local_time]
            if time_index + 1 < x_seq.shape[0]:
                if prepared_step is None:
                    prepared_step = prepare_fused_step(
                        x_seq[time_index + 1],
                        drive,
                        tuple(neuron_states),
                        tuple(psc_states),
                        tuple(neuron_parameters),
                        tuple(psc_parameters),
                        neuron_spec,
                        psc_spec,
                        backend=backend,
                        enable_fp_fusion=enable_fp_fusion,
                        precise_division=precise_division,
                        precise_exp=precise_exp,
                        neuron_prepared_parameters=neuron_prepared,
                        psc_prepared_parameters=psc_prepared,
                    )
                    working_outputs = (
                        prepared_step.allocate_outputs(),
                        prepared_step.allocate_outputs(),
                    )
                outputs = dict(working_outputs[slot])
                bindings = {"neuron_input": x_seq[time_index + 1], "drive": drive}
                for prefix, template, states, offset, record_time in (
                    ("neuron_", neuron_template, neuron_states, 0, local_time + 1),
                    ("psc_", psc_template, psc_states, ns, local_time),
                ):
                    for index, (name, value) in enumerate(
                        zip(template.state_names, states)
                    ):
                        bindings[prefix + name + "_init"] = value
                        if record_time < stop - start:
                            outputs[prefix + name + "_final"] = values[offset + index][
                                record_time
                            ]
                prepared_step.launch_into({**outputs, **bindings})
                neuron_states = [
                    outputs["neuron_" + name + "_final"]
                    for name in neuron_template.state_names
                ]
                psc_states = [
                    outputs["psc_" + name + "_final"]
                    for name in psc_template.state_names
                ]
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
                    prepared_parameters=psc_prepared,
                )
                for target, value in zip(values[ns:], psc_states):
                    target[local_time].copy_(value)
        cached_block = block
        cached_start = start
        cached_states = tuple(value[: stop - start] for value in values)
        cached_drives = drives[: stop - start]

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
    from .backends.fused import prepare_fused_backward

    input_dtype = x_seq.dtype
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
    prepared_parameters = prepare_parameters(
        neuron_spec, neuron_parameters, backward=True
    )
    psc_prepared_parameters = prepare_parameters(
        psc_spec, psc_parameters, backward=True
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
    grad_x = torch.empty(x_seq.shape, device=x_seq.device, dtype=torch.float32)
    grad_x_syn = torch.empty(
        x_syn_seq.shape, device=x_syn_seq.device, dtype=x_syn_seq.dtype
    )
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

    source_spikes = spikes_seq
    source_workspace = _prepare_source(
        source_spikes[0],
        x_syn_seq,
        initial_states[-1] if spec["delay"] else x_seq.new_empty(0),
        spec,
        None,
    )

    def source(time_index):
        ring = state(n_initial - 1, time_index - 1) if spec["delay"] else None
        return source_workspace(source_spikes[time_index], ring, time_index)[0]

    def add_parameters(values, start):
        for i, value in enumerate(values):
            if value is not None:
                gradients[start + i] += value.sum_to_size(parameters[start + i].shape)

    grad_neuron_states = list(grad_states[:ns])
    grad_psc_states = list(grad_states[ns : ns + ss])
    grad_ring = grad_states[-1] if spec["delay"] else None
    last_source = source(last)
    activity_seq = records.get(spec.get("gradient_activity"))

    def activity(time_index):
        if activity_seq is not None and spec.get("gradient_list"):
            from ...sparse.runtime.activity import GradientList

            return GradientList(activity_seq[time_index], last_source.numel())
        return (
            None
            if activity_seq is None
            else activity_seq[time_index].reshape(last_source.shape)
        )

    prepared_connection = _prepare_connection_forward(
        last_source, connection_parameters, topology, spec
    )

    def connection(source):
        return (
            prepared_connection(source)
            if prepared_connection is not None
            else _connection(source, connection_parameters, topology, spec)
        )

    drive = connection(last_source)
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
        prepared_parameters=psc_prepared_parameters,
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
    prepared_connection_backward = _prepare_connection_backward(
        last_source,
        connection_parameters,
        topology,
        spec,
        grad_drive.reshape(*last_source.shape[:-1], -1),
        connection_needs,
        activity=activity(last),
        grad_values=gradients[offset] if connection_parameters else None,
    )
    prepared_backward = None
    for time_index in range(last, -1, -1):
        step_source = source(time_index)
        grad_source, connection_gradients = _connection_backward(
            step_source,
            connection_parameters,
            topology,
            spec,
            grad_drive.reshape(*step_source.shape[:-1], -1),
            connection_needs,
            prepared=prepared_connection_backward,
            activity=activity(time_index),
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
            grad_spikes = (
                grad_ring_next[0]
                .to(spikes_seq.dtype)
                .reshape(spikes_seq[time_index].shape)
            )
            grad_ring = torch.cat(
                (grad_ring_next[1:], torch.zeros_like(grad_ring_next[:1])), dim=0
            )
        else:
            grad_spikes = grad_source.reshape(spikes_seq[time_index].shape)
        if x_syn_seq.numel():
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
                else connection(previous_source)
            )
            previous_psc = tuple(state(ns + i, time_index - 2) for i in range(ss))
            grad_psc_records = {
                name: gradient[time_index - 1]
                for i, name in enumerate(psc_state_names)
                if (gradient := grad_records.get(ns + i)) is not None
            }
            if prepared_backward is None:
                prepared_backward = prepare_fused_backward(
                    x_seq[time_index],
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
                    neuron_prepared_parameters=prepared_parameters,
                    psc_prepared_parameters=psc_prepared_parameters,
                )
            (
                gx,
                grad_drive,
                grad_neuron_states,
                grad_psc_states,
                grad_neuron_parameters,
                grad_psc_parameters,
            ) = prepared_backward.backward(
                x_seq[time_index],
                previous_drive,
                previous_neuron,
                previous_psc,
                grad_output[time_index] + grad_spikes,
                tuple(grad_neuron_states),
                tuple(grad_psc_states),
                authoritative_spike=spikes_seq[time_index],
                grad_neuron_records=neuron_records,
                grad_psc_records=grad_psc_records,
                destinations={"grad_neuron_input": grad_x[time_index]},
            )
            add_parameters(grad_psc_parameters, np)
        else:
            current = x_seq[0] + _current(initial_states[ns], spec)
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
                prepared_parameters=prepared_parameters,
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
        if time_index == 0:
            grad_x[time_index].copy_(gx)
        add_parameters(grad_neuron_parameters, 0)
    result = [
        grad_x.to(input_dtype),
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
            _spike_input(spikes, x_syn_seq, time_index).flatten(
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
            grad_spikes = grad_ring_next[0].to(spikes.dtype)
            grad_ring = torch.cat(
                (grad_ring_next[1:], torch.zeros_like(grad_ring_next[:1])), dim=0
            )
        else:
            grad_spikes, grad_ring = grad_source, None
        grad_spikes = grad_spikes.reshape(spikes.shape)
        if x_syn_seq.numel():
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
    gradients = _reverse_impl(
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
    # The custom-op boundary returns cotangents in the original storage dtype.
    # Calculation buffers remain private; no forward inputs are coerced.
    return [
        gradient.to(original.dtype) if gradient is not None else None
        for gradient, original in zip(
            gradients, (x_seq, x_syn_seq, *states, *parameters)
        )
    ]


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
    activity_index = spec.get("gradient_activity")
    nondifferentiable = []
    if activity_index is not None:
        position = history_names.index(activity_index)
        nondifferentiable.append(output[1 + len(states) + position])
    if spec.get("save_compute_spikes"):
        compute_spikes = output[-1]
        recorded_outputs = output[:-1]
        nondifferentiable.append(compute_spikes)
    else:
        compute_spikes = output[0]
        recorded_outputs = output
    if nondifferentiable:
        ctx.mark_non_differentiable(*nondifferentiable)
    ctx.save_for_backward(
        x_seq,
        x_syn_seq,
        *states,
        *parameters,
        *topology,
        *recorded_outputs[1 + len(states) :],
        compute_spikes,
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
