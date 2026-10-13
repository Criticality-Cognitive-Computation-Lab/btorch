"""Pair PSC adaptation and the following neuron update in one cell launch.

Connections remain independent. The recurrent caller executes the first
neuron alone, these paired stages between connections, and the last PSC
alone. A nonempty T-step sequence therefore uses T connections and T+1
cell launches.
"""

import re

import torch

from ..templates import get_template
from .engine import _compile, _dt_buffer, _mode_name, _prepare, _Source
from .fragments import get_fragments, scalar_layout


def _prefix(statements, prefix):
    """Rename declared scalar registers without interpreting their
    expressions."""
    functions = {"exp", "where", "maximum", "abs"}
    return tuple(
        re.sub(
            r"\b[A-Za-z_]\w*\b",
            lambda match: match[0] if match[0] in functions else prefix + match[0],
            statement,
        )
        for statement in statements
    )


def _build_fused_source(
    backend,
    neuron_spec,
    psc_spec,
    n,
    _state_receptors,
    _drive_receptors,
    neuron_histories=(),
    psc_histories=(),
    modes=0,
    *,
    store_outputs=True,
):
    """Render one direct PSC-to-neuron pipeline stage using shared fragments.

    Forward and reverse share the scalar compute stages. ``store_outputs`` is
    a Python generation-time choice: reverse recomputation declares only its
    inputs and keeps intermediates in registers, without forward output stores
    or pointers in the generated kernel.
    """
    source = _Source(backend, n, 1, modes)
    for prefix in ("neuron_", "psc_"):
        source.pointer(prefix + "dt", 1)
        source.load(prefix + "dt", prefix + "dt", "0")
    neuron = get_template(neuron_spec)
    psc = get_template(psc_spec)
    source.pointer("neuron_input", n)
    # ``receptors`` is the public connection topology. These private counts
    # describe the temporary PSC-state and drive layouts used by this stage.
    source.pointer("drive", n * _drive_receptors)
    if store_outputs:
        source.pointer("spike", n)
    source.load("neuron_external", "neuron_input", "index")
    source.lines.append("neuron_x = neuron_external")
    for receptor in range(_state_receptors):
        prefix = f"psc_{receptor}_"
        source.lines.append(prefix + "dt = psc_dt")
        index = f"index * {_state_receptors} + {receptor}"
        for name in psc.state_names:
            source.pointer("psc_" + name + "_init", n * _state_receptors)
            if store_outputs:
                source.pointer("psc_" + name + "_final", n * _state_receptors)
            source.load(prefix + name + "_prev", "psc_" + name + "_init", index)
        for name in psc.parameter_names:
            source.pointer("psc_" + name, n * _state_receptors)
            source.load(prefix + name, "psc_" + name, index)
        if _state_receptors == _drive_receptors:
            source.load(prefix + "x", "drive", index)
        elif _state_receptors == 1:
            source.lines.append(prefix + "x = 0.0")
            for drive_receptor in range(_drive_receptors):
                source.load(
                    prefix + "drive",
                    "drive",
                    f"index * {_drive_receptors} + {drive_receptor}",
                )
                source.lines.append(
                    prefix + "x = " + prefix + "x + " + prefix + "drive"
                )
        else:
            raise ValueError(
                "PSC state layout must retain every drive receptor or merge "
                "them into one compatible state"
            )
        source.statements(_prefix(get_fragments(psc_spec).forward, prefix))
        source.lines.append("neuron_x = neuron_x + " + prefix + "output")
        if store_outputs:
            for name in psc.state_names:
                source.store("psc_" + name + "_final", index, prefix + name + "_next")
        for name in psc_histories:
            source.pointer("psc_" + name + "_seq", n * _state_receptors)
            source.store("psc_" + name + "_seq", index, prefix + name + "_next")
    scalar_states, scalar_parameters = scalar_layout(neuron_spec, modes)
    for name in neuron.state_names:
        size = n * modes if name == "Iasc" else n
        source.pointer("neuron_" + name + "_init", size)
        if store_outputs:
            source.pointer("neuron_" + name + "_final", size)
    for name in scalar_states:
        base, _ = _mode_name(name, modes)
        source.load(
            "neuron_" + name + "_prev", "neuron_" + base + "_init", source.index(name)
        )
    for name in neuron.parameter_names:
        size = (
            n * modes
            if neuron_spec.model == "glif3" and name in ("k", "asc_amps")
            else n
        )
        source.pointer("neuron_" + name, size)
    for name in scalar_parameters:
        base, _ = _mode_name(name, modes)
        source.load("neuron_" + name, "neuron_" + base, source.index(name))
    source.statements(_prefix(get_fragments(neuron_spec, modes).forward, "neuron_"))
    if store_outputs:
        source.store("spike", "index", "neuron_output")
        for name in scalar_states:
            base, _ = _mode_name(name, modes)
            source.store(
                "neuron_" + base + "_final",
                source.index(name),
                "neuron_" + name + "_next",
            )
    for name in neuron_histories:
        source.pointer("neuron_" + name + "_seq", n * modes if name == "Iasc" else n)
        scalars = (
            tuple(f"Iasc_{i}" for i in range(modes)) if name == "Iasc" else (name,)
        )
        for scalar in scalars:
            value = scalar if name.endswith("_charged") else scalar + "_next"
            source.store(
                "neuron_" + name + "_seq", source.index(scalar), "neuron_" + value
            )
    return source


def generate_fused_source(*args, **kwargs):
    """Return inspectable source and buffer declarations for a paired stage."""
    source = _build_fused_source(*args, **kwargs)
    return source.render(), tuple(source.pointers)


def fused_step_forward(
    neuron_input,
    drive,
    neuron_states,
    psc_states,
    neuron_parameters,
    psc_parameters,
    neuron_spec,
    psc_spec,
    neuron_history_names=(),
    psc_history_names=(),
    *,
    backend="triton",
):
    """Return a spike, both final states and requested one-step histories.

    PSC states use an internal receptor layout while ``drive`` retains its
    receptor layout. A merged state has one internal PSC lane per neuron;
    compatible drive receptors are summed inside the cell kernel.
    """
    if neuron_input.device.type != "cuda":
        raise ValueError("Fused GPU cells require CUDA tensors")
    n = neuron_input.numel()
    neuron = get_template(neuron_spec)
    psc = get_template(psc_spec)
    neuron_shape = neuron_input.shape
    psc_shape = psc_states[0].shape
    _state_receptors = psc_states[0].numel() // n
    _drive_receptors = drive.numel() // n
    modes = (
        neuron_states[neuron.state_names.index("Iasc")].shape[-1]
        if neuron_spec.model == "glif3"
        else 0
    )
    buffers = {
        "neuron_dt": _dt_buffer(neuron_spec.dt, neuron_input),
        "psc_dt": _dt_buffer(psc_spec.dt, neuron_input),
        "neuron_input": neuron_input.float().contiguous(),
        "drive": drive.float().contiguous(),
        "spike": torch.empty_like(neuron_input, dtype=torch.float32),
    }
    for prefix, template, states, parameters, shape, history_names in (
        (
            "neuron_",
            neuron,
            neuron_states,
            neuron_parameters,
            neuron_shape,
            neuron_history_names,
        ),
        ("psc_", psc, psc_states, psc_parameters, psc_shape, psc_history_names),
    ):

        def target(name):
            return (
                (*shape, modes)
                if prefix == "neuron_"
                and (
                    name == "Iasc"
                    or (neuron_spec.model == "glif3" and name in ("k", "asc_amps"))
                )
                else shape
            )

        for name, value in zip(template.state_names, states):
            buffers[prefix + name + "_init"] = (
                value.float().expand(target(name)).contiguous()
            )
            buffers[prefix + name + "_final"] = torch.empty(
                target(name), device=neuron_input.device, dtype=torch.float32
            )
        for name, value in zip(template.parameter_names, parameters):
            buffers[prefix + name] = value.float().expand(target(name)).contiguous()
        for name in history_names:
            buffers[prefix + name + "_seq"] = torch.empty(
                (1, *target(name)), device=neuron_input.device, dtype=torch.float32
            )
    source, declarations = generate_fused_source(
        backend,
        neuron_spec,
        psc_spec,
        n,
        _state_receptors,
        _drive_receptors,
        neuron_history_names,
        psc_history_names,
        modes,
    )
    kernel = _compile(backend, source)
    arguments = tuple(buffers[name].reshape(-1) for name, _ in declarations)
    if backend == "triton":
        kernel[((n + 127) // 128,)](*arguments, BLOCK=128)
    elif backend == "cupy":
        import cupy

        with cupy.cuda.Device(neuron_input.device.index):
            with cupy.cuda.ExternalStream(
                torch.cuda.current_stream(neuron_input.device).cuda_stream
            ):
                arrays = tuple(cupy.from_dlpack(value.detach()) for value in arguments)
                kernel(((n + 127) // 128,), (128,), arrays)
    else:
        kernel(*arguments)
    return (
        buffers["spike"],
        tuple(buffers["neuron_" + name + "_final"] for name in neuron.state_names),
        tuple(buffers["psc_" + name + "_final"] for name in psc.state_names),
        tuple(buffers["neuron_" + name + "_seq"] for name in neuron_history_names),
        tuple(buffers["psc_" + name + "_seq"] for name in psc_history_names),
    )


def fused_step_backward(
    neuron_input,
    drive,
    neuron_states,
    psc_states,
    neuron_parameters,
    psc_parameters,
    neuron_spec,
    psc_spec,
    grad_spike,
    grad_neuron_states,
    grad_psc_states,
    *,
    grad_neuron_records=None,
    grad_psc_records=None,
    needs_neuron_parameters=None,
    needs_psc_parameters=None,
    backend="triton",
):
    """Reverse a paired stage in one launch with local charge recomputation.

    Returns:
        External-input, drive, neuron-state, PSC-state, neuron-parameter and
        PSC-parameter cotangents in their original tensor shapes and dtypes.
    """
    neuron, shape, modes, _, neuron_buffers = _prepare(
        neuron_input.unsqueeze(0), neuron_states, neuron_parameters, neuron_spec
    )
    psc, _, _, _, psc_buffers = _prepare(
        psc_states[0].unsqueeze(0), psc_states, psc_parameters, psc_spec
    )
    n = neuron_input.numel()
    neuron_parameter_names = set(
        name
        for name, need in zip(
            neuron.parameter_names,
            needs_neuron_parameters or (True,) * len(neuron.parameter_names),
        )
        if need
    )
    psc_parameter_names = set(
        name
        for name, need in zip(
            psc.parameter_names,
            needs_psc_parameters or (True,) * len(psc.parameter_names),
        )
        if need
    )
    _state_receptors = psc_states[0].numel() // n
    _drive_receptors = drive.numel() // n
    source = _build_fused_source(
        backend,
        neuron_spec,
        psc_spec,
        n,
        _state_receptors,
        _drive_receptors,
        modes=modes,
        store_outputs=False,
    )
    buffers = {
        "neuron_input": neuron_input.float().contiguous(),
        "drive": drive.float().contiguous(),
    }
    buffers.update(
        {
            "neuron_" + name: tensor
            for name, tensor in neuron_buffers.items()
            if name != "x_seq"
        }
    )
    buffers.update(
        {
            "psc_" + name: tensor
            for name, tensor in psc_buffers.items()
            if name != "x_seq"
        }
    )

    def cotangent(name, value, like):
        # Missing output losses are compile-time zero expressions, not GPU
        # zero allocations or unused pointer arguments in the generated ABI.
        if value is not None:
            buffers[name] = value.float().contiguous()
            source.pointer(name, like.numel())

    def load_cotangent(local, name, index):
        if name in buffers:
            source.load(local, name, index)
        else:
            source.lines.append(local + " = 0.0")

    def output_gradient(name, like):
        buffers[name] = torch.empty_like(like, dtype=torch.float32)
        source.pointer(name, like.numel())

    cotangent("grad_spike", grad_spike, neuron_input)
    load_cotangent("neuron_grad_output", "grad_spike", "index")
    neuron_record_grads, psc_record_grads = (
        grad_neuron_records or {},
        grad_psc_records or {},
    )
    for prefix, template, states, parameters, gradients, records in (
        (
            "neuron_",
            neuron,
            neuron_states,
            neuron_parameters,
            grad_neuron_states,
            neuron_record_grads,
        ),
        ("psc_", psc, psc_states, psc_parameters, grad_psc_states, psc_record_grads),
    ):
        for name, state, gradient in zip(template.state_names, states, gradients):
            cotangent(prefix + "grad_" + name, gradient, state)
            cotangent(prefix + "record_grad_" + name, records.get(name), state)
            output_gradient(prefix + "grad_" + name + "_init", state)
        for name in template.parameter_names:
            names = (
                neuron_parameter_names if prefix == "neuron_" else psc_parameter_names
            )
            if name in names:
                output_gradient(prefix + "grad_" + name, buffers[prefix + name])
    scalar_states, scalar_parameters = scalar_layout(neuron_spec, modes)
    for name in scalar_states:
        base, _ = _mode_name(name, modes)
        load_cotangent(
            "neuron_grad_s_" + name, "neuron_grad_" + base, source.index(name)
        )
        load_cotangent(
            "neuron_grad_r_" + name, "neuron_record_grad_" + base, source.index(name)
        )
        source.lines.append(
            f"neuron_grad_s_{name} = neuron_grad_s_{name} + neuron_grad_r_{name}"
        )
    for name in neuron.history_names:
        if not name.endswith("_charged"):
            continue
        cotangent(
            "neuron_record_grad_" + name, neuron_record_grads.get(name), neuron_input
        )
        load_cotangent("neuron_grad_r_" + name, "neuron_record_grad_" + name, "index")

    def reverse_statements(spec, prefix, names, n_modes=0):
        statements = []
        for statement in get_fragments(spec, n_modes).backward:
            lhs = statement.split(" = ", 1)[0]
            if lhs.startswith("grad_p_"):
                base, _ = _mode_name(lhs[len("grad_p_") :], n_modes)
                if base not in names:
                    continue
            statements.append(statement)
        return _prefix(statements, prefix)

    source.statements(
        reverse_statements(neuron_spec, "neuron_", neuron_parameter_names, modes)
    )
    output_gradient("grad_neuron_input", neuron_input)
    output_gradient("grad_drive", drive)
    source.store("grad_neuron_input", "index", "neuron_grad_x")
    for name in scalar_states:
        base, _ = _mode_name(name, modes)
        source.store(
            "neuron_grad_" + base + "_init", source.index(name), "neuron_grad_s_" + name
        )
    for name in scalar_parameters:
        base, _ = _mode_name(name, modes)
        if base in neuron_parameter_names:
            source.store(
                "neuron_grad_" + base, source.index(name), "neuron_grad_p_" + name
            )
    for receptor in range(_state_receptors):
        prefix = f"psc_{receptor}_"
        index = f"index * {_state_receptors} + {receptor}"
        source.lines.append(prefix + "grad_output = neuron_grad_x")
        for name in psc.state_names:
            load_cotangent(prefix + "grad_s_" + name, "psc_grad_" + name, index)
            load_cotangent(prefix + "grad_r_" + name, "psc_record_grad_" + name, index)
            source.lines.append(
                f"{prefix}grad_s_{name} = {prefix}grad_s_{name} + {prefix}grad_r_{name}"
            )
        source.statements(reverse_statements(psc_spec, prefix, psc_parameter_names))
        for name in psc.state_names:
            source.store("psc_grad_" + name + "_init", index, prefix + "grad_s_" + name)
        for name in psc.parameter_names:
            if name in psc_parameter_names:
                source.store("psc_grad_" + name, index, prefix + "grad_p_" + name)
        if _state_receptors == _drive_receptors:
            source.store("grad_drive", index, prefix + "grad_x")
        else:
            for receptor_index in range(_drive_receptors):
                source.store(
                    "grad_drive",
                    f"index * {_drive_receptors} + {receptor_index}",
                    prefix + "grad_x",
                )
    kernel = _compile(backend, source.render())
    arguments = tuple(buffers[name].reshape(-1) for name, _ in source.pointers)
    if backend == "triton":
        kernel[((n + 127) // 128,)](*arguments, BLOCK=128)
    elif backend == "cupy":
        import cupy

        with cupy.cuda.Device(neuron_input.device.index):
            with cupy.cuda.ExternalStream(
                torch.cuda.current_stream(neuron_input.device).cuda_stream
            ):
                arrays = tuple(cupy.from_dlpack(value.detach()) for value in arguments)
                kernel(((n + 127) // 128,), (128,), arrays)
    else:
        kernel(*arguments)
    reduce = lambda name, value: buffers[name].sum_to_size(value.shape).to(value.dtype)
    return (
        buffers["grad_neuron_input"].to(neuron_input.dtype),
        buffers["grad_drive"].to(drive.dtype),
        tuple(
            reduce("neuron_grad_" + name + "_init", value)
            for name, value in zip(neuron.state_names, neuron_states)
        ),
        tuple(
            reduce("psc_grad_" + name + "_init", value)
            for name, value in zip(psc.state_names, psc_states)
        ),
        tuple(
            reduce("neuron_grad_" + name, value)
            if name in neuron_parameter_names
            else None
            for name, value in zip(neuron.parameter_names, neuron_parameters)
        ),
        tuple(
            reduce("psc_grad_" + name, value) if name in psc_parameter_names else None
            for name, value in zip(psc.parameter_names, psc_parameters)
        ),
    )
