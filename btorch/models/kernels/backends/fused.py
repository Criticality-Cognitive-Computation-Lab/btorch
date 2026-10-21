"""Pair PSC adaptation and the following neuron update in one cell launch.

Connections remain independent. The recurrent caller executes the first
neuron alone, these paired stages between connections, and the last PSC
alone. A nonempty T-step sequence therefore uses T connections and T+1
cell launches.
"""

import functools
import re
from dataclasses import replace

import torch

from ..templates import get_template, model_template_revision
from .engine import (
    _compile,
    _fp32_parameters,
    _kernel_arguments,
    _launch_kernel,
    _layout,
    _mode_name,
    _parameter_index,
    _prepare,
    _prepare_glif_coefficients,
    _Source,
)
from .fragments import get_fragments, scalar_layout


def _prepare_psc_parameters(parameters, template, physical_target, logical_target):
    """Preserve PSC storage while accepting flat and receptor-axis layouts."""
    if len(parameters) != len(template.parameter_names):
        raise ValueError("PSC parameter count disagrees with the model template")
    buffers = {}
    layouts = []
    for name, converted in zip(template.parameter_names, _fp32_parameters(parameters)):
        try:
            layout = _layout(converted, logical_target)
        except ValueError as logical_error:
            try:
                layout = _layout(converted, physical_target)
            except ValueError:
                raise logical_error from None
        buffers[name] = converted
        layouts.append(layout)
    return buffers, tuple(layouts)


def _prefix(statements, prefix):
    """Rename declared scalar registers without interpreting their
    expressions."""
    functions = {"exp", "expm1", "fma", "where", "maximum", "abs"}
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
    neuron_parameter_layouts=None,
    psc_parameter_layouts=None,
    neuron_coefficient_layouts=None,
    *,
    store_outputs=True,
    precise_exp=False,
):
    """Render one direct PSC-to-neuron pipeline stage using shared fragments.

    Forward and reverse share the scalar compute stages. ``store_outputs`` is
    a Python generation-time choice: reverse recomputation declares only its
    inputs and keeps intermediates in registers, without forward output stores
    or pointers in the generated kernel.
    """
    source = _Source(backend, n, 1, modes, precise_exp=precise_exp)
    for prefix in ("neuron_", "psc_"):
        source.pointer(prefix + "dt", 1)
        source.load(prefix + "dt", prefix + "dt", "0")
    neuron = get_template(neuron_spec)
    psc = get_template(psc_spec)
    neuron_layouts = (
        dict(zip(neuron.parameter_names, neuron_parameter_layouts))
        if neuron_parameter_layouts is not None
        else None
    )
    psc_layouts = (
        dict(zip(psc.parameter_names, psc_parameter_layouts))
        if psc_parameter_layouts is not None
        else None
    )
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
            layout = psc_layouts[name] if psc_layouts is not None else None
            source.pointer(
                "psc_" + name,
                layout.span if layout is not None else n * _state_receptors,
            )
            parameter_index = (
                re.sub(
                    r"\bindex\b",
                    f"({index})",
                    _parameter_index(name, layout, 0, backend),
                )
                if layout is not None
                else index
            )
            source.load(prefix + name, "psc_" + name, parameter_index)
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
        layout = neuron_layouts[name] if neuron_layouts is not None else None
        size = (
            layout.span
            if layout is not None
            else (
                n * modes
                if neuron_spec.model == "glif3" and name in ("k", "asc_amps")
                else n
            )
        )
        source.pointer("neuron_" + name, size)
    for name in scalar_parameters:
        base, _ = _mode_name(name, modes)
        layout = neuron_layouts[base] if neuron_layouts is not None else None
        index = (
            _parameter_index(name, layout, modes, backend)
            if layout is not None
            else source.index(name)
        )
        source.load("neuron_" + name, "neuron_" + base, index)
    precomputed_glif_membrane = neuron_coefficient_layouts is not None
    if precomputed_glif_membrane:
        if neuron_spec.model != "glif3":
            raise ValueError("Precomputed membrane coefficients require GLIF3")
        coefficient_names = [("glif_phi_coefficient", "glif_phi")]
        if len(neuron_coefficient_layouts) == 2:
            coefficient_names.append(("glif_delta_coefficient", "glif_delta"))
        elif len(neuron_coefficient_layouts) != 1:
            raise ValueError("Invalid GLIF membrane coefficient count")
        for (pointer_name, scalar_name), layout in zip(
            coefficient_names, neuron_coefficient_layouts
        ):
            source.pointer("neuron_" + pointer_name, layout.span)
            source.load(
                "neuron_" + scalar_name,
                "neuron_" + pointer_name,
                _parameter_index("tau", layout, modes, backend),
            )
    if neuron_spec.surrogate == "poisson" and neuron_spec.poisson.spiking:
        pointer = "neuron_uniform" if store_outputs else "neuron_spike"
        source.pointer(pointer, n)
        source.load(
            "neuron_uniform" if store_outputs else "neuron_output",
            pointer,
            "index",
        )
    source.statements(
        _prefix(
            get_fragments(
                neuron_spec,
                modes,
                precomputed_glif_membrane=precomputed_glif_membrane,
                authoritative_output=(
                    not store_outputs
                    and neuron_spec.surrogate == "poisson"
                    and neuron_spec.poisson.spiking
                ),
            ).forward,
            "neuron_",
        )
    )
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


@functools.lru_cache(maxsize=256)
def _cached_fused_source(
    backend,
    neuron_spec,
    psc_spec,
    n,
    _state_receptors,
    _drive_receptors,
    neuron_histories,
    psc_histories,
    modes,
    neuron_parameter_layouts,
    psc_parameter_layouts,
    neuron_coefficient_layouts,
    store_outputs,
    precise_exp,
    neuron_revision,
    psc_revision,
):
    del neuron_revision, psc_revision
    source = _build_fused_source(
        backend,
        neuron_spec,
        psc_spec,
        n,
        _state_receptors,
        _drive_receptors,
        neuron_histories,
        psc_histories,
        modes,
        neuron_parameter_layouts,
        psc_parameter_layouts,
        neuron_coefficient_layouts,
        store_outputs=store_outputs,
        precise_exp=precise_exp,
    )
    return source.render(), tuple(source.pointers)


def generate_fused_source(
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
    precise_exp=False,
    neuron_parameter_layouts=None,
    psc_parameter_layouts=None,
    neuron_coefficient_layouts=None,
):
    """Return inspectable source and buffer declarations for a paired stage."""
    neuron = get_template(neuron_spec)
    psc = get_template(psc_spec)
    if neuron_parameter_layouts is not None and len(neuron_parameter_layouts) != len(
        neuron.parameter_names
    ):
        raise ValueError("Neuron parameter layout count disagrees with the template")
    if psc_parameter_layouts is not None and len(psc_parameter_layouts) != len(
        psc.parameter_names
    ):
        raise ValueError("PSC parameter layout count disagrees with the template")
    return _cached_fused_source(
        backend,
        replace(neuron_spec, dt=1.0),
        replace(psc_spec, dt=1.0),
        n,
        _state_receptors,
        _drive_receptors,
        tuple(neuron_histories),
        tuple(psc_histories),
        modes,
        None if neuron_parameter_layouts is None else tuple(neuron_parameter_layouts),
        None if psc_parameter_layouts is None else tuple(psc_parameter_layouts),
        None
        if neuron_coefficient_layouts is None
        else tuple(neuron_coefficient_layouts),
        store_outputs,
        precise_exp,
        model_template_revision(neuron_spec.model),
        model_template_revision(psc_spec.model),
    )


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
    enable_fp_fusion=True,
    precise_division=False,
    precise_exp=False,
    glif_membrane_coefficients=None,
    uniform=None,
):
    """Return a spike, both final states and requested one-step histories.

    PSC states use an internal receptor layout while ``drive`` retains its
    receptor layout. A merged state has one internal PSC lane per neuron;
    compatible drive receptors are summed inside the cell kernel.
    """
    if neuron_input.device.type != "cuda":
        raise ValueError("Fused GPU cells require CUDA tensors")
    n = neuron_input.numel()
    _state_receptors = psc_states[0].numel() // n
    _drive_receptors = drive.numel() // n
    (
        neuron,
        _,
        modes,
        neuron_target,
        neuron_buffers,
        neuron_parameter_layouts,
    ) = _prepare(
        neuron_input.unsqueeze(0),
        neuron_states,
        neuron_parameters,
        neuron_spec,
        preserve_parameters=True,
    )
    coefficient_buffers, neuron_coefficient_layouts = _prepare_glif_coefficients(
        glif_membrane_coefficients, neuron_target, backward=False
    )
    if coefficient_buffers and neuron_spec.model != "glif3":
        raise ValueError("Precomputed membrane coefficients require GLIF3")
    (
        psc,
        _,
        _,
        psc_target,
        psc_buffers,
        psc_parameter_layouts,
    ) = _prepare(
        psc_states[0].unsqueeze(0),
        psc_states,
        (),
        psc_spec,
        preserve_parameters=True,
    )
    psc_parameter_buffers, psc_parameter_layouts = _prepare_psc_parameters(
        psc_parameters,
        psc,
        psc_states[0].shape,
        (*neuron_input.shape, _state_receptors)
        if _state_receptors != 1
        else neuron_input.shape,
    )
    psc_buffers.update(psc_parameter_buffers)
    buffers = {
        "neuron_dt": neuron_buffers["dt"],
        "psc_dt": psc_buffers["dt"],
        "neuron_input": neuron_input.float().contiguous(),
        "drive": drive.float().contiguous(),
        "spike": torch.empty_like(neuron_input, dtype=torch.float32),
    }
    if neuron_spec.surrogate == "poisson" and neuron_spec.poisson.spiking:
        if uniform is None:
            raise ValueError("Spiking Poisson fused cells require an explicit uniform")
        buffers["neuron_uniform"] = uniform.float().contiguous()
    buffers.update(
        {"neuron_" + name: value for name, value in coefficient_buffers.items()}
    )
    for prefix, template, prepared, target, history_names in (
        (
            "neuron_",
            neuron,
            neuron_buffers,
            neuron_target,
            neuron_history_names,
        ),
        ("psc_", psc, psc_buffers, psc_target, psc_history_names),
    ):
        for name in template.state_names:
            buffers[prefix + name + "_init"] = prepared[name + "_init"]
            buffers[prefix + name + "_final"] = torch.empty(
                target(name), device=neuron_input.device, dtype=torch.float32
            )
        for name in template.parameter_names:
            buffers[prefix + name] = prepared[name]
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
        neuron_parameter_layouts=neuron_parameter_layouts,
        psc_parameter_layouts=psc_parameter_layouts,
        neuron_coefficient_layouts=neuron_coefficient_layouts,
        precise_exp=precise_exp,
    )
    kernel = _compile(
        backend, source, enable_fp_fusion, precise_division, device=neuron_input.device
    )
    strided_names = {
        *("neuron_" + name for name in neuron.parameter_names),
        *("psc_" + name for name in psc.parameter_names),
        *("neuron_" + name for name in ("glif_phi_coefficient",)),
    }
    arguments = _kernel_arguments(buffers, declarations, strided_names)
    _launch_kernel(backend, kernel, arguments, n, enable_fp_fusion=enable_fp_fusion)
    return (
        buffers["spike"],
        tuple(buffers["neuron_" + name + "_final"] for name in neuron.state_names),
        tuple(buffers["psc_" + name + "_final"] for name in psc.state_names),
        tuple(buffers["neuron_" + name + "_seq"] for name in neuron_history_names),
        tuple(buffers["psc_" + name + "_seq"] for name in psc_history_names),
    )


def _reverse_statements(
    spec, prefix, names, modes=0, *, precomputed_glif_membrane=False
):
    statements = []
    for statement in get_fragments(
        spec,
        modes,
        precomputed_glif_membrane=precomputed_glif_membrane,
    ).backward:
        lhs = statement.split(" = ", 1)[0]
        if lhs.startswith("grad_p_"):
            base, _ = _mode_name(lhs[len("grad_p_") :], modes)
            if base not in names:
                continue
        statements.append(statement)
    return _prefix(statements, prefix)


@functools.lru_cache(maxsize=256)
def _cached_fused_backward_source(
    backend,
    neuron_spec,
    psc_spec,
    n,
    _state_receptors,
    _drive_receptors,
    modes,
    neuron_parameter_names,
    psc_parameter_names,
    grad_spike_present,
    grad_neuron_states_present,
    grad_psc_states_present,
    grad_neuron_records_present,
    grad_psc_records_present,
    neuron_parameter_layouts,
    psc_parameter_layouts,
    neuron_coefficient_layouts,
    precise_exp,
    neuron_revision,
    psc_revision,
):
    del neuron_revision, psc_revision
    neuron = get_template(neuron_spec)
    psc = get_template(psc_spec)
    neuron_parameter_names = set(neuron_parameter_names)
    psc_parameter_names = set(psc_parameter_names)
    neuron_record_presence = dict(
        zip(neuron.history_names, grad_neuron_records_present)
    )
    psc_record_presence = dict(zip(psc.history_names, grad_psc_records_present))
    neuron_state_presence = dict(zip(neuron.state_names, grad_neuron_states_present))
    source = _build_fused_source(
        backend,
        neuron_spec,
        psc_spec,
        n,
        _state_receptors,
        _drive_receptors,
        modes=modes,
        neuron_parameter_layouts=neuron_parameter_layouts,
        psc_parameter_layouts=psc_parameter_layouts,
        neuron_coefficient_layouts=neuron_coefficient_layouts,
        store_outputs=False,
        precise_exp=precise_exp,
    )

    def neuron_size(name):
        return (
            n * modes
            if name == "Iasc"
            or (neuron_spec.model == "glif3" and name in ("k", "asc_amps"))
            else n
        )

    def psc_size(_name):
        return n * _state_receptors

    def cotangent(local, name, index, size, present):
        if present:
            source.pointer(name, size)
            source.load(local, name, index)
        else:
            source.lines.append(local + " = 0.0")

    cotangent("neuron_grad_output", "grad_spike", "index", n, grad_spike_present)
    for prefix, template, state_presence, record_presence, size, parameter_names in (
        (
            "neuron_",
            neuron,
            grad_neuron_states_present,
            neuron_record_presence,
            neuron_size,
            neuron_parameter_names,
        ),
        (
            "psc_",
            psc,
            grad_psc_states_present,
            psc_record_presence,
            psc_size,
            psc_parameter_names,
        ),
    ):
        for name, present in zip(template.state_names, state_presence):
            if present:
                source.pointer(prefix + "grad_" + name, size(name))
            if record_presence.get(name, False):
                source.pointer(prefix + "record_grad_" + name, size(name))
            source.pointer(prefix + "grad_" + name + "_init", size(name))
        for name in template.parameter_names:
            if name in parameter_names:
                source.pointer(prefix + "grad_" + name, size(name))

    scalar_states, scalar_parameters = scalar_layout(neuron_spec, modes)
    for name in scalar_states:
        base, _ = _mode_name(name, modes)
        cotangent(
            "neuron_grad_s_" + name,
            "neuron_grad_" + base,
            source.index(name),
            neuron_size(base),
            neuron_state_presence[base],
        )
        cotangent(
            "neuron_grad_r_" + name,
            "neuron_record_grad_" + base,
            source.index(name),
            neuron_size(base),
            neuron_record_presence.get(base, False),
        )
        source.lines.append(
            f"neuron_grad_s_{name} = neuron_grad_s_{name} + neuron_grad_r_{name}"
        )
    for name in neuron.history_names:
        if not name.endswith("_charged"):
            continue
        cotangent(
            "neuron_grad_r_" + name,
            "neuron_record_grad_" + name,
            "index",
            n,
            neuron_record_presence.get(name, False),
        )

    source.statements(
        _reverse_statements(
            neuron_spec,
            "neuron_",
            neuron_parameter_names,
            modes,
            precomputed_glif_membrane=neuron_coefficient_layouts is not None,
        )
    )
    source.pointer("grad_neuron_input", n)
    source.pointer("grad_drive", n * _drive_receptors)
    source.store("grad_neuron_input", "index", "neuron_grad_x")
    for name in scalar_states:
        base, _ = _mode_name(name, modes)
        source.store(
            "neuron_grad_" + base + "_init",
            source.index(name),
            "neuron_grad_s_" + name,
        )
    for name in scalar_parameters:
        base, _ = _mode_name(name, modes)
        if base in neuron_parameter_names:
            source.store(
                "neuron_grad_" + base,
                source.index(name),
                "neuron_grad_p_" + name,
            )
    for receptor in range(_state_receptors):
        prefix = f"psc_{receptor}_"
        index = f"index * {_state_receptors} + {receptor}"
        source.lines.append(prefix + "grad_output = neuron_grad_x")
        for name, present in zip(psc.state_names, grad_psc_states_present):
            cotangent(
                prefix + "grad_s_" + name,
                "psc_grad_" + name,
                index,
                psc_size(name),
                present,
            )
            cotangent(
                prefix + "grad_r_" + name,
                "psc_record_grad_" + name,
                index,
                psc_size(name),
                psc_record_presence.get(name, False),
            )
            if psc_spec.model != "alpha" or name != "psc":
                source.lines.append(
                    f"{prefix}grad_s_{name} = "
                    f"{prefix}grad_s_{name} + {prefix}grad_r_{name}"
                )
        psc_reverse = list(_reverse_statements(psc_spec, prefix, psc_parameter_names))
        if psc_spec.model == "alpha" and psc_record_presence.get("psc", False):
            # The Module first joins the carried-state and next-neuron paths,
            # then adds the public PSC record. Associating the record first
            # introduces FP32 errors that grow through long recurrent VJPs.
            output_update = (
                f"{prefix}grad_s_psc = {prefix}grad_s_psc + {prefix}grad_output"
            )
            try:
                output_index = psc_reverse.index(output_update)
            except ValueError as error:
                raise RuntimeError(
                    "AlphaPSC reverse must add its output cotangent explicitly"
                ) from error
            psc_reverse.insert(
                output_index + 1,
                f"{prefix}grad_s_psc = {prefix}grad_s_psc + {prefix}grad_r_psc",
            )
        source.statements(psc_reverse)
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
    return source.render(), tuple(source.pointers)


def _generate_fused_backward_source(
    backend,
    neuron_spec,
    psc_spec,
    n,
    _state_receptors,
    _drive_receptors,
    modes,
    neuron_parameter_names,
    psc_parameter_names,
    grad_spike_present,
    grad_neuron_states_present,
    grad_psc_states_present,
    grad_neuron_records_present,
    grad_psc_records_present,
    neuron_parameter_layouts=None,
    psc_parameter_layouts=None,
    neuron_coefficient_layouts=None,
    precise_exp=False,
):
    neuron = get_template(neuron_spec)
    psc = get_template(psc_spec)
    grad_neuron_states_present = tuple(grad_neuron_states_present)
    grad_psc_states_present = tuple(grad_psc_states_present)
    grad_neuron_records_present = tuple(grad_neuron_records_present)
    grad_psc_records_present = tuple(grad_psc_records_present)
    if len(grad_neuron_states_present) != len(neuron.state_names):
        raise ValueError("Neuron state cotangent count disagrees with the template")
    if len(grad_psc_states_present) != len(psc.state_names):
        raise ValueError("PSC state cotangent count disagrees with the template")
    if len(grad_neuron_records_present) != len(neuron.history_names):
        raise ValueError("Neuron record cotangent count disagrees with the template")
    if len(grad_psc_records_present) != len(psc.history_names):
        raise ValueError("PSC record cotangent count disagrees with the template")
    requested_neuron = set(neuron_parameter_names)
    requested_psc = set(psc_parameter_names)
    return _cached_fused_backward_source(
        backend,
        replace(neuron_spec, dt=1.0),
        replace(psc_spec, dt=1.0),
        n,
        _state_receptors,
        _drive_receptors,
        modes,
        tuple(name for name in neuron.parameter_names if name in requested_neuron),
        tuple(name for name in psc.parameter_names if name in requested_psc),
        bool(grad_spike_present),
        grad_neuron_states_present,
        grad_psc_states_present,
        grad_neuron_records_present,
        grad_psc_records_present,
        None if neuron_parameter_layouts is None else tuple(neuron_parameter_layouts),
        None if psc_parameter_layouts is None else tuple(psc_parameter_layouts),
        None
        if neuron_coefficient_layouts is None
        else tuple(neuron_coefficient_layouts),
        precise_exp,
        model_template_revision(neuron_spec.model),
        model_template_revision(psc_spec.model),
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
    authoritative_spike=None,
    grad_neuron_records=None,
    grad_psc_records=None,
    needs_neuron_parameters=None,
    needs_psc_parameters=None,
    backend="triton",
    enable_fp_fusion=True,
    precise_division=False,
    precise_exp=False,
    glif_membrane_coefficients=None,
):
    """Reverse a paired stage in one launch with local charge recomputation.

    Returns:
        External-input, drive, neuron-state, PSC-state, neuron-parameter and
        PSC-parameter cotangents in their original tensor shapes and dtypes.
    """
    n = neuron_input.numel()
    _state_receptors = psc_states[0].numel() // n
    _drive_receptors = drive.numel() // n
    neuron, _, modes, neuron_target, neuron_buffers, neuron_parameter_layouts = (
        _prepare(
            neuron_input.unsqueeze(0),
            neuron_states,
            neuron_parameters,
            neuron_spec,
            preserve_parameters=True,
        )
    )
    coefficient_buffers, neuron_coefficient_layouts = _prepare_glif_coefficients(
        glif_membrane_coefficients, neuron_target, backward=True
    )
    if coefficient_buffers and neuron_spec.model != "glif3":
        raise ValueError("Precomputed membrane coefficients require GLIF3")
    psc, _, _, psc_target, psc_buffers, psc_parameter_layouts = _prepare(
        psc_states[0].unsqueeze(0),
        psc_states,
        (),
        psc_spec,
        preserve_parameters=True,
    )
    psc_parameter_buffers, psc_parameter_layouts = _prepare_psc_parameters(
        psc_parameters,
        psc,
        psc_states[0].shape,
        (*neuron_input.shape, _state_receptors)
        if _state_receptors != 1
        else neuron_input.shape,
    )
    psc_buffers.update(psc_parameter_buffers)
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
    buffers = {
        "neuron_input": neuron_input.float().contiguous(),
        "drive": drive.float().contiguous(),
    }
    if neuron_spec.surrogate == "poisson" and neuron_spec.poisson.spiking:
        if authoritative_spike is None:
            raise ValueError("Poisson fused reverse requires authoritative spikes")
        buffers["neuron_spike"] = authoritative_spike.float().contiguous()
    buffers.update(
        {
            "neuron_" + name: tensor
            for name, tensor in neuron_buffers.items()
            if name != "x_seq"
        }
    )
    buffers.update(
        {"neuron_" + name: value for name, value in coefficient_buffers.items()}
    )
    buffers.update(
        {
            "psc_" + name: tensor
            for name, tensor in psc_buffers.items()
            if name != "x_seq"
        }
    )

    def cotangent(name, value):
        # Missing output losses are compile-time zero expressions, not GPU
        # zero allocations or unused pointer arguments in the generated ABI.
        if value is not None:
            buffers[name] = value.float().contiguous()

    def output_gradient(name, like):
        buffers[name] = torch.empty_like(like, dtype=torch.float32)

    cotangent("grad_spike", grad_spike)
    neuron_record_grads, psc_record_grads = (
        grad_neuron_records or {},
        grad_psc_records or {},
    )
    for prefix, template, states, parameter_layouts, gradients, records in (
        (
            "neuron_",
            neuron,
            neuron_states,
            neuron_parameter_layouts,
            grad_neuron_states,
            neuron_record_grads,
        ),
        (
            "psc_",
            psc,
            psc_states,
            psc_parameter_layouts,
            grad_psc_states,
            psc_record_grads,
        ),
    ):
        parameter_layouts = dict(zip(template.parameter_names, parameter_layouts))
        for name, state, gradient in zip(template.state_names, states, gradients):
            cotangent(prefix + "grad_" + name, gradient)
            cotangent(prefix + "record_grad_" + name, records.get(name))
            output_gradient(prefix + "grad_" + name + "_init", state)
        for name in template.parameter_names:
            names = (
                neuron_parameter_names if prefix == "neuron_" else psc_parameter_names
            )
            if name in names:
                layout = parameter_layouts[name]
                buffers[prefix + "grad_" + name] = torch.empty(
                    layout.target_shape,
                    device=neuron_input.device,
                    dtype=torch.float32,
                )
    for name in neuron.history_names:
        if not name.endswith("_charged"):
            continue
        cotangent("neuron_record_grad_" + name, neuron_record_grads.get(name))
    output_gradient("grad_neuron_input", neuron_input)
    output_gradient("grad_drive", drive)
    source, declarations = _generate_fused_backward_source(
        backend,
        neuron_spec,
        psc_spec,
        n,
        _state_receptors,
        _drive_receptors,
        modes,
        neuron_parameter_names,
        psc_parameter_names,
        grad_spike is not None,
        tuple(value is not None for value in grad_neuron_states),
        tuple(value is not None for value in grad_psc_states),
        tuple(
            neuron_record_grads.get(name) is not None for name in neuron.history_names
        ),
        tuple(psc_record_grads.get(name) is not None for name in psc.history_names),
        neuron_parameter_layouts,
        psc_parameter_layouts,
        neuron_coefficient_layouts,
        precise_exp=precise_exp,
    )
    kernel = _compile(
        backend, source, enable_fp_fusion, precise_division, device=neuron_input.device
    )
    strided_names = {
        *("neuron_" + name for name in neuron.parameter_names),
        *("psc_" + name for name in psc.parameter_names),
        *(
            "neuron_" + name
            for name in ("glif_phi_coefficient", "glif_delta_coefficient")
        ),
    }
    arguments = _kernel_arguments(buffers, declarations, strided_names)
    _launch_kernel(backend, kernel, arguments, n, enable_fp_fusion=enable_fp_fusion)
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
