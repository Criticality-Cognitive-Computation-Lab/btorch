"""Pair PSC adaptation and the following neuron update in one cell launch.

Connections remain independent. The recurrent caller executes the first
neuron alone, these paired stages between connections, and the last PSC
alone. A nonempty T-step sequence therefore uses T connections and T+1
cell launches.
"""

import functools
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from threading import RLock

import torch

from ..contracts import pointwise_dtype
from ..program import namespace_statements, substitute_expression
from ..templates import RecordCotangent, get_template, model_template_revision
from .engine import (
    _compile,
    _device_scope,
    _dt_buffer,
    _layout,
    _load_prepared_parameters,
    _mode_name,
    _parameter_buffers,
    _parameter_index,
    _prepare,
    _prepare_parameters,
    _sequence_layout,
    _Source,
    _tensor_storage,
)
from .fragments import get_fragments, scalar_layout
from .renderers import get_runtime


_FUSED_PLANS = OrderedDict()
_FUSED_PLAN_LOCK = RLock()


@dataclass(frozen=True)
class _FusedPlan:
    """Retain static paired-stage metadata, never invocation tensors."""

    factory: object
    state_names: tuple
    parameter_names: tuple
    state_shapes: dict

    def bind_inputs(self, neuron_input, drive, states, parameters, specs, coefficients):
        if (
            len(states) != 2
            or len(parameters) != 2
            or len(specs) != 2
            or len(coefficients) != 2
        ):
            raise ValueError("Paired preparation requires neuron and PSC groups")
        buffers = {
            "neuron_input": neuron_input,
            "drive": drive,
            "neuron_dt": _dt_buffer(specs[0].dt, neuron_input),
            "psc_dt": _dt_buffer(specs[1].dt, neuron_input),
        }
        for prefix, names, parameter_names, values, params, computed in zip(
            ("neuron_", "psc_"),
            self.state_names,
            self.parameter_names,
            states,
            parameters,
            coefficients,
        ):
            if len(values) != len(names) or len(params) != len(parameter_names):
                raise ValueError(
                    "Paired state or parameter count disagrees with the plan"
                )
            for name, value in zip(names, values):
                target = self.state_shapes[prefix + name + "_init"]
                buffers[prefix + name + "_init"] = (
                    value if value.shape == target else value.expand(target)
                )
            buffers.update(
                (prefix + name, value)
                for name, value in zip(parameter_names, _parameter_buffers(params))
            )
            buffers.update(
                (prefix + name, value.resolve_neg())
                for name, value in (computed or {}).items()
            )
        return buffers

    def bind(
        self,
        neuron_input,
        drive,
        states,
        parameters,
        specs,
        coefficients,
        uniform,
        spike_list,
        gradient_activity,
        extra_buffers,
    ):
        buffers = self.bind_inputs(
            neuron_input, drive, states, parameters, specs, coefficients
        )
        if extra_buffers:
            buffers.update(extra_buffers)
        if uniform is not None:
            buffers["neuron_uniform"] = uniform
        if spike_list is not None:
            buffers.update(spike_list.bindings())
        if gradient_activity is not None:
            buffers["gradient_activity"] = gradient_activity
        return self.factory(buffers=buffers)


def _fused_layout(value):
    return None if value is None else (*_sequence_layout(value), value.is_neg())


def _prepare_psc_parameters(parameters, template, physical_target, logical_target):
    """Preserve PSC storage while accepting flat and receptor-axis layouts."""
    if len(parameters) != len(template.parameter_names):
        raise ValueError("PSC parameter count disagrees with the model template")
    buffers = {}
    layouts = []
    for name, converted in zip(
        template.parameter_names, _parameter_buffers(parameters)
    ):
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
    neuron_prepared_layouts=None,
    psc_prepared_layouts=None,
    *,
    store_outputs=True,
    precise_exp=False,
    storage=(),
    spike_list=(),
    gradient_activity=False,
    clear_drive=False,
):
    """Render one direct PSC-to-neuron pipeline stage using shared fragments.

    Forward and reverse share the scalar compute stages. ``store_outputs`` is
    a Python generation-time choice: reverse recomputation declares only its
    inputs and keeps intermediates in registers, without forward output stores
    or pointers in the generated kernel.
    """
    source = _Source(
        backend,
        n,
        1,
        modes,
        precise_exp=precise_exp,
        storage=storage,
        scalar_dtype=dict(storage)["neuron_input"].dtype
        if "neuron_input" in dict(storage)
        else "float32",
    )
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
    source.statements(("neuron_x = neuron_external",))
    for receptor in range(_state_receptors):
        prefix = f"psc_{receptor}_"
        source.statements((prefix + "dt = psc_dt",))
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
                substitute_expression(
                    _parameter_index(name, layout, 0, source.renderer), {"index": index}
                )
                if layout is not None
                else index
            )
            source.load(prefix + name, "psc_" + name, parameter_index)
        if _state_receptors == _drive_receptors:
            source.load(prefix + "x", "drive", index)
        elif _state_receptors == 1:
            source.statements((prefix + "x = 0.0",))
            for drive_receptor in range(_drive_receptors):
                source.load(
                    prefix + "drive",
                    "drive",
                    f"index * {_drive_receptors} + {drive_receptor}",
                )
                source.statements(
                    (prefix + "x = " + prefix + "x + " + prefix + "drive",)
                )
        else:
            raise ValueError(
                "PSC state layout must retain every drive receptor or merge "
                "them into one compatible state"
            )
        psc_prepared_scalars = _load_prepared_parameters(
            source,
            psc,
            psc_prepared_layouts,
            pointer_prefix="psc_",
            scalar_prefix=prefix,
            index=index,
        )
        source.statements(
            namespace_statements(
                get_fragments(psc_spec, prepared_values=psc_prepared_scalars).forward,
                prefix,
            )
        )
        source.statements(("neuron_x = neuron_x + " + prefix + "output",))
        if store_outputs:
            for name in psc.state_names:
                source.store("psc_" + name + "_final", index, prefix + name + "_next")
        for name in psc_histories:
            source.pointer("psc_" + name + "_seq", n * _state_receptors)
            source.store("psc_" + name + "_seq", index, prefix + name + "_next")
    scalar_states, scalar_parameters = scalar_layout(neuron_spec, modes)
    for name in neuron.state_names:
        size = n * modes if neuron.extra_dims(name) else n
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
            else (n * modes if neuron.extra_dims(name) else n)
        )
        source.pointer("neuron_" + name, size)
    for name in scalar_parameters:
        base, _ = _mode_name(name, modes)
        layout = neuron_layouts[base] if neuron_layouts is not None else None
        index = (
            _parameter_index(name, layout, modes, source.renderer)
            if layout is not None
            else source.index(name)
        )
        source.load("neuron_" + name, "neuron_" + base, index)
    prepared_scalars = _load_prepared_parameters(
        source,
        neuron,
        neuron_prepared_layouts,
        pointer_prefix="neuron_",
        scalar_prefix="neuron_",
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
        namespace_statements(
            get_fragments(
                neuron_spec,
                modes,
                prepared_values=prepared_scalars,
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
        source.emit_spike_list("neuron_output", spike_list)
        if clear_drive:
            # This owner has consumed its receptor inputs. Publish zero for
            # the next propagation kernel without a separate clearing launch.
            for receptor in range(_drive_receptors):
                source.store("drive", f"index * {_drive_receptors} + {receptor}", "0.0")
        if gradient_activity:
            source.emit_gradient_activity(neuron_spec, prefix="neuron_")
        for name in scalar_states:
            base, _ = _mode_name(name, modes)
            source.store(
                "neuron_" + base + "_final",
                source.index(name),
                "neuron_" + name + "_next",
            )
    for name in neuron_histories:
        source.pointer(
            "neuron_" + name + "_seq", n * modes if neuron.extra_dims(name) else n
        )
        scalars = (
            tuple(f"{name}_{i}" for i in range(modes))
            if neuron.extra_dims(name)
            else (name,)
        )
        for scalar in scalars:
            value = scalar + "_next" if name in neuron.state_names else scalar
            source.store(
                "neuron_" + name + "_seq", source.index(scalar), "neuron_" + value
            )
    return source


def _build_helion_source(
    backend,
    neuron_spec,
    psc_spec,
    n,
    state_receptors,
    drive_receptors,
    propagation=(),
    neuron_histories=(),
    psc_histories=(),
    modes=0,
    neuron_parameter_layouts=None,
    psc_parameter_layouts=None,
    neuron_prepared_layouts=None,
    psc_prepared_layouts=None,
    *,
    precise_exp=False,
    storage=(),
    ring_capacity=None,
    history_capacity=None,
    extra_pointers=(),
    grid_sync_before_propagation=False,
    grid_sync_before_psc=False,
    pre_propagation=(),
    post_propagation=(),
    local_dtypes=(),
    spike_list=(),
):
    """Build the shared model phases used by a HELION recurrent step.

    The connection phase is the only injected code. Neuron and PSC equations,
    parameter broadcasting and storage addressing all come from the same
    source/fragments used by :func:`_build_fused_source`; this function only
    changes their order to ``neuron -> propagation -> PSC``. The caller wraps
    the returned source in its backend-specific cooperative launch and supplies
    the FutureCurrent/CSR statements.
    """
    if history_capacity is None:
        history_capacity = ring_capacity
    source = _Source(
        backend,
        n,
        1,
        modes,
        precise_exp=precise_exp,
        storage=storage,
        scalar_dtype=dict(storage)["neuron_input"].dtype
        if "neuron_input" in dict(storage)
        else "float32",
    )
    source.pointer("neuron_dt", 1)
    source.pointer("psc_dt", 1)
    source.load("neuron_dt", "neuron_dt", "0")
    source.load("psc_dt", "psc_dt", "0")
    source.pointer("neuron_input", n)
    drive_size = (
        int(ring_capacity) * n * drive_receptors
        if ring_capacity is not None
        else n * drive_receptors
    )
    source.pointer("drive", drive_size)
    for name, size in extra_pointers:
        source.pointer(name, size)
    if ring_capacity is not None or history_capacity is not None:
        source.pointer("helion_step", 1)
        source.load("helion_step", "helion_step", "0", dtype="int64")
    source.pointer("spike", n)
    source.load("neuron_external", "neuron_input", "index")
    source.statements(("neuron_x = neuron_external",))

    neuron = get_template(neuron_spec)
    psc = get_template(psc_spec)
    if psc.output_state is None:
        raise NotImplementedError(
            "HELION requires the PSC template to declare its output_state"
        )
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
    if neuron_layouts is not None and len(neuron_layouts) != len(
        neuron.parameter_names
    ):
        raise ValueError("Neuron parameter layout count disagrees with the template")
    if psc_layouts is not None and len(psc_layouts) != len(psc.parameter_names):
        raise ValueError("PSC parameter layout count disagrees with the template")

    # The template identifies the physical state supplying the old current;
    # do not infer it from a model name or a particular state spelling. The
    # PSC transition runs once after propagation, with the new drive.
    for receptor in range(state_receptors):
        prefix = f"psc_{receptor}_"
        index = f"index * {state_receptors} + {receptor}"
        source.statements((prefix + "dt = psc_dt",))
        for name in psc.state_names:
            source.pointer("psc_" + name + "_init", n * state_receptors)
            source.pointer("psc_" + name + "_final", n * state_receptors)
            source.load(prefix + name + "_prev", "psc_" + name + "_init", index)
        for name in psc.parameter_names:
            layout = psc_layouts[name] if psc_layouts is not None else None
            source.pointer(
                "psc_" + name,
                layout.span if layout is not None else n * state_receptors,
            )
            parameter_index = (
                substitute_expression(
                    _parameter_index(name, layout, 0, source.renderer),
                    {"index": index},
                )
                if layout is not None
                else index
            )
            source.load(prefix + name, "psc_" + name, parameter_index)
        source.statements(
            ("neuron_x = neuron_x + " + prefix + psc.output_state + "_prev",)
        )

    scalar_states, scalar_parameters = scalar_layout(neuron_spec, modes)
    for name in neuron.state_names:
        size = n * modes if neuron.extra_dims(name) else n
        source.pointer("neuron_" + name + "_init", size)
        source.pointer("neuron_" + name + "_final", size)
    for name in scalar_states:
        base, _ = _mode_name(name, modes)
        source.load(
            "neuron_" + name + "_prev",
            "neuron_" + base + "_init",
            source.index(name),
        )
    for name in neuron.parameter_names:
        layout = neuron_layouts[name] if neuron_layouts is not None else None
        size = (
            layout.span
            if layout is not None
            else (n * modes if neuron.extra_dims(name) else n)
        )
        source.pointer("neuron_" + name, size)
    for name in scalar_parameters:
        base, _ = _mode_name(name, modes)
        layout = neuron_layouts[base] if neuron_layouts is not None else None
        parameter_index = (
            _parameter_index(name, layout, modes, source.renderer)
            if layout is not None
            else source.index(name)
        )
        source.load("neuron_" + name, "neuron_" + base, parameter_index)
    neuron_prepared = _load_prepared_parameters(
        source,
        neuron,
        neuron_prepared_layouts,
        pointer_prefix="neuron_",
        scalar_prefix="neuron_",
    )
    source.statements(
        namespace_statements(
            get_fragments(
                neuron_spec,
                modes,
                prepared_values=neuron_prepared,
            ).forward,
            "neuron_",
        )
    )
    source.store("spike", "index", "neuron_output")
    if history_capacity is not None and any(
        name == "delay_history_record" for name, _ in extra_pointers
    ):
        # Record the canonical newest-first delay history while the neuron
        # output is resident.  Older entries come from prior raster rows or
        # the invocation's initial Delay.history snapshot.
        source.store(
            "delay_history_record",
            f"(helion_step * {history_capacity} * {n} + {0} * {n}) + index",
            "neuron_output",
        )
        for delay_index in range(1, int(history_capacity)):
            if backend == "triton":
                source.lines.extend(
                    (
                        f"history_past_{delay_index} = helion_step >= {delay_index}",
                        f"history_old_{delay_index} = tl.load(delay_history_ptr + "
                        f"({delay_index} - helion_step - 1) * {n} + index, "
                        f"mask=mask & ~history_past_{delay_index}, other=0.0)",
                        f"history_prev_{delay_index} = "
                        f"tl.load(delay_history_record_ptr + "
                        f"(helion_step - {delay_index}) * {history_capacity} * {n} "
                        f"+ index, mask=mask & history_past_{delay_index}, "
                        f"other=0.0)",
                        f"history_value_{delay_index} = "
                        f"tl.where(history_past_{delay_index}, "
                        f"history_prev_{delay_index}, history_old_{delay_index})",
                        "tl.store(delay_history_record_ptr + (helion_step * "
                        f"{history_capacity} * {n} + {delay_index} * {n}) + index, "
                        f"history_value_{delay_index}, mask=mask)",
                    )
                )
            else:
                source.lines.extend(
                    (
                        f"if helion_step >= {delay_index}:",
                        f"    history_value = delay_history_record_ptr["
                        f"(helion_step - {delay_index}) * {history_capacity} * {n} "
                        f"+ index]",
                        "else:",
                        f"    history_value = delay_history_ptr["
                        f"({delay_index} - helion_step - 1) * {n} + index]",
                        "delay_history_record_ptr[(helion_step * "
                        f"{history_capacity} * {n} + {delay_index} * {n}) "
                        f"+ index] = history_value",
                    )
                )
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
                "neuron_" + name + "_seq",
                source.index(scalar),
                "neuron_" + value,
            )

    # The injected CSR phase carries integer topology and ring indices. Mark
    # its scalar locals explicitly for TileLang, whose local allocation does
    # not infer types from raw injected statements.
    for name in (
        "row_start",
        "row_end",
        "edge",
        "dst",
        "slot",
        "target",
        "edge_delay",
        "history_age",
    ):
        source.local_dtypes[name] = "int64"
    source.local_dtypes.update(dict(local_dtypes))
    if spike_list:
        source.local_dtypes.update(
            list_count="int32", list_valid="bool", listed_source="int64"
        )
    source.emit_spike_list("neuron_output", spike_list)
    source.indexed_statements(pre_propagation)
    if grid_sync_before_propagation:
        source.grid_sync()
    source.indexed_statements(propagation)
    if grid_sync_before_psc:
        source.grid_sync()
    source.indexed_statements(post_propagation)
    for receptor in range(state_receptors):
        prefix = f"psc_{receptor}_"
        index = f"index * {state_receptors} + {receptor}"
        if state_receptors == drive_receptors:
            drive_index = index
            if ring_capacity is not None:
                drive_index = (
                    f"(helion_step % {ring_capacity}) * "
                    f"{n * drive_receptors} + ({index})"
                )
            source.load(prefix + "x", "drive", drive_index)
        elif state_receptors == 1:
            source.lines.append(prefix + "x = 0.0")
            for drive_receptor in range(drive_receptors):
                drive_index = f"index * {drive_receptors} + {drive_receptor}"
                if ring_capacity is not None:
                    drive_index = (
                        f"(helion_step % {ring_capacity}) * "
                        f"{n * drive_receptors} + {drive_index}"
                    )
                source.load(
                    prefix + "drive",
                    "drive",
                    drive_index,
                )
                source.lines.append(
                    prefix + "x = " + prefix + "x + " + prefix + "drive"
                )
        else:
            raise ValueError(
                "PSC state layout must retain every drive receptor or merge "
                "them into one compatible state"
            )
        psc_prepared = _load_prepared_parameters(
            source,
            psc,
            psc_prepared_layouts,
            pointer_prefix="psc_",
            scalar_prefix=prefix,
        )
        source.statements(
            namespace_statements(
                get_fragments(
                    psc_spec,
                    prepared_values=psc_prepared,
                ).forward,
                prefix,
            )
        )
        for name in psc.state_names:
            source.store("psc_" + name + "_final", index, prefix + name + "_next")
        # Clear the consumed drive slot after the PSC update; other ring
        # slots retain their pending delayed arrivals.
        source.lines.append(prefix + "drive_clear = 0.0")
        if state_receptors == drive_receptors:
            source.store("drive", drive_index, prefix + "drive_clear")
        elif state_receptors == 1:
            for drive_receptor in range(drive_receptors):
                clear_index = f"index * {drive_receptors} + {drive_receptor}"
                if ring_capacity is not None:
                    clear_index = (
                        f"(helion_step % {ring_capacity}) * "
                        f"{n * drive_receptors} + ({clear_index})"
                    )
                source.store("drive", clear_index, prefix + "drive_clear")
    return source


@functools.lru_cache(maxsize=256)
def _cached_helion_source(
    backend,
    neuron_spec,
    psc_spec,
    n,
    state_receptors,
    drive_receptors,
    propagation,
    neuron_histories,
    psc_histories,
    modes,
    neuron_parameter_layouts,
    psc_parameter_layouts,
    neuron_prepared_layouts,
    psc_prepared_layouts,
    precise_exp,
    ring_capacity,
    history_capacity,
    extra_pointers,
    grid_sync_before_propagation,
    grid_sync_before_psc,
    pre_propagation,
    post_propagation,
    local_dtypes,
    neuron_revision,
    psc_revision,
    storage=(),
    spike_list=(),
):
    del neuron_revision, psc_revision
    source = _build_helion_source(
        backend,
        neuron_spec,
        psc_spec,
        n,
        state_receptors,
        drive_receptors,
        propagation,
        neuron_histories,
        psc_histories,
        modes,
        neuron_parameter_layouts,
        psc_parameter_layouts,
        neuron_prepared_layouts,
        psc_prepared_layouts,
        precise_exp=precise_exp,
        storage=storage,
        ring_capacity=ring_capacity,
        history_capacity=history_capacity,
        extra_pointers=extra_pointers,
        grid_sync_before_propagation=grid_sync_before_propagation,
        grid_sync_before_psc=grid_sync_before_psc,
        pre_propagation=tuple(pre_propagation),
        post_propagation=post_propagation,
        local_dtypes=local_dtypes,
        spike_list=spike_list,
    )
    return source.render(), tuple(source.pointers)


def generate_helion_source(
    backend,
    neuron_spec,
    psc_spec,
    n,
    state_receptors,
    drive_receptors,
    propagation=(),
    neuron_histories=(),
    psc_histories=(),
    modes=0,
    *,
    precise_exp=False,
    ring_capacity=None,
    history_capacity=None,
    extra_pointers=(),
    grid_sync_before_propagation=False,
    grid_sync_before_psc=False,
    pre_propagation=(),
    post_propagation=(),
    local_dtypes=(),
    neuron_parameter_layouts=None,
    psc_parameter_layouts=None,
    neuron_prepared_layouts=None,
    psc_prepared_layouts=None,
    storage=(),
    spike_list=(),
    plasticity_model=None,
):
    """Return one HELION timestep source using shared model templates."""
    rendered, declarations = _cached_helion_source(
        backend,
        replace(neuron_spec, dt=1.0),
        replace(psc_spec, dt=1.0),
        n,
        state_receptors,
        drive_receptors,
        tuple(propagation),
        tuple(neuron_histories),
        tuple(psc_histories),
        modes,
        None if neuron_parameter_layouts is None else tuple(neuron_parameter_layouts),
        None if psc_parameter_layouts is None else tuple(psc_parameter_layouts),
        None if neuron_prepared_layouts is None else tuple(neuron_prepared_layouts),
        None if psc_prepared_layouts is None else tuple(psc_prepared_layouts),
        precise_exp,
        ring_capacity,
        history_capacity,
        tuple(extra_pointers),
        grid_sync_before_propagation,
        grid_sync_before_psc,
        tuple(pre_propagation),
        tuple(post_propagation),
        tuple(local_dtypes),
        model_template_revision(neuron_spec.model),
        model_template_revision(psc_spec.model),
        tuple(storage),
        tuple(spike_list),
    )
    if plasticity_model is not None:
        from ..plasticity import _native_source

        rendered = _native_source(rendered, backend, plasticity_model)
    return rendered, declarations


def helion_csr_push(
    backend,
    max_degree,
    n_neuron,
    n_receptor,
    ring_capacity,
    *,
    n_delay=1,
    preload_history=False,
    block_size=128,
    spike_list=False,
):
    """Return bounded source-major CSR push statements for HELION.

    Triton discovers active source rows cooperatively within each CTA and
    traverses each selected row with all lanes. ``t_crow``/``t_col``/``t_perm``
    are the existing source-major cache, and ``weight`` remains in canonical
    edge order. Each selected row traverses its actual CSR bounds; silent
    rows do not scan their edges.
    """
    if backend not in ("triton", "tilelang"):
        raise ValueError("HELION CSR push requires Triton or TileLang")
    max_degree = int(max_degree)
    if max_degree < 0 or n_receptor < 1:
        raise ValueError("Invalid HELION CSR push dimensions")
    if ring_capacity is None and (n_delay != 1 or preload_history):
        raise ValueError("Delayed HELION CSR push requires a drive ring")
    if preload_history and ring_capacity != n_delay:
        raise ValueError("Delayed HELION history ring must match n_delay")
    if spike_list and (ring_capacity is not None or n_delay != 1):
        raise NotImplementedError(
            "Fused spike-list static propagation currently requires zero delays; "
            "delayed queues must be retained before selecting arrival work"
        )
    if backend == "triton":
        if ring_capacity is None:
            from .helion_plasticity import spike_list_candidates

            discovery = (
                (
                    *spike_list_candidates(backend),
                    f"remaining = tl.where(list_valid, listed_source, {n_neuron})",
                )
                if spike_list
                else (
                    f"remaining = tl.where(mask & (neuron_output != 0.0), "
                    f"index, {n_neuron})",
                )
            )
            return (
                "lane = tl.arange(0, BLOCK)",
                *discovery,
                "active_source = tl.min(remaining, axis=0)",
                f"while active_source < {n_neuron}:",
                (
                    "    source_spike = tl.load(spike_ptr + active_source)"
                    if spike_list
                    else "    source_spike = tl.sum(tl.where(index == active_source, "
                    "neuron_output, 0.0), axis=0)"
                ),
                "    row_start = tl.load(t_crow_ptr + active_source)",
                "    row_end = tl.load(t_crow_ptr + active_source + 1)",
                f"    for edge_offset in tl.range(0, row_end - row_start, "
                f"{block_size}):",
                "        edge = row_start + edge_offset + lane",
                "        edge_mask = edge < row_end",
                "        dst = tl.load(t_col_ptr + edge, mask=edge_mask, other=0)",
                "        slot = tl.load(t_perm_ptr + edge, mask=edge_mask, other=0)",
                "        value = tl.load(weight_ptr + slot, mask=edge_mask, other=0.0)",
                "        target = dst",
                "        tl.atomic_add(drive_ptr + target, value * source_spike, "
                "mask=edge_mask)",
                "    remaining = tl.where(remaining == active_source, "
                f"{n_neuron}, remaining)",
                "    active_source = tl.min(remaining, axis=0)",
            )
        lines = ["lane = tl.arange(0, BLOCK)"]
        if preload_history:
            for age in range(n_delay):
                lines.extend(
                    (
                        f"history_value_{age} = tl.load(delay_history_ptr + "
                        f"{age * n_neuron} + index, mask=mask, other=0.0)",
                        f"remaining_history_{age} = tl.where(mask & "
                        f"(history_value_{age} != 0.0), index, {n_neuron})",
                        f"active_history_{age} = tl.min(remaining_history_{age}, "
                        "axis=0)",
                        f"while active_history_{age} < {n_neuron}:",
                        f"    source_history = tl.sum(tl.where(index == "
                        f"active_history_{age}, history_value_{age}, 0.0), "
                        "axis=0)",
                        f"    for edge_delay in tl.range({age + 1}, {n_delay}):",
                        f"        row_start = tl.load(t_crow_ptr + "
                        f"active_history_{age} * {n_delay} + edge_delay)",
                        "        row_end = tl.load(t_crow_ptr + active_history_"
                        f"{age} * {n_delay} + edge_delay + 1)",
                        f"        for edge_offset in tl.range(0, row_end - "
                        f"row_start, {block_size}):",
                        "            edge = row_start + edge_offset + lane",
                        "            edge_mask = edge < row_end",
                        "            dst = tl.load(t_col_ptr + edge, mask=edge_mask, "
                        "other=0)",
                        "            slot = tl.load(t_perm_ptr + edge, mask=edge_mask, "
                        "other=0)",
                        "            value = tl.load(weight_ptr + slot, "
                        "mask=edge_mask, "
                        "other=0.0)",
                        f"            target = ((helion_step + edge_delay - "
                        f"{age + 1}) % {n_delay}) * "
                        f"{n_neuron} * {n_receptor} + dst",
                        "            tl.atomic_add(drive_ptr + target, value * "
                        "source_history, "
                        "mask=edge_mask)",
                        f"    remaining_history_{age} = tl.where("
                        f"remaining_history_{age} == active_history_{age}, "
                        f"{n_neuron}, remaining_history_{age})",
                        f"    active_history_{age} = tl.min(remaining_history_{age}, "
                        "axis=0)",
                    )
                )
        lines.extend(
            (
                f"remaining = tl.where(mask & (neuron_output != 0.0), "
                f"index, {n_neuron})",
                "active_source = tl.min(remaining, axis=0)",
                f"while active_source < {n_neuron}:",
                "    source_spike = tl.sum(tl.where(index == active_source, "
                "neuron_output, 0.0), axis=0)",
                f"    for edge_delay in tl.range(0, {n_delay}):",
                f"        row_start = tl.load(t_crow_ptr + active_source * "
                f"{n_delay} + edge_delay)",
                "        row_end = tl.load(t_crow_ptr + active_source * "
                f"{n_delay} + edge_delay + 1)",
                f"        for edge_offset in tl.range(0, row_end - row_start, "
                f"{block_size}):",
                "            edge = row_start + edge_offset + lane",
                "            edge_mask = edge < row_end",
                "            dst = tl.load(t_col_ptr + edge, mask=edge_mask, other=0)",
                "            slot = tl.load(t_perm_ptr + edge, mask=edge_mask, "
                "other=0)",
                "            value = tl.load(weight_ptr + slot, mask=edge_mask, "
                "other=0.0)",
                "            target = ((helion_step + edge_delay) % "
                f"{n_delay}) * {n_neuron} * {n_receptor} + dst",
                "            tl.atomic_add(drive_ptr + target, value * source_spike, "
                "mask=edge_mask)",
                "    remaining = tl.where(remaining == active_source, "
                f"{n_neuron}, remaining)",
                "    active_source = tl.min(remaining, axis=0)",
            )
        )
        return tuple(lines)
    if ring_capacity is None:
        if spike_list:
            from .helion_plasticity import spike_list_candidates

            return (
                *spike_list_candidates(backend),
                "if list_valid:",
                "    row_start = t_crow_ptr[listed_source]",
                "    row_end = t_crow_ptr[listed_source + 1]",
                "    spike_value = spike_ptr[listed_source]",
                "    for edge in T.serial(row_start, row_end):",
                "        dst = t_col_ptr[edge]",
                "        slot = t_perm_ptr[edge]",
                "        value = weight_ptr[slot]",
                "        target = dst",
                "        T.atomic_add(drive_ptr[target], value * spike_value)",
            )
        return (
            "row_start = t_crow_ptr[index]",
            "row_end = t_crow_ptr[index + 1]",
            "spike_value = neuron_output",
            "if spike_value != 0.0:",
            "    for edge in T.serial(row_start, row_end):",
            "        dst = t_col_ptr[edge]",
            "        slot = t_perm_ptr[edge]",
            "        value = weight_ptr[slot]",
            "        target = dst",
            "        T.atomic_add(drive_ptr[target], value * spike_value)",
        )
    lines = ["spike_value = neuron_output"]
    if preload_history:
        for age in range(n_delay):
            lines.extend(
                (
                    f"history_value = delay_history_ptr[{age} * {n_neuron} + index]",
                    f"for edge_delay in T.serial({age + 1}, {n_delay}):",
                    f"    row_start = t_crow_ptr[index * {n_delay} + edge_delay]",
                    f"    row_end = t_crow_ptr[index * {n_delay} + edge_delay + 1]",
                    "    if history_value != 0.0:",
                    "        for edge in T.serial(row_start, row_end):",
                    "            dst = t_col_ptr[edge]",
                    "            slot = t_perm_ptr[edge]",
                    "            value = weight_ptr[slot]",
                    f"            target = ((helion_step + edge_delay - "
                    f"{age + 1}) % {n_delay}) * "
                    f"{n_neuron} * {n_receptor} + dst",
                    "            T.atomic_add(drive_ptr[target], value * "
                    "history_value)",
                )
            )
    lines.extend(
        (
            f"for edge_delay in T.serial(0, {n_delay}):",
            f"    row_start = t_crow_ptr[index * {n_delay} + edge_delay]",
            f"    row_end = t_crow_ptr[index * {n_delay} + edge_delay + 1]",
        )
    )
    lines.extend(
        (
            "    if spike_value != 0.0:",
            "        for edge in T.serial(row_start, row_end):",
            "            dst = t_col_ptr[edge]",
            "            slot = t_perm_ptr[edge]",
            "            value = weight_ptr[slot]",
            f"            target = ((helion_step + edge_delay) % {n_delay}) * "
            f"{n_neuron} * {n_receptor} + dst",
            "            T.atomic_add(drive_ptr[target], value * spike_value)",
        )
    )
    return tuple(lines)


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
    neuron_prepared_layouts,
    psc_prepared_layouts,
    store_outputs,
    precise_exp,
    neuron_revision,
    psc_revision,
    storage=(),
    spike_list=(),
    gradient_activity=False,
    clear_drive=False,
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
        neuron_prepared_layouts,
        psc_prepared_layouts,
        store_outputs=store_outputs,
        precise_exp=precise_exp,
        storage=storage,
        spike_list=spike_list,
        gradient_activity=gradient_activity,
        clear_drive=clear_drive,
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
    neuron_prepared_layouts=None,
    psc_prepared_layouts=None,
    storage=(),
    spike_list=(),
    gradient_activity=False,
    clear_drive=False,
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
        None if neuron_prepared_layouts is None else tuple(neuron_prepared_layouts),
        None if psc_prepared_layouts is None else tuple(psc_prepared_layouts),
        store_outputs,
        precise_exp,
        model_template_revision(neuron_spec.model),
        model_template_revision(psc_spec.model),
        tuple(storage),
        tuple(spike_list),
        gradient_activity,
        clear_drive,
    )


@dataclass
class _PreparedFusedLaunch:
    """Own positional arguments, their slot metadata and storage lifetimes."""

    launch: object
    arguments: list
    slots: dict
    owners: list
    constants: tuple = ()
    stream: object = None


@dataclass
class _FusedStep:
    """Invocation-local paired kernel bindings with caller-owned
    destinations."""

    backend: str
    kernel: object
    n: int
    buffers: dict
    declarations: tuple
    strided_names: set[str]
    output_shapes: dict[str, tuple[int, ...]]
    enable_fp_fusion: bool
    storage: tuple = ()
    source_factory: object = None
    precise_division: bool = False
    cooperative_grid: bool = False
    mutated_buffers: tuple[str, ...] = ()
    state_dtypes: tuple[torch.dtype, torch.dtype] | None = None
    variants: dict = field(default_factory=dict, init=False)
    argument_indices: dict[str, int] = field(init=False)
    arguments: list = field(init=False)
    runtime: object = field(init=False, repr=False)
    bound_launch: object = field(default=None, init=False, repr=False)
    bound_slots: tuple = field(default=(), init=False, repr=False)

    def __post_init__(self):
        self.runtime = get_runtime(self.backend)
        self.argument_indices = {
            name: index for index, (name, _size) in enumerate(self.declarations)
        }
        self.arguments = [None] * len(self.declarations)
        declared_inputs = tuple(
            declaration
            for declaration in self.declarations
            if declaration[0] in self.buffers
        )
        inputs = tuple(
            self.runtime.argument(self.buffers[name], size)
            for name, size in declared_inputs
        )
        for (name, _size), value in zip(declared_inputs, inputs):
            self.arguments[self.argument_indices[name]] = value

    def allocate_outputs(self):
        reference = self.buffers["neuron_input"]
        psc_dtype = pointwise_dtype(
            self.buffers["drive"],
            *(
                value
                for name, value in self.buffers.items()
                if name.startswith("psc_") and name != "psc_dt" and "_grad_" not in name
            ),
        )
        neuron_dtype = torch.promote_types(
            psc_dtype,
            pointwise_dtype(
                reference,
                *(
                    value
                    for name, value in self.buffers.items()
                    if name.startswith("neuron_")
                    and name not in ("neuron_dt", "neuron_uniform")
                    and "_grad_" not in name
                ),
            ),
        )
        if self.state_dtypes is not None:
            neuron_dtype, psc_dtype = self.state_dtypes
        accumulator_dtype = (
            torch.float64
            if any(value.dtype == torch.float64 for value in self.buffers.values())
            else torch.float32
        )
        outputs = {}
        for name, shape in self.output_shapes.items():
            state_name = name.removesuffix("_final").removesuffix("_seq")
            state = self.buffers.get(state_name + "_init", reference)
            dtype = (
                accumulator_dtype
                if name.startswith("grad_") or "_grad_" in name
                else psc_dtype
                if name.startswith("psc_")
                else neuron_dtype
            )
            outputs[name] = torch.empty(shape, device=state.device, dtype=dtype)
        return outputs

    def _layout_variant(self, bindings):
        """Select a typed storage ABI for this set of caller-owned views."""
        if self.source_factory is not None:
            buffers = {**self.buffers, **bindings}
            storage = _tensor_storage(buffers, self.strided_names)
            if storage != self.storage:
                variant = self.variants.get(storage)
                if variant is None:
                    source, declarations = self.source_factory(storage=storage)
                    kernel = _compile(
                        self.backend,
                        source,
                        self.enable_fp_fusion,
                        self.precise_division,
                        device=buffers["neuron_input"].device,
                    )
                    variant = _FusedStep(
                        self.backend,
                        kernel,
                        self.n,
                        buffers,
                        declarations,
                        self.strided_names,
                        self.output_shapes,
                        self.enable_fp_fusion,
                        storage=storage,
                        precise_division=self.precise_division,
                        cooperative_grid=self.cooperative_grid,
                        mutated_buffers=self.mutated_buffers,
                        state_dtypes=self.state_dtypes,
                    )
                    self.variants[storage] = variant
                return variant
        return self

    def _launch_scope(self):
        if self.cooperative_grid:
            return self._cooperative_launch_scope()
        return _device_scope(self.buffers["neuron_input"].device)

    @contextmanager
    def _cooperative_launch_scope(self):
        with _device_scope(self.buffers["neuron_input"].device):
            try:
                yield
            except RuntimeError as error:
                if self.cooperative_grid and any(
                    token in str(error).lower()
                    for token in ("cooperative", "resident", "too many resources")
                ):
                    raise NotImplementedError(
                        "HELION cooperative grid cannot reside on the selected CUDA "
                        "device; reduce the launch grid or use a supported device"
                    ) from error
                raise

    def launch_into(self, bindings):
        """Launch into supplied storage, caching encountered storage
        layouts."""
        variant = self._layout_variant(bindings)
        if variant is not self:
            variant.launch_into(bindings)
            return
        declarations = tuple(
            declaration
            for declaration in self.declarations
            if declaration[0] in bindings
        )
        values = tuple(
            self.runtime.argument(bindings[name], size) for name, size in declarations
        )
        for (name, _size), value in zip(declarations, values):
            self.arguments[self.argument_indices[name]] = value
        changing = tuple(self.argument_indices[name] for name, _ in declarations)
        with self._launch_scope():
            if self.bound_launch is None or self.bound_slots != changing:
                self.bound_launch = self.runtime.prepare_launch(
                    self.kernel,
                    self.arguments,
                    self.n,
                    changing=changing,
                    enable_fp_fusion=self.enable_fp_fusion,
                    writes=self._writes(),
                    launch_cooperative_grid=self.cooperative_grid,
                )
                self.bound_slots = changing
            self.bound_launch(self.arguments)

    def _writes(self):
        outputs = (
            self.output_shapes.keys()
            | set(self.mutated_buffers)
            | {
                "spike_indices",
                "spike_cumulative",
                "spike_counts",
                "spike_accumulation",
                "gradient_activity",
            }
        )
        return tuple(
            i for i, (name, _) in enumerate(self.declarations) if name in outputs
        )

    def prepare_launch(self, bindings):
        """Bind a trajectory's invariant layouts before its repeated stages.

        Only the supplied tensors' addresses may change after preparation.
        The trajectory owns their shapes, strides and dtypes; standalone calls
        and reverse steps with variable layouts retain ``launch_into``.
        The recurrent operator holds the CUDA device guard for the trajectory.
        """
        variant = self._layout_variant(bindings)
        buffers = {**variant.buffers, **bindings}
        declarations = variant.declarations
        args = [
            self.runtime.argument(buffers[name], size) for name, size in declarations
        ]
        updates = tuple(
            (index, name, size)
            for index, (name, size) in enumerate(declarations)
            if name in bindings
        )
        with variant._launch_scope():
            launch = self.runtime.prepare_launch(
                variant.kernel,
                args,
                self.n,
                changing=tuple(index for index, _name, _size in updates),
                enable_fp_fusion=self.enable_fp_fusion,
                writes=variant._writes(),
                launch_cooperative_grid=variant.cooperative_grid,
            )
        if self.runtime.supports_pointer_offsets:
            args, owners = launch.arguments, launch.owners
            submit, constants, stream = self.runtime.bind_submission(launch)
        else:
            owners = args
            submit = functools.partial(launch, args)
            constants, stream = (), None

        return _PreparedFusedLaunch(
            submit,
            args,
            {name: (index, size) for index, name, size in updates},
            owners,
            constants,
            stream,
        )


def prepare_fused_step(
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
    neuron_prepared_parameters=None,
    psc_prepared_parameters=None,
    uniform=None,
    spike_list=None,
    gradient_activity=None,
    source_builder=None,
    extra_buffers=None,
    drive_receptor_count=None,
    cooperative_grid=False,
    mutated_buffers=(),
    clear_drive=False,
    state_dtypes=None,
):
    """Prepare one paired kernel and its input layouts outside the step loop.

    PSC states use an internal receptor layout while ``drive`` retains its
    receptor layout. A merged state has one internal PSC lane per neuron;
    compatible drive receptors are summed inside the cell kernel.
    """
    if neuron_input.device.type != "cuda":
        raise ValueError("Fused GPU cells require CUDA tensors")
    semantic_specs = (replace(neuron_spec, dt=1.0), replace(psc_spec, dt=1.0))
    key = (
        backend,
        semantic_specs,
        model_template_revision(neuron_spec.model),
        model_template_revision(psc_spec.model),
        (_fused_layout(neuron_input), _fused_layout(drive)),
        tuple(
            tuple(_fused_layout(value) for value in group)
            for group in (
                neuron_states,
                psc_states,
                neuron_parameters,
                psc_parameters,
            )
        ),
        (_fused_layout(uniform), _fused_layout(gradient_activity)),
        tuple(
            tuple((name, _fused_layout(value)) for name, value in sorted(items.items()))
            if items
            else ()
            for items in (neuron_prepared_parameters, psc_prepared_parameters)
        ),
        ()
        if spike_list is None
        else tuple(
            (name, _fused_layout(value))
            for name, value in spike_list.bindings().items()
        ),
        tuple(neuron_history_names),
        tuple(psc_history_names),
        None
        if source_builder is None
        else (
            source_builder.func,
            source_builder.args,
            tuple(sorted(source_builder.keywords.items())),
        ),
        tuple(
            (name, _fused_layout(value))
            for name, value in sorted((extra_buffers or {}).items())
        ),
        drive_receptor_count,
        cooperative_grid,
        tuple(mutated_buffers),
        clear_drive,
        state_dtypes,
        enable_fp_fusion,
        precise_division,
        precise_exp,
    )
    with _FUSED_PLAN_LOCK:
        plan = _FUSED_PLANS.get(key)
        if plan is not None:
            _FUSED_PLANS.move_to_end(key)
    if plan is not None:
        return plan.bind(
            neuron_input,
            drive,
            (neuron_states, psc_states),
            (neuron_parameters, psc_parameters),
            (neuron_spec, psc_spec),
            (neuron_prepared_parameters, psc_prepared_parameters),
            uniform,
            spike_list,
            gradient_activity,
            extra_buffers,
        )
    n = neuron_input.numel()
    _state_receptors = psc_states[0].numel() // n
    _drive_receptors = (
        int(drive_receptor_count)
        if drive_receptor_count is not None
        else drive.numel() // n
    )
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
    coefficient_buffers, neuron_prepared_layouts = _prepare_parameters(
        neuron_prepared_parameters, neuron, neuron_target
    )
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
    psc_computed_buffers, psc_prepared_layouts = _prepare_parameters(
        psc_prepared_parameters,
        psc,
        lambda _name: (
            (*neuron_input.shape, _state_receptors)
            if _state_receptors != 1
            else neuron_input.shape
        ),
    )
    buffers = {
        "neuron_dt": neuron_buffers["dt"],
        "psc_dt": psc_buffers["dt"],
        "neuron_input": neuron_input,
        "drive": drive,
    }
    if extra_buffers:
        duplicate = set(extra_buffers).intersection(buffers)
        if duplicate:
            raise ValueError(
                f"Extra buffers duplicate prepared names: {sorted(duplicate)}"
            )
        buffers.update(extra_buffers)
    if neuron_spec.surrogate == "poisson" and neuron_spec.poisson.spiking:
        if uniform is None:
            raise ValueError("Spiking Poisson fused cells require an explicit uniform")
        buffers["neuron_uniform"] = uniform
    buffers.update(
        {"neuron_" + name: value for name, value in coefficient_buffers.items()}
    )
    buffers.update(
        {"psc_" + name: value for name, value in psc_computed_buffers.items()}
    )
    if spike_list is not None:
        buffers.update(spike_list.bindings())
    if gradient_activity is not None:
        buffers["gradient_activity"] = gradient_activity
    output_shapes = {"spike": tuple(neuron_input.shape)}
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
            output_shapes[prefix + name + "_final"] = target(name)
        for name in template.parameter_names:
            buffers[prefix + name] = prepared[name]
        for name in history_names:
            output_shapes[prefix + name + "_seq"] = (1, *target(name))
    strided_names = {
        *("neuron_" + name for name in neuron.parameter_names),
        *("psc_" + name for name in psc.parameter_names),
        *("neuron_" + name for name in coefficient_buffers),
        *("psc_" + name for name in psc_computed_buffers),
        *(extra_buffers or {}),
    }
    if source_builder is None:
        source_factory = functools.partial(
            generate_fused_source,
            backend,
            *semantic_specs,
            n,
            _state_receptors,
            _drive_receptors,
            neuron_history_names,
            psc_history_names,
            modes,
            neuron_parameter_layouts=neuron_parameter_layouts,
            psc_parameter_layouts=psc_parameter_layouts,
            neuron_prepared_layouts=neuron_prepared_layouts,
            psc_prepared_layouts=psc_prepared_layouts,
            precise_exp=precise_exp,
            spike_list=() if spike_list is None else spike_list.producer_layout,
            gradient_activity=gradient_activity is not None,
            clear_drive=clear_drive,
        )
    else:
        source_factory = functools.partial(
            source_builder,
            modes=modes,
            neuron_parameter_layouts=neuron_parameter_layouts,
            psc_parameter_layouts=psc_parameter_layouts,
            neuron_prepared_layouts=neuron_prepared_layouts,
            psc_prepared_layouts=psc_prepared_layouts,
        )
    storage = _tensor_storage(buffers, strided_names)
    source, declarations = source_factory(storage=storage)
    kernel = _compile(
        backend, source, enable_fp_fusion, precise_division, device=neuron_input.device
    )

    factory = functools.partial(
        _FusedStep,
        backend,
        kernel,
        n,
        declarations=declarations,
        strided_names=strided_names,
        output_shapes=output_shapes,
        enable_fp_fusion=enable_fp_fusion,
        storage=storage,
        source_factory=source_factory,
        precise_division=precise_division,
        cooperative_grid=cooperative_grid,
        mutated_buffers=(*mutated_buffers, *(("drive",) if clear_drive else ())),
        state_dtypes=state_dtypes,
    )

    plan = _FusedPlan(
        factory,
        (neuron.state_names, psc.state_names),
        (neuron.parameter_names, psc.parameter_names),
        {
            prefix + name + "_init": tuple(buffers[prefix + name + "_init"].shape)
            for prefix, template in (("neuron_", neuron), ("psc_", psc))
            for name in template.state_names
        },
    )
    with _FUSED_PLAN_LOCK:
        _FUSED_PLANS[key] = plan
        _FUSED_PLANS.move_to_end(key)
        if len(_FUSED_PLANS) > 256:
            _FUSED_PLANS.popitem(last=False)
    return factory(buffers=buffers)


def prepare_helion_step(
    neuron_input,
    future_current,
    neuron_states,
    psc_states,
    neuron_parameters,
    psc_parameters,
    neuron_spec,
    psc_spec,
    propagation,
    extra_buffers,
    neuron_history_names=(),
    psc_history_names=(),
    *,
    backend="triton",
    enable_fp_fusion=True,
    precise_division=False,
    precise_exp=False,
    neuron_prepared_parameters=None,
    psc_prepared_parameters=None,
    grid_sync_before_propagation=False,
    grid_sync_before_psc=False,
    pre_propagation=(),
    post_propagation=(),
    local_dtypes=(),
    cooperative_grid=True,
    ring_capacity=None,
    history_capacity=None,
    mutated_buffers=(),
    spike_list=None,
    state_dtypes=None,
    plasticity_model=None,
):
    """Prepare one HELION neuron/propagation/PSC launch.

    ``future_current`` is the invocation-owned no-delay drive buffer. The
    source builder and preparation object are the existing fused-kernel
    machinery with only the phase order and injected CSR propagation changed.
    """
    n = neuron_input.numel()
    state_receptors = psc_states[0].numel() // n
    if ring_capacity is None:
        if tuple(future_current.shape) != (n, state_receptors):
            raise ValueError("HELION drive shape disagrees with PSC state shape")
    elif tuple(future_current.shape) != (
        int(ring_capacity),
        n,
        state_receptors,
    ):
        raise ValueError("HELION delay ring shape disagrees with PSC state shape")
    source_builder = functools.partial(
        generate_helion_source,
        backend,
        # Cached source factories must retain semantic metadata only. Runtime
        # dt tensors can belong to a compiled CUDA Graph's private pool.
        replace(neuron_spec, dt=1.0),
        replace(psc_spec, dt=1.0),
        n,
        state_receptors,
        state_receptors,
        tuple(propagation),
        tuple(neuron_history_names),
        tuple(psc_history_names),
        precise_exp=precise_exp,
        ring_capacity=ring_capacity,
        history_capacity=history_capacity,
        grid_sync_before_propagation=grid_sync_before_propagation,
        grid_sync_before_psc=grid_sync_before_psc,
        pre_propagation=tuple(pre_propagation),
        post_propagation=tuple(post_propagation),
        local_dtypes=tuple(local_dtypes),
        spike_list=() if spike_list is None else spike_list.producer_layout,
        extra_pointers=tuple(
            (name, value.numel()) for name, value in extra_buffers.items()
        ),
        plasticity_model=plasticity_model,
    )
    buffers = dict(extra_buffers)
    if state_dtypes is None:
        psc = get_template(psc_spec)
        state_dtypes = (
            pointwise_dtype(
                neuron_input,
                *neuron_states,
                *neuron_parameters,
                psc_states[psc.state_names.index(psc.output_state)],
            ),
            pointwise_dtype(future_current, *psc_states, *psc_parameters),
        )
    if spike_list is not None:
        buffers.update(spike_list.bindings())
    return prepare_fused_step(
        neuron_input,
        future_current,
        neuron_states,
        psc_states,
        neuron_parameters,
        psc_parameters,
        neuron_spec,
        psc_spec,
        neuron_history_names,
        psc_history_names,
        backend=backend,
        enable_fp_fusion=enable_fp_fusion,
        precise_division=precise_division,
        precise_exp=precise_exp,
        neuron_prepared_parameters=neuron_prepared_parameters,
        psc_prepared_parameters=psc_prepared_parameters,
        source_builder=source_builder,
        extra_buffers=buffers,
        drive_receptor_count=state_receptors,
        cooperative_grid=cooperative_grid,
        state_dtypes=state_dtypes,
        mutated_buffers=(
            "drive",
            "grid_barrier",
            "grid_epoch",
            "spike_history",
            "delay_history_record",
            *mutated_buffers,
        ),
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
    neuron_prepared_parameters=None,
    psc_prepared_parameters=None,
    uniform=None,
):
    """Execute a standalone paired step using the shared prepared launcher."""
    prepared = prepare_fused_step(
        neuron_input,
        drive,
        neuron_states,
        psc_states,
        neuron_parameters,
        psc_parameters,
        neuron_spec,
        psc_spec,
        neuron_history_names,
        psc_history_names,
        backend=backend,
        enable_fp_fusion=enable_fp_fusion,
        precise_division=precise_division,
        precise_exp=precise_exp,
        neuron_prepared_parameters=neuron_prepared_parameters,
        psc_prepared_parameters=psc_prepared_parameters,
        uniform=uniform,
    )
    outputs = prepared.allocate_outputs()
    prepared.launch_into(outputs)
    neuron, psc = get_template(neuron_spec), get_template(psc_spec)
    return (
        outputs["spike"],
        tuple(outputs["neuron_" + name + "_final"] for name in neuron.state_names),
        tuple(outputs["psc_" + name + "_final"] for name in psc.state_names),
        tuple(outputs["neuron_" + name + "_seq"] for name in neuron_history_names),
        tuple(outputs["psc_" + name + "_seq"] for name in psc_history_names),
    )


def _reverse_statements(
    spec, prefix, names, modes=0, *, prepared_values=(), record_presence=None
):
    statements = []
    for statement in get_fragments(
        spec,
        modes,
        prepared_values=prepared_values,
    ).backward:
        if isinstance(statement, RecordCotangent):
            if record_presence and record_presence.get(statement.name, False):
                name = statement.name
                statements.append(f"grad_s_{name} = grad_s_{name} + grad_r_{name}")
            continue
        lhs = statement.split(" = ", 1)[0]
        if lhs.startswith("grad_p_"):
            base, _ = _mode_name(lhs[len("grad_p_") :], modes)
            if base not in names:
                continue
        statements.append(statement)
    return namespace_statements(tuple(statements), prefix)


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
    neuron_prepared_layouts,
    psc_prepared_layouts,
    precise_exp,
    neuron_revision,
    psc_revision,
    storage=(),
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
        neuron_prepared_layouts=neuron_prepared_layouts,
        psc_prepared_layouts=psc_prepared_layouts,
        store_outputs=False,
        precise_exp=precise_exp,
        storage=storage,
    )

    def neuron_size(name):
        return n * modes if neuron.extra_dims(name) else n

    def psc_size(_name):
        return n * _state_receptors

    def cotangent(local, name, index, size, present):
        dtype = (
            "float64"
            if any(layout.dtype == "float64" for _, layout in storage)
            else "float32"
        )
        if present:
            source.pointer(name, size)
            source.load(local, name, index, dtype=dtype)
        else:
            source.lines.append(local + " = 0.0")
            source.local_dtypes[local] = dtype

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
        if name in neuron.state_names:
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
            prepared_values=tuple(
                item.scalar_name
                for item in neuron.prepared_parameters
                if item.name in dict(neuron_prepared_layouts or ())
            ),
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
    deferred_records = get_fragments(psc_spec).deferred_records
    for receptor in range(_state_receptors):
        prefix = f"psc_{receptor}_"
        index = f"index * {_state_receptors} + {receptor}"
        source.statements((prefix + "grad_output = neuron_grad_x",))
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
            if name not in deferred_records:
                source.lines.append(
                    f"{prefix}grad_s_{name} = "
                    f"{prefix}grad_s_{name} + {prefix}grad_r_{name}"
                )
        psc_reverse = list(
            _reverse_statements(
                psc_spec,
                prefix,
                psc_parameter_names,
                record_presence=psc_record_presence,
                prepared_values=tuple(
                    item.scalar_name
                    for item in psc.prepared_parameters
                    if item.name in dict(psc_prepared_layouts or ())
                ),
            )
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
    neuron_prepared_layouts=None,
    psc_prepared_layouts=None,
    precise_exp=False,
    storage=(),
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
        None if neuron_prepared_layouts is None else tuple(neuron_prepared_layouts),
        None if psc_prepared_layouts is None else tuple(psc_prepared_layouts),
        precise_exp,
        model_template_revision(neuron_spec.model),
        model_template_revision(psc_spec.model),
        tuple(storage),
    )


@dataclass
class _FusedBackwardStep:
    """Own reverse working sets for one trajectory, independently of its
    tape."""

    step: _FusedStep
    neuron: object
    psc: object
    neuron_parameters: tuple
    psc_parameters: tuple
    neuron_parameter_names: set[str]
    psc_parameter_names: set[str]
    outputs: tuple = field(init=False)
    index: int = field(default=0, init=False)
    launches: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        # Alternate destinations: incoming adjoints remain readable throughout
        # the launch that produces the preceding timestep's adjoints.
        self.outputs = (self.step.allocate_outputs(), self.step.allocate_outputs())

    def backward(
        self,
        neuron_input,
        drive,
        neuron_states,
        psc_states,
        grad_spike,
        grad_neuron_states,
        grad_psc_states,
        *,
        authoritative_spike=None,
        grad_neuron_records=None,
        grad_psc_records=None,
        destinations=None,
    ):
        bindings = dict(self.outputs[self.index % 2])
        bindings.update(destinations or {})
        self.index += 1
        bindings["neuron_input"] = neuron_input
        bindings["drive"] = drive
        if "neuron_spike" in self.step.buffers:
            bindings["neuron_spike"] = authoritative_spike
        if grad_spike is not None:
            bindings["grad_spike"] = grad_spike
        for prefix, template, states, gradients, records in (
            (
                "neuron_",
                self.neuron,
                neuron_states,
                grad_neuron_states,
                grad_neuron_records or {},
            ),
            ("psc_", self.psc, psc_states, grad_psc_states, grad_psc_records or {}),
        ):
            for name, state, gradient in zip(template.state_names, states, gradients):
                key = prefix + name + "_init"
                target = self.step.buffers[key].shape
                bindings[key] = state if state.shape == target else state.expand(target)
                if gradient is not None:
                    bindings[prefix + "grad_" + name] = gradient
            for name, gradient in records.items():
                if gradient is not None:
                    bindings[prefix + "record_grad_" + name] = gradient
        stream = torch.cuda.current_stream(self.step.buffers["neuron_input"].device)
        signature = (
            stream.cuda_stream,
            tuple((name, _sequence_layout(value)) for name, value in bindings.items()),
        )
        prepared = self.launches.get(signature)
        if prepared is None:
            prepared = self.step.prepare_launch(bindings)
            self.launches[signature] = prepared
        args, owners = prepared.arguments, prepared.owners
        with _device_scope(self.step.buffers["neuron_input"].device):
            if self.step.runtime.supports_pointer_offsets:
                for name, (index, _span) in prepared.slots.items():
                    value = bindings[name]
                    owners[index] = value
                    args[index] = value.data_ptr()
                prepared.launch(*args, *prepared.constants)
            else:
                for name, (index, span) in prepared.slots.items():
                    args[index] = self.step.runtime.argument(bindings[name], span)
                prepared.launch()

        def reduce(name, value):
            result = bindings[name]
            if result.shape != value.shape:
                result = result.sum_to_size(value.shape)
            return result if result.dtype == value.dtype else result.to(value.dtype)

        return (
            bindings["grad_neuron_input"]
            if destinations is not None
            else bindings["grad_neuron_input"].to(neuron_input.dtype),
            bindings["grad_drive"].to(drive.dtype),
            tuple(
                reduce("neuron_grad_" + name + "_init", value)
                for name, value in zip(self.neuron.state_names, neuron_states)
            ),
            tuple(
                reduce("psc_grad_" + name + "_init", value)
                for name, value in zip(self.psc.state_names, psc_states)
            ),
            tuple(
                reduce("neuron_grad_" + name, value)
                if name in self.neuron_parameter_names
                else None
                for name, value in zip(
                    self.neuron.parameter_names, self.neuron_parameters
                )
            ),
            tuple(
                reduce("psc_grad_" + name, value)
                if name in self.psc_parameter_names
                else None
                for name, value in zip(self.psc.parameter_names, self.psc_parameters)
            ),
        )


def prepare_fused_backward(
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
    neuron_prepared_parameters=None,
    psc_prepared_parameters=None,
):
    """Prepare a paired reverse kernel and private gradient working sets."""
    if neuron_input.device.type != "cuda":
        raise ValueError("Fused GPU cells require CUDA tensors")
    semantic_specs = (replace(neuron_spec, dt=1.0), replace(psc_spec, dt=1.0))
    key = (
        "backward",
        backend,
        semantic_specs,
        tuple(model_template_revision(spec.model) for spec in semantic_specs),
        tuple(
            _fused_layout(value) for value in (neuron_input, drive, authoritative_spike)
        ),
        tuple(
            tuple(_fused_layout(value) for value in group)
            for group in (
                neuron_states,
                psc_states,
                neuron_parameters,
                psc_parameters,
                (grad_spike,),
                grad_neuron_states,
                grad_psc_states,
            )
        ),
        tuple(
            tuple((name, _fused_layout(value)) for name, value in sorted(items.items()))
            if items
            else ()
            for items in (
                neuron_prepared_parameters,
                psc_prepared_parameters,
                grad_neuron_records,
                grad_psc_records,
            )
        ),
        None if needs_neuron_parameters is None else tuple(needs_neuron_parameters),
        None if needs_psc_parameters is None else tuple(needs_psc_parameters),
        enable_fp_fusion,
        precise_division,
        precise_exp,
    )
    neuron, psc = get_template(neuron_spec), get_template(psc_spec)
    neuron_parameter_names = {
        name
        for name, needed in zip(
            neuron.parameter_names,
            needs_neuron_parameters or (True,) * len(neuron.parameter_names),
        )
        if needed
    }
    psc_parameter_names = {
        name
        for name, needed in zip(
            psc.parameter_names,
            needs_psc_parameters or (True,) * len(psc.parameter_names),
        )
        if needed
    }
    with _FUSED_PLAN_LOCK:
        plan = _FUSED_PLANS.get(key)
        if plan is not None:
            _FUSED_PLANS.move_to_end(key)
    if plan is not None:
        buffers = plan.bind_inputs(
            neuron_input,
            drive,
            (neuron_states, psc_states),
            (neuron_parameters, psc_parameters),
            (neuron_spec, psc_spec),
            (neuron_prepared_parameters, psc_prepared_parameters),
        )
        if neuron_spec.surrogate == "poisson" and neuron_spec.poisson.spiking:
            if authoritative_spike is None:
                raise ValueError("Poisson fused reverse requires authoritative spikes")
            buffers["neuron_spike"] = authoritative_spike
        if grad_spike is not None:
            buffers["grad_spike"] = grad_spike
        for prefix, template, gradients, records in (
            ("neuron_", neuron, grad_neuron_states, grad_neuron_records or {}),
            ("psc_", psc, grad_psc_states, grad_psc_records or {}),
        ):
            buffers.update(
                (prefix + "grad_" + name, value)
                for name, value in zip(template.state_names, gradients)
                if value is not None
            )
            buffers.update(
                (prefix + "record_grad_" + name, value)
                for name, value in records.items()
                if value is not None
            )
        return _FusedBackwardStep(
            plan.factory(buffers=buffers),
            neuron,
            psc,
            tuple(neuron_parameters),
            tuple(psc_parameters),
            neuron_parameter_names,
            psc_parameter_names,
        )
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
    coefficient_buffers, neuron_prepared_layouts = _prepare_parameters(
        neuron_prepared_parameters, neuron, neuron_target
    )
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
    buffers = {
        "neuron_input": neuron_input,
        "drive": drive,
    }
    if neuron_spec.surrogate == "poisson" and neuron_spec.poisson.spiking:
        if authoritative_spike is None:
            raise ValueError("Poisson fused reverse requires authoritative spikes")
        buffers["neuron_spike"] = authoritative_spike
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
            buffers[name] = value

    output_shapes = {}

    def output_gradient(name, like):
        output_shapes[name] = tuple(like.shape)

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
                output_shapes[prefix + "grad_" + name] = layout.target_shape
    for name in neuron.history_names:
        if name in neuron.state_names:
            continue
        cotangent("neuron_record_grad_" + name, neuron_record_grads.get(name))
    output_gradient("grad_neuron_input", neuron_input)
    output_gradient("grad_drive", drive)
    psc_computed_buffers, psc_prepared_layouts = _prepare_parameters(
        psc_prepared_parameters,
        psc,
        lambda _name: (
            (*neuron_input.shape, _state_receptors)
            if _state_receptors != 1
            else neuron_input.shape
        ),
    )
    buffers.update(
        {"psc_" + name: value for name, value in psc_computed_buffers.items()}
    )
    strided_names = {
        *("neuron_" + name for name in neuron.parameter_names),
        *("psc_" + name for name in psc.parameter_names),
        *("neuron_" + name for name in coefficient_buffers),
        *("psc_" + name for name in psc_computed_buffers),
    }
    source_factory = functools.partial(
        _generate_fused_backward_source,
        backend,
        *semantic_specs,
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
        neuron_prepared_layouts,
        psc_prepared_layouts,
        precise_exp=precise_exp,
    )
    storage = _tensor_storage(buffers, strided_names)
    source, declarations = source_factory(storage=storage)
    kernel = _compile(
        backend, source, enable_fp_fusion, precise_division, device=neuron_input.device
    )

    factory = functools.partial(
        _FusedStep,
        backend,
        kernel,
        n,
        declarations=declarations,
        strided_names=strided_names,
        output_shapes=output_shapes,
        enable_fp_fusion=enable_fp_fusion,
        storage=storage,
        source_factory=source_factory,
        precise_division=precise_division,
    )
    plan = _FusedPlan(
        factory,
        (neuron.state_names, psc.state_names),
        (neuron.parameter_names, psc.parameter_names),
        {
            prefix + name + "_init": tuple(buffers[prefix + name + "_init"].shape)
            for prefix, template in (("neuron_", neuron), ("psc_", psc))
            for name in template.state_names
        },
    )
    with _FUSED_PLAN_LOCK:
        _FUSED_PLANS[key] = plan
        _FUSED_PLANS.move_to_end(key)
        if len(_FUSED_PLANS) > 256:
            _FUSED_PLANS.popitem(last=False)
    step = factory(buffers=buffers)
    return _FusedBackwardStep(
        step,
        neuron,
        psc,
        tuple(neuron_parameters),
        tuple(psc_parameters),
        neuron_parameter_names,
        psc_parameter_names,
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
    neuron_prepared_parameters=None,
    psc_prepared_parameters=None,
):
    """Reverse a paired stage in one launch with local charge recomputation.

    Returns:
        External-input, drive, neuron-state, PSC-state, neuron-parameter and
        PSC-parameter cotangents in their original tensor shapes and dtypes.
    """
    prepared = prepare_fused_backward(
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
        authoritative_spike=authoritative_spike,
        grad_neuron_records=grad_neuron_records,
        grad_psc_records=grad_psc_records,
        needs_neuron_parameters=needs_neuron_parameters,
        needs_psc_parameters=needs_psc_parameters,
        backend=backend,
        enable_fp_fusion=enable_fp_fusion,
        precise_division=precise_division,
        precise_exp=precise_exp,
        neuron_prepared_parameters=neuron_prepared_parameters,
        psc_prepared_parameters=psc_prepared_parameters,
    )
    return prepared.backward(
        neuron_input,
        drive,
        neuron_states,
        psc_states,
        grad_spike,
        grad_neuron_states,
        grad_psc_states,
        authoritative_spike=authoritative_spike,
        grad_neuron_records=grad_neuron_records,
        grad_psc_records=grad_psc_records,
    )
