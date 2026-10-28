"""Pair PSC adaptation and the following neuron update in one cell launch.

Connections remain independent. The recurrent caller executes the first
neuron alone, these paired stages between connections, and the last PSC
alone. A nonempty T-step sequence therefore uses T connections and T+1
cell launches.
"""

import functools
from dataclasses import dataclass, field, replace

import torch

from ..program import namespace_statements, substitute_expression
from ..templates import RecordCotangent, get_template, model_template_revision
from .engine import (
    _compile,
    _device_scope,
    _layout,
    _load_prepared_parameters,
    _mode_name,
    _parameter_buffers,
    _parameter_index,
    _prepare,
    _prepare_parameters,
    _Source,
    _tensor_storage,
)
from .fragments import get_fragments, scalar_layout
from .renderers import get_runtime


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
):
    """Render one direct PSC-to-neuron pipeline stage using shared fragments.

    Forward and reverse share the scalar compute stages. ``store_outputs`` is
    a Python generation-time choice: reverse recomputation declares only its
    inputs and keeps intermediates in registers, without forward output stores
    or pointers in the generated kernel.
    """
    source = _Source(backend, n, 1, modes, precise_exp=precise_exp, storage=storage)
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
        source.lines.append("neuron_x = neuron_x + " + prefix + "output")
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
    )


@dataclass
class _PreparedFusedLaunch:
    """Own positional arguments, their slot metadata and storage lifetimes."""

    launch: object
    arguments: list
    slots: dict
    owners: list


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
    variants: dict = field(default_factory=dict, init=False)
    argument_indices: dict[str, int] = field(init=False)
    arguments: list = field(init=False)
    runtime: object = field(init=False, repr=False)

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
        return {
            name: torch.empty(shape, device=reference.device, dtype=torch.float32)
            for name, shape in self.output_shapes.items()
        }

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
                    )
                    self.variants[storage] = variant
                return variant
        return self

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
        with _device_scope(self.buffers["neuron_input"].device):
            self.runtime.launch(
                self.kernel,
                self.arguments,
                self.n,
                enable_fp_fusion=self.enable_fp_fusion,
                debug=None,
                num_warps=None,
            )

    def prepare_launch(self, bindings):
        """Bind a trajectory's invariant layouts before its repeated stages.

        Only the supplied tensors' addresses may change after preparation.
        The trajectory owns their shapes, strides and dtypes; standalone calls
        and reverse steps with variable layouts retain ``launch_into``.
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
        device = buffers["neuron_input"].device
        with _device_scope(device):
            launch = self.runtime.prepare_launch(
                variant.kernel,
                args,
                self.n,
                changing=tuple(index for index, _name, _size in updates),
                enable_fp_fusion=self.enable_fp_fusion,
            )
        if self.runtime.supports_pointer_offsets:
            args, owners = launch.arguments, launch.owners
            submit = launch.submit
        else:
            owners = args
            submit = functools.partial(launch, args)

        def run():
            with _device_scope(device):
                submit()

        return _PreparedFusedLaunch(
            run,
            args,
            {name: (index, size) for index, name, size in updates},
            owners,
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
):
    """Prepare one paired kernel and its input layouts outside the step loop.

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
    }
    source_factory = functools.partial(
        generate_fused_source,
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
        neuron_prepared_layouts=neuron_prepared_layouts,
        psc_prepared_layouts=psc_prepared_layouts,
        precise_exp=precise_exp,
        spike_list=() if spike_list is None else spike_list.producer_layout,
        gradient_activity=gradient_activity is not None,
    )
    storage = _tensor_storage(buffers, strided_names)
    source, declarations = source_factory(storage=storage)
    kernel = _compile(
        backend, source, enable_fp_fusion, precise_division, device=neuron_input.device
    )

    return _FusedStep(
        backend,
        kernel,
        n,
        buffers,
        declarations,
        strided_names,
        output_shapes,
        enable_fp_fusion,
        storage=storage,
        source_factory=source_factory,
        precise_division=precise_division,
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
                bindings[key] = state.expand(self.step.buffers[key].shape)
                if gradient is not None:
                    bindings[prefix + "grad_" + name] = gradient
            for name, gradient in records.items():
                if gradient is not None:
                    bindings[prefix + "record_grad_" + name] = gradient
        self.step.launch_into(bindings)

        def reduce(name, value):
            return bindings[name].sum_to_size(value.shape).to(value.dtype)

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
        neuron_prepared_layouts,
        psc_prepared_layouts,
        precise_exp=precise_exp,
    )
    storage = _tensor_storage(buffers, strided_names)
    source, declarations = source_factory(storage=storage)
    kernel = _compile(
        backend, source, enable_fp_fusion, precise_division, device=neuron_input.device
    )

    step = _FusedStep(
        backend,
        kernel,
        n,
        buffers,
        declarations,
        strided_names,
        output_shapes,
        enable_fp_fusion,
        storage=storage,
        source_factory=source_factory,
        precise_division=precise_division,
    )
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
