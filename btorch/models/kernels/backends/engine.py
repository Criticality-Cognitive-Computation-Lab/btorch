"""Render fixed scalar templates into independent sequence kernels.

One lane owns a neuron across time. Connections never enter these
kernels. Parameter VJPs accumulate in lane registers across time and
reduce across broadcast batch axes afterwards; no time-by-edge
intermediate is created.
"""

import functools
import hashlib
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ....sparse.runtime.activity import BLOCK_SIZE
from ....utils.kernel import storage_span
from ..contracts import surrogate_support
from ..program import substitute_expression
from ..templates import (
    RecordCotangent,
    ResidualTemplate,
    get_template,
    has_generated_model,
    model_template_revision,
)
from .fragments import get_fragments, scalar_layout
from .renderers import get_renderer, get_runtime


@dataclass(frozen=True)
class _ParameterLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    target_shape: tuple[int, ...]
    span: int


@dataclass(frozen=True)
class _TensorStorage:
    shape: tuple[int | None, ...]
    stride: tuple[int, ...]
    dtype: str
    span: int
    physical_index: bool = False
    dynamic_first: bool = False


def _tensor_storage(buffers, physical_names=(), dynamic_names=()):
    """Describe views requiring typed pointers or noncanonical addressing."""
    entries = []
    for name, value in sorted(buffers.items()):
        dynamic_first = name in dynamic_names and value.ndim > 0
        if value.dtype == torch.float32 and value.is_contiguous() and not dynamic_first:
            continue
        shape = tuple(value.shape)
        if dynamic_first:
            # The time extent is invocation data.  The trailing dimensions and
            # strides still describe the caller's storage layout.
            shape = (None, *shape[1:])
            span = 0
        else:
            span = (
                0
                if not value.numel()
                else 1
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(value.shape, value.stride())
                )
            )
        entries.append(
            (
                name,
                _TensorStorage(
                    shape,
                    tuple(value.stride()),
                    str(value.dtype).removeprefix("torch."),
                    span,
                    name in physical_names,
                    dynamic_first,
                ),
            )
        )
    return tuple(entries)


@dataclass(frozen=True)
class _GradientPresence:
    output: bool
    final_states: tuple[bool, ...]
    histories: tuple[bool, ...]


@dataclass(frozen=True)
class _GradientTargets:
    input: bool
    states: tuple[bool, ...]


def _normalize_gradient_presence(template, history_names, presence):
    if presence is None:
        return _GradientPresence(
            True,
            (True,) * len(template.state_names),
            (True,) * len(history_names),
        )
    if not isinstance(presence, _GradientPresence):
        presence = _GradientPresence(*presence)
    presence = _GradientPresence(
        bool(presence.output),
        tuple(bool(value) for value in presence.final_states),
        tuple(bool(value) for value in presence.histories),
    )
    if len(presence.final_states) != len(template.state_names):
        raise ValueError("Final-state gradient mask count disagrees with the template")
    if len(presence.histories) != len(history_names):
        raise ValueError(
            "History gradient mask count disagrees with requested histories"
        )
    return presence


def _normalize_gradient_targets(template, targets):
    if targets is None:
        return _GradientTargets(True, (True,) * len(template.state_names))
    if not isinstance(targets, _GradientTargets):
        targets = _GradientTargets(*targets)
    targets = _GradientTargets(
        bool(targets.input), tuple(bool(value) for value in targets.states)
    )
    if len(targets.states) != len(template.state_names):
        raise ValueError("State gradient target count disagrees with the template")
    return targets


def supports(spec, device):
    """Restrict built-in GPU kernels to their explicitly declared models."""
    return device.type == "cuda" and has_generated_model(spec.model)


def _mode_name(name, modes):
    base, separator, component = name.rpartition("_")
    return (
        (base, int(component))
        if modes and separator and component.isdigit()
        else (name, None)
    )


def _parameter_index(scalar, layout, modes, renderer):
    """Render a storage offset for one broadcast parameter element."""
    _, mode = _mode_name(scalar, modes)
    logical_shape = layout.target_shape
    target_shape = logical_shape[:-1] if mode is not None else logical_shape
    padding = len(logical_shape) - len(layout.shape)
    terms = []
    for axis, (dimension, stride) in enumerate(zip(layout.shape, layout.stride)):
        target_axis = padding + axis
        target_dimension = logical_shape[target_axis]
        if dimension not in (1, target_dimension):
            raise ValueError(
                f"Parameter shape {layout.shape} cannot broadcast to {logical_shape}"
            )
        if dimension == 1 or stride == 0:
            continue
        if mode is not None and target_axis == len(logical_shape) - 1:
            terms.append(str(mode * stride))
            continue
        else:
            lane_axis = target_axis
            trailing = 1
            for size in target_shape[lane_axis + 1 :]:
                trailing *= size
            division = renderer.division
            coordinate = "index" if trailing == 1 else f"index {division} {trailing}"
            if target_shape[lane_axis] != 1:
                coordinate = f"({coordinate}) % {target_shape[lane_axis]}"
        terms.append(coordinate if stride == 1 else f"({coordinate}) * {stride}")
    return " + ".join(terms) or "0"


class _Source:
    def __init__(
        self,
        backend,
        n,
        n_step,
        modes,
        precise_exp=False,
        storage=(),
        dynamic_steps=False,
    ):
        self.backend, self.n, self.n_step, self.modes = backend, n, n_step, modes
        self.dynamic_steps = dynamic_steps
        self.precise_exp = precise_exp
        self.storage = dict(storage)
        self.renderer = get_renderer(
            backend, n, n_step, self.storage, precise_exp, dynamic_steps
        )
        self.pointers = []
        self.lines = []
        self.epilogue = []
        self.has_spike_list = False
        self.has_gradient_list = False

    def pointer(self, name, size):
        if name in self.storage:
            if size is not None:
                size = self.storage[name].span
        if (name, size) not in self.pointers:
            self.pointers.append((name, size))

    def index(self, scalar, time=None):
        _, mode = _mode_name(scalar, self.modes)
        index = "index" if mode is None else f"index * {self.modes} + {mode}"
        if time is not None:
            stride = self.n if mode is None else self.n * self.modes
            index = f"({time}) * {stride} + {index}"
        return index

    def storage_index(self, pointer, index):
        return self.renderer.storage_index(self.storage.get(pointer), index)

    def load(self, local, pointer, index):
        lane_index = "index" in index
        index = self.storage_index(pointer, index)
        expr = self.renderer.load(
            pointer, index, self.storage.get(pointer), lane_index=lane_index
        )
        self.lines.append(f"{local} = {expr}")

    def store(self, pointer, index, local):
        index = self.storage_index(pointer, index)
        self.lines.append(self.renderer.store(pointer, index, local))

    def statements(self, lines):
        self.lines.extend(lines)

    def emit_spike_list(self, value, layout):
        """Append an optional sparse output sink after shared model math."""
        if not layout:
            return
        n_source, n_output, capacity = layout
        if capacity != -(-self.n // BLOCK_SIZE) * BLOCK_SIZE:
            raise ValueError("Spike-list capacity must match the producer block layout")
        for name, size in (
            ("spike_indices", capacity),
            ("spike_cumulative", capacity),
            ("spike_counts", capacity // BLOCK_SIZE),
            ("spike_degree", n_source),
            ("spike_accumulation", n_output),
        ):
            self.pointer(name, size)
        emission = self.renderer.emit_spike_list(value, self.n, layout)
        self.lines.extend(emission.lines)
        self.epilogue.extend(emission.epilogue)
        self.has_spike_list |= emission.feature == "spike_list"

    def emit_gradient_activity(self, spec, *, prefix="", time=None):
        """Store exact output-adjoint support declared by the model
        template."""
        scalar = get_template(spec).spike_gradient_input
        width = surrogate_support(spec)
        if scalar is None or width is None:
            raise ValueError(
                "Gradient activity requires declared compact spike support"
            )
        self.pointer(
            "gradient_activity",
            None if self.dynamic_steps else self.n * self.n_step,
        )
        layout = self.storage.get("gradient_activity")
        emission = self.renderer.emit_gradient_activity(
            scalar, width, prefix, time, self.n, self.n_step, layout
        )
        self.lines.extend(emission.lines)
        self.has_gradient_list |= emission.feature == "gradient_list"

    def render(self):
        return self.renderer.render(
            self.lines,
            tuple(self.pointers),
            self.has_spike_list,
            self.has_gradient_list,
            self.epilogue,
        )


@functools.lru_cache(maxsize=256)
def _generate_source(
    backend,
    spec,
    n,
    n_step,
    history_names,
    modes=0,
    backward=False,
    parameter_mask=None,
    parameter_layouts=None,
    gradient_presence=None,
    template_revision=1,
    residual_template=None,
    prepared_parameter_layouts=None,
    gradient_targets=None,
    precise_exp=False,
    storage=(),
    spike_list=(),
    gradient_activity=False,
):
    # The revision participates in the cache key even though current built-in
    # templates are immutable within a revision.
    del template_revision
    template = get_template(spec)
    states, parameters = scalar_layout(spec, modes)
    prepared_declarations = {item.name: item for item in template.prepared_parameters}
    prepared_scalars = tuple(
        prepared_declarations[name].scalar_name
        for name, _ in (prepared_parameter_layouts or ())
    )
    residual = residual_template or ResidualTemplate(template.state_names)
    fragments = get_fragments(
        spec,
        modes,
        prepared_values=prepared_scalars,
        authoritative_output=backward and residual.save_output,
    )
    checkpoint_state = residual.checkpoint_state
    checkpoint_interval = residual.checkpoint_interval
    if checkpoint_state is not None and (
        template.replay_state is None
        or checkpoint_state != template.checkpoint_state
        or checkpoint_interval != template.checkpoint_interval
    ):
        raise NotImplementedError(
            "Checkpoint replay must be declared by the model definition"
        )
    dynamic_steps = get_runtime(backend).supports_dynamic_steps
    checkpoint_count = (
        max((n_step - 1) // checkpoint_interval, 0) if checkpoint_state else 0
    )
    source = _Source(
        backend,
        n,
        n_step,
        modes,
        precise_exp,
        storage,
        dynamic_steps,
    )
    checkpoint_block = f"(time {source.renderer.division} {checkpoint_interval or 1})"
    source.pointer("dt", 1)
    source.load("dt", "dt", "0")
    source.lines.extend(source.renderer.validate_dt(template.required_dt))
    step_count = "n_steps" if dynamic_steps else str(n_step)
    size = lambda name: n * modes if template.extra_dims(name) else n
    source.pointer("x_seq", None if dynamic_steps else n * n_step)
    if spec.surrogate == "poisson" and spec.poisson.spiking and not backward:
        source.pointer("uniform_seq", None if dynamic_steps else n * n_step)
    for name in template.state_names:
        source.pointer(name + "_init", size(name))
    if parameter_layouts is not None and len(parameter_layouts) != len(
        template.parameter_names
    ):
        raise ValueError("Parameter layout count disagrees with the model template")
    layouts = (
        dict(zip(template.parameter_names, parameter_layouts))
        if parameter_layouts is not None
        else {}
    )
    for name in template.parameter_names:
        source.pointer(name, layouts[name].span if name in layouts else size(name))
    for name in parameters:
        base, _ = _mode_name(name, modes)
        index = (
            _parameter_index(name, layouts[base], modes, source.renderer)
            if base in layouts
            else source.index(name)
        )
        source.load(name, base, index)
    _load_prepared_parameters(source, template, prepared_parameter_layouts)
    if not backward:
        forward_fragments = fragments.forward
        source.pointer("output_seq", None if dynamic_steps else n * n_step)
        for name in template.state_names:
            source.pointer(name + "_final", size(name))
        for name in history_names:
            source.pointer(
                name + "_seq", None if dynamic_steps else size(name) * n_step
            )
        if checkpoint_state is not None and (dynamic_steps or checkpoint_count):
            source.pointer(
                checkpoint_state + "_checkpoint",
                None if dynamic_steps else size(checkpoint_state) * checkpoint_count,
            )
        for name in states:
            base, _ = _mode_name(name, modes)
            source.load(name + "_prev", base + "_init", source.index(name))
        source.lines.append(f"for time in range({step_count}):")
        start = len(source.lines)
        source.load("x", "x_seq", source.index("x", "time"))
        if spec.surrogate == "poisson" and spec.poisson.spiking:
            source.load("uniform", "uniform_seq", source.index("x", "time"))
        source.statements(forward_fragments)
        source.store("output_seq", source.index("output", "time"), "output")
        if gradient_activity:
            source.emit_gradient_activity(spec, time="time")
        if spike_list:
            if n_step != 1:
                raise ValueError("A recurrent spike sink belongs to one producer step")
            source.emit_spike_list("output", spike_list)
        for name in history_names:
            scalars = (
                tuple(f"{name}_{i}" for i in range(modes))
                if template.extra_dims(name)
                else (name,)
            )
            for scalar in scalars:
                value = scalar + "_next" if name in template.state_names else scalar
                source.store(name + "_seq", source.index(scalar, "time"), value)
        if checkpoint_state is not None and (dynamic_steps or checkpoint_count):
            checkpoint_start = len(source.lines)
            source.lines.append(
                f"if (time + 1) % {checkpoint_interval} == 0 and "
                f"time + 1 < {step_count}:"
            )
            scalars = (
                tuple(f"{checkpoint_state}_{i}" for i in range(modes))
                if template.extra_dims(checkpoint_state)
                else (checkpoint_state,)
            )
            for scalar in scalars:
                source.store(
                    checkpoint_state + "_checkpoint",
                    source.index(
                        scalar,
                        f"((time + 1) {source.renderer.division} "
                        f"{checkpoint_interval}) - 1",
                    ),
                    scalar + "_next",
                )
            source.lines[checkpoint_start + 1 :] = [
                "    " + line for line in source.lines[checkpoint_start + 1 :]
            ]
            source.lines.append("END_IF")
        source.statements(f"{name}_prev = {name}_next" for name in states)
        source.lines[start:] = ["    " + line for line in source.lines[start:]]
        source.lines.append("END")
        for name in states:
            base, _ = _mode_name(name, modes)
            source.store(base + "_final", source.index(name), name + "_prev")
    else:
        presence = _normalize_gradient_presence(
            template, history_names, gradient_presence
        )
        targets = _normalize_gradient_targets(template, gradient_targets)
        if presence.output:
            source.pointer("grad_output_seq", None if dynamic_steps else n * n_step)
        if targets.input:
            source.pointer("grad_x_seq", None if dynamic_steps else n * n_step)
        if residual.save_output:
            source.pointer("output_seq", None if dynamic_steps else n * n_step)
        if checkpoint_state is not None and (dynamic_steps or checkpoint_count):
            source.pointer(
                checkpoint_state + "_checkpoint",
                None if dynamic_steps else size(checkpoint_state) * checkpoint_count,
            )
        for name, final_present in zip(template.state_names, presence.final_states):
            if name != checkpoint_state:
                source.pointer(
                    name + "_seq", None if dynamic_steps else size(name) * n_step
                )
            if final_present:
                source.pointer("grad_" + name + "_final", size(name))
            if targets.states[template.state_names.index(name)]:
                source.pointer("grad_" + name + "_init", size(name))
        history_presence = dict(zip(history_names, presence.histories))
        for name, history_present in zip(history_names, presence.histories):
            if history_present:
                source.pointer(
                    "grad_" + name + "_seq",
                    None if dynamic_steps else size(name) * n_step,
                )
        needed_params = set(
            template.parameter_names if parameter_mask is None else parameter_mask
        )
        for name in template.parameter_names:
            if name in needed_params:
                source.pointer("grad_" + name, size(name))
        for name in parameters:
            base, _ = _mode_name(name, modes)
            if base in needed_params:
                source.lines.append(f"sum_grad_{name} = 0.0")
        final_presence = dict(zip(template.state_names, presence.final_states))
        for name in states:
            base, _ = _mode_name(name, modes)
            if final_presence[base]:
                source.load(
                    "grad_s_" + name,
                    "grad_" + base + "_final",
                    source.index(name),
                )
            else:
                source.lines.append("grad_s_" + name + " = 0.0")
        for name in states:
            base, _ = _mode_name(name, modes)
            source.load("initial_" + name, base + "_init", source.index(name))
        source.lines.append(f"for time in range({step_count} - 1, -1, -1):")
        start = len(source.lines)
        source.load("x", "x_seq", source.index("x", "time"))
        if presence.output:
            source.load(
                "grad_output", "grad_output_seq", source.index("output", "time")
            )
        else:
            source.lines.append("grad_output = 0.0")
        for name in states:
            base, _ = _mode_name(name, modes)
            index = source.index(name)
            if base == checkpoint_state:
                source.lines.append(f"{name}_prev = initial_{name}")
                if dynamic_steps:
                    checkpoint_start = len(source.lines)
                    source.lines.append(f"if time >= {checkpoint_interval}:")
                    checkpoint_index = source.index(
                        name, f"maximum({checkpoint_block} - 1, 0)"
                    )
                    source.load(
                        name + "_checkpoint_value",
                        base + "_checkpoint",
                        checkpoint_index,
                    )
                    source.lines.append(f"{name}_prev = {name}_checkpoint_value")
                    source.lines[checkpoint_start + 1 :] = [
                        "    " + line for line in source.lines[checkpoint_start + 1 :]
                    ]
                    source.lines.append("END_IF")
                elif checkpoint_count:
                    checkpoint_index = source.index(
                        name, f"maximum({checkpoint_block} - 1, 0)"
                    )
                    source.load(
                        name + "_checkpoint_value",
                        base + "_checkpoint",
                        checkpoint_index,
                    )
                    source.lines.append(
                        f"{name}_prev = where(time >= {checkpoint_interval}, "
                        f"{name}_checkpoint_value, initial_{name})"
                    )
                replay_start = len(source.lines)
                source.lines.append(
                    f"for replay_offset in range({checkpoint_interval - 1}):"
                )
                replay_time = (
                    f"{checkpoint_block} * {checkpoint_interval} + replay_offset"
                )
                replay_output = f"replay_output_{_mode_name(name, modes)[1]}"
                source.load(
                    replay_output,
                    "output_seq",
                    source.index("output", f"minimum({replay_time}, {step_count} - 1)"),
                )
                mode = _mode_name(name, modes)[1]
                condition = f"{replay_time} < time"
                source.statements(
                    template.replay_state(name, mode, replay_output, condition)
                )
                source.lines[replay_start + 1 :] = [
                    "    " + line for line in source.lines[replay_start + 1 :]
                ]
                source.lines.append("END_REPLAY")
                continue
            history_index = source.index(name, "maximum(time - 1, 0)")
            source.load(name + "_history_value", base + "_seq", history_index)
            source.lines.append(
                f"{name}_prev = where(time > 0, {name}_history_value, initial_{name})"
            )
            if base in history_names:
                if history_presence[base]:
                    source.load(
                        "grad_r_" + name,
                        "grad_" + base + "_seq",
                        source.index(name, "time"),
                    )
                else:
                    source.lines.append("grad_r_" + name + " = 0.0")
                if base not in fragments.deferred_records:
                    source.lines.append(
                        f"grad_s_{name} = grad_s_{name} + grad_r_{name}"
                    )
        for name in template.history_names:
            if name in template.state_names:
                continue
            if name in history_names and history_presence[name]:
                source.load(
                    "grad_r_" + name,
                    "grad_" + name + "_seq",
                    source.index(name, "time"),
                )
            else:
                source.lines.append("grad_r_" + name + " = 0.0")
        if residual.save_output:
            source.load("output", "output_seq", source.index("output", "time"))
        source.statements(fragments.forward)
        # Parameter VJPs are independent leaves of the handwritten reverse
        # stage. Removing unrequested leaves preserves every state adjoint and
        # input derivative needed by a trainable upstream encoder.
        for statement in fragments.backward:
            if isinstance(statement, RecordCotangent):
                if statement.name in history_names:
                    for name in states:
                        if _mode_name(name, modes)[0] == statement.name:
                            source.lines.append(
                                f"grad_s_{name} = grad_s_{name} + grad_r_{name}"
                            )
                continue
            lhs = statement.split(" = ", 1)[0]
            if lhs.startswith("grad_p_"):
                base, _ = _mode_name(lhs[len("grad_p_") :], modes)
                if base not in needed_params:
                    continue
            source.statements((statement,))
        if targets.input:
            source.store("grad_x_seq", source.index("x", "time"), "grad_x")
        for name in parameters:
            base, _ = _mode_name(name, modes)
            if base in needed_params:
                source.lines.append(
                    f"sum_grad_{name} = sum_grad_{name} + grad_p_{name}"
                )
        source.lines[start:] = ["    " + line for line in source.lines[start:]]
        source.lines.append("END")
        state_targets = dict(zip(template.state_names, targets.states))
        for name in states:
            base, _ = _mode_name(name, modes)
            if state_targets[base]:
                source.store(
                    "grad_" + base + "_init",
                    source.index(name),
                    "grad_s_" + name,
                )
        for name in parameters:
            base, _ = _mode_name(name, modes)
            if base in needed_params:
                source.store("grad_" + base, source.index(name), "sum_grad_" + name)
    return source.render(), tuple(source.pointers)


def generate_source(
    backend,
    spec,
    n,
    n_step,
    history_names,
    modes=0,
    backward=False,
    parameter_mask=None,
    parameter_layouts=None,
    gradient_presence=None,
    residual_template=None,
    prepared_parameter_layouts=None,
    gradient_targets=None,
    precise_exp=False,
    storage=(),
    spike_list=(),
    gradient_activity=False,
):
    """Generate inspectable direct DSL source for a static cell sequence.

    The physical timestep is loaded from a runtime tensor and therefore is not
    part of the generated source identity. Normalizing it here also prevents a
    cached source entry from retaining a caller-owned timestep tensor.

    Returns:
        Source text and ordered ``(buffer_name, element_count)`` declarations.
    """
    template = get_template(spec)
    history_names = tuple(history_names)
    parameter_layouts = None if parameter_layouts is None else tuple(parameter_layouts)
    prepared_parameter_layouts = (
        None
        if prepared_parameter_layouts is None
        else tuple(prepared_parameter_layouts)
    )
    if backward:
        if parameter_mask is not None:
            requested = set(parameter_mask)
            parameter_mask = tuple(
                name for name in template.parameter_names if name in requested
            )
        gradient_presence = _normalize_gradient_presence(
            template, history_names, gradient_presence
        )
        gradient_targets = _normalize_gradient_targets(template, gradient_targets)
    else:
        parameter_mask = None
        gradient_presence = None
        gradient_targets = None
    residual_template = residual_template or ResidualTemplate(template.state_names)
    semantic_spec = replace(spec, dt=1.0)
    supports_dynamic_steps = get_runtime(backend).supports_dynamic_steps
    cache_steps = n_step if not supports_dynamic_steps or spike_list else 0
    return _generate_source(
        backend,
        semantic_spec,
        n,
        cache_steps,
        history_names,
        modes,
        backward,
        parameter_mask,
        parameter_layouts,
        gradient_presence,
        model_template_revision(spec.model),
        residual_template,
        prepared_parameter_layouts,
        gradient_targets,
        precise_exp,
        tuple(storage),
        tuple(spike_list),
        gradient_activity,
    )


@functools.lru_cache(maxsize=256)
def _compile_cached(
    backend,
    source,
    enable_fp_fusion=True,
    precise_division=False,
    tilelang_toolchain_stamp=None,
    tilelang_target=None,
):
    """Compile source through the selected backend runtime."""
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm generated kernels before CUDA Graph capture")
    return get_runtime(backend).compile(
        source,
        enable_fp_fusion,
        precise_division,
        tilelang_toolchain_stamp,
        tilelang_target,
    )


def _compiler_version(executable, description):
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"Could not fingerprint TileLang {description} at {executable}"
        ) from error
    return result.stdout + result.stderr


@functools.lru_cache(maxsize=1)
def _tilelang_toolchain_stamp():
    from tilelang.contrib.nvcc import find_cuda_path, get_nvcc_compiler

    nvcc = Path(get_nvcc_compiler()).expanduser()
    cuda_root = Path(find_cuda_path()).expanduser()
    try:
        nvcc = nvcc.resolve(strict=True)
        cuda_root = cuda_root.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"Could not resolve TileLang CUDA toolchain: nvcc={nvcc}, "
            f"CUDA_HOME={cuda_root}"
        ) from error
    ptxas = cuda_root / "bin" / "ptxas"
    libdevice = cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc"
    try:
        ptxas = ptxas.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"Could not fingerprint TileLang ptxas at {ptxas}"
        ) from error
    try:
        libdevice_bytes = libdevice.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"Could not fingerprint TileLang libdevice at {libdevice}"
        ) from error
    identity = "\n".join(
        (
            str(nvcc),
            _compiler_version(nvcc, "NVCC"),
            str(ptxas),
            _compiler_version(ptxas, "ptxas"),
            hashlib.sha256(libdevice_bytes).hexdigest(),
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()


@functools.lru_cache(maxsize=32)
def _tilelang_target(device_index):
    """Resolve the installed backend's target for one physical CUDA device."""
    from tilelang.backend.target import determine_target

    with torch.cuda.device(device_index):
        return str(determine_target("cuda", return_object=True))


def _current_tilelang_target():
    return _tilelang_target(torch.cuda.current_device())


def _device_scope(device):
    """Select a tensor's CUDA device only when it differs from the current
    one."""
    if device is None or device.index == torch.cuda.current_device():
        return nullcontext()
    return torch.cuda.device(device)


def _compile(
    backend, source, enable_fp_fusion=True, precise_division=False, *, device=None
):
    # Source generation is device-independent. Compiled TileLang callables
    # own a concrete target; its compiler still owns executable/device reuse.
    with _device_scope(device):
        runtime = get_runtime(backend)
        toolchain_stamp, target = runtime.compile_context(
            _tilelang_toolchain_stamp, _current_tilelang_target
        )
        return _compile_cached(
            backend,
            source,
            enable_fp_fusion,
            precise_division,
            toolchain_stamp,
            target,
        )


def _launch_kernel(
    runtime,
    kernel,
    args,
    n,
    *,
    steps=None,
    enable_fp_fusion=True,
    debug=None,
    num_warps=None,
    writes=(),
):
    """Launch primitive or fused kernels with the same lane and stream
    contract."""
    if not n:
        return
    with _device_scope(args[0].device):
        runtime.launch(
            kernel,
            args,
            n,
            steps=steps,
            enable_fp_fusion=enable_fp_fusion,
            debug=debug,
            num_warps=num_warps,
            writes=writes,
        )


def _launch(
    backend,
    spec,
    buffers,
    n,
    n_step,
    history_names,
    modes,
    backward=False,
    parameter_mask=None,
    parameter_layouts=None,
    gradient_presence=None,
    enable_fp_fusion=True,
    precise_division=False,
    residual_template=None,
    prepared_parameter_layouts=None,
    gradient_targets=None,
    precise_exp=False,
    spike_list=(),
    gradient_activity=False,
    outputs=(),
):
    if not n:
        return
    runtime = get_runtime(backend)
    dynamic_steps = runtime.supports_dynamic_steps
    strided_names = set(get_template(spec).parameter_names)
    if prepared_parameter_layouts is not None:
        strided_names.update(name for name, _ in prepared_parameter_layouts)
    source, pointers = generate_source(
        backend,
        spec,
        n,
        n_step,
        history_names,
        modes,
        backward,
        parameter_mask,
        parameter_layouts,
        prepared_parameter_layouts=prepared_parameter_layouts,
        gradient_presence=gradient_presence,
        gradient_targets=gradient_targets,
        residual_template=residual_template,
        precise_exp=precise_exp,
        storage=_tensor_storage(
            buffers,
            strided_names,
            ()
            if not dynamic_steps
            else {
                name
                for name in buffers
                if name == "x_seq"
                or name == "uniform_seq"
                or name == "output_seq"
                or name == "gradient_activity"
                or name.endswith("_seq")
                or name.endswith("_checkpoint")
            },
        ),
        spike_list=spike_list,
        gradient_activity=gradient_activity,
    )
    kernel = _compile(
        backend,
        source,
        enable_fp_fusion,
        precise_division,
        device=buffers["x_seq"].device,
    )
    args = tuple(storage_span(buffers[name], size) for name, size in pointers)
    num_warps = None
    if backward and get_template(spec).reverse_parameter_warps:
        targets = _normalize_gradient_targets(get_template(spec), gradient_targets)
        if not targets.input:
            for parameter, warps in get_template(spec).reverse_parameter_warps:
                if parameter_mask is None or parameter in parameter_mask:
                    num_warps = warps
                    break
    _launch_kernel(
        runtime,
        kernel,
        args,
        n,
        steps=n_step if dynamic_steps else None,
        enable_fp_fusion=enable_fp_fusion,
        debug=True,
        num_warps=num_warps,
        writes=tuple(i for i, (name, _) in enumerate(pointers) if name in outputs),
    )


def _dt_buffer(dt, reference):
    """Keep the invocation timestep on device without reading tensor values."""
    if isinstance(dt, torch.Tensor):
        if dt.numel() != 1:
            raise ValueError("Cell timestep must be scalar")
        return dt.to(device=reference.device, dtype=torch.float32).reshape(1)
    # ``full`` fills device storage directly; unlike ``tensor([dt], device=...)``
    # it does not introduce a host-to-device copy during CUDA graph capture.
    return torch.full((1,), dt, device=reference.device, dtype=torch.float32)


def _kernel_arguments(buffers, declarations, strided_names):
    """Bind the declared ABI while preserving broadcast parameter storage."""
    return tuple(
        storage_span(buffers[name], size)
        if name in strided_names
        else buffers[name].reshape(-1)
        for name, size in declarations
    )


def _layout(value, target_shape):
    if len(value.shape) > len(target_shape):
        raise ValueError(
            f"Parameter shape {tuple(value.shape)} cannot broadcast to {target_shape}"
        )
    padding = len(target_shape) - value.ndim
    for dimension, target in zip(value.shape, target_shape[padding:]):
        if dimension not in (1, target):
            raise ValueError(
                "Parameter shape "
                f"{tuple(value.shape)} cannot broadcast to {target_shape}"
            )
    span = (
        0
        if not value.numel()
        else 1
        + sum(
            (dimension - 1) * stride
            for dimension, stride in zip(value.shape, value.stride())
        )
    )
    return _ParameterLayout(
        tuple(value.shape), tuple(value.stride()), tuple(target_shape), span
    )


def _parameter_buffers(parameters):
    """Resolve lazy metadata once per identity while preserving storage
    dtype."""
    converted = {}
    for tensor in parameters:
        if id(tensor) not in converted:
            converted[id(tensor)] = tensor.resolve_neg()
    return tuple(converted[id(tensor)] for tensor in parameters)


def _prepare(x_seq, states, parameters, spec, *, preserve_parameters=False):
    if x_seq.device.type != "cuda":
        raise ValueError("GPU cell backends require CUDA tensors")
    template = get_template(spec)
    shape = x_seq.shape[1:]
    modes = (
        states[template.state_names.index(template.state_extra_dims[0][0])].shape[-1]
        if template.state_extra_dims
        else 0
    )
    target = lambda name: (*shape, modes) if template.extra_dims(name) else shape
    buffers = {"x_seq": x_seq, "dt": _dt_buffer(spec.dt, x_seq)}
    buffers.update(
        {
            name + "_init": tensor.expand(target(name))
            for name, tensor in zip(template.state_names, states)
        }
    )
    parameter_layouts = None
    if preserve_parameters:
        converted = _parameter_buffers(parameters)
        parameter_layouts = tuple(
            _layout(tensor, target(name))
            for name, tensor in zip(template.parameter_names, converted)
        )
        buffers.update(dict(zip(template.parameter_names, converted)))
    else:
        buffers.update(
            {
                name: tensor.expand(target(name))
                for name, tensor in zip(template.parameter_names, parameters)
            }
        )
    prepared = template, shape, modes, target, buffers
    return (*prepared, parameter_layouts) if preserve_parameters else prepared


def _load_prepared_parameters(
    source, template, layouts, *, pointer_prefix="", scalar_prefix="", index="index"
):
    """Bind template-declared computed values in primitive or paired source."""
    declarations = {item.name: item for item in template.prepared_parameters}
    scalars = []
    for name, layout in layouts or ():
        declaration = declarations[name]
        source.pointer(pointer_prefix + name, layout.span)
        offset = _parameter_index(
            declaration.shape_source, layout, source.modes, source.renderer
        )
        offset = substitute_expression(offset, {"index": index})
        source.load(
            scalar_prefix + declaration.scalar_name, pointer_prefix + name, offset
        )
        scalars.append(declaration.scalar_name)
    return tuple(scalars)


def _prepare_parameters(values, template, target):
    if not values:
        return {}, None
    declarations = {item.name: item for item in template.prepared_parameters}
    undeclared = values.keys() - declarations.keys()
    if undeclared:
        raise ValueError(f"Undeclared prepared parameters: {sorted(undeclared)}")
    converted = {name: value.resolve_neg() for name, value in values.items()}
    layouts = tuple(
        (name, _layout(value, target(declarations[name].shape_source)))
        for name, value in converted.items()
    )
    return converted, layouts


def forward(
    backend,
    x_seq,
    states,
    parameters,
    spec,
    history_names,
    *,
    enable_fp_fusion=True,
    precise_division=False,
    residual_template=None,
    prepared_parameters=None,
    uniform_seq=None,
    precise_exp=False,
    spike_list=None,
    gradient_activity=None,
):
    """Execute inference or training forward using exactly requested
    storage."""
    template, shape, modes, target, buffers, parameter_layouts = _prepare(
        x_seq, states, parameters, spec, preserve_parameters=True
    )
    coefficient_buffers, prepared_parameter_layouts = _prepare_parameters(
        prepared_parameters, template, target
    )
    buffers.update(coefficient_buffers)
    if spike_list is not None:
        buffers.update(spike_list.bindings())
    if gradient_activity is not None:
        buffers["gradient_activity"] = gradient_activity
    if spec.surrogate == "poisson" and spec.poisson.spiking:
        if uniform_seq is None:
            raise ValueError("Spiking Poisson kernels require explicit uniforms")
        buffers["uniform_seq"] = uniform_seq
    n_step = x_seq.shape[0]
    residual = residual_template or ResidualTemplate(template.state_names)
    checkpoint_count = (
        max((n_step - 1) // residual.checkpoint_interval, 0)
        if residual.checkpoint_state
        else 0
    )
    n = x_seq[0].numel() if n_step else x_seq.numel() // max(n_step, 1)
    # Empty sequences preserve initial state with the same FP32 policy.
    if not n_step:
        checkpoint = (
            (
                x_seq.new_empty(
                    (0, *target(residual.checkpoint_state)), dtype=torch.float32
                ),
            )
            if residual.checkpoint_state
            else ()
        )
        return (
            torch.empty(x_seq.shape, device=x_seq.device, dtype=torch.float32),
            tuple(buffers[name + "_init"].clone() for name in template.state_names),
            tuple(
                x_seq.new_empty((0, *target(name)), dtype=torch.float32)
                for name in history_names
            )
            + checkpoint,
        )
    buffers["output_seq"] = torch.empty(
        x_seq.shape, device=x_seq.device, dtype=torch.float32
    )
    for name in template.state_names:
        buffers[name + "_final"] = torch.empty(
            target(name), device=x_seq.device, dtype=torch.float32
        )
    for name in history_names:
        buffers[name + "_seq"] = torch.empty(
            (n_step, *target(name)), device=x_seq.device, dtype=torch.float32
        )
    if residual.checkpoint_state is not None:
        buffers[residual.checkpoint_state + "_checkpoint"] = torch.empty(
            (checkpoint_count, *target(residual.checkpoint_state)),
            device=x_seq.device,
            dtype=torch.float32,
        )
    outputs = {
        "output_seq",
        "gradient_activity",
        "spike_indices",
        "spike_cumulative",
        "spike_counts",
        "spike_accumulation",
        *(name + "_final" for name in template.state_names),
        *(name + "_seq" for name in history_names),
    }
    if residual.checkpoint_state is not None:
        outputs.add(residual.checkpoint_state + "_checkpoint")
    _launch(
        backend,
        spec,
        buffers,
        n,
        n_step,
        history_names,
        modes,
        parameter_layouts=parameter_layouts,
        prepared_parameter_layouts=prepared_parameter_layouts,
        enable_fp_fusion=enable_fp_fusion,
        precise_division=precise_division,
        residual_template=residual,
        precise_exp=precise_exp,
        spike_list=() if spike_list is None else spike_list.producer_layout,
        gradient_activity=gradient_activity is not None,
        outputs=outputs,
    )
    return (
        buffers["output_seq"],
        tuple(buffers[name + "_final"] for name in template.state_names),
        tuple(buffers[name + "_seq"] for name in history_names)
        + (
            (buffers[residual.checkpoint_state + "_checkpoint"][:checkpoint_count],)
            if residual.checkpoint_state is not None
            else ()
        ),
    )


def backward(
    backend,
    x_seq,
    states,
    parameters,
    spec,
    history_names,
    histories,
    grad_output,
    grad_final_states,
    grad_histories,
    needs_input_grad=None,
    *,
    enable_fp_fusion=True,
    precise_division=False,
    residual_template=None,
    residual_output=None,
    residuals=(),
    prepared_parameters=None,
    precise_exp=False,
):
    """Reverse a sequence with local charge recomputation and state
    adjoints."""
    template = get_template(spec)
    if len(states) != len(template.state_names) or len(parameters) != len(
        template.parameter_names
    ):
        raise ValueError("State or parameter count disagrees with the model template")
    if len(grad_final_states) != len(template.state_names):
        raise ValueError("Final-state gradient count disagrees with the template")
    if len(grad_histories) != len(history_names):
        raise ValueError("History gradient count disagrees with requested histories")
    needs = needs_input_grad or (True,) * (1 + len(states) + len(parameters))
    if not x_seq.shape[0]:
        grad_states = tuple(
            (
                torch.zeros_like(state)
                if gradient is None
                else gradient.sum_to_size(state.shape).to(state.dtype)
            )
            if needs[1 + index]
            else None
            for index, (state, gradient) in enumerate(
                zip(states, grad_final_states, strict=True)
            )
        )
        parameter_offset = 1 + len(states)
        return (
            torch.empty_like(x_seq) if needs[0] else None,
            grad_states,
            tuple(
                torch.zeros_like(parameter) if needs[parameter_offset + index] else None
                for index, parameter in enumerate(parameters)
            ),
        )
    template, shape, modes, target, buffers, parameter_layouts = _prepare(
        x_seq, states, parameters, spec, preserve_parameters=True
    )
    coefficient_buffers, prepared_parameter_layouts = _prepare_parameters(
        prepared_parameters, template, target
    )
    buffers.update(coefficient_buffers)
    residual = residual_template or ResidualTemplate(template.state_names)
    gradient_presence = _GradientPresence(
        grad_output is not None,
        tuple(grad is not None for grad in grad_final_states),
        tuple(grad is not None for grad in grad_histories),
    )
    gradient_targets = _GradientTargets(
        bool(needs[0]), tuple(bool(value) for value in needs[1 : 1 + len(states)])
    )
    n_step, n = x_seq.shape[0], buffers["x_seq"].numel() // max(x_seq.shape[0], 1)
    if grad_output is not None:
        buffers["grad_output_seq"] = grad_output
    if gradient_targets.input:
        buffers["grad_x_seq"] = torch.empty(
            x_seq.shape, device=x_seq.device, dtype=torch.float32
        )
    for name, history, grad in zip(history_names, histories, grad_histories):
        buffers[name + "_seq"] = history
        if grad is not None:
            buffers["grad_" + name + "_seq"] = grad
    for name, grad in zip(template.state_names, grad_final_states):
        if name + "_seq" not in buffers and name != residual.checkpoint_state:
            raise RuntimeError(f"Training history must include state {name!r}")
        if grad is not None:
            buffers["grad_" + name + "_final"] = grad
        if gradient_targets.states[template.state_names.index(name)]:
            buffers["grad_" + name + "_init"] = torch.empty(
                target(name), device=x_seq.device, dtype=torch.float32
            )
    if residual.save_output:
        if residual_output is None:
            raise RuntimeError("Stochastic reverse requires authoritative output")
        buffers["output_seq"] = residual_output
    if residual.checkpoint_state is not None:
        if len(residuals) != 1:
            raise RuntimeError(
                "Checkpointed GLIF reverse requires output and checkpoint residuals"
            )
        buffers[residual.checkpoint_state + "_checkpoint"] = residuals[0]
    parameter_mask = tuple(
        name
        for i, name in enumerate(template.parameter_names)
        if needs[1 + len(states) + i]
    )
    for name in parameter_mask:
        buffers["grad_" + name] = torch.empty(
            target(name), device=x_seq.device, dtype=torch.float32
        )
    outputs = {
        "grad_x_seq",
        *("grad_" + name for name in parameter_mask),
        *("grad_" + name + "_init" for name in template.state_names),
    }
    _launch(
        backend,
        spec,
        buffers,
        n,
        n_step,
        history_names,
        modes,
        backward=True,
        parameter_mask=parameter_mask,
        parameter_layouts=parameter_layouts,
        gradient_presence=gradient_presence,
        gradient_targets=gradient_targets,
        enable_fp_fusion=enable_fp_fusion,
        precise_division=precise_division,
        residual_template=residual,
        prepared_parameter_layouts=prepared_parameter_layouts,
        precise_exp=precise_exp,
        outputs=outputs,
    )
    grad_states = tuple(
        buffers["grad_" + name + "_init"].sum_to_size(value.shape).to(value.dtype)
        if needs[1 + i]
        else None
        for i, (name, value) in enumerate(zip(template.state_names, states))
    )
    grad_parameters = tuple(
        buffers["grad_" + name].sum_to_size(value.shape).to(value.dtype)
        if name in parameter_mask
        else None
        for name, value in zip(template.parameter_names, parameters)
    )
    return (
        buffers["grad_x_seq"].to(x_seq.dtype) if gradient_targets.input else None,
        grad_states,
        grad_parameters,
    )
