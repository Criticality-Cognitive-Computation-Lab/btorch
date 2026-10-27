"""Render fixed scalar templates into independent sequence kernels.

One lane owns a neuron across time. Connections never enter these
kernels. Parameter VJPs accumulate in lane registers across time and
reduce across broadcast batch axes afterwards; no time-by-edge
intermediate is created.
"""

import ast
import functools
import gc
import hashlib
import linecache
import re
import subprocess
import uuid
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ..contracts import surrogate_support
from ..templates import _MODELS, ResidualTemplate, get_template, model_template_revision
from .fragments import _exp_euler_forward, get_fragments, scalar_layout


@dataclass(frozen=True)
class _ParameterLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    target_shape: tuple[int, ...]
    span: int


@dataclass(frozen=True)
class _TensorStorage:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    span: int
    physical_index: bool = False


def _tensor_storage(buffers, physical_names=()):
    """Describe views requiring typed pointers or noncanonical addressing."""
    return tuple(
        (
            name,
            _TensorStorage(
                tuple(value.shape),
                tuple(value.stride()),
                str(value.dtype).removeprefix("torch."),
                0
                if not value.numel()
                else 1
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(value.shape, value.stride())
                ),
                name in physical_names,
            ),
        )
        for name, value in sorted(buffers.items())
        if value.dtype != torch.float32 or not value.is_contiguous()
    )


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
    return device.type == "cuda" and spec.model in _MODELS


def _mode_name(name, modes):
    for base in ("Iasc", "k", "asc_amps"):
        if modes and name.startswith(base + "_") and name[len(base) + 1 :].isdigit():
            return base, int(name[len(base) + 1 :])
    return name, None


def _parameter_index(scalar, layout, modes, backend):
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
            division = "/" if backend == "cupy" else "//"
            coordinate = "index" if trailing == 1 else f"index {division} {trailing}"
            if target_shape[lane_axis] != 1:
                coordinate = f"({coordinate}) % {target_shape[lane_axis]}"
        terms.append(coordinate if stride == 1 else f"({coordinate}) * {stride}")
    return " + ".join(terms) or "0"


class _Source:
    def __init__(self, backend, n, n_step, modes, precise_exp=False, storage=()):
        self.backend, self.n, self.n_step, self.modes = backend, n, n_step, modes
        self.precise_exp = precise_exp
        self.storage = dict(storage)
        self.pointers = []
        self.lines = []
        self.has_spike_list = False
        self.has_gradient_list = False

    def pointer(self, name, size):
        if name in self.storage:
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
        layout = self.storage.get(pointer)
        if layout is None or layout.physical_index:
            return index
        terms = []
        trailing = 1
        division = "/" if self.backend == "cupy" else "//"
        for size, stride in reversed(tuple(zip(layout.shape, layout.stride))):
            if size > 1 and stride:
                coordinate = f"(({index}) {division} {trailing}) % {size}"
                terms.append(f"({coordinate}) * {stride}")
            trailing *= size
        return " + ".join(reversed(terms)) or "0"

    def load(self, local, pointer, index):
        lane_index = "index" in index
        index = self.storage_index(pointer, index)
        if self.backend == "cupy":
            # Clamp addresses with integer intrinsics; fminf would turn an
            # index into a float and lose precision for large trajectories.
            index = re.sub(r"(?<![\w.])minimum\(", "min(", index)
            index = re.sub(r"(?<![\w.])maximum\(", "max(", index)
        expr = (
            (
                f"tl.load({pointer}_ptr" + ("" if index == "0" else f" + {index}") + ")"
                if "index" not in index
                else f"tl.load({pointer}_ptr + {index}, mask=mask, other=0.0)"
            )
            if self.backend == "triton"
            else f"{pointer}_ptr[{index}]"
        )
        layout = self.storage.get(pointer)
        if layout is not None and layout.dtype != "float32":
            # Storage precision is independent of the existing FP32 scalar
            # equations. Convert in registers, never by copying input tensors.
            expr = (
                f"({expr}).to(tl.float32)"
                if self.backend == "triton"
                else f"float({expr})"
                if self.backend == "cupy"
                else f'T.Cast("float32", {expr})'
            )
        if self.backend == "triton" and lane_index and "index" not in index:
            # An expanded scalar has one storage address but remains one
            # logical value per lane, including loop-carried state adjoints.
            expr = f"tl.broadcast_to({expr}, (BLOCK,))"
        self.lines.append(f"{local} = {expr}")

    def store(self, pointer, index, local):
        index = self.storage_index(pointer, index)
        self.lines.append(
            f"tl.store({pointer}_ptr + {index}, {local}, mask=mask)"
            if self.backend == "triton"
            else f"{pointer}_ptr[{index}] = {local}"
        )

    def statements(self, lines):
        self.lines.extend(lines)

    def emit_spike_list(self, value, layout):
        """Append an optional sparse output sink after shared model math."""
        if not layout:
            return
        if self.backend != "triton":
            raise ValueError("Direct spike-list emission requires the Triton renderer")
        n_source, n_output, capacity = layout
        if n_output % self.n:
            raise ValueError(
                "Producer output must have whole receptor lanes per neuron"
            )
        self.has_spike_list = True
        for name, size in (
            ("spike_indices", capacity),
            ("spike_cumulative", capacity),
            ("spike_counts", capacity // 128),
            ("spike_degree", n_source),
            ("spike_accumulation", n_output),
        ):
            self.pointer(name, size)
        self.lines.append("tl.static_assert(BLOCK == 128)")
        self.lines.append(
            f"_emit_spike_list({value}, index, mask, spike_degree_ptr, "
            "spike_indices_ptr, spike_cumulative_ptr, spike_counts_ptr, "
            f"{n_source}, BLOCK)"
        )
        # Each lane clears exactly the receptor slots whose drive it consumed.
        # Other CTAs may still be reading their own slots: clearing by a flat
        # output index instead would race when there are multiple receptors.
        receptors = n_output // self.n
        for receptor in range(receptors):
            self.lines.append(
                f"tl.store(spike_accumulation_ptr + index * {receptors} + {receptor}, "
                "0.0, mask=mask)"
            )

    def emit_gradient_activity(self, spec, *, prefix="", time=None):
        """Store exact output-adjoint support declared by the model
        template."""
        scalar = get_template(spec).spike_gradient_input
        width = surrogate_support(spec)
        if scalar is None or width is None:
            raise ValueError(
                "Gradient activity requires declared compact spike support"
            )
        self.pointer("gradient_activity", self.n * self.n_step)
        # Keep NaNs active conservatively; they are not known derivative zeros.
        expression = f"where(abs({prefix}{scalar}) >= {width!r}, 0.0, 1.0)"
        layout = self.storage.get("gradient_activity")
        if self.backend == "triton" and layout is not None and layout.dtype == "int32":
            from ....sparse.runtime.activity import GradientList

            capacity = ((self.n + 127) // 128) * 128
            blocks = capacity // 128
            size = GradientList.size(self.n)
            base = "gradient_activity_ptr"
            if time is not None:
                base += f" + {time} * {size}"
            self.has_gradient_list = True
            self.lines.extend(
                [
                    "tl.static_assert(BLOCK == 128)",
                    f"gradient_tape = {base}",
                    f"gradient_active = mask & ~ (abs({prefix}{scalar}) >= {width!r})",
                    "tl.store(gradient_tape + index, gradient_active, mask=mask)",
                    "_emit_active_list(gradient_active, index, "
                    f"gradient_tape + {capacity}, "
                    f"gradient_tape + {3 * capacity}, BLOCK)",
                    "_emit_active_list(gradient_active | "
                    f"(mask & ({prefix}output != 0.0)), "
                    f"index, gradient_tape + {2 * capacity}, "
                    f"gradient_tape + {3 * capacity + blocks}, BLOCK)",
                ]
            )
            return
        self.store("gradient_activity", self.index("output", time), expression)

    def render(self):
        if self.backend == "triton":
            lines = []
            for line in self.lines:
                if line.strip() in ("END_IF", "END_REPLAY"):
                    continue
                if line.endswith(" = 0.0"):
                    line = (
                        line.rsplit(" = ", 1)[0]
                        + " = tl.full((BLOCK,), 0.0, tl.float32)"
                    )
                for old, new in (
                    ("fma", "tl.fma"),
                    ("exp", "libdevice.exp" if self.precise_exp else "tl.exp"),
                    ("expm1", "libdevice.expm1"),
                    ("where", "tl.where"),
                    ("maximum", "tl.maximum"),
                    ("minimum", "tl.minimum"),
                    ("abs", "tl.abs"),
                ):
                    line = re.sub(rf"(?<![\w.]){old}\(", new + "(", line)
                lines.append(line)
            args = ", ".join(name + "_ptr" for name, _ in self.pointers)
            return (
                "import triton\nimport triton.language as tl\n"
                "from triton.language.extra import libdevice\n"
                + (
                    "from btorch.sparse.runtime.kernels_triton_spikelist "
                    "import _emit_active_list\n"
                    if self.has_gradient_list
                    else ""
                )
                + (
                    "from btorch.sparse.runtime.kernels_triton_spikelist "
                    "import _emit_spike_list\n"
                    if self.has_spike_list
                    else ""
                )
                + "@triton.jit\n"
                f"def cell_kernel({args}, BLOCK: tl.constexpr):\n"
                "    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
                f"    mask = index < {self.n}\n"
                + "\n".join("    " + line for line in lines)
                + "\n"
            )
        if self.backend == "cupy":
            declared = {"index", "time", "replay"}
            lines = []
            for line in self.lines:
                indentation = len(line) - len(line.lstrip())
                line = line.strip()
                if line.startswith("for time in"):
                    reverse = "-1, -1" in line
                    line = (
                        f"for (int time = {self.n_step - 1}; time >= 0; --time) {{"
                        if reverse
                        else f"for (int time = 0; time < {self.n_step}; ++time) {{"
                    )
                elif line.startswith("for replay_offset in range("):
                    end = line.removeprefix("for replay_offset in range(").removesuffix(
                        "):"
                    )
                    line = (
                        f"for (int replay_offset = 0; replay_offset < {end}; "
                        "++replay_offset) {"
                    )
                elif line.startswith("if ") and line.endswith(":"):
                    condition = line[3:-1].replace(" and ", " && ").replace("//", "/")
                    line = f"if ({condition}) {{"
                elif line in ("END", "END_IF", "END_REPLAY"):
                    line = "}"
                else:
                    for old, new in (
                        ("fma", "fmaf"),
                        ("exp", "expf"),
                        ("expm1", "expm1f"),
                        ("maximum", "fmaxf"),
                        ("minimum", "fminf"),
                        ("abs", "fabsf"),
                    ):
                        line = re.sub(rf"(?<![\w.]){old}\(", new + "(", line)
                    lhs = line.split(" = ", 1)[0]
                    if lhs.isidentifier() and lhs not in declared:
                        declared.add(lhs)
                        line = "float " + line
                    # Keep constants in FP32, including operations before assignment.
                    line = re.sub(
                        r"(?<![\w.])(\d+\.\d+(?:e[-+]?\d+)?)(?![\w.])", r"\1f", line
                    )
                    line += ";"
                line = line.replace("//", "/")
                lines.append(" " * indentation + line)
            types = {
                "float16": "__half",
                "bfloat16": "__nv_bfloat16",
                "float64": "double",
                "bool": "bool",
                "uint8": "unsigned char",
                "int8": "signed char",
                "int16": "short",
                "int32": "int",
                "int64": "long long",
            }
            args = ", ".join(
                types.get(self.storage[name].dtype, "float") + "* " + name + "_ptr"
                if name in self.storage
                else "float* " + name + "_ptr"
                for name, _ in self.pointers
            )
            return (
                "#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n"
                "__device__ __forceinline__ float where(\n"
                "    bool condition, float x, float y) { return condition ? x : y; }\n"
                f'extern "C" __global__ void cell_kernel({args}) {{\n'
                "  int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
                f"  if (index < {self.n}) {{\n"
                + "\n".join("    " + line for line in lines)
                + "\n  }\n}\n"
            )
        # TileLang mutable register arrays make state and adjoint updates across
        # the serial time loop explicit, avoiding SSA rebinding ambiguities.
        locals_ = sorted(
            {
                line.strip().split(" = ", 1)[0]
                for line in self.lines
                if " = " in line and line.strip().split(" = ", 1)[0].isidentifier()
            }
        )
        lines = []
        for line in self.lines:
            if line.strip() in ("END", "END_IF", "END_REPLAY"):
                continue
            line = line.replace("range(", "T.serial(")
            line = re.sub(
                r"(?<![\w.])fma\(",
                'T.call_pure_extern("float32", "fmaf", ',
                line,
            )
            if self.precise_exp:
                line = re.sub(
                    r"(?<![\w.])exp\(",
                    'T.call_pure_extern("float32", "expf", ',
                    line,
                )
            exp_substitutions = () if self.precise_exp else (("exp", "T.exp"),)
            for old, new in (
                *exp_substitutions,
                ("expm1", "_btorch_expm1"),
                ("where", "T.if_then_else"),
                ("maximum", "T.max"),
                ("minimum", "T.min"),
                ("abs", "T.abs"),
            ):
                line = re.sub(rf"(?<![\w.]){old}\(", new + "(", line)
            for name in locals_:
                line = re.sub(rf"\b{name}\b", name + "[0]", line)
            lines.append(line)
        args = ", ".join(
            f'{name}_ptr: T.Tensor(({size},), "'
            f'{self.storage[name].dtype if name in self.storage else "float32"}")'
            for name, size in self.pointers
        )
        return (
            "import tilelang\nimport tilelang.language as T\n"
            "def _btorch_expm1(value):\n"
            '    return T.call_pure_extern("float32", "expm1f", value)\n'
            "@tilelang.jit\ndef build_kernel():\n"
            f"    @T.prim_func\n    def cell_kernel({args}):\n"
            f"        with T.Kernel({(self.n + 127) // 128}, threads=128) as block:\n"
            + "".join(
                f'            {name} = T.alloc_local((1,), "float32")\n'
                for name in locals_
            )
            + "            index = block * 128 + T.get_thread_binding()\n"
            + f"            if index < {self.n}:\n"
            + "\n".join("                " + line for line in lines)
            + "\n    return cell_kernel\n"
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
        spec.model != "glif3" or checkpoint_state != "Iasc" or checkpoint_interval != 16
    ):
        raise NotImplementedError(
            "Only block-16 GLIF adaptation checkpoints are supported"
        )
    checkpoint_count = (
        max((n_step - 1) // checkpoint_interval, 0) if checkpoint_state else 0
    )
    checkpoint_block = "(time >> 4)"
    source = _Source(backend, n, n_step, modes, precise_exp, storage)
    source.pointer("dt", 1)
    source.load("dt", "dt", "0")
    if backend == "triton":
        source.lines.append(
            "tl.device_assert((index != 0) | ((dt == dt) & (dt > 0.0) & "
            '(dt < float("inf"))), "dt must be finite and positive")'
        )
        if spec.model == "alpha_billeh":
            source.lines.append(
                "tl.device_assert((index != 0) | (dt == 1.0), "
                '"AlphaPSCBilleh requires dt == 1")'
            )
    elif backend == "cupy":
        source.lines.append(
            "if (index == 0 && !(isfinite(dt) && dt > 0.0)) "
            '__assertfail("dt must be finite and positive", __FILE__, __LINE__, '
            "__func__, sizeof(char))"
        )
        if spec.model == "alpha_billeh":
            source.lines.append(
                "if (index == 0 && !(dt == 1.0)) "
                '__assertfail("AlphaPSCBilleh requires dt == 1", __FILE__, '
                "__LINE__, __func__, sizeof(char))"
            )
    else:
        # Use TileLang's stack-aware assertion path, retaining diagnostics.
        # One predicate covers both validity checks without two device asserts.
        source.lines.append(
            "T.device_assert(index != 0 or (T.isfinite(dt) and dt > 0.0), "
            '"dt must be finite and positive")'
        )
        if spec.model == "alpha_billeh":
            source.lines.append(
                "T.device_assert(index != 0 or dt == 1.0, "
                '"AlphaPSCBilleh requires dt == 1")'
            )
    size = lambda name: n * modes if template.extra_dims(name) else n
    source.pointer("x_seq", n * n_step)
    if spec.surrogate == "poisson" and spec.poisson.spiking and not backward:
        source.pointer("uniform_seq", n * n_step)
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
            _parameter_index(name, layouts[base], modes, backend)
            if base in layouts
            else source.index(name)
        )
        source.load(name, base, index)
    _load_prepared_parameters(source, template, prepared_parameter_layouts)
    if not backward:
        forward_fragments = fragments.forward
        source.pointer("output_seq", n * n_step)
        for name in template.state_names:
            source.pointer(name + "_final", size(name))
        for name in history_names:
            source.pointer(name + "_seq", size(name) * n_step)
        if checkpoint_state is not None and checkpoint_count:
            source.pointer(
                checkpoint_state + "_checkpoint",
                size(checkpoint_state) * checkpoint_count,
            )
        for name in states:
            base, _ = _mode_name(name, modes)
            source.load(name + "_prev", base + "_init", source.index(name))
        source.lines.append(f"for time in range({n_step}):")
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
                tuple(f"Iasc_{i}" for i in range(modes)) if name == "Iasc" else (name,)
            )
            for scalar in scalars:
                value = scalar if name.endswith("_charged") else scalar + "_next"
                source.store(name + "_seq", source.index(scalar, "time"), value)
        if checkpoint_state is not None and checkpoint_count:
            checkpoint_start = len(source.lines)
            source.lines.append(
                f"if (time + 1) % {checkpoint_interval} == 0 and time + 1 < {n_step}:"
            )
            scalars = (
                tuple(f"Iasc_{i}" for i in range(modes))
                if checkpoint_state == "Iasc"
                else (checkpoint_state,)
            )
            for scalar in scalars:
                source.store(
                    checkpoint_state + "_checkpoint",
                    source.index(scalar, "((time + 1) >> 4) - 1"),
                    scalar + "_next",
                )
            source.lines[checkpoint_start + 1 :] = [
                "    " + line for line in source.lines[checkpoint_start + 1 :]
            ]
            source.lines.append("END_IF")
        source.statements(f"{name}_prev = {name}_next" for name in states)
        source.lines[start:] = ["    " + line for line in source.lines[start:]]
        if backend == "cupy":
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
            source.pointer("grad_output_seq", n * n_step)
        if targets.input:
            source.pointer("grad_x_seq", n * n_step)
        if residual.save_output:
            source.pointer("output_seq", n * n_step)
        if checkpoint_state is not None and checkpoint_count:
            source.pointer(
                checkpoint_state + "_checkpoint",
                size(checkpoint_state) * checkpoint_count,
            )
        for name, final_present in zip(template.state_names, presence.final_states):
            if name != checkpoint_state:
                source.pointer(name + "_seq", size(name) * n_step)
            if final_present:
                source.pointer("grad_" + name + "_final", size(name))
            if targets.states[template.state_names.index(name)]:
                source.pointer("grad_" + name + "_init", size(name))
        history_presence = dict(zip(history_names, presence.histories))
        for name, history_present in zip(history_names, presence.histories):
            if history_present:
                source.pointer("grad_" + name + "_seq", size(name) * n_step)
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
        source.lines.append(f"for time in range({n_step - 1}, -1, -1):")
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
                if not checkpoint_count:
                    source.load(name + "_prev", base + "_init", index)
                else:
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
                    source.index("output", f"minimum({replay_time}, {n_step - 1})"),
                )
                mode = _mode_name(name, modes)[1]
                condition = f"{replay_time} < time"
                source.lines += [
                    f"replay_L_{mode} = -k_{mode}",
                    f"replay_D_{mode} = replay_L_{mode} * {name}_prev",
                    *_exp_euler_forward(f"replay_{mode}", f"replay_L_{mode}", "dt"),
                    f"{name}_prev = where({condition}, {name}_prev + "
                    f"replay_{mode}_phi * replay_D_{mode}, {name}_prev)",
                    f"{name}_prev = where({condition}, {name}_prev + "
                    f"asc_amps_{mode} * {replay_output}, {name}_prev)",
                ]
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
                source.lines.append(f"grad_s_{name} = grad_s_{name} + grad_r_{name}")
        for name in ("v_charged", "u_charged"):
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
        if backend == "cupy":
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
    return _generate_source(
        backend,
        semantic_spec,
        n,
        n_step,
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


class _TritonPreciseDivision(ast.NodeTransformer):
    """Lower generated floating division without touching integer floor
    division."""

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.Div):
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="tl", ctx=ast.Load()),
                    attr="div_rn",
                    ctx=ast.Load(),
                ),
                args=(node.left, node.right),
                keywords=(),
            ),
            node,
        )


def _triton_precise_division(source):
    tree = _TritonPreciseDivision().visit(ast.parse(source))
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


@functools.lru_cache(maxsize=256)
def _compile_cached(
    backend,
    source,
    enable_fp_fusion=True,
    precise_division=False,
    tilelang_toolchain_stamp=None,
    tilelang_target=None,
):
    if backend == "cupy":
        import cupy

        options = []
        if not enable_fp_fusion:
            options.append("--fmad=false")
        if precise_division:
            options.append("--prec-div=true")
        kernel = cupy.RawKernel(source, "cell_kernel", options=tuple(options))
        kernel.compile()
        return kernel
    if backend == "triton" and precise_division:
        source = _triton_precise_division(source)
    if backend == "tilelang":
        # TileLang's fast-math pass controls broader approximations. Its device
        # compiler flags enter both of its disk-cache keys. A harmless macro
        # prevents binaries built by another CUDA toolkit from satisfying this
        # compile while keeping contraction and division independent.
        if tilelang_toolchain_stamp is None:
            raise RuntimeError("TileLang compilation requires a toolchain stamp")
        flags = [f"-DBTORCH_TILELANG_TOOLCHAIN_{tilelang_toolchain_stamp}=1"]
        if not enable_fp_fusion:
            flags.append("--fmad=false")
        if precise_division:
            flags.append("--prec-div=true")
        if tilelang_target is None:
            raise RuntimeError("TileLang compilation requires a resolved device target")
        source = source.replace(
            "@tilelang.jit",
            f"@tilelang.jit(target={tilelang_target!r}, compile_flags={flags!r})",
            1,
        )
    filename = f"<btorch-{backend}-cell-{uuid.uuid4().hex}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {}
    try:
        exec(compile(source, filename, "exec"), namespace)
        kernel = (
            namespace["cell_kernel"]
            if backend == "triton"
            else namespace["build_kernel"]()
        )
        if backend == "tilelang":
            # TileLang's Python builder leaves temporary frame cycles holding
            # its caller. Release them at cold compilation, before CUDA Graph
            # Trees checkpoint the trajectory's saved-tensor lifetimes.
            gc.collect()
        # Eviction may leave a kernel alive in a caller or the backend cache.
        # Keep inspectable source until that object itself is released.
        weakref.finalize(kernel, linecache.cache.pop, filename, None)
    except BaseException:
        linecache.cache.pop(filename, None)
        raise
    return kernel


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
        toolchain_stamp = _tilelang_toolchain_stamp() if backend == "tilelang" else None
        target = _current_tilelang_target() if backend == "tilelang" else None
        return _compile_cached(
            backend,
            source,
            enable_fp_fusion,
            precise_division,
            toolchain_stamp,
            target,
        )


def _launch_cupy(kernel, args, n):
    """Launch a RawKernel against Torch-owned storage on Torch's stream."""
    import cupy
    import numpy

    with cupy.cuda.Device(args[0].device.index):
        stream = cupy.cuda.Stream.from_external(
            torch.cuda.current_stream(args[0].device)
        )
        with stream:
            # RawKernel copies each NumPy scalar's bytes into the matching
            # pointer parameter. The tensor tuple stays alive through the
            # asynchronous launch, while avoiding per-argument DLPack and
            # CuPy array wrappers.
            pointers = tuple(numpy.uintp(arg.data_ptr()) for arg in args)
            kernel(((n + 127) // 128,), (128,), pointers)


def _launch_kernel(
    backend,
    kernel,
    args,
    n,
    *,
    enable_fp_fusion=True,
    debug=None,
    num_warps=None,
):
    """Launch primitive or fused kernels with the same lane and stream
    contract."""
    if not n:
        return
    with _device_scope(args[0].device):
        if backend == "triton":
            options = {}
            if debug is not None:
                options["debug"] = debug
            if num_warps is not None:
                options["num_warps"] = num_warps
            kernel[((n + 127) // 128,)](
                *args, BLOCK=128, enable_fp_fusion=enable_fp_fusion, **options
            )
        elif backend == "cupy":
            _launch_cupy(kernel, args, n)
        else:
            kernel(*args)


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
):
    if not n:
        return
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
        storage=_tensor_storage(buffers, strided_names),
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
    args = tuple(_storage_span(buffers[name], size) for name, size in pointers)
    num_warps = None
    if backend == "triton" and backward and spec.model == "alpha":
        targets = _normalize_gradient_targets(get_template(spec), gradient_targets)
        if not targets.input and (
            parameter_mask is None or "tau_syn" in parameter_mask
        ):
            # Without the input-VJP store, the tau reverse encounters a
            # load-dependency scheduling cliff with four warps on sm120.
            # Two warps retain independent lanes per thread and recover
            # latency while preserving the omitted-output storage contract.
            num_warps = 2
    _launch_kernel(
        backend,
        kernel,
        args,
        n,
        enable_fp_fusion=enable_fp_fusion,
        debug=True,
        num_warps=num_warps,
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
        _storage_span(buffers[name], size)
        if name in strided_names
        else buffers[name].reshape(-1)
        for name, size in declarations
    )


def _storage_span(value, span):
    """Expose reachable storage without changing logical parameter offsets."""
    return value.as_strided((span,), (1,), value.storage_offset())


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
            declaration.shape_source, layout, source.modes, source.backend
        )
        offset = re.sub(r"\bindex\b", f"({index})", offset)
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
    )
    return (
        buffers["output_seq"],
        tuple(buffers[name + "_final"] for name in template.state_names),
        tuple(buffers[name + "_seq"] for name in history_names)
        + (
            (buffers[residual.checkpoint_state + "_checkpoint"],)
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
