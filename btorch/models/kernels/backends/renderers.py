"""Backend syntax for the shared scalar kernel source program.

The scalar templates emit backend-independent assignments and control
markers. Renderers own pointer access, math intrinsics, dtype conversion
and the kernel wrapper for one target backend.
"""

from __future__ import annotations

import ast
import functools
import gc
import linecache
import re
import uuid
import weakref
from collections import OrderedDict
from dataclasses import dataclass

import torch

from ....sparse.runtime.activity import BLOCK_SIZE


_MATH_NAMES = {
    "abs",
    "exp",
    "expm1",
    "fma",
    "maximum",
    "minimum",
    "where",
}
_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
_LOW_PRECISION_DTYPES = {"float16", "bfloat16"}
_GRID_SYNC_MARKER = "__BTORCH_GRID_SYNC__"
_TILE_ALL_LANES_MARKER = "__BTORCH_TILE_ALL_LANES__"
_BLOCK_EPILOGUE_MARKER = "__BTORCH_BLOCK_EPILOGUE__"


def _split_grid_phases(lines):
    phases = [[]]
    barriers = 0
    for line in lines:
        if line == _GRID_SYNC_MARKER:
            barriers += 1
            phases.append([])
        else:
            phases[-1].append(line)
    return phases, barriers


@dataclass(frozen=True)
class RendererEmission:
    """Emit lane-local statements and an optional block-wide epilogue."""

    lines: tuple[str, ...]
    feature: str | None = None
    epilogue: tuple[str, ...] = ()


class _TritonPreciseDivision(ast.NodeTransformer):
    """Lower only floating division for Triton's precise-division mode."""

    @staticmethod
    def _constant(expression):
        return not any(
            isinstance(node, (ast.Name, ast.Call, ast.Attribute, ast.Subscript))
            for node in ast.walk(expression)
        )

    def visit_BinOp(self, node):  # noqa: N802 - ast API
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.Div):
            return node
        left_constant = self._constant(node.left)
        right_constant = self._constant(node.right)
        if left_constant and right_constant:
            return node
        if left_constant != right_constant:
            # Libdevice requires equal operand types. Match a literal's type
            # to the tensor rather than allowing Triton to default it to FP32.
            literal, tensor = (
                (node.left, node.right) if left_constant else (node.right, node.left)
            )
            literal = ast.Call(
                func=ast.Attribute(ast.Name("tl", ast.Load()), "cast", ast.Load()),
                args=[literal, ast.Attribute(tensor, "dtype", ast.Load())],
                keywords=[],
            )
            if left_constant:
                node.left = literal
            else:
                node.right = literal
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="libdevice", ctx=ast.Load()),
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


def _source_filename(backend, source):
    filename = f"<btorch-{backend}-cell-{uuid.uuid4().hex}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    return filename


def _retain_source(kernel, filename):
    weakref.finalize(kernel, linecache.cache.pop, filename, None)
    return kernel


def _parse_statement(line: str) -> ast.AST:
    """Parse one scalar assignment or expression without textual rewriting."""
    return ast.parse(line.strip(), mode="exec").body[0]


class _StoragePrecision(ast.NodeTransformer):
    """Keep pointwise arithmetic in its operand dtype.

    Division and libdevice transcendentals lack low-precision
    interfaces. Promote only those operations and round their results
    immediately, as Torch's pointwise operations do. Mixed tensor dtypes
    follow Torch promotion.
    """

    def __init__(self, renderer, dtypes, default_dtype):
        self.renderer, self.dtypes, self.default_dtype = renderer, dtypes, default_dtype

    @staticmethod
    def _dtype(node):
        return getattr(node, "_value_dtype", None)

    def _promote(self, nodes):
        types = [self._dtype(node) for node in nodes if self._dtype(node) is not None]
        if not types:
            return None
        result = getattr(torch, types[0])
        for dtype in types[1:]:
            result = torch.promote_types(result, getattr(torch, dtype))
        return str(result).removeprefix("torch.")

    def _cast(self, node, dtype):
        if (
            self.renderer.name == "tilelang"
            and dtype in _LOW_PRECISION_DTYPES
            and self._dtype(node) in ("int64", "uint64")
        ):
            # TileLang's CUDA scalar wrappers have no unambiguous int64
            # constructor. Convert the integer operand, without widening the
            # surrounding floating-point operation or its stored state.
            node = self._cast(node, "float32")
        result = ast.parse(
            self.renderer.cast_load(ast.unparse(node), dtype), mode="eval"
        ).body
        result._value_dtype = dtype
        return result

    def visit_Name(self, node):  # noqa: N802
        node._value_dtype = self.dtypes.get(node.id)
        return node

    def _pointer_dtype(self, node):
        pointers = [
            item.id.removesuffix("_ptr")
            for item in ast.walk(node)
            if isinstance(item, ast.Name) and item.id.endswith("_ptr")
        ]
        if len(pointers) != 1:
            return None
        layout = self.renderer.storage.get(pointers[0])
        return layout.dtype if layout is not None else "float32"

    def visit_Subscript(self, node):  # noqa: N802
        node = self.generic_visit(node)
        node._value_dtype = self._pointer_dtype(node.value)
        return node

    def visit_UnaryOp(self, node):  # noqa: N802
        node = self.generic_visit(node)
        node._value_dtype = self._dtype(node.operand)
        return node

    def visit_BinOp(self, node):  # noqa: N802
        node = self.generic_visit(node)
        dtype = self._promote((node.left, node.right))
        node._value_dtype = dtype
        if dtype in _LOW_PRECISION_DTYPES and isinstance(node.op, (ast.Div, ast.Pow)):
            node.left, node.right = (
                self._cast(node.left, "float32"),
                self._cast(node.right, "float32"),
            )
            return self._cast(node, dtype)
        if dtype in _FLOAT_DTYPES:
            for name in ("left", "right"):
                operand = getattr(node, name)
                if (
                    self._dtype(operand) is not None
                    and self._dtype(operand) != "bool"
                    and self._dtype(operand) != dtype
                ) or (
                    dtype in _LOW_PRECISION_DTYPES
                    and self._dtype(operand) is None
                    and _TritonPreciseDivision._constant(operand)
                ):
                    setattr(node, name, self._cast(operand, dtype))
        return node

    def visit_Call(self, node):  # noqa: N802
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr == "load"
        ):
            node._value_dtype = self._pointer_dtype(node.args[0])
            return node
        if not isinstance(node.func, ast.Name) or node.func.id not in (
            *_MATH_NAMES,
            "pow",
        ):
            return node
        dtype = self._promote(node.args[1:] if node.func.id == "where" else node.args)
        if dtype is None and node.func.id == "where":
            # A spike is a floating tensor in the threshold operand's dtype.
            # Untyped 0/1 branches must not silently make it FP32.
            dtype = getattr(node.args[0], "_comparison_dtype", self.default_dtype)
        node._value_dtype = dtype
        needs_math_abi = node.func.id in ("exp", "expm1", "pow") or (
            node.func.id == "fma" and self.renderer.name == "tilelang"
        )
        if dtype in _LOW_PRECISION_DTYPES and needs_math_abi:
            node.args = [self._cast(argument, "float32") for argument in node.args]
            return self._cast(node, dtype)
        if dtype in _FLOAT_DTYPES:
            start = 1 if node.func.id == "where" else 0
            for index in range(start, len(node.args)):
                argument = node.args[index]
                if (
                    self._dtype(argument) in _FLOAT_DTYPES
                    and self._dtype(argument) != dtype
                ) or (
                    dtype in _LOW_PRECISION_DTYPES
                    and self._dtype(argument) is None
                    and _TritonPreciseDivision._constant(argument)
                ):
                    node.args[index] = self._cast(argument, dtype)
        return node

    def visit_Compare(self, node):  # noqa: N802
        node = self.generic_visit(node)
        node._comparison_dtype = self._promote((node.left, *node.comparators))
        node._value_dtype = "bool"
        return node

    def visit_Assign(self, node):  # noqa: N802
        node = self.generic_visit(node)
        if isinstance(node.targets[0], ast.Name):
            self.dtypes[node.targets[0].id] = self._dtype(
                node.value
            ) or self.dtypes.get(node.targets[0].id, self.default_dtype)
        return node


class _MathTransformer(ast.NodeTransformer):
    def __init__(
        self,
        calls: dict[str, str],
        *,
        floor_division: bool = False,
        fast_math: bool | None = None,
    ):
        self.calls = calls
        self.floor_division = floor_division
        self.fast_math = fast_math

    def visit_Call(self, node):  # noqa: N802 - ast API
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            replacement = self.calls.get(node.func.id)
            if self.fast_math is not None and node.func.id == "exp":
                node.args.append(ast.Constant(int(self.fast_math)))
            if replacement is not None:
                if replacement == "__tilelang_fma__":
                    node.func = ast.parse("T.call_pure_extern", mode="eval").body
                    node.args = [
                        ast.Constant("float32"),
                        ast.Constant("fmaf"),
                        *node.args,
                    ]
                else:
                    node.func = ast.parse(replacement, mode="eval").body
        return node

    def visit_BinOp(self, node):  # noqa: N802 - ast API
        node = self.generic_visit(node)
        if self.floor_division and isinstance(node.op, ast.FloorDiv):
            node.op = ast.Div()
        return node


class _LocalArrayTransformer(ast.NodeTransformer):
    def __init__(self, names: set[str]):
        self.names = names

    def visit_Name(self, node):  # noqa: N802 - ast API
        if node.id in self.names:
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id=node.id, ctx=ast.Load()),
                    slice=ast.Constant(0),
                    ctx=node.ctx,
                ),
                node,
            )
        return node


def transform_statement(
    line: str,
    calls: dict[str, str],
    *,
    floor_division: bool = False,
    fast_math: bool | None = None,
) -> str:
    """Lower scalar math calls in one Python-like source statement."""
    tree = _MathTransformer(
        calls, floor_division=floor_division, fast_math=fast_math
    ).visit(_parse_statement(line))
    return ast.unparse(ast.fix_missing_locations(tree))


def local_statement(line: str, names: set[str]) -> str:
    """Render scalar registers as one-element TileLang local arrays."""
    tree = _LocalArrayTransformer(names).visit(_parse_statement(line))
    return ast.unparse(ast.fix_missing_locations(tree))


def local_expression(expression: str, names: set[str]) -> str:
    """Render local-array references inside a standalone expression."""
    tree = _LocalArrayTransformer(names).visit(ast.parse(expression, mode="eval"))
    return ast.unparse(ast.fix_missing_locations(tree.body))


class SourceRenderer:
    """Backend-independent renderer interface used by ``_Source``."""

    def storage_statements(self, lines, dtypes, default_dtype):
        """Resolve pointwise precision once while preparing scalar
        templates."""
        transform = _StoragePrecision(self, dtypes, default_dtype)
        result = []
        for line in lines:
            if line.strip().endswith(":"):
                result.append(line)
                continue
            indentation = line[: len(line) - len(line.lstrip())]
            tree = transform.visit(_parse_statement(line))
            result.append(indentation + ast.unparse(ast.fix_missing_locations(tree)))
        return tuple(result)

    name = ""
    calls: dict[str, str] = {}
    floor_division = False

    def __init__(self, n, n_step, storage, precise_exp=False, dynamic_steps=False):
        self.n = n
        self.n_step = n_step
        self.storage = storage
        self.precise_exp = precise_exp
        self.dynamic_steps = dynamic_steps

    def storage_index(self, layout, index):
        if layout is None or layout.physical_index:
            return index
        terms = []
        trailing = 1
        axes = tuple(enumerate(zip(layout.shape, layout.stride)))
        for axis, (size, stride) in reversed(axes):
            if layout.dynamic_first and axis == 0:
                coordinate = f"(({index}) {self.division} {trailing})"
            elif size is not None and size > 1 and stride:
                coordinate = f"(({index}) {self.division} {trailing}) % {size}"
            else:
                coordinate = None
            if coordinate is not None and stride:
                terms.append(f"({coordinate}) * {stride}")
            if size is not None:
                trailing *= size
        return " + ".join(reversed(terms)) or "0"

    @property
    def division(self):
        return "/" if self.floor_division else "//"

    def load(self, pointer, index, layout, *, lane_index, dtype="float32"):
        expression = f"{pointer}_ptr[{index}]"
        if (
            dtype in _FLOAT_DTYPES
            and (layout.dtype if layout is not None else "float32") != dtype
        ):
            expression = self.cast_load(expression, dtype)
        return expression

    def cast_load(self, expression, dtype):
        return expression

    def store(self, pointer, index, local):
        return f"{pointer}_ptr[{index}] = {local}"

    def emit_spike_list(self, value, n, layout):
        raise ValueError(
            f"Direct spike-list emission is not supported by {self.name!r}"
        )

    def validate_dt(self, required_dt=None):
        """Emit the backend assertion for runtime timestep validation."""
        raise NotImplementedError(f"{self.name} does not define timestep validation")

    def emit_gradient_activity(self, scalar, width, prefix, time, n, n_step, layout):
        expression = f"where(abs({prefix}{scalar}) >= {width!r}, 0.0, 1.0)"
        index = "index" if time is None else f"({time}) * {n} + index"
        return RendererEmission((self.store("gradient_activity", index, expression),))

    def _math(self, line):
        stripped = line.strip()
        if stripped.startswith(("for ", "if ", "while ")):
            return stripped
        calls = dict(self.calls)
        return transform_statement(
            line,
            calls,
            floor_division=self.floor_division,
            fast_math=not self.precise_exp,
        )

    def render(
        self,
        lines,
        pointers,
        has_spike_list,
        has_gradient_list,
        epilogue=(),
        *,
        scalar_dtype="float32",
        local_dtypes=None,
        grid_barriers=0,
    ):
        raise NotImplementedError


class TritonRenderer(SourceRenderer):
    name = "triton"
    calls = {
        "abs": "tl.abs",
        "exp": "_btorch_exp",
        "expm1": "_btorch_expm1",
        "fma": "tl.fma",
        "maximum": "tl.maximum",
        "minimum": "tl.minimum",
        "pow": "libdevice.pow",
        "where": "tl.where",
    }

    def validate_dt(self, required_dt=None):
        lines = [
            "tl.device_assert((index != 0) | ((dt == dt) & (dt > 0.0) & "
            '(dt < float("inf"))), "dt must be finite and positive")'
        ]
        if required_dt is not None:
            lines.append(
                f"tl.device_assert((index != 0) | (dt == {required_dt!r}), "
                f'"requires dt == {required_dt:g}")'
            )
        return tuple(lines)

    def load(self, pointer, index, layout, *, lane_index, dtype="float32"):
        value_type = layout.dtype if layout is not None else dtype
        other = "0" if value_type.startswith(("int", "uint", "bool")) else "0.0"
        expression = (
            f"tl.load({pointer}_ptr)"
            if index == "0"
            else f"tl.load({pointer}_ptr + {index}, mask=mask, other={other})"
            if "index" in index
            else f"tl.load({pointer}_ptr + {index})"
        )
        if (
            dtype in _FLOAT_DTYPES
            and (layout.dtype if layout is not None else "float32") != dtype
        ):
            expression = self.cast_load(expression, dtype)
        if lane_index and "index" not in index:
            expression = f"tl.broadcast_to({expression}, (BLOCK,))"
        return expression

    def cast_load(self, expression, dtype):
        return f"tl.cast({expression}, tl.{dtype})"

    def store(self, pointer, index, local):
        return f"tl.store({pointer}_ptr + {index}, {local}, mask=mask)"

    def emit_spike_list(self, value, n, layout):
        n_source, n_output, capacity = layout
        if n_output % n:
            raise ValueError(
                "Producer output must have whole receptor lanes per neuron"
            )
        receptors = n_output // n
        lines = [
            "tl.static_assert(BLOCK == 128)",
            f"_emit_spike_list({value}, index, mask, spike_degree_ptr, "
            "spike_indices_ptr, spike_cumulative_ptr, spike_counts_ptr, "
            f"{n_source}, BLOCK)",
        ]
        lines.extend(
            f"tl.store(spike_accumulation_ptr + index * {receptors} + {receptor}, "
            "0.0, mask=mask)"
            for receptor in range(receptors)
        )
        return RendererEmission(tuple(lines), "spike_list")

    def emit_gradient_activity(self, scalar, width, prefix, time, n, n_step, layout):
        if layout is not None and layout.dtype == "int32":
            from ....sparse.runtime.activity import GradientList

            capacity = ((n + 127) // 128) * 128
            blocks = capacity // 128
            size = GradientList.size(n)
            base = "gradient_activity_ptr"
            if time is not None:
                base += f" + {time} * {size}"
            return RendererEmission(
                (
                    "tl.static_assert(BLOCK == 128)",
                    f"gradient_tape = {base}",
                    f"gradient_active = mask & ~ (abs({prefix}{scalar}) >= {width!r})",
                    "tl.store(gradient_tape + index, gradient_active, mask=mask)",
                    "_emit_active_list(gradient_active, index, "
                    f"gradient_tape + {capacity}, "
                    f"gradient_tape + {3 * capacity}, BLOCK)",
                    "_emit_active_list(gradient_active | "
                    f"(mask & ({prefix}output != 0.0)), index, "
                    f"gradient_tape + {2 * capacity}, "
                    f"gradient_tape + {3 * capacity + blocks}, BLOCK)",
                ),
                "gradient_list",
            )
        return super().emit_gradient_activity(
            scalar, width, prefix, time, n, n_step, layout
        )

    def render(
        self,
        lines,
        pointers,
        has_spike_list,
        has_gradient_list,
        epilogue=(),
        *,
        scalar_dtype="float32",
        local_dtypes=None,
        grid_barriers=0,
    ):
        local_dtypes = local_dtypes or {}
        phases, marker_count = _split_grid_phases(lines)
        if marker_count != grid_barriers:
            raise ValueError(
                "Grid synchronization marker count disagrees with renderer state"
            )
        rendered = []
        for phase_index, phase in enumerate(phases):
            for raw in phase:
                indentation = raw[: len(raw) - len(raw.lstrip())]
                line = raw.strip()
                if line in ("END", "END_IF", "END_REPLAY"):
                    continue
                if self.dynamic_steps and line.startswith("for time in range("):
                    reverse = "-1, -1" in line
                    if reverse:
                        rendered.append(
                            indentation + "for reverse_time in tl.range(0, n_steps):"
                        )
                        rendered.append(
                            indentation + "    time = n_steps - 1 - reverse_time"
                        )
                    else:
                        rendered.append(
                            indentation + "for time in tl.range(0, n_steps):"
                        )
                    continue
                if line.rstrip().endswith(" = 0.0"):
                    statement = _parse_statement(line)
                    if isinstance(statement, ast.Assign):
                        dtype = local_dtypes.get(statement.targets[0].id, scalar_dtype)
                        line = (
                            f"{ast.unparse(statement.targets[0])} = "
                            f"tl.full((BLOCK,), 0.0, tl.{dtype})"
                        )
                else:
                    line = self._math(line)
                rendered.append(indentation + line)
            if phase_index < marker_count:
                rendered.append(
                    "_grid_sync(grid_barrier_ptr, grid_epoch_ptr, grid_epoch, "
                    f"PHASE_INDEX={phase_index + 1}, N_PHASES={marker_count}, "
                    f"N_CTAS={(self.n + 127) // 128}, BLOCK=BLOCK)"
                )
        args = ", ".join(name + "_ptr" for name, _ in pointers)
        if self.dynamic_steps:
            args += ", n_steps"
        imports = (
            "import triton\nimport triton.language as tl\n"
            "from triton.language.extra import libdevice\n"
            "from btorch.models.kernels.backends.math_triton "
            "import exp as _btorch_exp, expm1 as _btorch_expm1\n"
        )
        if marker_count:
            imports += (
                "from btorch.models.kernels.backends.synchronization "
                "import _grid_sync\n"
            )
        if has_gradient_list:
            imports += (
                "from btorch.sparse.runtime.kernels_triton_spikelist "
                "import _emit_active_list\n"
            )
        if has_spike_list:
            imports += (
                "from btorch.sparse.runtime.kernels_triton_spikelist "
                "import _emit_spike_list\n"
            )
        return (
            imports
            + "@triton.jit("
            + ("do_not_specialize=['n_steps'], " if self.dynamic_steps else "")
            + "do_not_specialize_on_alignment="
            + repr([name + "_ptr" for name, _ in pointers])
            + ")\n"
            + f"def cell_kernel({args}, BLOCK: tl.constexpr):\n"
            + "    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
            + f"    mask = index < {self.n}\n"
            + ("    grid_epoch = tl.load(grid_epoch_ptr)\n" if marker_count else "")
            + "\n".join("    " + line for line in rendered)
            + "\n"
        )


class CuPyRenderer(SourceRenderer):
    name = "cupy"
    floor_division = True
    calls = {
        "abs": "fabsf",
        "exp": "_btorch_exp",
        "expm1": "_btorch_expm1",
        "fma": "fmaf",
        "maximum": "fmaxf",
        "minimum": "fminf",
        "where": "where",
    }

    def storage_index(self, layout, index):
        # Pointer offsets are integer arithmetic. Floating fmaxf/fminf would
        # make a valid indexed load fail CUDA compilation.
        expression = super().storage_index(layout, index)
        return transform_statement(
            expression, {"maximum": "max", "minimum": "min"}, floor_division=True
        )

    def validate_dt(self, required_dt=None):
        lines = [
            "if (index == 0 && !(isfinite(dt) && dt > 0.0)) "
            '__assertfail("dt must be finite and positive", __FILE__, __LINE__, '
            "__func__, sizeof(char))"
        ]
        if required_dt is not None:
            lines.append(
                "if (index == 0 && !(dt == "
                f"{required_dt!r})) "
                f'__assertfail("requires dt == {required_dt:g}", __FILE__, __LINE__, '
                "__func__, sizeof(char))"
            )
        return tuple(lines)

    def emit_gradient_activity(self, scalar, width, prefix, time, n, n_step, layout):
        expression = f"where(abs({prefix}{scalar}) >= {width!r}, 0.0, 1.0)"
        index = "index" if time is None else f"({time}) * {n} + index"
        return RendererEmission((f"gradient_activity_ptr[{index}] = {expression}",))

    def cast_load(self, expression, dtype):
        return f"float({expression})"

    def render(
        self,
        lines,
        pointers,
        has_spike_list,
        has_gradient_list,
        epilogue=(),
        *,
        scalar_dtype="float32",
        local_dtypes=None,
        grid_barriers=0,
    ):
        del scalar_dtype, local_dtypes
        if grid_barriers:
            raise NotImplementedError(
                "Grid synchronization is unavailable for the CuPy renderer"
            )
        declared = {"index", "time", "replay"}
        rendered = []
        for raw in lines:
            indentation = len(raw) - len(raw.lstrip())
            line = raw.strip()
            if line.startswith("for time in"):
                reverse = "-1, -1" in line
                if self.dynamic_steps:
                    line = (
                        "for (int time = n_steps - 1; time >= 0; --time) {"
                        if reverse
                        else "for (int time = 0; time < n_steps; ++time) {"
                    )
                else:
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
                    "for (int replay_offset = 0; "
                    f"replay_offset < {end}; ++replay_offset) {{"
                )
            elif line.startswith("if ") and line.endswith(":"):
                condition = (
                    transform_statement(line[3:-1], self.calls, floor_division=True)
                    .replace(" and ", " && ")
                    .replace(" or ", " || ")
                )
                line = f"if ({condition}) {{"
            elif line in ("END", "END_IF", "END_REPLAY"):
                line = "}"
            elif line.startswith("if (index == 0 &&"):
                # Runtime validation is already a complete CUDA statement.
                # It is intentionally kept outside the Python scalar DSL.
                line += ";"
            else:
                line = self._math(line)
                statement = _parse_statement(line)
                if isinstance(statement, ast.Assign) and isinstance(
                    statement.targets[0], ast.Name
                ):
                    name = statement.targets[0].id
                    if name not in declared:
                        declared.add(name)
                        line = ("int " if name == "n_steps" else "float ") + line
                line = re.sub(
                    r"(?<![\w.])(\d+\.\d+(?:e[-+]?\d+)?)(?![\w.])",
                    r"\1f",
                    line,
                )
                line += ";"
            line = line.replace("//", "/")
            rendered.append(" " * indentation + line)
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
            for name, _ in pointers
        )
        if self.dynamic_steps:
            args += ", int n_steps"
        return (
            "#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n"
            'extern "C" __device__ float __nv_expf(float);\n'
            'extern "C" __device__ float __nv_expm1f(float);\n'
            "__device__ __forceinline__ float _btorch_exp(float x, bool fast_math) "
            "{ return fast_math ? __expf(x) : __nv_expf(x); }\n"
            "__device__ __forceinline__ float _btorch_expm1(float x) "
            "{ return __nv_expm1f(x); }\n"
            "__device__ __forceinline__ float where(bool condition, float x, float y) "
            "{ return condition ? x : y; }\n"
            f'extern "C" __global__ void cell_kernel({args}) {{\n'
            "  int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
            f"  if (index < {self.n}) {{\n"
            + "\n".join("    " + line for line in rendered)
            + "\n  }\n}\n"
        )


class TileLangRenderer(SourceRenderer):
    name = "tilelang"
    calls = {
        "abs": "T.abs",
        "exp": "_btorch_exp",
        "expm1": "_btorch_expm1",
        "fma": "__tilelang_fma__",
        "maximum": "T.max",
        "minimum": "T.min",
        "pow": "T.pow",
        "where": "T.if_then_else",
    }

    def emit_spike_list(self, value, n, layout):
        n_source, n_output, capacity = layout
        if n_output % n:
            raise ValueError(
                "Producer output must have whole receptor lanes per neuron"
            )
        receptors = n_output // n
        return RendererEmission(
            (
                "spike_selected[T.get_thread_binding()] = "
                f'T.Cast("int32", {value} != 0.0)',
            ),
            "spike_list",
            (
                "_emit_spike_list(spike_selected, spike_degree_ptr, "
                "spike_indices_ptr, spike_cumulative_ptr, spike_counts_ptr, "
                f"block, {n_source}, {BLOCK_SIZE})",
                f"for spike_clear_lane in T.Parallel({BLOCK_SIZE}):",
                f"    if block * {BLOCK_SIZE} + spike_clear_lane < {n}:",
                *(
                    f"        spike_accumulation_ptr[(block * {BLOCK_SIZE} + "
                    f"spike_clear_lane) * "
                    f"{receptors} + {receptor}] = 0.0"
                    for receptor in range(receptors)
                ),
            ),
        )

    def validate_dt(self, required_dt=None):
        layout = self.storage.get("dt")
        finite_operand = (
            "T.Cast('float32', dt)"
            if layout is not None and layout.dtype == "bfloat16"
            else "dt"
        )
        lines = [
            "T.device_assert(index != 0 or "
            f"(T.isfinite({finite_operand}) and dt > 0.0), "
            '"dt must be finite and positive")'
        ]
        if required_dt is not None:
            lines.append(
                f"T.device_assert(index != 0 or dt == {required_dt!r}, "
                f'"requires dt == {required_dt:g}")'
            )
        return tuple(lines)

    def _math(self, line, scalar_dtype="float32"):
        if line.strip() == "else:" or line.strip().startswith(
            ("for ", "if ", "elif ", "while ")
        ):
            return line.strip()
        calls = dict(self.calls)
        tree = _MathTransformer(calls, fast_math=not self.precise_exp).visit(
            _parse_statement(line)
        )

        # FP64 explicit FMA keeps the matching CUDA ABI; exponentials are
        # handled by the shared dtype-aware helpers imported below.
        class _ExpandFma(ast.NodeTransformer):
            def visit_Call(self, node):  # noqa: N802
                node = self.generic_visit(node)
                if (
                    scalar_dtype == "float64"
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "call_pure_extern"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "fmaf"
                ):
                    node.args[:2] = [ast.Constant("float64"), ast.Constant("fma")]
                return node

        tree = _ExpandFma().visit(tree)
        return ast.unparse(ast.fix_missing_locations(tree))

    def cast_load(self, expression, dtype):
        return f'T.Cast("{dtype}", {expression})'

    def render(
        self,
        lines,
        pointers,
        has_spike_list,
        has_gradient_list,
        epilogue=(),
        *,
        scalar_dtype="float32",
        local_dtypes=None,
        grid_barriers=0,
    ):
        local_dtypes = local_dtypes or {}
        phases, marker_count = _split_grid_phases(lines)
        if marker_count != grid_barriers:
            raise ValueError(
                "Grid synchronization marker count disagrees with renderer state"
            )
        assignments = tuple(
            statement
            for line in lines
            if " = " in line and not line.strip().startswith("for ")
            for statement in (_parse_statement(line.strip()),)
            if isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], (ast.Name, ast.Tuple))
        )
        shared = {
            statement.targets[0].id: ast.unparse(statement)
            for statement in assignments
            if isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == "T"
            and statement.value.func.attr == "alloc_shared"
        }
        locals_ = sorted(
            {
                node.id
                for statement in assignments
                for node in ast.walk(statement.targets[0])
                if isinstance(node, ast.Name)
            }
            - shared.keys()
        )
        fixed_threads = (
            has_spike_list or bool(grid_barriers) or _TILE_ALL_LANES_MARKER in lines
        )
        local_names = set(locals_)
        rendered_phases = []
        for phase_index, phase in enumerate(phases):
            # A producer may publish its block-wide queue before a grid
            # barrier, rather than at kernel exit. These statements already
            # use physical lanes and must stay outside the neuron mask.
            block_epilogue = []
            if _BLOCK_EPILOGUE_MARKER in phase:
                boundary = phase.index(_BLOCK_EPILOGUE_MARKER)
                block_epilogue = phase[boundary + 1 :]
                phase = phase[:boundary]
            phase_rendered = []
            all_lanes = False
            for raw in phase:
                if raw == _TILE_ALL_LANES_MARKER:
                    all_lanes = True
                    continue
                indentation = raw[: len(raw) - len(raw.lstrip())]
                line = raw.strip()
                if " = T.alloc_shared(" in line:
                    continue
                if line in ("END", "END_IF", "END_REPLAY"):
                    continue
                line = line.replace("range(", "T.serial(")
                line = self._math(line, scalar_dtype)
                if line.startswith("for "):
                    target, iterable = line[4:].split(" in ", 1)
                    target = target.strip()
                    iterable = local_expression(
                        iterable.rstrip(":"), local_names - {target}
                    )
                    line = f"for {target} in {iterable}:"
                elif line.startswith("if "):
                    condition = local_expression(line[3:].rstrip(":"), local_names)
                    line = f"if {condition}:"
                elif line.startswith("elif "):
                    condition = local_expression(line[5:].rstrip(":"), local_names)
                    line = f"elif {condition}:"
                elif line.startswith("while "):
                    condition = local_expression(line[6:].rstrip(":"), local_names)
                    line = f"while {condition}:"
                elif line != "else:":
                    line = local_statement(line, local_names)
                phase_rendered.append(indentation + line)
            rendered_phases.append((phase_rendered, all_lanes, block_epilogue))
        args = ", ".join(
            f'{name}_ptr: T.Tensor(({size},), "'
            f'{self.storage[name].dtype if name in self.storage else "float32"}")'
            for name, size in pointers
        )
        return (
            "import tilelang\nimport tilelang.language as T\n"
            + f"TUNABLE_THREADS = {not fixed_threads!r}\n"
            + (
                "from btorch.sparse.runtime.tilelang_csr import emit_spike_list "
                "as _emit_spike_list\n"
                if has_spike_list
                else ""
            )
            + "from btorch.models.kernels.backends.math_tilelang "
            "import exp as _btorch_exp, expm1 as _btorch_expm1\n"
            + (
                "@tilelang.jit\ndef build_kernel():\n"
                if fixed_threads
                else "@tilelang.jit\ndef build_kernel(threads=128):\n"
            )
            + f"    @T.prim_func\n    def cell_kernel({args}):\n"
            + (
                f"        with T.Kernel({-(-self.n // BLOCK_SIZE)}, "
                f"threads={BLOCK_SIZE}) as block:\n"
                if fixed_threads
                else f"        with T.Kernel(T.ceildiv({self.n}, threads), "
                "threads=threads) as block:\n"
            )
            + "".join("            " + line + "\n" for line in shared.values())
            + "".join(
                f"            {name} = T.alloc_local((1,), "
                f'"{local_dtypes.get(name, scalar_dtype)}")\n'
                for name in locals_
            )
            + (
                "            spike_selected = "
                f'T.alloc_fragment(({BLOCK_SIZE},), "int32")\n'
                "            T.annotate_layout({spike_selected: tilelang.Fragment(\n"
                f"                ({BLOCK_SIZE},), forward_thread_fn=lambda i: i, "
                "forward_index_fn=lambda i: 0)})\n"
                "            T.clear(spike_selected)\n"
                if has_spike_list
                else ""
            )
            + (
                f"            index = block * {BLOCK_SIZE} + T.get_thread_binding()\n"
                if fixed_threads
                else "            index = block * threads + T.get_thread_binding()\n"
            )
            + "".join(
                (
                    ("\n".join("            " + line for line in phase) + "\n")
                    if all_lanes
                    else (
                        f"            if index < {self.n}:\n"
                        + "\n".join("                " + line for line in phase)
                        + "\n"
                    )
                )
                + "".join("            " + line + "\n" for line in block_epilogue)
                + ("            T.sync_grid()\n" if i < marker_count else "")
                for i, (phase, all_lanes, block_epilogue) in enumerate(rendered_phases)
            )
            + "\n".join("            " + line for line in epilogue)
            + "\n    return cell_kernel\n"
        )


class RendererRuntime:
    """Compilation and launch operations for one source backend."""

    name = ""
    supports_dynamic_steps = False
    supports_pointer_offsets = False
    supports_spike_list = False

    def __init__(self):
        # Recipes contain only executable objects and immutable launch
        # metadata. Invocation tensors are deliberately excluded so a
        # primitive call cannot retain an autograd graph or a previous output.
        self._recipes = OrderedDict()

    @staticmethod
    def _argument_contract(args):
        layouts = []
        for value in args:
            # Source generation already specializes the original logical
            # shapes/strides. These launch arguments are storage spans, so
            # only pointer ABI dtype/device affect the recipe; sequence length
            # and storage addresses must remain dynamic for Triton/CuPy.
            layouts.append((value.dtype, value.device))
        return tuple(layouts)

    def _recipe_key(
        self, kernel, args, n, steps, enable_fp_fusion, debug, num_warps, writes
    ):
        contract = self._argument_contract(args)
        return (
            id(kernel),
            contract,
            n,
            steps is not None,
            enable_fp_fusion,
            debug,
            num_warps,
            tuple(writes),
        )

    def _remember_recipe(self, key, recipe):
        self._recipes[key] = recipe
        self._recipes.move_to_end(key)
        while len(self._recipes) > 128:
            self._recipes.popitem(last=False)
        return recipe

    def prepare_recipe(
        self, kernel, args, n, *, steps, enable_fp_fusion, debug, num_warps, writes=()
    ):
        """Prepare a tensor-free launch recipe for repeated primitive calls."""
        del kernel, args, n, steps, enable_fp_fusion, debug, num_warps, writes
        raise NotImplementedError

    def get_recipe(
        self,
        kernel,
        args,
        n,
        *,
        steps=None,
        enable_fp_fusion,
        debug,
        num_warps,
        writes=(),
    ):
        key = self._recipe_key(
            kernel,
            args,
            n,
            steps,
            enable_fp_fusion,
            debug,
            num_warps,
            writes,
        )
        recipe = self._recipes.get(key)
        if recipe is None:
            recipe = self._remember_recipe(
                key,
                self.prepare_recipe(
                    kernel,
                    args,
                    n,
                    steps=steps,
                    enable_fp_fusion=enable_fp_fusion,
                    debug=debug,
                    num_warps=num_warps,
                    writes=writes,
                ),
            )
        return recipe

    def launch_prepared(self, kernel, args, n, **options):
        self.get_recipe(kernel, args, n, **options)(args, options.get("steps"))

    def argument(self, value, span):
        """Expose the pointer ABI without changing a tensor's storage."""
        del span
        return value

    def bind_submission(self, launch):
        """Bind an invocation's repeated submissions to its active stream."""
        return launch.launch, launch.constants, None

    def bind_rows(self, value, span):
        """Prepare a sequence's argument views for a tensor-based ABI."""
        return tuple(self.argument(row, span) for row in value.unbind(0))

    def prepare_launch(
        self,
        kernel,
        args,
        n,
        *,
        changing,
        enable_fp_fusion,
        writes=(),
        read_writes=(),
        launch_cooperative_grid=False,
    ):
        """Bind launch policy for an invocation with stable argument
        layouts."""
        if launch_cooperative_grid:
            raise NotImplementedError("This renderer has no cooperative launch binding")
        del args, changing, writes, read_writes
        return functools.partial(
            self.launch_prepared,
            kernel,
            n=n,
            enable_fp_fusion=enable_fp_fusion,
            debug=None,
            num_warps=None,
        )

    def compile_context(self, toolchain_stamp, target):
        del toolchain_stamp, target
        return None, None

    def compile(self, source, enable_fp_fusion, precise_division, stamp, target):
        raise NotImplementedError


class TritonRuntime(RendererRuntime):
    supports_spike_list = True
    name = "triton"
    supports_dynamic_steps = True
    supports_pointer_offsets = True

    def bind_submission(self, prepared):
        import torch

        stream = torch.cuda.current_stream(prepared.owners[0].device)
        # Retain the stream separately; partial forwards directly to Triton's
        # public grid callable, including its profiling hooks.
        return (
            functools.partial(prepared.launch, stream=stream.cuda_stream),
            prepared.constants,
            stream,
        )

    def prepare_recipe(
        self, kernel, args, n, *, steps, enable_fp_fusion, debug, num_warps, writes=()
    ):
        from ....utils.triton import prepare_launch_recipe

        grid = ((n + 127) // 128,)
        options = {}
        if debug is not None:
            options["debug"] = debug
        if num_warps is not None:
            options["num_warps"] = num_warps
        call_args = (*args, steps) if steps is not None else args
        options = {
            **options,
            "enable_fp_fusion": enable_fp_fusion,
        }
        launch, constexprs = prepare_launch_recipe(
            kernel,
            call_args,
            grid,
            (("BLOCK", 128),),
            writes=writes,
            work=n,
            runtime_step_index=len(args) if steps is not None else None,
            **options,
        )

        def invoke(current_args, current_steps, _kernel=kernel):
            del _kernel
            call_args = (
                (*current_args, current_steps)
                if current_steps is not None
                else current_args
            )
            launch(*call_args, *constexprs)

        return invoke

    def prepare_launch(
        self,
        kernel,
        args,
        n,
        *,
        changing,
        enable_fp_fusion,
        writes=(),
        read_writes=(),
        launch_cooperative_grid=False,
    ):
        # Generated cells do not specialize pointer alignment. Cache executable
        # metadata only; every invocation binds fresh addresses and owners.
        from ....utils.kernel import PreparedLaunch
        from ....utils.triton import prepare_launch

        key = (
            "bound",
            kernel,
            self._argument_contract(args),
            n,
            enable_fp_fusion,
            tuple(writes),
            launch_cooperative_grid,
        )
        del read_writes  # Triton already restores every writable trial buffer.
        recipe = self._recipes.get(key)
        if recipe is None:
            prepared = prepare_launch(
                kernel,
                args,
                ((n + 127) // 128,),
                (("BLOCK", 128),),
                changing=changing,
                writes=writes,
                work=0 if launch_cooperative_grid else n,
                enable_fp_fusion=enable_fp_fusion,
                launch_cooperative_grid=launch_cooperative_grid,
            )
            self._remember_recipe(key, (prepared.launch, prepared.constants))
            return prepared
        launch, constants = recipe
        return PreparedLaunch(launch, list(args), tuple(changing), constants)

    def compile(self, source, enable_fp_fusion, precise_division, stamp, target):
        del enable_fp_fusion, stamp, target
        if precise_division:
            source = _triton_precise_division(source)
        filename = _source_filename(self.name, source)
        namespace = {}
        try:
            exec(compile(source, filename, "exec"), namespace)
            return _retain_source(namespace["cell_kernel"], filename)
        except BaseException:
            linecache.cache.pop(filename, None)
            raise


class CuPyRuntime(RendererRuntime):
    name = "cupy"
    supports_dynamic_steps = True
    supports_pointer_offsets = True

    def bind_submission(self, prepared):
        import cupy
        import torch

        owner = torch.cuda.current_stream(prepared.owners[0].device)
        stream = cupy.cuda.Stream.from_external(owner)
        return (
            functools.partial(prepared.launch, stream=stream),
            prepared.constants,
            (owner, stream),
        )

    def prepare_recipe(
        self, kernel, args, n, *, steps, enable_fp_fusion, debug, num_warps, writes=()
    ):
        del enable_fp_fusion, debug, num_warps
        import cupy
        import numpy
        import torch

        from ....utils.autotune import cupy_block

        device = args[0].device.index
        block = cupy_block(kernel, args, n, writes, steps=steps)

        def invoke(current_args, current_steps):
            stream = cupy.cuda.Stream.from_external(torch.cuda.current_stream(device))
            with cupy.cuda.Device(device), stream:
                pointers = tuple(numpy.uintp(arg.data_ptr()) for arg in current_args)
                call_args = (
                    (*pointers, numpy.int32(current_steps))
                    if current_steps is not None
                    else pointers
                )
                kernel((-(-n // block),), (block,), call_args)

        return invoke

    def prepare_launch(
        self,
        kernel,
        args,
        n,
        *,
        changing,
        enable_fp_fusion,
        writes=(),
        read_writes=(),
        launch_cooperative_grid=False,
    ):
        if launch_cooperative_grid:
            raise NotImplementedError("CuPy does not support cooperative grid launch")
        del enable_fp_fusion, read_writes
        import cupy
        import torch

        from ....utils.autotune import cupy_block
        from ....utils.kernel import PreparedLaunch

        device = args[0].device.index
        block = cupy_block(kernel, args, n, writes)
        grid = (-(-n // block),)
        last_stream = cupy.cuda.Stream.from_external(torch.cuda.current_stream(device))

        def launch(*pointers, stream=None):
            nonlocal last_stream
            # Resolve the caller's stream at submission, including capture.
            # Recurrent bindings supply their invocation-owned stream directly.
            if stream is None:
                current = torch.cuda.current_stream(device)
                if current.cuda_stream != last_stream.ptr:
                    last_stream = cupy.cuda.Stream.from_external(current)
                stream = last_stream
            kernel(grid, (block,), pointers, stream=stream)

        return PreparedLaunch(launch, list(args), tuple(changing), ())

    def compile(self, source, enable_fp_fusion, precise_division, stamp, target):
        del stamp, target
        import cupy

        options = []
        if not enable_fp_fusion:
            options.append("--fmad=false")
        if precise_division:
            options.append("--prec-div=true")
        kernel = cupy.RawKernel(source, "cell_kernel", options=tuple(options))
        kernel.compile()
        return kernel


class TileLangRuntime(RendererRuntime):
    supports_spike_list = True
    name = "tilelang"

    def __init__(self):
        super().__init__()
        self.factories = weakref.WeakKeyDictionary()

    def prepare_recipe(
        self, kernel, args, n, *, steps, enable_fp_fusion, debug, num_warps, writes=()
    ):
        del steps, enable_fp_fusion, debug, num_warps
        from ....utils.autotune import outputs_are_disjoint, tilelang_kernel

        factory = self.factories.get(kernel)
        executable = kernel
        if factory is not None:
            executable = tilelang_kernel(
                factory,
                (),
                args,
                writes,
                4,
                work=n,
                overwrite=outputs_are_disjoint(args, writes),
            )

        def invoke(current_args, _current_steps, _kernel=kernel):
            del _kernel
            executable(*current_args)

        return invoke

    def prepare_launch(
        self,
        kernel,
        args,
        n,
        *,
        changing,
        enable_fp_fusion,
        writes=(),
        read_writes=(),
        launch_cooperative_grid=False,
    ):
        del changing, enable_fp_fusion, launch_cooperative_grid
        from ....utils.autotune import outputs_are_disjoint, tilelang_kernel

        factory = self.factories.get(kernel)
        if factory is not None:
            # Separate destinations can overwrite trial outputs. An in-place
            # input may occur at only one argument slot, so explicit read/write
            # metadata is needed in addition to checking storage aliases.
            kernel = tilelang_kernel(
                factory,
                (),
                args,
                writes,
                4,
                work=n,
                overwrite=not read_writes and outputs_are_disjoint(args, writes),
            )
        return lambda arguments: kernel(*arguments)

    def argument(self, value, span):
        # Unlike pointer-only runtimes, TileLang validates the flat buffer ABI.
        from ....utils.kernel import storage_span

        return storage_span(value, span)

    def bind_rows(self, value, span):
        # Build the ABI table with two Torch metadata operations, rather than
        # select + as_strided for every argument at every timestep.
        span = self.argument(value[0], span).numel()
        return value.as_strided(
            (value.shape[0], span), (value.stride(0), 1), value.storage_offset()
        ).unbind(0)

    def compile_context(self, toolchain_stamp, target):
        return toolchain_stamp(), target()

    def compile(self, source, enable_fp_fusion, precise_division, stamp, target):
        if stamp is None:
            raise RuntimeError("TileLang compilation requires a toolchain stamp")
        if target is None:
            raise RuntimeError("TileLang compilation requires a resolved device target")
        flags = [f"-DBTORCH_TILELANG_TOOLCHAIN_{stamp}=1"]
        if not enable_fp_fusion:
            flags.append("--fmad=false")
        if precise_division:
            flags.append("--prec-div=true")
        source = source.replace(
            "@tilelang.jit",
            f"@tilelang.jit(target={target!r}, compile_flags={flags!r})",
            1,
        )
        filename = _source_filename(self.name, source)
        namespace = {}
        try:
            exec(compile(source, filename, "exec"), namespace)
            kernel = namespace["build_kernel"]()
            if namespace.get("TUNABLE_THREADS", False):
                self.factories[kernel] = namespace["build_kernel"]
            gc.collect()
            return _retain_source(kernel, filename)
        except BaseException:
            linecache.cache.pop(filename, None)
            raise


_RUNTIME_TYPES = {
    "triton": TritonRuntime,
    "cupy": CuPyRuntime,
    "tilelang": TileLangRuntime,
}


@functools.lru_cache(maxsize=3)
def get_runtime(name):
    """Return the compiler and bounded launch recipes for one source
    backend."""
    runtime = _RUNTIME_TYPES.get(name)
    if runtime is None:
        raise ValueError(f"Unknown source runtime {name!r}")
    return runtime()


def get_renderer(name, n, n_step, storage, precise_exp=False, dynamic_steps=False):
    """Construct the syntax renderer for one registered source backend."""
    renderer = {
        "triton": TritonRenderer,
        "cupy": CuPyRenderer,
        "tilelang": TileLangRenderer,
    }.get(name)
    if renderer is None:
        raise ValueError(f"Unknown source renderer {name!r}")
    return renderer(n, n_step, storage, precise_exp, dynamic_steps)
