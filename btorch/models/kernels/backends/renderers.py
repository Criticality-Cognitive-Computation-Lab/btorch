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
from dataclasses import dataclass


_MATH_NAMES = {
    "abs",
    "exp",
    "expm1",
    "fma",
    "maximum",
    "minimum",
    "where",
}
_FLOAT_DTYPES = {"float16", "bfloat16", "float64"}


@dataclass(frozen=True)
class RendererEmission:
    """Source lines and capability flags emitted by a renderer helper."""

    lines: tuple[str, ...]
    feature: str | None = None


class _TritonPreciseDivision(ast.NodeTransformer):
    """Lower only floating division for Triton's precise-division mode."""

    def visit_BinOp(self, node):  # noqa: N802 - ast API
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


class _MathTransformer(ast.NodeTransformer):
    def __init__(self, calls: dict[str, str], *, floor_division: bool = False):
        self.calls = calls
        self.floor_division = floor_division

    def visit_Call(self, node):  # noqa: N802 - ast API
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            replacement = self.calls.get(node.func.id)
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
) -> str:
    """Lower scalar math calls in one Python-like source statement."""
    tree = _MathTransformer(calls, floor_division=floor_division).visit(
        _parse_statement(line)
    )
    return ast.unparse(ast.fix_missing_locations(tree))


def local_statement(line: str, names: set[str]) -> str:
    """Render scalar registers as one-element TileLang local arrays."""
    tree = _LocalArrayTransformer(names).visit(_parse_statement(line))
    return ast.unparse(ast.fix_missing_locations(tree))


class SourceRenderer:
    """Backend-independent renderer interface used by ``_Source``."""

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

    def load(self, pointer, index, layout, *, lane_index):
        expression = f"{pointer}_ptr[{index}]"
        if layout is not None and layout.dtype in _FLOAT_DTYPES:
            expression = self.cast_load(expression, layout.dtype)
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
        if stripped.startswith(("for ", "if ")):
            return stripped
        calls = dict(self.calls)
        if self.precise_exp:
            calls["exp"] = self.precise_exp_call
        return transform_statement(
            line,
            calls,
            floor_division=self.floor_division,
        )

    def render(self, lines, pointers, has_spike_list, has_gradient_list):
        raise NotImplementedError


class TritonRenderer(SourceRenderer):
    name = "triton"
    calls = {
        "abs": "tl.abs",
        "exp": "tl.exp",
        "expm1": "libdevice.expm1",
        "fma": "tl.fma",
        "maximum": "tl.maximum",
        "minimum": "tl.minimum",
        "where": "tl.where",
    }
    precise_exp_call = "libdevice.exp"

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

    def load(self, pointer, index, layout, *, lane_index):
        expression = (
            f"tl.load({pointer}_ptr)"
            if index == "0"
            else f"tl.load({pointer}_ptr + {index}, mask=mask, other=0.0)"
            if "index" in index
            else f"tl.load({pointer}_ptr + {index})"
        )
        if layout is not None and layout.dtype in _FLOAT_DTYPES:
            expression = self.cast_load(expression, layout.dtype)
        if lane_index and "index" not in index:
            expression = f"tl.broadcast_to({expression}, (BLOCK,))"
        return expression

    def cast_load(self, expression, dtype):
        return f"({expression}).to(tl.float32)"

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

    def render(self, lines, pointers, has_spike_list, has_gradient_list):
        rendered = []
        for raw in lines:
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
                    rendered.append(indentation + "for time in tl.range(0, n_steps):")
                continue
            if line.rstrip().endswith(" = 0.0"):
                statement = _parse_statement(line)
                if isinstance(statement, ast.Assign):
                    line = (
                        f"{ast.unparse(statement.targets[0])} = "
                        "tl.full((BLOCK,), 0.0, tl.float32)"
                    )
            else:
                line = self._math(line)
            rendered.append(indentation + line)
        args = ", ".join(name + "_ptr" for name, _ in pointers)
        if self.dynamic_steps:
            args += ", n_steps"
        imports = (
            "import triton\nimport triton.language as tl\n"
            "from triton.language.extra import libdevice\n"
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
            + (
                "@triton.jit(do_not_specialize=['n_steps'])\n"
                if self.dynamic_steps
                else "@triton.jit\n"
            )
            + f"def cell_kernel({args}, BLOCK: tl.constexpr):\n"
            + "    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
            + f"    mask = index < {self.n}\n"
            + "\n".join("    " + line for line in rendered)
            + "\n"
        )


class CuPyRenderer(SourceRenderer):
    name = "cupy"
    floor_division = True
    calls = {
        "abs": "fabsf",
        "exp": "expf",
        "expm1": "expm1f",
        "fma": "fmaf",
        "maximum": "fmaxf",
        "minimum": "fminf",
        "where": "where",
    }
    precise_exp_call = "expf"

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

    def render(self, lines, pointers, has_spike_list, has_gradient_list):
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
        "exp": "T.exp",
        "expm1": "_btorch_expm1",
        "fma": "__tilelang_fma__",
        "maximum": "T.max",
        "minimum": "T.min",
        "where": "T.if_then_else",
    }
    precise_exp_call = "__tilelang_exp__"

    def validate_dt(self, required_dt=None):
        lines = [
            "T.device_assert(index != 0 or (T.isfinite(dt) and dt > 0.0), "
            '"dt must be finite and positive")'
        ]
        if required_dt is not None:
            lines.append(
                f"T.device_assert(index != 0 or dt == {required_dt!r}, "
                f'"requires dt == {required_dt:g}")'
            )
        return tuple(lines)

    def _math(self, line):
        if line.strip().startswith(("for ", "if ")):
            return line.strip()
        calls = dict(self.calls)
        if self.precise_exp:
            calls["exp"] = "__tilelang_exp__"
        tree = _MathTransformer(calls).visit(_parse_statement(line))

        # TileLang's pure extern call needs the same ABI as fma.
        class _ExpandExp(ast.NodeTransformer):
            def visit_Call(self, node):  # noqa: N802 - ast API
                node = self.generic_visit(node)
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "__tilelang_exp__"
                ):
                    node.func = ast.parse("T.call_pure_extern", mode="eval").body
                    node.args = [
                        ast.Constant("float32"),
                        ast.Constant("expf"),
                        *node.args,
                    ]
                return node

        tree = _ExpandExp().visit(tree)
        return ast.unparse(ast.fix_missing_locations(tree))

    def cast_load(self, expression, dtype):
        return f'T.Cast("float32", {expression})'

    def render(self, lines, pointers, has_spike_list, has_gradient_list):
        locals_ = sorted(
            {
                statement.targets[0].id
                for line in lines
                if " = " in line and not line.strip().startswith("for ")
                for statement in (_parse_statement(line.strip()),)
                if isinstance(statement, ast.Assign)
                and isinstance(statement.targets[0], ast.Name)
            }
        )
        rendered = []
        for raw in lines:
            indentation = raw[: len(raw) - len(raw.lstrip())]
            line = raw.strip()
            if line in ("END", "END_IF", "END_REPLAY"):
                continue
            line = line.replace("range(", "T.serial(")
            line = self._math(line)
            if not line.startswith(("for ", "if ")):
                line = local_statement(line, set(locals_))
            rendered.append(indentation + line)
        args = ", ".join(
            f'{name}_ptr: T.Tensor(({size},), "'
            f'{self.storage[name].dtype if name in self.storage else "float32"}")'
            for name, size in pointers
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
            + "\n".join("                " + line for line in rendered)
            + "\n    return cell_kernel\n"
        )


class RendererRuntime:
    """Compilation and launch operations for one source backend."""

    name = ""
    supports_dynamic_steps = False
    supports_pointer_offsets = False

    def argument(self, value, span):
        """Expose the pointer ABI without changing a tensor's storage."""
        del span
        return value

    def bind_rows(self, value, span):
        """Prepare a sequence's argument views for a tensor-based ABI."""
        return tuple(self.argument(row, span) for row in value.unbind(0))

    def prepare_launch(self, kernel, args, n, *, changing, enable_fp_fusion):
        """Bind launch policy for an invocation with stable argument
        layouts."""
        del args, changing
        return functools.partial(
            self.launch,
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

    def launch(
        self, kernel, args, n, *, steps=None, enable_fp_fusion, debug, num_warps
    ):
        raise NotImplementedError


class TritonRuntime(RendererRuntime):
    name = "triton"
    supports_dynamic_steps = True
    supports_pointer_offsets = True

    def prepare_launch(self, kernel, args, n, *, changing, enable_fp_fusion):
        # Row views can change alignment even when shape/strides are constant.
        # Specialize dtype/layout once, but never assume the first row's pointer
        # alignment for changing arguments. Static parameters retain alignment.
        from ....utils.triton import prepare_launch

        return prepare_launch(
            kernel,
            args,
            ((n + 127) // 128,),
            (("BLOCK", 128),),
            changing=changing,
            enable_fp_fusion=enable_fp_fusion,
        )

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

    def launch(
        self, kernel, args, n, *, steps=None, enable_fp_fusion, debug, num_warps
    ):
        options = {}
        if debug is not None:
            options["debug"] = debug
        if num_warps is not None:
            options["num_warps"] = num_warps
        call_args = (*args, steps) if steps is not None else args
        kernel[((n + 127) // 128,)](
            *call_args, BLOCK=128, enable_fp_fusion=enable_fp_fusion, **options
        )


class CuPyRuntime(RendererRuntime):
    name = "cupy"
    supports_dynamic_steps = True
    supports_pointer_offsets = True

    def prepare_launch(self, kernel, args, n, *, changing, enable_fp_fusion):
        del enable_fp_fusion
        import cupy
        import torch

        from ....utils.kernel import PreparedLaunch

        device = args[0].device.index
        grid = ((n + 127) // 128,)
        stream = cupy.cuda.Stream.from_external(torch.cuda.current_stream(device))

        def launch(*pointers):
            nonlocal stream
            # Resolve the caller's stream at submission, including capture.
            # The fused invocation already owns the device guard.
            current = torch.cuda.current_stream(device)
            if current.cuda_stream != stream.ptr:
                stream = cupy.cuda.Stream.from_external(current)
            kernel(grid, (128,), pointers, stream=stream)

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

    def launch(
        self, kernel, args, n, *, steps=None, enable_fp_fusion, debug, num_warps
    ):
        del enable_fp_fusion, debug, num_warps
        import cupy
        import numpy
        import torch

        with cupy.cuda.Device(args[0].device.index):
            stream = cupy.cuda.Stream.from_external(
                torch.cuda.current_stream(args[0].device)
            )
            with stream:
                pointers = tuple(numpy.uintp(arg.data_ptr()) for arg in args)
                call_args = (
                    (*pointers, numpy.int32(steps)) if steps is not None else pointers
                )
                kernel(((n + 127) // 128,), (128,), call_args)


class TileLangRuntime(RendererRuntime):
    name = "tilelang"

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
            gc.collect()
            return _retain_source(kernel, filename)
        except BaseException:
            linecache.cache.pop(filename, None)
            raise

    def launch(
        self, kernel, args, n, *, steps=None, enable_fp_fusion, debug, num_warps
    ):
        del n, enable_fp_fusion, debug, num_warps, steps
        kernel(*args)


_RUNTIME_TYPES = {
    "triton": TritonRuntime,
    "cupy": CuPyRuntime,
    "tilelang": TileLangRuntime,
}


@functools.lru_cache(maxsize=3)
def get_runtime(name):
    """Return the stateless compiler/launcher for one source backend."""
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
