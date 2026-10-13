"""Render fixed scalar templates into independent sequence kernels.

One lane owns a neuron across time. Connections never enter these
kernels. Parameter VJPs accumulate in lane registers across time and
reduce across broadcast batch axes afterwards; no time-by-edge
intermediate is created.
"""

import functools
import linecache
import re

import torch

from ..templates import get_template
from .fragments import get_fragments, scalar_layout


def supports(spec, device):
    """Restrict built-in GPU kernels to their explicitly declared models."""
    return device.type == "cuda" and spec.model in (
        "if",
        "lif",
        "glif3",
        "izhikevich",
        "exponential",
        "alpha",
        "alpha_billeh",
        "dual_exponential",
    )


def _mode_name(name, modes):
    for base in ("Iasc", "k", "asc_amps"):
        if modes and name.startswith(base + "_") and name[len(base) + 1 :].isdigit():
            return base, int(name[len(base) + 1 :])
    return name, None


class _Source:
    def __init__(self, backend, n, n_step, modes):
        self.backend, self.n, self.n_step, self.modes = backend, n, n_step, modes
        self.pointers = []
        self.lines = []

    def pointer(self, name, size):
        if (name, size) not in self.pointers:
            self.pointers.append((name, size))

    def index(self, scalar, time=None):
        _, mode = _mode_name(scalar, self.modes)
        index = "index" if mode is None else f"index * {self.modes} + {mode}"
        if time is not None:
            stride = self.n if mode is None else self.n * self.modes
            index = f"({time}) * {stride} + {index}"
        return index

    def load(self, local, pointer, index):
        expr = (
            (
                f"tl.load({pointer}_ptr)"
                if index == "0"
                else f"tl.load({pointer}_ptr + {index}, mask=mask, other=0.0)"
            )
            if self.backend == "triton"
            else f"{pointer}_ptr[{index}]"
        )
        self.lines.append(f"{local} = {expr}")

    def store(self, pointer, index, local):
        self.lines.append(
            f"tl.store({pointer}_ptr + {index}, {local}, mask=mask)"
            if self.backend == "triton"
            else f"{pointer}_ptr[{index}] = {local}"
        )

    def statements(self, lines):
        self.lines.extend(lines)

    def render(self):
        if self.backend == "triton":
            lines = []
            for line in self.lines:
                if line.endswith(" = 0.0"):
                    line = (
                        line.rsplit(" = ", 1)[0]
                        + " = tl.full((BLOCK,), 0.0, tl.float32)"
                    )
                for old, new in (
                    ("exp", "tl.exp"),
                    ("where", "tl.where"),
                    ("maximum", "tl.maximum"),
                    ("abs", "tl.abs"),
                ):
                    line = re.sub(rf"(?<![\w.]){old}\(", new + "(", line)
                lines.append(line)
            args = ", ".join(name + "_ptr" for name, _ in self.pointers)
            return (
                "import triton\nimport triton.language as tl\n@triton.jit\n"
                f"def cell_kernel({args}, BLOCK: tl.constexpr):\n"
                "    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
                f"    mask = index < {self.n}\n"
                + "\n".join("    " + line for line in lines)
                + "\n"
            )
        if self.backend == "cupy":
            declared = {"index", "time"}
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
                elif line == "END":
                    line = "}"
                else:
                    for old, new in (
                        ("exp", "expf"),
                        ("maximum", "fmaxf"),
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
                lines.append(" " * indentation + line)
            args = ", ".join("float* " + name + "_ptr" for name, _ in self.pointers)
            return (
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
            if line.strip() == "END":
                continue
            line = line.replace("range(", "T.serial(")
            for old, new in (
                ("exp", "T.exp"),
                ("where", "T.if_then_else"),
                ("maximum", "T.max"),
                ("abs", "T.abs"),
            ):
                line = re.sub(rf"(?<![\w.]){old}\(", new + "(", line)
            for name in locals_:
                line = re.sub(rf"\b{name}\b", name + "[0]", line)
            lines.append(line)
        args = ", ".join(
            f'{name}_ptr: T.Tensor(({size},), "float32")'
            for name, size in self.pointers
        )
        return (
            "import tilelang\nimport tilelang.language as T\n"
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


def generate_source(
    backend,
    spec,
    n,
    n_step,
    history_names,
    modes=0,
    backward=False,
    parameter_mask=None,
):
    """Generate inspectable direct DSL source for a static cell sequence.

    Returns:
        Source text and ordered ``(buffer_name, element_count)`` declarations.
    """
    template = get_template(spec)
    states, parameters = scalar_layout(spec, modes)
    fragments = get_fragments(spec, modes)
    source = _Source(backend, n, n_step, modes)
    source.pointer("dt", 1)
    source.load("dt", "dt", "0")
    size = lambda name: (
        n * modes
        if name == "Iasc" or (spec.model == "glif3" and name in ("k", "asc_amps"))
        else n
    )
    source.pointer("x_seq", n * n_step)
    for name in template.state_names:
        source.pointer(name + "_init", size(name))
    for name in template.parameter_names:
        source.pointer(name, size(name))
    for name in parameters:
        base, _ = _mode_name(name, modes)
        source.load(name, base, source.index(name))
    if not backward:
        source.pointer("output_seq", n * n_step)
        for name in template.state_names:
            source.pointer(name + "_final", size(name))
        for name in history_names:
            source.pointer(name + "_seq", size(name) * n_step)
        for name in states:
            base, _ = _mode_name(name, modes)
            source.load(name + "_prev", base + "_init", source.index(name))
        source.lines.append(f"for time in range({n_step}):")
        start = len(source.lines)
        source.load("x", "x_seq", source.index("x", "time"))
        source.statements(fragments.forward)
        source.store("output_seq", source.index("output", "time"), "output")
        for name in history_names:
            scalars = (
                tuple(f"Iasc_{i}" for i in range(modes)) if name == "Iasc" else (name,)
            )
            for scalar in scalars:
                value = scalar if name.endswith("_charged") else scalar + "_next"
                source.store(name + "_seq", source.index(scalar, "time"), value)
        source.statements(f"{name}_prev = {name}_next" for name in states)
        source.lines[start:] = ["    " + line for line in source.lines[start:]]
        if backend == "cupy":
            source.lines.append("END")
        for name in states:
            base, _ = _mode_name(name, modes)
            source.store(base + "_final", source.index(name), name + "_prev")
    else:
        source.pointer("grad_output_seq", n * n_step)
        source.pointer("grad_x_seq", n * n_step)
        for name in template.state_names:
            source.pointer(name + "_seq", size(name) * n_step)
            source.pointer("grad_" + name + "_final", size(name))
            source.pointer("grad_" + name + "_init", size(name))
        for name in history_names:
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
        for name in states:
            base, _ = _mode_name(name, modes)
            source.load("grad_s_" + name, "grad_" + base + "_final", source.index(name))
        source.lines.append(f"for time in range({n_step - 1}, -1, -1):")
        start = len(source.lines)
        source.load("x", "x_seq", source.index("x", "time"))
        source.load("grad_output", "grad_output_seq", source.index("output", "time"))
        for name in states:
            base, _ = _mode_name(name, modes)
            index = source.index(name)
            history_index = source.index(name, "time - 1")
            if backend == "triton":
                source.lines += [
                    f"{name}_prev = tl.load({base}_seq_ptr + {history_index}, "
                    "mask=mask & (time > 0), other=0.0)",
                    f"{name}_prev = tl.where(time > 0, {name}_prev, "
                    f"tl.load({base}_init_ptr + {index}, mask=mask, other=0.0))",
                ]
            elif backend == "cupy":
                source.lines += [
                    f"{name}_prev = (time > 0 ? {base}_seq_ptr[{history_index}] "
                    f": {base}_init_ptr[{index}])"
                ]
            else:
                # Clamp the unused branch index so either evaluation strategy
                # remains within the history allocation.
                source.lines += [
                    f"{name}_prev = where(time > 0, {base}_seq_ptr["
                    f"maximum(time - 1, 0) * {size(base)} + {index}], "
                    f"{base}_init_ptr[{index}])"
                ]
            if base in history_names:
                source.load(
                    "grad_r_" + name,
                    "grad_" + base + "_seq",
                    source.index(name, "time"),
                )
                source.lines.append(f"grad_s_{name} = grad_s_{name} + grad_r_{name}")
        for name in ("v_charged", "u_charged"):
            if name in history_names:
                source.load(
                    "grad_r_" + name,
                    "grad_" + name + "_seq",
                    source.index(name, "time"),
                )
            else:
                source.lines.append("grad_r_" + name + " = 0.0")
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
        for name in states:
            base, _ = _mode_name(name, modes)
            source.store("grad_" + base + "_init", source.index(name), "grad_s_" + name)
        for name in parameters:
            base, _ = _mode_name(name, modes)
            if base in needed_params:
                source.store("grad_" + base, source.index(name), "sum_grad_" + name)
    return source.render(), tuple(source.pointers)


@functools.lru_cache(maxsize=256)
def _compile(backend, source):
    if backend == "cupy":
        import cupy

        kernel = cupy.RawKernel(source, "cell_kernel")
        kernel.compile()
        return kernel
    namespace = {}
    filename = f"<btorch-{backend}-cell-{len(linecache.cache)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return (
        namespace["cell_kernel"] if backend == "triton" else namespace["build_kernel"]()
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
):
    if not n:
        return
    source, pointers = generate_source(
        backend, spec, n, n_step, history_names, modes, backward, parameter_mask
    )
    kernel = _compile(backend, source)
    args = tuple(buffers[name].reshape(-1) for name, _ in pointers)
    if backend == "triton":
        kernel[((n + 127) // 128,)](*args, BLOCK=128)
    elif backend == "cupy":
        import cupy

        with cupy.cuda.Device(args[0].device.index):
            stream = cupy.cuda.ExternalStream(
                torch.cuda.current_stream(args[0].device).cuda_stream
            )
            with stream:
                # DLPack exports storage only. The original differentiable
                # tensors remain explicit custom-op inputs and are handled by
                # the registered VJP; detach here only satisfies its exporter.
                arrays = tuple(cupy.from_dlpack(arg.detach()) for arg in args)
                kernel(((n + 127) // 128,), (128,), arrays)
    else:
        kernel(*args)


def _dt_buffer(dt, reference):
    """Keep the invocation timestep on device without reading tensor values."""
    if isinstance(dt, torch.Tensor):
        if dt.numel() != 1:
            raise ValueError("Cell timestep must be scalar")
        return dt.to(device=reference.device, dtype=torch.float32).reshape(1)
    # ``full`` fills device storage directly; unlike ``tensor([dt], device=...)``
    # it does not introduce a host-to-device copy during CUDA graph capture.
    return torch.full((1,), dt, device=reference.device, dtype=torch.float32)


def _prepare(x_seq, states, parameters, spec):
    if x_seq.device.type != "cuda":
        raise ValueError("GPU cell backends require CUDA tensors")
    template = get_template(spec)
    shape = x_seq.shape[1:]
    modes = (
        states[template.state_names.index("Iasc")].shape[-1]
        if spec.model == "glif3"
        else 0
    )
    target = lambda name: (
        (*shape, modes)
        if name == "Iasc" or (spec.model == "glif3" and name in ("k", "asc_amps"))
        else shape
    )
    buffers = {"x_seq": x_seq.float().contiguous(), "dt": _dt_buffer(spec.dt, x_seq)}
    buffers.update(
        {
            name + "_init": tensor.float().expand(target(name)).contiguous()
            for name, tensor in zip(template.state_names, states)
        }
    )
    buffers.update(
        {
            name: tensor.float().expand(target(name)).contiguous()
            for name, tensor in zip(template.parameter_names, parameters)
        }
    )
    return template, shape, modes, target, buffers


def forward(backend, x_seq, states, parameters, spec, history_names):
    """Execute inference or training forward using exactly requested
    storage."""
    template, shape, modes, target, buffers = _prepare(x_seq, states, parameters, spec)
    n_step = x_seq.shape[0]
    n = x_seq[0].numel() if n_step else x_seq.numel() // max(n_step, 1)
    # Empty sequences preserve initial state with the same FP32 policy.
    if not n_step:
        return (
            x_seq.float().clone(),
            tuple(buffers[name + "_init"].clone() for name in template.state_names),
            tuple(
                x_seq.new_empty((0, *target(name)), dtype=torch.float32)
                for name in history_names
            ),
        )
    buffers["output_seq"] = torch.empty_like(x_seq, dtype=torch.float32)
    for name in template.state_names:
        buffers[name + "_final"] = torch.empty(
            target(name), device=x_seq.device, dtype=torch.float32
        )
    for name in history_names:
        buffers[name + "_seq"] = torch.empty(
            (n_step, *target(name)), device=x_seq.device, dtype=torch.float32
        )
    _launch(backend, spec, buffers, n, n_step, history_names, modes)
    return (
        buffers["output_seq"],
        tuple(buffers[name + "_final"] for name in template.state_names),
        tuple(buffers[name + "_seq"] for name in history_names),
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
):
    """Reverse a sequence with local charge recomputation and state
    adjoints."""
    template, shape, modes, target, buffers = _prepare(x_seq, states, parameters, spec)
    needs = needs_input_grad or (True,) * (1 + len(states) + len(parameters))
    n_step, n = x_seq.shape[0], buffers["x_seq"].numel() // max(x_seq.shape[0], 1)
    zeros = lambda value, dimensions: (
        torch.zeros(dimensions, device=x_seq.device, dtype=torch.float32)
        if value is None
        else value.float().contiguous()
    )
    buffers["grad_output_seq"] = zeros(grad_output, x_seq.shape)
    buffers["grad_x_seq"] = torch.empty_like(x_seq, dtype=torch.float32)
    for name, history, grad in zip(history_names, histories, grad_histories):
        buffers[name + "_seq"] = history.contiguous()
        buffers["grad_" + name + "_seq"] = zeros(grad, history.shape)
    for name, grad in zip(template.state_names, grad_final_states):
        if name + "_seq" not in buffers:
            raise RuntimeError(f"Training history must include state {name!r}")
        buffers["grad_" + name + "_final"] = zeros(grad, target(name))
        buffers["grad_" + name + "_init"] = torch.empty(
            target(name), device=x_seq.device, dtype=torch.float32
        )
    parameter_mask = tuple(
        name
        for i, name in enumerate(template.parameter_names)
        if needs[1 + len(states) + i]
    )
    for name in parameter_mask:
        buffers["grad_" + name] = torch.empty(
            target(name), device=x_seq.device, dtype=torch.float32
        )
    if n_step:
        _launch(
            backend,
            spec,
            buffers,
            n,
            n_step,
            history_names,
            modes,
            True,
            parameter_mask,
        )
    else:
        for name in template.state_names:
            buffers["grad_" + name + "_init"] = buffers["grad_" + name + "_final"]
        for name in parameter_mask:
            buffers["grad_" + name].zero_()
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
        buffers["grad_x_seq"].to(x_seq.dtype) if needs[0] else None,
        grad_states,
        grad_parameters,
    )
