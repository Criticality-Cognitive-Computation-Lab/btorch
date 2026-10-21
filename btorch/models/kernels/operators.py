"""Register complete sequence boundaries with fake kernels and explicit
VJPs."""

import functools
import math

import torch
from torch import Tensor

from .contracts import CellSpec, HistoryPlan, PoissonOptions
from .options import KernelOptions, resolve_model_options
from .registry import _NATIVE_GPU_BACKENDS, cell_backend_revision
from .templates import (
    ResidualTemplate,
    decode_residual_template,
    get_reference_callbacks,
    get_template,
    model_template_revision,
    residual_template,
)


if not hasattr(torch.library, "custom_op"):
    raise NotImplementedError(
        "Cell sequence backends require PyTorch >= 2.4 with torch.library.custom_op"
    )


@torch.library.custom_op("btorch_cells::environment_dt", mutates_args=())
def environment_dt(x_seq: Tensor, model: str) -> Tensor:
    """Read the invocation timestep without specializing the compiled graph."""
    from .. import environ

    dt = float(environ.get("dt"))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if model == "alpha_billeh" and dt != 1:
        raise ValueError("AlphaPSCBilleh requires dt == 1")
    return torch.full((), dt, device=x_seq.device, dtype=torch.float32)


@environment_dt.register_fake
def _fake_environment_dt(x_seq, model):
    return x_seq.new_empty((), dtype=torch.float32)


def _setup_environment_dt(ctx, inputs, output):
    # The environment scalar is metadata, not a differentiable model input.
    ctx.mark_non_differentiable(output)


environment_dt.register_autograd(
    lambda ctx, gradient: (None, None), setup_context=_setup_environment_dt
)


_poisson_library = torch.library.Library("btorch_cells", "FRAGMENT")
_poisson_library.define(
    "poisson_uniforms(Tensor x_seq) -> Tensor",
    tags=(torch.Tag.nondeterministic_seeded,),
)


def _poisson_uniforms_impl(x_seq: Tensor) -> Tensor:
    """Draw one contiguous FP32 uniform tensor per logical timestep."""
    reference = torch.empty(x_seq.shape[1:], device=x_seq.device, dtype=torch.float32)
    if not x_seq.shape[0]:
        return reference.new_empty((0, *reference.shape))
    return torch.stack(tuple(torch.rand_like(reference) for _ in range(x_seq.shape[0])))


def _fake_poisson_uniforms(x_seq):
    return x_seq.new_empty(x_seq.shape, dtype=torch.float32)


def _setup_poisson_uniforms(ctx, inputs, output):
    ctx.mark_non_differentiable(output)


_poisson_library.impl(
    "poisson_uniforms", _poisson_uniforms_impl, "CompositeExplicitAutograd"
)
torch.library.register_fake(
    "btorch_cells::poisson_uniforms", _fake_poisson_uniforms, lib=_poisson_library
)
torch.library.register_autograd(
    "btorch_cells::poisson_uniforms",
    lambda ctx, gradient: (None,),
    setup_context=_setup_poisson_uniforms,
    lib=_poisson_library,
)


poisson_uniforms = torch.ops.btorch_cells.poisson_uniforms.default


def encode_spec(spec: CellSpec) -> str:
    """Encode immutable compile-specialized scalar choices."""
    values = (
        spec.model,
        str(int(spec.hard_reset)),
        str(int(spec.detach_reset)),
        spec.surrogate,
        str(spec.alpha),
        str(spec.damping),
        str(int(spec.refractory)),
    )
    if spec.poisson is None:
        return "|".join(values)
    poisson = spec.poisson
    return "|".join(
        (
            *values,
            str(int(poisson.spiking)),
            repr(poisson.tau),
            repr(poisson.rho),
            repr(poisson.leak),
            repr(poisson.k),
        )
    )


def decode_spec(encoded: str, dt: float | Tensor = 1.0) -> CellSpec:
    """Restore a cell specification inside the tensor operator boundary."""
    values = encoded.split("|")
    if len(values) not in (7, 12):
        raise ValueError("Invalid encoded cell specification")
    model, hard_reset, detach_reset, surrogate, alpha, damping, refractory = values[:7]
    poisson = (
        None
        if len(values) == 7
        else PoissonOptions(
            bool(int(values[7])),
            float(values[8]),
            float(values[9]),
            float(values[10]),
            float(values[11]),
        )
    )
    return CellSpec(
        model,
        dt,
        bool(int(hard_reset)),
        bool(int(detach_reset)),
        surrogate,
        float(alpha),
        float(damping),
        bool(int(refractory)),
        poisson,
    )


def _backend(name, spec, device):
    from .registry import resolve_cell_backend

    return resolve_cell_backend(name, spec, device)


def _split_history_plan(value, spec):
    names, separator, encoded = value.partition("#")
    residual = (
        decode_residual_template(encoded)
        if separator
        else ResidualTemplate(get_template(spec).state_names)
    )
    return names, residual


def _validate_compile_policy(
    backend, enable_fp_fusion, precise_division=False, precise_exp=False
):
    if (not enable_fp_fusion or precise_division or precise_exp) and backend not in (
        "torch",
        *_NATIVE_GPU_BACKENDS,
    ):
        raise NotImplementedError(
            "Nondefault compilation policies are supported only by built-in "
            "cell backends"
        )


def _uses_integrated_dt_validation(callback, backend, x_seq):
    if backend not in _NATIVE_GPU_BACKENDS or x_seq.device.type != "cuda":
        return False
    if not x_seq.shape[0] or not x_seq.numel():
        return False
    while isinstance(callback, functools.partial):
        callback = callback.func
    from .backends import engine

    return callback is engine.forward


def _validate_parameter_shapes(parameters, spec, states, target_shape, *, label=""):
    """Validate model-aware parameter broadcasting without tensor reads."""
    template = get_template(spec)
    modes = (
        states[template.state_names.index("Iasc")].shape[-1]
        if spec.model == "glif3"
        else 0
    )
    prefix = f"{label} " if label else ""
    for name, value in zip(template.parameter_names, parameters):
        target = (
            (*target_shape, modes)
            if spec.model == "glif3" and name in ("k", "asc_amps")
            else target_shape
        )
        try:
            broadcast = torch.broadcast_shapes(value.shape, target)
        except RuntimeError as error:
            raise ValueError(
                f"{prefix}parameter {name!r} shape {tuple(value.shape)} cannot "
                f"broadcast to {tuple(target)}"
            ) from error
        if broadcast != target:
            raise ValueError(
                f"{prefix}parameter {name!r} shape {tuple(value.shape)} cannot "
                f"broadcast to {tuple(target)}"
            )


def _validate_poisson_fp32(spec, tensors):
    """Reject unqualified native Poisson precision before consuming RNG."""
    if spec.poisson is None:
        return
    for name, value in tensors:
        if value.dtype != torch.float32:
            raise ValueError(
                "Native Poisson lowering requires FP32 inputs, states, "
                f"parameters, and uniforms; {name} has dtype {value.dtype}"
            )


def _forward(
    x_seq,
    states,
    parameters,
    encoded,
    history_names,
    backend,
    enable_fp_fusion,
    precise_division,
    dt,
    residual=None,
    uniform_seq=None,
    precise_exp=False,
):
    spec = decode_spec(encoded, dt)
    _validate_compile_policy(backend, enable_fp_fusion, precise_division, precise_exp)
    callback = _backend(backend, spec, x_seq.device).forward
    if not _uses_integrated_dt_validation(callback, backend, x_seq):
        torch._assert_async(
            torch.isfinite(dt) & (dt > 0), "dt must be finite and positive"
        )
        if spec.model == "alpha_billeh":
            torch._assert_async(dt == 1, "AlphaPSCBilleh requires dt == 1")
    arguments = (
        x_seq,
        tuple(states),
        tuple(parameters),
        spec,
        tuple(history_names.split("|")) if history_names else (),
    )
    if backend in _NATIVE_GPU_BACKENDS and x_seq.device.type == "cuda":
        output, final, histories = callback(
            *arguments,
            uniform_seq=(
                uniform_seq if uniform_seq is not None and uniform_seq.numel() else None
            ),
            enable_fp_fusion=enable_fp_fusion,
            precise_division=precise_division,
            precise_exp=precise_exp,
            **({"residual_template": residual} if residual is not None else {}),
        )
    else:
        output, final, histories = callback(*arguments)
    return [output, *final, *histories]


@torch.library.custom_op("btorch_cells::inference", mutates_args=())
def _inference(
    x_seq: Tensor,
    states: list[Tensor],
    parameters: list[Tensor],
    encoded: str,
    history_names: str,
    backend: str,
    dt: Tensor,
    enable_fp_fusion: bool = True,
    precise_division: bool = False,
    uniform_seq: Tensor | None = None,
    precise_exp: bool = False,
) -> list[Tensor]:
    return _forward(
        x_seq,
        states,
        parameters,
        encoded,
        history_names,
        backend,
        enable_fp_fusion,
        precise_division,
        dt,
        uniform_seq=uniform_seq,
        precise_exp=precise_exp,
    )


@torch.library.custom_op("btorch_cells::training", mutates_args=())
def _training(
    x_seq: Tensor,
    states: list[Tensor],
    parameters: list[Tensor],
    encoded: str,
    history_names: str,
    backend: str,
    dt: Tensor,
    enable_fp_fusion: bool = True,
    precise_division: bool = False,
    uniform_seq: Tensor | None = None,
    precise_exp: bool = False,
) -> list[Tensor]:
    names, residual = _split_history_plan(history_names, decode_spec(encoded))
    return _forward(
        x_seq,
        states,
        parameters,
        encoded,
        names,
        backend,
        enable_fp_fusion,
        precise_division,
        dt,
        residual,
        uniform_seq,
        precise_exp,
    )


def _fake_forward(
    x_seq,
    states,
    parameters,
    encoded,
    history_names,
    backend,
    dt,
    enable_fp_fusion=True,
    precise_division=False,
    uniform_seq=None,
    precise_exp=False,
):
    template = get_template(decode_spec(encoded))
    state_map = dict(zip(template.state_names, states))
    records = []
    for name in history_names.split("|") if history_names else ():
        source = state_map[template.history_state_name(name)]
        records.append(
            source.new_empty((x_seq.shape[0], *source.shape), dtype=torch.float32)
        )
    return [
        torch.empty_like(x_seq, dtype=torch.float32),
        *(torch.empty_like(value, dtype=torch.float32) for value in states),
        *records,
    ]


_inference.register_fake(_fake_forward)


@_training.register_fake
def _fake_training(
    x_seq,
    states,
    parameters,
    encoded,
    history_names,
    backend,
    dt,
    enable_fp_fusion=True,
    precise_division=False,
    uniform_seq=None,
    precise_exp=False,
):
    history_names, residual = _split_history_plan(history_names, decode_spec(encoded))
    result = _fake_forward(
        x_seq,
        states,
        parameters,
        encoded,
        history_names,
        backend,
        dt,
        enable_fp_fusion,
        precise_division,
        uniform_seq,
        precise_exp,
    )
    if residual.checkpoint_state is not None:
        template = get_template(decode_spec(encoded))
        state = states[template.state_names.index(residual.checkpoint_state)]
        checkpoints = max((x_seq.shape[0] - 1) // residual.checkpoint_interval, 0)
        result.append(state.new_empty((checkpoints, *state.shape), dtype=torch.float32))
    return result


@torch.library.custom_op(
    "btorch_cells::reverse",
    mutates_args=(),
    schema="(Tensor x_seq, Tensor[] states, Tensor[] parameters, str encoded, "
    "str history_names, Tensor[] histories, Tensor[] residuals, Tensor? grad_output, "
    "str residual_encoded, Tensor? residual_output, "
    "Tensor?[] grad_final, Tensor?[] grad_history, str backend, bool[] needs, "
    "bool enable_fp_fusion, bool precise_division, bool precise_exp, "
    "int backend_revision, "
    "int model_revision, Tensor dt) "
    "-> Tensor?[]",
)
def _reverse(
    x_seq: Tensor,
    states: list[Tensor],
    parameters: list[Tensor],
    encoded: str,
    history_names: str,
    histories: list[Tensor],
    residuals: list[Tensor],
    grad_output: Tensor | None,
    residual_encoded: str,
    residual_output: Tensor | None,
    grad_final: list[Tensor | None],
    grad_history: list[Tensor | None],
    backend: str,
    needs: list[bool],
    enable_fp_fusion: bool,
    precise_division: bool,
    precise_exp: bool,
    backend_revision: int,
    model_revision: int,
    dt: Tensor,
) -> list[Tensor | None]:
    spec = decode_spec(encoded, dt)
    _validate_compile_policy(backend, enable_fp_fusion, precise_division, precise_exp)
    if (
        cell_backend_revision(backend) != backend_revision
        or model_template_revision(spec.model) != model_revision
    ):
        raise RuntimeError(
            "Cell registration changed after forward; complete all outstanding "
            "backward calls before replacing a backend or model template"
        )
    # Native GPU reverse kernels lower absent cotangents to zero registers.
    # Torch and third-party callbacks retain the original dense-tensor contract.
    native_optional = backend in _NATIVE_GPU_BACKENDS and (x_seq.device.type == "cuda")
    if not native_optional:
        grad_output = (
            grad_output if grad_output is not None else torch.zeros_like(x_seq)
        )
        grad_final = [
            gradient if gradient is not None else torch.zeros_like(state)
            for gradient, state in zip(grad_final, states)
        ]
        grad_history = [
            gradient if gradient is not None else torch.zeros_like(history)
            for gradient, history in zip(grad_history, histories)
        ]
    callback = _backend(backend, spec, x_seq.device).backward
    residual = decode_residual_template(residual_encoded)
    arguments = (
        x_seq,
        tuple(states),
        tuple(parameters),
        spec,
        tuple(history_names.split("|")) if history_names else (),
        tuple(histories),
        grad_output,
        tuple(grad_final),
        tuple(grad_history),
    )
    if backend in _NATIVE_GPU_BACKENDS and x_seq.device.type == "cuda":
        gradients = callback(
            *arguments,
            needs_input_grad=tuple(needs),
            enable_fp_fusion=enable_fp_fusion,
            precise_division=precise_division,
            precise_exp=precise_exp,
            residual_template=residual,
            residual_output=residual_output,
            residuals=tuple(residuals),
        )
    else:
        gradients = callback(*arguments, needs_input_grad=tuple(needs))
    gx, gs, gp = gradients
    # Optional tensor outputs avoid allocating zero tensors for frozen inputs.
    return [
        value.clone() if value is not None and x_seq.shape[0] == 0 else value
        for value in (gx, *gs, *gp)
    ]


@_reverse.register_fake
def _fake_reverse(
    x_seq,
    states,
    parameters,
    encoded,
    history_names,
    histories,
    residuals,
    grad_output,
    residual_encoded,
    residual_output,
    grad_final,
    grad_history,
    backend,
    needs,
    enable_fp_fusion,
    precise_division,
    precise_exp,
    backend_revision,
    model_revision,
    dt,
):
    return [
        torch.empty_like(value) if needed else None
        for value, needed in zip((x_seq, *states, *parameters), needs)
    ]


def _setup_context(ctx, inputs, output):
    ctx.n_input_args = len(inputs)
    (
        x_seq,
        states,
        parameters,
        encoded,
        history_names,
        backend,
        dt,
        *policy,
    ) = inputs
    enable_fp_fusion = policy[0] if policy else True
    precise_division = policy[1] if len(policy) > 1 else False
    precise_exp = policy[3] if len(policy) > 3 else False
    history_names, residual = _split_history_plan(history_names, decode_spec(encoded))
    residual_encoded = residual.encoded
    ctx.n_states = len(states)
    ctx.n_parameters = len(parameters)
    ctx.encoded = encoded
    ctx.history_names = history_names
    ctx.n_histories = len(tuple(filter(None, history_names.split("|"))))
    ctx.residual_encoded = residual_encoded
    ctx.save_residual_output = residual.save_output
    ctx.backend = backend
    ctx.enable_fp_fusion = enable_fp_fusion
    ctx.precise_division = precise_division
    ctx.precise_exp = precise_exp
    ctx.backend_revision = cell_backend_revision(backend)
    ctx.model_revision = model_template_revision(decode_spec(encoded).model)
    # Histories are also returned to callers. save_for_backward retains the
    # exact same storage; it does not clone or duplicate requested records.
    ctx.primary_psc = decode_spec(encoded).model in (
        "alpha",
        "alpha_billeh",
        "exponential",
        "dual_exponential",
    )
    ctx.save_for_backward(
        x_seq,
        *states,
        *parameters,
        *([output[0]] if residual.save_output else []),
        *([output[0]] if ctx.primary_psc else []),
        *output[1 + len(states) :],
        dt,
    )
    ctx.set_materialize_grads(False)


def _backward(ctx, grad_results):
    if torch.is_grad_enabled():
        raise NotImplementedError(
            "Cell sequence kernels support first-order gradients only; "
            "create_graph=True is unsupported"
        )
    saved = ctx.saved_tensors
    x_seq = saved[0]
    states = list(saved[1 : 1 + ctx.n_states])
    parameters = list(saved[1 + ctx.n_states : 1 + ctx.n_states + ctx.n_parameters])
    offset = 1 + ctx.n_states + ctx.n_parameters
    residual_output = saved[offset] if ctx.save_residual_output else None
    offset += int(ctx.save_residual_output)
    saved_outputs = list(saved[offset:-1])
    psc_history = saved_outputs.pop(0) if ctx.primary_psc else None
    histories = saved_outputs[: ctx.n_histories]
    residuals = saved_outputs[ctx.n_histories :]
    grad_output = grad_results[0]
    grad_final = list(grad_results[1 : 1 + ctx.n_states])
    grad_history = list(
        grad_results[1 + ctx.n_states : 1 + ctx.n_states + ctx.n_histories]
    )
    history_names = ctx.history_names
    if ctx.primary_psc:
        histories.append(psc_history)
        grad_history.append(None)
        history_names = history_names + ("|" if history_names else "") + "psc"
    needs_x, needs_states, needs_parameters = ctx.needs_input_grad[:3]
    needs = [needs_x, *needs_states, *needs_parameters]
    results = _reverse(
        x_seq,
        states,
        parameters,
        ctx.encoded,
        history_names,
        histories,
        residuals,
        grad_output,
        ctx.residual_encoded,
        residual_output,
        grad_final,
        grad_history,
        ctx.backend,
        needs,
        ctx.enable_fp_fusion,
        ctx.precise_division,
        ctx.precise_exp,
        ctx.backend_revision,
        ctx.model_revision,
        saved[-1],
    )
    results = [value if needed else None for value, needed in zip(results, needs)]
    gradients = (
        results[0],
        results[1 : 1 + ctx.n_states],
        results[1 + ctx.n_states :],
        None,
        None,
        None,
        None,
    )
    return (
        *gradients,
        *(None for _ in range(len(ctx.needs_input_grad) - len(gradients))),
    )


_training.register_autograd(_backward, setup_context=_setup_context)


def run_multi_step(
    x_seq: Tensor,
    states: tuple[Tensor, ...],
    parameters: tuple[Tensor, ...],
    spec: CellSpec,
    history: HistoryPlan = HistoryPlan(),
    *,
    backend: str | KernelOptions = "torch",
    uniform_seq: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """Run an FP32 state sequence, choosing training from actual autograd
    inputs.

    Args:
        x_seq: Input current or connection drive in time-first layout.
        states: Initial state tensors in :class:`ModelTemplate` order.
        parameters: Participating tensor parameters, retaining their graphs.
        spec: Immutable equation and surrogate options.
        history: Post-step or charged-state records requested by the caller.
        backend: Registered cell backend name or explicit kernel options.
        uniform_seq: Optional explicit Poisson uniforms for validation. Native
            stochastic execution draws from Torch's global generator when None.

    Returns:
        Flat tuple of output, final states and requested histories.
    """
    options = resolve_model_options(backend, spec.model)
    backend = options.neuron
    _validate_compile_policy(
        backend,
        options.enable_fp_fusion,
        options.precise_division,
        options.precise_exp,
    )
    template = get_template(spec)
    if len(states) != len(template.state_names) or len(parameters) != len(
        template.parameter_names
    ):
        raise ValueError("State or parameter count disagrees with the model template")
    if x_seq.ndim < 2 or x_seq.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("Cell input requires time-first FP32, FP16 or BF16 tensors")
    for name in history.names:
        if name not in template.history_names:
            raise ValueError(f"Unsupported history {name!r} for {spec.model}")
    for name, value in zip(template.state_names, states):
        extra_dims = dict(template.state_extra_dims).get(name, 0)
        expected = (
            (*x_seq.shape[1:], *value.shape[-extra_dims:])
            if extra_dims
            else x_seq.shape[1:]
        )
        if (
            value.shape != expected
            or value.device != x_seq.device
            or not value.is_floating_point()
        ):
            raise ValueError(f"Invalid initial {name} shape, dtype or device")
    for value in parameters:
        if value.device != x_seq.device or not value.is_floating_point():
            raise ValueError(
                "Cell parameters must be floating tensors on the input device"
            )
    stochastic = spec.surrogate == "poisson" and spec.poisson is not None
    precision_tensors = [
        ("x_seq", x_seq),
        *(
            (f"initial {name}", value)
            for name, value in zip(template.state_names, states)
        ),
        *(
            (f"parameter {name}", value)
            for name, value in zip(template.parameter_names, parameters)
        ),
    ]
    if uniform_seq is not None:
        precision_tensors.append(("uniform_seq", uniform_seq))
    _validate_poisson_fp32(spec, precision_tensors)
    if isinstance(spec.dt, Tensor):
        if spec.dt.ndim or not spec.dt.is_floating_point():
            raise ValueError("dt tensors must be floating-point scalars")
        if (
            stochastic
            and spec.dt.device.type == "cpu"
            and (not torch.isfinite(spec.dt).item() or spec.dt.item() <= 0)
        ):
            raise ValueError("dt must be finite and positive")
    if stochastic and not x_seq.shape[0]:
        raise ValueError("Poisson cell sequences require at least one timestep")
    if stochastic and backend == "torch":
        raise NotImplementedError(
            "Poisson tensor lowering requires a native GPU backend; plain Torch "
            "Modules remain the stochastic reference"
        )
    if stochastic and backend not in _NATIVE_GPU_BACKENDS:
        raise NotImplementedError(
            "Poisson tensor lowering currently supports the built-in native "
            "GPU backends only"
        )
    if stochastic:
        target_shape = x_seq.shape[1:]
        _validate_parameter_shapes(parameters, spec, states, target_shape)
        _backend(backend, spec, x_seq.device)
        expected = (x_seq.shape[0], *x_seq.shape[1:])
        if uniform_seq is not None and not spec.poisson.spiking:
            raise ValueError("Non-spiking Poisson execution does not use uniforms")
        if uniform_seq is None and spec.poisson.spiking:
            uniform_seq = poisson_uniforms(x_seq)
        elif uniform_seq is not None and (
            uniform_seq.shape != expected
            or uniform_seq.device != x_seq.device
            or uniform_seq.dtype != torch.float32
            or uniform_seq.requires_grad
        ):
            raise ValueError(
                "Poisson uniforms must be non-differentiable FP32 values with "
                "the time-first input shape and device"
            )
    elif uniform_seq is not None:
        raise ValueError("Explicit uniforms require a Poisson surrogate")
    if uniform_seq is None:
        uniform_seq = x_seq.new_empty(0, dtype=torch.float32)
    needs_backward = torch.is_grad_enabled() and any(
        value.requires_grad for value in (x_seq, *states, *parameters)
    )
    callbacks = get_reference_callbacks(spec)
    if (
        needs_backward
        and backend == "torch"
        and callbacks is not None
        and callbacks[1] is None
    ):
        raise NotImplementedError(
            f"Model template {spec.model!r} does not provide a backward_step; "
            "use inference mode or register a backend with a backward implementation"
        )
    if (
        needs_backward
        and backend not in ("torch", "triton", "tilelang", "cupy")
        and _backend(backend, spec, x_seq.device).backward is None
    ):
        raise NotImplementedError(
            f"Cell backend {backend!r} does not provide a backward implementation"
        )
    checkpoint_supported = (
        needs_backward
        and backend in _NATIVE_GPU_BACKENDS
        and x_seq.device.type == "cuda"
    )
    residual_names = tuple(dict.fromkeys((*history.requested, *history.backward)))
    residual = (
        residual_template(spec, residual_names)
        if checkpoint_supported
        else ResidualTemplate(template.state_names)
        if needs_backward
        else None
    )
    names = (
        tuple(
            dict.fromkeys((*history.requested, *residual.histories, *history.backward))
        )
        if residual is not None
        else history.requested
    )
    primary_psc = spec.model in (
        "alpha",
        "alpha_billeh",
        "exponential",
        "dual_exponential",
    )
    if primary_psc:
        names = tuple(name for name in names if name != "psc")
    operator = _training if needs_backward else _inference
    arguments = (
        x_seq,
        list(states),
        list(parameters),
        encode_spec(spec),
        "|".join(names) + (("#" + residual.encoded) if residual is not None else ""),
    )
    result = operator(
        *arguments,
        backend,
        spec.dt.to(device=x_seq.device, dtype=torch.float32)
        if isinstance(spec.dt, Tensor)
        else torch.full((), spec.dt, dtype=torch.float32, device=x_seq.device),
        options.enable_fp_fusion,
        options.precise_division,
        uniform_seq,
        options.precise_exp,
    )
    n_states = len(states)
    record_map = dict(zip(names, result[1 + n_states :]))
    if primary_psc:
        record_map["psc"] = result[0]
    return (
        result[0].to(x_seq.dtype),
        *result[1 : 1 + n_states],
        *(record_map[name] for name in history.requested),
    )
