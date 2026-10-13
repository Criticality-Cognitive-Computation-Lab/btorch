"""Register complete sequence boundaries with fake kernels and explicit
VJPs."""

import math

import torch
from torch import Tensor

from .contracts import CellSpec, HistoryPlan
from .registry import cell_backend_revision
from .templates import get_reference_callbacks, get_template, model_template_revision


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


def encode_spec(spec: CellSpec) -> str:
    """Encode immutable compile-specialized scalar choices."""
    return "|".join(
        (
            spec.model,
            str(int(spec.hard_reset)),
            str(int(spec.detach_reset)),
            spec.surrogate,
            str(spec.alpha),
            str(spec.damping),
            str(int(spec.refractory)),
        )
    )


def decode_spec(encoded: str, dt: float | Tensor = 1.0) -> CellSpec:
    """Restore a cell specification inside the tensor operator boundary."""
    model, hard_reset, detach_reset, surrogate, alpha, damping, refractory = (
        encoded.split("|")
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
    )


def _backend(name, spec, device):
    from .registry import resolve_cell_backend

    return resolve_cell_backend(name, spec, device)


def _forward(x_seq, states, parameters, encoded, history_names, backend, dt):
    spec = decode_spec(encoded, dt)
    torch._assert_async(torch.isfinite(dt) & (dt > 0), "dt must be finite and positive")
    if spec.model == "alpha_billeh":
        torch._assert_async(dt == 1, "AlphaPSCBilleh requires dt == 1")
    output, final, histories = _backend(backend, spec, x_seq.device).forward(
        x_seq,
        tuple(states),
        tuple(parameters),
        spec,
        tuple(history_names.split("|")) if history_names else (),
    )
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
) -> list[Tensor]:
    return _forward(x_seq, states, parameters, encoded, history_names, backend, dt)


@torch.library.custom_op("btorch_cells::training", mutates_args=())
def _training(
    x_seq: Tensor,
    states: list[Tensor],
    parameters: list[Tensor],
    encoded: str,
    history_names: str,
    backend: str,
    dt: Tensor,
) -> list[Tensor]:
    return _forward(x_seq, states, parameters, encoded, history_names, backend, dt)


def _fake_forward(x_seq, states, parameters, encoded, history_names, backend, dt):
    template = get_template(decode_spec(encoded))
    state_map = dict(zip(template.state_names, states))
    records = []
    for name in history_names.split("|") if history_names else ():
        source = state_map.get(name)
        if source is None:
            source = state_map[dict(template.history_sources)[name]]
        records.append(
            source.new_empty((x_seq.shape[0], *source.shape), dtype=torch.float32)
        )
    return [
        torch.empty_like(x_seq, dtype=torch.float32),
        *(torch.empty_like(value, dtype=torch.float32) for value in states),
        *records,
    ]


_inference.register_fake(_fake_forward)
_training.register_fake(_fake_forward)


@torch.library.custom_op(
    "btorch_cells::reverse",
    mutates_args=(),
    schema="(Tensor x_seq, Tensor[] states, Tensor[] parameters, str encoded, "
    "str history_names, Tensor[] histories, Tensor grad_output, "
    "Tensor[] grad_final, Tensor[] grad_history, str backend, bool[] needs, "
    "int backend_revision, int model_revision, Tensor dt) "
    "-> Tensor?[]",
)
def _reverse(
    x_seq: Tensor,
    states: list[Tensor],
    parameters: list[Tensor],
    encoded: str,
    history_names: str,
    histories: list[Tensor],
    grad_output: Tensor,
    grad_final: list[Tensor],
    grad_history: list[Tensor],
    backend: str,
    needs: list[bool],
    backend_revision: int,
    model_revision: int,
    dt: Tensor,
) -> list[Tensor | None]:
    spec = decode_spec(encoded, dt)
    if (
        cell_backend_revision(backend) != backend_revision
        or model_template_revision(spec.model) != model_revision
    ):
        raise RuntimeError(
            "Cell registration changed after forward; complete all outstanding "
            "backward calls before replacing a backend or model template"
        )
    gradients = _backend(backend, spec, x_seq.device).backward(
        x_seq,
        tuple(states),
        tuple(parameters),
        spec,
        tuple(history_names.split("|")) if history_names else (),
        tuple(histories),
        grad_output,
        tuple(grad_final),
        tuple(grad_history),
        needs_input_grad=tuple(needs),
    )
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
    grad_output,
    grad_final,
    grad_history,
    backend,
    needs,
    backend_revision,
    model_revision,
    dt,
):
    return [
        torch.empty_like(value) if needed else None
        for value, needed in zip((x_seq, *states, *parameters), needs)
    ]


def _setup_context(ctx, inputs, output):
    x_seq, states, parameters, encoded, history_names, backend, dt = inputs
    ctx.n_states = len(states)
    ctx.n_parameters = len(parameters)
    ctx.encoded = encoded
    ctx.history_names = history_names
    ctx.backend = backend
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
    histories = list(saved[1 + ctx.n_states + ctx.n_parameters : -1])
    grad_output = (
        grad_results[0] if grad_results[0] is not None else torch.zeros_like(x_seq)
    )
    grad_final = [
        g if g is not None else torch.zeros_like(s)
        for g, s in zip(grad_results[1 : 1 + ctx.n_states], states)
    ]
    history_outputs = histories[1:] if ctx.primary_psc else histories
    grad_history = [
        g if g is not None else torch.zeros_like(h)
        for g, h in zip(grad_results[1 + ctx.n_states :], history_outputs)
    ]
    history_names = ctx.history_names
    if ctx.primary_psc:
        psc_history, histories = histories[0], histories[1:]
        histories.append(psc_history)
        grad_history.append(torch.zeros_like(psc_history))
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
        grad_output,
        grad_final,
        grad_history,
        ctx.backend,
        needs,
        ctx.backend_revision,
        ctx.model_revision,
        saved[-1],
    )
    results = [value if needed else None for value, needed in zip(results, needs)]
    return (
        results[0],
        results[1 : 1 + ctx.n_states],
        results[1 + ctx.n_states :],
        None,
        None,
        None,
        None,
    )


_training.register_autograd(_backward, setup_context=_setup_context)


def run_multi_step(
    x_seq: Tensor,
    states: tuple[Tensor, ...],
    parameters: tuple[Tensor, ...],
    spec: CellSpec,
    history: HistoryPlan = HistoryPlan(),
    *,
    backend: str = "torch",
) -> tuple[Tensor, ...]:
    """Run an FP32 state sequence, choosing training from actual autograd
    inputs.

    Args:
        x_seq: Input current or connection drive in time-first layout.
        states: Initial state tensors in :class:`ModelTemplate` order.
        parameters: Participating tensor parameters, retaining their graphs.
        spec: Immutable equation and surrogate options.
        history: Post-step or charged-state records requested by the caller.
        backend: Registered cell backend name.

    Returns:
        Flat tuple of output, final states and requested histories.
    """
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
    names = (
        tuple(
            dict.fromkeys(
                (*history.requested, *template.state_names, *history.backward)
            )
        )
        if needs_backward
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
    result = operator(
        x_seq,
        list(states),
        list(parameters),
        encode_spec(spec),
        "|".join(names),
        backend,
        spec.dt.to(device=x_seq.device, dtype=torch.float32)
        if isinstance(spec.dt, Tensor)
        else torch.full((), spec.dt, dtype=torch.float32, device=x_seq.device),
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
