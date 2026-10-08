"""Registered custom operators: the ``torch.compile`` boundary.

Model code hands raw tensor buffers to these operators; everything behind
them (representation, algorithm, backend) is opaque to Dynamo and to autograd,
which see one node with an explicit backward. The operators therefore compile
with ``fullgraph=True`` and never expose PyTorch sparse tensors to the tracer.

All operators share one layout: the operator ``A`` (``M x N``) as
destination-major CSR ``(crow, col)`` with ``values [*vb, E]``, plus the
source-major CSR of the same edges ``(t_crow, t_col, t_perm)`` used by the
backward and by source-driven kernels. Inputs are dense ``x [*vb, *sample,
N]``; the logical spike tensor stays dense so surrogate gradients are exact.
"""

from __future__ import annotations

from itertools import count

import torch
from torch import Tensor
from torch.library import custom_op

from . import kernels_aten, kernels_triton, kernels_triton_push
from .backend import (
    RegisteredOperator,
    RouteBinding,
    RouteImplementation,
    RouteKey,
    RouteSpec,
    registry,
)


registry.register("csr_matvec", "aten", kernels_aten.csr_matvec)
registry.register("edge_grad", "aten", kernels_aten.edge_grad)
registry.register("spike_push", "aten", kernels_aten.spike_push)
registry.register(
    "csr_matvec",
    "torch_sparse",
    kernels_aten.csr_matvec_torch_sparse,
    priority=-1,
    available=kernels_aten.torch_sparse_available,
)


# Optional CUDA backends; each checks its own availability lazily.
kernels_triton.register(registry)
kernels_triton_push.register(registry)


def _out_shape(values: Tensor, x: Tensor, n_out: int) -> tuple[int, ...]:
    n_vb = values.ndim - 1
    batch = torch.broadcast_shapes(values.shape[:-1], x.shape[:n_vb])
    return (*batch, *x.shape[n_vb:-1], n_out)


def _kernel(name: str, x: Tensor):
    return registry.resolve(name, x.device.type)


def _run(name: str, x: Tensor, *args):
    """Call kernel ``name`` of the device of ``x`` with ``args``.

    Kernels run in the dtypes they are given, independent of ambient
    AMP: otherwise an autocast region would change the output dtype of
    the eager kernel but not of its traced (fake) counterpart. The
    autocast context is only entered when there is something to disable;
    entering it costs about as much as launching a small kernel.
    """
    device = x.device.type
    kernel = registry.resolve(name, device)
    if torch.is_autocast_enabled(device):
        with torch.autocast(device_type=device, enabled=False):
            return kernel(*args)
    return kernel(*args)


def _check_real(values: Tensor, x: Tensor) -> None:
    if values.is_complex() or x.is_complex():
        raise TypeError("Sparse propagation operators support real dtypes only.")


def _fake_out(values: Tensor, x: Tensor, n_out: int) -> Tensor:
    _check_real(values, x)
    dtype = torch.promote_types(values.dtype, x.dtype)
    return x.new_empty(_out_shape(values, x, n_out), dtype=dtype)


# --------------------------------------------------------------- raw kernels
# Not differentiable themselves; they are the building blocks of the backward.


@custom_op("btorch::csr_matvec", mutates_args=())
def csr_matvec(crow: Tensor, col: Tensor, values: Tensor, x: Tensor) -> Tensor:
    """``y[..., m] = sum_{k in row m} values[..., k] * x[..., col[k]]``."""
    return _run("csr_matvec", x, crow, col, values, x)


@csr_matvec.register_fake
def _(crow, col, values, x):
    return _fake_out(values, x, crow.shape[0] - 1)


@custom_op("btorch::csr_edge_grad", mutates_args=())
def csr_edge_grad(
    crow: Tensor, col: Tensor, grad: Tensor, x: Tensor, n_vb: int
) -> Tensor:
    """Per-entry gradient ``sum_samples grad[..., row[k]] * x[...,
    col[k]]``."""
    return _run("edge_grad", x, crow, col, grad, x, n_vb)


@csr_edge_grad.register_fake
def _(crow, col, grad, x, n_vb):
    return grad.new_empty(*grad.shape[:n_vb], col.shape[0])


# ------------------------------------------------------ differentiable ops


def _pull(crow, col, values, x):
    _check_real(values, x)
    return _run("csr_matvec", x, crow, col, values, x)


def _push_or_pull_kernels(
    crow, col, values, x, t_crow, t_col, t_perm, max_density, task_values=None
):
    n_out = crow.shape[0] - 1
    if values.ndim == 1 and x.numel() > 0:
        flat = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
        dtype = values.dtype
        if x.dtype != dtype:
            dtype = torch.promote_types(dtype, x.dtype)
        if registry.has("spike_push_dense", x.device.type):
            # Compaction happens on the device inside the kernel: no host
            # synchronisation and no data-dependent launch shape, so this
            # path can be captured in a CUDA graph.
            push = _kernel("spike_push_dense", x)
            out = push(
                t_crow,
                t_col,
                t_perm,
                values,
                flat,
                n_out,
                task_values=task_values,
            )
        else:
            active_idx, ptr = kernels_aten.pack_spikes(flat)
            if active_idx.shape[0] > max_density * flat.numel():
                return _kernel("csr_matvec", x)(crow, col, values, x)
            push = _kernel("spike_push", x)
            out = push(t_crow, t_col, t_perm, values, flat, active_idx, ptr, n_out)
        if out.dtype != dtype:
            out = out.to(dtype)
        return out if x.ndim == 2 else out.reshape(*x.shape[:-1], n_out)
    return _kernel("csr_matvec", x)(crow, col, values, x)


def _push_or_pull(
    crow, col, values, x, t_crow, t_col, t_perm, max_density, task_values=None
):
    _check_real(values, x)
    args = (crow, col, values, x, t_crow, t_col, t_perm, max_density, task_values)
    device = x.device.type
    if torch.is_autocast_enabled(device):  # see ``_run``
        with torch.autocast(device_type=device, enabled=False):
            return _push_or_pull_kernels(*args)
    return _push_or_pull_kernels(*args)


def _fused_backward(device: str):
    """The ``"csr_backward"`` kernel of the selected backend, if it has one.

    A backend may provide both gradients in one optional kernel. It is used
    only when that same backend is also the one selected for the two
    kernels it replaces, so forcing or registering another backend for
    ``"csr_matvec"`` or ``"edge_grad"`` keeps taking effect in the backward.
    """
    if not registry.has("csr_backward", device):
        return None
    name = registry.name("csr_backward", device)
    if name != registry.name("csr_matvec", device):
        return None
    if name != registry.name("edge_grad", device):
        return None
    return registry.resolve("csr_backward", device)


def _grad_kernels(saved, grad, need_values: bool, need_x: bool):
    crow, col, values, x, t_crow, t_col, t_perm = saved
    device = x.device.type
    grad_values = grad_x = None
    fused = _fused_backward(device)
    if fused is not None:
        grad_values, grad_x = fused(
            crow, col, t_crow, t_col, t_perm, values, x, grad, need_values, need_x
        )
    else:
        if need_values:
            edge_grad = registry.resolve("edge_grad", device)
            grad_values = edge_grad(crow, col, grad, x, values.ndim - 1)
        if need_x:
            # Transposed product as a pull over the source-major CSR: no
            # atomics, deterministic, and always a dense gradient w.r.t. the
            # input.
            matvec = registry.resolve("csr_matvec", device)
            grad_x = matvec(t_crow, t_col, values.index_select(-1, t_perm), grad)
    if grad_values is not None:
        if grad_values.shape != values.shape:
            grad_values = grad_values.sum_to_size(values.shape)
        if grad_values.dtype != values.dtype:
            grad_values = grad_values.to(values.dtype)
    if grad_x is not None:
        if grad_x.shape != x.shape:
            grad_x = grad_x.sum_to_size(x.shape)
        if grad_x.dtype != x.dtype:
            grad_x = grad_x.to(x.dtype)
    return grad_values, grad_x


def _grads(saved, grad, need_values: bool, need_x: bool):
    """Shared backward: sampled product for values, transposed pull for x.

    ``saved`` is ``(crow, col, values, x, t_crow, t_col, t_perm)``. Returns
    ``(grad_values, grad_x)`` with the shapes and dtypes of ``values`` and
    ``x``; an entry that is not needed is ``None``.
    """
    device = saved[3].device.type
    if torch.is_autocast_enabled(device):  # see ``_run``
        with torch.autocast(device_type=device, enabled=False):
            return _grad_kernels(saved, grad, need_values, need_x)
    return _grad_kernels(saved, grad, need_values, need_x)


@custom_op("btorch::csr_propagate_backward", mutates_args=())
def csr_propagate_backward(
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    grad: Tensor,
    need_values: bool,
    need_x: bool,
) -> tuple[Tensor, Tensor]:
    """Both gradients of a propagation in one operator call.

    The backward of :func:`csr_propagate` and :func:`spike_propagate` in
    compiled graphs: one operator boundary instead of one per gradient. Not
    differentiable itself.

    Args:
        crow: ``[M + 1]`` row pointers of the destination-major CSR.
        col: ``[E]`` source index of each entry.
        values: ``[*vb, E]`` entry values of the forward.
        x: ``[*vb, *sample, N]`` input of the forward.
        t_crow: ``[N + 1]`` pointers of the source-major CSR.
        t_col: ``[E]`` destination index of each source-major entry.
        t_perm: ``[E]`` CSR position of each source-major entry.
        grad: ``[*vb, *sample, M]`` gradient w.r.t. the output.
        need_values: Compute the gradient w.r.t. ``values``.
        need_x: Compute the gradient w.r.t. ``x``.

    Returns:
        ``(grad_values, grad_x)`` shaped like ``values`` and ``x``. A
        gradient that is not needed is an empty ``[0]`` tensor.
    """
    saved = (crow, col, values, x, t_crow, t_col, t_perm)
    grad_values, grad_x = _grads(saved, grad, need_values, need_x)
    if grad_values is None:
        grad_values = values.new_empty(0)
    if grad_x is None:
        grad_x = x.new_empty(0)
    return grad_values, grad_x


def _bound_grads(
    saved,
    grad: Tensor,
    need_values: bool,
    need_x: bool,
    edge_grad,
    matvec,
    fused=None,
):
    """Compute backward with the kernels captured by one route binding."""
    crow, col, values, x, t_crow, t_col, t_perm = saved
    grad_values = grad_x = None
    if fused is not None:
        grad_values, grad_x = fused(
            crow,
            col,
            t_crow,
            t_col,
            t_perm,
            values,
            x,
            grad,
            need_values,
            need_x,
        )
    else:
        if need_values:
            grad_values = edge_grad(crow, col, grad, x, values.ndim - 1)
        if need_x:
            grad_x = matvec(t_crow, t_col, values.index_select(-1, t_perm), grad)
    if grad_values is not None:
        if grad_values.shape != values.shape:
            grad_values = grad_values.sum_to_size(values.shape)
        if grad_values.dtype != values.dtype:
            grad_values = grad_values.to(values.dtype)
    if grad_x is not None:
        if grad_x.shape != x.shape:
            grad_x = grad_x.sum_to_size(x.shape)
        if grad_x.dtype != x.dtype:
            grad_x = grad_x.to(x.dtype)
    return grad_values, grad_x


def _build_pull(kernels: dict[str, object]) -> RouteImplementation:
    matvec = kernels["csr_matvec"]
    edge_grad = kernels["edge_grad"]
    fused = kernels.get("csr_backward")
    prepare = kernels.get("prepare_pull")

    def forward(crow, col, values, x, t_crow, t_col, t_perm, max_density, task_values):
        return matvec(crow, col, values, x)

    def backward(
        crow,
        col,
        values,
        x,
        t_crow,
        t_col,
        t_perm,
        grad,
        need_values,
        need_x,
    ):
        return _bound_grads(
            (crow, col, values, x, t_crow, t_col, t_perm),
            grad,
            need_values,
            need_x,
            edge_grad,
            matvec,
            fused,
        )

    return RouteImplementation(forward, backward, prepare)


def _build_adaptive_push(kernels: dict[str, object]) -> RouteImplementation:
    push = kernels["spike_push"]
    matvec = kernels["csr_matvec"]
    edge_grad = kernels["edge_grad"]

    def forward(crow, col, values, x, t_crow, t_col, t_perm, max_density, task_values):
        if values.ndim != 1 or x.numel() == 0:
            return matvec(crow, col, values, x)
        flat = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
        active_idx, ptr = kernels_aten.pack_spikes(flat)
        if active_idx.shape[0] > max_density * flat.numel():
            return matvec(crow, col, values, x)
        out = push(
            t_crow,
            t_col,
            t_perm,
            values,
            flat,
            active_idx,
            ptr,
            crow.shape[0] - 1,
        )
        return out if x.ndim == 2 else out.reshape(*x.shape[:-1], crow.shape[0] - 1)

    def backward(
        crow,
        col,
        values,
        x,
        t_crow,
        t_col,
        t_perm,
        grad,
        need_values,
        need_x,
    ):
        return _bound_grads(
            (crow, col, values, x, t_crow, t_col, t_perm),
            grad,
            need_values,
            need_x,
            edge_grad,
            matvec,
        )

    return RouteImplementation(forward, backward)


def _build_dense_push(kernels: dict[str, object]) -> RouteImplementation:
    push = kernels["spike_push_dense"]
    matvec = kernels["csr_matvec"]
    prepare = kernels.get("prepare_push")
    prepare_values = kernels.get("prepare_task_values")
    fused = kernels.get("csr_backward")

    def forward(crow, col, values, x, t_crow, t_col, t_perm, max_density, task_values):
        if values.ndim != 1 or x.numel() == 0:
            return matvec(crow, col, values, x)
        flat = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
        out = push(
            t_crow,
            t_col,
            t_perm,
            values,
            flat,
            crow.shape[0] - 1,
            task_values=(
                task_values if task_values is not None and task_values.numel() else None
            ),
            require_task=True,
        )
        return out if x.ndim == 2 else out.reshape(*x.shape[:-1], crow.shape[0] - 1)

    def backward(
        crow,
        col,
        values,
        x,
        t_crow,
        t_col,
        t_perm,
        grad,
        need_values,
        need_x,
    ):
        return _bound_grads(
            (crow, col, values, x, t_crow, t_col, t_perm),
            grad,
            need_values,
            need_x,
            kernels["edge_grad"] if "edge_grad" in kernels else kernels["csr_backward"],
            matvec,
            fused,
        )

    return RouteImplementation(forward, backward, prepare, prepare_values)


@csr_propagate_backward.register_fake
def _(crow, col, values, x, t_crow, t_col, t_perm, grad, need_values, need_x):
    grad_values = values.new_empty(values.shape if need_values else (0,))
    grad_x = x.new_empty(x.shape if need_x else (0,))
    return grad_values, grad_x


def _setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs[:7])


def _backward(ctx, grad: Tensor):
    need = ctx.needs_input_grad
    grad_values, grad_x = csr_propagate_backward(
        *ctx.saved_tensors, grad, need[2], need[3]
    )
    return (
        None,
        None,
        grad_values if need[2] else None,
        grad_x if need[3] else None,
    ) + (None,) * (len(need) - 4)


@custom_op("btorch::csr_propagate", mutates_args=())
def csr_propagate(
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
) -> Tensor:
    """Destination-driven sparse propagation ``y = A @ x`` along the last axis.

    Args:
        crow: ``[M + 1]`` row pointers of the destination-major CSR.
        col: ``[E]`` source index of each entry.
        values: ``[*vb, E]`` entry values in CSR order (differentiable).
        x: ``[*vb, *sample, N]`` dense input (differentiable).
        t_crow: ``[N + 1]`` pointers of the source-major CSR.
        t_col: ``[E]`` destination index of each source-major entry.
        t_perm: ``[E]`` CSR position of each source-major entry.

    Returns:
        ``[*vb, *sample, M]``.
    """
    return _pull(crow, col, values, x)


@csr_propagate.register_fake
def _(crow, col, values, x, t_crow, t_col, t_perm):
    return _fake_out(values, x, crow.shape[0] - 1)


csr_propagate.register_autograd(_backward, setup_context=_setup_context)


@custom_op("btorch::spike_propagate", mutates_args=())
def spike_propagate(
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    max_density: float,
    task_values: Tensor | None = None,
) -> Tensor:
    """Sparse propagation that exploits sparse activity in ``x``.

    Mathematically identical to :func:`csr_propagate`. The forward visits
    only the out-edges of active sources. With a backend that compacts the
    input on the device (Triton on CUDA) this is a single launch without host
    synchronisation. The reference path packs the non-zero entries on the
    host side and falls back to the destination-driven product when the input
    density exceeds ``max_density``. The backward is the same dense one, so
    gradients w.r.t. ``x`` are exact for every entry, including the silent
    ones.

    Args:
        crow: ``[M + 1]`` row pointers of the destination-major CSR.
        col: ``[E]`` source index of each entry.
        values: ``[E]`` entry values in CSR order (differentiable).
        x: ``[*sample, N]`` dense input (differentiable).
        t_crow: ``[N + 1]`` pointers of the source-major CSR.
        t_col: ``[E]`` destination index of each source-major entry.
        t_perm: ``[E]`` CSR position of each source-major entry.
        max_density: Largest non-zero fraction of ``x`` handled by the
            host-packed source-driven path.

    Returns:
        ``[*sample, M]``.
    """
    prepared = None if task_values is None or task_values.numel() == 0 else task_values
    return _push_or_pull(
        crow, col, values, x, t_crow, t_col, t_perm, max_density, prepared
    )


@spike_propagate.register_fake
def _(crow, col, values, x, t_crow, t_col, t_perm, max_density, task_values=None):
    return _fake_out(values, x, crow.shape[0] - 1)


spike_propagate.register_autograd(_backward, setup_context=_setup_context)


# ------------------------------------------------------------ eager fast path


class _Propagate(torch.autograd.Function):
    """Eager twin of the registered operators.

    Same kernels and the same backward, without the per-call cost of the
    operator dispatch, which dominates for small networks. Never traced:
    :func:`propagate` routes compiled code to the registered operators.
    """

    @staticmethod
    def forward(
        ctx, crow, col, values, x, t_crow, t_col, t_perm, max_density, task_values
    ):
        ctx.save_for_backward(crow, col, values, x, t_crow, t_col, t_perm)
        if max_density is None:
            return _pull(crow, col, values, x)
        prepared = None if task_values.numel() == 0 else task_values
        return _push_or_pull(
            crow, col, values, x, t_crow, t_col, t_perm, max_density, prepared
        )

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            # ``create_graph=True``: the kernels are not differentiable, and a
            # silently pruned branch would give wrong second-order gradients.
            raise RuntimeError(
                "Sparse propagation does not support double backward "
                "(create_graph=True)."
            )
        need = ctx.needs_input_grad
        grad_values, grad_x = _grads(ctx.saved_tensors, grad, need[2], need[3])
        return None, None, grad_values, grad_x, None, None, None, None, None


def propagate(
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    max_density: float | None = None,
    task_values: Tensor | None = None,
) -> Tensor:
    """Sparse propagation ``y = A @ x``; the entry point connections call.

    Under ``torch.compile`` this is one registered operator
    (:func:`csr_propagate`, or :func:`spike_propagate` when ``max_density``
    is given). In eager mode the same kernels run through a lighter autograd
    node. Results and gradients are identical on both routes.

    Args:
        crow: ``[M + 1]`` row pointers of the destination-major CSR.
        col: ``[E]`` source index of each entry.
        values: ``[*vb, E]`` entry values in CSR order.
        x: ``[*vb, *sample, N]`` dense input.
        t_crow: ``[N + 1]`` pointers of the source-major CSR.
        t_col: ``[E]`` destination index of each source-major entry.
        t_perm: ``[E]`` CSR position of each source-major entry.
        max_density: ``None`` for destination-driven propagation; a density
            limit to use the source-driven path for sparse activity.

    Returns:
        ``[*vb, *sample, M]``.
    """
    if torch.compiler.is_compiling():
        if max_density is None:
            return csr_propagate(crow, col, values, x, t_crow, t_col, t_perm)
        prepared = values.new_empty(0) if task_values is None else task_values
        return spike_propagate(
            crow,
            col,
            values,
            x,
            t_crow,
            t_col,
            t_perm,
            max_density,
            prepared,
        )
    return _propagate_eager(
        crow, col, values, x, t_crow, t_col, t_perm, max_density, task_values
    )


@torch.compiler.disable
def _propagate_eager(
    crow, col, values, x, t_crow, t_col, t_perm, max_density, task_values=None
):
    """Eager route.

    Never traced: if a compiled frame is skipped and runs in
    the interpreter, Dynamo must not start compiling the kernel wrappers.
    """
    if not (torch.is_grad_enabled() and (values.requires_grad or x.requires_grad)):
        if max_density is None:
            return _pull(crow, col, values, x)
        return _push_or_pull(
            crow,
            col,
            values,
            x,
            t_crow,
            t_col,
            t_perm,
            max_density,
            task_values,
        )
    prepared = values.new_empty(0) if task_values is None else task_values
    return _Propagate.apply(
        crow, col, values, x, t_crow, t_col, t_perm, max_density, prepared
    )


# Route bindings are created during planning, outside recurrent timesteps.
# The integer token is the only route state crossing a compiled op boundary;
# its table entry is immutable for the lifetime of the process.
_BOUND_ROUTES: dict[int, RouteBinding] = {}
_BOUND_ROUTE_TOKENS: dict[tuple, int] = {}
_ROUTE_TOKENS = count(1)


def bind_route(route: RouteBinding) -> int:
    """Publish a planned route and return its stable execution token."""
    fingerprint = route.snapshot.fingerprint
    existing = _BOUND_ROUTE_TOKENS.get(fingerprint)
    if existing is not None:
        return existing
    token = next(_ROUTE_TOKENS)
    _BOUND_ROUTES[token] = route
    _BOUND_ROUTE_TOKENS[fingerprint] = token
    return token


def _route(token: int) -> RouteBinding:
    try:
        return _BOUND_ROUTES[token]
    except KeyError as error:
        raise RuntimeError(f"Unknown sparse route token {token}.") from error


@custom_op("btorch::route_propagate_backward", mutates_args=())
def route_propagate_backward(
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    grad: Tensor,
    need_values: bool,
    need_x: bool,
    route_token: int,
) -> tuple[Tensor, Tensor]:
    """Execute the backward bound to the same exact route as forward."""
    route = _route(route_token)
    grad_values, grad_x = route.implementation.backward(
        crow,
        col,
        values,
        x,
        t_crow,
        t_col,
        t_perm,
        grad,
        need_values,
        need_x,
    )
    if grad_values is None:
        grad_values = values.new_empty(0)
    if grad_x is None:
        grad_x = x.new_empty(0)
    return grad_values, grad_x


@route_propagate_backward.register_fake
def _(
    crow,
    col,
    values,
    x,
    t_crow,
    t_col,
    t_perm,
    grad,
    need_values,
    need_x,
    route_token,
):
    return (
        values.new_empty(values.shape if need_values else (0,)),
        x.new_empty(x.shape if need_x else (0,)),
    )


@custom_op("btorch::route_propagate", mutates_args=())
def route_propagate(
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    max_density: float,
    task_values: Tensor,
    route_token: int,
) -> Tensor:
    """Execute one previously bound complete sparse route."""
    route = _route(route_token)
    prepared = None if task_values.numel() == 0 else task_values
    return route.implementation.forward(
        crow,
        col,
        values,
        x,
        t_crow,
        t_col,
        t_perm,
        max_density,
        prepared,
    )


@route_propagate.register_fake
def _(
    crow,
    col,
    values,
    x,
    t_crow,
    t_col,
    t_perm,
    max_density,
    task_values,
    route_token,
):
    return _fake_out(values, x, crow.shape[0] - 1)


def _route_setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs[:7])
    ctx.route_token = inputs[9]


def _route_backward(ctx, grad: Tensor):
    need = ctx.needs_input_grad
    grad_values, grad_x = route_propagate_backward(
        *ctx.saved_tensors,
        grad,
        need[2],
        need[3],
        ctx.route_token,
    )
    return (
        None,
        None,
        grad_values if need[2] else None,
        grad_x if need[3] else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


route_propagate.register_autograd(_route_backward, setup_context=_route_setup_context)


class _BoundPropagate(torch.autograd.Function):
    """Eager autograd wrapper that keeps one exact route for backward."""

    @staticmethod
    def forward(
        ctx,
        route_token,
        crow,
        col,
        values,
        x,
        t_crow,
        t_col,
        t_perm,
        max_density,
        task_values,
    ):
        ctx.route_token = route_token
        ctx.save_for_backward(crow, col, values, x, t_crow, t_col, t_perm)
        route = _route(route_token)
        prepared = None if task_values.numel() == 0 else task_values
        return route.implementation.forward(
            crow,
            col,
            values,
            x,
            t_crow,
            t_col,
            t_perm,
            max_density,
            prepared,
        )

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Sparse propagation does not support double backward "
                "(create_graph=True)."
            )
        need = ctx.needs_input_grad
        route = _route(ctx.route_token)
        grad_values, grad_x = route.implementation.backward(
            *ctx.saved_tensors, grad, need[3], need[4]
        )
        return (
            None,
            None,
            None,
            grad_values if need[3] else None,
            grad_x if need[4] else None,
            None,
            None,
            None,
            None,
            None,
        )


def propagate_bound(
    route_token: int,
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    max_density: float | None = None,
    task_values: Tensor | None = None,
) -> Tensor:
    """Execute a route selected before the current forward call."""
    prepared = values.new_empty(0) if task_values is None else task_values
    density = 0.0 if max_density is None else max_density
    if torch.compiler.is_compiling():
        return route_propagate(
            crow,
            col,
            values,
            x,
            t_crow,
            t_col,
            t_perm,
            density,
            prepared,
            route_token,
        )
    if not (torch.is_grad_enabled() and (values.requires_grad or x.requires_grad)):
        route = _route(route_token)
        return route.implementation.forward(
            crow,
            col,
            values,
            x,
            t_crow,
            t_col,
            t_perm,
            density,
            None if prepared.numel() == 0 else prepared,
        )
    return _BoundPropagate.apply(
        route_token,
        crow,
        col,
        values,
        x,
        t_crow,
        t_col,
        t_perm,
        density,
        prepared,
    )


def propagate_binding(
    route: RouteBinding,
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    x: Tensor,
    t_crow: Tensor,
    t_col: Tensor,
    t_perm: Tensor,
    max_density: float | None = None,
    task_values: Tensor | None = None,
) -> Tensor:
    """Execute an already-bound route without a token-table lookup."""
    prepared = values.new_empty(0) if task_values is None else task_values
    density = 0.0 if max_density is None else max_density
    if not (torch.is_grad_enabled() and (values.requires_grad or x.requires_grad)):
        return route.implementation.forward(
            crow,
            col,
            values,
            x,
            t_crow,
            t_col,
            t_perm,
            density,
            None if prepared.numel() == 0 else prepared,
        )
    token = bind_route(route)
    return _BoundPropagate.apply(
        token,
        crow,
        col,
        values,
        x,
        t_crow,
        t_col,
        t_perm,
        density,
        prepared,
    )


def _register_route_specs() -> None:
    """Register complete built-in routes after all low-level kernels exist."""
    registry.register_route(
        RouteSpec(
            key=RouteKey("propagate", "post_pre_csr", "pull", "aten"),
            kernel_bindings=(("csr_matvec", "aten"), ("edge_grad", "aten")),
            build=_build_pull,
        ),
        device=("cpu", "cuda"),
        priority=0,
    )
    registry.register_route(
        RouteSpec(
            key=RouteKey("propagate", "post_pre_csr", "pull", "torch_sparse"),
            kernel_bindings=(
                ("csr_matvec", "torch_sparse"),
                ("edge_grad", "aten"),
            ),
            build=_build_pull,
        ),
        device=("cpu", "cuda"),
        priority=-1,
        available=kernels_aten.torch_sparse_available,
    )
    registry.register_route(
        RouteSpec(
            key=RouteKey("propagate", "post_pre_csr", "pull", "triton"),
            kernel_bindings=(
                ("csr_matvec", "triton"),
                ("edge_grad", "triton"),
                ("csr_backward", "triton"),
                ("prepare_pull", "triton"),
            ),
            build=_build_pull,
        ),
        device="cuda",
        priority=10,
        available=kernels_triton.is_available,
    )
    registry.register_route(
        RouteSpec(
            key=RouteKey("propagate", "post_pre_csr", "adaptive-push", "aten"),
            kernel_bindings=(
                ("spike_push", "aten"),
                ("csr_matvec", "aten"),
                ("edge_grad", "aten"),
            ),
            build=_build_adaptive_push,
            operator=RegisteredOperator(supports_capture=False),
        ),
        device=("cpu", "cuda"),
        priority=0,
    )
    registry.register_route(
        RouteSpec(
            key=RouteKey("propagate", "post_pre_csr", "push", "triton"),
            kernel_bindings=(
                ("spike_push_dense", "triton"),
                ("csr_matvec", "triton"),
                ("edge_grad", "triton"),
                ("csr_backward", "triton"),
                ("prepare_push", "triton"),
                ("prepare_task_values", "triton"),
            ),
            build=_build_dense_push,
            supports=lambda context: (
                context is not None
                and context.dtype == torch.float32
                and not context.value_batched
            ),
        ),
        device="cuda",
        priority=10,
        available=kernels_triton_push.is_available,
    )


_register_route_specs()


def pack_spikes(x: Tensor) -> tuple[Tensor, Tensor]:
    """Pack the non-zero entries of a dense spike tensor ``[B, N]``.

    The packed indices are an execution aid only: the dense tensor remains
    the logical (and differentiable) spike representation.

    Returns:
        ``(active_idx, ptr)``; sample ``b`` owns
        ``active_idx[ptr[b]:ptr[b + 1]]``.
    """
    return kernels_aten.pack_spikes(x)
