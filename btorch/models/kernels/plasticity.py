"""Native canonical-edge STDP updates using the shared scalar renderers."""

from __future__ import annotations

import functools

import torch

from .backends.engine import (
    _compile,
    _launch_kernel,
    _Source,
    _storage_span,
    _tensor_storage,
)
from .backends.renderers import get_runtime


PARAMETERS = ("a_plus", "a_minus", "tau_plus", "tau_minus", "w_min", "w_max")


def update(
    backend,
    weight,
    pre_trace,
    post_trace,
    arrivals,
    posts,
    pre_index,
    post_index,
    delay_index,
    n_delay,
    parameters,
    dt,
    *,
    records=None,
):
    """Apply one owned per-edge update without separate gathers or trace
    passes.

    Integer topology stays integer; FP64 weights use FP64 arithmetic.
    FP16 and BF16 storage use FP32 registers and round on store.
    Canonical weights and trace buffers are mutable; all event tensors
    and rule parameters are read only. A recurrent caller must consume
    old weights before invoking this.
    """
    if backend not in ("triton", "tilelang"):
        raise NotImplementedError("Native STDP supports Triton and TileLang")
    if not weight.numel():
        return
    buffers = dict(
        weight=weight,
        pre_trace=pre_trace,
        post_trace=post_trace,
        arrivals=arrivals,
        posts=posts,
        pre_index=pre_index,
        post_index=post_index,
        dt=dt,
    )
    if delay_index is not None:
        buffers["delay_index"] = delay_index
    buffers.update(parameters)
    buffers.update(
        {name + "_history": value for name, value in (records or {}).items()}
    )
    rendered, declarations = _render(
        backend,
        weight.numel(),
        n_delay,
        _tensor_storage(buffers),
        tuple((name, value.numel()) for name, value in buffers.items()),
        "float64" if weight.dtype == torch.float64 else "float32",
    )
    kernel = _compile(
        backend,
        rendered,
        False,
        weight.dtype != torch.float64,
        device=weight.device,
    )
    arguments = tuple(_storage_span(buffers[name], size) for name, size in declarations)
    _launch_kernel(
        get_runtime(backend),
        kernel,
        arguments,
        weight.numel(),
        enable_fp_fusion=False,
    )


@functools.lru_cache(maxsize=128)
def _render(backend, n, n_delay, storage, sizes, scalar_dtype):
    """Reuse source for a prepared layout independently of tensor addresses."""
    source = _Source(
        backend,
        n,
        1,
        0,
        storage=storage,
        precise_exp=True,
        scalar_dtype=scalar_dtype,
    )
    sizes = dict(sizes)
    for name, size in sizes.items():
        source.pointer(name, size)
    source.load("pre", "pre_index", "index", dtype="int64")
    source.load("post", "post_index", "index", dtype="int64")
    if "delay_index" not in sizes:
        delay = "0"
    else:
        source.load("delay", "delay_index", "index", dtype="int64")
        delay = "delay"
    source.load("arrival", "arrivals", f"pre * {n_delay} + {delay}")
    source.load("post_event", "posts", "post")
    for name in ("weight", "pre_trace", "post_trace"):
        source.load(name, name, "index")
    for name in PARAMETERS:
        source.load(name, name, "0" if sizes[name] == 1 else "index")
    source.load("dt", "dt", "0")
    source.statements(
        (
            "x = pre_trace * exp(-dt / tau_plus)",
            "y = post_trace * exp(-dt / tau_minus)",
            "delta = a_plus * post_event * x - a_minus * arrival * y",
            "next_weight = maximum(w_min, minimum(w_max, weight + delta))",
            "next_pre = x + arrival",
            "next_post = y + post_event",
        )
    )
    source.store("weight", "index", "next_weight")
    source.store("pre_trace", "index", "next_pre")
    source.store("post_trace", "index", "next_post")
    for name, value in (
        ("pre_trace", "next_pre"),
        ("post_trace", "next_post"),
        ("weight", "next_weight"),
    ):
        if name + "_history" in sizes:
            source.store(name + "_history", "index", value)
    return source.render(), tuple(source.pointers)
