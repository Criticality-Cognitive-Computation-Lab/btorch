"""Declare an external STDP rule and validate its native RSNN execution.

Run ``python examples/oot_stdp.py`` for the Torch reference, or select
``--backend triton --device cuda --capture`` for native chunk replay.
TileLang uses the same declaration. ``--recurrent helion`` selects fused
execution; ``--compile`` checks the public recurrent operator boundary.
"""

import argparse

import torch

from btorch.models import KernelOptions, environ
from btorch.models.connection import (
    EdgeWeight,
    PlasticityRule,
    SparseConnection,
    Synapse,
)
from btorch.models.functional import init_net_state
from btorch.models.kernels import (
    PlasticityStage,
    PlasticityTemplate,
    PlasticityTrace,
    register_plasticity_template,
)
from btorch.models.neurons import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC


class BurstSTDP(PlasticityRule):
    """Use squared pre-trace potentiation with additive depression.

    Both terms read strictly earlier, decayed traces. Current events are
    accumulated afterwards. The connection applies and clips the increment.

    Args:
        a_plus: Potentiation amplitude.
        a_minus: Depression amplitude.
        tau: Positive exponential trace time constant in simulation units.
        w_min: Lower canonical-edge weight bound.
        w_max: Upper canonical-edge weight bound.
    """

    kernel_model = "tutorial_burst_stdp"
    state_domains = {"pre_trace": "edge", "post_trace": "edge"}

    def __init__(self, a_plus=0.002, a_minus=0.003, tau=20.0, w_min=0.0, w_max=0.05):
        super().__init__(w_min=w_min, w_max=w_max)
        for name, value in (("a_plus", a_plus), ("a_minus", a_minus), ("tau", tau)):
            self.register_buffer(name, torch.as_tensor(value))
        if not torch.isfinite(self.tau).all() or not (self.tau > 0).all():
            raise ValueError("tau must be finite and positive")
        self.register_memory("pre_trace", 0.0, (), persistent=True)
        self.register_memory("post_trace", 0.0, (), persistent=True)

    def single_step_forward(self, weight, events, dt):
        decay = torch.exp(-dt / self.tau)
        pre, post = (
            self.pre_trace.reshape(-1) * decay,
            self.post_trace.reshape(-1) * decay,
        )
        delta = (
            self.a_plus * events.post_spike * pre.square()
            - self.a_minus * events.arrival * post
        )
        self.pre_trace = (pre + events.arrival).reshape_as(self.pre_trace)
        self.post_trace = (post + events.post_spike).reshape_as(self.post_trace)
        return delta


def native_step(backend):
    """Return an ordinary native helper; btorch never interprets its math."""
    if backend == "triton":
        import triton
        import triton.language as tl
        from triton.language.extra.cuda import libdevice

        @triton.jit
        def step(
            weight,
            pre_trace,
            post_trace,
            a_plus,
            a_minus,
            tau,
            w_min,
            w_max,
            arrival,
            post_event,
            dt,
            INCLUDE_LTP: tl.constexpr,
            INCLUDE_LTD: tl.constexpr,
        ):
            dtype: tl.constexpr = weight.dtype
            compute_dtype: tl.constexpr = (
                tl.float64 if dtype == tl.float64 else tl.float32
            )
            ratio = (-dt / tau).to(dtype)
            decay = libdevice.exp(ratio.to(compute_dtype)).to(dtype)
            pre = (pre_trace * decay).to(dtype)
            post = (post_trace * decay).to(dtype)
            plus = tl.full((), 0.0, dtype)
            minus = tl.full((), 0.0, dtype)
            if INCLUDE_LTP:
                plus = ((a_plus * post_event).to(dtype) * (pre * pre).to(dtype)).to(
                    dtype
                )
            if INCLUDE_LTD:
                minus = ((a_minus * arrival).to(dtype) * post).to(dtype)
            delta = (plus - minus).to(dtype)
            new_weight = tl.maximum(
                w_min, tl.minimum(w_max, (weight + delta).to(dtype))
            )
            return new_weight, (pre + arrival).to(dtype), (post + post_event).to(dtype)

        return step
    if backend == "tilelang":
        import tilelang.language as T

        # TileLang calls this function with scalar expressions while building
        # its kernel. No tracing or Python callback runs at simulation time.
        def step(
            weight,
            pre_trace,
            post_trace,
            a_plus,
            a_minus,
            tau,
            w_min,
            w_max,
            arrival,
            post_event,
            dt,
            include_ltp,
            include_ltd,
        ):
            dtype = weight.dtype
            compute_dtype = "float32" if dtype in ("float16", "bfloat16") else dtype
            ratio = T.Cast(
                dtype, -T.Cast(compute_dtype, dt) / T.Cast(compute_dtype, tau)
            )
            decay = T.Cast(
                dtype,
                T.call_pure_extern(
                    compute_dtype,
                    "exp" if dtype == "float64" else "expf",
                    T.Cast(compute_dtype, ratio),
                ),
            )
            pre = T.Cast(dtype, pre_trace * decay)
            post = T.Cast(dtype, post_trace * decay)
            plus = (
                T.Cast(
                    dtype, T.Cast(dtype, a_plus * post_event) * T.Cast(dtype, pre * pre)
                )
                if include_ltp
                else T.Cast(dtype, 0.0)
            )
            minus = (
                T.Cast(dtype, T.Cast(dtype, a_minus * arrival) * post)
                if include_ltd
                else T.Cast(dtype, 0.0)
            )
            delta = T.Cast(dtype, plus - minus)
            new_weight = T.max(w_min, T.min(w_max, T.Cast(dtype, weight + delta)))
            return (
                new_weight,
                T.Cast(dtype, pre + arrival),
                T.Cast(dtype, post + post_event),
            )

        return step
    raise ValueError("Choose triton or tilelang for native learning")


def native_template(backend, function):
    """Bind positional tensors and a native function to the existing
    scheduler."""
    return PlasticityTemplate(
        states=("pre_trace", "post_trace"),
        parameters=("a_plus", "a_minus", "tau", "w_min", "w_max"),
        stages=(
            PlasticityStage(
                domain="edge",
                reads=(
                    "weight",
                    "pre_trace",
                    "post_trace",
                    "a_plus",
                    "a_minus",
                    "tau",
                    "w_min",
                    "w_max",
                    "arrival",
                    "post_event",
                    "dt",
                ),
                outputs=("weight", "pre_trace", "post_trace"),
                **{backend: function},
            ),
        ),
        traces=(
            PlasticityTrace("pre_trace", "tau", "arrival"),
            PlasticityTrace("post_trace", "tau", "post_event"),
        ),
        event_driven=True,
    )


def build_network(backend, device, dtype, *, capture=False, rule=None):
    """Bind rule memories through the normal sparse connection lifecycle."""
    pre = torch.arange(129)
    connection = SparseConnection.from_edges(
        pre,
        (pre + 1) % 129,
        129,
        129,
        Synapse(
            weight=EdgeWeight(torch.full((129,), 0.001), trainable=False),
            plasticity=BurstSTDP() if rule is None else rule,
        ),
    )
    network = RecurrentNN(
        LIF(129, tau=1.0, v_threshold=0.5, v_reset=0.0, hard_reset=True),
        ExponentialPSC(129, tau_syn=2.0, linear=connection),
        backend=backend,
        chunk_size=4,
        cudagraph=capture,
        update_state_names=(
            "neuron.v",
            "synapse.psc",
            "synapse.linear.weight.value",
            "synapse.linear.plasticity.pre_trace",
            "synapse.linear.plasticity.post_trace",
        ),
    ).to(device=device, dtype=dtype)
    init_net_state(network, device=device, dtype=dtype, persistent=True)
    return network


def run_example(
    backend="torch",
    device="cpu",
    *,
    dtype=torch.float32,
    recurrent="standard",
    compile=False,
    capture=False,
):
    """Compare spike, state, trace and weight trajectories with eager Torch."""
    if compile and backend == "torch":
        raise ValueError(
            "This example compiles the native recurrent path; choose triton or tilelang"
        )
    if (capture or backend != "torch") and torch.device(device).type != "cuda":
        raise ValueError("Native kernels and CUDA Graph capture require a CUDA device")
    if backend != "torch":
        register_plasticity_template(
            BurstSTDP.kernel_model,
            native_template(backend, native_step(backend)),
            replace=True,
        )
    options = (
        "torch"
        if backend == "torch"
        else KernelOptions(
            neuron=backend,
            connection=backend,
            connection_method="push",
            recurrent=recurrent,
            spike_selection="spike_list",
        )
    )
    # Three bounded chunks exercise trace persistence. No BPTT or truncation:
    # online plasticity intentionally runs with gradients disabled.
    with torch.no_grad(), environ.context(dt=1.0):
        reference = build_network("torch", "cpu", dtype)
        candidate = build_network(options, device, dtype, capture=capture)
        inputs = (torch.arange(12 * 129).reshape(12, 129) % 7 == 0).to(dtype) * 1.1
        expected_z, expected_records = reference(inputs)
        # RecurrentNN keeps graph preparation/replay and chunk bookkeeping in
        # Python while compiling the tensor work inside those boundaries.
        execute = torch.compile(candidate) if compile else candidate
        actual_z, actual_records = execute(inputs.to(device))
        torch.testing.assert_close(actual_z.cpu(), expected_z, atol=0, rtol=0)
        relative = max(1e-8, 4 * torch.finfo(dtype).eps)
        absolute = 1e-9 if dtype == torch.float64 else 2e-6
        for name, expected in expected_records.items():
            torch.testing.assert_close(
                actual_records[name].cpu(), expected, atol=absolute, rtol=relative
            )
    return actual_z, actual_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("torch", "triton", "tilelang"), default="torch"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float64", "float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--recurrent", choices=("standard", "helion"), default="standard"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    run_example(
        args.backend,
        args.device,
        dtype=getattr(torch, args.dtype),
        recurrent=args.recurrent,
        compile=args.compile,
        capture=args.capture,
    )
    print("Spike, neuron/PSC, trace and weight trajectories match eager Torch")
