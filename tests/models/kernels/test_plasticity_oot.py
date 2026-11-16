"""Validate native out-of-tree STDP through the existing recurrent
schedules."""

import json
import math
from dataclasses import replace

import pytest
import torch

from btorch.models import environ
from btorch.models.kernels import (
    KernelOptions,
    PlasticityStage,
    register_plasticity_template,
)
from btorch.models.kernels.plasticity import (
    _native_source,
    _render,
    _resolve_native_function,
)
from examples.oot_stdp import BurstSTDP, build_network, native_step, native_template
from tests.models.kernels.helpers import require_cuda_backend


def _scaled_step(backend, factor):
    base = native_step(backend)
    if backend == "triton":
        import triton
        import triton.language as tl

        # A nested JIT helper must capture its language module in the body;
        # annotation-only references are not Python closure variables. The
        # captured Python scale is a compile-time constant, not a tensor input.
        factor = tl.constexpr(factor)

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
            return base(
                weight,
                pre_trace,
                post_trace,
                tl.cast(a_plus * factor, a_plus.dtype),
                a_minus,
                tau,
                w_min,
                w_max,
                arrival,
                post_event,
                dt,
                INCLUDE_LTP,
                INCLUDE_LTD,
            )

        return step
    import tilelang.language as T

    def step(weight, pre_trace, post_trace, a_plus, *remaining):
        return base(
            weight,
            pre_trace,
            post_trace,
            T.Cast(a_plus.dtype, a_plus * factor),
            *remaining,
        )

    return step


def _reference_step(
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
    """CPU callable used only to check compile-time function binding."""
    pre, post = pre_trace * torch.exp(-dt / tau), post_trace * torch.exp(-dt / tau)
    plus = a_plus * post_event * pre.square() if include_ltp else 0.0
    minus = a_minus * arrival * post if include_ltd else 0.0
    return (weight + plus - minus).clamp(w_min, w_max), pre + arrival, post + post_event


def test_external_memory_rule_keeps_analytical_trace_order_and_lifecycle():
    from btorch.models.connection import EdgeWeight, SparseConnection, Synapse

    connection = SparseConnection.from_edges(
        torch.tensor([0]),
        torch.tensor([0]),
        1,
        1,
        Synapse(
            weight=EdgeWeight(torch.tensor([0.01]), trainable=False),
            plasticity=BurstSTDP(tau=2.0),
        ),
    ).double()
    rule = connection.plasticity
    weight, pre, post = connection.weight.value.item(), 0.0, 0.0
    with torch.no_grad():
        # Arrival, post, coincidence: read old decayed traces before adding
        # current events. This oracle does not call the rule or native helper.
        for arrival, post_event in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
            pre, post = pre * math.exp(-0.5), post * math.exp(-0.5)
            weight = max(
                0.0,
                min(
                    rule.w_max.item(),
                    weight
                    + rule.a_plus.item() * post_event * pre * pre
                    - rule.a_minus.item() * arrival * post,
                ),
            )
            pre, post = pre + arrival, post + post_event
            connection.update_plasticity(
                torch.tensor([arrival], dtype=torch.float64),
                torch.tensor([post_event], dtype=torch.float64),
                dt=1.0,
            )
            for actual, expected in (
                (connection.weight.value, weight),
                (rule.pre_trace, pre),
                (rule.post_trace, post),
            ):
                torch.testing.assert_close(
                    actual.reshape(()), torch.tensor(expected, dtype=torch.float64)
                )
    checkpoint = {name: value.clone() for name, value in rule.state_dict().items()}
    rule.reset()
    assert rule.pre_trace.count_nonzero() == rule.post_trace.count_nonzero() == 0
    rule.load_state_dict(checkpoint)
    torch.testing.assert_close(rule.pre_trace, checkpoint["pre_trace"])
    torch.testing.assert_close(rule.post_trace, checkpoint["post_trace"])


def test_native_registration_rejects_conflicting_event_ownership():
    template = native_template("triton", _reference_step)
    incomplete = replace(template.stages[0], outputs=("weight", "pre_trace"))
    with pytest.raises(ValueError, match="all declared exponential traces"):
        register_plasticity_template(
            "lab_missing_trace_writer", replace(template, stages=(incomplete,))
        )
    with pytest.raises(ValueError, match="built-in"):
        register_plasticity_template("pair_stdp", template)
    # Public customization accepts native callables, not the rejected math DSL.
    with pytest.raises(TypeError, match="callable"):
        PlasticityStage("edge", ("weight",), ("weight",), triton="weight + 1")


def test_replaced_native_rule_rejects_a_stale_mutating_operator():
    from btorch.models.kernels.recurrent_operator import _online_impl

    name = "lab_stale_native_operator"
    template = native_template("triton", _reference_step)
    register_plasticity_template(name, template)
    encoded = json.dumps({"plasticity": name, "plasticity_revision": 1})
    register_plasticity_template(name, template, replace=True)
    weight, trace = torch.tensor([0.4]), torch.tensor([0.2])
    with pytest.raises(RuntimeError, match="registration changed"):
        _online_impl(
            torch.ones(1, 1),
            torch.empty(0),
            [],
            [],
            [],
            weight,
            [trace],
            [],
            [],
            [],
            encoded,
            [],
            "triton",
            torch.tensor(1.0),
        )
    torch.testing.assert_close(weight, torch.tensor([0.4]), atol=0, rtol=0)
    torch.testing.assert_close(trace, torch.tensor([0.2]), atol=0, rtol=0)


def test_external_rule_validates_the_learning_renderer_before_mutation():
    # In the two-kernel schedule, propagation and STDP share the connection
    # renderer, which may differ from the neuron renderer. A TileLang-only
    # learning rule must therefore work with a Triton neuron producer. This
    # CPU preflight checks capability rejection without requiring GPU execution.
    name = "lab_connection_owned_learning"

    class ExternalBurst(BurstSTDP):
        kernel_model = name

    register_plasticity_template(name, native_template("tilelang", _reference_step))
    options = KernelOptions(
        neuron="triton",
        connection="tilelang",
        connection_method="push",
        spike_selection="spike_list",
    )
    network = build_network(options, "cpu", torch.float32, rule=ExternalBurst())
    rule = network.synapse.linear.plasticity
    rule.pre_trace.fill_(0.25)
    weight = network.synapse.linear.weight.value.detach().clone()
    with torch.no_grad():
        network._begin_stdp_run(torch.zeros(129))
        network._end_stdp_run()
        network.backend = replace(options, connection="triton")
        with pytest.raises(NotImplementedError, match="no triton function"):
            network._begin_stdp_run(torch.zeros(129))
    torch.testing.assert_close(network.synapse.linear.weight.value, weight)
    torch.testing.assert_close(rule.pre_trace, torch.full_like(rule.pre_trace, 0.25))


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_native_replacement_refreshes_separate_and_helion_sources(backend):
    from btorch.models.kernels.backends.helion_plasticity import prepare_stdp_phases

    name = f"lab_native_source_{backend}"
    template = native_template(backend, _reference_step)
    sizes = tuple(
        {
            **dict.fromkeys(template.parameters, 1),
            **dict.fromkeys(template.states, 4),
            "weight": 4,
            "dt": 1,
            "arrivals": 3,
            "posts": 3,
            "pre_index": 4,
            "post_index": 4,
        }.items()
    )

    def sources():
        ordinary, _ = _render(backend, name, 0, 4, 1, (), sizes, "float32", ())
        fused = prepare_stdp_phases(
            backend, name, 3, 1, 1, 3, 4, (1,) * len(template.parameters)
        )
        return ordinary, _native_source(
            "\n".join((*fused.propagation, *fused.post_propagation)), backend, name
        )

    register_plasticity_template(name, template)
    before = sources()
    # The compiler receives the user's actual function object; it is never
    # symbolically traced or translated into an equation representation.
    assert _resolve_native_function(name, backend, 0, 1) is _reference_step
    register_plasticity_template(name, template, replace=True)
    after = sources()
    for old, new in zip(before, after, strict=True):
        assert f"_btorch_stdp_{name}_v1_s0" in old
        assert f"_btorch_stdp_{name}_v2_s0" in new
        assert old != new
    other = "tilelang" if backend == "triton" else "triton"
    with pytest.raises(NotImplementedError, match="no .* function"):
        _native_source("", other, name)


@pytest.mark.parametrize(
    "neuron_backend,backend,recurrent,discovery,dtype",
    (
        ("triton", "triton", "standard", "auto", torch.float64),
        ("triton", "tilelang", "standard", "spike_list", torch.float16),
        ("triton", "triton", "helion", "spike_list", torch.float16),
        ("tilelang", "tilelang", "helion", "scan", torch.bfloat16),
    ),
)
def test_external_native_stdp_replacement_and_chunk_replay(
    neuron_backend, backend, recurrent, discovery, dtype
):
    """Check four intended schedules, not a model/options Cartesian product.

    Native function replacement must change real weight trajectories and
    invalidate captures. Nonzero traces, strided drives, a partial final
    CTA and three chunks expose snapshot ordering and persistent state
    behavior.
    """
    require_cuda_backend(backend)
    if neuron_backend != backend:
        require_cuda_backend(neuron_backend)
    name = f"lab_native_{backend}_{recurrent}_{discovery}"

    class ExternalBurst(BurstSTDP):
        kernel_model = name

    options = KernelOptions(
        neuron=neuron_backend,
        connection=backend,
        recurrent=recurrent,
        connection_method="push",
        spike_selection=discovery,
    )
    inputs = (torch.arange(12 * 129).reshape(12, 129) % 7 == 0).to(dtype) * 1.1
    padded = torch.zeros(12, 258, dtype=dtype, device="cuda")
    padded[:, ::2] = inputs.cuda()
    inputs_cuda = padded[:, ::2]
    actual = None
    captures = None
    with torch.no_grad(), environ.context(dt=1.0):
        for index, factor in enumerate((0.5, 0.25)):
            register_plasticity_template(
                name,
                native_template(backend, _scaled_step(backend, factor)),
                replace=bool(index),
            )
            expected = build_network(
                "torch", "cpu", dtype, rule=ExternalBurst(a_plus=0.002 * factor)
            )
            rule = expected.synapse.linear.plasticity
            rule.pre_trace = torch.linspace(0.1, 0.5, 129, dtype=dtype).reshape_as(
                rule.pre_trace
            )
            rule.post_trace = torch.linspace(0.5, 0.1, 129, dtype=dtype).reshape_as(
                rule.post_trace
            )
            if actual is None:
                actual = build_network(
                    options, "cuda", dtype, capture=True, rule=ExternalBurst()
                )
            actual.load_state_dict(expected.state_dict())
            actual.synapse.linear.plasticity.a_plus.fill_(torch.tensor(0.002).item())
            target_z, target_records = expected(inputs)
            actual_z, actual_records = actual(inputs_cuda)
            torch.testing.assert_close(actual_z.cpu(), target_z, atol=0, rtol=0)
            relative = max(1e-8, 4 * torch.finfo(dtype).eps)
            absolute = 1e-9 if dtype == torch.float64 else 2e-6
            for key in target_records:
                torch.testing.assert_close(
                    actual_records[key].cpu(),
                    target_records[key],
                    atol=absolute,
                    rtol=relative,
                    msg=lambda message: f"{key}: {message}",
                )
            if captures is not None:
                assert actual._cudagraph_runner.n_captures > captures
            captures = actual._cudagraph_runner.n_captures
        if backend == "triton" and recurrent == "standard":
            # Compile and manual CUDA Graph replay must compose. Keep the
            # bounded graph runner enabled: its Python preparation/replay
            # bookkeeping is intentionally outside Dynamo's tensor regions.
            # Build a fresh oracle because the previous sequence advanced it.
            expected = build_network(
                "torch", "cpu", dtype, rule=ExternalBurst(a_plus=0.002 * 0.25)
            )
            actual.load_state_dict(expected.state_dict())
            actual.synapse.linear.plasticity.a_plus.fill_(torch.tensor(0.002).item())
            target_z, target_records = expected(inputs)
            actual_z, actual_records = torch.compile(actual)(inputs_cuda)
            torch.testing.assert_close(actual_z.cpu(), target_z, atol=0, rtol=0)
            for key in target_records:
                torch.testing.assert_close(
                    actual_records[key].cpu(), target_records[key], atol=1e-9, rtol=1e-8
                )
