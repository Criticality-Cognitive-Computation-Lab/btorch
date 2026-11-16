"""Parity tests for native per-edge STDP kernels.

The eager connection tests define the mathematical oracle.  This file only
checks the renderer ABI: canonical indices, delayed arrival addressing,
heterogeneous rule tensors, mutable traces, and non-contiguous views.
"""

from __future__ import annotations

import io
from functools import partial

import pytest
import torch

from btorch.models import environ, functional
from btorch.models.connection import (
    ExternalSignal,
    InhibitorySTDP,
    ModulatedSTDP,
    ModulatedSTDPOptions,
    NearestPairSTDP,
    PairSTDP,
    PowerPairSTDP,
    PowerPairSTDPOptions,
    TripletSTDP,
    TripletSTDPOptions,
    VoltageSTDP,
    VoltageSTDPOptions,
)
from btorch.models.kernels import KernelOptions
from btorch.models.kernels.plasticity import (
    get_plasticity_template,
    prepare_update,
    update,
)
from tests.models.connection.test_plasticity import (
    N_DELAY,
    _arrivals,
    _connection,
    _posts,
    _recurrent_inputs,
    _recurrent_model,
    _rule,
    _shared_pair_connection,
)
from tests.models.kernels.helpers import require_cuda_backend


def _heterogeneous_rule() -> PairSTDP:
    """Use input-edge parameters whose canonical reorder is observable."""
    return _rule(
        a_plus=torch.tensor([0.08, 0.12, 0.16, 0.20]),
        a_minus=torch.tensor([0.07, 0.11, 0.15, 0.19]),
        tau_plus=torch.tensor([4.0, 5.0, 6.0, 7.0]),
        tau_minus=torch.tensor([5.0, 6.0, 7.0, 8.0]),
        w_min=torch.tensor([0.0, 0.01, 0.0, 0.02]),
        w_max=torch.tensor([0.20, 0.25, 0.30, 0.35]),
    )


def _inhibitory_rule():
    """Mix signed bounds and constant weakening over routed canonical edges."""
    return InhibitorySTDP(
        eta=torch.tensor([0.04, 0.05, 0.06, 0.07], dtype=torch.float64),
        tau=torch.tensor([3.0, 4.0, 5.0, 6.0], dtype=torch.float64),
        alpha=torch.tensor([0.2, 0.0, 0.3, 0.4], dtype=torch.float64),
        polarity=torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float64),
        w_min=torch.tensor([-0.5, 0.0, -0.5, 0.0], dtype=torch.float64),
        w_max=torch.tensor([0.0, 0.5, 0.0, 0.5], dtype=torch.float64),
    )


def _event_sequences(connection):
    """Build distinct arrival and post-event streams from canonical slots."""
    arrivals = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
        ]
    )
    posts = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    return (
        torch.stack([_arrivals(connection, row) for row in arrivals]),
        torch.stack([_posts(connection, row) for row in posts]),
    )


def _strided_time_sequence(values: torch.Tensor, *, device, dtype) -> torch.Tensor:
    """Materialize a logical sequence as a view with a gap in its stride."""
    storage = torch.zeros(
        (*values.shape[:-1], values.shape[-1] * 2), device=device, dtype=dtype
    )
    storage[..., ::2].copy_(values.to(device=device, dtype=dtype))
    return storage[..., ::2]


def _run_native_case(
    backend: str, dtype: torch.dtype, *, strided: bool, rule_factory=_heterogeneous_rule
) -> None:
    """Compare every native step with its eager rule oracle."""
    cpu_events = _connection(rule_factory())
    eager = _connection(rule_factory()).to(device="cuda", dtype=dtype)
    native = _connection(rule_factory()).to(device="cuda", dtype=dtype)
    model = type(native.plasticity).__dict__["kernel_model"]
    template = get_plasticity_template(model)
    arrivals_cpu, posts_cpu = _event_sequences(cpu_events)
    arrivals = (
        _strided_time_sequence(arrivals_cpu, device="cuda", dtype=dtype)
        if strided
        else arrivals_cpu.to(device="cuda", dtype=dtype)
    )
    posts = (
        _strided_time_sequence(posts_cpu, device="cuda", dtype=dtype)
        if strided
        else posts_cpu.to(device="cuda", dtype=dtype)
    )

    # Use strided mutable destinations in the main FP32 case.  The native ABI
    # exposes their reachable storage and derives logical offsets from strides.
    if strided:
        weight_storage = torch.empty(2 * native.nnz, device="cuda", dtype=dtype)
        native_weight = weight_storage[::2]
        native_weight.copy_(native.weight.value)
        native_states = {
            name: torch.zeros(2 * native.nnz, device="cuda", dtype=dtype)[::2]
            for name in template.states
        }
    else:
        native_weight = native.weight.value.detach().clone()
        native_states = {
            name: getattr(native.plasticity, name).detach().clone()
            for name in template.states
        }

    native_rule = native.plasticity
    parameters = {}
    for name in template.parameters:
        value = getattr(native_rule, name)
        if strided and value.numel() > 1:
            storage = torch.empty(2 * value.numel(), device="cuda", dtype=dtype)
            value_view = storage[::2]
            value_view.copy_(value)
            value = value_view
        parameters[name] = value
    dt = torch.tensor(1.0, device="cuda", dtype=dtype)
    # A promoted dense execution operand must receive the rounded canonical
    # storage values. Gaps represent missing edges and must remain untouched.
    placement = torch.arange(native.nnz, device="cuda") * 3 + 1
    mirror = torch.full(
        (3 * native.nnz,),
        -1.0,
        device="cuda",
        dtype=torch.float32 if dtype == torch.float16 else dtype,
    )
    # A recurrent invocation resolves the law and launch bindings only once.
    # Later event views change pointers while weights and traces stay owned.
    transition = prepare_update(
        backend,
        model,
        native_weight,
        native_states,
        arrivals[0],
        posts[0],
        native.pre,
        native.post,
        native.delay,
        N_DELAY,
        parameters,
        dt,
        weight_destination=(mirror, placement),
    )
    for arrivals_t, posts_t in zip(arrivals, posts, strict=True):
        with torch.no_grad():
            eager.update_plasticity(arrivals_t, posts_t, dt=dt)
            transition(arrivals_t, posts_t)
        torch.testing.assert_close(
            mirror[placement], native_weight.to(mirror.dtype), atol=0, rtol=0
        )
        assert (mirror.reshape(-1, 3)[:, (0, 2)] == -1).all()
        for actual, expected in (
            (native_weight, eager.weight.value),
            *(
                (value, getattr(eager.plasticity, name))
                for name, value in native_states.items()
            ),
        ):
            torch.testing.assert_close(
                actual,
                expected,
                atol=2e-5
                if dtype == torch.float16
                else 2e-12
                if dtype == torch.float64
                else 2e-7,
                rtol=2e-3
                if dtype == torch.float16
                else 2e-12
                if dtype == torch.float64
                else 2e-6,
            )


@pytest.mark.parametrize(
    ("backend", "dtype", "kind"),
    (
        ("triton", torch.float64, "pair"),
        ("tilelang", torch.float32, "pair"),
        ("triton", torch.float64, "modulated_shared"),
        ("tilelang", torch.float32, "modulated_edge"),
        pytest.param(
            "triton", torch.float16, "modulated_edge", id="triton-fp16-modulated-edge"
        ),
        pytest.param(
            "triton", torch.bfloat16, "modulated_edge", id="triton-bf16-modulated-edge"
        ),
        pytest.param(
            "tilelang",
            torch.float16,
            "modulated_shared",
            id="tilelang-fp16-modulated-shared",
        ),
        pytest.param(
            "tilelang",
            torch.bfloat16,
            "modulated_shared",
            id="tilelang-bf16-modulated-shared",
        ),
        pytest.param(
            "triton", torch.float16, "modulated_mixed", id="triton-fp16-fp32-signal"
        ),
        pytest.param(
            "tilelang",
            torch.bfloat16,
            "modulated_mixed",
            id="tilelang-bf16-fp32-signal",
        ),
    ),
)
def test_prepared_shared_traces_follow_delayed_streams_and_isolated_posts(
    backend, dtype, kind
):
    """Reuse compact trace writers across coincident, delayed and silent
    events.

    Receptor duplicates share a source/delay trace, while an isolated
    post still owns its trace. Strided memories and changing history
    destinations exercise the same prepared ABI used by recurrent chunk
    execution. Guard columns detect writes past unequal, partial-block
    edge, segment and post domains.
    """
    require_cuda_backend(backend)

    def factory(pair):
        if kind == "pair":
            return pair
        return ModulatedSTDP(
            ModulatedSTDPOptions(
                a_plus=pair.a_plus,
                a_minus=pair.a_minus,
                tau_plus=pair.tau_plus,
                tau_minus=pair.tau_minus,
                tau_eligibility=torch.linspace(2.0, 5.0, 6),
                eta=(1e-5 if dtype == torch.float16 else 1e-38)
                if kind == "modulated_mixed"
                else 0.5,
                baseline=0.2,
            ),
            modulator=ExternalSignal("segment" if kind.endswith("shared") else "post"),
            trace_scope="neuron" if kind.endswith("shared") else "connection",
            w_min=pair.w_min,
            w_max=pair.w_max,
        )

    eager = (
        _shared_pair_connection("neuron", rule_factory=factory).cuda().to(dtype=dtype)
    )
    native = (
        _shared_pair_connection("neuron", rule_factory=factory).cuda().to(dtype=dtype)
    )
    rule = native.plasticity
    tolerance = (
        {"atol": 2e-5, "rtol": 2e-3}
        if dtype == torch.float16
        else {"atol": 2e-3, "rtol": 2e-2}
        if dtype == torch.bfloat16
        else {}
    )
    if kind != "pair":
        # Nonzero eligibility makes silent-bin modulation observable even
        # before this invocation has produced a new pair.
        rule.eligibility.fill_(0.1)
        eager.plasticity.eligibility.fill_(0.1)
    domains = tuple(rule.state_domains.items())
    template = get_plasticity_template(rule._kernel_model, domains, rule.signal_domains)
    weight = native.weight.value.detach().clone()
    states = {
        name: _strided_time_sequence(getattr(rule, name), device="cuda", dtype=dtype)
        for name in template.states
    }
    source = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
        ],
        device="cuda",
        dtype=dtype,
    )
    posts = torch.tensor(
        [[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1], [0, 0, 0, 0], [0, 1, 1, 0]],
        device="cuda",
        dtype=dtype,
    )
    guarded_records = {
        name: value.new_full(
            (len(source), *value.shape[:-1], value.shape[-1] + 2), -1234
        )
        for name, value in {**states, "weight": weight}.items()
    }
    records = {name: value[..., 1:-1] for name, value in guarded_records.items()}
    topology = native.cache.prepare_delay_segments(3, 2)
    inverse = native.cache.prepare_edge_segments(3, 2)
    signals = {}
    if kind != "pair":
        count = topology[1].numel() if kind.endswith("shared") else native.n_post
        signals["modulator"] = _strided_time_sequence(
            torch.linspace(-1.0, 1.0, len(source) * count).reshape(len(source), count),
            device="cuda",
            dtype=dtype,
        )
        if kind == "modulated_mixed":
            # A contiguous FP32 signal has no layout descriptor. Use finite
            # FP32 values outside the trace dtype's range, with a small eta
            # to keep updates finite and inside the weight bounds. Casting
            # this signal to storage precision would overflow and clip the
            # weights, rather than merely introduce a tolerable rounding error.
            signals["modulator"] = torch.full(
                (len(source), count),
                70000.0 if dtype == torch.float16 else 3.4e38,
                device="cuda",
                dtype=torch.float32,
            )
    dt = weight.new_tensor(0.5)
    transition = prepare_update(
        backend,
        rule._kernel_model,
        weight,
        states,
        source[0],
        posts[0],
        native.pre,
        native.post,
        native.delay,
        2,
        {name: getattr(rule, name) for name in template.parameters},
        dt,
        records={name: value[0] for name, value in records.items()},
        state_domains=domains,
        signal_domains=rule.signal_domains,
        signals={name: value[0] for name, value in signals.items()},
        segments={
            "segment_pre": topology[1],
            "segment_delay": topology[2],
            "segment_index": inverse,
        },
    )
    with torch.no_grad():
        for step, (arrival, post) in enumerate(zip(source, posts, strict=True)):
            step_signals = {name: value[step] for name, value in signals.items()}
            eager.update_plasticity(arrival, post, dt=dt, signals=step_signals)
            transition(
                arrival,
                post,
                records={name: value[step] for name, value in records.items()},
                signals=step_signals,
            )
            for name, value in {**transition.states, "weight": weight}.items():
                expected = (
                    eager.weight.value
                    if name == "weight"
                    else getattr(eager.plasticity, name)
                )
                expected = expected.to(dtype=dtype)
                torch.testing.assert_close(
                    value,
                    expected,
                    msg=f"{backend}/{dtype}/{name}/step={step}",
                    **tolerance,
                )
                torch.testing.assert_close(records[name][step], value)
            if step == 2:
                # Commit an odd chunk, then reuse this preparation for an
                # even continuation. Buffer parity is local to each commit.
                transition.finish()
                for name, value in states.items():
                    torch.testing.assert_close(
                        value, getattr(eager.plasticity, name).to(dtype), **tolerance
                    )
        transition.finish()
        for name, value in states.items():
            torch.testing.assert_close(
                value, getattr(eager.plasticity, name).to(dtype), **tolerance
            )
        for value in guarded_records.values():
            torch.testing.assert_close(
                value[..., 0], torch.full_like(value[..., 0], -1234)
            )
            torch.testing.assert_close(
                value[..., -1], torch.full_like(value[..., -1], -1234)
            )


def _triplet_rule():
    """Make slow traces and edge-specific triplet terms affect the
    trajectory."""
    return TripletSTDP(
        TripletSTDPOptions(
            a_plus=0.08,
            a_minus=0.07,
            a3_plus=torch.tensor([0.01, 0.03, 0.05, 0.07]),
            a3_minus=torch.tensor([0.02, 0.04, 0.06, 0.08]),
            tau_plus=4.0,
            tau_minus=5.0,
            tau_x=torch.tensor([8.0, 9.0, 10.0, 11.0]),
            tau_y=torch.tensor([12.0, 13.0, 14.0, 15.0]),
        ),
        w_max=0.35,
    )


@pytest.mark.parametrize(
    ("backend", "dtype"),
    (("triton", torch.float32), ("tilelang", torch.float64)),
)
def test_native_voltage_stdp_post_owned_filters_and_silent_learning(backend, dtype):
    """Advance post filters once while incoming edges learn on silent steps.

    Distinct edge/post counts and heterogeneous post coefficients expose
    accidental edge indexing. Every signal, mutable state and history is
    a strided view. The eager rule has an independent
    impulse/convolution oracle in the connection tests; here we exercise
    the two native stage renderers.
    """
    require_cuda_backend(backend)

    def rule():
        return VoltageSTDP(
            VoltageSTDPOptions(
                a_plus=torch.tensor([0.02, 0.025, 0.01, 0.03]),
                a_minus=torch.tensor([0.05, 0.07, 0.06, 0.04]),
                tau_pre=torch.tensor([2.0, 3.0, 4.0, 5.0]),
                tau_plus=torch.tensor([1.0, 2.0, 3.0]),
                tau_minus=torch.tensor([0.8, 1.5, 2.5]),
                theta_plus=torch.tensor([-50.0, -45.0, -45.0]),
                theta_minus=torch.tensor([-60.0, -58.0, -62.0]),
                v_rest=torch.tensor([-70.0, -65.0, -68.0]),
            ),
            w_max=0.72,
        )

    eager = _connection(rule()).to(device="cuda", dtype=dtype)
    native = _connection(rule()).to(device="cuda", dtype=dtype)
    template = get_plasticity_template("voltage_stdp")
    weight = _strided_time_sequence(
        native.weight.value.detach(), device="cuda", dtype=dtype
    )
    states = {
        name: _strided_time_sequence(
            getattr(native.plasticity, name), device="cuda", dtype=dtype
        )
        for name in template.states
    }
    records = {
        name: torch.empty(value.numel() * 2, device="cuda", dtype=dtype)[::2]
        for name, value in {"weight": weight, **states}.items()
    }
    voltages = _strided_time_sequence(
        torch.tensor(
            [
                [-55.0, -40.0, -65.0],
                [-35.0, -30.0, -25.0],
                [-30.0, -35.0, -40.0],
                [-25.0, -30.0, -35.0],
                [-70.0, -65.0, -68.0],
            ]
        ),
        device="cuda",
        dtype=dtype,
    )
    arrivals, posts = _event_sequences(_connection())
    arrivals = arrivals.to(device="cuda", dtype=dtype)
    # The third and fourth steps have no arrivals or posts. Voltage-gated LTP
    # must still evolve the existing eligibility and weights.
    arrivals[2:4].zero_()
    posts = posts.to(device="cuda", dtype=dtype)
    posts[2:4].zero_()
    parameters = {
        name: _strided_time_sequence(
            getattr(native.plasticity, name).reshape(-1), device="cuda", dtype=dtype
        )
        for name in template.parameters
    }
    dt = torch.tensor(0.5, device="cuda", dtype=dtype)
    silent_ltp = False
    for step in range(len(voltages)):
        old_weight = weight.clone()
        with torch.no_grad():
            eager.update_plasticity(
                arrivals[step], posts[step], dt=dt, signals={"voltage": voltages[step]}
            )
            update(
                backend,
                "voltage_stdp",
                weight,
                states,
                arrivals[step],
                posts[step],
                native.pre,
                native.post,
                native.delay,
                N_DELAY,
                parameters,
                dt,
                records=records,
                signals={"voltage": voltages[step]},
            )
        expected = {
            "weight": eager.weight.value,
            **{name: getattr(eager.plasticity, name) for name in template.states},
        }
        for name, actual in {"weight": weight, **states}.items():
            torch.testing.assert_close(
                actual,
                expected[name],
                atol=2e-12 if dtype == torch.float64 else 3e-6,
                rtol=2e-12 if dtype == torch.float64 else 3e-6,
            )
            torch.testing.assert_close(records[name], actual)
        if step in (2, 3):
            silent_ltp |= bool((weight > old_weight).any())
    assert silent_ltp


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_recurrent_native_voltage_stdp_live_signals_and_chunks(backend):
    """Route charged or time-first voltage without a full charged history.

    Each renderer covers a different binding: Triton reads the neuron's
    charged record, TileLang consumes a strided external sequence through the
    shared chunk inputs. Both compare every physical/learning final state and
    requested learning trajectories with the eager recurrent scheduler.
    """
    require_cuda_backend(backend)
    options = VoltageSTDPOptions(
        a_plus=0.08,
        a_minus=0.06,
        tau_pre=4.0,
        tau_plus=2.0,
        tau_minus=3.0,
        theta_plus=0.4,
        theta_minus=0.1,
        v_rest=0.0,
    )
    signal = (
        {}
        if backend == "triton"
        else {"voltage": ExternalSignal(domain="post", time_first=True)}
    )
    eager = _recurrent_model(
        rule=VoltageSTDP(options, **signal),
        pre_spike_v=True,
    ).to("cuda")
    native = _recurrent_model(
        rule=VoltageSTDP(options, **signal),
        pre_spike_v=True,
        # Charged signals must follow interior producer storage, not retain
        # the second step's view until a chunk boundary. The external case
        # separately covers splitting time-first signals into shorter chunks.
        chunk_size=200 if backend == "triton" else 2,
    ).to("cuda")
    native.load_state_dict(eager.state_dict())
    native.backend = KernelOptions(neuron=backend, connection="triton")
    native.cudagraph = backend == "triton"
    # Record learning state with its actual domain shape, but no full charged
    # trajectory. A live voltage binding must use current workspace storage.
    names = tuple(
        "synapse.linear.plasticity." + name
        for name in ("pre_trace", "voltage_plus", "voltage_minus")
    )
    eager.update_state_names = native.update_state_names = names
    inputs = _recurrent_inputs().cuda()
    kwargs = {}
    if signal:
        kwargs["plasticity_signals"] = {
            "voltage": _strided_time_sequence(
                torch.linspace(0.2, 1.5, inputs.numel()).reshape_as(inputs),
                device="cuda",
                dtype=inputs.dtype,
            )
        }
    with torch.no_grad(), environ.context(dt=0.5):
        expected, eager_records = eager(inputs, **kwargs)
        actual, native_records = native(inputs, **kwargs)
    torch.testing.assert_close(actual, expected)
    for name in names:
        torch.testing.assert_close(
            native_records[name], eager_records[name], atol=2e-6, rtol=3e-6
        )
    for name, value in eager.state_dict().items():
        torch.testing.assert_close(
            native.state_dict()[name], value, atol=2e-6, rtol=3e-6
        )


@pytest.mark.parametrize("recurrent", ("standard", "helion"))
def test_native_voltage_stdp_graph_and_compile_refresh_external_signal(recurrent):
    """Replay changed live voltage without capture-time learning or stale
    data."""
    require_cuda_backend("triton")
    options = VoltageSTDPOptions(
        a_plus=0.01,
        a_minus=0.01,
        theta_plus=0.4,
        theta_minus=0.1,
        v_rest=0.0,
    )
    inputs = _recurrent_inputs().cuda()
    for mode in ("graph", "compile"):

        def build():
            result = _recurrent_model(
                rule=VoltageSTDP(
                    options, voltage=ExternalSignal(domain="post", time_first=True)
                ),
                chunk_size=2 if mode == "graph" else 200,
            ).cuda()
            result.update_state_names = ()
            result.backend = KernelOptions(
                neuron="triton",
                connection="triton",
                recurrent=recurrent,
                connection_method="push" if recurrent == "helion" else "pull",
            )
            return result

        expected, actual = build(), build()
        actual.load_state_dict(expected.state_dict())
        actual.cudagraph = mode == "graph"
        execute = (
            torch.compile(actual, fullgraph=True, mode="reduce-overhead")
            if mode == "compile"
            else actual
        )
        # Same sequence shape, changed signal values, continued physical and
        # learning state. Warmup must restore both post filters and edge traces.
        with torch.no_grad(), environ.context(dt=0.5):
            for voltage in (0.6, 1.2):
                signals = {"voltage": torch.full_like(inputs, voltage)}
                reference, _ = expected(inputs, plasticity_signals=signals)
                output, _ = execute(inputs, plasticity_signals=signals)
                torch.testing.assert_close(output, reference)
                for name, value in expected.state_dict().items():
                    torch.testing.assert_close(
                        actual.state_dict()[name], value, atol=2e-6, rtol=3e-6
                    )


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_native_triplet_matches_eager_strided(backend):
    """Four old trace snapshots govern coincident events, silence and
    clipping."""
    require_cuda_backend(backend)
    _run_native_case(backend, torch.float32, strided=True, rule_factory=_triplet_rule)


def _power_rule():
    """Exercise distinct power laws, signed intervals and one fixed weight."""
    pair = _heterogeneous_rule()
    return PowerPairSTDP(
        PowerPairSTDPOptions(
            a_plus=pair.a_plus,
            a_minus=pair.a_minus,
            tau_plus=pair.tau_plus,
            tau_minus=pair.tau_minus,
            mu_plus=torch.tensor([0.0, 1.0, 0.5, 2.0]),
            mu_minus=torch.tensor([0.0, 1.0, 1.5, 0.25]),
        ),
        w_min=torch.tensor([-0.25, 0.0, -0.3, 0.25]),
        w_max=torch.tensor([0.0, 0.25, 0.3, 0.25]),
    )


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_native_power_pair_matches_eager_strided(backend):
    """Both renderers preserve zero, unit and fractional heterogeneous
    powers."""
    require_cuda_backend(backend)
    _run_native_case(backend, torch.float32, strided=True, rule_factory=_power_rule)


def test_native_power_pair_tilelang_fp64():
    """Power evaluation remains FP64 rather than narrowing through float32."""
    require_cuda_backend("tilelang")
    _run_native_case("tilelang", torch.float64, strided=False, rule_factory=_power_rule)


@pytest.mark.parametrize(
    ("backend", "pairing", "dtype"),
    [
        ("triton", "symmetric", torch.float32),
        ("tilelang", "restricted", torch.float64),
    ],
)
def test_native_nearest_pair_matches_eager(backend, pairing, dtype):
    """Check both pairing policies with routed, strided renderer bindings."""
    require_cuda_backend(backend)

    def rule():
        pair = _heterogeneous_rule()
        return NearestPairSTDP(
            a_plus=pair.a_plus,
            a_minus=pair.a_minus,
            tau_plus=pair.tau_plus,
            tau_minus=pair.tau_minus,
            w_min=pair.w_min,
            w_max=pair.w_max,
            pairing=pairing,
        )

    _run_native_case(backend, dtype, strided=True, rule_factory=rule)


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_native_stdp_matches_eager_with_heterogeneous_routed_strided_fp32(backend):
    """Both native renderers preserve FP32 routed multi-step semantics."""
    require_cuda_backend(backend)
    _run_native_case(backend, torch.float32, strided=True)


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_native_stdp_matches_eager_fp64(backend):
    """Each renderer performs the recurrence natively in FP64."""
    require_cuda_backend(backend)
    _run_native_case(backend, torch.float64, strided=False)


def test_native_stdp_triton_fp16_anchor():
    """The FP16 path matches the same-dtype eager update on strided storage."""
    require_cuda_backend("triton")
    _run_native_case("triton", torch.float16, strided=True)


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_recurrent_native_stdp_matches_torch_with_delay_receptors_and_chunks(backend):
    """A four-trace rule preserves delayed recurrence and all state records."""
    require_cuda_backend(backend)
    requested = (
        "neuron.v",
        "synapse.psc",
        "synapse.delayed_spike.history",
        "synapse.linear.plasticity.pre_trace",
        "synapse.linear.plasticity.post_trace",
        "synapse.linear.plasticity.pre_slow",
        "synapse.linear.plasticity.post_slow",
        "synapse.linear.weight.value",
    )
    reference = _recurrent_model(chunk_size=8, rule=_triplet_rule()).to(device="cuda")
    native = _recurrent_model(chunk_size=8, rule=_triplet_rule()).to(device="cuda")
    # Domain declarations describe named storage, not the renderer's positional
    # ABI. Reordering them must never relabel fast/slow history tensors.
    native.synapse.linear.plasticity.state_domains = dict(
        reversed(tuple(native.synapse.linear.plasticity.state_domains.items()))
    )
    reference.update_state_names = requested
    native.update_state_names = requested
    native.backend = KernelOptions(
        neuron=backend,
        # Exercise mutable online weights in the renderer's native sparse
        # path as well; a packed static snapshot would go stale after step 0.
        connection=backend,
        connection_method="pull",
        enable_fp_fusion=False,
        fast_math=False,
    )
    native.load_state_dict(reference.state_dict())
    # A prepared pull must keep reading the mutable parameter, rather than
    # retaining a private contiguous snapshot after the first STDP update.
    weight = native.synapse.linear.weight.value
    storage = weight.new_full((2 * weight.numel(),), -9.0)
    view = storage[::2]
    view.copy_(weight)
    native.synapse.linear.weight.value = torch.nn.Parameter(view, requires_grad=False)
    inputs = _recurrent_inputs().to(device="cuda")

    with torch.no_grad(), environ.context(dt=1.0):
        reference_first, reference_records_first = reference(inputs[:4])
        reference_second, reference_records_second = reference(inputs[4:])
        native_first, native_records_first = native(inputs[:4])
        native_second, native_records_second = native(inputs[4:])
    torch.testing.assert_close(storage[1::2], torch.full_like(weight, -9.0))

    for actual, expected in (
        (native_first, reference_first),
        (native_second, reference_second),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    for actual_records, expected_records in (
        (native_records_first, reference_records_first),
        (native_records_second, reference_records_second),
    ):
        assert set(actual_records) == set(expected_records) == set(requested)
        for name in requested:
            torch.testing.assert_close(
                actual_records[name],
                expected_records[name],
                atol=2e-4,
                rtol=2e-4,
            )

    for actual, expected in (
        (native.synapse.linear.weight.value, reference.synapse.linear.weight.value),
        (
            native.synapse.linear.plasticity.pre_trace,
            reference.synapse.linear.plasticity.pre_trace,
        ),
        (
            native.synapse.linear.plasticity.post_trace,
            reference.synapse.linear.plasticity.post_trace,
        ),
        (
            native.synapse.linear.plasticity.pre_slow,
            reference.synapse.linear.plasticity.pre_slow,
        ),
        (
            native.synapse.linear.plasticity.post_slow,
            reference.synapse.linear.plasticity.post_slow,
        ),
        (native.neuron.v, reference.neuron.v),
        (native.synapse.psc, reference.synapse.psc),
        (
            native.synapse.delayed_spike.history,
            reference.synapse.delayed_spike.history,
        ),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_online_stdp_graph_warmup_and_compile_preserve_learning_state():
    """Capture must not learn during warmup; compiled mutations stay visible.

    Both runners consume changed inputs on a second call without
    resetting. Compare the full physical/trace state and canonical
    weights, so a raster that happens not to diverge cannot hide extra
    warmup learning.
    """
    from btorch.models.functional import named_hidden_states

    require_cuda_backend("triton")
    options = KernelOptions(
        neuron="triton",
        connection="triton",
        connection_method="pull",
        enable_fp_fusion=False,
        fast_math=False,
    )
    inputs = _recurrent_inputs().cuda()
    for mode, rule_factory in (
        ("graph", _power_rule),
        ("compile", _triplet_rule),
        ("graph", lambda: NearestPairSTDP(pairing="restricted")),
        # Shared traces have different domain shapes and three ordered stages.
        # Both runners must restore compact memories as well as learned weights.
        ("graph", lambda: PairSTDP(trace_scope="neuron")),
        ("compile", lambda: PairSTDP(trace_scope="neuron")),
    ):
        expected = _recurrent_model(chunk_size=200, rule=rule_factory()).cuda()
        actual = _recurrent_model(chunk_size=200, rule=rule_factory()).cuda()
        actual.load_state_dict(expected.state_dict())
        expected.backend = actual.backend = options
        actual.cudagraph = mode == "graph"
        execute = (
            torch.compile(actual, fullgraph=True, mode="reduce-overhead")
            if mode == "compile"
            else actual
        )
        with torch.no_grad(), environ.context(dt=1.0):
            for step, current in enumerate((inputs[:3], inputs[2:] + 0.1)):
                # Nearest pairing is a live coefficient, not a Python branch:
                # a warmed graph must see policy changes at the same address.
                if step and isinstance(
                    actual.synapse.linear.plasticity, NearestPairSTDP
                ):
                    actual.synapse.linear.plasticity.restricted.fill_(False)
                    expected.synapse.linear.plasticity.restricted.fill_(False)
                y, records = expected(current)
                z, observed = execute(current)
                torch.testing.assert_close(z, y)
                torch.testing.assert_close(observed, records, atol=2e-4, rtol=2e-4)
                torch.testing.assert_close(
                    named_hidden_states(actual),
                    named_hidden_states(expected),
                    atol=2e-4,
                    rtol=2e-4,
                )
                torch.testing.assert_close(
                    actual.synapse.linear.weight.value,
                    expected.synapse.linear.weight.value,
                    atol=2e-4,
                    rtol=2e-4,
                )


@pytest.mark.parametrize("method", ("auto", "pull"))
def test_native_online_weights_follow_restored_canonical_slots(method):
    """A restored slot order need not coincide with a route's CSR order.

    State dictionaries own canonical edge IDs. Reorder all edge-aligned
    state consistently in a checkpoint, then verify that later
    propagation reads the learned weights for those IDs rather than the
    CSR positional values.
    """
    import copy

    require_cuda_backend("triton")
    expected = _recurrent_model(chunk_size=200).cuda()
    actual = _recurrent_model(chunk_size=200).cuda()
    checkpoint = copy.deepcopy(expected.state_dict())
    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    for name, value in tuple(checkpoint.items()):
        if (
            ".linear." not in name
            or not isinstance(value, torch.Tensor)
            or not value.ndim
        ):
            continue
        if name.endswith(".indices"):
            checkpoint[name] = value.index_select(1, permutation)
        elif value.shape[-1] == 4 and (
            name.endswith((".delay", ".receptor", ".weight.value"))
            or ".plasticity." in name
        ):
            checkpoint[name] = value.index_select(-1, permutation)
    expected.load_state_dict(checkpoint)
    actual.load_state_dict(checkpoint)
    assert not actual.synapse.linear.cache.identity_perm
    actual.backend = KernelOptions(
        neuron="triton",
        connection="auto" if method == "auto" else "triton",
        connection_method=method,
        enable_fp_fusion=False,
        fast_math=False,
    )
    inputs = _recurrent_inputs().cuda()
    with torch.no_grad(), environ.context(dt=1.0):
        for current in (inputs[:3], inputs[3:]):
            y, _ = expected(current)
            z, _ = actual(current)
            torch.testing.assert_close(z, y)
            torch.testing.assert_close(actual.neuron.v, expected.neuron.v)
            torch.testing.assert_close(actual.synapse.psc, expected.synapse.psc)
            torch.testing.assert_close(
                actual.synapse.linear.weight.value,
                expected.synapse.linear.weight.value,
                atol=2e-5,
                rtol=2e-5,
            )


def _izhikevich_dual_exponential_stdp_model(backend):
    """Build one signed, routed Izhikevich recurrent network for parity."""
    from btorch.models.connection import EdgeWeight, SparseConnection, Synapse
    from btorch.models.neurons import Izhikevich
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import DualExponentialPSC, HeterSynapsePSC

    # The semantic edge list is intentionally unsorted.  Every (source,
    # target, receptor) slot is unique, while both receptors are active.
    # Signed values exercise STDP in both directions.  Delay expansion is
    # covered by the existing heterogeneous LIF parity test; this case keeps
    # the primary Izhikevich example focused on its no-delay execution path.
    pre = torch.tensor([2, 0, 1, 2])
    post = torch.tensor([1, 0, 2, 2])
    receptor = torch.tensor([1, 0, 1, 0])
    weights = torch.tensor([-0.08, 0.06, -0.05, 0.07])
    rule = PairSTDP(
        a_plus=torch.tensor([0.08, 0.12, 0.16, 0.20]),
        a_minus=torch.tensor([0.07, 0.11, 0.15, 0.19]),
        tau_plus=torch.tensor([4.0, 5.0, 6.0, 7.0]),
        tau_minus=torch.tensor([5.0, 6.0, 7.0, 8.0]),
        w_min=torch.tensor([-0.25, -0.20, -0.30, -0.18]),
        w_max=torch.tensor([0.25, 0.20, 0.30, 0.18]),
    )
    connection = SparseConnection.from_edges(
        pre,
        post,
        3,
        3,
        Synapse(
            weight=EdgeWeight(weights, trainable=False),
            receptor=receptor,
            n_receptor=2,
            plasticity=rule,
        ),
    )
    synapse = HeterSynapsePSC(
        3,
        2,
        None,
        connection,
        base_psc=DualExponentialPSC,
        tau_rise=1.5,
        tau_decay=4.0,
    )
    neuron = Izhikevich(
        3,
        v_threshold=0.7,
        v_reset=0.0,
        v_rest=0.0,
        v_peak=1.0,
        c_m=1.0,
        k=0.1,
        a=0.03,
        b=0.1,
        d=0.05,
        hard_reset=True,
    )
    return RecurrentNN(
        neuron,
        synapse,
        backend=backend,
        update_state_names=(
            "neuron.v",
            "neuron.u",
            "synapse.psc",
            "synapse.base_psc.psc",
            "synapse.base_psc.g_rise",
            "synapse.base_psc.g_decay",
            "synapse.linear.plasticity.pre_trace",
            "synapse.linear.plasticity.post_trace",
            "synapse.linear.weight.value",
        ),
    )


@pytest.mark.parametrize(
    ("backend", "recurrent"),
    (("triton", "standard"), ("tilelang", "standard"), ("triton", "helion")),
)
def test_native_izhikevich_dual_exponential_stdp_matches_eager_per_renderer(
    backend, recurrent
):
    """Native renderers preserve Izhikevich, PSC, and STDP state semantics.

    This is a model-independence check: the kernel path must agree with the
    ordinary Torch modules for a signed two-receptor connection,
    including a split continuation boundary.  The deterministic current
    produces recurrent pre/post events, so a matching raster alone cannot
    hide a mismatch in dual-exponential memories, traces, or
    canonical weights.
    """
    require_cuda_backend(backend)
    from btorch.models import functional
    from btorch.models.functional import init_net_state

    options = KernelOptions(
        neuron=backend,
        connection=backend,
        recurrent=recurrent,
        # This complementary fixture covers live weights in source-major
        # execution, including the native TileLang sparse backend.
        connection_method="push",
        enable_fp_fusion=False,
        fast_math=False,
    )
    reference = _izhikevich_dual_exponential_stdp_model("torch")
    init_net_state(reference, dtype=torch.float32, device="cpu", persistent=True)
    native = _izhikevich_dual_exponential_stdp_model(options).to(
        device="cuda", dtype=torch.float32
    )
    init_net_state(native, dtype=torch.float32, device="cuda", persistent=True)
    native.load_state_dict(reference.state_dict())
    if recurrent == "helion":
        # Mix scalar and edge-aligned parameters in this additional anchor.
        # Scalar loads must broadcast to the cooperative CSR lanes without
        # applying a vector memory mask to a scalar pointer.
        for model in (reference, native):
            rule = model.synapse.linear.plasticity
            rule.a_plus = rule.a_plus[:1]
            rule.tau_minus = rule.tau_minus[:1]
    initial_weights = reference.synapse.linear.weight.value.detach().clone()

    inputs = torch.tensor(
        [
            [0.8, 0.2, 0.0],
            [0.8, 0.8, 0.3],
            [0.0, 0.8, 0.8],
            [0.9, 0.0, 0.9],
            [0.0, 0.9, 0.0],
            [0.85, 0.1, 0.7],
            [0.0, 0.85, 0.85],
            [0.8, 0.0, 0.8],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad(), environ.context(dt=1.0):
        for current in (inputs[:4], inputs[4:]):
            expected, expected_records = reference(current)
            actual, actual_records = native(current.cuda())
            torch.testing.assert_close(actual.cpu(), expected, atol=4e-4, rtol=4e-4)
            assert set(actual_records) == set(expected_records)
            for name, expected_state in expected_records.items():
                torch.testing.assert_close(
                    actual_records[name].cpu(),
                    expected_state,
                    atol=4e-4,
                    rtol=4e-4,
                )

    # Compare every persistent physical memory after continuation, not just
    # the selected records returned by the recurrent wrapper.
    expected_states = functional.named_hidden_states(reference)
    actual_states = functional.named_hidden_states(native)
    assert set(actual_states) == set(expected_states)
    for name, expected_state in expected_states.items():
        torch.testing.assert_close(
            actual_states[name].cpu(), expected_state, atol=4e-4, rtol=4e-4
        )
    assert int(expected.sum().item()) > 0
    assert not torch.equal(reference.synapse.linear.weight.value, initial_weights)


_HELION_STDP_RECORDS = (
    "neuron.v",
    "synapse.psc",
    "synapse.delayed_spike.history",
    "synapse.linear.plasticity.pre_trace",
    "synapse.linear.plasticity.post_trace",
    "synapse.linear.weight.value",
)
_HELION_STDP_INPUTS = torch.tensor(
    [
        [0.8, 0.8, 0.8, 0.8],
        [0.8, 0.8, 0.8, 0.8],
        [0.8, 0.8, 0.8, 0.8],
        [0.8, 0.8, 0.8, 0.8],
        [0.9, 0.7, 0.9, 0.7],
        [0.9, 0.7, 0.9, 0.7],
        [0.9, 0.7, 0.9, 0.7],
        [0.9, 0.7, 0.9, 0.7],
    ]
)


def _helion_triplet_rule(pair):
    """Retain the fixture's pair terms and add heterogeneous slow traces."""
    return TripletSTDP(
        TripletSTDPOptions(
            a_plus=pair.a_plus,
            a_minus=pair.a_minus,
            a3_plus=torch.linspace(0.1, 0.6, pair.tau_plus.numel()),
            a3_minus=torch.linspace(0.6, 0.1, pair.tau_plus.numel()),
            tau_plus=pair.tau_plus,
            tau_minus=pair.tau_minus,
            tau_x=torch.linspace(10.0, 15.0, pair.tau_plus.numel()),
            tau_y=torch.linspace(20.0, 25.0, pair.tau_plus.numel()),
        ),
        w_min=pair.w_min,
        w_max=pair.w_max,
    )


def _helion_power_rule(pair):
    """Retain trace timing and exercise signed, fractional and fixed bounds."""
    lower = pair.w_min.clone()
    lower[3] = pair.w_max[3]
    return PowerPairSTDP(
        PowerPairSTDPOptions(
            a_plus=pair.a_plus,
            a_minus=pair.a_minus,
            tau_plus=pair.tau_plus,
            tau_minus=pair.tau_minus,
            mu_plus=torch.tensor([0.0, 1.0, 0.5, 2.0, 0.0, 0.5]),
            mu_minus=torch.tensor([0.0, 1.0, 1.5, 0.25, 2.0, 0.5]),
        ),
        w_min=lower,
        w_max=pair.w_max,
    )


def _helion_nearest_rule(pair, *, pairing="symmetric"):
    """Replace accumulation while retaining heterogeneous pair coefficients."""
    return NearestPairSTDP(
        a_plus=pair.a_plus,
        a_minus=pair.a_minus,
        tau_plus=pair.tau_plus,
        tau_minus=pair.tau_minus,
        w_min=pair.w_min,
        w_max=pair.w_max,
        pairing=pairing,
    )


def _helion_inhibitory_rule(pair, *, trace_scope="connection"):
    """Use mixed signed encodings without inferring signs from live weights."""
    polarity = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    return InhibitorySTDP(
        eta=pair.a_plus * 0.1,
        tau=4.0 if trace_scope == "neuron" else pair.tau_plus,
        alpha=0.2,
        polarity=polarity,
        w_min=torch.where(polarity < 0, -0.1, 0.0),
        w_max=torch.where(polarity > 0, 0.1, 0.0),
        trace_scope=trace_scope,
    )


def _helion_stdp_lif_model(
    backend,
    *,
    device,
    dtype=torch.float32,
    chunk_size=4,
    rule_factory=None,
    delayed=True,
    dense=False,
):
    """Build a small D²CSR-shaped delayed connection for HELION parity."""
    from btorch.models.connection import EdgeWeight, SparseConnection, Synapse
    from btorch.models.functional import init_net_state
    from btorch.models.neurons import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import ExponentialPSC, HeterSynapsePSC

    # The semantic edge order is intentionally unsorted.  Sources 127 and 128
    # straddle a 128-lane CTA boundary, as do several destination targets;
    # this makes outgoing (PastSpike) and incoming (LTP) publication visible
    # in the partial final CTA without adding a backend/shape matrix.
    pre = torch.tensor([128, 0, 127, 64, 128, 1])
    post = torch.tensor([128, 127, 0, 128, 64, 1])
    delay = torch.tensor([2, 0, 1, 2, 0, 1])
    receptor = torch.tensor([1, 0, 1, 0, 1, 0])
    rule = PairSTDP(
        a_plus=torch.tensor([0.75, 0.55, 0.85, 0.65, 0.95, 0.45]),
        a_minus=torch.tensor([0.70, 0.50, 0.80, 0.60, 0.90, 0.40]),
        tau_plus=torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        tau_minus=torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        w_min=torch.tensor([-0.05, -0.02, -0.04, -0.03, -0.06, -0.01]),
        w_max=torch.tensor([0.051, 0.061, 0.071, 0.081, 0.091, 0.101]),
    )
    if dense:
        # More than 128 outgoing edges per source activates multiple consumer
        # CTAs per producer block. Weak weights keep externally selected events
        # separated from recurrent threshold changes.
        pre = torch.arange(129).repeat_interleave(129).flip(0)
        post = torch.arange(129).repeat(129).flip(0)
        delay = torch.zeros_like(pre)
        receptor = (pre + post).remainder(2)
        rule = PairSTDP(
            a_plus=0.0002,
            a_minus=0.0003,
            tau_plus=torch.linspace(3.0, 8.0, pre.numel()),
            tau_minus=torch.linspace(4.0, 9.0, pre.numel()),
            w_min=-0.0001,
            w_max=0.0001,
        )
    if rule_factory is not None:
        rule = rule_factory(rule)
    connection = SparseConnection.from_edges(
        pre,
        post,
        129,
        129,
        Synapse(
            weight=EdgeWeight(
                torch.full((pre.numel(),), 0.00005 if dense else 0.05),
                trainable=False,
            ),
            delay=delay if delayed else None,
            receptor=receptor,
            n_delay=3 if delayed else None,
            n_receptor=2,
            plasticity=rule,
        ),
    )
    model = RecurrentNN(
        LIF(
            129,
            v_threshold=0.5,
            v_reset=0.0,
            tau=1.0,
            hard_reset=True,
        ),
        HeterSynapsePSC(
            129,
            2,
            None,
            connection,
            base_psc=ExponentialPSC,
            tau_syn=2.0,
            **(dict(max_delay_steps=3, use_circular_buffer=False) if delayed else {}),
        ),
        backend=backend,
        update_state_names=(
            *_HELION_STDP_RECORDS[: 3 if delayed else 2],
            *("synapse.linear.plasticity." + name for name in rule.state_domains),
            "synapse.linear.weight.value",
        ),
        chunk_size=chunk_size,
    ).to(device=device, dtype=dtype)
    init_net_state(model, dtype=dtype, device=device, persistent=True)
    return model


def _seed_helion_stdp_state(model) -> None:
    """Seed nonzero traces and pending arrivals before the first timestep."""
    with torch.no_grad():
        rule = model.synapse.linear.plasticity
        rule.pre_trace.copy_(
            torch.linspace(
                0.2,
                0.7,
                rule.pre_trace.numel(),
                device=rule.pre_trace.device,
                dtype=rule.pre_trace.dtype,
            )
        )
        rule.post_trace.copy_(
            torch.linspace(
                0.6,
                0.1,
                rule.post_trace.numel(),
                device=rule.post_trace.device,
                dtype=rule.post_trace.dtype,
            )
        )
        # Nonzero slow traces must survive lazy chunk-boundary materialization,
        # not be reconstructed from the fast traces or current raster.
        for name in ("pre_slow", "post_slow"):
            if name in rule.state_domains:
                getattr(rule, name).copy_(rule.pre_trace * 0.7 + rule.post_trace * 0.3)
        # A populated delay history makes the first step exercise the same
        # arrival-time convention as later steps, including zero-lag events.
        if model.synapse.delayed_spike is not None:
            history = model.synapse.delayed_spike.history
            history.copy_(torch.ones_like(history))


@pytest.mark.parametrize(
    ("backend", "rule_factory"),
    (("triton", None), ("tilelang", _helion_triplet_rule)),
)
def test_fused_spike_list_stdp_uses_both_event_directions_across_chunks(
    backend, rule_factory
):
    """Compare producer-list learning with the eager canonical-edge oracle.

    Sources and destinations straddle CTA 128; some active neurons have
    no outgoing edges. Sparse pulses, silence and a simultaneous all-
    neuron pulse exercise outgoing combined LTP/LTD and incoming-only
    LTP without reading unused queue entries. Seeded traces make silent
    exponential evolution observable, including at the next chunk's
    boundary.
    """
    require_cuda_backend(backend)
    expected = _helion_stdp_lif_model(
        "torch", device="cpu", delayed=False, rule_factory=rule_factory
    )
    _seed_helion_stdp_state(expected)
    actual = _helion_stdp_lif_model(
        KernelOptions(
            neuron=backend,
            recurrent="helion",
            connection_method="push",
            spike_selection="spike_list",
            enable_fp_fusion=False,
            fast_math=False,
        ),
        device="cuda",
        delayed=False,
        rule_factory=rule_factory,
    )
    actual.load_state_dict(expected.state_dict())
    # TileLang also exercises publication inside Graph capture; this does
    # not multiply every rule by every execution mode.
    actual.cudagraph = backend == "tilelang"
    execute = (
        torch.compile(actual, fullgraph=True, mode="reduce-overhead")
        if backend == "triton"
        else actual
    )
    inputs = torch.zeros(8, 129)
    inputs[0, [0, 1, 64, 127, 128]] = 0.8
    inputs[2, [1, 64, 128]] = 0.8
    inputs[4] = 0.8
    inputs[6, [0, 127]] = 0.8
    initial_weight = expected.synapse.linear.weight.value.clone()
    with torch.no_grad(), environ.context(dt=1.0):
        for chunk in inputs.split(4):
            spikes, records = expected(chunk)
            observed, observed_records = execute(chunk.cuda())
            torch.testing.assert_close(observed.cpu(), spikes, atol=0, rtol=0)
            _assert_helion_stdp_records(observed_records, records)
            for name, value in functional.named_hidden_states(expected).items():
                torch.testing.assert_close(
                    functional.named_hidden_states(actual)[name].cpu(),
                    value,
                    atol=5e-4,
                    rtol=5e-4,
                )
    assert not torch.equal(expected.synapse.linear.weight.value, initial_weight)


@pytest.mark.parametrize(
    ("neuron_backend", "connection_backend", "selection", "rule_factory"),
    (
        ("tilelang", "triton", "scan", _helion_triplet_rule),
        ("triton", "tilelang", "spike_list", None),
    ),
)
def test_split_sparse_stdp_records_follow_event_owners(
    neuron_backend, connection_backend, selection, rule_factory
):
    """Record active and inactive edges without cross-block history races.

    Mix renderers instead of constructing a model/options matrix. Initial
    weights exceed bounds: active edges must transmit the old weight before
    updating, while silent edges must still clip and record. Subsequent
    pulses activate both edge directions and the partial final producer
    block. Dense fanout activates two consumer partitions per producer block;
    their outgoing/incoming updates and inactive histories must have unique
    owners. Strided canonical state must remain writable without packing.
    """
    require_cuda_backend(neuron_backend)
    require_cuda_backend(connection_backend)
    expected = _helion_stdp_lif_model(
        "torch", device="cpu", delayed=False, dense=True, rule_factory=rule_factory
    )
    _seed_helion_stdp_state(expected)
    with torch.no_grad():
        expected.synapse.linear.weight.value.copy_(
            expected.synapse.linear.plasticity.w_max + 0.01
        )
    actual = _helion_stdp_lif_model(
        KernelOptions(
            neuron=neuron_backend,
            connection=connection_backend,
            connection_method="push",
            recurrent="standard",
            spike_selection=selection,
            enable_fp_fusion=False,
            fast_math=False,
        ),
        device="cuda",
        delayed=False,
        dense=True,
        rule_factory=rule_factory,
    )
    actual.load_state_dict(expected.state_dict())
    for owner, names in (
        (actual.synapse.linear.weight, ("value",)),
        (
            actual.synapse.linear.plasticity,
            actual.synapse.linear.plasticity.state_domains,
        ),
    ):
        for name in names:
            value = getattr(owner, name)
            padded = value.new_full((2 * value.numel(),), float("nan"))
            view = padded[::2].view_as(value)
            view.copy_(value)
            setattr(owner, name, view)
    actual.cudagraph = neuron_backend == "tilelang"
    execute = (
        actual
        if actual.cudagraph
        else torch.compile(actual, fullgraph=True, mode="reduce-overhead")
    )
    inputs = torch.zeros(8, 129)
    inputs[0, 0] = 0.8
    inputs[2, 128] = 0.8
    inputs[3] = 0.8
    inputs[5, [1, 64, 127]] = 0.8
    inputs[7, [0, 128]] = 0.8
    with torch.no_grad(), environ.context(dt=1.0):
        for chunk in inputs.split(4):
            spikes, records = expected(chunk)
            observed, observed_records = execute(chunk.cuda())
            torch.testing.assert_close(observed.cpu(), spikes, atol=0, rtol=0)
            _assert_helion_stdp_records(observed_records, records)
            for name, value in functional.named_hidden_states(expected).items():
                torch.testing.assert_close(
                    functional.named_hidden_states(actual)[name].cpu(),
                    value,
                    atol=5e-4,
                    rtol=5e-4,
                )


def _helion_stdp_multipass_lif_model(
    backend,
    *,
    device,
    dtype=torch.float32,
    rule_factory=None,
    pre_spike_v=False,
    hub_rows=False,
):
    """Build a delayed model with more CSR segments than one padded CTA."""
    from btorch.models.connection import EdgeWeight, SparseConnection, Synapse
    from btorch.models.functional import init_net_state
    from btorch.models.neurons import LIF
    from btorch.models.rnn import RecurrentNN
    from btorch.models.synapse import ExponentialPSC, HeterSynapsePSC

    n = 129
    source = torch.arange(n).repeat(3)
    delay = torch.arange(3).repeat_interleave(n)
    target = (source + delay + 1) % n
    receptor = delay.remainder(2)
    if hub_rows:
        # Replace source 0's original delay-zero edge with its hub row so
        # canonical coordinates remain distinct; plastic edges cannot coalesce.
        source, target, delay, receptor = (
            value[1:] for value in (source, target, delay, receptor)
        )
        # Outgoing source 0 and incoming post 0 each have a row longer than
        # 128 edges. All three cooperation widths must execute multiple row
        # iterations; the surrounding degree-one rows exercise inactive groups.
        source = torch.cat((source, torch.zeros(n, dtype=torch.long), torch.arange(n)))
        target = torch.cat((target, torch.arange(n), torch.zeros(n, dtype=torch.long)))
        delay = torch.cat(
            (delay, torch.zeros(n, dtype=torch.long), torch.full((n,), 2))
        )
        receptor = torch.cat(
            (
                receptor,
                torch.zeros(n, dtype=torch.long),
                torch.ones(n, dtype=torch.long),
            )
        )
    n_edge = source.numel()
    rule = PairSTDP(
        a_plus=torch.linspace(0.20, 0.45, n_edge),
        a_minus=torch.linspace(0.15, 0.40, n_edge),
        tau_plus=torch.linspace(2.0, 5.0, n_edge),
        tau_minus=torch.linspace(3.0, 6.0, n_edge),
        w_min=torch.full((n_edge,), -0.05),
        w_max=torch.full((n_edge,), 0.15),
    )
    if rule_factory is not None:
        rule = rule_factory(rule)
    connection = SparseConnection.from_edges(
        source,
        target,
        n,
        n,
        Synapse(
            weight=EdgeWeight(torch.full((n_edge,), 0.05), trainable=False),
            delay=delay,
            receptor=receptor,
            n_delay=3,
            n_receptor=2,
            plasticity=rule,
        ),
    )
    model = RecurrentNN(
        LIF(
            n,
            v_threshold=0.5,
            v_reset=0.0,
            tau=1.0,
            hard_reset=True,
            pre_spike_v=pre_spike_v,
        ),
        HeterSynapsePSC(
            n,
            2,
            None,
            connection,
            base_psc=ExponentialPSC,
            tau_syn=2.0,
            max_delay_steps=3,
            use_circular_buffer=False,
        ),
        backend=backend,
        update_state_names=(
            *_HELION_STDP_RECORDS[:3],
            *("synapse.linear.plasticity." + name for name in rule.state_domains),
            "synapse.linear.weight.value",
        ),
        chunk_size=2,
    ).to(device=device, dtype=dtype)
    init_net_state(model, dtype=dtype, device=device, persistent=True)
    return model


@pytest.mark.parametrize(
    ("backend", "dtype"),
    (
        ("triton", torch.float32),
        ("tilelang", torch.float64),
        ("triton", torch.float16),
        ("tilelang", torch.bfloat16),
    ),
)
def test_helion_voltage_stdp_continuous_edges_delays_and_chunk_state(backend, dtype):
    """Match dense continuous updates across partial CTAs and edge passes.

    Seeded eligibility and pending arrivals make both LTD and silent LTP
    observable immediately. There are more edges than padded neuron
    lanes; every canonical edge must evolve, even when neither endpoint
    fires.
    """
    require_cuda_backend(backend)

    def voltage(pair):
        return VoltageSTDP(
            VoltageSTDPOptions(
                a_plus=0.01,
                a_minus=0.02,
                tau_pre=torch.linspace(2.0, 5.0, pair.a_plus.numel()),
                tau_plus=torch.linspace(1.0, 3.0, 129),
                tau_minus=torch.linspace(2.0, 4.0, 129),
                theta_plus=0.25,
                theta_minus=0.1,
                v_rest=0.0,
            ),
            w_min=pair.w_min,
            w_max=pair.w_max,
        )

    options = KernelOptions(
        neuron=backend,
        recurrent="helion",
        connection_method="push",
        enable_fp_fusion=False,
        fast_math=False,
    )
    expected = _helion_stdp_multipass_lif_model(
        "torch",
        device="cpu",
        dtype=dtype,
        rule_factory=voltage,
        pre_spike_v=True,
    )
    actual = _helion_stdp_multipass_lif_model(
        options,
        device="cuda",
        dtype=dtype,
        rule_factory=voltage,
        pre_spike_v=True,
    )
    with torch.no_grad():
        expected.synapse.linear.plasticity.pre_trace.fill_(0.2)
        expected.synapse.delayed_spike.history.fill_(1.0)
        actual.load_state_dict(expected.state_dict())
        weight = actual.synapse.linear.weight.value
        storage = weight.new_full((2 * weight.numel(),), -9.0)
        storage[::2].copy_(weight)
        actual.synapse.linear.weight.value = torch.nn.Parameter(
            storage[::2], requires_grad=False
        )
    inputs = torch.tensor([0.8, 0.15, 0.0, 0.35, 0.2, 0.9], dtype=dtype)
    inputs = inputs[:, None].expand(-1, 129)
    tolerance = (
        2e-11
        if dtype == torch.float64
        else 4 * torch.finfo(dtype).eps
        if dtype in (torch.float16, torch.bfloat16)
        else 3e-6
    )
    with torch.no_grad():
        for index, chunk in enumerate(inputs.split(3)):
            with environ.context(dt=1.0 if index == 0 else 0.5):
                reference, reference_records = expected(chunk)
                output, records = actual(chunk.cuda())
                torch.testing.assert_close(output.cpu(), reference)
                _assert_helion_stdp_records(
                    records,
                    reference_records,
                    atol=tolerance,
                    rtol=tolerance,
                )
                for name, value in expected.state_dict().items():
                    torch.testing.assert_close(
                        actual.state_dict()[name].cpu(),
                        value,
                        atol=tolerance,
                        rtol=tolerance,
                    )
    torch.testing.assert_close(storage[1::2], torch.full_like(weight, -9.0))


def _assert_helion_stdp_records(actual, expected, *, atol=5e-4, rtol=5e-4) -> None:
    """Compare returned physical memories, including copied edge histories."""
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(
            actual[name].cpu(),
            expected[name],
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.parametrize(
    ("backend", "parallelism"),
    (
        ("triton", "thread"),
        ("triton", "warp"),
        ("triton", "block"),
        ("tilelang", "thread"),
        ("tilelang", "warp"),
        ("tilelang", "block"),
    ),
)
def test_helion_stdp_traversal_modes_multipass_segments_match_eager(
    backend, parallelism, monkeypatch
):
    """Exercise each connection traversal across a padded CTA boundary.

    There are three distinct delay segments for every one of 129
    sources, giving S=387 segments.  The padded source capacity is 256
    lanes, so the propagation phase must discover and process segments
    in multiple passes; seeded delay history and traces make the first
    pass observable. Outgoing and incoming hub rows exceed 128 edges so
    every mode also exercises cooperative row tails and multiple
    iterations.
    """
    from btorch.models.kernels.backends import helion_plasticity

    require_cuda_backend(backend)
    # Select an internal traversal specialization without altering the public
    # rule/model API. Each case checks the same complete state/weight history,
    # including cross-group arrival/post coincidences and inactive tail lanes.
    monkeypatch.setattr(
        helion_plasticity,
        "prepare_stdp_phases",
        partial(helion_plasticity.prepare_stdp_phases, parallelism=parallelism),
    )
    options = KernelOptions(
        neuron=backend,
        recurrent="helion",
        connection_method="push",
        enable_fp_fusion=False,
        fast_math=False,
    )
    expected = _helion_stdp_multipass_lif_model("torch", device="cpu", hub_rows=True)
    actual = _helion_stdp_multipass_lif_model(options, device="cuda", hub_rows=True)
    _seed_helion_stdp_state(expected)
    actual.load_state_dict(expected.state_dict())
    history = expected.synapse.delayed_spike.history
    with torch.no_grad():
        history.copy_(
            (torch.arange(history.numel()).reshape_as(history) % 3 == 0).to(history)
        )
        actual.load_state_dict(expected.state_dict())

    inputs = torch.full((6, 129), 0.8)
    with torch.no_grad(), environ.context(dt=1.0):
        for chunk in inputs.split(2):
            expected_spikes, expected_records = expected(chunk)
            actual_spikes, actual_records = actual(chunk.cuda())
            torch.testing.assert_close(
                actual_spikes.cpu(), expected_spikes, atol=0, rtol=0
            )
            _assert_helion_stdp_records(actual_records, expected_records)

    expected_states = functional.named_hidden_states(expected)
    actual_states = functional.named_hidden_states(actual)
    assert set(actual_states) == set(expected_states)
    for name, expected_state in expected_states.items():
        torch.testing.assert_close(
            actual_states[name].cpu(), expected_state, atol=5e-4, rtol=5e-4
        )


@pytest.mark.parametrize(
    ("backend", "dtype", "rule_factory"),
    (
        ("triton", torch.float32, _helion_triplet_rule),
        ("tilelang", torch.float32, _helion_triplet_rule),
        ("triton", torch.float64, None),
        ("triton", torch.float16, None),
        ("tilelang", torch.bfloat16, _helion_triplet_rule),
        ("triton", torch.float32, _helion_power_rule),
        ("tilelang", torch.float32, _helion_power_rule),
        ("triton", torch.float32, _helion_nearest_rule),
        (
            "tilelang",
            torch.float32,
            partial(_helion_nearest_rule, pairing="restricted"),
        ),
    ),
)
def test_helion_online_stdp_matches_eager_delayed_lif_across_chunks(
    backend, dtype, rule_factory
):
    """Match D²CSR/PastSpike updates with eager STDP across two invocations.

    The fixture has heterogeneous edge parameters, receptors, delays and
    nonzero initial traces.  The first returned history is cloned before
    the second call; it must remain unchanged while the kernel lazily
    materializes the next chunk's edge traces and weights.
    """
    require_cuda_backend(backend)
    options = KernelOptions(
        neuron=backend,
        recurrent="helion",
        connection_method="push",
        enable_fp_fusion=False,
        fast_math=False,
    )
    expected = _helion_stdp_lif_model(
        "torch", device="cpu", dtype=dtype, rule_factory=rule_factory
    )
    actual = _helion_stdp_lif_model(
        options, device="cuda", dtype=dtype, rule_factory=rule_factory
    )
    expected.synapse.linear.plasticity.a_plus.fill_(0.5)
    actual.load_state_dict(expected.state_dict())
    # Canonical edge IDs are logical slots, not physical storage offsets.
    # Keep padding between weights, traces and heterogeneous parameters so
    # raw sparse loads/stores must respect views without packing each step.
    connection = actual.synapse.linear
    for owner, names in (
        (connection.weight, ("value",)),
        (
            connection.plasticity,
            (
                *connection.plasticity.state_domains,
                *get_plasticity_template(
                    type(connection.plasticity).kernel_model
                ).parameters,
            ),
        ),
    ):
        for name in names:
            value = getattr(owner, name)
            storage = value.new_full((2 * value.numel(),), float("nan"))
            view = storage[::2].view(value.shape)
            view.copy_(value)
            setattr(owner, name, view)
    inputs = _HELION_STDP_INPUTS.to(dtype=dtype).repeat(1, 33)[:, :129]
    input_storage = torch.empty(
        (inputs.shape[0], 2 * inputs.shape[1]), device="cuda", dtype=dtype
    )
    actual_inputs = input_storage[:, ::2]
    actual_inputs.copy_(inputs)
    initial_weights = expected.synapse.linear.weight.value.detach().clone()
    quiet_input = torch.zeros((1, 129), dtype=dtype)
    tolerance = (
        2e-10
        if dtype == torch.float64
        else 4 * torch.finfo(dtype).eps
        if dtype in (torch.float16, torch.bfloat16)
        else 5e-4
    )
    with torch.no_grad(), environ.context(dt=1.0):
        # A quiet step still applies the rule's combined clamp.  This catches
        # a HELION path that only materializes weight updates when events exist.
        quiet_weight = expected.synapse.linear.plasticity.w_max + 0.2
        expected.synapse.linear.weight.value.copy_(quiet_weight)
        expected.synapse.delayed_spike.history.zero_()
        actual.load_state_dict(expected.state_dict())
        quiet_expected, quiet_expected_records = expected(quiet_input)
        quiet_actual, quiet_actual_records = actual(quiet_input.cuda())
        torch.testing.assert_close(quiet_actual.cpu(), quiet_expected, atol=0, rtol=0)
        _assert_helion_stdp_records(
            quiet_actual_records,
            quiet_expected_records,
            atol=tolerance,
            rtol=tolerance,
        )
        torch.testing.assert_close(
            expected.synapse.linear.weight.value,
            expected.synapse.linear.plasticity.w_max,
            atol=0,
            rtol=0,
        )

        # Reset physical memories, restore the original weights, and seed the
        # actual learning trajectory independently of the quiet preflight.
        functional.reset_net_state(expected)
        functional.reset_net_state(actual)
        expected.synapse.linear.weight.value.copy_(initial_weights)
        actual.load_state_dict(expected.state_dict())
        _seed_helion_stdp_state(expected)
        actual.load_state_dict(expected.state_dict())
        # Read-only parameters may broadcast even though writable weights
        # cannot. Keep this zero-stride view through both actual chunks;
        # the reference retains its equivalent full-sized parameter buffer.
        rule = actual.synapse.linear.plasticity
        rule.a_plus = rule.a_plus[:1].expand_as(rule.a_plus)
    first_actual_records = None
    first_actual_snapshot = None
    with torch.no_grad():
        for chunk_index, (chunk, actual_chunk) in enumerate(
            zip(
                (inputs[:4], inputs[4:]),
                (actual_inputs[:4], actual_inputs[4:]),
                strict=True,
            )
        ):
            dt = 1.0 if chunk_index == 0 else 0.5
            with environ.context(dt=dt):
                expected_spikes, expected_records = expected(chunk)
                actual_spikes, actual_records = actual(actual_chunk)
                if first_actual_records is None:
                    # tau=1 and the first all-neuron pulse force simultaneous
                    # postsynaptic spikes.  The zero-delay edges therefore also
                    # exercise the strict-past coincident-event rule.
                    assert torch.all(expected_spikes[0] != 0)
                torch.testing.assert_close(
                    actual_spikes.cpu(), expected_spikes, atol=0, rtol=0
                )
                _assert_helion_stdp_records(
                    actual_records,
                    expected_records,
                    atol=tolerance,
                    rtol=tolerance,
                )
                if first_actual_records is None:
                    first_actual_records = actual_records
                    first_actual_snapshot = {
                        name: value.clone() for name, value in actual_records.items()
                    }

    # The second call must not rewrite tensors returned by the first call.
    assert first_actual_records is not None
    assert first_actual_snapshot is not None
    for name, snapshot in first_actual_snapshot.items():
        torch.testing.assert_close(first_actual_records[name], snapshot)
        assert snapshot.shape[0] == 4

    expected_states = functional.named_hidden_states(expected)
    actual_states = functional.named_hidden_states(actual)
    assert set(actual_states) == set(expected_states)
    for name, expected_state in expected_states.items():
        torch.testing.assert_close(
            actual_states[name].cpu(),
            expected_state,
            atol=tolerance,
            rtol=tolerance,
        )

    weights = actual.synapse.linear.weight.value
    bounds = actual.synapse.linear.plasticity.w_max
    assert torch.isclose(weights, bounds, atol=5e-4, rtol=0).any()


def test_helion_online_stdp_triton_graph_and_compile_continue():
    """Exercise one Triton graph and one fullgraph continuation path.

    Each runner receives two changed chunks from the same seeded state.
    The eager model is rebuilt per runner, so graph capture and Dynamo
    compilation are checked independently without a backend-by-mode
    Cartesian matrix.
    """
    require_cuda_backend("triton")
    options = KernelOptions(
        neuron="triton",
        recurrent="helion",
        connection_method="push",
        enable_fp_fusion=False,
        fast_math=False,
    )
    inputs = _HELION_STDP_INPUTS.repeat(1, 33)[:, :129]
    with torch.no_grad(), environ.context(dt=1.0):
        for mode, rule_factory in (
            ("graph", _helion_triplet_rule),
            ("compile", _helion_power_rule),
            ("compile", partial(_helion_nearest_rule, pairing="restricted")),
        ):
            expected = _helion_stdp_lif_model(
                "torch", device="cpu", rule_factory=rule_factory
            )
            actual = _helion_stdp_lif_model(
                options, device="cuda", rule_factory=rule_factory
            )
            _seed_helion_stdp_state(expected)
            actual.load_state_dict(expected.state_dict())
            actual.cudagraph = mode == "graph"
            execute = (
                torch.compile(actual, fullgraph=True, mode="reduce-overhead")
                if mode == "compile"
                else actual
            )
            for step, chunk in enumerate((inputs[:4], inputs[4:])):
                if step and isinstance(
                    actual.synapse.linear.plasticity, NearestPairSTDP
                ):
                    # Compiled calls must load the live policy without replacing
                    # storage or clearing the eligibility learned in chunk 1.
                    actual.synapse.linear.plasticity.restricted.fill_(False)
                    expected.synapse.linear.plasticity.restricted.fill_(False)
                expected_spikes, expected_records = expected(chunk)
                actual_spikes, actual_records = execute(chunk.cuda())
                torch.testing.assert_close(
                    actual_spikes.cpu(), expected_spikes, atol=0, rtol=0
                )
                _assert_helion_stdp_records(actual_records, expected_records)
            expected_states = functional.named_hidden_states(expected)
            actual_states = functional.named_hidden_states(actual)
            for name, expected_state in expected_states.items():
                torch.testing.assert_close(
                    actual_states[name].cpu(), expected_state, atol=5e-4, rtol=5e-4
                )


@pytest.mark.parametrize(
    ("backend", "rule_kind"), (("triton", "pair"), ("tilelang", "voltage"))
)
def test_helion_trial_scratch_preserves_live_learning_and_recurrent_state(
    backend, rule_kind, monkeypatch
):
    """Run real cooperative trial steps without consuming live learning state.

    Test an event rule and a continuous rule on different renderers
    rather than multiplying every rule/backend combination. Shared
    output/input storage, delayed drive, recorded histories and private
    barrier clocks must all be isolated by the same scratch utility used
    by native tuning.
    """
    from btorch.models.kernels.backends.renderers import get_runtime
    from btorch.utils.autotune import scratch_arguments

    require_cuda_backend(backend)
    runtime = get_runtime(backend)
    original_prepare = runtime.prepare_launch
    trials = []

    def prepare_with_trial(kernel, arguments, n, **options):
        if options.get("launch_cooperative_grid", False):
            before = [value.clone() for value in arguments]
            scratch, reset = scratch_arguments(arguments, options["writes"])
            candidate = original_prepare(kernel, scratch, n, **options)
            candidate(scratch)
            torch.cuda.synchronize()
            # Missing any mutation (including canonical weights, lazy clocks
            # or future drive) makes a scratch trial modify a live argument.
            for value, saved in zip(arguments, before, strict=True):
                torch.testing.assert_close(value, saved, atol=0, rtol=0, equal_nan=True)
            assert any(
                not torch.equal(scratch[index], before[index])
                for index in options["writes"]
            )
            reset()
            for value, saved in zip(scratch, before, strict=True):
                torch.testing.assert_close(value, saved, atol=0, rtol=0, equal_nan=True)
            trials.append(True)
        return original_prepare(kernel, arguments, n, **options)

    monkeypatch.setattr(runtime, "prepare_launch", prepare_with_trial)
    rule_factory = None
    if rule_kind == "voltage":
        rule_factory = lambda pair: VoltageSTDP(  # noqa: E731 - local fixture
            VoltageSTDPOptions(
                a_plus=0.01,
                a_minus=0.02,
                theta_plus=0.25,
                theta_minus=0.1,
                v_rest=0.0,
            ),
            w_min=pair.w_min,
            w_max=pair.w_max,
        )
    expected = _helion_stdp_multipass_lif_model(
        "torch",
        device="cpu",
        rule_factory=rule_factory,
        pre_spike_v=rule_kind == "voltage",
    )
    actual = _helion_stdp_multipass_lif_model(
        KernelOptions(
            neuron=backend,
            recurrent="helion",
            connection_method="push",
            fast_math=False,
            enable_fp_fusion=False,
        ),
        device="cuda",
        rule_factory=rule_factory,
        pre_spike_v=rule_kind == "voltage",
    )
    rule = expected.synapse.linear.plasticity
    if rule_kind == "pair":
        _seed_helion_stdp_state(expected)
    else:
        rule.pre_trace.fill_(0.2)
        rule.voltage_plus.fill_(0.3)
        rule.voltage_minus.fill_(0.2)
    expected.synapse.delayed_spike.history.fill_(1)
    actual.load_state_dict(expected.state_dict())
    original_weight = actual.synapse.linear.weight.value.clone()
    inputs = torch.full((6, 129), 0.8)
    with torch.no_grad(), environ.context(dt=1.0):
        for chunk in inputs.split(2):
            expected_spikes, expected_records = expected(chunk)
            actual_spikes, actual_records = actual(chunk.cuda())
            torch.testing.assert_close(actual_spikes.cpu(), expected_spikes)
            _assert_helion_stdp_records(actual_records, expected_records)
    assert trials
    assert not torch.equal(actual.synapse.linear.weight.value, original_weight)


@pytest.mark.parametrize(
    ("backend", "delayed", "selection"),
    (("triton", False, "spike_list"), ("tilelang", True, "scan")),
)
def test_helion_shared_traces_publish_after_all_edge_readers(
    backend, delayed, selection
):
    """Keep strict-past compact traces read-only until the propagation fence.

    A partial second CTA, simultaneous events and receptor duplicates
    distinguish domain writers from per-edge writes. The delayed FP64
    case has more segments than padded lanes and consumes pending
    history. The list case traverses multiple edge tiles, captures and
    compiles continued chunks without a separate trace materialization
    launch.
    """
    require_cuda_backend(backend)

    def rule(pair):
        return PairSTDP(
            a_plus=pair.a_plus,
            a_minus=pair.a_minus,
            tau_plus=3.0,
            tau_minus=4.0,
            w_min=pair.w_min,
            w_max=pair.w_max,
            trace_scope="neuron",
        )

    options = KernelOptions(
        neuron=backend,
        connection=backend,
        recurrent="helion",
        connection_method="push",
        spike_selection=selection,
        enable_fp_fusion=False,
        fast_math=False,
    )
    dtype = torch.float64 if delayed else torch.float32
    factory = _helion_stdp_multipass_lif_model if delayed else _helion_stdp_lif_model
    fixture = {} if delayed else {"delayed": False, "dense": True}
    expected = factory("torch", device="cpu", rule_factory=rule, dtype=dtype, **fixture)
    actual = factory(options, device="cuda", rule_factory=rule, dtype=dtype, **fixture)
    _seed_helion_stdp_state(expected)
    actual.load_state_dict(expected.state_dict())
    for owner in (actual.synapse.linear.plasticity,):
        for name in owner.state_domains:
            value = getattr(owner, name)
            storage = value.new_empty(value.numel() * 2)
            view = storage[::2]
            view.copy_(value)
            setattr(owner, name, view)
    actual.cudagraph = not delayed
    inputs = _HELION_STDP_INPUTS.repeat(1, 33)[:, :129].to(dtype=dtype)
    execute = actual
    with torch.no_grad(), environ.context(dt=1.0):
        for step, chunk in enumerate(inputs.split(4)):
            target, target_records = expected(chunk)
            result, records = execute(chunk.cuda())
            torch.testing.assert_close(result.cpu(), target, atol=0, rtol=0)
            _assert_helion_stdp_records(records, target_records)
            for name, state in functional.named_hidden_states(expected).items():
                torch.testing.assert_close(
                    functional.named_hidden_states(actual)[name].cpu(),
                    state,
                    atol=5e-4,
                    rtol=5e-4,
                )
            if not delayed and step == 0:
                # Capture correctness is checked above; fullgraph must bind the
                # same compact persistent state on the next continuation.
                actual.cudagraph = False
                torch.compiler.reset()
                execute = torch.compile(actual, fullgraph=True, mode="reduce-overhead")
    if not delayed:
        # Fused incoming traversal uses src_seg_ptr directly: no duplicate
        # canonical-edge inverse is needed for a zero-delay population.
        assert actual.synapse.linear.cache.edge_segment.numel() == 0


@pytest.mark.parametrize(
    ("backend", "selection"), (("triton", "scan"), ("tilelang", "spike_list"))
)
def test_split_shared_traces_keep_old_snapshots_across_buffer_swaps(backend, selection):
    """Use next buffers instead of a global barrier in ordinary event learning.

    Dense rows force multiple consumer partitions, whose trace readers
    may overlap other CTAs' next-state writers. Odd/even chunks, strided
    memories, requested histories and capture restoration check both
    buffer parities.
    """
    require_cuda_backend(backend)

    def rule(pair):
        return PairSTDP(
            a_plus=pair.a_plus,
            a_minus=pair.a_minus,
            tau_plus=3.0,
            tau_minus=4.0,
            w_min=pair.w_min,
            w_max=pair.w_max,
            trace_scope="neuron",
        )

    options = KernelOptions(
        neuron=backend,
        connection=backend,
        recurrent="standard",
        connection_method="push",
        spike_selection=selection,
        enable_fp_fusion=False,
        fast_math=False,
    )
    expected = _helion_stdp_lif_model(
        "torch",
        device="cpu",
        rule_factory=rule,
        delayed=False,
        dense=True,
    )
    actual = _helion_stdp_lif_model(
        options,
        device="cuda",
        rule_factory=rule,
        delayed=False,
        dense=True,
    )
    # Outer calls define the parity boundaries; do not split the three-step
    # chunks again internally before they exercise the odd-length copy-out.
    expected.chunk_size = actual.chunk_size = 3
    _seed_helion_stdp_state(expected)
    actual.load_state_dict(expected.state_dict())
    for name in actual.synapse.linear.plasticity.state_domains:
        value = getattr(actual.synapse.linear.plasticity, name)
        storage = value.new_empty(2 * value.numel())
        view = storage[::2]
        view.copy_(value)
        setattr(actual.synapse.linear.plasticity, name, view)
    actual.cudagraph = backend == "triton"
    inputs = _HELION_STDP_INPUTS.repeat(1, 33)[:, :129]
    execute = actual
    captures = None
    with torch.no_grad(), environ.context(dt=1.0):
        for index, chunk in enumerate((inputs[:3], inputs[3:6], inputs[6:])):
            target, target_records = expected(chunk)
            result, records = execute(chunk.cuda())
            torch.testing.assert_close(result.cpu(), target, atol=0, rtol=0)
            _assert_helion_stdp_records(records, target_records)
            for name, state in functional.named_hidden_states(expected).items():
                torch.testing.assert_close(
                    functional.named_hidden_states(actual)[name].cpu(),
                    state,
                    atol=5e-4,
                    rtol=5e-4,
                )
            if backend == "triton" and index == 0:
                captures = actual._cudagraph_runner.n_captures
                assert captures > 0
            if backend == "triton" and index == 1:
                # Reuse the captured odd-length chunk with new inputs and live
                # learned memories before testing even-length compiled execution.
                assert actual._cudagraph_runner.n_captures == captures
                actual.cudagraph = False
                torch.compiler.reset()
                execute = torch.compile(actual, fullgraph=True, mode="reduce-overhead")
    assert actual.synapse.linear.cache.edge_segment.numel() == 0


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_auto_dense_stdp_refreshes_live_weights_across_graph_and_compiled_chunks(
    backend,
):
    """Reuse dense propagation while keeping canonical online weights live."""
    require_cuda_backend(backend)

    def rule(pair):
        if backend == "triton":
            return _helion_triplet_rule(pair)
        return PairSTDP(
            a_plus=pair.a_plus,
            a_minus=pair.a_minus,
            tau_plus=3.0,
            tau_minus=4.0,
            w_min=pair.w_min,
            w_max=pair.w_max,
            trace_scope="neuron",
        )

    # The two anchors cover edge-owned triplet and shared pair stages without
    # a family/backend/options Cartesian matrix. Receptor routing leaves empty
    # dense entries, while clipping changes weights on the very first step.
    expected = _helion_stdp_lif_model(
        "torch",
        device="cpu",
        rule_factory=rule,
        delayed=backend == "tilelang",
        chunk_size=3,
    )
    actual = _helion_stdp_lif_model(
        KernelOptions(
            neuron=backend,
            enable_fp_fusion=False,
            fast_math=False,
        ),
        device="cuda",
        rule_factory=rule,
        delayed=backend == "tilelang",
        chunk_size=3,
    )
    _seed_helion_stdp_state(expected)
    actual.load_state_dict(expected.state_dict())
    initial = expected.synapse.linear.weight.value.detach().clone()
    actual.cudagraph = True
    inputs = _HELION_STDP_INPUTS.repeat(1, 33)[:, :129]
    execute = actual
    captures = None
    with torch.no_grad(), environ.context(dt=1.0):
        for index, chunk in enumerate((inputs[:3], inputs[3:6], inputs[6:])):
            target, target_records = expected(chunk)
            result, records = execute(chunk.cuda())
            torch.testing.assert_close(result.cpu(), target, atol=0, rtol=0)
            _assert_helion_stdp_records(records, target_records)
            for name, state in functional.named_hidden_states(expected).items():
                torch.testing.assert_close(
                    functional.named_hidden_states(actual)[name].cpu(),
                    state,
                    atol=5e-4,
                    rtol=5e-4,
                )
            if index == 0:
                captures = actual._cudagraph_runner.n_captures
                assert captures > 0
            if index == 1:
                assert actual._cudagraph_runner.n_captures == captures
                actual.cudagraph = False
                torch.compiler.reset()
                execute = torch.compile(actual, fullgraph=True, mode="reduce-overhead")
    assert not torch.equal(expected.synapse.linear.weight.value, initial)
    assert actual.synapse.linear._plan.route.key.representation == "dense"


@pytest.mark.parametrize(
    ("backend", "dtype"), (("triton", torch.float16), ("tilelang", torch.float64))
)
def test_native_inhibitory_stdp_preserves_signed_learning_and_storage(backend, dtype):
    """Check signed weakening/pairing and promoted execution mirrors per
    step."""
    require_cuda_backend(backend)
    _run_native_case(backend, dtype, strided=True, rule_factory=_inhibitory_rule)


def _check_inhibitory_recurrence(options, trace_scope, *, delayed=True):
    """Compare every state/weight after captured and compiled continuation."""
    factory = partial(
        _helion_inhibitory_rule,
        trace_scope=trace_scope,
    )
    expected = _helion_stdp_lif_model(
        "torch", device="cpu", rule_factory=factory, chunk_size=3, delayed=delayed
    )
    actual = _helion_stdp_lif_model(
        options,
        device="cuda",
        rule_factory=factory,
        chunk_size=3,
        delayed=delayed,
    )
    _seed_helion_stdp_state(expected)
    with torch.no_grad():
        expected.synapse.linear.weight.value.copy_(
            0.05 * expected.synapse.linear.plasticity.polarity
        )
    actual.load_state_dict(expected.state_dict())
    actual.cudagraph = True
    inputs = _HELION_STDP_INPUTS.repeat(1, 33)[:, :129]
    with torch.no_grad(), environ.context(dt=1.0):
        for index, chunk in enumerate((inputs[:6], inputs[6:])):
            target, target_records = expected(chunk)
            execute = (
                actual
                if index == 0
                else torch.compile(actual, fullgraph=True, mode="reduce-overhead")
            )
            result, records = execute(chunk.cuda())
            torch.testing.assert_close(result.cpu(), target, atol=0, rtol=0)
            _assert_helion_stdp_records(records, target_records)
            for name, state in functional.named_hidden_states(expected).items():
                torch.testing.assert_close(
                    functional.named_hidden_states(actual)[name].cpu(),
                    state,
                    atol=5e-4,
                    rtol=5e-4,
                )
            if index == 0:
                assert actual._cudagraph_runner.n_captures == 1
                # A fresh module must continue from public physical memories,
                # not a captured kernel's private scratch or lazy clocks.
                checkpoint = io.BytesIO()
                torch.save(actual.state_dict(), checkpoint)
                checkpoint.seek(0)
                actual = _helion_stdp_lif_model(
                    options,
                    device="cuda",
                    rule_factory=factory,
                    chunk_size=3,
                    delayed=delayed,
                )
                actual.load_state_dict(torch.load(checkpoint, weights_only=True))
                actual.cudagraph = False
                torch.compiler.reset()


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_native_inhibitory_recurrence_preserves_arrivals_and_graph_learning(backend):
    """Keep independent STDP stages correct before exercising fused
    schedules."""
    require_cuda_backend(backend)
    _check_inhibitory_recurrence(
        KernelOptions(
            neuron=backend,
            connection=backend,
            connection_method="pull",
            enable_fp_fusion=False,
            fast_math=False,
        ),
        "neuron" if backend == "triton" else "connection",
    )


@pytest.mark.parametrize(
    ("backend", "recurrent", "selection", "layout", "delayed"),
    (
        ("triton", "helion", "scan", "connection", True),
        ("triton", "helion", "scan", "neuron", True),
        ("tilelang", "helion", "spike_list", "neuron", False),
        ("tilelang", "standard", "spike_list", "neuron", False),
    ),
)
def test_event_owned_inhibitory_stdp_preserves_signed_snapshots(
    backend, recurrent, selection, layout, delayed
):
    """Reuse event ownership for signed learning in both timestep schedules.

    Delayed lazy edge traces and shared list-driven traces cover
    distinct ownership contracts; the ordinary consumer uses next
    buffers instead of cooperative barriers. Each case checks actual
    Graph capture reuse before compiled continuation, not just the final
    raster.
    """
    require_cuda_backend(backend)
    _check_inhibitory_recurrence(
        KernelOptions(
            neuron=backend,
            connection=backend,
            connection_method="push",
            recurrent=recurrent,
            spike_selection=selection,
            enable_fp_fusion=False,
            fast_math=False,
        ),
        layout,
        delayed=delayed,
    )


@pytest.mark.parametrize(
    ("backend", "recurrent", "layout", "signal_domain", "dtype"),
    (
        ("triton", "standard", "neuron", "scalar", torch.float32),
        ("tilelang", "standard", "neuron", "segment", torch.float32),
        ("triton", "helion", "neuron", "scalar", torch.float32),
        ("tilelang", "helion", "neuron", "segment", torch.float64),
        ("triton", "helion", "connection", "post", torch.float16),
        ("tilelang", "helion", "neuron", "segment", torch.bfloat16),
    ),
)
def test_native_modulated_recurrence_keeps_silent_credit_across_graph_and_compile(
    backend,
    recurrent,
    layout,
    signal_domain,
    dtype,
):
    """Keep mixed edge/shared state and fresh teaching signals across chunks.

    Separate execution covers the auto-dense mirror and native pull.
    Fused execution covers shared trace commits after edge readers,
    independent edge eligibility, and low-precision pointwise learning.
    One shared case has more delayed segments than physical neuron
    lanes. All cases refresh teaching signals, reuse captures, and
    restore into a fresh compiled model.
    """
    require_cuda_backend(backend)

    def factory(pair):
        return ModulatedSTDP(
            ModulatedSTDPOptions(
                a_plus=pair.a_plus,
                a_minus=pair.a_minus,
                tau_plus=3.0,
                tau_minus=4.0,
                tau_eligibility=torch.linspace(3.0, 8.0, pair.a_plus.numel()),
                eta=0.01,
                baseline=0.2,
            ),
            modulator=ExternalSignal(signal_domain, time_first=True),
            trace_scope=layout,
            w_min=pair.w_min,
            w_max=pair.w_max,
        )

    options = KernelOptions(
        neuron=backend,
        connection="auto"
        if backend == "triton" and recurrent == "standard"
        else backend,
        connection_method="push"
        if recurrent == "helion"
        else "auto"
        if backend == "triton"
        else "pull",
        recurrent=recurrent,
        enable_fp_fusion=False,
        fast_math=False,
    )
    builder = partial(
        _helion_stdp_multipass_lif_model
        if recurrent == "helion" and dtype == torch.float64
        else _helion_stdp_lif_model,
        rule_factory=factory,
        dtype=dtype,
    )
    expected = builder("torch", device="cpu")
    actual = builder(options, device="cuda")
    expected.chunk_size = actual.chunk_size = 3
    _seed_helion_stdp_state(expected)
    expected.synapse.linear.plasticity.eligibility.copy_(
        torch.linspace(-0.02, 0.03, expected.synapse.linear.nnz)
    )
    actual.load_state_dict(expected.state_dict())
    # Eligibility stays canonical/in place. Separate execution swaps compact
    # snapshots; the fused owner commits them only after the edge-reader fence.
    for name in actual.synapse.linear.plasticity.state_domains:
        value = getattr(actual.synapse.linear.plasticity, name)
        storage = value.new_empty(2 * value.numel())
        view = storage[::2]
        view.copy_(value)
        setattr(actual.synapse.linear.plasticity, name, view)
    actual.cudagraph = True
    inputs = _HELION_STDP_INPUTS.repeat(1, 33)[:, :129].to(dtype)
    n_signal = (
        1
        if signal_domain == "scalar"
        else 129
        if signal_domain == "post"
        else expected.synapse.linear.plasticity.pre_trace.numel()
    )
    rewards = torch.linspace(-1.0, 1.0, len(inputs) * n_signal).reshape(
        len(inputs), n_signal
    )
    if signal_domain == "scalar":
        rewards = rewards[:, 0]
    tolerance = max(5e-4, 4 * torch.finfo(dtype).eps)
    with torch.no_grad(), environ.context(dt=1.0):
        for index, (start, end) in enumerate(((0, 6), (6, len(inputs)))):
            target, target_records = expected(
                inputs[start:end], plasticity_signals={"modulator": rewards[start:end]}
            )
            execute = (
                actual
                if index == 0
                else torch.compile(actual, fullgraph=True, mode="reduce-overhead")
            )
            result, records = execute(
                inputs[start:end].cuda(),
                plasticity_signals={"modulator": rewards[start:end].cuda()},
            )
            torch.testing.assert_close(result.cpu(), target, atol=0, rtol=0)
            _assert_helion_stdp_records(
                records, target_records, atol=tolerance, rtol=tolerance
            )
            for name, value in functional.named_hidden_states(expected).items():
                torch.testing.assert_close(
                    functional.named_hidden_states(actual)[name].cpu(),
                    value,
                    atol=tolerance,
                    rtol=tolerance,
                )
            if index == 0:
                assert actual._cudagraph_runner.n_captures == 1
                checkpoint = io.BytesIO()
                torch.save(actual.state_dict(), checkpoint)
                checkpoint.seek(0)
                actual = builder(options, device="cuda")
                actual.chunk_size = 3
                actual.load_state_dict(torch.load(checkpoint, weights_only=True))
                actual.cudagraph = False
                torch.compiler.reset()
    if layout == "neuron":
        assert (
            actual.synapse.linear.cache.edge_segment.numel()
            == actual.synapse.linear.nnz
        )
    if backend == "triton" and recurrent == "standard":
        assert actual.synapse.linear._plan.route.key.representation == "dense"


@pytest.mark.parametrize(
    ("backend", "dtype"),
    (("triton", torch.float16), ("tilelang", torch.bfloat16)),
)
def test_zero_delay_shared_learning_uses_compact_source_segments(backend, dtype):
    """Keep missing sources and silent eligibility correct without an edge map.

    Sources 1, 2, 4, 5 and 7 have no outgoing edges: source IDs cannot
    stand in for segment IDs. The source pointer already records this
    distinction. Pair learning covers FP16; continuous modulated
    learning covers BF16 with an FP32 teaching signal. Both reuse
    compact snapshots across five steps.
    """
    require_cuda_backend(backend)
    from btorch.models.connection import SparseConnection, Synapse
    from btorch.models.kernels.plasticity import get_plasticity_template, prepare_update

    def connection():
        rule = PairSTDP(
            a_plus=0.1,
            a_minus=0.05,
            tau_plus=3.0,
            tau_minus=4.0,
            trace_scope="neuron",
        )
        if backend == "tilelang":
            rule = ModulatedSTDP(
                ModulatedSTDPOptions(
                    a_plus=0.1,
                    a_minus=0.05,
                    tau_plus=3.0,
                    tau_minus=4.0,
                    tau_eligibility=5.0,
                    eta=0.01,
                ),
                trace_scope="neuron",
            )
        return SparseConnection.from_edges(
            torch.tensor([0, 0, 3, 3, 6]),
            torch.tensor([1, 4, 0, 7, 4]),
            8,
            8,
            Synapse(weight=torch.full((5,), 0.5, dtype=dtype), plasticity=rule),
        )

    expected, actual = connection(), connection().cuda()
    rule = actual.plasticity
    topology = actual.cache.prepare_delay_segments(8, 1)
    domains = tuple(rule.state_domains.items())
    template = get_plasticity_template(rule._kernel_model, domains, rule.signal_domains)
    inputs = torch.tensor(
        [
            [1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=dtype,
    )
    posts = inputs.roll(1, dims=1)
    signals = {"modulator": torch.tensor(0.2)} if backend == "tilelang" else {}
    device_signals = {name: value.cuda() for name, value in signals.items()}
    transition = prepare_update(
        backend,
        rule._kernel_model,
        actual.weight.value,
        {name: getattr(rule, name) for name in template.states},
        inputs[0].cuda(),
        posts[0].cuda(),
        actual.pre,
        actual.post,
        None,
        1,
        {name: getattr(rule, name) for name in template.parameters},
        actual.weight.value.new_tensor(0.5),
        signals=device_signals,
        state_domains=domains,
        signal_domains=rule.signal_domains,
        segments={
            "segment_pre": topology[1],
            "segment_delay": topology[2],
            "source_segment_ptr": topology[0],
        },
    )
    tolerance = 4 * torch.finfo(dtype).eps
    with torch.no_grad():
        for arrival, post in zip(inputs, posts, strict=True):
            expected.update_plasticity(arrival, post, dt=0.5, signals=signals)
            transition(arrival.cuda(), post.cuda(), signals=device_signals)
            torch.testing.assert_close(
                actual.weight.value.cpu(),
                expected.weight.value,
                atol=tolerance,
                rtol=tolerance,
            )
            for name, value in transition.states.items():
                torch.testing.assert_close(
                    value.cpu(),
                    getattr(expected.plasticity, name),
                    atol=tolerance,
                    rtol=tolerance,
                )
        transition.finish()
    # No O(edges) inverse is prepared to read these compact segment traces.
    assert actual.cache.edge_segment.numel() == 0
