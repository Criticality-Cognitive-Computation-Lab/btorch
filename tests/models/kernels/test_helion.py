"""Compare genuine fused timesteps with the ordinary recurrent schedule."""

from dataclasses import replace

import pytest
import torch

from btorch.models import KernelOptions, environ, functional
from btorch.models.connection import EdgeWeight, PairSTDP, SparseConnection, Synapse
from btorch.models.neurons import GLIF3, Izhikevich
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import AlphaPSC, DualExponentialPSC, HeterSynapsePSC
from tests.models.kernels.helpers import require_cuda_backend


def _network(backend, dtype=torch.float32):
    # 129 neurons leave 127 inactive lanes in the final CTA. Every lane still
    # has to reach the cooperative barrier; putting it inside the neuron mask
    # would deadlock. Signed receptor channels also expose drive-layout errors.
    n = 129
    source = torch.arange(n).repeat(2)
    target = torch.cat(((torch.arange(n) + 1) % n, (torch.arange(n) + 7) % n))
    receptor = source % 2
    connection = SparseConnection.from_edges(
        source,
        target,
        n,
        n,
        Synapse(
            weight=EdgeWeight(torch.where(receptor == 0, 0.03, -0.02), trainable=False),
            receptor=receptor,
            n_receptor=2,
        ),
    )
    model = RecurrentNN(
        Izhikevich(
            n,
            v_threshold=0.7,
            v_reset=0.0,
            v_rest=0.0,
            v_peak=1.0,
            c_m=1.0,
            k=0.1,
            a=torch.linspace(0.025, 0.035, n),
            b=0.1,
            d=0.05,
            hard_reset=True,
        ),
        HeterSynapsePSC(
            n,
            2,
            None,
            connection,
            base_psc=DualExponentialPSC,
            tau_rise=torch.tensor([1.0, 1.5]).repeat(n),
            tau_decay=torch.tensor([4.0, 6.0]).repeat(n),
        ),
        backend=backend,
        update_state_names=(
            "neuron.v",
            "neuron.u",
            "synapse.psc",
            "synapse.base_psc.psc",
            "synapse.base_psc.g_rise",
            "synapse.base_psc.g_decay",
        ),
        chunk_size=200,
    ).to(device="cuda", dtype=dtype)
    functional.init_net_state(model, batch_size=1, device="cuda", dtype=dtype)
    return model


@pytest.mark.parametrize(
    ("backend", "dtype", "execution", "weight_dtype"),
    (
        ("triton", torch.float16, "compile", None),
        ("tilelang", torch.bfloat16, "graph", None),
        ("triton", torch.float16, "compile", torch.float64),
        ("tilelang", torch.bfloat16, "graph", torch.float32),
    ),
)
def test_low_precision_recurrent_physics_keeps_storage_dtype(
    backend, dtype, execution, weight_dtype
):
    """Continue same-dtype physical memories without a hidden FP32 policy.

    Pair the ordinary scheduler with compiled execution and the fused
    static scheduler with CUDA Graph replay. These are focused
    renderer/lifecycle anchors, rather than a Cartesian model, dtype and
    execution-mode matrix.
    """
    require_cuda_backend(backend)
    reference = _network(
        KernelOptions(neuron="torch", connection="torch", connection_method="pull"),
        dtype,
    )
    model = _network(
        KernelOptions(
            neuron=backend,
            recurrent="helion" if execution == "graph" else "standard",
            connection_method="push",
            enable_fp_fusion=False,
            fast_math=False,
        ),
        dtype,
    )
    model.load_state_dict(reference.state_dict())
    if weight_dtype is not None:
        for network in (model, reference):
            weight = network.synapse.linear.weight
            weight.value = weight.value.to(weight_dtype)
    model.cudagraph = execution == "graph"
    if execution == "compile":
        torch._dynamo.reset()
        runner = torch.compile(model, backend="aot_eager", fullgraph=True)
    else:
        runner = model
    inputs = torch.linspace(0.3, 0.9, 129, device="cuda", dtype=dtype).expand(8, 1, 129)
    tolerance = 4 * torch.finfo(dtype).eps
    chunks = (
        (inputs[:1], inputs[1:] * 0.9)
        if execution == "compile" and weight_dtype is not None
        else (inputs, inputs * 0.9)
    )
    with torch.no_grad(), environ.context(dt=1.0):
        for current in chunks:
            expected, expected_records = reference(current)
            actual, records = runner(current)
            # Native sequence APIs return the public raster in input dtype;
            # physical memories and records retain their promoted model dtype.
            torch.testing.assert_close(actual, expected.to(dtype), atol=0, rtol=0)
            torch.testing.assert_close(
                records, expected_records, atol=tolerance, rtol=tolerance
            )
            for name, value in functional.named_hidden_states(model).items():
                wanted = functional.named_hidden_states(reference)[name]
                torch.testing.assert_close(
                    value, wanted, atol=tolerance, rtol=tolerance
                )


@pytest.mark.parametrize(
    ("backend", "dtype"),
    (("triton", torch.float32), ("tilelang", torch.float32), ("triton", torch.float64)),
)
def test_helion_izhikevich_dual_psc_preserves_physics_and_continuation(backend, dtype):
    """Fuse the schedule without changing PSC timing or recorded memories."""
    require_cuda_backend(backend)
    expected = _network(
        KernelOptions(neuron="torch", connection="torch", connection_method="pull"),
        dtype,
    )
    actual = _network(
        KernelOptions(
            neuron=backend,
            recurrent="helion",
            connection_method="push",
            enable_fp_fusion=False,
            fast_math=False,
        ),
        dtype,
    )
    actual.load_state_dict(expected.state_dict())
    # A noncontiguous time sequence exercises storage addressing without a
    # forced contiguous conversion. The changed second chunk must continue
    # both adaptation and all three PSC memories from the first chunk.
    inputs = torch.linspace(0.3, 0.9, 129, device="cuda", dtype=dtype).expand(
        31, 1, 129
    )
    tolerance = 2e-10 if dtype == torch.float64 else 4e-4
    with torch.no_grad(), environ.context(dt=1.0):
        for current in (inputs[:13], inputs[13:] * 1.1):
            spikes, records = expected(current)
            observed, actual_records = actual(current)
            torch.testing.assert_close(observed, spikes, atol=0, rtol=0)
            torch.testing.assert_close(
                actual_records, records, atol=tolerance, rtol=tolerance
            )
            actual_states = functional.named_hidden_states(actual)
            for name, value in functional.named_hidden_states(expected).items():
                torch.testing.assert_close(
                    actual_states[name],
                    value,
                    atol=tolerance,
                    rtol=tolerance,
                )
    assert spikes.count_nonzero() > 0


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_fused_static_spike_list_preserves_neuron_and_psc_history(backend):
    """Use the existing producer in a fused kernel without plasticity state."""
    require_cuda_backend(backend)
    expected = _network(
        KernelOptions(neuron="torch", connection="torch", connection_method="pull")
    )
    actual = _network(
        KernelOptions(
            neuron=backend,
            recurrent="helion",
            connection_method="push",
            spike_selection="spike_list",
            enable_fp_fusion=False,
            fast_math=False,
        )
    )
    actual.load_state_dict(expected.state_dict())
    inputs = torch.linspace(0.3, 0.9, 129, device="cuda").expand(17, 1, 129)
    with torch.no_grad(), environ.context(dt=1.0):
        for chunk in (inputs[:7], inputs[7:] * 1.1):
            spikes, records = expected(chunk)
            observed, observed_records = actual(chunk)
            torch.testing.assert_close(observed, spikes, atol=0, rtol=0)
            torch.testing.assert_close(observed_records, records, atol=4e-4, rtol=4e-4)
    assert spikes.count_nonzero() > 0


def test_helion_graph_chunks_preserve_continuation_and_history_ownership(monkeypatch):
    """Warm bounded chunks and keep earlier records independent of replay."""
    require_cuda_backend("triton")
    options = KernelOptions(
        neuron="triton",
        recurrent="helion",
        connection_method="push",
        enable_fp_fusion=False,
        fast_math=False,
    )
    expected = _network(
        KernelOptions(neuron="torch", connection="torch", connection_method="pull")
    )
    actual = _network(options)
    actual.load_state_dict(expected.state_dict())
    expected.chunk_size = actual.chunk_size = 32
    actual.cudagraph = True
    lengths = []
    original = actual._stacked_chunk_forward

    def observe_chunk(current, **kwargs):
        lengths.append(current.shape[0])
        return original(current, **kwargs)

    monkeypatch.setattr(actual, "_stacked_chunk_forward", observe_chunk)
    inputs = torch.linspace(0.3, 0.9, 129, device="cuda").expand(73, 1, 129)
    with torch.no_grad(), environ.context(dt=1.0):
        first = actual(inputs)
        reference = expected(inputs)
        torch.testing.assert_close(first, reference, atol=4e-4, rtol=4e-4)
        retained = (
            first[0].clone(),
            {name: value.clone() for name, value in first[1].items()},
        )
        captures = actual._cudagraph_runner.n_captures
        assert max(lengths) == 32  # Never warm/capture the full 73-step call.
        assert 1 <= captures <= 2  # Full pieces and, optionally, the remainder shape.
        changed = inputs[:64] * 1.05
        observed = actual(changed)
        reference = expected(changed)
        torch.testing.assert_close(observed, reference, atol=4e-4, rtol=4e-4)
        torch.testing.assert_close(
            functional.named_hidden_states(actual),
            functional.named_hidden_states(expected),
            atol=4e-4,
            rtol=4e-4,
        )
        assert actual._cudagraph_runner.n_captures == captures
        torch.testing.assert_close(first, retained, atol=0, rtol=0)
        # A schedule change must capture its own work while continuing the
        # same physical state, rather than replay the earlier HELION graph.
        actual.backend = replace(options, recurrent="standard")
        observed = actual(changed)
        reference = expected(changed)
        torch.testing.assert_close(observed, reference, atol=4e-4, rtol=4e-4)
        torch.testing.assert_close(
            functional.named_hidden_states(actual),
            functional.named_hidden_states(expected),
            atol=4e-4,
            rtol=4e-4,
        )
        assert actual._cudagraph_runner.n_captures > captures


def test_helion_compile_continues_physical_memories():
    """Keep recurrent memories outside the compiler's reusable output pool.

    Fullgraph compilation covers one recurrent chunk. The second call
    changes the input and continues all physical memories without a
    reset. Public results follow Torch's normal compiled-output lifetime
    rules; callers that retain them across graph iterations clone them
    outside the compiled call.
    """
    require_cuda_backend("triton")
    # Other model/stride compile tests share MemoryModule.forward's code object.
    # Give this independent continuation scenario its own compilation cache.
    torch.compiler.reset()
    options = KernelOptions(
        neuron="triton",
        recurrent="helion",
        connection_method="push",
        enable_fp_fusion=False,
        fast_math=False,
    )
    expected = _network(options)
    actual = _network(options)
    actual.load_state_dict(expected.state_dict())
    execute = torch.compile(actual, fullgraph=True, mode="reduce-overhead")
    inputs = torch.linspace(0.3, 0.9, 129, device="cuda").expand(32, 1, 129)
    with torch.no_grad(), environ.context(dt=1.0):
        for current in (inputs, inputs * 1.05):
            reference = expected(current)
            observed = execute(current)
            torch.testing.assert_close(observed, reference, atol=4e-4, rtol=4e-4)
            torch.testing.assert_close(
                functional.named_hidden_states(actual),
                functional.named_hidden_states(expected),
                atol=4e-4,
                rtol=4e-4,
            )


_GLIF_ALPHA_RECORDS = (
    "neuron.v",
    "neuron.Iasc",
    "synapse.psc",
    "synapse.base_psc.psc",
    "synapse.base_psc.h",
    "synapse.delayed_spike.history",
)


def _delayed_glif_alpha_model(options, *, device="cuda", plasticity=None, records=None):
    """Build the shared heterogeneous GLIF3/AlphaPSC delayed fixture."""
    n = 129
    source = torch.arange(n).repeat(2)
    receptor = torch.arange(2 * n) % 2
    connection = SparseConnection.from_edges(
        source,
        (source + torch.arange(2 * n) % 3 + 1) % n,
        n,
        n,
        Synapse(
            weight=EdgeWeight(torch.where(receptor == 0, 0.02, -0.01), trainable=False),
            delay=torch.arange(2 * n) % 5,
            receptor=receptor,
            n_receptor=2,
            plasticity=plasticity,
        ),
    )
    model = RecurrentNN(
        GLIF3(
            n,
            v_threshold=0.8,
            v_reset=0.0,
            v_rest=0.0,
            c_m=1.0,
            tau=5.0,
            k=[0.1, 0.2],
            asc_amps=[-0.01, -0.02],
            hard_reset=True,
        ),
        HeterSynapsePSC(n, 2, None, connection, base_psc=AlphaPSC, tau_syn=3.0),
        backend=options,
        update_state_names=_GLIF_ALPHA_RECORDS if records is None else records,
        chunk_size=3,
    ).to(device=device, dtype=torch.float32)
    functional.init_net_state(model, batch_size=1, device=device, dtype=torch.float32)
    return model


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_helion_delayed_glif_alpha_rebuilds_pending_arrivals_across_chunks(backend):
    """Preserve arrival-time weights, delay history and physical records.

    Start with pending spikes, run a chunk shorter than the delay
    horizon, then edit canonical weights before continuing. A backend-
    owned pending current carried across calls would incorrectly retain
    the old weights. Comparing the entire delay trajectory also exposes
    off-by-one errors that a final spike count would miss.
    """
    require_cuda_backend(backend)

    expected = _delayed_glif_alpha_model(
        KernelOptions(neuron="torch", connection="torch", connection_method="pull")
    )
    actual = _delayed_glif_alpha_model(
        KernelOptions(
            neuron=backend,
            recurrent="helion",
            connection_method="push",
            enable_fp_fusion=False,
            fast_math=False,
        )
    )
    # Exercise delayed replay as part of the Triton anchor; the TileLang
    # anchor independently validates its renderer without a Cartesian matrix.
    actual.cudagraph = backend == "triton"
    # History is newest first. These initial events must propagate even
    # before the newly simulated neurons have emitted their first spike.
    with torch.no_grad(), environ.context(dt=1.0):
        history = expected.synapse.delayed_spike.history
        history.copy_(
            torch.arange(history.numel(), device="cuda").reshape_as(history) % 3 == 0
        )
        actual.load_state_dict(expected.state_dict())
        actual.synapse.delayed_spike.history.copy_(history)
        inputs = torch.linspace(0.15, 0.45, 129, device="cuda").expand(14, 1, 129)
        for index, current in enumerate((inputs[:2], inputs[2:11], inputs[11:])):
            if index == 1:
                expected.synapse.linear.weight.value.mul_(1.5)
                actual.synapse.linear.weight.value.mul_(1.5)
            reference = expected(current)
            observed = actual(current)
            torch.testing.assert_close(observed[0], reference[0], atol=0, rtol=0)
            torch.testing.assert_close(observed[1], reference[1], atol=4e-4, rtol=4e-4)
            torch.testing.assert_close(
                functional.named_hidden_states(actual),
                functional.named_hidden_states(expected),
                atol=4e-4,
                rtol=4e-4,
            )
        assert reference[0].count_nonzero() > 0


def _glif_alpha_rule(n_edge):
    """Create heterogeneous per-edge PairSTDP parameters for the online
    gate."""
    return PairSTDP(
        a_plus=torch.linspace(0.08, 0.20, n_edge),
        a_minus=torch.linspace(0.07, 0.19, n_edge),
        tau_plus=torch.linspace(4.0, 7.0, n_edge),
        tau_minus=torch.linspace(5.0, 8.0, n_edge),
        w_min=torch.full((n_edge,), -0.05),
        w_max=torch.full((n_edge,), 0.08),
    )


def test_helion_online_delayed_glif_alpha_matches_eager_across_chunks():
    """Match delayed GLIF3/AlphaPSC online STDP across a split continuation."""
    require_cuda_backend("triton")
    records = (
        *_GLIF_ALPHA_RECORDS,
        "synapse.linear.plasticity.pre_trace",
        "synapse.linear.plasticity.post_trace",
        "synapse.linear.weight.value",
    )
    options = KernelOptions(
        neuron="triton",
        recurrent="helion",
        connection_method="push",
        enable_fp_fusion=False,
        fast_math=False,
    )
    expected = _delayed_glif_alpha_model(
        KernelOptions(neuron="torch", connection="torch", connection_method="pull"),
        device="cpu",
        plasticity=_glif_alpha_rule(258),
        records=records,
    )
    actual = _delayed_glif_alpha_model(
        options,
        plasticity=_glif_alpha_rule(258),
        records=records,
    )
    with torch.no_grad():
        history = expected.synapse.delayed_spike.history
        history.copy_(
            (torch.arange(history.numel()).reshape_as(history) % 3 == 0).to(history)
        )
        rule = expected.synapse.linear.plasticity
        rule.pre_trace.copy_(torch.linspace(0.1, 0.6, rule.pre_trace.numel()))
        rule.post_trace.copy_(torch.linspace(0.6, 0.1, rule.post_trace.numel()))
        actual.load_state_dict(expected.state_dict())
        # Delay history is a running memory rather than a persistent buffer
        # in this fixture. Seed it explicitly on both sides of the oracle.
        actual.synapse.delayed_spike.history.copy_(history)

    inputs = torch.linspace(0.15, 0.45, 129).expand(14, 1, 129)
    initial_weights = expected.synapse.linear.weight.value.detach().clone()
    with torch.no_grad(), environ.context(dt=1.0):
        for current in (inputs[:5], inputs[5:] * 1.1):
            expected_spikes, expected_records = expected(current)
            actual_spikes, actual_records = actual(current.cuda())
            torch.testing.assert_close(
                actual_spikes.cpu(), expected_spikes, atol=0, rtol=0
            )
            assert set(actual_records) == set(records)
            for name in records:
                torch.testing.assert_close(
                    actual_records[name].cpu(),
                    expected_records[name],
                    atol=4e-4,
                    rtol=4e-4,
                )
            torch.testing.assert_close(
                {
                    name: value.cpu()
                    for name, value in functional.named_hidden_states(actual).items()
                },
                functional.named_hidden_states(expected),
                atol=4e-4,
                rtol=4e-4,
            )
    assert expected_spikes.count_nonzero() > 0
    assert not torch.equal(expected.synapse.linear.weight.value, initial_weights)
