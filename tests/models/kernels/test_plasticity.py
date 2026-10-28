"""Parity tests for native per-edge STDP kernels.

The eager connection tests define the mathematical oracle.  This file only
checks the renderer ABI: canonical indices, delayed arrival addressing,
heterogeneous rule tensors, mutable traces, and non-contiguous views.
"""

from __future__ import annotations

import pytest
import torch

from btorch.models import environ
from btorch.models.connection import PairSTDP
from btorch.models.kernels import KernelOptions
from btorch.models.kernels.plasticity import PARAMETERS, update
from tests.models.connection.test_plasticity import (
    N_DELAY,
    _arrivals,
    _connection,
    _posts,
    _recurrent_inputs,
    _recurrent_model,
    _rule,
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


def _run_native_case(backend: str, dtype: torch.dtype, *, strided: bool) -> None:
    """Compare every native step with the eager PairSTDP oracle."""
    cpu_events = _connection(_heterogeneous_rule())
    eager = _connection(_heterogeneous_rule()).to(device="cuda", dtype=dtype)
    native = _connection(_heterogeneous_rule()).to(device="cuda", dtype=dtype)
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
        pre_storage = torch.empty(2 * native.nnz, device="cuda", dtype=dtype)
        post_storage = torch.empty(2 * native.nnz, device="cuda", dtype=dtype)
        native_weight = weight_storage[::2]
        native_pre = pre_storage[::2].zero_()
        native_post = post_storage[::2].zero_()
        native_weight.copy_(native.weight.value)
    else:
        native_weight = native.weight.value.detach().clone()
        native_pre = native.plasticity.pre_trace.detach().clone()
        native_post = native.plasticity.post_trace.detach().clone()

    native_rule = native.plasticity
    parameters = {}
    for name in PARAMETERS:
        value = getattr(native_rule, name)
        if strided and value.numel() > 1:
            storage = torch.empty(2 * value.numel(), device="cuda", dtype=dtype)
            value_view = storage[::2]
            value_view.copy_(value)
            value = value_view
        parameters[name] = value
    dt = torch.tensor(1.0, device="cuda", dtype=dtype)
    for arrivals_t, posts_t in zip(arrivals, posts, strict=True):
        with torch.no_grad():
            eager.update_plasticity(arrivals_t, posts_t, dt=dt)
            update(
                backend,
                native_weight,
                native_pre,
                native_post,
                arrivals_t,
                posts_t,
                native.pre,
                native.post,
                native.delay,
                N_DELAY,
                parameters,
                dt,
            )
        torch.testing.assert_close(
            native_weight,
            eager.weight.value,
            atol=(
                2e-5
                if dtype == torch.float16
                else 2e-12
                if dtype == torch.float64
                else 2e-7
            ),
            rtol=(
                2e-3
                if dtype == torch.float16
                else 2e-12
                if dtype == torch.float64
                else 2e-6
            ),
        )
        torch.testing.assert_close(
            native_pre,
            eager.plasticity.pre_trace,
            atol=(
                2e-5
                if dtype == torch.float16
                else 2e-12
                if dtype == torch.float64
                else 2e-7
            ),
            rtol=(
                2e-3
                if dtype == torch.float16
                else 2e-12
                if dtype == torch.float64
                else 2e-6
            ),
        )
        torch.testing.assert_close(
            native_post,
            eager.plasticity.post_trace,
            atol=(
                2e-5
                if dtype == torch.float16
                else 2e-12
                if dtype == torch.float64
                else 2e-7
            ),
            rtol=(
                2e-3
                if dtype == torch.float16
                else 2e-12
                if dtype == torch.float64
                else 2e-6
            ),
        )


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
    """The FP16 storage path keeps FP32 update accuracy within storage
    error."""
    require_cuda_backend("triton")
    _run_native_case("triton", torch.float16, strided=True)


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_recurrent_native_stdp_matches_torch_with_delay_receptors_and_chunks(backend):
    """Native recurrent STDP preserves outputs, records, and mutable state."""
    require_cuda_backend(backend)
    requested = (
        "neuron.v",
        "synapse.psc",
        "synapse.delayed_spike.history",
        "synapse.linear.plasticity.pre_trace",
        "synapse.linear.plasticity.post_trace",
        "synapse.linear.weight.value",
    )
    reference = _recurrent_model(chunk_size=8).to(device="cuda")
    native = _recurrent_model(chunk_size=8).to(device="cuda")
    reference.update_state_names = requested
    native.update_state_names = requested
    native.backend = KernelOptions(
        neuron=backend,
        connection="triton",
        enable_fp_fusion=False,
        precise_division=True,
        precise_exp=True,
    )
    native.load_state_dict(reference.state_dict())
    inputs = _recurrent_inputs().to(device="cuda")

    with torch.no_grad(), environ.context(dt=1.0):
        reference_first, reference_records_first = reference(inputs[:4])
        reference_second, reference_records_second = reference(inputs[4:])
        native_first, native_records_first = native(inputs[:4])
        native_second, native_records_second = native(inputs[4:])

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
        precise_division=True,
        precise_exp=True,
    )
    inputs = _recurrent_inputs().cuda()
    for mode in ("graph", "compile"):
        expected = _recurrent_model(chunk_size=200).cuda()
        actual = _recurrent_model(chunk_size=200).cuda()
        actual.load_state_dict(expected.state_dict())
        expected.backend = actual.backend = options
        actual.cudagraph = mode == "graph"
        execute = (
            torch.compile(actual, fullgraph=True, mode="reduce-overhead")
            if mode == "compile"
            else actual
        )
        with torch.no_grad(), environ.context(dt=1.0):
            for current in (inputs[:3], inputs[3:] + 0.1):
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


def test_native_online_weights_follow_restored_canonical_slots():
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
        if ".linear." not in name or not value.ndim:
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
        connection="triton",
        connection_method="pull",
        enable_fp_fusion=False,
        precise_division=True,
        precise_exp=True,
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


@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_native_izhikevich_dual_exponential_stdp_matches_eager_per_renderer(backend):
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
        connection="triton",
        enable_fp_fusion=False,
        precise_division=True,
        precise_exp=True,
    )
    reference = _izhikevich_dual_exponential_stdp_model("torch")
    init_net_state(reference, dtype=torch.float32, device="cpu", persistent=True)
    native = _izhikevich_dual_exponential_stdp_model(options).to(
        device="cuda", dtype=torch.float32
    )
    init_net_state(native, dtype=torch.float32, device="cuda", persistent=True)
    native.load_state_dict(reference.state_dict())
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
