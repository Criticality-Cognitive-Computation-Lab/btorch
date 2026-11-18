"""CPU component contracts for heterogeneous conductance and current routing.

The oracle uses an explicit dense matrix and exp(-dt/tau), never another PSC
forward path. Nonzero initial state distinguishes stateful stepping from a
convolution starting at rest. Created 2026-11-18.
"""

import io
import math

import pandas as pd
import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state, named_hidden_states
from btorch.models.synapse import HeterConductancePSC


DT = 0.35
B, R, T = 2, 3, 7
DTYPE = torch.float64
TAU = torch.tensor([1.7, 4.3, 8.1], dtype=DTYPE)
REVERSAL = torch.tensor([0.0, -78.0, -25.0], dtype=DTYPE)


def _metadata(mode):
    # Deliberately scramble table rows: names must use receptor_index rather
    # than their row number or an alphabetical ordering.
    if mode == "connection":
        return pd.DataFrame(
            {"receptor_type": ["slow", "exc", "inh"], "receptor_index": [2, 0, 1]}
        )
    if mode == "neuron":
        return pd.DataFrame(
            {
                "pre_receptor_type": ["M", "E", "I"],
                "post_receptor_type": ["E", "E", "E"],
                "receptor_index": [2, 0, 1],
            }
        )
    return None


def _case(shape=(4,), delays=1):
    n = math.prod(shape)
    generator = torch.Generator(device="cpu").manual_seed(8721)
    return {
        "shape": shape,
        "delays": delays,
        "weight": torch.rand(n * R, n * delays, generator=generator, dtype=DTYPE),
        "g": torch.rand(B, n * R, generator=generator, dtype=DTYPE) + 0.2,
        "history": torch.rand(B, delays, n, generator=generator, dtype=DTYPE),
        "z": torch.rand(T, B, *shape, generator=generator, dtype=DTYPE),
        "v": -85 + 95 * torch.rand(T, B, *shape, generator=generator, dtype=DTYPE),
    }


def _build(
    case,
    *,
    mode="connection",
    circular=False,
    declared=True,
    tau=TAU,
    reversal=REVERSAL,
):
    weight = case["weight"]
    linear = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, device="cpu", dtype=DTYPE
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    options = {}
    if case["delays"] > 1:
        if declared:
            linear.n_delay = case["delays"]
        else:
            # The explicit override also supports delay-unaware dense layers.
            options["max_delay_steps"] = case["delays"]
    synapse = HeterConductancePSC(
        case["shape"],
        R,
        _metadata(mode),
        linear,
        tau_syn=tau,
        reversal_potential=reversal,
        step_mode="m",
        backend="torch",
        use_circular_buffer=circular,
        **options,
    )
    init_net_state(synapse, batch_size=B, device="cpu", dtype=DTYPE, persistent=True)
    synapse.base_psc.psc = case["g"].clone()
    if synapse.delayed_spike is not None:
        # Seed the public delay interface oldest first. The oracle itself does
        # not call Delay or read this implementation's flattened history.
        for d in reversed(range(case["delays"])):
            synapse.delayed_spike.update(case["history"][:, d])
    return synapse


def _slots(value, shape):
    """Expand the documented parameter forms independently of the component."""
    value = torch.as_tensor(value, dtype=DTYPE)
    if value.numel() == math.prod(shape) * R:
        value = value.reshape(*shape, R)
    return torch.broadcast_to(value, (*shape, R)).reshape(-1)


def _oracle(case, *, tau=TAU, reversal=REVERSAL, voltage=None):
    n = math.prod(case["shape"])
    z = case["z"].reshape(T, B, n)
    voltage = case["v"] if voltage is None else voltage.expand_as(case["v"])
    g = case["g"].clone()
    decay = torch.exp(-DT / _slots(tau, case["shape"]))
    e = _slots(reversal, case["shape"]).reshape(n, R)
    currents, states = [], []
    for t in range(T):
        # Column pre*D+d receives z[t-d, pre], with prehistory for t-d<0.
        delayed = torch.stack(
            [
                z[t - d] if t >= d else case["history"][:, d - t - 1]
                for d in range(case["delays"])
            ],
            dim=-1,
        ).reshape(B, n * case["delays"])
        g = g * decay + delayed @ case["weight"].T
        v = voltage[t].reshape(B, n, 1)
        currents.append(
            (g.reshape(B, n, R) * (e - v)).sum(-1).reshape(B, *case["shape"])
        )
        states.append(g.clone())
    return torch.stack(currents), torch.stack(states)


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def _snapshot(synapse):
    return named_hidden_states(synapse, clone=True)


def _assert_snapshot(synapse, before):
    after = _snapshot(synapse)
    assert after.keys() == before.keys()
    for key in before:
        torch.testing.assert_close(
            after[key], before[key], rtol=0, atol=0, msg=lambda msg: f"{key}: {msg}"
        )


@pytest.mark.parametrize("delays", [1, 3])
@pytest.mark.parametrize("circular", [False, True])
def test_single_sequence_and_chunks_match_independent_dense_oracle(delays, circular):
    """Every step uses its own voltage and retains live g across chunk cuts."""
    case = _case(delays=delays)
    whole = _build(case, circular=circular)
    chunked = _build(case, circular=circular)
    single = _build(case, circular=circular)
    expected, states = _oracle(case)
    with torch.no_grad(), environ.context(dt=DT):
        actual = whole(case["z"], v_seq=case["v"])
        # Split before and after the delay depth and insert a one-step chunk.
        pieces = [
            chunked(case["z"][a:b], v_seq=case["v"][a:b])
            for a, b in [(0, 2), (2, 3), (3, T)]
        ]
        for t in range(T):
            _assert_close(
                single.single_step_forward(case["z"][t], case["v"][t]), expected[t]
            )
            _assert_close(single.get_conductance(), states[t])
    _assert_close(actual, expected)
    _assert_close(torch.cat(pieces), expected)
    for synapse in (whole, chunked, single):
        _assert_close(synapse.get_conductance(), states[-1])
        _assert_close(synapse.psc, expected[-1])
        if delays > 1:
            # A double-buffered inner PSC would shift all delayed arrivals.
            assert synapse.base_psc.delayed_spike is None
            for d in range(delays):
                _assert_close(synapse.delayed_spike.get_delay(d), case["z"][-1 - d])


@pytest.mark.parametrize(
    "mode, keys",
    [
        ("connection", ["exc", "inh", "slow"]),
        ("neuron", [("E", "E"), ("I", "E"), ("M", "E")]),
        (None, [0, 1, 2]),
    ],
)
def test_voltage_readout_and_named_channel_access_use_current_not_g(mode, keys):
    """Changing V re-evaluates g(E-V), including inhibitory sign changes."""
    case = _case(shape=(2, 2))
    synapse = _build(case, mode=mode)
    initial = _snapshot(synapse)
    g = case["g"].reshape(B, 2, 2, R)
    for voltage in (
        torch.full((1, 2, 2), -90.0, dtype=DTYPE),
        torch.full((1, 2, 2), -40.0, dtype=DTYPE),
    ):
        current = g * (REVERSAL - voltage.unsqueeze(-1))
        _assert_close(synapse.current_charge(voltage), current.sum(-1))
        _assert_close(synapse.get_current(voltage), current.flatten(1))
        _assert_close(synapse.get_psc(v=voltage), current.flatten(1))
        for r, key in enumerate(keys):
            _assert_close(synapse.get_conductance(key), g[..., r])
            _assert_close(synapse.get_current(voltage, key), current[..., r])
            _assert_close(synapse.get_psc(key, v=voltage), current[..., r])
            # An explicit PSC is already current, so it needs no live voltage.
            _assert_close(synapse.get_psc(key, psc=current.flatten(1)), current[..., r])
        _assert_close(synapse.get_conductance(), case["g"])
    # Read-only queries must neither decay g nor overwrite the last step cache.
    _assert_snapshot(synapse, initial)


@pytest.mark.parametrize("tau_form", ["scalar", "receptor", "flat", "shaped"])
@pytest.mark.parametrize("reversal_form", ["scalar", "receptor", "flat", "shaped"])
def test_multidimensional_neurons_parameter_forms_and_voltage_broadcast(
    tau_form,
    reversal_form,
):
    """Parameter layouts expand by receptor; singleton time/batch V broadcasts."""
    case = _case(shape=(2, 2))

    def parameter(form, base):
        if form == "scalar":
            return base[0]
        if form == "receptor":
            return base
        # Distinct neuron and receptor values expose accidental tiled ordering.
        shaped = base + torch.arange(4, dtype=DTYPE).reshape(2, 2, 1) * 0.3
        return shaped.flatten() if form == "flat" else shaped

    tau = parameter(tau_form, TAU)
    reversal = parameter(reversal_form, REVERSAL)
    synapse = _build(case, tau=tau, reversal=reversal)
    v = torch.tensor([[[[-65.0, -42.0], [-81.0, -20.0]]]], dtype=DTYPE)
    expected, states = _oracle(case, tau=tau, reversal=reversal, voltage=v)
    with torch.no_grad(), environ.context(dt=DT):
        actual = synapse(case["z"], v_seq=v)
    assert actual.shape == (T, B, 2, 2)
    _assert_close(actual, expected)
    _assert_close(synapse.get_conductance(), states[-1])


@pytest.mark.parametrize("circular", [False, True])
def test_explicit_delay_override_has_delivery_step_semantics(circular):
    """An isolated impulse arrives at t=d, with pre*D+d column ordering."""
    case = _case(delays=3)
    case["weight"].zero_()
    case["g"].zero_()
    case["history"].zero_()
    case["z"].zero_()
    # Three receptor slots at post 1 receive source 2 with delays 0, 1, 2.
    for d in range(3):
        case["weight"][1 * R + d, 2 * 3 + d] = d + 1
    case["z"][0, :, 2] = 1
    synapse = _build(case, circular=circular, declared=False)
    expected, states = _oracle(case)
    with torch.no_grad(), environ.context(dt=DT):
        _assert_close(synapse(case["z"], v_seq=case["v"]), expected)
    # Check the arrivals themselves, not merely equivalence of two APIs.
    for r in range(R):
        assert torch.count_nonzero(states[:r, :, 1 * R + r]) == 0
        _assert_close(states[r, :, 1 * R + r], torch.full((B,), r + 1, dtype=DTYPE))


@pytest.mark.parametrize("circular", [False, True])
def test_persistent_state_dict_roundtrip_resumes_g_delay_and_current(circular):
    """Checkpoint at a nonzero ring cursor, then resume against the dense oracle."""
    case = _case(delays=3)
    source = _build(case, circular=circular)
    restored = _build(case, circular=circular)
    expected, states = _oracle(case)
    with torch.no_grad(), environ.context(dt=DT):
        source(case["z"][:2], v_seq=case["v"][:2])
        # Use actual serialization rather than an aliasing state_dict snapshot.
        stream = io.BytesIO()
        torch.save(source.state_dict(), stream)
        stream.seek(0)
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        _assert_snapshot(restored, _snapshot(source))
        _assert_close(restored(case["z"][2:], v_seq=case["v"][2:]), expected[2:])
        _assert_close(source(case["z"][2:], v_seq=case["v"][2:]), expected[2:])
    _assert_close(restored.get_conductance(), states[-1])
    _assert_snapshot(restored, _snapshot(source))


@pytest.mark.parametrize("delays", [1, 3])
@pytest.mark.parametrize("time_length", [0, 1])
def test_empty_sequence_returns_neuron_shape_without_advancing_state(
    delays, time_length
):
    """Both documented voltage time lengths are legal for a zero-step chunk."""
    case = _case(shape=(2, 2), delays=delays)
    synapse = _build(case, circular=True)
    before = _snapshot(synapse)
    voltage = torch.full((time_length, 1, 2, 2), -65.0, dtype=DTYPE)
    with environ.context(dt=DT):
        actual = synapse(case["z"][:0], v_seq=voltage)
    assert actual.shape == (0, B, 2, 2)
    assert actual.dtype == DTYPE
    assert actual.device.type == "cpu"
    _assert_snapshot(synapse, before)


@pytest.mark.parametrize("delays", [1, 3])
@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
def test_wrong_source_batch_is_rejected_before_state_changes(
    delays, circular, sequence
):
    """A broadcastable voltage cannot make an incompatible source batch valid."""
    case = _case(delays=delays)
    synapse = _build(case, circular=circular)
    # State was initialized for B lanes; a different batch must be rejected
    # before either decay or ring writes, even with a valid neuron/source axis.
    bad_source = torch.ones(T, B + 1, 4, dtype=DTYPE)
    voltage = torch.full((1, 1, 4), -60.0, dtype=DTYPE)
    with torch.no_grad(), environ.context(dt=DT):
        synapse.single_step_forward(case["z"][0], case["v"][0])
        before = _snapshot(synapse)
        with pytest.raises((ValueError, RuntimeError)):
            if sequence:
                synapse(bad_source, v_seq=voltage)
            else:
                synapse.single_step_forward(bad_source[0], voltage[0])
    _assert_snapshot(synapse, before)


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize(
    "bad",
    [
        "single_missing_v",
        "single_none_v",
        "single_bad_v",
        "single_expanding_v",
        "single_bad_z",
        "sequence_missing_v",
        "sequence_scalar_v",
        "sequence_bad_time",
        "sequence_bad_v",
        "sequence_expanding_v",
        "sequence_bad_z",
        "empty_bad_time",
        "get_psc_missing_v",
        "get_current_none_v",
        "current_missing_v",
        "delayed_receptor_axis",
    ],
)
def test_invalid_inputs_raise_before_conductance_or_delay_mutation(bad, circular):
    """Rejected calls leave g, total-current cache, history and cursor intact."""
    case = _case(delays=3)
    synapse = _build(case, circular=circular)
    z, v = case["z"], case["v"]
    calls = {
        "single_missing_v": lambda: synapse.single_step_forward(z[0]),
        "single_none_v": lambda: synapse.single_step_forward(z[0], None),
        "single_bad_v": lambda: synapse.single_step_forward(z[0], v[0, :, :3]),
        "single_expanding_v": lambda: synapse.single_step_forward(
            z[0], v[0].unsqueeze(0)
        ),
        "single_bad_z": lambda: synapse.single_step_forward(z[0, :, :3], v[0]),
        "sequence_missing_v": lambda: synapse(z),
        "sequence_scalar_v": lambda: synapse(z, v_seq=torch.tensor(-65.0)),
        "sequence_bad_time": lambda: synapse(z, v_seq=v[:2]),
        "sequence_bad_v": lambda: synapse(z, v_seq=v[..., :3]),
        "sequence_expanding_v": lambda: synapse(z, v_seq=v.unsqueeze(1)),
        "sequence_bad_z": lambda: synapse(z[..., :3], v_seq=v),
        "empty_bad_time": lambda: synapse(z[:0], v_seq=v[:2]),
        "get_psc_missing_v": lambda: synapse.get_psc(),
        "get_current_none_v": lambda: synapse.get_current(None),
        "current_missing_v": lambda: synapse.current_charge(),
        "delayed_receptor_axis": lambda: synapse.single_step_forward(
            z[0].unsqueeze(-1).expand(B, 4, R), v[0]
        ),
    }
    # First make the cache and cursor nontrivial, so failures cannot hide a
    # reset or a partial advance behind a freshly initialized zero state.
    with torch.no_grad(), environ.context(dt=DT):
        synapse.single_step_forward(z[0], v[0])
        before = _snapshot(synapse)
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            calls[bad]()
    _assert_snapshot(synapse, before)


@pytest.mark.parametrize("delays", [1, 3])
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("invalid", ["dtype", "device"])
def test_source_dtype_device_preflight_preserves_all_memory(delays, sequence, invalid):
    """Reject incompatible dense sources before decay or circular-history writes."""
    case = _case(delays=delays)
    synapse = _build(case, circular=True)
    with torch.no_grad(), environ.context(dt=DT), torch.autocast("cpu", enabled=False):
        synapse.single_step_forward(case["z"][0], case["v"][0])
        before = _snapshot(synapse)
        # Meta exercises the device check without allocating CUDA tensors. A
        # float32 source conflicts with this fixture's float64 dense weights.
        source = case["z"].to(
            device="meta" if invalid == "device" else "cpu",
            dtype=DTYPE if invalid == "device" else torch.float32,
        )
        with pytest.raises((ValueError, RuntimeError), match=invalid):
            if sequence:
                synapse(source, v_seq=case["v"])
            else:
                synapse.single_step_forward(source[0], case["v"][0])
    _assert_snapshot(synapse, before)


@pytest.mark.parametrize("time_length", [0, 1])
@pytest.mark.parametrize("voltage_dtype", [torch.float32, torch.float64])
def test_empty_current_promotes_dtype_and_keeps_zero_operand_gradients(
    time_length,
    voltage_dtype,
):
    """Empty current retains g/E/z/V dependencies and ordinary COBA promotion."""
    case = _case(shape=(2, 2), delays=3)
    synapse = _build(case, circular=True).to(dtype=torch.float32)
    g = synapse.base_psc.psc.detach().requires_grad_()
    synapse.base_psc.psc = g
    e = synapse.base_psc.E.requires_grad_()
    source = torch.empty(0, B, 2, 2, dtype=torch.float32, requires_grad=True)
    voltage = torch.full(
        (time_length, 1, 2, 2),
        -65.0,
        dtype=voltage_dtype,
        requires_grad=True,
    )
    before = _snapshot(synapse)
    with environ.context(dt=DT):
        actual = synapse(source, v_seq=voltage)
    # Independent g(E-V) arithmetic establishes the dtype even for T=0.
    empty_v = voltage.expand(0, B, 2, 2).unsqueeze(-1)
    expected = (g.reshape(B, 2, 2, R) * (e.reshape(2, 2, R) - empty_v)).sum(-1)
    assert actual.shape == (0, B, 2, 2)
    assert actual.dtype == expected.dtype
    _assert_close(actual, expected)
    # Do not allow unused operands: differentiating an empty current should
    # return zero tensors, including the singleton-time broadcast voltage.
    gradients = torch.autograd.grad(actual.sum(), (g, e, source, voltage))
    for gradient, operand in zip(gradients, (g, e, source, voltage), strict=True):
        torch.testing.assert_close(gradient, torch.zeros_like(operand), rtol=0, atol=0)
    _assert_snapshot(synapse, before)


@pytest.mark.parametrize("time_length", [0, 1])
def test_empty_current_backward_when_only_conductance_requires_grad(time_length):
    """An empty chunk's loss remains differentiable with only live g trainable."""
    case = _case(delays=3)
    synapse = _build(case, circular=True)
    g = synapse.base_psc.psc.detach().requires_grad_()
    synapse.base_psc.psc = g
    before = _snapshot(synapse)
    source = case["z"][:0]
    voltage = torch.full((time_length, 1, 4), -65.0, dtype=DTYPE)
    assert not source.requires_grad and not voltage.requires_grad
    assert not synapse.base_psc.E.requires_grad
    with environ.context(dt=DT):
        output = synapse(source, v_seq=voltage)
    assert output.requires_grad
    output.sum().backward()
    torch.testing.assert_close(g.grad, torch.zeros_like(g), rtol=0, atol=0)
    _assert_snapshot(synapse, before)


@pytest.mark.parametrize("origin", ["conductance", "voltage"])
@pytest.mark.parametrize("receptor", [None, "inh", 1])
def test_live_get_psc_nan_validation_can_be_explicitly_disabled(origin, receptor):
    """Live current queries honor default/explicit validation and its opt-out."""
    case = _case()
    synapse = _build(case)
    voltage = case["v"][0].clone()
    if origin == "conductance":
        # Flat slot 1 is the inhibitory receptor of neuron 0.
        synapse.base_psc.psc[0, 1] = torch.nan
    else:
        voltage[0, 0] = torch.nan
    initial_g = synapse.get_conductance().clone()
    with pytest.raises(ValueError, match="NaN"):
        synapse.get_psc(receptor, v=voltage)
    with pytest.raises(ValueError, match="NaN"):
        synapse.get_psc(receptor, v=voltage, validate_nan=True)
    actual = synapse.get_psc(receptor, v=voltage, validate_nan=False)
    assert torch.isnan(actual).any()
    torch.testing.assert_close(synapse.get_conductance(), initial_g, equal_nan=True)
