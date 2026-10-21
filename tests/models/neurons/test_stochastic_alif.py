"""State and gradient contracts for ALIF/ELIF with Poisson firing."""

from contextlib import nullcontext

import pytest
import torch

from btorch.models import environ
from btorch.models.functional import init_net_state
from btorch.models.neurons.alif import ALIF, ELIF
from btorch.models.surrogate import PoissonRandomSpike


DT = 0.5
SEED = 7391
DEVICES = (
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA runtime is unavailable"
        ),
    ),
)


def _device_index(device: torch.device) -> int:
    assert device.type == "cuda"
    return torch.cuda.current_device() if device.index is None else device.index


def _fork_rng(device: torch.device):
    devices = [] if device.type == "cpu" else [_device_index(device)]
    return torch.random.fork_rng(devices=devices)


def _manual_seed(device: torch.device, seed: int) -> None:
    if device.type == "cpu":
        torch.random.default_generator.manual_seed(seed)
    else:
        torch.cuda.default_generators[_device_index(device)].manual_seed(seed)


def _get_rng_state(device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return torch.get_rng_state().clone()
    return torch.cuda.get_rng_state(_device_index(device)).clone()


def _set_rng_state(device: torch.device, state: torch.Tensor) -> None:
    if device.type == "cpu":
        torch.set_rng_state(state)
    else:
        torch.cuda.set_rng_state(state, _device_index(device))


def _make_neuron(
    neuron_cls: type[ALIF],
    *,
    device: torch.device | str = "cpu",
    step_mode: str = "m",
    detach_reset: bool = False,
    damping_factor: float = 0.5,
    refractory: bool = True,
    spiking: bool = True,
) -> ALIF:
    """Build a finite, genuinely stochastic ALIF-family example."""
    kwargs = {
        "n_neuron": 4,
        "v_threshold": 1.0,
        "v_reset": 0.0,
        "c_m": 1.0,
        "g_leak": 0.2,
        "E_leak": 0.0,
        "E_k": -1.0,
        "g_k_init": 0.05,
        "tau_adapt": 8.0,
        "dg_k": 0.12,
        "tau_ref": 2.0 if refractory else None,
        "surrogate_function": PoissonRandomSpike(
            spiking=spiking,
            tau=2.5,
            rho=0.7,
            leak=0.2,
            k=1.1,
            damping_factor=damping_factor,
        ),
        "detach_reset": detach_reset,
        "hard_reset": True,
        "pre_spike_v": True,
        "step_mode": step_mode,
        "device": device,
        "dtype": torch.float32,
    }
    if neuron_cls is ELIF:
        # Keep the exponential upswing active but well below overflow. The
        # stochastic firing rule, rather than a saturated voltage, decides each
        # event in this validation case.
        kwargs.update(delta_T=0.5, v_T=0.8)
    neuron = neuron_cls(**kwargs)
    init_net_state(neuron, batch_size=2, device=device, dtype=torch.float32)
    return neuron


def _batched_drive(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return a time-major drive with distinct batch lanes and neurons."""
    base = torch.tensor(
        [[0.05, 0.20, 0.50, 0.90], [0.80, 0.40, 0.15, 0.65]],
        device=device,
        dtype=torch.float32,
    )
    return torch.stack([base + 0.08 * ((step % 4) - 1.5) for step in range(13)])


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
def test_poisson_multistep_chunk_continuation_preserves_state_and_rng(
    neuron_cls, device_name
):
    """Chunking preserves stochastic trajectories, model state, and RNG.

    Restoring the global RNG before the second execution makes the full
    call and the two chunks consume the same logical random draws. The
    chunked model is not reset between chunks, so this also exercises
    continuation of voltage, adaptation, pre-spike voltage, and
    refractory state.
    """
    device = torch.device(device_name)
    drive = _batched_drive(device)
    whole = _make_neuron(neuron_cls, device=device)
    chunked = _make_neuron(neuron_cls, device=device)

    # fork_rng makes this test deterministic without perturbing the RNG stream
    # used by tests that execute after it.
    with _fork_rng(device), torch.no_grad(), environ.context(dt=DT):
        _manual_seed(device, SEED)
        initial_rng = _get_rng_state(device)
        whole_spikes = whole(drive)
        whole_rng = _get_rng_state(device)

        _set_rng_state(device, initial_rng)
        split = 5
        chunked_spikes = torch.cat((chunked(drive[:split]), chunked(drive[split:])))
        chunked_rng = _get_rng_state(device)

    assert whole_spikes.shape == drive.shape
    assert torch.equal(whole_spikes, chunked_spikes)
    # A saturated all-zero or all-one case would not validate random event
    # placement or stochastic reset/adaptation continuation.
    assert whole_spikes.eq(0).any()
    assert whole_spikes.eq(1).any()

    for name in ("v", "g_k", "v_pre_spike", "refractory"):
        torch.testing.assert_close(getattr(whole, name), getattr(chunked, name))
        assert torch.isfinite(getattr(whole, name)).all()
    assert torch.equal(whole_rng, chunked_rng)


def _two_forward_vjp(
    neuron_cls: type[ALIF],
    *,
    device: torch.device,
    detach_reset: bool,
    damping_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    """Run two stateful forwards before one VJP under an isolated RNG
    stream."""
    neuron = _make_neuron(
        neuron_cls,
        device=device,
        step_mode="s",
        detach_reset=detach_reset,
        damping_factor=damping_factor,
        refractory=False,
    )
    drive = _batched_drive(device)[:2].detach().requires_grad_()
    with _fork_rng(device), environ.context(dt=DT):
        _manual_seed(device, 421)
        spikes = torch.stack((neuron(drive[0]), neuron(drive[1])))
        # Final-state cotangents exercise both the voltage reset and the
        # spike-triggered adaptation branch without adding a direct spike loss.
        loss = 0.7 * neuron.v.sum() + 1.3 * neuron.g_k.sum()
        (drive_grad,) = torch.autograd.grad(loss, drive)
        final_rng = _get_rng_state(device)
    final = (neuron.v.detach().clone(), neuron.g_k.detach().clone())
    return spikes.detach(), drive_grad, final, final_rng


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
def test_poisson_two_forward_vjp_respects_detach_reset(neuron_cls, device_name):
    """Each stochastic forward owns its VJP context and detach_reset cuts
    reset.

    Poisson damping changes only the surrogate derivative, not sampled
    spikes. Therefore final-state gradients are damping-independent when
    reset events are detached, but damping-dependent when reset and
    adaptation retain their spike derivative. Replaying the same two-
    forward computation also checks that the first stochastic autograd
    context is not overwritten by the second.
    """
    device = torch.device(device_name)
    detached_low = _two_forward_vjp(
        neuron_cls, device=device, detach_reset=True, damping_factor=0.2
    )
    detached_high = _two_forward_vjp(
        neuron_cls, device=device, detach_reset=True, damping_factor=0.8
    )
    attached_low = _two_forward_vjp(
        neuron_cls, device=device, detach_reset=False, damping_factor=0.2
    )
    attached_high = _two_forward_vjp(
        neuron_cls, device=device, detach_reset=False, damping_factor=0.8
    )
    replay = _two_forward_vjp(
        neuron_cls, device=device, detach_reset=False, damping_factor=0.2
    )

    # Controlled RNG gives identical binary events and final forward state for
    # every damping/detach choice; only the reset derivative is under test.
    for result in (detached_high, attached_low, attached_high, replay):
        assert torch.equal(result[0], detached_low[0])
        for actual, expected in zip(result[2], detached_low[2]):
            torch.testing.assert_close(actual, expected)
        assert torch.equal(result[3], detached_low[3])
    assert detached_low[0].eq(0).any()
    assert detached_low[0].eq(1).any()

    torch.testing.assert_close(detached_low[1], detached_high[1])
    assert torch.isfinite(detached_low[1]).all()
    assert torch.isfinite(attached_low[1]).all()
    assert attached_low[1][0].abs().sum() > 0
    assert attached_low[1][1].abs().sum() > 0
    assert not torch.allclose(attached_low[1], attached_high[1])

    # Exact replay covers two independent Poisson autograd contexts followed by
    # one backward, the pattern that would expose shared/singleton saved state.
    torch.testing.assert_close(attached_low[1], replay[1])


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
def test_poisson_modes_preserve_stochastic_output_and_rng(neuron_cls, device_name):
    """Module mode and autograd recording do not change stochastic firing."""
    device = torch.device(device_name)
    drive = _batched_drive(device)
    results = []
    cases = (
        (True, nullcontext()),
        (False, nullcontext()),
        (True, torch.no_grad()),
        (False, torch.inference_mode()),
    )

    with _fork_rng(device), environ.context(dt=DT):
        for training, grad_context in cases:
            neuron = _make_neuron(neuron_cls, device=device)
            neuron.train(training)
            _manual_seed(device, SEED)
            case_drive = drive.detach().clone().requires_grad_()
            with grad_context:
                spikes = neuron(case_drive)
                records_grad = spikes.requires_grad
                state = tuple(
                    getattr(neuron, name).detach().clone()
                    for name in ("v", "g_k", "v_pre_spike", "refractory")
                )
            results.append(
                (spikes.detach().clone(), state, _get_rng_state(device), records_grad)
            )

    expected_spikes, expected_state, expected_rng, _ = results[0]
    assert expected_spikes.eq(0).any() and expected_spikes.eq(1).any()
    assert [records_grad for *_, records_grad in results] == [True, True, False, False]
    for spikes, state, rng, _ in results[1:]:
        assert torch.equal(spikes, expected_spikes)
        for value, expected in zip(state, expected_state):
            torch.testing.assert_close(value, expected)
        assert torch.equal(rng, expected_rng)


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
def test_poisson_nonspiking_primitive_does_not_advance_rng(neuron_cls, device_name):
    """The deterministic primitive remains differentiable and consumes no
    RNG."""
    device = torch.device(device_name)
    neuron = _make_neuron(
        neuron_cls,
        device=device,
        refractory=False,
        spiking=False,
    )
    drive = _batched_drive(device)[:3].detach().requires_grad_()

    with _fork_rng(device), environ.context(dt=DT):
        _manual_seed(device, 117)
        initial_rng = _get_rng_state(device)
        output = neuron(drive)
        final_rng = _get_rng_state(device)
        (gradient,) = torch.autograd.grad(
            output.square().sum() + neuron.v.sum() + neuron.g_k.sum(), drive
        )

    assert torch.equal(initial_rng, final_rng)
    assert ((output != 0) & (output != 1)).any()
    assert torch.isfinite(output).all()
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
@pytest.mark.parametrize("neuron_cls", [ALIF, ELIF])
def test_poisson_cuda_graph_replay_preserves_state_and_rng(neuron_cls):
    """CUDA Graph replay matches eager stochastic state continuation exactly.

    CUDA Graphs replay operations against the addresses seen during
    capture. The explicit input and memory buffers below therefore
    remain alive and at fixed addresses while their values change
    between invocations. Each eager comparison starts from the replay's
    exact CUDA generator state; matching the final state proves both
    paths consumed the same random-number schedule.
    """
    device = torch.device("cuda")
    captured_neuron = _make_neuron(neuron_cls, device=device)
    eager_neuron = _make_neuron(neuron_cls, device=device)
    state_names = ("v", "g_k", "v_pre_spike", "refractory")
    initial_state = tuple(
        getattr(captured_neuron, name).detach().clone() for name in state_names
    )
    external_state = tuple(value.clone() for value in initial_state)
    external_addresses = tuple(value.data_ptr() for value in external_state)
    static_drive = _batched_drive(device)[:4].clone()

    base_drive = _batched_drive(device)[:4]
    trials = (
        base_drive * 0.73 + 0.04,
        base_drive.flip(0) * 1.11 - 0.03,
        base_drive.roll(1, dims=0) * 0.61 + 0.17,
    )

    with _fork_rng(device), torch.no_grad(), environ.context(dt=DT):
        _manual_seed(device, 1907)

        # Warm the same Module on a side stream so capture does not need to
        # initialize kernels or allocator paths. Its state is rebound to the
        # stable external buffers immediately afterward.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                captured_neuron(static_drive)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        for name, buffer, value in zip(
            state_names, external_state, initial_state, strict=True
        ):
            buffer.copy_(value)
            setattr(captured_neuron, name, buffer)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_spikes = captured_neuron(static_drive)

        # Start replay validation from a fresh, known generator position,
        # independent of RNG bookkeeping during warmup and capture.
        _manual_seed(device, 8123)
        captured_carried = tuple(value.clone() for value in initial_state)
        eager_carried = tuple(value.clone() for value in initial_state)
        replay_outputs = []

        for drive in trials:
            static_drive.copy_(drive)
            for buffer, value in zip(external_state, captured_carried, strict=True):
                buffer.copy_(value)

            replay_initial_rng = _get_rng_state(device)
            graph.replay()
            replay_final_rng = _get_rng_state(device)
            replay_spikes = captured_spikes.clone()
            replay_state = tuple(
                getattr(captured_neuron, name).clone() for name in state_names
            )

            _set_rng_state(device, replay_initial_rng)
            for name, value in zip(state_names, eager_carried, strict=True):
                setattr(eager_neuron, name, value.clone())
            eager_spikes = eager_neuron(drive)
            eager_final_rng = _get_rng_state(device)
            eager_state = tuple(
                getattr(eager_neuron, name).clone() for name in state_names
            )

            assert not torch.equal(replay_initial_rng, replay_final_rng)
            assert torch.equal(replay_final_rng, eager_final_rng)
            assert torch.equal(replay_spikes, eager_spikes)
            for replay_value, eager_value in zip(
                replay_state, eager_state, strict=True
            ):
                torch.testing.assert_close(
                    replay_value, eager_value, rtol=0.0, atol=0.0
                )

            replay_outputs.append(replay_spikes)
            captured_carried = replay_state
            eager_carried = eager_state

        assert tuple(value.data_ptr() for value in external_state) == external_addresses
        replay_outputs = torch.cat(replay_outputs)
        assert replay_outputs.eq(0).any()
        assert replay_outputs.eq(1).any()
