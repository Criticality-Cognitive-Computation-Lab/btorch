"""Executable contracts for the derived fused conductance inference class.

CPU checks protect unsupported model rejection and the unchanged inherited
reference. GPU checks must run within an allocated Slurm GPU on cccl2.
"""

import copy

import numpy as np
import pytest
import torch

from btorch.models import KernelOptions, environ
from btorch.models.connection import SparseConnection, Synapse
from btorch.models.functional import init_net_state
from btorch.models.neurons import LIF, Izhikevich
from btorch.models.rnn import ConductanceRecurrentNN, FusedConductanceRecurrentNN
from btorch.models.synapse import HeterConductancePSC


def model(device="cpu", n=64):
    # Use unique destinations per source/receptor to make the small reference
    # deterministic, yet exercise all receptor slots and both firing/silence.
    pre = torch.arange(n)
    post = (pre + 1) % n
    receptor = pre % 3
    connection = SparseConnection.from_edges(
        pre,
        post,
        n,
        n,
        Synapse(weight=torch.full((n,), 0.0625), receptor=receptor, n_receptor=3),
    )
    neuron = Izhikevich(
        n,
        c_m=100.0,
        k=0.7,
        v_rest=-60.0,
        v_threshold=-40.0,
        v_peak=25.0,
        v_reset=-55.0,
        a=0.03,
        b=-2.0,
        d=30.0,
        hard_reset=True,
    )
    synapse = HeterConductancePSC(
        n,
        3,
        None,
        connection,
        tau_syn=[5.0, 7.0, 11.0],
        reversal_potential=[0.0, -70.0, -70.0],
    )
    net = ConductanceRecurrentNN(
        neuron,
        synapse,
        backend=KernelOptions(
            connection="triton" if device == "cuda" else "torch",
            connection_method="push" if device == "cuda" else "pull",
        ),
        update_state_names=(),
        cudagraph=False,
    ).to(device)
    init_net_state(net, batch_size=1, device=device, dtype=torch.float32)
    net.neuron.v.copy_(net.neuron.v_rest)
    net.neuron.u.zero_()
    # A nonzero initial state exposes accidental consumption of capture steps.
    net.synapse.base_psc.psc.fill_(0.03125)
    return net


def test_reference_inheritance_keeps_cpu_schedule():
    torch.set_num_threads(1)
    with torch.no_grad(), environ.context(dt=0.2):
        reference = model()
        native = FusedConductanceRecurrentNN.from_reference(copy.deepcopy(reference))
        for current in [0.0, 500.0, 0.0]:
            x = torch.full((1, 64), current)
            torch.testing.assert_close(
                native.single_step_forward(x)[0],
                reference.single_step_forward(x)[0],
                rtol=0,
                atol=0,
            )
            for a, b in [
                (native.neuron.v, reference.neuron.v),
                (native.neuron.u, reference.neuron.u),
                (native.synapse.base_psc.psc, reference.synapse.base_psc.psc),
            ]:
                torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_incompatible_neuron_rejected_without_mutation():
    reference = model()
    before = reference.synapse.base_psc.psc.clone()
    with pytest.raises(TypeError, match="Izhikevich"):
        FusedConductanceRecurrentNN(LIF(64), reference.synapse)
    torch.testing.assert_close(before, reference.synapse.base_psc.psc, rtol=0, atol=0)


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


GPU = pytest.mark.usefixtures("cuda_device")


@GPU
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("n", [64, 131])
def test_fused_matches_reference_and_chunk_boundaries(graph, n):
    torch.set_num_threads(1)
    with torch.no_grad(), environ.context(dt=0.2):
        reference = model("cuda", n=n)
        native = FusedConductanceRecurrentNN.from_reference(
            copy.deepcopy(reference), cudagraph=graph
        )
        probes = torch.arange(n, device="cuda")
        groups = (probes % 3).to(torch.int32)
        previous = None
        # Graph setup must restore a nonzero g and current V/u. Repeated chunks
        # use new inputs; zero chunks test clearing of lists/accumulation.
        for amplitude in [500.0, 0.0, 700.0, 0.0]:
            x = torch.full((12, 1, n), amplitude, device="cuda")
            before_v = native.neuron.v.clone()
            result = native.run_chunk(
                x, hold_steps=5, probe_ids=probes, group_index=groups, current_stride=5
            )
            assert "algorithm = push" in result["metadata"]["route"]
            assert "backend = triton" in result["metadata"]["route"]
            assert result["metadata"]["cudagraph"] == graph
            if previous is not None:
                torch.testing.assert_close(
                    previous[0], previous[1], atol=3e-5, rtol=2e-6
                )
            torch.testing.assert_close(
                result["pre_v_mv"][0], before_v[0], atol=0, rtol=0
            )
            expected_spikes = []
            voltage = []
            conductance = []
            currents = []
            for t in range(60):
                v = reference.neuron.v.clone()
                g = reference.synapse.base_psc.psc.reshape(1, n, 3).clone()
                cur = g * (
                    torch.tensor([0.0, -70.0, -70.0], device="cuda") - v.unsqueeze(-1)
                )
                z = reference.single_step_forward(x[t // 5])[0]
                expected_spikes.append(z.clone())
                voltage.append(v[0])
                conductance.append(g[0])
                currents.append(cur[0])
            torch.testing.assert_close(
                result["spikes"], torch.stack(expected_spikes), atol=0, rtol=0
            )
            torch.testing.assert_close(
                result["pre_v_mv"], torch.stack(voltage), atol=3e-5, rtol=2e-6
            )
            torch.testing.assert_close(
                result["pre_g_ns"], torch.stack(conductance), atol=1e-6, rtol=2e-6
            )
            torch.testing.assert_close(
                result["pre_current_pa"], torch.stack(currents), atol=2e-5, rtol=3e-6
            )
            torch.testing.assert_close(
                native.neuron.v, reference.neuron.v, atol=3e-5, rtol=2e-6
            )
            torch.testing.assert_close(
                native.neuron.u, reference.neuron.u, atol=3e-5, rtol=2e-6
            )
            torch.testing.assert_close(
                native.synapse.base_psc.psc,
                reference.synapse.base_psc.psc,
                atol=1e-6,
                rtol=2e-6,
            )
            torch.testing.assert_close(
                native.synapse.psc, reference.synapse.psc, atol=2e-5, rtol=3e-6
            )
            torch.testing.assert_close(
                result["epsc_pa"],
                torch.stack(currents)[::5, :, 0],
                atol=2e-5,
                rtol=3e-6,
            )
            torch.testing.assert_close(
                result["ipsc_pa"],
                torch.stack(currents)[::5, :, 1:].sum(-1),
                atol=2e-5,
                rtol=3e-6,
            )
            packed = np.packbits(
                result["spikes"][:, 0].cpu().numpy().astype(np.uint8),
                axis=1,
                bitorder="little",
            )
            assert np.array_equal(packed, result["packed_spikes"].cpu().numpy())
            for group in range(3):
                torch.testing.assert_close(
                    result["group_spikes"][:, group],
                    result["spikes"][:, 0, groups == group].sum(-1).int(),
                    atol=0,
                    rtol=0,
                )
            archive = result["pre_v_mv"].clone()
            previous = (archive, torch.stack(voltage).clone())


@GPU
def test_weight_change_refreshes_binding_and_bad_state_is_reported():
    with torch.no_grad(), environ.context(dt=0.2):
        native = FusedConductanceRecurrentNN.from_reference(model("cuda"))
        x = torch.zeros((10, 1, 64), device="cuda")
        probes = torch.arange(64, device="cuda")
        native.run_chunk(x, probe_ids=probes)
        native.synapse.linear.weight.value.mul_(0)
        b = native.run_chunk(x, probe_ids=probes)
        assert b["metadata"]["prepared_calls"] == 1
        native.neuron.v[0, 0] = float("nan")
        with pytest.raises(FloatingPointError):
            native.run_chunk(x, probe_ids=probes)


@GPU
def test_standard_forward_without_optional_recording_and_reset():
    # The ordinary RNN interface must select the fused CUDA path too, and
    # support no-probe/no-group output without attempting writes to empty data.
    with torch.no_grad(), environ.context(dt=0.2):
        reference = model("cuda")
        native = FusedConductanceRecurrentNN.from_reference(copy.deepcopy(reference))
        x = torch.full((30, 1, 64), 500.0, device="cuda")
        expected = torch.stack([reference.single_step_forward(v)[0].clone() for v in x])
        actual, states = native.multi_step_forward(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert states == {}
        native.clear_fused_cache()
        functional_state = native.synapse.base_psc.psc.clone()
        native.clear_fused_cache()
        torch.testing.assert_close(
            native.synapse.base_psc.psc, functional_state, rtol=0, atol=0
        )


@GPU
def test_layout_eviction_keeps_the_physical_trajectory():
    # Evicting a derived graph/recording layout must not reset any physical
    # state. Different chunk lengths represent one continuous trajectory.
    with torch.no_grad(), environ.context(dt=0.2):
        reference = model("cuda")
        native = FusedConductanceRecurrentNN.from_reference(copy.deepcopy(reference))
        for length in [1, 2, 3, 4, 5, 1]:
            x = torch.full((length, 1, 64), 500.0, device="cuda")
            expected = torch.stack(
                [reference.single_step_forward(v)[0].clone() for v in x]
            )
            actual = native.run_chunk(x, cudagraph=False)["spikes"]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(
                native.neuron.v, reference.neuron.v, atol=3e-5, rtol=2e-6
            )
            assert len(native._fused_chunks) <= 4


@GPU
@pytest.mark.parametrize("graph", [False, True])
def test_fine_step_current_sums_reset_on_every_chunk(graph):
    # Retained fine-step histories independently expose a missed reset or an
    # accumulation at the coarse input/recording rate instead of every cell step.
    with torch.no_grad(), environ.context(dt=0.2):
        native = FusedConductanceRecurrentNN.from_reference(
            model("cuda", n=131), cudagraph=graph
        )
        probes = torch.arange(131, device="cuda")
        for amplitude in [500.0, 0.0, 700.0]:
            result = native.run_chunk(
                torch.full((15, 1, 131), amplitude, device="cuda"),
                hold_steps=5,
                probe_ids=probes,
                current_stride=5,
            )
            expected = result["pre_current_pa"].double().sum(0)
            assert result["cell_current_sum_pa"].dtype == torch.float64
            torch.testing.assert_close(
                result["cell_current_sum_pa"], expected, rtol=2e-13, atol=1e-10
            )
