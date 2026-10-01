"""End-to-end pipeline test chaining the btorch layers.

The test doubles as a compact usage example. It walks through the typical
workflow of the library:

1. ``btorch.connectome``: a synthetic connectome table (pandas DataFrame with
   ``pre_simple_id`` / ``post_simple_id`` / ``syn_count``) is turned into a
   scipy sparse weight matrix with :func:`make_sparse_mat`.
2. ``btorch.models``: the matrix becomes a :class:`SparseConn`, which is used
   as the recurrent weight of an :class:`ExponentialPSC` synapse. A
   :class:`LIF` population and the synapse are wrapped in a
   :class:`RecurrentNN`, which unrolls the network over time.
3. ``btorch.analysis``: spike-train metrics (firing rate, ISI CV, Fano
   factor) are computed from the resulting raster.
4. ``btorch.io``: the simulation memories are written to / read from an
   xarray Dataset and must round-trip exactly.

Everything runs on CPU in a few seconds and is fully seeded, so two runs with
the same seed must produce bit-identical rasters.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from btorch.analysis import firing_rate, isi_cv
from btorch.analysis.spiking import fano
from btorch.connectome.connection import make_sparse_mat
from btorch.models import environ
from btorch.models.functional import init_net_state, reset_net
from btorch.models.linear import SparseConn
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC


N_NEURON = 40  # population size
N_EDGE = 300  # number of (pre, post) rows in the synthetic connectome
N_STEP = 200  # simulation length (time steps, dt = 1 ms)
DT = 1.0


def build_connectome(seed: int) -> pd.DataFrame:
    """Create a synthetic connectome table with a seeded RNG.

    Each row is a (pre, post) neuron pair with a synapse count. Duplicated
    pairs are allowed on purpose: real connectomes list one row per
    neuropil, and ``make_sparse_mat`` sums the counts of duplicates.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "pre_simple_id": rng.integers(0, N_NEURON, N_EDGE),
            "post_simple_id": rng.integers(0, N_NEURON, N_EDGE),
            "syn_count": rng.integers(1, 6, N_EDGE),
        }
    )


def build_network(connectome: pd.DataFrame) -> RecurrentNN:
    """Connectome table -> sparse matrix -> LIF + ExponentialPSC RNN."""
    # (1) connectome -> scipy sparse array of shape (pre, post).
    weight = make_sparse_mat(connectome, shape=(N_NEURON, N_NEURON))

    # Scale synapse counts to a small weight and make the last 20% of the
    # neurons inhibitory (negative outgoing weights), as in cortical models.
    weight = weight.tocsr().astype(np.float32) * 0.15
    n_exc = int(N_NEURON * 0.8)
    sign = np.ones(N_NEURON, dtype=np.float32)
    sign[n_exc:] = -3.0  # stronger inhibition balances the excitation
    weight = weight.multiply(sign[:, None]).tocoo()

    # (2) sparse matrix -> SparseConn (fixed topology, optional Dale's law
    # which we disable because signs are already encoded in the weights).
    conn = SparseConn(weight, enforce_dale=False, dtype=torch.float32)

    # (3) neuron + synapse, wrapped into a recurrent network. The neuron sees
    # the synaptic current of the *previous* step plus the external input.
    neuron = LIF(
        n_neuron=N_NEURON,
        v_threshold=1.0,
        v_reset=0.0,
        tau=20.0,
        tau_ref=2.0,
        step_mode="s",
    )
    synapse = ExponentialPSC(N_NEURON, tau_syn=5.0, linear=conn, step_mode="s")
    net = RecurrentNN(
        neuron=neuron,
        synapse=synapse,
        step_mode="m",  # "m": the module consumes a whole [T, ...] sequence
        update_state_names=("neuron.v", "synapse.psc"),
        grad_checkpoint=False,
    )
    init_net_state(net, dtype=torch.float32)
    return net


def simulate(seed: int) -> tuple[torch.Tensor, dict]:
    """Build everything from ``seed`` and return (spike raster, states)."""
    connectome = build_connectome(seed)
    net = build_network(connectome)

    # Seeded external drive of shape [T, N]; mean 0.08 is enough to cross the
    # threshold of 1.0 through leaky integration, noise makes spiking irregular.
    gen = torch.Generator().manual_seed(seed)
    x_ext = 0.08 + 0.05 * torch.randn(N_STEP, N_NEURON, generator=gen)

    reset_net(net)
    with torch.no_grad(), environ.context(dt=DT):
        spikes, states = net(x_ext)
    return spikes, states


def test_pipeline_end_to_end():
    spikes, states = simulate(seed=0)

    # ---- simulation output -------------------------------------------------
    # Raster: one binary value per (time step, neuron).
    assert spikes.shape == (N_STEP, N_NEURON)
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}
    assert spikes.sum() > 0, "network must be active, otherwise metrics are trivial"
    # Recorded states are returned per time step as well.
    assert states["neuron.v"].shape == (N_STEP, N_NEURON)
    assert torch.isfinite(states["neuron.v"]).all()

    # ---- determinism -------------------------------------------------------
    # Rebuilding the connectome and network from the same seed gives the very
    # same raster (all randomness is seeded; the CPU backend is deterministic).
    spikes_again, _ = simulate(seed=0)
    assert torch.equal(spikes, spikes_again)
    # A different seed gives a different connectome and input -> other raster.
    spikes_other, _ = simulate(seed=1)
    assert not torch.equal(spikes, spikes_other)

    # ---- analysis ----------------------------------------------------------
    # firing_rate with batch_axis=1 averages over neurons -> population rate
    # trace of shape [T] in spikes / ms (dt is 1 ms here).
    pop_rate = firing_rate(spikes, width=10, dt=DT, batch_axis=1)
    assert pop_rate.shape == (N_STEP,)
    assert torch.isfinite(pop_rate).all()
    # The mean rate must agree with the raw spike count (no smoothing).
    raw_rate = firing_rate(spikes, width=None, dt=DT, batch_axis=1).mean()
    assert raw_rate.item() == pytest.approx(spikes.mean().item() / DT, rel=1e-5)

    # ISI CV per neuron; neurons with < 3 spikes yield NaN by convention,
    # so only the active ones are checked for finiteness.
    cv, _ = isi_cv(spikes, dt=DT)
    assert cv.shape == (N_NEURON,)
    active = spikes.sum(0) >= 3
    assert active.any()
    assert torch.isfinite(cv[active]).all()
    assert (cv[active] >= 0).all()

    # Fano factor per neuron from spike counts in 20-step windows.
    ff, _ = fano(spikes, window=20)
    assert ff.shape == (N_NEURON,)
    assert torch.isfinite(ff[active]).all()


def test_pipeline_io_round_trip():
    # xarray is an optional dependency of btorch.io.
    pytest.importorskip("xarray")
    from btorch.io.serialization import memories_to_xarray, xarray_to_memories

    spikes, states = simulate(seed=0)
    memories = {"spike": spikes, "v": states["neuron.v"]}

    # dim_counts=(1, 0, 1): (time, batch, neuron) with no batch axis, matching
    # the [T, N] layout of the simulation. Spikes are stored sparsely.
    ds = memories_to_xarray(memories, dim_counts=(1, 0, 1), force_sparse=["spike"])
    assert ds["spike"].attrs.get("_btorch_sparse")  # marker of sparse encoding
    assert ds["v"].shape == (N_STEP, N_NEURON)

    loaded = xarray_to_memories(ds)

    # Exact round trip: spikes come back as a (bool) dense raster and the
    # voltages are unchanged.
    assert loaded["spike"].shape == (N_STEP, N_NEURON)
    np.testing.assert_array_equal(loaded["spike"].astype(np.float32), spikes.numpy())
    np.testing.assert_array_equal(loaded["v"], states["neuron.v"].numpy())

    # Metrics computed on the reloaded data equal those of the original run.
    cv_orig, _ = isi_cv(spikes, dt=DT)
    cv_loaded, _ = isi_cv(torch.as_tensor(loaded["spike"].astype(np.float32)), dt=DT)
    torch.testing.assert_close(cv_orig, cv_loaded, equal_nan=True)
