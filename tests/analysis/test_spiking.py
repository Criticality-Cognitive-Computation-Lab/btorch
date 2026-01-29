import numpy as np
import torch

from btorch.analysis.spiking import (
    compute_raster,
    cv_from_spikes,
    fano_factor_from_spikes,
    firing_rate,
    kurtosis_from_spikes,
)


def test_cv_from_spikes_handles_sparse_activity():
    spike_data = np.array(
        [
            [1, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
        ]
    )

    cv_values, isi_total, isi_stats = cv_from_spikes(spike_data, dt_ms=1.0)

    assert np.isnan(cv_values[1])
    np.testing.assert_allclose(cv_values[0], 0.0)
    np.testing.assert_allclose(isi_total["cv"], 0.0)
    assert isi_stats[0]["n_spikes"] == 2


def test_fano_and_kurtosis_from_spikes_basic():
    spike = np.array([[[1]], [[0]], [[1]], [[0]]])
    fano = fano_factor_from_spikes(spike, window=2, overlap=0)
    np.testing.assert_array_equal(fano.shape, (1, 1))
    np.testing.assert_allclose(fano, 0.0)

    kurt = kurtosis_from_spikes(spike, window=2, overlap=0, fisher=True)
    np.testing.assert_allclose(kurt, -3.0)


def test_firing_rate_numpy_and_torch():
    spikes = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    fr_mean = firing_rate(spikes, width=3, dt=1.0, axis=-1)
    assert fr_mean.shape == (3,)

    fr_per_neuron = firing_rate(spikes, width=3, dt=1.0, axis=None)
    assert fr_per_neuron.shape == spikes.shape

    torch_spikes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    torch_fr = firing_rate(torch_spikes, width=3, dt=0.5, axis=None)
    assert torch_fr.shape == torch_spikes.shape
    assert torch_fr.dtype == torch_spikes.dtype


def test_raster_plot_returns_indices_and_times():
    sp_matrix = np.array([[1, 0], [0, 1], [1, 1]])
    times = np.array([0.0, 1.0, 2.0])
    neuron_idx, spike_times = compute_raster(sp_matrix, times)
    np.testing.assert_array_equal(neuron_idx, np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(spike_times, np.array([0.0, 1.0, 2.0, 2.0]))
