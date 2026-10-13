"""Execute the workflows linked from both language versions of the
tutorials."""

import torch

from examples.kernel_backend_training import run_example as train_example
from examples.oot_cell_backend import run_example as extension_example


def test_sparse_training_tutorial_preserves_the_simulated_trajectory():
    # The published example resets between an inference run and two training
    # chunks with the same input. Their spikes must agree; the example also
    # verifies real encoder/neuron/readout gradients and state-record losses.
    loss, simulated, trained = train_example()
    assert torch.isfinite(loss)
    torch.testing.assert_close(trained, simulated)


def test_external_tutorial_verifies_callbacks_then_trains_a_normal_module():
    # Readers should be able to execute the public registration and parity
    # workflow without pytest as a runtime dependency or an optional GPU package.
    # The example additionally checks an upstream gradient through eval-mode LIF.
    report = extension_example()
    report.assert_close()
