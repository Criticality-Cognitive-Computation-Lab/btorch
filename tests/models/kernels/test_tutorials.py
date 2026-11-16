"""Execute the workflows linked from both language versions of the
tutorials."""

import pytest
import torch

from examples.kernel_backend_training import run_example as train_example


def test_sparse_training_tutorial_preserves_the_simulated_trajectory():
    # The published example resets between an inference run and two training
    # chunks with the same input. Their spikes must agree; the example also
    # verifies real encoder/neuron/readout gradients and state-record losses.
    loss, simulated, trained = train_example()
    assert torch.isfinite(loss)
    torch.testing.assert_close(trained, simulated)


def test_external_tutorial_empty_sequence_preserves_state_and_its_derivative():
    # An empty mixed-precision sequence does not advance or promote physical
    # state. This boundary case needs no GPU: the adapter must not launch its
    # Triton kernel, and the final voltage retains an identity derivative.
    pytest.importorskip("triton")
    from btorch.models.kernels import NeuronSpec
    from examples.oot_cell_backend import forward

    initial = torch.linspace(0.1, 0.6, 12, dtype=torch.float16).reshape(2, 6)[:, ::2]
    initial.requires_grad_()
    drive = torch.empty(0, 2, 3, dtype=torch.float64)
    parameters = tuple(
        torch.tensor(value, dtype=torch.float64) for value in (1.0, 0.0, 1.0, 4.0)
    )
    output, (final,), histories = forward(
        drive, (initial,), parameters, NeuronSpec("lif"), ("v", "v_charged")
    )
    torch.testing.assert_close(final, initial)
    (gradient,) = torch.autograd.grad(final.sum(), initial)
    torch.testing.assert_close(gradient, torch.ones_like(initial))
    assert output.shape == drive.shape and all(
        value.shape == drive.shape for value in histories
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_external_tutorial_verifies_native_kernel_then_trains_a_normal_module():
    # This tutorial implements an actual Triton forward rather than aliasing
    # a built-in callback. Its numerical/gradient check and ordinary Module
    # usage should both work on the device the kernel is written for.
    pytest.importorskip("triton")
    from examples.oot_cell_backend import run_example as extension_example

    report = extension_example()
    report.assert_close()
