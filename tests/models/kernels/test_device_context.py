"""Exercise native cache and launch ownership across CUDA devices."""

import pytest
import torch

from btorch.models import KernelOptions, environ
from btorch.models.connection import SparseConnection
from btorch.models.functional import init_net_state
from btorch.models.neurons.lif import LIF
from btorch.models.rnn import RecurrentNN
from btorch.models.synapse import ExponentialPSC
from tests.models.kernels.helpers import GPU_BACKENDS, require_cuda_backend


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible GPUs")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_recurrent_kernels_follow_inputs_without_changing_current_device(backend):
    require_cuda_backend(backend)

    def network(device, implementation):
        connection = SparseConnection.from_edges(
            torch.tensor([0, 0, 1, 2, 2], device=device),
            torch.tensor([1, 2, 2, 0, 1], device=device),
            3,
            3,
            values=torch.tensor([0.1, -0.03, 0.12, 0.04, 0.08], device=device),
        )
        model = RecurrentNN(
            LIF(3, tau=2.0, v_threshold=0.5, v_reset=0.0, c_m=1.0),
            ExponentialPSC(3, 2.5, connection),
            backend=(
                KernelOptions(
                    neuron=implementation,
                    connection="tilelang" if implementation == "tilelang" else "triton",
                    connection_method="push",
                )
                if implementation != "torch"
                else "torch"
            ),
            update_state_names=("neuron.v", "synapse.psc"),
        ).to(device)
        init_net_state(model, batch_size=2, device=device, dtype=torch.float32)
        return model

    # Compile the same primitive/paired programs on GPU 0 first. Reusing their
    # source on GPU 1 must select a callable for that device, even when device 0
    # remains current. Include the actual source-major push binding, rather
    # than an identity connection that cannot expose a sparse device mismatch.
    # The actual Torch Module schedule supplies independent VJPs.
    with torch.cuda.device(0), environ.context(dt=0.25):
        for index in (0, 1):
            device = torch.device("cuda", index)
            candidate, reference = network(device, backend), network(device, "torch")
            x = torch.linspace(0.4, 1.2, 9 * 2 * 3, device=device).reshape(9, 2, 3)
            x.requires_grad_()
            xr = x.detach().clone().requires_grad_()
            output, records = candidate(x)
            expected, reference_records = reference(xr)
            torch.testing.assert_close(output, expected)
            for name, value in records.items():
                torch.testing.assert_close(value, reference_records[name])
            loss = output.sum() + sum(
                value.square().sum() for value in records.values()
            )
            reference_loss = expected.sum() + sum(
                value.square().sum() for value in reference_records.values()
            )
            gradient = torch.autograd.grad(loss, x)[0]
            reference_gradient = torch.autograd.grad(reference_loss, xr)[0]
            torch.testing.assert_close(
                gradient, reference_gradient, atol=3e-5, rtol=3e-4
            )
            assert output.device == device
            assert torch.cuda.current_device() == 0
