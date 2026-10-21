"""Kernel-suite fixtures that exercise the registered sequence boundary."""

import pytest

from btorch.models.kernels import register_cell_backend
from btorch.models.kernels.torch_backend import backward_sequence, forward_sequence


@pytest.fixture(scope="session")
def reference_backend():
    """Use the real Torch sequence callbacks as a CPU operator-path oracle.

    A registered backend enters the same recurrent tensor operator as
    GPU backends, while a plain Torch Module uses its normal recurrent
    schedule. Register once on demand, so importing tests does not
    mutate the registry.
    """
    name = "test_torch_sequence_reference"
    register_cell_backend(
        name,
        forward=forward_sequence,
        backward=backward_sequence,
        supports=lambda _spec, device: device.type == "cpu",
    )
    return name
