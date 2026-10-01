"""Use a Triton sparse connection efficiently in a manual time loop.

``RecurrentNN`` and ``make_rnn`` prepare their sparse modules automatically.
Only code that calls ``SparseConn`` directly in a Python time loop needs the
explicit preparation scope demonstrated below.
"""

import numpy as np
import scipy.sparse
import torch

from btorch.models.functional import prepare_sparse_modules
from btorch.models.linear import SparseConn


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This Triton sparse example requires a CUDA device.")

    n_neuron = 1024
    fanout = 16
    timesteps = 100
    batch_size = 1
    rng = np.random.default_rng(42)

    source = np.repeat(np.arange(n_neuron), fanout)
    destination = rng.integers(0, n_neuron, size=source.size)
    weight = rng.normal(scale=0.1, size=source.size).astype(np.float32)
    matrix = scipy.sparse.coo_array(
        (weight, (source, destination)),
        shape=(n_neuron, n_neuron),
    )
    connection = SparseConn(
        matrix,
        enforce_dale=False,
        sparse_backend="triton",
        device="cuda",
        dtype=torch.float32,
    )
    spikes = (
        torch.rand(timesteps, batch_size, n_neuron, device="cuda") < 0.01
    ).float()

    # Avoid calling connection(spikes[t]) in an unprepared loop. Without this
    # scope, a direct SparseConn call cannot know that more timesteps follow and
    # must repack its edge weights on every call.
    with torch.inference_mode(), prepare_sparse_modules(
        connection, batch_size=batch_size
    ):
        currents = torch.stack(
            [connection(spikes[t]) for t in range(timesteps)]
        )

    print(f"spikes: {tuple(spikes.shape)}")
    print(f"synaptic currents: {tuple(currents.shape)}")

    # No explicit scope is needed when the connection belongs to RecurrentNN or
    # a make_rnn wrapper: their multi_step_forward already prepares all sparse
    # modules once around the complete time loop.


if __name__ == "__main__":
    main()
