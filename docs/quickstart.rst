Quickstart
==========

Installation
------------

As ``btorch`` is not yet on PyPI, install from source:

.. code-block:: bash

   git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
   cd btorch
   pip install -e . --config-settings editable_mode=strict


Basic Neuron Usage
------------------

Create and run a simple LIF neuron:

.. code-block:: python

   import torch
   from btorch.models.neurons import LIF

   # Create a single LIF neuron
   neuron = LIF(n_neuron=100, tau=20.0, v_threshold=1.0)

   # Run for 100 timesteps
   spikes = []
   for t in range(100):
       # Random input current
       x = torch.randn(100) * 0.5
       spike = neuron(x)
       spikes.append(spike)

   # Stack spikes into a tensor [time, neurons]
   spike_train = torch.stack(spikes)


Using Heterogeneous Parameters
------------------------------

Neurons support per-neuron parameters:

.. code-block:: python

   import torch
   from btorch.models.neurons import LIF

   # Each neuron has its own time constant
   taus = torch.rand(100) * 30 + 10  # 10-40 ms range

   neuron = LIF(n_neuron=100, tau=taus, v_threshold=1.0)


Basic Analysis Usage
--------------------

Analyze spike trains:

.. code-block:: python

   import numpy as np
   from btorch.analysis import isi_cv, fano, firing_rate

   # Generate sample spike data [time, batch, neurons]
   spike_data = np.random.rand(1000, 10, 50) > 0.95

   # Coefficient of variation of ISIs
   cv, isi_total, isi_stats = isi_cv(spike_data, dt_ms=1.0)

   # Fano factor (variance/mean of spike counts)
   fano_values, fano_stats = fano(spike_data, window_ms=100, dt_ms=1.0)

   # Firing rate via convolution
   rates = firing_rate(spike_data, dt_ms=1.0, smooth_ms=50)


Shape Conventions
-----------------

- Input shape: ``(*batch, n_neuron)`` where ``*batch`` can be any number of dimensions
- ``n_neuron`` is stored as a tuple; use ``.size`` for total neuron count
- Use ``init_net_state(..., batch_size=(...))`` for multi-dimensional batch setups
