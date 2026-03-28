Btorch Documentation
====================

**Btorch** is a brain-inspired Torch library for neuromorphic research, providing stateful neuron models, connectome utilities, and analysis tools.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   analysis
   connection_conversion
   visualisation
   api/index


Overview
--------

Btorch provides:

- **Neuron Models**: LIF, ALIF, GLIF3, Izhikevich neurons with torch.compile compatibility
- **Connectome Tools**: Sparse connectivity matrices, Flywire-compatible data handling
- **Analysis**: Spike train analysis, dynamic metrics, statistical tools
- **Surrogate Gradients**: Custom gradient functions for spiking neural networks

Key Features
------------

- Heterogeneous neuron parameters
- Enhanced shape/dtype checking for stateful modules
- torch.compile and ONNX-compatible
- Gradient checkpointing and truncated BPTT support
- Sparse connectivity matrix support

Installation
------------

Install from source:

.. code-block:: bash

   git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
   cd btorch
   pip install -e . --config-settings editable_mode=strict


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
