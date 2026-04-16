---
hide:
  - navigation
  - toc
---

# Btorch

**A brain-inspired Torch library for neuromorphic research.**

![Btorch Overview](assets/images/btorch-overview.png)

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } __Neuron Models__

    ---

    LIF, ALIF, GLIF3, and Izhikevich neurons with
    `torch.compile` compatibility and heterogeneous parameters.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

-   :material-graph-outline:{ .lg .middle } __Connectome Tools__

    ---

    Sparse connectivity matrices, delay expansion, and
    Flywire-compatible data handling.

    [:octicons-arrow-right-24: Connection Conversion](connection_conversion.md)

-   :material-chart-line:{ .lg .middle } __Analysis & Visualisation__

    ---

    Spike train analysis, dynamic metrics, and plotting utilities
    for large-scale simulations.

    [:octicons-arrow-right-24: Analysis](analysis.md)

-   :material-book-open-variant:{ .lg .middle } __Tutorials__

    ---

    Step-by-step guides for building RSNNs, training SNNs,
    and using dataclass-first configuration.

    [:octicons-arrow-right-24: Tutorials](tutorials/training.md)

</div>

## Installation

Install from source:

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
pip install -e . --config-settings editable_mode=strict
```

## Key Features

- **Stateful Modules**: Built-in memory management for spiking neurons
- **Shape Safety**: Enhanced dtype and dimension handling for scala and hetergenous parameters
- **`torch.compile` Ready**: Compatible with PyTorch 2.x compilation
- **Sparse Connectivity**: First-class support for large sparse matrices
- **Truncated BPTT**: Easy gradient truncation for long sequences
