# Changelog

All notable changes to btorch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- **Two-compartment neuron** (`TwoCompartmentGLIF`): soma-apical neuron with nonlinear apical plateau, bidirectional coupling, and optional adaptive threshold. See [tutorial](tutorials/mixed_neurons.md).
- **Mixed neuron population** (`MixedNeuronPopulation`): single recurrent layer mixing multiple neuron types (e.g. GLIF3 + TwoCompartmentGLIF) with automatic current slicing and spike concatenation.
- **Heterogeneous RNN** (`HeteroRecurrentNN`): drop-in replacement for `RecurrentNN` that accepts a `MixedNeuronPopulation`.
- **GeNN/CUDA backend** (`btorch.backend`): experimental spike simulation backend targeting GeNN for GPU-accelerated large-scale networks.

### Changed
- Conda environment file renamed from `dev-requirements.yaml` to `environment.yml`.
