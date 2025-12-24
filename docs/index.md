# Btorch Documentation (Placeholder)

Welcome! Documentation is under construction. Planned sections include:

- Quickstart installation and minimal examples
- API reference for models, utils, and connectome helpers
- Tutorials and benchmarks

For now, see `README.md` for usage notes. Contributions to the docs are welcome.

## Shape Conventions

- Most stateful models accept inputs shaped as `(*batch, *neuron)`.
- `n_neuron` is stored as a tuple of neuron axis sizes (ints normalize to tuples).
- Use `.size` for the total neuron count when needed.
- Use `init_net_state(..., batch_size=(...))` for multi-dim batch setups.
