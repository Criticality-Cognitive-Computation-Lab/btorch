# AGENT.md

Btorch is a brain-inspired Torch library for neuromorphic research. Follow the patterns in the codebase: stateful Modules, careful tensor shape/dtype handling, and practical usability over over-engineering.

## Repo focus

- Primary package: `btorch/` (models, connectome tools, analysis, visualization).
- Docs live in `docs/`; examples in `examples/`.
- Avoid adding new heavy dependencies unless explicitly requested.
- Connectome tooling lives under `btorch/connectome/`; keep data transforms explicit and reversible.
- Neuron models live under `btorch/models/neurons/`; follow existing parameter/state patterns.

## Coding guidelines

- Match existing module structure and naming.
- Use `jaxtyping` for tensor shapes where it clarifies intent.
- Use modern Python type annotations (`|`, `list`, `dict`); avoid
  `typing.Optional`/`typing.Union` unless absolutely required.
- Be explicit about device, dtype, and batch/time dimensions.
- Preserve torch.compile compatibility and ONNX friendliness where applicable.
- Prefer buffers/register_memory for stateful tensors; validate shapes/dtypes.
- Keep code and comments within the formatter line length (88 chars); wrap long docstrings and comments.
- Use English for code, comments, and docstrings; conversation can be in other languages.
- In tests, include thorough explanatory comments so they are understandable and can serve as examples.

## Documentation format

Use **Google style** docstrings (preferred) with this structure:

```python
"""One-line summary (imperative, no period).

Extended description. Explain what, not how. Use math notation for equations:

.. math::
    \tau \frac{dv}{dt} = -(v - v_{reset}) + R \cdot I(t)

Args:
    param1: Description with type inferred from annotation.
    param2: Multi-line descriptions
        should indent.

Returns:
    Description of return value.

Raises:
    ValueError: When input is invalid.

Examples:
    Basic usage:

    >>> result = function_name(arg1, arg2)
    >>> print(result)
    expected_output

Notes:
    Additional implementation details.

References:
    [1] Author, Title, Journal, Year.
"""
```

Key patterns:

- Document `__init__` parameters in class docstring, not `__init__` method
- Include mathematical model equations in `.. math::` blocks
- Include cross-references with `:func:` and `:class:`

## Tests and checks

- Lint: `ruff check .`
- Tests: `pytest tests`
- Docs: `python -m sphinx.cmd.build docs docs/_build/html`

Run the smallest relevant subset when possible; call out if tests were not run.
Recommended targeted runs:

- Neurons: `pytest tests/models -k neuron`
- Connectome: `pytest tests/connectome`
- Visualization: `pytest tests/visualisation`
If unsure, run the full suite.

## Documentation

- Update `docs/` for user-facing changes.
- Update `README.md` for installation or workflow changes.

## Contribution workflow

- Use pre-commit formatting before PRs: `pre-commit install --install-hooks`.
- Keep changes modular; avoid mixing refactors with feature work.

## Design principles

- Usability over performance; simple over clever; customizability over abstraction.
- Keep network models cohesive (single file/folder principle).
