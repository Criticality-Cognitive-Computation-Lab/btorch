# Contributing

Thank you for your interest in improving btorch! This page covers the basics of getting set up and submitting changes.

## Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
```

2. **Install development dependencies**

```bash
pip install -e ".[dev]"
```

3. **Install pre-commit hooks**

```bash
pre-commit install --install-hooks
```

## Running Checks

Before opening a pull request, run the following checks:

```bash
# Lint
ruff check .

# Tests
pytest tests

# Docs build
python -m sphinx.cmd.build docs docs/_build/html
```

For faster feedback, run targeted subsets:

```bash
pytest tests/models -k neuron
pytest tests/connectome
pytest tests/visualisation
```

## Code Style

- Match existing module structure and naming.
- Use modern Python type annotations (`|`, `list`, `dict`).
- Use `jaxtyping` for tensor shapes where it clarifies intent.
- Keep lines within 88 characters.
- Use Google-style docstrings.
- Write comments and docstrings in English.

## Documentation

- Update `docs/` for any user-facing API changes.
- Update `README.md` for installation or workflow changes.
- New API pages should use `mkdocstrings` (`::: btorch.module.path`) rather than hand-written signatures.

## Pull Request Workflow

1. Create a feature branch from `main`.
2. Make focused, atomic commits.
3. Ensure all checks pass.
4. Open a PR with a clear description of the change and its motivation.

## Questions?

Open a [GitHub Discussion](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/discussions) or check the [FAQ](faq.md).
