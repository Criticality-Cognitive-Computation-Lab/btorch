# Agent Documentation Guidelines

## Goal
Migrate btorch from MkDocs to Sphinx + MyST-NB for professional documentation with auto-generated API reference.

## Stack
- **Sphinx** (core)
- **sphinx-rtd-theme** (theme)
- **myst-nb** (Markdown + Jupyter support)
- **sphinx-autodoc-typehints** (type annotation extraction)
- **sphinx-gallery** (executable examples)
- **napoleon** (Google/NumPy docstring support)

## Docstring Style

Use **Google style** (preferred) or **NumPy style**. Follow this structure:

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

## Priority Modules (add docstrings in this order)

1. **Neurons**: btorch/models/neurons/ - LIF, GLIF, ALIF, Izhikevich
2. **Synapses**: btorch/models/synapse.py
3. **Base**: btorch/models/base.py - BaseNode, BaseLayer
4. **Analysis**: btorch/analysis/spiking.py, statistics.py, metrics.py
5. **Connectome**: btorch/connectome/connection.py, augment.py
6. **Visualization**: btorch/visualisation/
7. **Utils**: btorch/utils/

## Key Patterns

### Module docstrings
```python
"""Short description.

Extended description with module purpose.

Key Classes:
    ClassName: Brief description.

Functions:
    function_name: Brief description.

Example:
    >>> from btorch.models.neurons import LIF
    >>> neuron = LIF(n_neuron=100)
"""
```

### Class docstrings
- Document `__init__` parameters in class docstring, not `__init__` method
- Include mathematical model equations in `.. math::` blocks
- List attributes with their tensor shapes
- Provide runnable examples with expected output

### Function docstrings  
- Use jaxtyping annotations for tensor shapes: `Float[Tensor, "*batch n_neuron"]`
- Document batch/time dimension conventions
- Include cross-references with `:func:` and `:class:`

## File Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main entry
├── api/                 # Auto-generated API (use sphinx-apidoc)
│   ├── index.rst
│   ├── models.rst
│   ├── analysis.rst
│   └── ...
├── user_guide/          # Narrative docs (MyST markdown)
├── tutorials/           # Jupyter notebooks (.md with myst-nb frontmatter)
└── examples/            # Sphinx-gallery examples (plot_*.py)
```

## Build Commands

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build HTML
cd docs && make html

# Live reload
cd docs && sphinx-autobuild . _build/html

# Check links
cd docs && make linkcheck
```

## References
- BrainState docs: https://brainstate.readthedocs.io/
- Google style guide: https://google.github.io/styleguide/pyguide.html
- Napoleon docs: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
