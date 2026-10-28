"""Bind registered scalar mathematics without model-name dispatch."""

from ..templates import get_template


def scalar_layout(spec, modes=0):
    """Expand declared trailing component axes into scalar lane registers."""
    template = get_template(spec)

    def expand(names):
        return tuple(
            scalar
            for name in names
            for scalar in (
                tuple(f"{name}_{i}" for i in range(modes))
                if template.extra_dims(name)
                else (name,)
            )
        )

    return expand(template.state_names), expand(template.parameter_names)


def get_fragments(spec, modes=0, *, prepared_values=(), authoritative_output=False):
    """Resolve one model-local declaration during source preparation."""
    template = get_template(spec)
    if template.fragments is None:
        raise NotImplementedError(
            f"Model {spec.model!r} has no generated scalar definition"
        )
    return template.fragments(
        spec,
        modes,
        prepared_values=prepared_values,
        authoritative_output=authoritative_output,
    )
