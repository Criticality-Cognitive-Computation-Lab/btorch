"""Namespace scalar stage statements structurally during kernel preparation.

Only scalar assignments and elementary expressions are represented. This is
not an equation inference system: registered definitions own their mathematics,
reverse ordering and residual contracts.
"""

import ast
from functools import lru_cache


_FUNCTIONS = frozenset(("exp", "expm1", "where", "maximum", "minimum", "abs", "fma"))


@lru_cache(maxsize=2048)
def namespace_statements(statements: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    """Bind scalar register identities without rewriting expression strings."""

    class Namespace(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in _FUNCTIONS:
                return node
            return ast.copy_location(ast.Name(prefix + node.id, node.ctx), node)

    return tuple(
        ast.unparse(ast.fix_missing_locations(Namespace().visit(ast.parse(line))))
        for line in statements
    )


def substitute_expression(expression: str, bindings: dict[str, str]) -> str:
    """Bind named scalar operands in a storage index expression."""

    class Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id not in bindings:
                return node
            return ast.copy_location(
                ast.parse(bindings[node.id], mode="eval").body, node
            )

    return ast.unparse(
        ast.fix_missing_locations(Bind().visit(ast.parse(expression, mode="eval")))
    )
