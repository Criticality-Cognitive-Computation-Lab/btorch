"""Discover public API modules from the checkout without importing btorch."""

import ast
from pathlib import Path


EXCLUDES = {"btorch.backend", "btorch.types", "btorch.config"}
DEEP_PAGES = {
    "btorch.analysis.dynamic_tools.": (
        "api/analysis_dynamic_tools.md",
        "Analysis — Dynamic Tools",
    ),
    "btorch.models.neurons": ("api/neurons.md", "Neurons"),
}


def has_public_api(path: Path) -> bool:
    """Find public definitions or reexports without loading dependencies."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                return True
        elif isinstance(node, ast.ImportFrom):
            if any(not alias.name.startswith("_") for alias in node.names):
                return True
    return False


def discover(package_dir: Path) -> list[str]:
    """List current source modules, independent of any installed checkout."""
    modules = []
    for path in sorted(package_dir.rglob("*.py")):
        rel = path.relative_to(package_dir.parent)
        parts = (
            rel.parts[:-1] if path.name == "__init__.py" else rel.with_suffix("").parts
        )
        name = ".".join(parts)
        if len(parts) < 2 or any(part.startswith("_") for part in parts):
            continue
        if name in EXCLUDES:
            continue
        if len(parts) > 3 and not any(name.startswith(p) for p in DEEP_PAGES):
            continue
        if has_public_api(path):
            modules.append(name)
    return sorted(set(modules))
