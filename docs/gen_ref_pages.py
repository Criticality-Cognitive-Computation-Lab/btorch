"""Generate mkdocstrings API reference pages dynamically from source.

Discovery rules:
- Walk btorch and find all public modules with top-level classes/functions.
- Skip internal backends and type-only modules (EXCLUDES).
- Skip modules deeper than package.module unless explicitly allowed (DEEP_PAGES).
- Auto-group direct submodules into their parent package page.
- Top-level modules (depth 1) become standalone pages.
"""

import ast
import importlib
import pkgutil
from pathlib import Path

import mkdocs_gen_files


EXCLUDES = {
    "btorch.backend",
    "btorch.types",
    "btorch.config",
    "btorch.jit",
}

# Deep prefixes that get their own dedicated page instead of being grouped.
DEEP_PAGES = {
    "btorch.analysis.dynamic_tools.": (
        "api/analysis_dynamic_tools.md",
        "Analysis — Dynamic Tools",
    ),
}


def _is_public(name: str) -> bool:
    return all(not part.startswith("_") for part in name.split("."))


def _has_public_api(file_path: Path) -> bool:
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                return True
    return False


def _discover(package: str) -> list[str]:
    pkg = importlib.import_module(package)
    base_path = Path(pkg.__file__).parent
    modules: list[str] = []
    for _, name, ispkg in pkgutil.walk_packages([str(base_path)], prefix=f"{package}."):
        if not _is_public(name) or name in EXCLUDES:
            continue
        depth = len(name.split(".")) - 1
        if depth >= 3 and not any(name.startswith(p) for p in DEEP_PAGES):
            continue
        rel = name.split(".")[1:]
        file_path = (
            base_path / "/".join(rel) / "__init__.py"
            if ispkg
            else base_path / ("/".join(rel) + ".py")
        )
        if file_path.exists() and _has_public_api(file_path):
            modules.append(name)
    return sorted(set(modules))


all_modules = _discover("btorch")
grouped: dict[str, list[str]] = {}
standalone: list[str] = []

for mod in all_modules:
    if any(mod.startswith(p) for p in DEEP_PAGES):
        continue
    parts = mod.split(".")
    if len(parts) == 2:
        standalone.append(mod)
    else:
        parent = ".".join(parts[:2])
        grouped.setdefault(parent, []).append(mod)

nav = mkdocs_gen_files.Nav()


def _write_page(path: str, title: str, modules: list[str]) -> None:
    with mkdocs_gen_files.open(path, "w") as fd:
        fd.write(f"# {title}\n\n")
        for mod in modules:
            fd.write(f"::: {mod}\n")
            fd.write("    options:\n      members: true\n\n")
    nav["API Reference", title] = path


# Package-level grouped pages
for parent, mods in sorted(grouped.items()):
    slug = parent.replace("btorch.", "") + ".md"
    title = parent.replace("btorch.", "").replace(".", " ").title()
    _write_page(f"api/{slug}", title, mods)

# Standalone top-level modules
for mod in standalone:
    slug = mod.replace("btorch.", "") + ".md"
    title = mod.replace("btorch.", "").title()
    _write_page(f"api/{slug}", title, [mod])

# Deep-prefix pages
for prefix, (path, title) in DEEP_PAGES.items():
    present = [m for m in all_modules if m.startswith(prefix)]
    if present:
        _write_page(path, title, present)

# Index page
with mkdocs_gen_files.open("api/index.md", "w") as fd:
    fd.write("# API Reference\n\n")
    fd.write("Auto-generated reference for all public btorch modules.\n\n")
    fd.write("## Module Index\n\n")
    for parent in sorted(grouped):
        slug = parent.replace("btorch.", "") + ".md"
        title = parent.replace("btorch.", "").replace(".", " ").title()
        fd.write(f"- [{title}]({Path(slug).name})\n")
    for mod in standalone:
        slug = mod.replace("btorch.", "") + ".md"
        fd.write(f"- [{mod.replace('btorch.', '').title()}]({Path(slug).name})\n")
    for prefix, (path, title) in DEEP_PAGES.items():
        fd.write(f"- [{title}]({Path(path).name})\n")
    fd.write("\n")

nav["API Reference", "Index"] = "api/index.md"

with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
