"""Generate static API reference pages to disk for Zensical builds.

This mirrors docs/gen_ref_pages.py but writes files directly to
docs/en/docs/api/ and docs/zh/docs/api/ instead of using mkdocs-gen-
files.
"""

import ast
import importlib
import pkgutil
import shutil
from pathlib import Path


EXCLUDES = {
    "btorch.backend",
    "btorch.types",
    "btorch.config",
    "btorch.jit",
}

DEEP_PAGES = {
    "btorch.analysis.dynamic_tools.": (
        "api/analysis_dynamic_tools.md",
        "Analysis — Dynamic Tools",
    ),
    "btorch.models.neurons": (
        "api/neurons.md",
        "Neurons",
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
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if not alias.name.startswith("_"):
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


def _write_page_disk(base_dir: Path, path: str, title: str, modules: list[str]):
    dest = base_dir / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}\n\n"]
    for mod in modules:
        lines.append(f"::: {mod}\n")
        lines.append("    options:\n")
        lines.append("      members: true\n\n")
    dest.write_text("".join(lines), encoding="utf-8")


def generate(lang: str = "en"):
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    api_dir = docs_dir / lang / "docs" / "api"

    if api_dir.exists():
        shutil.rmtree(api_dir)
    api_dir.mkdir(parents=True, exist_ok=True)

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

    # Package-level grouped pages
    for parent, mods in sorted(grouped.items()):
        slug = parent.replace("btorch.", "") + ".md"
        title = parent.replace("btorch.", "").replace(".", " ").title()
        _write_page_disk(api_dir, slug, title, mods)

    # Standalone top-level modules
    for mod in standalone:
        slug = mod.replace("btorch.", "") + ".md"
        title = mod.replace("btorch.", "").title()
        _write_page_disk(api_dir, slug, title, [mod])

    # Deep-prefix pages
    for prefix, (path, title) in DEEP_PAGES.items():
        present = [m for m in all_modules if m.startswith(prefix)]
        if present:
            # path from DEEP_PAGES includes 'api/' prefix; strip it since
            # _write_page_disk writes under api_dir already.
            rel_path = path.removeprefix("api/")
            _write_page_disk(api_dir, rel_path, title, present)

    # Index page
    index_path = api_dir / "index.md"
    lines = ["# API Reference\n\n"]
    lines.append("Auto-generated reference for all public btorch modules.\n\n")
    lines.append("## Module Index\n\n")
    for parent in sorted(grouped):
        slug = parent.replace("btorch.", "") + ".md"
        title = parent.replace("btorch.", "").replace(".", " ").title()
        lines.append(f"- [{title}]({Path(slug).name})\n")
    for mod in standalone:
        slug = mod.replace("btorch.", "") + ".md"
        lines.append(f"- [{mod.replace('btorch.', '').title()}]({Path(slug).name})\n")
    for prefix, (path, title) in DEEP_PAGES.items():
        lines.append(f"- [{title}]({Path(path).name})\n")
    lines.append("\n")
    index_path.write_text("".join(lines), encoding="utf-8")

    print(f"Generated API pages in {api_dir}")


if __name__ == "__main__":
    generate("en")
    generate("zh")
