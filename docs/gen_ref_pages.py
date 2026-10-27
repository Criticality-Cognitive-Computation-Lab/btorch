"""Generate mkdocstrings API reference pages dynamically from source.

Discovery rules:
- Walk btorch and find all public modules with top-level classes/functions.
- Skip internal backends and type-only modules (EXCLUDES).
- Skip modules deeper than package.module unless explicitly allowed (DEEP_PAGES).
- Auto-group direct submodules into their parent package page.
- Top-level modules (depth 1) become standalone pages.
"""

import sys
from pathlib import Path

import mkdocs_gen_files


sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_reference import DEEP_PAGES, discover  # noqa: E402


all_modules = discover(Path(__file__).resolve().parents[1] / "btorch")
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

# Keep package docs and submodule docs on the same page.
standalone_only = []
for mod in standalone:
    if mod in grouped:
        grouped[mod].insert(0, mod)
    else:
        standalone_only.append(mod)
standalone = standalone_only

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
        if any(mod.startswith(prefix) for mod in all_modules):
            fd.write(f"- [{title}]({Path(path).name})\n")
    fd.write("\n")

nav["API Reference", "Index"] = "api/index.md"

with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
