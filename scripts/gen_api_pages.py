"""Generate static API reference pages to disk for Zensical builds.

This mirrors docs/gen_ref_pages.py but writes files directly to
docs/en/docs/api/ and docs/zh/docs/api/ instead of using mkdocs-gen-
files.
"""

import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "docs"))
from api_reference import DEEP_PAGES, discover  # noqa: E402


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

    all_modules = discover(REPO_ROOT / "btorch")
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
        if any(mod.startswith(prefix) for mod in all_modules):
            lines.append(f"- [{title}]({Path(path).name})\n")
    lines.append("\n")
    index_path.write_text("".join(lines), encoding="utf-8")

    print(f"Generated API pages in {api_dir}")


if __name__ == "__main__":
    generate("en")
    generate("zh")
