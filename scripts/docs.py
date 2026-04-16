"""Build orchestration for btorch multilingual docs.

Builds each language into a unified `site/` directory for GitHub Pages:
- English at `site/`
- Other languages under `site/<lang>/`
"""

from __future__ import annotations

import multiprocessing
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import typer
import yaml


app = typer.Typer(help="Btorch docs build orchestrator")

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
SITE_DIR = Path(__file__).resolve().parent.parent / "site"
LANGUAGES_FILE = DOCS_DIR / "language_names.yml"


def _discover_languages() -> list[str]:
    """Return sorted list of language codes with mkdocs.yml."""
    langs = []
    for lang_dir in sorted(DOCS_DIR.iterdir()):
        if lang_dir.is_dir() and (lang_dir / "mkdocs.yml").exists():
            langs.append(lang_dir.name)
    return langs


def _sync_shared(language: str) -> None:
    """Copy shared assets and stylesheets from English docs to target
    language."""
    if language == "en":
        return
    en_docs = DOCS_DIR / "en" / "docs"
    lang_docs = DOCS_DIR / language / "docs"
    for name in ("assets", "stylesheets"):
        src = en_docs / name
        dest = lang_docs / name
        if not src.exists():
            continue
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        typer.echo(f"Synced {name}: {src} -> {dest}")


def _get_en_nav_paths() -> list[str]:
    """Extract relative doc paths from English mkdocs.yml nav."""
    en_yml = DOCS_DIR / "en" / "mkdocs.yml"
    data = yaml.safe_load(en_yml.read_text(encoding="utf-8"))
    nav = data.get("nav", [])
    paths: list[str] = []

    def _walk(items):
        for item in items:
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        paths.append(v)
                    elif isinstance(v, list):
                        _walk(v)

    _walk(nav)
    return paths


@app.command()
def build_lang(
    language: str = typer.Argument(..., help="Language code to build"),
) -> None:
    """Build a single language into the unified site directory."""
    config_path = DOCS_DIR / language / "mkdocs.yml"
    if not config_path.exists():
        typer.echo(f"Config not found: {config_path}", err=True)
        raise typer.Exit(1)

    if language == "en":
        dest = SITE_DIR
        default_output = DOCS_DIR / "en" / "site"
    else:
        dest = SITE_DIR / language
        default_output = DOCS_DIR / language / "site"

    if dest.exists():
        shutil.rmtree(dest)
    if default_output.exists():
        shutil.rmtree(default_output)

    # Sync shared assets/stylesheets into the target language docs
    _sync_shared(language)

    # Generate API pages to disk before building
    api_script = Path(__file__).resolve().parent / "gen_api_pages.py"
    typer.echo("Generating API reference pages...")
    subprocess.run(["python", str(api_script)], check=True)

    cmd = [
        "zensical",
        "build",
        "--config-file",
        str(config_path),
    ]
    typer.echo(f"Building {language} -> {dest}")
    subprocess.run(cmd, check=True)

    # Zensical builds into a 'site' folder next to the config file.
    # Move it to the unified site directory.
    if default_output.exists():
        shutil.move(str(default_output), str(dest))


@app.command()
def build_all() -> None:
    """Build all languages in parallel."""
    langs = _discover_languages()
    if not langs:
        typer.echo("No language configs found.", err=True)
        raise typer.Exit(1)

    # Ensure English builds first so root index exists, then parallel rest
    if "en" in langs:
        build_lang("en")
        rest = [lang for lang in langs if lang != "en"]
    else:
        rest = langs

    if rest:
        with multiprocessing.Pool(processes=min(len(rest), 4)) as pool:
            pool.map(build_lang, rest)

    typer.echo(f"All languages built into {SITE_DIR}")


@app.command()
def live(
    language: str = typer.Argument("en", help="Language code to serve"),
    dev_addr: str = typer.Option("127.0.0.1:8000", help="Address to bind"),
) -> None:
    """Run mkdocs serve for a language."""
    config_path = DOCS_DIR / language / "mkdocs.yml"
    if not config_path.exists():
        typer.echo(f"Config not found: {config_path}", err=True)
        raise typer.Exit(1)

    _sync_shared(language)

    cmd = [
        "zensical",
        "serve",
        "--config-file",
        str(config_path),
        "--dev-addr",
        dev_addr,
    ]
    subprocess.run(cmd, check=True)


@app.command()
def update_languages() -> None:
    """Regenerate extra.alternate in docs/en/mkdocs.yml using absolute links
    derived from site_url."""
    if not LANGUAGES_FILE.exists():
        typer.echo(f"{LANGUAGES_FILE} not found", err=True)
        raise typer.Exit(1)

    names = yaml.safe_load(LANGUAGES_FILE.read_text(encoding="utf-8"))
    langs = _discover_languages()

    en_yml = DOCS_DIR / "en" / "mkdocs.yml"
    en_data = (
        yaml.safe_load(en_yml.read_text(encoding="utf-8")) if en_yml.exists() else {}
    )
    site_url = en_data.get("site_url", "")
    prefix = urlparse(site_url).path.rstrip("/") if site_url else ""

    def _abs_link(target: str) -> str:
        if target == "en":
            return f"{prefix}/" if prefix else "/"
        return f"{prefix}/{target}/" if prefix else f"/{target}/"

    alternate = []
    for other in langs:
        link = _abs_link(other)
        name = names.get(other, other)
        alternate.append({"link": link, "name": f"{other} - {name}", "lang": other})

    en_data.setdefault("extra", {})["alternate"] = alternate
    en_yml.write_text(
        yaml.dump(en_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    typer.echo(f"Updated language alternates in {en_yml}")


@app.command()
def ensure_non_translated(
    language: str = typer.Argument(..., help="Language code"),
) -> None:
    """Delete translated files that should stay English-only (e.g. api/)."""
    lang_docs = DOCS_DIR / language / "docs"
    if not lang_docs.exists():
        return

    non_translated = {"api"}
    for name in non_translated:
        target = lang_docs / name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            typer.echo(f"Removed non-translated: {target}")


if __name__ == "__main__":
    app()
