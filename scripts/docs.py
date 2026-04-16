"""Build orchestration for btorch multilingual docs.

Builds each language into a unified `site/` directory for GitHub Pages:
- English at `site/`
- Other languages under `site/<lang>/`
"""

from __future__ import annotations

import multiprocessing
import shutil
import subprocess
import sys
from pathlib import Path

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
    else:
        dest = SITE_DIR / language

    if dest.exists():
        shutil.rmtree(dest)

    cmd = [
        sys.executable,
        "-m",
        "mkdocs",
        "build",
        "--config-file",
        str(config_path),
        "--site-dir",
        str(dest),
    ]
    typer.echo(f"Building {language} -> {dest}")
    subprocess.run(cmd, check=True)


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

    cmd = [
        sys.executable,
        "-m",
        "mkdocs",
        "serve",
        "--config-file",
        str(config_path),
        "--dev-addr",
        dev_addr,
    ]
    subprocess.run(cmd, check=True)


@app.command()
def update_languages() -> None:
    """Regenerate extra.alternate in docs/en/mkdocs.yml from
    language_names.yml."""
    if not LANGUAGES_FILE.exists():
        typer.echo(f"{LANGUAGES_FILE} not found", err=True)
        raise typer.Exit(1)

    names = yaml.safe_load(LANGUAGES_FILE.read_text(encoding="utf-8"))
    langs = _discover_languages()

    alternate = []
    for lang in langs:
        link = "/" if lang == "en" else f"/{lang}/"
        name = names.get(lang, lang)
        alternate.append({"link": link, "name": f"{lang} - {name}", "lang": lang})

    en_yml = DOCS_DIR / "en" / "mkdocs.yml"
    data = yaml.safe_load(en_yml.read_text(encoding="utf-8"))
    data.setdefault("extra", {})["alternate"] = alternate

    en_yml.write_text(
        yaml.dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    typer.echo("Updated language alternates in docs/en/mkdocs.yml")


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
