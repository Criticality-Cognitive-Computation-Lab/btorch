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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml
from omegaconf import OmegaConf


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
        print(f"Synced {name}: {src} -> {dest}")


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


def build_lang(cfg: DocsConf) -> None:
    """Build a single language into the unified site directory."""
    language = cfg.language
    config_path = DOCS_DIR / language / "mkdocs.yml"
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        raise SystemExit(1)

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

    _sync_shared(language)

    if language != "en":
        nav_script = Path(__file__).resolve().parent / "translate.py"
        subprocess.run(
            [
                sys.executable,
                str(nav_script),
                "command=update-nav",
                f"language={language}",
            ],
            check=True,
        )

    api_script = Path(__file__).resolve().parent / "gen_api_pages.py"
    print("Generating API reference pages...")
    subprocess.run(["python", str(api_script)], check=True)

    cmd = [
        "zensical",
        "build",
        "--config-file",
        str(config_path),
    ]
    print(f"Building {language} -> {dest}")
    subprocess.run(cmd, check=True)

    if default_output.exists():
        shutil.move(str(default_output), str(dest))


def build_all(cfg: DocsConf) -> None:
    """Build all languages in parallel."""
    langs = _discover_languages()
    if not langs:
        print("No language configs found.", file=sys.stderr)
        raise SystemExit(1)

    if "en" in langs:
        build_lang(DocsConf(command="build-lang", language="en"))
        rest = [lang for lang in langs if lang != "en"]
    else:
        rest = langs

    if rest:
        with multiprocessing.Pool(processes=min(len(rest), 4)) as pool:
            pool.map(
                lambda lang: build_lang(DocsConf(command="build-lang", language=lang)),
                rest,
            )

    print(f"All languages built into {SITE_DIR}")


def live(cfg: DocsConf) -> None:
    """Run mkdocs serve for a language."""
    config_path = DOCS_DIR / cfg.language / "mkdocs.yml"
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        raise SystemExit(1)

    _sync_shared(cfg.language)

    if cfg.language != "en":
        nav_script = Path(__file__).resolve().parent / "translate.py"
        subprocess.run(
            [
                sys.executable,
                str(nav_script),
                "command=update-nav",
                f"language={cfg.language}",
            ],
            check=True,
        )

    cmd = [
        "zensical",
        "serve",
        "--config-file",
        str(config_path),
        "--dev-addr",
        cfg.dev_addr,
    ]
    subprocess.run(cmd, check=True)


def update_languages(cfg: DocsConf) -> None:
    """Regenerate extra.alternate in docs/en/mkdocs.yml using absolute links
    derived from site_url."""
    if not LANGUAGES_FILE.exists():
        print(f"{LANGUAGES_FILE} not found", file=sys.stderr)
        raise SystemExit(1)

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
    print(f"Updated language alternates in {en_yml}")


def ensure_non_translated(cfg: DocsConf) -> None:
    """Delete translated files that should stay English-only (e.g. api/)."""
    lang_docs = DOCS_DIR / cfg.language / "docs"
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
            print(f"Removed non-translated: {target}")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class DocsConf:
    """Top-level config for docs commands.

    Set ``command`` to select which operation to run.
    """

    command: str = ""
    language: str = "en"
    dev_addr: str = "127.0.0.1:8000"


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


_COMMANDS: dict[str, Callable] = {
    "build-lang": build_lang,
    "build-all": build_all,
    "live": live,
    "update-languages": update_languages,
    "ensure-non-translated": ensure_non_translated,
}


def _print_help() -> None:
    print("Usage: python docs.py command=<cmd> [options]")
    print()
    print("Commands:")
    for name in sorted(_COMMANDS):
        print(f"  {name}")
    print()
    print("Common options:")
    print("  language=<code>  Language code (default: en)")
    print("  dev_addr=<addr>  Address to bind (default: 127.0.0.1:8000)")


def main() -> None:
    defaults = OmegaConf.structured(DocsConf())
    cli_cfg = OmegaConf.from_cli()

    if "command" not in cli_cfg or not cli_cfg.command:
        _print_help()
        raise SystemExit(1)

    cfg = OmegaConf.to_object(OmegaConf.unsafe_merge(defaults, cli_cfg))

    if cfg.command not in _COMMANDS:
        print(f"Unknown command: {cfg.command}", file=sys.stderr)
        _print_help()
        raise SystemExit(1)

    _COMMANDS[cfg.command](cfg)


if __name__ == "__main__":
    main()
