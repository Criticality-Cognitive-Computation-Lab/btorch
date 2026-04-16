"""Translation CLI for btorch docs.

Supports AI-driven translation with freeze markers for human edits.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import typer
from openai import OpenAI


app = typer.Typer(help="Btorch docs translation CLI")

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
GENERAL_PROMPT_PATH = Path(__file__).resolve().parent / "general-llm-prompt.md"

NON_TRANSLATED_SECTIONS = {"api"}

client: OpenAI | None = None


def _get_client() -> OpenAI:
    global client
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key and not base_url:
            typer.echo("OPENAI_API_KEY not set", err=True)
            raise typer.Exit(1)
        kwargs: dict[str, str] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        client = OpenAI(**kwargs)
    return client


def _load_prompt(language: str) -> str:
    parts = []
    if GENERAL_PROMPT_PATH.exists():
        parts.append(GENERAL_PROMPT_PATH.read_text(encoding="utf-8"))
    lang_prompt = DOCS_DIR / language / "llm-prompt.md"
    if lang_prompt.exists():
        parts.append(lang_prompt.read_text(encoding="utf-8"))
    if not parts:
        typer.echo(f"No prompt files found for {language}", err=True)
        raise typer.Exit(1)
    return "\n\n".join(parts)


def _mirror_path(en_path: Path, language: str) -> Path:
    rel = en_path.relative_to(DOCS_DIR / "en" / "docs")
    return DOCS_DIR / language / "docs" / rel


def _is_non_translated(rel_path: Path) -> bool:
    parts = rel_path.parts
    for section in NON_TRANSLATED_SECTIONS:
        if section in parts:
            return True
    return False


def _extract_freeze_blocks(text: str) -> dict[str, str]:
    pattern = re.compile(
        r"( *<!--\s*translate:\s*freeze\s*-->\n.*?\n"
        r" *<!--\s*translate:\s*end-freeze\s*-->)",
        re.DOTALL,
    )
    blocks = {}
    for i, match in enumerate(pattern.finditer(text), start=1):
        key = f"__FREEZE_BLOCK_{i}__"
        blocks[key] = match.group(1)
    return blocks


def _replace_freeze_blocks(text: str, blocks: dict[str, str]) -> str:
    for key, val in blocks.items():
        text = text.replace(key, val)
    return text


def _call_llm(prompt: str, model: str = "gpt-4o") -> str:
    c = _get_client()
    response = c.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical documentation translator. "
                    "Translate the provided Markdown file accurately, preserving "
                    "all code blocks, URLs, math notation, admonitions, and "
                    "permalink anchors. Do not add extra commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _build_prompt(
    base_prompt: str,
    en_text: str,
    old_translation: str | None = None,
) -> str:
    prompt_parts = [base_prompt]
    prompt_parts.append(
        "Translate the following Markdown document. "
        "Preserve all code blocks, inline code, URLs, permalinks, and admonitions. "
        "If freeze markers exist, preserve them exactly.\n"
    )
    if old_translation:
        prompt_parts.append(
            "An older translation is provided below. Reuse as much as possible; "
            "only update sections that correspond to changed English text.\n\n"
            "--- OLD TRANSLATION ---\n"
            f"{old_translation}\n"
            "--- END OLD TRANSLATION ---\n\n"
        )
    prompt_parts.append(f"--- ENGLISH SOURCE ---\n{en_text}\n--- END ---")
    return "\n\n".join(prompt_parts)


def _translate_file(en_path: Path, language: str, model: str | None = None) -> str:
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    en_text = en_path.read_text(encoding="utf-8")
    base_prompt = _load_prompt(language)
    old_translation: str | None = None

    mirror = _mirror_path(en_path, language)
    freeze_blocks: dict[str, str] = {}
    if mirror.exists():
        old_translation = mirror.read_text(encoding="utf-8")
        freeze_blocks = _extract_freeze_blocks(old_translation)
        if freeze_blocks:
            placeholder_text = old_translation
            for key, val in freeze_blocks.items():
                placeholder_text = placeholder_text.replace(val, key)
            old_translation = placeholder_text

    prompt = _build_prompt(base_prompt, en_text, old_translation)
    result = _call_llm(prompt, model=model)

    if freeze_blocks:
        result = _replace_freeze_blocks(result, freeze_blocks)
    return result


@app.command()
def translate_page(
    language: str = typer.Option(..., help="Target language code"),
    en_path: str = typer.Option(..., help="Path to English Markdown file"),
    model: str = typer.Option(
        os.environ.get("OPENAI_MODEL", "gpt-4o"), help="LLM model"
    ),
    dry_run: bool = typer.Option(False, help="Print instead of writing"),
) -> None:
    """Translate a single English page."""
    src = Path(en_path).resolve()
    if not src.exists():
        typer.echo(f"File not found: {src}", err=True)
        raise typer.Exit(1)

    rel = src.relative_to(DOCS_DIR / "en" / "docs")
    if _is_non_translated(rel):
        typer.echo(f"Skipping non-translated section: {rel}")
        raise typer.Exit(0)

    result = _translate_file(src, language, model=model)
    if dry_run:
        typer.echo(result)
    else:
        dest = _mirror_path(src, language)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result, encoding="utf-8")
        typer.echo(f"Translated: {dest}")


@app.command()
def add_missing(
    language: str = typer.Option(..., help="Target language code"),
    max_pages: int = typer.Option(50, help="Max pages to translate"),
    model: str = typer.Option(
        os.environ.get("OPENAI_MODEL", "gpt-4o"), help="LLM model"
    ),
) -> None:
    """Translate English pages that are missing in the target language."""
    en_docs = DOCS_DIR / "en" / "docs"
    lang_docs = DOCS_DIR / language / "docs"

    en_files = sorted(en_docs.rglob("*.md"))
    count = 0
    for en_file in en_files:
        rel = en_file.relative_to(en_docs)
        if _is_non_translated(rel):
            continue
        dest = lang_docs / rel
        if dest.exists():
            continue
        result = _translate_file(en_file, language, model=model)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result, encoding="utf-8")
        typer.echo(f"Created: {dest}")
        count += 1
        if count >= max_pages:
            typer.echo(f"Reached max_pages={max_pages}")
            break
    typer.echo(f"Added {count} missing translations")


@app.command()
def update_outdated(
    language: str = typer.Option(..., help="Target language code"),
    max_pages: int = typer.Option(50, help="Max pages to translate"),
    model: str = typer.Option(
        os.environ.get("OPENAI_MODEL", "gpt-4o"), help="LLM model"
    ),
) -> None:
    """Re-translate pages where the English source is newer than the
    translation."""
    en_docs = DOCS_DIR / "en" / "docs"
    lang_docs = DOCS_DIR / language / "docs"

    en_files = sorted(en_docs.rglob("*.md"))
    count = 0
    for en_file in en_files:
        rel = en_file.relative_to(en_docs)
        if _is_non_translated(rel):
            continue
        dest = lang_docs / rel
        if not dest.exists():
            continue

        def _last_commit_time(path: Path) -> int:
            try:
                out = subprocess.check_output(
                    ["git", "log", "-1", "--format=%ct", str(path)],
                    text=True,
                ).strip()
                return int(out) if out else 0
            except Exception:
                return 0

        en_time = _last_commit_time(en_file)
        tr_time = _last_commit_time(dest)
        if en_time <= tr_time:
            continue

        result = _translate_file(en_file, language, model=model)
        dest.write_text(result, encoding="utf-8")
        typer.echo(f"Updated: {dest}")
        count += 1
        if count >= max_pages:
            typer.echo(f"Reached max_pages={max_pages}")
            break
    typer.echo(f"Updated {count} outdated translations")


@app.command()
def remove_removable(
    language: str = typer.Option(..., help="Target language code"),
) -> None:
    """Delete translated pages whose English source no longer exists."""
    en_docs = DOCS_DIR / "en" / "docs"
    lang_docs = DOCS_DIR / language / "docs"

    if not lang_docs.exists():
        typer.echo("No translated docs found")
        return

    tr_files = sorted(lang_docs.rglob("*.md"))
    removed = 0
    for tr_file in tr_files:
        rel = tr_file.relative_to(lang_docs)
        if _is_non_translated(rel):
            continue
        src = en_docs / rel
        if not src.exists():
            tr_file.unlink()
            typer.echo(f"Removed: {tr_file}")
            removed += 1
            # Clean up empty parent dirs
            parent = tr_file.parent
            while parent != lang_docs and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
    typer.echo(f"Removed {removed} obsolete translations")


@app.command()
def update_and_add(
    language: str = typer.Option(..., help="Target language code"),
    max_pages: int = typer.Option(50, help="Max pages to translate per step"),
    model: str = typer.Option(
        os.environ.get("OPENAI_MODEL", "gpt-4o"), help="LLM model"
    ),
) -> None:
    """Run add-missing then update-outdated."""
    add_missing(language=language, max_pages=max_pages, model=model)
    update_outdated(language=language, max_pages=max_pages, model=model)


if __name__ == "__main__":
    app()
