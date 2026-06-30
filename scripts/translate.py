"""Translation CLI for btorch docs.

Supports AI-driven translation with freeze markers for human edits. Uses
OmegaConf for structured config and CLI parsing.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from openai import OpenAI


DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
GENERAL_PROMPT_PATH = Path(__file__).resolve().parent / "general-llm-prompt.md"

NON_TRANSLATED_SECTIONS = {"api"}

client: OpenAI | None = None


def _discover_readme_langs(repo_root: Path) -> dict[str, tuple[str, str]]:
    """Discover README files and return {lang_code: (display_name, filename)}.

    Filenames like ``README.md`` map to ``en``; ``README.zh.md`` maps to ``zh``.
    Display names are read from ``docs/language_names.yml`` if available,
    otherwise the language code itself is used as the fallback.
    """
    langs: dict[str, tuple[str, str]] = {}
    names: dict[str, str] = {}
    names_file = repo_root / "docs" / "language_names.yml"
    if names_file.exists():
        names = yaml.safe_load(names_file.read_text(encoding="utf-8")) or {}

    for path in sorted(repo_root.glob("README*")):
        if not path.is_file() or path.suffix != ".md":
            continue
        name = path.stem
        if name == "README":
            code = "en"
        elif name.startswith("README."):
            code = name.split(".", 1)[1]
        else:
            continue
        display = names.get(code, code)
        langs[code] = (display, path.name)
    return langs


def _generate_readme_switcher(
    current_lang: str, langs: dict[str, tuple[str, str]]
) -> str:
    """Generate the HTML language switcher bar for a README."""
    links: list[str] = []
    for code, (name, filename) in sorted(langs.items()):
        if code == current_lang:
            links.append(f"<b>{name}</b>")
        else:
            links.append(f'<a href="{filename}">{name}</a>')
    return (
        '<h4 align="center">\n'
        "    <p>\n"
        "        " + " |\n        ".join(links) + "\n"
        "    </p>\n"
        "</h4>"
    )


def _inject_readme_switcher(
    readme_path: Path, language: str, langs: dict[str, tuple[str, str]]
) -> None:
    """Inject or update the language switcher in a README file.

    The switcher is placed immediately after the first H1 heading.
    """
    if not readme_path.exists():
        return
    text = readme_path.read_text(encoding="utf-8")
    switcher = _generate_readme_switcher(language, langs)

    switcher_pattern = re.compile(
        r'<h4 align="center">\s*<p>\s*'
        r'((?:<a href="[^"]+">[^<]+</a>|<b>[^<]+</b>)\s*\|\s*)*'
        r'(?:<a href="[^"]+">[^<]+</a>|<b>[^<]+</b>)\s*'
        r"</p>\s*</h4>",
        re.DOTALL,
    )

    if switcher_pattern.search(text):
        new_text = switcher_pattern.sub(switcher, text)
        if new_text != text:
            readme_path.write_text(new_text, encoding="utf-8")
            print(f"Updated language switcher in {readme_path}")
        return

    h1_pattern = re.compile(r"^(# .+)$", re.MULTILINE)
    match = h1_pattern.search(text)
    if match:
        insert_pos = match.end()
        new_text = text[:insert_pos] + "\n\n" + switcher + text[insert_pos:]
        readme_path.write_text(new_text, encoding="utf-8")
        print(f"Injected language switcher in {readme_path}")


def _sync_readme_switchers() -> None:
    """Ensure all README files have up-to-date language switchers."""
    repo_root = Path(__file__).resolve().parent.parent
    langs = _discover_readme_langs(repo_root)
    for code, (_, filename) in langs.items():
        readme_path = repo_root / filename
        _inject_readme_switcher(readme_path, code, langs)


def _get_client() -> OpenAI:
    global client
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key and not base_url:
            print("OPENAI_API_KEY not set", file=sys.stderr)
            raise SystemExit(1)
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
        print(f"No prompt files found for {language}", file=sys.stderr)
        raise SystemExit(1)
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


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TranslateConf:
    """Top-level config for translate commands.

    Set ``command`` to select which operation to run.  Other fields are
    interpreted depending on the command.
    """

    command: str = ""
    language: str = ""
    en_path: str = ""
    since_ref: str = ""
    model: str = "gpt-4o"
    max_pages: int = 50
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def translate_page(cfg: TranslateConf) -> None:
    """Translate a single English page."""
    if not cfg.en_path:
        print("Missing required option: en_path", file=sys.stderr)
        raise SystemExit(1)
    src = Path(cfg.en_path).resolve()
    if not src.exists():
        print(f"File not found: {src}", file=sys.stderr)
        raise SystemExit(1)

    rel = src.relative_to(DOCS_DIR / "en" / "docs")
    if _is_non_translated(rel):
        print(f"Skipping non-translated section: {rel}")
        return

    result = _translate_file(src, cfg.language, model=cfg.model)
    if cfg.dry_run:
        print(result)
    else:
        dest = _mirror_path(src, cfg.language)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result, encoding="utf-8")
        print(f"Translated: {dest}")


def add_missing(cfg: TranslateConf) -> None:
    """Translate English pages that are missing in the target language."""
    en_docs = DOCS_DIR / "en" / "docs"
    lang_docs = DOCS_DIR / cfg.language / "docs"

    en_files = sorted(en_docs.rglob("*.md"))
    count = 0
    for en_file in en_files:
        rel = en_file.relative_to(en_docs)
        if _is_non_translated(rel):
            continue
        dest = lang_docs / rel
        if dest.exists():
            continue
        result = _translate_file(en_file, cfg.language, model=cfg.model)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result, encoding="utf-8")
        print(f"Created: {dest}")
        count += 1
        if count >= cfg.max_pages:
            print(f"Reached max_pages={cfg.max_pages}")
            break
    print(f"Added {count} missing translations")


def update_outdated(cfg: TranslateConf) -> None:
    """Re-translate pages where the English source is newer than the
    translation."""
    en_docs = DOCS_DIR / "en" / "docs"
    lang_docs = DOCS_DIR / cfg.language / "docs"

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

        result = _translate_file(en_file, cfg.language, model=cfg.model)
        dest.write_text(result, encoding="utf-8")
        print(f"Updated: {dest}")
        count += 1
        if count >= cfg.max_pages:
            print(f"Reached max_pages={cfg.max_pages}")
            break
    print(f"Updated {count} outdated translations")


def remove_removable(cfg: TranslateConf) -> None:
    """Delete translated pages whose English source no longer exists."""
    en_docs = DOCS_DIR / "en" / "docs"
    lang_docs = DOCS_DIR / cfg.language / "docs"

    if not lang_docs.exists():
        print("No translated docs found")
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
            print(f"Removed: {tr_file}")
            removed += 1
            parent = tr_file.parent
            while parent != lang_docs and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
    print(f"Removed {removed} obsolete translations")


def translate_changed(cfg: TranslateConf) -> None:
    """Translate only English doc pages changed since a given git ref."""
    if not cfg.since_ref:
        print("Missing required option: since_ref", file=sys.stderr)
        raise SystemExit(1)
    en_docs = DOCS_DIR / "en" / "docs"
    try:
        out = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                cfg.since_ref,
                "HEAD",
                "--",
                str(en_docs),
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        print("Failed to get changed files from git", file=sys.stderr)
        raise SystemExit(1)

    changed = [
        line.strip() for line in out.splitlines() if line.strip().endswith(".md")
    ]
    if not changed:
        print("No changed doc files to translate")
        return

    count = 0
    for file in changed:
        src = Path(file).resolve()
        if not src.exists():
            print(f"Skipping deleted file: {file}")
            continue
        rel = src.relative_to(en_docs)
        if _is_non_translated(rel):
            print(f"Skipping non-translated section: {rel}")
            continue
        result = _translate_file(src, cfg.language, model=cfg.model)
        dest = _mirror_path(src, cfg.language)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result, encoding="utf-8")
        print(f"Translated: {dest}")
        count += 1
        if count >= cfg.max_pages:
            print(f"Reached max_pages={cfg.max_pages}")
            break
    print(f"Translated {count} changed pages")


def translate_readme(cfg: TranslateConf) -> None:
    """Translate README.md to README.<language>.md."""
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "README.md"
    if not src.exists():
        print("README.md not found", file=sys.stderr)
        raise SystemExit(1)

    dest = repo_root / f"README.{cfg.language}.md"
    en_text = src.read_text(encoding="utf-8")
    base_prompt = _load_prompt(cfg.language)
    old_translation: str | None = None

    freeze_blocks: dict[str, str] = {}
    if dest.exists():
        old_translation = dest.read_text(encoding="utf-8")
        freeze_blocks = _extract_freeze_blocks(old_translation)
        if freeze_blocks:
            placeholder_text = old_translation
            for key, val in freeze_blocks.items():
                placeholder_text = placeholder_text.replace(val, key)
            old_translation = placeholder_text

    prompt = _build_prompt(base_prompt, en_text, old_translation)
    result = _call_llm(prompt, model=cfg.model)

    if freeze_blocks:
        result = _replace_freeze_blocks(result, freeze_blocks)

    if cfg.dry_run:
        print(result)
    else:
        dest.write_text(result, encoding="utf-8")
        print(f"Translated: {dest}")
        _sync_readme_switchers()


def update_and_add(cfg: TranslateConf) -> None:
    """Run add-missing then update-outdated."""
    add_missing(cfg)
    update_outdated(cfg)


def sync_readme_switchers(cfg: TranslateConf) -> None:
    """Ensure all README files have up-to-date language switchers."""
    _sync_readme_switchers()


def _extract_h1(path: Path) -> str | None:
    """Extract the first H1 heading from a Markdown file."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return None


def _collect_section_headers(nav: list) -> list[str]:
    """Collect all section header titles from a nav structure."""
    headers: list[str] = []

    def _walk(items):
        for item in items:
            if isinstance(item, dict):
                for title, value in item.items():
                    if isinstance(value, list):
                        headers.append(title)
                        _walk(value)

    _walk(nav)
    # Deduplicate while preserving order
    return list(dict.fromkeys(headers))


def _translate_headers(headers: list[str], language: str) -> dict[str, str]:
    """Batch-translate section header titles via LLM."""
    if not headers:
        return {}

    prompt = (
        f"Translate the following documentation section titles into {language}.\n"
        "Return each translation on a new line, in the same order.\n"
        "Do not add numbering, bullets, or commentary.\n\n"
        + "\n".join(f"- {h}" for h in headers)
        + "\n"
    )
    result = _call_llm(prompt).strip()
    lines = [line.strip() for line in result.splitlines() if line.strip()]

    translations: dict[str, str] = {}
    for i, header in enumerate(headers):
        if i < len(lines):
            cleaned = lines[i].lstrip("- ").lstrip("* ").strip()
            translations[header] = cleaned
        else:
            translations[header] = header
    return translations


def _translate_nav_item(
    item: str | dict,
    language: str,
    header_translations: dict[str, str],
) -> str | dict:
    """Recursively rebuild nav items with translated headers and H1 page
    titles."""
    lang_docs = DOCS_DIR / language / "docs"

    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        result: dict[str, str | list] = {}
        for title, value in item.items():
            if isinstance(value, str):
                h1 = _extract_h1(lang_docs / value)
                result[h1 if h1 else title] = value
            elif isinstance(value, list):
                translated_title = header_translations.get(title, title)
                result[translated_title] = [
                    _translate_nav_item(child, language, header_translations)
                    for child in value
                ]
            else:
                result[title] = value
        return result
    return item


def update_nav(cfg: TranslateConf) -> None:
    """Generate a nav override for a language mkdocs.yml.

    Page titles are extracted from translated H1 headings; section
    headers are translated via LLM.
    """
    en_yml = DOCS_DIR / "en" / "mkdocs.yml"
    lang_yml = DOCS_DIR / cfg.language / "mkdocs.yml"

    if not en_yml.exists():
        print(f"English config not found: {en_yml}", file=sys.stderr)
        raise SystemExit(1)

    en_data = yaml.safe_load(en_yml.read_text(encoding="utf-8"))
    en_nav = en_data.get("nav", [])

    headers = _collect_section_headers(en_nav)
    header_translations = _translate_headers(headers, cfg.language)

    translated_nav = [
        _translate_nav_item(item, cfg.language, header_translations) for item in en_nav
    ]

    if lang_yml.exists():
        lang_data = yaml.safe_load(lang_yml.read_text(encoding="utf-8"))
    else:
        lang_data = {}

    lang_data["nav"] = translated_nav

    lang_yml.write_text(
        yaml.dump(lang_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Updated nav in {lang_yml}")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


_COMMANDS: dict[str, callable] = {
    "translate-page": translate_page,
    "add-missing": add_missing,
    "update-outdated": update_outdated,
    "remove-removable": remove_removable,
    "translate-changed": translate_changed,
    "translate-readme": translate_readme,
    "update-and-add": update_and_add,
    "sync-readme-switchers": sync_readme_switchers,
    "update-nav": update_nav,
}


def _print_help() -> None:
    print("Usage: python translate.py command=<cmd> [options]")
    print()
    print("Commands:")
    for name in sorted(_COMMANDS):
        print(f"  {name}")
    print()
    print("Common options:")
    print("  language=<code>   Target language code (e.g. zh)")
    print("  model=<name>     LLM model (default: gpt-4o)")
    print("  max_pages=<n>    Max pages to translate (default: 50)")
    print("  dry_run=true     Print instead of writing")
    print("  en_path=<path>  Path to English Markdown file")
    print("  since_ref=<ref> Git ref to diff against")


def main() -> None:
    defaults = OmegaConf.structured(TranslateConf())
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
