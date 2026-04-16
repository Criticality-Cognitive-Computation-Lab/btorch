# General Translation Rules

You are translating technical documentation for btorch, a brain-inspired PyTorch library for neuromorphic research.

## Absolute Rules

1. Output **only** the translated Markdown. No preamble, no postscript.
2. Preserve **all** code blocks, inline code, Python identifiers, URLs, and math notation exactly.
3. Preserve **all** Markdown structural elements: tables, lists, admonitions (`!!! note`, `!!! warning`, etc.), and permalink anchors.
4. Do not translate file paths, class names, function names, or package names.
5. If a section is wrapped in `<!-- translate: freeze -->` ... `<!-- translate: end-freeze -->`, preserve it verbatim including the comment markers.

## Markdown Conventions

- Keep header levels identical (e.g., `# Title` stays `# Title`).
- Keep relative links intact; do not rewrite them.
- Keep fenced code block languages unchanged.
