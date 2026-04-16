# Development

Install precommit hooks for auto formatting.

PR without precommit formatting will not be accepted!

```bash
pre-commit install --install-hooks
```

Highly recommended to use [jaxtyping](https://docs.kidger.site/jaxtyping/) to mark expected array shape,
see [good example of using jaxtyping](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer)

When using AI coding assistants, pls use [`desloppify`](https://github.com/peteromallet/desloppify) to verify code quality before submitting PRs. Consider installing it as a pre-push git hook to catch issues early.

## Run the tests

```bash
ruff check .
pytest tests
python scripts/docs.py build-all
```

## Local Demonstration Workflow

Before relying on CI/CD, the entire docs pipeline can be verified locally in ~5 minutes.

### 1. Install docs dependencies

```bash
pip install -e .[docs]
```

### 2. Serve English docs locally

```bash
python scripts/docs.py live --language en
# open http://127.0.0.1:8000
```

### 3. Run a single-page AI translation

```bash
export OPENAI_API_KEY=...
# Optional: use a different API provider (e.g. DeepSeek, Azure, local proxy)
export OPENAI_BASE_URL=https://api.deepseek.com
# Optional: use a different model (default is gpt-4o)
export OPENAI_MODEL=deepseek-chat
python scripts/translate.py translate-page \
  --language zh \
  --en-path docs/en/docs/installation.md
# inspect docs/zh/docs/installation.md
```

### 4. Serve Chinese docs locally

```bash
python scripts/docs.py live --language zh
# open http://127.0.0.1:8000/zh/
```

### 5. Build the full unified site

```bash
python scripts/docs.py build-all
# site/ now contains:
#   index.html          (English default)
#   zh/index.html       (Chinese)
```

### 6. Test incremental update (minimal-diff)

- Edit one sentence in `docs/en/docs/installation.md`
- Run `python scripts/translate.py update-outdated --language zh`
- Verify `git diff docs/zh/docs/installation.md` only changes the corresponding sentence

### 7. Test manual-fix preservation

- Add `<!-- translate: freeze -->` around a paragraph in `docs/zh/docs/installation.md`
- Edit the matching English source
- Re-run `update-outdated`
- Verify the frozen paragraph stays intact
