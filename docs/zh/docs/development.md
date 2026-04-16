# 开发

安装 precommit 钩子以进行自动格式化。

未经 precommit 格式化的 PR 将不被接受！

```bash
pre-commit install --install-hooks
```

强烈建议使用 [jaxtyping](https://docs.kidger.site/jaxtyping/) 来标记预期的数组形状，请参阅 [使用 jaxtyping 的优秀示例](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer)。

使用 AI 编程助手时，请在提交 PR 之前使用 [`desloppify`](https://github.com/peteromallet/desloppify) 验证代码质量。建议将其安装为 pre-push git 钩子，以便及早发现问题。

## 运行测试

```bash
ruff check .
pytest tests
python scripts/docs.py build-all
```

## 本地演示工作流

在依赖 CI/CD 之前，可以在本地约 5 分钟内验证整个文档流水线。

### 1. 安装文档依赖

```bash
pip install -e .[docs]
```

或者如果你不需要运行 AI 自动生成：

```bash
pip install zensical>=0.0.32 mkdocstrings-python mkdocs-gen-files
pip install -e .
```

### 2. 在本地预览英文文档

```bash
python scripts/docs.py live en
# 打开 http://127.0.0.1:8000
```

注意：`zensical serve` 不会对文档文件夹外的源文件变更（例如 `btorch/`）自动重新加载。编辑文档字符串时请手动刷新浏览器。

### 3. 运行单页 AI 翻译

```bash
export OPENAI_API_KEY=...
# 可选：使用不同的 API 提供商（例如 DeepSeek、Azure、本地代理）
export OPENAI_BASE_URL=https://api.deepseek.com
# 可选：使用不同的模型（默认为 gpt-4o）
export OPENAI_MODEL=deepseek-chat
python scripts/translate.py translate-page \
  --language zh \
  --en-path docs/en/docs/installation.md
# 检查 docs/zh/docs/installation.md
```

### 4. 在本地预览中文文档

```bash
python scripts/docs.py live zh
# 打开 http://127.0.0.1:8000/zh/
```

### 5. 构建完整的统一站点

```bash
python scripts/docs.py build-all
# site/ 现在包含：
#   index.html          （英文默认）
#   zh/index.html       （中文）
```

### 6. 测试增量更新（最小差异）

- 修改 `docs/en/docs/installation.md` 中的一个句子
- 运行 `python scripts/translate.py update-outdated --language zh`
- 验证 `git diff docs/zh/docs/installation.md` 仅更改了相应的句子

### 7. 测试手动修复保留

- 在 `docs/zh/docs/installation.md` 的一段文字周围添加 `<!-- translate: freeze -->`
- 修改对应的英文源码
- 重新运行 `update-outdated`
- 验证被冻结的段落保持不变
