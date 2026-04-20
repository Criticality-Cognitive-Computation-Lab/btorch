# 贡献指南

感谢你对改进 btorch 的兴趣！本页面涵盖了基本的环境搭建和提交更改的流程。

## 开发环境搭建

1. **克隆仓库**

```bash
git clone https://github.com/Criticality-Cognitive-Computation-Lab/btorch.git
cd btorch
```

2. **安装开发依赖**

```bash
pip install -e ".[dev]"
```

3. **安装 pre-commit 钩子**

```bash
pre-commit install --install-hooks
```

## 运行检查

在提交拉取请求之前，请运行以下检查：

```bash
# 代码检查
ruff check .

# 测试
pytest tests

# 文档构建
python scripts/docs.py command=build-all
```

为了更快获得反馈，可以运行针对性的子集：

```bash
pytest tests/models -k neuron
pytest tests/connectome
pytest tests/visualisation
```

## 代码风格

- 遵循现有的模块结构和命名规范。
- 使用现代 Python 类型注解（`|`、`list`、`dict`）。
- 在能够明确意图的地方使用 `jaxtyping` 标记张量形状。
- 每行保持在 88 个字符以内。
- 使用 Google 风格的文档字符串。
- 注释和文档字符串使用英文。

## 文档

- 任何面向用户的 API 变更都需要以英文更新 `docs/en/docs/`， ci会自动同步中文文档`docs/zh/docs/`。
- 安装或工作流变更需要更新 `README.md`。
- 新的 API 页面应使用 `mkdocstrings`（`::: btorch.module.path`），而不是手写签名。

## 拉取请求工作流

1. 从 `main` 创建功能分支。
2. 提交聚焦的、原子化的提交。
3. 确保所有检查通过。
4. 提交 PR，并清楚描述变更及其动机。

## 有问题？

打开 [GitHub Discussion](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/discussions) 或查看 [FAQ](faq.md)。
