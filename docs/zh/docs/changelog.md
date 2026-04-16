# 更新日志

btorch 的所有重要变更都将记录在此文件中。

本格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
并且本项目遵循 [语义化版本控制](https://semver.org/spec/v2.0.0.html)。

## [未发布]

### 新增
- 扩展 API 参考覆盖范围：visualisation、io、utils、models（functional、history、environ、init、scale、regularizer、ode、connection_conversion）、analysis（dynamic_tools）和 connectome（connection、augment）。
- 新增核心概念指南：《有状态模块》、《`dt` 环境》和《替代梯度》。
- 新增教程：《教程 1：构建 RSNN》和《教程 2：训练 SNN》。
- 新增指南：《OmegaConf 配置指南》。
- 新增页面：常见问题、技能参考、示例库、贡献指南。
- MkDocs Material UX 升级：编辑/查看链接、面包屑导航、返回顶部按钮、可共享搜索 URL 和版本徽章。

### 变更
- 更新 `api/neurons.md` 和 `api/analysis.md` 以公开完整的公共 API。
- 改进 `btorch.models.functional`、`environ`、`init`、`scale`、`regularizer` 和 `btorch.utils.conf` 中 Google 风格的文档字符串。

### 移除
- 移除冗余的 `zh/docs/api/` 英文 mkdocstrings 块（等待完整翻译工作流更新后彻底清理）。
