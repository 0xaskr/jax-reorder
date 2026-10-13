# 核心文档的 AI 修改保护

## 目标与完成条件

按用户要求删除英文 README 内容，仅保留中文入口；为 README、Agent 入口、架构和设计规范增加默认只读边界，代码与开发记录仍可正常修改。

## 设计与取舍

- 使用 Codex 原生 permission profile，继承 `:workspace`，保留内置受保护目录；核心文档逐项声明 `read`。
- 使用 `on-request` 和 `approvals_reviewer = "user"`，越界请求由用户审批。
- AGENTS 说明保护范围与授权要求；明确普通的“同步文档”不构成修改核心文档的授权。
- 规则作用于加载该配置的 Codex 沙箱；其他 AI 客户端需另行配置其写入权限。
- 配置依据：[OpenAI Docs 权限配置](https://learn.chatgpt.com/docs/permissions)、[配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)。

## 工作项

- [x] 删除英文 README，修正文档引用与包元数据。
- [x] 检查邻仓与本机 Codex：邻仓 `.codex` 为空；本机为 0.162.1。
- [x] 在临时副本中验证读取、拒绝写入/删除/替换，以及普通文件可写。
- [x] 安装项目配置并更新 AGENTS。
- [x] 检查文档引用、包构建与最终差异，归档计划。

## 决策与进展

- 工作区在任务期间将中文版改名为 `README.md`，并同步了导航与包元数据。保留这些调整，将只读规则同步到当前文件名；保留已有 Git 暂存状态。
- 最终保护范围为 `README.md`、`AGENTS.md`、`CLAUDE.md`、`ARCHITECTURE.md`、`docs/index.md`、`docs/PLANS.md`、`docs/design-docs/`。权限配置自身也继承 `.codex/` 的只读保护。

## 验证结果

- `python3 /tmp/jax-reorder-test-protected-docs.py /tmp/jax-reorder-protected-docs.toml`：Codex 0.162.1 下通过。8 个文件可读，50 次覆盖、追加、删除、同目录重命名、移出、替换和目录写入尝试均被拒绝；源码与开发记录可增改移删。
- 测试使用独立临时仓库，并将同一 TOML 配置传给 `codex sandbox`；没有对项目真实文档进行破坏性试写。当前会话无法嵌套启动沙箱，验证通过权限审查后在会话沙箱外启动独立测试沙箱。
- `codex app-server --strict-config` 的 `config/read`：确认从项目 `.codex/config.toml` 加载；有效值为 `default_permissions = "protected-docs"`、`approval_policy = "on-request"`、`approvals_reviewer = "user"`，未设置旧 `sandbox_mode`。只读取配置，未发起模型任务。
- `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv build --offline --out-dir /tmp/jax-reorder-protected-docs-dist`：源码包与 wheel 构建成功，wheel 元数据包含当前中文版 README。
- 文档链接检查（全量和 `--changed`）、`git diff --check` 通过；本次未修改算法行为。

## 使用边界

当前会话权限不会因写入配置自动刷新；新会话需加载 `protected-docs`。旧 `sandbox_mode` 或启动时 `--sandbox` 会覆盖新 profile，不能混用。
