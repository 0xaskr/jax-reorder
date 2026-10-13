# 文档索引

仓库内保存 AI 完成任务所需的规则、设计、计划和验证证据。按下表寻找对应内容，并在修改时维护相应索引。

| 入口 | 内容 |
| --- | --- |
| [AGENTS.md](../AGENTS.md) | Agent 规则、命令与技能入口 |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | 项目架构与比较契约 |
| [design-docs](design-docs/index.md) | 跨功能设计决定与核心原则 |
| [development-docs](development-docs/index.md) | 具体算法或工具的设计、实验与验证 |
| [references](references/index.md) | 工具参考、迁移来源与上游资料 |
| [PLANS.md](PLANS.md) | 执行计划规范 |
| [执行计划](exec-plans/active/) | 进行中的工作 |
| [已完成计划](exec-plans/completed/) | 完成记录与验证证据 |
| [技术债](exec-plans/tech-debt-tracker.md) | 已知缺口及状态 |
| [系统验证矩阵](development-docs/verification-matrix.md) | 覆盖、失败分类、CI 与版本升级证据 |

## 维护规则

- 每项规则或事实有明确的维护位置；入口通过链接引用，避免多份正文漂移。
- 新增、移动或删除文档时同步更新索引与引用。
- 复杂任务维护计划，完成后归档；未解决问题进入技术债记录。
- README 面向使用者；内部设计和计划默认使用中文。
- 标明规划、实现和验证状态。上游实验记录保留其项目、版本和设备上下文。
- 能由确定性检查保障的约束放入脚本、lint 或测试。文档修改运行 `uv run python tools/check_docs.py --changed` 和 `git diff --check`。
