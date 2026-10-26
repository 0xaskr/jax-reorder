# 文档索引

仓库内保存 AI 完成任务所需的规则、设计、计划和验证证据。按下表寻找对应内容，并在修改时维护相应索引。

| 入口 | 内容 |
| --- | --- |
| [AGENTS.md](../AGENTS.md) | Agent 规则、命令与技能入口 |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | 项目架构与比较契约 |
| [design-docs](design-docs/index.md) | 根索引逐项标明状态；`current/` 放现行设计、契约与验证方法，`archive/` 放历史方案 |
| [references](references/index.md) | 工具参考、迁移来源与上游资料 |
| [PLANS.md](PLANS.md) | 执行计划规范 |
| [执行计划索引](exec-plans/index.md) | 进行中的工作、已完成阶段及实验/验收归档 |
| [已完成计划](exec-plans/completed/) | 完成记录与验证证据 |
| [待处理清单](exec-plans/index.md#待处理) | 已知缺口及状态 |
| [系统验证矩阵](design-docs/current/verification-matrix.md) | 覆盖、失败分类、CI 与版本升级证据 |

## 维护规则

- 每项规则或事实有明确的维护位置；入口通过链接引用，避免多份正文漂移。
- 设计与验证方法放 `design-docs/current/`，历史方案放 `design-docs/archive/` 并标明替代关系；`design-docs/` 根目录只保留 `index.md`。实施过程、实验和实际验收结果放 exec-plans。
- 新增、移动或删除文档时同步更新索引与引用。
- 复杂任务维护计划，完成后归档；未解决问题登记到执行计划索引的“待处理”清单，并链接对应计划。
- README 面向使用者；内部设计和计划默认使用中文。
- 标明规划、实现和验证状态。上游实验记录保留其项目、版本和设备上下文。
- 能由确定性检查保障的约束放入脚本、lint 或测试。文档修改运行 `uv run python tools/check_docs.py --changed` 和 `git diff --check`。
