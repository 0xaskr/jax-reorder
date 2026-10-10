# AI 协作与工具使用

AI 根据仓库内的契约和证据完成设计、实现、验证与维护。入口是 [AGENTS.md](../AGENTS.md)，Claude Code 通过 [CLAUDE.md](../CLAUDE.md) 读取同一份规则。

## 工作闭环

1. 读取当前任务相关的契约、代码和已有计划，明确原始实现、候选实现与比较条件。
2. 在 [执行计划](PLANS.md) 中拆解复杂工作；将算法假设和取舍写入开发设计。
3. AI 提出候选方案，确定性脚本负责执行、校验、计量和记录失败。
4. 读取结构化结果，根据证据调整方案。原始输入、版本、规则和判定标准保持可追溯。
5. 完成后审查差异、归档计划、更新质量记录；非显然的调试经验提炼为窄范围技能。

短任务直接完成，按需要读取资料。已有授权覆盖的后续步骤持续执行；不因每一步实现选择重复等待确认。

## 技能选择

当前未配置项目技能。新增技能的约定见 [技能说明](references/skills.md)。

## 本地工具

通过项目环境运行 tracing 示例和文档检查：

```bash
uv run python examples/inspect_jaxpr.py
uv run python scripts/check_docs.py
```

执行 Python 模块、追踪函数或运行候选实现时，先明确输入代码与执行范围。

独立辅助工具保存在 `scripts/`。通过 `uv sync --locked --group ai-tools` 安装额外依赖，运行时使用 `uv run --group ai-tools`；支持范围以各脚本的说明为准，回归命令见 [贡献指引](../CONTRIBUTING.md)。

## 自动维护

GitHub 工作流位于 [.github/workflow-templates](../.github/workflow-templates/)，当前不会自动执行。

`scripts/pm_agent.py` 是确定性的 Plane 自动化工具，可扫描过期任务、催更、汇总周报和分配任务；它本身不调用大模型。实际外部写入需要明确配置目标，并使用 `--apply`。
