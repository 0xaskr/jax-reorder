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

当前未配置项目技能。

## 本地工具

通过项目环境运行 tracing 示例和文档检查：

```bash
uv run python examples/inspect_jaxpr.py
uv run python tools/check_docs.py --changed
```

执行 Python 模块、追踪函数或运行候选实现时，先明确输入代码与执行范围。

依赖通过 `uv sync --locked` 安装；当前没有 `ai-tools` 依赖组。AI 候选保存在 `examples/`，由 AI 直接编写并 hardcode；运行器只生成输入和收集证据，不生成候选改写。回归入口见 [系统验证矩阵](development-docs/verification-matrix.md)。

## 自动维护

GitHub 工作流位于 [.github/workflows](../.github/workflows/)，推送和 PR 触发锁定 CPU 验证；候选 JAX 升级检查需手动触发。当前未配置 Plane/Falcon 自动维护或外部消息发送。
