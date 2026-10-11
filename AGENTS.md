# AGENTS.md

本文件是 AI 在 `jax-reorder` 中工作的简明入口。按任务需要进入设计、计划和工具，详细约束在对应文档中维护。

## 从这里开始

- 项目介绍与运行方式：[中文 README](README.zh.md)、[英文 README](README.md)。
- 架构与比较契约：[ARCHITECTURE.md](ARCHITECTURE.md)。
- 算法设计、审查与验证：[BFS 开发记录](docs/development-docs/bfs-reordering.md)。
- 文档知识库：[docs/index.md](docs/index.md)。
- AI 工作闭环与工具：[docs/AI_WORKFLOW.md](docs/AI_WORKFLOW.md)。
- 执行计划与技术债：[docs/PLANS.md](docs/PLANS.md)。

## 当前形态与目录

项目通过 jaxpr 结构规范化比较 AI 生成的 JAX 代码与原始代码。已实现完整图比较、函数契约与一阶求导检查；支持状态和边界见架构与开发记录。

| 路径 | 用途 |
| --- | --- |
| [examples/inspect_jaxpr.py](examples/inspect_jaxpr.py) | 当前 tracing 入口；可运行示例放在 examples |
| [src/jax_reorder](src/jax_reorder/) | 实现代码目录，BFS、语义编码及比较实现 |
| [tests](tests/) | 算法、CPU 原语与 AI 固定用例测试 |
| [docs/development-docs](docs/development-docs/index.md) | 算法、工具设计与实验记录 |
| [docs/design-docs](docs/design-docs/index.md) | 跨功能设计决定 |
| [.github/workflows](.github/workflows/) | CPU CI |

## 必须遵守

- 以架构文档为比较契约的维护入口。保留运算、操作数与输入输出顺序、依赖、常量、抽象值、primitive 参数和嵌套语义。
- 结构匹配与代数匹配分开；浮点交换律或结合律改写必须发出警告，不承诺误差很小或逐位等价。
- 不支持的 effects、不透明参数和嵌套构造明确失败。不能忽略语义来获得匹配，也不能从不匹配推出程序不等价。
- 比较仅针对追踪输入、静态参数和执行环境，并包含函数输入输出 pytree；不宣称一般 Python 等价性。
- IR 语法校验、抽样数值验证和性能数据分别报告，不能替代规范化表示的精确比较。
- 保留用户已有改动，限定任务范围；不提交凭据、环境或生成产物。Git 提交、推送和外部写入按用户授权执行。

## 工作方式

- 简单任务直接完成。跨文件功能、算法或迁移按 [计划规范](docs/PLANS.md) 维护进度、决定与验证记录。
- 新增或修改算法前或同时更新开发设计，写明支持范围、反例和验证方法。
- 优先结构化接口和确定性工具；用证据区分假设、已确认根因与未验证条件。
- 完成后审查差异、同步文档、归档计划。只有具有复用价值且根因已确认的调试经验才沉淀为技能。

## 常用命令

使用 Python 3.12 和 uv，与 [.python-version](.python-version) 保持一致。

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run python examples/inspect_jaxpr.py
uv run pytest
uv run python examples/bfs_reordering.py
uv run python tests/lax_inventory.py
```

- 提交前运行 Ruff、格式检查和 tracing 示例；算法变更还需运行 pytest；验证证据见开发记录。
- 缓存只读时设置 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache`。
- 使用 `uv add` / `uv add --dev` 管理依赖，提交依赖变更时包含 `uv.lock`。
- CI 已启用 `uv run pytest`；新增规范化规则应覆盖正例、反例与拒绝路径。

## 技能入口

当前未配置项目技能。

## 文档维护

- README 面向使用者，中英文同步；AGENTS 保持导航和必要规则。
- 比较契约在架构文档中维护，具体设计与实验进入 development-docs，长期决策进入 design-docs。
- 复杂工作的 active 计划完成后移到 completed；未解决问题进入技术债记录。
- 新增、移动或删除文档时更新索引与引用，运行 `uv run python tools/check_docs.py --changed` 与 `git diff --check`。
- 上游快照保留原项目上下文，本项目的当前状态与验证结论独立记录。
