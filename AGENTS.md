# AGENTS.md

本文件是 AI 在 `jax-reorder` 中工作的简明入口。按任务需要进入设计、计划和工具，详细约束在对应文档中维护。

## 从这里开始

- 项目介绍与使用方式和示例：[中文 README](README.md)。
- 架构与比较契约：[ARCHITECTURE.md](ARCHITECTURE.md)。
- 算法设计、契约与验证方法：[设计文档索引](docs/design-docs/index.md)。
- 文档知识库：[docs/index.md](docs/index.md)。
- 执行计划与待处理事项：[执行计划索引](docs/exec-plans/index.md)。

## 当前形态与目录

项目通过 jaxpr 结构规范化比较 AI 生成的 JAX 代码与原始代码。已实现完整图比较、函数契约与一阶求导检查；支持状态和边界见架构与开发记录。实现按 `core`（结构规范化与比较）→ `algebra`（SymPy 代数后端）→ `functions`（函数契约编排）分层，新代码不得反向依赖。

| 路径 | 用途 |
| --- | --- |
| [examples/inspect_jaxpr.py](examples/inspect_jaxpr.py) | 当前 tracing 入口；可运行示例放在 examples |
| [src/jax_reorder](src/jax_reorder/) | 实现代码：`core/` 结构层、`algebra/` 数学层、`functions.py` 契约编排 |
| [tests](tests/) | 算法、CPU 原语与 AI 固定用例测试 |
| [docs/design-docs](docs/design-docs/index.md) | 根目录仅保留索引；`current/` 放现行设计与验证方法，`archive/` 放历史方案 |
| [docs/exec-plans](docs/exec-plans/index.md) | 执行计划、实施过程、实验与验收记录 |
| [.github/workflows](.github/workflows/) | CPU CI |

## 必须遵守

- 以架构文档为比较契约的维护入口。保留运算、操作数与输入输出顺序、依赖、常量、抽象值、primitive 参数和嵌套语义。
- 结构匹配与代数匹配分开；浮点交换律或结合律改写必须发出警告，不承诺误差很小或逐位等价。
- 不支持的 effects、不透明参数和嵌套构造明确失败。不能忽略语义来获得匹配，也不能从不匹配推出程序不等价。
- 比较仅针对追踪输入、静态参数和执行环境，并包含函数输入输出 pytree；不宣称一般 Python 等价性。
- IR 语法校验、抽样数值验证和性能数据分别报告，不能替代规范化表示的精确比较。
- 保留用户已有改动，限定任务范围；不提交凭据、环境或生成产物。Git 提交、推送和外部写入按用户授权执行。

## 受保护文档

- 默认只读：`README.md`、`AGENTS.md`、`CLAUDE.md`、`ARCHITECTURE.md`、`docs/index.md`、`docs/PLANS.md` 与 `docs/design-docs/` 整个目录。
- 未经用户明确授权具体文件和修改范围，不得修改、删除、移动、重命名或替换这些文档。“实现功能”“整理代码”“同步文档”等泛化要求不构成授权；发现需要调整时先给出建议改动，再请求授权。已有明确授权时按授权范围执行。
- 设计与契约写入 `docs/design-docs/current/`，历史方案归入 `docs/design-docs/archive/`，实验、实施与任务进展写入 `docs/exec-plans/`；受保护文档的编辑遵循上述授权要求。
- Codex 使用 [.codex/config.toml](.codex/config.toml) 中的 `protected-docs` 权限配置，继承工作区默认保护，核心文档只读，越界审批交给用户。AI 不得自行放宽配置、切换到更宽权限或绕过保护。
- 新会话需加载该权限配置；不要混用旧 `sandbox_mode` / `--sandbox`，或用 Full Access 覆盖它。其他 AI 客户端也须遵守本节规则，其文件写入拦截需在对应客户端另行配置。

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

- README 面向使用者；AGENTS 保持导航和必要规则。
- 总体比较契约在架构文档中维护，具体算法、接口与长期决策进入 `docs/design-docs/current/`；`docs/design-docs/` 根目录只保留 `index.md`。实施、实验与验收进入 exec-plans。
- 复杂工作的 active 计划完成后移到 completed；未解决问题登记到执行计划索引的“待处理”清单，并链接对应计划。
- 新增、移动或删除文档时更新索引与引用，运行 `uv run python tools/check_docs.py --changed` 与 `git diff --check`。
- 上游快照保留原项目上下文，本项目的当前状态与验证结论独立记录。
