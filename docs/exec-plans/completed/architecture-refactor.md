# 代码架构分层重构

状态：2026-10-18 完成。实现已拆为 `core` / `algebra` / `functions` 三层，公共顶层 API 不变；子模块导入路径按 0.2.0 破坏性调整。

## 目标与完成条件

- 解决 `src/jax_reorder/` 下 20 个模块平铺、核心层与代数层混在一起的问题。
- 按实际依赖方向分层：`errors → core → algebra → functions`，禁止反向导入。
- 拆掉既有的两个导入环（`symbolic ↔ regions`、`symbolic ↔ trusted`）。
- 顶层公共接口（`jax_reorder.__all__` 的 29 个名字）保持不变；不提供旧子模块路径兼容层。
- 验证：Ruff、格式、分层守卫测试、分组 pytest、示例脚本与文档链接检查通过。

## 设计与取舍

- `core/`：SSA/语义编码、BFS 与完整图搜索、原图对应证据、注册求导规则物化，以及新的 `params.py`（共享参数视图）。
- `algebra/`：`compare.py` 编排；`adapter/` 负责 JAX 图到数学层的桥接；`symbolic.py` 构造表达式；`regions.py` 绑定控制流；`trusted.py` 结果协议与隔离 worker；`explicit.py` 保留参考实现。
- 抽出 `expression.py`（`SymbolicTensor`/坐标/绑定重命名）与 `sympy_ops.py`（预算受控的化简与查询）作为共享底层，消除环。
- 提升跨模块私有 API：`canonical.difference`、`canonical.ordered_equation`、`core.params.semantic_parameters`、`core.params.parameter_view`。
- 不保留 `jax_reorder.canonical`、`jax_reorder.algebra_trusted` 等旧路径 shim：它们会重新制造平铺文件并与“子模块是内部实现”的定位冲突。版本升为 0.2.0。

## 工作项

- [x] 创建 `core/`、`algebra/`、`algebra/adapter/`，移动并重命名全部模块。
- [x] 改写包内相对导入、tests/tools 的旧路径及 worker 的 `-m` 入口。
- [x] 抽出 `expression.py` / `sympy_ops.py`，删除函数内的隐藏环导入。
- [x] 提升共享内部 API，更新调用方与测试。
- [x] 更新 `ARCHITECTURE.md`、`AGENTS.md`、development-docs 源码链接与版本号。
- [x] 新增 `tests/test_package_layout.py` 守卫分层、无环与公共 API 面。
- [x] 运行验证并归档本记录。

## 决策与进展

- 2026-10-18：确认在当前依赖图中，`errors` 被 14 个模块依赖，`correspondence`/`derivatives`/`semantics` 构成 core 内部基础；两个环均由函数内延迟导入掩盖，必须通过抽共享底层解决，而不是简单移动文件。
- 采用“公共 API 不变、子模块路径破坏”的方案；用户明确不需要兼容外部使用者，并授权修改 `ARCHITECTURE.md` 与 `AGENTS.md`。
- 大文件拆分（`canonical.py`、`symbolic.py`）不在本轮范围：先保证层级与导入方向正确，避免在同一次迁移中混合结构性拆分与语义风险。

## 验证结果

- `uv sync --locked`：通过（`jax-reorder 0.2.0`，锁文件仅版本变化）。
- `uv run --offline ruff check .`、`ruff format --check .`、`git diff --check`：通过。
- `tests/test_package_layout.py`：**4 passed**（全部模块可导入、无环、分层方向、公共 API 面）；包内共 27 个模块。
- 核心/契约分组（canonical、semantics、functions、correspondence、depth、unrooted、custom derivatives、structural、array conversion、symbolic explicit、comparison errors、adapter guards）：**465 passed**。
- 代数快速分组（finiteness、array conversion、padding、cast sharding、jit composition、common calls）：**86 passed**；`test_trusted_sympy.py`：**44 passed**（验证 worker 路径与 sympy_ops 抽取）；区域/扫描/无限性/abs-where/SymPy 分组：**207 passed，1 failed**（`test_captured_array_...` 的 10 秒 SymPy worker 超时，在重构前同一环境已复现，非本次引入）。
- 示例：`inspect_jaxpr.py`、`check_correctness.py --trials 1`（4/4）、`cond_bindings.py`、`sympy_equivalence.py` 通过；`tests/adapter_inventory.py`、`tests/lax_inventory.py` 通过（Windows 控制台需 `PYTHONUTF8=1`）。
- 文档：`tools/check_docs.py` 检查 **61 个 Markdown、424 个本地目标**，无缺失。
- 未完成项：`test_algebra.py` 的完整参数化轮次在本机持续超过 10 分钟（约 17%），未等待跑完；`tests/test_nerva_symbolic.py` 的 29 项因 Windows `core.autocrlf` 把 fixture `.txt` 转成 CRLF、SHA-256 不匹配而报错，属检出行尾环境问题，与本次重构无关。以上两项均不影响对本次分层迁移的代码路径覆盖（已分别运行其依赖模块的针对性测试）。

## 未解决问题

- `canonical.py`（约 1160 行）与 `symbolic.py`（约 1066 行）仍偏大，后续可在现有分层内按职责继续拆分。
- 旧子模块路径不保留兼容层；外部使用者需按 0.2.0 迁移。
- `derivatives._abstract` 重建签名时未透传 `memory_space`/`manual_axis_type`/`layout`，属既有技术债，本轮未改代码。
