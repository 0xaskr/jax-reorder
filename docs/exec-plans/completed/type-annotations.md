# 核心库类型信息补全

## 目标与完成条件

补全 `src/jax_reorder` 中异常字段、函数签名、回调和容器元素的类型，避免类型信息从未标注的入口传播丢失。保留运行行为、比较语义和已有用户改动。为核心库增加持续类型检查，完成 Ruff、tracing 示例与回归验证。

## 设计与取舍

- 按用户补充要求，阶段与失败分类使用 `StrEnum`，类默认值使用对应的 `ClassVar`；异常重定位返回 `Self`，诊断载荷允许异构值。
- 图、坐标、预算、结果、回调使用具体类型；JAX 的异构 primitive 参数和任意 Python pytree 保留明确的动态边界。
- 注解按现有单向分层组织；仅为类型检查需要的循环引用使用 `TYPE_CHECKING`。
- 本次覆盖生产库；测试中的刻意非法输入、上游快照与实验样例不做批量类型迁移。
- 受保护文档不修改；任务记录通过开发文档索引与 active 目录入口维护。

## 工作项

- [x] 读取权限配置、比较契约，盘点现有类型缺口及用户改动。
- [x] 补全异常、结构层、代数层和函数入口的类型信息。
- [x] 加入持续检查，核对注解与真实调用方式一致。
- [x] 运行验证、审查差异并归档计划。

## 决策与进展

- 初始工作区仅 `errors.py` 有用户新增的 `# Base Class` 注释，必须保留。
- 仓库未配置类型检查器；多数内部函数缺少完整签名，一些数据类字段只有裸容器类型。
- 完成全部 224 个函数/方法（含嵌套回调与协议方法）及 36 个数组回调的签名标注。修正生成器、可空返回值及异构容器的声明；未改比较规则。
- 新增 `basedpyright` 开发依赖及 CI 检查、Ruff ANN 规则和调用方 `assert_type` 检查；验证已有 `py.typed` 标记随 wheel 发布。设计与边界见[开发记录](../../design-docs/current/type-annotations.md)。
- 类型检查不推测未声明类型契约的 SymPy 源码，保留原语参数、pytree 和精确编码载荷的动态边界。
- 运行时导入图检查原先把 `TYPE_CHECKING` 分支也视为执行依赖；调整为仅在环检查时排除该分支，分层检查仍覆盖全部导入。
- 检查器在 WSL 挂载目录安装时发生 rename 错误，改用 `/tmp/jax-reorder-types-venv` 的锁定依赖环境；静态检查临时配置仅替换环境路径，规则与仓库配置一致。随后使用离线 copy 模式成功恢复本地 `.venv`；直接运行 `uv run --no-sync basedpyright` 也通过。
- 工作过程中出现用户暂存、注释调整及新增 `.gitattributes`，保留这些改动。类型迁移完成后，用户明确授权将全部改动提交并推送至 `origin/main`。
- 首轮全量遇到上游快照校验失败：Windows 检出把固定快照转成 CRLF。逐个核对清单 SHA-256 与 `HEAD` 原始字节后，将 148 个快照文件恢复为 LF；恢复后 `git diff HEAD -- tests/fixtures` 为空。未放宽哈希检查或修改上游内容。

## 验证结果

- Ruff 检查与格式检查通过。
- basedpyright：30 个文件，0 errors、0 warnings。
- 签名 AST 盘点：224 个函数/方法，缺失参数或返回类型为 0。
- 异常、包分层与整数索引最终定向回归：27 passed，11.14 秒。
- tracing 示例通过。
- wheel 构建通过，检查包含 `jax_reorder/py.typed` 及全部 29 个 Python 模块。
- 首轮全量 pytest：1648 passed、245 failed、65 errors，耗时 1232.55 秒。308 个失败/错误源于快照换行导致的哈希不符；另外两项达到 worker 的 10 秒超时。
- 恢复快照后 `pytest -q --lf` 串行复测全部 310 项：309 passed、1 failed、1 个上游 Pallas 弃用警告，耗时 1025.33 秒。JIT 捕获常量用例通过；所有快照相关用例通过。
- 剩余失败为 `tests/test_algebra_negative_infinity.py::test_captured_array_keeps_finite_elements_and_negative_infinity_exact`，仍触发默认 10 秒 worker 超时。通过 `git archive HEAD src/jax_reorder` 提取修改前源码到临时目录，以 `PYTHONPATH` 指向该源码并复测同一节点，也在 11.51 秒后报告相同超时；调用栈确认加载基线目录。
- 因此共 1957 个不同用例分阶段通过，1 个既有超时仍可复现；不宣称单次全量全绿。默认超时和比较算法未为验收而调整。
- `uv lock --check --offline`、文档链接检查与 `git diff HEAD --check` 通过；受保护文档未修改。

最终检查使用锁定依赖。Ruff、basedpyright 和 tracing 示例直接通过本地 `uv run --no-sync` 运行；pytest 使用同一锁定依赖的 `/tmp/jax-reorder-types-venv` 环境。日志保存在 `/tmp/jax-reorder-type-*.log`，不作为仓库产物提交。

## 未解决问题

类型迁移已完成。既有负无穷捕获常量用例在本机超过默认 worker 时限，已补充到 [ALG-005](../index.md#待处理)，需要单独的性能分析与算法验证。
