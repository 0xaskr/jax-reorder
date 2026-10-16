# 三种比较结论与分类异常迁移

状态：2026-10-16 完成。代码、调用方、异常回归和文档已迁移；后续用户已授权提交并推送 `origin/main`。

## 目标与完成条件

落实[结果契约](../../references/equivalence-results.md)：完整比较只返回结构等价、数学等价、数学不等价；删除 UNKNOWN，未完成比较直接抛出可定位的分类异常。保留数学判断和资源边界，不把失败改成负结论。

## 设计与取舍

- `compare_jaxprs` 和显式 `equivalence="structural"` 是结构阶段诊断，可返回 `different`；默认函数比较和 `compare_algebraic_jaxprs` 完成两阶段流程，不能返回 `different`。
- 公共异常携带 stage、path、reason、failure_kind；已有非法图、不支持和对应校验异常接入统一基类。数学未决、契约、预算、超时、追踪/编译和内部错误分别分类。
- worker 使用独立失败消息，父进程还原异常分类及远端 traceback；成功结果不包含失败状态。保留查询、预算和时间诊断。
- 函数请求中任何模式失败即抛错，不再以某模式的不等价掩盖其他模式失败。批处理可以捕获异常，但错误记录与正常结果分开。
- 不修改受保护文档；现有参考文档及开发记录记录当前实现。不开新依赖、不自动提交或推送。

## 工作项

- [x] 迁移异常类型、结构比较及资源边界。
- [x] 迁移数学后端、进程通信和函数接口。
- [x] 迁移示例/批处理以及既有失败回归，补充公共契约和故障注入测试。
- [x] 完成 pytest、Ruff、示例、文档和差异检查，归档计划。

## 决策与进展

- 2026-10-16：确认现有 UNKNOWN 覆盖结构校验、数学转换、条件/CAS 未决、worker 故障、追踪/编译和多模式汇总，需跨层修改。

- 比较错误通过 `ComparisonError` 及子类传递；调用选项格式错误继续抛 `ValueError`。两者都不生成失败结果。
- 删除 `ModeComparison` / `SympyResult` 的失败字段，worker 改为独立 `WorkerFailure` 消息；父进程还原错误类型和远端 traceback。
- 普通图搜索和注册求导规则枚举的预算统一使用资源异常，避免按字符串或“不支持”推断失败类别。
- 反例断言收紧后发现旧 unknown 隐藏的真实 bug：布尔 Piecewise 被当作算术 Expr 相减。改为仅对布尔分支调用 SymPy `rewrite(ITE)`；数值 Piecewise 保留原查询方式，新增双向与数值反例回归。
- `max` 分配式及含负无穷的捕获数组负例可能耗尽默认 10 秒。保留该边界，测试只接受明确数学未决或超时，不接受内部错误；没有通过扩大预算或强行判不等修饰结果。

## 验证结果

- 结构、函数和新异常契约首轮专项 188 项：187 通过，1 项故障注入误选了已被 SymPy 自动合并的相同公式；改用不同公式后进入预期未决路径。
- 数学/区域专项首轮 172 项：161 通过，11 个旧“非等价”断言需按实际条件未决、契约冲突或转换缺口迁为明确异常。未修改数学算法来改变结果。
- 迁移中的全量命令 `uv run --no-sync --offline pytest -q --tb=short`：**1895 passed / 34 failed / 1 warning，1582.76 秒**。运行期间后续完成的测试迁移不影响已加载测试；34 个失败均已完成迁移并纳入最后复测。
- 修正后的结构证据、求导、区域、worker 与可信后端专项：**245 passed，164.57 秒**。
- 收紧数学拒绝断言后的 8 文件回归：**352 passed / 4 failed，445.85 秒**。四项分别是两种实际超时、未决大 Sum 的异常类型断言，以及上述布尔 Piecewise 错误。修正后前述三项 **3 passed**，布尔/部分定义域相关 **12 passed**。
- 最后命令 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache python3 /tmp/rerun_comparison_failures.py` 组装全部首轮失败节点（按最新测试名称）、完整 `test_trusted_sympy.py`、`test_comparison_errors.py`，以及捕获数组、部分函数、GLM CLI 和全部 DeepSeek 负例：**110 passed，191.95 秒**。全部 34 个首轮失败均包含在内，包含完整文件时不重复选择其节点。
- `uv run --no-sync --offline pytest --collect-only -q`：当前 **1950 tests collected**，比迁移时首轮全量新增 21 项异常/逻辑条件回归。完成后没有再重复整套全量；以上记录区分全量与分组复测，计数不可相加。
- `uv run --no-sync --offline ruff check .`、`ruff format --check .`、`python tests/adapter_inventory.py`、`python examples/inspect_jaxpr.py`、`python examples/bfs_reordering.py`、`python examples/sympy_equivalence.py`、`python examples/check_correctness.py --trials 1`、`python tools/check_docs.py --changed` 和 `git diff --check` 通过。
- 命令均使用 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache`。首轮与复测日志、复测节点清单及组装脚本在 `/tmp/jax-reorder-exceptions-*` 和 `/tmp/rerun_comparison_failures.py`，属于本地临时证据，不纳入仓库。唯一 warning 是原有上游 Pallas 弃用提示。

## 未解决问题

受保护架构文档仍描述旧接口，本轮遵守其文件修改授权边界；当前契约以结果参考及开发记录为准。
