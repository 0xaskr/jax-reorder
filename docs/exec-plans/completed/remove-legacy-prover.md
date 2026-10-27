# 删除旧代数证明器并整理 SymPy 比较路径

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

## 目标与完成条件

用户明确要求放弃旧证明器、删除相关代码并重构。生产包仅保留结构比较与可信 SymPy 数学比较；不再导出旧代数证书类型或验证 API，不再运行或通过测试 helper 保留旧数学引擎。

- 删除旧多项式规范化、证据生成/重放、独立标量规则及仅供旧 IR 使用的转换代码。
- 将新路径仍使用的元数据编码、有限性分析、预算和索引适配从旧 Lowerer/IR 中解耦。
- 结构快速路径直接返回 GraphCorrespondence；SymPy 查询结果继续使用 backend_result。
- 迁移有价值的公式正反例、绑定/定义域/资源回归；删除仅测试旧证书格式或旧搜索实现的测试。
- 全量测试、Ruff、示例、文档链接与差异检查通过。保留定义域与 unknown 的原有边界，不因删除旧代码而放宽判断。

## 设计与取舍

设计入口：[代数比较开发记录](../../design-docs/current/algebraic-equivalence.md)。此次是明确的 API 删除：AlgebraProof、PolynomialInequivalenceProof、verify_algebraic_proof、verify_inequivalence_proof 以及结果中的 proof/inequivalence_proof 不再提供兼容外壳。结构证据及 verify_correspondence 仍属于第一阶段。

已有 Finiteness 和 JIT 参数检查间接依赖 Lowerer，必须先提取实际使用的小范围语义工具，再删除旧转换器。索引坐标适配可以保留，但删除仅服务 Factor/Term 和证明步骤的代码。数学判断继续信任 SymPy；迁移回归发现的表示丢失与查询资源问题，在转换边界和查询顺序中修复，不增加独立数学证明器。

受保护文档不在本次授权范围内，不修改。历史完成计划保留为历史记录；当前接口、迁移映射和验收结果记录在开发文档及本计划。

## 工作项

- [x] 盘点生产依赖和旧测试，记录保留/迁移/删除范围。
- [x] 提取共享语义工具，删除旧引擎与公开证书接口，重构结果类型。
- [x] 迁移公式与拒绝路径测试，清除旧测试 helper 和专用证书测试。
- [x] 分组检查并修复回归，完成全量运行及受影响范围复测。
- [x] 审查差异、记录 API 变化与验证证据，归档计划。

## 决策与进展

- 开始时工作区干净。当前默认比较已使用 SymPy，但旧引擎仍在包内，测试仍生成旧证书。
- 已确认新路径间接使用 Lowerer 的 encode、非有限值检查，以及 Finiteness 所需的预算/规则/形状校验；这些依赖将逐项拆出。
- 迁移旧反例发现现有 SymPy 路径把 `x[:2]` 与 `x[1:]` 判成等价。复现表明 `convert_indexed_to_array` 将 `x[i+1]` 的整个坐标视作轴标签，转换后丢失偏移；这是本地误用转换 API，不能把其结果当成带偏移的完整张量。ArrayExpr 快路径限于符号轴和完整范围的纯收缩，切片/步长/翻转/坐标权重继续交给保留 Indexed 的 SymPy 查询，并增加正反例回归。

## 验证结果

- 首轮分组测试：358 passed，6 failed。两例是相同 log 定义域上的有效等价；一例是大指数求和化简次序；一例确认切片误判；其余是旧资源策略断言与新后端表示不同。均保留场景，按新契约修复或迁移。
- Ruff、格式和当前文档链接检查通过。最终源码的离线 sdist/wheel 构建通过；wheel 内不含 12 个已删除模块，新辅助模块及上游许可证完整。隔离导入验证结构/等价/不等价/unknown 四条路径，以及切片偏移反例，全部通过。
- 包未在普通环境安装 hatchling，首次 `--no-build-isolation` 构建失败；使用标准隔离构建及现有离线缓存成功，无依赖修改。
- 迁移重验 78 passed，唯一失败是新增坐标权重测试无意使用了整数 arange→float 转换；改为明确 float32 iota 后通过。转换边界文件最终 10 passed。
- 主数学入口文件 91 passed；大轴/确定性预算定向 5 passed。全量验收随后暴露 Piecewise 常量求和的化简顺序超时，已定位并修复。
- 源码复查补充自由轴完整性检查，防止行/列广播在 ArrayExpr 中丢失轴身份；相应反例已通过。完整套件初次运行途中更新此边界，已中止并重新启动，未将中止结果视作验收。
- 全量首次完整尝试在 269 passed 后发现 JIT 捕获常量用例超时：新增两侧化简对 Piecewise 生成过多分支。含 Piecewise 的表达式恢复先做有预算的有限展开，定向回归 1 passed；无分段的大指数求和仍使用两侧化简。
- 完整运行 1930 项：1928 passed、2 failed、1 个已有上游弃用 warning，1613.10 秒。两项失败为 Nerva 稳定 softmax/log-softmax 超时；独立复现及调用栈确认 Max 在两侧预化简中生成分段条件，refine 成本膨胀。将 Piecewise 排除范围扩展至会转成分段函数的 Max/Min/Abs，保留先消项/有限展开的原有路径。两项在原 15 秒限制下复测通过，总耗时 4.32 秒；未修改判等断言。
- 运行过程中检测到外部将 Nerva 公共入口测试超时从 15 秒改为 180 秒。保留这项并发修改；上面的 15 秒复测通过临时 pytest 插件覆盖参数，不改仓库文件。全量运行已加载原 15 秒测试，不将放宽超时计为修复。
- 最后一次查询调整后，受影响的 8 个数学测试文件 **299 passed，333.63 秒**。全量结果与修复后定向复测分开记录；最后源码未再重复运行整个套件。复测期间源码和测试文件哈希保持不变。
- 最终 Ruff、177 个文件的格式检查、tracing 与两个数学示例、文档链接、`git diff --check` 均通过。受保护文档与依赖清单无变动；仅保留已说明的 Nerva 超时并发修改。清理验收时尚未提交；后续用户已授权提交并推送 `origin/main`。

复现命令（只读缓存环境可设置 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache`）：

```bash
uv run pytest -q
uv run pytest -q tests/test_algebra.py tests/test_algebra_abs_where.py tests/test_algebra_array_conversion.py tests/test_algebra_jit_composition.py tests/test_algebra_negative_infinity.py tests/test_algebra_sympy.py tests/test_nerva_symbolic.py tests/test_trusted_sympy.py
uv run ruff check .
uv run ruff format --check .
uv run python tests/inspect_jaxpr.py
uv run python tests/sympy_equivalence.py
uv run python tests/algebraic_equivalence.py
uv run python tools/check_docs.py --changed
uv build --offline --out-dir /tmp/jax-reorder-without-legacy-verified
git diff --check
```

## 测试迁移清单

- 旧标量恒等式、完整输出不等价、源常量变化、资源边界分别迁到公共比较接口；保留反例、输出顺序、空域、整数/复数与部分函数拒绝路径。
- 有限性、cast sharding 和 padding 改为测试实际 freeze/SymbolicBuilder 输出；独立整数坐标解释器继续检查正负坐标及 div/mod。
- 原语清单的结构快速路径测试继续全量覆盖，并阻断数学 worker；结构对应篡改直接交给 verify_correspondence。
- 删除旧证书系数/见证篡改、旧规则重放、旧 normalizer 搜索内核及重复的旧 IR 展开清单。这些验证目标随旧实现删除。
- 新增切片偏移/步长/翻转/部分求和/坐标权重/广播轴反例、同映射正例和大矩阵结合 ArrayExpr 回归。

## 未解决问题

当前没有新增问题；已有不支持的数学条件与一般递归仍按现有设计返回 unknown。
