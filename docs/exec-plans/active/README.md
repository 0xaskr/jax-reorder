# 进行中的计划

当前进行中的执行计划：

- [数学等价后端接入、完整验证与 main 发布](symbolic-math-integration.md)：实现和全量 2095 项验证完成，按用户最新授权直接提交并推送 main。

[开源数学等价后端的并行评估](../completed/open-source-equivalence-backends.md)已完成：三个 subagent 与主代理覆盖 21 个候选/组件，运行符号探针与真实 GLM QK/PV helper 的 jaxpr→SymPy 转换。完整 layer、生产适配和独立证据复核仍待完成，见 ALG-004。

[`-inf` mask 的受限代数支持](../completed/negative-infinity-masks.md)已完成：保留特殊值及其依赖语义，阻止非有限路径进入实数乘加展开；新增 41 项回归，全量 1952 passed。GLM 图比较现为 4 个已证明正例和 6 个 unknown，剩余边界见 ALG-003。

[GLM-5.2 核心层的 jaxpr 健壮性测试纠偏](../completed/glm52-jaxpr-robustness.md)保存初始结果：仅抽象追踪核心的 5+5 改写，当时为 2 个结构正例与 8 个 unknown；后续 `-inf` 支持与最新结果见上述完成计划。

[GLM-5.2 纯 JAX decode DSA/MLA 初版](../completed/glm52-decode-dsa-mla.md)保存历史结果；其外围改写和 CPU 数值验收偏离目标，已按上述计划改为核心层的 jaxpr 对比。

[DeepSeek-V4.1-Flash 真实 JAX 层的代数改写验证](../completed/deepseek-v41-layer.md)已完成并归档：固定原始 mHC 层、两种 AI 代数改写、前向/JVP/VJP 证明、独立数值参照与 CPU 计时；新增 68 项回归，1870 个用例均完成通过验证。

[修复仓库审查确认的求导与内部绑定问题](../completed/review-correctness-fixes.md)已完成并归档：修复 VJP 整数输出规则遗漏和无根私有输入换序，新增 27 项回归，全量 1802 项通过。

[标量索引 IR 重构](../completed/algebraic-reduction-gaps.md)已完成并归档：四组需求及通用域规则接入，全量 1775 项通过。

[JAX 函数代数等价比较](../completed/algebraic-equivalence.md)已完成并归档：全量 1600 项通过，受保护文档按用户要求保持原状。

[计算图等价比较的完善与系统验证](../completed/graph-equivalence-validation.md)已完成并归档。

新计划按 [计划规范](../../PLANS.md)编写；完成记录见 [completed](../completed/)。
