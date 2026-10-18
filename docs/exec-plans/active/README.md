# 进行中的计划

[核心库类型信息补全](../completed/type-annotations.md)已完成并归档：异常阶段与分类改用 `StrEnum`，补全核心库签名、回调与容器类型，接入持续检查。分阶段 1957 项通过，1 项既有超时在修改前源码也可复现；详见验证记录。

[代码架构分层重构](../completed/architecture-refactor.md)已完成并归档：`jax_reorder` 从 20 个平铺模块拆为 `core/`、`algebra/`（含 `adapter/`）与 `functions.py` 三层，拆解 symbolic/regions/trusted 导入环并新增分层守卫测试。顶层公共 API 不变，子模块路径按 0.2.0 破坏性调整；核心分组 465 passed，详见归档记录。

[三种比较结论与分类异常迁移](../completed/comparison-exceptions.md)已完成并归档：删除 UNKNOWN，统一结构、SymPy worker、函数模式和批处理的异常通道；首轮全量发现的 34 项迁移失败均已修复，最终 110 项复测通过。后续用户已授权提交并推送 `origin/main`。

[将 jax2sympy 转换代码并入项目](../completed/inline-symbolic-converter.md)已完成并归档：生产逻辑直接并入构造器，完整 60 个 primitive 映射与逐元素转换入口迁入普通模块，删除 vendor 包，许可证保留在文件头。相关回归 **741 passed**，另有 46 项转换/后端检查通过；尚未提交或推送。

[删除旧代数证明器并整理 SymPy 比较路径](../completed/remove-legacy-prover.md)已完成并归档：移除旧证书接口、12 个旧模块及专用测试，解耦转换辅助代码，公式回归迁至可信 SymPy 路径。全量 1930 项中发现的两项超时已修复，受影响范围复测 **299 passed**；详见归档中的分阶段验证记录。后续用户已授权提交并推送 `origin/main`。

[可信 SymPy 后端：开源组件复用与架构迁移](../completed/trusted-sympy-backend.md)已完成并归档：本地 jax2sympy fork、默认可信 SymPy 数学链路、Nerva 两层测试与可终止 worker 均已接入。全量 **2166 passed**，1 个已有上游弃用警告；按后续用户授权提交到 `origin/main`。已无进行中的迁移计划。

[数学等价后端接入、完整验证与 main 发布](../completed/symbolic-math-integration.md)已完成：SymPy 候选与独立复核、完整多项式输出的不等价证据、guarded-softmax/PV 规则已接入。全量 2095 passed；GLM 三组签名均为 5 个正例已证明和 5 个负例 unknown。按用户最新授权直接推送 main。

[开源数学等价后端的并行评估](../completed/open-source-equivalence-backends.md)已完成：三个 subagent 与主代理覆盖 21 个候选/组件，运行符号探针与真实 GLM QK/PV helper 的 jaxpr→SymPy 转换。该评估阶段的生产适配工作已在上述接入计划推进；当前保留边界见 ALG-004。

[`-inf` mask 的受限代数支持](../completed/negative-infinity-masks.md)已完成：保留特殊值及其依赖语义，阻止非有限路径进入实数乘加展开；新增 41 项回归，全量 1952 passed。该阶段 GLM 图比较为 4 个已证明正例和 6 个 unknown；最新结果与剩余边界见上述接入记录及 ALG-003。

[GLM-5.2 核心层的 jaxpr 健壮性测试纠偏](../completed/glm52-jaxpr-robustness.md)保存初始结果：仅抽象追踪核心的 5+5 改写，当时为 2 个结构正例与 8 个 unknown；后续 `-inf` 支持与最新结果见上述完成计划。

[GLM-5.2 纯 JAX decode DSA/MLA 初版](../completed/glm52-decode-dsa-mla.md)保存历史结果；其外围改写和 CPU 数值验收偏离目标，已按上述计划改为核心层的 jaxpr 对比。

[DeepSeek-V4.1-Flash 真实 JAX 层的代数改写验证](../completed/deepseek-v41-layer.md)已完成并归档：固定原始 mHC 层、两种 AI 代数改写、前向/JVP/VJP 证明、独立数值参照与 CPU 计时；新增 68 项回归，1870 个用例均完成通过验证。

[修复仓库审查确认的求导与内部绑定问题](../completed/review-correctness-fixes.md)已完成并归档：修复 VJP 整数输出规则遗漏和无根私有输入换序，新增 27 项回归，全量 1802 项通过。

[标量索引 IR 重构](../completed/algebraic-reduction-gaps.md)已完成并归档：四组需求及通用域规则接入，全量 1775 项通过。

[JAX 函数代数等价比较](../completed/algebraic-equivalence.md)已完成并归档：全量 1600 项通过，受保护文档按用户要求保持原状。

[计算图等价比较的完善与系统验证](../completed/graph-equivalence-validation.md)已完成并归档。

新计划按 [计划规范](../../PLANS.md)编写；完成记录见 [completed](../completed/)。
