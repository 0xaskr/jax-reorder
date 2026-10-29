# 历史验证记录索引

本目录保存固定案例、来源、实验方法与阶段验收。归档不表示对应测试资产废弃；文中的测试数量、
性能数据、接口和“当前缺口”只描述原记录时点，2026-10-26 的文档迁移没有重跑这些验证。
现行能力见[贡献者指南](../../../../CONTRIBUTING.md)，现行契约与算法见[设计索引](../../../design-docs/index.md)。

| 记录 | 状态与阅读范围 |
| --- | --- |
| [AI 组合案例与 LlamaMLP](bfs-cases.md) | 历史验证、固定测试资产说明；保留案例预判、来源与追踪原语，不外推当前全量覆盖 |
| [真实业务层固定改写](business-layer-cases.md) | 历史验收；五类业务计算、独立候选和数值参照，记录中的结构模式不代表项目没有代数比较 |
| [集合通信 jaxpr](collective-jaxpr.md) | 历史通信验收；参数与 mesh 保留原则仍有参考价值，正常结果 `unknown` 的接口口径已被分类异常替代 |
| [DeepSeek-V4.1-Flash mHC](deepseek-v41-layer.md) | 历史模型验证与 CPU 性能实验；旧独立数学证明器、校验器和索引实现细节已被可信 SymPy 路线替代 |
| [完整 Attention 与后端边界](full-attention.md) | 历史固定配置验收；保留完整层来源、求导与原后端拒绝证据，不把原图自比较当作候选认证 |
| [SGLang-JAX 模型层覆盖](sglang-model-layers.md) | 历史验收；36 个选定层及固定配置，后续完整层结果单列，不合并为当前测试总数 |
| [GLM-5.2 DSA/MLA 核心图](glm52-decode-dsa-mla.md) | 核心图测试纠偏与阶段验收；固定候选仍保留，正常结果 `unknown` 和旧证明器叙述已过期，剩余缺口查技术债 |
| [KDA backward mega 历史与 AI 候选](kda-backward-history-results.json) | 固定来源与源码哈希、53 组完整结构、106 次 CPU interpreter 执行及 AI 正反例；资源/数学停在原始 scratch 读取，详见 [执行计划](../kda-backward-history.md) |
| [KDA 初始化候选与资源诊断](kda-resource-diagnostics-results.json) | shape、占位输出及尾窗填充初始化候选，精确 DAG 共享、完整局部 AI 对比、数值与预算证据；不认证原始 base，当前边界见[后续计划](../../active/kda-resource-completion.md) |
| [KDA 原始代码的父输出比较](kda-parent-boundary-results.json) | 六配置资源自比较、默认 FP32 历史/AI，14 个唯一比较及一次阶段复测；8 个等价含 6 个自比较，6 个数学未完成另列，详见[当前验收与缺口](../../active/kda-resource-completion.md#原始代码的父输出验收) |

`unknown` 接口迁移见[分类异常完成记录](../comparison-exceptions.md)，数学后端替换见
[可信 SymPy 迁移](../trusted-sympy-backend.md)与[旧证明器删除](../remove-legacy-prover.md)。
未解决问题维护于[待处理清单](../../index.md#待处理)，不继续改写历史验收数字。
