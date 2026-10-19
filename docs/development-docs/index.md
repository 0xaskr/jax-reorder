# 开发文档索引

规范化算法、比较实现和辅助工具的具体设计、实验与验证放在本目录。当前已实现完整图比较及函数/一阶求导契约；以 [比较契约](../../ARCHITECTURE.md) 为起点。BFS 第一轮记录保留历史范围，当前状态见系统验证矩阵。

| 主题 | 文档 |
| --- | --- |
| JAX / Pallas 等价比较：范围、共享语义层、接口与验证记录 | [pallas-jax-equivalence.md](pallas-jax-equivalence.md) |
| 核心库类型边界、异常枚举与持续类型检查 | [type-annotations.md](type-annotations.md) |
| BFS 计算图规范化、比较接口与验证 | [bfs-reordering.md](bfs-reordering.md) |
| 可信 SymPy 数学判等、定义条件与结果接口 | [algebraic-equivalence.md](algebraic-equivalence.md) |
| 开源数学等价后端的并行评估与符号探针 | [open-source-equivalence-backends.md](open-source-equivalence-backends.md) |
| 符号计算与代码等价研究概览及论文入口 | [2026-10-16 research](equivalence-backends/symbolic-code-equivalence-research-2026-10-16.md) |
| 可修改的 JAX 转换器、符号张量与 Nerva 成套公式资产 | [architecture-reuse.md](equivalence-backends/architecture-reuse.md) |
| 旧标量索引 IR 与证明器的历史设计 | [algebraic-reduction-gaps.md](algebraic-reduction-gaps.md) |
| DeepSeek-V4.1-Flash 原始 mHC 层的代数改写与验证 | [deepseek-v41-layer.md](deepseek-v41-layer.md) |
| GLM-5.2 纯 JAX decode DSA/MLA 核心的 jaxpr 对比与能力缺口 | [glm52-decode-dsa-mla.md](glm52-decode-dsa-mla.md) |
| cond 闭包捕获与实参/形参同步规范化 | [cond-bindings.md](cond-bindings.md) |
| 循环与调用区域的接口绑定规范化 | [region-bindings.md](region-bindings.md) |
| 原始图对应证据、独立校验与同构论证 | [original-correspondence.md](original-correspondence.md) |
| 函数契约、自定义求导及模式范围 | [function-and-derivative-contract.md](function-and-derivative-contract.md) |
| 完整图、对称接口与深度压力 | [complete-graph.md](complete-graph.md) |
| 完整 Attention/block、索引器、求导与原始后端边界 | [full-attention.md](full-attention.md) |
| 系统验证矩阵、CI 分组与版本升级检查 | [verification-matrix.md](verification-matrix.md) |
| CPU lax 逐项原语清单与验证绑定 | [lax-coverage.md](lax-coverage.md) |
| AI 组合判断、反例与模型提取来源 | [bfs-cases.md](bfs-cases.md) |
| 通信组合、参数反例与 CPU 多设备验证 | [collective-jaxpr.md](collective-jaxpr.md) |
| 真实层级场景、AI hardcode 改写与独立数值参照 | [business-layer-cases.md](business-layer-cases.md) |
| SGLang-JAX 全部模型的层级对抗验证 | [sglang-model-layers.md](sglang-model-layers.md) |

开发文档应覆盖：

- 问题与输入输出契约：签名、静态参数、常量、dtype、shape、pytree 与 effects。
- 支持范围和拒绝条件，结构模式与代数模式的区别。
- 表示与算法：依赖关系、确定性要求、复杂度、替代方案和已知局限。
- 验证：正例、反例、失败案例、环境版本、复现命令与结果。
- 证据与进展：假设、已确认结论、未解决问题及对应计划。

新增文档时在此添加入口；实现废弃时同步归档或删除文档，并更新调用方链接。
