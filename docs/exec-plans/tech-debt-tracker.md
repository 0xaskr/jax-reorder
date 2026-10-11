# 技术债与已知缺口

优先级使用 P0（阻塞）、P1（当前迭代）、P2（后续）；状态使用待处理、进行中、已解决。每条记录附实际依据，并在解决后补充验证或提交链接。

| 编号 | 范围 | 问题 | 优先级 | 状态 | 依据 |
| --- | --- | --- | --- | --- | --- |
| IR-001 | Oryx 接入 | 已测依赖组合不兼容 JAX 0.11.2；模式匹配缺少闭包值、共享关系及 effects 的完整检查，需要当前 JAX 适配设计 | P1 | 待处理 | 首次评估文档当前缺失，待补复现记录 |
| AI-001 | 上游技能登记 | 原登记表的 3 个技能文件在源仓库中缺失 | P2 | 待处理 | 原仓库 AGENTS.md 登记表 |
| AI-002 | 外部工具 | Falcon、Plane 与 TPU/GPU 未在本项目配置或验证 | P2 | 待处理 | 当前仓库没有对应工作流模板或运行记录 |

## BFS 第一轮之后的工作

| 编号 | 问题 | 优先级 | 状态 | 依据 |
| --- | --- | --- | --- | --- |
| BFS-001 | 未参与输出的纯子图、常量及对称结构已加入完整图细化与精确搜索；预算不足仍明确 unknown | P2 | 已解决 | [完整图设计及测试](../development-docs/complete-graph.md) |
| BFS-002 | 已增加函数级 pytree、静态签名、环境及分模式比较；纯函数与未声明 Python 行为仍由调用方约定 | P2 | 已解决 | [函数契约](../development-docs/function-and-derivative-contract.md) |
| BFS-003 | 升级时仍需审核私有 IR 与 reduce callable 例外；现已固定字段快照、版本拒绝及独立候选检查 | P2 | 进行中 | [升级审核](../development-docs/verification-matrix.md#jax-升级审核) |
| BFS-004 | 深层迭代与共享区域复用已完成；持久化跨进程格式、代数规则仍未实现，且不在本轮契约内 | P2 | 待处理 | [深度证据](../development-docs/complete-graph.md)、[架构](../../ARCHITECTURE.md) |
| BFS-005 | cond 未使用槽位、scan/while 捕获与反馈、jit/shard_map 私有输入输出已联合规范化 | P1 | 已解决 | [区域绑定](../development-docs/region-bindings.md)、[系统验证计划](completed/graph-equivalence-validation.md) |
| BFS-006 | custom JVP/VJP 注册一阶规则、残差、透传、symbolic zeros 已适配；剩余边界见 AD-001/002 | P1 | 已解决 | [求导契约与测试](../development-docs/function-and-derivative-contract.md) |

## 系统验证后保留的边界

| 编号 | 问题 | 优先级 | 状态 | 依据 |
| --- | --- | --- | --- | --- |
| AD-001 | 仅存在于求导规则、没有对应原调用实参的动态捕获，底层注册规则物化仍 unknown；函数入口显式变换可另行比较 | P2 | 待处理 | [规则绑定说明](../development-docs/function-and-derivative-contract.md)、[回归](../../tests/test_custom_derivatives.py) |
| AD-002 | 延迟求导规则新增有序操作、反向日志及未解释 effects 需专用适配；当前明确拒绝，不能从普通父图推断求导顺序 | P2 | 待处理 | [规则边界](../development-docs/function-and-derivative-contract.md) |
| AD-003 | 高阶求导及任意变换组合没有整体保证；前向等价也可能因 add_any 顺序改变而 VJP 图不同 | P2 | 待处理 | [反馈用例](../../tests/test_feedback_bindings.py)、[完整层求导](../development-docs/full-attention.md#求导与未使用索引器) |
| LAYER-001 | Pallas 内核语义未解释：GLM5 absorbed/Decoder、Wan 交叉注意力 CPU lowering 拒绝；EPMoE CPU interpret 可执行但比较仍 unknown | P2 | 待处理 | [实际后端边界](../development-docs/full-attention.md#已确认的拒绝与未支持) |
| LAYER-002 | 原 FluxSingleTransformerBlock Explicit sharding 与 GLM5 单 head 配置被锁定 JAX 追踪拒绝；未篡改原计算绕过 | P2 | 待处理 | [原始拒绝回归](../../tests/test_full_attention.py) |
| LAYER-003 | GLM5 稀疏 DSA、完整 MoE Decoder、其他配置和多主机/GPU/TPU 未验证 | P2 | 待处理 | [完整层范围](../development-docs/full-attention.md) |
| DOC-001 | `.agents/README.md` 仍有 scaffold 旧描述；当前工作区该目录只读，本轮以 AGENTS/架构/开发文档为当前事实来源 | P2 | 待处理 | [项目入口](../../AGENTS.md)、[当前架构](../../ARCHITECTURE.md) |
