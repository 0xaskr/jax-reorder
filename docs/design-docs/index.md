# 设计文档索引

本目录维护算法、接口契约、约束、方案取舍和验证方法。原开发文档已按内容归并：
设计留在这里，任务过程、实验和阶段验收进入[执行记录](../exec-plans/index.md)。
使用方法见 [README](../../README.md)，支持状态、未完成能力和主动排除项见[贡献者指南](../../CONTRIBUTING.md)。

根目录只保留本索引，正文按状态存放：

- [current/](current/)：当前生效的设计、契约与验证方法，Pallas 专题放在其 `pallas/` 子目录。
- [archive/](archive/)：已被替代的方案与历史设计，替代范围和当前入口见下表。

状态于 2026-10-26 整理：**当前生效**表示仍维护的设计依据，不表示全部目标已经实现或没有缺陷；
**已被替代**表示旧接口、算法或阶段边界不能继续指导实现；**历史参考**保留当时事实或决策理由。
历史归档不等于实验全部成功，也不等于其中每条原则都已失效。

## 当前生效的设计与契约

| 文档 | 生效内容与边界 |
| --- | --- |
| [核心工程原则](current/core-beliefs.md) | 语义保留、证据与验证、明确边界等长期约束 |
| [代码导读与公共入口](current/code-map.md) | 当前 API 选择、模块职责与调用链；实现缺陷链接待处理清单 |
| [完整图与深层区域](current/complete-graph.md) | 无根计算、对称接口、有界搜索及迭代式处理 |
| [区域接口与循环捕获](current/region-bindings.md) | 实参/形参、返回端口和循环反馈同步置换 |
| [原图对应证据](current/original-correspondence.md) | 图同构论证、独立复核与共同信任基础 |
| [函数与一阶求导契约](current/function-and-derivative-contract.md) | pytree、签名、环境、forward/JVP/VJP 及异常通道 |
| [可信 SymPy 代数比较](current/algebraic-equivalence.md) | 当前数学后端、定义条件与结果接口；普通 JAX 纯死计算删减仍是待实现决定，整数缺陷仍待修复 |
| [类型与异常分类](current/type-annotations.md) | 核心类型边界、失败分类及持续类型检查 |
| [系统验证矩阵](current/verification-matrix.md) | 规则与测试映射、CI 分组和 JAX 版本升级审核方法 |
| [Pallas 设计导读](current/pallas-jax-equivalence.md) | 完整图、设备局部、资源与数学路径的范围及阅读路线 |
| [Pallas 契约/API](current/pallas/contracts.md) | 调用接口、轴/实参来源、局部结论与执行前提 |
| [Pallas grid 与内存](current/pallas/grid-memory.md) | 地址、存储版本、窗口、初始化、mask 和 DMA 规则 |
| [Pallas 通信与执行绑定](current/pallas/communication.md) | 必要依赖、计数、进展及 core 绑定义务 |
| [Pallas 当前边界与验证规则](current/pallas/validation.md) | 未完成能力、主动排除项、已知缺陷及证据分层；历史批次另归档 |

[ARCHITECTURE.md](../../ARCHITECTURE.md) 仍是总体架构入口，其中旧结果字段与接口说明尚待同步；
具体当前接口以本表的专题和源码为准，待办见 [DOC-002](../exec-plans/index.md#待处理)。

自动生成的[CPU lax 原语清单](../references/lax-coverage.md)单独维护在参考目录，
记录测试绑定关系；它不属于设计方案，也不代替实际执行证据。

## 已被替代的设计

这些文档明确过期于所列范围，保留原方案供追溯；独立的实施与验收章节已分流到执行归档。

| 旧设计 | 已过期的内容 | 当前替代入口 |
| --- | --- | --- |
| [首轮 BFS 规范化](archive/bfs-reordering.md) | 输出可达子图为主、无根图部分完成、旧 `unknown` 结果；BFS 与语义保留原则仍延续 | [完整图](current/complete-graph.md)、[原图对应](current/original-correspondence.md)、[函数契约](current/function-and-derivative-contract.md) |
| [首轮 cond 绑定](archive/cond-bindings.md) | 未使用槽位的拒绝范围、固定私有结果端口与旧状态接口 | [区域绑定](current/region-bindings.md)、[完整图](current/complete-graph.md) |
| [旧代数证明器与 CAS 候选设计](archive/algebraic-equivalence.md) | 稀疏项表引擎、独立代数证书、CAS 仅生成候选及旧公开证明 API | [可信 SymPy](current/algebraic-equivalence.md) |
| [旧索引标量 IR](archive/algebraic-reduction-gaps.md) | 已删除的旧引擎、证明格式及其实现分层；索引与定义条件约束仍有参考价值 | [可信 SymPy](current/algebraic-equivalence.md) |

## 历史设计与替代关系

以下是 Pallas 早期需求与方案依据，部分原则延续；不把旧文的“当前/尚未支持”作为现状。

| 历史文档 | 状态与替代关系 |
| --- | --- |
| [范围修订与原始摘要](archive/pallas/design-00-context.md) | 局部范围、资源/数学分层等决定延续；“尚未接入公共入口”等阶段状态已被[当前导读](current/pallas-jax-equivalence.md)替代 |
| [早期要求](archive/pallas/design-01.md) | 语义保留与浮点解释等要求延续；早期跨分块优先安排已调整，以[当前契约](current/pallas/contracts.md)为准 |
| [目标架构草案](archive/pallas/design-03.md) | 分层原则延续；“不新增 Pallas API”已被替代；全部消费者统一视图仍未完成，不能把目标架构当成现状 |
| [比较语义与可判定范围](archive/pallas/design-04.md) | 结论与拒绝原则延续；首批无 DMA/无 scratch 等范围已被[内存](current/pallas/grid-memory.md)与[通信](current/pallas/communication.md)规则扩展 |
| [通信与进展方案](archive/pallas/design-05.md) | 因果、初始化、计数及进展约束延续；早期协议准入以[当前通信设计](current/pallas/communication.md)替代 |
| [原语与参数审核方案](archive/pallas/design-06.md) | 完整字段和分层能力审核原则延续；原字段清单只代表当时版本，不是当前全覆盖声明 |
| [验证方案](archive/pallas/validation-design.md) | 证据层级和隔离原则延续；测试设施、预算及执行方式以[当前验证规则](current/pallas/validation.md)为准 |

## 实验与执行记录

| 归档入口 | 内容与状态 |
| --- | --- |
| [结构/代数历史验收](../exec-plans/completed/history/index.md) | BFS、cond 与代数后端的实施、失败、修复、规模实验和验收数字 |
| [模型与通信验证](../exec-plans/completed/validation/index.md) | 固定业务/模型用例、运行方式和阶段结果；测试资产没有因归档而废弃 |
| [数学后端研究](../exec-plans/completed/research/index.md) | CAS、重写、SMT、编译器与组件复用的选型/探针；研究参考不等于当前已接入功能 |
| [Pallas 实施与调查历史](../exec-plans/completed/pallas-history/index.md) | 27 份阶段记录、初期调查与旧计划、锁定后端观察和验收快照 |

当前任务和剩余问题从[执行计划索引](../exec-plans/index.md)与[待处理清单](../exec-plans/index.md#待处理)进入。

## 维护方式

- 新设计正文写入 `current/`，根目录只保留 `index.md`。设计写明问题、输入输出契约、约束、方案、理由、反例与验证方法；执行计划链接设计，不重复维护正文。
- 实施过程、实验环境、实际命令及结果写入 `exec-plans/`；已有历史附件冻结，只修复引用或补充明确的状态说明。
- 新设计和替代关系同步更新本页。设计变更不自动表示代码完成，验证数字只归属实际运行的源码与批次。
- 废弃方案移入 `archive/` 并给出替代入口；不能仅以文件日期判断整篇过期。
