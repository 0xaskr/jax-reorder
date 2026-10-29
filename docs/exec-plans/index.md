# 执行计划索引

本页汇总正在进行的计划、已完成阶段与历史记录。状态随对应计划更新；
详细目标、决定和验证结果保留在各计划中。

## 正在进行

| 计划 | 当前重点 |
| --- | --- |
| [KDA 完整资源与数学比较后续](active/kda-resource-completion.md) | 已获授权并实现父输出边界、未消费读取及本地填充传播；原始六配置资源自比较通过，继续跟踪大公式耗时、一般精度对应和历史编译策略 |
| [JAX / Pallas 语义层](active/pallas-semantic-layer.md) | 优先修复整数归约/点积的数学语义；继续扩展数据条件、动态地址、并行共享 staging、索引/路由/终止及布局证明 |

尚未解决事项统一维护在下方[待处理清单](#待处理)，具体复现和修复要求放入对应计划。
具体支持状态及范围取舍见[贡献者指南](../../CONTRIBUTING.md)。

## 待处理

本节维护尚未解决事项的编号、优先级、状态和依据；具体支持范围与主动排除项见贡献者指南。
P0 表示阻塞，P1 表示当前迭代，P2 表示后续。启动或完成任务时更新本节与相应计划，避免另设重复总表。

| 编号 | 问题 | 优先级 | 状态 | 依据 |
| --- | --- | --- | --- | --- |
| BFS-003 | 升级时仍需审核私有 IR 与 reduce callable 例外；现已固定字段快照、版本拒绝及独立候选检查 | P2 | 进行中 | [升级审核](../design-docs/current/verification-matrix.md#jax-升级审核) |
| BFS-004 | 深层迭代与共享区域复用已完成；持久化跨进程格式仍未实现。索引代数规则现已另行实现，见 ALG-001/002 的后续边界 | P2 | 待处理 | [深度证据](../design-docs/current/complete-graph.md)、[代数设计](../design-docs/current/algebraic-equivalence.md) |
| AD-001 | 仅存在于求导规则、没有对应原调用实参的动态捕获，底层注册规则物化仍抛出不支持异常；函数入口显式变换可另行比较 | P2 | 待处理 | [规则绑定说明](../design-docs/current/function-and-derivative-contract.md)、[回归](../../tests/test_custom_derivatives.py) |
| AD-002 | 延迟求导规则新增有序操作、反向日志及未解释 effects 需专用适配；当前明确拒绝，不能从普通父图推断求导顺序 | P2 | 待处理 | [规则边界](../design-docs/current/function-and-derivative-contract.md) |
| AD-003 | 高阶求导及任意变换组合没有整体保证；add_any 累加换序的 VJP 图差异可由默认代数模式单独证明，区域接口改变仍可能无法完成比较并抛出异常 | P2 | 待处理 | [反馈用例](../../tests/test_feedback_bindings.py)、[代数边界](../design-docs/current/algebraic-equivalence.md#当前边界) |
| PALLAS-001 | 已支持受限远端条件初始化、有界动态页 DMA 和设备局部数学比较；仍缺更一般的数据谓词关系、并行共享 staging、索引/路由/终止及物理布局证明。执行绑定、预算与源后端限制分别列明；core_map/mpmd_map、输入输出共享 HBM 暂不支持 | P1 | 进行中 | [准确范围](../design-docs/current/pallas/validation.md)、[当前计划](active/pallas-semantic-layer.md) |
| PALLAS-002 | 第一版之外的全局扩展：一般多轴 mesh 跨设备值组装、动态路由/通信协议及多主机验证。有限单轴协议与多维 kernel grid 已有专用规则，不构成一般全局证明；这些缺口不阻塞设备局部图比较 | P2 | 待处理 | [通信范围](../design-docs/current/pallas/communication.md)、[通信验收](completed/communication-protocols.md) |
| PALLAS-006 | KDA 7 月历史版的 collective_id/custom barrier 编译策略尚未审核；CP inactive 不能直接作为忽略这些参数的理由 | P2 | 待处理 | [原始代码验收](active/kda-resource-completion.md#新的后续问题) |
| LAYER-002 | 原 FluxSingleTransformerBlock Explicit sharding 与 GLM5 单 head 配置被锁定 JAX 追踪拒绝；未篡改原计算绕过 | P2 | 待处理 | [原始拒绝回归](../../tests/test_full_attention.py) |
| LAYER-003 | GLM5 稀疏 DSA、完整 MoE Decoder、其他配置和多主机/GPU/TPU 未验证 | P2 | 待处理 | [完整层范围](completed/validation/full-attention.md) |
| DOC-001 | `.agents/README.md` 仍有 scaffold 旧描述；当前工作区该目录只读，本轮以 AGENTS/架构/开发文档为当前事实来源 | P2 | 待处理 | [项目入口](../../AGENTS.md)、[当前架构](../../ARCHITECTURE.md) |
| DOC-002 | 代数功能与公共 Pallas 结构入口已实现，受保护的 ARCHITECTURE、AGENTS 和 docs/PLANS 部分说明及完成索引尚未同步，待后续具体授权；当前支持状态集中维护于贡献者指南和开发文档 | P2 | 待处理 | [贡献者指南](../../CONTRIBUTING.md)、[代数设计](../design-docs/current/algebraic-equivalence.md)、[Pallas 验收归档](completed/pallas-structural-comparison.md) |
| ALG-001 | 符号 reshape/静态切片、连续域合并已接入；动态索引、对角读取、任意步长覆盖、额外嵌套原语及特殊精度/分片仍待适配 | P2 | 待处理 | [代数边界与原语回归](../design-docs/current/algebraic-equivalence.md#当前边界)、[结构蕴含测试](../../tests/test_algebra_structural.py) |
| ALG-002 | 独立 scan 累加、已审核选择包装和严格默认纯 JIT 的残差组合已适配；DeepSeek-V4.1-Flash mHC 的矩阵/分块改写可分别证明前向与一阶求导。一般递推、变化 carry 的历史输出、非默认调用及其他跨区域改写仍无通用证明 | P2 | 待处理 | [代数设计](../design-docs/current/algebraic-equivalence.md#当前边界)、[真实 mHC 层](completed/validation/deepseek-v41-layer.md) |
| ALG-003 | `-inf` 及其依赖语义已保留，guarded-softmax 有限性规则使 GLM 的 5 个正例全部证明（2 结构、3 代数）。5 个负例仍缺完整输出反例：需适配近似 top-k 的精确环境语义、动态 gather 和 scan，不能把 indexer 输出任意化或以局部差异代替。禁止删除 mask 或替换有限常量绕过 | P1 | 待处理 | [核心层 jaxpr 记录](completed/validation/glm52-decode-dsa-mla.md)、[精确缺口回归](../../tests/test_glm52_decode_dsa_mla.py) |
| ALG-004 | 默认数学路径使用可信 SymPy；jax2sympy 的生产逻辑已直接并入项目构造器，其余数组转换代码保存在参考模块。支持完整标量/索引表达、非线性组合、原始条件及受限区域；旧独立证据和兼容 API 已删除。仍不完备：不透明排序/索引残差、未建模的整数 dtype 值域、无法确认的部分函数条件不能生成源程序负结论；一般递推与数据依赖终止仍明确拒绝或报告未决异常。已确认未正确拒绝的整数转换见 ALG-006～008 | P1 | 待处理 | [当前实现与边界](../design-docs/current/algebraic-equivalence.md#2026-10-16-可信-sympy-默认数学后端)、[迁移计划](completed/trusted-sympy-backend.md) |
| ALG-005 | 符号 worker 默认墙钟限制按用户要求由 10 秒调整为 60 秒，可显式覆盖；展开预算保留，尚无进程 RSS 硬上限。时间范围不包括父进程 tracing/结构规范化；复杂 Piecewise/张量条件仍可能超预算或超时，构造规模基线不能代替大图查询性能保证 | P2 | 待处理 | [资源与规模记录](../design-docs/current/algebraic-equivalence.md#实现与结果接口)、[规模脚本](../../tools/experiments/equivalence_backends/trusted_sympy_scaling.py) |
| ALG-009 | 普通 JAX 的纯死计算删减已确定设计，`freeze` 与 `SymbolicBuilder.graph/finish` 仍转换并保留这些计算的定义条件；Pallas 数学投影已有受限实现，不代表普通 JAX 已完成 | P2 | 待处理 | [决定与实际范围](../design-docs/current/algebraic-equivalence.md#2026-10-19-决定在代数-lowering-中删减未使用的纯计算)、[现有行为回归](../../tests/test_algebra_inequivalence.py) |
| ALG-010 | HIGHEST 的完整点积及有序输出对应已接入，支持有限标量缩放迁移；更一般的布局/代数改写对应仍明确拒绝，不能只按精度集合放行 | P2 | 部分完成 | [配对范围](../design-docs/current/algebraic-equivalence.md#2026-10-29相同非默认点积精度)、[正反例](../../tests/test_pallas_precision_pairing.py) |
| ALG-011 | 原始 KDA 的 dg 完整公式构造与坐标替换触发 SymPy 高开销，缩放正反例仍达默认 60 秒；需保持全部输出、内存定义性和完整输入反例前提的性能改进 | P1 | 待处理 | [限时诊断与范围](active/kda-resource-completion.md#新的后续问题) |
| ALG-006 | 整数归约/点积遗漏溢出语义，可能错误判等价 | P1 | 待处理 | [复现与修复要求](active/pallas-semantic-layer.md#2026-10-26-审查确认的正确性缺陷) |
| ALG-007 | iota 遗漏 dtype 值域，可能错误判等价 | P1 | 待处理 | [复现与修复要求](active/pallas-semantic-layer.md#2026-10-26-审查确认的正确性缺陷) |
| ALG-008 | 空整数极值归约单位元错误，可能错误判不等价 | P2 | 待处理 | [复现与修复要求](active/pallas-semantic-layer.md#2026-10-26-审查确认的正确性缺陷) |
| UPSTREAM-001 | 真实 SNR 动态索引的定义条件未决及 Piecewise 展开超时 | P2 | 待处理 | [Diffusers 实测与定位](completed/github-top10-tpu-validation.md#本轮剩余缺口) |
| UPSTREAM-002 | 共享纯 JIT 的定义条件与 attention 残差阻止部分数学负例证明 | P2 | 待处理 | [真实 MLP/attention 反例](completed/github-top10-tpu-validation.md#本轮剩余缺口) |
| UPSTREAM-003 | 普通 JAX shard_map/psum 数学转换尚未适配 | P2 | 待处理 | [ALX 双设备实测](completed/github-top10-tpu-validation.md#本轮剩余缺口) |
| UPSTREAM-004 | Splash 外层 broadcast、选取主输出的完整函数数学组合尚未支持 | P2 | 待处理 | [真实 Pallas wrapper 边界](completed/github-top10-tpu-validation.md#本轮剩余缺口) |

### 待核验旧记录

以下三项缺少完整的现行依据，保留原编号、优先级和问题描述；核实后再决定是否纳入开发任务。
此处的“待核验”不表示已解决，也不把旧记录直接当成当前能力缺口。

| 编号 | 范围 | 原记录与待核验内容 | 原优先级 | 状态 | 依据 |
| --- | --- | --- | --- | --- | --- |
| IR-001 | Oryx 接入 | 已测依赖组合不兼容 JAX 0.11.2；模式匹配缺少闭包值、共享关系及 effects 的完整检查，需要当前 JAX 适配设计 | P1 | 待核验（原待处理） | 首次评估文档当前缺失，待补复现记录 |
| AI-001 | 上游技能登记 | 原登记表的 3 个技能文件在源仓库中缺失 | P2 | 待核验（原待处理） | 原仓库 AGENTS.md 登记表 |
| AI-002 | 外部工具 | Falcon、Plane 与 TPU/GPU 未在本项目配置或验证 | P2 | 待核验（原待处理） | 当前仓库没有对应工作流模板或运行记录 |

### 已解决问题编号

已解决条目的原优先级、结论和证据随对应完成记录归档；此表仅保留检索入口。

| 编号 | 完成记录 |
| --- | --- |
| BFS-001 | [已解决记录](completed/graph-equivalence-validation.md#已解决问题编号) |
| BFS-002 | [已解决记录](completed/graph-equivalence-validation.md#已解决问题编号) |
| BFS-005 | [已解决记录](completed/graph-equivalence-validation.md#已解决问题编号) |
| BFS-006 | [已解决记录](completed/graph-equivalence-validation.md#已解决问题编号) |
| LAYER-001 | [已解决记录](completed/pallas-structural-comparison.md#已解决问题编号) |
| PALLAS-003 | [已解决记录](completed/pallas-broadcast-coordinates.md#已解决问题编号) |
| PALLAS-004 | [完全未引用私有分配的精确对称剪枝](completed/kda-comparison-fixes.md#已解决问题编号) |
| PALLAS-005 | [授权后的未消费读取、父输出边界与本地 DMA 填充传播；原始 KDA 六配置资源通过](active/kda-resource-completion.md#原始代码的父输出验收) |
| TEST-001 | [共同定义域反例与重复条件检查修复](completed/kda-comparison-fixes.md#已解决问题编号) |

旧 Pallas 边界、10 秒 CAS 失败与 60 秒复测保存在[历史问题快照](completed/context-history/tech-debt-2026-10-21.md)，不覆盖本节现行状态。

## 已完成

以下按归档阶段列出。“已完成”指对应计划声明的范围；早期实现可能已被后续方案替代，
历史测试结果不能当作当前源码的全量验收。后续发现的缺陷与剩余覆盖继续在当前计划和本页待处理清单中跟踪。

### JAX 结构、契约与验证

| 已完成计划 | 阶段内容 |
| --- | --- |
| [前十仓库 TPU JAX/Pallas 验证](completed/github-top10-tpu-validation.md) | 十库与两库额外覆盖；真实源码/AI 正反例、五项修复与 360 项整合回归；剩余四类能力单独跟踪 |
| [BFS 计算图规范化与比较](completed/bfs-reordering.md) | 第一轮结构规范化与完整比较基础 |
| [cond 内部参数绑定规范化](completed/cond-bindings.md) | 捕获槽位与实参/形参同步置换 |
| [带通信的 jaxpr 验证](completed/collective-jaxpr.md) | CPU 集合通信的结构与数值验证 |
| [计算图等价比较的完善与系统验证](completed/graph-equivalence-validation.md) | 完整图、函数契约、一阶求导与区域绑定 |
| [求导与内部绑定审查修复](completed/review-correctness-fixes.md) | custom VJP 整数输出与无根私有输入绑定 |
| [真实层级场景的 AI 固定改写](completed/business-layer-cases.md) | 五类业务计算与固定正负候选 |
| [SGLang-JAX 全模型层级对抗验证](completed/sglang-model-layers.md) | 固定上游版本的真实层与 AI 改写验证 |

### 数学比较与后端

| 已完成计划 | 阶段内容 |
| --- | --- |
| [JAX 函数代数等价比较](completed/algebraic-equivalence.md) | 首版代数规则、证据与函数自动模式 |
| [标量索引 IR 与代数缺口补齐](completed/algebraic-reduction-gaps.md) | reshape、静态分块、scan 累加等改写 |
| [开源数学等价后端评估](completed/open-source-equivalence-backends.md) | 首轮方案评估；生产集成另立计划 |
| [数学等价后端接入](completed/symbolic-math-integration.md) | 早期 SymPy 集成与验证 |
| [可信 SymPy 后端迁移](completed/trusted-sympy-backend.md) | 默认数学后端与架构迁移 |
| [将 jax2sympy 转换代码并入项目](completed/inline-symbolic-converter.md) | 转换逻辑、参考实现与许可证整理 |
| [删除旧代数证明器](completed/remove-legacy-prover.md) | 清理旧数学引擎与兼容接口 |
| [`-inf` mask 的受限代数支持](completed/negative-infinity-masks.md) | 保留非有限值与定义条件的受限规则 |
| [DeepSeek-V4.1-Flash 真实层验证](completed/deepseek-v41-layer.md) | 真实 JAX 层的代数改写与求导比较 |
| [GLM-5.2 核心层 jaxpr 测试纠偏](completed/glm52-jaxpr-robustness.md) | 改为核心内部的抽象图测试；未解决证明缺口另跟踪 |

### Pallas

| 已完成计划 | 阶段内容 |
| --- | --- |
| [完整结构比较](completed/pallas-structural-comparison.md) | 公共结构入口与定向验证 |
| [远端 DMA 规范化](completed/remote-dma-normalization.md) | 有限协议依赖、存储版本与输出组装 |
| [通信协议扩展](completed/communication-protocols.md) | 信号量、复用、部分等待与混合 DMA |
| [覆盖补齐与代数接入](completed/pallas-coverage-and-algebra.md) | 受支持投影接入公共数学入口 |
| [比较审查修复](completed/pallas-review-fixes.md) | 一维收缩范围与远端执行绑定 |
| [一维广播坐标修复](completed/pallas-broadcast-coordinates.md) | singleton 输入与广播坐标 |
| [外层浮点 iota 准入修复](completed/pallas-float-iota.md) | 拒绝把浮点 iota 当作精确整数坐标 |
| [有界整数坐标转浮点](completed/pallas-bounded-integer-casts.md) | 已证明范围内的精确转换 |
| [常量与远端布尔转换修复](completed/pallas-constant-and-boolean-casts.md) | 大整数常量舍入与布尔转换 |
| [scratch 范围与布尔运算修复](completed/pallas-scratch-boolean-review.md) | 存储范围与布尔语义审查 |
| [条件义务与布尔点积修复](completed/pallas-cond-boolean-dot.md) | cond 条件保留与布尔收缩语义 |
| [设备局部等价比较收口](completed/pallas-local-equivalence-closure.md) | 局部入口、受限条件初始化与循环审查 |
| [顺序输入 index_map](completed/pallas-ordered-index-map.md) | 当前 kernel 完成后计算下一输入映射 |
| [顺序输出映射与纯死计算 lowering](completed/pallas-output-map-and-pure-lowering.md) | 输出生命周期与受限数学删减 |
| [有界动态页与在途 DMA](completed/pallas-bounded-page-dma.md) | 逐笔页号、源版本、等待与输出关联 |
| [KDA 比较与既有回归修复](completed/kda-comparison-fixes.md) | 私有分配搜索剪枝、pad/exp2 与 Ref/编译参数覆盖、共同定义域见证缓存及回归核验 |
| [KDA backward 历史与 AI 候选验证](completed/kda-backward-history.md) | 最新 mega、三个历史版本与五个 AI 候选；53 组完整结构、106 次数值执行，局部资源/数学的原始读取阻塞单独记录 |

### 工程与文档

| 已完成计划 | 阶段内容 |
| --- | --- |
| [代码架构分层重构](completed/architecture-refactor.md) | core / algebra / functions 分层 |
| [比较结论与分类异常迁移](completed/comparison-exceptions.md) | 正常结果与未完成比较的异常通道分离 |
| [核心库类型信息补全](completed/type-annotations.md) | 类型契约与持续类型检查 |
| [核心文档修改保护](completed/protected-documents.md) | 核心文档的默认只读约定与权限配置 |
| [测试与文档按需加载整理](completed/test-doc-context.md) | 验证设施复用与文档分层 |
| [测试文件按职责合并](completed/test-file-consolidation.md) | 合并重复入口与零散测试文件 |
| [设计文档与执行记录归位](completed/documentation-consolidation.md) | 按内容迁移设计、历史方案和实验/验收记录，修复引用并明确生效与替代关系 |
| [测试资产与依赖分组整理](completed/test-assets-consolidation.md) | 39 个示例文件并入 tests，分离 dev/test 依赖并明确真实代码与 AI 正反例规范；迁移阶段仅做静态检查 |

## 历史记录

- [结构与代数的实施/验收记录](completed/history/index.md)：从混合设计文档拆出的首轮修复、规模实验和阶段结果。
- [模型与通信验证](completed/validation/index.md)：固定用例、来源、运行方式与历史验收；测试资产继续保留。
- [后端研究与选型](completed/research/index.md)：历史探针、研究资料和已被后续方案替代的建议。
- [Pallas 实施、调查与验收](completed/pallas-history/index.md)：27 个阶段、初期调查、旧实施计划和后端观察。
- [GLM-5.2 初版 decode DSA/MLA 对抗测试](completed/glm52-decode-dsa-mla.md)：已被核心层 jaxpr 测试纠偏取代，不作为当前能力声明。
- [历史上下文快照](completed/context-history/README.md)：保存旧计划索引、Pallas 进展和技术债快照，按需追溯。

## 维护方式

- 新建计划放入 `active/`，并在“正在进行”中登记。
- 阶段完成后移入 `completed/`，更新本页链接与分类；剩余问题写入本页待处理清单，并链接后续计划。
- 历史快照单独保留，不将其“当前状态”混入进行中清单。
- 现行算法、接口、约束与验证方法维护在[设计目录](../design-docs/index.md)，计划只记录任务与执行证据并链接设计。
- 计划内容要求见[计划规范](../PLANS.md)。
