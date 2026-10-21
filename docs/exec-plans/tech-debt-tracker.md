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
| BFS-004 | 深层迭代与共享区域复用已完成；持久化跨进程格式仍未实现。索引代数规则现已另行实现，见 ALG-001/002 的后续边界 | P2 | 待处理 | [深度证据](../development-docs/complete-graph.md)、[代数设计](../development-docs/algebraic-equivalence.md) |
| BFS-005 | cond 未使用槽位、scan/while 捕获与反馈、jit/shard_map 私有输入输出已联合规范化 | P1 | 已解决 | [区域绑定](../development-docs/region-bindings.md)、[系统验证计划](completed/graph-equivalence-validation.md) |
| BFS-006 | custom JVP/VJP 注册一阶规则、残差、透传、symbolic zeros 已适配；剩余边界见 AD-001/002 | P1 | 已解决 | [求导契约与测试](../development-docs/function-and-derivative-contract.md) |

## 系统验证后保留的边界

| 编号 | 问题 | 优先级 | 状态 | 依据 |
| --- | --- | --- | --- | --- |
| AD-001 | 仅存在于求导规则、没有对应原调用实参的动态捕获，底层注册规则物化仍 unknown；函数入口显式变换可另行比较 | P2 | 待处理 | [规则绑定说明](../development-docs/function-and-derivative-contract.md)、[回归](../../tests/test_custom_derivatives.py) |
| AD-002 | 延迟求导规则新增有序操作、反向日志及未解释 effects 需专用适配；当前明确拒绝，不能从普通父图推断求导顺序 | P2 | 待处理 | [规则边界](../development-docs/function-and-derivative-contract.md) |
| AD-003 | 高阶求导及任意变换组合没有整体保证；add_any 累加换序的 VJP 图差异可由默认代数模式单独证明，区域接口改变仍可能 unknown | P2 | 待处理 | [反馈用例](../../tests/test_feedback_bindings.py)、[代数边界](../development-docs/algebraic-equivalence.md#当前边界) |
| LAYER-001 | 公共 Pallas 结构入口与大图接口细化已接入；GLM5 absorbed Attention/Decoder 的 prefill/decode 四个真实配置在 search_budget=1 下通过完整结构比较，Wan/EPMoE 原图自比较通过。CPU 正常 lowering 仍拒绝 TPU 内核；自比较不计为任意改写候选认证 | P2 | 已解决 | [实际后端边界](../development-docs/full-attention.md#已确认的拒绝与未支持)、[接口细化](../development-docs/region-bindings.md) |
| PALLAS-001 | 当前设备局部覆盖缺口：远端条件初始化、并行共享 staging、一般索引/路由/终止及物理布局；执行绑定、预算与源后端限制分别列明。core_map/mpmd_map、输入输出共享 HBM 暂不支持 | P1 | 进行中 | [准确范围](../development-docs/pallas/validation.md)、[当前计划](active/pallas-semantic-layer.md) |
| PALLAS-002 | 第一版之外的全局扩展：一般多轴 mesh 跨设备值组装、动态路由/通信协议及多主机验证。有限单轴协议与多维 kernel grid 已有专用规则，不构成一般全局证明；这些缺口不阻塞设备局部图比较 | P2 | 待处理 | [通信范围](../development-docs/pallas/communication.md)、[通信验收](completed/communication-protocols.md) |
| LAYER-002 | 原 FluxSingleTransformerBlock Explicit sharding 与 GLM5 单 head 配置被锁定 JAX 追踪拒绝；未篡改原计算绕过 | P2 | 待处理 | [原始拒绝回归](../../tests/test_full_attention.py) |
| LAYER-003 | GLM5 稀疏 DSA、完整 MoE Decoder、其他配置和多主机/GPU/TPU 未验证 | P2 | 待处理 | [完整层范围](../development-docs/full-attention.md) |
| DOC-001 | `.agents/README.md` 仍有 scaffold 旧描述；当前工作区该目录只读，本轮以 AGENTS/架构/开发文档为当前事实来源 | P2 | 待处理 | [项目入口](../../AGENTS.md)、[当前架构](../../ARCHITECTURE.md) |
| DOC-002 | 代数功能与公共 Pallas 结构入口已实现，受保护的 README、ARCHITECTURE、AGENTS 和 docs/PLANS 部分说明及完成索引尚未同步；用户明确要求不修改，待后续具体授权 | P2 | 待处理 | [代数设计](../development-docs/algebraic-equivalence.md)、[Pallas 验收归档](completed/pallas-structural-comparison.md) |
| ALG-001 | 符号 reshape/静态切片、连续域合并已接入；动态索引、对角读取、任意步长覆盖、额外嵌套原语及特殊精度/分片仍待适配 | P2 | 待处理 | [代数边界与原语回归](../development-docs/algebraic-equivalence.md#当前边界)、[结构蕴含测试](../../tests/test_algebra_structural.py) |
| ALG-002 | 独立 scan 累加、已审核选择包装和严格默认纯 JIT 的残差组合已适配；DeepSeek-V4.1-Flash mHC 的矩阵/分块改写可分别证明前向与一阶求导。一般递推、变化 carry 的历史输出、非默认调用及其他跨区域改写仍无通用证明 | P2 | 待处理 | [代数设计](../development-docs/algebraic-equivalence.md#当前边界)、[真实 mHC 层](../development-docs/deepseek-v41-layer.md) |
| ALG-003 | `-inf` 及其依赖语义已保留，guarded-softmax 有限性规则使 GLM 的 5 个正例全部证明（2 结构、3 代数）。5 个负例仍缺完整输出反例：需适配近似 top-k 的精确环境语义、动态 gather 和 scan，不能把 indexer 输出任意化或以局部差异代替。禁止删除 mask 或替换有限常量绕过 | P1 | 待处理 | [核心层 jaxpr 记录](../development-docs/glm52-decode-dsa-mla.md)、[精确缺口回归](../../tests/test_glm52_decode_dsa_mla.py) |
| ALG-004 | 默认数学路径使用可信 SymPy；jax2sympy 的生产逻辑已直接并入项目构造器，其余数组转换代码保存在参考模块。支持完整标量/索引表达、非线性组合、原始条件及受限区域；旧独立证据和兼容 API 已删除。仍不完备：不透明排序/索引残差、未建模的整数 dtype 值域、无法确认的部分函数条件不能生成源程序负结论；一般递推与数据依赖终止仍 unknown | P1 | 待处理 | [当前实现与边界](../development-docs/algebraic-equivalence.md#2026-10-16-可信-sympy-默认数学后端)、[迁移计划](completed/trusted-sympy-backend.md) |
| ALG-005 | 符号 worker 默认墙钟限制按用户要求由 10 秒调整为 60 秒，可显式覆盖；展开预算保留，尚无进程 RSS 硬上限。时间范围不包括父进程 tracing/结构规范化；复杂 Piecewise/张量条件仍可能超预算或超时，构造规模基线不能代替大图查询性能保证 | P2 | 待处理 | [资源与规模记录](../development-docs/algebraic-equivalence.md#实现与结果接口)、[规模脚本](../../tools/experiments/equivalence_backends/trusted_sympy_scaling.py) |


Pallas 阶段边界、10 秒 CAS 失败及修改前复现、随后 60 秒复测均保存在
[历史技术债快照](completed/context-history/tech-debt-2026-10-21.md)；当前问题只维护上表和主题边界，避免重复整段验证记录。
