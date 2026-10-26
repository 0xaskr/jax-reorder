# 历史上下文快照（2026-10-21 压缩前）

> 非默认加载。原记录含已完成或被后续覆盖的边界；当前状态见[Pallas 入口](../../../design-docs/current/pallas-jax-equivalence.md)。

# 进行中的计划

[通信协议扩展](../communication-protocols.md)已完成并归档：普通信号量、跨代复用、
部分等待及本地/远端 DMA 混合已接入必要因果证明、版本传播与公共比较；修复 semaphore_read
源顺序。独立逐单位状态机、源反例和官方通信回归通过；CPU read lowering 和真实 TPU 仍有边界。

[远端 DMA 语义规范化](../remote-dma-normalization.md)已完成并归档：单轴/单程序有限协议
已连接跨设备区间版本、初始化及全局组装；直接/中继、请求—回复、切片和 scratch 复用通过。
新增静态/数学 31 项、官方通信模拟 15 项通过；多代/部分等待随后由上述通信协议计划补齐，
一般协议及真实 TPU 仍是边界。

[Pallas 覆盖补齐与代数接入](../pallas-coverage-and-algebra.md)已完成并归档：
40 类官方构造正反例及 48 项原生 schema 有真实追踪来源；一维全局值投影已接公共 SymPy，
支持跨分块及 Pallas↔JAX。最终数学组合 82 passed；默认超时按用户要求改为 60 秒，
五个历史超时与 worker 专项 49 passed。多维投影、一般分片/远程协议及真实 TPU 证据仍有边界。

[Pallas 完整结构比较](../pallas-structural-comparison.md)已完成并归档：公共结构入口、
Ref/信号量资源对应及副作用顺序已接入；Tokamax/vLLM 17 组 AI 正反例通过。
结构相关 1029 项、全部 Pallas 262 项、深度/规模 6 项回归通过；大型图预算和 TPU 验证仍有边界。

[JAX / Pallas 语义中间层：设计记录与后续计划](../../active/pallas-semantic-layer.md)：实施中。
内部 `ProgramView`、符号 blocked 域、分片契约、存储版本、结构/normalized 证据链已落地；
新增单 program 本地 DMA、有限远端全局版本和 VMEM scratch 复用检查，定向验证通过；
2026-10-19 用户将第一版聚焦为分布式程序的设备局部 kernel 比较，mesh 可以存在。
已补公开局部入口、外层常量/实参/逻辑轴绑定、有限控制流、稳定 read 控制依赖，以及
独立多 program 的 scratch/DMA/顺序 counter 和安全 alias。静态多维 grid、轴换序/反向、
混合进制拆合、矩形访存及受限尾块已在 core 完成覆盖与地址证明；新增 62 项、既有回归
428 项通过，CPU 成功执行与预期 lowering 拒绝分列记录。结构匹配与可选通信检查
分别报告；完整全局输出组装与跨协议整体等价证明留到后续，既有全函数契约保持独立。
开源 kernel 驱动阶段于 2026-10-20 完成：官方 matmul/FlashAttention 与 Tokamax/vLLM 原函数的状态投影及
独立 CPU 对照通过；串行后缀按代码重放，dot/reduce 接 SymPy，matmul 128/256 分块正例和
三个更新反例均获数学结论，FlashAttention grid/内部循环移动可直接规范化匹配。
修复 RoPE 包装的 staging Ref 绑定错误。该阶段 Pallas 624 项与其余 1961 项回归去重后，
全仓库 2585 项测试分批全部通过，详见
[开源验证记录](../pallas-history/stage-15.md#915-开源-kernel-驱动的测试与修复2026-10-20完成)。
随后完成[真实双缓冲 DMA 子阶段](../pallas-history/stage-16.md#916-真实双缓冲-dma-的跨迭代证明2026-10-20完成)：
固定 JAX 官方 copy/matmul，补局部 scope、DMA semaphore 数组、对齐矩形搬运及跨迭代
生命周期；合法调度等价，漏 wait、错槽和提前覆盖拒绝，搬错块单独检查输出差异。
本轮新增 42 项回归，Pallas 最终 665 项分批通过；公开代数入口给出搬错块的 0/1 反例。
随后补齐有限串行 grid 的跨输出块状态：上游持久 scratch 与 HBM grid DMA 基线、
AI 的 A/B/A/B 槽位复用和显式输出回读、换序等价及初始化/等待/错误目标反例。
切换窗口会使输出 staging 失效，独立大 grid 保留符号证明，输入输出共享 HBM
仍按用户决定暂不支持。该阶段仍缺的多输出不同窗口寿命随后按下段补齐；
一般动态循环/索引、非默认布局和远端流水仍待证明；平铺读取与固定槽位分段值的
CAS 恒等式也单列未决。该子阶段新增 38 项，当时 Pallas 703 项分批验证通过，
含 98 项官方 CPU 解释器测试。
详见[跨输出块记录](../pallas-history/stage-17.md#917-跨输出块状态复用2026-10-20完成)。
随后补齐[多输出独立 index_map 窗口](../pallas-history/stage-18.md#918-多输出独立-index_map-窗口2026-10-20)：
每个输出分别按拥有者前缀/连续串行后缀确定初始化、封存与失效，支持逐块输出与
跨步累加、三层窗口、不同局部块形状和长窗口 DMA。grid 串行轴改为内部循环有
normalized 等价及重放证据；错更新/错 map 给出完整输出反例。新增 39 项分批通过，
包含 11 项官方 CPU 用例；296 项既有静态/数学、40 项既有 CPU 用例也通过。
首批发现普通一维多输出的规则标签回归，修复后最终 144 项复测通过；去重共
375 项验证，无遗留失败。Ruff/格式/类型/tracing/文档/diff 检查通过，本子阶段完成。
CPU 数值/竞态检测与资源证明分开报告。
随后完成[有界 grid 展开](../pallas-history/stage-19.md#919-grid-并行批量轴与串行状态展开2026-10-20)：
串行实例按实际 index_map 合并各输出的连续分组窗口，保留输入 staging、
DMA/信号量状态和真实资源身份；并行拥有者保持符号，局部 dot/reduce 维度
不改变。`j//2`、`j//3` 与拆分 grid 在 core 直接判等，展开不调用 SymPy。
并行普通信号量支持显式单 core 执行条件并记录证据；任意多 core 分配仍待
证明。新增 43 项通过，含 8 项 CPU 用例；修正一项更早 DMA 拒绝的报错断言
后，最终 Pallas collection 785 项分批全部通过。内部循环的投影/CPU 对照
与未决图比较分别记录，没有为后者改变 grid 展开策略。
随后完成[并行覆盖补齐](../pallas-history/stage-20.md#920-并行补齐语义覆盖与验证2026-10-20完成)：
静态不规则 index_map/分组尾块、精确步长与完整 Ref 视图、有界读取/while、
普通及远端信号量数组、受限 HBM 远端串行 grid，以及 grid/内部循环的精确
坐标分区匹配。全部规则仍在 core。交叉审查修复单例轴 mask 坐标误判及
重复预算计费；最终 967 项 Pallas 用例分批通过。本机独立安装 libtpu 后，
29 个具体源码通过 TPU v5p 2x2x1 离线后端编译，没有连接实机。
交错轴共享状态、任意多 core 绑定、一般动态写/路由/布局与锁定后端不支持的
语法仍明确单列，不能将 AOT 编译通过当成这些语义的证明。
随后完成[剩余覆盖专项与大图预算](../pallas-history/stage-21.md#921-剩余覆盖专项与大图预算2026-10-20完成)：
有限远端自动 IO/padding DMA、交错独立/连续后缀、显式 core 流拓扑、有界动态写、
静态部分 Ref 与受审查布局开关已落地。banked 持久状态需显式单 core 前提，
普通 pallas_call 不自动保证调用者的 core 映射。主工作区 Pallas 1141 项分批通过，
共享核心 312 项通过；GLM5 absorbed Attention/Decoder 四配置在接口预算 1 下通过。
27 份具体源码离线 TPU 编译成功，后端拒绝与仅审计探针单列。任意动态控制、
动态 Ref、一般物理布局及缺失执行绑定仍保留明确边界。
随后补齐[时序重访、尾块有效域与 Ref 组合](../pallas-history/stage-22.md#922-时序重访尾块有效域与-ref-组合2026-10-20完成)：
显式 ANY HBM 可按已证明的 DMA/wait 时序重复覆盖；动态 Ref 链和精确矩形
本地变换 DMA 在 core 组合到原 storage。合法输入 padding 的初始化义务独立
进入代数层，局部 Sum 按实际 mask 收缩，384/256 分块的 sum/mean 四种
公共 CAS 正例通过。默认自动缓冲重访仍缺源写回顺序契约，远端变换 DMA、
padded Ref 变换及部分大型 CAS 查询保留边界；原生后端拒绝和超时单列。
具体最终验收及独立反例见该记录。
2026-10-21 的四线补齐已合入：精确有洞 DMA 与一维远端端点映射、padded Ref
组合、同步 Buffered(1,ANY) 远端恢复、有界整数分支/循环 carry，以及真实
384/256 错 mask、错 mean 分母和大型 Max 的确定 CAS 结果。默认/IMMEDIATE
重访、本地输出 ANY、无跨设备约束的数据路由、多维原存储远端组装和任意
物理布局仍保持明确边界。代码、组合正反例、离线编译与实际集成回归分开记录于
[四线验证记录](../pallas-history/stage-23.md#923-四类剩余问题并行补齐2026-10-21完成)。
随后用户要求补齐本地 ANY 与多维远端原存储：两条实现已合并，12 项组合
及最终 1,571 个 Pallas 用例分批验证通过。本地保存每个窗口及 owner 的最后版本，
只恢复 HBM 有效区；远端保留原 shape、精确分段足迹和任意单分片轴的坐标。
源码正反例、padding 生命周期及源后端边界见
[本轮记录](../pallas-history/stage-24.md#924-本地-any-恢复与多维远端原存储2026-10-21完成)。
本地符号并行 owner 同时涵盖完整块与尾块的 ANY 恢复已进一步补齐：core
保留全局 writer 证明，按有效范围分区并分别重放串行历史，再按各窗口 inverse
组装输出。新场景 30 项及分区单测 6 项、完整 NumPy 对照、9 份离线 AOT 已通过；
最终 1,601 项 Pallas、6 项分区和 368 项共享核心回归全部通过，记录见
[并行尾块恢复](../pallas-history/stage-25.md#925-并行-owner-的-any-尾块恢复2026-10-21完成)。
普通尾块和 masked 访存进一步支持安全切片、部分写入、显式 other 与正确
where 清理无效值；本地及远端共享有效足迹分析，仍检查实际 DMA/初始化义务。
原始 where 的受限适配也已补齐；完整集合与共享 core 去重 2,167 项通过，
初次回归的文案/预算问题已修复。结果与锁定 TPU 后端的 masked load 限制见
[普通尾块与 masked 访存](../pallas-history/stage-26.md#926-普通尾块与-masked-访存2026-10-21完成)。
JAX 迁移继续单列跟踪，暂不接 MLIR。实现、回归和 CPU 对照详见
[本地验证记录](../pallas-history/stage-14.md#914-多维-grid-与访存2026-10-19)。

[核心库类型信息补全](../type-annotations.md)已完成并归档：异常阶段与分类改用 `StrEnum`，补全核心库签名、回调与容器类型，接入持续检查。分阶段 1957 项通过，1 项既有超时在修改前源码也可复现；详见验证记录。

[代码架构分层重构](../architecture-refactor.md)已完成并归档：`jax_reorder` 从 20 个平铺模块拆为 `core/`、`algebra/`（含 `adapter/`）与 `functions.py` 三层，拆解 symbolic/regions/trusted 导入环并新增分层守卫测试。顶层公共 API 不变，子模块路径按 0.2.0 破坏性调整；核心分组 465 passed，详见归档记录。

[三种比较结论与分类异常迁移](../comparison-exceptions.md)已完成并归档：删除 UNKNOWN，统一结构、SymPy worker、函数模式和批处理的异常通道；首轮全量发现的 34 项迁移失败均已修复，最终 110 项复测通过。后续用户已授权提交并推送 `origin/main`。

[将 jax2sympy 转换代码并入项目](../inline-symbolic-converter.md)已完成并归档：生产逻辑直接并入构造器，完整 60 个 primitive 映射与逐元素转换入口迁入普通模块，删除 vendor 包，许可证保留在文件头。相关回归 **741 passed**，另有 46 项转换/后端检查通过；尚未提交或推送。

[删除旧代数证明器并整理 SymPy 比较路径](../remove-legacy-prover.md)已完成并归档：移除旧证书接口、12 个旧模块及专用测试，解耦转换辅助代码，公式回归迁至可信 SymPy 路径。全量 1930 项中发现的两项超时已修复，受影响范围复测 **299 passed**；详见归档中的分阶段验证记录。后续用户已授权提交并推送 `origin/main`。

[可信 SymPy 后端：开源组件复用与架构迁移](../trusted-sympy-backend.md)已完成并归档：本地 jax2sympy fork、默认可信 SymPy 数学链路、Nerva 两层测试与可终止 worker 均已接入。全量 **2166 passed**，1 个已有上游弃用警告；按后续用户授权提交到 `origin/main`。已无进行中的迁移计划。

[数学等价后端接入、完整验证与 main 发布](../symbolic-math-integration.md)已完成：SymPy 候选与独立复核、完整多项式输出的不等价证据、guarded-softmax/PV 规则已接入。全量 2095 passed；GLM 三组签名均为 5 个正例已证明和 5 个负例 unknown。按用户最新授权直接推送 main。

[开源数学等价后端的并行评估](../open-source-equivalence-backends.md)已完成：三个 subagent 与主代理覆盖 21 个候选/组件，运行符号探针与真实 GLM QK/PV helper 的 jaxpr→SymPy 转换。该评估阶段的生产适配工作已在上述接入计划推进；当前保留边界见 ALG-004。

[`-inf` mask 的受限代数支持](../negative-infinity-masks.md)已完成：保留特殊值及其依赖语义，阻止非有限路径进入实数乘加展开；新增 41 项回归，全量 1952 passed。该阶段 GLM 图比较为 4 个已证明正例和 6 个 unknown；最新结果与剩余边界见上述接入记录及 ALG-003。

[GLM-5.2 核心层的 jaxpr 健壮性测试纠偏](../glm52-jaxpr-robustness.md)保存初始结果：仅抽象追踪核心的 5+5 改写，当时为 2 个结构正例与 8 个 unknown；后续 `-inf` 支持与最新结果见上述完成计划。

[GLM-5.2 纯 JAX decode DSA/MLA 初版](../glm52-decode-dsa-mla.md)保存历史结果；其外围改写和 CPU 数值验收偏离目标，已按上述计划改为核心层的 jaxpr 对比。

[DeepSeek-V4.1-Flash 真实 JAX 层的代数改写验证](../deepseek-v41-layer.md)已完成并归档：固定原始 mHC 层、两种 AI 代数改写、前向/JVP/VJP 证明、独立数值参照与 CPU 计时；新增 68 项回归，1870 个用例均完成通过验证。

[修复仓库审查确认的求导与内部绑定问题](../review-correctness-fixes.md)已完成并归档：修复 VJP 整数输出规则遗漏和无根私有输入换序，新增 27 项回归，全量 1802 项通过。

[标量索引 IR 重构](../algebraic-reduction-gaps.md)已完成并归档：四组需求及通用域规则接入，全量 1775 项通过。

[JAX 函数代数等价比较](../algebraic-equivalence.md)已完成并归档：全量 1600 项通过，受保护文档按用户要求保持原状。

[计算图等价比较的完善与系统验证](../graph-equivalence-validation.md)已完成并归档。

[远端多维 grid 与并行事件流](../pallas-history/stage-27.md#927-远端多维-grid-与并行事件流2026-10-21完成)子阶段完成：多轴坐标、自动 staging/ANY 恢复、跨 program DMA；并行使用明确的完整单 core 点流条件，并分别验证并行资源冲突和实际顺序进展。分批去重 1,183 项通过，六份离线 AOT 成功；范围与边界见记录。

新计划按 [计划规范](../../../PLANS.md)编写；完成记录见 [completed](..)。
