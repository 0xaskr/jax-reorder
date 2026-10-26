# 历史上下文快照（2026-10-21 压缩前）

> 非默认加载。原记录含已完成或被后续覆盖的边界；当前状态见[Pallas 入口](../../../design-docs/current/pallas-jax-equivalence.md)。

﻿# JAX / Pallas 语义中间层：设计记录与后续计划

创建：2026-10-18。状态：**实施中**（用户已要求实现设计）。
设计依据：[Pallas / JAX 等价比较设计](../../../design-docs/current/pallas-jax-equivalence.md)
（尤其 §3、§4.2、§8）。用户已确认：不把 JAX/Pallas 直接翻译成 SymPy 作为比较主路径；
中间抽象层由本项目自研；MLIR 虽长期看好但现阶段接入成本过大，暂不接入。

## 目标与完成条件

2026-10-19 最新范围：用户要求比较**分布式程序中单个 device 执行的 kernel**，
不是只允许单设备程序。mesh（包括多维 mesh）可以存在，第一版不展开或证明整个 mesh
上的全局计算。见[设备局部比较边界](../../../design-docs/current/pallas-jax-equivalence.md)。
设备坐标、轴大小、分片和通信目标若影响局部图，仍须保留依赖及接口绑定；不能无条件删除 mesh。
单设备视角不限制为一维张量、单维 grid 或单 program。局部入口已实现，当前规范化覆盖
静态多维规则分块、矩形访存与受限控制/状态；一般动态索引和布局等仍是后续覆盖目标。
用户进一步确认同时关注 device_id 引起的计算/分支差异和通信逻辑问题；不要求具体设备编号。
仅在相关证明需要时取得设备范围及对端行为，缺失则明确报错；不能用结构相同代替通信有效性。

2026-10-20 范围调整：用户确认暂不支持输入输出共用同一块 HBM。该项移出近期补齐
清单，不作为本阶段完成的阻塞项。下一优先项是在输入输出存储独立时，补齐跨输出
块的状态复用（切换到其他块后再回来继续更新）。历史受限 alias 实现记录保留，
不构成一般共享 HBM 的支持承诺。

2026-10-19 范围修订：用户要求 Pallas 层先做完整计算图比较，计数/进展模拟和跨分块数学证明
不作为结构准入。公共结构入口与 GitHub 正反例见[本轮结构计划](../pallas-structural-comparison.md)；
下列 ProgramView 统一消费、跨分块及远程通信语义规范化继续作为后续工作，不等于本轮已全部实现。

- 自研 Python 类型化源语义视图（`ProgramView`：值/区域/操作/Ref/effects/来源），只由
  `core/adapters` 解释 JAX/Pallas 源对象；`Graph`、分析与数学投影从视图派生。
- 第一版优先设备局部完整计算图和必要环境的绑定、局部内存与同步覆盖，再推进跨分块等代数比较。
  局部逻辑涉及的路由/等待检查按需使用对端约束；跨协议整体等价、完整分片组装与拓扑证明
  改为后续扩展，已有有限实现和验证继续保留。
  JAX 消费者迁移继续单列跟踪，不能将局部结论报告为全分布式程序等价。
- 当前已有公共完整结构入口，以及 A↔B 跨分块 normalized 内部链路、受限的 M1 分片和
  M2 本地 DMA/VMEM scratch。2026-10-19 已将一维完整投影接入公共 SymPy 和 Pallas↔JAX；
  同日接通有限远端通信的全局值投影和单轴分片组装；一般协议/分片及现有 JAX 消费者统一迁移仍待实现。
- 不新增运行时依赖；暂不接入 MLIR、StableHLO、ISL/Z3 等外部组件。

## 设计与取舍

- `ProgramView` 是唯一源事实；适配器独占 JAX/Pallas 对象解释权；视图不含比较算法、不构造 SymPy 对象。
- 远程 DMA/信号量由 core 语义图层解决：保留设备、Ref、传输区间及同步拓扑，完成跨设备
  存储版本传播、初始化与输出组装后，才提供纯值表达式给 SymPy。CAS 只判断表达式关系，
  不补通信协议或可见性语义；这些规范化前提不作为原图结构比较的额外准入要求。
- 源事实从 JAX 内部对象提取：`pallas_call` 参数、`GridMapping`/`BlockMapping`、已 trace 的
  `index_map_jaxpr`、kernel jaxpr、Ref/effects。私有 API 字段纳入 adapter inventory 审计。
- 生产索引域在 core 证明 blocked 轴对应/反向及混合进制拆合的逆映射，不枚举 grid 或
  全局输出元素。受限 lane gather 在分析预算内穷尽固定 lane 域；串行后缀在预算内
  重放实际状态更新。地址、覆盖、初始化及冲突判断均不调用 SymPy。
- 结果契约保持三类状态 + 分类异常；`normalized` 已作为内部证据种类实现。
- 当前规范化的已知边界（显式拒绝）：kernel 直接捕获数组常量、非默认布局、动态 grid、
  未证明的动态索引、vmapped dims、Element/BoundedSlice 块维度、非默认 JIT、未知 effect/原语。
  显式数组常量实参、scalar prefetch、静态标量/矩形访问、有限控制流、独立多 program
  的 DMA/VMEM scratch 与同坐标 alias 已接入；本轮扩展多维尾块的安全切片、
  几何 mask 和未定义值的显式 select/where 清理，详见 §9.26 的消费义务，
  非连续 DMA 仅补默认布局下对齐的 32 位矩形；一般动态索引和非默认执行策略仍需要专用证明。
  parallel 普通信号量缺少 core 目标绑定时拒绝，不能视作每个 core 自动私有。
  远端 DMA/静态设备分支已连接有限进展/冲突检查、初始化、区间版本和全局值规范化，
  支持连续切片；后续已接普通信号量、跨代/部分等待与混合 DMA。有限远端循环、
  数组信号量和受限 ANY/HBM 一维串行 grid 已补齐，仍限单 mesh 轴与可证明的
  有限控制；一般远端自动缓冲多 program 继续拒绝。计数读取仅接受可证明的稳定值。

## 工作项

### 2026-10-21 远端多维 grid 与并行事件流（完成）

用户要求将已有语义层 grid 展开接入远端 DMA。完成条件是多轴坐标、
index_map 和串行状态重放一致，parallel owner 之间没有虚构顺序；
实际共享存储、信号量及 DMA 版本仍受协议检查。设计见开发记录 §9.27。

- [x] 全坐标绑定与多维串行 I/O staging、窗口重访及值重放。
- [x] 同设备多事件流，保留串行顺序，检查并行真实资源冲突和计数消费归属。
- [x] 真实 Pallas 正反例、完整输出 oracle、证据重放与已有远端/协议回归。
- [x] 独立审查、质量门、本地结果记录和剩余边界。

不增加 SymPy 规则，不假定并行 program 自动获得私有 scratch 或 core 绑定；
core_map/mpmd_map、共享输入输出 HBM 和实机验证继续不在范围内。

并行远端最终采用显式 `CoreExecutionBinding`：单 core 完整网格点流在所有
device 相同；仅传 core 数量不足以确定次序。独立 owner 协议证明冲突，另按
完整调用顺序证明真实单 core 进展。新增静态 99 项、协议多流 30 项通过；
CPU 20 项中 18 项为完整数值/race，2 项为 interpreter 限制路径。六份源码
离线 v5:2x2x1 AOT 生成 Executable，未运行 TPU。详见开发记录 §9.27。
最终新静态 99 项、既有远端/本地/协议 507 项、共享 core/函数/代数 557 项、
CPU 20 项分批去重 **1,183 项通过**；47 项最终证据重放复测不重复计数。
Ruff、格式、类型、tracing、清单及文档检查通过；工件保存在
`/tmp/jax-reorder-remote-grid-ekhm4ft4/`。此子阶段完成，整体计划仍 active。

### 2026-10-21 普通尾块与 masked 访存（完成）

用户要求补齐上一轮第 1 项：普通非 ANY 的安全尾部切片、masked load/store，
以及远端协议/值投影中的普通 masked 访问。设计与反例见开发记录 §9.26。
基于当前工作区快照分别实现本地与远端路径，独立审查后合并；保留此前改动。

- [x] 复现完整对齐限制、mask 未生效前初始化检查和远端 mask 拒绝。
- [x] 按实际索引与 mask 确定访问/写入区域；inactive lane 不访问或初始化
  storage，读取无 other 的未定义值不能擅自补零或逃逸到有效输出。
- [x] 本地与远端的初始化、版本、DMA 冲突和通信时序采用一致的有效足迹，
  支持已证明安全的切片及有效值选择，不委托 SymPy 证明内存安全。
- [x] 真实源码、完整 NumPy/CPU 正反例、证据重放、预算和独立审查；适用
  源码离线 TPU AOT；冻结合并后的源码完成回归、质量门和本地记录。

本轮不扩展远端多维 grid、未知数据路由/终止或物理布局，不修改代数规则、
默认预算及 60 秒时限；core_map/mpmd_map、输入输出共享 HBM 继续排除。
不需要实机，不提交/推送。工件：`/tmp/jax-reorder-mask-access-f46cey_k/`。

已合入静态/数据 mask 的足迹与值检查、未定义值清理和原始 `jnp.where` 所需
的 kernel 内受限 replicated reshard/broadcast 适配。完整 collection 覆盖
1,793 项 Pallas、6 项分区，最终共享 core 368 项，去重 **2,167 项通过**。
初次静态的旧错误文案和四缓冲预算回归已修，最终源码定向 423 项全部通过；
四缓冲预算恢复到原 99,360/100,000。9 份 AOT 场景及 where/控制探针 2 份
编译成功，16 份含 masked load 的源码按锁定 TPU 后端拒绝单列。独立
1,200 组/79,131 点 oracle、源码哈希和质量门通过，完整结果与边界见 §9.26。


### 2026-10-21 并行 owner 的 ANY 尾块恢复（完成）

用户要求修复 §9.24 保留的符号并行 owner 同时涵盖完整块/尾块时的恢复拒绝。
在 core 根据有效恢复区域拆分并行域，分别重放串行窗口；恢复不引入跨 owner
的执行顺序，不初始化 padding，不调用 SymPy。设计与验证见开发记录 §9.25。

- [x] 复现 `(16,384)` 输出、`(8,256)` block、`grid=(2,3)`、
  `index_map(p,t)=(t%2,p)` 的 A/B/A 重访；区分完整块与尾块的有效范围。
- [x] 实现预算受控的并行域拆分和输出组装，保留窗口/owner 身份、版本、
  各输出独立生命周期及实际 DMA/wait 义务；避免枚举完整 tensor。
- [x] 补完整值正例、错误范围/旧 padding/缺初始化与同步反例，检查多维尾块、
  不同映射、多输出和证据重放；独立审查，适用源码 CPU/离线 TPU AOT。
- [x] 合并相关回归和质量门，更新完成范围与剩余边界。

最终主树 1,601 个 Pallas、6 个分区、368 个共享核心用例全部通过，合计
1,975 个不同 ID，其中 36 项新增；9 份离线 TPU AOT 成功。独立完整输出、
分区覆盖、DMA 证据篡改及 device 绑定检查通过。原始日志、补丁和 SHA 清单
在 `/tmp/jax-reorder-parallel-tail-dwz8eocj/`，详细范围和结果见开发记录 §9.25。

不修改代数规则、默认预算或 60 秒时限，不扩大远端 grid、未知路由/终止、
core_map/mpmd_map 或输入输出共享 HBM；不使用实机，不提交/推送。

### 2026-10-21 本地 ANY 恢复与多维远端原存储（完成）

用户明确要求补齐 §9.23 后的第 1、2 项；恢复由语义层按源流水契约保留
存储版本和时序，不引入 SymPy 地址/通信证明。以当前完整工作区为基线，
两项在隔离副本并行实现，主线程合并后验证共享接口和交叉案例。

- [x] 本地输出 `Buffered(1, RevisitMode.ANY)`：A→B→A 重访保留已写值，
  连续重复、首访、部分覆盖、多输出和初始化错误按实际源语义处理；
  grid 展开后保留串行依赖，不凭空添加源程序没有的同步。
- [x] 多维远端原存储：保留真实 shape、逻辑坐标、线性地址之间的映射，
  覆盖二维/三维原 Ref、矩形/有洞传输、变换组合及输出组装；源版本、
  DMA 完成计数、覆盖、冲突和信号量拓扑仍对应真实足迹。
- [x] 独立审查与交叉案例：正例、错值/错误坐标反例、缺同步/未初始化拒绝，
  证据重放与独立 NumPy/CPU 对照；适用源码离线 TPU AOT。
- [x] 合并回归、质量门与本地验证记录，更新剩余边界。

两条实现已合并并冻结：完整静态 1,345 项、CPU 通信 224 项通过；之后
仅补 2 个远端尾块测试并通过，最终 1,571 个 Pallas ID 全部覆盖。交叉
12 项、共享结构 368 项及 23 份源码离线 AOT 另行记录；独立审查的完整
坐标/协议集合检查与最终生产哈希均通过。类型、Ruff、格式、tracing、
inventory 和文档质量门通过，详见开发记录 §9.24。

保留符号并行 owner 跨完整块/尾块的本地恢复边界；远端仍限单具体 mesh
轴及有限一维串行 grid。多维有洞协议可能需显式增加分析预算，复杂错列
Piecewise 的下游 CAS 仍可能 60 秒未决；不作为已解决项。本子阶段完成。

不改动代数规则或 60 秒时限，不扩展 core_map/mpmd_map、输入输出共享 HBM、
多轴 mesh 全局证明或实机要求；受保护文档与依赖保持不变，不提交/推送。

### 2026-10-20～21 四类剩余问题并行补齐（完成）

用户要求分别开 subagent 继续实现 §9.22 后的四类缺口。以当前完整工作区快照
为基线，在隔离副本开发并由主线程合入；设计见开发记录 §9.23。

- [x] 组合覆盖：远端 DMA 的 Ref 端点映射、padded Ref 变换及有洞足迹；
  逐项保留原存储身份、有效域、真实搬运集合与顺序，补正例、错值和资源反例。
- [x] 自动缓冲重访：核对锁定后端的实际流水契约，补有源依据的时序/恢复规则；
  不用编译接受或自行添加同步代替缺失的源事实。
- [x] 动态行为与布局：用真实可 trace/lower 的具体索引、路由、循环和布局案例，
  扩展可证明子集；区分实现缺口、缺少范围/执行前提和源后端拒绝。
- [x] 代数性能：解决真实 384/256 错 mask、错误 mean 分母和大规模 Max 的
  已知未决/超时，保持默认 60 秒，且不以抽样或忽略定义域替代证明。
- [x] 交叉审查、相关回归、证据重放、CPU 对照、适用源码的离线 TPU AOT，
  统一记录完成范围和仍缺的必要事实。

完整 Pallas 分批验证：1,269 项静态比较、210 项 CPU 通信测试通过；最终
索引预算与极值绑定变量安全修正后，362 项定向回归通过。共享结构 368 项、
通用代数 765 个原 ID 分段覆盖及 21 份源码离线 AOT 见开发记录 §9.23；
没有把中断/已修失败或源后端拒绝算作连续全绿。独立审查确认极值 1/5
反例不再误判，最终主树与审查版本一致。Ruff、格式、类型、tracing、inventory
和文档检查通过。本子阶段完成，整体计划继续 active。

继续排除 core_map/mpmd_map、输入输出共享 HBM、多轴 mesh 全局证明与实机要求。
不新增未授权的运行时假设；一般数据依赖终止或未知布局必须有具体来源及证明条件。
受保护文档、依赖与已有用户改动保持不变，不提交或推送。

### 2026-10-20 时序重访、尾块有效域与 Ref 组合（完成）

用户确认继续按语义层/代数层分工补齐三项。设计见开发记录 §9.22。
以当前完整工作区为基线，在隔离副本并行修改，主线程合并共享文件并验证；
不修改受保护文档，不增加硬件要求，`core_map`/`mpmd_map` 与输入输出共享 HBM
继续排除。每项支持范围由源码正例、值差异反例及资源错误拒绝共同限定。

- [x] 远端窗口：显式 `ANY` HBM 的 A/B/A 完整与部分覆盖，按协议依赖及存储版本重放。
  保留输出暂存失效、无隐式回读、未完成 DMA/缺失同步拒绝。默认自动缓冲重访
  缺少原生写回完成顺序契约，保留拒绝；AOT 接受不作为时序证明。
- [x] 尾块：语义层保留合法 padding 的逐位置有效域；代数 lowering 根据真实选择
  分支证明定义性，按算子规则处理求和范围，保持输出形状、mean 分母和非单位元贡献。
- [x] Ref：组合 reshape/transpose/index 到原存储坐标，接入有界动态索引与
  受支持的 DMA 端点；不以包围矩形替代有洞足迹，不忽略源后端约束。
- [x] 集成审查、对应证据重放、正反例、相关回归与质量门，更新范围和实际验证记录。

交叉审查补齐了混合远端 kernel 对局部变换 DMA start/wait 的拒绝：不能丢弃
变换信息后复用一维协议解释器。尾块值经过 scratch 和本地 reshape DMA 后
仍保留初始化义务；未经屏蔽的输出不能用相同未定义域或代数消去获得等价结论。

最终 Pallas collection 1308 项，分批验证覆盖全部用例：首轮非通信 1069 项、
CPU 通信 193 项通过；尾块增量最终 82 项与 4 项包守卫通过。另有通用代数/
符号功能 756 项、共享结构 70 项通过。包分层检查发现并修复
`symbolic → pallas → symbolic` 导入环；没有放宽守卫。17 份源码离线 TPU
编译成功（含 2 份自动窗口仅审计），2 份 transpose lowering 拒绝单列。
完整日志、交叉审查和真实尺寸 CAS 超时见开发记录 §9.22。质量门、tracing、
adapter inventory 与文档检查通过。本子阶段结束，整体计划继续 active。

### 2026-10-20 剩余覆盖专项与大图预算（完成）

用户要求多 subagent 继续完成 §9.20 列出的五类缺口。以当前完整工作区为
基线，分别在隔离目录推进，主线程审查合并；不覆盖前轮实现，不修改受保护
文档、不连接实机。输入输出共享 HBM 仍排除，语义规则不调用 SymPy。

- [x] 远端流水：有限 rank-one 串行自动 staging/writeback、多输出窗口与有效 padding DMA。
- [x] 调度：交错独立/连续串行后缀、显式单 core banked 状态与条件化多 core 信号量绑定。
- [x] 动态语义：有限地址写/算术控制、静态部分 Ref 链与受审查 layout-pass 开关。
- [x] 规模：接口不变量细化，四个真实 GLM5 absorbed/Decoder 配置在预算 1 下通过。
- [x] 交叉审查、正反例、27 份离线编译成功源码与主工作区全部 1141 项 Pallas 分批验证。

不能用任意路由/终止假设填补源程序缺少的约束；可证明的有限子集落地，
无法闭合的情况给出最小案例和明确缺失事实。每项分别记录语义比较、
数值验证与后端接受结果，新增规则保留预算及证据重放。

交叉审查收紧了 banked 状态的 core 归属前提与省略 signal 目标的原生语义，
并消除新 helper 的导入环。统一静态首轮 964 passed / 1 failed；唯一失败是更早
预算拒绝造成的旧报错位置断言，保留入口与直接 owner 预收费双重检查后，最终
新增功能/包分层组合 152 passed。全部官方 CPU 通信 176 passed；
共享结构/区域/代数/规模 312 passed，GLM5 四配置通过。质量门、tracing、
adapter inventory 通过。完整证据见开发记录 §9.21。

2026-10-20 用户确认：`core_map`、`mpmd_map` 暂不支持，其入口适配与自动
core 绑定不列为当前补齐项或验收阻塞。已有显式 `CoreExecutionBinding` 与
`execution_cores=1` 的条件证明保留；离线编译参考不构成这两个入口的支持。

剩余的任意动态路由/终止、未知 core 分配下的持久私有状态、动态 Ref 变换、
一般物理布局与远端窗口重访已留在技术债；没有宣称支持任意合法 Pallas。

### 2026-10-20 并行补齐语义覆盖与验证（完成）

用户要求多个 subagent 一次推进当前剩余问题。工作基于本轮开始时的完整
工作区快照，在隔离目录实现后逐项审查、合入；保留已有未提交改动。
设计与结果记录见开发记录 §9.20。grid、地址、控制流和资源规则继续在 core，
不以 SymPy 代替语义证明；输入输出共享 HBM 继续排除。

- [x] Grid：分组尾块、静态不规则 index_map 与精确有限覆盖；交错轴的源码可编译但状态调度未证明，保留拒绝。
- [x] 通信：普通/远端信号量数组、有限远端循环与受限 HBM 串行 grid；多 core 用两种绑定的正反例固定边界。
- [x] 语法：精确步长、有界动态读/while、完整静态 Ref 视图；atomic 等锁定后端拒绝单列，不伪造语义。
- [x] 规范化：完整坐标分区在 core 证明分组 grid 与内部循环，保留浮点运算顺序。
- [x] 验证：独立安装 jax[tpu]，29 个具体用例在 CPU 主机完成 TPU v5p 2x2x1 离线后端编译，无实机执行。
- [x] 合入审查与回归：修复单例轴 mask 坐标误判和单矩形重复预算计费，最终 967 项 Pallas 用例分批通过，质量门通过；未证明范围单列。

不承诺解决任意数据依赖终止或任意并发通信；每个新增支持项都需要源事实、
适用前提、正例、反例和拒绝路径。用户先提供 falcon 2x2x1，随后明确不需
实机，改用 AOT；未连接或启动 falcon。CPU 数值验证与静态证明分别记录。
完成记录见[开发记录 §9.20](../pallas-history/stage-20.md#920-并行补齐语义覆盖与验证2026-10-20完成)。
本子阶段完成，整体计划继续 active；任意多 core 绑定、交错轴调度、一般动态写/路由、
部分动态视图和非默认物理布局等仍需独立规则。

### 2026-10-20 Grid 并行批量轴与串行状态展开（完成）

用户提出将并行 grid 轴放入值的批量维度，将串行轴展开为通过实际状态依赖连接
的 kernel 实例；无需额外插入 barrier，program_id 直接由实例坐标绑定。
采用这个方向统一后续 grid lowering，具体语义和验收范围见
[开发记录 §9.19](../pallas-history/stage-19.md#919-grid-并行批量轴与串行状态展开2026-10-20)。

- [x] 明确 program 批量轴、真实 Ref 地址、串行状态传递及有界展开的职责。
- [x] 复用状态解释器，按实例化后的 index_map 块序列证明覆盖与独立输出窗口。
- [x] 补齐 `j//2`、`j//3`、混合轴和拆分 grid 的规范化等价与对应反例。
- [x] 内部循环单独核对投影与 CPU 输出，未取得图等价结论的路径明确记录。
- [x] 保持跨 program DMA 的显式等待、并行冲突拒绝及独立大 grid 的预算回归。
- [x] 完成扩展回归、质量检查及结果归档。

串行连接保留必要的读写和副作用依赖，不新增 barrier 或隐式 DMA 等待；
parallel 轴不增加全局顺序。
用户已要求继续解决自动访存、资源共享和展开规模三项。实施先复用独立
program 快速路径，再补有界串行实例及实际输出块窗口；同时验证输入搬入、
输出失效与 DMA 生命周期。并行坐标保持符号，局部 dot/reduce 维度不改变。
普通信号量的 core 目标按锁定后端实际语义审查，缺少调度绑定时不得猜测。
输入输出共享 HBM 继续排除。

用户随后明确 grid 展开只涉及语义图、不涉及 SymPy。本轮生产改动仅在
core；地址、覆盖、顺序、存储和通信生命周期均在此完成。新测试直接验证
core normalized 证据；内部循环的数值对照与跨图等价结论分列。额外代数
实验的未决及超时单列为边界，相关试验性展开策略已撤掉。

新增 43 项通过，包含 8 项官方 CPU 用例；`j//2`、`j//3` 与拆分 grid
直接获得 core normalized 证据。并行普通信号量在显式
`execution_cores=1` 条件下检查并重放，任意多 core 调度仍拒绝。
全 Pallas 首批 778 passed、1 项报错文本断言失败，修正为实际更早的 DMA
窗口冲突后，相关模块 62 passed；最终 collection 785 项已分批全部通过。
共享守卫 32 项及 Ruff/格式/类型/tracing/inventory 检查通过。具体条件、
中途失败和剩余边界见开发记录 §9.19；整体语义层计划继续 active。

### 2026-10-20 按输出 index_map 分别证明窗口（完成）

用户要求补齐多输出不同窗口寿命。以 `grid=(M,K)`、`y(i,j)=(i,j)`、
`s(i,j)=(i,)` 为具体入口：`y` 每步结束一个块，`s` 保留同一块完成串行
累加。初始化和更新必须由实际代码决定，不由 index_map 推测运算。
设计见[开发记录 §9.18](../pallas-history/stage-18.md#918-多输出独立-index_map-窗口2026-10-20)。

- [x] 复现不同窗口拒绝，增加具体两输出和上游 scratch 派生用例。
- [x] 每个输出分别证明 ownership/连续串行后缀，以后缀并集决定有界重放。
- [x] 分别封存、失效和投影输出窗口，仅检查实际切换资源的 DMA 冲突。
- [x] 补齐嵌套窗口、输出换序、初始化/错误 map/并行 writer/DMA/预算反例。
- [x] 完成静态证据重放、官方 CPU 对照、受影响回归和质量检查，更新结果与边界。

输入输出共用 HBM 继续排除；本轮不放宽自动 staging 的非连续复访契约，
不引入一般动态索引或新的 CAS 恒等式规则。

新增 39 项已分批通过：28 项静态/数学及 11 项官方 TPU CPU 用例；还有
296 项既有静态/数学、40 项既有 CPU 用例通过。grid 中的串行轴改为 kernel
内循环可规范化匹配，错误汇总更新或 map 有完整输出反例。首批回归发现普通
一维多输出的规则标签变化，收紧扩展入口后最终 144 项复测通过。去重共 375 项，
Ruff/格式/类型/tracing/文档/diff 检查通过，无遗留失败。本子阶段完成，
整体语义层计划继续 active；一般动态地址/布局、远端流水等边界见技术债。

### 2026-10-20 跨输出块状态复用（完成）

用户要求推进下一项，输入输出共享 HBM 已排除。本轮以真实上游源代码和独立正反例
核对输出窗口的生命周期；完成条件是合法显式状态复用可规范化/重放，错误的跨块
依赖或未初始化读取明确拒绝，实际 CPU 执行与完整输出证明分列。

- [x] 固定上游案例与执行语义，复现当前边界，区分自动 staging 与显式 DMA/状态。
- [x] 更新开发设计，按实际存储和 grid 顺序维护状态，保留连续输出写入契约。
- [x] 增加合法复用、换序、错误槽位/初始化/等待等正反例和完整输出证据。
- [x] 完成官方 CPU 对照、受影响回归、类型/格式检查并记录结果与剩余范围。

初步确认：官方 TPU 文档要求自动 BlockSpec 中对同一输出块的多次写入连续。
因此不会把简单放宽 trailing-suffix 检查作为实现；合法的非连续复用需由显式
scratch/ANY Ref/DMA 的实际状态给出依据，必要时补这些组合的覆盖。

固定 JAX `test_hoisted_scratch_space` 和 `test_hbm_hbm_grid_dma` 原始函数；完整源码
与锁定 GitHub revision 逐字节核对，保留 SHA-256、Git blob 和 Apache 许可证。
新增 scratch 输出历史、A/B/A/B 槽位复用，以及显式 DMA 回读已有输出后更新；
重置状态与写错目标给出完整输出反例，缺初始化、漏等待、暂存区失效和在途冲突拒绝。
串行状态路径也用于一维 Ref；parallel 前缀保持符号、serial 后缀有限重放，预算先行。
CPU 8 项已通过。SymPy 对显式按块 stack 参考式判等；平铺 `x * 2` 仍无法合并
分块读取恒等式，明确返回数学未决并单列边界，不放宽 core 地址/生命周期证明。

新增 38 项回归，Pallas 最终 collection 703 项分批全部通过（605 静态/数学、98 CPU）。
最后一维入口/局部状态/DMA/semantic-slice 147 项复测通过；共享层 232 项通过。
Ruff、格式、类型、tracing、adapter inventory、文档与 diff 检查通过。详细命令、
中途失败及修正、剩余范围见
[跨输出块验证](../pallas-history/stage-17.md#917-跨输出块状态复用2026-10-20完成)。
本子阶段完成；当时遗留的多输出不同窗口寿命已由上方 index_map 子阶段补齐，
一般动态地址/布局、远端流水和真实 TPU 验证仍未完成，整体计划保留 active。
输入输出共享 HBM 继续排除。

### 2026-10-20 真实双缓冲 DMA 与跨迭代依赖（完成）

用户要求以真实开源双缓冲 DMA kernel 补齐串行通信，继续用具体正反例推动。
本阶段固定上游版本并保留原函数；只证明设备局部、有限可确定迭代与地址下的
DMA 槽位、存储版本和等待关系，不扩大到一般动态路由或全 mesh 协议。

- [x] 固定上游双缓冲函数及来源，运行原始追踪并记录首个真实缺口。
- [x] 写明跨迭代资源生命周期规则；保留 pending DMA、来源快照和 semaphore 代次。
- [x] 接入所需源适配与 core 证明，逐次验证合法预取、等待移动和缓冲轮换。
- [x] 构造漏等待、等待错槽、提前覆盖、错误数据块等源级反例，分开报告拒绝和数值差异。
- [x] 运行官方 CPU 解释器、证据重放及受影响回归，记录剩余边界并完成本子阶段。

固定 JAX 0.11.2 的完整 copy/matmul 与真实 emit_pipeline。补齐 ANY 整块存储、局部
run_scoped、DMA semaphore 数组、有限循环动态切片、降秩矩形搬运；跨轮次保留 pending
和版本，按实际 staging 生命周期放行串行 grid。合法等待移动、2→4 buffer 和同步
调度等价；错槽、漏 wait、提前覆盖和未排空退出拒绝；搬错数据块单独报告输出差异。
官方 CPU 解释器覆盖完整输出，资源证明与数值/竞态结果分开记录。实现、测试矩阵和
剩余边界见 [双缓冲 DMA 验证](../pallas-history/stage-16.md#916-真实双缓冲-dma-的跨迭代证明2026-10-20完成)。
本子阶段完成；一般动态布局/索引、跨输出块状态复用、远端流水和真实 TPU 验证仍未完成，
整体语义层计划保持 active。输入输出共享 HBM 按用户决定暂不支持，移出近期待办。
本轮新增 41 项 Pallas 回归和 1 项整数见证边界测试。Pallas 最终 665 项分批通过；
最后公开入口补上整数分块选择与有界 0/1 反例，搬错块获得完整输出不等价结论。
共享层和数学回归按批次记录，未把旧阶段的全仓库计数当成本轮重跑结果。

### 2026-10-19 开源 kernel 驱动的测试与修复（2026-10-20 完成）

用户要求先用开源真实 Pallas kernel，并执行“测试 → 发现问题 → 修复 → 继续测试”，
直到仓库中本轮覆盖的 Pallas cases 无未解决的验证问题。先保留上游实现及来源，
不以手写简化 kernel 代替基线；AI 正反例和源程序的实际 CPU 执行另列证据。

- [x] 跑当前 Pallas 全部定向测试，记录基线与原有拒绝路径。
- [x] 固定 JAX 官方 matmul/FlashAttention 及已有 Tokamax、vLLM 源案例，核对版本和源码。
- [x] 分别验证源图结构、core 状态投影、数学结论和 CPU 数值/资源行为，定位首个实际缺口。
- [x] 对确认的仓库问题增加失败回归并修复；串行维按源代码的实际状态更新解释，不默认求和。
- [x] 重跑受影响项及完整 Pallas 矩阵：当前 624 项均通过，真实执行与预期拒绝/停滞分列。
- [x] 更新本地结果、阶段状态及剩余边界。

补充回归覆盖其余 1961 项，与 Pallas 624 项按最终 collection 去重后，当前全仓库
2585 项分批全部通过。Ruff、格式、类型、tracing、adapter inventory 和文档检查通过；
CPU 数值/资源执行与预期拒绝/停滞分别报告，未做真实 TPU 验证。完整结果见
[开源验证记录](../pallas-history/stage-15.md#915-开源-kernel-驱动的测试与修复2026-10-20完成)。
本子阶段完成；一般语法覆盖、串行通信/alias 等仍使整体语义层计划保持 active。

按最新范围优先推进设备局部入口和覆盖；下列 M1/M2/M3 保留已有实现与后续工作的记录。

### 2026-10-19 局部入口与覆盖补齐（本轮完成）

用户在定向验证后要求补齐三类缺口。本轮不改变既有全函数比较契约，新增局部结构入口；
规范化先覆盖已固定的合法用例，不能用抽样或删除副作用放宽证明。

- [x] 提取单个调用及父实参/常量/轴绑定，新增公开设备局部结构比较入口和可重放证据。
- [x] 接纳 scalar prefetch、标量 Ref 索引、有限 scan/while 的源事实，保持区域端口与 effects。
- [x] 在局部值/资源解释中支持有限循环、逐分支证明、稳定 semaphore_read 控制依赖。
- [x] 证明多 program scratch 初始化、counter 归零、DMA 完成与安全 alias；保留跨 program 依赖拒绝。
- [x] 将外层常量和纯运算绑定接入完整输出投影，验证公共数学入口和旧远端规则回归。
- [x] 更新定向用例、反例、独立 CPU 对照和开发记录，执行类型/格式/文档及算法检查。

实现规则和验证边界见开发记录 §9.13。动态循环没有有限终止证据、数据决定远端路由、
跨 program 资源传递和非默认 core 调度不能依赖一次执行的成功；此类保留结构事实并明确拒绝证明。

本轮主回归 **427 passed**，最终 alias 复查 **35 passed**（与主回归重叠，另增一个反例），
独立 CPU 检查 **27 passed**（25 项实际执行、2 项预期 read lowering 拒绝）。Ruff、格式、
类型、tracing、adapter inventory 和文档检查通过；未做真实 TPU 验证。实现和完整命令见
[§9.13](../pallas-history/stage-13.md#913-局部入口与覆盖补齐规则2026-10-19)。
本子阶段完成；下一子阶段补多维地址域。一般控制/通信、JAX 消费者迁移等仍使整体计划保持 active。

### 既有工作项

#### 2026-10-19 多维 grid 与访存覆盖（本子阶段完成）

用户要求下一步补 grid/访存，并明确 grid、index_map 和 kernel 坐标的对应应由语义层处理。
本轮以 core 的精确整数域及矩形存储证明为主，不将地址/覆盖/依赖判断交给 SymPy。

- [x] 源适配保留多个静态 grid 轴、多维 Blocked、每轴索引及矩形切片；未知变换继续拒绝。
- [x] 组合 grid 域、index_map 与局部坐标，证明轴对应/换序/反向、混合进制拆合的覆盖与地址对应。
- [x] 完成多维局部读写、部分切片、初始化/版本和资源检查；尾块与不支持组合显式区分。
- [x] 接通局部入口及完整值投影，保留旧一维行为；代数仅消费已证明的值表达式。
- [x] 增加二维/三维、轴错位、重叠/漏写、越界和大域预算测试，运行独立 CPU 对照与回归。
- [x] 将结果及剩余边界写入开发记录，更新本子阶段状态。

新增完整批次 **62 passed**；既有 Pallas 主回归 **428 passed，58 deselected**，
既有局部 CPU 专项 **11 passed**。
新增 CPU 检查有 11 项 TPU 专用 interpreter 实际成功、1 项明确断言 masked load lowering
拒绝，另有通用 `interpret=True` 尾块数值成功。完整证据见
[§9.14](../pallas-history/stage-14.md#914-多维-grid-与访存2026-10-19)。
一般动态/非线性索引、非连续 DMA、真实 TPU 验证及完整多轴 mesh 全局通信仍有边界。

- [x] 设备局部比较入口：抽取单 `pallas_call`，明确局部输入输出、设备环境量及通信资源的绑定；
      mesh 可存在，不将全局投影或协议证明作为局部结构比较准入，也不修改既有全函数契约。
      验证相同局部图的环境对应，以及改变轴依赖、局部输入来源、通信目标/计数/等待顺序的反例。
- [ ] device_id 相关逻辑：保留符号输入、运算、分支及路由/等待依赖；不要求具体编号。
      按需从外层 jaxpr、调用/轴绑定取得必要范围，并取得相关对端的通信行为，不能仅凭范围
      相同合并设备坐标。验证末端路由、错误目标、缺失信号、循环等待与缺上下文拒绝路径。
      结构匹配和通信有效性单列证据；该项未完成时不能宣称已支持完整的设备局部逻辑检查。
- [x] `core/model.py`：Program/Region/Operation/Value/Literal/Ref/effects/Origin/SourceHandle，
      Pallas `GridAxis`/`BlockMappingView`/`KernelDomainView`/`PallasCallView`。
- [x] `core/index_domain.py`：整数索引表达式、index-map 区域求值、blocked 覆盖/唯一写入检查。
- [x] `core/adapters/jax.py`：本 slice 的受审查原语注册表和区域适配（SSA 检查、类型、effects、JIT 绑定）。
- [x] `core/adapters/pallas/call.py`：解码单 `pallas_call` 的 grid、BlockMapping、index map、Ref、
      effects、interpret/alias/compiler 参数；显式拒绝未审查 schema。
- [x] `tests/test_pallas_semantic_slice.py`：A/B fact 保留、program_id、覆盖/尾块、部分写拒绝。
- [x] `core/memory.py` 存储版本与纯值投影、`core/normalization.py` 规则与
      `NormalizedEvidence`；`core/view_graph.py` 的内部结构快路径与原视图复核。
- [x] adapter inventory 扩展：Pallas/Ref/indexer 字段和原语参数已加入现有版本快照。
- [ ] 文档同步：设计文档与执行记录；受保护文档（README/ARCHITECTURE/PLANS）待单独授权。
- [x] M1 扩展片段：JIT/shard_map、相同 mesh/分片契约、受限尾块与完整 CompilerParams inventory。
- [x] M2 本地片段：start/wait 源事实、单 program 完整块 DMA、信号量代次、VMEM scratch
      初始化/版本/复用、读写冲突检查、证据重放与官方 interpreter 对照。
- [x] M2 本地区间：静态连续切片、拆分合并、区间版本、累计消费、独立进程 oracle 与 CI job。
- [x] M2 远端分析组件：mesh tuple 路由、axis_index/条件区域、独立发送/接收完成、单代全量计数消费、
      依赖环与访问区间检查、源图证据重放；该组件不产生等价结论。
- [x] M2 语义层整合：连接远端协议依赖与存储版本传播，检查生命周期、初始化、只读输入及
      完整输出覆盖，形成跨设备全局输出。验证直接/中继、请求—回复、等待环与缓冲区复用；
      该工作在 core 完成，不交由 SymPy 推断通信语义。范围和验证见[远端计划](../remote-dma-normalization.md)。
- [x] M2 协议扩展：普通信号量、跨代复用、部分等待与本地/远端混合；累计计数、必要因果和
      合法求值见证分离，修复 semaphore_read 源顺序。见[通信归档](../communication-protocols.md)。
- [ ] M3：现有 JAX 结构/数学消费者统一适配，公共入口证据分派，JVP/VJP 回归。
      公共结构与一维代数分派已接入，验证见[覆盖与代数记录](../pallas-coverage-and-algebra.md)；
      全部消费者统一迁移及 JVP/VJP 尚未完成，故不勾选整个 M3。
- [ ] 第一版局部覆盖：多维 grid/访存、多 program、控制流、atomic、scratch/alias 与局部同步的组合；
      静态多维规则分块、矩形访存及上述受限组合已完成；一般动态索引、atomic、布局和同步组合
      仍待扩展。区分结构适配和语义规范化的缺口，保留未知语义拒绝及单设备 TPU 验证缺口。
- [x] 首个串行 grid 规则：输出窗口的 trailing arbitrary 后缀按实际代码区分覆盖、加法更新和一般递推；
      保留初值、分支及跨迭代版本，dot/reduce 值图接入 SymPy。非后缀复用和串行通信/alias 尚未扩展。
- [ ] 代数后续：扩展局部值投影，并按[已定规则](../../../design-docs/current/algebraic-equivalence.md)
      实现未使用纯计算的删减；不能以放宽 effects 或删除环境依赖获得匹配。
- [ ] 后续全局扩展：多轴 mesh 的全局组装、动态跨设备路由/协议、真实多设备及多主机验证；不阻塞第一版。

## 决策与进展

- 2026-10-18 实施启动：核对工作区仅有设计文档和执行计划改动，未发现已有实现。
  已读取 protected-docs 配置；核心文档保持只读。按设计 §4.2 修正旧计划中枚举程序实例的建议。
  锁定环境实际为 JAX 0.11.2；`BlockMapping.index_map_jaxpr` 实际可为 `Jaxpr`，effects 绑定 Var；
  公共 `pl.load/store` 已移除，尾块探针需使用锁定版本内部 masked primitives。
- 已实现内部链路，生产域分析不枚举程序或元素；`2**24 + 13` abstract tracing 通过预算 256
  的跨分块证明。支持正向/反向 blocked、坐标 mask、128/256 跨分块和非负 div/rem 的逆映射恒等式。
- Ref 解释器先检查全部操作，再按当前版本读写；不会把未初始化输出填零。完整有效域只有一个写入者；
  普通 store 的未使用旧值可丢弃，但实际使用旧值明确拒绝。读集合相同不代表地址对应相同。
- 结构证据目前保持源方程顺序，独立复核未规范化视图；normalized 证据重放规则前提并检查终端值 DAG。
  默认 JIT 的参数/区域绑定和调用边界进入证据；非默认 JIT 契约明确拒绝。
- 首轮失败暴露了 `mul.out_dtype` 与超越函数 `accuracy` 参数遗漏，按实测 schema 修复；
  JAX tracing 会消除未使用 get，完整图读取反例改为显式插入保留的源方程。
- DAG 投影与相等检查使用共享遍历，80 次重复平方不会展开成指数规模树；整数索引各中间运算检查溢出。
- M1 扩展：恢复 shard_map 的手动轴环境，保存 mesh/设备排列/端口分片；Pallas 清除 Ref variation
  的绑定政策单独保留。输出复制性由实际值依赖证明，不能依赖关闭后的 check_vma 声明。
  Mosaic 17 个 CompilerParams 字段逐项保存；未知 flags 拒绝，空 FrozenDict 单独编码。
- M2 本地片段：DMA 适配不预配对，分析只在 start 已发起、源版本稳定和完整 wait 匹配时发布目标；
  每个 semaphore 同时最多一次传输，完整消费后可复用；不同 semaphore 允许共享只读源。
  VMEM scratch 不初始化为零，整块写产生新版本；源/目的在途保护使用索引，不枚举执行调度。
  结论显式携带有效本地 DMA 最终完成的平台假设。程序自身缺少等待/提前覆盖不是可豁免假设。
- shard_map + scratch 组合检查确认：VMEM 构造器的空抽象 mesh 并非环境外构造特有现象，
  环境内构造也相同。保留原描述符，并显式记录内核局部 mesh 绑定；其它已命名 mesh 不自动重绑。
- M2 本地区间扩展替代此前每信号量单次未完成传输限制。相同存储对/dtype 的对齐切片可以共同计数；
  部分 wait 只消费数量，不指定已完成 producer，消费整代总量后才解除全部在途保护并发布目标版本。
  区间版本证书允许同一缓冲区的不交叠区域独立更新，源/目的坐标差进入值公式；生产路径按边界切分。
  静态 stride/OOB/缺口/重叠/计数不足或过量/代次混用均有拒绝路径；大形状不分配数据。
- 官方 interpreter 移至逐例隔离进程：启动/编译 90 秒，READY 后执行 30 秒，超时终止进程组；
  增加故意挂起和持有管道的子进程测试，验证阶段报告与清理。CI 单独串行运行通信 marker。
- 2026-10-19：远端分析现从完整适配视图实例化固定设备的有限命令；条件分支只由静态设备坐标决定。
  start→发送完成→接收完成保持分离；一代一次完整消费建立全部 producer 前驱，计数不守恒先拒绝。
  交叠冲突由依赖可达关系检查，不能以某个安全拓扑排序代替所有调度的证明。双设备请求—回复保留
  提前 start，延后到等待回复之后会检出环。默认 TPU device barrier 的源参数进入准入，跳过 barrier 拒绝。
  远端证据单列未完成的生命周期、初始化/覆盖及全局值条件；只有模型内进展与冲突次序，不返回等价。

- 早期讨论阶段曾只记录结论、暂不实现；后续用户要求实现。2026-10-19 再次澄清第一版
  聚焦分布式程序的设备局部视角，mesh 可以存在，整体分布式证明转为后续目标。

## 验证结果

### 2026-10-19 设备局部边界定向验证（已完成）

用户要求继续验证并把结果写入本地。本轮只检验当前能力与拒绝边界，不将新的局部入口
或一般协议分析的设计写成已实现。复用真实 Pallas tracing、公共结构比较、现有视图
规范化和隔离 CPU oracle；没有 TPU 硬件，危险等待用例只做静态检查。

- [x] 外层上下文：具名轴/父参数绑定、多维 mesh、抽出 kernel 后缺失上下文的行为。
- [x] 动态通信：设备条件与数据条件分支、循环、稳定/不稳定信号量读数、错误路由与等待环。
- [x] 多 program 资源：普通分块、scratch/DMA/semaphore、alias 的结构保留与规范化边界。
- [x] 固化可复现用例，执行相关回归和独立 CPU 对照，记录分类异常与未验证条件。
- [x] 将结果写入开发记录与技术债，并完成文档、格式及差异检查。

新增 [31 项回归](../../../../tests/test_pallas_local_scope.py)，静态/无通信 CPU 检查 24 passed，
联合通信 23 passed，相关既有回归 226 passed，共 273 项；未执行全仓库 pytest。
详见[逐类结果、命令及环境](../pallas-history/stage-12.md#912-设备局部视角的定向验证2026-10-19)。
普通 Python/NumPy 静态标量预处理可以通过；合法 NumPy 数组参数暴露外层常量绑定缺口。
其余定位到 scalar prefetch/标量索引、稳定 read 控制流、循环和多 program 生命周期。
CPU 通信 23 项中两项是 read lowering 的预期拒绝，没有 TPU 硬件证据。
本轮只新增测试与记录，设备局部入口及上面的覆盖工作项继续保持未完成。
全仓库 Ruff 检查及格式检查通过（245 个 Python 文件）；文档检查通过（11 个文件、
213 个本地链接），`git diff --check` 通过。该验证子任务完成，整体实现计划继续保持 active。

### 历史验证批次

- 2026-10-19 设备局部范围澄清：仅更新开发设计、活动计划/索引与技术债，不修改比较算法。
  `tools/check_docs.py --changed` 检查 11 个文件、207 个本地链接通过；`git diff --check` 通过。
  本次未重跑算法测试；下列实现批次的验证范围保持原样。
  同日补充 device_id 相关逻辑与通信检查边界，仍仅更新文档，沿用上述检查命令。

- 2026-10-19 远端语义整合：协议、版本和全局组装的正反例及证据重放，官方双/三设备
  interpreter 对照、公共数学入口与既有局部规范化回归，见[本轮记录](../remote-dma-normalization.md)。
  该记录更新下列历史批次中“远端仅有分析”的状态。

- 最终扩展组合：上述 Pallas/分层文件加 canonical/semantics/correspondence/algebra_structural，
  `-m 'not pallas_communication' -q --tb=short`：**502 passed，21 deselected，41.25s**；
  随后追加三设备 switch/clamp 与对应拒绝路径，见下方专项记录。
- 最后 Pallas/协议/清理/分层静态专项（含 clamp 上下界与中间溢出反例）：
  **175 passed，25 deselected，22.18s**。该批次覆盖全部当前 Pallas 文件，未重复全量 SymPy 测试。
- 本地与双设备全部隔离通信模拟重跑：**21 passed，166.55s**；三设备直接发送/中继（两种模式）
  另 **4 passed，37.65s**，合计 **25 个通信用例**。三设备 worker 同时核对原生源图的协议证据。
  三设备初次使用 lax.switch 时 **4 failed**，原因是源图新增 clamp 尚未适配；已补充受审查 schema
  和静态整数控制求值后通过，没有删除 clamp 或仅以模拟结果绕过源适配。
- Ruff、格式检查（224 文件）、basedpyright、adapter inventory、文档链接及差异空白检查通过；
  tracing 示例与 **1,280** 个小域 oracle 通过。受保护文档和权限配置无 staged/unstaged 修改。
- 本地区间与既有结构组合回归（上文六文件、oracle 清理测试及 canonical/semantics/correspondence/
  algebra_structural）：**480 passed，16 deselected，37.17s**，这是远端适配扩展前的批次。
- 远端源图/事件/清理定向检查：**25 passed，5 deselected，8.98s**；包括独立计数状态机与证据篡改。
- `uv run pytest tests/test_pallas_remote.py -m pallas_communication -q --tb=short`：
  **5 passed，46.75s**；四个正常模拟与一个已有静态等待环证据的执行超时观察，按阶段回收进程组。
- 隔离 runner 清理逻辑改进后重跑本地区间 eager 模拟：**5 passed，42.45s**。
- 本地区间最终静态批次：`uv run pytest tests/test_pallas_dma.py tests/test_pallas_dma_regions.py
  tests/test_pallas_storage.py tests/test_pallas_semantic_slice.py tests/test_pallas_sharding.py
  tests/test_package_layout.py -m 'not pallas_communication' -q --tb=short`：**147 passed，19.60s**。
- 独立进程 CPU oracle：上述 DMA 两文件加 `-m pallas_communication`：**16 passed，122.75s**；
  包含串行/重排/复用、拆分总等待/分次等待、半块交换、中继、不交叠版本更新，两种 DMA 模式均通过。
- `uv run pytest tests/test_pallas_oracle.py -q --tb=short`：**4 passed，4.21s**，包含编译前/执行阶段
  超时、非零退出、错误结果，以及父进程已退出但子进程忽略 SIGTERM 的清理路径。
- 适配 inventory 新增完整/切片 DMA PyTree 静态元数据与 `TransformedRef` 字段，重新生成锁定快照后通过；
  Ruff 与 basedpyright 已通过本地区间生产代码检查。之前 uv editable 重建受 DNS 限制失败，依赖构建已恢复，
  失败时未执行的测试已重新运行，未把构建失败计为测试结果。
- `uv run pytest tests/test_pallas_dma.py tests/test_pallas_sharding.py
  tests/test_pallas_semantic_slice.py tests/test_package_layout.py -q --tb=short`：最终 **126 passed，17.82s**
  （session `16795`）。包含共享源保护、100 次版本预算、shard_map + scratch、不同 mesh 拒绝路径；
  最后的输入无关整数差异回归确认不等价证据也携带 DMA 完成假设。
- 上述四个文件加 `test_canonical.py test_semantics.py test_correspondence.py test_algebra_structural.py`：
  最新 **454 passed，33.22s**（session `65486`）；此前 VMEM 扩展前批次 **441 passed**。
- DMA 专项 **40 passed** 后补充不同 scratch mesh 反例，已纳入上述 454 项；eager/on_wait
  数值对照均显式断言官方 race detector 未报告冲突。
- `uv run pytest tests/test_pallas_semantic_slice.py tests/test_package_layout.py tests/test_canonical.py
  tests/test_semantics.py tests/test_correspondence.py tests/test_algebra_structural.py -q`：**382 passed**
  （默认 JIT 扩展前的回归批次；扩展后新增路径已包含在上述 57 项中）。
- `uv run python tests/pallas_domain_oracle.py`：**1,280** 个小域覆盖/漏写案例通过；pytest 另交叉检查生产域证明。
- `uv run python tests/adapter_inventory.py`、`uv run python tests/lax_inventory.py`：通过。
- `uv run python examples/inspect_jaxpr.py`、`uv run python examples/bfs_reordering.py`：通过。
- Ruff、格式检查（212 文件）、basedpyright、`tools/check_docs.py --changed`、`git diff --check` 已通过；
  adapter/lax inventory、两个运行示例和 1,280 个小域 oracle 已在当前扩展后重跑通过。
- 原位置的全量 `uv run pytest -q --tb=short` 出现多项失败后中止（session `35056` 已返回 exit 130），
  输出 `/tmp/jax-reorder-pallas-pytest.log`，不报告通过。
- 首个失败 `test_elementwise_polynomial_rules[float32-pair6]` 单独复现为 SymPy worker **10 秒超时**
  （session `40680`，1 failed，29.44s）。HEAD 源码复制到 `/tmp` 后同一测试 **1 passed，18.01s**；
  当前源码复制到 `/tmp` 后 **1 passed，15.85s**。逐文件核对：既有实现模块内容均未改动，新实现模块 9 个。
  这些证据与源码位置/文件系统开销影响超时一致，尚不能宣称全部全量失败的根因已确认。
- 先前 slice 的全量验证使用当时同内容源码副本（不包含后续 sharding/CompilerParams/DMA 扩展）：
  `PYTHONPATH=/tmp/jax-reorder-pallas-current-rgxyl_23/src .venv/bin/python -m pytest -x -q --tb=short`。
  session **23664** 已结束：**1 failed, 116 passed，1067.60s**；失败为
  `test_absolute_matches_both_half_axis_spellings[float32-<lambda>0]` 的 SymPy worker 10 秒超时。
  输出 `/tmp/jax-reorder-pallas-pytest-local-src.log`；未修改 worker 时间阈值，不报告全量通过。
- DMA 独立数值对照使用官方 `pltpu.InterpretParams` 的 eager/on_wait 两种模式及 race detector；
  直接复制、串行中转、独立算术重排、scratch 复用成功。反例只做静态拒绝，避免执行潜在死锁 kernel。
- 所有证据均为 CPU tracing/测试；没有真实 TPU 编译、多设备 TPU 执行或并发证明。

## 未解决问题

- 局部入口已有显式 scope、实参/逻辑轴绑定和可选资源证明；嵌套 shard 环境、计算过的全局
  分片实参值证明与一般动态对端协议仍有边界。全函数比较结果不能当作局部结论，反之亦然。
- M1 尚缺非默认 CompilerParams 的重写规则和更一般地址/状态组合。
  现有符号规则只证明一维正向/反向 blocked；非线性/一般映射不能偷偷退回枚举。
- 一般 mask、数据依赖终止循环、atomic、alias 的语义规范化仍未普遍支持，不代表结构图均无法比较。
  一维独立多 program、有限循环、动态局部分支及同坐标 alias 已接入；scratch 在调用期间复用，
  每 program 需证明初始化/归零/DMA 完成。parallel 普通信号量需要另补 core 目标绑定和硬件证明。
- 已有远端单轴/单程序全局投影及普通信号量、跨代复用、部分等待、混合 DMA 规则。
  更一般的全局协议、分片组装与多主机验证属于第一版范围之外；局部图中的通信操作仍须保留。
  分段整数坐标转浮点的 CAS 条件尚未证明，core 投影和 CPU 对照通过不代替数学证明。
- 公共结构快路径已接通既有 BFS/歧义搜索，一维值投影已接入公共数学路径；现有 JAX
  全部消费者尚未统一迁移到 ProgramView。
  M3 继续单列跟踪；多设备整体证明不再是第一版设备局部比较的完成条件。
- MLIR 分析后端、ISL/Z3 选型是 M1 之后的 spike，不阻塞当前设计。
