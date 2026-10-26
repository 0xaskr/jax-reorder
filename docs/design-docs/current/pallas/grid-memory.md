# Pallas Grid 与内存

从[任务入口](../pallas-jax-equivalence.md)到此；地址、覆盖、生命周期与定义性全部属于 core。
当前范围于 2026-10-26 对照源码与固定测试核对；下列历史验收未在本次文档整理中重跑。

## 投影路径与模块分工

| 模块 | 输入与证明职责 |
| --- | --- |
| [memory.py](../../../../src/jax_reorder/core/memory.py) | 定义共享 `ValueExpr`、`MemoryProjection` 和版本模型；其 `functionalize_call` 处理独立一维 blocked program 的直线状态消除，要求对齐窗口和完整 writer |
| [local_memory.py](../../../../src/jax_reorder/core/local_memory.py) | 本地控制流与独立 program 证明；检查各动态分支、初始化和计数闭合，遇到需要跨 program 状态的受支持情形时转入 tensor 路径 |
| [tensor_memory.py](../../../../src/jax_reorder/core/tensor_memory.py) | 多维存储状态、Ref 变换、mask、窗口重访、串行 index_map、有限动态地址和本地 DMA；保留跨 program 版本，按预算重放必要状态 |
| [remote_memory.py](../../../../src/jax_reorder/core/remote_memory.py) | 先取得跨设备协议证明，再传播发送源版本与目标值；完整函数组装全局输出，局部范围保留每个 `(输出端口, device)` |

[normalization.py](../../../../src/jax_reorder/core/normalization.py) 选择完整函数的简单、本地扩展或远端投影；
[local.py](../../../../src/jax_reorder/core/local.py) 则在局部实参/轴绑定后选择本地、远端或显式 core 证明。
独立 program 快路径与有状态回退共享预算；回退仍须满足自身前提，不能把一次准入失败当成已证明安全。

## Grid、地址与窗口

- `program_id`、`num_programs`、index_map 与局部坐标使用同一 grid 计划。
  静态串行坐标在预算内重放；parallel 轴优先保留符号批量模板，不逐 tensor 元素枚举。
- 并行 program 轴不进入 kernel 原有 dot/reduce 的收缩轴，也不直接成为用户输出 shape。
- 一维块的 reduce/dot 同样保留完整输入范围与绑定坐标；是否逐元素取决于运算语义，
  不取决于数组 rank。简单、控制流和外层实参投影使用同一准入规则。
- 广播保留每个输入的 shape，长度为 1 的输入轴固定读取坐标 0；kernel 嵌套区域与外层
  实参共用坐标路径判定。同 shape 逐元素和标量广播继续使用原有快速路径。
- 外层和局部一维投影从当前运算的 shape 绑定完整坐标范围；scratch 不借用输出 block
  的范围，标量保留设备/program 作用域。不同 scratch 大小和部分 Ref 窗口按实际需求
  选择扩展路径，不依赖 jit 包装；坐标上下界不是输入数据值的上下界。
- 空 reshape 保留空域与来源，不构造除零坐标；归约消费者仍使用真实空形状。
  [空域修复与条件义务验收](../../../exec-plans/completed/pallas-cond-boolean-dot.md)。
- 各输出分别根据实际 index_map 划分窗口；同窗口可继续更新，换窗口按源 staging 契约提交/失效。
  串行意味着保留实际状态依赖；是否累加由 kernel 代码决定。
- scalar-prefetch 参与串行输入/输出 index_map 时，当前 kernel 完成状态更新后才计算下一 program 的映射；
  本轮已绑定地址保持不变。这一执行前提进入结果，默认自动流水线是否满足它仍由调用方保证。
- 证明全局覆盖、逆映射及唯一 writer/owner 后才能组装输出；共享 HBM/Ref 不按 owner 克隆。
- 串行实例传递内存版本、scratch、计数器和在途 DMA。实例结束不隐含 wait、清零或 barrier。
  多维远端使用同一计划，额外通信义务见[通信与执行绑定](communication.md)。

核心源码：[grid_execution.py](../../../../src/jax_reorder/core/grid_execution.py)、
[tensor_domain.py](../../../../src/jax_reorder/core/tensor_domain.py)、
[tensor_memory.py](../../../../src/jax_reorder/core/tensor_memory.py)。

## 重访与 ANY 恢复

`RevisitMode.ANY` 与内存空间 ANY 是不同概念。A→A 保留当前暂存内容；允许的 A→B→A
恢复 A 上次实际写回 HBM 的版本。首访没有旧值，多个输出分别保存生命周期。
仅恢复 HBM 有效域和暂存 Ref 的交集；未提交的 padding 在窗口切换后不恢复。
Buffered(1, ANY) 的同步恢复按锁定源契约处理；默认/IMMEDIATE 非连续重访、双缓冲 ANY
仍受源规则限制，不因比较器能描述状态就认为源后端合法。

并行 owner 跨完整块/尾块时，按所有输出的有效区要求拆成精确矩形分区。
原 writer/owner 证明继续有效，相邻相同分区可合并，区域仍可代表大量 program。
不添加跨 owner 顺序；非单调映射可能分区爆炸，耗尽预算明确失败。
显式 `program_point` / core observer 路径不额外推断恢复顺序。
源码：[tensor_owner_partition.py](../../../../src/jax_reorder/core/tensor_owner_partition.py)。

## Mask 与定义性

- 以原 Ref、结果坐标和 mask 推导真实 active 矩形；几何条件精确分区，未知数据 mask 使用已证明安全范围的保守包络。
- store 只更新 active lane，inactive 保留原版本/初始化。load 只对实际读取施加越界、初始化与在途 DMA 冲突义务。
- 无 `other` 的 masked load，以及 masked swap 的 inactive 返回值都保留未定义性；不能假定为零或可靠旧值。
- 正确 select/where 可在同一 predicate 或已证明区域关系下屏蔽无效分支；不猜测不同数据 predicate 的代数关系。
- 未定义值不得进入有效输出、索引、mask/控制条件或实际 DMA 源；`0 * undefined` 仍未定义。
  reduce/dot 检查全部参与输入，归约后再 mask 不能修复已经读入的 padding。
- 自动输入已分配 VMEM padding 可先读再正确清理；越界 HBM、scratch/output 的 active 未初始化读取仍拒绝。
  普通输出最终只要求有效 HBM 域有定义，不要求未提交 padding 初始化。
- 远端数据 masked store 可保留 inactive 区域的条件未定义状态；互补写入、同谓词保护读取等已审核模式
  可完成初始化证明。实际消费仍须有定义，不推断任意不同数据谓词之间的关系；漏写、错误谓词、
  未定义值用于控制/索引/DMA 或缺 wait 仍拒绝，见[条件初始化回归](../../../../tests/test_pallas_remote_mask.py)。

源码：[masked_access.py](../../../../src/jax_reorder/core/masked_access.py)、
[remote_coordinates.py](../../../../src/jax_reorder/core/remote_coordinates.py)、
[remote_memory.py](../../../../src/jax_reorder/core/remote_memory.py)。

## Ref 变换、DMA 与预算

reshape/transpose/index 组合必须映回同一真实存储，保留 lane 次序、维度及分片元数据。
有洞矩形不能用包围区间替代；一个源 DMA 的计数不能因分析拆分矩形而改变。
受限非连续 DMA 只接受已审核默认布局/对齐规则，非默认物理布局与任意向量索引尚未普遍支持。
已审核的布局编译选项（如 `needs_layout_passes=False`）可保留并投影；该范围不扩展到任意物理布局或执行选项。

只读输入源的有界运行时页号支持多笔 DMA 同时在途：每笔 start 保存自己的页选择与源版本，wait 提交对应值，
候选源保护足迹与实际传输量分别记录。动态可写源/目标仍需完整状态分支证明；未知范围、在途冲突及错误等待
继续拒绝。准入、数学正反例和证据篡改见[动态地址测试](../../../../tests/test_pallas_dynamic_views.py)。

规则区域保持符号化；有限动态地址、串行展开、分区与表达式工作共享预算。
快速路径和扩展路径不能重置预算或吞掉超限异常。对严格 Python `True` 的访问复用已证明 boxes，
避免重复 predicate 分区；这条优化保留真实 mask、定义性与 DMA 检查。

## 按需证据

| 修改什么 | 原始设计、反例及验收 |
| --- | --- |
| 有界动态页、多笔在途 DMA 与实际传输量 | [动态页 DMA 验收](../../../exec-plans/completed/pallas-bounded-page-dma.md) |
| 串行状态驱动 index_map 与独立输出窗口 | [顺序输出映射验收](../../../exec-plans/completed/pallas-output-map-and-pure-lowering.md) |
| 远端条件初始化与局部完整输出 | [设备局部等价收口](../../../exec-plans/completed/pallas-local-equivalence-closure.md) |
| scratch 中间坐标与局部路径选择 | [修复与复审](../../../exec-plans/completed/pallas-scratch-boolean-review.md) |
| 一维 reduce/dot 范围与投影准入 | [审查修复验收](../../../exec-plans/completed/pallas-review-fixes.md) |
| 一维广播与外层坐标范围 | [坐标修复验收](../../../exec-plans/completed/pallas-broadcast-coordinates.md) |
| Grid 展开、独立输出窗口 | [阶段 19](../../../exec-plans/completed/pallas-history/stage-19.md)、[阶段 18](../../../exec-plans/completed/pallas-history/stage-18.md) |
| 跨迭代 DMA、输出块状态 | [阶段 16](../../../exec-plans/completed/pallas-history/stage-16.md)、[阶段 17](../../../exec-plans/completed/pallas-history/stage-17.md) |
| Ref 组合、有洞传输、尾块有效域 | [阶段 22](../../../exec-plans/completed/pallas-history/stage-22.md)、[阶段 23](../../../exec-plans/completed/pallas-history/stage-23.md) |
| 本地 ANY / 多维远端存储 | [阶段 24](../../../exec-plans/completed/pallas-history/stage-24.md) |
| 并行 owner 尾块恢复与独立有限 oracle | [阶段 25](../../../exec-plans/completed/pallas-history/stage-25.md) |
| Mask、未定义消费、原始 where、预算回归 | [阶段 26](../../../exec-plans/completed/pallas-history/stage-26.md) |
