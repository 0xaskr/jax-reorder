# Pallas Grid 与内存

从[任务入口](../pallas-jax-equivalence.md)到此；地址、覆盖、生命周期与定义性全部属于 core。

## Grid、地址与窗口

- `program_id`、`num_programs`、index_map 与局部坐标使用同一 grid 计划。
  静态串行坐标在预算内重放；parallel 轴优先保留符号批量模板，不逐 tensor 元素枚举。
- 并行 program 轴不进入 kernel 原有 dot/reduce 的收缩轴，也不直接成为用户输出 shape。
- 各输出分别根据实际 index_map 划分窗口；同窗口可继续更新，换窗口按源 staging 契约提交/失效。
  串行意味着保留实际状态依赖；是否累加由 kernel 代码决定。
- 证明全局覆盖、逆映射及唯一 writer/owner 后才能组装输出；共享 HBM/Ref 不按 owner 克隆。
- 串行实例传递内存版本、scratch、计数器和在途 DMA。实例结束不隐含 wait、清零或 barrier。
  多维远端使用同一计划，额外通信义务见[通信与执行绑定](communication.md)。

核心源码：[grid_execution.py](../../../src/jax_reorder/core/grid_execution.py)、
[tensor_domain.py](../../../src/jax_reorder/core/tensor_domain.py)、
[tensor_memory.py](../../../src/jax_reorder/core/tensor_memory.py)。

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
源码：[tensor_owner_partition.py](../../../src/jax_reorder/core/tensor_owner_partition.py)。

## Mask 与定义性

- 以原 Ref、结果坐标和 mask 推导真实 active 矩形；几何条件精确分区，未知数据 mask 使用已证明安全范围的保守包络。
- store 只更新 active lane，inactive 保留原版本/初始化。load 只对实际读取施加越界、初始化与在途 DMA 冲突义务。
- 无 `other` 的 masked load，以及 masked swap 的 inactive 返回值都保留未定义性；不能假定为零或可靠旧值。
- 正确 select/where 可在同一 predicate 或已证明区域关系下屏蔽无效分支；不猜测不同数据 predicate 的代数关系。
- 未定义值不得进入有效输出、索引、mask/控制条件或实际 DMA 源；`0 * undefined` 仍未定义。
  reduce/dot 检查全部参与输入，归约后再 mask 不能修复已经读入的 padding。
- 自动输入已分配 VMEM padding 可先读再正确清理；越界 HBM、scratch/output 的 active 未初始化读取仍拒绝。
  普通输出最终只要求有效 HBM 域有定义，不要求未提交 padding 初始化。
- 远端数据 masked store 当前仍要求可能保留的 inactive 区域已初始化；该保守限制尚待补齐。

源码：[masked_access.py](../../../src/jax_reorder/core/masked_access.py)、
[remote_coordinates.py](../../../src/jax_reorder/core/remote_coordinates.py)、
[remote_memory.py](../../../src/jax_reorder/core/remote_memory.py)。

## Ref 变换、DMA 与预算

reshape/transpose/index 组合必须映回同一真实存储，保留 lane 次序、维度及分片元数据。
有洞矩形不能用包围区间替代；一个源 DMA 的计数不能因分析拆分矩形而改变。
受限非连续 DMA 只接受已审核默认布局/对齐规则，非默认物理布局与任意向量索引尚未普遍支持。

规则区域保持符号化；有限动态地址、串行展开、分区与表达式工作共享预算。
快速路径和扩展路径不能重置预算或吞掉超限异常。对严格 Python `True` 的访问复用已证明 boxes，
避免重复 predicate 分区；这条优化保留真实 mask、定义性与 DMA 检查。

## 按需证据

| 修改什么 | 原始设计、反例及验收 |
| --- | --- |
| Grid 展开、独立输出窗口 | [阶段 19](history/stage-19.md)、[阶段 18](history/stage-18.md) |
| 跨迭代 DMA、输出块状态 | [阶段 16](history/stage-16.md)、[阶段 17](history/stage-17.md) |
| Ref 组合、有洞传输、尾块有效域 | [阶段 22](history/stage-22.md)、[阶段 23](history/stage-23.md) |
| 本地 ANY / 多维远端存储 | [阶段 24](history/stage-24.md) |
| 并行 owner 尾块恢复与独立有限 oracle | [阶段 25](history/stage-25.md) |
| Mask、未定义消费、原始 where、预算回归 | [阶段 26](history/stage-26.md) |
